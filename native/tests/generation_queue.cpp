#include "kadan/generation_queue.hpp"
#include <iostream>

using Queue = kadan::serving::GenerationQueue;
using Kind = Queue::Kind;
void expect(bool value) { if (!value) throw std::runtime_error("assertion_failed"); }
template<class F> void rejects(F function) {
    bool rejected = false;
    try { function(); } catch (const std::runtime_error&) { rejected = true; }
    expect(rejected);
}
const Queue::Model text{"qwen/revision/context", kadan::Workload::llm, {20, 80}};
const Queue::Model image{"image/revision/size", kadan::Workload::image, {30, 90}};
void fifo() {
    Queue q({100, 100});
    auto a = q.submit(text), b = q.submit(text), c = q.submit(image), d = q.submit(text);
    auto load = q.poll(); expect(load.kind == Kind::load && load.request == a);
    expect(q.poll().reservation == load.reservation);
    q.loaded(a, true); expect(q.poll().kind == Kind::execute);
    q.completed(a, true);
    auto reuse = q.poll(); expect(reuse.kind == Kind::execute && reuse.request == b);
    expect(reuse.reservation == load.reservation);
    q.completed(b, true);
    auto cleanup = q.poll(); expect(cleanup.kind == Kind::cleanup);
    expect(q.snapshot().used == text.bytes);
    q.cleaned(cleanup.reservation, false);
    expect(q.poll().kind == Kind::cleanup && q.snapshot().used == text.bytes);
    q.cleaned(cleanup.reservation, true);
    expect(q.snapshot().used == kadan::Footprint({0, 0}));
    auto next = q.poll(); expect(next.kind == Kind::load && next.request == c);
    q.loaded(c, true); q.completed(c, true);
    q.cleaned(q.poll().reservation, true);
    expect(q.poll().request == d); // No same-model bypass across an image.
    q.loaded(d, true); q.completed(d, true); q.stop();
    q.cleaned(q.poll().reservation, true);
    expect(q.snapshot().residents == 0 && q.poll().kind == Kind::idle);
}
void video_fifo() {
    Queue q({100,100});
    auto a=q.submit(text);
    auto b=q.submit({"h3/revision",kadan::Workload::video,{50,0}});
    auto c=q.submit(image);
    q.poll();q.loaded(a,true);q.completed(a,true);
    q.cleaned(q.poll().reservation,true);
    auto v=q.poll();expect(v.request==b&&v.kind==Kind::load);q.loaded(b,true);
    q.cancel(b);q.completed(b,false);
    q.cleaned(v.reservation,false);expect(q.poll().request==b&&q.pending()==2);
    q.cleaned(v.reservation,true);expect(q.poll().request==c);q.loaded(c,true);q.completed(c,false);q.cleaned(q.poll().reservation,true);
    expect(q.snapshot().residents==0);
}
void cancellation() {
    for (bool during_load : {false, true}) {
        Queue q({100, 100});
        auto a = q.submit(text), b = q.submit(image), c = q.submit(text);
        auto action = q.poll();
        if (!during_load) q.loaded(a, true);
        expect(q.cancel(b) && !q.cancel(b));
        expect(q.cancel(a) && q.cancellation_requested());
        expect(q.snapshot().used == text.bytes);
        if (during_load) q.loaded(a, true); else q.completed(a, true);
        expect(q.poll().kind == Kind::cleanup);
        q.cleaned(action.reservation, false);
        expect(q.pending() == 2);
        q.cleaned(action.reservation, true);
        expect(q.poll().request == c && !q.cancellation_requested());
        q.loaded(c, false); // Partial allocation failure still needs cleanup.
        expect(q.snapshot().used == text.bytes);
        q.cleaned(q.poll().reservation, true);
        expect(q.pending() == 0 && q.snapshot().residents == 0);
    }
}
void bounds_and_shutdown() {
    Queue q({100, 100}, 2);
    auto bad = text; bad.bytes = {101, 0};
    rejects([&] { q.submit(bad); });
    auto a = q.submit(text); q.submit(image);
    rejects([&] { q.submit(text); });
    expect(q.snapshot().residents == 0);
    auto load = q.poll();
    rejects([&] { q.loaded(a + 1, true); });
    rejects([&] { q.completed(a, true); });
    q.stop(); expect(q.pending() == 1 && q.cancellation_requested());
    rejects([&] { q.submit(text); });
    q.loaded(a, false);
    rejects([&] { q.cleaned(load.reservation + 1, true); });
    q.cleaned(load.reservation, true);
    rejects([&] { q.cleaned(load.reservation, true); });
    expect(q.poll().kind == Kind::idle && q.snapshot().used == kadan::Footprint({0, 0}));
    Queue queued({100, 100}); queued.submit(text); queued.stop();
    expect(queued.pending() == 0 && queued.poll().kind == Kind::idle);
    Queue changed({100, 100});
    auto first = changed.submit(text); bad = text; bad.bytes = {21, 80}; changed.submit(bad);
    changed.poll(); changed.loaded(first, true); changed.completed(first, true);
    expect(changed.poll().kind == Kind::cleanup); // Configuration/accounting changes cannot reuse.
}
int main() {
    fifo(); video_fifo(); cancellation(); bounds_and_shutdown();
    std::cout << "generation queue tests passed\n";
}
