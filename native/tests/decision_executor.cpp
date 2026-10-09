#include "kadan/decision_executor.hpp"
#include "decision_tokenizer.hpp"
#include "kadan/generation_queue.hpp"
#include <cblas.h>
#include <fstream>
#include <iostream>

using namespace kadan;
using kadan::decision::Json;
using Executor = decision::Executor;
using Queue = serving::GenerationQueue;
using Kind = Queue::Kind;
constexpr Bytes MiB = 1024 * 1024;
void expect(bool ok) {
    if (!ok)
        throw std::runtime_error("assertion_failed");
}
template <class F> void rejects(F &&call) {
    bool failed = false;
    try {
        call();
    } catch (const std::exception &) {
        failed = true;
    }
    expect(failed);
}
const std::string request =
    R"({"state":"red apple","questions":[{"key":"x","type":"Choice","instructions":"color?","options":[{"key":"red","description":"red"},{"key":"blue","description":"blue"}]},{"key":"s","type":"Score","instructions":"quality?","levels":["bad","good","great"]},{"key":"n","type":"Noul","instructions":"red?","trueWhen":"red","falseWhen":"blue"}]})";

void tokenizer(const std::string &root) {
    std::atomic_bool cancel{false};
    decision::Tokenizer tok(decision::read_json(root + "/tokenizer/tokenizer.json", 8 * MiB));
    expect(tok.encode("abc", cancel) == std::vector<int>{261});
    expect(tok.encode("caf\xc3\xa9", cancel) == tok.encode("cafe\xcc\x81", cancel));
    expect(tok.encode("a[SEP]b", cancel) == std::vector<int>({97, 257, 98}));
    rejects([&] { tok.encode("\xff", cancel); });
    rejects([&] { tok.encode("literal [MASK]", cancel); });
    expect(tok.encode("[MAS\xe2\x84\xaa]", cancel) == std::vector<int>({91, 77, 65, 83, 75, 93}));
    auto config = decision::read_json(root + "/tokenizer/tokenizer.json", 8 * MiB);
    config["added_tokens"][2]["normalized"] = true;
    decision::Tokenizer normalized_mask(config);
    rejects([&] { normalized_mask.encode("[MAS\xe2\x84\xaa]", cancel); });
    cancel = true;
    rejects([&] { tok.encode("abc", cancel); });
    rejects([] { decision::parse_json("{\"a\":1,\"a\":2}"); });
    rejects([] { decision::parse_json(std::string(40, '[') + "0" + std::string(40, ']')); });
}

void lifecycle(const std::string &root) {
    auto resources = std::make_shared<Resources>(Footprint{Executor::envelope_bytes});
    std::atomic_bool cancel{false};
    Executor executor(resources);
    rejects([&] { executor.execute(request, cancel); });
    cancel = true;
    rejects([&] { executor.load(root, cancel); });
    expect(resources->snapshot().used[0] == 0);
    cancel = false;
    int visited = 0;
    rejects([&] {
        executor.load(root, cancel, 0, [&](auto) {
            if (++visited == 4)
                cancel = true;
        });
    });
    expect(visited == 4 && !executor.loaded() && resources->snapshot().used[0] == 0);
    cancel = false;
    rejects([&] { executor.load(root + "/missing", cancel); });
    expect(resources->snapshot().residents == 0);
    executor.load(root, cancel);
    auto retained = resources->snapshot().used[0];
    expect(retained > 128 * MiB && retained < 129 * MiB);
    auto answer = Json::parse(executor.execute(request, cancel));
    expect(answer["answers"].size() == 3);
    expect(answer["answers"][0]["value"].is_string());
    expect(answer["answers"][1]["value"].is_number());
    expect(answer["answers"][2]["probabilities"].is_null());
    // Full numerical reference values are checked by the fixture driver.
    expect(resources->snapshot().used[0] == retained);
    expect(Json::parse(executor.execute(request, cancel)) == answer);
    for (auto stage : {"tokenized", "encoder", "head", "publication"}) {
        bool reached = false;
        rejects([&] {
            executor.execute(request, cancel, [&](auto now) {
                expect(resources->snapshot().used[0] == retained + 256 * MiB);
                rejects([&] { executor.unload(); });
                if (now == stage) {
                    reached = true;
                    cancel = true;
                }
            });
        });
        expect(reached && resources->snapshot().used[0] == retained);
        cancel = false;
    }
    for (auto invalid : {"{}", "[]", "null", "{\"state\":0,\"questions\":[]}"})
        rejects([&] { executor.execute(invalid, cancel); });
    for (int failure = 0; failure < 6; ++failure) {
        auto bad = Json::parse(request);
        if (failure == 0)
            bad["questions"][1]["key"] = "x";
        if (failure == 1)
            bad["questions"][0]["options"][1]["key"] = "red";
        if (failure == 2)
            bad["state"] = std::string(600, 'x');
        if (failure == 3)
            bad["questions"][0]["instructions"] = "[MASK]";
        if (failure == 4)
            bad["questions"][0]["options"][0]["description"] = std::string(100, 'z');
        if (failure == 5)
            bad["questions"][0]["type"] = "Unknown";
        rejects([&] { executor.execute(bad.dump(), cancel); });
        expect(resources->snapshot().used[0] == retained);
    }
    auto single = Json::parse(request);
    single["questions"] = Json::array({single["questions"][1]});
    single["questions"][0]["levels"] = Json::array({"one"});
    auto one = Json::parse(executor.execute(single.dump(), cancel));
    expect(one["answers"][0]["value"] == 0 && one["answers"][0]["confidence"] == 1);
    executor.unload();
    executor.unload();
    expect(resources->snapshot().used[0] == 0 && resources->snapshot().residents == 0);
    auto tight = std::make_shared<Resources>(Footprint{128 * MiB});
    Executor denied(tight);
    rejects([&] { denied.load(root, cancel); });
    expect(!denied.loaded() && tight->snapshot().used[0] == 0);
    auto no_scratch = std::make_shared<Resources>(Footprint{129 * MiB});
    Executor limited(no_scratch);
    limited.load(root, cancel);
    auto before = no_scratch->snapshot().used[0];
    rejects([&] { limited.execute(request, cancel); });
    expect(no_scratch->snapshot().used[0] == before);
    limited.unload();
    expect(no_scratch->snapshot().residents == 0);
    std::cout << answer.dump() << '\n';
}

void fifo(const std::string &root) {
    auto resources = std::make_shared<Resources>(Footprint{Executor::envelope_bytes});
    Queue queue(resources);
    Executor executor(resources);
    std::atomic_bool cancel{false};
    Queue::Model model{root, Workload::decision, {Executor::envelope_bytes}};
    auto a = queue.submit(model), b = queue.submit(model);
    auto middle = queue.submit({"image/revision", Workload::image, {64 * MiB}});
    auto removed = queue.submit(model), last = queue.submit(model);
    expect(queue.cancel(removed));
    auto load = queue.poll();
    expect(load.kind == Kind::load && load.request == a);
    executor.load(root, cancel, load.reservation);
    queue.loaded(a, true);
    auto run = [&](Handle id) {
        auto action = queue.poll();
        expect(action.kind == Kind::execute && action.request == id);
        executor.execute(request, cancel);
        expect(resources->snapshot().used[0] == Executor::envelope_bytes);
        queue.completed(id, true);
    };
    run(a);
    expect(queue.poll().reservation == load.reservation);
    run(b);
    auto cleanup = queue.poll();
    expect(cleanup.kind == Kind::cleanup);
    queue.cleaned(cleanup.reservation, false);
    expect(queue.poll().kind == Kind::cleanup && executor.loaded());
    executor.unload();
    expect(resources->snapshot().used[0] == Executor::envelope_bytes);
    queue.cleaned(cleanup.reservation, true);
    expect(resources->snapshot().used[0] == 0);
    auto image = queue.poll();
    expect(image.kind == Kind::load && image.request == middle);
    queue.loaded(middle, false);
    queue.cleaned(image.reservation, true);
    auto final = queue.poll();
    expect(final.kind == Kind::load && final.request == last);
    executor.load(root, cancel, final.reservation);
    queue.loaded(last, true);
    // Active cancellation retains reservation until all model ownership is gone.
    queue.cancel(last);
    cancel = true;
    rejects([&] { executor.execute(request, cancel); });
    queue.completed(last, true);
    expect(resources->snapshot().used[0] == Executor::envelope_bytes);
    executor.unload();
    queue.cleaned(final.reservation, true);
    queue.stop();
    expect(queue.poll().kind == Kind::idle && resources->snapshot().used[0] == 0);
    auto wrong = resources->reserve(Workload::image, {Executor::envelope_bytes});
    cancel = false;
    rejects([&] { executor.load(root, cancel, wrong); });
    resources->released(wrong);
}
int main(int argc, char **argv) {
    expect(argc == 2);
    openblas_set_num_threads(1);
    tokenizer(argv[1]);
    lifecycle(argv[1]);
    fifo(argv[1]);
}
