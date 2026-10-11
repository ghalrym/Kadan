#include "kadan/device_workspace.hpp"
#include <iostream>
#include <set>
#include <string>
using namespace kadan;
namespace {
void check(bool ok){if(!ok)throw std::runtime_error("workspace_test_failed");}
template<class F>void fails(F function){try{function();}catch(const std::exception&){return;}throw std::runtime_error("expected_failure");}
struct Counters {
    Bytes allocated=0,peak=0;
    unsigned allocations=0,frees=0,creates=0,destroys=0,resets=0;
    bool handle=false;
    std::string fail,cleanup_fail;
    std::vector<std::string> events;
    std::set<std::thread::id> threads;
};
struct FakeOperations final:DeviceOperations {
    Resources& resources;std::shared_ptr<Counters> counts;int device;
    FakeOperations(Resources& r,std::shared_ptr<Counters> c,int d):resources(r),counts(std::move(c)),device(d){}
    void event(const char* name){counts->events.emplace_back(name);counts->threads.insert(std::this_thread::get_id());if(counts->fail==name||counts->cleanup_fail==name)throw std::runtime_error(name);}
    void select(int d)override{check(d==device);event("select");}
    void allocate(Bytes bytes)override{
        check(!counts->allocated&&resources.snapshot().used[device+1]>=bytes);
        event("allocate");counts->allocated=bytes;counts->peak=std::max(counts->peak,bytes);++counts->allocations;
    }
    void free()override{check(counts->allocated&&!counts->handle);event("free");counts->allocated=0;++counts->frees;}
    void create_handle()override{check(!counts->handle);event("create");counts->handle=true;++counts->creates;}
    void configure_handle()override{check(counts->handle);event("configure");}
    void destroy_handle()override{check(counts->handle);event("destroy");counts->handle=false;++counts->destroys;}
    void synchronize()override{event("synchronize");}
    void reset()override{check(!counts->allocated&&!counts->handle);event("reset");++counts->resets;}
};
std::unique_ptr<DeviceOperations> fake(Resources& r,std::shared_ptr<Counters> c,int d=0){return std::make_unique<FakeOperations>(r,std::move(c),d);}
void operation_and_cleanup_errors(){
    Resources r({1024,512});auto c=std::make_shared<Counters>();DeviceWorkspace w(r,0,Workload::image,fake(r,c));
    c->fail="configure";c->cleanup_fail="destroy";bool combined=false;
    try{w.ensure(128);}catch(const OperationCleanupError& e){combined=true;check(exception_message(e.operation)=="configure"&&exception_message(e.cleanup)=="destroy");check(std::string(e.what()).find("configure")!=std::string::npos&&std::string(e.what()).find("destroy")!=std::string::npos);}
    check(combined&&w.quarantined()&&r.snapshot().used[1]==128);
    Resources other({1024,512});auto d=std::make_shared<Counters>();DeviceThread thread(other,0,Workload::image,fake(other,d));std::atomic_bool failed=false;
    auto result=thread.submit_work([&](DeviceWorkspace& workspace){workspace.ensure(64);d->fail="synchronize";throw std::invalid_argument("request_error");},&failed);
    combined=false;try{result.get();}catch(const OperationCleanupError& e){combined=true;check(exception_message(e.operation)=="request_error"&&exception_message(e.cleanup)=="synchronize");}
    check(combined&&failed.load()&&other.snapshot().used[1]==64);
    bool original=false;try{rethrow_after_cleanup(std::make_exception_ptr(std::invalid_argument("original")),[]{});}catch(const std::invalid_argument& e){original=std::string(e.what())=="original";}check(original);
}
void reuse_growth_and_release(){
    Resources r({1024,512});auto c=std::make_shared<Counters>();DeviceWorkspace w(r,0,Workload::image,fake(r,c));
    w.ensure(128);w.synchronize();w.ensure(64);w.ensure(128);
    check(c->allocations==1&&c->creates==1&&w.retained_bytes()==128&&r.snapshot().used[1]==128);
    w.ensure(512);check(c->allocations==2&&c->frees==1&&c->peak==512&&r.snapshot().residents==1);
    w.release();check(r.snapshot().residents==0&&c->allocated==0&&!c->handle);
    const auto events=c->events.size();w.release();check(c->events.size()==events+1); // Synchronization is harmless/idempotent.
    w.reset();check(c->resets==1);const auto after=c->events.size();w.reset();check(c->events.size()==after);
    w.ensure(32);w.reset();check(c->resets==2&&r.snapshot().used[1]==0);
}
void accounting_and_construction_failures(){
    for(const auto* fault:{"allocate","create","configure"}){
        Resources r({1024,512});auto c=std::make_shared<Counters>();c->fail=fault;DeviceWorkspace w(r,0,Workload::tts,fake(r,c));
        fails([&]{w.ensure(128);});check(!w.quarantined()&&r.snapshot().residents==0&&!c->allocated&&!c->handle);
        c->fail.clear();w.ensure(128);w.reset();check(r.snapshot().residents==0);
    }
    Resources r({1024,512});auto c=std::make_shared<Counters>();DeviceWorkspace w(r,0,Workload::tts,fake(r,c));
    auto other=r.reserve(Workload::llm,{0,256});w.ensure(128);
    fails([&]{w.ensure(300);});check(!w.quarantined()&&r.snapshot().used[1]==256&&c->peak==128);
    r.released(other);w.ensure(512);w.release();check(r.snapshot().residents==0);
    fails([&]{w.ensure(513);});check(r.snapshot().residents==0);
}
void uncertain_cleanup_stays_charged(){
    for(const auto* fault:{"synchronize","destroy","free"}){
        Resources r({1024,512});auto c=std::make_shared<Counters>();DeviceWorkspace w(r,0,Workload::image,fake(r,c));
        w.ensure(128);c->fail=fault;fails([&]{w.release();});
        check(w.quarantined()&&r.snapshot().used[1]==128);
        const auto events=c->events.size();c->fail.clear();fails([&]{w.ensure(64);});fails([&]{w.reset();});
        check(c->events.size()==events&&r.snapshot().residents==1);
    }
    // Growth must also quarantine a failed destruction rather than retry it.
    Resources r({1024,512});auto c=std::make_shared<Counters>();DeviceWorkspace w(r,0,Workload::image,fake(r,c));
    w.ensure(128);c->fail="destroy";fails([&]{w.ensure(256);});check(w.quarantined()&&r.snapshot().used[1]==128&&c->allocations==1);
}
void reset_failure_keeps_context_accounting(){
    Resources r({1024,512});auto context=r.reserve(Workload::tts,{0,64});auto c=std::make_shared<Counters>();DeviceWorkspace w(r,0,Workload::tts,fake(r,c));
    w.ensure(128);c->fail="reset";fails([&]{w.reset();});
    check(w.quarantined()&&r.snapshot().used[1]==64&&r.snapshot().residents==1);
    // The owner has received no reset acknowledgement and must not release context.
    check(r.reservation(context).bytes[1]==64);
}
void persistent_threads_and_cancel(){
    Resources r({1024,512,512});auto a=std::make_shared<Counters>(),b=std::make_shared<Counters>();
    {
        DeviceThread first(r,0,Workload::image,fake(r,a,0)),second(r,1,Workload::image,fake(r,b,1));
        for(int i=0;i<3;++i){
            auto x=first.submit_work([](DeviceWorkspace& w){w.ensure(128);});
            auto y=second.submit_work([](DeviceWorkspace& w){w.ensure(256);});x.get();y.get();
        }
        check(a->threads.size()==1&&b->threads.size()==1&&*a->threads.begin()!=*b->threads.begin());
        check(a->allocations==1&&b->allocations==1&&r.snapshot().used==Footprint({0,128,256}));
        std::atomic_bool failed=false;
        auto cancelled=first.submit_work([](DeviceWorkspace& w){w.ensure(64);throw std::runtime_error("cancelled");},&failed);
        fails([&]{cancelled.get();});check(failed.load()&&!a->allocated&&r.snapshot().used[1]==0&&r.snapshot().used[2]==256);
        second.submit([](DeviceWorkspace& w){w.release();}).get();
        check(r.snapshot().residents==0);
        first.submit_work([](DeviceWorkspace& w){w.ensure(64);}).get();
        first.submit([](DeviceWorkspace& w){w.reset();}).get();
        second.submit([](DeviceWorkspace& w){w.reset();}).get();
        check(a->threads.size()==1&&b->threads.size()==1&&a->resets==1&&b->resets==1);
        // Destruction drains a pending job before releasing its resources.
        first.submit_work([](DeviceWorkspace& w){w.ensure(64);});
    }
    check(r.snapshot().residents==0&&a->threads.size()==1);
}
void selection_failure_never_uses_default_device(){
    Resources r({1024,512});auto c=std::make_shared<Counters>();c->fail="select";
    DeviceWorkspace w(r,0,Workload::tts,fake(r,c));fails([&]{w.ensure(128);});
    check(w.quarantined()&&c->events==std::vector<std::string>{"select"}&&r.snapshot().residents==0);
}
void destruction_does_not_acknowledge_failed_cleanup(){
    Resources r({1024,512});auto c=std::make_shared<Counters>();
    {
        DeviceThread thread(r,0,Workload::image,fake(r,c));
        thread.submit_work([](DeviceWorkspace& w){w.ensure(128);}).get();
        c->fail="synchronize";
    }
    check(r.snapshot().used[1]==128&&c->allocated==128&&c->frees==0);
}
void bounded_mailbox(){
    Resources r({1024,512});auto c=std::make_shared<Counters>();DeviceThread t(r,0,Workload::tts,fake(r,c));
    std::promise<void> entered,finish;auto finished=finish.get_future();
    auto active=t.submit([&](DeviceWorkspace&){entered.set_value();finished.wait();});entered.get_future().get();
    auto pending=t.submit([](DeviceWorkspace&){});bool rejected=false;
    try {t.submit([](DeviceWorkspace&){});}catch(const std::runtime_error&){rejected=true;}
    finish.set_value();active.get();pending.get();check(rejected);
}
}
int main(){
    operation_and_cleanup_errors();reuse_growth_and_release();accounting_and_construction_failures();uncertain_cleanup_stays_charged();
    reset_failure_keeps_context_accounting();persistent_threads_and_cancel();selection_failure_never_uses_default_device();destruction_does_not_acknowledge_failed_cleanup();bounded_mailbox();
    std::cout<<"Mock device workspace lifecycle passed (no CUDA or inference)\n";
}
