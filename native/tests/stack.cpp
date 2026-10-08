#include "stack_checks.hpp"
#include <iostream>
namespace {
template<class F>void rejects(F f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}stack_test::check(caught,"expected_error");}
}
int main(){try{
    using namespace stack_test;StackFixture f;auto c=f.config();auto p=kadan::stack::plan(c);
    rejects([&]{kadan::stack::Reference m(c,f.weights(),p.host_numeric_bytes-1);});auto bad=c;bad.layer[0].attention=kadan::decoder::Attention::full;rejects([&]{kadan::stack::plan(bad);});
    kadan::stack::Reference model(c,f.weights(),p.host_numeric_bytes);
    for(int replay=0;replay<2;++replay){unsigned next=0;for(std::size_t t=0;t<5;++t){unsigned input=t<2?stack_golden::input_ids[t]:next;check(input==stack_golden::input_ids[t],"generation_feedback");auto selected=model.step(input,t!=0);next=selected.token;check(next==stack_golden::selected[t]&&!selected.eos&&model.tokens()==t+1,"selection");verify(model,p,t);}model.reset();zero(model,p);}
    std::array<float,16>a{},b{};rejects([&]{model.read_output(a,b);});
    // Capacity guard refuses further work without publishing another token.
    for(std::size_t i=0;i<p.capacity;++i)model.step(2,false);rejects([&]{model.step(2,false);});check(model.valid()&&model.tokens()==p.capacity,"capacity");model.reset();
    std::atomic_bool cancelled=true;auto result=kadan::stack::Selection{99,false};rejects([&]{result=model.step(2,true,&cancelled);});check(result.token==99&&!model.valid()&&model.tokens()==0,"cancelled_result");model.reset();zero(model,p);
    rejects([&]{model.step(16);});check(!model.valid()&&model.tokens()==0,"bad_id");model.reset();
    // Equal logits select the lowest ID; EOS is terminal only when requested.
    StackFixture tied;tied.packed.fill(0);auto eos=c;eos.eos=0;kadan::stack::Reference ending(eos,tied.weights(),p.host_numeric_bytes);
    auto s=ending.step(2,false);check(s.token==0&&s.eos&&!ending.finished(),"prompt_eos");s=ending.step(7);check(s.token==0&&s.eos&&ending.finished()&&ending.tokens()==2,"eos");rejects([&]{ending.step(0);});check(ending.tokens()==2,"after_eos");ending.reset();zero(ending,p);
    // Last-layer failure occurs after three complete layers mutated their state.
    StackFixture late;for(std::size_t e=0;e<4;++e){late.layer[3].moe.globals[3*e]=1e20f;late.layer[3].moe.globals[3*e+1]=1e20f;}
    kadan::stack::Reference broken(c,late.weights(),p.host_numeric_bytes);result={99,false};rejects([&]{result=broken.step(2);});check(result.token==99&&!broken.valid()&&broken.tokens()==0,"late_publication");rejects([&]{broken.step(2);});rejects([&]{broken.read_output(a,b);});broken.reset();zero(broken,p);
    StackFixture head_failure;head_failure.global=1e38f;head_failure.scales.fill(0x38);
    for(std::size_t r=0;r<16;++r)for(std::size_t j=0;j<16;j+=2){auto code=[&](std::size_t k){return 4+(stack_golden::normalized[0][k]<0?8:0);};head_failure.packed[r*8+j/2]=std::uint8_t(code(j)|(code(j+1)<<4));}
    kadan::stack::Reference failed_head(c,head_failure.weights(),p.host_numeric_bytes);result={99,false};rejects([&]{result=failed_head.step(2);});check(result.token==99&&!failed_head.valid()&&failed_head.tokens()==0,"head_failure_publication");failed_head.reset();zero(failed_head,p);
    std::cout<<"Mixed-stack CPU goldens, selection, EOS/capacity/failure/reset passed; device arena "<<p.device_bytes<<", CPU numeric bytes "<<p.host_numeric_bytes<<".\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
