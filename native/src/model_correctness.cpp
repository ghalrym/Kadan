#include "kadan/cuda_model.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <charconv>
#include <csignal>
#include <fcntl.h>
#include <iostream>
#include <string_view>
#include <unistd.h>
namespace {
void require(bool v,const char* e){if(!v)throw std::invalid_argument(e);}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
std::size_t number(const char* s){std::size_t n=0;std::string_view v(s);auto[e,c]=std::from_chars(v.data(),v.data()+v.size(),n);require(c==std::errc{}&&e==v.data()+v.size(),"invalid_number");return n;}
// Fail-stop watchdog deliberately exits the process without attempting recovery
// from a possibly stuck CUDA call. This harness never serves another request.
void expired(int){_exit(124);}
struct Output {
    int fd=-1;explicit Output(const char* path){fd=::open(path,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600);if(fd<0)throw std::runtime_error("output_create_failed");}
    ~Output(){if(fd>=0)::close(fd);}
    void write(const void* p,std::size_t n){auto*s=static_cast<const char*>(p);while(n){auto v=::write(fd,s,n);if(v<0&&errno==EINTR)continue;if(v<=0)throw std::runtime_error("output_write_failed");s+=v;n-=std::size_t(v);}}
    void finish(){if(::fsync(fd))throw std::runtime_error("output_sync_failed");auto old=fd;fd=-1;if(::close(old))throw std::runtime_error("output_close_failed");}
};
}
int main(int argc,char**argv){
    // All arguments explicit; no default root/device or execution on an empty command.
    if(argc<11||argc>18||std::string_view(argv[1])!="--execute"){
        std::cerr<<"Usage: kadan-model-correctness --execute ROOT DEVICE CAPACITY HOST_BYTES DEVICE_BYTES HEADROOM_BYTES TIMEOUT_SECONDS OUTPUT_FILE TOKEN_ID [up to 8 IDs total]\n";return 2;
    }
    try{static_assert(std::endian::native==std::endian::little);auto device=number(argv[3]);require(device<kadan::Resources::max_devices,"device_range");kadan::cuda::ModelOptions o;o.capacity=number(argv[4]);o.device_headroom=number(argv[7]);auto timeout=number(argv[8]);
        require(o.capacity>0&&o.capacity<=128&&timeout>0&&timeout<=900,"correctness_bounds");std::array<unsigned,8> tokens{};const auto count=std::size_t(argc-10);require(count<=o.capacity,"input_capacity");for(std::size_t i=0;i<count;++i){auto n=number(argv[10+i]);require(n<=UINT32_MAX,"token_range");tokens[i]=unsigned(n);}
        kadan::Footprint cap(device+2);cap[0]=number(argv[5]);cap[device+1]=number(argv[6]);auto r=std::make_shared<kadan::Resources>(cap);
        // Metadata-only preflight validates all IDs and admission before any GPU
        // call. Its reservation is released only after the parser/layout die.
        kadan::Footprint host(cap.size());host[0]=kadan::cuda::Model::host_bytes(o);auto ticket=r->reserve(kadan::Workload::llm,host);std::size_t vocab=0,arena=0;
        {auto budget=std::make_shared<kadan::checkpoint::MemoryBudget>(o.metadata_bytes);kadan::checkpoint::ModelManifest m(argv[2],budget);vocab=m.architecture().vocab;for(std::size_t i=0;i<count;++i)require(tokens[i]<vocab,"token_range");auto g=kadan::model::read_generation(argv[2],vocab,budget);kadan::model::Layout l(m,o.capacity,g,budget);arena=l.device_bytes();require(arena<=cap[device+1]&&o.device_headroom<=cap[device+1]-arena,"device_envelope");}
        r->released(ticket);require(host[0]<=cap[0]&&vocab*sizeof(float)<=cap[0]-host[0],"host_envelope");host[0]=vocab*sizeof(float);auto diagnostics=r->reserve(kadan::Workload::llm,host);std::vector<float> logits(vocab);
        Output output(argv[9]);require(std::signal(SIGALRM,expired)!=SIG_ERR,"watchdog_signal");alarm(unsigned(timeout));check(cudaSetDevice(int(device)));
        kadan::cuda::Model model(argv[2],o,int(device),r);require(model.vocabulary()==vocab&&model.device_bytes()==arena,"preflight_changed");
        // Incomplete files have no DONE marker and MUST be discarded. Header and
        // each record are little endian; logits are FP32 containers of BF16 values.
        const std::array<std::uint32_t,4> header{0x4b4d4331,1,std::uint32_t(vocab),std::uint32_t(count)};output.write(header.data(),sizeof(header));
        for(std::size_t i=0;i<count;++i){auto selected=model.step(tokens[i],false);model.read_logits(logits);std::array<std::uint32_t,4> record{tokens[i],selected.token,unsigned(selected.eos),unsigned(model.tokens())};output.write(record.data(),sizeof(record));output.write(logits.data(),logits.size()*sizeof(float));}
        model.close();logits.clear();logits.shrink_to_fit();r->released(diagnostics);require(r->snapshot().residents==0,"reservation_leak");const std::uint32_t done=0x444f4e45;output.write(&done,sizeof(done));output.finish();alarm(0);
        std::cout<<"Completed "<<count<<" explicit input IDs; logits artifact complete; arena_bytes="<<arena<<"; zero final reservations. Correctness capture only, no throughput measurement.\n";
    }catch(const std::exception&e){std::cerr<<"FAIL_STOP: "<<e.what()<<"; discard incomplete output; no retry\n";return 1;}
}
