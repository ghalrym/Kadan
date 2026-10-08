#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <chrono>
#include <cstdio>
#include <map>
#include <vector>
#include <cstdint>
#include <csignal>
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point a){return std::chrono::duration<double,std::milli>(Clock::now()-a).count();}
template<class T>T sym(const char*n){return reinterpret_cast<T>(dlsym(RTLD_NEXT,n));}
struct Stat {unsigned long n=0;double gpu=0,host=0;};
static std::map<std::uintptr_t,Stat> stats; static double sync_ms=0,copy_ms[5]={};static unsigned long sync_n=0,copy_n[5]={};
struct Pair{cudaEvent_t a,b;std::uintptr_t key;};static std::vector<Pair> pending;
static void report();
static volatile sig_atomic_t boundary=0;static int phase=0;
__attribute__((constructor)) static void setup(){std::signal(SIGUSR1,[](int){boundary=1;});}
static void phase_check(){if(boundary){report();stats.clear();sync_ms=0;sync_n=0;for(int i=0;i<5;i++){copy_ms[i]=0;copy_n[i]=0;}phase++;boundary=0;}}
static void collect(){
 auto elapsed=sym<decltype(&cudaEventElapsedTime)>("cudaEventElapsedTime");auto destroy=sym<decltype(&cudaEventDestroy)>("cudaEventDestroy");
 for(auto&p:pending){float value=0;if(elapsed(&value,p.a,p.b)==cudaSuccess)stats[p.key].gpu+=value;destroy(p.a);destroy(p.b);}pending.clear();
}
extern "C" cudaError_t __cudaLaunchKernel(const void*f,dim3 grid,dim3 block,void**args,size_t shared,cudaStream_t stream){
 phase_check();
 auto create=sym<decltype(&cudaEventCreate)>("cudaEventCreate");auto record=sym<decltype(&cudaEventRecord)>("cudaEventRecord");
 Dl_info info{};const void* caller=__builtin_return_address(0);dladdr(caller,&info);auto key=reinterpret_cast<std::uintptr_t>(caller)-reinterpret_cast<std::uintptr_t>(info.dli_fbase);
 Pair p{};p.key=key;create(&p.a);create(&p.b);record(p.a,stream);auto t=Clock::now();auto e=sym<decltype(&cudaLaunchKernel)>("__cudaLaunchKernel")(f,grid,block,args,shared,stream);stats[key].host+=ms(t);stats[key].n++;record(p.b,stream);pending.push_back(p);return e;
}
extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t s){auto t=Clock::now();auto e=sym<decltype(&cudaStreamSynchronize)>("cudaStreamSynchronize")(s);sync_ms+=ms(t);sync_n++;if(e==cudaSuccess)collect();if(sync_n%5000==0)report();return e;}
extern "C" cudaError_t cudaMemcpy(void*d,const void*s,size_t n,cudaMemcpyKind kind){auto t=Clock::now();auto e=sym<decltype(&cudaMemcpy)>("cudaMemcpy")(d,s,n,kind);copy_ms[kind]+=ms(t);copy_n[kind]++;return e;}
__attribute__((destructor)) static void report(){char path[128];snprintf(path,sizeof(path),"/tmp/kadan-profile-phase-%d.tsv",phase);FILE*f=fopen(path,"w");if(!f)return;fprintf(f,"kernel_offset\tcalls\tgpu_ms\tlaunch_host_ms\n");for(auto&[k,s]:stats)fprintf(f,"%lx\t%lu\t%.6f\t%.6f\n",k,s.n,s.gpu,s.host);fprintf(f,"sync\t%lu\t%.6f\n",sync_n,sync_ms);for(int i=0;i<5;i++)fprintf(f,"copy%d\t%lu\t%.6f\n",i,copy_n[i],copy_ms[i]);fclose(f);}
