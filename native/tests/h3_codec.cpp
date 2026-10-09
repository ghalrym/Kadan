#include "h3_codec.hpp"
#include <atomic>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <chrono>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>
void check(bool b){if(!b)throw std::runtime_error("codec_test_failed");}
template<class F>void fails(F f,const char* expected){try{f();}catch(const std::exception& e){check(std::string(e.what())==expected);return;}throw std::runtime_error("expected_failure");}
int main(int argc,char** argv){try{
 if(std::filesystem::path(argv[0]).filename()=="ffmpeg") {std::ifstream in(argv[6]);std::string mode;in>>mode;std::ofstream out(argv[argc-1]);out<<"test codec payload";out.close();std::cout<<"frame="<<(mode=="truncated"?1:22)<<"\nprogress=end\n";if(mode=="fail")return 7;if(mode=="wait")poll(nullptr,0,10000);return 0;}
 char pattern[]="/tmp/kadan-codec-test-XXXXXX";check(mkdtemp(pattern)!=nullptr);std::filesystem::path root(pattern);std::filesystem::create_symlink(std::filesystem::canonical(argv[0]),root/"ffmpeg");const auto old=std::string(getenv("PATH"));setenv("PATH",root.c_str(),1);auto input=(root/"input").string(),output=(root/"output").string();auto mode=[&](const char* m){std::ofstream f(input);f<<m;};std::atomic_bool cancel=false;
 mode("ok");kadan::video::h3::encode_mp4(input,output,cancel);check(std::filesystem::file_size(output)==18);fails([&]{kadan::video::h3::encode_mp4(input,output,cancel);},"h3_codec_output_exists");std::filesystem::remove(output);
 mode("ok");kadan::video::h3::encode_mp4(input,output,cancel,22);std::filesystem::remove(output);
 mode("truncated");fails([&]{kadan::video::h3::encode_mp4(input,output,cancel,22);},"h3_codec_frame_count");check(!std::filesystem::exists(output));
 mode("fail");fails([&]{kadan::video::h3::encode_mp4(input,output,cancel);},"h3_codec_failed");check(!std::filesystem::exists(output));
 mode("wait");std::thread stop([&]{std::this_thread::sleep_for(std::chrono::milliseconds(100));cancel=true;});fails([&]{kadan::video::h3::encode_mp4(input,output,cancel);},"h3_generation_cancelled");stop.join();check(!std::filesystem::exists(output));check(waitpid(-1,nullptr,WNOHANG)==-1&&errno==ECHILD);
 cancel=false;setenv("PATH","/nonexistent",1);fails([&]{kadan::video::h3::encode_mp4(input,output,cancel);},"h3_codec_unavailable");setenv("PATH",old.c_str(),1);std::filesystem::remove_all(root);std::cout<<"PASS codec success, no overwrite, failure, cancellation and reaping\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
