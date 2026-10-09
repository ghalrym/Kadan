#include "h3_codec.hpp"
#include <cerrno>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fcntl.h>
#include <poll.h>
#include <spawn.h>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
namespace kadan::video::h3 {
namespace {
void check(bool ok,const char* error){if(!ok)throw std::runtime_error(error);}
struct Child {
    pid_t pid=-1;
    void reap(){if(pid>0){kill(pid,SIGKILL);while(waitpid(pid,nullptr,0)<0&&errno==EINTR){}pid=-1;}}
    ~Child(){reap();}
};
struct Actions {
    posix_spawn_file_actions_t value;
    Actions(){check(posix_spawn_file_actions_init(&value)==0,"h3_codec_spawn_setup");}
    ~Actions(){posix_spawn_file_actions_destroy(&value);}
};
}
void encode_mp4(const std::string& input,const std::string& output,const std::atomic_bool& cancel){
    check(!cancel.load(),"h3_generation_cancelled");
    check(!std::filesystem::exists(output),"h3_codec_output_exists");
    Actions actions;
    check(posix_spawn_file_actions_addopen(&actions.value,STDIN_FILENO,"/dev/null",O_RDONLY,0)==0,"h3_codec_spawn_setup");
    check(posix_spawn_file_actions_addopen(&actions.value,STDOUT_FILENO,"/dev/null",O_WRONLY,0)==0,"h3_codec_spawn_setup");
    check(posix_spawn_file_actions_addopen(&actions.value,STDERR_FILENO,"/dev/null",O_WRONLY,0)==0,"h3_codec_spawn_setup");
    const char* args[]={"ffmpeg","-nostdin","-v","error","-n","-i",input.c_str(),"-an","-c:v","libx264","-preset","ultrafast","-threads","1","-pix_fmt","yuv420p","-movflags","+faststart","-fs","16777216","-f","mp4",output.c_str(),nullptr};
    Child child;check(posix_spawnp(&child.pid,"ffmpeg",&actions.value,nullptr,const_cast<char**>(args),environ)==0,"h3_codec_unavailable");
    try {
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::minutes(5);
        while(true){check(!cancel.load(),"h3_generation_cancelled");check(std::chrono::steady_clock::now()<deadline,"h3_codec_timeout");int status=0;const auto done=waitpid(child.pid,&status,WNOHANG);if(done==child.pid){child.pid=-1;check(WIFEXITED(status)&&WEXITSTATUS(status)==0,"h3_codec_failed");break;}check(done==0||(done<0&&errno==EINTR),"h3_codec_wait");poll(nullptr,0,25);}
        const auto bytes=std::filesystem::file_size(output);check(bytes>0&&bytes<16777216,"h3_codec_output_size");
    } catch (...) {child.reap();unlink(output.c_str());throw;}
}
}
