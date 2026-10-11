#include "h3_codec.hpp"
#include <cerrno>
#include <charconv>
#include <limits>
#include <vector>
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
struct Progress {
    int descriptors[2]{-1,-1};std::string pending;std::size_t frames=0;bool ended=false;
    Progress(){check(pipe2(descriptors,O_CLOEXEC)==0,"h3_codec_pipe");const auto flags=fcntl(descriptors[0],F_GETFL);if(flags<0||fcntl(descriptors[0],F_SETFL,flags|O_NONBLOCK)<0){close(descriptors[0]);close(descriptors[1]);throw std::runtime_error("h3_codec_pipe");}}
    ~Progress(){for(int fd:descriptors)if(fd>=0)close(fd);}
    void close_writer(){close(descriptors[1]);descriptors[1]=-1;}
    void drain(){char bytes[1024];while(true){const auto n=read(descriptors[0],bytes,sizeof(bytes));if(n==0||(n<0&&(errno==EAGAIN||errno==EWOULDBLOCK)))break;if(n<0&&errno==EINTR)continue;check(n>0,"h3_codec_progress");for(ssize_t i=0;i<n;++i){if(bytes[i]!='\n'){check(pending.size()<1024,"h3_codec_progress");pending+=bytes[i];continue;}if(pending.compare(0,6,"frame=")==0){std::size_t count=0;const auto first=pending.data()+6,last=pending.data()+pending.size();const auto parsed=std::from_chars(first,last,count);check(parsed.ec==std::errc{}&&parsed.ptr==last,"h3_codec_progress");frames=count;}if(pending=="progress=end")ended=true;pending.clear();}}}
};
struct Actions {
    posix_spawn_file_actions_t value;
    Actions(){check(posix_spawn_file_actions_init(&value)==0,"h3_codec_spawn_setup");}
    ~Actions(){posix_spawn_file_actions_destroy(&value);}
};
}
void encode_mp4(const std::string& input,const std::string& output,const std::atomic_bool& cancel,std::size_t expected_frames,const std::string& audio){
    check(!cancel.load(),"h3_generation_cancelled");
    check(!std::filesystem::exists(output),"h3_codec_output_exists");
    const auto input_bytes=std::filesystem::file_size(input);check(input_bytes<std::numeric_limits<std::uintmax_t>::max()-16777216,"h3_codec_input_size");const auto byte_limit=input_bytes+16777216;const auto limit=std::to_string(byte_limit);
    Progress progress;Actions actions;
    check(posix_spawn_file_actions_addopen(&actions.value,STDIN_FILENO,"/dev/null",O_RDONLY,0)==0,"h3_codec_spawn_setup");
    check(posix_spawn_file_actions_adddup2(&actions.value,progress.descriptors[1],STDOUT_FILENO)==0,"h3_codec_spawn_setup");
    check(posix_spawn_file_actions_addopen(&actions.value,STDERR_FILENO,"/dev/null",O_WRONLY,0)==0,"h3_codec_spawn_setup");
    std::vector<const char*> args{"ffmpeg","-nostdin","-v","error","-n","-i",input.c_str()};
    if(audio.empty())args.insert(args.end(),{"-an"});
    else args.insert(args.end(),{"-i",audio.c_str(),"-map","0:v:0","-map","1:a:0","-c:a","aac","-b:a","192k"});
    args.insert(args.end(),{"-c:v","libx264","-preset","ultrafast","-threads","1","-pix_fmt","yuv420p","-movflags","+faststart","-fs",limit.c_str(),"-progress","pipe:1","-nostats","-f","mp4",output.c_str(),nullptr});
    // Deliberately no -shortest: rounded audio length must never remove a video frame.
    Child child;check(posix_spawnp(&child.pid,"ffmpeg",&actions.value,nullptr,const_cast<char**>(args.data()),environ)==0,"h3_codec_unavailable");progress.close_writer();
    try {
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::minutes(5);
        while(true){progress.drain();check(!cancel.load(),"h3_generation_cancelled");check(std::chrono::steady_clock::now()<deadline,"h3_codec_timeout");int status=0;const auto done=waitpid(child.pid,&status,WNOHANG);if(done==child.pid){child.pid=-1;check(WIFEXITED(status)&&WEXITSTATUS(status)==0,"h3_codec_failed");break;}check(done==0||(done<0&&errno==EINTR),"h3_codec_wait");poll(nullptr,0,25);}
        progress.drain();check(!expected_frames||(progress.ended&&progress.frames==expected_frames),"h3_codec_frame_count");
        const auto bytes=std::filesystem::file_size(output);check(bytes>0&&bytes<byte_limit,"h3_codec_output_size");
    } catch (...) {child.reap();unlink(output.c_str());throw;}
}
}
