#pragma once
#include <cstddef>
#include <iosfwd>
#include <string_view>
#include "kadan/resources.hpp"
#include "kadan/weight_backing.hpp"
#include <memory>
namespace kadan::serving {
constexpr std::size_t max_capacity=262144, max_frame=64;
constexpr std::size_t metadata_bytes=256*1024*1024, staging_bytes=1024*1024, control_bytes=1024*1024;
std::size_t number(std::string_view);
struct Info {std::size_t vocabulary,capacity,arena_bytes,host_bytes;};
struct Token {unsigned id;bool eos;std::size_t committed;};
class Engine {
public:
    virtual ~Engine()=default;
    virtual Info info()const=0;
    virtual void reset()=0;
    virtual Token step(unsigned,bool)=0;
    // Must verify physical cleanup and zero reservations before returning.
    virtual void close()=0;
};
// Parent supervises deadlines and process termination. Any exception is terminal;
// caller emits a fixed error frame, exits nonzero, and never reuses this engine.
void session(Engine&,std::istream&,std::ostream&);
class ResidentEngine : public Engine {
public:
    virtual void begin_request()=0;
    virtual void end_request()=0;
    virtual void park()=0;
    virtual WeightBacking::Stats cache_stats()const { return {}; }
};
// Unique worker incarnation; every v2 command/reply also carries its FIFO ID.
std::string session_identity();
void resident_session(ResidentEngine&,std::shared_ptr<Resources>,std::string_view identity,
                      std::istream&,std::ostream&);
}
