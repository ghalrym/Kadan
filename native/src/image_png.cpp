#include "kadan/image_generation.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
namespace kadan::image {
namespace {
using Bytes=std::vector<std::uint8_t>;
void need(bool b,const char* e){if(!b)throw std::runtime_error(e);}
void be(Bytes& b,std::uint32_t n){for(int i=24;i>=0;i-=8)b.push_back(std::uint8_t(n>>i));}
std::uint32_t crc(std::span<const std::uint8_t> data){std::uint32_t x=0xffffffff;for(auto b:data){x^=b;for(int i=0;i<8;++i)x=(x>>1)^(0xedb88320u&std::uint32_t(-int(x&1)));}return ~x;}
void chunk(Bytes& b,const char* tag,const Bytes& data){be(b,std::uint32_t(data.size()));auto start=b.size();b.insert(b.end(),tag,tag+4);b.insert(b.end(),data.begin(),data.end());be(b,crc(std::span(b).subspan(start)));}
}
void publish_png(const std::string& path,std::span<const float> rgba,std::size_t height,std::size_t width,const std::atomic_bool& cancel){need(height>0&&width>0&&height<=3072&&width<=3072&&height*width<=4608*1024&&rgba.size()==4*height*width,"image_png_shape");Bytes raw;raw.reserve(height*(1+width*4));for(std::size_t y=0;y<height;++y){need(!cancel.load(),"image_cancelled");raw.push_back(0);for(std::size_t x=0;x<width;++x)for(std::size_t c=0;c<4;++c){float v=rgba[(c*height+y)*width+x];need(std::isfinite(v),"image_png_nonfinite");raw.push_back(std::uint8_t(std::lround(std::clamp(v*.5f+.5f,0.f,1.f)*255)));}}
 // RFC1950/1951 stored DEFLATE blocks: bounded native PNG transport, no model code.
 Bytes z{0x78,0x01};for(std::size_t at=0;at<raw.size();){auto n=std::min(std::size_t(65535),raw.size()-at);z.push_back(at+n==raw.size()?1:0);auto len=std::uint16_t(n),inv=std::uint16_t(~len);z.push_back(len&255);z.push_back(len>>8);z.push_back(inv&255);z.push_back(inv>>8);z.insert(z.end(),raw.begin()+at,raw.begin()+at+n);at+=n;}std::uint32_t a=1,b=0;for(auto v:raw){a=(a+v)%65521;b=(b+a)%65521;}be(z,(b<<16)|a);
 Bytes png{137,80,78,71,13,10,26,10},header;be(header,std::uint32_t(width));be(header,std::uint32_t(height));header.insert(header.end(),{8,6,0,0,0});chunk(png,"IHDR",header);chunk(png,"IDAT",z);chunk(png,"IEND",{});need(!cancel.load(),"image_cancelled");int fd=open(path.c_str(),O_CREAT|O_EXCL|O_WRONLY|O_CLOEXEC|O_NOFOLLOW,0600);need(fd>=0,"image_png_create");std::size_t at=0;while(at<png.size()){auto n=write(fd,png.data()+at,png.size()-at);if(n<=0||cancel.load()){close(fd);unlink(path.c_str());throw std::runtime_error("image_png_write");}at+=n;}if(close(fd)!=0){unlink(path.c_str());throw std::runtime_error("image_png_close");}
}
}
