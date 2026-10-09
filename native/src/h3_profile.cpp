#include "kadan/h3_profile.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace kadan::video {
namespace {
std::size_t nearest_even(double x){const double lo=std::floor(x),fraction=x-lo;auto n=std::size_t(lo);return n+(fraction>.5||(fraction==.5&&n%2));}
}
H3Profile h3_api_profile(std::size_t edge,std::string_view aspect,std::size_t seconds){
    if((edge!=480&&edge!=768)||seconds<4||seconds>15)throw std::invalid_argument("h3_api_profile");
    double ratio=1;if(aspect=="16:9")ratio=16.0/9;else if(aspect=="9:16")ratio=9.0/16;else if(aspect!="1:1")throw std::invalid_argument("h3_api_aspect");
    double width=edge*std::max(1.0,ratio),height=edge*std::max(1.0,1/ratio);
    if(width*height>768.0*1344){const double scale=std::sqrt((768.0*1344)/(width*height));width*=scale;height*=scale;}
    const auto w=std::max<std::size_t>(1,nearest_even(width/32))*32,h=std::max<std::size_t>(1,nearest_even(height/32))*32;
    const auto requested=seconds*24,frames=requested+(5+17-requested%17)%17;
    const auto t=(frames-5)/17*5+2,a=(frames*5)/3+((frames*5)%3>=2);
    return {w,h,frames,t,a,t*(h/32)*(w/32),2*a,7*(h/16)*(w/16)};
}
}
