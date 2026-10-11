#include "kadan/image_schedule.hpp"
#include <iomanip>
#include <iostream>
int main(int argc,char** argv){try{if(argc!=3)throw std::runtime_error("usage: kadan-image-schedule TOKENS STEPS");auto s=kadan::image::schedule(std::stoul(argv[1]),std::stoul(argv[2]));std::cout<<std::setprecision(9)<<'[';for(std::size_t i=0;i<=s.steps;++i)std::cout<<(i?",":"")<<s.sigma[i];std::cout<<"]\n";}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
