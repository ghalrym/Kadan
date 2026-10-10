#pragma once
#include <exception>
#include <stdexcept>
#include <string>
namespace kadan {
inline std::string exception_message(const std::exception_ptr& error){
    try{std::rethrow_exception(error);}catch(const std::exception& e){return e.what();}catch(...){return "non-standard exception";}
}
class OperationCleanupError final:public std::runtime_error {
public:
    const std::exception_ptr operation,cleanup;
    OperationCleanupError(std::exception_ptr original,std::exception_ptr failure):
        std::runtime_error("operation: "+exception_message(original)+"; cleanup: "+exception_message(failure)),operation(original),cleanup(failure){}
};
template<class Cleanup> [[noreturn]] void rethrow_after_cleanup(std::exception_ptr original,Cleanup cleanup){
    try{cleanup();}catch(...){throw OperationCleanupError(original,std::current_exception());}
    std::rethrow_exception(original);
}
}
