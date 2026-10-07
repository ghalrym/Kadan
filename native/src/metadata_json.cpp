#include "metadata_json.hpp"
#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <fcntl.h>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace kadan::checkpoint::json {
namespace {
void require(bool ok, const char* message) { if (!ok) throw std::invalid_argument(message); }
struct Fd {
    int value;
    explicit Fd(int fd) : value(fd) { if (fd < 0) throw std::runtime_error("metadata_open"); }
    ~Fd() { ::close(value); }
    Fd(const Fd&) = delete;
};
class Parser {
    std::string_view input;
    std::pmr::memory_resource* resource;
    std::size_t pos = 0, nodes = 0;
    void ws() { while (pos < input.size() && (input[pos]==' ' || input[pos]=='\n' || input[pos]=='\r' || input[pos]=='\t')) ++pos; }
    bool take(char c) { ws(); if (pos == input.size() || input[pos]!=c) return false; ++pos; return true; }
    void expect(char c) { require(take(c), "metadata_json_syntax"); }
    unsigned hex() {
        require(pos < input.size(), "metadata_json_escape"); const char c=input[pos++];
        if (c >= '0' && c <= '9') return c-'0';
        if (c >= 'a' && c <= 'f') return c-'a'+10;
        if (c >= 'A' && c <= 'F') return c-'A'+10;
        throw std::invalid_argument("metadata_json_escape");
    }
    std::pmr::string string() {
        expect('"'); std::pmr::string out(resource);
        while (pos < input.size()) {
            unsigned char c=input[pos++]; if (c=='"') return out;
            require(c >= 32 && c < 128, "metadata_ascii");
            if (c=='\\') {
                require(pos < input.size(), "metadata_json_escape"); c=input[pos++];
                switch (c) {
                case '"': case '\\': case '/': break;
                case 'b': c=8; break; case 'f': c=12; break; case 'n': c=10; break;
                case 'r': c=13; break; case 't': c=9; break;
                case 'u': { unsigned n=0; for (int i=0;i<4;++i) n=n*16+hex(); require(n && n<128,"metadata_ascii"); c=n; break; }
                default: throw std::invalid_argument("metadata_json_escape");
                }
            }
            require(out.size()<512,"metadata_string_limit"); out+=static_cast<char>(c);
        }
        throw std::invalid_argument("metadata_json_string");
    }
    Value value(unsigned depth) {
        require(depth<=32 && ++nodes<=262144,"metadata_structure_limit"); ws();
        require(pos<input.size(),"metadata_json_syntax");
        if (input[pos]=='{' || input[pos]=='[') {
            const bool object=input[pos++]=='{'; const char close=object?'}':']';
            Value out(object?Value::Kind::object:Value::Kind::array,resource);
            if (!take(close)) {
                do {
                    std::pmr::string name(resource);
                    if (object) { name=string(); expect(':'); }
                    auto child=value(depth+1); child.name=std::move(name); out.children.push_back(std::move(child));
                } while (take(',')); expect(close);
            }
            if (object) {
                std::sort(out.children.begin(),out.children.end(),[](const auto& a,const auto& b){return a.name<b.name;});
                for (std::size_t i=1;i<out.children.size();++i)
                    require(out.children[i-1].name!=out.children[i].name,"metadata_duplicate_key");
            }
            return out;
        }
        if (input[pos]=='"') { Value out(Value::Kind::string,resource); out.text=string(); return out; }
        for (const auto literal : {std::string_view("true"),std::string_view("false"),std::string_view("null")})
            if (input.substr(pos,literal.size())==literal) {
                pos+=literal.size(); Value out(literal=="null"?Value::Kind::null:Value::Kind::boolean,resource); out.text=literal; return out;
            }
        Value out(Value::Kind::number,resource); const auto start=pos;
        if (input[pos]=='-') ++pos;
        require(pos<input.size() && input[pos]>='0' && input[pos]<='9',"metadata_number");
        if (input[pos]=='0') ++pos; else while (pos<input.size() && input[pos]>='0' && input[pos]<='9') ++pos;
        const auto digits=[&] { const auto begin=pos; while (pos<input.size() && input[pos]>='0' && input[pos]<='9') ++pos; require(pos>begin,"metadata_number"); };
        if (pos<input.size() && input[pos]=='.') { ++pos; digits(); }
        if (pos<input.size() && (input[pos]=='e'||input[pos]=='E')) {
            ++pos; if (pos<input.size() && (input[pos]=='+'||input[pos]=='-')) ++pos; digits();
        }
        require(pos-start<=64,"metadata_number"); out.text=input.substr(start,pos-start); out.number(); return out;
    }
public:
    Parser(std::string_view text,std::pmr::memory_resource* r):input(text),resource(r) {}
    Value parse() { auto out=value(0); ws(); require(pos==input.size(),"metadata_json_trailing"); return out; }
};
}
const Value* Value::find(std::string_view key) const {
    require(kind==Kind::object,"metadata_expected_object");
    const auto it=std::lower_bound(children.begin(),children.end(),key,[](const auto& v,auto k){return std::string_view(v.name)<k;});
    return it!=children.end() && it->name==key?&*it:nullptr;
}
const Value& Value::at(std::string_view key) const { const auto* v=find(key); require(v,"metadata_missing_field"); return *v; }
std::string_view Value::string() const { require(kind==Kind::string,"metadata_expected_string"); return text; }
std::uint64_t Value::integer() const {
    require(kind==Kind::number,"metadata_expected_integer"); std::uint64_t out=0;
    const auto [end,error]=std::from_chars(text.data(),text.data()+text.size(),out);
    require(error==std::errc{} && end==text.data()+text.size(),"metadata_expected_integer"); return out;
}
double Value::number() const {
    require(kind==Kind::number,"metadata_expected_number"); double out=0;
    const auto [end,error]=std::from_chars(text.data(),text.data()+text.size(),out);
    require(error==std::errc{} && end==text.data()+text.size() && std::isfinite(out),"metadata_number"); return out;
}
bool Value::boolean() const { require(kind==Kind::boolean,"metadata_expected_boolean"); return text=="true"; }
Value read(const char* root,std::string_view name,std::size_t limit,const std::shared_ptr<MemoryBudget>& budget) {
    require(budget && limit && !name.empty() && name.size()<=255 && name!="." && name!="..","metadata_limits_or_name");
    for (unsigned char c:name) require((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='.'||c=='_'||c=='-',"metadata_basename");
    std::array<char,256> path{}; std::copy(name.begin(),name.end(),path.begin());
    Fd directory(::open(root,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC));
    Fd file(::openat(directory.value,path.data(),O_RDONLY|O_NONBLOCK|O_NOFOLLOW|O_CLOEXEC));
    struct stat before{}; require(::fstat(file.value,&before)==0 && S_ISREG(before.st_mode) && before.st_size>0 &&
        static_cast<std::uint64_t>(before.st_size)<=limit,"metadata_file_size");
    std::pmr::string text(budget.get()); text.resize(before.st_size);
    std::size_t offset=0;
    while (offset<text.size()) {
        auto n=::pread(file.value,text.data()+offset,std::min(text.size()-offset,std::size_t{1024*1024}),offset);
        if (n<0 && errno==EINTR) continue;
        if (n<=0) throw std::runtime_error("metadata_short_read");
        offset+=n;
    }
    struct stat after{}; require(::fstat(file.value,&after)==0 && before.st_size==after.st_size &&
        before.st_mtim.tv_sec==after.st_mtim.tv_sec && before.st_mtim.tv_nsec==after.st_mtim.tv_nsec &&
        before.st_ctim.tv_sec==after.st_ctim.tv_sec && before.st_ctim.tv_nsec==after.st_ctim.tv_nsec,"metadata_changed");
    return Parser(text,budget.get()).parse();
}
} // namespace kadan::checkpoint::json
