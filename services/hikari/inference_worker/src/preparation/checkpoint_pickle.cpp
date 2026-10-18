#include "checkpoint_pickle.hpp"

#include <memory>
#include <unordered_map>

namespace kadan::serving {
namespace {
// A data-only reader for the verified torch.save format. GLOBAL/REDUCE
// accept only storage/tensor descriptors and OrderedDict; nothing is executed.
class CheckpointPickle {
public:
    CheckpointPickle(std::string bytes, const std::atomic_bool& cancel)
        : cancel_(cancel), bytes_(std::move(bytes)) {
        preparation_check(
            bytes_.size() <= 4 * 1024 * 1024, "Checkpoint pickle size bound"
        );
    }

    Json read() {
        std::size_t operations = 0;
        while (at_ < bytes_.size()) {
            if (operations % 1024 == 0)
                preparation_cancel(cancel_);
            preparation_check(
                ++operations <= 1000000 && stack_.size() <= 100000,
                "Checkpoint pickle bound"
            );
            switch (byte()) {
            case 0x80:
                preparation_check(byte() <= 5, "Unsupported pickle protocol");
                break;
            case 0x95:
                integer(8);
                break;  // Frame length; byte bounds still apply.
            case '}':
                push(Json::object());
                break;
            case ']':
            case ')':
                push(Json::array());
                break;
            case '(':
                marks_.push_back(stack_.size());
                break;
            case 'N':
                push(nullptr);
                break;
            case 0x88:
                push(true);
                break;
            case 0x89:
                push(false);
                break;
            case 'K':
                push(integer(1));
                break;
            case 'M':
                push(integer(2));
                break;
            case 'J':
                push(std::int32_t(integer(4)));
                break;
            case 0x8a: {
                const auto length = byte();
                preparation_check(length <= 8, "Pickle integer bound");
                push(integer(length));
                break;
            }
            case 'X':
                push(text(integer(4)));
                break;
            case 0x8c:
            case 'U':
                push(text(byte()));
                break;
            case 'T':
                push(text(integer(4)));
                break;
            case 'q':
                memo(integer(1));
                break;
            case 'r':
                memo(integer(4));
                break;
            case 0x94:
                memo(memo_.size());
                break;
            case 'h':
                get(integer(1));
                break;
            case 'j':
                get(integer(4));
                break;
            case 't':
                tuple(mark());
                break;
            case 0x85:
                tuple(stack_.size() - 1);
                break;
            case 0x86:
                preparation_check(stack_.size() >= 2, "Invalid pickle tuple");
                tuple(stack_.size() - 2);
                break;
            case 0x87:
                preparation_check(stack_.size() >= 3, "Invalid pickle tuple");
                tuple(stack_.size() - 3);
                break;
            case 'u':
                setitems(mark());
                break;
            case 's':
                preparation_check(stack_.size() >= 3, "Invalid pickle item");
                setitems(stack_.size() - 2);
                break;
            case 'e':
                append(mark());
                break;
            case 'a':
                preparation_check(stack_.size() >= 2, "Invalid pickle append");
                append(stack_.size() - 1);
                break;
            case 'c': {
                const auto module = line(), name = line();
                global(module + "." + name);
                break;
            }
            case 0x93: {
                auto name = pop(), module = pop();
                global(module->get<std::string>() + "." + name->get<std::string>());
                break;
            }
            case 'Q':
                persistent();
                break;
            case 'R':
                reduce();
                break;
            case 'b': {
                const auto attributes = pop();
                preparation_check(
                    !stack_.empty() && attributes->is_object(),
                    "Unsupported pickle attributes"
                );
                // OrderedDict's _metadata describes module versions, not weights.
                for (auto it = attributes->begin(); it != attributes->end(); ++it)
                    preparation_check(
                        it.key() == "_metadata", "Unsupported pickle attribute"
                    );
                break;
            }
            case '.':
                preparation_check(
                    stack_.size() == 1 && marks_.empty() && at_ == bytes_.size(),
                    "Invalid pickle termination"
                );
                return std::move(*stack_.back());
            default:
                throw RequestError("Unsupported checkpoint pickle opcode");
            }
        }
        throw RequestError("Truncated checkpoint pickle");
    }

private:
    unsigned char byte() {
        preparation_check(at_ < bytes_.size(), "Truncated pickle");
        return static_cast<unsigned char>(bytes_[at_++]);
    }

    std::uint64_t integer(std::size_t size) {
        std::uint64_t result = 0;
        for (std::size_t i = 0; i < size; ++i)
            result |= std::uint64_t(byte()) << (i * 8);
        return result;
    }

    std::string text(std::size_t size) {
        preparation_check(
            size <= 4 * 1024 * 1024 && size <= bytes_.size() - at_,
            "Pickle string bound"
        );
        auto value = bytes_.substr(at_, size);
        at_ += size;
        return value;
    }

    std::string line() {
        const auto end = bytes_.find('\n', at_);
        preparation_check(
            end != std::string::npos && end - at_ <= 256, "Invalid pickle global"
        );
        auto value = text(end - at_);
        ++at_;
        return value;
    }

    // Account every deep copy before it occurs. Memo references cannot amplify
    // a short pickle into an unbounded tree, and nesting cannot exhaust the stack.
    void charge(const Json& value, unsigned depth = 0) {
        preparation_check(depth <= 32, "Checkpoint pickle nesting bound");
        std::size_t amount = 32;
        if (value.is_string())
            amount += value.get_ref<const std::string&>().size();
        preparation_check(
            amount <= 64 * 1024 * 1024 - copied_, "Checkpoint pickle copy bound"
        );
        copied_ += amount;
        if (value.is_structured())
            for (auto it = value.begin(); it != value.end(); ++it) {
                if (value.is_object()) {
                    preparation_check(
                        it.key().size() <= 64 * 1024 * 1024 - copied_,
                        "Checkpoint pickle key bound"
                    );
                    copied_ += it.key().size();
                }
                charge(it.value(), depth + 1);
            }
    }

    void global(const std::string& name) {
        preparation_check(
            name == "collections.OrderedDict" || name == "torch.FloatStorage"
                || name == "torch.HalfStorage" || name == "torch.BFloat16Storage"
                || name == "torch._utils._rebuild_tensor_v2"
                || name == "torch._utils._rebuild_tensor"
                || name == "torch._utils._rebuild_parameter",
            "Unsupported checkpoint pickle global: " + name
        );
        push({{"global", name}});
    }

    void push(Json value) {
        charge(value);
        stack_.push_back(std::make_shared<Json>(std::move(value)));
    }

    std::shared_ptr<Json> pop() {
        preparation_check(!stack_.empty(), "Empty pickle stack");
        auto value = stack_.back();
        stack_.pop_back();
        return value;
    }

    std::size_t mark() {
        preparation_check(!marks_.empty(), "Missing pickle mark");
        auto at = marks_.back();
        marks_.pop_back();
        return at;
    }

    void memo(std::size_t index) {
        preparation_check(index < 100000 && !stack_.empty(), "Pickle memo bound");
        memo_[index] = stack_.back();
    }

    void get(std::size_t index) {
        preparation_check(memo_.contains(index), "Invalid pickle memo");
        stack_.push_back(memo_.at(index));
    }

    void tuple(std::size_t start) {
        preparation_check(start <= stack_.size(), "Invalid pickle tuple");
        Json value = Json::array();
        for (std::size_t i = start; i < stack_.size(); ++i) {
            charge(*stack_[i]);
            value.push_back(*stack_[i]);
        }
        stack_.resize(start);
        push(std::move(value));
    }

    void setitems(std::size_t start) {
        preparation_check(
            start > 0 && start <= stack_.size() && (stack_.size() - start) % 2 == 0
                && stack_[start - 1]->is_object(),
            "Invalid pickle dictionary"
        );
        auto& object = *stack_[start - 1];
        for (std::size_t i = start; i < stack_.size(); i += 2) {
            charge(*stack_[i + 1]);
            object[stack_[i]->get<std::string>()] = *stack_[i + 1];
        }
        stack_.resize(start);
    }

    void append(std::size_t start) {
        preparation_check(
            start > 0 && start <= stack_.size() && stack_[start - 1]->is_array(),
            "Invalid pickle list"
        );
        for (std::size_t i = start; i < stack_.size(); ++i) {
            charge(*stack_[i]);
            stack_[start - 1]->push_back(*stack_[i]);
        }
        stack_.resize(start);
    }

    void persistent() {
        const auto record_owner = pop();
        const auto& record = *record_owner;
        preparation_check(
            record.is_array() && record.size() == 5 && record[0] == "storage"
                && record[4].is_number_integer() && record[4] >= 0,
            "Unsupported pickle persistent object"
        );
        const auto global = record[1].at("global").get<std::string>();
        std::string dtype;
        if (global == "torch.FloatStorage")
            dtype = "F32";
        else if (global == "torch.HalfStorage")
            dtype = "F16";
        else if (global == "torch.BFloat16Storage")
            dtype = "BF16";
        else
            throw RequestError("Unsupported checkpoint storage dtype");
        const auto key = record[2].get<std::string>();
        preparation_check(
            !key.empty() && key.size() < 128
                && key.find_first_not_of("0123456789") == std::string::npos,
            "Invalid checkpoint storage key"
        );
        push({{"storage", key}, {"dtype", dtype}, {"elements", record[4]}});
    }

    void reduce() {
        const auto args_owner = pop(), callable_owner = pop();
        const auto& args = *args_owner;
        const auto& callable = *callable_owner;
        const auto global = callable.at("global").get<std::string>();
        preparation_check(args.is_array(), "Invalid pickle reducer arguments");
        if (global == "collections.OrderedDict") {
            preparation_check(args.empty(), "Unsupported OrderedDict initializer");
            push(Json::object());
        } else if (
            global == "torch._utils._rebuild_tensor_v2"
            || global == "torch._utils._rebuild_tensor"
        ) {
            preparation_check(
                args.size() >= 4 && args.size() <= 7 && args[0].contains("storage"),
                "Invalid tensor descriptor"
            );
            preparation_check(
                args[1].is_number_integer() && args[1] >= 0 && args[2].is_array()
                    && args[2].size() <= 8 && args[3].is_array()
                    && args[3].size() == args[2].size(),
                "Invalid tensor descriptor dimensions"
            );
            for (const auto field : {2u, 3u})
                for (const auto& dimension : args[field])
                    preparation_check(
                        dimension.is_number_integer() && dimension >= 0,
                        "Invalid tensor descriptor dimension"
                    );
            charge(args[0]);
            charge(args[1]);
            charge(args[2]);
            charge(args[3]);
            Json tensor = args[0];
            tensor["offset"] = args[1];
            tensor["shape"] = args[2];
            tensor["stride"] = args[3];
            push(std::move(tensor));
        } else if (global == "torch._utils._rebuild_parameter") {
            preparation_check(
                args.size() == 3 && args[0].contains("storage"),
                "Invalid parameter descriptor"
            );
            charge(args[0]);
            push(args[0]);
        } else
            throw RequestError("Unsupported checkpoint pickle global: " + global);
    }

    const std::atomic_bool& cancel_;
    std::string bytes_;
    std::size_t at_ = 0, copied_ = 0;
    std::vector<std::shared_ptr<Json>> stack_;
    std::vector<std::size_t> marks_;
    std::unordered_map<std::size_t, std::shared_ptr<Json>> memo_;
};

}  // namespace

Json read_checkpoint_pickle(std::string bytes, const std::atomic_bool& cancel) {
    return CheckpointPickle(std::move(bytes), cancel).read();
}
}  // namespace kadan::serving
