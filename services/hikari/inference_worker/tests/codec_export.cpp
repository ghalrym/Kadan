#include "../src/preparation/codec_export.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

using namespace kadan::serving;

namespace {
std::atomic_bool running{false};

template <class F> void rejects(F action) {
    bool rejected = false;
    try {
        action();
    } catch (const std::exception&) {
        rejected = true;
    }
    assert(rejected);
}

std::string pickle_string(std::string_view value) {
    std::string result = "X";
    for (unsigned i = 0; i < 4; ++i)
        result += char(value.size() >> (8 * i));
    result += value;
    return result;
}

std::string pickle_integer(std::uint32_t value) {
    std::string result = "J";
    for (unsigned i = 0; i < 4; ++i)
        result += char(value >> (8 * i));
    return result;
}

void metadata_units() {
    std::string pickle = "\x80\x02"
                         "ccollections\nOrderedDict\n)R";
    pickle +=
        pickle_string("decoder.weight_v") + "ctorch._utils\n_rebuild_tensor_v2\n((";
    pickle += pickle_string("storage") + "ctorch\nFloatStorage\n";
    pickle += pickle_string("0") + pickle_string("cpu") + "K\x06tQ";
    pickle += std::string("K\0", 2) + "K\x02K\x03\x86K\x03K\x01\x86\x89";
    pickle += "ccollections\nOrderedDict\n)RtRs.";
    const auto metadata = read_checkpoint_pickle(pickle, running);
    const auto& tensor = metadata.at("decoder.weight_v");
    assert(tensor.at("dtype") == "F32");
    assert(tensor.at("shape") == Json::array({2, 3}));
    assert(tensor.at("stride") == Json::array({3, 1}));
    assert(tensor.at("storage") == "0" && tensor.at("elements") == 6);
    assert(tensor.at("offset") == 0);

    // The official codec also stores a large causal mask. Parse its descriptor
    // without allocating or reading its Boolean payload, retaining weight dtypes.
    auto with_mask = pickle.substr(0, pickle.size() - 1);
    with_mask += pickle_string("encoder.block.4.block.5.causal_mask")
                 + "ctorch._utils\n_rebuild_tensor_v2\n((";
    with_mask += pickle_string("storage") + "ctorch\nBoolStorage\n"
                 + pickle_string("116") + pickle_string("cpu")
                 + pickle_integer(268435456) + "tQ" + pickle_integer(0);
    with_mask += pickle_integer(16384) + pickle_integer(16384) + "\x86"
                 + pickle_integer(16384) + pickle_integer(1) + "\x86\x89"
                 + "ccollections\nOrderedDict\n)RtRs.";
    const auto mixed = read_checkpoint_pickle(with_mask, running);
    const auto& mask = mixed.at("encoder.block.4.block.5.causal_mask");
    assert(mask.at("dtype") == "BOOL");
    assert(mask.at("storage") == "116" && mask.at("elements") == 268435456);
    assert(mask.at("offset") == 0);
    assert(mask.at("shape") == Json::array({16384, 16384}));
    assert(mask.at("stride") == Json::array({16384, 1}));
    assert(mixed.at("decoder.weight_v") == tensor);
    // Recognizing a storage descriptor must not make the global callable.
    rejects([&] { read_checkpoint_pickle("ctorch\nBoolStorage\n)R.", running); });
    rejects([&] { read_checkpoint_pickle("ctorch\nIntStorage\n.", running); });
    rejects([&] {
        read_checkpoint_pickle(pickle.substr(0, pickle.size() - 1), running);
    });
    rejects([&] { read_checkpoint_pickle(pickle + ".", running); });
    rejects([&] {
        read_checkpoint_pickle(
            "\x80\x02"
            "cos\nsystem\n)R.",
            running
        );
    });
    // Even an unused unrecognized global is rejected, without invoking it.
    rejects([&] {
        read_checkpoint_pickle(
            "\x80\x02"
            "cos\nsystem\n.",
            running
        );
    });
    rejects([&] { read_checkpoint_pickle("\x80\x02h\x01.", running); });
    rejects([&] {
        read_checkpoint_pickle(std::string(4 * 1024 * 1024 + 1, 'N'), running);
    });
    // Memoized data may be copied, but a tiny recursively growing descriptor
    // cannot evade the total-copy or nesting bounds.
    std::string amplified = std::string("\x80\x02]q\0", 5);
    for (unsigned i = 0; i < 40; ++i)
        amplified += std::string("h\0a", 3);
    amplified += '.';
    rejects([&] { read_checkpoint_pickle(amplified, running); });
    std::atomic_bool cancelled{true};
    rejects([&] { read_checkpoint_pickle(pickle, cancelled); });
}

void normalization_units() {
    std::array<float, 2> row{3, 4};
    fold_codec_weight_norm(row, 2);
    assert(std::abs(row[0] - 1.2f) < 1e-6f && std::abs(row[1] - 1.6f) < 1e-6f);
    row = {3, 4};
    fold_codec_weight_norm(row, -5);
    assert(row[0] == -3 && row[1] == -4);
    row = {3, 4};
    fold_codec_weight_norm(row, 0);
    assert(row[0] == 0 && row[1] == 0);
    row = {std::numeric_limits<float>::denorm_min(), 0};
    fold_codec_weight_norm(row, 1);
    assert(row[0] == 1 && row[1] == 0);
    row = {0, 0};
    rejects([&] { fold_codec_weight_norm(row, 1); });
    row = {1, std::numeric_limits<float>::infinity()};
    rejects([&] { fold_codec_weight_norm(row, 1); });
    row = {1, 1};
    rejects([&] {
        fold_codec_weight_norm(row, std::numeric_limits<float>::quiet_NaN());
    });
    rejects([&] { fold_codec_weight_norm({}, 1); });
}
}  // namespace

int main() {
    metadata_units();
    normalization_units();
    std::cout << "Codec preparation metadata and weight-normalization units passed; no "
                 "model execution.\n";
}
