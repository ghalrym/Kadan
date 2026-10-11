#include <array>
#include <bit>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <unordered_map>

#include "archive_reader.hpp"
#include "kadan/preparation.hpp"

namespace kadan::serving {
namespace {
void number(std::ofstream& output, std::uint32_t value) {
    static_assert(std::endian::native == std::endian::little);
    output.write(reinterpret_cast<const char*>(&value), 4);
}

void numbers(std::ofstream& output, const std::vector<std::uint32_t>& values) {
    number(output, values.size());
    for (auto value : values)
        number(output, value);
}

std::vector<std::string> characters(std::string_view input) {
    std::vector<std::string> result;
    for (std::size_t i = 0; i < input.size();) {
        const auto start = i++;
        while (i < input.size()
               && (static_cast<unsigned char>(input[i]) & 0xc0) == 0x80)
            ++i;
        result.emplace_back(input.substr(start, i - start));
    }
    return result;
}

class Ranks {
public:
    explicit Ranks(const fs::path& file) {
        std::istringstream input(read_bytes(file, 4 * 1024 * 1024));
        std::string token;
        std::uint32_t rank;
        std::set<std::uint32_t> assigned;
        while (input >> token >> rank) {
            preparation_check(rank < 52000, "Whisper token rank bound");
            // The pinned multilingual ranks include an empty byte token at 50256.
            const auto raw = token == "=" ? std::string() : base64_decode(token);
            preparation_check(
                (token != "=" || rank == 50256) && raw.size() <= 128
                    && ranks_.emplace(raw, rank).second,
                "Invalid Whisper token rank"
            );
            if (pieces.size() <= rank)
                pieces.resize(rank + 1);
            preparation_check(
                assigned.insert(rank).second, "Duplicate Whisper token rank"
            );
            pieces[rank] = raw;
        }
        preparation_check(
            input.eof() && !pieces.empty(), "Invalid Whisper token ranks"
        );
        preparation_check(
            assigned.size() == pieces.size(), "Missing Whisper token rank"
        );
    }

    std::vector<std::uint32_t> encode(const std::string& text) const {
        if (const auto token = ranks_.find(text); token != ranks_.end())
            return {token->second};
        std::vector<std::string> parts;
        for (char c : text)
            parts.emplace_back(1, c);
        while (parts.size() > 1) {
            std::size_t best = parts.size();
            std::uint32_t rank = UINT32_MAX;
            for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
                const auto found = ranks_.find(parts[i] + parts[i + 1]);
                if (found != ranks_.end() && found->second < rank) {
                    best = i;
                    rank = found->second;
                }
            }
            if (best == parts.size())
                break;
            parts[best] += parts[best + 1];
            parts.erase(parts.begin() + static_cast<std::ptrdiff_t>(best + 1));
        }
        std::vector<std::uint32_t> result;
        result.reserve(parts.size());
        for (const auto& part : parts)
            result.push_back(ranks_.at(part));
        return result;
    }

    std::vector<std::string> pieces;

private:
    std::unordered_map<std::string, std::uint32_t> ranks_;
};

void mel_filters(
    const fs::path& archive_path,
    const fs::path& destination,
    const std::atomic_bool& cancel
) {
    ArchiveReader archive(archive_path);
    std::set<unsigned> found;
    while (archive.next()) {
        unsigned bins = 0;
        if (archive.name() == "mel_80.npy")
            bins = 80;
        if (archive.name() == "mel_128.npy")
            bins = 128;
        if (!bins)
            continue;
        preparation_check(found.insert(bins).second, "Duplicate mel filters");
        const auto bytes = archive.read(1024 * 1024, cancel);
        preparation_check(
            bytes.size() > 12 && bytes.substr(0, 6) == std::string("\x93NUMPY", 6),
            "Invalid mel filter array"
        );
        const auto version = static_cast<unsigned char>(bytes[6]);
        preparation_check(
            version == 1 || version == 2, "Unsupported mel filter array version"
        );
        const std::size_t start = version == 1 ? 10 : 12;
        std::size_t header_size = 0;
        for (std::size_t i = 8; i < start; ++i)
            header_size |= std::size_t(static_cast<unsigned char>(bytes[i]))
                           << ((i - 8) * 8);
        preparation_check(
            header_size < bytes.size() - start, "Invalid mel filter header"
        );
        const auto header = bytes.substr(start, header_size);
        std::string shape = "(";
        shape += std::to_string(bins);
        shape += ", 201)";
        preparation_check(
            header.find("'<f4'") != std::string::npos
                && header.find("'fortran_order': False") != std::string::npos
                && header.find(shape) != std::string::npos
                && bytes.size() - start - header_size == bins * 201 * 4,
            "Invalid mel filter shape or dtype"
        );
        std::ofstream output(
            destination / ("mel-" + std::to_string(bins) + ".f32"), std::ios::binary
        );
        output.write(
            bytes.data() + start + header_size, bytes.size() - start - header_size
        );
        output.close();
        preparation_check(!output.fail(), "Cannot write mel filters");
    }
    preparation_check(found.size() == 2, "Missing mel filters");
}
}  // namespace

void export_whisper_assets(
    std::size_t vocabulary,
    const fs::path& data,
    const fs::path& destination,
    const std::atomic_bool& cancel
) {
    preparation_check(
        vocabulary == 51864 || vocabulary == 51865 || vocabulary == 51866,
        "Unsupported Whisper vocabulary"
    );
    preparation_check(!fs::exists(destination), "Whisper asset destination exists");
    const auto manifest = read_document(data / "manifest.json");
    for (const auto* name :
         {"gpt2.tiktoken",
          "multilingual.tiktoken",
          "mel_filters.npz",
          "whisper_languages.json"})
        preparation_check(
            file_digest(data / name, cancel) == manifest.at(name).get<std::string>(),
            "Whisper asset hash mismatch"
        );
    const bool multi = vocabulary >= 51865;
    const auto language_count = vocabulary - 51765 - unsigned(multi);
    Ranks ranks(data / (multi ? "multilingual.tiktoken" : "gpt2.tiktoken"));
    const auto eos = std::uint32_t(ranks.pieces.size()), sot = eos + 1;
    const auto languages = read_document(data / "whisper_languages.json");
    preparation_check(
        language_count <= languages.size(), "Whisper language count mismatch"
    );
    ranks.pieces.emplace_back("<|endoftext|>");
    ranks.pieces.emplace_back("<|startoftranscript|>");
    for (std::size_t i = 0; i < language_count; ++i)
        ranks.pieces.push_back("<|" + languages[i].get<std::string>() + "|>");
    const auto transcribe = std::uint32_t(ranks.pieces.size() + 1),
               no_timestamps = std::uint32_t(ranks.pieces.size() + 5);
    for (const auto* text :
         {"<|translate|>",
          "<|transcribe|>",
          "<|startoflm|>",
          "<|startofprev|>",
          "<|nospeech|>",
          "<|notimestamps|>"})
        ranks.pieces.emplace_back(text);
    for (unsigned i = 0; i <= 1500; ++i) {
        std::ostringstream stamp;
        stamp << "<|" << i / 50 << '.' << std::setw(2) << std::setfill('0')
              << (i % 50) * 2 << "|>";
        ranks.pieces.push_back(stamp.str());
    }
    preparation_check(ranks.pieces.size() == vocabulary, "Whisper vocabulary mismatch");
    std::set<std::uint32_t> suppressed;
    auto symbols = characters("\"#()*+/:;<=>@[\\]^_`{|}~「」『』");
    std::istringstream groups(
        "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪"
    );
    std::string group;
    while (groups >> group)
        symbols.push_back(group);
    const auto musical = characters("♩♪♫♬♭♮♯");
    for (const auto& symbol : musical)
        symbols.push_back(symbol);
    suppressed.insert(ranks.encode(" -").front());
    suppressed.insert(ranks.encode(" '").front());
    for (const auto& symbol : symbols)
        for (const auto& text : {symbol, " " + symbol}) {
            const auto tokens = ranks.encode(text);
            if (tokens.size() == 1
                || std::find(musical.begin(), musical.end(), symbol) != musical.end())
                suppressed.insert(tokens.front());
        }
    for (auto id = eos + 1; id < vocabulary; ++id)
        suppressed.insert(id);
    std::vector<std::uint32_t> prompt = {sot};
    if (multi) {
        prompt.push_back(eos + 2);
        prompt.push_back(transcribe);
    }
    prompt.push_back(no_timestamps);
    auto first = ranks.encode(" ");
    first.push_back(eos);
    fs::create_directories(destination.parent_path());
    const auto stage = fs::path(destination.string() + ".partial");
    fs::remove_all(stage);
    fs::create_directory(stage);
    try {
        std::ofstream output(stage / "english.tokens", std::ios::binary);
        output << "KDWVOC01";
        number(output, vocabulary);
        number(output, eos);
        numbers(output, prompt);
        numbers(output, {suppressed.begin(), suppressed.end()});
        numbers(output, first);
        for (const auto& piece : ranks.pieces) {
            preparation_cancel(cancel);
            number(output, piece.size());
            output.write(piece.data(), piece.size());
        }
        output.close();
        preparation_check(!output.fail(), "Cannot write Whisper tokens");
        mel_filters(data / "mel_filters.npz", stage, cancel);
        write_document(
            stage / "assets.json",
            {{"language", "en"},
             {"whisper_version", "20250625"},
             {"vocabulary", vocabulary},
             {"prompt", prompt},
             {"eos", eos}}
        );
        preparation_cancel(cancel);
        fs::rename(stage, destination);
    } catch (...) {
        fs::remove_all(stage);
        throw;
    }
}

void export_speech_tokenizer(
    const fs::path& checkpoint,
    const fs::path& destination,
    const std::atomic_bool& cancel
) {
    preparation_cancel(cancel);
    preparation_check(!fs::exists(destination), "Speech tokenizer destination exists");
    const auto vocab = read_document(checkpoint / "vocab.json"),
               config = read_document(checkpoint / "tokenizer_config.json");
    Json merges = Json::array();
    std::istringstream input(read_bytes(checkpoint / "merges.txt", 32 * 1024 * 1024));
    std::string line;
    while (std::getline(input, line)) {
        preparation_cancel(cancel);
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.empty() || line.starts_with("#version:"))
            continue;
        const auto split = line.find(' ');
        preparation_check(
            split != std::string::npos
                && line.find(' ', split + 1) == std::string::npos,
            "Invalid speech tokenizer merge"
        );
        merges.push_back({line.substr(0, split), line.substr(split + 1)});
    }
    Json added = Json::array();
    for (auto it = config.at("added_tokens_decoder").begin();
         it != config.at("added_tokens_decoder").end();
         ++it) {
        preparation_check(
            !it.key().empty()
                && it.key().find_first_not_of("0123456789") == std::string::npos,
            "Invalid speech token ID"
        );
        auto token = it.value();
        token["id"] = std::stoul(it.key());
        added.push_back(token);
    }
    const std::string pattern =
        R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    Json tokenizer = {
        {"version", "1.0"},
        {"truncation", nullptr},
        {"padding", nullptr},
        {"added_tokens", added},
        {"normalizer", {{"type", "NFC"}}},
        {"pre_tokenizer",
         {{"type", "Sequence"},
          {"pretokenizers",
           Json::array(
               {{{"type", "Split"},
                 {"pattern", {{"Regex", pattern}}},
                 {"behavior", "Isolated"},
                 {"invert", false}},
                {{"type", "ByteLevel"},
                 {"add_prefix_space", false},
                 {"trim_offsets", true},
                 {"use_regex", false}}}
           )}}},
        {"post_processor",
         {{"type", "TemplateProcessing"},
          {"single", Json::array({{{"Sequence", {{"id", "A"}, {"type_id", 0}}}}})},
          {"pair",
           Json::array(
               {{{"Sequence", {{"id", "A"}, {"type_id", 0}}}},
                {{"Sequence", {{"id", "B"}, {"type_id", 1}}}}}
           )},
          {"special_tokens", Json::object()}}},
        {"decoder",
         {{"type", "ByteLevel"},
          {"add_prefix_space", true},
          {"trim_offsets", true},
          {"use_regex", true}}},
        {"model",
         {{"type", "BPE"},
          {"dropout", nullptr},
          {"unk_token", nullptr},
          {"continuing_subword_prefix", ""},
          {"end_of_word_suffix", ""},
          {"fuse_unk", false},
          {"byte_fallback", false},
          {"ignore_merges", false},
          {"vocab", vocab},
          {"merges", merges}}}
    };
    fs::create_directories(destination.parent_path());
    const auto stage = fs::path(destination.string() + ".partial");
    fs::remove_all(stage);
    fs::create_directory(stage);
    try {
        write_document(stage / "tokenizer.json", tokenizer);
        preparation_cancel(cancel);
        fs::rename(stage, destination);
    } catch (...) {
        fs::remove_all(stage);
        throw;
    }
}
}  // namespace kadan::serving
