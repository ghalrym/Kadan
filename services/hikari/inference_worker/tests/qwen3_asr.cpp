#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "kadan/qwen3_asr.hpp"
#include "kadan/stt.hpp"

namespace {
void test_slaney_mel_filters() {
    const auto filters = kadan::stt::slaney_mel_filters(128, 16000, 400);
    assert(filters.size() == 128 * 201);
    double total = 0.0;
    for (float v : filters) {
        assert(std::isfinite(v) && v >= 0.0f);
        total += v;
    }
    assert(total > 0.0);
    // Every mel band should have non-zero energy across FFT bins.
    for (std::size_t m = 0; m < 128; ++m) {
        double band = 0.0;
        for (std::size_t k = 0; k < 201; ++k)
            band += filters[m * 201 + k];
        assert(band > 0.0);
    }
}

void test_feat_extract_output_length() {
    assert(kadan::stt::feat_extract_output_length(0) == 0);
    assert(kadan::stt::feat_extract_output_length(1) == 1);
    assert(kadan::stt::feat_extract_output_length(50) == 7);
    assert(kadan::stt::feat_extract_output_length(100) == 13);
    assert(kadan::stt::feat_extract_output_length(101) == 14);
    assert(kadan::stt::feat_extract_output_length(200) == 26);
    assert(kadan::stt::feat_extract_output_length(3000) == 390);
}

void test_sinusoids() {
    const auto emb = kadan::stt::sinusoids_position_embedding(13, 1024);
    assert(emb.size() == 13 * 1024);
    // At pos 0: sin(0) == 0 for first half, cos(0) == 1 for second half.
    for (std::size_t i = 0; i < 512; ++i) {
        assert(std::fabs(emb[i]) < 1e-6f);
        assert(std::fabs(emb[512 + i] - 1.0f) < 1e-6f);
    }
}

void test_language_normalization() {
    assert(kadan::stt::normalize_language_hint("en") == "English");
    assert(kadan::stt::normalize_language_hint("ENGLISH") == "English");
    assert(kadan::stt::normalize_language_hint("zh") == "Chinese");
    assert(kadan::stt::normalize_language_hint("ja") == "Japanese");
    assert(!kadan::stt::normalize_language_hint("").has_value());
    assert(!kadan::stt::normalize_language_hint("auto").has_value());
    assert(!kadan::stt::normalize_language_hint("klingon").has_value());
}

void test_parse_asr_output() {
    {
        const auto r =
            kadan::stt::parse_asr_output("language English<asr_text>Hello, world.");
        assert(r.language == "English");
        assert(r.text == "Hello, world.");
    }
    {
        const auto r = kadan::stt::parse_asr_output("language None<asr_text>");
        assert(r.language.empty());
        assert(r.text.empty());
    }
    {
        const auto r =
            kadan::stt::parse_asr_output("Direct forced output.", "English");
        assert(r.language == "English");
        assert(r.text == "Direct forced output.");
    }
    {
        const auto r = kadan::stt::parse_asr_output("Plain text without tag");
        assert(r.language.empty());
        assert(r.text == "Plain text without tag");
    }
    {
        std::string repeated = "hi ";
        for (int i = 0; i < 25; ++i)
            repeated += "ha";
        const auto fixed = kadan::stt::detect_and_fix_repetitions(repeated, 20);
        assert(fixed == "hi ha");
    }
}

void test_catalog_all_apache_2() {
    const auto catalog_path =
        std::filesystem::path(KADAN_WORKER_SOURCE_DATA) / "catalog.json";
    std::ifstream file(catalog_path);
    assert(bool(file));
    const auto doc = nlohmann::json::parse(file);
    const auto& models = doc.at("models");
    assert(models.is_array() && !models.empty());
    bool has_asr = false;
    for (const auto& entry : models) {
        assert(entry.at("license").get<std::string>() == "apache-2.0");
        assert(
            entry.contains("license_url") && entry.at("license_url").is_string()
            && !entry.at("license_url").get<std::string>().empty()
        );
        if (entry.at("id").get<std::string>() == "qwen3-asr-1.7b")
            has_asr = true;
    }
    assert(has_asr);
}

void test_config_parsing() {
    const auto dir = std::filesystem::temp_directory_path() / "kadan-qwen3-asr-cfg-test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    nlohmann::json cfg = {
        {"model_type", "qwen3_asr"},
        {"support_languages", {"English", "Chinese"}},
        {"thinker_config",
         {{"audio_start_token_id", 151669},
          {"audio_end_token_id", 151670},
          {"audio_token_id", 151676},
          {"audio_config",
           {{"d_model", 1024},
            {"downsample_hidden_size", 480},
            {"encoder_attention_heads", 16},
            {"encoder_ffn_dim", 4096},
            {"encoder_layers", 24},
            {"num_mel_bins", 128},
            {"max_source_positions", 1500},
            {"n_window", 50},
            {"n_window_infer", 800},
            {"output_dim", 2048}}},
          {"text_config",
           {{"vocab_size", 151936},
            {"hidden_size", 2048},
            {"intermediate_size", 6144},
            {"num_hidden_layers", 28},
            {"num_attention_heads", 16},
            {"num_key_value_heads", 8},
            {"head_dim", 128},
            {"max_position_embeddings", 65536},
            {"rms_norm_eps", 1e-6},
            {"rope_theta", 1000000}}}}}
    };
    {
        std::ofstream out(dir / "config.json");
        out << cfg.dump(2);
    }
    auto budget = std::make_shared<kadan::checkpoint::MemoryBudget>(4 * 1024 * 1024);
    const auto parsed = kadan::stt::read_qwen3_asr_config(dir.c_str(), budget);
    assert(parsed.audio.encoder_layers == 24);
    assert(parsed.text.num_hidden_layers == 28);
    assert(parsed.text.hidden_size == 2048);
    assert(parsed.audio_token_id == 151676);
    std::filesystem::remove_all(dir);
}
}  // namespace

int main() {
    test_slaney_mel_filters();
    test_feat_extract_output_length();
    test_sinusoids();
    test_language_normalization();
    test_parse_asr_output();
    test_catalog_all_apache_2();
    test_config_parsing();
    return 0;
}
