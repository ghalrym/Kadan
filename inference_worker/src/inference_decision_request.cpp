#include <type_traits>

#include "kadan/preparation.hpp"

namespace kadan::serving {
void prepare_input(
    DecisionRequest&,
    InferenceRequest& request,
    ModelStore& store,
    const std::atomic_bool& cancel
) {
    request.model = "laya";
    request.configuration.checkpoint = store.acquire(request.model, cancel);
}

Json present_decisions(const std::vector<DecisionAnswer>& answers) {
    Json output = Json::array();
    for (const auto& answer : answers) {
        Json row{{"key", answer.key}, {"confidence", answer.confidence}};
        std::visit(
            [&](const auto& result) {
                using Result = std::decay_t<decltype(result)>;
                row["value"] = result.value;
                if constexpr (std::is_same_v<Result, NoulResult>) {
                    row["type"] = "Noul";
                    row["probabilities"] = nullptr;
                } else {
                    row["type"] =
                        std::is_same_v<Result, ChoiceResult> ? "Choice" : "Score";
                    row["probabilities"] = result.probabilities;
                }
            },
            answer.result
        );
        output.push_back(std::move(row));
    }
    return output;
}
}  // namespace kadan::serving
