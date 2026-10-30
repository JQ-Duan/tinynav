#include "frontend/features/feature_matcher.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace tinynav::frontend {
namespace {

infrastructure::Tensor feature_tensor(const infrastructure::TensorSpec& spec,
                                      const FeatureSet& features) {
    infrastructure::Tensor tensor{spec.shape, {}};
    const auto n = static_cast<std::int64_t>(features.keypoints.size());
    if (spec.name.rfind("kpts", 0) == 0) {
        if (spec.shape != std::vector<std::int64_t>{1, n, 2}) throw std::invalid_argument("LightGlue keypoint slots do not match SuperPoint");
        for (std::size_t i = 0; i < features.keypoints.size(); ++i) {
            tensor.values.push_back(features.valid[i] ? features.keypoints[i].x : 0.0F);
            tensor.values.push_back(features.valid[i] ? features.keypoints[i].y : 0.0F);
        }
    } else if (spec.name.rfind("desc", 0) == 0) {
        if (spec.shape != std::vector<std::int64_t>{1, n, features.descriptors.cols}) throw std::invalid_argument("LightGlue descriptor shape does not match SuperPoint");
        tensor.values.reserve(features.descriptors.total());
        for (int i = 0; i < features.descriptors.rows; ++i) {
            const float* row = features.descriptors.ptr<float>(i);
            for (int j = 0; j < features.descriptors.cols; ++j) tensor.values.push_back(features.valid[static_cast<std::size_t>(i)] ? row[j] : 0.0F);
        }
    } else if (spec.name.rfind("mask", 0) == 0) {
        if (spec.shape != std::vector<std::int64_t>{1, n, 1} && spec.shape != std::vector<std::int64_t>{1, n}) throw std::invalid_argument("Invalid LightGlue mask shape");
        for (auto valid : features.valid) tensor.values.push_back(valid ? 1.0F : 0.0F);
    } else {
        if (spec.shape != std::vector<std::int64_t>{2} && spec.shape != std::vector<std::int64_t>{1, 2}) throw std::invalid_argument("Invalid LightGlue image size shape");
        if (features.image_size.width > 16777216 || features.image_size.height > 16777216) throw std::invalid_argument("Image dimensions exceed exact tensor integer range");
        tensor.values = {static_cast<float>(features.image_size.width), static_cast<float>(features.image_size.height)};
    }
    infrastructure::validate_tensor(tensor);
    return tensor;
}

}  // namespace

FeatureMatcher::FeatureMatcher(infrastructure::InferenceSession& session, FeatureMatcherConfig config)
    : session_(session), config_(config), inputs_(session.inputs()) {
    if (!std::isfinite(config_.match_threshold) || config_.match_threshold < 0 || config_.match_threshold > 1) {
        throw std::invalid_argument("LightGlue threshold must be in [0,1]");
    }
    std::set<std::string> expected{"kpts0", "kpts1", "desc0", "desc1", "mask0", "mask1", "img_shape0", "img_shape1", "threshold"};
    for (const auto& input : inputs_) {
        if (!expected.erase(input.name)) throw std::invalid_argument("Unexpected or duplicate LightGlue input: " + input.name);
        infrastructure::tensor_element_count(input.shape);
        if (input.name == "threshold" && infrastructure::tensor_element_count(input.shape) != 1) throw std::invalid_argument("Expected scalar LightGlue threshold");
    }
    if (!expected.empty()) throw std::invalid_argument("Incomplete LightGlue inputs");
}

FeatureMatcher::~FeatureMatcher() = default;

MatchingResult FeatureMatcher::match(const FeatureSet& reference, const FeatureSet& current) {
    infrastructure::TensorMap inputs;
    try {
        validate_features(reference);
        validate_features(current);
        if (std::none_of(reference.valid.begin(), reference.valid.end(), [](auto value) { return value != 0; }) ||
            std::none_of(current.valid.begin(), current.valid.end(), [](auto value) { return value != 0; })) {
            return {FeatureStatus::Ready, {}, {}};
        }
        for (const auto& spec : inputs_) {
            if (spec.name == "threshold") inputs.emplace(spec.name, infrastructure::Tensor{spec.shape, {config_.match_threshold}});
            else inputs.emplace(spec.name, feature_tensor(spec, spec.name.back() == '0' ? reference : current));
        }
    } catch (const std::exception& error) {
        return {FeatureStatus::InvalidInput, {}, error.what()};
    }
    infrastructure::TensorMap outputs;
    try {
        outputs = session_.run(inputs);
    } catch (const std::exception& error) {
        return {FeatureStatus::InferenceError, {}, error.what()};
    }
    std::vector<FeatureMatch> matches;
    try {
        const auto index_it = outputs.find("match_indices");
        const auto score_it = outputs.find("score");
        if (index_it == outputs.end() || score_it == outputs.end()) throw std::invalid_argument("Missing LightGlue match_indices/score");
        const auto& indices = index_it->second;
        const auto& scores = score_it->second;
        infrastructure::validate_tensor(indices);
        infrastructure::validate_tensor(scores);
        const auto shape = std::vector<std::int64_t>{1, static_cast<std::int64_t>(reference.keypoints.size())};
        if (indices.shape != shape || scores.shape != shape) throw std::invalid_argument("Expected LightGlue outputs [1,N_reference]");
        std::vector<std::optional<FeatureMatch>> best(current.keypoints.size());
        for (std::size_t i = 0; i < reference.keypoints.size(); ++i) {
            const float index = indices.values[i];
            if (!std::isfinite(index) || index < -1 || std::trunc(index) != index ||
                (index >= 0 && static_cast<double>(index) >= static_cast<double>(current.keypoints.size()))) {
                throw std::invalid_argument("Invalid LightGlue match index");
            }
            if (index == -1 || !reference.valid[i]) continue;
            const auto j = static_cast<std::size_t>(index);
            if (!current.valid[j]) continue;
            const float score = scores.values[i];
            if (!std::isfinite(score) || score < 0 || score > 1) throw std::invalid_argument("Invalid LightGlue match score");
            if (score < config_.match_threshold) continue;
            if (!best[j] || score > best[j]->score) best[j] = FeatureMatch{i, j, score};
        }
        for (const auto& match : best) if (match) matches.push_back(*match);
        std::sort(matches.begin(), matches.end(), [](const auto& a, const auto& b) { return a.reference_index < b.reference_index; });
    } catch (const std::exception& error) {
        return {FeatureStatus::InvalidModelOutput, {}, error.what()};
    }
    return {FeatureStatus::Ready, std::move(matches), {}};
}

}  // namespace tinynav::frontend
