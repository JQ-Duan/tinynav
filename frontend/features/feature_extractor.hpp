#pragma once

#include "frontend/features/feature_types.hpp"
#include "infrastructure/inference/inference_session.hpp"

#include <optional>
#include <string>

namespace tinynav::frontend {

enum class FeatureStatus { Ready, InvalidInput, InferenceError, InvalidModelOutput };

struct FeatureExtractorConfig {
    float keypoint_threshold{0.0005F};
};

struct ExtractionResult {
    FeatureStatus status{FeatureStatus::InvalidInput};
    std::optional<FeatureSet> features;
    std::string detail;
};

// SuperPoint adapter for TinyNav's image/keypoint_threshold ->
// kpts/descps/mask export. The session must outlive this object; caller serializes
// access. Preserve the original model's 0..255 input range (no extra /255).
class FeatureExtractor {
public:
    explicit FeatureExtractor(infrastructure::InferenceSession& session,
                              FeatureExtractorConfig config = {});
    ~FeatureExtractor();

    ExtractionResult extract(const cv::Mat& left_gray, std::uint64_t frame_id,
                             std::int64_t timestamp_ns,
                             const cv::Matx33d& intrinsics);

private:
    infrastructure::InferenceSession& session_;
    const FeatureExtractorConfig config_;
    cv::Size network_size_;
    std::vector<std::int64_t> threshold_shape_;
};

}  // namespace tinynav::frontend
