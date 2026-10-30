#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tinynav::frontend {

// Preserve the model's row indices, including padded rows, for LightGlue and
// later VIO track association. Coordinates are in the original left image.
struct FeatureSet {
    std::uint64_t frame_id{};
    std::int64_t timestamp_ns{};
    cv::Size image_size;
    cv::Matx33d intrinsics;
    std::vector<cv::Point2f> keypoints;
    cv::Mat descriptors;  // Owned CV_32FC1, N rows by descriptor dimension.
    std::vector<std::uint8_t> valid;
};

struct FeatureMatch {
    std::size_t reference_index{};
    std::size_t current_index{};
    float score{};
};

// Throw invalid_argument for malformed data. Invalid padded rows may contain
// arbitrary model values; valid rows must have finite coordinates/descriptors.
void validate_features(const FeatureSet& features);

}  // namespace tinynav::frontend
