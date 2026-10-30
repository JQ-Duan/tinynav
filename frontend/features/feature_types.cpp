#include "frontend/features/feature_types.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace tinynav::frontend {

void validate_features(const FeatureSet& features) {
    if (features.image_size.width <= 0 || features.image_size.height <= 0 ||
        features.keypoints.size() != features.valid.size() ||
        features.keypoints.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("Invalid feature dimensions or mask size");
    }
    for (double value : features.intrinsics.val) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Non-finite camera intrinsics");
        }
    }
    const auto& k = features.intrinsics;
    if (k(0, 0) <= 0 || k(1, 1) <= 0 || std::abs(k(0, 1)) > 1e-9 ||
        std::abs(k(1, 0)) > 1e-9 || std::abs(k(2, 0)) > 1e-9 ||
        std::abs(k(2, 1)) > 1e-9 || std::abs(k(2, 2) - 1.0) > 1e-9) {
        throw std::invalid_argument("Expected rectified pinhole intrinsics");
    }
    if (features.keypoints.empty()) {
        if (!features.descriptors.empty()) {
            throw std::invalid_argument("Descriptors without keypoints");
        }
        return;
    }
    if (features.descriptors.type() != CV_32FC1 || features.descriptors.dims != 2 ||
        features.descriptors.rows != static_cast<int>(features.keypoints.size()) ||
        features.descriptors.cols <= 0) {
        throw std::invalid_argument("Expected one float descriptor row per keypoint");
    }
    for (std::size_t i = 0; i < features.keypoints.size(); ++i) {
        if (!features.valid[i]) continue;
        const auto& point = features.keypoints[i];
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || point.x < 0 ||
            point.y < 0 || point.x >= features.image_size.width ||
            point.y >= features.image_size.height) {
            throw std::invalid_argument("Valid keypoint is outside the image");
        }
        const float* row = features.descriptors.ptr<float>(static_cast<int>(i));
        for (int j = 0; j < features.descriptors.cols; ++j) {
            if (!std::isfinite(row[j])) {
                throw std::invalid_argument("Non-finite valid descriptor");
            }
        }
    }
}

}  // namespace tinynav::frontend
