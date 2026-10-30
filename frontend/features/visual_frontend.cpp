#include "frontend/features/visual_frontend.hpp"

#include <utility>

namespace tinynav::frontend {

VisualFrontend::VisualFrontend(FeatureExtractor& extractor, FeatureMatcher& matcher,
                               GeometryCheckerConfig geometry_config)
    : extractor_(extractor), matcher_(matcher), geometry_(geometry_config) {}

VisualFrontend::~VisualFrontend() = default;

VisualResult VisualFrontend::process(const PreparedFrame& current, const FeatureSet* reference) {
    VisualResult result;
    if (current.left_gray.empty() || current.left_gray.type() != CV_8UC1 ||
        current.left_gray.dims != 2 || current.depth_meters.type() != CV_32FC1 ||
        current.depth_meters.dims != 2 || current.depth_meters.size() != current.left_gray.size() ||
        current.depth_valid_mask.type() != CV_8UC1 || current.depth_valid_mask.dims != 2 ||
        current.depth_valid_mask.size() != current.left_gray.size() ||
        current.calibration.width != static_cast<std::uint32_t>(current.left_gray.cols) ||
        current.calibration.height != static_cast<std::uint32_t>(current.left_gray.rows)) {
        result.detail = "Prepared left image, depth, mask, and calibration dimensions must agree";
        return result;
    }
    if (reference != nullptr) {
        result.reference_frame_id = reference->frame_id;
        if (reference->timestamp_ns >= current.timestamp_ns || reference->frame_id == current.id) {
            result.detail = "Reference must be a distinct earlier frame";
            return result;
        }
        try {
            validate_features(*reference);
        } catch (const std::exception& error) {
            result.detail = error.what();
            return result;
        }
    }
    auto extraction = extractor_.extract(current.left_gray, current.id, current.timestamp_ns,
                                         current.calibration.left_intrinsics);
    if (extraction.status != FeatureStatus::Ready) {
        result.status = VisualStatus::ExtractionFailed;
        result.detail = extraction.detail;
        return result;
    }
    result.current_features = std::move(extraction.features);
    if (reference == nullptr) {
        result.status = VisualStatus::FeaturesOnly;
        return result;
    }
    auto matching = matcher_.match(*reference, *result.current_features);
    if (matching.status != FeatureStatus::Ready) {
        result.status = VisualStatus::MatchingFailed;
        result.detail = matching.detail;
        return result;
    }
    result.matches = std::move(matching.matches);
    result.geometry = geometry_.check(*reference, *result.current_features, result.matches,
                                      current.depth_meters, current.depth_valid_mask);
    if (result.geometry->status != GeometryStatus::Ready) {
        result.status = VisualStatus::GeometryRejected;
        result.detail = result.geometry->detail;
        return result;
    }
    result.status = VisualStatus::Ready;
    return result;
}

}  // namespace tinynav::frontend
