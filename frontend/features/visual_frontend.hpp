#pragma once

#include "frontend/features/feature_matcher.hpp"
#include "frontend/features/geometry_checker.hpp"
#include "frontend/sensors/data_preparer.hpp"

namespace tinynav::frontend {

enum class VisualStatus {
    Ready,
    FeaturesOnly,
    InvalidInput,
    ExtractionFailed,
    MatchingFailed,
    GeometryRejected,
};

struct VisualResult {
    VisualStatus status{VisualStatus::InvalidInput};
    std::optional<std::uint64_t> reference_frame_id;
    std::optional<FeatureSet> current_features;
    std::vector<FeatureMatch> matches;  // Candidates before geometric rejection.
    std::optional<GeometryResult> geometry;
    std::string detail;
};

// One visual stage after DataPreparer. The caller owns its reference keyframe
// and decides when to replace it; failed tracking never replaces it implicitly.
// No IMU integration, keyframe selection, or absolute-pose estimation occurs.
// Caller serializes access; extractor and matcher must outlive this object.
class VisualFrontend {
public:
    VisualFrontend(FeatureExtractor& extractor, FeatureMatcher& matcher,
                   GeometryCheckerConfig geometry_config = {});
    ~VisualFrontend();

    // Without a reference, return FeaturesOnly for initialization. With one,
    // Ready requires a valid geometric constraint. Use geometry.inlier_matches
    // for VIO, never the unfiltered matches. PreparedFrame and its IMU stay with
    // the caller; this stage does not commit/discard the FrameAssembler.
    VisualResult process(const PreparedFrame& current,
                         const FeatureSet* reference = nullptr);

private:
    FeatureExtractor& extractor_;
    FeatureMatcher& matcher_;
    GeometryChecker geometry_;
};

}  // namespace tinynav::frontend
