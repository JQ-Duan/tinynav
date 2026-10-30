#pragma once

#include "frontend/features/feature_extractor.hpp"

namespace tinynav::frontend {

struct FeatureMatcherConfig {
    float match_threshold{0.1F};
};

struct MatchingResult {
    FeatureStatus status{FeatureStatus::InvalidInput};
    std::vector<FeatureMatch> matches;
    std::string detail;
};

// LightGlue adapter. Match reference left-image features to current left-image
// features; preserve feature indices. No replacement matcher is used on error.
// Caller serializes access; the inference session must outlive this object.
class FeatureMatcher {
public:
    explicit FeatureMatcher(infrastructure::InferenceSession& session,
                            FeatureMatcherConfig config = {});
    ~FeatureMatcher();

    MatchingResult match(const FeatureSet& reference, const FeatureSet& current);

private:
    infrastructure::InferenceSession& session_;
    const FeatureMatcherConfig config_;
    std::vector<infrastructure::TensorSpec> inputs_;
};

}  // namespace tinynav::frontend
