#pragma once

#include "frontend/features/feature_types.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace tinynav::frontend {

struct GeometryCheckerConfig {
    double min_depth_m{0.1};
    double max_depth_m{10.0};
    std::size_t min_correspondences{6};
    std::size_t min_inliers{6};
    int ransac_iterations{100};
    float reprojection_error_px{2.0F};
    double ransac_confidence{0.999};
};

enum class GeometryStatus {
    Ready,
    InvalidInput,
    InsufficientCorrespondences,
    DegenerateGeometry,
    EstimationFailed,
    InsufficientInliers,
};

struct GeometryResult {
    GeometryStatus status{GeometryStatus::InvalidInput};
    // Maps current-camera coordinates to reference-camera coordinates.
    // Failure never carries an identity pose or usable visual constraints.
    std::optional<cv::Matx44d> T_reference_current;
    std::vector<FeatureMatch> inlier_matches;
    std::vector<cv::Point3f> points_current;
    std::vector<cv::Point2f> points_reference;
    std::size_t input_match_count{};
    std::size_t correspondence_count{};
    std::size_t inlier_count{};
    std::string detail;
};

class GeometryChecker {
public:
    explicit GeometryChecker(GeometryCheckerConfig config = {});
    ~GeometryChecker();

    // Depth is in meters and registered to the CURRENT left image. Image
    // coordinates are already rectified, so no distortion is applied here.
    // Returned match indices remain indices into the original feature rows.
    GeometryResult check(const FeatureSet& reference,
                         const FeatureSet& current,
                         const std::vector<FeatureMatch>& matches,
                         const cv::Mat& current_depth_meters,
                         const cv::Mat& depth_valid_mask) const;

private:
    const GeometryCheckerConfig config_;
};

}  // namespace tinynav::frontend
