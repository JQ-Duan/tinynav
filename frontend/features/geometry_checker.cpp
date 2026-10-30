#include "frontend/features/geometry_checker.hpp"

#include <opencv2/calib3d.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace tinynav::frontend {
namespace {

bool finite_pose(const cv::Mat& rotation, const cv::Mat& translation) {
    return cv::checkRange(rotation) && cv::checkRange(translation);
}

bool collinear(const cv::Mat& samples) {
    cv::Mat mean;
    cv::reduce(samples, mean, 0, cv::REDUCE_AVG, CV_64F);
    cv::Mat centered = samples - cv::repeat(mean, samples.rows, 1);
    cv::Mat singular_values;
    cv::SVD::compute(centered, singular_values, cv::SVD::NO_UV);
    return singular_values.total() < 2 ||
           singular_values.at<double>(0) <= std::numeric_limits<double>::epsilon() ||
           singular_values.at<double>(1) <= singular_values.at<double>(0) * 1e-6;
}

bool degenerate(const std::vector<cv::Point3f>& objects,
                const std::vector<cv::Point2f>& images) {
    cv::Mat object_samples(static_cast<int>(objects.size()), 3, CV_64F);
    cv::Mat image_samples(static_cast<int>(images.size()), 2, CV_64F);
    for (int i = 0; i < object_samples.rows; ++i) {
        const auto& object = objects[static_cast<std::size_t>(i)];
        const auto& image = images[static_cast<std::size_t>(i)];
        object_samples.at<double>(i, 0) = object.x;
        object_samples.at<double>(i, 1) = object.y;
        object_samples.at<double>(i, 2) = object.z;
        image_samples.at<double>(i, 0) = image.x;
        image_samples.at<double>(i, 1) = image.y;
    }
    // Planar scenes remain valid; only coincident/collinear samples are rejected.
    return collinear(object_samples) || collinear(image_samples);
}

}  // namespace

GeometryChecker::GeometryChecker(GeometryCheckerConfig config) : config_(config) {
    if (!std::isfinite(config_.min_depth_m) || !std::isfinite(config_.max_depth_m) ||
        config_.min_depth_m < 0 || config_.max_depth_m <= config_.min_depth_m ||
        config_.min_correspondences < 6 || config_.min_inliers < 6 ||
        config_.ransac_iterations <= 0 || !std::isfinite(config_.reprojection_error_px) ||
        config_.reprojection_error_px <= 0 || !std::isfinite(config_.ransac_confidence) ||
        config_.ransac_confidence <= 0 || config_.ransac_confidence >= 1) {
        throw std::invalid_argument("Invalid PnP geometry configuration");
    }
}

GeometryChecker::~GeometryChecker() = default;

GeometryResult GeometryChecker::check(const FeatureSet& reference,
                                      const FeatureSet& current,
                                      const std::vector<FeatureMatch>& matches,
                                      const cv::Mat& current_depth_meters,
                                      const cv::Mat& depth_valid_mask) const {
    GeometryResult result;
    result.input_match_count = matches.size();
    try {
        validate_features(reference);
        validate_features(current);
        if (current_depth_meters.dims != 2 || current_depth_meters.type() != CV_32FC1 ||
            current_depth_meters.size() != current.image_size ||
            depth_valid_mask.dims != 2 || depth_valid_mask.type() != CV_8UC1 ||
            depth_valid_mask.size() != current.image_size) {
            result.detail = "Expected current-image-sized float depth and byte validity mask";
            return result;
        }

        std::vector<cv::Point3f> points_current;
        std::vector<cv::Point2f> points_reference;
        std::vector<FeatureMatch> correspondences;
        std::unordered_set<std::size_t> used_reference;
        std::unordered_set<std::size_t> used_current;
        const auto& k = current.intrinsics;
        for (const auto& match : matches) {
            if (match.reference_index >= reference.keypoints.size() ||
                match.current_index >= current.keypoints.size() ||
                !reference.valid[match.reference_index] || !current.valid[match.current_index] ||
                !std::isfinite(match.score) || used_reference.count(match.reference_index) ||
                used_current.count(match.current_index)) {
                continue;
            }
            const auto& pixel = current.keypoints[match.current_index];
            const int u = static_cast<int>(pixel.x);
            const int v = static_cast<int>(pixel.y);
            const float depth = current_depth_meters.at<float>(v, u);
            if (!depth_valid_mask.at<std::uint8_t>(v, u) || !std::isfinite(depth) ||
                depth <= config_.min_depth_m || depth >= config_.max_depth_m) {
                continue;
            }
            const double x = (pixel.x - k(0, 2)) * depth / k(0, 0);
            const double y = (pixel.y - k(1, 2)) * depth / k(1, 1);
            if (!std::isfinite(x) || !std::isfinite(y) ||
                std::abs(x) > std::numeric_limits<float>::max() ||
                std::abs(y) > std::numeric_limits<float>::max()) {
                continue;
            }
            used_reference.insert(match.reference_index);
            used_current.insert(match.current_index);
            points_current.emplace_back(static_cast<float>(x), static_cast<float>(y), depth);
            points_reference.push_back(reference.keypoints[match.reference_index]);
            correspondences.push_back(match);
        }
        result.correspondence_count = correspondences.size();
        if (correspondences.size() < config_.min_correspondences ||
            correspondences.size() < config_.min_inliers) {
            result.status = GeometryStatus::InsufficientCorrespondences;
            result.detail = "Too few unique matches with usable current-frame depth";
            return result;
        }
        if (degenerate(points_current, points_reference)) {
            result.status = GeometryStatus::DegenerateGeometry;
            result.detail = "Coincident or collinear correspondences cannot constrain this pose";
            return result;
        }

        cv::Mat rotation_vector;
        cv::Mat translation;
        cv::Mat inliers;
        const bool solved = cv::solvePnPRansac(
            points_current, points_reference, reference.intrinsics, cv::noArray(),
            rotation_vector, translation, false, config_.ransac_iterations,
            config_.reprojection_error_px, config_.ransac_confidence, inliers, cv::SOLVEPNP_EPNP);
        if (!solved || inliers.empty() || !finite_pose(rotation_vector, translation)) {
            result.status = GeometryStatus::EstimationFailed;
            result.detail = "PnP did not return a finite pose with inliers";
            return result;
        }

        cv::Mat rotation;
        cv::Rodrigues(rotation_vector, rotation);
        if (!cv::checkRange(rotation)) {
            result.status = GeometryStatus::EstimationFailed;
            result.detail = "PnP returned a non-finite rotation";
            return result;
        }
        cv::Matx44d transform = cv::Matx44d::eye();
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                transform(row, col) = rotation.at<double>(row, col);
            }
            transform(row, 3) = translation.at<double>(row);
        }

        std::vector<cv::Point2f> reprojections;
        cv::projectPoints(points_current, rotation_vector, translation,
                          reference.intrinsics, cv::noArray(), reprojections);
        std::vector<std::size_t> checked_inliers;
        const double max_error_squared =
            static_cast<double>(config_.reprojection_error_px) * config_.reprojection_error_px;
        for (int i = 0; i < inliers.rows; ++i) {
            const int index = inliers.at<int>(i);
            if (index < 0 || static_cast<std::size_t>(index) >= correspondences.size()) continue;
            const auto n = static_cast<std::size_t>(index);
            const auto& object = points_current[n];
            const double z = transform(2, 0) * object.x + transform(2, 1) * object.y +
                             transform(2, 2) * object.z + transform(2, 3);
            const cv::Point2f error = reprojections[n] - points_reference[n];
            const double squared_error = static_cast<double>(error.x) * error.x +
                                         static_cast<double>(error.y) * error.y;
            if (std::isfinite(z) && z > 0 && std::isfinite(squared_error) &&
                squared_error <= max_error_squared) {
                checked_inliers.push_back(n);
            }
        }
        result.inlier_count = checked_inliers.size();
        if (checked_inliers.size() < config_.min_inliers) {
            result.status = GeometryStatus::InsufficientInliers;
            result.detail = "Too few inliers after positive-depth and reprojection checks";
            return result;
        }
        for (const auto index : checked_inliers) {
            result.inlier_matches.push_back(correspondences[index]);
            result.points_current.push_back(points_current[index]);
            result.points_reference.push_back(points_reference[index]);
        }
        result.T_reference_current = transform;
        result.status = GeometryStatus::Ready;
        return result;
    } catch (const std::invalid_argument& error) {
        result.status = GeometryStatus::InvalidInput;
        result.detail = error.what();
    } catch (const cv::Exception& error) {
        result.status = GeometryStatus::EstimationFailed;
        result.detail = error.what();
    }
    return result;
}

}  // namespace tinynav::frontend
