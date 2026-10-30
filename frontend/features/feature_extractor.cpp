#include "frontend/features/feature_extractor.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace tinynav::frontend {
namespace {

const infrastructure::Tensor& output(const infrastructure::TensorMap& tensors,
                                     const std::string& name) {
    const auto found = tensors.find(name);
    if (found == tensors.end()) throw std::invalid_argument("Missing SuperPoint output: " + name);
    infrastructure::validate_tensor(found->second);
    return found->second;
}

}  // namespace

FeatureExtractor::FeatureExtractor(infrastructure::InferenceSession& session,
                                   FeatureExtractorConfig config)
    : session_(session), config_(config) {
    if (!std::isfinite(config_.keypoint_threshold) || config_.keypoint_threshold < 0 ||
        config_.keypoint_threshold > 1) {
        throw std::invalid_argument("SuperPoint threshold must be in [0,1]");
    }
    const auto inputs = session_.inputs();
    if (inputs.size() != 2) throw std::invalid_argument("Expected two SuperPoint inputs");
    bool image_found = false;
    for (const auto& input : inputs) {
        if (input.name == "image") {
            if (image_found || input.shape.size() != 4 || input.shape[0] != 1 ||
                input.shape[1] != 1 || input.shape[2] <= 0 || input.shape[3] <= 0 ||
                input.shape[2] > std::numeric_limits<int>::max() ||
                input.shape[3] > std::numeric_limits<int>::max()) {
                throw std::invalid_argument("Expected SuperPoint image [1,1,H,W]");
            }
            network_size_ = cv::Size(static_cast<int>(input.shape[3]), static_cast<int>(input.shape[2]));
            image_found = true;
        } else if (input.name == "keypoint_threshold") {
            if (!threshold_shape_.empty() || infrastructure::tensor_element_count(input.shape) != 1) {
                throw std::invalid_argument("Expected scalar SuperPoint threshold");
            }
            threshold_shape_ = input.shape;
        } else {
            throw std::invalid_argument("Unsupported SuperPoint input: " + input.name);
        }
    }
    if (!image_found || threshold_shape_.empty()) throw std::invalid_argument("Incomplete SuperPoint inputs");
}

FeatureExtractor::~FeatureExtractor() = default;

ExtractionResult FeatureExtractor::extract(const cv::Mat& left_gray, std::uint64_t frame_id,
                                           std::int64_t timestamp_ns,
                                           const cv::Matx33d& intrinsics) {
    FeatureSet features;
    features.frame_id = frame_id;
    features.timestamp_ns = timestamp_ns;
    features.image_size = left_gray.size();
    features.intrinsics = intrinsics;
    infrastructure::TensorMap inputs;
    try {
        if (left_gray.empty() || left_gray.dims != 2 || left_gray.type() != CV_8UC1) {
            throw std::invalid_argument("SuperPoint requires a nonempty CV_8UC1 left image");
        }
        validate_features(features);
        cv::Mat resized;
        cv::resize(left_gray, resized, network_size_, 0, 0, cv::INTER_LINEAR);
        infrastructure::Tensor image;
        image.shape = {1, 1, network_size_.height, network_size_.width};
        image.values.reserve(resized.total());
        for (int y = 0; y < resized.rows; ++y) {
            const auto* row = resized.ptr<std::uint8_t>(y);
            for (int x = 0; x < resized.cols; ++x) image.values.push_back(static_cast<float>(row[x]));
        }
        inputs.emplace("image", std::move(image));
        inputs.emplace("keypoint_threshold", infrastructure::Tensor{threshold_shape_, {config_.keypoint_threshold}});
    } catch (const std::exception& error) {
        return {FeatureStatus::InvalidInput, std::nullopt, error.what()};
    }
    infrastructure::TensorMap outputs;
    try {
        outputs = session_.run(inputs);
    } catch (const std::exception& error) {
        return {FeatureStatus::InferenceError, std::nullopt, error.what()};
    }
    try {
        const auto& mask = output(outputs, "mask");
        const auto& keypoints = output(outputs, "kpts");
        const auto& descriptors = output(outputs, "descps");
        if (!((mask.shape.size() == 2 && mask.shape[0] == 1) ||
              (mask.shape.size() == 3 && mask.shape[0] == 1 && mask.shape[2] == 1))) {
            throw std::invalid_argument("Expected SuperPoint mask [1,N] or [1,N,1]");
        }
        const auto count = mask.shape[1];
        const bool row_layout = keypoints.shape == std::vector<std::int64_t>{1, count, 2};
        if (count > std::numeric_limits<int>::max() ||
            (!row_layout && keypoints.shape != std::vector<std::int64_t>{1, 2, count}) ||
            descriptors.shape.size() != 3 || descriptors.shape[0] != 1 ||
            descriptors.shape[1] != count || descriptors.shape[2] > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("Incompatible SuperPoint keypoint/descriptor shapes");
        }
        const int dimension = static_cast<int>(descriptors.shape[2]);
        features.keypoints.resize(static_cast<std::size_t>(count));
        features.valid.resize(static_cast<std::size_t>(count), 0);
        features.descriptors = cv::Mat::zeros(static_cast<int>(count), dimension, CV_32FC1);
        const double sx = static_cast<double>(left_gray.cols) / network_size_.width;
        const double sy = static_cast<double>(left_gray.rows) / network_size_.height;
        for (std::size_t i = 0; i < features.keypoints.size(); ++i) {
            if (!std::isfinite(mask.values[i]) || (mask.values[i] != 0 && mask.values[i] != 1)) {
                throw std::invalid_argument("SuperPoint mask must contain zeros and ones");
            }
            if (mask.values[i] == 0) continue;
            const float x = keypoints.values[row_layout ? 2 * i : i];
            const float y = keypoints.values[row_layout ? 2 * i + 1 : features.keypoints.size() + i];
            if (!std::isfinite(x) || !std::isfinite(y)) throw std::invalid_argument("Non-finite SuperPoint keypoint");
            const cv::Point2f point(static_cast<float>((x + 0.5) * sx - 0.5),
                                   static_cast<float>((y + 0.5) * sy - 0.5));
            // Half-pixel resizing can move an edge point outside the original
            // image. Keep its slot masked instead of clamping its coordinates.
            if (point.x < 0 || point.y < 0 || point.x >= left_gray.cols || point.y >= left_gray.rows) continue;
            features.keypoints[i] = point;
            features.valid[i] = 1;
            auto* row = features.descriptors.ptr<float>(static_cast<int>(i));
            for (int j = 0; j < dimension; ++j) {
                row[j] = descriptors.values[i * static_cast<std::size_t>(dimension) + static_cast<std::size_t>(j)];
            }
        }
        validate_features(features);
    } catch (const std::exception& error) {
        return {FeatureStatus::InvalidModelOutput, std::nullopt, error.what()};
    }
    return {FeatureStatus::Ready, std::move(features), {}};
}

}  // namespace tinynav::frontend
