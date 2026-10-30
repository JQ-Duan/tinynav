#include "frontend/sensors/data_preparer.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tinynav::frontend {
namespace {

using Image = sensor_msgs::msg::Image;
using CameraInfo = sensor_msgs::msg::CameraInfo;

PreparationResult failure(PreparationStatus status, const char* detail) {
    return {status, std::nullopt, detail};
}

bool valid_camera_info(const CameraInfo& info) {
    if (info.width == 0 || info.height == 0 || info.binning_x > 1 ||
        info.binning_y > 1 || info.roi.x_offset != 0 || info.roi.y_offset != 0 ||
        (info.roi.width != 0 && info.roi.width != info.width) ||
        (info.roi.height != 0 && info.roi.height != info.height)) {
        return false;
    }
    const auto finite = [](double value) { return std::isfinite(value); };
    if (!std::all_of(info.k.begin(), info.k.end(), finite) ||
        !std::all_of(info.r.begin(), info.r.end(), finite) ||
        !std::all_of(info.p.begin(), info.p.end(), finite) ||
        info.k[0] <= 0.0 || info.k[4] <= 0.0 ||
        info.p[0] <= 0.0 || info.p[5] <= 0.0) {
        return false;
    }
    constexpr double tolerance = 1e-9;
    return std::abs(info.p[1]) <= tolerance &&
           std::abs(info.p[4]) <= tolerance &&
           std::abs(info.p[7]) <= tolerance &&
           std::abs(info.p[8]) <= tolerance &&
           std::abs(info.p[9]) <= tolerance &&
           std::abs(info.p[10] - 1.0) <= tolerance &&
           std::abs(info.p[11]) <= tolerance;
}

cv::Matx33d rectified_intrinsics(const CameraInfo& info) {
    return {info.p[0], info.p[1], info.p[2],
            info.p[4], info.p[5], info.p[6],
            info.p[8], info.p[9], info.p[10]};
}

int image_channels(const std::string& encoding) {
    if (encoding == "mono8" || encoding == "8UC1") {
        return 1;
    }
    if (encoding == "rgb8" || encoding == "bgr8") {
        return 3;
    }
    if (encoding == "rgba8" || encoding == "bgra8") {
        return 4;
    }
    return 0;
}

bool valid_layout(const Image& image, std::size_t pixel_bytes) {
    if (image.header.stamp.nanosec >= 1'000'000'000 || image.is_bigendian > 1 ||
        image.width == 0 || image.height == 0 ||
        image.width > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        image.height > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    const auto row_size = static_cast<std::uint64_t>(image.width) * pixel_bytes;
    const auto storage_size = static_cast<std::uint64_t>(image.step) * image.height;
    return pixel_bytes > 0 && image.step >= row_size && image.data.size() >= storage_size;
}

cv::Mat grayscale(const Image& image, int channels) {
    cv::Mat pixels(static_cast<int>(image.height), static_cast<int>(image.width),
                   CV_MAKETYPE(CV_8U, channels));
    const auto row_bytes = static_cast<std::size_t>(image.width) * channels;
    for (int row = 0; row < pixels.rows; ++row) {
        std::memcpy(pixels.ptr(row),
                    image.data.data() + static_cast<std::size_t>(row) * image.step,
                    row_bytes);
    }
    if (channels == 1) {
        return pixels;
    }
    int conversion = cv::COLOR_BGR2GRAY;
    if (image.encoding == "rgb8") {
        conversion = cv::COLOR_RGB2GRAY;
    } else if (image.encoding == "rgba8") {
        conversion = cv::COLOR_RGBA2GRAY;
    } else if (image.encoding == "bgra8") {
        conversion = cv::COLOR_BGRA2GRAY;
    }
    cv::Mat gray;
    cv::cvtColor(pixels, gray, conversion);
    return gray;
}

std::uint16_t read_u16(const std::uint8_t* bytes, bool big_endian) {
    if (big_endian) {
        return static_cast<std::uint16_t>((std::uint16_t{bytes[0]} << 8) | bytes[1]);
    }
    return static_cast<std::uint16_t>((std::uint16_t{bytes[1]} << 8) | bytes[0]);
}

float read_f32(const std::uint8_t* bytes, bool big_endian) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "ROS 32FC1 requires IEEE-754 binary32");
    std::uint32_t bits = 0;
    for (int index = 0; index < 4; ++index) {
        const int source = big_endian ? index : 3 - index;
        bits = (bits << 8) | bytes[source];
    }
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void convert_depth(const Image& image, double scale, PreparedFrame& output) {
    output.depth_meters.create(static_cast<int>(image.height),
                               static_cast<int>(image.width), CV_32FC1);
    output.depth_valid_mask.create(output.depth_meters.size(), CV_8UC1);
    const bool floating = image.encoding == "32FC1";
    const std::size_t pixel_bytes = floating ? 4 : 2;
    for (int row = 0; row < output.depth_meters.rows; ++row) {
        const auto* source = image.data.data() + static_cast<std::size_t>(row) * image.step;
        auto* values = output.depth_meters.ptr<float>(row);
        auto* mask = output.depth_valid_mask.ptr<std::uint8_t>(row);
        for (int column = 0; column < output.depth_meters.cols; ++column) {
            const auto* pixel = source + static_cast<std::size_t>(column) * pixel_bytes;
            const double raw = floating ? static_cast<double>(read_f32(pixel, image.is_bigendian))
                                        : static_cast<double>(read_u16(pixel, image.is_bigendian));
            const double meters = raw * scale;
            float value = std::numeric_limits<float>::quiet_NaN();
            if (std::isfinite(meters) && meters > 0.0 &&
                meters <= std::numeric_limits<float>::max()) {
                const auto converted = static_cast<float>(meters);
                if (converted > 0.0F) {
                    value = converted;
                }
            }
            values[column] = value;
            mask[column] = std::isfinite(value) ? 255 : 0;
        }
    }
}

bool prepare_imu(const SynchronizedFrame& input, PreparedFrame& output) {
    const auto& interval = input.imu;
    if (interval.samples.empty() || interval.end_ns != input.timestamp_ns ||
        (input.first_frame && interval.start_ns != interval.end_ns) ||
        (!input.first_frame && interval.start_ns >= interval.end_ns)) {
        return false;
    }
    std::optional<TimestampNs> previous;
    output.imu.reserve(interval.samples.size());
    for (const auto& sample : interval.samples) {
        if (!sample || sample->header.stamp.nanosec >= 1'000'000'000 ||
            sample->linear_acceleration_covariance[0] == -1.0 ||
            sample->angular_velocity_covariance[0] == -1.0) {
            return false;
        }
        const auto stamp = sensor_timestamp_ns(sample->header.stamp);
        if (previous && stamp <= *previous) {
            return false;
        }
        if (!previous) {
            output.imu_frame_id = sample->header.frame_id;
        } else if (sample->header.frame_id != output.imu_frame_id) {
            return false;
        }
        const auto& a = sample->linear_acceleration;
        const auto& g = sample->angular_velocity;
        PreparedImuSample measurement{stamp, {a.x, a.y, a.z}, {g.x, g.y, g.z}};
        const auto finite = [](double value) { return std::isfinite(value); };
        if (!std::all_of(measurement.acceleration_mps2.begin(),
                         measurement.acceleration_mps2.end(), finite) ||
            !std::all_of(measurement.angular_velocity_radps.begin(),
                         measurement.angular_velocity_radps.end(), finite)) {
            return false;
        }
        output.imu.push_back(measurement);
        previous = stamp;
    }
    output.imu_start_ns = interval.start_ns;
    output.imu_end_ns = interval.end_ns;
    return output.imu.front().timestamp_ns <= interval.start_ns &&
           output.imu.back().timestamp_ns >= interval.end_ns;
}

}  // namespace

DataPreparer::DataPreparer(DataPreparerConfig config) : config_(config) {
    if (!std::isfinite(config_.float_depth_scale) || config_.float_depth_scale <= 0.0 ||
        (config_.integer_depth_scale &&
         (!std::isfinite(*config_.integer_depth_scale) || *config_.integer_depth_scale <= 0.0))) {
        throw std::invalid_argument("Depth conversion scales must be finite and positive");
    }
}

DataPreparer::~DataPreparer() = default;

PreparationStatus DataPreparer::set_calibration(const CameraInfo& left,
                                               const CameraInfo& right) {
    calibration_.reset();
    if (!valid_camera_info(left) || !valid_camera_info(right) ||
        left.width != right.width || left.height != right.height) {
        return PreparationStatus::InvalidCalibration;
    }
    const double baseline = left.p[3] / left.p[0] - right.p[3] / right.p[0];
    if (!std::isfinite(baseline) || baseline <= 0.0) {
        return PreparationStatus::InvalidCalibration;
    }
    calibration_ = StereoCalibration{left.width, left.height,
                                     rectified_intrinsics(left), rectified_intrinsics(right),
                                     baseline, left.header.frame_id, right.header.frame_id};
    return PreparationStatus::Ready;
}

void DataPreparer::clear_calibration() {
    calibration_.reset();
}

PreparationResult DataPreparer::prepare(const SynchronizedFrame& input) const {
    if (!calibration_) {
        return failure(PreparationStatus::CalibrationMissing, "Stereo calibration is not available");
    }
    if (!input.left || !input.right || !input.depth) {
        return failure(PreparationStatus::InvalidFrame, "A camera image is missing");
    }
    const auto& left = *input.left;
    const auto& right = *input.right;
    const auto& depth = *input.depth;
    const auto left_channels = image_channels(left.encoding);
    const auto right_channels = image_channels(right.encoding);
    if (left_channels == 0 || right_channels == 0) {
        return failure(PreparationStatus::UnsupportedImageEncoding,
                       "Expected 8-bit mono, RGB, BGR, RGBA, or BGRA images");
    }
    if (depth.encoding != "32FC1" && depth.encoding != "16UC1") {
        return failure(PreparationStatus::UnsupportedDepthEncoding,
                       "Expected 32FC1 or 16UC1 depth");
    }
    if (!valid_layout(left, left_channels) || !valid_layout(right, right_channels) ||
        !valid_layout(depth, depth.encoding == "32FC1" ? 4 : 2) ||
        sensor_timestamp_ns(left.header.stamp) != input.timestamp_ns) {
        return failure(PreparationStatus::InvalidFrame, "Invalid image layout or frame timestamp");
    }
    for (const auto* image : {&left, &right, &depth}) {
        if (image->width != calibration_->width || image->height != calibration_->height) {
            return failure(PreparationStatus::InvalidCalibration,
                           "Image dimensions do not match the stereo calibration");
        }
    }
    if ((!calibration_->left_frame_id.empty() && !left.header.frame_id.empty() &&
         calibration_->left_frame_id != left.header.frame_id) ||
        (!calibration_->right_frame_id.empty() && !right.header.frame_id.empty() &&
         calibration_->right_frame_id != right.header.frame_id)) {
        return failure(PreparationStatus::InvalidCalibration,
                       "Image coordinate frame does not match its CameraInfo");
    }
    double depth_scale = config_.float_depth_scale;
    if (depth.encoding == "16UC1") {
        if (!config_.integer_depth_scale) {
            return failure(PreparationStatus::MissingDepthScale,
                           "16UC1 depth requires an explicit meters-per-unit scale");
        }
        depth_scale = *config_.integer_depth_scale;
    }

    PreparedFrame output;
    output.id = input.id;
    output.timestamp_ns = input.timestamp_ns;
    output.right_timestamp_ns = sensor_timestamp_ns(right.header.stamp);
    output.depth_timestamp_ns = sensor_timestamp_ns(depth.header.stamp);
    output.first_frame = input.first_frame;
    output.calibration = *calibration_;
    if (!prepare_imu(input, output)) {
        return failure(PreparationStatus::InvalidImu,
                       "IMU samples must be ordered, finite, and cover the complete interval");
    }
    output.left_gray = grayscale(left, left_channels);
    output.right_gray = grayscale(right, right_channels);
    convert_depth(depth, depth_scale, output);
    return {PreparationStatus::Ready, std::move(output), {}};
}

}  // namespace tinynav::frontend
