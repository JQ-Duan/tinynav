#pragma once

#include "frontend/sensors/frame_assembler.hpp"

#include <opencv2/core.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace tinynav::frontend {

struct DataPreparerConfig {
    // ZED 32FC1 input is in meters. Other publishers must specify their scale.
    double float_depth_scale{1.0};
    // Encoding alone does not specify units; e.g. explicitly use 0.001 for mm.
    std::optional<double> integer_depth_scale;
};

struct StereoCalibration {
    std::uint32_t width{};
    std::uint32_t height{};
    cv::Matx33d left_intrinsics;
    cv::Matx33d right_intrinsics;
    double baseline_m{};
    std::string left_frame_id;
    std::string right_frame_id;
};

struct PreparedImuSample {
    TimestampNs timestamp_ns{};
    std::array<double, 3> acceleration_mps2{};
    std::array<double, 3> angular_velocity_radps{};
};

struct PreparedFrame {
    std::uint64_t id{};
    TimestampNs timestamp_ns{};
    TimestampNs right_timestamp_ns{};
    TimestampNs depth_timestamp_ns{};
    bool first_frame{};
    cv::Mat left_gray;          // CV_8UC1, owned storage.
    cv::Mat right_gray;         // CV_8UC1, owned storage.
    cv::Mat depth_meters;       // CV_32FC1, invalid pixels are NaN.
    cv::Mat depth_valid_mask;   // CV_8UC1, valid = 255, invalid = 0.
    TimestampNs imu_start_ns{};
    TimestampNs imu_end_ns{};
    std::string imu_frame_id;
    std::vector<PreparedImuSample> imu;
    StereoCalibration calibration;
};

enum class PreparationStatus {
    Ready,
    CalibrationMissing,
    InvalidCalibration,
    InvalidFrame,
    UnsupportedImageEncoding,
    UnsupportedDepthEncoding,
    MissingDepthScale,
    InvalidImu,
};

struct PreparationResult {
    PreparationStatus status{PreparationStatus::InvalidFrame};
    std::optional<PreparedFrame> frame;
    std::string detail;
};

// The caller serializes calibration updates and preparation. Input images are
// already rectified; depth must already be registered to the left image.
// No clock correction, image resizing, IMU integration, or bias removal occurs.
// Successful results own their pixels independently of the ROS input messages.
class DataPreparer {
public:
    explicit DataPreparer(DataPreparerConfig config = {});
    ~DataPreparer();

    // Uses rectified intrinsics from P and a horizontal stereo baseline.
    // Binning/cropped ROI are not applied implicitly. Invalid updates clear
    // the previous calibration so it cannot be silently reused.
    PreparationStatus set_calibration(const sensor_msgs::msg::CameraInfo& left,
                                      const sensor_msgs::msg::CameraInfo& right);
    void clear_calibration();

    // Does not commit/discard the assembler candidate. Commit only after this
    // returns Ready and the downstream consumer accepts the result.
    PreparationResult prepare(const SynchronizedFrame& frame) const;

private:
    const DataPreparerConfig config_;
    std::optional<StereoCalibration> calibration_;
};

}  // namespace tinynav::frontend
