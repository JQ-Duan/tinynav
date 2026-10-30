#pragma once

#include "frontend/sensors/imu_buffer.hpp"

#include <sensor_msgs/msg/image.hpp>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>

namespace tinynav::frontend {

struct SynchronizedFrame {
    std::uint64_t id{};
    TimestampNs timestamp_ns{};
    bool first_frame{};
    sensor_msgs::msg::Image::ConstSharedPtr left;
    sensor_msgs::msg::Image::ConstSharedPtr right;
    sensor_msgs::msg::Image::ConstSharedPtr depth;
    ImuInterval imu;
};

struct FrameAssemblerConfig {
    std::size_t queue_capacity{10};
    TimestampNs synchronization_tolerance_ns{0};
};

enum class FramePushStatus {
    Accepted,
    AcceptedWithEviction,
    DuplicateTimestamp,
    OutOfOrder,
    InvalidMessage,
};

enum class FrameAssemblyStatus {
    Ready,
    WaitingForImages,
    WaitingForImu,
    ImuHistoryUnavailable,
    ImuGapTooLarge,
    InvalidImuInterval,
    NonIncreasingFrameTime,
};

struct FrameAssemblyResult {
    FrameAssemblyStatus status{FrameAssemblyStatus::WaitingForImages};
    std::optional<SynchronizedFrame> frame;
};

struct FrameAssemblerStats {
    std::uint64_t committed_frames{};
    std::uint64_t discarded_frames{};
    std::uint64_t unmatched_images{};
    std::uint64_t evicted_images{};
    std::uint64_t rejected_messages{};
};

// The caller serializes access. ImuBuffer may receive data concurrently and
// must outlive this object. Inputs already share a clock domain and remain
// immutable, including through any aliases held by the caller.
//
// Match the oldest left image with the oldest right/depth images within the
// configured tolerance. No timestamps, pixel values, or IMU values are changed.
// Image payload validation belongs to DataPreparer.
class FrameAssembler {
public:
    explicit FrameAssembler(ImuBuffer& imu_buffer,
                            FrameAssemblerConfig config = {});
    ~FrameAssembler();

    FrameAssembler(const FrameAssembler&) = delete;
    FrameAssembler& operator=(const FrameAssembler&) = delete;

    FramePushStatus push_left(sensor_msgs::msg::Image::ConstSharedPtr image);
    FramePushStatus push_right(sensor_msgs::msg::Image::ConstSharedPtr image);
    FramePushStatus push_depth(sensor_msgs::msg::Image::ConstSharedPtr image);

    // A matched camera candidate remains fixed while waiting for IMU coverage.
    // Once Ready, repeated calls return the same frame and IMU snapshot until
    // commit/discard/reset. Queue eviction cannot invalidate this snapshot.
    FrameAssemblyResult peek_frame();

    // Commit only after preparation and delivery succeed. Discard leaves the
    // previous committed timestamp unchanged, preserving the next IMU interval.
    // A stale or unknown id returns false without changing assembler state.
    bool commit_frame(std::uint64_t id);
    bool discard_frame(std::uint64_t id);

    // An anchor is the timestamp of an already established prior state.
    // Reset clears pending images, ordering state, and statistics, but not the
    // shared IMU buffer. Frame identifiers are never reused across resets.
    void reset(std::optional<TimestampNs> anchor = std::nullopt);
    FrameAssemblerStats statistics() const;

private:
    using ImagePtr = sensor_msgs::msg::Image::ConstSharedPtr;

    struct QueuedImage {
        TimestampNs timestamp_ns;
        ImagePtr message;
    };

    struct CameraCandidate {
        TimestampNs timestamp_ns;
        ImagePtr left;
        ImagePtr right;
        ImagePtr depth;
    };

    FramePushStatus push(ImagePtr image, std::deque<QueuedImage>& queue,
                         std::optional<TimestampNs>& last_time);
    bool too_early(TimestampNs candidate, TimestampNs anchor) const;
    bool too_late(TimestampNs candidate, TimestampNs anchor) const;
    void discard_early(std::deque<QueuedImage>& queue, TimestampNs anchor);
    void remove_matched_images();

    ImuBuffer& imu_buffer_;
    const FrameAssemblerConfig config_;
    std::deque<QueuedImage> left_images_;
    std::deque<QueuedImage> right_images_;
    std::deque<QueuedImage> depth_images_;
    std::optional<TimestampNs> last_left_time_;
    std::optional<TimestampNs> last_right_time_;
    std::optional<TimestampNs> last_depth_time_;
    std::optional<TimestampNs> committed_time_;
    std::optional<CameraCandidate> candidate_;
    std::optional<SynchronizedFrame> pending_;
    std::uint64_t next_frame_id_{1};
    FrameAssemblerStats statistics_;
};

}  // namespace tinynav::frontend
