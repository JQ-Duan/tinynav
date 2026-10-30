#pragma once

#include <builtin_interfaces/msg/time.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <vector>

namespace tinynav::frontend {

using TimestampNs = std::int64_t;
using ImuSamplePtr = sensor_msgs::msg::Imu::ConstSharedPtr;

// Convert representation only. The publisher already supplies a shared clock.
TimestampNs sensor_timestamp_ns(const builtin_interfaces::msg::Time& stamp);

struct ImuInterval {
    TimestampNs start_ns{};
    TimestampNs end_ns{};
    // Raw samples from the last one <= start through the first one >= end.
    // start == end is valid for an initial anchor and implies no integration.
    std::vector<ImuSamplePtr> samples;
};

enum class ImuPushStatus {
    Accepted,
    AcceptedWithEviction,
    DuplicateTimestamp,
    OutOfOrder,
    InvalidSample,
};

enum class ImuReadStatus {
    Ready,
    Empty,
    InvalidInterval,
    HistoryUnavailable,
    AwaitingData,
    GapTooLarge,
};

struct ImuReadResult {
    ImuReadStatus status{ImuReadStatus::Empty};
    ImuInterval interval;
};

struct ImuBufferConfig {
    std::size_t capacity{4096};
    // Retain one boundary sample at/before latest - duration when available.
    // The hard capacity limit can still evict that sample.
    std::optional<TimestampNs> history_duration_ns;
    // Gap rejection is opt-in and does not alter the original measurements.
    std::optional<TimestampNs> max_gap_ns;
};

struct ImuBufferStats {
    std::uint64_t received{};
    std::uint64_t accepted{};
    std::uint64_t rejected{};
    std::uint64_t evicted{};
    std::size_t stored{};
};

// Thread-safe history, independently readable by VIO and propagation.
// Reads never consume samples. Each consumer maintains its own time cursor.
// Incoming messages must remain immutable, including through external aliases.
class ImuBuffer {
public:
    explicit ImuBuffer(ImuBufferConfig config = {});
    ~ImuBuffer();

    ImuPushStatus push(ImuSamplePtr sample);

    // Non-Ready results contain the requested bounds and no partial samples.
    // Snapshots keep their messages alive after clear() or history eviction.
    // This method does not interpolate, integrate, or modify timestamps.
    ImuReadResult read_interval(TimestampNs start_ns, TimestampNs end_ns) const;

    std::size_t size() const;
    std::optional<TimestampNs> oldest_timestamp() const;
    std::optional<TimestampNs> latest_timestamp() const;
    ImuBufferStats statistics() const;

    // Clears samples and counters. Coordinate stream resets with all consumers.
    void clear();

private:
    const ImuBufferConfig config_;
    mutable std::mutex mutex_;
    std::deque<ImuSamplePtr> samples_;
    ImuBufferStats statistics_;
};

}  // namespace tinynav::frontend
