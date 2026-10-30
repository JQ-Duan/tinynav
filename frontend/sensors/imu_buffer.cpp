#include "frontend/sensors/imu_buffer.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tinynav::frontend {
namespace {

bool valid_sample(const ImuSamplePtr& sample) {
    if (!sample || sample->header.stamp.nanosec >= 1'000'000'000 ||
        sample->linear_acceleration_covariance[0] == -1.0 ||
        sample->angular_velocity_covariance[0] == -1.0) {
        return false;
    }
    const auto& acceleration = sample->linear_acceleration;
    const auto& angular_velocity = sample->angular_velocity;
    return std::isfinite(acceleration.x) && std::isfinite(acceleration.y) &&
           std::isfinite(acceleration.z) && std::isfinite(angular_velocity.x) &&
           std::isfinite(angular_velocity.y) && std::isfinite(angular_velocity.z);
}

std::uint64_t ordered_distance(TimestampNs earlier, TimestampNs later) {
    return static_cast<std::uint64_t>(later) - static_cast<std::uint64_t>(earlier);
}

}  // namespace

TimestampNs sensor_timestamp_ns(const builtin_interfaces::msg::Time& stamp) {
    if (stamp.nanosec >= 1'000'000'000) {
        throw std::invalid_argument("Sensor timestamp nanoseconds must be below one second");
    }
    // int32 seconds and uint32 subsecond nanoseconds fit within int64 nanoseconds.
    return static_cast<TimestampNs>(stamp.sec) * 1'000'000'000 + stamp.nanosec;
}

ImuBuffer::ImuBuffer(ImuBufferConfig config) : config_(std::move(config)) {
    if (config_.capacity < 2) {
        throw std::invalid_argument("IMU history capacity must be at least two");
    }
    if (config_.history_duration_ns && *config_.history_duration_ns <= 0) {
        throw std::invalid_argument("IMU history duration must be positive");
    }
    if (config_.max_gap_ns && *config_.max_gap_ns <= 0) {
        throw std::invalid_argument("IMU maximum gap must be positive");
    }
}

ImuBuffer::~ImuBuffer() = default;

ImuPushStatus ImuBuffer::push(ImuSamplePtr sample) {
    const bool valid = valid_sample(sample);
    const std::lock_guard<std::mutex> lock(mutex_);
    ++statistics_.received;
    if (!valid) {
        ++statistics_.rejected;
        return ImuPushStatus::InvalidSample;
    }
    const auto stamp = sensor_timestamp_ns(sample->header.stamp);
    if (!samples_.empty()) {
        const auto last = sensor_timestamp_ns(samples_.back()->header.stamp);
        if (stamp <= last) {
            ++statistics_.rejected;
            return stamp == last ? ImuPushStatus::DuplicateTimestamp
                                 : ImuPushStatus::OutOfOrder;
        }
    }

    samples_.push_back(std::move(sample));
    ++statistics_.accepted;
    std::size_t evicted = 0;
    while (samples_.size() > config_.capacity) {
        samples_.pop_front();
        ++evicted;
    }
    if (config_.history_duration_ns &&
        stamp >= std::numeric_limits<TimestampNs>::min() +
                     *config_.history_duration_ns) {
        const auto cutoff = stamp - *config_.history_duration_ns;
        while (samples_.size() > 1 &&
               sensor_timestamp_ns(samples_[1]->header.stamp) <= cutoff) {
            samples_.pop_front();
            ++evicted;
        }
    }
    statistics_.evicted += evicted;
    return evicted == 0 ? ImuPushStatus::Accepted : ImuPushStatus::AcceptedWithEviction;
}

ImuReadResult ImuBuffer::read_interval(TimestampNs start_ns,
                                     TimestampNs end_ns) const {
    ImuReadResult result;
    result.interval.start_ns = start_ns;
    result.interval.end_ns = end_ns;
    if (end_ns < start_ns) {
        result.status = ImuReadStatus::InvalidInterval;
        return result;
    }
    const std::lock_guard<std::mutex> lock(mutex_);
    if (samples_.empty()) {
        return result;
    }
    if (start_ns < sensor_timestamp_ns(samples_.front()->header.stamp)) {
        result.status = ImuReadStatus::HistoryUnavailable;
        return result;
    }
    if (end_ns > sensor_timestamp_ns(samples_.back()->header.stamp)) {
        result.status = ImuReadStatus::AwaitingData;
        return result;
    }

    const auto before = [](const ImuSamplePtr& sample, TimestampNs stamp) {
        return sensor_timestamp_ns(sample->header.stamp) < stamp;
    };
    auto first = std::lower_bound(samples_.begin(), samples_.end(), start_ns, before);
    if (sensor_timestamp_ns((*first)->header.stamp) > start_ns) {
        --first;
    }
    const auto last = std::lower_bound(first, samples_.end(), end_ns, before);
    if (config_.max_gap_ns) {
        for (auto current = first; current != last; ++current) {
            const auto next = std::next(current);
            if (ordered_distance(sensor_timestamp_ns((*current)->header.stamp),
                                 sensor_timestamp_ns((*next)->header.stamp)) >
                static_cast<std::uint64_t>(*config_.max_gap_ns)) {
                result.status = ImuReadStatus::GapTooLarge;
                return result;
            }
        }
    }
    result.interval.samples.assign(first, std::next(last));
    result.status = ImuReadStatus::Ready;
    return result;
}

std::size_t ImuBuffer::size() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return samples_.size();
}

std::optional<TimestampNs> ImuBuffer::oldest_timestamp() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (samples_.empty()) {
        return std::nullopt;
    }
    return sensor_timestamp_ns(samples_.front()->header.stamp);
}

std::optional<TimestampNs> ImuBuffer::latest_timestamp() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (samples_.empty()) {
        return std::nullopt;
    }
    return sensor_timestamp_ns(samples_.back()->header.stamp);
}

ImuBufferStats ImuBuffer::statistics() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    auto result = statistics_;
    result.stored = samples_.size();
    return result;
}

void ImuBuffer::clear() {
    const std::lock_guard<std::mutex> lock(mutex_);
    samples_.clear();
    statistics_ = {};
}

}  // namespace tinynav::frontend
