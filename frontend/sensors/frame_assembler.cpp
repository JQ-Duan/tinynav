#include "frontend/sensors/frame_assembler.hpp"

#include <stdexcept>
#include <utility>

namespace tinynav::frontend {

FrameAssembler::FrameAssembler(ImuBuffer& imu_buffer, FrameAssemblerConfig config)
    : imu_buffer_(imu_buffer), config_(config) {
    if (config_.queue_capacity == 0) {
        throw std::invalid_argument("Frame queue capacity must be positive");
    }
    if (config_.synchronization_tolerance_ns < 0) {
        throw std::invalid_argument("Synchronization tolerance cannot be negative");
    }
}

FrameAssembler::~FrameAssembler() = default;

FramePushStatus FrameAssembler::push_left(ImagePtr image) {
    return push(std::move(image), left_images_, last_left_time_);
}

FramePushStatus FrameAssembler::push_right(ImagePtr image) {
    return push(std::move(image), right_images_, last_right_time_);
}

FramePushStatus FrameAssembler::push_depth(ImagePtr image) {
    return push(std::move(image), depth_images_, last_depth_time_);
}

FramePushStatus FrameAssembler::push(ImagePtr image,
                                     std::deque<QueuedImage>& queue,
                                     std::optional<TimestampNs>& last_time) {
    if (!image || image->header.stamp.nanosec >= 1'000'000'000) {
        ++statistics_.rejected_messages;
        return FramePushStatus::InvalidMessage;
    }
    const auto stamp = sensor_timestamp_ns(image->header.stamp);
    if (last_time && stamp <= *last_time) {
        ++statistics_.rejected_messages;
        return stamp == *last_time ? FramePushStatus::DuplicateTimestamp
                                  : FramePushStatus::OutOfOrder;
    }
    queue.push_back({stamp, std::move(image)});
    last_time = stamp;
    if (queue.size() > config_.queue_capacity) {
        queue.pop_front();
        ++statistics_.evicted_images;
        return FramePushStatus::AcceptedWithEviction;
    }
    return FramePushStatus::Accepted;
}

bool FrameAssembler::too_early(TimestampNs candidate, TimestampNs anchor) const {
    return candidate < anchor &&
           static_cast<std::uint64_t>(anchor) - static_cast<std::uint64_t>(candidate) >
               static_cast<std::uint64_t>(config_.synchronization_tolerance_ns);
}

bool FrameAssembler::too_late(TimestampNs candidate, TimestampNs anchor) const {
    return candidate > anchor &&
           static_cast<std::uint64_t>(candidate) - static_cast<std::uint64_t>(anchor) >
               static_cast<std::uint64_t>(config_.synchronization_tolerance_ns);
}

void FrameAssembler::discard_early(std::deque<QueuedImage>& queue,
                                    TimestampNs anchor) {
    while (!queue.empty() && too_early(queue.front().timestamp_ns, anchor)) {
        queue.pop_front();
        ++statistics_.unmatched_images;
    }
}

void FrameAssembler::remove_matched_images() {
    left_images_.pop_front();
    right_images_.pop_front();
    depth_images_.pop_front();
}

FrameAssemblyResult FrameAssembler::peek_frame() {
    if (pending_) {
        return {FrameAssemblyStatus::Ready, pending_};
    }

    while (!candidate_) {
        if (left_images_.empty()) {
            return {};
        }
        const auto stamp = left_images_.front().timestamp_ns;
        discard_early(right_images_, stamp);
        discard_early(depth_images_, stamp);
        if (right_images_.empty() || depth_images_.empty()) {
            return {};
        }
        if (too_late(right_images_.front().timestamp_ns, stamp) ||
            too_late(depth_images_.front().timestamp_ns, stamp)) {
            left_images_.pop_front();
            ++statistics_.unmatched_images;
            continue;
        }
        if (committed_time_ && stamp <= *committed_time_) {
            remove_matched_images();
            ++statistics_.discarded_frames;
            return {FrameAssemblyStatus::NonIncreasingFrameTime, std::nullopt};
        }
        candidate_ = CameraCandidate{stamp, left_images_.front().message,
                                     right_images_.front().message,
                                     depth_images_.front().message};
        remove_matched_images();
    }

    auto imu = imu_buffer_.read_interval(
        committed_time_.value_or(candidate_->timestamp_ns), candidate_->timestamp_ns);
    switch (imu.status) {
    case ImuReadStatus::Empty:
    case ImuReadStatus::AwaitingData:
        return {FrameAssemblyStatus::WaitingForImu, std::nullopt};
    case ImuReadStatus::HistoryUnavailable:
        return {FrameAssemblyStatus::ImuHistoryUnavailable, std::nullopt};
    case ImuReadStatus::GapTooLarge:
        return {FrameAssemblyStatus::ImuGapTooLarge, std::nullopt};
    case ImuReadStatus::InvalidInterval:
        return {FrameAssemblyStatus::InvalidImuInterval, std::nullopt};
    case ImuReadStatus::Ready:
        break;
    }

    if (next_frame_id_ == 0) {
        throw std::overflow_error("Frame identifiers exhausted");
    }
    SynchronizedFrame frame;
    frame.id = next_frame_id_++;
    frame.timestamp_ns = candidate_->timestamp_ns;
    frame.first_frame = !committed_time_.has_value();
    frame.left = std::move(candidate_->left);
    frame.right = std::move(candidate_->right);
    frame.depth = std::move(candidate_->depth);
    frame.imu = std::move(imu.interval);
    pending_ = std::move(frame);
    candidate_.reset();
    return {FrameAssemblyStatus::Ready, pending_};
}

bool FrameAssembler::commit_frame(std::uint64_t id) {
    if (!pending_ || pending_->id != id) {
        return false;
    }
    committed_time_ = pending_->timestamp_ns;
    pending_.reset();
    ++statistics_.committed_frames;
    return true;
}

bool FrameAssembler::discard_frame(std::uint64_t id) {
    if (!pending_ || pending_->id != id) {
        return false;
    }
    pending_.reset();
    ++statistics_.discarded_frames;
    return true;
}

void FrameAssembler::reset(std::optional<TimestampNs> anchor) {
    left_images_.clear();
    right_images_.clear();
    depth_images_.clear();
    last_left_time_.reset();
    last_right_time_.reset();
    last_depth_time_.reset();
    committed_time_ = anchor;
    candidate_.reset();
    pending_.reset();
    statistics_ = {};
}

FrameAssemblerStats FrameAssembler::statistics() const {
    return statistics_;
}

}  // namespace tinynav::frontend
