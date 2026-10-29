#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>

namespace tinynav::frontend {

using ImageMessage = sensor_msgs::msg::Image;
using ImagePtr = ImageMessage::ConstSharedPtr;
using ImuMessage = sensor_msgs::msg::Imu;
using ImuPtr = ImuMessage::ConstSharedPtr;

struct StreamStatistics {
    std::uint64_t received;
    std::uint64_t dropped;
    std::uint64_t rejected;
    std::size_t queued;
};

struct SensorStatistics {
    StreamStatistics left_image;
    StreamStatistics right_image;
    StreamStatistics depth;
    StreamStatistics imu;
};

// Receives raw ROS messages without changing their timestamps, units, or data.
// Queues retain the newest messages when full. dropped counts application
// evictions only; DDS losses cannot be measured from these queues.
//
// These are ingress queues, not IMU history. One consumer should read the IMU
// queue and distribute its samples to initialization, VIO, and propagation.
// Queue reads and statistics are safe alongside subscription callbacks.
// Topic, capacity, and QoS parameters are startup-only (read-only afterward).
// IMU acceleration and angular velocity must be available; orientation need not be.
class SensorReceiver final : public rclcpp::Node {
public:
    SensorReceiver();
    explicit SensorReceiver(const rclcpp::NodeOptions& options);
    ~SensorReceiver() override;

    // FIFO reads remove one message. An empty queue resets output to nullptr.
    bool try_pop_left_image(ImagePtr& output);
    bool try_pop_right_image(ImagePtr& output);
    bool try_pop_depth(ImagePtr& output);
    bool try_pop_imu(ImuPtr& output);

    // received includes rejected messages; rejected messages are never queued.
    SensorStatistics statistics() const;

private:
    void receive_image(ImagePtr message, std::deque<ImagePtr>& queue,
                       StreamStatistics& statistics, bool is_depth);
    void receive_imu(ImuPtr message);
    bool pop_image(std::deque<ImagePtr>& queue, ImagePtr& output);

    std::size_t image_queue_capacity_;
    std::size_t imu_queue_capacity_;
    mutable std::mutex mutex_;
    SensorStatistics statistics_;
    std::deque<ImagePtr> left_images_;
    std::deque<ImagePtr> right_images_;
    std::deque<ImagePtr> depth_images_;
    std::deque<ImuPtr> imu_samples_;
    rclcpp::Subscription<ImageMessage>::SharedPtr left_subscription_;
    rclcpp::Subscription<ImageMessage>::SharedPtr right_subscription_;
    rclcpp::Subscription<ImageMessage>::SharedPtr depth_subscription_;
    rclcpp::Subscription<ImuMessage>::SharedPtr imu_subscription_;
};

}  // namespace tinynav::frontend
