#include "frontend/sensors/sensor_receiver.hpp"

#include <rcl_interfaces/msg/parameter_descriptor.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace tinynav::frontend {
namespace {

template <typename Value>
Value startup_parameter(rclcpp::Node& node, const char* name,
                        const Value& default_value) {
    rcl_interfaces::msg::ParameterDescriptor descriptor;
    descriptor.read_only = true;
    descriptor.description = "Applied when the sensor receiver starts.";
    return node.declare_parameter<Value>(name, default_value, descriptor);
}

std::size_t positive_size(rclcpp::Node& node, const char* name,
                          std::int64_t default_value) {
    const auto value = startup_parameter(node, name, default_value);
    if (value <= 0 || static_cast<std::uint64_t>(value) >
                          std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(std::string(name) + " must be a positive size");
    }
    return static_cast<std::size_t>(value);
}

std::string topic_parameter(rclcpp::Node& node, const char* name,
                            const char* default_value) {
    const auto topic = startup_parameter(node, name, std::string(default_value));
    if (topic.empty()) {
        throw std::invalid_argument(std::string(name) + " cannot be empty");
    }
    return topic;
}

rclcpp::QoS sensor_qos(std::size_t depth, bool reliable) {
    rclcpp::QoS qos{rclcpp::KeepLast(depth)};
    qos.durability_volatile();
    if (reliable) {
        qos.reliable();
    } else {
        qos.best_effort();
    }
    return qos;
}

std::size_t pixel_size(const std::string& encoding) {
    // Supported packed formats. No decoding or unit conversion is performed.
    if (encoding == "mono8" || encoding == "8UC1") {
        return 1;
    }
    if (encoding == "mono16" || encoding == "16UC1") {
        return 2;
    }
    if (encoding == "rgb8" || encoding == "bgr8") {
        return 3;
    }
    if (encoding == "rgba8" || encoding == "bgra8" || encoding == "32FC1") {
        return 4;
    }
    return 0;
}

bool valid_image(const ImageMessage& message, bool is_depth) {
    if (message.header.stamp.nanosec >= 1'000'000'000 ||
        message.width == 0 || message.height == 0 || message.is_bigendian > 1) {
        return false;
    }
    if (is_depth && message.encoding != "32FC1" && message.encoding != "16UC1") {
        return false;
    }
    const auto bytes_per_pixel = pixel_size(message.encoding);
    if (bytes_per_pixel == 0) {
        return false;
    }
    const auto row_bytes = static_cast<std::uint64_t>(message.width) * bytes_per_pixel;
    const auto storage_bytes = static_cast<std::uint64_t>(message.step) * message.height;
    // ROS Image data covers step * height, including any row padding.
    return message.step >= row_bytes && message.data.size() >= storage_bytes;
}

bool valid_imu(const ImuMessage& message) {
    const auto& acceleration = message.linear_acceleration;
    const auto& angular_velocity = message.angular_velocity;
    // Orientation is optional and is not used by this input stage.
    return message.header.stamp.nanosec < 1'000'000'000 &&
           message.linear_acceleration_covariance[0] != -1.0 &&
           message.angular_velocity_covariance[0] != -1.0 &&
           std::isfinite(acceleration.x) && std::isfinite(acceleration.y) &&
           std::isfinite(acceleration.z) && std::isfinite(angular_velocity.x) &&
           std::isfinite(angular_velocity.y) && std::isfinite(angular_velocity.z);
}

template <typename MessagePtr>
void enqueue(MessagePtr message, std::deque<MessagePtr>& queue,
             std::size_t capacity, StreamStatistics& statistics) {
    queue.push_back(std::move(message));
    if (queue.size() > capacity) {
        queue.pop_front();
        ++statistics.dropped;
    }
}

template <typename MessagePtr>
bool pop(std::deque<MessagePtr>& queue, MessagePtr& output) {
    if (queue.empty()) {
        output.reset();
        return false;
    }
    output = std::move(queue.front());
    queue.pop_front();
    return true;
}

}  // namespace

SensorReceiver::SensorReceiver() : SensorReceiver(rclcpp::NodeOptions{}) {}

SensorReceiver::SensorReceiver(const rclcpp::NodeOptions& options)
    : rclcpp::Node("tinynav_sensor_receiver", options), statistics_{} {
    const auto left_topic = topic_parameter(
        *this, "left_image_topic", "/camera/camera/infra1/image_rect_raw");
    const auto right_topic = topic_parameter(
        *this, "right_image_topic", "/camera/camera/infra2/image_rect_raw");
    const auto depth_topic = topic_parameter(
        *this, "depth_topic", "/camera/camera/depth/image_rect_raw");
    const auto imu_topic = topic_parameter(*this, "imu_topic", "/camera/camera/imu");

    image_queue_capacity_ = positive_size(*this, "image_queue_capacity", 10);
    imu_queue_capacity_ = positive_size(*this, "imu_queue_capacity", 4096);
    const auto image_qos_depth = positive_size(*this, "image_qos_depth", 10);
    const auto imu_qos_depth = positive_size(*this, "imu_qos_depth", 4096);
    const auto image_reliable = startup_parameter(*this, "image_reliable", true);
    const auto imu_reliable = startup_parameter(*this, "imu_reliable", true);
    const auto image_qos = sensor_qos(image_qos_depth, image_reliable);
    const auto imu_qos = sensor_qos(imu_qos_depth, imu_reliable);

    left_subscription_ = create_subscription<ImageMessage>(
        left_topic, image_qos, [this](ImagePtr message) {
            receive_image(std::move(message), left_images_, statistics_.left_image, false);
        });
    right_subscription_ = create_subscription<ImageMessage>(
        right_topic, image_qos, [this](ImagePtr message) {
            receive_image(std::move(message), right_images_, statistics_.right_image, false);
        });
    depth_subscription_ = create_subscription<ImageMessage>(
        depth_topic, image_qos, [this](ImagePtr message) {
            receive_image(std::move(message), depth_images_, statistics_.depth, true);
        });
    imu_subscription_ = create_subscription<ImuMessage>(
        imu_topic, imu_qos, [this](ImuPtr message) {
            receive_imu(std::move(message));
        });

    RCLCPP_INFO(get_logger(), "Left image: %s", left_topic.c_str());
    RCLCPP_INFO(get_logger(), "Right image: %s", right_topic.c_str());
    RCLCPP_INFO(get_logger(), "Depth: %s", depth_topic.c_str());
    RCLCPP_INFO(get_logger(), "IMU: %s", imu_topic.c_str());
}

SensorReceiver::~SensorReceiver() = default;

void SensorReceiver::receive_image(ImagePtr message, std::deque<ImagePtr>& queue,
                                   StreamStatistics& statistics, bool is_depth) {
    const bool valid = message && valid_image(*message, is_depth);
    const std::lock_guard<std::mutex> lock(mutex_);
    ++statistics.received;
    if (!valid) {
        ++statistics.rejected;
        return;
    }
    enqueue(std::move(message), queue, image_queue_capacity_, statistics);
}

void SensorReceiver::receive_imu(ImuPtr message) {
    const bool valid = message && valid_imu(*message);
    const std::lock_guard<std::mutex> lock(mutex_);
    ++statistics_.imu.received;
    if (!valid) {
        ++statistics_.imu.rejected;
        return;
    }
    enqueue(std::move(message), imu_samples_, imu_queue_capacity_, statistics_.imu);
}

bool SensorReceiver::pop_image(std::deque<ImagePtr>& queue, ImagePtr& output) {
    const std::lock_guard<std::mutex> lock(mutex_);
    return pop(queue, output);
}

bool SensorReceiver::try_pop_left_image(ImagePtr& output) {
    return pop_image(left_images_, output);
}

bool SensorReceiver::try_pop_right_image(ImagePtr& output) {
    return pop_image(right_images_, output);
}

bool SensorReceiver::try_pop_depth(ImagePtr& output) {
    return pop_image(depth_images_, output);
}

bool SensorReceiver::try_pop_imu(ImuPtr& output) {
    const std::lock_guard<std::mutex> lock(mutex_);
    return pop(imu_samples_, output);
}

SensorStatistics SensorReceiver::statistics() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    auto result = statistics_;
    result.left_image.queued = left_images_.size();
    result.right_image.queued = right_images_.size();
    result.depth.queued = depth_images_.size();
    result.imu.queued = imu_samples_.size();
    return result;
}

}  // namespace tinynav::frontend
