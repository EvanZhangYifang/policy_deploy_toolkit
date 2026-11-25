#pragma once

#include <cstdint>
#include <mutex>
#include <array>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

namespace policy_deploy {

// A selector message is an operator request, copied atomically for each tick.
class RosMixedModeReceiver {
public:
    struct Snapshot {
        int requested_mode = -1;
        std::uint64_t request_sequence = 0;
    };
    explicit RosMixedModeReceiver(rclcpp::Node& node);
    Snapshot snapshot() const;

private:
    using Array = std_msgs::msg::Float64MultiArray;
    rclcpp::Node& node_;
    mutable std::mutex mutex_;
    Snapshot value_;
    rclcpp::Subscription<Array>::SharedPtr selector_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_subscription_;
    std::array<std::int64_t, 4> button_indices_;
    std::array<bool, 4> button_down_{};
};

} // namespace policy_deploy
