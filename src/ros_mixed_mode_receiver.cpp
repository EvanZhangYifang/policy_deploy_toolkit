#include "ros_mixed_mode_receiver.h"
#include <algorithm>
#include <stdexcept>

namespace policy_deploy {

RosMixedModeReceiver::RosMixedModeReceiver(rclcpp::Node& node) : node_(node)
{
    // Mode order is BB, BF, FB, FF; defaults follow the common Xbox layout.
    button_indices_ = {node_.declare_parameter<int>("mode_button_bb", 2),
                       node_.declare_parameter<int>("mode_button_bf", 3),
                       node_.declare_parameter<int>("mode_button_fb", 1),
                       node_.declare_parameter<int>("mode_button_ff", 0)};
    for (std::size_t i = 0; i < button_indices_.size(); ++i) {
        if (button_indices_[i] < 0 ||
            std::find(button_indices_.begin(), button_indices_.begin() + i, button_indices_[i]) != button_indices_.begin() + i)
            throw std::invalid_argument("Mode button indices must be distinct and nonnegative");
    }
    joy_subscription_ = node_.create_subscription<sensor_msgs::msg::Joy>(
        "/joy", rclcpp::SensorDataQoS(), [this](sensor_msgs::msg::Joy::ConstSharedPtr msg) {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto max_index = *std::max_element(button_indices_.begin(), button_indices_.end());
            if (msg->buttons.size() <= static_cast<std::size_t>(max_index)) {
                RCLCPP_WARN_THROTTLE(node_.get_logger(), *node_.get_clock(), 1000,
                    "Ignoring joystick mode buttons: button array is too short");
                return;
            }
            int pressed = -1;
            int down_count = 0;
            for (std::size_t i = 0; i < button_indices_.size(); ++i) {
                const bool down = msg->buttons[button_indices_[i]] != 0;
                if (down) {
                    ++down_count;
                    if (!button_down_[i]) pressed = static_cast<int>(i);
                }
                button_down_[i] = down;
            }
            // Reject combinations; release all other face buttons before retrying.
            if (down_count == 1 && pressed >= 0) {
                value_.requested_mode = pressed;
                ++value_.request_sequence;
            }
        });
    selector_subscription_ = node_.create_subscription<Array>(
        "~/mode_selectors", rclcpp::QoS(1).best_effort(), [this](Array::ConstSharedPtr msg) {
            if(msg->data.size() != 2 ||
               (msg->data[0] != -1.0 && msg->data[0] != 1.0) ||
               (msg->data[1] != -1.0 && msg->data[1] != 1.0)) {
                RCLCPP_WARN_THROTTLE(node_.get_logger(), *node_.get_clock(), 1000,
                    "Ignoring mode request: expected exactly two values, each -1 or +1");
                return;
            }
            const int mode = (msg->data[0] > 0 ? 2 : 0) + (msg->data[1] > 0 ? 1 : 0);
            std::lock_guard<std::mutex> lock(mutex_);
            // Every valid message is a new confirmation, including same-mode retries.
            value_.requested_mode = mode;
            ++value_.request_sequence;
        });
}

RosMixedModeReceiver::Snapshot RosMixedModeReceiver::snapshot() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return value_;
}

} // namespace policy_deploy
