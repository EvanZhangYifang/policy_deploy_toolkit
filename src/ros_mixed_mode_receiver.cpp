#include "ros_mixed_mode_receiver.h"

namespace policy_deploy {

RosMixedModeReceiver::RosMixedModeReceiver(rclcpp::Node& node) : node_(node)
{
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
