#include "ros_mixed_mode_receiver.h"

#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>

using namespace std::chrono_literals;
using Array = std_msgs::msg::Float64MultiArray;

static void check(bool ok, const char* error) {
    if(!ok) throw std::runtime_error(error);
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<rclcpp::Node>("mixed_receiver_test",
            rclcpp::NodeOptions().use_intra_process_comms(true).arguments(
                {"--ros-args", "-r", "/joy:=/mixed_receiver_test_joy"}));
        policy_deploy::RosMixedModeReceiver receiver(*node);
        auto modes = node->create_publisher<Array>("~/mode_selectors", 1);
        auto joy = node->create_publisher<sensor_msgs::msg::Joy>("/joy", rclcpp::SensorDataQoS());
        rclcpp::executors::SingleThreadedExecutor executor;
        executor.add_node(node);
        auto spin = [&] {
            const auto end = std::chrono::steady_clock::now() + 35ms;
            do { executor.spin_some(); std::this_thread::sleep_for(1ms); }
            while(std::chrono::steady_clock::now() < end);
        };
        auto deadline = std::chrono::steady_clock::now() + 3s;
        while(!modes->get_subscription_count() || !joy->get_subscription_count()) {
            check(std::chrono::steady_clock::now() < deadline, "topic discovery failed"); spin();
        }
        check(receiver.snapshot().requested_mode == -1 && receiver.snapshot().request_sequence == 0,
              "receiver should require an explicit selector");
        Array mode; mode.data = {-1, 1}; modes->publish(mode); spin();
        auto s = receiver.snapshot();
        check(s.requested_mode == 1 && s.request_sequence == 1, "BF selector not accepted");
        modes->publish(mode); spin();
        check(receiver.snapshot().request_sequence == 2, "same-mode retry must be delivered");
        mode.data = {0, 1}; modes->publish(mode); spin();
        check(receiver.snapshot().requested_mode == 1 && receiver.snapshot().request_sequence == 2,
              "invalid selector changed request");
        mode.data = {std::numeric_limits<double>::quiet_NaN(), 1}; modes->publish(mode); spin();
        check(receiver.snapshot().request_sequence == 2, "non-finite selector changed request");
        mode.data = {1, -1}; modes->publish(mode); spin();
        check(receiver.snapshot().requested_mode == 2 && receiver.snapshot().request_sequence == 3,
              "FB request missing");
        mode.data = {1, 1}; modes->publish(mode); spin();
        check(receiver.snapshot().requested_mode == 3 && receiver.snapshot().request_sequence == 4,
              "FF request missing");
        mode.data = {-1, -1}; modes->publish(mode); spin();
        check(receiver.snapshot().requested_mode == 0 && receiver.snapshot().request_sequence == 5,
              "BB request missing");
        sensor_msgs::msg::Joy buttons; buttons.buttons = {0, 0, 0, 0};
        joy->publish(buttons); spin();
        const int indices[] = {2, 3, 1, 0};
        for (int i = 0; i < 4; ++i) {
            buttons.buttons[indices[i]] = 1; joy->publish(buttons); spin();
            s = receiver.snapshot();
            check(s.requested_mode == i && s.request_sequence == static_cast<std::uint64_t>(6 + i),
                  "joystick mode request missing");
            joy->publish(buttons); spin();
            check(receiver.snapshot().request_sequence == s.request_sequence, "held button retriggered");
            buttons.buttons[indices[i]] = 0; joy->publish(buttons); spin();
        }
        buttons.buttons = {1, 1, 0, 0}; joy->publish(buttons); spin();
        check(receiver.snapshot().request_sequence == 9, "button combination accepted");
        buttons.buttons = {0, 0, 0, 0}; joy->publish(buttons); spin();
        buttons.buttons = {1}; joy->publish(buttons); spin();
        check(receiver.snapshot().request_sequence == 9, "short joystick message accepted");
        buttons.buttons = {1, 0, 0, 0}; joy->publish(buttons); spin();
        check(receiver.snapshot().request_sequence == 10 && receiver.snapshot().requested_mode == 3,
              "button retry after release missing");
    } catch(const std::exception& e) {
        std::cerr << e.what() << '\n'; rclcpp::shutdown(); return 1;
    }
    rclcpp::shutdown();
}
