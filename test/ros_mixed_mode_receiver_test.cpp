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
            rclcpp::NodeOptions().use_intra_process_comms(true));
        policy_deploy::RosMixedModeReceiver receiver(*node);
        auto modes = node->create_publisher<Array>("~/mode_selectors", 1);
        rclcpp::executors::SingleThreadedExecutor executor;
        executor.add_node(node);
        auto spin = [&] {
            const auto end = std::chrono::steady_clock::now() + 35ms;
            do { executor.spin_some(); std::this_thread::sleep_for(1ms); }
            while(std::chrono::steady_clock::now() < end);
        };
        auto deadline = std::chrono::steady_clock::now() + 3s;
        while(!modes->get_subscription_count()) {
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
    } catch(const std::exception& e) {
        std::cerr << e.what() << '\n'; rclcpp::shutdown(); return 1;
    }
    rclcpp::shutdown();
}
