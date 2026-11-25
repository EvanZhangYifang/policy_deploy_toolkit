#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "deploy_onnx.h"
#include "ros_adaptation_receiver.h"
#include "ros_command_receiver.h"
#include "ros_sensor_receiver.h"
#include "ros_mixed_mode_receiver.h"
#include "mixed_mode_transition.h"
#include "mixed_startup_guard.h"
#include "mixed_startup_mode.h"
#include "manual_mode_request.h"
#include "ros_policy_term_publisher.h"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <xbot2_interface/robotinterface2.h>
#include <xbot2_interface/ros2/config_from_param.hpp>

namespace {

using namespace std::chrono_literals;
using policy_deploy::RosCommandReceiver;
using policy_deploy::RosSensorReceiver;

class PolicyDeployNode final : public rclcpp::Node {
public:
    PolicyDeployNode()
        : rclcpp::Node("policy_deploy_node")
    {
        _node_start_time = now();
        _imu_name = declare_parameter<std::string>("imu_name", "imu_link");
        _obs_group_override = declare_parameter<std::string>("obs_group_override", "");
        _allow_missing_robot_joints = declare_parameter<bool>("allow_missing_robot_joints", false);
        _command_timeout_s = declare_parameter<double>("command_timeout_s", 0.5);
        _sensor_timeout_s = declare_parameter<double>("sensor_timeout_s", 0.5);
        _mixed_initial_mode = policy_deploy::parse_mixed_initial_mode(
            declare_parameter<std::string>("initial_mode", "bb"));

        RCLCPP_INFO(get_logger(), "Obs group override: %s", _obs_group_override.c_str());
        RCLCPP_INFO(get_logger(), "Allow missing robot joints: %s", _allow_missing_robot_joints ? "true" : "false");

        if(_command_timeout_s < 0.0)
        {
            throw std::runtime_error("Parameter 'command_timeout_s' must be non-negative");
        }

        if(_sensor_timeout_s < 0.0)
        {
            throw std::runtime_error("Parameter 'sensor_timeout_s' must be non-negative");
        }
    }

    void init(const std::string& model_path, const std::string& model_metadata_path)
    {
        // build robotinterface
        auto cfg = XBot::ConfigOptionsFromParams(shared_from_this(), "xbotcore/", 2s);
        _robot = XBot::RobotInterface::getRobot(cfg);
        _imu = _robot->getImu(_imu_name);
        if(!_imu)
        {
            throw std::runtime_error(
                std::format("IMU sensor '{}' not found in robot model", _imu_name));
        }

        // v index to joint index (needed for control mode mapping)
        // note: for simplicity, the policy only deals with v indices, whereas
        // control modes are indexed by joint indices
        _vid_to_jid.assign(_robot->getNv(), -1);
        for(int jid = 0; jid < _robot->getJointNum(); ++jid)    {
            int vid = _robot->getVIndexFromVName(_robot->getJointNames()[jid]);
            if(vid >= 0)
            {
                _vid_to_jid[vid] = jid;
            }
        }

        // fill robot info for policy
        XBot::policy::RobotInfo robot_info;
        robot_info.joint_names = _robot->getVNames();
        _robot->getPose(_imu->getName(), "base_link", robot_info.base_T_imu);
        for(const auto& [name, link] : _robot->getUrdf()->links_)
        {
            robot_info.body_names.push_back(name);
        }
        robot_info.body_pose_in_base = [this](const std::string& name) {
            return _robot->getPose(name, "base_link");
        };

        double joint_velocity_limit_scale = declare_parameter<double>("joint_velocity_limit_scale", 1.0);
        RCLCPP_INFO(get_logger(), "Joint velocity limit scale: %f", joint_velocity_limit_scale);

        Eigen::VectorXd qmin, qmax;
        _robot->getJointLimits(qmin, qmax);
        robot_info.joint_pos_min = qmin.cast<float>();
        robot_info.joint_pos_max = qmax.cast<float>();
        Eigen::VectorXd vmax;
        _robot->getVelocityLimits(vmax);
        robot_info.joint_vel_max = vmax.cast<float>() * joint_velocity_limit_scale;

        // build policy wrapper
        _policy = std::make_unique<XBot::policy::OnnxPolicy>(model_path, model_metadata_path, _obs_group_override, robot_info, _allow_missing_robot_joints);
        if(_policy->is_mixed_mode())
        {
            _mixed_info = _policy->policyInfo();
            _base_R_imu = robot_info.base_T_imu.linear();
            std::array<Eigen::VectorXd, 4> references;
            std::array<bool, 4> found{};
            for(const auto& mode : _mixed_info.mixed_modes)
            {
                const int mode_id = (mode.selectors[0] > 0 ? 2 : 0) + (mode.selectors[1] > 0 ? 1 : 0);
                if(found[mode_id] || mode.joint_reference_rad.size() != 12)
                    throw std::runtime_error("Invalid mixed policy mode table");
                found[mode_id] = true;
                references[mode_id] = Eigen::Map<const Eigen::VectorXd>(mode.joint_reference_rad.data(), 12);
                _mixed_selectors[mode_id] = mode.selectors;
            }
            if(!std::all_of(found.begin(), found.end(), [](bool x) { return x; }) ||
               _mixed_info.joint_id_policy_to_robot.size() != 12)
                throw std::runtime_error("Mixed policy requires four modes and twelve leg joints");
            _mixed_lower.resize(12); _mixed_upper.resize(12); _mixed_velocity_limit.resize(12);
            for(int i = 0; i < 12; ++i)
            {
                const int robot_id = _mixed_info.joint_id_policy_to_robot[i];
                if(robot_id < 0 || robot_id >= robot_info.joint_pos_min.size())
                    throw std::runtime_error("All mixed policy leg joints must be present on the robot");
                _mixed_lower[i] = robot_info.joint_pos_min[robot_id];
                _mixed_upper[i] = robot_info.joint_pos_max[robot_id];
                _mixed_velocity_limit[i] = robot_info.joint_vel_max[robot_id];
            }
            _mixed_references = references;
            _mixed_transition = std::make_unique<policy_deploy::MixedModeTransition>(_mixed_info.control_dt, references);
            _mixed_startup_guard = std::make_unique<policy_deploy::MixedStartupGuard>(_mixed_info.control_dt);
            _mixed_receiver = std::make_unique<policy_deploy::RosMixedModeReceiver>(*this);
            _mixed_status_publisher = create_publisher<std_msgs::msg::String>("~/transition/status", 10);
            RCLCPP_WARN(get_logger(), "Operator-confirmed transitions enabled: foot contact and body linear velocity are NOT monitored. Confirm a standing, stationary robot before each mode request.");
            if(_mixed_initial_mode < 0)
                RCLCPP_INFO(get_logger(), "Mixed policy: waiting for starting-mode selector and valid feedback");
            else
                RCLCPP_INFO(get_logger(), "Mixed policy: initial mode index %d; waiting for valid feedback", _mixed_initial_mode);
        }

        // construct policy wrapper outputs
        _outputs = std::make_unique<XBot::policy::Outputs>(
            _policy->policyInfo().action_size,
            robot_info.joint_names.size());

        _command_receiver = std::make_unique<RosCommandReceiver>(*this, *_policy, _command_timeout_s);
        _adaptation_receiver = std::make_unique<policy_deploy::RosAdaptationReceiver>(*this);
        _sensor_receiver = std::make_unique<RosSensorReceiver>(*this, *_policy, _sensor_timeout_s, robot_info.base_T_imu.linear());
        _observation_publisher = create_publisher<std_msgs::msg::Float32MultiArray>("~/policy/observations", 10);
        _action_publisher = create_publisher<std_msgs::msg::Float32MultiArray>("~/policy/actions", 10);
        _observation_term_publisher = std::make_unique<policy_deploy::RosPolicyTermPublisher>(
            *this, "~/policy/observations", _policy->observation_specs());
        _action_term_publisher = std::make_unique<policy_deploy::RosPolicyTermPublisher>(
            *this, "~/policy/actions", _policy->action_specs());
        _q_ref_publisher = create_publisher<sensor_msgs::msg::JointState>(
            "~/policy/q_ref", rclcpp::QoS(10).reliable());
        const auto& recurrent_state_names = _policy->recurrent_state_names();
        for(std::size_t i = 0; i < recurrent_state_names.size(); ++i)
        {
            const auto topic = "~/policy/hidden_states/state_" + std::to_string(i);
            _hidden_state_publishers.push_back(
                create_publisher<std_msgs::msg::Float32MultiArray>(topic, 10));
            RCLCPP_INFO(get_logger(), "Publishing recurrent state '%s' on '%s'",
                        recurrent_state_names[i].c_str(), topic.c_str());
        }

        // buffers
        _vec_nq.resize(_robot->getNq());
        _vec_nj.setZero(_robot->getJointNum());

        const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(_policy->policyInfo().control_dt));

        _timer = create_timer(period, [this]() { control_timer_callback(); });
    }

private:
    void publish_mixed_status(bool feedback_valid)
    {
        std_msgs::msg::String msg;
        msg.data = "phase=" + _mixed_transition->phase() +
                   " active_mode=" + std::to_string(_mixed_transition->active_mode()) +
                   " feedback=" + (feedback_valid ? "valid" : "invalid_or_stale") +
                   " support_check=manual_not_monitored" +
                   " blend=" + std::to_string(_mixed_transition->policy_blend()) +
                   " reason=" + _mixed_transition->status();
        _mixed_status_publisher->publish(msg);
    }

    bool prepare_mixed_step()
    {
        const auto sample = _mixed_receiver->snapshot();
        policy_deploy::TransitionInput input;
        input.joint_position.resize(12); input.joint_velocity.resize(12);
        Eigen::VectorXd last_target(12);
        bool finite_joints = true;
        for(int i = 0; i < 12; ++i)
        {
            const int id = _mixed_info.joint_id_policy_to_robot[i];
            input.joint_position[i] = _inputs.q[id];
            input.joint_velocity[i] = _inputs.v[id];
            last_target[i] = _outputs->q_des[id];
            finite_joints = finite_joints && std::isfinite(input.joint_position[i]) &&
                            std::isfinite(input.joint_velocity[i]) && std::isfinite(last_target[i]);
        }
        const double imu_norm = _inputs.w_R_imu.norm();
        const bool imu_valid = _inputs.w_R_imu.coeffs().allFinite() && std::isfinite(imu_norm) &&
                               imu_norm > 1e-12 && _inputs.imu_omega.allFinite();
        if(imu_valid)
        {
            const Eigen::Matrix3d world_R_base = _inputs.w_R_imu.normalized().toRotationMatrix() * _base_R_imu.transpose();
            input.gravity_body = world_R_base.transpose() * Eigen::Vector3d(0, 0, -1);
            input.angular_velocity_body = _base_R_imu * _inputs.imu_omega;
        }
        auto command = _inputs.command.find("base_velocity");
        const bool command_valid = command != _inputs.command.end() && command->second.size() == 3 &&
                                   command->second.allFinite();
        if(command_valid) input.requested_velocity = command->second;
        input.valid = finite_joints && imu_valid && command_valid;

        if(_mixed_transition->active_mode() < 0)
        {
            const auto startup = policy_deploy::select_mixed_startup_mode(
                _mixed_initial_mode, sample.requested_mode, sample.request_sequence);
            // A visual check is specific to an instant. A request published
            // before startup completes must not trigger a later transition.
            if(_mixed_initial_mode >= 0 && sample.request_sequence != _mixed_request_sequence)
            {
                _mixed_request_sequence = sample.request_sequence;
                RCLCPP_WARN(get_logger(), "Ignoring mode request before startup is ready; visually check stance and send it again when idle");
            }
            if(!input.valid || startup.mode < 0)
            {
                _mixed_startup_guard->reset();
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                    "Waiting for initial mode and valid required feedback");
                publish_mixed_status(input.valid);
                return false;
            }
            const auto& reference = _mixed_references[startup.mode];
            if(!_mixed_startup_guard->update(startup.mode, input, reference, last_target,
                                             std::chrono::steady_clock::now()))
            {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "%s", _mixed_startup_guard->reason().c_str());
                publish_mixed_status(input.valid);
                return false;
            }
            _mixed_transition->initialize(startup.mode, last_target);
            // Prior requests are discarded when a mode is configured: an
            // earlier visual confirmation may no longer be valid.
            _mixed_request_sequence = _mixed_initial_mode >= 0
                ? sample.request_sequence : startup.consumed_sequence;
        }
        if(sample.request_sequence != _mixed_request_sequence)
        {
            _mixed_request_sequence = sample.request_sequence;
            const auto decision = policy_deploy::manual_mode_request(
                input.valid, _mixed_transition->phase() == "idle",
                input.requested_velocity.cwiseAbs().maxCoeff());
            if(decision == policy_deploy::ManualRequestDecision::feedback_invalid)
                RCLCPP_WARN(get_logger(), "Ignoring mode request: required feedback is invalid; visually check stance and retry");
            else if(decision == policy_deploy::ManualRequestDecision::busy)
                RCLCPP_WARN(get_logger(), "Ignoring mode request: transition not idle; visually check stance and retry when idle");
            else if(decision == policy_deploy::ManualRequestDecision::command_nonzero)
                RCLCPP_WARN(get_logger(), "Ignoring mode request: commanded velocity must be zero; visually confirm the robot is stationary and retry");
            else if(decision == policy_deploy::ManualRequestDecision::accept &&
                    sample.requested_mode != _mixed_transition->active_mode())
                _mixed_transition->request(sample.requested_mode);
        }
        const auto gated_command = _mixed_transition->advance(input);
        publish_mixed_status(input.valid);
        if(!input.valid)
        {
            RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
                "Mixed policy stopped: required feedback is invalid or stale");
            return false;
        }
        command->second = gated_command;
        _inputs.mode_selectors = _mixed_selectors[_mixed_transition->active_mode()];
        return true;
    }

    void filter_mixed_outputs()
    {
        Eigen::VectorXd target(12);
        for(int i = 0; i < 12; ++i)
            target[i] = _outputs->q_des[_mixed_info.joint_id_policy_to_robot[i]];
        const auto filtered = _mixed_transition->filter(target, _mixed_lower, _mixed_upper, _mixed_velocity_limit);
        for(int i = 0; i < 12; ++i)
            _outputs->q_des[_mixed_info.joint_id_policy_to_robot[i]] = filtered[i];
    }

    void fill_ctrl_mode()
    {
        for(int vi = 0; vi < _robot->getNv(); ++vi)
        {
            int jid = _vid_to_jid[vi];
            if(jid >= 0)
            {
                _vec_nj(jid) = _outputs->ctrl_mode(vi);
            }
        }
    }

    void control_timer_callback()
    {
        auto& outputs = *_outputs;
        auto t_start = std::chrono::steady_clock::now();

        // wait for robot to be updated
        while(rclcpp::ok() && !_robot->sense())
        {
            std::this_thread::sleep_for(1ms);
        }

        if(!rclcpp::ok())
        {
            return;
        }

        const auto feedback_sample_time = now();
        _robot->positionToMinimal(_robot->getPositionReferenceFeedback(), _inputs.q_ref);
        const int offset = _robot->isFloatingBase() ? 6 : 0;
        const auto& names = _robot->getVNames();
        sensor_msgs::msg::JointState q_ref_msg;
        q_ref_msg.header.stamp = feedback_sample_time;
        q_ref_msg.name.assign(names.begin() + offset, names.end());
        q_ref_msg.position.assign(_inputs.q_ref.data() + offset,
                                  _inputs.q_ref.data() + _inputs.q_ref.size());
        _q_ref_publisher->publish(q_ref_msg);

        _inputs.w_R_imu = _imu->getOrientation();
        _sensor_receiver->update_imu(_inputs.w_R_imu);

        // fill inputs for policy
        _inputs.last_action = outputs.raw_action; // for the first iteration, last action is zero
        const auto ros_now = now();
        _command_receiver->fill_inputs(_inputs, ros_now);
        if(!_sensor_receiver->fill_inputs(_inputs, ros_now))
        {
            return;
        }
        _inputs.time_sec = (now() - _node_start_time).seconds();
        _inputs.q = _robot->getJointPositionMinimal();
        _inputs.v = _robot->getJointVelocity();
        _inputs.tau = _robot->getJointEffort();
        _inputs.v_ref = _robot->getVelocityReferenceFeedback();
        _inputs.tau_ref = _robot->getEffortReferenceFeedback();
        _inputs.k = _robot->getStiffness();
        _inputs.d = _robot->getDamping();
        _inputs.imu_omega = _imu->getAngularVelocity();
        _inputs.imu_acc = _imu->getLinearAcceleration();

        // initialize outputs as needed
        if(_inputs.step == 0)
        {
            outputs.q_des = _inputs.q_ref;
        }

        _adaptation_receiver->fill_inputs(_inputs);

        // Mixed-mode selection and guarded transitions run before observation assembly.
        if(_mixed_transition && !prepare_mixed_step())
        {
            return;
        }

        // run policy, then apply the standing transition's physical target filter
        if(!_policy->run(_inputs, outputs))
        {
            RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "Policy inference failed; holding last reference");
            return;
        }
        if(_mixed_transition) filter_mixed_outputs();

        // increment step counter
        _inputs.step++;

        std_msgs::msg::Float32MultiArray observation_msg;
            observation_msg.data.assign(
                outputs.raw_observation.data(),
                outputs.raw_observation.data() + outputs.raw_observation.size());
        _observation_publisher->publish(observation_msg);

        std_msgs::msg::Float32MultiArray action_msg;
        action_msg.data.resize(outputs.raw_action.size());
        for(Eigen::Index i = 0; i < outputs.raw_action.size(); ++i)
        {
            action_msg.data[i] = static_cast<float>(outputs.raw_action[i]);
        }
        _action_publisher->publish(action_msg);
        _observation_term_publisher->publish(outputs.raw_observation);
        _action_term_publisher->publish(outputs.raw_action);

        const auto& recurrent_state_names = _policy->recurrent_state_names();
        for(std::size_t i = 0; i < outputs.raw_recurrent_states.size(); ++i)
        {
            std_msgs::msg::Float32MultiArray state_msg;
                const auto& recurrent_state = outputs.raw_recurrent_states[i];
                state_msg.data.assign(recurrent_state.data(), recurrent_state.data() + recurrent_state.size());
            std_msgs::msg::MultiArrayDimension state_dimension;
            state_dimension.label = recurrent_state_names[i];
            state_dimension.size = static_cast<uint32_t>(state_msg.data.size());
            state_dimension.stride = state_dimension.size;
            state_msg.layout.dim.push_back(state_dimension);
            _hidden_state_publishers[i]->publish(state_msg);
        }

        //
        fill_ctrl_mode();

        // send commands to robot
        _robot->setControlMode(_vec_nj);
        _robot->minimalToPosition(outputs.q_des, _vec_nq);
        _robot->setPositionReference(_vec_nq);
        _robot->setVelocityReference(outputs.v_des);
        _robot->setEffortReference(outputs.tau_des);
        if(!_policy_start_time_initialized)
        {
            // Startup may wait indefinitely for feedback or a valid pose. Start
            // the gain window only after the first successful policy step.
            _loop_start = t_start;
            _policy_start_time_initialized = true;
        }
        if(t_start - _loop_start < 1s)
        {
            // note: we stop sending impedance after one second,
            // so we can tune it online from a separate node (e.g. gui)
            _robot->setStiffness(outputs.k_des);
            _robot->setDamping(outputs.d_des);
        }
#ifdef ENABLE_COMMAND_TIMESTAMPS
        _robot->setCommandTimestamp(_robot->getStateTimestamp());
#endif
        _robot->move();
    }

    XBot::RobotInterface::UniquePtr _robot;
    XBot::ImuSensor::ConstPtr _imu;
    std::string _imu_name;
    std::string _obs_group_override;
    bool _allow_missing_robot_joints{false};
    double _command_timeout_s{0.0};
    double _sensor_timeout_s{0.0};
    int _mixed_initial_mode{-1};
    XBot::policy::PolicyInfo _mixed_info{};
    Eigen::Matrix3d _base_R_imu = Eigen::Matrix3d::Identity();
    std::array<Eigen::VectorXd, 4> _mixed_references;
    std::array<Eigen::Vector2d, 4> _mixed_selectors;
    Eigen::VectorXd _mixed_lower, _mixed_upper, _mixed_velocity_limit;
    std::unique_ptr<policy_deploy::RosMixedModeReceiver> _mixed_receiver;
    std::unique_ptr<policy_deploy::MixedModeTransition> _mixed_transition;
    std::unique_ptr<policy_deploy::MixedStartupGuard> _mixed_startup_guard;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr _mixed_status_publisher;
    std::uint64_t _mixed_request_sequence{0};
    std::unique_ptr<XBot::policy::OnnxPolicy> _policy;
    std::unique_ptr<RosCommandReceiver> _command_receiver;
    std::unique_ptr<policy_deploy::RosAdaptationReceiver> _adaptation_receiver;
    std::unique_ptr<RosSensorReceiver> _sensor_receiver;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr _observation_publisher;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr _action_publisher;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr _q_ref_publisher;
    std::vector<rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr> _hidden_state_publishers;
    std::unique_ptr<policy_deploy::RosPolicyTermPublisher> _observation_term_publisher;
    std::unique_ptr<policy_deploy::RosPolicyTermPublisher> _action_term_publisher;
    XBot::policy::Inputs _inputs;
    std::unique_ptr<XBot::policy::Outputs> _outputs;
    std::vector<int> _vid_to_jid;
    rclcpp::Time _node_start_time{0, 0, RCL_ROS_TIME};
    Eigen::VectorXd _vec_nq;
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> _vec_nj;
    std::chrono::steady_clock::time_point _loop_start;
    bool _policy_start_time_initialized{false};
    rclcpp::TimerBase::SharedPtr _timer;
};

} // namespace

int main(int argc, char *argv[]) {

    rclcpp::init(argc, argv);

    if(argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <model_metadata_path>" << std::endl;
        rclcpp::shutdown();
        return 1;
    }

    try
    {
        auto node = std::make_shared<PolicyDeployNode>();
        node->init(argv[1], argv[2]);
        rclcpp::spin(node);
    }
    catch(const std::exception& e)
    {
        std::cerr << "xbot2_deploy_policy: " << e.what() << std::endl;
        rclcpp::shutdown();
        return 1;
    }

    rclcpp::shutdown();
    return 0;

}
