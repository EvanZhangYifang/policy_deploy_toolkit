#ifndef __XBOT2_DEPLOY_POLICY_TYPES_H
#define __XBOT2_DEPLOY_POLICY_TYPES_H

#include <Eigen/Dense>
#include <functional>
#include <map>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace XBot::policy {

struct RobotInfo {
    std::vector<std::string> joint_names;
    Eigen::Affine3d base_T_imu;
    Eigen::VectorXf joint_pos_min, joint_pos_max;
    Eigen::VectorXf joint_vel_max;
    // Live link poses in the command/base frame, refreshed by RobotInterface::sense().
    std::vector<std::string> body_names;
    std::function<Eigen::Affine3d(const std::string&)> body_pose_in_base;
};

struct MixedMode {
    std::string name;
    Eigen::Vector2d selectors;
    std::vector<double> joint_reference_rad;
};

struct PolicyInfo {
    double control_dt;
    std::vector<std::string> joint_names;
    std::vector<double> stiffness, damping;
    std::vector<double> joint_default_pos;
    std::vector<double> joint_default_vel;
    std::vector<int> joint_id_policy_to_robot;
    std::vector<int> joint_id_robot_to_policy;
    std::map<std::string, int> command_size;
    int action_size;
    int height_scan_size;
    std::vector<MixedMode> mixed_modes;
    double mixed_action_scale = 0.0;
};

// Returns the unique configured mode for an exact, validated selector pair.
inline const MixedMode& mixed_mode_for_selectors(const PolicyInfo& info, const Eigen::Vector2d& selectors)
{
    for(const auto& mode : info.mixed_modes)
    {
        if((mode.selectors.array() == selectors.array()).all()) return mode;
    }
    throw std::runtime_error("Unrecognized mixed-policy mode selectors");
}

// A named contiguous slice of the model observation or action vector.
struct PolicyTermSpec {
    std::string name;
    int offset;
    int size;
};

struct Inputs {
    uint64_t step = 0;
    double time_sec = 0;

    std::map<std::string, Eigen::VectorXd> command; 
    Eigen::VectorXd last_action;
    // Valid selector is required for mixed policies; no default mode is assumed.
    Eigen::Vector2d mode_selectors = Eigen::Vector2d::Constant(std::numeric_limits<double>::quiet_NaN());

    Eigen::VectorXd q; 
    Eigen::VectorXd v; 
    Eigen::VectorXd tau;

    Eigen::VectorXd q_ref;
    Eigen::VectorXd v_ref;
    Eigen::VectorXd tau_ref;
    Eigen::VectorXd k;
    Eigen::VectorXd d;

    Eigen::Quaterniond w_R_imu;
    Eigen::Vector3d imu_omega;
    Eigen::Vector3d imu_acc;

    Eigen::VectorXd height_scan;

    // [mass delta (kg), CoM offset x/y/z (m), friction foot 1/2/3/4], in training order.
    Eigen::Matrix<double, 8, 1> adaptation_parameters =
        (Eigen::Matrix<double, 8, 1>() << 0., 0., 0., 0., 1., 1., 1., 1.).finished();
};

struct Outputs {
    Eigen::VectorXf raw_observation;
    std::vector<Eigen::VectorXf> raw_recurrent_states;
    Eigen::VectorXd raw_action;

    Eigen::VectorXd q_des;
    Eigen::VectorXd v_des;
    Eigen::VectorXd tau_des;
    Eigen::VectorXd k_des;
    Eigen::VectorXd d_des;

    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> ctrl_mode;

    Outputs(int action_size, int num_joints) {
        raw_action = Eigen::VectorXd::Zero(action_size);
        q_des = Eigen::VectorXd::Zero(num_joints);
        v_des = Eigen::VectorXd::Zero(num_joints);
        tau_des = Eigen::VectorXd::Zero(num_joints);
        k_des = Eigen::VectorXd::Zero(num_joints);
        d_des = Eigen::VectorXd::Zero(num_joints);
        ctrl_mode = decltype(ctrl_mode)::Zero(num_joints);
    }
};

}

#endif // __XBOT2_DEPLOY_POLICY_TYPES_H
