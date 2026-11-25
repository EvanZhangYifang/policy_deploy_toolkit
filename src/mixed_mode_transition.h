#pragma once

#include <Eigen/Dense>
#include <array>
#include <string>

namespace policy_deploy {

// Simulator-inspired standing transition. Inputs must be fresh and expressed
// in the frames documented in the deployment README.
struct TransitionInput {
    bool valid = false;
    // The operator confirms standing and zero body velocity before each request.
    // The controller cannot measure foot contact or base translation.
    Eigen::VectorXd joint_position, joint_velocity;
    Eigen::Vector3d gravity_body = Eigen::Vector3d::Zero();
    Eigen::Vector3d angular_velocity_body = Eigen::Vector3d::Zero();
    Eigen::Vector3d requested_velocity = Eigen::Vector3d::Zero();
};

class MixedModeTransition {
public:
    MixedModeTransition(double dt, const std::array<Eigen::VectorXd, 4>& references);
    // Initialize only after the physical starting mode and actual last target
    // are known. A blocked controller must be explicitly reinitialized.
    void initialize(int mode, const Eigen::VectorXd& last_target);
    bool request(int mode);
    // Call before actor observation and inference; mode can commit here.
    Eigen::Vector3d advance(const TransitionInput& input);
    // Call after inference; result is the physical position target to send.
    Eigen::VectorXd filter(const Eigen::VectorXd& policy_target,
                           const Eigen::VectorXd& lower, const Eigen::VectorXd& upper,
                           const Eigen::VectorXd& velocity_limit);
    int active_mode() const { return active_; }
    const std::string& phase() const { return phase_; }
    const std::string& status() const { return status_; }
    double policy_blend() const { return blend_; }

private:
    void set_phase(const std::string& phase);
    Eigen::VectorXd planted(const Eigen::Vector4d& knee) const;
    Eigen::VectorXd balanced(const Eigen::VectorXd& target) const;
    bool valid_input(const TransitionInput& input) const;
    double dt_;
    std::array<Eigen::VectorXd, 4> references_;
    int active_ = -1, requested_ = -1;
    std::string phase_ = "uninitialized", status_;
    double elapsed_ = 0, phase_time_ = 0, hold_time_ = 0, lost_time_ = 0;
    double blend_elapsed_ = 0, blend_ = 0;
    bool hold_target_ = false, handoff_ready_ = false, resume_started_ = false, release_limit_ = false;
    Eigen::VectorXd last_target_, script_target_;
    TransitionInput input_;
};
}
