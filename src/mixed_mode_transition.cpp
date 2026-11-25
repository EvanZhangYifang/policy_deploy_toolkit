#include "mixed_mode_transition.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace policy_deploy {
namespace {
constexpr int N = 12;
constexpr double pi = 3.14159265358979323846;
double smooth(double x) { x = std::clamp(x, 0.0, 1.0); return x*x*x*(10.0 + x*(-15.0 + 6.0*x)); }
void require_vector(const Eigen::VectorXd& v, const char* name) {
    if (v.size() != N || !v.allFinite()) throw std::invalid_argument(std::string(name) + " must contain 12 finite values");
}
}

MixedModeTransition::MixedModeTransition(double dt, const std::array<Eigen::VectorXd, 4>& references)
    : dt_(dt), references_(references) {
    if (!std::isfinite(dt) || dt <= 0 || dt > 0.1) throw std::invalid_argument("invalid control period");
    for (const auto& r : references_) require_vector(r, "mode reference");
}

void MixedModeTransition::initialize(int mode, const Eigen::VectorXd& last_target) {
    if (mode < 0 || mode >= 4) throw std::invalid_argument("invalid mode");
    require_vector(last_target, "last target");
    active_ = requested_ = mode;
    last_target_ = last_target;
    script_target_.resize(0);
    elapsed_ = phase_time_ = hold_time_ = lost_time_ = blend_elapsed_ = blend_ = 0;
    hold_target_ = handoff_ready_ = resume_started_ = release_limit_ = false;
    phase_ = "idle"; status_.clear();
}

bool MixedModeTransition::request(int mode) {
    if (mode < 0 || mode >= 4) throw std::invalid_argument("invalid mode");
    if (phase_ != "idle") return false;
    if (mode == active_) return false;
    requested_ = mode;
    elapsed_ = phase_time_ = hold_time_ = lost_time_ = blend_elapsed_ = blend_ = 0;
    hold_target_ = handoff_ready_ = resume_started_ = false;
    script_target_.resize(0);
    phase_ = "braking"; status_.clear();
    return true;
}

void MixedModeTransition::set_phase(const std::string& phase) {
    phase_ = phase; phase_time_ = hold_time_ = lost_time_ = 0;
}

bool MixedModeTransition::valid_input(const TransitionInput& i) const {
    return i.valid && i.joint_position.size() == N && i.joint_velocity.size() == N &&
        i.joint_position.allFinite() && i.joint_velocity.allFinite() && i.gravity_body.allFinite() &&
        i.angular_velocity_body.allFinite() && i.requested_velocity.allFinite();
}

Eigen::Vector3d MixedModeTransition::advance(const TransitionInput& i) {
    if (active_ < 0) throw std::logic_error("transition controller not initialized");
    if (phase_ == "blocked") return Eigen::Vector3d::Zero();
    if (!valid_input(i)) {
        // A missing or non-finite state is never interpreted as a safe transition.
        if (phase_ != "idle") { phase_ = "blocked"; status_ = "Transition feedback unavailable or invalid"; }
        return Eigen::Vector3d::Zero();
    }
    input_ = i;
    if (phase_ == "idle") return i.requested_velocity;
    elapsed_ += dt_; phase_time_ += dt_;
    // The operator confirms support and zero translation; neither is sensed.
    const bool upright = i.gravity_body.z() <= -std::cos(7.0*pi/180.0);
    const bool still = upright && i.angular_velocity_body.norm() <= .18 &&
        i.joint_velocity.cwiseAbs().maxCoeff() <= .45;
    if (elapsed_ > 15.0) { phase_ = "blocked"; status_ = "Transition timed out"; return Eigen::Vector3d::Zero(); }
    if (phase_ == "braking") {
        if (phase_time_ >= .44) set_phase("waiting");
        // Keep drive command zero throughout the transition.
        return Eigen::Vector3d::Zero();
    }
    if (phase_ == "waiting") {
        hold_time_ = still ? hold_time_ + dt_ : 0.0;
        status_ = "Waiting for an upright, low-motion stance (support not sensed)";
        if (hold_time_ + 1e-9 >= .14) { set_phase("extending"); status_.clear(); }
        return Eigen::Vector3d::Zero();
    }
    if (phase_ == "extending" || phase_ == "bending") {
        if (!upright) {
            lost_time_ += dt_; hold_time_ = 0; phase_time_ -= dt_; hold_target_ = true;
            status_ = "Paused: upright posture lost (support not sensed)";
            if (lost_time_ + 1e-9 >= 1.0) {
                phase_ = "blocked";
                status_ = "Upright posture lost (support not sensed)";
            }
            return Eigen::Vector3d::Zero();
        }
        lost_time_ = 0; hold_target_ = false; status_.clear();
        Eigen::Vector4d source, dest, measured;
        constexpr double mirror[4] = {1, -1, 1, -1};
        for (int j = 0; j < 4; ++j) {
            source[j] = references_[active_][3*j+2] * mirror[j];
            dest[j] = references_[requested_][3*j+2] * mirror[j];
            measured[j] = i.joint_position[3*j+2] * mirror[j];
        }
        if (phase_ == "extending") {
            double p = smooth(phase_time_ / 1.08);
            script_target_ = planted(source * (1.0-p));
            if (p >= 1.0 && measured.cwiseAbs().maxCoeff() <= .18 && still) set_phase("bending");
        } else {
            double p = smooth(phase_time_ / .96);
            script_target_ = planted(dest*p) + p*(references_[requested_] - planted(dest));
            const double error = (i.joint_position - references_[requested_]).cwiseAbs().maxCoeff();
            const bool quiet = i.angular_velocity_body.norm() <= .25 &&
                i.joint_velocity.cwiseAbs().maxCoeff() <= 2.50;
            hold_time_ = (error <= .30 && quiet) ? hold_time_ + dt_ : 0;
            if (hold_time_ + 1e-9 >= .06) {
                active_ = requested_;
                handoff_ready_ = resume_started_ = false;
                blend_elapsed_ = dt_; blend_ = smooth(blend_elapsed_ / .80);
                set_phase("resuming");
            }
        }
        return Eigen::Vector3d::Zero();
    }
    if (phase_ == "resuming") {
        if (!upright) {
            lost_time_ += dt_; phase_time_ -= dt_; hold_target_ = true;
            handoff_ready_ = resume_started_ = false;
            status_ = "Paused: upright posture lost during handoff (support not sensed)";
            if (lost_time_ + 1e-9 >= .20) {
                phase_ = "blocked";
                status_ = "Upright posture lost during handoff (support not sensed)";
            }
            return Eigen::Vector3d::Zero();
        }
        lost_time_ = 0; hold_target_ = false;
        blend_elapsed_ += dt_; blend_ = smooth(blend_elapsed_ / .80);
        if (!resume_started_) {
            if (!handoff_ready_) { status_ = "Waiting for smooth policy handoff"; return Eigen::Vector3d::Zero(); }
            resume_started_ = true; phase_time_ = 0; status_.clear();
        }
        if (phase_time_ + 1e-9 >= .40 && handoff_ready_) {
            set_phase("idle"); script_target_.resize(0); release_limit_ = true;
        }
        return Eigen::Vector3d::Zero();
    }
    throw std::logic_error("unexpected transition phase");
}

Eigen::VectorXd MixedModeTransition::planted(const Eigen::Vector4d& knees) const {
    Eigen::VectorXd q(N);
    constexpr double mirror[4] = {1, -1, 1, -1}, fore_aft[4] = {1, 1, -1, -1};
    for (int j = 0; j < 4; ++j) {
        double k = knees[j], reach = .346 + .300 * std::cos(k), offset = .300 * std::sin(k);
        double length = std::hypot(reach, offset), hp = -std::atan2(offset, reach);
        double rolling = (.04 / (length + .04)) * (hp + k);
        hp -= rolling; length *= std::cos(rolling);
        double hr = std::acos(std::clamp(.1463 / std::sqrt(length*length + .2226*.2226), -1., 1.)) - std::atan(length / .2226);
        q[3*j] = mirror[j]*fore_aft[j]*hr; q[3*j+1] = mirror[j]*hp; q[3*j+2] = mirror[j]*k;
    }
    return q;
}

Eigen::VectorXd MixedModeTransition::balanced(const Eigen::VectorXd& target) const {
    constexpr double fx[4] = {.39,.39,-.39,-.39}, fy[4] = {.23,-.23,.23,-.23};
    double roll = -.30*input_.gravity_body.y() + .045*input_.angular_velocity_body.x();
    double pitch = .30*input_.gravity_body.x() + .045*input_.angular_velocity_body.y();
    Eigen::VectorXd result = target;
    for (int j=0;j<4;++j) {
        double dz = roll*fy[j] - pitch*fx[j];
        double dk = std::clamp(5.*dz,-.12,.12)*std::clamp(result[3*j+2]/.20,-1.,1.);
        result[3*j+2] += dk; result[3*j+1] -= .46*dk;
    }
    return result;
}

Eigen::VectorXd MixedModeTransition::filter(const Eigen::VectorXd& policy_target,
                                              const Eigen::VectorXd& lower, const Eigen::VectorXd& upper,
                                              const Eigen::VectorXd& velocity_limit) {
    if (active_ < 0) throw std::logic_error("transition controller not initialized");
    require_vector(policy_target, "policy target"); require_vector(lower, "lower limits"); require_vector(upper, "upper limits");
    if ((lower.array() > upper.array()).any()) throw std::invalid_argument("invalid position limits");
    if (velocity_limit.size() != N || (velocity_limit.array().isNaN()).any() ||
        (velocity_limit.array() < 0).any()) throw std::invalid_argument("invalid velocity limits");
    if (phase_ == "idle" && !release_limit_) { last_target_ = policy_target; return policy_target; }
    Eigen::VectorXd desired = policy_target;
    if (phase_ == "blocked" || hold_target_) desired = last_target_;
    else if ((phase_ == "extending" || phase_ == "bending") && script_target_.size() == N) desired = balanced(script_target_);
    else if (phase_ == "resuming" && script_target_.size() == N) desired = (1.-blend_)*balanced(script_target_) + blend_*policy_target;
    desired = desired.cwiseMin(upper).cwiseMax(lower);
    const Eigen::VectorXd max_step = velocity_limit.cwiseMin(2.0) * dt_;
    const Eigen::VectorXd delta = (desired - last_target_).cwiseMax(-max_step).cwiseMin(max_step);
    last_target_ += delta;
    if (!hold_target_) {
        bool reached = (desired - last_target_).cwiseAbs().maxCoeff() <= .02;
        handoff_ready_ = reached && (phase_ != "resuming" || blend_ >= 1.);
    }
    if (phase_ == "idle" && handoff_ready_ && (policy_target-desired).cwiseAbs().maxCoeff() <= 1e-6) release_limit_ = false;
    return last_target_;
}
}
