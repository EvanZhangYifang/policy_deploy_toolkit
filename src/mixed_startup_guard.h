#pragma once

#include "mixed_mode_transition.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>

namespace policy_deploy {

// Require the selected starting pose and an upright, low-motion state on every
// sample for 0.14 seconds. Time gaps reset the hold; time is monotonic and is
// supplied by the caller to make the behavior independently testable.
class MixedStartupGuard {
public:
    explicit MixedStartupGuard(double control_dt) : max_gap_(std::max(0.1, 2.5 * control_dt)) {}
    void reset() { have_sample_ = false; elapsed_ = 0; mode_ = -1; }
    const std::string& reason() const { return reason_; }
    bool update(int mode, const TransitionInput& input, const Eigen::VectorXd& reference,
                const Eigen::VectorXd& last_target, std::chrono::steady_clock::time_point now)
    {
        if(mode < 0 || mode >= 4 || !input.valid || input.joint_position.size() != 12 ||
           input.joint_velocity.size() != 12 || reference.size() != 12 || last_target.size() != 12 ||
           !input.joint_position.allFinite() || !input.joint_velocity.allFinite() ||
           !reference.allFinite() || !last_target.allFinite() ||
           !input.gravity_body.allFinite() || !input.angular_velocity_body.allFinite()) {
            reset(); reason_ = "Waiting for valid starting mode and state"; return false;
        }
        if((input.joint_position-reference).cwiseAbs().maxCoeff() > 0.3 ||
           (last_target-reference).cwiseAbs().maxCoeff() > 0.3) {
            reset(); reason_ = "Initial joints or position reference differ from selected mode by more than 0.3 rad";
            return false;
        }
        constexpr double pi = 3.14159265358979323846;
        const bool still = input.gravity_body.z() <= -std::cos(7.0*pi/180.0) &&
            input.angular_velocity_body.norm() <= 0.18 && input.joint_velocity.cwiseAbs().maxCoeff() <= 0.45;
        if(!still) {
            reset(); reason_ = "Waiting for an upright, low-motion stance (support not sensed)"; return false;
        }
        const double gap = have_sample_ ? std::chrono::duration<double>(now-last_).count() : 0;
        if(!have_sample_ || mode != mode_ || gap < 0 || gap > max_gap_) {
            have_sample_ = true; mode_ = mode; elapsed_ = 0;
        } else elapsed_ += gap;
        last_ = now;
        if(elapsed_ + 1e-9 < 0.14) {
            reason_ = "Waiting 0.14 seconds for upright, low-motion stance (support not sensed)";
            return false;
        }
        reason_.clear();
        return true;
    }
private:
    double max_gap_;
    bool have_sample_ = false;
    double elapsed_ = 0;
    int mode_ = -1;
    std::chrono::steady_clock::time_point last_{};
    std::string reason_;
};

} // namespace policy_deploy
