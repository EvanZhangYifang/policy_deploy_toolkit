#include "mixed_startup_guard.h"
#include "mixed_startup_mode.h"
#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace std::chrono_literals;
using namespace policy_deploy;
static void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
int main() {
    try {
        require(parse_mixed_initial_mode("bb") == 0 && parse_mixed_initial_mode("bf") == 1 &&
                parse_mixed_initial_mode("fb") == 2 && parse_mixed_initial_mode("ff") == 3 &&
                parse_mixed_initial_mode("") == -1, "invalid initial mode mapping");
        bool rejected = false;
        try { parse_mixed_initial_mode("invalid"); } catch(const std::invalid_argument&) { rejected = true; }
        require(rejected, "invalid initial mode was accepted");
        const auto default_start = select_mixed_startup_mode(0, -1, 0);
        require(default_start.mode == 0 && default_start.consumed_sequence == 0,
                "configured startup must not require a selector");
        const auto pending = select_mixed_startup_mode(0, 1, 7);
        require(pending.mode == 0 && pending.consumed_sequence != 7,
                "configured startup discarded a pending transition request");
        const auto waiting = select_mixed_startup_mode(-1, -1, 0);
        require(waiting.mode == -1, "empty initial mode must wait for selector");
        const auto selector_start = select_mixed_startup_mode(-1, 1, 7);
        require(selector_start.mode == 1 && selector_start.consumed_sequence == 7,
                "selector-based startup did not consume its initial selector");
        MixedStartupGuard guard(.02);
        TransitionInput input; input.valid = true;
        input.joint_position = Eigen::VectorXd::Zero(12); input.joint_velocity = Eigen::VectorXd::Zero(12);
        input.gravity_body << 0,0,-1;
        const Eigen::VectorXd q = Eigen::VectorXd::Zero(12);
        auto t = std::chrono::steady_clock::time_point{};
        auto sample = [&] { const bool ready=guard.update(0,input,q,q,t); t+=20ms; return ready; };
        for(int i=0; i<7; ++i) require(!sample(), "started before stable hold elapsed");
        require(sample(), "did not start after stable hold");
        input.joint_velocity[0]=.46; require(!sample(), "moving joint accepted");
        input.joint_velocity[0]=0; input.gravity_body << .2,0,-.98; require(!sample(), "tilted body accepted");
        input.gravity_body << 0,0,-1; input.joint_position[0]=.31; require(!sample(), "wrong pose accepted");
        input.joint_position[0]=0; input.valid=false; require(!sample(), "stale inputs accepted");
        input.valid=true; require(!sample(), "invalid feedback did not reset hold");
        t+=200ms; require(!sample(), "sample gap did not reset hold");
        for(int i=0; i<6; ++i) require(!sample(), "time gap recovery skipped hold");
        require(sample(), "time gap recovery failed");
        require(!guard.update(1,input,q,q,t), "mode change did not reset hold");
        // Operator-confirmed support still requires genuine upright,
        // angular and joint-speed feedback.
        MixedStartupGuard manual(.02);
        input.gravity_body << 0,0,-1; input.angular_velocity_body.setZero();
        input.joint_velocity.setZero(); input.joint_position=q; input.valid=true;
        for(int i=0; i<7; ++i) { require(!manual.update(0,input,q,q,t), "manual startup skipped hold"); t+=20ms; }
        require(manual.update(0,input,q,q,t), "operator-confirmed startup failed after hold"); t+=20ms;
        input.angular_velocity_body.x()=.19;
        require(!manual.update(0,input,q,q,t), "manual startup accepted excessive angular speed"); t+=20ms;
        input.angular_velocity_body.setZero(); input.joint_velocity[0]=.46;
        require(!manual.update(0,input,q,q,t), "manual startup accepted fast joints"); t+=20ms;
        input.joint_velocity.setZero(); input.gravity_body << .2,0,-.98;
        require(!manual.update(0,input,q,q,t), "manual startup accepted tilt"); t+=20ms;
        input.gravity_body << 0,0,-1; input.valid=false;
        require(!manual.update(0,input,q,q,t), "manual startup accepted unavailable real feedback");
        std::cout << "Mixed startup posture guard passed.\n";
        return 0;
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
