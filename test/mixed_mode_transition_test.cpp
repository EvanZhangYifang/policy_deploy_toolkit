#include "mixed_mode_transition.h"
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
using policy_deploy::MixedModeTransition;
using policy_deploy::TransitionInput;

static Eigen::VectorXd vec(std::initializer_list<double> v) {
    Eigen::VectorXd r(v.size()); int i=0; for (double x:v) r[i++]=x; return r;
}
int main() {
    const std::array<Eigen::VectorXd,4> refs = {
        vec({.15,-.7,1.4,-.15,.7,-1.4,-.15,-.7,1.4,.15,.7,-1.4}),
        vec({.15,-.7,1.4,-.15,.7,-1.4,-.15,.7,-1.4,.15,-.7,1.4}),
        vec({.15,.7,-1.4,-.15,-.7,1.4,-.15,-.7,1.4,.15,.7,-1.4}),
        vec({.15,.7,-1.4,-.15,-.7,1.4,-.15,.7,-1.4,.15,-.7,1.4})};
    const auto lo = Eigen::VectorXd::Constant(12,-4.), hi = Eigen::VectorXd::Constant(12,4.);
    const auto vmax = Eigen::VectorXd::Constant(12,7.6);
    TransitionInput s; s.valid=true; s.joint_position=refs[0];
    s.joint_velocity=Eigen::VectorXd::Zero(12); s.gravity_body << 0,0,-1;
    // Transition output is bounded by the actual robot velocity limit even
    // when that limit is below the simulator controller's 2 rad/s cap.
    MixedModeTransition limited(.02,refs); limited.initialize(0,refs[0]); assert(limited.request(1));
    s.valid=true; s.joint_position=refs[0];
    limited.advance(s);
    Eigen::VectorXd slow = Eigen::VectorXd::Constant(12,.5);
    Eigen::VectorXd limited_q=limited.filter(refs[0]+Eigen::VectorXd::Ones(12),lo,hi,slow);
    assert((limited_q-refs[0]).cwiseAbs().maxCoeff() <= .0100001);
    // Missing state also blocks an active transition.
    MixedModeTransition missing(.02,refs); missing.initialize(0,refs[0]); assert(missing.request(1));
    s.valid=false; missing.advance(s); assert(missing.phase()=="blocked");
    // Real IMU posture and joint feedback remain mandatory.
    MixedModeTransition manual(.02,refs); manual.initialize(0,refs[0]); assert(manual.request(1));
    TransitionInput m; m.valid=true;
    m.joint_position=refs[0]; m.joint_velocity=Eigen::VectorXd::Zero(12); m.gravity_body << 0,0,-1;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    bool manual_committed=false; Eigen::VectorXd manual_last=refs[0];
    m.requested_velocity << .4,.1,.2;  // joystick changed after operator request
    for (int i=0;i<750 && manual.phase()!="idle";++i) {
        int old=manual.active_mode(); auto command=manual.advance(m);
        assert(command.norm()==0);
        if (manual.active_mode()!=old) {
            manual_committed=true; assert(manual.active_mode()==1);
            assert((m.joint_position-refs[1]).cwiseAbs().maxCoeff()<=.3+1e-12);
        }
        Eigen::VectorXd q=manual.filter(refs[manual.active_mode()],lo,hi,vmax);
        assert(q.allFinite() && (q-manual_last).cwiseAbs().maxCoeff()<=.0400001);
        m.joint_position=q; manual_last=q;
    }
    assert(manual_committed && manual.phase()=="idle");
    assert((manual.advance(m) - m.requested_velocity).norm()<1e-12);
    // Exercise all selector pairs with operator-confirmed support.
    for (int source=0;source<4;++source) for (int target=0;target<4;++target) {
        if (source==target) continue;
        MixedModeTransition pair(.02,refs); pair.initialize(source,refs[source]); assert(pair.request(target));
        m.joint_position=refs[source]; Eigen::VectorXd last_pair=refs[source]; bool committed=false;
        for (int i=0;i<750 && pair.phase()!="idle";++i) {
            int old=pair.active_mode(); assert(pair.advance(m).norm()==0);
            if (pair.active_mode()!=old) {
                committed=true; assert(pair.active_mode()==target);
                assert((m.joint_position-refs[target]).cwiseAbs().maxCoeff()<=.3+1e-12);
            }
            Eigen::VectorXd q=pair.filter(refs[pair.active_mode()],lo,hi,vmax);
            assert(q.allFinite() && (q-last_pair).cwiseAbs().maxCoeff()<=.0400001);
            m.joint_position=q; last_pair=q;
        }
        assert(committed && pair.phase()=="idle" && pair.active_mode()==target);
    }
    // Lose upright posture during motion: hold and ultimately block with operator-confirmed support.
    MixedModeTransition tilted(.02,refs); tilted.initialize(0,refs[0]); assert(tilted.request(1));
    m.joint_position=refs[0]; manual_last=refs[0];
    for (int i=0;i<100 && tilted.phase()!="extending";++i) {
        tilted.advance(m); manual_last=tilted.filter(refs[0],lo,hi,vmax); m.joint_position=manual_last;
    }
    assert(tilted.phase()=="extending"); m.gravity_body << .2,0,-.98;
    for (int i=0;i<50;++i) {
        tilted.advance(m); auto q=tilted.filter(refs[0],lo,hi,vmax); assert((q-manual_last).norm()<1e-12);
    }
    assert(tilted.phase()=="blocked" && tilted.status().find("support not sensed")!=std::string::npos);
    // An invalid IMU or joint signal remains fatal during transitions.
    for (int kind=0;kind<2;++kind) {
        MixedModeTransition bad(.02,refs); bad.initialize(0,refs[0]); assert(bad.request(1));
        m.gravity_body << 0,0,-1; m.joint_position=refs[0]; m.angular_velocity_body.setZero();
        if(kind==0) m.angular_velocity_body.x()=nan; else m.joint_position[0]=nan;
        bad.advance(m); assert(bad.phase()=="blocked");
    }
    std::cout << "12 directed transitions, blending, tilt loss, and invalid state passed\n";
}
