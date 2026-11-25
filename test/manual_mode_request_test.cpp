#include "manual_mode_request.h"
#include <limits>
#include <stdexcept>
using namespace policy_deploy;
static void check(bool good) { if(!good) throw std::runtime_error("manual mode request gate failed"); }
int main() {
    check(manual_mode_request(false, true, 0) == ManualRequestDecision::feedback_invalid);
    check(manual_mode_request(true, false, 0) == ManualRequestDecision::busy);
    check(manual_mode_request(true, true, .0011) == ManualRequestDecision::command_nonzero);
    check(manual_mode_request(true, true, std::numeric_limits<double>::quiet_NaN()) == ManualRequestDecision::command_nonzero);
    check(manual_mode_request(true, true, 0) == ManualRequestDecision::accept);
    check(manual_mode_request(true, true, .001) == ManualRequestDecision::accept);
}
