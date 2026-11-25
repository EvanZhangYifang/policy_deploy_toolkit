#pragma once

#include <cmath>

namespace policy_deploy {

// A manual selector message is a time-sensitive operator confirmation. Never
// queue it for a later transition after the visual check may have gone stale.
enum class ManualRequestDecision { accept, feedback_invalid, busy, command_nonzero };

inline ManualRequestDecision manual_mode_request(bool feedback_valid, bool idle, double max_command_magnitude)
{
    if(!feedback_valid) return ManualRequestDecision::feedback_invalid;
    if(!idle) return ManualRequestDecision::busy;
    if(!std::isfinite(max_command_magnitude) || max_command_magnitude > 1e-3 || max_command_magnitude < 0)
        return ManualRequestDecision::command_nonzero;
    return ManualRequestDecision::accept;
}
}
