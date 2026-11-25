#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace policy_deploy {

// An empty initial_mode intentionally delegates the initial stance to the
// first selector. With a configured starting mode, the caller discards requests
// received before startup so the operator can provide a fresh confirmation.
inline int parse_mixed_initial_mode(const std::string& mode)
{
    if (mode.empty()) return -1;
    if (mode == "bb") return 0;
    if (mode == "bf") return 1;
    if (mode == "fb") return 2;
    if (mode == "ff") return 3;
    throw std::invalid_argument("initial_mode must be bb, bf, fb, ff, or empty");
}

struct MixedStartupSelection {
    int mode;
    std::uint64_t consumed_sequence;
};

inline MixedStartupSelection select_mixed_startup_mode(int configured_mode, int selector_mode,
                                                        std::uint64_t selector_sequence)
{
    if (configured_mode < -1 || configured_mode > 3 || selector_mode < -1 || selector_mode > 3)
        throw std::invalid_argument("invalid mixed mode index");
    if (configured_mode >= 0) return {configured_mode, 0};
    return {selector_mode, selector_sequence};
}
}
