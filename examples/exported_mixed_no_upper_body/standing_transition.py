"""Flat-ground, guarded, standing mode changes for interactive mixed-policy playback.

The state machine is intentionally separate from the training action term. It
assumes a single simulator environment and is not a real-robot controller.
"""
from __future__ import annotations

import math

import torch

from .mdp import MODE_NAMES, MODE_JOINT_POSITIONS


def _tensor(value):
    return value.torch if hasattr(value, "torch") else value


def _smoothstep(value: float) -> float:
    x = max(0.0, min(1.0, value))
    return x**3 * (10.0 + x * (-15.0 + 6.0 * x))


def planted_leg_pose(knees: torch.Tensor) -> torch.Tensor:
    """Coordinated leg targets for the current USD's 0.346 m/0.300 m legs.

    Current and archived leg-1 USD joint transforms match (including the
    contact frame); full-body dynamics still require separate validation.
    """
    reach = 0.346 + 0.300 * knees.cos()
    offset = 0.300 * knees.sin()
    length = torch.hypot(reach, offset)
    hip_pitch = -torch.atan2(offset, reach)
    rolling_pitch = (0.04 / (length + 0.04)) * (hip_pitch + knees)
    hip_pitch -= rolling_pitch
    length *= rolling_pitch.cos()
    hip_roll = (0.1463 / (length.square() + 0.2226**2).sqrt()).clamp(-1, 1).acos()
    hip_roll -= torch.atan(length / 0.2226)
    fore_aft = torch.tensor([1.0, 1.0, -1.0, -1.0], device=knees.device)
    mirror = torch.tensor([1.0, -1.0, 1.0, -1.0], device=knees.device)
    return torch.stack((fore_aft * hip_roll, hip_pitch, knees), dim=-1).mul(mirror[:, None]).reshape(1, 12)


class StandingModeTransition:
    """Guarded standing switch around a fixed-mode policy, for one flat env.

    Call ``advance`` before computing policy observations, then pass policy
    output through ``filter_actions`` before the simulator step. This preserves
    the 47-input policy; the scripted switch is not learned by that policy.
    """

    # Early handoff is based on measured legs, not the scripted target. These
    # playback thresholds are simulator parameters, not physical robot limits.
    bend_duration = 0.96
    handoff_joint_tolerance = 0.30
    handoff_planar_speed = 0.08
    handoff_vertical_speed = 0.40
    handoff_angular_speed = 0.25
    handoff_joint_speed = 2.50
    handoff_hold_time = 0.06
    handoff_blend_duration = 0.80

    def __init__(self, env):
        if env.num_envs != 1:
            raise ValueError("Standing mode playback requires exactly one environment")
        self.env = env
        self.robot = env.scene["robot"]
        self.sensor = env.scene.sensors["contact_forces"]
        self.foot_ids, _ = self.sensor.find_bodies([f"contact_{i}" for i in range(1, 5)], preserve_order=True)
        if len(self.foot_ids) != 4:
            raise ValueError("Standing transitions require four foot contact sensors")
        self.action = env.action_manager.get_term("joint_pos")
        if tuple(self.action._joint_names) != tuple(f"{joint}_{leg}" for leg in range(1, 5) for joint in ("hip_roll", "hip_pitch", "knee_pitch")):
            raise ValueError("Unexpected mixed-policy joint order")
        self.dt = float(env.step_dt)
        self._mirror = torch.tensor([1.0, -1.0, 1.0, -1.0], device=env.device)
        self._table = torch.tensor([MODE_JOINT_POSITIONS[m] for m in MODE_NAMES], device=env.device)
        self.reference_changed = False
        self.reset()

    def reset(self):
        """Cancel a pending switch after env.reset, retaining the active mode."""
        self.active_mode = MODE_NAMES[int(self.env.mixed_mode_ids[0].item())]
        self.requested_mode = self.active_mode
        self.phase = "idle"
        self.reason = ""
        self.progress = 0.0
        self.elapsed = 0.0
        self.phase_time = 0.0
        self.hold_time = 0.0
        self.lost_time = 0.0
        self.reference_changed = True
        self._last_target = None
        self._target = None
        self._handoff_ready = False
        self._resume_started = False
        self._release_limit = False
        self._anchor = None
        self._hold_target = False
        self.handoff_pose_error = None
        self.handoff_bend_progress = None
        self.policy_blend = 0.0
        self._policy_blend_elapsed = 0.0

    def request_mode(self, mode: str):
        if mode not in MODE_NAMES:
            raise ValueError("Select BB, BF, FB or FF")
        if self.phase != "idle":
            raise RuntimeError("A transition is already in progress; reset if it is blocked")
        if mode == self.active_mode:
            return False
        self.requested_mode = mode
        self.phase = "braking"
        self.reason = ""
        self.elapsed = self.phase_time = self.hold_time = self.lost_time = 0.0
        self.progress = 0.0
        self.handoff_pose_error = None
        self.handoff_bend_progress = None
        self.policy_blend = 0.0
        self._policy_blend_elapsed = 0.0
        self._target = None
        return True

    def status(self):
        return {"phase": self.phase, "active_mode": self.active_mode,
                "requested_mode": self.requested_mode, "pending": self.phase != "idle",
                "progress": self.progress, "reason": self.reason,
                "handoff_pose_error_rad": self.handoff_pose_error,
                "handoff_bend_progress": self.handoff_bend_progress,
                "policy_blend": self.policy_blend}

    def _support(self):
        data = self.robot.data
        forces = _tensor(self.sensor.data.net_forces_w)[0, self.foot_ids]
        supported = bool((forces[:, 2].abs() > 16.0).all().item())
        upright = float(_tensor(data.projected_gravity_b)[0, 2]) <= -math.cos(math.radians(7))
        still = (supported and upright
                 and float(torch.linalg.vector_norm(_tensor(data.root_lin_vel_b)[0])) <= 0.04
                 and float(torch.linalg.vector_norm(_tensor(data.root_ang_vel_b)[0])) <= 0.18
                 and float(_tensor(data.joint_vel)[0, self.action._joint_ids].abs().max()) <= 0.45)
        return supported, upright, still

    def _phase(self, phase):
        self.phase = phase
        self.phase_time = self.hold_time = self.lost_time = 0.0

    def _commit(self):
        idx = MODE_NAMES.index(self.requested_mode)
        ref = self._table[idx:idx + 1]
        defaults = _tensor(self.robot.data.default_joint_pos)
        defaults[:, self.action._joint_ids] = ref
        self.action._offset = ref.clone()
        self.env.mixed_mode_reference[:] = ref
        self.env.mixed_mode_ids[:] = idx
        # The policy observes the previous raw action relative to its current
        # reference. Rebase the last physical target when the reference
        # changes so the observation remains in the destination coordinates.
        if self._last_target is None:
            raise RuntimeError("Cannot commit a mode before applying transition targets")
        rebased_action = (self._last_target - ref) / self.action._scale
        self.env.action_manager.action[:] = rebased_action
        self.env.action_manager.prev_action[:] = rebased_action
        self.action._raw_actions[:] = rebased_action
        self.active_mode = self.requested_mode
        self.reference_changed = True
        # The previous readiness was measured against the scripted target,
        # not the destination policy's target.
        self._handoff_ready = False
        self._resume_started = False
        # Policy contributes from the first handoff step. Its influence is
        # faded in while the scripted balance reference remains active.
        self._policy_blend_elapsed = self.dt
        self.policy_blend = _smoothstep(self._policy_blend_elapsed / self.handoff_blend_duration)

    def _balance_target(self, target):
        """Bounded planar and posture feedback during planted-leg movement."""
        if self._anchor is None:
            return target
        data = self.robot.data
        position = _tensor(data.root_pos_w)[0, :2]
        heading = _tensor(data.heading_w)[0]
        error = (self._anchor - position).clamp(-0.15, 0.15)
        c, s = heading.cos(), heading.sin()
        body_error = torch.stack((c * error[0] + s * error[1], -s * error[0] + c * error[1]))
        body_error -= 0.32 * _tensor(data.root_lin_vel_b)[0, :2]
        signs = torch.tensor([[-1., -1., 1., 1.], [1., -1., 1., -1.]], device=target.device)
        hip = (-0.65 * body_error[[1, 0], None] * signs).clamp(-0.10, 0.10)
        result = target.clone().reshape(4, 3)
        result[:, 0] += hip[0]
        result[:, 1] += hip[1]
        gravity = _tensor(data.projected_gravity_b)[0, :2]
        angular = _tensor(data.root_ang_vel_b)[0]
        roll = -0.30 * gravity[1] + 0.045 * angular[0]
        pitch = 0.30 * gravity[0] + 0.045 * angular[1]
        foot_x = torch.tensor([0.39, 0.39, -0.39, -0.39], device=target.device)
        foot_y = torch.tensor([0.23, -0.23, 0.23, -0.23], device=target.device)
        dz = roll * foot_y - pitch * foot_x
        knee = (5.0 * dz).clamp(-0.12, 0.12) * (result[:, 2] / 0.20).clamp(-1.0, 1.0)
        result[:, 2] += knee
        result[:, 1] -= 0.46 * knee
        return result.reshape(1, 12)

    def advance(self, requested_velocity):
        """Advance one 20 ms control step and return the gated drive command."""
        if len(requested_velocity) != 3 or any(not math.isfinite(float(x)) for x in requested_velocity):
            raise ValueError("Velocity command must contain three finite values")
        self.reference_changed = False
        cmd = [float(v) for v in requested_velocity]
        if self.phase == "idle":
            return cmd
        if self.phase == "blocked":
            return [0.0, 0.0, 0.0]
        self.elapsed += self.dt
        self.phase_time += self.dt
        supported, upright, still = self._support()
        if self.elapsed > 15.0:
            self.phase = "blocked"
            self.reason = "Transition timed out; reset the simulation before retrying"
            return [0.0, 0.0, 0.0]
        if self.phase == "braking":
            scale = 1.0 - _smoothstep(self.phase_time / 0.44)
            if self.phase_time >= 0.44:
                self._phase("waiting")
                scale = 0.0
            return [v * scale for v in cmd]
        if self.phase == "waiting":
            self.hold_time = self.hold_time + self.dt if still else 0.0
            self.reason = "Waiting for four loaded feet and a stable, upright stance"
            if self.hold_time >= 0.14:
                self.reason = ""
                self._anchor = _tensor(self.robot.data.root_pos_w)[0, :2].clone()
                self._phase("extending")
            return [0.0, 0.0, 0.0]
        if self.phase in ("extending", "bending", "settling"):
            if not (supported and upright):
                self.lost_time += self.dt
                self.hold_time = 0.0  # handoff requires continuous stable support
                self.phase_time -= self.dt  # freeze the commanded path while support is lost
                self._hold_target = True
                self.reason = "Paused: foot support or upright posture lost"
                if self.lost_time >= 1.0:
                    self.phase = "blocked"
                    self.reason = "Support or upright posture lost; reset the simulation"
                return [0.0, 0.0, 0.0]
            self.lost_time = 0.0
            self._hold_target = False
            self.reason = ""
            source = self._table[MODE_NAMES.index(self.active_mode), 2::3] * self._mirror
            target = self._table[MODE_NAMES.index(self.requested_mode), 2::3] * self._mirror
            measured = _tensor(self.robot.data.joint_pos)[0, self.action._joint_ids].reshape(4, 3)[:, 2] * self._mirror
            if self.phase == "extending":
                p = _smoothstep(self.phase_time / 1.08)
                self._target = planted_leg_pose(source * (1.0 - p))
                self.progress = 0.5 * p
                if p >= 1.0 and bool((measured.abs() <= 0.18).all().item()) and still:
                    self._phase("bending")
            elif self.phase == "bending":
                p = _smoothstep(self.phase_time / self.bend_duration)
                self._target = planted_leg_pose(target * p)
                # Correct only the small geometric mismatch at the trained
                # endpoint. Blending whole poses would distort the knee path.
                nominal = self._table[MODE_NAMES.index(self.requested_mode):MODE_NAMES.index(self.requested_mode)+1]
                self._target += p * (nominal - planted_leg_pose(target))
                self.progress = 0.5 + 0.5 * p
                # Start the policy handoff once the physical legs approach
                # the destination. Do not wait for p == 1 or a scripted hold.
                data = self.robot.data
                joints = _tensor(data.joint_pos)[:, self.action._joint_ids]
                error = float((joints - nominal).abs().max().item())
                quiet_enough = (
                    float(torch.linalg.vector_norm(_tensor(data.root_lin_vel_b)[0, :2])) <= self.handoff_planar_speed
                    and float(_tensor(data.root_lin_vel_b)[0, 2].abs()) <= self.handoff_vertical_speed
                    and float(torch.linalg.vector_norm(_tensor(data.root_ang_vel_b)[0])) <= self.handoff_angular_speed
                    and float(_tensor(data.joint_vel)[0, self.action._joint_ids].abs().max()) <= self.handoff_joint_speed
                )
                near = error <= self.handoff_joint_tolerance and quiet_enough
                self.hold_time = self.hold_time + self.dt if near else 0.0
                if self.hold_time + 1e-9 >= self.handoff_hold_time:
                    self.handoff_pose_error = error
                    self.handoff_bend_progress = p
                    self._commit()
                    self._phase("resuming")
            else:
                raise RuntimeError("Unexpected scripted settling phase")
            return [0.0, 0.0, 0.0]
        if self.phase == "resuming":
            if not (supported and upright):
                self.lost_time += self.dt
                self.phase_time -= self.dt
                self._hold_target = True
                self._handoff_ready = False
                self._resume_started = False
                self.reason = "Paused: foot support or upright posture lost during handoff"
                if self.lost_time >= 0.20:
                    self.phase = "blocked"
                    self.reason = "Support or upright posture lost during handoff; reset the simulation"
                return [0.0, 0.0, 0.0]
            self.lost_time = 0.0
            self._hold_target = False
            self._policy_blend_elapsed += self.dt
            self.policy_blend = _smoothstep(self._policy_blend_elapsed / self.handoff_blend_duration)
            if self.reason.startswith("Paused:"):
                self.reason = ""
            if not self._resume_started:
                if not self._handoff_ready:
                    self.reason = "Waiting for a smooth policy handoff"
                    return [0.0, 0.0, 0.0]
                self._resume_started = True
                self.phase_time = 0.0
                self.reason = ""
            scale = _smoothstep(self.phase_time / 0.40)
            # Remain rate-limited until the policy target has actually caught
            # up; a fixed-time handoff can otherwise create a target jump.
            if self.phase_time >= 0.40 and self._handoff_ready:
                self._phase("idle")
                self.progress = 1.0
                self._target = None
                # Also constrain the first idle output: the newly evaluated
                # policy target may differ from the previous step's target.
                self._release_limit = True
            return [v * scale for v in cmd]
        raise RuntimeError(f"Unknown transition phase: {self.phase}")

    def filter_actions(self, actions):
        """Rate-limit commanded positions and map them to the existing action space."""
        if self.phase == "idle" and not self._release_limit:
            self._last_target = (actions * self.action._scale + self.action._offset).detach().clone()
            return actions
        policy_target = actions * self.action._scale + self.action._offset
        if self._last_target is None:
            self._last_target = _tensor(self.robot.data.joint_pos)[:, self.action._joint_ids].clone()
        if self.phase == "blocked" or self._hold_target:
            desired = self._last_target
        elif self.phase in ("extending", "bending", "settling") and self._target is not None:
            desired = self._balance_target(self._target)
        elif self.phase == "resuming" and self._target is not None:
            desired = torch.lerp(self._balance_target(self._target), policy_target, self.policy_blend)
        else:
            desired = policy_target
        limits = _tensor(self.robot.data.soft_joint_pos_limits)[:, self.action._joint_ids]
        desired = torch.maximum(torch.minimum(desired, limits[..., 1]), limits[..., 0])
        delta = (desired - self._last_target).clamp(-2.0 * self.dt, 2.0 * self.dt)
        self._last_target = self._last_target + delta
        if not self._hold_target:
            reached = bool((desired - self._last_target).abs().max().item() <= 0.02)
            self._handoff_ready = reached and (self.phase != "resuming" or self.policy_blend >= 1.0)
        if self.phase == "idle" and self._handoff_ready and bool((policy_target - desired).abs().max().item() <= 1e-6):
            self._release_limit = False
        return (self._last_target - self.action._offset) / self.action._scale
