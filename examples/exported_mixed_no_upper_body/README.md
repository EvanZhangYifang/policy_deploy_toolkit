# Four-mode Kyon policy: deployment handoff

This folder is intended to be copied on its own. `policy.onnx` is the deterministic actor from the mixed-mode PPO checkpoint. It accepts a float32 observation of shape `[1, 47]` and returns a float32 action of shape `[1, 12]`. The ONNX graph includes the checkpoint's learned observation normalization; **do not normalize its input a second time**. `metadata.json` records joint names, mode references, action scale, and other exported settings. The inference test vectors in this folder can be used to check that a new runtime loads and evaluates the model correctly.

## Contents and verification

- `policy.onnx`: deterministic 47-input, 12-output actor with normalization embedded.
- `metadata.json`: policy contract, mode-specific references, selectors and model/checkpoint hashes.
- `deploy_metadata.yaml`: the same policy contract in the mixed-mode `policy_deploy_toolkit` metadata format.
- `test_vectors.json`: four checkpoint-derived observation/action pairs.
- `verify_mixed_policy.py`: standalone ONNX Runtime verification of the model hash and all four vectors.
- `standing_transition.py`: simulator controller source for reference only.

After copying this folder, install Python, NumPy and ONNX Runtime in the target environment and run:

```bash
cd /path/to/exported_mixed_no_upper_body
python verify_mixed_policy.py
```

All four modes should report `PASS`. This checks inference equivalence for the supplied vectors; it does not validate the robot interface or physical behavior. Use `deploy_metadata.yaml` with a version of `policy_deploy_toolkit` that supports mixed-mode policies and the ROS mode selector; older toolkit versions cannot load this policy. Consult the toolkit's instructions for the ROS interface.

This policy controls the **12 leg joints only**. The upper body does not appear in its observation or action vectors. However, it was trained with the upper body physically present. Removing it changes mass, inertia and center of mass; export without upper-body joints is not evidence that the policy is safe or stable on a robot without the upper body. The policy was trained in fixed modes, not on mode transitions. `standing_transition.py` in this folder is a copy of the flat-ground simulation playback controller as a reference; it depends on Isaac Lab, PyTorch and the original package and is not a standalone or real-robot controller.

## Policy input and output

Run one inference every **0.02 s (50 Hz)**. Values are in radians, radians/second and meters/second, in the robot's conventions and body frame, with +x forward, +y left and +z up. Resolve the IMU mounting transform and joint signs against the simulator conventions before testing. The policy observation is concatenated exactly as follows; indices are zero-based and ranges exclude the end:

| Indices | Size | Value |
|---|---:|---|
| 0:3 | 3 | Base angular velocity `[wx, wy, wz]` in the base frame, rad/s. |
| 3:6 | 3 | Unit gravity vector in the base frame: `R_world_base.T @ [0, 0, -1]`; approximately `[0, 0, -1]` when upright. |
| 6:9 | 3 | Body-frame base velocity command `[vx, vy, yaw_rate]`, in m/s, m/s, rad/s. |
| 9:21 | 12 | Measured joint positions minus the active mode's reference positions, in the joint order below. |
| 21:33 | 12 | Joint velocity minus default velocity, which is zero for these joints. |
| 33:45 | 12 | Previous **applied** action in active-mode coordinates, not the current network output. |
| 45:47 | 2 | Active mode selectors from the table below. |

All terms except the selectors were clipped to `[-100, 100]` during observation preparation. There is no additional observation scaling. Noise was used during training; do not inject artificial training noise into deployment. The actor uses no observation history buffer beyond the previous applied action. It does not use a terrain height scan.

The joint order is:

```text
hip_roll_1, hip_pitch_1, knee_pitch_1,
hip_roll_2, hip_pitch_2, knee_pitch_2,
hip_roll_3, hip_pitch_3, knee_pitch_3,
hip_roll_4, hip_pitch_4, knee_pitch_4
```

The reference vectors below use precisely that order. They are also available in `metadata.json`.

| Mode | Selectors | Reference joint positions (rad) |
|---|---|---|
| BB | `[-1, -1]` | `[0.15, -0.7, 1.4, -0.15, 0.7, -1.4, -0.15, -0.7, 1.4, 0.15, 0.7, -1.4]` |
| BF | `[-1, 1]` | `[0.15, -0.7, 1.4, -0.15, 0.7, -1.4, -0.15, 0.7, -1.4, 0.15, -0.7, 1.4]` |
| FB | `[1, -1]` | `[0.15, 0.7, -1.4, -0.15, -0.7, 1.4, -0.15, -0.7, 1.4, 0.15, 0.7, -1.4]` |
| FF | `[1, 1]` | `[0.15, 0.7, -1.4, -0.15, -0.7, 1.4, -0.15, 0.7, -1.4, 0.15, -0.7, 1.4]` |

For fixed-mode operation, the network's 12 outputs `a` are raw position actions; use `q_target = q_ref[mode] + 0.5 * a`. The action scale is 0.5 rad and the training action was not clipped. The next observation's previous action is `(actual_last_position_target - q_ref[mode]) / 0.5`. In normal unfiltered playback this equals the preceding raw actor output, but it must reflect the *filtered command actually applied* during transitions or clamping. At startup choose a known mode that matches the robot's actual starting pose, initialize the last commanded target consistently, and derive the initial previous action from it. Never initialize a mode solely by changing the two selectors while leaving a mismatched reference.

The saved simulator motor gains were stiffness 300 and damping 10 for the leg joints, with actuator effort/saturation parameters of 120. These are simulation parameters, not approved robot gains or guaranteed motor torque limits. Set hardware gains, limits and safety according to the robot's actuator and control system.

Example inference, assuming ONNX Runtime Python is installed:

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession('policy.onnx', providers=['CPUExecutionProvider'])
name = session.get_inputs()[0].name
obs = np.zeros((1, 47), dtype=np.float32)
# Replace EVERY field with live measurements, active reference, previous
# applied action, command and selector as described above before use.
action = session.run(None, {name: obs})[0][0]  # shape (12,)
```

The example only illustrates model invocation; zeros are not valid live robot measurements.

## Transition design from the simulation

This section describes the original simulation controller. The deployment toolkit uses operator confirmation and IMU/joint feedback; it does not implement the simulator's contact-force or odometry checks.

The standing transition is a **separate state machine** outside the ONNX graph. The following describes the simulation reference so another environment can implement it; it is not a validated hardware procedure. It assumes level ground and all four feet can remain planted. Before using anything on hardware, provide and validate joint position/velocity, correctly transformed gravity and angular velocity, body-frame linear velocity, world planar body position and heading, and per-foot contact/load estimates. A four-contact estimate and state estimator must be suitable for the physical robot. If any required input is missing, stale, inconsistent or unreliable, block transitions and follow the robot's safety controller. Tune and validate limits independently; the numeric values here are the simulation settings, not hardware safety limits.

Maintain `active_mode`, `requested_mode`, `q_ref`, last commanded physical joint target `q_last`, previous applied action, phase timers and a stance anchor. Reject a new mode request unless idle. The active mode remains the source until the handoff condition; set velocity command to zero during transition except for the controlled ramps below. At each 20 ms tick:

1. Read robot state and update the state machine, including command gating. During the commit tick below, update selector, reference and rebased previous action **together before building the next observation**.
2. Form the 47-value observation and run the destination or source actor for the currently active mode. Convert its raw action to a physical `q_policy = q_ref + 0.5 * a`.
3. Choose the appropriate scripted, blended, held or policy target as described below. Clip the desired target to the robot's independently validated soft joint limits. Rate-limit its per-joint change to at most `2.0 * dt` radians from `q_last` while transitioning, yielding `q_cmd`; send `q_cmd` through the robot's safety and actuator interface.
4. Set `q_last = q_cmd`; update previous applied action to `(q_cmd - q_ref) / 0.5`. Detect and handle missing/nonfinite inputs, bad estimator values, timing misses and actuator faults under the robot's own safety design.

Simulation phases:

- **Braking:** follow the source policy while ramping the requested velocity command toward zero over 0.44 s with `1 - smoothstep(t / 0.44)`.
- **Waiting:** remain at zero command until all four feet are loaded, the robot is upright, and it is still continuously for 0.14 s. Save world planar base position as the stance anchor.
- **Extending:** smoothly straighten the signed knee angles from source nominal to zero over 1.08 s and generate a planted-foot target using the geometry below. Start bending only once the trajectory is complete, all measured knee magnitudes are at most 0.18 rad, and the robot again meets the full still check.
- **Bending:** smoothly move signed knee angles from zero toward the destination nominal over 0.96 s using the planted-foot geometry and endpoint correction below. As soon as **all 12 measured joints** remain within 0.30 rad of the destination reference for 0.06 s, with the handoff support, posture and motion checks satisfied, commit the destination mode without waiting for the bend trajectory to finish.
- **Commit:** atomically change the selector and joint reference to the destination, and rebase previous applied action to `(q_last - q_ref_destination) / 0.5`. This avoids a discontinuity in the network's joint-relative and previous-action observations. Recompute observations immediately in the destination coordinates.
- **Resuming:** start blending the balanced scripted target into the destination policy target immediately. Blend fraction is `smoothstep(blend_elapsed / 0.80)`; rate-limit the resulting target. Keep the velocity command zero until blend reaches 1 and the rate-limited target is within 0.02 rad of its current desired target in every joint. Then smoothly ramp the requested velocity command over 0.40 s. Keep rate limiting until the target catches up and also constrain the first idle target.
- **Fault/blocked:** freeze the last position target when support or upright posture is lost during extending or bending; freeze the phase timer and clear the handoff debounce. If support/posture does not recover within 1.0 s, block. During resuming, freeze the blend and target on support loss; block after 0.20 s. Block after 15 s overall. A blocked state holds the last target and zero velocity command in the simulator; the physical robot must instead use its separately validated fault and fall-response strategy. Do not automatically resume or assume holding a pose is safe.

Here `smoothstep(x)` is clamped to `[0,1]` and equals `x^3 * (10 - 15*x + 6*x^2)`.

The simulation's support check considers all four vertical world-frame contact-force magnitudes `> 16 N` and upright if projected gravity z `<= -cos(7 degrees)`. The waiting/extension still check additionally requires root linear-speed norm `<= 0.04 m/s`, angular-speed norm `<= 0.18 rad/s`, and maximum absolute leg joint speed `<= 0.45 rad/s`. During the handoff debounce, require the same support/upright checks plus planar body speed `<= 0.08 m/s`, absolute vertical body speed `<= 0.40 m/s`, body angular-speed norm `<= 0.25 rad/s` and max absolute joint speed `<= 2.50 rad/s`. All these thresholds are from simulation; sensor noise, false contacts, different body motion and control delays must be handled in the target system.

### Planted-leg geometry

For each leg, `mirror = [1, -1, 1, -1]` and `fore_aft = [1, 1, -1, -1]`. The signed knee value `k` equals that leg's nominal knee angle multiplied by its mirror sign. For an interpolation fraction `p = smoothstep(t / duration)`, use `k_source * (1-p)` during extension and `k_destination * p` during bending. The geometry in meters is the following vectorized formula, applied per leg:

```text
reach = 0.346 + 0.300*cos(k)
offset = 0.300*sin(k)
length = hypot(reach, offset)
hip_pitch = -atan2(offset, reach)
rolling_pitch = (0.04 / (length + 0.04)) * (hip_pitch + k)
hip_pitch = hip_pitch - rolling_pitch
length = length * cos(rolling_pitch)
hip_roll = acos(clamp(0.1463 / sqrt(length^2 + 0.2226^2), -1, 1))
           - atan(length / 0.2226)
q_geometry_leg = mirror * [fore_aft * hip_roll, hip_pitch, k]
```

For bending, correct the small geometry/reference mismatch using `q_script = geometry(k_destination*p) + p*(q_ref_destination - geometry(k_destination))`. This ensures the geometric endpoint equals the trained destination reference; all arithmetic is in policy joint order.

### Simulation balance correction

The playback controller applies a bounded balance correction to `q_script` using the stance anchor, world planar position, body heading `yaw`, body planar velocity, projected gravity and body angular velocity. This is heuristic feedback specific to the simulator, not a hardware balance controller:

```text
e_world = clamp(anchor_xy - body_position_xy, -0.15, 0.15)
e_body = [[cos(yaw), sin(yaw)], [-sin(yaw), cos(yaw)]] @ e_world
         - 0.32 * body_linear_velocity_xy
roll_signs  = [-1, -1,  1,  1]
pitch_signs = [ 1, -1,  1, -1]
q_hip_roll[i]  += clamp(-0.65 * e_body.y * roll_signs[i],  -0.10, 0.10)
q_hip_pitch[i] += clamp(-0.65 * e_body.x * pitch_signs[i], -0.10, 0.10)
roll  = -0.30 * gravity_body.y + 0.045 * angular_velocity_body.x
pitch =  0.30 * gravity_body.x + 0.045 * angular_velocity_body.y
foot_x = [0.39, 0.39, -0.39, -0.39]
foot_y = [0.23, -0.23, 0.23, -0.23]
dz[i] = roll * foot_y[i] - pitch * foot_x[i]
dk[i] = clamp(5.0 * dz[i], -0.12, 0.12) * clamp(q_knee[i] / 0.20, -1, 1)
q_knee[i] += dk[i]
q_hip_pitch[i] -= 0.46 * dk[i]
```

During the blend interpolate `q_desired = (1-blend)*q_balanced_script + blend*q_policy`; then apply position and rate limits. Loss of support freezes both the blend and target.

## Validation before physical use

First verify the inference test vectors and observation/action ordering in the new runtime. In simulation, exercise fixed-mode standing and commands in each mode, then all 12 directed transitions on level ground, including deliberate support loss, timeouts, bad sensors, rate and joint-limit faults. Compare the implementation against the included simulation reference and record contact, joint errors, commanded targets and blend state. The previous CPU tests covered only one deterministic seed on level ground. In a 60-second BB→BF test the body tilt slowly increased to about 5.7 degrees, so the previously observed growing body wave is not proven resolved. Validate long-duration standing, disturbances and different robot configurations independently before any physical deployment.
