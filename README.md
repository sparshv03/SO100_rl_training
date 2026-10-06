# Training an SO-100 Arm for Pick-and-Place with SAC

Curriculum-based reinforcement learning for a simulated SO-100 robotic arm in MuJoCo, using Soft Actor-Critic (SAC) and nothing but reward shaping and environment design: no demonstrations, no hand-crafted controllers, no trajectory planners.

**Authors:** Sparsh Vaidya

**Course:** Reinforcement Learning-II, NYU (May 2026)

---

## Overview

Pick-and-place is a multi-phase manipulation problem (approach, grasp, lift, transport, release) with sparse, delayed rewards and a continuous action space. This project asks how far a single algorithm (SAC) can get when guided only by carefully designed reward functions and a curriculum of increasingly harder subtasks.

A policy is trained on one task and then used to warm-start the next:

```
TouchCube  ->  GraspCube  ->  PickAndPlace
```

The most useful outcome of the project is a diagnosis of **why warm-started curriculum transfer breaks**: reward scale consistency across curriculum stages turned out to be a necessary condition for successful transfer (see [Key Findings](#key-findings)).

## Results at a Glance

| Run | Task | Steps | Final ep_rew_mean | Success rate |
|---|---|---|---|---|
| 1 | TouchCube | 300k | 0.87 | 0% |
| 2 | TouchCube | 1M | ~150 | 38-40% |
| 3 | GraspCube | 2.68M | 342 | 27% |
| 4 | PickAndPlace v1 | 2.68M | 97 (down from 271) | 0% |
| 5 | PickAndPlace v2 | 2.49M | 401 (up from 150, peak ~540) | 0% |

- **TouchCube** was solved: the agent reliably reaches and touches the cube (~213 steps per episode on average).
- **GraspCube** did not converge to a reliable grasp-and-lift. The policy settled into a local optimum of repeatedly approaching and touching the cube.
- **PickAndPlace v1** collapsed because of a reward scale mismatch after warm-starting.
- **PickAndPlace v2** fixed the scale issue and showed a healthy, rising reward curve, but did not complete the full five-stage chain (transport and release) within the training budget.

This is an honest negative-plus-diagnostic result rather than a solved benchmark. The failures are documented in detail because they were the most informative part of the project.

## Demos

### TouchCube
The trained policy approaches the cube and makes gripper contact within 5 cm. This is the one task that was solved (38-40% success).





https://github.com/user-attachments/assets/a4265d62-c119-46cd-9bcb-d6a7e6b18307



### Grasp but no pick
GraspCube behavior: the policy approaches the cube and makes contact but never commits to closing the gripper and lifting. It oscillates near the cube instead (the local optimum discussed in Key Findings).


https://github.com/user-attachments/assets/875d90f1-12d7-4926-ba9e-c25b2b0ffc53




### Pick-and-place failure
PickAndPlace v2: the policy approaches, touches, and partially lifts the cube, but does not complete transport and release inside the bin, so episodes time out at 500 steps.


https://github.com/user-attachments/assets/63b37268-0a2b-4937-b256-76afcd19b8f0



## Environment

Built on a modified fork of [`gym-so100-c`](https://github.com/ilonajulczuk/gym-so100-c), with physics from MuJoCo via `dm_control`, wrapped as a Gymnasium environment.

| Item | Details |
|---|---|
| Robot | SO-100, 6 actuated joints (5 arm joints + gripper) |
| Physics | MuJoCo, 20 ms timestep (50 Hz), scene defined in `so100_transfer_cube.xml` |
| Observation | 15-D vector: cube position (3), bin position (3), end-effector position (3), joint angles (6) |
| Action | 6-D in [-1, 1]: `[waist, shoulder, elbow, forearm_roll, wrist_rotate, gripper]`, mapped to joint position targets |
| Cube spawn | x in [-0.25, -0.15] m, y in [0.30, 0.60] m, z = 0.05 m (uniform random) |
| Bin | Fixed position across episodes |
| Observations | Pixel inputs are not used; the state is kept compact on purpose |

Contact detection uses MuJoCo's contact pairs directly (eight gripper jaw-pad geoms against `red_box`), and bin containment is checked against the bin's axis-aligned bounding box.

## Task Design and Rewards

All tasks share the robot, simulator, observation and action spaces. **The reward function is the only thing that changes between curriculum phases.** Every task returns `4.0` on success, and the maximum non-success shaping reward per step is kept below that so the success signal is never ambiguous.

| Task | Max steps | Success condition | Reward highlights |
|---|---|---|---|
| `SO100TouchCube` | 300 | Gripper touching cube and end-effector within 5 cm | 5-tier nested distance shaping (max 2.0), +1.0 contact bonus, -0.2 step penalty |
| `SO100GraspCube` | 400 | Touching, not touching table, cube lifted >= 4 cm | Approach shaping capped at 1.5, touch +1.0, lift +1.0 and scaled lift up to +0.5, -0.05 step penalty |
| `SO100PickAndPlace` (v2) | 500 | Cube inside bin and no gripper contact | Approach (1.5), touch (1.0), lift (0.5), transport toward bin (0.4), over-bin (0.2), -0.05 step penalty |

## Training Setup

- **Algorithm:** SAC from [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), with two-layer 256-unit MLP policy and critic networks
- **Parallelism:** `SubprocVecEnv` with 4 environments (~16 fps stable). Eight environments were tried first but caused CPU thermal throttling that dropped throughput from ~22 fps to 4-5 fps.
- **Observation normalization:** `VecNormalize` with reward normalization disabled so logged rewards stay interpretable. Normalization statistics must be saved and loaded alongside the model.
- **Compute:** Training is CPU-bound (physics stepping dominates), so a GPU gives little benefit for this setup.
- **Checkpointing:** Every 10k steps the model, `VecNormalize` stats, and replay buffer (~400 MB) are saved, so training can resume without a cold-start exploration phase.
- **Logging:** TensorBoard (`outputs/logs/TASK_NAME/`) plus a plain-text training log.

```bash
tensorboard --logdir outputs/logs
```

### Practical bugs fixed along the way

- **Rendering:** Wayland/GLX conflicts were resolved by switching to the OSMesa backend with `MUJOCO_GL=osmesa`.
- **Checkpoint naming mismatch:** manually saved models and `CheckpointCallback` checkpoints name their `VecNormalize` files differently, so the resume logic handles both conventions.
- **Replay buffer `n_envs` mismatch:** a buffer saved with 8 environments crashes when loaded into a 4-environment run, so the buffer is discarded and recreated when `n_envs` differs.
- **Visualization bug:** rendering was going to a different environment instance, fixed with a `get_inner_env` accessor.

## Key Findings

1. **Reward scale must stay consistent across curriculum steps.**
   Reducing the approach reward ceiling from 1.5 to 0.6 when warm-starting (PickAndPlace v1) wrecked previously learned approach behavior, even though the task structure was unchanged. The critic still expected high values for near-cube states, the TD error pushed those Q-values down, and the actor learned to avoid those states. Restoring the approach scale (v2) preserved the skill and let reward keep climbing. Practical rule: when warm-starting, add new reward terms freely, but do not shrink shaping that the policy was already trained on.

2. **Dense shaping can trap SAC in a local optimum for contact-rich tasks.**
   On GraspCube, hovering near the cube and repeatedly touching it yields steady shaping reward, while the coordinated "close the gripper and lift" sequence is temporarily worse under the reward and hard to discover with entropy-driven exploration. The factorised SAC action distribution also makes tightly coupled gripper and lift timing hard to learn.

3. **SAC finds rare rewards quickly once it sees them.**
   In TouchCube, success jumped from 0% to ~40% within about 2,400 steps of the first contact event, thanks to replay-buffer reuse.

## Debugging Tools

Three helper scripts live in `my_scripts/`:

| Script | Purpose |
|---|---|
| `visualize_policy.py` | Runs a trained SAC policy (with its `VecNormalize` stats) in a pygame window with a HUD showing distance to cube, contact status, lift height, and per-step reward. Supports `--save-video` for headless MP4 rendering. |
| `teleop_keyboard.py` | Keyboard teleoperation: number keys select joints, arrow keys move them, `O`/`C` open/close the gripper, `V` toggles recording, `R` resets. |
| `reward_probe.py` | Manually pose the robot and see a live breakdown of every reward component (distance, contact flag, shaping, bonus, step penalty, total). Used to verify rewards before each training run. |

```bash
# Visualize a trained policy
python my_scripts/visualize_policy.py --task PickAndPlace

# Render to video without a display
python my_scripts/visualize_policy.py --task PickAndPlace --save-video out.mp4
```

## Installation

```bash
# Clone this repository, then install the environment from local source
# (installing gym_so100 from a package index caused issues; local install fixed it)
pip install -e <path-to-gym-so100-c>

# Core dependencies
pip install stable-baselines3 mujoco dm_control gymnasium pygame tensorboard

# Headless rendering
export MUJOCO_GL=osmesa
```

> Add your training entry-point commands here (for example how to launch TouchCube, GraspCube, and PickAndPlace runs and how to resume from a checkpoint).

## Limitations and Future Work

- The full pick-and-place chain (transport and release) was not achieved within the ~2.5M-step budget, and GraspCube never converged.
- Results are from a single seed per run and simulation only, with no real-hardware evaluation.
- **Hindsight Experience Replay (HER)** could densify learning signal around near-miss grasp transitions.
- **Seeding the replay buffer with 10-20 teleoperated demonstrations** (collected with `teleop_keyboard.py`) could give SAC an initial trajectory distribution for grasp-and-lift.
- **Sim-to-real transfer** of the TouchCube policy to a physical SO-100, using external pose tracking for the state-based observation.

## References

1. Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor," ICML 2018.
2. Haarnoja et al., "Soft Actor-Critic Algorithms and Applications," arXiv:1812.05905, 2018.
3. Raffin et al., "Stable-Baselines3: Reliable Reinforcement Learning Implementations," JMLR 2021.
4. Todorov, Erez, Tassa, "MuJoCo: A Physics Engine for Model-Based Control," IROS 2012.
5. Tunyasuvunakool et al., "dm_control: Software Package for Physics-Based Simulation and Reinforcement Learning," Software Impacts, 2020.
6. Andrychowicz et al., "Hindsight Experience Replay," NeurIPS 2017.
7. Bengio et al., "Curriculum Learning," ICML 2009.
8. Ng, Harada, Russell, "Policy Invariance under Reward Transformations," ICML 1999.
9. [`gym-so100-c`](https://github.com/ilonajulczuk/gym-so100-c) and [LeRobot](https://github.com/huggingface/lerobot).
