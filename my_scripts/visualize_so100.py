import os
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import mujoco.viewer

# scene_so100.xml is a <mujocoinclude> fragment — use the full scene file
model = mujoco.MjModel.from_xml_path(
    "/home/sparsh-ubuntu/rl_projects/SO100_RL/gym_so100-c/gym_so100/assets/so100_transfer_cube.xml"
)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
