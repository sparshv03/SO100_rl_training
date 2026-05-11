import mujoco

model = mujoco.MjModel.from_xml_path("/home/sparsh-ubuntu/rl_projects/so100_rl/gym_so100/assets/scene_so100.xml")

print("Number of actuators:", model.nu)

for i in range(model.nu):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))