import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
from ros_msg import RosMsg
from mujoco_data import MujocoData
import glfw


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def send_pose_msg(d, body_name, ros_msg):
    posi = MujocoData.get_position(d, body_name)
    quat = MujocoData.get_quat(d, body_name)
    ros_msg.sent_pose(posi, quat)

class CameraViewer(object):
    def __init__(self):
        self.window = None
        self.context = None
        self.camera = None

def _get_camera_viewer(m, camera_name):
    camera_viewer = CameraViewer()
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    camera_viewer.window = glfw.create_window(320, 240, "camera_viewer", None, None)
    if not camera_viewer.window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window for left camera")
    glfw.make_context_current(camera_viewer.window)
    camera_viewer.context = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_100)
    camera_viewer.camera = mujoco.MjvCamera()
    camera_viewer.camera.fixedcamid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_viewer.camera.fixedcamid < 0:
        raise ValueError(f"Camera '{camera_name}' not found in model")
    camera_viewer.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    return camera_viewer

def _get_camera_image(m, d, camera_viewer):
    width, height = glfw.get_window_size(camera_viewer.window)
    scene = mujoco.MjvScene(m, maxgeom=1000)

    # 创建图像缓冲区
    img = np.zeros((height, width, 3), dtype=np.uint8)  # RGB 图像
    depth = np.zeros((height, width), dtype=np.float32)  # 深度图像

    # 渲染图像
    viewport = mujoco.MjrRect(0, 0, width, height)
    mujoco.mjv_updateScene(m, d, mujoco.MjvOption(), None, camera_viewer.camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, camera_viewer.context)    
    img = np.zeros((height, width, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(img, None, viewport, camera_viewer.context)

    # 处理深度数据
    depth_min = 0.1  # 最近深度（单位：米）
    depth_max = 10.0  # 最远深度（单位：米）
    depth = depth_min + (depth_max - depth_min) * (1 - depth)

    return depth

def send_cloud_msg(m, d, camera_viewer, ros_msg):
    depth = _get_camera_image(m, d, camera_viewer)    
    width, height = glfw.get_window_size(camera_viewer.window)

    # 生成点云
    pointcloud = []
    for v in range(height):
        for u in range(width):
            z = depth[v, u]
            if z > 0:
                x = (u - width / 2) * z / 100  # 假设焦距为 100
                y = (v - height / 2) * z / 100  # 假设焦距为 100
                pointcloud.append([x, y, z])
    cloud = np.array(pointcloud)
    print(cloud)


if __name__ == "__main__":
    # get config file name from command line
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    print(LEGGED_GYM_ROOT_DIR)
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    policy = torch.jit.load(policy_path)

    ros_msg = RosMsg()

    with mujoco.viewer.launch_passive(m, d) as viewer:
        camera_viewer = _get_camera_viewer(m, 'depth_camera')

        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9 : 9 + num_actions] = qj
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

                # send ros msg
                send_pose_msg(d, 'pelvis', ros_msg)
                send_cloud_msg(m, d, camera_viewer, ros_msg)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
