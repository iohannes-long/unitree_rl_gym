from threading import Thread
from ros_msg import RosMsg
from mujoco_data import MujocoData
import glfw
import time
from queue import Queue
import mujoco
import numpy as np
import cv2


class CameraViewer(object):
    def __init__(self):
        self.name = None
        self.window = None
        self.context = None
        self.camera = None

class StatusOp:
    def __init__(self, m, d, pose_body_name, camera_name):
       self._m = m
       self._d = d
       self._pose_body_name = pose_body_name
       self._camera_viewer = self._get_camera_viewer(m, camera_name)
       self._ros_msg = RosMsg()

       self._queue = Queue()
       self._t = Thread(target=self._run, args=(), daemon=True)
       self._t.start()

    def _run(self):
       while True:
           msg =self._queue.get()
           if msg == 1:
               self._send_pose_msg(self._d, self._pose_body_name, self._ros_msg)
           elif msg == 2:
               self._send_cloud_msg(self._m, self._d, self._camera_viewer, self._ros_msg)
           else:
               pass        

    def update(self):
        self._queue.put(1)
        self._queue.put(2)

    def _send_pose_msg(self, d, body_name, ros_msg):
        posi = MujocoData.get_position(d, body_name)
        quat = MujocoData.get_quat(d, body_name)
        ros_msg.send_pose(posi, quat)

    def _get_camera_viewer(self, m, camera_name):
        camera_viewer = CameraViewer()
        camera_viewer.name = camera_name
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        camera_viewer.window = glfw.create_window(640, 480, "camera_viewer", None, None)
        if not camera_viewer.window:
            glfw.terminate()
            raise RuntimeError(f"Failed to create GLFW window for {camera_name} camera")
        glfw.make_context_current(camera_viewer.window)
        camera_viewer.context = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_100)
        camera_viewer.camera = mujoco.MjvCamera()
        camera_viewer.camera.fixedcamid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_viewer.camera.fixedcamid < 0:
            raise ValueError(f"Camera '{camera_name}' not found in model")
        camera_viewer.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        return camera_viewer

    def _get_camera_image(self, m, d, camera_viewer):
        if not camera_viewer.window:
            glfw.terminate()
            raise RuntimeError(f"Failed to create GLFW window for {camera_viewer.name} camera")
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
        mujoco.mjr_readPixels(img, depth, viewport, camera_viewer.context)

        # 处理深度数据
        depth_min = 0.1  # 最近深度（单位：米）
        depth_max = 10.0  # 最远深度（单位：米）
        depth = depth_min + (depth_max - depth_min) * (1 - depth)

        cv2.namedWindow("H1 Cameras", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        cv2.moveWindow("H1 Cameras", 1100, 0)            
        cv2.imshow("H1 Cameras", img)
        cv2.waitKey(1)

        return img, depth

    def _send_cloud_msg(self, m, d, camera_viewer, ros_msg):
        img, depth = self._get_camera_image(m, d, camera_viewer)    
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
        ros_msg.send_cloud(cloud)