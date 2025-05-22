from threading import Thread
from ros_msg import RosMsg
from mujoco_data import MujocoData
import glfw
import time
from queue import Queue
import mujoco
import numpy as np
import cv2
from deep_to_cloud import DeepToCloud


class StatusOp:
    def __init__(self, m, d, pose_body_name, camera_name):
       self._m = m
       self._d = d
       self._pose_body_name = pose_body_name
       self._camera_id = m.camera(camera_name).id
       self._camera_renderer = self._get_camera_renderer(m)
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
               self._send_cloud_msg(self._camera_renderer, self._camera_id, self._d, self._ros_msg)
           else:
               pass        

    def update(self):
        self._queue.put(1)
        self._queue.put(2)

    def _send_pose_msg(self, d, body_name, ros_msg):
        posi = MujocoData.get_position(d, body_name)
        quat = MujocoData.get_quat(d, body_name)
        ros_msg.send_pose(posi, quat)

    def _get_camera_renderer(self, m):
        renderer = mujoco.Renderer(m, 480, 640)
        return renderer
    
    def _get_rgb_image(self, renderer, camera_id, d):
        renderer.update_scene(d, camera=camera_id)
        rgb = renderer.render()
        return rgb

    def _get_depth_image(self, renderer, camera_id, d):
        renderer.enable_depth_rendering()
        renderer.update_scene(d, camera=camera_id)
        depth = renderer.render()

        # 深度图后处理（可选）
        # 1. 将最近距离设为0
        depth -= depth.min()

        # 2. 计算有效深度范围（排除背景/无限远点）
        valid_depths = depth[depth < float('inf')]

        # 3. 基于有效深度进行归一化
        if len(valid_depths) > 0:
            depth /= 2 * valid_depths.mean()
        
        # 4. 裁剪并转换为8位图像
        pixels = 255 * np.clip(depth, 0, 1)

        depth_img = pixels.astype(np.uint8)
        renderer.disable_depth_rendering()

        return depth_img

    def _send_cloud_msg(self, renderer, camera_id, d, ros_msg):
        rgb_img = self._get_rgb_image(renderer, camera_id, d)
        depth_img = self._get_depth_image(renderer, camera_id, d)

        cv2.namedWindow("H1 Cameras", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        cv2.moveWindow("H1 Cameras", 1100, 0)            
        cv2.imshow("H1 Cameras", rgb_img)
        cv2.waitKey(1)

        pcd = DeepToCloud.get_pcd(rgb_img, depth_img)
        intensity = depth_img.flatten()[depth_img.flatten() > 0]
        ros_msg.send_cloud(pcd.points, intensity)