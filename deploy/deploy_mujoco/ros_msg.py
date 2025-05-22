import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion
import threading
import queue
from sensor_msgs.msg import PointCloud2, PointField
import rospy
import numpy as np


class RosMsg:
    def __init__(self):
        rospy.init_node('ros_msg_pub_node', anonymous=True)

        self._pub_pose = rospy.Publisher('robot_pose', PoseStamped, queue_size=10)
        self._queue_pose = queue.Queue()
        self._t_pose = threading.Thread(target=self._run_pose, daemon=True)
        self._t_pose.start()

        self._pub_cloud = rospy.Publisher('robot_cloud', PointCloud2, queue_size=10)
        self._queue_cloud = queue.Queue()
        self._t_cloud = threading.Thread(target=self._run_cloud, daemon=True)
        self._t_cloud.start()

        self._FRAME_ID = "ros_msg"
        
    def _run_pose(self):
        while True:
            pose_msg = self._queue_pose.get()
            self._pub_pose.publish(pose_msg)

    def _run_cloud(self):
        while True:
            cloud_msg = self._queue_cloud.get()
            self._pub_cloud.publish(cloud_msg)
                
    def send_pose(self, position_xyz, quaternion_xyzw):
        posi_x, posi_y, posi_z = position_xyz
        quat_x, quat_y, quat_z, quat_w = quaternion_xyzw
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = self._FRAME_ID
        pose_msg.pose.position = Point(x=posi_x, y=posi_y, z=posi_z)
        pose_msg.pose.orientation = Quaternion(x=quat_x, y=quat_y, z=quat_z, w=quat_w)
        self._queue_pose.put(pose_msg)

    def send_cloud(self, points, intensity):
        msg = PointCloud2()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self._FRAME_ID
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        msg.is_bigendian = False
        msg.point_step = 16  # 每个点占用16字节（x, y, z, intensity）
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = False

        # 将点云和强度信息打包到字节流中
        points_with_intensity = np.zeros((len(points), 4), dtype=np.float32)
        points_with_intensity[:, :3] = points
        points_with_intensity[:, 3] = intensity
        msg.data = points_with_intensity.tobytes()
        self._queue_cloud.put(msg)