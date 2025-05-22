import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion
import threading
import queue
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import Header

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

    def send_cloud(self, np_points):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'
        fields = [
            point_cloud2.PointField(name='x', offset=0, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='y', offset=4, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='z', offset=8, datatype=point_cloud2.PointField.FLOAT32, count=1)
        ]
        cloud_msg = point_cloud2.create_cloud(header, fields, np_points)
        self._queue_cloud.put(cloud_msg)