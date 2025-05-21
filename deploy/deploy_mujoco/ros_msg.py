import rospy
from geometry_msgs.msg import PoseStamped, Point, Quaternion
import threading
import queue
import time

class RosMsg:
    def __init__(self):
        rospy.init_node('ros_msg_pub_node', anonymous=True)
        self._pub_pose = rospy.Publisher('h1_pose', PoseStamped, queue_size=10)
        self._queue_pose = queue.Queue()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        self._FRAME_ID = "ros_msg"
        
    def _run(self):
        while True:
            try:
                pose_msg = self._queue_pose.get(block=False)
                self._pub_pose.publish(pose_msg)
            except queue.Empty:
                pass
            time.sleep(0.1)
                
    def sent_pose(self, position_xyz, quaternion_xyzw):
        posi_x, posi_y, posi_z = position_xyz
        quat_x, quat_y, quat_z, quat_w = quaternion_xyzw
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = self._FRAME_ID
        pose_msg.pose.position = Point(x=posi_x, y=posi_y, z=posi_z)
        pose_msg.pose.orientation = Quaternion(x=quat_x, y=quat_y, z=quat_z, w=quat_w)
        self._queue_pose.put(pose_msg)
        