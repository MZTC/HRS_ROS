#!/usr/bin/env python3
import rospy
import tf
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry

class SimOdomPublisher:
    def __init__(self):
        rospy.init_node('sim_odom_publisher')
        
        # --- 参数配置 ---
        self.model_name = rospy.get_param('~model_name', 'scout')
        self.world_frame = rospy.get_param('~world_frame', 'world')
        self.base_frame = rospy.get_param('~base_frame', 'base_link')
        self.pub_odom_topic = rospy.get_param('~odom_topic', '/odom')
        self.publish_tf = rospy.get_param('~publish_tf', True)

        self.tf_broadcaster = tf.TransformBroadcaster()
        self.odom_pub = rospy.Publisher(self.pub_odom_topic, Odometry, queue_size=1)
        
        # 【核心修复】：记录上一次发布的时间戳
        self.last_publish_time = rospy.Time(0)
        
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback, queue_size=1)
        rospy.loginfo(f"Sim Odom Fix: Preventing redundant timestamps for {self.model_name}")

    def model_states_callback(self, msg):
        try:
            # 获取当前仿真时间
            now = rospy.Time.now()

            # 【核心逻辑】：如果时间戳没有更新，直接跳过，不发布重复数据
            if now <= self.last_publish_time:
                return

            if self.model_name in msg.name:
                idx = msg.name.index(self.model_name)
            else:
                indices = [i for i, s in enumerate(msg.name) if self.model_name in s]
                if not indices: return
                idx = indices[0]

            curr_pose = msg.pose[idx]
            curr_twist = msg.twist[idx]

            # 发布 TF
            if self.publish_tf:
                self.tf_broadcaster.sendTransform(
                    (curr_pose.position.x, curr_pose.position.y, curr_pose.position.z),
                    (curr_pose.orientation.x, curr_pose.orientation.y, curr_pose.orientation.z, curr_pose.orientation.w),
                    now,
                    self.base_frame,
                    self.world_frame
                )

            # 发布 Odom
            odom = Odometry()
            odom.header.stamp = now
            odom.header.frame_id = self.world_frame
            odom.child_frame_id = self.base_frame
            odom.pose.pose = curr_pose
            odom.twist.twist = curr_twist
            self.odom_pub.publish(odom)

            # 更新最后一次发布时间
            self.last_publish_time = now

        except Exception:
            pass

if __name__ == '__main__':
    publisher = SimOdomPublisher()
    rospy.spin()