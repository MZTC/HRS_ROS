#!/usr/bin/env python3
import rospy
import tf
import numpy as np
from geometry_msgs.msg import PoseStamped, Twist

class ScoutControllerV3:
    def __init__(self):
        rospy.init_node('mop_controller')
        self.tf_listener = tf.TransformListener()
        
        # --- 参数化配置 ---
        self.goal_topic = rospy.get_param('~goal_topic', '/local_move_goal')
        self.cmd_topic = rospy.get_param('~cmd_topic', '/cmd_vel')
        
        self.v_max = rospy.get_param('~v_max', 0.7)
        self.w_max = rospy.get_param('~w_max', 1.0)
        self.lookahead = rospy.get_param('~arrival_tolerance', 0.4)
        
        self.world_frame = rospy.get_param('~world_frame', 'world')
        self.robot_frame = rospy.get_param('~robot_frame', 'base_link')

        self.target_pose = None
        self.sub_goal = rospy.Subscriber(self.goal_topic, PoseStamped, self.goal_callback)
        self.pub_cmd = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        
        self.rate = rospy.Rate(20)

    def goal_callback(self, msg):
        self.target_pose = msg

    def run(self):
        while not rospy.is_shutdown():
            if not self.target_pose:
                self.rate.sleep()
                continue
            
            try:
                (trans, rot) = self.tf_listener.lookupTransform(self.world_frame, self.robot_frame, rospy.Time(0))
                dx = self.target_pose.pose.position.x - trans[0]
                dy = self.target_pose.pose.position.y - trans[1]
                _, _, yaw = tf.transformations.euler_from_quaternion(rot)
                
                target_yaw = np.arctan2(dy, dx)
                alpha = (target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
                dist = np.hypot(dx, dy)
                
                cmd = Twist()
                if dist > self.lookahead:
                    # 角度偏差补偿：偏差越大，线速度越小
                    cmd.linear.x = self.v_max * np.cos(alpha) * min(1.0, dist)
                    # 防止倒车
                    if cmd.linear.x < 0: cmd.linear.x = 0.1 
                    
                    cmd.angular.z = np.clip(2.0 * alpha, -self.w_max, self.w_max)
                else:
                    cmd.linear.x = 0; cmd.angular.z = 0
                
                self.pub_cmd.publish(cmd)
            except:
                continue
            self.rate.sleep()

if __name__ == '__main__':
    ScoutControllerV3().run()