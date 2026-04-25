#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import tf
import csv
import os
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String  # 改用标准字符串消息

class PathManager:
    def __init__(self):
        rospy.init_node('path_manager_node')

        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.save_dir = rospy.get_param("~save_dir", os.path.expanduser("~/paths/"))
        self.publish_rate = rospy.get_param("~rate", 10)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.listener = tf.TransformListener()
        self.recorded_path = Path()
        self.recorded_path.header.frame_id = self.odom_frame

        self.path_pub = rospy.Publisher('/robot_path', Path, queue_size=10, latch=True)
        
        # 改为订阅话题：向该话题发送消息即触发保存
        rospy.Subscriber('/save_path_cmd', String, self.handle_save_topic)

        rospy.loginfo("Path Manager Ready. To save, run:")
        rospy.loginfo("rostopic pub -1 /save_path_cmd std_msgs/String 'data: \"your_name\"'")

    def handle_save_topic(self, msg):
        """处理话题传来的保存指令"""
        file_name = msg.data.strip()
        if not file_name:
            file_name = "path_" + str(rospy.get_time())
        
        if not file_name.endswith('.csv'):
            file_name += '.csv'
        
        full_path = os.path.join(self.save_dir, file_name)

        if len(self.recorded_path.poses) > 0:
            if self.execute_save(full_path):
                self.recorded_path.poses = [] # 清空
                rospy.loginfo("SUCCESS: Path saved to %s", full_path)
            else:
                rospy.logerr("ERROR: Save failed.")
        else:
            rospy.logwarn("SKIP: Path is empty.")

    def execute_save(self, path):
        try:
            with open(path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
                for p in self.recorded_path.poses:
                    pos = p.pose.position
                    ori = p.pose.orientation
                    writer.writerow([pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w])
            return True
        except Exception as e:
            rospy.logerr("File writing error: %s", str(e))
            return False

    def run(self):
        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            try:
                now = rospy.Time(0)
                # 等待变换可用
                self.listener.waitForTransform(self.odom_frame, self.base_frame, now, rospy.Duration(0.1))
                (trans, rot) = self.listener.lookupTransform(self.odom_frame, self.base_frame, now)
                
                current_pose = PoseStamped()
                current_pose.header.stamp = rospy.Time.now()
                current_pose.header.frame_id = self.odom_frame
                current_pose.pose.position.x, current_pose.pose.position.y, current_pose.pose.position.z = trans
                current_pose.pose.orientation.x, current_pose.pose.orientation.y, current_pose.pose.orientation.z, current_pose.pose.orientation.w = rot

                if len(self.recorded_path.poses) > 0:
                    last_p = self.recorded_path.poses[-1].pose.position
                    dist = ((trans[0]-last_p.x)**2 + (trans[1]-last_p.y)**2)**0.5
                    if dist < 0.1: 
                        rate.sleep()
                        continue

                self.recorded_path.poses.append(current_pose)
                self.path_pub.publish(self.recorded_path)

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
            rate.sleep()

if __name__ == '__main__':
    try:
        mgr = PathManager()
        mgr.run()
    except rospy.ROSInterruptException:
        pass