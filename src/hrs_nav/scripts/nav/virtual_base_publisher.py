#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import tf
from geometry_msgs.msg import TransformStamped

def publish_aligned_base():
    rospy.init_node('virtual_base_publisher')
    
    br = tf.TransformBroadcaster()
    listener = tf.TransformListener()
    
    # 设置发布频率
    rate = rospy.Rate(50.0)
    
    target_frame = "base_link_aligned"
    source_frame = "world" # 或者 "odom"
    reference_frame = "base_link"

    rospy.loginfo("Starting to publish %s aligned with %s", target_frame, source_frame)

    while not rospy.is_shutdown():
        try:
            # 1. 获取从 world 到 base_link 的当前变换
            # 获取最新的变换 (Time(0))
            (trans, rot) = listener.lookupTransform(source_frame, reference_frame, rospy.Time(0))
            
            # 2. 发布新的坐标系
            # 位置(trans) 使用 base_link 的位置
            # 姿态(quaternion) 使用单位四元数 [0, 0, 0, 1]，使其轴向与 source_frame (world) 平行
            br.sendTransform(
                trans,
                (0.0, 0.0, 0.0, 1.0), 
                rospy.Time.now(),
                target_frame,
                source_frame
            )
            
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
            
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_aligned_base()
    except rospy.ROSInterruptException:
        pass