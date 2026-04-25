#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import tf
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

class PointCloudManualTransformer:
    def __init__(self):
        rospy.init_node('pc_manual_transformer', anonymous=True)

        # 参数配置
        self.target_frame = rospy.get_param('~target_frame', 'world')
        self.source_topic = rospy.get_param('~source_topic', '/heightmap_pointcloud')
        self.target_topic = rospy.get_param('~target_topic', '/heightmap_pointcloud_in_world')

        # 初始化传统 tf 监听器
        self.listener = tf.TransformListener()

        # 发布者和订阅者
        self.pub = rospy.Publisher(self.target_topic, PointCloud2, queue_size=1)
        self.sub = rospy.Subscriber(self.source_topic, PointCloud2, self.callback)

        rospy.loginfo("使用传统 tf 转换点云: [%s] -> [%s]", self.source_topic, self.target_frame)

    def callback(self, cloud_msg):
        try:
            # 获取变换矩阵
            self.listener.waitForTransform(self.target_frame, cloud_msg.header.frame_id, rospy.Time(0), rospy.Duration(5.0))
            (trans, rot) = self.listener.lookupTransform(self.target_frame, cloud_msg.header.frame_id, rospy.Time(0))
            matrix = self.listener.fromTranslationRotation(trans, rot)

            # 高效处理：利用 numpy 批量转换，且保留原始 field
            # 获取原始所有字段的数据
            pc_data = pc2.read_points(cloud_msg, skip_nans=True)
            points_attr = [list(p) for p in pc_data] # 这里包含了 x,y,z,intensity...
            
            new_points = []
            for p in points_attr:
                # 仅对前三个元素(x,y,z)进行矩阵运算
                pt_old = np.array([p[0], p[1], p[2], 1.0])
                pt_new = np.dot(matrix, pt_old)
                # 拼回原来的属性 (如 intensity)
                new_points.append([pt_new[0], pt_new[1], pt_new[2]] + p[3:])

            # 创建输出点云时，必须复制原有的 fields 结构！
            cloud_out = pc2.create_cloud(cloud_msg.header, cloud_msg.fields, new_points)
            cloud_out.header.frame_id = self.target_frame
            
            self.pub.publish(cloud_out)

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn("TF 查找失败: %s", str(e))

if __name__ == '__main__':
    try:
        PointCloudManualTransformer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

