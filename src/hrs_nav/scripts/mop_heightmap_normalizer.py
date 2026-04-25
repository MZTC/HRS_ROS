#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import tf
import cv2
import numpy as np
from gazebo_msgs.msg import ModelStates

class HeightmapNormalizer:
    def __init__(self):
        rospy.init_node('heightmap_normalizer_node')

        # ==================== 参数获取 ====================
        self.target_model = rospy.get_param('~target_model', 'scout/')
        self.world_frame  = rospy.get_param('~world_frame', 'world')
        self.map_frame    = rospy.get_param('~map_frame', 'map')
        self.world_size_x = rospy.get_param('~world_size_x', 100.0)
        self.world_size_y = rospy.get_param('~world_size_y', 100.0)
        self.max_z        = rospy.get_param('~max_z', 10.0)
        self.image_path   = rospy.get_param('~image_path', 'T1.png')
        self.z_offset     = rospy.get_param('~z_offset', 0.0)
        
        # 新增：同步周期设置为 2.0 秒
        self.sync_interval = rospy.get_param('~sync_interval', 2.0) 
        # =================================================

        # 加载高程图
        self.img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None:
            rospy.logerr("【错误】无法加载高程图: %s", self.image_path)
            return
        self.h, self.w = self.img.shape

        self.br = tf.TransformBroadcaster()
        self.delta_z = 0.0
        self.is_initialized = False

        self.print_params()

        # 订阅 Gazebo 状态
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback)

    def print_params(self):
        print("\n" + "="*40)
        print("   高度动态归一化节点   ")
        print("="*40)
        print(f"目标模型    : {self.target_model}")
        print(f"同步周期    : {self.sync_interval} 秒")
        print(f"高程图路径  : {self.image_path}")
        print(f"图像尺寸    : {self.w} x {self.h}")
        print(f"物理范围    : X={self.world_size_x}m, Y={self.world_size_y}m")
        print(f"最大高度(Z) : {self.max_z}m")
        print(f"Z轴偏移     : {self.z_offset}m")
        print(f"TF 映射     : {self.world_frame} -> {self.map_frame}")
        print("="*40 + "\n")

    def get_expected_z_from_map(self, x, y):
        u = int((x / self.world_size_x + 0.5) * (self.w - 1))
        v = int((0.5 - y / self.world_size_y) * (self.h - 1))
        u = max(0, min(self.w - 1, u))
        v = max(0, min(self.h - 1, v))
        pixel_val = self.img[v, u]
        # 使用居中映射公式
        # return (pixel_val / 255.0) * self.max_z
        return ((pixel_val - 127.5) / 127.5) * self.max_z

    def callback(self, data):
        """回调函数仅负责根据最新的位置计算 delta_z"""
        try:
            idx = data.name.index(self.target_model)
            pose = data.pose[idx]
            
            actual_z = pose.position.z
            expected_z = self.get_expected_z_from_map(pose.position.x, pose.position.y)
            
            # 更新 delta_z 缓存
            self.delta_z = actual_z - expected_z + self.z_offset
            
            if not self.is_initialized:
                rospy.loginfo("初始高度差锁定: %.3f m", self.delta_z)
                self.is_initialized = True
        except (ValueError, IndexError):
            pass

    def run(self):
        # 设置循环频率为 0.5Hz (即 2秒一次)
        rate = rospy.Rate(1.0 / self.sync_interval)
        
        while not rospy.is_shutdown():
            if self.is_initialized:
                # 定时发布 TF
                self.br.sendTransform(
                    (0, 0, self.delta_z),
                    (0, 0, 0, 1),
                    rospy.Time.now(),
                    self.map_frame,
                    self.world_frame
                )
                # rospy.loginfo("已发布高度补偿 TF: %.3f m", self.delta_z)
            
            rate.sleep()

if __name__ == '__main__':
    try:
        node = HeightmapNormalizer()
        node.run()
    except rospy.ROSInterruptException:
        pass