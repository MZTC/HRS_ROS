#!/usr/bin/python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg

class HeightmapToPointCloud:
    def __init__(self):
        rospy.init_node('heightmap_to_pc2_node')

        # --- 配置区 ---
        self.image_path = rospy.get_param('~image_path', 'your_map.png')
        self.world_x = rospy.get_param('~world_x', 20.0) 
        self.world_y = rospy.get_param('~world_y', 20.0)
        self.max_z = rospy.get_param('~max_z', 2.0)
        self.frame_id = rospy.get_param('~frame_id', 'map')
        self.z_offset = rospy.get_param('~z_offset', 0.0)
        
        # 采样步长：边缘补全建议设为 1 或 2
        self.stride = rospy.get_param('~density_stride', 1) 
        self.v_step = rospy.get_param('~vertical_step', 0.5)
        self.pub_duaration = rospy.get_param('~pub_duaration', 10)

        # 垂直补点阈值：当相邻像素高度差超过此值时，进行补点（单位：米）
        self.edge_threshold = 0.1 
        
        self.pub = rospy.Publisher('/heightmap_pointcloud', PointCloud2, queue_size=1)
        
        # 处理并存储点云消息
        self.pc2_msg = self.process_heightmap()

    def process_heightmap(self):
        # 以灰度模式读取图片
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            rospy.logerr(f"无法读取图片，请检查路径: {self.image_path}")
            return None

        h, w = img.shape
        rospy.loginfo(f"读取图片成功: {w}x{h}. 正在计算边缘增强点云...")

        # 1. 基础映射：计算每个像素对应的 (x, y, z)
        res_x = self.world_x / float(w)
        res_y = self.world_y / float(h)
        
        # 矢量化计算基础点
        cols = np.arange(0, w, self.stride)
        rows = np.arange(0, h, self.stride)
        jj, ii = np.meshgrid(cols, rows)
        
        # 物理坐标映射
        x_base = (jj - w/2.0) * res_x
        y_base = (h/2.0 - ii) * res_y
        
        # 高度映射 (0-255 -> 0-max_z)
        # 注意：这里改为 0 为地面，你也可以根据需要改回 (intensity-127.5) 的逻辑
        intensity_base = img[ii, jj].astype(np.float32)
        z_base = ((intensity_base - 127.5) / 127.5) * self.max_z + self.z_offset
        # 2. 边缘补齐逻辑 (修正维度问题版)
        base_points = np.stack([x_base, y_base, z_base, intensity_base], axis=-1).reshape(-1, 4)
        points_list = [base_points]
        
        # 用来临时存放所有新增的垂直点，最后统一转换维度
        extra_points = []

        for r in range(0, x_base.shape[0] - 1):
            for c in range(0, x_base.shape[1] - 1):
                curr_z = z_base[r, c]
                right_z = z_base[r, c+1]
                down_z = z_base[r+1, c]
                
                def get_vertical_points(z1, z2, x, y, intensity):
                    diff = abs(z1 - z2)
                    if diff > self.edge_threshold:
                        num_points = int(diff / self.v_step)
                        if num_points > 0:
                            z_pts = np.linspace(z1, z2, num_points + 2)[1:-1]
                            # 重点：这里生成 (num_points, 4) 的二维数组
                            pts = np.zeros((len(z_pts), 4))
                            pts[:, 0] = x
                            pts[:, 1] = y
                            pts[:, 2] = z_pts
                            pts[:, 3] = intensity
                            return pts
                    return None

                # 检查右侧
                v_pts_r = get_vertical_points(curr_z, right_z, x_base[r,c], y_base[r,c], intensity_base[r,c])
                if v_pts_r is not None:
                    extra_points.append(v_pts_r)

                # 检查下方
                v_pts_d = get_vertical_points(curr_z, down_z, x_base[r,c], y_base[r,c], intensity_base[r,c])
                if v_pts_d is not None:
                    extra_points.append(v_pts_d)

        # 3. 合并
        if extra_points:
            # 将 list 中的多个二维数组合并成一个大的二维数组
            extra_points_concat = np.concatenate(extra_points, axis=0)
            points_list.append(extra_points_concat)

        # 现在 points_list 里的所有元素都是二维的 (N, 4)
        all_points = np.concatenate(points_list, axis=0).astype(np.float32)

        print("\n" + "="*30)
        print(f" 物理范围 : X={self.world_x}m, Y={self.world_y}m")
        print(f" Z轴范围  : {all_points[:,2].min():.2f}m ~ {all_points[:,2].max():.2f}m")
        print(f" 总点数   : {len(all_points)} (已补齐边缘)")
        print("="*30)

        return self.create_point_cloud(all_points, self.frame_id)

    def create_point_cloud(self, points, frame_id):
        msg = PointCloud2()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = frame_id
        
        msg.height = 1
        msg.width = len(points)
        
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1),
        ]

        msg.is_bigendian = False
        msg.point_step = 16 
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = points.tobytes()

        return msg

    def run(self):
        if self.pc2_msg is None:
            return
            
        rate = rospy.Rate(1/self.pub_duaration) 
        while not rospy.is_shutdown():
            self.pc2_msg.header.stamp = rospy.Time.now()
            self.pub.publish(self.pc2_msg)
            rate.sleep()

if __name__ == '__main__':
    try:
        node = HeightmapToPointCloud()
        node.run()
    except rospy.ROSInterruptException:
        pass