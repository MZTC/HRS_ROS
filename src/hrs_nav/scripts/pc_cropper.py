#!/usr/bin/env python3
import rospy
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Header

class PointCloudCropper:
    def __init__(self):
        rospy.init_node('custom_pc_cropper')

        # --- 参数读取 ---
        self.crop_size_x = rospy.get_param('~crop_x', 4.0)  
        self.crop_size_y = rospy.get_param('~crop_y', 4.0)
        self.model_name  = rospy.get_param('~model_name', 'scout')
        self.world_frame = rospy.get_param('~world_frame', 'world')
        
        in_topic  = rospy.get_param('~input_topic', '/gazebo_full_ground_cloud')
        out_topic = rospy.get_param('~output_topic', '/velodyne_points/points_downsampled')
        publish_rate = rospy.get_param('~publish_rate', 50.0) # 每秒发布10次

        # --- 内部状态 ---
        self.full_cloud_data = None
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.has_new_pos = False # 标志位：是否有新位置

        # --- 定义点云字段 (只定义一次，节省开销) ---
        self.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1),
        ]

        # --- 订阅与发布 ---
        # 1. 全量地图通常很大，我们只接一次或者低频接收
        rospy.Subscriber(in_topic, PointCloud2, self.full_cloud_cb)
        
        # 2. 订阅位置（这个频率很高，现在它不会被阻塞了）
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_cb)
        
        self.pub = rospy.Publisher(out_topic, PointCloud2, queue_size=1)

        # 3. 【核心重构】：开启定时器，独立于回调之外执行裁剪
        rospy.Timer(rospy.Duration(1.0 / publish_rate), self.timer_cb)

        rospy.loginfo(f"Cropper started. Rate: {publish_rate}Hz, Model: {self.model_name}")

    def full_cloud_cb(self, msg):
        """ 解析全量地图，这步很重，只在收到新地图时跑一次 """
        # rospy.loginfo("Processing full ground cloud... please wait.")
        gen = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity"))
        self.full_cloud_data = np.array(list(gen), dtype=np.float32)
        # rospy.loginfo(f"Full cloud loaded: {len(self.full_cloud_data)} points.")

    def model_cb(self, msg):
        """ 高频更新位置，现在它只是简单的赋值，绝不阻塞 """
        if self.model_name in msg.name:
            idx = msg.name.index(self.model_name)
            self.robot_x = msg.pose[idx].position.x
            self.robot_y = msg.pose[idx].position.y
            self.has_new_pos = True

    def timer_cb(self, event):
        """ 定时器回调：负责干重活 """
        if self.full_cloud_data is not None and self.has_new_pos:
            self.crop_and_publish()

    def crop_and_publish(self):
        # 1. 计算裁剪边界
        half_x = self.crop_size_x / 2.0
        half_y = self.crop_size_y / 2.0
        
        x_min, x_max = self.robot_x - half_x, self.robot_x + half_x
        y_min, y_max = self.robot_y - half_y, self.robot_y + half_y

        # 2. NumPy 快速裁剪
        points = self.full_cloud_data
        mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & \
               (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
        
        cropped = points[mask]

        if len(cropped) == 0:
            return
        # cropped = self.density_boost(cropped)
        # 3. 封装消息
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.world_frame
        
        # 使用高效的 create_cloud
        out_msg = pc2.create_cloud(header, self.fields, cropped)
        self.pub.publish(out_msg)

    def density_boost(self, points):
        """
        通过对每个点进行微小偏移复制，人为增加点云密度（廉价但有效）
        """
        offset = 0.05 # 2厘米偏移
        p1 = points.copy()
        p1[:, 0] += offset
        p2 = points.copy()
        p2[:, 1] += offset
        return np.vstack((points, p1, p2))

if __name__ == '__main__':
    try:
        PointCloudCropper()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass