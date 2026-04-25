#!/usr/bin/env python3
import rospy
import numpy as np
import tf
import threading
from sensor_msgs.msg import PointCloud2
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float32MultiArray, MultiArrayDimension


class LocalElevationMap:
    def __init__(self):
        rospy.init_node("local_elevation_map")

        # --- 参数配置 ---
        self.resolution = rospy.get_param("~resolution", 0.03)
        self.length_x = rospy.get_param("~length_x", 15.0)  # 对应你裁剪的 15m
        self.length_y = rospy.get_param("~length_y", 15.0)
        self.map_frame = rospy.get_param("~map_frame", "world")
        self.robot_frame = rospy.get_param("~robot_frame", "base_link")
        self.publish_rate = rospy.get_param("~publish_rate", 20.0)

        # 计算网格维度
        self.nx = int(round(self.length_x / self.resolution))
        self.ny = int(round(self.length_y / self.resolution))

        # --- TF 与 线程安全 ---
        self.tf_listener = tf.TransformListener()
        self.lock = threading.Lock()
        self.cloud_cache = None

        # --- 订阅与发布 ---
        self.sub = rospy.Subscriber("/heightmap_pointcloud_crop", PointCloud2, self.cloud_cb, queue_size=1)
        self.pub = rospy.Publisher("/local_elevation_map", GridMap, queue_size=1)

        # 启动定时转换器 (独立于回调线程)
        rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.process_timer_cb)

        rospy.loginfo(f"Elevation Map Node Started. Center: {self.robot_frame}")

    def cloud_cb(self, msg):
        """ 极速接收点云 """
        try:
            # 根据 PointCloud2 的 point_step 解析
            depth = msg.point_step // 4
            arr = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, depth)
            with self.lock:
                self.cloud_cache = arr[:, :3].copy()
        except Exception as e:
            pass

    def get_robot_pose(self):
        """ 获取机器人相对于 world 的位置 """
        try:
            # 获取最新的变换
            (trans, rot) = self.tf_listener.lookupTransform(self.map_frame, self.robot_frame, rospy.Time(0))
            return trans[0], trans[1] # 返回 x, y
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            # rospy.logwarn("TF Waiting for transform...")
            return None, None

    def process_timer_cb(self, event):
        # ===== 1. 取点云 =====
        with self.lock:
            if self.cloud_cache is None:
                return
            points = self.cloud_cache.copy()

        # ===== 2. 机器人中心 =====
        cx, cy = self.get_robot_pose()
        if cx is None:
            return

        # ===== 3. 地图边界 (计算最大值作为基准) =====
        # 为了旋转 180 度，我们将索引 0 映射到物理坐标的最大值 (max_x, max_y)
        max_x = cx + self.length_x / 2.0
        max_y = cy + self.length_y / 2.0

        # ===== 4. 坐标 → 网格索引（实现 180 度翻转）=====
        # 使用 max - points 替代 points - min
        # 当 points 接近 max 时，索引为 0；当 points 接近 min 时，索引接近 nx/ny
        u = ((max_x - points[:, 0]) / self.resolution).astype(np.int32)
        v = ((max_y - points[:, 1]) / self.resolution).astype(np.int32)

        # ===== 5. 边界过滤 =====
        mask = (u >= 0) & (u < self.nx) & (v >= 0) & (v < self.ny)
        u, v, z = u[mask], v[mask], points[mask, 2]

        if u.size == 0:
            return

        # ===== 6. 构建 grid（使用 ny x nx）=====
        # GridMap 插件通常期望第一维是“列方向的长度”，但在 NumPy 习惯中我们对应为 Rows
        grid = np.full((self.ny, self.nx), -np.inf, dtype=np.float32)

        # ===== 7. 填充 =====
        # 这里的 flat_idx 计算方式决定了内存排布
        flat_idx = v * self.nx + u
        np.maximum.at(grid.ravel(), flat_idx, z)

        # ===== 8. 恢复空值 =====
        grid[grid == -np.inf] = np.nan

        # ===== 9. 发布 =====
        self.publish_grid_map(grid, cx, cy)

    def publish_grid_map(self, grid_data, cx, cy):
        # grid_data shape 是 (ny, nx)，即 (rows, cols)
        rows, cols = grid_data.shape 

        gm = GridMap()
        gm.info.header.stamp = rospy.Time.now()
        gm.info.header.frame_id = self.map_frame
        gm.info.resolution = float(self.resolution)
        gm.info.length_x = float(self.length_x)
        gm.info.length_y = float(self.length_y)
        gm.info.pose.position.x = float(cx)
        gm.info.pose.position.y = float(cy)
        gm.info.pose.orientation.w = 1.0

        gm.layers.append("elevation")
        gm.basic_layers = ["elevation"]

        data_msg = Float32MultiArray()

        # --- 这里的 Label 必须与 ny, nx 的顺序严格一致 ---
        # 既然你定义 flat_idx = v * nx + u (v是row, u是col)
        # 那么维度顺序应该是 [ny, nx]
        
        dim_row = MultiArrayDimension()
        dim_row.label = "column_index"  # 注意：GridMap 插件通常把第一维叫 column_index
        dim_row.size = rows             # 对应 ny
        dim_row.stride = rows * cols

        dim_col = MultiArrayDimension()
        dim_col.label = "row_index"     # 第二维叫 row_index
        dim_col.size = cols             # 对应 nx
        dim_col.stride = cols

        data_msg.layout.dim = [dim_row, dim_col]
        
        # 使用 numpy 的默认 C-order 扁平化
        data_msg.data = grid_data.astype(np.float32).ravel().tolist()

        gm.data.append(data_msg)
        self.pub.publish(gm)


if __name__ == "__main__":
    try:
        LocalElevationMap()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass