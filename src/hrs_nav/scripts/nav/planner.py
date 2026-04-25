import requests
import numpy as np
from grid_map_msgs.msg import GridMap
from geometry_msgs.msg import PoseStamped, Point
import rospy
from nav_msgs.msg import Path
import tf2_ros
import tf2_geometry_msgs


def process_grid_map(grid_map_msg:GridMap):
    # --- 1. 动态获取地图尺寸与数据 ---
    res = grid_map_msg.info.resolution
    # 稳健计算行列数
    rows = int(round(grid_map_msg.info.length_x / res))
    cols = int(round(grid_map_msg.info.length_y / res))
    center_x = int(rows//2)
    center_y = int(rows//2)
    
    try:
        # 获取 elevation 层索引
        layer_idx = grid_map_msg.layers.index("elevation")
    except ValueError:
        layer_idx = 0 
        

    # 提取数据并处理 NaN (API 推理不允许 NaN)
    h_map_np = np.array(grid_map_msg.data[layer_idx].data).reshape(rows, cols)
    h_map_np = np.nan_to_num(h_map_np, nan=0.0)
    h_map_np = h_map_np[:, ::-1]

    h_map_np = h_map_np - h_map_np[center_x][center_y]
    return res, h_map_np

class Planner:
    def __init__(self, planner_addr):
        """
        :param planner_url: Flask API 的完整地址 (如 http://localhost:5000/predict)
        """
        self.planner_url = f"{planner_addr}/plan"
        self.tf2_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer)

    def run_planner_inference(self, grid_map_msg, subgoal_in_base_a: PoseStamped, frame_id="base_link_aligned"):
        """
        核心推理逻辑：
        1. 从 GridMap 消息提取 elevation 数据
        2. 调用远程 Flask API
        3. 转换并返回 nav_msgs/Path 对象
        @subgoal: 必须为base_link_aligned坐标系下的！
        """

        
        try:
            # 获取 elevation 层索引
            layer_idx = grid_map_msg.layers.index("elevation")
        except ValueError:
            layer_idx = 0 
            
        try:
            # 提取数据并处理 NaN (API 推理不允许 NaN)
            res, h_map_np = process_grid_map(grid_map_msg) 
            h_map_data = h_map_np.tolist()

        except Exception as e:
            rospy.logerr("[Planner] GridMap preprocess failed: %s", str(e))
            return None

        # --- 2. 构造 API 请求负载 ---
        # subgoal_m 应该是相对于机器人中心的物理坐标 [x, y] (单位：米)
        payload = {
            "h_map": h_map_data,
            "goal": [subgoal_in_base_a.pose.position.x, subgoal_in_base_a.pose.position.y],
            "resolution": res,
        }
        
        # --- 3. 调用 Flask 服务 ---
        try:
            response = requests.post(self.planner_url, json=payload, timeout=0.8)
            res_data = response.json()
            
            if res_data.get('status') == 'success':
                # trajectory = res_data['full_trajectory']
                trajectory = res_data['way_points']
                # --- 4. 封装为 Path 消息并返回 ---
                local_path = self.generate_path_msg(trajectory, frame_id)
                global_path = self.transform_path_to_odom(local_path)
                return global_path
                
        except Exception as e:
            rospy.logwarn("[Planner] API communication error: %s", str(e))
        
        return None

    def generate_path_msg(self, trajectory, frame_id):
        """
        内部转换工具：将轨迹坐标点转为标准的 ROS Path 消息
        """
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = frame_id # 关键点：设置为对齐后的坐标系

        for pt in trajectory:
            pose = PoseStamped()
            pose.header.stamp = path.header.stamp
            pose.header.frame_id = frame_id
            
            # API 返回的是 odom2idx 的反变换结果，即相对于 map 中心的偏移米
            pose.pose.position.x = pt[0]
            pose.pose.position.y = pt[1]
            pose.pose.position.z = pt[2] # 局部规划通常忽略高度，或后续根据地图高度补全
            
            # 默认无旋转（路标点本身不携带方向信息）
            pose.pose.orientation.w = 1.0
            
            path.poses.append(pose)
            
        return path

    def transform_path_to_odom(self, local_path):
        """
        将 Path 消息从 base_link_aligned 转换到 odom
        """
        global_path = Path()
        global_path.header.stamp = rospy.Time.now()
        global_path.header.frame_id = "odom"

        try:
            for local_pose in local_path.poses:
                # 强制使用最新的变换，防止 Extrapolation Error
                local_pose.header.stamp = rospy.Time(0)
                
                # 使用 tf_buffer 转换每一个点
                global_pose = self.tf2_buffer.transform(
                    local_pose, 
                    "odom", 
                    timeout=rospy.Duration(0.5)
                )
                global_path.poses.append(global_pose)
            
            return global_path
        except Exception as e:
            rospy.logerr("[Planner] Path transformation failed: %s", str(e))
            return None