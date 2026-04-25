import requests
import numpy as np
from grid_map_msgs.msg import GridMap
from geometry_msgs.msg import PoseStamped, Point
import rospy

class SubGoal:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
    
    def __str__(self):
        return f"x:{self.x}, y:{self.y}, z:{self.z}"

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

def normalize_vector(v):
    """
    将向量 v 转化为单位向量。

    参数：
    v (numpy.ndarray): 输入的向量

    返回：
    numpy.ndarray: 规范化后的单位向量
    """
    norm = np.linalg.norm(v)
    # if norm == 0:
    #     return v * 0
    return v / (norm + 1e-9)

def manhattan_distance(point1, point2):
    """
    计算两个点之间的曼哈顿距离。

    参数:
    point1 (tuple): 第一个点的坐标 (x1, y1)
    point2 (tuple): 第二个点的坐标 (x2, y2)

    返回:
    int: 两点之间的曼哈顿距离
    """
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

class DecisionMaker:

    def __init__(self, agent_addr, size=512):
        self.clear_url = f"{agent_addr}/clear"
        self.add_url = f"{agent_addr}/add"
        self.decision_url = f"{agent_addr}/decision"
        self.size = size

    def clear(self):
        
        try:
            response = requests.post(self.clear_url, timeout=0.5)
            res_data = response.json()
            
            if res_data.get('status') == 'success':
                return True
            else:
                return False
                
        except Exception as e:
            rospy.logwarn("[DecisionMaker/Clear] API communication error: %s", str(e))
            return False

    
    def add(self, curr_pos:PoseStamped):
        """
        将当前机器人位置加入历史列表
        """
        x = curr_pos.pose.position.x
        y = curr_pos.pose.position.y

        payload = {
            "point": [x, y]
        }
        try:
            response = requests.post(self.add_url, json=payload, timeout=0.5)
            res_data = response.json()
            
            if res_data.get('status') == 'success':
                return True
            else:
                return False
                
        except Exception as e:
            rospy.logwarn("[DecisionMaker/Add] API communication error: %s", str(e))
            return False
    
    def desion(self, grid_map: GridMap, curr_pos:PoseStamped, global_goal:PoseStamped, step=0):
        """
        @grid_map: 局部地图
        @goal: 全局目标
        @curr_pos: 机器人当前位置
        """
        try:
            res, h_map_np = process_grid_map(grid_map_msg=grid_map)
        except Exception as e:
            rospy.logerr("[DecisionMaker] GridMap preprocess failed: %s", str(e))
            return None
        

        curr_x, curr_y = curr_pos.pose.position.x, curr_pos.pose.position.y
        gg_x, gg_y = global_goal.pose.position.x, global_goal.pose.position.y

        curr_p = np.array((curr_x, curr_y))
        gg_p = np.array((gg_x, gg_y))

        vec = normalize_vector(gg_p - curr_p)
        dist = manhattan_distance(gg_p, curr_p) / self.size 

        payload = {
            "h_map": h_map_np.tolist(),
            "goal": [vec[0], vec[1], dist],
            "resolution": res,
            "size": self.size,
            "step": step
        }

        # --- 3. 调用 Flask 服务 ---
        try:
            response = requests.post(self.decision_url, json=payload, timeout=1)
            res_data = response.json()
            
            if res_data.get('status') == 'success':
                sub_goal_data = res_data["sub_goal"]
                
                # sub_goal_data 是相对于 base_link_aligned 的偏移量 (dx, dy, dz)
                dx, dy, dz = sub_goal_data[0], sub_goal_data[1], sub_goal_data[2]

                # 创建全局目标的 PoseStamped
                abs_sub_goal = PoseStamped()
                # 显式指定坐标系为 odom，这样后续规划器就能知道这是一个全局固定点
                abs_sub_goal.header.frame_id = "odom" 
                abs_sub_goal.header.stamp = rospy.Time.now()

                # 核心转换公式：全局位置 = 当前位置 + 局部偏移
                # 因为 base_link_aligned 轴向与 odom 一致，所以直接相加即可
                abs_sub_goal.pose.position.x = curr_pos.pose.position.x + dx
                abs_sub_goal.pose.position.y = curr_pos.pose.position.y + dy
                abs_sub_goal.pose.position.z = curr_pos.pose.position.z + dz
                
                # 姿态通常可以保持与 global_goal 一致，或者设为默认
                abs_sub_goal.pose.orientation = global_goal.pose.orientation

                rospy.loginfo("[Decision] New Global Sub-goal in Odom: x=%.2f, y=%.2f", 
                              abs_sub_goal.pose.position.x, abs_sub_goal.pose.position.y)
                
                return abs_sub_goal
                
        except Exception as e:
            rospy.logwarn("[Decision] API communication error: %s", str(e))
        
        return None
