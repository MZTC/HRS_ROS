#!/usr/bin/env python3
import rospy
import tf
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray

class ObstacleAwareLocalPlannerVis:
    def __init__(self):
        rospy.init_node('mop_local_planner')
        self.tf_listener = tf.TransformListener()
        
        # --- 参数化配置 ---
        self.path_topic = rospy.get_param('~path_topic', '/mop/path')
        self.obs_topic = rospy.get_param('~obs_topic', '/velodyne_points')
        self.goal_topic = rospy.get_param('~goal_topic', '/local_move_goal')
        self.lookahead_dist = rospy.get_param('~lookahead_dist', 1.5) # 建议减小一点，更贴合路径
        self.obs_threshold = rospy.get_param('~collision_radius', 0.4) # 根据车宽调整
        self.world_frame = rospy.get_param('~world_frame', 'world')
        self.robot_frame = rospy.get_param('~robot_frame', 'base_link')
        self.z_min = rospy.get_param('~z_min', 0.1)  # 调高一点，避开地面干扰
        self.z_max = rospy.get_param('~z_max', 0.8)
        
        self.global_path = None
        self.obstacles = []
        self.last_nearest_idx = 0 # 记录上一次匹配到的路径索引，防止倒退

        # --- 订阅与发布 ---
        self.sub_path = rospy.Subscriber(self.path_topic, Path, self.path_callback)
        self.sub_obs = rospy.Subscriber(self.obs_topic, PointCloud2, self.obs_callback)
        self.pub_local_goal = rospy.Publisher(self.goal_topic, PoseStamped, queue_size=1)
        
        # --- 可视化发布者 ---
        self.pub_vis_obs = rospy.Publisher('~vis_obstacles', Marker, queue_size=1)
        self.pub_vis_search = rospy.Publisher('~vis_search_space', MarkerArray, queue_size=1)
        
        self.rate = rospy.Rate(10)

    def path_callback(self, msg):
        # 收到新路径时，重置索引记录
        self.global_path = msg
        self.last_nearest_idx = 0
        rospy.loginfo("Local Planner: New Global Path Received")

    def obs_callback(self, msg):
        # 提取指定高度范围内的点云
        points = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        # 注意：这里假设雷达点已经在 robot_frame 下，如果不是，需要 TF 转换
        self.obstacles = [[p[0], p[1], p[2]] for p in points if self.z_min < p[2] < self.z_max]
        self.visualize_obstacles()

    def is_collision(self, x, y):
        if not self.obstacles: return False
        # 这里进行简单的 2D 碰撞检测
        obs_arr = np.array(self.obstacles)[:, :2]
        dists = np.linalg.norm(obs_arr - np.array([x, y]), axis=1)
        return np.any(dists < self.obs_threshold)

    def find_pure_goal(self, curr_pos):
        """ 
        核心修复：基于‘最近点索引’的单向搜索逻辑
        防止目标点跳回已经走过的路径点
        """
        if not self.global_path or len(self.global_path.poses) == 0:
            return None

        poses = self.global_path.poses
        
        # 1. 在当前索引附近寻找最近点，更新进度
        # 为了性能和稳定性，只从 last_nearest_idx 开始往后找 20 个点范围内最近的
        search_range = poses[self.last_nearest_idx : self.last_nearest_idx + 50]
        if not search_range:
            search_range = poses[self.last_nearest_idx:]
            
        dists_to_robot = [np.hypot(p.pose.position.x - curr_pos[0], 
                                   p.pose.position.y - curr_pos[1]) for p in search_range]
        
        current_nearest_in_range = np.argmin(dists_to_robot)
        self.last_nearest_idx += current_nearest_in_range

        # 2. 从最近点开始向后搜索第一个超过 lookahead_dist 的点
        for i in range(self.last_nearest_idx, len(poses)):
            dist = np.hypot(poses[i].pose.position.x - curr_pos[0], 
                            poses[i].pose.position.y - curr_pos[1])
            if dist > self.lookahead_dist:
                return [poses[i].pose.position.x, poses[i].pose.position.y, poses[i].pose.position.z]
        
        # 3. 如果后面没有足够远的点，则锁定终点
        last_p = poses[-1].pose.position
        return [last_p.x, last_p.y, last_p.z]

    def search_safe_goal(self, target):
        """ 当主目标碰撞时，在周围采样一个安全的替代点 """
        marker_array = MarkerArray()
        final_target = target
        found = False

        # 采样策略：多层圆环
        radii = [0.5, 1.0, 1.5]
        angles = np.linspace(0, 2*np.pi, 16)
        
        idx = 0
        for r in radii:
            for angle in angles:
                nx = target[0] + r * np.cos(angle)
                ny = target[1] + r * np.sin(angle)
                
                m = Marker()
                m.header.frame_id = self.world_frame
                m.header.stamp = rospy.Time.now()
                m.ns = "search_points"
                m.id = idx
                m.type = Marker.SPHERE
                m.scale.x = m.scale.y = m.scale.z = 0.15
                m.pose.position.x, m.pose.position.y, m.pose.position.z = nx, ny, target[2]
                
                if not found and not self.is_collision(nx, ny):
                    final_target = [nx, ny, target[2]]
                    found = True
                    m.color.g = 1.0; m.color.a = 1.0 # 选中的点为绿色
                else:
                    m.color.r = 1.0; m.color.a = 0.2 # 碰撞或未选为红色透明
                
                marker_array.markers.append(m)
                idx += 1
        
        self.pub_vis_search.publish(marker_array)
        return final_target

    def visualize_obstacles(self):
        if not self.obstacles: return
        marker = Marker()
        marker.header.frame_id = self.robot_frame 
        marker.header.stamp = rospy.Time.now()
        marker.ns = "local_obs"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.scale.x = 0.1; marker.scale.y = 0.1
        marker.color.r = 1.0; marker.color.a = 0.6
        for p in self.obstacles:
            pt = Point(); pt.x, pt.y, pt.z = p[0], p[1], p[2]
            marker.points.append(pt)
        self.pub_vis_obs.publish(marker)

    def publish_goal(self, target):
        msg = PoseStamped()
        msg.header.frame_id = self.world_frame
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = target
        msg.pose.orientation.w = 1.0
        self.pub_local_goal.publish(msg)

    def run(self):
        while not rospy.is_shutdown():
            try:
                # 获取机器人当前在世界坐标系下的位置
                (trans, rot) = self.tf_listener.lookupTransform(self.world_frame, self.robot_frame, rospy.Time(0))
                
                if self.global_path:
                    target = self.find_pure_goal(trans)
                    
                    if target:
                        # 检查前瞻点是否碰撞
                        if self.is_collision(target[0], target[1]):
                            target = self.search_safe_goal(target)
                        else:
                            # 没碰撞时清除搜索可视化
                            clear_msg = MarkerArray()
                            c = Marker(); c.action = Marker.DELETEALL
                            clear_msg.markers.append(c)
                            self.pub_vis_search.publish(clear_msg)
                        
                        self.publish_goal(target)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass
            
            self.rate.sleep()

if __name__ == '__main__':
    try:
        planner = ObstacleAwareLocalPlannerVis()
        planner.run()
    except rospy.ROSInterruptException:
        pass