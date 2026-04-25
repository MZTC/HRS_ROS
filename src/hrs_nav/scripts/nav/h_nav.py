#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import rospkg
import os

# 自动获取 hrs_nav 包的物理路径并拼接 scripts 目录
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('hrs_nav')
sys.path.append(f"{pkg_path}/scripts")

import rospy
import tf
import numpy as np
import threading
import math
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from grid_map_msgs.msg import GridMap
from visualization_msgs.msg import Marker
from nav.planner import Planner
from nav.decision_maker import DecisionMaker
import tf2_ros
import tf2_geometry_msgs


class HierarchicalNavigationNode:
    """
    local_subgoal: 在odom/world坐标系下的

    odom-a 坐标系就是 base_link_aligned 坐标系

    """
    def __init__(self):
        rospy.init_node('hierarchical_planner_node')

        # --- 参数配置 ---
        self.plan_frequency = rospy.get_param('~plan_frequency', 3)      # 局部规划频率
        self.goal_tolerance = rospy.get_param('~goal_tolerance', 5)      # 全局目标容差 (米)
        self.subgoal_tolerance = rospy.get_param('~subgoal_tolerance', 10)# 触发新SubGoal的距离 (米)
        self.push_threshold = rospy.get_param('~push_threshold', 5)      # 记录轨迹的距离阈值 (米)
        
        # --- 状态变量 ---
        self.global_goal = None
        self.local_subgoal = None

        self.local_map = None
        self.is_navigating = False
        self.last_pushed_pose = None  # 上次推送给决策器的位置

        self.step = 0
        
        # --- 工具实例化 ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 神经网络接口
        self.planner = Planner(
            planner_addr=os.getenv("PLANNER_ADDR")
        )

        self.decision_maker = DecisionMaker(
            agent_addr=os.getenv("AGENT_ADDR"),
            size=1024
        )

        # --- 订阅与发布 ---
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.global_goal_callback)
        rospy.Subscriber('/local_elevation_map', GridMap, self.map_callback)
        
        self.subgoal_pub = rospy.Publisher('/local_subgoal', Marker, queue_size=10)
        self.global_goal_pub = rospy.Publisher('/global_goal_vis', Marker, queue_size=10)
        self.path_pub = rospy.Publisher('/local_planned_path', Path, queue_size=10)
        
        # --- 启动核心驱动线程 ---
        # 该线程负责：1.轨迹推送 2.SubGoal切换判断 3.全局到达判断
        self.monitor_thread = threading.Thread(target=self.navigation_monitor_worker)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        # --- 启动局部规划定时任务 ---
        rospy.Timer(rospy.Duration(1.0/self.plan_frequency), self.low_level_planning_loop)
        
        # 调试用：固定发布一个全局目标
        # rospy.Timer(rospy.Duration(2.0), self.debug_global_goal)

        self.decision_maker.clear()
        

    # ==========================================
    # 核心逻辑：导航监控线程 (事件驱动)
    # ==========================================
    def navigation_monitor_worker(self):
        rate = rospy.Rate(10) # 10Hz 监控
        while not rospy.is_shutdown():
            if self.is_navigating and self.global_goal:
                curr_pose = self.get_current_pose()

                if not curr_pose:
                    rate.sleep()
                    continue

                # 1. 轨迹记录逻辑：按位移距离推送
                self._check_and_push_pose(curr_pose)

                # 2. 全局目标靠近判断：停止导航
                dist_to_global = self._calculate_dist(curr_pose, self.global_goal)
                if dist_to_global < self.goal_tolerance:
                    rospy.loginfo("Reached Global Goal. Stopping Navigation.")
                    self.stop_navigation()
                    rate.sleep()
                    continue

                # 3. SubGoal 靠近判断：获取新的 SubGoal
                # 如果当前没有 SubGoal，或者距离当前 SubGoal 足够近，则触发决策更新
                if self.local_subgoal is None:
                    rospy.loginfo("Initial Sub-goal request.")
                    self._trigger_high_level_decision(curr_pose)
                else:
                    # 注意：假设 SubGoal 是在 base_link 坐标系下的相对位置
                    local_subgoal_in_base_a = self.local_subgoal_in_base_a
                    if local_subgoal_in_base_a is None: continue

                    dist_to_sub = math.sqrt(local_subgoal_in_base_a.pose.position.x**2 + local_subgoal_in_base_a.pose.position.y**2)
                    if dist_to_sub < self.subgoal_tolerance:
                        rospy.loginfo("Near Sub-goal (%.2fm), requesting new one.", dist_to_sub)
                        self._trigger_high_level_decision(curr_pose)

            rate.sleep()

    def odom2base_a(self, pose_in_odom: PoseStamped):
        """
        将 odom 系下的点转换到 base_link_aligned 系
        """
        try:
            # 使用 transform 直接变换
            # target_frame: "base_link_aligned"
            pose_in_odom.header.stamp = rospy.Time(0)

            pose_in_odoma = self.tf_buffer.transform(
                pose_in_odom, 
                "base_link_aligned", 
                timeout=rospy.Duration(0.5)
            )
            return pose_in_odoma
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr("Transform odom to odom-a failed: %s", str(e))
            return None

    def base_a2odom(self, pose_in_odoma: PoseStamped):
        """
        将 base_link_aligned 系下的点转换回 odom 系
        """
        try:
            # target_frame: "odom"
            pose_in_odom.header.stamp = rospy.Time(0)
            pose_in_odom = self.tf_buffer.transform(
                pose_in_odoma, 
                "odom", 
                timeout=rospy.Duration(0.5)
            )
            return pose_in_odom
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr("Transform odom-a to odom failed: %s", str(e))
            return None


    @property
    def local_subgoal_in_base_a(self) -> PoseStamped:
        if self.local_subgoal is None: return None

        local_subgoal_in_odom = self.local_subgoal
        local_subgoal_in_base_a = self.odom2base_a(local_subgoal_in_odom)
        return local_subgoal_in_base_a



    def _check_and_push_pose(self, curr_pose):
        """逻辑1：记录线程。当移动超过阈值，推送当前位置"""
        if self.last_pushed_pose is None:
            self.decision_maker.add(curr_pose)
            self.last_pushed_pose = curr_pose
            return

        dist_moved = self._calculate_dist(curr_pose, self.last_pushed_pose)
        if dist_moved > self.push_threshold:
            self.decision_maker.add(curr_pose)
            self.last_pushed_pose = curr_pose

    def _trigger_high_level_decision(self, curr_pose):
        """逻辑2：当靠近 SubGoal 时，进行新的目标获取"""
        if self.local_map is None:
            rospy.logwarn_throttle(5, "Local map not received yet.")
            return

        rospy.loginfo("Triggering High-Level RL Decision...")
        new_subgoal = self.decision_maker.desion(
            grid_map=self.local_map,
            curr_pos=curr_pose, 
            global_goal=self.global_goal
        )
        
        if new_subgoal:
            self.local_subgoal = new_subgoal
            self.publish_subgoal_marker(self.local_subgoal)
            self.step += 1
        else:
            rospy.logerr("RL Decision Maker returned None.")

    def stop_navigation(self):
        """逻辑3：停止导航"""
        self.is_navigating = False
        self.global_goal = None
        self.local_subgoal = None
        self.last_pushed_pose = None
        self.decision_maker.clear()  # 清除 RL 端的历史状态

        self.path_pub.publish(Path()) # 清除路径显示

    # ==========================================
    # 基础回调与辅助函数
    # ==========================================
    def global_goal_callback(self, msg):
        rospy.loginfo("Global Goal Update.")
        self.global_goal = msg
        self.is_navigating = True
        self.decision_maker.clear() # 开始新任务前清空旧轨迹
        
        self.step = 0

        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "global_goal"
        marker.type = Marker.CYLINDER
        marker.pose.position.x = self.global_goal.pose.position.x
        marker.pose.position.y = self.global_goal.pose.position.y
        marker.scale.z = marker.scale.x = marker.scale.y = 1.5
        marker.color.a, marker.color.r = 1.0, 1.0 # 红色
        self.global_goal_pub.publish(marker)

    def map_callback(self, msg):
        self.local_map = msg

    def get_current_pose(self):
        try:
            # tf2 使用 lookup_transform，返回的是 TransformStamped 对象
            # 参数：目标坐标系，源坐标系，查询时间（Time(0)表示最新），超时时间
            transform = self.tf_buffer.lookup_transform(
                'odom', 
                'base_link', 
                rospy.Time(0), 
                rospy.Duration(0.5)
            )
            
            pose = PoseStamped()
            pose.header = transform.header # 包含 odom frame 和时间戳
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z
            pose.pose.orientation = transform.transform.rotation
            
            return pose
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("[TF2] Failed to get pose: %s", str(e))
            return None

    def _calculate_dist(self, p1, p2):
        return math.sqrt((p1.pose.position.x - p2.pose.position.x)**2 + 
                         (p1.pose.position.y - p2.pose.position.y)**2)

    def low_level_planning_loop(self, event):
        """固定频率的局部避障规划"""
        if not self.is_navigating or self.local_subgoal is None or self.local_map is None:
            return

        local_subgoal_in_base_a = self.local_subgoal_in_base_a
        if local_subgoal_in_base_a is None: return
    
        path_msg = self.planner.run_planner_inference(
            grid_map_msg=self.local_map, 
            subgoal_in_base_a=local_subgoal_in_base_a
        )
        if path_msg:
            path_msg.header.stamp = rospy.Time.now()
            self.path_pub.publish(path_msg)

    def publish_subgoal_marker(self, subgoal_point: PoseStamped):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "subgoal"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = subgoal_point.pose.position.x
        marker.pose.position.y = subgoal_point.pose.position.y
        marker.pose.position.z = subgoal_point.pose.position.z

        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 1
        marker.color.a, marker.color.r = 1.0, 1.0  # 绿色
        marker.color.g, marker.color.b = 1.0, 1.0
        self.subgoal_pub.publish(marker)

    def debug_global_goal(self, event):

        if self.global_goal is None:
            pose = PoseStamped()
            pose.header.frame_id = 'odom'
            pose.pose.position.x, pose.pose.position.y = 0, 0
            self.global_goal = pose
            self.is_navigating = True
            
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "global_goal"
        marker.type = Marker.CYLINDER
        marker.pose.position.x = self.global_goal.pose.position.x
        marker.pose.position.y = self.global_goal.pose.position.y
        marker.scale.x = marker.scale.y = 0.5
        marker.scale.z = 0.1
        marker.color.a, marker.color.r = 1.0, 1.0 # 红色
        self.global_goal_pub.publish(marker)

if __name__ == '__main__':
    try:
        node = HierarchicalNavigationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass