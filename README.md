# HRS_ROS 导航系统

[<video src="video/demo_low.mp4" controls width="100%"></video>
![Sample Video](video/demo_low.mp4)](https://github.com/user-attachments/assets/302ada8d-d725-4287-a5ac-58e0ad5f99d1)

本项目是一个集成**高层强化学习决策**与**底层路径规划**的 ROS 导航框架，专门针对非结构化地形设计。

---

## 📦 1. 环境依赖

- **操作系统**: Ubuntu 18.04  
- **ROS版本**: Melodic  
- **仿真平台**: Gazebo  

---

## 🚀 2. 安装步骤 (Install)

### 2.1 克隆仓库

进入 `catkin` 工作空间目录：

```bash
git clone https://github.com/MZTC/HRS_ROS.git
cd HRS_ROS

```

---

### 2.2 安装 Elevation Map 依赖

```bash
# look https://github.com/ANYbotics/elevation_mapping
```

---

### 2.3 编译项目

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

---

### 2.4 配置 Gazebo 模型路径

将项目中的 `gazebo_model` 目录下的子目录复制到系统 Gazebo 模型路径中：

```bash
cp -r HRS_ROS/gazebo_model/* ~/.gazebo/models/
```
---


## ▶️ 3. 使用指南 (Usage)

请按照以下顺序在不同终端中运行：

---

### 3.1 启动仿真环境

```bash
roslaunch hrs_nav start_gazebo.launch world:=T96
```

---

### 3.2 启动地图处理与控制

```bash
roslaunch hrs_nav start_mapping.launch world:=T96 length_x:=20 length_y:=20
```

该步骤将启动：

- 高程图构建（Elevation Mapping）
- 底层控制器
- RViz 可视化

---

### 3.3 启动导航模块

#### 记录轨迹

```bash
rosrun hrs_nav record_path.py
```

#### 启动导航决策

```bash
export PLANNER_ADDR=http://xxx:10067
export AGENT_ADDR=http://xxx:10068
rosrun hrs_nav h_nav.py
```

---

## 🧠 4. 系统结构

```
hrs_nav/
├── scripts/      # 模型推理与决策逻辑
├── utils/        # 坐标转换、地图处理与可视化
├── launch/       # 启动文件
```

---

## 🔧 5. [核心模块](src/hrs_nav/scripts/nav/h_nav.py)说明

- **High-Level Policy**  
  基于强化学习的高层决策模块，输出子目标（Sub-goals）

- **Low-Level Planner**  
  基于几何成本的轨迹优化器，生成可执行路径

---

## 🌍 6. 实验说明

- 在 Gazebo 仿真环境中构建 **200m × 200m** 场景  
- 每个场景包含多个独立导航任务（如 A→B、B→C）  
- 机器人仅依赖：
  - 局部高程图
  - 模糊目标位置  
- 输入与训练分布不同，仅进行简单归一化  

👉 可视化轨迹为**真实执行轨迹**，而非规划拼接结果  

---

## 📌 7. 特点

- ✅ 分层强化学习 + 几何规划  
- ✅ 适用于非结构化地形  
- ✅ 低延迟实时导航  
- ✅ 传感器无关（sensor-agnostic）  
- ✅ 支持大规模环境（200m+）

