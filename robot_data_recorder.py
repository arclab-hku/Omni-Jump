#!/usr/bin/env python3
"""
robot_data_recorder.py

该脚本提供了 RobotDataRecorder 类，用于在 Isaac Gym 环境下记录四足机器人的数据，
数据格式参考 jump00.txt 文件要求（每帧 61 维数据），数据的各部分含义如下：

【数据格式（每帧 61 维度）】
    curr_pose (31 维):
        - root_pos                [0:3]   机器人根部在世界坐标系中的位置 (3 维)
        - root_rot                [3:7]   机器人根部旋转（四元数，格式 [x, y, z, w]，4 维)
        - joint_pose              [7:19]  四足机器人 12 个关节角度，顺序：FL, FR, RL, RR（共 12 维）
        - tar_toe_pos_local       [19:31] 4 个足端在体坐标系中的位置，每个 3 维（共 12 维）

    del_linear_vel (3 维):
        - 机器人根部线速度（体坐标系）， [31:34]

    del_angular_vel (3 维):
        - 机器人根部角速度（体坐标系）， [34:37]

    joint_velocity (12 维):
        - 机器人关节角速度（与 joint_pose 顺序一致）， [37:49]

    toe_velocity (12 维):
        - 四个足端速度（体坐标系），每个 3 维，总共 12 维 [49:61]

【附加参数】
    - LoopMode: "Wrap"
    - EnableCycleOffsetPosition: True
    - EnableCycleOffsetRotation: True
    - MotionWeight: 2.5
    - FrameDuration: env.dt

采集流程：
    每个仿真步（帧）调用 record_frame() 方法采集 61 维数据，并存储到 frames 列表中，
    最后调用 save() 方法将所有数据保存为 txt 文件，文件格式参考 jump00.txt 文件标准。

注意：
    请确保 Isaac Gym 环境对象 env 的以下属性存在且已更新：
        env.root_states, env.dof_pos, env.dof_vel, env.feet_pos, env.feet_vel,
        env.base_lin_vel, env.base_ang_vel
    若数据为 torch.Tensor，请调用 .cpu().numpy() 转换为 numpy 数组。

该脚本文件可供 test.py 导入调用，也可直接运行进行简单测试。
"""

import json
import numpy as np
from scipy.spatial.transform import Rotation as R

class RobotDataRecorder:
    def __init__(self, frame_duration, 
                 loop_mode="Wrap",
                 enable_cycle_offset_position=True,
                 enable_cycle_offset_rotation=True,
                 motion_weight=2.5):
        """
        初始化数据记录器

        参数:
            frame_duration: 帧间隔（秒），通常取自 env.dt
            loop_mode: 循环模式，默认为 "Wrap"
            enable_cycle_offset_position: 是否开启周期性位置偏移（True/False）
            enable_cycle_offset_rotation: 是否开启周期性旋转偏移（True/False）
            motion_weight: 动作权重，默认 2.5
        """
        self.frame_duration = frame_duration
        self.loop_mode = loop_mode
        self.enable_cycle_offset_position = enable_cycle_offset_position
        self.enable_cycle_offset_rotation = enable_cycle_offset_rotation
        self.motion_weight = motion_weight

        # 存储每一帧数据，每帧为一个含有 61 个数字的列表
        self.frames = []

    def record_frame(self, env, robot_index=0):
        """
        从当前 env 仿真状态记录一帧数据

        参数:
            env: Isaac Gym 仿真环境对象，必须包含对应属性
            robot_index: 要采集数据的机器人索引（默认 0）
        """
        # 若数据为 torch.Tensor，则转换为 numpy 数组
        if hasattr(env.root_states, "cpu"):
            root_states = env.root_states.cpu().numpy()
        else:
            root_states = env.root_states
        if hasattr(env.dof_pos, "cpu"):
            dof_pos = env.dof_pos.cpu().numpy()
        else:
            dof_pos = env.dof_pos
        if hasattr(env.dof_vel, "cpu"):
            dof_vel = env.dof_vel.cpu().numpy()
        else:
            dof_vel = env.dof_vel
        if hasattr(env.feet_pos, "cpu"):
            feet_pos = env.feet_pos.cpu().numpy()
        else:
            feet_pos = env.feet_pos
        if hasattr(env.foot_vel, "cpu"):
            feet_vel = env.foot_vel.cpu().numpy()
        else:
            feet_vel = env.foot_vel
        if hasattr(env.base_lin_vel, "cpu"):
            base_lin_vel = env.base_lin_vel.cpu().numpy()
        else:
            base_lin_vel = env.base_lin_vel
        if hasattr(env.base_ang_vel, "cpu"):
            base_ang_vel = env.base_ang_vel.cpu().numpy()
        else:
            base_ang_vel = env.base_ang_vel

        # 1. 机器人根部位置与旋转（世界坐标系）
        robot_pos = root_states[robot_index, 0:3]  # 3 维位置
        robot_quat = root_states[robot_index, 3:7]   # 4 维四元数 [x, y, z, w]

        # 2. 机器人关节角度（12 维），假设取 dof_pos 的前 12 个
        joint_pose = dof_pos[robot_index, :12]

        # 3. 机器人足端位置（世界坐标）转换到体坐标系，得到目标足端局部位置（12 维）
        r_inv = R.from_quat(robot_quat).inv()
        toe_local_all = []
        for i in range(4):
            foot_world = feet_pos[robot_index, i, :]  # 3 维
            toe_local = r_inv.apply(foot_world - robot_pos)
            toe_local_all.extend(toe_local.tolist())
        toe_local_all = np.array(toe_local_all)  # 12 维

        # 4. 当前位姿： curr_pose = [root_pos(3), root_rot(4), joint_pose(12), toe_local(12)] 共 31 维
        curr_pose = np.concatenate([robot_pos, robot_quat, joint_pose, toe_local_all])

        # 5. 根部线速度（世界）转换到体坐标： del_linear_vel (3 维)
        world_lin_vel = base_lin_vel[robot_index]
        del_linear_vel = r_inv.apply(world_lin_vel)

        # 6. 根部角速度（世界）转换到体坐标： del_angular_vel (3 维)
        world_ang_vel = base_ang_vel[robot_index]
        del_angular_vel = r_inv.apply(world_ang_vel)

        # 7. 机器人关节角速度（12 维），假设取 dof_vel 前 12 个
        joint_velocity = dof_vel[robot_index, :12]

        # 8. 足端速度（世界转换到体坐标），每个足 3 维，4 个足共 12 维
        toe_velocity_all = []
        foot_vel_world = np.zeros((4, 3))
        for i in range(4):
            foot_vel_world[i, 0] = feet_vel[robot_index, 3*i] 
            foot_vel_world[i, 1] = feet_vel[robot_index, 3*i+1]
            foot_vel_world[i, 2] = feet_vel[robot_index, 3*i+2] 
            toe_vel_local = r_inv.apply(foot_vel_world[i,:])
            toe_velocity_all.extend(toe_vel_local.tolist())
        
        #toe_velocity_all = np.array(toe_velocity_all)

        # 拼接一帧数据，共 61 维：
        # [curr_pose (31), del_linear_vel (3), del_angular_vel (3),
        #  joint_velocity (12), toe_velocity_all (12)]
        frame = np.concatenate([
            curr_pose,           # 31 维
            del_linear_vel,      # 3 维
            del_angular_vel,     # 3 维
            joint_velocity,      # 12 维
            toe_velocity_all     # 12 维
        ])
        if frame.shape[0] != 61:
            raise ValueError(f"记录数据维度错误，期望 61 维，获得 {frame.shape[0]} 维")

        # 添加本帧数据（转换为列表形式）
        self.frames.append(frame.tolist())

    def save(self, file_path):
        """
        将所有记录帧保存到指定的 txt 文件中，文件格式参考 jump00.txt 文件要求，
        其中文件首部包含参数信息，随后每一行对应一帧 61 维数据，数值之间以逗号分隔。

        文件输出示例如下：

            LoopMode: Wrap
            FrameDuration: 0.01677foot_vel
            ...
        """
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            for frame in self.frames:
                # 将每一帧的数值转换为字符串，并以逗号分隔
                line = ",".join(str(x) for x in frame)
                f.write(line + "\n")
        print(f"机器人数据已保存为 txt 文件: {file_path}")

# 如果直接运行该文件，则使用简单示例进行测试
if __name__ == "__main__":
    # 构造虚拟环境以测试数据记录功能
    class DummyEnv:
        def __init__(self):
            self.dt = 0.01677
            # 模拟一个机器人
            self.root_states = np.array([[0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0]])
            self.dof_pos = np.zeros((1, 12))
            self.dof_vel = np.zeros((1, 12))
            self.feet_pos = np.array([[
                [0.15, 0.10, 0.0],
                [0.15, -0.10, 0.0],
                [-0.15, 0.10, 0.0],
                [-0.15, -0.10, 0.0]
            ]])
            self.feet_vel = np.zeros((1, 4, 3))
            self.base_lin_vel = np.zeros((1, 3))
            self.base_ang_vel = np.zeros((1, 3))
    
    dummy_env = DummyEnv()
    recorder = RobotDataRecorder(frame_duration=dummy_env.dt)
    for _ in range(5):
        recorder.record_frame(dummy_env, robot_index=0)
    # 保存为 txt 文件
    recorder.save("dummy_robot_data.txt")