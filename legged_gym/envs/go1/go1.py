from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import nn
# from torch.tensor import Tensor
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from .go1_config_baseline import Go1BaseCfg

LEG_NUM = 4
LEG_DOF = 3
LEN_HIST = 5
MODEL_IN_SIZE = 2 * LEG_DOF * LEN_HIST

class Go1CameraMixin:
    def __init__(self, *args, **kwargs):
        self.follow_cam = None
        self.floating_cam = None
        super().__init__(*args, **kwargs)

    def init_aux_cameras(self, follow_cam=False, float_cam=False):
        if follow_cam:
            self.follow_cam, follow_trans = self.make_handle_trans(
                1920, 1080, 0, (1.0, -1.0, 0.0), (0.0, 0.0, 3 * 3.14 / 4)
            )
            body_handle = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], "base"
            )
            self.gym.attach_camera_to_body(
                self.follow_cam,  # camera_handle,
                self.envs[0],
                body_handle,
                follow_trans,
                gymapi.FOLLOW_POSITION,
            )

        if float_cam:
            self.floating_cam, _ = self.make_handle_trans(
                # 1280, 720, 0, (0, 0, 0), (0, 0, 0), hfov=50
                1920, 1080, 0, (0, 0, 0), (0, 0, 0)
            )
            camera_position = gymapi.Vec3(5, 5, 5)
            camera_target = gymapi.Vec3(0, 0, 0)
            self.gym.set_camera_location(
                self.floating_cam, self.envs[0], camera_position, camera_target
            )

    def make_handle_trans(self, width, height, env_idx, trans, rot, hfov=None):
        camera_props = gymapi.CameraProperties()
        camera_props.width = width
        camera_props.height = height
        camera_props.enable_tensors = True
        if hfov is not None:
            camera_props.horizontal_fov = hfov
        camera_handle = self.gym.create_camera_sensor(self.envs[env_idx], camera_props)
        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(*trans)
        local_transform.r = gymapi.Quat.from_euler_zyx(*rot)
        return camera_handle, local_transform


class Go1(Go1CameraMixin, LeggedRobot):
    cfg: Go1BaseCfg
    def __init__(
        self, cfg, sim_params, physics_engine, sim_device, headless, record=False
    ):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.camera_handles = []
        print("GO1 INIT")
        self.init_aux_cameras(cfg.env.follow_cam, cfg.env.float_cam)


    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # Additionaly empty actuator network hidden states
        self.sea_hidden_state_per_env[:, env_ids] = 0.0
        self.sea_cell_state_per_env[:, env_ids] = 0.0
    def _init_buffers(self):
        super()._init_buffers()
        # Additionally initialize actuator network hidden state tensors
        self.sea_input = torch.zeros(
            self.num_envs * self.num_actions,
            1,
            2,
            device=self.device,
            requires_grad=False,
        )
        self.sea_hidden_state = torch.zeros(
            2,
            self.num_envs * self.num_actions,
            8,
            device=self.device,
            requires_grad=False,
        )
        self.sea_cell_state = torch.zeros(
            2,
            self.num_envs * self.num_actions,
            8,
            device=self.device,
            requires_grad=False,
        )
        self.sea_hidden_state_per_env = self.sea_hidden_state.view(
            2, self.num_envs, self.num_actions, 8
        )
        self.sea_cell_state_per_env = self.sea_cell_state.view(
            2, self.num_envs, self.num_actions, 8
        )

    def _compute_torques(self, actions):
        return super()._compute_torques(actions)







