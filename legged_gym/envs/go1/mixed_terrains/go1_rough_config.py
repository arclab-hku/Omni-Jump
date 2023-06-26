

from legged_gym.envs import Go1BaseCfg, Go1BaseCfgPPO


class Go1RoughCfg(Go1BaseCfg):
    class env(Go1BaseCfg.env):
        num_observations = 49

    class terrain(Go1BaseCfg.terrain):
        mesh_type = 'trimesh'
        measure_heights = True
        measure_obs_heights = False



    class rewards(Go1BaseCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 1 # TODO: 0.2, 0.3, 0.4, 0.5

        class scales(Go1BaseCfg.rewards.scales):
            lin_vel_z = -4.0
            ang_vel_xy = -0.01
            torques = -0.00025
            # feet_air_time = 2.0
            dof_pos_limits = -10.0
            base_height = -0.1  # I added this reward to get the robot a high higher up
            hip_motion = -0.05
            thigh_motion = -0.05
            calf_motion = -0.05
            orientation = -5.0

    class RMA(Go1BaseCfg.RMA):
        class adaptor(Go1BaseCfg.RMA.adaptor):
            propHistoryLen = 30
            privInfoDim = 204
        class privInfo(Go1BaseCfg.RMA.privInfo):
                enableMass = True
                enableCOM = True
                enableFriction = True
                enableMotorStrength = True
                enableMeasuredHeight = True
                enableMeasuredVel = False
                enableOnlyVel = False


class Go1RoughCfgPPO(Go1BaseCfgPPO):

    class algorithm(Go1BaseCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(Go1BaseCfgPPO.runner):
        run_name = ''
        max_iterations = 1000  # number of policy updates
        save_interval = 250  # check for potential saves every this many iterations
        load_run = -1 # -1 = last run
        experiment_name = 'go1_rough'

    class RMA(Go1BaseCfgPPO.RMA):
        export_policy = False
        priv_mlp_units = [256, 128, 32]
        priv_info = False
        priv_info_dim = 204
        proprio_adapt = False
        checkpoint_model = None