

from legged_gym.envs.aliengo.aliengo_config_baseline import AliengoBaseCfg, AliengoBaseCfgPPO


class AliengoRoughCfg(AliengoBaseCfg):
    class env(AliengoBaseCfg.env):
        num_observations = 49
        # train_type = "lin_vel"  # standard, priv, lbc, encoder

    class terrain(AliengoBaseCfg.terrain):
        mesh_type = 'trimesh'
        measure_heights = True
        measure_obs_heights = False
        # terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]

    # class commands(AliengoBaseCfg.commands):
    #     curriculum = False
    #     max_curriculum = 1.
    #     num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
    #     resampling_time = 10.  # time before command are changed[s]
    #     heading_command = True  # if true: compute ang vel command from heading error
    #
    #     class ranges(AliengoBaseCfg.commands.ranges):
    #             lin_vel_x = [0.0, 1.0]  # min max [m/s]
    #             lin_vel_y = [0.0, 0.0]  # min max [m/s]
    #             ang_vel_yaw = [-1, 1]  # min max [rad/s]
    #             heading = [-3.14, 3.14]


    class rewards(AliengoBaseCfg.rewards):
        soft_dof_pos_limit = 0.9
        max_contact_force = 500.  # forces above this value are penalized
        base_height_target = 1 # TODO: 0.2, 0.3, 0.4, 0.5
        terminate_base_height = False
        class scales(AliengoBaseCfg.rewards.scales):
            lin_vel_z = -4.0
            ang_vel_xy = -0.01
            torques = -0.00025
            feet_air_time = 2.0
            dof_pos_limits = -10.0
            base_height = -0.5  # I added this reward to get the robot a high higher up
            # orientation = -5.0
            # hip_motion = -0.05
            # thigh_motion = -0.05
            # calf_motion = -0.05

    class RMA(AliengoBaseCfg.RMA):
        class adaptor(AliengoBaseCfg.RMA.adaptor):
            propHistoryLen = 30
            privInfoDim = 204



        class privInfo(AliengoBaseCfg.RMA.privInfo):
            enableMass = True
            enableCOM = True
            enableFriction = True
            enableMotorStrength = True
            enableMeasuredHeight = True


class AliengoRoughCfgPPO(AliengoBaseCfgPPO):

    class algorithm(AliengoBaseCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(AliengoBaseCfgPPO.runner):
        run_name = ''
        max_iterations = 2000  # number of policy updates
        save_interval = 250  # check for potential saves every this many iterations
        load_run = -1 # -1 = last run
        resume = False
        experiment_name = 'aliengo_rough'

    class RMA(AliengoBaseCfgPPO.RMA):
        export_policy = False
        priv_mlp_units = [256, 128, 32]
        priv_info = False
        priv_info_dim = 204
        proprio_adapt = False
        checkpoint_model = None