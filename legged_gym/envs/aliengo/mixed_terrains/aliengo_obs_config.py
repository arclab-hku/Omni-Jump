

from legged_gym.envs import AliengoBaseCfg, AliengoBaseCfgPPO


class AliengoObsCfg(AliengoBaseCfg):
    class env(AliengoBaseCfg.env):
        num_observations = 235
        train_type = "lin_vel"  # standard, priv, lbc, encoder
        num_proprio_obs = 45
    class terrain(AliengoBaseCfg.terrain):
        mesh_type = 'trimesh_obs'
        measure_heights = True
        measure_obs_heights = True
        contact = False
        creat_map = False
        unlin_vel = False

        terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 1]

    class domain_rand(AliengoBaseCfg.domain_rand):
        added_mass_range = [-1.0, 1.0]


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
            # feet_step = -1.0
            stumble = -1.0
    class commands(AliengoBaseCfg.commands):
        class ranges:
            lin_vel_x = [0.0, 1.0]  # min max [m/s]
            lin_vel_y = [0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
            heading = [-3.14, 3.14]
class AliengoObsCfgPPO(AliengoBaseCfgPPO):
    class obsSize:
        encoder_hidden_dims = [128, 64, 32]
    class algorithm(AliengoBaseCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(AliengoBaseCfgPPO.runner):
        run_name = ''
        alg = "encoder"
        max_iterations = 1000  # number of policy updates
        save_interval = 200  # check for potential saves every this many iterations
        load_run = 'Apr12_19-23-55_' # -1 = last run
        resume = True
        # resume_path = ''  # updated from load_run and chkpt
        experiment_name = 'aliengo_obs'

