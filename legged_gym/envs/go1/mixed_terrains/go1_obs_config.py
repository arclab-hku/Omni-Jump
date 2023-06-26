

from legged_gym.envs import Go1BaseCfg, Go1BaseCfgPPO

class Go1ObsCfg(Go1BaseCfg):
    class env(Go1BaseCfg.env):
        num_observations = 48

    class terrain(Go1BaseCfg.terrain):
        mesh_type = 'trimesh_obs'
        measure_heights = True
        measure_obs_heights = False
        contact = False
        creat_map = False
        unlin_vel = False

        terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 1]



    class rewards(Go1BaseCfg.rewards):
        soft_dof_pos_limit = 0.9
        max_contact_force = 500.  # forces above this value are penalized
        base_height_target = 1 # TODO: 0.2, 0.3, 0.4, 0.5

        class scales(Go1BaseCfg.rewards.scales):
            lin_vel_z = -4.0
            ang_vel_xy = -0.01
            torques = -0.00025
            feet_air_time = 2.0
            dof_pos_limits = -10.0
            base_height = -0.1 # I added this reward to get the robot a high higher up
            feet_step = -1.0
            stumble = -1.0

class Go1ObsCfgPPO(Go1BaseCfgPPO):

    class algorithm(Go1BaseCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(Go1BaseCfgPPO.runner):
        run_name = ''
        max_iterations = 1000  # number of policy updates
        save_interval = 200  # check for potential saves every this many iterations
        load_run = 'base_48' # -1 = last run
        resume = True
        # resume_path = '/home/dong/hku/legged_gym/logs/rough_aliengo/base_48/model_400.pt'  # updated from load_run and chkpt
        experiment_name = 'go1_obs'

