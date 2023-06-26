

from legged_gym.envs import AliengoBaseCfg, AliengoBaseCfgPPO


class AliengoHisCfg(AliengoBaseCfg):
    class env(AliengoBaseCfg.env):
        num_observations = 235
        train_type = "lin_vel"  # standard, priv, lbc, encoder
        num_proprio_obs = 45
    class terrain(AliengoBaseCfg.terrain):
        mesh_type = 'trimesh'
        measure_heights = True
        measure_obs_heights = True
        Encoder = False
        contact = False
        creat_map = False
        unlin_vel = False


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
    class RMA:
        class hora:
          propHistoryLen = 30

class AliengoHisCfgPPO(AliengoBaseCfgPPO):
    class obsSize:
        encoder_hidden_dims = [128, 64, 32]
        cnn_out_size = 32
        num_dm_encoder_obs = 190
    class runner(AliengoBaseCfgPPO.runner):
        run_name = ''
        alg = "his"
        max_iterations = 1000  # number of policy updates
        save_interval = 500  # check for potential saves every this many iterations
        load_run = -1 # -1 = last run
        resume = True
        experiment_name = 'aliengo_his'
        teacher_policy ="/home/dong/hku/legged_gym/logs/aliengo_encoder/weight/model_1000.pt"

    class His(AliengoBaseCfgPPO.His):
        batch_size = 10

