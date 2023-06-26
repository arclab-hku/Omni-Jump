from legged_gym.envs import AliengoBaseCfg, AliengoBaseCfgPPO

"""
changes from a1 to go1
- pd gains
- starting height
- target height?
- action scale
"""

class AliengoLbcCfg(AliengoBaseCfg):
    class env(AliengoBaseCfg.env):
        num_observations = 48
        save_im = True
        # camera_type = "d"  # rgb
        num_privileged_obs = None  # 187
        train_type = "standard"  # standard, priv, His, lbc

    class terrain(AliengoBaseCfg.terrain):
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2, 1.0]
        mesh_type = "trimesh"


    class domain_rand(AliengoBaseCfg.domain_rand):
        added_mass_range = [-5.0, 5.0]


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
            feet_step = -1.0
            stumble = -1.0
    # class evals(LeggedRobotCfg.evals):
    #     feet_stumble = True
    #     feet_step = True
    #     crash_freq = True
    #     any_contacts = True

    # class commands(LeggedRobotCfg.commands):
    #     class ranges:
    #         lin_vel_x = [0.0, 1.0]  # min max [m/s]
    #         lin_vel_y = [0.0, 0.0]  # min max [m/s]
    #         ang_vel_yaw = [-0.3, 0.3]  # min max [rad/s]
    #         heading = [0.0, 1.14]



class AliengoLbcCfgPPO(AliengoBaseCfgPPO):
    # class obsSize(AliengoBaseCfgPPO.obsSize):
    #     encoder_hidden_dims = [128, 64, 32]
    #     cnn_out_size = 32
    #     num_dm_encoder_obs = 187

    class runner(AliengoBaseCfgPPO.runner):
        alg = "lbc"
        run_name = ''
        max_iterations = 1000  # number of policy updates
        save_interval = 200  # check for potential saves every this many iterations
        load_run = -1
        experiment_name = "aliengo_lbc"


        resume = False #True for eval, false for train
        resume_path = ""

        teacher_policy =""

    class lbc(AliengoBaseCfgPPO.lbc):
        batch_size = 10
