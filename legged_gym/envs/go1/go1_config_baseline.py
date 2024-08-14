from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Go1BaseCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):  # comment if use visual
        # num_envs = 4096
        num_envs = 4096  # was getting a seg fault
        # num_envs = 100  # was getting a seg fault
        num_actions = 12
        num_observations = 45
        # num_proprio_obs = 48
        camera_res = [1280, 720]
        camera_type = "d"  # rgb
        num_privileged_obs = 200 + 5 +12# 187, 5 means 4 mass and 1 z height
        train_type = "EST"  # standard, priv, lbc, standard, RMA, EST, Dream, GenHis

        follow_cam = False
        float_cam = False

        measure_obs_heights = False
        num_env_priv_obs = 17  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_histroy_obs = 5


    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'
        jump = True
        origin_zero_z = False#True

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.34]  # x,y,z [m]
        rel_foot_pos = [[0.181,0.181,-0.195,-0.195], # x feet pos which are equal to the x pos of hip joint
                        [0.12675,-0.12675,0.12675,-0.12675], # y feet pos is the same as the initial values
                        [-0.32,-0.32,-0.32,-0.32]] 
        default_joint_angles = { # = target angles [rad] when action = 0.0

            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': -0.0 ,  # [rad]
            'RR_hip_joint': -0.0,   # [rad]

            'FL_thigh_joint': 0.7220,     # [rad]
            'RL_thigh_joint': 0.7220,   # [rad]
            'FR_thigh_joint': 0.7220,     # [rad]
            'RR_thigh_joint': 0.7220,   # [rad]

            'FL_calf_joint': -1.4441,   # [rad]
            'RL_calf_joint': -1.4441,    # [rad]
            'FR_calf_joint': -1.4441,  # [rad]
            'RR_calf_joint': -1.4441,    # [rad]
        }
        # default_joint_angles = {  # = target angles [rad] when action = 0.0
        #     "FL_hip_joint": -0.05,  # [rad]
        #     "RL_hip_joint": -0.05,  # [rad]
        #     "FR_hip_joint": 0.05,  # [rad]
        #     "RR_hip_joint": 0.05,  # [rad]

        #     "FL_thigh_joint": 0.8,  # [rad]
        #     "RL_thigh_joint": 1.0,  # [rad]
        #     "FR_thigh_joint": 0.8,  # [rad]
        #     "RR_thigh_joint": 1.0,  # [rad]

        #     "FL_calf_joint": -1.5,  # [rad]
        #     "RL_calf_joint": -1.5,  # [rad]
        #     "FR_calf_joint": -1.5,  # [rad]
        #     "RR_calf_joint": -1.5,  # [rad]
        # }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "POSE"#"actuator_net"
        # stiffness = {'joint': 20.}  # [N*m/rad]
        stiffness = {"joint": 30.0}  # [N*m/rad]
        # damping = {'joint': 0.5}     # [N*m*s/rad]
        damping = {"joint": 0.8}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_actuator_network = True
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/unitree_go1_join_brick_stairs_it550.pt"

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1_description/urdf/go1.urdf'
        name = "go1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base", "trunk", "hip"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        # flip_visual_attachments = False  # Some .obj meshes must be flipped from y-up to z-up

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-2.0, 2.0]

        randomize_friction = True
        friction_range = [0.2, 1.25]

        randomize_center = True
        center_range = [-0.05, 0.05]

        randomize_motor_strength = True
        added_motor_strength = [0.9, 1.1]

        randomize_lag_timesteps = True   # actuator net: True
        added_lag_timesteps = 6

        randomize_Motor_Offset = True  # actuator net: True
        added_Motor_OffsetRange = [-0.02, 0.02]


    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.5
        max_contact_force = 500.0
        only_positive_rewards = True
        foot_height_target = 0.09

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.0#1.2#1.0#1.0#0.6#1.5#1.0
            tracking_ang_vel = 0.#0.5#0.5#0.35
            tracking_yaw = 0.#5.0#0.7#0.6
            lin_vel_z = 0.#1.5#3.5#-1.0 #-0.5 # base_lin_vel_z
            lin_vel_z_world = 0.8#0.5#2.5#1.8
            lin_disz_world = 0.#1.0
            ang_vel_xy = -0.6#-0.6
            headup = 0#0.5#0.5
            uf_forces = 0#2.5
            orientation = -0.8#0 #0#0.2 positive means encourage the robot to stand upright
            upright = 0.#-0.2 # negative means encourage the robot to stand upright
            vel_switch = 0.#1.0
            tracking_pos = 0#1.5
            tracking_feet_pos = 0#0.8

            torques = 0.#-0.00001
            dof_acc = -2.5e-7
            base_height = 0#0.2#0.3#0.1
            feet_air_time = 0.#1.0

            feet_distance = 0.8#1.0#0.4#0.65
            early_contact = 0.#10.#1.0
            max_height = 0.#5.#1.0#10.0#1.5#0.8  # add has_jumped mask and used the simple env
            task_max_height = 20.0#20.0
            base_height_flight = 1.0 #0.8 # Reward for being in the air, only active the first jump
            base_height_stance = 0.5 #0.4 # Reward fo            
            jumping = 1.0#1.0
            has_jumped = 0.#5.0 # remember to delete the has_jumped cutoff in check_termination

            feet_stumble = 0.#-0.5
            collision = -1.0
            action_rate = -0.01#-0.005
            # #### motion
            default_pose = -0.12 # dont be too big!
            f_hip_motion = 0.#-0.08
            r_hip_motion = 0.#-0.08
            f_thigh_motion = 0.#-0.06
            r_thigh_motion = 0.#-0.06
            f_calf_motion = 0.#-0.05
            r_calf_motion = 0.#-0.05

            # f_hip_motion_height = -0.08
            # r_hip_motion_height = -0.08
            # f_thigh_motion_height = -0.06
            # r_thigh_motion_height = -0.06
            # f_calf_motion_height = -0.06
            # r_calf_motion_height = -0.06

            flfr_gait_diff = -0.2#-0.2
            flfr_gait_diff2 = 0.#-0.015
            rlrr_gait_diff = -0.2#-0.2
            rlrr_gait_diff2 = 0.#-0.015

            flrl_gait_diff = 0.#-0.2#-0.2
            frrr_gait_diff = 0.#-0.2#-0.2

            FL_phase_height = 0.#0.3
            FR_phase_height = 0.#0.3
            RL_phase_height = 0.#0.3
            RR_phase_height = 0.#0.3
            #### smoothness
            # dream_smoothness = -0.001
            # power_joint = -1e-4
            # foot_clearance = -0.001
            # foot_height = -0.01

            ##imitation reward
            imitation_root_pos = 0.#10.0
            imitation_joint_angle = 0.#-0.06
            imitation_feet_pos = 0.#-0.6#-1.0 #-0.8

        # class scales(LeggedRobotCfg.rewards.scales):
        #     tracking_lin_vel = 1.0
        #     tracking_ang_vel = 0.5
        #     lin_vel_z = -2.0
        #     ang_vel_xy = -0.05
        #     orientation = -0.2
        #     torques = -0.00001
        #     dof_acc = -2.5e-7
        #     base_height = -0.0
        #     feet_air_time = 1.0
        #     collision = -1.0
        #     action_rate = -0.01
        #     # #### motion
        #     f_hip_motion = -0.08
        #     r_hip_motion = -0.08
        #     f_thigh_motion = -0.04
        #     r_thigh_motion = -0.04
        #     f_calf_motion = -0.04
        #     r_calf_motion = -0.04

            #### smoothness
            # dream_smoothness = -0.001
            # power_joint = -1e-4
            # foot_clearance = -0.001
            # foot_height = -0.01


    class evals(LeggedRobotCfg.evals):
        feet_stumble = True
        feet_step = True
        crash_freq = True
        any_contacts = True

    class privInfo(LeggedRobotCfg.privInfo):
        enableMotorStrength = True
        enableMeasuredVel = True
        enableMeasuredHeight = True
        enableForce = True
        enable_priv_Zheights_weights = True
        enable_priv_feet_height = True


class Go1BaseCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        max_iterations = 8000  # number of policy updates
        resume = False
        save_interval = 50  # check for potential saves every this many iterations
        experiment_name = 'go1'
        export_policy = False

    class Encoder(LeggedRobotCfgPPO.Encoder):
        priv_mlp_units = [258, 128, 8+12]#[258, 128, 3]  # 3 is for the vel estimator. 
        priv_info = False
        priv_info_dim = 200+5+12
        estLen = 3+1+4+12
        proprio_adapt = False
        checkpoint_model = None
        proprio_adapt_out_dim = 11


        HistoryLen = 5
        Hist_info_dim = 45 * HistoryLen