from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class AliengoBaseCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        # num_envs = 4096
        num_envs = 4096  # was getting a seg fault
        # num_envs = 100  # was getting a seg fault
        num_actions = 12
        num_observations = 45+1 # 45 only consider 3-dim commands #+ 10 # 10 more dims on command 
        # num_proprio_obs = 48
        camera_res = [1280, 720]
        camera_type = "d"  # rgb
        num_privileged_obs = 200 + 5 +12 + 3 + 2 - 4 - 8# 187, 5 means 4 mass and 1 z height and 3 world_ang_vel， 2 for XY position tracking
        train_type = "EST"  # standard, priv, lbc, standard, RMA, EST, Dream, GenHis

        follow_cam = False
        float_cam = False

        measure_obs_heights = False
        num_env_priv_obs = 17  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_histroy_obs = 20
        pass_has_jumped = True

        save_action = False



    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'#'trimesh'#'stone'#'QRC'#
        jump = True
        origin_zero_z = False#True
        vis_type = 'train' #'test'


    class init_state(LeggedRobotCfg.init_state):
        # reference pose for passing QRC while walking
        rel_foot_pos = [[0.239,0.239,-0.301,-0.302], # x
                        [0.154,-0.154,0.154,-0.154], # y
                        [-0.305,-0.305,-0.299,-0.299]]# z  # relative to the COM pos [FL FR RL RR]
        # rel_foot_pos = [[0.228,0.228,-0.253,-0.253], # x
        #                 [0.138,-0.138,0.137,-0.137], # y
        #                 [-0.465,-0.465,-0.465,-0.465]] # z  # relative to the COM pos 
        pos = [0.1, 0.0, 0.36]  # x,y,z [m] # aliengo 0.39
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "FL_hip_joint": 0.06,  # [rad]
            "RL_hip_joint": 0.06,  # [rad]
            "FR_hip_joint": -0.06,  # [rad]
            "RR_hip_joint": -0.06,  # [rad]

            "FL_thigh_joint": 0.9,  # [rad]
            "RL_thigh_joint": 1.1,  # [rad]
            "FR_thigh_joint": 0.9,  # [rad]
            "RR_thigh_joint": 1.1,  # [rad]

            "FL_calf_joint": -1.8,  # [rad]
            "RL_calf_joint": -1.8,  # [rad]
            "FR_calf_joint": -1.8,  # [rad]
            "RR_calf_joint": -1.8,  # [rad]
        }

        default_joint_angles_peak = {  # = target angles [rad] when action = 0.0
            "FL_hip_joint": -0.05,  # [rad]
            "RL_hip_joint": -0.05,  # [rad]
            "FR_hip_joint": 0.05,  # [rad]
            "RR_hip_joint": 0.05,  # [rad]

            "FL_thigh_joint": 1.0,  # [rad]
            "RL_thigh_joint": 0.95,  # [rad]
            "FR_thigh_joint": 1.0,  # [rad]
            "RR_thigh_joint": 0.95,  # [rad]

            "FL_calf_joint": -2.35,  # [rad]
            "RL_calf_joint": -2.55,  # [rad]
            "FR_calf_joint": -2.35,  # [rad]
            "RR_calf_joint": -2.55,  # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "P"#"actuator_net1" #"actuactor_net"#"POSE"#
        # stiffness = {'joint': 20.}  # [N*m/rad]
        stiffness = {"joint": 40.0}  # [N*m/rad]
        # damping = {'joint': 0.5}     # [N*m*s/rad]
        damping = {"joint": 1.2}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_actuator_network = True
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/unitree_aliengo_2rd_f100_it4000_ly2_mlp_dec27_dec28.pt"

        max_delay_steps = 5

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/aliengo_description/urdf/aliengo_old.urdf"
        name = "aliengo"
        foot_name = "foot"#["FL_foot", "FR_foot", "RL_foot", "RR_foot"]#"foot"
        penalize_contacts_on = ["hip", "thigh", "calf"]
        terminate_after_contacts_on = ["base", "trunk", "hip", "thigh"]#["base"]#["base", "trunk", "hip", "thigh"]#["base", "trunk", "hip"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-3.0, 3.0]
        randomize_joint_friction = True
        randomize_friction = True
        friction_range = [0.2, 1.25]

        randomize_center = True
        center_range = [-0.05, 0.05]

        randomize_motor_strength = True
        added_motor_strength = [0.9, 1.1]

        randomize_lag_timesteps = True   # actuator net: True
        added_lag_timesteps = 4

        randomize_Motor_Offset = True  # actuator net: True
        added_Motor_OffsetRange = [-0.02, 0.02]

        randomize_has_jumped = True # Randomize if the robot has jumped or not at the start of the episode
        has_jumped_random_prob = 0.8
        reset_has_jumped = True # Whether to reset the has_jumped to False at a random point of the episode
        manual_has_jumped_reset_time = 0 # Manual time at which to reset has_jumped (in steps)

        class ranges():
            joint_friction_range = [0.0, 0.04]            

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.9
        max_contact_force = 1000.0
        only_positive_rewards = True
        foot_height_target = 0.09
        max_height_reward_sigma = 0.05

        class scales(LeggedRobotCfg.rewards.scales):
            task_pos = 0.#2.5
            task_ori = 0.#2.0
            tracking_lin_vel = 2.5#2.0#2.5#1.2#1.0#1.2#1.0#1.0#0.6#1.5#1.0
            tracking_ang_vel = 1.2#1.0#1.2#0.6#0.5
            tracking_pitch_vel = 0.#4.0
            tracking_yaw = 0.#0.7#0.6
            tracking_pitch = 0.
            lin_vel_z = 0.#1.5#3.5#-1.0 #-0.5 # base_lin_vel_z
            lin_vel_z_world = 0.#0.5#0.5
            lin_disz_world = 0.#1.0
            ang_vel_xy = 0.#-1.0#-0.6 # penalize on yaw
            headup = 0#0.5#0.5
            uf_forces = 0#2.5
            orientation = -1.0#-1.0#-0.6#-0.5#0.2 positive means encourage the robot to stand upright
            upright = 0.#-0.2 # negative means encourage the robot to stand upright
            vel_switch = 0.#1.0
            tracking_pos = 0#1.5
            tracking_feet_pos = 0.#0.8

            torque_limits = -0.01 #0.
            torques = -1e-6 #-1e-7
            dof_acc = -2.5e-7
            base_height = 0#0.2#0.3#0.1
            feet_air_time = 0.#1.0

            stick_to_ground = 0.#0.5

            feet_distance = 0.#0.8#1.0#0.4#0.65
            feet_pos = 0.6#0.4#0.6 # maybe need to be smaller
            early_contact = 0.#1.0
            max_height = 0.#2.5#4.#5.#1.0#10.0#1.5#0.8  # add has_jumped mask and used the simple env
            task_max_height = 1.5#0.6#1.0#15.0#20.0
            base_height_flight = 0.#0.8 #0.8 # Reward for being in the air, only active the first jump
            base_height_stance = 0.#0.8 #0.4 # Reward fo            
            jumping = 30.0#12#20.0#1.0
            has_jumped = 0.#5.0 # remember to delete the has_jumped cutoff in check_termination

            pitch_tracking = 0.#3.0#1.0 #1.0 需要它大，reward他 
            pitch_vel_tracking = 0.#0.3 # penalize the acceleration difference.

            feet_stumble = 0.#-0.5
            collision = -1.0
            action_rate = -0.01#-0.005
            # #### motion
            default_pose = -0.1 # dont be too big!
            tracking_air_angle = 0.#-0.6#-0.6
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

            flfr_gait_diff = -0.04#-0.08#-0.2
            flfr_gait_diff2 = 0.#-0.015
            rlrr_gait_diff = -0.04#-0.08#-0.2
            rlrr_gait_diff2 = 0.#-0.015

            flrl_gait_diff = -0.0#-0.2#-0.
            frrr_gait_diff = -0.0#-0.2#-0.

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
        enable_priv_Zheights_weights = False#True
        enable_priv_feet_height = True
        enable_priv_ang_vel = True
        enable_priv_ZXYheights = True


class AliengoBaseCfgPPO(LeggedRobotCfgPPO):

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        max_iterations = 10000  # number of policy updates
        resume = False
        save_interval = 50  # check for potential saves every this many iterations
        experiment_name = 'aliengo'
        export_policy = False
        export_onnx_policy = False

    class Encoder(LeggedRobotCfgPPO.Encoder):
        priv_mlp_units = [256, 128, 8+12+3+2-4-8]#[258, 128, 3]  # 3 is for the vel estimator. 
        priv_info = False
        priv_info_dim = 200+5+12+3+2-4-8 # +2 for the XY position tracking
        estLen = 3+1+4+12+3+2-4-8 # +2 for the XY position tracking
        proprio_adapt = False
        checkpoint_model = None
        proprio_adapt_out_dim = 11
        HistoryLen = 20
        Hist_info_dim = (45+1) * HistoryLen