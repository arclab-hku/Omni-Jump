from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class AliengoBaseCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        # num_envs = 4096
        num_envs = 4096#4096  # was getting a seg fault
        # num_envs = 100  # was getting a seg fault
        num_actions = 12
        num_observations = 45
        # num_proprio_obs = 48
        camera_res = [1280, 720]
        camera_type = "d"  # rgb
        num_privileged_obs = 200#+264  # 187 +264 = 451
        train_type = "EST"  # standard, priv, lbc, standard, RMA, EST, Dream, GenHis

        follow_cam = False
        float_cam = False

        measure_obs_heights = False
        num_env_priv_obs = 17  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_histroy_obs = 5
        pass_has_jumped = True



    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'#'stone'#'QRC'#
        jump = True
        origin_zero_z = False#True


    class init_state(LeggedRobotCfg.init_state):
        # reference pose for passing QRC while walking
        rel_foot_pos = [[0.239,0.239,-0.302,-0.302], # x
                        [0.118,-0.118,0.118,-0.118], # y
                        [-0.314,-0.314,-0.308,-0.308]]# z  # relative to the COM pos [FL FR RL RR]
        # rel_foot_pos = [[0.228,0.228,-0.253,-0.253], # x
        #                 [0.138,-0.138,0.137,-0.137], # y
        #                 [-0.465,-0.465,-0.465,-0.465]] # z  # relative to the COM pos 
        pos = [0.0, 0.0, 0.39]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "FL_hip_joint": -0.05,  # [rad]
            "RL_hip_joint": -0.05,  # [rad]
            "FR_hip_joint": 0.05,  # [rad]
            "RR_hip_joint": 0.05,  # [rad]

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
        control_type = "POSE"#"actuator_net"#"actuator_net"#"actuator_net"#"POSE"#"actuator_net"#
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

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/aliengo_description/urdf/aliengo.urdf"
        name = "aliengo"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]#["base", "trunk", "hip"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-3.0, 3.0]

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


    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.9
        max_contact_force = 1000.0
        only_positive_rewards = True
        foot_height_target = 0.09
        max_height_reward_sigma = 0.05

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 0.#1.0#0.6#1.5#1.0
            tracking_ang_vel = 0.5#0.5#0.35
            lin_vel_z = 0.#1.5#3.5#-1.0 #-0.5 # base_lin_vel_z
            lin_vel_z_world = 0.5#2.5#1.8
            ang_vel_xy = 0.#-0.6
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
            max_height = 0.#10.0#1.5#0.8  # add has_jumped mask and used the simple env
            task_max_height = 20.0
            base_height_flight = 0.8 # Reward for being in the air, only active the first jump
            base_height_stance = 0.4 # Reward fo            
            jumping = 1.0#1.0

            feet_stumble = 0.#-0.5
            collision = -1.0
            action_rate = -0.01#-0.005
            # #### motion
            default_pose = -0.16
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

            flfr_gait_diff = -0.2
            flfr_gait_diff2 = 0.#-0.015
            rlrr_gait_diff = -0.2
            rlrr_gait_diff2 = 0.#-0.015

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
        priv_mlp_units = [258, 128, 3]
        priv_info = False
        priv_info_dim = 200#+264
        velLen = 3
        proprio_adapt = False
        checkpoint_model = None
        proprio_adapt_out_dim = 11


        HistoryLen = 5
        Hist_info_dim = 45 * HistoryLen