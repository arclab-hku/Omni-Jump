from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class AliengoBaseCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_observations = 45
        num_vel_obs = 200
        train_type = "Dream"  # standard, RMA, EST, Dream

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        # "trimesh_no_stair"
        # plane
        # 'trimesh'
        measure_heights = True
        measure_obs_heights = False
        vel = False
        contact = False
        # terrain_length = 10.
        # terrain_width = 10.
        #
        # max_init_terrain_level = 8  # starting curriculum state

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 1]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,  # [rad]
            'RL_hip_joint': 0.0,  # [rad]
            'FR_hip_joint': 0.0,  # [rad]
            'RR_hip_joint': 0.0,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.0,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1.0,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip_joint': 30, 'thigh_joint': 50., 'calf_joint': 50.}  # [N*m/rad]
        # stiffness = {"joint": 40.0}
        damping = {'hip_joint': 2., 'thigh_joint': 2., 'calf_joint': 2.}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_actuator_network = False

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/aliengo_ye/urdf/aliengo.urdf'
        name = "aliengo"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-1.0, 2.0]
        randomize_limb_mass = True
        added_limb_percentage = [-0.2, 0.2]

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.5
        foot_height_target = 0.09
        terminate_base_height = False

        class scales(LeggedRobotCfg.rewards.scales):
            lin_vel_z = -2.0
            ang_vel_xy = -0.01
            orientation = -0.2
            dof_acc = -2.5e-7

            action_rate = -0.01 / 3

            collision = -1
            feet_air_time_base = 1.0
            # dof_pos_limits = -10.0

            # torques = -1e-4
            # base_height = -0.5  # I added this reward to get the robot a high higher up

            # ### dream
            # dof_vel = -1e-5

            # orientation_base = -0.02
            dream_smoothness = -0.001
            power_joint = -1e-4
            foot_clearance = -0.01
            foot_height = -0.01
            #### motion
            # motion_base = -0.1

            # motion = -0.1
            f_hip_motion = -0.1
            r_hip_motion = -0.1
            f_thigh_motion = -0.05
            r_thigh_motion = -0.05
            f_calf_motion = -0.05
            r_calf_motion = -0.05

    class RMA(LeggedRobotCfg.RMA):
        class adaptor(LeggedRobotCfg.RMA.adaptor):
            propHistoryLen = 6
            privInfoDim = 17

        class randomization(LeggedRobotCfg.RMA.randomization):
            # Randomization Property
            randomizeMass = True
            randomizeMassLower = -1
            randomizeMassUpper = 2
            randomizeCOM = True
            randomizeCOMLower = -0.05
            randomizeCOMUpper = 0.05
            randomizeFriction = True
            randomizeFrictionLower = 0.2
            randomizeFrictionUpper = 1.25
            randomizeMotorStrength = True
            randomizeMotorStrengthLower = 0.9
            randomizeMotorStrengthUpper = 1.1

        class privInfo(LeggedRobotCfg.RMA.privInfo):
            enableMass = True
            enableCOM = True
            enableFriction = True
            enableMotorStrength = True
            enableMeasuredHeight = True
            enableMeasuredVel = True
            enableForce = True


class AliengoBaseCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        max_iterations = 1000  # number of policy updates
        resume = False
        save_interval = 200  # check for potential saves every this many iterations
        experiment_name = 'aliengo'
    class RMA(LeggedRobotCfgPPO.RMA):
        export_policy = False
        priv_mlp_units = [128, 64, 19]
        decoder_mlp_units = [64, 128, 48]
        priv_info = False
        priv_info_dim = 187+2
        proprio_adapt = False
        checkpoint_model = None
        proprio_adapt_out_dim = 19
        HistoryLen = 6
        velLen = 3
        Hist_info_dim = 45 * HistoryLen