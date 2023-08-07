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
        num_privileged_obs = 200  # 187
        train_type = "EST"  # standard, priv, lbc, standard, RMA, EST, Dream

        follow_cam = False
        float_cam = False

        measure_obs_heights = False
        num_env_priv_obs = 17  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.38]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "FL_hip_joint": 0.1,  # [rad]
            "RL_hip_joint": 0.1,  # [rad]
            "FR_hip_joint": -0.1,  # [rad]
            "RR_hip_joint": -0.1,  # [rad]

            "FL_thigh_joint": 0.8,  # [rad]
            "RL_thigh_joint": 1.0,  # [rad]
            "FR_thigh_joint": 0.8,  # [rad]
            "RR_thigh_joint": 1.0,  # [rad]

            "FL_calf_joint": -1.5,  # [rad]
            "RL_calf_joint": -1.5,  # [rad]
            "FR_calf_joint": -1.5,  # [rad]
            "RR_calf_joint": -1.5,  # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "P"
        # stiffness = {'joint': 20.}  # [N*m/rad]
        stiffness = {"joint": 40.0}  # [N*m/rad]
        # damping = {'joint': 0.5}     # [N*m*s/rad]
        damping = {"joint": 2.0}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go1.urdf'
        name = "go1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on =  ["base", "trunk", "hip"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-5.0, 5.0]
        randomize_friction = True
        friction_range = [0.2, 1.25]

        randomize_center = True
        center_range = [-0.05, 0.05]

        randomize_motor_strength = True
        added_motor_strength = [0.9, 1.1]

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.5
        max_contact_force = 500.0
        only_positive_rewards = True
        foot_height_target = 0.09

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.2
            torques = -0.00001
            dof_acc = -2.5e-7
            base_height = -0.0
            feet_air_time = 1.0
            collision = -1.0
            action_rate = -0.01
            # #### motion
            # f_hip_motion = -0.02
            # r_hip_motion = -0.02
            # f_thigh_motion = -0.02
            # r_thigh_motion = -0.02
            # f_calf_motion = -0.02
            # r_calf_motion = -0.02

            #### smoothness
            # dream_smoothness = -0.001
            # power_joint = -1e-4
            # foot_clearance = -0.01
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


class Go1BaseCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        max_iterations = 2000  # number of policy updates
        resume = False
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = 'go1'
        export_policy = False

    class Encoder(LeggedRobotCfgPPO.Encoder):
        priv_mlp_units = [258, 128, 11]
        priv_info = False
        priv_info_dim = 200
        velLen = 3
        proprio_adapt = False
        checkpoint_model = None
        proprio_adapt_out_dim = 11