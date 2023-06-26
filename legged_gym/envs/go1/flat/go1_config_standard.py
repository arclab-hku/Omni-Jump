from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Go1BaseCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):  # comment if use visual
        num_observations = 48
        num_vel_obs = 198
        train_type = "standard"  # standard, priv, lbc

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'
        # 'trimesh'
        measure_heights = True
        measure_obs_heights = False
        vel = False
        contact = False

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
        pos = [0.0, 0.0, 0.38]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,  # [rad]
            'RL_hip_joint': 0.1,  # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

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
        damping = {'hip_joint': 2., 'thigh_joint': 2., 'calf_joint': 2.}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_actuator_network = False
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/go1_net.pt"

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf'
        name = "go1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base", "trunk", "hip"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        randomize_limb_mass = True
        added_limb_percentage = [-0.2, 0.2]

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        max_contact_force = 500
        base_height_target = 1  # TODO: 0.2, 0.3, 0.4, 0.5
        terminate_base_height = False

        class scales(LeggedRobotCfg.rewards.scales):
            lin_vel_z = -4.0
            ang_vel_xy = -0.01
            torques = -0.00025
            feet_air_time = 2.0
            dof_pos_limits = -10.0
            base_height = -0.5  # I added this reward to get the robot a high higher up
    class RMA(LeggedRobotCfg.RMA):
            class adaptor(LeggedRobotCfg.RMA.adaptor):
                propHistoryLen = 30
                privInfoDim = 21
            class privInfo(LeggedRobotCfg.RMA.privInfo):
                enableMass = False
                enableCOM = False
                enableFriction = False
                enableMotorStrength = False
                enableMeasuredHeight = True
                enableMeasuredVel = True
                enableOnlyVel = False


class Go1BaseCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        max_iterations = 1000  # number of policy updates
        resume = False
        save_interval = 200  # check for potential saves every this many iterations

        experiment_name = 'go1'
