# This file contains common configuration classes for robots supported by LeggedGym-Ex
# These common configuration classes are used to define default values for various configuration parameters, which can be overridden by task-specific configuration classes.
# Author: Yasen Jia
# Date: 2026-02-14

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym import SIMULATOR

# =============================================================================
# CONFIG MIXINS - Reusable configuration blocks
# =============================================================================
# These mixin classes can be combined to create robot-specific configs.
# Mixins provide common patterns that reduce duplication across robot variants.

def get_simulator_suffix():
    """Helper function to get simulator suffix for run names."""
    if SIMULATOR == "genesis":
        return '_genesis'
    elif SIMULATOR == "isaacgym":
        return '_isaacgym'
    elif SIMULATOR == "isaaclab":
        return '_isaaclab'
    return ''

#----- Common configuration for Unitree Go2 on flat terrain -----#
class Go2FlatCommonCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.4] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': 0.0 ,  # [rad]
            'RR_hip_joint': 0.0,   # [rad]

            'FL_thigh_joint': 0.8, # [rad]
            'RL_thigh_joint': 0.8, # [rad]
            'FR_thigh_joint': 0.8, # [rad]
            'RR_thigh_joint': 0.8, # [rad]

            'FL_calf_joint': -1.5, # [rad]
            'RL_calf_joint': -1.5, # [rad]
            'FR_calf_joint': -1.5, # [rad]
            'RR_calf_joint': -1.5, # [rad]
        }
    
    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        # control_type = 'P'
        stiffness = {'joint': 20.}   # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        action_scale = 0.25 # action scale: target angle = actionScale * action + defaultAngle
        dt = 0.02  # control frequency 50Hz
        decimation = 4 # decimation: Number of control action updates @ sim DT per policy DT
        
    class asset(LeggedRobotCfg.asset):
        # Common
        name = "go2"
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/unitree_robotics/go2/urdf/go2.urdf'
        xml_file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/unitree_robotics/go2/go2.xml'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base", "Head"]
        # full name of the base link
        base_link_name = "base"
        dof_names = [           # align with the real robot
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint"
        ]
        dof_armature = [0.01] * 12 # use a small armature for go2
        # For Genesis
        links_to_keep = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        dof_vel_limits = [30.1, 30.1, 15.7, 
                          30.1, 30.1, 15.7, 
                          30.1, 30.1, 15.7, 
                          30.1, 30.1, 15.7]

#----- Common configuration for Unitree Go2 on rough terrain -----#
class Go2RoughCommonCfg(Go2FlatCommonCfg):
    class terrain( LeggedRobotCfg.terrain ):
        if SIMULATOR in ["isaacgym", "isaaclab"]:
            mesh_type = "trimesh"
        else:
            mesh_type = "heightfield"
        border_size = 20.0 # [m]
        curriculum = True
        # rough terrain only:
        obtain_terrain_info_around_feet = True
        measure_heights = True
        measured_points_x = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4] # 9x9=81
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4]
        terrain_length = 8.0
        terrain_width = 8.0
        platform_size = 4.0
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 10  # number of terrain cols (types)
        # terrain types: [smooth slope, random uniform, stairs up, stairs down, discrete]
        terrain_proportions = [0.2, 0.1, 0.25, 0.25, 0.2]
        
    class init_state( Go2FlatCommonCfg.init_state ):
        pass
    class control( Go2FlatCommonCfg.control ):
        pass
    class asset( Go2FlatCommonCfg.asset ):
        obtain_link_contact_states = True
        contact_state_link_names = ["thigh", "calf", "foot", "base", "hip"]
        penalize_contacts_on = ["thigh", "calf", "base", "Head", "hip"]
        terminate_after_contacts_on = []
    
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        foot_clearance_target = 0.09 # desired foot clearance above ground [m]
        foot_height_offset = 0.022   # height of the foot coordinate origin above ground [m]
        foot_clearance_tracking_sigma = 0.01
        only_positive_rewards = True
        class scales( LeggedRobotCfg.rewards.scales ):
            # limitation
            dof_pos_limits = -2.0
            collision = -1.0
            # command tracking
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            # smooth
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            dof_power = -2.e-4
            dof_acc = -2.e-7
            action_rate = -0.01
            action_smoothness = -0.01
            # gait
            feet_air_time = 1.0
            foot_clearance = 0.2
            feet_contact_stand_still = 0.5
            dof_close_to_default_stand_still = -0.5
