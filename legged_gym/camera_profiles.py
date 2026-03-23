"""Shared camera profiles for Warp pointcloud collection and NSR runtime."""

ANYMAL_LEGACY_4CAM_PROFILE = {
    "num_sensors": 4,
    "resolution": (80, 60),
    "horizontal_fov_deg": 75.0,
    "near_clip": 0.1,
    "far_clip": 50.0,
    "near_plane": 0.1,
    "far_plane": 50.0,
    "decimation": 1,
    "calculate_depth": False,
    "return_pointcloud": True,
    "pointcloud_in_world_frame": True,
    "euler": [
        (0.0, 2.09, 0.0),
        (0.0, 2.09, 3.1416),
        (0.0, 2.09, 1.5708),
        (0.0, 2.09, -1.5708),
    ],
    "pos": [
        (0.3, 0.0, 0.1),
        (-0.3, 0.0, 0.1),
        (0.0, 0.3, 0.1),
        (0.0, -0.3, 0.1),
    ],
}

# Intel RealSense D435i depth stream is approximately 86-87 deg x 57-58 deg.
# Using a 16:9 render target keeps the implied vertical FOV close to the real sensor.
GO2_D435I_4CAM_PROFILE = {
    "num_sensors": 4,
    "resolution": (96, 54),
    "horizontal_fov_deg": 87.0,
    "near_clip": 0.28,
    "far_clip": 5.0,
    "near_plane": 0.28,
    "far_plane": 5.0,
    "decimation": 1,
    "calculate_depth": False,
    "return_pointcloud": True,
    "pointcloud_in_world_frame": True,
    "euler": [
        (0.0, 2.09, 0.0),
        (0.0, 2.09, 3.1416),
        (0.0, 2.09, 1.5708),
        (0.0, 2.09, -1.5708),
    ],
    "pos": [
        (0.23, 0.0, 0.10),
        (-0.23, 0.0, 0.10),
        (0.0, 0.17, 0.10),
        (0.0, -0.17, 0.10),
    ],
}


def get_pointcloud_camera_profile(task_name):
    task_name = str(task_name).lower()
    if "go2" in task_name:
        return GO2_D435I_4CAM_PROFILE
    return ANYMAL_LEGACY_4CAM_PROFILE


def apply_pointcloud_camera_profile(sensor_cfg, profile):
    sensor_cfg.num_sensors = int(profile["num_sensors"])
    sensor_cfg.depth_camera_config.resolution = tuple(profile["resolution"])
    sensor_cfg.depth_camera_config.horizontal_fov_deg = float(profile["horizontal_fov_deg"])
    sensor_cfg.depth_camera_config.near_clip = float(profile["near_clip"])
    sensor_cfg.depth_camera_config.far_clip = float(profile["far_clip"])
    sensor_cfg.depth_camera_config.near_plane = float(profile["near_plane"])
    sensor_cfg.depth_camera_config.far_plane = float(profile["far_plane"])
    sensor_cfg.depth_camera_config.decimation = int(profile["decimation"])
    sensor_cfg.depth_camera_config.calculate_depth = bool(profile["calculate_depth"])
    sensor_cfg.depth_camera_config.return_pointcloud = bool(profile["return_pointcloud"])
    sensor_cfg.depth_camera_config.pointcloud_in_world_frame = bool(profile["pointcloud_in_world_frame"])
    sensor_cfg.depth_camera_config.euler = list(profile["euler"])
    sensor_cfg.depth_camera_config.pos = list(profile["pos"])
