import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    rs_launch = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            [os.path.join(get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py")]
        ),
        launch_arguments={
            "enable_rgbd": "true",
            "align_depth.enable": "true",
            "enable_sync": "true",
            "enable_color": "true",
            "enable_depth": "true",
        }.items(),
    )

    return LaunchDescription([rs_launch])
