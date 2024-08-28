from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit


def generate_launch_description():
    nachi_launch = ExecuteProcess(
        cmd=["ros2", "launch", "rl_nachi", "nachi_bringup.launch.py"],
        output="screen",
    )

    rs_launch = ExecuteProcess(
        cmd=["ros2", "launch", "rl_nachi", "realsense.launch.py"],
        output="screen",
    )

    shutdown = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=nachi_launch,
            on_exit=[Shutdown()],
        )
    )

    return LaunchDescription(
        [
            nachi_launch,
            rs_launch,
            shutdown,
        ]
    )
