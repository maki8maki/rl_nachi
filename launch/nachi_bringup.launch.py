from launch import LaunchDescription
from launch.actions import RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

log_level_controller_manager = "INFO"
log_level_rviz = "INFO"
log_level_robot_state_publisher = "INFO"
log_level_joint_state_broadcaster = "INFO"
log_level_oss_controller = "INFO"
log_level_custom_controller = "INFO"


def generate_launch_description():
    # Get URDF via xacro
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [
                    FindPackageShare("rl_nachi"),
                    "urdf",
                    "mz04.urdf",
                ]
            ),
        ]
    )
    robot_description = {"robot_description": robot_description_content}

    # load robot and controller parameter
    robot_controllers = PathJoinSubstitution(
        [FindPackageShare("nachi_bringup"), "config", "nachi_controller_param.yaml"]
    )
    # load controller manager
    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description, robot_controllers],
        output="screen",
        # remappings=[("joint_states", "nachi_joint_states")],
        arguments=["--ros-args", "--log-level", log_level_controller_manager],
    )

    # Get robot state publisher config file and Load node
    robot_state_pub_config = PathJoinSubstitution(
        [FindPackageShare("nachi_bringup"), "config", "robot_state_publisher.yaml"]
    )
    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description, robot_state_pub_config],
        arguments=["--ros-args", "--log-level", log_level_robot_state_publisher],
    )

    # load joint state broadcaster config file and load node
    joint_state_broadcaster_config_file = PathJoinSubstitution(
        [FindPackageShare("nachi_bringup"), "config", "joint_state_broadcaster.yaml"]
    )
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
            "--ros-args",
            "--log-level",
            log_level_joint_state_broadcaster,
        ],
        parameters=[robot_description, joint_state_broadcaster_config_file],
    )

    # each controller and node, rviz2 must be loaded in sequence (not same timing)
    # load joint group angle controller
    joint_group_angle_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_group_angle_controller",
            "-c",
            "/controller_manager",
            "--ros-args",
            "--log-level",
            log_level_oss_controller,
        ],
    )
    # Delay start of joint_group_angle_controller after `joint_state_broadcaster`
    delay_joint_group_angle_controller_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[joint_group_angle_controller_spawner],
        )
    )

    # load joint group position controller
    joint_group_position_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_group_position_controller",
            "-c",
            "/controller_manager",
            "--ros-args",
            "--log-level",
            log_level_oss_controller,
        ],
    )
    # Delay start of joint_group_position_controller after `joint_group_angle_controller`
    delay_joint_group_position_controller_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_group_angle_controller_spawner,
            on_exit=[joint_group_position_controller_spawner],
        )
    )

    # load nachi service controller
    nachi_service_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "nachi_service_controller",
            "-c",
            "/controller_manager",
            "--ros-args",
            "--log-level",
            log_level_custom_controller,
        ],
    )
    # Delay start of nachi_service_controller after `joint_group_position_controller`
    delay_nachi_service_controller_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_group_position_controller_spawner,
            on_exit=[nachi_service_controller_spawner],
        )
    )

    # load nachi servomotor state controller
    nachi_servomotor_state_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "nachi_servomotor_state_controller",
            "-c",
            "/controller_manager",
            "--ros-args",
            "--log-level",
            log_level_custom_controller,
        ],
    )
    # Delay start of nachi_servomotor_state_controller after `nachi_service_controller`
    delay_nachi_servomotor_state_controller_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=nachi_service_controller_spawner,
            on_exit=[nachi_servomotor_state_controller_spawner],
        )
    )

    # regular node should be launched samet timing
    # load controller_stopper node
    nachi_controller_stopper = Node(
        package="nachi_controller_stopper",
        executable="nachi_controller_stopper",
        output="screen",
        arguments=["--ros-args", "--log-level", log_level_custom_controller],
    )
    # Delay start of nachi_controller_stopper after `nachi_servomotor_state_controller`
    delay_nachi_controller_stopper = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=nachi_servomotor_state_controller_spawner,
            on_exit=[nachi_controller_stopper],
        )
    )

    # load "nachi_param.yaml" in nachi_service_provider launch file
    nachi_param_yaml = PathJoinSubstitution([FindPackageShare("nachi_bringup"), "config", "nachi_param.yaml"])
    # load service provider config file and load node
    nachi_service_provider_config_file = PathJoinSubstitution(
        [FindPackageShare("nachi_service_provider"), "config", "nachi_service_provider_param.yaml"]
    )
    nachi_service_provider = Node(
        package="nachi_service_provider",
        executable="nachi_service_provider",
        output="screen",
        parameters=[nachi_param_yaml, nachi_service_provider_config_file],
        arguments=["--ros-args", "--log-level", log_level_custom_controller],
    )
    # Delay start of nachi_service_provider after `nachi_servomotor_state_controller`
    delay_nachi_service_provider = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=nachi_servomotor_state_controller_spawner,
            on_exit=[nachi_service_provider],
        )
    )

    # load rviz config file and rviz node
    rviz_config_file = PathJoinSubstitution([FindPackageShare("rl_nachi"), "rviz", "show_robot.rviz"])
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file, "--ros-args", "--log-level", log_level_rviz],
        on_exit=Shutdown(),
    )
    # Delay rviz start after `nachi_servomotor_state_controller`
    delay_rviz_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=nachi_servomotor_state_controller_spawner,
            on_exit=[rviz_node],
        )
    )

    nodes = [
        control_node,
        robot_state_pub_node,
        joint_state_broadcaster_spawner,
        delay_joint_group_angle_controller_spawner,
        delay_joint_group_position_controller_spawner,
        delay_nachi_service_controller_spawner,
        delay_nachi_servomotor_state_controller_spawner,
        delay_nachi_controller_stopper,
        delay_nachi_service_provider,
        delay_rviz_spawner,
    ]

    return LaunchDescription(nodes)
