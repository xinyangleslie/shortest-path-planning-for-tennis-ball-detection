import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


URDF_PATH = "/opt/ros/jazzy/share/turtlebot3_description/urdf/turtlebot3_burger.urdf"
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLANNER_SCRIPT = os.path.join(_PROJECT_ROOT, "robot_cluster_pickup_after_demo.py")
WORK_DIR = _PROJECT_ROOT


def launch_planner(context, *args, **kwargs):
    collect_seconds = LaunchConfiguration("collect_seconds").perform(context)
    cluster_eps = LaunchConfiguration("cluster_eps").perform(context)
    target_clusters = LaunchConfiguration("target_clusters").perform(context)
    transition_weight = LaunchConfiguration("transition_weight").perform(context)
    save_snapshot = LaunchConfiguration("save_snapshot").perform(context).strip()
    load_snapshot = LaunchConfiguration("load_snapshot").perform(context).strip()

    cmd = [
        "/usr/bin/python3",
        PLANNER_SCRIPT,
        "--collect-seconds",
        collect_seconds,
        "--cluster-eps",
        cluster_eps,
        "--target-clusters",
        target_clusters,
        "--transition-weight",
        transition_weight,
    ]
    if save_snapshot:
        cmd.extend(["--save", save_snapshot])
    if load_snapshot:
        cmd.extend(["--load", load_snapshot])

    return [ExecuteProcess(cmd=cmd, cwd=WORK_DIR, output="screen")]


def generate_launch_description():
    robot_description = Command(["xacro ", URDF_PATH, " namespace:="])

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )

    return LaunchDescription([
        DeclareLaunchArgument("collect_seconds", default_value="2.0"),
        DeclareLaunchArgument("cluster_eps", default_value="0.75"),
        DeclareLaunchArgument("target_clusters", default_value="3"),
        DeclareLaunchArgument("transition_weight", default_value="1.0"),
        DeclareLaunchArgument("save_snapshot", default_value=""),
        DeclareLaunchArgument("load_snapshot", default_value=""),
        robot_state_publisher,
        OpaqueFunction(function=launch_planner),
    ])
