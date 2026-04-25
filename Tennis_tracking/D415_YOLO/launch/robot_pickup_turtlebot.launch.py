import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch.substitutions import Command
from launch_ros.actions import Node


URDF_PATH = "/opt/ros/jazzy/share/turtlebot3_description/urdf/turtlebot3_burger.urdf"
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_SCRIPT = os.path.join(_PROJECT_ROOT, "robot", "robot_path_sim.py")


def generate_launch_description():
    robot_description = Command(["xacro ", URDF_PATH, " namespace:="])

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )

    robot_pickup_sim = ExecuteProcess(
        cmd=["/usr/bin/python3", SIM_SCRIPT],
        output="screen",
    )

    return LaunchDescription([
        robot_state_publisher,
        robot_pickup_sim,
    ])
