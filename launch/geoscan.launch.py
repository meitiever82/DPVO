#!/usr/bin/env python3
"""Launch DPVO ROS2 node on the GeoScan B1 bag.

All node parameters and the image/pose topics are configurable as launch
arguments, e.g.:

    ros2 launch src/DPVO/launch/geoscan.launch.py stride:=2 image_topic:=/left_camera/image

Set play_bag:=true to also auto-play the bag (via the monotonic replayer,
which avoids the ros2-bag-play out-of-order delivery bug on GeoScan):

    ros2 launch src/DPVO/launch/geoscan.launch.py play_bag:=true
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

# .../src/DPVO
DPVO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DPVO deps (torch/yacs/dpvo CUDA ext) live in this venv; system python lacks them.
# rclpy is still picked up from the sourced ROS distro via PYTHONPATH.
VENV_PY = os.path.join(DPVO_DIR, '.venv', 'bin', 'python')
NODE_PY = VENV_PY if os.path.exists(VENV_PY) else 'python3'
DEFAULT_BAG = os.path.expanduser('~/Documents/Datasets/geoscan/B1/2026-02-12-16-47-48')
DEFAULT_CAMCHAIN = os.path.expanduser('~/Documents/Datasets/geoscan/geoscan_camchain-imucam.yaml')


def generate_launch_description():
    lc = LaunchConfiguration

    launch_args = [
        DeclareLaunchArgument('network', default_value=os.path.join(DPVO_DIR, 'dpvo.pth')),
        DeclareLaunchArgument('config', default_value=os.path.join(DPVO_DIR, 'config/geoscan/435i.yaml')),
        DeclareLaunchArgument('camchain', default_value=DEFAULT_CAMCHAIN, description='Kalibr camchain yaml (cam0 intrinsics); empty = built-in default'),
        DeclareLaunchArgument('stride', default_value='1', description='process every Nth frame'),
        DeclareLaunchArgument('save_trajectory', default_value=os.path.join(DPVO_DIR, 'results/dpvo_ros_geoscan_traj.txt')),
        DeclareLaunchArgument('backend_thresh', default_value='64.0'),
        DeclareLaunchArgument('image_topic', default_value='/left_camera/image'),
        DeclareLaunchArgument('pose_topic', default_value='/dpvo/pose'),
        DeclareLaunchArgument('pose_frame_id', default_value='dpvo_world'),
        DeclareLaunchArgument('publish_pose', default_value='true'),
        # optional bag auto-play
        DeclareLaunchArgument('play_bag', default_value='false', description='auto-play the bag after the node starts'),
        DeclareLaunchArgument('bag', default_value=DEFAULT_BAG),
        DeclareLaunchArgument('start_offset', default_value='10', description='bag play --start-offset (skip static startup)'),
    ]

    dpvo_node = ExecuteProcess(
        cmd=[
            NODE_PY, os.path.join(DPVO_DIR, 'dpvo_ros_node.py'),
            '--ros-args',
            '-p', ['network:=', lc('network')],
            '-p', ['config:=', lc('config')],
            '-p', ['camchain:=', lc('camchain')],
            '-p', ['stride:=', lc('stride')],
            '-p', ['save_trajectory:=', lc('save_trajectory')],
            '-p', ['backend_thresh:=', lc('backend_thresh')],
            '-p', ['image_topic:=', lc('image_topic')],
            '-p', ['pose_topic:=', lc('pose_topic')],
            '-p', ['pose_frame_id:=', lc('pose_frame_id')],
            '-p', ['publish_pose:=', lc('publish_pose')],
        ],
        cwd=DPVO_DIR,  # so `import dpvo` resolves, matching manual `python3 dpvo_ros_node.py`
        output='screen',
    )

    # Give the node ~5s to load weights and start subscribing before replaying.
    # DPVO uses a single image topic, so there is no cross-topic ordering issue
    # and native `ros2 bag play` is preferred (the monotone replayer's
    # deserialize/reserialize path truncates large Image messages over DDS).
    play_bag = TimerAction(
        period=5.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'bag', 'play', lc('bag'),
                    '--topics', lc('image_topic'),
                    '--start-offset', lc('start_offset'),
                ],
                output='screen',
                condition=IfCondition(lc('play_bag')),
            )
        ],
    )

    return LaunchDescription(launch_args + [dpvo_node, play_bag])
