import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


def generate_launch_description():

    pkg_path = get_package_share_directory('my_robot_description')
    nav2_params = os.path.join(pkg_path, 'config', 'nav2_params.yaml')

    return LaunchDescription([

        # SLAM for robot 1
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'odom_frame': 'my_robot/odom',
                'map_frame': 'map',
                'base_frame': 'my_robot/base_footprint',
                'scan_topic': '/my_robot/scan',
                'mode': 'mapping',
                'transform_publish_period': 0.02,
                'map_update_interval': 2.0,
                'resolution': 0.05,
                'max_laser_range': 12.0,
                'minimum_time_interval': 0.3,
                'transform_timeout': 0.5,
                'tf_buffer_duration': 30.0,
                'stack_size_to_use': 40000000,
            }]
        ),

        # Nav2 starts after 8 seconds to let SLAM initialize
        TimerAction(
            period=20.0,
            actions=[

                Node(
                    package='nav2_controller',
                    executable='controller_server',
                    name='controller_server',
                    output='screen',
                    parameters=[nav2_params, {'use_sim_time': True}],
                    remappings=[
                        ('scan', '/my_robot/scan'),
                        ('cmd_vel', '/my_robot/cmd_vel'),
                        ('odom', '/my_robot/odom'),
                    ]
                ),

                Node(
                    package='nav2_planner',
                    executable='planner_server',
                    name='planner_server',
                    output='screen',
                    parameters=[nav2_params, {'use_sim_time': True}],
                    remappings=[
                        ('scan', '/my_robot/scan'),
                    ]
                ),

                Node(
                    package='nav2_behaviors',
                    executable='behavior_server',
                    name='behavior_server',
                    output='screen',
                    parameters=[nav2_params, {'use_sim_time': True}],
                    remappings=[
                        ('cmd_vel', '/my_robot/cmd_vel'),
                    ]
                ),

                Node(
                    package='nav2_bt_navigator',
                    executable='bt_navigator',
                    name='bt_navigator',
                    output='screen',
                    parameters=[nav2_params, {
                        'use_sim_time': True,
                        'global_frame': 'map',
                        'robot_base_frame': 'my_robot/base_link',
                        'odom_topic': '/my_robot/odom',
                    }]
                ),

                Node(
                    package='nav2_lifecycle_manager',
                    executable='lifecycle_manager',
                    name='lifecycle_manager_navigation',
                    output='screen',
                    parameters=[{
                        'use_sim_time': True,
                        'autostart': True,
                        'node_names': [
                            'controller_server',
                            'planner_server',
                            'behavior_server',
                            'bt_navigator',
                        ]
                    }]
                ),

            ]
        ),

    ])
