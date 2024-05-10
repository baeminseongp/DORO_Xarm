# -*- coding: utf-8 -*-
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    inference_params = LaunchConfiguration(
        "inference_params",
        default=os.path.join(
            get_package_share_directory(
                "doro_xarm"), "configs", "inference.yaml"
        ),
    )
    model_file = LaunchConfiguration(
        "model_file",
        default=os.path.join(
            get_package_share_directory(
                "doro_xarm"), "model", "modelcnn.pt"
        ),
    )
    data_path = LaunchConfiguration(
        "data_path",
        default=os.path.join(
            get_package_share_directory(
                "doro_xarm"), "data"
        ),
    )

    inference = Node(
        package="doro_xarm",
        executable="inference_node",
        name="inference_node",
        output="screen",
        parameters=[inference_params,
                    {"model_file":model_file, 
                     "data_path":data_path}],
        namespace="",
    )

    ld = LaunchDescription()

    ld.add_action(inference)

    return ld