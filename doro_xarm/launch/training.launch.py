# -*- coding: utf-8 -*-
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    training_params = LaunchConfiguration(
        "training_params",
        default=os.path.join(
            get_package_share_directory(
                "doro_xarm"), "configs", "training.yaml"
        ),
    )
    train_path = LaunchConfiguration(
        "train_path",
        default=os.path.join(
            get_package_share_directory(
                "doro_xarm"), "data", "dataset"
        ),
    )
    test_path = LaunchConfiguration(
        "test_path",
        default=os.path.join(
            get_package_share_directory(
                "doro_xarm"), "data", "testset"
        ),
    )
    model_path = LaunchConfiguration(
        "model_path",
        default=os.path.join(
            get_package_share_directory(
                "doro_xarm"), "model",
        ),
    )

    training = Node(
        package="doro_xarm",
        executable="training_node",
        name="training_node",
        output="screen",
        parameters=[training_params,
                    {"train_path":train_path, 
                     "test_path":test_path,
                     "model_path":model_path}],
        namespace="",
    )

    ld = LaunchDescription()

    ld.add_action(training)

    return ld