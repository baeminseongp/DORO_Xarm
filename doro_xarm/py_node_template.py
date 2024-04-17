#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node


class PyTemplateNode(Node):
    """A template for a basic Python Node."""

    def __init__(self):
        super().__init__("dummy_py")

        self._counter = 0

        self.get_logger().info("Hello ROS2!")
        self.create_timer(1, self.timer_callback)

    def timer_callback(self):
        """Increments a counter and logs a message"""
        self._counter += 1
        self.get_logger().info(f"Hello ({self._counter})")


def main(args=None):
    rclpy.init(args=args)  # Initialize the ROS communication

    node = PyTemplateNode()  # Initialize the node
    rclpy.spin(node)  # Keep the node running until it is killed

    rclpy.shutdown()  # Shutdown the ROS communication


if __name__ == "__main__":
    main()
