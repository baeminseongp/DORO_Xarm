#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node

from rcl_interfaces.msg import ParameterDescriptor
from example_interfaces.msg import String


class PyParametrizedTemplateNode(Node):
    """A template for a simple parametrized publisher Python Node."""

    def __init__(self):
        super().__init__("dummy_parametrized_py")

        # Declaring a parameter with default value
        # The type is inferred from default value
        self.declare_parameter("timer_delta_ms", 1000)
        self.declare_parameter("word", "BAZINGA")

        # Get values from parameters (can be done when the parameters are needed)
        self._timer_delta_ms = self.get_parameter("timer_delta_ms").value
        self._word = self.get_parameter("word").value

        self._publisher = self.create_publisher(String, "dummy_topic", 10)
        self._timer = self.create_timer(
            self._timer_delta_ms / 1000, self.publish_message
        )

        self.get_logger().info("Parametrized dummy publisher node started.")

    def publish_message(self):
        """Publishes a message with parametrized content."""
        msg = String()
        msg.data = "Hello, here is the secret word: {}".format(self._word)
        self._publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)  # Initialize the ROS communication

    node = PyParametrizedTemplateNode()  # Initialize the node
    rclpy.spin(node)  # Keep the node running until it is killed

    rclpy.shutdown()  # Shutdown the ROS communication


if __name__ == "__main__":
    main()
