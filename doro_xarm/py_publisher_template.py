#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node

from example_interfaces.msg import String


class PyPublisherTemplateNode(Node):
    """A template for a publisher Python Node."""

    def __init__(self):
        super().__init__("dummy_publisher_py")

        self._publisher = self.create_publisher(String, "dummy_topic", 10)
        self._timer = self.create_timer(1, self.publish_message)

        self.get_logger().info("Dummy publisher node started.")

    def publish_message(self):
        """Publishes a message."""
        msg = String()
        msg.data = "Hello from a dummy publisher!"
        self._publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)  # Initialize the ROS communication

    node = PyPublisherTemplateNode()  # Initialize the node
    rclpy.spin(node)  # Keep the node running until it is killed

    rclpy.shutdown()  # Shutdown the ROS communication


if __name__ == "__main__":
    main()
