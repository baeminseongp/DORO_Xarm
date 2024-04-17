#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node

from example_interfaces.msg import String


class PySubscriberTemplateNode(Node):
    """A template for a subscriber Python Node."""

    def __init__(self):
        super().__init__("dummy_subscriber_py")

        self._subscriber = self.create_subscription(
            String, "dummy_topic", self.callback_msg_received, 10
        )

        self.get_logger().info("Dummy publisher node started.")

    def callback_msg_received(self, msg):
        """Runs when a message is received."""
        self.get_logger().info("Received message: '{}'.".format(msg.data))


def main(args=None):
    rclpy.init(args=args)  # Initialize the ROS communication

    node = PySubscriberTemplateNode()  # Initialize the node
    rclpy.spin(node)  # Keep the node running until it is killed

    rclpy.shutdown()  # Shutdown the ROS communication


if __name__ == "__main__":
    main()
