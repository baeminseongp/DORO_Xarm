#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node

from example_interfaces.srv import AddTwoInts


class PyServiceServerNode(Node):
    """A template for a service server Python Node."""

    def __init__(self):
        super().__init__("dummy_server_py")

        self.server_ = self.create_service(
            AddTwoInts, "dummy_service", self.callback_service_call
        )
        self.get_logger().info("Dummy service server node started.")

    def callback_service_call(self, request, response):
        """Runs when a service has been called with a request."""
        response.sum = request.a + request.b
        self.get_logger().info(
            "Computed: {} + {} = {}".format(request.a, request.b, response.sum)
        )
        return response


def main(args=None):
    rclpy.init(args=args)  # Initialize the ROS communication

    node = PyServiceServerNode()  # Initialize the node
    rclpy.spin(node)  # Keep the node running until it is killed

    rclpy.shutdown()  # Shutdown the ROS communication


if __name__ == "__main__":
    main()
