#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from functools import partial

from example_interfaces.srv import AddTwoInts


class PyServiceClientNode(Node):
    """A template for a service client Python Node."""

    def __init__(self):
        super().__init__("dummy_client_py")

        self._client = self.create_client(AddTwoInts, "dummy_service")
        self.wait_for_server()  # Wait until the server is available

        # Example service calls
        self.call_service_server(2, 2)
        self.call_service_server(8, 1)
        self.call_service_server(7, 9)

    def call_service_server(self, a, b):
        """Calls service server asynchronously with given arguments."""
        # Create a request
        request = AddTwoInts.Request()
        request.a, request.b = a, b

        # Call the service
        future = self._client.call_async(request)

        # Add a callback for the response
        future.add_done_callback(partial(self.callback_done, a=a, b=b))

    def callback_done(self, future, a, b):
        """Runs when a response from the server is received.

        Note: The arguments for the request are included, so when the callback
        is called we know what the request has been.
        """
        try:
            response = future.result()
            self.get_logger().info(
                "Response from server: {} + {} = {}".format(a, b, response.sum)
            )
        except Exception as e:
            self.get_logger().error("Service call failed %r" % (e,))

    def wait_for_server(self):
        """Waits for the server in a loop."""
        while not self._client.wait_for_service(1.0):
            self.get_logger().warn("Waiting for dummy server...")


def main(args=None):
    rclpy.init(args=args)  # Initialize the ROS communication

    node = PyServiceClientNode()  # Initialize the node
    rclpy.spin(node)  # Keep the node running until it is killed

    rclpy.shutdown()  # Shutdown the ROS communication


if __name__ == "__main__":
    main()
