import rclpy
from rclpy.node import Node
from doro_interfaces.srv import Index

class MyClient(Node):
    def __init__(self):
        super().__init__('my_client')

        self.client = self.create_client(Index, 'index')

        # Wait for the server to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')

        # Call the service
        self.call_service()

    def call_service(self):
        
        request = Index.Request()
        request.index = [1, 1, 2, 
                         2, 2, 0, 
                         1, 1, 0]

        # Call the service
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            # Process the response message
            response = future.result()
            self.get_logger().info(f'Response: {response.result}')
            # Do something with the response data
        else:
            self.get_logger().info('Service call failed')

def main(args=None):
    rclpy.init(args=args)
    client = MyClient()
    rclpy.shutdown()

if __name__ == '__main__':
    main()