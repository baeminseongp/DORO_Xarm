import socket

class ImageClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.client_socket = None

    def connect(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.host, self.port))

    def request_image(self, image_filename, save_path):
        self.client_socket.sendall(image_filename.encode('utf-8'))
        self.receive_image(save_path)

    def receive_image(self, image_filename):
        try:
            with open(image_filename, 'wb') as f:
                while True:
                    data = self.client_socket.recv(1024)
                    if not data:
                        break
                    f.write(data)
        except Exception as e:
            print("Error receiving image:", e)

    def close(self):
        self.client_socket.close()

