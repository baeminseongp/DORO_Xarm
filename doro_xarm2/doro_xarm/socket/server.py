import subprocess
import socket
import sys
class ImageServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server_socket = None

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"server start: {self.host}:{self.port}")

    def serve(self):
        while True:
            conn, addr = self.server_socket.accept()
            print("connected client:", addr)

            image_filename = conn.recv(1024).decode('utf-8').strip()
            print("request image file :", image_filename)

            self.capture_and_save_image(image_filename)

            self.send_image(conn, image_filename)

            conn.close()

    def send_image(self, conn, image_filename):
        try:
            with open(image_filename, 'rb') as f:
                image_data = f.read()
                conn.sendall(image_data)
        except FileNotFoundError:
            conn.sendall(b"File not found")

    def capture_and_save_image(self, output_path):
        cmd = ["libcamera-jpeg", "-o", "-", "|", "tee", output_path, ">", "/dev/null"]
        subprocess.run(" ".join(str(x) for x in cmd), shell=True)

    def __del__(self):
        if self.server_socket:
            self.server_socket.close()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script.py <host> <port>")
        sys.exit(1)
    host = str(sys.argv[1])
    port = int(sys.argv[2])
    server = ImageServer(host, port)
    server.start()
    server.serve()