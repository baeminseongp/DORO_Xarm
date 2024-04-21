# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

import os
import torch
from torchvision import transforms
from time import sleep

from doro_xarm import *
doro_xarm_path = get_package_share_directory("doro_xarm")

class InferenceNode(Node):
    def __init__(self):
        super().__init__("inference_node")
        self.init_parameters()
        self.im_sock = ImageClient(self.ip, self.port)
        self.image_queue = []
        self.predictions = []
        # while True:
        #     try:
        #         self.im_sock.connect()
        #         break
        #     except Exception as e:
        #         self.get_logger().error(f"Failed to connect to server: {e}")
        #         sleep(1)

    def init_parameters(self):
        self.declare_parameter("device", "cpu")
        self.declare_parameter("model_file", doro_xarm_path+"model/modelcnn.pt")
        self.declare_parameter("ip", "localhost")
        self.declare_parameter("port", 12345)
        self.declare_parameter("data_path", doro_xarm_path+"/data")
        
        self.device = self.get_parameter("device").value
        self.model_file = self.get_parameter("model_file").value
        self.ip = self.get_parameter("ip").value
        self.port = self.get_parameter("port").value
        self.data_path = self.get_parameter("data_path").value
        
        self.get_logger().info("Parameters initialized.")

    def capture_image(self):
        self.image_queue.append("image" + str(self.get_clock().now().to_msg().sec) + ".jpg")
        self.image_path = self.data_path + "/" + self.image_queue[-1]
        self.im_sock.request_image(self.image_queue[-1],
                                   self.image_path)                                     # 이미지 요청
        self.get_logger().info(f"Received image: {self.image_queue[-1]}")        

    def set_custom_dataset(self):
        # self.save_path = self.data_path + "/" + self.image_queue[-1][:-4]
        # os.makedirs(self.save_path, exist_ok=True)                  # 이미지 저장 경로 생성
        # self.get_logger().info(f"save_path: {self.save_path}") 
        # imgcut(self.image_path, self.save_path)
        self.get_logger().info("Images cropped.")
        self.save_path = self.data_path + "/image1713626842"
        image_files = get_image_files(self.save_path)
        # self.image_queue.pop(0)
        self.infer_dataset = CustomDataset(image_files, transform=self.trans)
        self.get_logger().info("Dataset created.")



    def set_image_transform(self):
        self.trans = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.RandomHorizontalFlip(),  # 랜덤하게 이미지를 좌우 반전합니다.
            transforms.RandomRotation(15),  # 이미지를 최대 15도까지 랜덤하게 회전시킵니다.
            transforms.ToTensor(), # 배열 -> Tensor로 변환합니다.
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 픽셀 값의 범위를 0~1로 조절합니다.
            transforms.Grayscale(num_output_channels=1)   # grayscale로 변환한다.
        ])


    def inference(self):
        model = CNN().to(self.device)
        check_point = torch.load(self.model_file, map_location=torch.device(self.device))
        model.load_state_dict(check_point)
        model.eval()

        with torch.no_grad():
            for idx in range(len(self.infer_dataset)):
                img = self.infer_dataset[idx]
                out = model(img.unsqueeze(0))
                _, predict = torch.max(out.data, 1)
                self.predictions.append(predict.item())
        
        self.get_logger().info(f"Predictions: {self.predictions}")

    def run(self):
        # self.capture_image()       # 이미지 캡처
        self.set_image_transform() # 이미지 변환 설정
        self.set_custom_dataset()  # custom 데이터셋 설정
        self.inference()           # 추론 수행



    def __del__(self):
        # self.im_sock.close()
        self.get_logger().info("Inference node terminated.")

def main():
    rclpy.init()
    mode = CNN()
    node = InferenceNode()
    node.run()
    try:
        node.get_logger().info("Beginning client, shut down with CTRL-C")
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down.\n")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()