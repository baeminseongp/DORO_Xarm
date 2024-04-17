# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from deeplearning.cnn import CNN
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

doro_xarm_path = get_package_share_directory("doro_xarm")

class TrainingNode(Node):
    
    def __init__(self):
        super().__init__("training_node")
        self.init_parameters()
        self.model = CNN()

    def init_parameters(self):
        self.declare_parameter("train_path", doro_xarm_path + "/data/dataset")
        self.declare_parameter("test_path", doro_xarm_path + "/data/testset")
        self.declare_parameter("model_path", doro_xarm_path + "/model")
        self.declare_parameter("batch_size", 10)
        self.declare_parameter("epochs", 30)
        self.declare_parameter("device", "cpu")

        self.train_path = self.get_parameter("train_path").value
        self.test_path = self.get_parameter("test_path").value
        self.model_path = self.get_parameter("model_path").value
        self.batch_size = self.get_parameter("batch_size").value
        self.epochs = self.get_parameter("epochs").value
        self.device = self.get_parameter("device").value


        self.get_logger().info("Parameters initialized.")

    def set_deeplearning_parameters(self):
        self.model.set_device(self.device)
        self.model.set_optimizer(torch.optim.Adam(self.model.parameters(), lr=0.001))
        self.model.set_loss(nn.CrossEntropyLoss())

    def model_training(self):
        for epoch in range(1, self.epochs + 1):
            self.model.train(self.trainloader, log_interval=10, epoch=epoch)
            test_loss, test_accuracy = self.model.evaluate(self.testloader)
            self.get_logger().info(
                "Epoch: {}, Test Loss: {:.4f}, Test Accuracy: {:.2f}%".format(epoch, test_loss, test_accuracy))

    def set_image_transform(self):
        self.trans = transforms.Compose([
            transforms.Resize((100, 100)),                                  # 이미지 크기 조정
            transforms.RandomHorizontalFlip(),                              # 랜덤하게 이미지 좌우 반전
            transforms.RandomRotation(15),                                  # 랜덤하게 이미지 회전
            transforms.ToTensor(),                                          # 이미지를 텐서로 변환
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),         # 이미지를 정규화
            transforms.Grayscale(num_output_channels=1)                     # grayscale로 변환한다.
        ])

    def load_dataset(self):
        trainset = torchvision.datasets.ImageFolder(root=self.train_path, 
                                                    transform = self.trans) 
        testset = torchvision.datasets.ImageFolder(root=self.test_path,
                                                   transform= self.trans)
        self.trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        self.testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False)

    def run(self):
        self.set_image_transform()
        self.load_dataset()
        self.set_deeplearning_parameters()
        self.model_training()
        self.get_logger().info("Training completed.")
        torch.save(self.model, self.model_path + "/modelcnn.pt")
        self.get_logger().info("Model saved.")


if __name__ == "__main__":
    rclpy.init()
    node = TrainingNode()
    node.run()
    node.save_model()
    try:
        node.get_logger().info("Beginning client, shut down with CTRL-C")
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down.\n")

    node.destroy_node()
    rclpy.shutdown()
