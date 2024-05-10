# DORO_Xarm

## Description
이 패키지는 DORO Xarm 프로그램을 위해 기획된 프로젝트입니다.\
패키지의 구성은 다음과 같습니다.

- doro_interface : custom msg, srv으로 구성되어 있는 패키지
- doro_xarm : CNN기반 O,X 추론 패키지
- tik_tak_toe : ox기반 다음 tik_tak_toe를 결정해주는패키지


### 📄 1. doro_interface

해당 패키지는 ros2 custom msg와 srv를 담고 있는 패키지입니다.\
현재는 index.srv 파일만 존재하는 상황입니다.

#### index.srv
```
int64[9] index #ox에 대한 정보
---
int32 result   #어디다 두어야하는지에 대한 결과 값
```

### 🧐 2. doro_xarm

해당 패키지는 좀더 복잡한 구조를 가지고 있습니다.
```
📦doro_xarm
 ┣ 📂configs              //ros2 config
 ┃ ┣ 📜inference.yaml     
 ┃ ┗ 📜training.yaml        
 ┣ 📂data
 ┃ ┣ 📜dataset.zip        //dataset 압축파일
 ┃ ┣ 📜test_image.zip     
 ┃ ┗ 📜testset.zip        //testset 압축파일
 ┣ 📂doro_xarm
 ┃ ┣ 📂deeplearning
 ┃ ┃ ┣ 📜__init__.py
 ┃ ┃ ┣ 📜cnn.py           //cnn 모델
 ┃ ┃ ┗ 📜dataset.py       //inference에 쓰이는 dataset을 parsing하는 코드
 ┃ ┣ 📂socket
 ┃ ┃ ┣ 📜__init__.py
 ┃ ┃ ┣ 📜client.py        // rpi에 사진을 요청하는 코드
 ┃ ┃ ┗ 📜server.py        // rpi에 들어있는 서버 코드
 ┃ ┣ 📂utils
 ┃ ┃ ┣ 📜__init__.py
 ┃ ┃ ┗ 📜ox_crop.py       // 찍힌 9개의 o,xd의 pixel을 crop하는 코드
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜inference.py       // ROS2 inferece node code(main)
 ┃ ┗ 📜training.py        // ROS2 training node code(main)
 ┣ 📂launch
 ┃ ┣ 📜inference.launch.py
 ┃ ┗ 📜training.launch.py
 ┣ 📂model
 ┃ ┗ 📜modelcnn.pt        // 기존의 training dataset으로 만들어진 pt파일
 ┣ 📂resource
 ┃ ┗ 📜doro_xarm
 ┣ 📂test
 ┃ ┣ 📜test_copyright.py
 ┃ ┣ 📜test_flake8.py
 ┃ ┗ 📜test_pep257.py
 ┣ 📜README.md
 ┣ 📜package.xml
 ┣ 📜setup.cfg
 ┗ 📜setup.py
```
해당 패키지는 전부 ros2 install directory안에서 상호작용을 진행하게 됩니다.\
즉, 저장되는 사진을 찾기 위해서는 install/doro_xarm directory를 확인해보면 됩니다.

- rpi에서 찍어오는 사진
- 압축이 풀리는 결과
- rpi의 사진이 9장으로 잘려 저장된 사진
- etc...

ROS2에서는 모든 설치와 상호작용을 install에서 진행하는 것을 권장하고 있고, 이 패키지는 모든 상호작용을 source가 아닌 install directory에서 진행합니다.

### 🤡 3. tik_tak_toe

tik_tak_toe패키지는 tik_tak_toe_node와 이를 테스트하기 위한 client_test노드 두가지가 존재합니다.\
tik_tak_tow_node는 현재 tik_tak_toe game의 상황의 9개의ox를 전달하면 결과를 반환합니다.\
이 결과값을 다른 노드에서 필요로 한다면, 추가적으로 client 코드를 작성해주면 됩니다.

## Installation
1. Clone the repository:
    ```shell
    git clone https://github.com/username/DORO_Xarm.git
    ```
## build
1. build
    ```shell
    $ colcon build --symlink-install
    ```

## Usage
1. Launch the DORO_xarm training node
    ```shell
    $ ros2 launch doro_xarm training.launch.py
    ```
2. Launch the DORO_xarm inference node
    ```shell
    $ ros2 launch doro_xarm inference.launch.py
    ```
3. run the tik_tak_toe node
    ```shell
    $ ros2 run tik_tak_toe tik_tak_toe_node.py
    ```

