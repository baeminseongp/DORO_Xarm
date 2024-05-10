# -*- coding: utf-8 -*-
from setuptools import find_packages, setup
from glob import glob
package_name = "doro_xarm"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/configs", glob("configs/*.*")),
        ("share/" + package_name + "/model", glob("model/*.pt")),
        ("share/" + package_name + "/data", glob("data/*.*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="joenghan",
    maintainer_email="kimjh9813@naver.com",
    description="TODO: Package description",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "inference_node = doro_xarm.inference:main",
            "training_node = doro_xarm.training:main",
        ],
    },
)
