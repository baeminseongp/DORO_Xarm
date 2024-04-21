# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

package_name = "tik_tak_toe"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
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
            "tik_tak_toe_node = tik_tak_toe.tik_tak_toe_node:main",
            "client_test = tik_tak_toe.client_test:main",
        ],
    },
)
