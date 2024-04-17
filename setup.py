# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

package_name = "doro_xarm"

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
            "dummy_py_node = py_pkg_template.py_node_template:main",
            "dummy_publisher_py_node = py_pkg_template.py_publisher_template:main",
            "dummy_subscriber_py_node = py_pkg_template.py_subscriber_template:main",
            "dummy_parametrized_py_node = py_pkg_template.py_parametrized_template:main",
            "dummy_server_py_node = py_pkg_template.py_service_server_template:main",
            "dummy_client_py_node = py_pkg_template.py_service_client_template:main",
        ],
    },
)
