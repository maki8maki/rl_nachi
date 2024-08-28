import os
from glob import glob

from setuptools import find_packages, setup

package_name = "rl_nachi"


def package_files(directory, data_files: list):
    for path, _, _ in os.walk(directory):
        data_files.append((os.path.join("share/", package_name, path), glob(os.path.join(path, "*.*"))))
    return data_files


data_files = []
data_files.append(("share/ament_index/resource_index/packages", ["resource/" + package_name]))
data_files.append(("share/" + package_name, ["package.xml"]))
data_files = package_files("launch/", data_files)
data_files = package_files("rviz/", data_files)
data_files = package_files("urdf/", data_files)
data_files = package_files("rl_nachi/config", data_files)
data_files = package_files("rl_nachi/model", data_files)

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="robot",
    maintainer_email="robot@todo.todo",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "main = rl_nachi.main:main",
            "sb3 = rl_nachi.sb3:main",
        ],
    },
)
