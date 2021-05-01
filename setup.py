from setuptools import find_packages, setup

setup(
    name="tf2-keras-template",
    version="0.0.1",
    description="This is template repository for tensorflow keras model development.",
    python_requires=">=3.6",
    install_requires=["tensorflow>=2"],
    url="https://github.com/psj8252/tf2-keras-template.git",
    author="Park Sangjun",
    packages=find_packages(exclude=["tests"]),
)
