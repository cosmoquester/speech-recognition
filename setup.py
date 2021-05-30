from setuptools import find_packages, setup

setup(
    name="speech-recognition",
    version="0.0.1",
    description="Develope speech recognition models with tensorflow 2",
    python_requires=">=3.6",
    install_requires=["tensorflow>=2", "tensorflow-io", "tensorflow-text", "tqdm", "omegaconf"],
    url="https://github.com/psj8252/speech-recognition.git",
    author="Park Sangjun",
    packages=find_packages(exclude=["tests"]),
)
