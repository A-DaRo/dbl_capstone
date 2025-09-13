# setup.py
from setuptools import setup, find_packages

setup(
    name="coral_mtl",
    version="0.1.0",
    description="Multi-task coral segmentation model",
    package_dir={"": "src"},                # Tell setuptools packages are under 'src'
    packages=find_packages(where="src"),   # Find all packages under 'src'
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "transformers",
    ],
)