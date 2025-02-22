from setuptools import setup, find_packages

setup(
    name="medsynnet",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.8.0",
        "numpy>=1.19.2",
        "pandas>=1.2.4",
        "scikit-learn>=0.24.2",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "diffusers>=0.12.0",
        "transformers>=4.18.0",
        "flwr>=1.0.0",
    ],
    python_requires=">=3.8",
) 