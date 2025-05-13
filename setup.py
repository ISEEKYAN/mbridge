from setuptools import find_packages, setup

setup(
    name="mbridge",
    version="0.1.0",
    description="Bridge between Reinforcement Learning and Megatron-Core",
    author="Yan Bai",
    author_email="bayan@nvidia.com",
    packages=find_packages(),
    install_requires=[
        # "megatron-core>=0.11.0",
    ],
    python_requires=">=3.8",
)
