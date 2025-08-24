from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="drivable-corridor",
    version="0.1.0",
    author="sriram-0311",
    author_email="ramesh.anu@northeastern.edu",
    description="CNN-based drivable corridor detection for autonomous vehicles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sriram-0311/Drivable-Corridor",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "isort>=5.9",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "drivable-train=scripts.train:main",
            "drivable-inference=scripts.inference:main",
        ],
    },
)
