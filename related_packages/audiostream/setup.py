"""Setup configuration for audiostream package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="audiostream",
    version="0.1.0",
    author="Theremin Project",
    description="Audio input feature extraction - real-time pitch, onset, rhythm detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thorwhalen/theremin",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "sounddevice>=0.4.0",
        "aubio>=0.4.9",
        "numpy>=1.19.0",
    ],
)
