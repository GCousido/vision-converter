from setuptools import setup, find_packages

setup(
    name="vision-converter",
    version="0.1.0",
    description="This project consist of a library and a CLI for converting datasets between annotation formats.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Guillermo Cousido Martínez",
    author_email="guillermo.cousido@gmail.com",
    python_requires=">=3.12",
    packages=find_packages(),
    install_requires=[
        "click>=8.1.8",
        "pillow>=11.2.1",
        "tensorflow>=2.19.0",
        "pycocotools>=2.0.10",
        "numpy>=2.1.3"
    ],
    keywords=[
        "dataset", "converter", "computer vision", "cli", "yolo", "coco", "pascal voc"
    ],
    entry_points={
        "console_scripts": [
            "vconverter=vision_converter.cli.main:vconverter",
        ],
    },
    extras_require={
        "dev": [
            "pytest>=8.3.5",
            "pytest-mock>=3.14.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
    ],
    include_package_data=True,
    license="MIT",
)
