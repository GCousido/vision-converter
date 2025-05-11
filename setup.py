from setuptools import setup, find_packages

setup(
    name="datasetconverter",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "click",
        "pillow"
    ],
    entry_points={
        'console_scripts': [
            'dconverter=cli.main:dconverter',
        ],
    },
)
