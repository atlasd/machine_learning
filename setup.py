from setuptools import setup, find_packages

test = ["pytest"]

setup(
    name="machine-learning",
    package_dir={"machine_learning": "machine_learning"},
    version="0.0.1",
    packages=find_packages(),
    extras_require={"test": test},
)
