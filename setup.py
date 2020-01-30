from setuptools import setup

test = ["pytest"]

setup(
    name="machine_learning",
    package_dir={"machine_learning": "machine_learning"},
    version="0.0.1",
    extra_requires={"test": test},
)
