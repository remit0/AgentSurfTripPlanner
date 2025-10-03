# setup.py
from setuptools import setup, find_packages

setup(
    package_dir={"": "AGENTSURFTRIPPLANNER/lib/python"},
    packages=find_packages(
        where="AGENTSURFTRIPPLANNER/lib/python",
        exclude=["tests"],
    ),
)
