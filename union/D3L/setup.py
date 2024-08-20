import logging
import os

from setuptools import find_packages, setup

logging.basicConfig()

ROOT = os.path.abspath(os.path.dirname(__file__))
def load_requirements() -> list:
    """Load requirements from file, parse them as a Python list!"""
    with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as f:
        all_reqs = f.read().split("\n")
    install_requires = [x.strip() for x in all_reqs if "git+" not in x]

    return install_requires

setup(
    name='D3l',
    version='0.1',
    packages=find_packages(exclude=["contrib", "test-docs", "tests*"]),
    install_requires=load_requirements(),
    include_package_data=True,
    url='https://arxiv.org/pdf/2011.10427.pdf',
    license='Apache License',
    author='Alex Bogatu',
    author_email='alex.bogatu89@yahoo.com',
    description='D3L Data Discovery Framework.'
)
