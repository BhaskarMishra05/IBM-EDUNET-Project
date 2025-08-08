from typing import List
from datetime import datetime
from setuptools import find_packages, setup
def get_requirements(file_path: str) -> List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements= [req.replace('\n','') for req in requirements]
        return requirements
setup(
    name='salaryprediction',
    version='0.0.0',
    author='BhaskarMishra',
    author_email='bhaskarmishra1590@gmail.com',
    packages= find_packages(),
    install_requires= get_requirements('requirements.txt')
)