from setuptools import find_packages,setup
from typing import List

def get_requirements()-> List[str]:
    requirements_list : list[str] = []

    requirements_list




setup(
name = "10yrs_chd_predictor",
version = "0.0.1",
author = "vishal",
author_email = "support@havinosh.com",
packages = find_packages(),
install_requires = get_requirements() #["numpy","pandas"]

)