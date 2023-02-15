from setuptools import find_packages, setup

NAME = "pipelines"
DESCRIPTION = """Data engineering pipelines for the Moovio Plus Product.
Moovio is a fictional product created by the Databricks Curriculum Team to
support teaching data engineering.
"""
REPO = "https://github.com/databricks-academy/databricks-project/"
URL = REPO + "tree/published/example/databricks_job_with_wheel/src/pipelines"
AUTHOR = "@joshuacook"
REQUIRES_PYTHON = ">=3.6.0"
VERSION = "0.1.0"

REQUIRED = ["pyspark"]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    packages=find_packages(),
    python_requires=REQUIRES_PYTHON,
    url=URL,
    author=AUTHOR,
)
