from setuptools import setup, find_packages


__version__ = "0.0.0"

REPO_NAME = "Chicken_Disease_Classification"
AUTHOR_NAME = "Paras Jain"
SRC_REPO = "CNN_Classifier"
AUTHOR_EMAIL = "Paras.Jain@eclerx.com"

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_NAME,
    author_email=AUTHOR_EMAIL,
    description="A Python Package for CNN Application",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/pjain809/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/pjain809/{REPO_NAME}/issues"
    },
    package_dir={"": "src"},
    packages=find_packages(where="src")
)
