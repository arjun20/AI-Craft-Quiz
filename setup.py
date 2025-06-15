from setuptools import find_packages,setup

setup(
    name='mcqgenrator',
    version='0.0.1',
    author='Arjun',
    author_email='arjuncoding@gmail.com',
    install_requires=["openai","langchain","streamlit","python-dotenv","PyPDF2"],
    # packages=find_packages()
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)

