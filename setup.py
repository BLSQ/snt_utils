from setuptools import setup, find_packages

setup(
    name="snt_lib",
    version="0.45.0",
    packages=find_packages(),  # Automatically finds snt_lib
    install_requires=[
        "openhexa.sdk>=2.8.0",
        "papermill>=2.6.0",
        "pandas>=2.2.3",
        "sqlalchemy==2.0.39",
        "pyarrow>=18.1.0",
        "psycopg2-binary==2.9.10",
        "requests>=2.32.3",
        "nbclient>=0.10.2",
        "GitPython>=3.1.44",
    ],
)
