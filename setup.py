from setuptools import setup, find_packages

setup(
    name='regression-language-model',
    version='0.1.0',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'torch', 'numpy', 'pandas', 'scikit-learn', 'streamlit', 'fastapi', 'uvicorn', 'plotly'
    ],
)
