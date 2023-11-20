from setuptools import setup, find_packages

with open('requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()

with open('README.md', encoding='utf-8') as file:
    long_description = file.read()

setup(
    name='self_ood',
    version='0.0.1',
    description='Self-supervised OOD.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mishgon/self_ood',
    packages=find_packages(include=('self_ood',)),
    python_requires='>=3.6',
    install_requires=requirements,
)
