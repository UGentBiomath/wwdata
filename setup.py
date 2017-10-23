#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'pandas',
    'numpy',
    'dateutils',
    'scipy',
    'matplotlib',
    'statsmodels',
    'xlrd',
    #'tkinter'
]

setup_requirements = [
    # TODO(cdemulde): put setup requirements (distutils extensions, etc.) here
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='wwdata',
    version='0.1.0',
    description="Data analysis package aimed at data obtained in the context of (waste)water",
    long_description=readme + '\n\n' + history,
    author="Chaim De Mulder",
    author_email='demulderchaim@gmail.com',
    url='https://github.com/cdemulde/wwdata',
    packages=find_packages(include=['wwdata']),
    include_package_data=True,
    install_requires=requirements,
    license="GNU General Public License v3",
    zip_safe=False,
    keywords='wwdata',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=requirements,
    setup_requires=setup_requirements,
)
