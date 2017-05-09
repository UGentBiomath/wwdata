#!/usr/bin/env python

from setuptools import setup

setup(name='wwdata',
	  version='0.1',
	  description='Analysis of datasets obtained in the (waste)water sector',
	  url='https://cdmulde.github.io/wwdata/',
 	  author='Chaim De Mulder',
	  author_email='demulderchaim@gmail.com',
	  license='GNU AGPL',
	  classifiers=[
		'Intended audience :: water professionals',
		'Topic :: data analysis'],
	  packages=['wwdata'],
	  keywords='wastewater data analysis',
	  install_requires=['pandas','numpy','dateutils','scipy','matplotlib','statsmodels'])
