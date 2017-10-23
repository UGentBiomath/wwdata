# -*- coding: utf-8 -*-

"""Top-level package for wwdata."""

__author__ = """Chaim De Mulder"""
__email__ = 'demulderchaim@gmail.com'
__version__ = '0.1.0'


from .Class_HydroData import HydroData
from .Class_LabExperimBased import LabExperimBased
print('LabExperimBased imported')
from .Class_LabSensorBased import LabSensorBased
print('LabSensorBased imported')
from .Class_OnlineSensorBased import OnlineSensorBased
print('OnlineSensorBased imported')

from .time_conversion_functions import *
from .data_reading_functions import *
