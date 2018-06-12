# -*- coding: utf-8 -*-

"""Top-level package for wwdata."""

__author__ = """Chaim De Mulder"""
__email__ = 'demulderchaim@gmail.com'
__version__ = '0.2.0'


from .Class_HydroData import HydroData
from .Class_LabExperimBased import LabExperimBased
from .Class_LabSensorBased import LabSensorBased
from .Class_OnlineSensorBased import OnlineSensorBased

from .time_conversion_functions import *
from .data_reading_functions import *
