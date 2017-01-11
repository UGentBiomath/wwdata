#book# -*- coding: utf-8 -*-
"""
    Class_LabSensorBased provides functionalities for data handling of data obtained in lab experiments with online sensors in the field of (waste)water treatment.
    Copyright (C) 2016 Chaim De Mulder

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
@authors: chaimdemulder, stijnvanhoey
contact: chaim.demulder@ugent.be
"""
import sys
import os
from os import listdir
import pandas as pd
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt   #plotten in python
import datetime as dt

from wwdata.Class_HydroData import HydroData
from wwdata.data_reading_functions import *
from wwdata.time_conversion_functions import *

class LabSensorBased(HydroData):
    """
    Superclass for a HydroData object, expanding the functionalities with 
    specific functions for data gathered is lab experiments
    
    Attributes
    ----------
    (see HydroData docstring)
    
    """
    
    def __init__(self,data,experiment_tag='None'):
        """
        initialisation of a LabSensorBased object, based on a previously defined
        HydroData object. 
        
        Parameters
        ----------
        (currently no additional data needed to the HydroData object creation)
        """
        HydroData.__init__(self,data,timedata_column,experiment_tag='No tag given')
    
    def drop_peaks(self,data_name,cutoff,inplace=True,log_file=None):
        """
        Filters out the peaks larger than a cut-off value in a dataseries
        
        Parameters
        ----------
        data_name : str
            the name of the column to use for the removal of peak values
        cutoff : int
            cut off value to use for the removing of peaks; values with an 
            absolute value larger than this cut off will be removed from the data
        inplace : bool
            indicates whether a new dataframe is created and returned or whether
            the operations are executed on the existing dataframe (nothing is
            returned)
        log_file : str
            string containing the directory to a log file to be written out 
            when using this function
            
        Returns
        -------
        LabSensorBased object (if inplace=False)
            the dataframe from which the double values of 'data' are removed
        None (if inplace=True)
        """
        original = len(self.data)
        if inplace == False:
            data = self.data.copy()
            data.drop(data[abs(data[data_name]) > cutoff].index,inplace=True)
            data.reset_index(drop=True,inplace=True)
            new = len(data)
            if log_file == None:
                _print_removed_output(original,new)
            elif type(log_file) == str:
                _log_removed_output(log_file,original,new)
            else :
                raise TypeError('Please provide the location of the log file as \
                                a string type, or leave the argument if no log \
                                file is needed.')                
            
            return self.__class__(data,data.columns)
            
        elif inplace == True:
            self.drop(self.data[abs(self.data[data_name]) > cutoff].index,
                                inplace=True)
            self.data.reset_index(drop=True,inplace=True)
            new = len(self.data)
            if log_file == None:
                _print_removed_ouiput(original,new)
            elif type(log_file) == str:
                _log_removed_output(log_file,original,new)
            else :
                raise TypeError('Please provide the location of the log file as \
                                a string type, or leave the argument if no log \
                                file is needed.') 
    
    def _select_slope(self,ydata,down=True,limit=0):#,based_on_max=True):#,bounds=[1,1]):
    
        #TO BE ADJUSTED BASED ON ALL FUNCTIONS FILE!
        """
        Selects down- or upward sloping data from a given dataseries, based on 
        the maximum in the dataseries. This requires only one maximum to be 
        present in the dataset.
        
        Parameters
        ----------
        ydata : str
            name of the column containing the data for which slopes, either up 
            or down, need to be selected
        down : bool
            if True, the downwards slopes are selected, if False, the upward 
            slopes
        based_on_max : bool
            if True, the data is selected based on the maximum of the data, if
            false it is based on the minimum
        bounds : array
            array containing two integer values, indicating the extra margin of 
            values that needs to be dropped from the dataset to avoid selecting 
            irregular data (e.g. not straightened out after reaching of maximum)
        
        Returns
        -------    
        LabSensorBased object:
            a dataframe from which the non-down or -upward sloping data are dropped
        """
        #if based_on_max == True:
        drop_index = self.data[ydata].idxmax()
        if down == True:
            try:
                print('Selecting downward slope:',drop_index,\
                'datapoints dropped,',len(self.data)-drop_index,\
                'datapoints left.')
            
                self.data = self.data[drop_index:]
                self.data.reset_index(drop=True,inplace=True)
                return self.__class__(self.data,self.columns)
            except:#IndexError:
                print('Not enough datapoints left for selection')
    
        elif down == False:
            try:
                print('Selecting upward slope:',len(self.data)-drop_index,\
                'datapoints dropped,',drop_index,'datapoints left.')
            
                self.data = self.data[:drop_index]
                self.data.reset_index(drop=True,inplace=True)
                return self.__class__(self.data,self.columns)
            except:#IndexError:
                print('Not enough datapoints left for selection')
        
    #    elif based_on_max == False:
    #        drop_index = dataframe[ydata].idxmin()
    #        if down == True:
    #            try:
    #                print 'Selecting downward slope:',drop_index+sum(bounds),\
    #                'datapoints dropped,',len(dataframe)-drop_index-sum(bounds),\
    #                'datapoints left.'
    #                
    #                dataframe = dataframe[bounds[0]:drop_index-bounds[1]]
    #                dataframe.reset_index(drop=True,inplace=True)
    #                return dataframe
    #            except IndexError:
    #                print 'Not enough datapoints left for selection'
    #    
    #        elif down == False:
    #            try:
    #                print 'Selecting upward slope:',len(dataframe)-drop_index+sum(bounds),\
    #                'datapoints dropped,',drop_index-sum(bounds),'datapoints left.'
    #                
    #                dataframe = dataframe[drop_index+bounds[0]:-bounds[1]]
    #                dataframe.reset_index(drop=True,inplace=True)
    #                return dataframe
    #            except IndexError:
    #                print 'Not enough datapoints left for selection'
    #   
    
    
    
            
##############################
###   NON-CLASS FUNCTIONS  ###
##############################

def _print_removed_output(original,new,type_):
    """
    function printing the output of functions that remove datapoints. 
    
    Parameters
    ----------
    original : int
        original length of the dataset
    new : int
        length of the new dataset
    type_ : str
        'removed' or 'dropped'

    """
    print('Original dataset:',original,'datapoints')
    print('New dataset:',new,'datapoints')
    print(original-new,'datapoints ',type_)

def _log_removed_output(log_file,original,new,type_):
    """
    function writing the output of functions that remove datapoints to a log file. 
    
    Parameters
    ----------
    log_file : str
        string containing the directory to the log file to be written out
    original : int
        original length of the dataset
    new : int
        length of the new dataset
    type_ : str
        'removed' or 'dropped'
    """   
    log_file = open(log_file,'a')
    log_file.write(str('\nOriginal dataset: '+str(original)+' datapoints; new dataset: '+
                    str(new)+' datapoints'+str(original-new)+' datapoints ',type_))
    log_file.close()
                
