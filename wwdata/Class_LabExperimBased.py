# -*- coding: utf-8 -*-
"""
Class_LabExperimBased provides functionalities for data handling of data obtained in lab experiments in the field of (waste)water treatment.
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
along with this program.  If not, see http://www.gnu.org/licenses/.
"""

import sys
#import os
#from os import listdir
#import pandas as pd
#import scipy as sp
#import numpy as np
#import datetime as dt
import matplotlib.pyplot as plt   #plotten in python
import warnings as wn

from wwdata.Class_HydroData import HydroData

class LabExperimBased(HydroData):
    """
    Superclass for a HydroData object, expanding the functionalities with
    specific functions for data gathered is lab experiments.

    Attributes
    ----------
    timedata_column : str
        name of the column containing the time data
    data_type : str
        type of the data provided
    experiment_tag : str
        A tag identifying the experiment; can be a date or a code used by
        the producer/owner of the data.
    time_unit : str
        The time unit in which the time data is given
    units : array
        The units of the variables in the columns
    """

    def __init__(self,data,timedata_column='index',data_type='NAT',
                 experiment_tag='No tag given',time_unit=None):
        """
        initialisation of a LabExperimBased object, based on a previously defined
        HydroData object.
        """
        HydroData.__init__(self,data,timedata_column=timedata_column,data_type=data_type,
                           experiment_tag=experiment_tag,time_unit=time_unit)


    def hours(self,time_column='index'):
        """
        calculates the hours from the relative values

        Parameters
        ----------
        time_column : string
            column containing the relative time values; default to index
        """
        if time_column == 'index':
            self.data['index']=self.time.values
            self.data['h']= (self.data['indexes'])*24 + self.data['indexes'].shift(1)
            self.data['h'].fillna(0,inplace=True)
            self.data.drop('index', axis=1, inplace=True)
        else:
            self.data['h']= (self.data[time_column])*24 + self.data[time_column].shift(1)
            self.data['h'].fillna(0,inplace=True)


    def add_conc(self,column_name,x,y,new_name='default'):
        """
        calculates the concentration values of the given column and adds them as
        a new column to the DataFrame.

        Parameters
        ----------
        column_name : str
            column with values
        x : int
            ...
        y : int
            ...
        new_name : str
            name of the new column, default to 'column_name + mg/L'
        """
        if new_name == 'default':
            new_name = column_name + ' ' + 'mg/L'

        self.data[new_name] = self.data[column_name].values*x*y

    ## Instead of this function: define a dataframe/dict with conversion or
    ## concentration factors, so that you can have a function that automatically
    ## converts all parameters in the frame to concentrations

    def check_ph(self,ph_column='pH',thresh=0.4):
        """
        gives the maximal change in pH

        Parameters
        ----------
        ph_column : str
            column with pH-values, default to 'pH'
        threshold : int
            threshold value for warning, default to '0.4'
        """
        dph = self.data[ph_column].max()-self.data[ph_column].min()
        if dph > thresh:
            wn.warn('Strong change in pH during experiment!')
        else:
            self.delta_ph = dph

    def in_out(self,columns):
        """
        (start_values-end_values)

        Parameters
        ----------
        columns : array of strings
        """
        inv=0
        outv=0
        indexes= self.time.values
        for column in columns:
            inv += self.data[column][indexes[0]]
        for column in columns:
            outv += self.data[column][indexes[-1]]
        in_out = inv-outv

        return in_out


    def removal(self,columns):
        """
        total removal of nitrogen
        (1-(end_values/start_values))

        Parameters
        ----------
        columns : array of strings
        """
        inv=0
        outv=0
        indexes= self.time.values
        for column in columns:
            inv += self.data[column][indexes[0]]
        for column in columns:
            outv += self.data[column][indexes[-1]]
        removal = 1-(outv/inv)

        return removal

    def calc_slope(self,columns,time_column='h'):
        """
        calculates the slope of the selected columns

        Parameters
        ----------
        columns : array of strings
            columns to calculate the slope for
        time_column : str
            time used for calculation; default to 'h'

        """
        for column in columns:
            self.data[column + " " +'slope'] = (self.data[column].shift(1)-self.data[column])\
            /(self.data[time_column]-self.data[time_column].shift(1))

    def plot(self,columns,time_column='index'):
        """
        calculates the slope of the selected columns

        Parameters
        ----------
        columns : array of strings
            columns to plot
        time_column : str
            time used for calculation; default to 'h'
        """
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        if time_column=='index':
            for column in columns:
                ax.plot(self.time,self.data[column],marker='o')
        else:
            for column in columns:
                ax.plot(self.data[time_column],self.data[column],marker='o')
        ax.legend()

        return fig,ax

#######################################

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
