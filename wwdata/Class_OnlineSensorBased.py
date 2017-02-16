"""
    Class_OnlineSensorBased provides functionalities for data handling of data obtained with online sensors in the field of (waste)water treatment.
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

#import sys
#import os
#from os import listdir
import pandas as pd
#import scipy as sp
import numpy as np
import matplotlib.pyplot as plt   #plotten in python
import datetime as dt
import warnings as wn

from wwdata.Class_HydroData import HydroData
#from data_reading_functions import _print_removed_output,_log_removed_output
#from time_conversion_functions import *

class OnlineSensorBased(HydroData):
    """
    Superclass for a HydroData object, expanding the functionalities with 
    specific functions for data gathered at full scale by continous measurements
    
    Attributes
    ----------
    (see HydroData docstring)
    
    """
    def __init__(self,data,timedata_column='index',data_type='WWTP',
                 experiment_tag='No tag given',time_unit=None):
        """
        initialisation of a FullScaleSensorBased object, based on a previously defined
        HydroData object. 
        
        Parameters
        ----------
        (currently no additional data needed to the HydroData object creation)
        """
        HydroData.__init__(self,data=data,timedata_column=timedata_column,
                           data_type=data_type,experiment_tag=experiment_tag,
                           time_unit=time_unit)
        self.filled = pd.DataFrame(index=self.index())
        self.meta_filled = pd.DataFrame(self.meta_valid,index=self.data.index)
    
    def time_to_index(self,drop=True,inplace=True,verify_integrity=False):
        """CONFIRMED
        using pandas set_index function to set the columns with timevalues
        as index"""
        # Drop second layer of indexing to make dataframe handlable
        # self.data.columns = self.data.columns.get_level_values(0)
        
        if self.timename == 'index':
            raise IndexError('There already is a timeseries in the dataframe index!')
        if isinstance(self.time[0],str):
            raise ValueError('Time values of type "str" can not be used as index')
            
        if inplace == False:
            new_data = self.set_index(self.timename,drop=drop,inplace=False,
                                      verify_integrity=verify_integrity)
            #self.columns = np.array(new_data.columns)
            return self.__class__(new_data,timedata_column='index',
                                  data_type=self.data_type,experiment_tag=self.tag,
                                  time_unit=self.time_unit)
        elif inplace == True:
            self.set_index(self.timename,drop=drop,inplace=True,
                           verify_integrity=verify_integrity)
            #self.columns = np.array(self.data.columns)
            #self.timename = 'index'
            #self.time = self.index()
    
    def calc_total_proportional(self,Q_tot,Q,conc,new_name='new',unit='mg/l',
                                filled=False):
        """CONFIRMED
        Calculates the total concentration of an incoming flow, based on the 
        given total flow and the separate incoming flows and concentrations
        
        Parameters
        ----------
        Q_tot : str
            name of the column containing the total flow
        Q : array of str
            names of the columns containing the separate flows
        conc : array of str
            names of the columns containing the separate concentration values
        new_name : str
            name of the column to be added
        filled : bool
            if true, use self.filled to calculate proportions from
        !!Order of columns in Q and conc must match!!
        
        Returns
        -------
        None;
        creates a hydropy object with added column for the proportional concentration
        """
        if filled:
            index = self.filled.index
            sum_ = pd.Series(0, index=index)
            for i in range(0,len(Q)):
                sum_ = sum_ + self.filled[Q[i]] * self.filled[conc[i]]
            self.filled[new_name] = sum_ / self.filled[Q_tot]
            
        else:
            index = self.index()
            sum_ = pd.Series(0, index=index)
            for i in range(0,len(Q)):
                sum_ = sum_ + self.data[Q[i]] * self.data[conc[i]]
            
            self.data[new_name] = sum_ / self.data[Q_tot]
            self.columns = np.array(self.data.columns)

        try:
            self.units = pd.concat([self.units,
                                    pd.DataFrame([[new_name,unit]],columns=self.units.columns)],
                                    ignore_index=True)                
        except:
	    wn.warn('Something might have gone wrong with the updating of the units. '+
                    'Check self.units to make sure everything is still okay.')

	return None
    
    def calc_daily_average(self,column_name,time='float',plot=False):
        """
        calculates the daily average of values in the given column and returns them as a 2D-array,
        containing the days and the average values on the respective days. Plotting is possible.
    
        Parameters
        ----------
        column_name : str
            name of the column containing the data to calculate the average values for
        time : str
            type of the time indication. Available: "float"
        plot : bool
            plot or not
    
        Returns
        -------
        pd.Dataframe :
            pandas dataframe, containing the daily means with standard deviations
            for the selected column
        """
        self.daily_average = {}             
        series = self.data[column_name]   
        
        if isinstance(series.index[0],float):
            days = np.arange(series.index[0],series.index[-1],1)
            means = [series[x:x+1].mean() for x in days]
            stds = [series[x:x+1].std() for x in days]
            
            to_return = pd.DataFrame([days,means,stds]).transpose()
            to_return.columns = ['day','mean','std']
 
        elif isinstance(self.data.index[0],pd.tslib.Timestamp):
            means = series.resample('d').mean()
            stds = series.resample('d').std()
            
            to_return = pd.DataFrame([means.index,means.values,stds.values]).transpose()
            to_return.columns = ['day','mean','std']
 
        if plot==True:
            fig = plt.figure(figsize=(16,6))
            ax = fig.add_subplot(111)
            ax.errorbar(to_return['day'],to_return['mean'],yerr=to_return['std'],fmt='o')
            #ax.plot(to_return['day'],(to_return['mean']+to_return['std']),'b',alpha=0.5)
            #ax.plot(to_return['day'],(to_return['mean']-to_return['std']),'b',alpha=0.5)
            #ax.fill_between(to_return['day'],to_return['mean'],(to_return['mean']+to_return['std']),
            #                color='grey', alpha='0.3')
            #ax.fill_between(to_return['day'],to_return['mean'],(to_return['mean']-to_return['std']),
            #                color='grey', alpha='0.3')
            ax.set_title(column_name)
        
        self.daily_average[column_name] = to_return
        
#==============================================================================
# FILLING FUNCTIONS
#==============================================================================
    def _reset_meta_filled(self,data_name=None):
        """
        reset the meta dataframe, possibly for only a certain data series, 
        should wrong labels have been assigned at some point
        """
        if data_name == None:
            self.meta_filled = pd.DataFrame(self.meta_valid,index=self.data.index)
        else:
            try:
                self.meta_filled[data_name] = self.meta_valid[data_name]
            except:
                pass
                #wn.warn(data_name + ' is not contained in self.meta_valid yet, so cannot\
                #be removed from it!')
               
    def add_to_filled(self,column_names):
        """
        columns_names : array
        """
        self._plot = 'filled'
        # Create/adjust self.filled
        self.filled = self.filled.reindex(self.index())
        for column in column_names:
            if not column in self.filled.columns:
                self.filled[column] = self.data[column]   
            else:
                pass                
                #wn.warn('self.filled already contains a column named ' + 
                #    column + '. The original columns was kept.')
        
    def fill_missing_interpolation(self,to_fill,range_,arange,method='index',plot=False,
                                   clear=False):
        """
        Fills the missing values in a dataset (to_fill), based specified 
        interpolation algorithm (method). This happens only if the number of 
        consecutive missing values is smaller than range_.
        
        Parameters
        ----------
        to_fill : str
            name of the column containing the data to be filled
        range_ : int
            the maximum range that the absence of values can be to still
            allow interpolation to fill in values
        method : str
            interpolation method to be used by the .interpolate function. See
            pandas docstrings for more info
        plot : bool
            whether or not to plot the new dataset
        clear : bool
            whether or not to clear the previoulsy filled values and start from
            the self.meta_valid dataset again for this particular dataseries. 
            
        Returns
        -------
        None;
        creates/updates self.filled, containing the adjusted dataset and updates
        meta_filled with the correct labels.
        """
        ###
        # CHECKS
        ###
        self._plot = 'filled'
        wn.warn('When making use of filling functions, please make sure to '+
        'start filling small gaps and progressively move to larger gaps. This '+
        'ensures the proper working of the package algorithms.')
        if clear:
            self._reset_meta_filled(to_fill)
        self.meta_filled = self.meta_filled.reindex(self.index(),fill_value='!!')
        
        if not to_fill in self.meta_filled.columns:
            # if the to_fill column doesn't exist yet in the meta_filled dataset,
            # add it, and fill it with the meta_valid values; if this last one
            # doesn't exist yet, create it with 'original' tags.
            try:
                self.meta_filled[to_fill] = self.meta_valid[to_fill]
            except:
                self.add_to_meta_valid([to_fill])
                self.meta_filled[to_fill] = self.meta_valid[to_fill]
        else:
            # where the meta_filled dataset contains original values, update with
            # the values from meta_valid; in case a filling round was done before
            # any filtering; not supposed to happen, but cases exist.
            try:
                self.meta_filled[to_fill].loc[self.meta_filled[to_fill]=='original'] = \
                self.meta_valid[to_fill].loc[self.meta_filled[to_fill]=='original']
            except:
                self.add_to_meta_valid([to_fill])
                self.meta_filled[to_fill].loc[self.meta_filled[to_fill]=='original'] = \
                self.meta_valid[to_fill].loc[self.meta_filled[to_fill]=='original']
        if not to_fill in self.filled:
            self.add_to_filled([to_fill])
        
        # Give warning when replacing data from rain events and at the same time
        # check if arange has the right type
        try:
            rain = (self.data_type == 'WWTP') and \
                   (self.highs['highs'].loc[arange[0]:arange[1]].sum() > 1)
        except TypeError:
            raise TypeError("Slicing not possible for index type " + \
            str(type(self.data.index[0])) + " and arange argument type " + \
            str(type(arange[0])) + ". Try changing the type of the arange " + \
            "values to one compatible with " + str(type(self.data.index[0])) + \
            " slicing.")    

        if rain :
            wn.warn('Data points obtained during a rain event will be replaced. \
            Make sure you are confident in this replacement method for the \
            filling of gaps in the data during rain events.')
        
        ###
        # CALCULATIONS
        ###
        # Create a mask to replace the filtered datapoints with nan-values, 
        # if consecutive occurence lower than range_
        mask_df = pd.DataFrame(index = self.meta_valid[arange[0]:arange[1]].index)
        mask_df['count'] = (self.meta_valid[to_fill][arange[0]:arange[1]] != self.meta_valid[to_fill][arange[0]:arange[1]].\
                            shift()).astype(int).cumsum().astype(str)
        group = mask_df.groupby('count').size()
        group.index = mask_df.groupby('count').size().index.astype(str)
        
        # Compare the values in 'count' with the ones in the group-by object.
        # mask_df now contains the amount of consecutive true or false datapoints,
        # for every datapoint
        replace_dict = {'count':dict(group)}
        mask_df = mask_df.replace(replace_dict)
        
        # Based on the mask and whether a datapoint is filtered, replace with
        # nan values
        filtered_based = pd.DataFrame(self.meta_filled.loc[self.meta_filled[to_fill] == 'filtered'].index.values)
        mask_based = pd.DataFrame(mask_df.loc[mask_df['count'] < range_].index.values)
        indexes_to_replace = pd.merge(filtered_based,mask_based,how='inner')
        self.filled[to_fill] = self.filled[to_fill].drop(indexes_to_replace[0])
        
        ###
        # FILLING
        ###
        # Use the .interpolate() method to interpolate for nan values
        self.filled[to_fill] = self.filled[to_fill].interpolate(method=method)
        #                                                            limit_direction='both')
        
        # Adjust in the self.meta_valid dataframe
        self.meta_filled.loc[indexes_to_replace[0],to_fill] = 'filled_interpol'
        
        if plot:
            self.plot_analysed(to_fill)
        
        return None
  
    def fill_missing_ratio(self,to_fill,to_use,ratio,arange,
                             filtered_only=True,plot=False,clear=False):#,use_smoothing=True):
        """
        Fills the missing values in a dataset (to_fill), based on the ratio this 
        data shows when comparing to other data (to_use). This happens within 
        the range given by arange.
        
        Parameters
        ----------
        to_fill : str
            name of the column with data to fill
        to_use : str
            name of the column to use, in combination with the given ratio, to 
            fill in some of the missing data
        ratio : float
            ratio to multiply the to_use data with to obtain data for filling in 
            in the to_fill data column
        arange : array of two values
            the range within which missing/filtered values need to be replaced
        filtered_only : boolean
            if True, fills only the datapoints labeled as filtered. If False, 
            fills/replaces all datapoints in the given range
        plot : bool
            whether or not to plot the new dataset
        clear : bool
            whether or not to clear the previoulsy filled values and start from
            the self.meta_valid dataset again for this particular dataseries.            
        
        Returns
        -------
        None;
        creates/updates self.filled, containing the adjusted dataset and updates
        meta_filled with the correct labels.
        """
        ###
        # CHECKS
        ###
        self._plot = 'filled'
        wn.warn('When making use of filling functions, please make sure to '+
        'start filling small gaps and progressively move to larger gaps. This '+
        'ensures the proper working of the package algorithms.')
        if clear:
            self._reset_meta_filled(to_fill)
        self.meta_filled = self.meta_filled.reindex(self.index(),fill_value='!!')
        
        if not to_fill in self.meta_filled.columns:
            # if the to_fill column doesn't exist yet in the meta_filled dataset,
            # add it, and fill it with the meta_valid values; if this last one
            # doesn't exist yet, create it with 'original' tags.
            try:
                self.meta_filled[to_fill] = self.meta_valid[to_fill]
            except:
                self.add_to_meta_valid([to_fill])
                self.meta_filled[to_fill] = self.meta_valid[to_fill]
        else:
            # where the meta_filled dataset contains original values, update with
            # the values from meta_valid; in case a filling round was done before
            # any filtering; not supposed to happen, but cases exist.
            try:
                self.meta_filled[to_fill].loc[self.meta_filled[to_fill]=='original'] = \
                self.meta_valid[to_fill].loc[self.meta_filled[to_fill]=='original']
            except:
                self.add_to_meta_valid([to_fill])
                self.meta_filled[to_fill].loc[self.meta_filled[to_fill]=='original'] = \
                self.meta_valid[to_fill].loc[self.meta_filled[to_fill]=='original']
                
        if not to_fill in self.filled:
            self.add_to_filled([to_fill])

        # Give warning when replacing data from rain events and at the same time
        # check if arange has the right type
        try:
            rain = (self.data_type == 'WWTP') and \
                   (self.highs['highs'].loc[arange[0]:arange[1]].sum() > 1)
        except TypeError:
            raise TypeError("Slicing not possible for index type " + \
            str(type(self.data.index[0])) + " and arange argument type " + \
            str(type(arange[0])) + ". Try changing the type of the arange " + \
            "values to one compatible with " + str(type(self.data.index[0])) + \
            " slicing.")    

        if rain :
            wn.warn('Data points obtained during a rain event will be replaced. \
            Make sure you are confident in this replacement method for the \
            filling of gaps in the data during rain events.')
         
        ###
        # FILLING
        ###            
        if filtered_only:
            indexes_to_replace = pd.DataFrame(self.meta_valid.\
                                    loc[arange[0]:arange[1]].\
                                    loc[self.meta_filled[to_fill] == 'filtered'].index.values)
            self.filled.loc[indexes_to_replace[0],to_fill] = self.data.loc[indexes_to_replace[0],to_use]*ratio
            # Adjust in the self.meta_filled dataframe
            self.meta_filled.loc[indexes_to_replace[0],to_fill] = 'filled_ratio'
            
        if not filtered_only:
            self.filled.loc[arange[0]:arange[1],to_fill] = self.data.loc[arange[0]:arange[1],to_use]*ratio
            # Adjust in the self.meta_valid dataframe
            self.meta_filled[to_fill].loc[arange[0]:arange[1]] = 'filled_ratio'

        if plot:
            self.plot_analysed(to_fill)        
        
        return None

    def fill_missing_correlation(self,to_fill,to_use,arange,corr_range,
                                 zero_intercept=False,filtered_only=True,
                                 plot=False,clear=False):
        """
        Fills the missing values in a dataset (to_fill), based on the correlation
        this data shows when comparing to other data (to_use). This happens within 
        the range given by arange.
        
        Parameters
        ----------
        to_fill : str
            name of the column with data to fill
        to_use : str
            name of the column to use, in combination with the given ratio, to 
            fill in some of the missing data
        arange : array of two values
            the range within which missing/filtered values need to be replaced
        corr_range : array of two values
            the range to use for the calculation of the correlation
        filtered_only : boolean
            if True, fills only the datapoints labeled as filtered. If False, 
            fills/replaces all datapoints in the given range
        plot : bool
            whether or not to plot the new dataset
        clear : bool
            whether or not to clear the previoulsy filled values and start from
            the self.meta_valid dataset again for this particular dataseries.            
        
        Returns
        -------
        None;
        creates/updates self.filled, containing the adjusted dataset and updates
        meta_filled with the correct labels.
        """
        ###
        # CHECKS
        ###
        self._plot = 'filled'
        wn.warn('When making use of filling functions, please make sure to '+
        'start filling small gaps and progressively move to larger gaps. This '+
        'ensures the proper working of the package algorithms.')
        if clear:
            self._reset_meta_filled(to_fill)
        self.meta_filled = self.meta_filled.reindex(self.index(),fill_value='!!')
        
        if not to_fill in self.meta_filled.columns:
            # if the to_fill column doesn't exist yet in the meta_filled dataset,
            # add it, and fill it with the meta_valid values; if this last one
            # doesn't exist yet, create it with 'original' tags.
            try:
                self.meta_filled[to_fill] = self.meta_valid[to_fill]
            except:
                self.add_to_meta_valid([to_fill])
                self.meta_filled[to_fill] = self.meta_valid[to_fill]
        else:
            # where the meta_filled dataset contains original values, update with
            # the values from meta_valid; in case a filling round was done before
            # any filtering; not supposed to happen, but cases exist.
            try:
                self.meta_filled[to_fill].loc[self.meta_filled[to_fill]=='original'] = \
                self.meta_valid[to_fill].loc[self.meta_filled[to_fill]=='original']
            except:
                self.add_to_meta_valid([to_fill])
                self.meta_filled[to_fill].loc[self.meta_filled[to_fill]=='original'] = \
                self.meta_valid[to_fill].loc[self.meta_filled[to_fill]=='original']
                
        if not to_fill in self.filled:
            self.add_to_filled([to_fill])
       
        # Give warning when replacing data from rain events and at the same time
        # check if arange has the right type
        try:
            rain = (self.data_type == 'WWTP') and \
                   (self.highs['highs'].loc[arange[0]:arange[1]].sum() > 1)
        except TypeError:
            raise TypeError("Slicing not possible for index type " + \
            str(type(self.data.index[0])) + " and arange argument type " + \
            str(type(arange[0])) + ". Try changing the type of the arange " + \
            "values to one compatible with " + str(type(self.data.index[0])) + \
            " slicing.")    

        if rain :
            wn.warn('Data points obtained during a rain event will be replaced.' + \
            ' Make sure you are confident in this replacement method for the' + \
            ' filling of gaps in the data during rain events.')
        
        ###
        # CALCULATIONS
        ###
        slope,intercept,r_sq = self.get_correlation(to_use,to_fill,corr_range,
                                                    zero_intercept=zero_intercept)
        if intercept < 0:
                wn.warn('The intercept was calculated to be lower than '+
                '0, which might lead to negative data values when data is replaced '+
                'based on this correlation. Try setting "zero_intercept" to True '+
                'to avoid.')                                            
        ###
        # FILLING
        ###
        if filtered_only:
            indexes_to_replace = pd.DataFrame(self.meta_valid.\
                                            loc[arange[0]:arange[1]].\
                                            loc[self.meta_filled[to_fill] == 'filtered'].index.values)
            self.filled.loc[indexes_to_replace[0],to_fill] = \
                            self.data.loc[indexes_to_replace[0],to_use]*slope + intercept
            # Adjust in the self.meta_filled dataframe
            self.meta_filled.loc[indexes_to_replace[0],to_fill] = 'filled_correlation'
            
        if not filtered_only:
            self.filled.loc[arange[0]:arange[1],to_fill] = \
                            self.data.loc[arange[0]:arange[1],to_use]*slope + intercept
            # Adjust in the self.meta_filled dataframe
            self.meta_filled[to_fill].loc[arange[0]:arange[1]] = 'filled_correlation'
        
        if plot:
            self.plot_analysed(to_fill)
            
        return None   
        
    def fill_missing_standard(self,to_fill,arange,filtered_only=True,plot=False,
                              clear=False):
        """
        Fills the missing values in a dataset (to_fill), based on the average
        daily profile calculated by calc_daily_profile(). This happens within 
        the range given by arange.
        
        Parameters
        ----------
        to_fill : str
            name of the column with data to fill
        arange : array of two values
            the range within which missing/filtered values need to be replaced
        filtered_only : boolean
            if True, fills only the datapoints labeled as filtered. If False, 
            fills/replaces all datapoints in the given range
        plot : bool
            whether or not to plot the new dataset
        clear : bool
            whether or not to clear the previoulsy filled values and start from
            the self.meta_valid dataset again for this particular dataseries.
            
        Returns
        -------
        None;
        creates/updates self.filled, containing the adjusted dataset and updates
        meta_filled with the correct labels.
        """
        ###
        # CHECKS
        ###
        self._plot = 'filled'
        wn.warn('When making use of filling functions, please make sure to '+
        'start filling small gaps and progressively move to larger gaps. This '+
        'ensures the proper working of the package algorithms.')
            
        # several checks on availability of the right columns in the necessary
        # dataframes/dictionaries
        if clear:
            self._reset_meta_filled(to_fill)
        self.meta_filled = self.meta_filled.reindex(self.index(),fill_value='!!')
        
        if not to_fill in self.meta_filled.columns:
            # if the to_fill column doesn't exist yet in the meta_filled dataset,
            # add it, and fill it with the meta_valid values; if this last one
            # doesn't exist yet, create it with 'original' tags.
            try:
                self.meta_filled[to_fill] = self.meta_valid[to_fill]
            except:
                self.add_to_meta_valid([to_fill])
                self.meta_filled[to_fill] = self.meta_valid[to_fill]
        else:
            # where the meta_filled dataset contains original values, update with
            # the values from meta_valid; in case a filling round was done before
            # any filtering; not supposed to happen, but cases exist.
            try:
                self.meta_filled[to_fill].loc[self.meta_filled[to_fill]=='original'] = \
                self.meta_valid[to_fill].loc[self.meta_filled[to_fill]=='original']
            except:
                self.add_to_meta_valid([to_fill])
                self.meta_filled[to_fill].loc[self.meta_filled[to_fill]=='original'] = \
                self.meta_valid[to_fill].loc[self.meta_filled[to_fill]=='original']
                
        if not to_fill in self.filled:
            self.add_to_filled([to_fill])
                
        try:
            if not isinstance(self.daily_profile,dict):
                raise TypeError("self.daily_profile should be a dictionary Type. \
                Run calc_daily_profile() to get an average daily profile for " + to_fill)
        except AttributeError:
            raise AttributeError("self.daily_profile doesn't exist yet, meaning "+
            "there is not data available to replace other data with. Run "+
            "calc_daily_profile() to get an average daily profile for " + to_fill)
  
        # Give warning when replacing data from rain events and at the same time
        # check if arange has the right type
        try:
            rain = (self.data_type == 'WWTP') and \
                   (self.highs['highs'].loc[arange[0]:arange[1]].sum() > 1)
        except TypeError:
            raise TypeError("Slicing not possible for index type " + \
            str(type(self.data.index[0])) + " and arange argument type " + \
            str(type(arange[0])) + ". Try changing the type of the arange " + \
            "values to one compatible with " + str(type(self.data.index[0])) + \
            " slicing.")    

        if rain :
            wn.warn('Data points obtained during a rain event will be replaced. '+
            'Make sure you are confident in this replacement method for the '+
            'filling of gaps in the data during rain events.')
         
        ###
        # CALCULATIONS
        ###
        daily_profile = pd.DataFrame([self.daily_profile[to_fill].index.values,
                                      self.daily_profile[to_fill]['avg'].values])
        daily_profile = daily_profile.transpose()                              
        daily_profile.index = self.daily_profile[to_fill].index
        daily_profile.columns = ['time','data']
        
        ###
        # FILLING
        ###  
        if filtered_only:
            indexes_to_replace = pd.DataFrame(self.meta_filled.\
                                            loc[arange[0]:arange[1]].\
                                            loc[self.meta_filled[to_fill] == 'filtered'].index.values,
                                            columns=['indexes'])
        elif not filtered_only:
            indexes_to_replace = pd.DataFrame(self.meta_filled.loc[arange[0]:arange[1]].index.values,
                                              columns=['indexes'])
                                              
        if isinstance(self.data.index[0],dt.datetime):
            indexes_to_replace['day'] = pd.Index(indexes_to_replace['indexes']).time
            indexes_to_replace['values'] = [daily_profile['data'][index_value] for index_value in indexes_to_replace['day']]
        elif isinstance(self.data.index[0],float):
            indexes_to_replace['day'] = indexes_to_replace['indexes'].apply(lambda x: x-int(x))
            indexes_to_replace['time_index'] = indexes_to_replace['day'].apply(find_nearest_time,args=(daily_profile,'time'))
            indexes_to_replace['values'] = indexes_to_replace['time_index'].apply(vlookup_day,args=(daily_profile,'data'))
        
        self.filled[to_fill][indexes_to_replace['indexes']] = indexes_to_replace['values'].values
        
        # Adjust in the self.meta_valid dataframe
        self.meta_filled[to_fill][indexes_to_replace['indexes']] = 'filled_average_profile'        
        
        if plot:
            self.plot_analysed(to_fill)
            
        return None
        
    def fill_missing_model(self,to_fill,to_use,arange,filtered_only=True,
                           unit='d',plot=False,clear=False):
        """
        Fills the missing values in a dataset (to_fill), based on the modeled
        values given in to_use. This happens within the range given by arange.
        
        Parameters
        ----------
        to_fill : str
            name of the column with data to fill
        to_use : pd.Series
            pandas series containing the modeled data with which the filtered
            data can be replaced
        arange : array of two values
            the range within which missing/filtered values need to be replaced
        filtered_only : boolean
            if True, fills only the datapoints labeled as filtered. If False, 
            fills/replaces all datapoints in the given range
        unit : str
            the unit in which the modeled values are given; datetime values will 
            be converted to values with that unit. Possible: sec, min, hr, d
        plot : bool
            whether or not to plot the new dataset
        clear : bool
            whether or not to clear the previoulsy filled values and start from
            the self.meta_valid dataset again for this particular dataseries.
            
        Returns
        -------
        None;
        creates/updates self.filled, containing the adjusted dataset and updates
        meta_filled with the correct labels.
        """
        ###
        # CHECKS
        ###
        self._plot = 'filled'
        wn.warn('When making use of filling functions, please make sure to '+
        'start filling small gaps and progressively move to larger gaps. This '+
        'ensures the proper working of the package algorithms.')
            
        # several checks on availability of the right columns in the necessary
        # dataframes/dictionaries
        if clear:
            self._reset_meta_filled(to_fill)
        self.meta_filled = self.meta_filled.reindex(self.index(),fill_value='!!')
        
        if not to_fill in self.meta_filled.columns:
            # if the to_fill column doesn't exist yet in the meta_filled dataset,
            # add it, and fill it with the meta_valid values; if this last one
            # doesn't exist yet, create it with 'original' tags.
            try:
                self.meta_filled[to_fill] = self.meta_valid[to_fill]
            except:
                self.add_to_meta_valid([to_fill])
                self.meta_filled[to_fill] = self.meta_valid[to_fill]
        else:
            # where the meta_filled dataset contains original values, update with
            # the values from meta_valid; in case a filling round was done before
            # any filtering; not supposed to happen, but cases exist.
            try:
                self.meta_filled[to_fill].loc[self.meta_filled[to_fill]=='original'] = \
                self.meta_valid[to_fill].loc[self.meta_filled[to_fill]=='original']
            except:
                self.add_to_meta_valid([to_fill])
                self.meta_filled[to_fill].loc[self.meta_filled[to_fill]=='original'] = \
                self.meta_valid[to_fill].loc[self.meta_filled[to_fill]=='original']
                
        if not to_fill in self.filled:
            self.add_to_filled([to_fill])
        
        # Give warning when replacing data from rain events and at the same time
        # check if arange has the right type
        try:
            rain = (self.data_type == 'WWTP') and \
                   (self.highs['highs'].loc[arange[0]:arange[1]].sum() > 1)
        except TypeError:
            raise TypeError("Slicing not possible for index type " + \
            str(type(self.data.index[0])) + " and arange argument type " + \
            str(type(arange[0])) + ". Try changing the type of the arange " + \
            "values to one compatible with " + str(type(self.data.index[0])) + \
            " slicing.")    

        if rain :
            wn.warn('Data points obtained during a rain event will be replaced. '+
            'Make sure you are confident in this replacement method for the '+
            'filling of gaps in the data during rain events.')       
        
        ###
        # CALCULATIONS
        ###
        #model_values = to_use.name
        model_values = pd.DataFrame(index = to_use.index)
        model_values['time'] = to_use.index
        model_values['data'] = to_use.values
        
        ###
        # FILLING
        ###
        if filtered_only:
            indexes_to_replace = pd.DataFrame(self.meta_filled.\
                                            loc[arange[0]:arange[1]].\
                                            loc[self.meta_filled[to_fill] == 'filtered'].index.values,
                                            columns=['indexes'])   
        if not filtered_only:
            indexes_to_replace = pd.DataFrame(self.meta_filled.\
                                            loc[arange[0]:arange[1]].index.values,
                                            columns=['indexes'])
                                            
        if isinstance(self.data.index[0],dt.datetime):
            try:
                indexes_to_replace['abs_indexes'] = absolute_to_relative(indexes_to_replace['indexes'],
                                                    start_date=self.data.index[0],unit=unit)
                indexes_to_replace['time_index'] = indexes_to_replace['abs_indexes'].\
                                                    apply(find_nearest_time,args=(model_values,'time'))
            except(IndexError):
                raise IndexError('No indexes were found to replace. Check the '+\
                'range in which you want to replace values, or check if filtered '+\
                'values actually exist in the meta_filled dataset.')
            
        elif isinstance(self.data.index[0],float):
            indexes_to_replace['time_index'] = indexes_to_replace['indexes'].apply(find_nearest_time,args=(model_values,'time'))
            
        indexes_to_replace['values'] = indexes_to_replace['time_index'].apply(vlookup_day,args=(model_values,'data'))
        
        self.filled[to_fill][indexes_to_replace['indexes']] = indexes_to_replace['values'].values
        # Adjust in the self.meta_valid dataframe
        self.meta_filled[to_fill][indexes_to_replace['indexes']] = 'filled_infl_model'        
            
            #self.filled.loc[arange[0]:arange[1],to_fill] = to_use.values
            # Adjust in the self.meta_valid dataframe
            #self.meta_filled.loc[arange[0]:arange[1],to_fill] = 'filled_model'
        
        if plot:
            self.plot_analysed(to_fill)
            
        return None
        
    def fill_missing_daybefore(self,to_fill,arange,range_to_replace=[1,4],
                               filtered_only=True,plot=False,clear=False):
        """
        Fills the missing values in a dataset (to_fill), based on the data values
        from the day before the range starts. These data values are based on 
        the self.filled dataset and therefor can contain filled datapoints as well. 
        This happens within the range given by arange.
        
        Parameters
        ----------
        to_fill : str
            name of the column with data to fill
        arange : array of two values
            the range within which missing/filtered values need to be replaced
        range_to_replace : array of two int/float values
            the minimum and maximum amount of time (i.e. min and max size of 
            gaps in data) where missing datapoints can be replaced using this 
            function, i.e. using values of the last day before measurements 
            went bad.
        filtered_only : boolean
            if True, fills only the datapoints labeled as filtered. If False, 
            fills/replaces all datapoints in the given range
        plot : bool
            whether or not to plot the new dataset
        clear : bool
            whether or not to clear the previoulsy filled values and start from
            the self.meta_valid dataset again for this particular dataseries.
            
        Returns
        -------
        None;
        creates/updates self.filled, containing the adjusted dataset and updates
        meta_filled with the correct labels.
        """
        ###
        # CHECKS
        ###
        self._plot = 'filled'
        wn.warn('When making use of filling functions, please make sure to '+
        'start filling small gaps and progressively move to larger gaps. This '+
        'ensures the proper working of the package algorithms.')
        # index checks
        #if arange[0] < 1 or arange[1] > self.index()[-1]:
        #    raise IndexError('Index out of bounds. Check whether the values of \
        #    "arange" are within the index range of the data. Mind that the first \
        #    day of data cannot be replaced with this algorithm!')
            
        # several checks on availability of the right columns in the necessary
        # dataframes/dictionaries
        if clear:
            self._reset_meta_filled(to_fill)
        self.meta_filled = self.meta_filled.reindex(self.index(),fill_value='!!')
        
        if not to_fill in self.meta_filled.columns:
            # if the to_fill column doesn't exist yet in the meta_filled dataset,
            # add it, and fill it with the meta_valid values; if this last one
            # doesn't exist yet, create it with 'original' tags.
            try:
                self.meta_filled[to_fill] = self.meta_valid[to_fill]
            except:
                self.add_to_meta_valid([to_fill])
                self.meta_filled[to_fill] = self.meta_valid[to_fill]
        else:
            # where the meta_filled dataset contains original values, update with
            # the values from meta_valid; in case a filling round was done before
            # any filtering; not supposed to happen, but cases exist.
            try:
                self.meta_filled[to_fill].loc[self.meta_filled[to_fill]=='original'] = \
                self.meta_valid[to_fill].loc[self.meta_filled[to_fill]=='original']
            except:
                self.add_to_meta_valid([to_fill])
                self.meta_filled[to_fill].loc[self.meta_filled[to_fill]=='original'] = \
                self.meta_valid[to_fill].loc[self.meta_filled[to_fill]=='original']
                
        if not to_fill in self.filled:
            self.add_to_filled([to_fill])
        
        # Give warning when replacing data from rain events and at the same time
        # check if arange has the right type
        try:
            rain = (self.data_type == 'WWTP') and \
                   (self.highs['highs'].loc[arange[0]:arange[1]].sum() > 1)
        except TypeError:
            raise TypeError("Slicing not possible for index type " + \
            str(type(self.data.index[0])) + " and arange argument type " + \
            str(type(arange[0])) + ". Try changing the type of the arange " + \
            "values to one compatible with " + str(type(self.data.index[0])) + \
            " slicing.")    

        if rain :
            wn.warn('Data points obtained during a rain event will be replaced. '+
            'Make sure you are confident in this replacement method for the '+
            'filling of gaps in the data during rain events.')       
        
        ###
        # CALCULATIONS
        ###
        # Get data to fill the missing data with, i.e. data from the day before,
        # and convert indices to relative ones per day; parallel for 
        # self.meta_filled
        
        if isinstance(self.data.index[0],dt.datetime):
            oneday = dt.timedelta(1)
            time = pd.Series((self.filled[to_fill][arange[0]-oneday:arange[0]].index).time)    
        elif isinstance(self.data.index[0],float):
            oneday = 1
            time = pd.Series(self.filled[to_fill][arange[0]-oneday:arange[0]].index).apply(lambda x: x-int(x))
            
        day_before = pd.DataFrame(self.filled[to_fill][arange[0]-oneday:arange[0]].values,
                                  index=time)
        day_before.columns = ['data']
        day_before = day_before.reset_index().drop_duplicates('index',keep='first').\
                     set_index('index')
        #day_before_meta = self.meta_filled[to_fill][arange[0]-1:arange[0]]
        #day_before_meta.index = pd.Series(day_before_meta.index).apply(lambda x: x-int(x))
        
        range_to_replace[0] = range_to_replace[0] * len(day_before)
        range_to_replace[1] = range_to_replace[1] * len(day_before)
        
        # Create a mask to replace the filtered datapoints with nan-values, 
        # if consecutive occurence lower than range_
        mask_df = pd.DataFrame(index = self.meta_valid[arange[0]:arange[1]].index)
        mask_df['count'] = (self.meta_valid[to_fill][arange[0]:arange[1]] != self.meta_valid[to_fill][arange[0]:arange[1]].\
                            shift()).astype(int).cumsum().astype(str)
        group = mask_df.groupby('count').size()
        group.index = mask_df.groupby('count').size().index.astype(str)
                
        # Compare the values in 'count' with the ones in the group-by object.
        # mask_df now contains the amount of consecutive true or false datapoints,
        # for every datapointday
        replace_dict = {'count':dict(group)}
        mask_df = mask_df.replace(replace_dict)
        
        ###
        # FILLING
        ###
        # Based on the mask and whether a datapoint is filtered, replace with
        # nan values
        if filtered_only:
            filtered_based = pd.DataFrame(self.meta_valid.loc[arange[0]:arange[1]].\
                                           loc[self.meta_filled[to_fill] == 'filtered'].index.values,
                                           columns = ['indexes'])
        if not filtered_only:
            filtered_based = pd.DataFrame(self.meta_filled.loc[arange[0]:arange[1]].index.values,
                                           columns=['indexes'])    
        
        mask_based = pd.DataFrame(mask_df.loc[mask_df['count'] < range_to_replace[1]].\
                                          loc[mask_df['count'] > range_to_replace[0]].\
                                          index.values,columns=['indexes'])
        #mask_based.columns = ['indexes']
        indexes_to_replace = pd.merge(filtered_based,mask_based,how='inner')
        
        # look up the values to replace with in the day_before dataset
        if isinstance(self.data.index[0],dt.datetime):
            indexes_to_replace['day'] = pd.Index(indexes_to_replace['indexes']).time
            indexes_to_replace['values'] = [day_before['data'][index_value] for index_value in indexes_to_replace['day']]
        elif isinstance(self.data.index[0],float):
            indexes_to_replace['day'] = indexes_to_replace['indexes'].apply(lambda x: x-int(x))
            indexes_to_replace['time_index'] = indexes_to_replace['day'].apply(find_nearest_time,args=(day_before,'time'))
            indexes_to_replace['values'] = indexes_to_replace['time_index'].apply(vlookup_day,args=(day_before,'data'))
        
        self.filled[to_fill][indexes_to_replace['indexes']] = indexes_to_replace['values'].values
        
        # Adjust in the self.meta_valid dataframe
        self.meta_filled[to_fill][indexes_to_replace['indexes']] = 'filled_profile_day_before'        
        
        if plot:
            self.plot_analysed(to_fill)
            
        return None
        
#==============================================================================
# LOOKUP FUNCTIONS
#==============================================================================

def find_nearest_time(value,df,column):
    """
    """
    return (np.abs(df[column]-value)).argmin()

def vlookup_day(value,df,column):
    """
    """
    return df[column].loc[value]


####START ADJUSTING HERE NEXT TIME! 
  
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
            raise TypeError('Please provide the location of the log file as '+
                            'a string type, or leave the argument if no log '+
                            'file is needed.')                
        
        return self.__class__(data,self.timename)
        
    elif inplace == True:
        self.drop(self.data[abs(self.data[data_name]) > cutoff].index,
                            inplace=True)
        self.data.reset_index(drop=True,inplace=True)
        new = len(self.data)
        if log_file == None:
            _print_removed_output(original,new)
        elif type(log_file) == str:
            _log_removed_output(log_file,original,new)
        else :
            raise TypeError('Please provide the location of the log file as '+
                            'a string type, or leave the argument if no log '+
                            'file is needed.') 


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
            return self.__class__(self.data,self.timename)
        except:#IndexError:
            print('Not enough datapoints left for selection')

    elif down == False:
        try:
            print('Selecting upward slope:',len(self.data)-drop_index,\
            'datapoints dropped,',drop_index,'datapoints left.')
        
            self.data = self.data[:drop_index]
            self.data.reset_index(drop=True,inplace=True)
            return self.__class__(self.data,self.timename)
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
            
def go_WEST(raw_data,time_data,WEST_name_conversion):
    """
    Saves a WEST compatible file (influent or other inputs)
    
    parameters
    ----------
    raw_data: str or pd DataFrame
    
    time_data: 
        
    WEST_name_conversion: pd DataFrame with column names: WEST, units and RAW
        dataframe containing three columns: the column names for the WEST-compatible file,
        the units to appear in the WEST-compatible file and the column names of the raw
        data file.
        
    output
    ------
    None
    
    """
    #if type(raw_data) == str:
    #    try data = pd.read_csv(raw_data,sep= '\t')
    #    except print('Provide valid file name (including path) to read.')
    #else:
    data = raw_data
    #if not data.columns == WEST_name_conversion['raw_data_name']
    #    print('raw data columns should be the same as the raw data colum values given in WEST_name_conversion')
    #    return None
    
    WEST_compatible = pd.DataFrame()
    for i in range(0,len(WEST_name_conversion)):
        WEST_compatible[WEST_name_conversion['WEST'][i]] = data[WEST_name_conversion['RAW'][i]]
    help_df = pd.DataFrame(WEST_name_conversion['units']).transpose()
    help_df.columns = [WEST_compatible.columns]
    WEST_compatible = help_df.append(WEST_compatible)
    WEST_compatible.insert(0,'#t',time_data)
    WEST_compatible['#t']['units']='#d'
    return WEST_compatible 


    
###############################################################################
##                          HELP FUNCTIONS                                   ##
###############################################################################

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

def total_seconds(timedelta_value):
    return timedelta_value.total_seconds()
                
def absolute_to_relative(series,start_date,unit='d',decimals=5):
    """
    converts a pandas series with datetime timevalues to relative timevalues 
    in the given unit, starting from start_date
    
    parameters
    ----------
    series : pd.Series
        series of datetime of comparable values
    unit : str 
        unit to which to convert the time values (sec, min, hr or d)
    
    output
    ------
    
    """
    try:
        time_delta = series - series[0]
    except('IndexError'):
        raise IndexError('The passed series appears to be empty. To calculate ' + \
        'a relative timeseries, an absolute timeseries is necessary.')
    start = total_seconds(series[0] - start_date)
    
    relative = time_delta.map(total_seconds)
    if unit == 'sec':
        relative = np.array(relative) + start
    elif unit == 'min':
        relative = (np.array(relative) + start) / (60) 
    elif unit == 'hr':
        relative = (np.array(relative) + start) / (60*60)
    elif unit == 'd':
        relative = (np.array(relative) + start) / (60*60*24)
    
    return relative.round(decimals)
    
