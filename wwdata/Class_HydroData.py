# -*- coding: utf-8 -*-
"""
    Class_HydroData provides functionalities for handling data obtained in the context of (waste)water treatment.
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
import os
#from os import listdir
import pandas as pd
import scipy as sp
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt   #plotten in python
import warnings as wn

import wwdata.data_reading_functions #imports the functions in data_reading_functions.py: the ones without underscore are included, the ones with underscore need to be called by hp.data_reading_functions.function()
#import time_conversion_functions #import timedelta_to_abs, _get_datetime_info,\
#make_datetime,to_datetime_singlevalue

class HydroData():
    """
    """
    def __init__(self,data,timedata_column='index',data_type='WWTP',
                 experiment_tag='No tag given',time_unit=None,
                 units=[]):
        """
        initialisation of a HydroData object.

        Attributes
        ----------
        timedata_column : str
            name of the column containing the time data
        experiment_tag : str
            A tag identifying the experiment; can be a date or a code used by
            the producer/owner of the data.
        time_unit : str
            The time unit in which the time data is given
        units : array
            The units of the variables in the columns
        """
        if isinstance(data, pd.DataFrame):
            self.data = data.copy()
        else:
            try:
                self.data = pd.DataFrame(data.copy())
            except:
                raise Exception("Input data not convertable to DataFrame.")
        if timedata_column == 'index':
            self.timename = 'index'
            self.time = self.data.index
        else:
            self.timename = timedata_column
            self.time = self.data[timedata_column].values.ravel()
        self.columns = np.array(self.data.columns)
        self.data_type = data_type
        self.tag = experiment_tag
        self.time_unit = time_unit
        self.meta_valid = pd.DataFrame(index=self.data.index)
        self.units = units
        #self.highs = pd.DataFrame(data=0,columns=['highs'],index=self.data.index)
        #wn.warn('WARNING: Some functions in the OnlineSensorBased Class assume ' + \
        #'equidistant data!!! This is primarily of importance when indexes are ' + \
        #'missing!')

    def set_tag(self,tag):
        """CONFIRMED
        """
        self.tag = tag

    def set_units(self,units):
        """
        """
        if isinstance(units, pd.DataFrame):
            self.units = units.copy()
        else:
            try:
                self.units = pd.DataFrame(units.copy())
            except:
                raise Exception("Unit data not convertable to DataFrame type.")

    def set_time_unit(self,unit):
        """
        """

        self.time_unit = unit

    def head(self, n=5):
        """CONFIRMED
        piping pandas head function"""
        return self.data.head(n)

    def tail(self, n=5):
        """CONFIRMED
        piping pandas tail function"""
        return self.data.tail(n)

    def index(self):
        """
        piping pandas index function
        """
        return self.data.index

    #####################
    ###   FORMATTING
    #####################

    def fill_index(self,arange,index_type='float'):
        """
        function to fill in missing index values
        """
        wn.warn('This function assumes equidistant data and fills the indexes '+\
        'accordingly')
        first_part = self.data[self.data.index < arange[0]]

        if isinstance(self.data.index[0],dt.datetime):
            delta_time = self.data.index[1]-self.data.index[0]
            index = [arange[0] + delta_time * x for x in range(0, int((arange[1]-arange[0])/delta_time))]
        elif isinstance(self.data.index[0],float):
            day_length = float(len(self.data[0:1]))
            index = np.arange(arange[0],arange[1],(arange[1]-arange[0])/day_length)

        fill_part = pd.DataFrame(index=index,columns=self.data.columns)

        last_part = self.data[self.data.index > arange[1]]
        self.data = first_part.append(fill_part).append(last_part)
        self._update_time()

    def _reset_meta_valid(self,data_name=None):
        """
        reset the meta dataframe, possibly for only a certain data series,
        should wrong labels have been assigned at some point
        """
        if data_name == None:
            self.meta_valid = pd.DataFrame(index=self.data.index)
        else:
            try:
                self.meta_valid[data_name] = pd.Series(['original']*len(self.meta_valid),index=self.index())
        #self.meta_valid.drop(data_name,axis=1)
            except:
                pass
                #wn.warn(data_name + ' is not contained in self.meta_valid yet, so cannot\
                #be removed from it!')

    def drop_index_duplicates(self):
        """
        drop rows with a duplicate index, ASSUMING THEY HAVE THE SAME DATA IN
        THEM!! Also updates the meta_valid dataframe
        """
        #len_orig = len(self.data)
        self.data = self.data.groupby(self.index()).first()
        self.meta_valid = self.meta_valid.groupby(self.meta_valid.index).first()
        self._update_time()
        if isinstance(self.index()[1],str):
            wn.warn('Rows may change order using this function based on '+ \
            'string values. Convert to datetime, int or float and use '+ \
            '.sort_index() or .sort_value() to avoid. (see also hp.to_datetime())')

    def replace(self,to_replace,value,inplace=False):
        """CONFIRMED
        piping pandas replace function"""
        if inplace == False:
            return self.__class__(self.data.replace(to_replace,value,inplace=False),
                                  self.data.timename,self.data_type,
                                  self.tag,self.time_unit)
        elif inplace == True:
            return self.data.replace(to_replace,value,inplace=inplace)

    def set_index(self,keys,key_is_time=False,drop=True,inplace=False,
                  verify_integrity=False,save_prev_index=True):
        """
        piping pandas set_index function, including the option to define the new index
        as being the time data
        key_is_time : bool
            when true, the new index will we known as the time data from here on
        (other arguments cfr pd.set_index)
        """
        if save_prev_index:
            self.prev_index = self.data.index

        if not inplace:
            if key_is_time:
                if isinstance(self.time[0],str):
                    raise ValueError('Time values of type "str" can not be used as index')
                timedata_column = 'index'
            elif key_is_time == False:
                timedata_column = self.timename

            data = self.data.set_index(keys,drop=drop,inplace=False,
                                       verify_integrity=verify_integrity)

            return self.__class__(pd.DataFrame(data),timedata_column=timedata_column,
                                  data_type=self.data_type,experiment_tag=self.tag,
                                  time_unit=self.time_unit)

        elif inplace:
            if key_is_time:
                if self.timename == 'index':
                    raise IndexError('There already is a timeseries in the dataframe index!')
                if isinstance(self.time[0],str):
                    raise ValueError('Time values of type "str" can not be used as index')

            self.data.set_index(keys,drop=drop,inplace=True,
                                verify_integrity=verify_integrity)
            self.columns = np.array(self.data.columns)
            self._update_meta_valid_index()

            if key_is_time:
                self.timename = 'index'
                self.time = self.data.index

    def _update_time(self):
        """
        adjust the value of self.time, needed in some functions
        """
        if self.timename == 'index':
            self.time = self.index()
        else:
            self.time = self.data[self.timename]

    def _update_meta_valid_index(self):
        """
        update the index of the meta_valid dataframe to be the same as the one of the dataframe
        with the data
        """
        self.meta_valid.index = self.index()

    def to_float(self,columns='all'):
        """
        convert values in given columns to float values

        Parameters
        ---------
        columns : array of strings
            column names of the columns where values need to be converted to floats

        """
        if columns == 'all':
            columns = self.columns#.levels[0]
        for column in columns:
            try:
                self.data[column] = self.data[column].astype(float)
            except TypeError:
                print('Data type of column '+ str(column) + ' not convertible to float')
        self._update_time()

    def to_datetime(self,time_column='index',time_format='%dd-%mm-%yy',
                    unit='D'):
        """CONFIRMED
        flexibly piping pandas to_datetime function

        Parameter
        ---------
        time_column : str
            column name of the column where values need to be converted to date-
            time values. Default 'index' converts index values to datetime
        time_format : str
            the format to use by to_datetime function to convert strings to
            datetime format
        unit : str
            unit to use by to_datetime function to convert int or float values
            to datetime format
        """
        if time_column == 'index':
            if isinstance(self.time[0],int) or isinstance(self.time[0],float):
                self.data.index = pd.to_datetime(self.time,unit=unit)
                self.data.sort_index(inplace=True)
            elif isinstance(self.time[0],str):
                self.data.index = pd.to_datetime(self.time,format=time_format)
                self.data.sort_index(inplace=True)
        else:
            if isinstance(self.time[0],int) or isinstance(self.time[0],float):
                self.data.index = pd.to_datetime(self.data[time_column],unit=unit)
                self.data.sort_values(inplace=True)
            elif isinstance(self.time[0],str):
                self.data[time_column] = pd.to_datetime(self.data[time_column].values.ravel(),
                                                        format=time_format)
                self.data.sort_values(time_column,inplace=True)
        self._update_time()

    def absolute_to_relative(self,time_data='index',unit='d',inplace=True,
                             save_abs=True,decimals=5):
        """
        converts a pandas series with datetime timevalues to relative timevalues
        in the given unit, starting from 0

        parameters
        ----------
        time_data : str
            name of the column containing the time data. If this is the index
            column, just give 'index' (also default)
        unit : str
            unit to which to convert the time values (sec, min, hr or d)

        output
        ------

        """
        if time_data == 'index':
            timedata = self.time
        else:
            timedata = self.data[time_data]
        time_delta = timedata - timedata[0]

        relative = time_delta.map(total_seconds)
        if unit == 'sec':
            relative = np.array(relative)
        elif unit == 'min':
            relative = np.array(relative) / (60)
        elif unit == 'hr':
            relative = np.array(relative) / (60*60)
        elif unit == 'd':
            relative = np.array(relative) / (60*60*24)
        self.time_unit = unit

        if inplace == False:
            data = self.data.copy()
            data['time_rel'] = relative.round(decimals)
            return self.__class__(data,self.timename)
        elif inplace == True:
            if save_abs == True:
                self.data['time_abs'] = timedata
                self.columns = np.array(self.data.columns)
            if time_data == 'index':
                self.data.index = relative.round(decimals)
                self._update_time()
                self.columns = np.array(self.data.columns)
                return None
            else:
                self.data[time_data] = relative.round(decimals)
                return None

    def write(self,filename,filepath=os.getcwd(),method='all'):
        """

        Parameters
        ----------
        filepath : str
            the path the output file should be saved to
        filename : str
            the name of the output file
        method : str (all,filtered,filled)
            depending on the method choice, different values will be written out:
            all values, only the filtered values or the filled values
        for_WEST : bool
        include_units : bool


        Returns
        -------
        None; write an output file
        """

        if method == 'all':
            self.data.to_csv(os.path.join(filepath,filename),sep='\t')
        elif method == 'filtered':
            to_write = self.data.copy()
            for column in self.meta_valid.columns:
                to_write[column] = self.data[column][self.meta_valid[column]=='original']
            to_write.to_csv(os.path.join(filepath,filename),sep='\t')
        elif method == 'filled':
            self.filled.to_csv(os.path.join(filepath,filename),sep='\t')

    #######################
    ### DATA EXPLORATION
    #######################

    def get_avg(self,name=None,only_checked=True):
        """
        Gets the averages of all or certain columns in a dataframe

        Parameters
        ----------
        name : arary of str
            name(s) of the column(s) containing the data to be averaged;
            defaults to ['none'] and will calculate average for every column

        Returns
        -------
        pd.DataFrame :
            pandas dataframe, containing the average slopes of all or certain
            columns
        """
        mean = []
        if only_checked:
            df = self.data.copy()
            df[self.meta_valid == 'filtered']=np.nan

            if name == None:
                mean = df.mean()
            elif isinstance(name,str):
                mean = df[name].mean()
            else:
                for i in name:
                    mean.append(df[name].mean())

        else:
            if name == None:
                mean = self.data.mean()
            elif isinstance(name,str):
                mean = self.data[name].mean()
            else:
                for i in name:
                    mean.append(self.data[name].mean())

        return mean

    def get_std(self,name=None,only_checked=True):
        """
        Gets the standard deviations of all or certain columns in a dataframe

        Parameters
        ----------
        dataframe : pd.DataFrame
            dataframe containing the columns to calculate the standard deviation for
        name : arary of str
            name(s) of the column(s) containing the data to calculate standard
            deviation for; defaults to ['none'] and will calculate standard
            deviation for every column
        plot : bool
            if True, plots the calculated standard deviations, defaults to False

        Returns
        -------
        pd.DataFrame :
            pandas dataframe, containing the average slopes of all or certain
            columns
        """
        std=[]
        if only_checked:
            df = self.data.copy()
            df[self.meta_valid == 'filtered']=np.nan

            if name == None:
                std = df.std()
            elif isinstance(name,str):
                std = df[name].std()
            else:
                for i in name:
                    std.append(df[name].std())

        else:
            if name == None:
                std = self.data.std()
            elif isinstance(name,str):
                std = self.data[name].std()
            else:
                for i in name:
                    std.append(self.data[name].std())

        return std

    def get_highs(self,data_name,bound_value,arange,method='percentile',plot=False):
        """
        creates a dataframe with tags indicating what indices have data-values
        higher than a certain value; example: the definition/tagging of rain
        events.

        data_name : str
            name of the column to execute the function on
        bound_value : float
            the boundary value above which points will be tagged
        arange : array of two values
            the range within which high values need to be tagged
        method: str (value or percentile)
            when percentile, the bound value is a given percentile above which
            data points will be tagged, when value, bound_values is used directly
            to tag data points.
        """
        self._reset_highs()
        try:
            data_to_use = self.data[data_name][arange[0]:arange[1]].copy()
        except TypeError:
            raise TypeError("Slicing not possible for index type " + \
            str(type(self.data.index[0])) + " and arange argument type " + \
            str(type(arange[0])) + ". Try changing the type of the arange " + \
            "values to one compatible with " + str(type(self.data.index[0])) + \
            " slicing.")   

        # get indexes where flow is higher then bound_value
        if method is 'value':
            bound_value = bound_value
        elif method is 'percentile':
            bound_value = data_to_use.dropna().quantile(bound_value)

        indexes = data_to_use.loc[data_to_use > bound_value].index
        self.highs['highs'].loc[indexes] = 1

        if plot:
            fig = plt.figure(figsize=(16,6))
            ax = fig.add_subplot(111)
            ax.plot(data_to_use[self.highs['highs']==0].index,
                    data_to_use[self.highs['highs']==0],
                    '-g')
            ax.plot(data_to_use[self.highs['highs']==1].index,
                    data_to_use[self.highs['highs']==1],
                    '.b',label='high')
            ax.legend(fontsize=17)
            ax.tick_params(labelsize=15)
            ax.set_ylabel(data_name,size=17)
            ax.set_xlabel('Time',size=17)

    def _reset_highs(self):
        """
        """
        self.highs = pd.DataFrame(data=0,columns=['highs'],index=self.index())

    ##############
    ### FILTERING
    ##############

    def add_to_meta_valid(self,column_names):
        """
        Adds (a) column(s) with the given column_name(s) to the self.meta_filled
        DataFrame, where all tags are set to 'original'. This makes sure that
        also data that already is very reliable can be used further down the
        process (e.g. filling etc.)

        Parameters
        ----------
        column_names : array
            array containing the names of the columns to add to the meta_valied
            dataframe
        """
        self._plot = 'valid'
        # Create/adjust self.filled
        self.meta_valid = self.meta_valid.reindex(self.index())
        for column in column_names:
            if not column in self.meta_valid.columns:
                self.meta_valid[column] = 'original'
            else:
                pass
                wn.warn('self.meta_valid already contains a column named ' +
                    column + '. The original column was kept.')


    def tag_nan(self,data_name,arange=None,clear=False):
        """
        adds a tag 'filtered' in self.meta_valid for every NaN value in the given
        column

        Parameters
        ----------
        data_name : str
            column name of the column to apply the function to
        arange : array of two values
            the range within which nan values need to be tagged
        clear : bool
            when true, resets the tags in meta_valid for the data in column
            data_name

        Returns
        -------
        None

        """
        self._plot='valid'

        if clear:
            self._reset_meta_valid(data_name)
        self.meta_valid = self.meta_valid.reindex(self.index(),fill_value='!!')
        
        if not data_name in self.meta_valid.columns:
            # if the data_name column doesn't exist yet in the meta_valid dataset,
            # add it
            self.add_to_meta_valid([data_name])
        
        if arange == None:
            len_orig = len(self.data[data_name])
            self.meta_valid[data_name] = np.where(np.isnan(self.data[data_name]),
                                                  'filtered','original')
            len_new = self.data[data_name].count()
    
        else:
            # check if arange has the right type
            try:
                len_orig = len(self.data[data_name][arange[0]:arange[1]])
            except TypeError:
                raise TypeError("Slicing not possible for index type " + \
                                str(type(self.data.index[0])) + " and arange "+\
                                "argument type " + str(type(arange[0])) + " or " +\
                                str(type(arange[1])) + ". Try changing the type "+\
                                "of the arange values to one compatible with " + \
                                str(type(self.data.index[0])) + " slicing.") 
                
            self.meta_valid[data_name][arange[0]:arange[1]] = np.where(np.isnan(self.data[data_name][arange[0]:arange[1]]),
                                                                       'filtered','original')
            len_new = self.data[data_name][arange[0]:arange[1]].count()
    
        _print_removed_output(len_orig,len_new,'NaN tagging')

    def tag_doubles(self,data_name,bound,arange=None,clear=False,inplace=False,log_file=None,
                       plot=False,final=False):
        '''
        tags double values that subsequently occur in a measurement series.
        This is relevant in case a sensor has failed and produces a constant
        signal. A band is provided within which the signal can vary and still
        be filtered out

        Parameters
        ----------
        data_name : str
            column name of the column from which double values will be sought
        bound : float
            boundary value of the band to use. When the difference between a
            point and the next one is smaller then the bound value, the latter
            datapoint is tagged as 'filtered'.
        arange : array of two values
            the range within which double values need to be tagged
        clear : bool
            if True, the tags added to datapoints before will be removed and put
            back to 'original'.
        inplace : bool
            indicates whether a new dataframe is created and returned or whether
            the operations are executed on the existing dataframe (nothing is
            returned). (This argument only comes into play when the 'final'
            argument is True)
        log_file : str
            string containing the directory to a log file to be written out
            when using this function
        plot : bool
             whether or not to make a plot of the newly tagged data points
        final : bool
            if true, the values are actually replaced with nan values (either
            inplace or in a new hp object)

        Returns
        -------
        HydroData object (if inplace=False)
            the dataframe from which the double values of 'data' are removed or
            replaced
        None (if inplace=True)
        '''
        self._plot = 'valid'
        len_orig = self.data[data_name].count()

        # Make temporary object for operations
        df_temp = self.__class__(self.data.copy(),timedata_column=self.timename,
                                 data_type=self.data_type,experiment_tag=self.tag,
                                 time_unit=self.time_unit)
        # Make a mask with False values for double values to be dropped
        bound_mask = abs(self.data[data_name].dropna().diff()) >= bound
        # Make sure the indexes are still the same in the mask and df_temp, so the
        # tagging can happen
        bound_mask = bound_mask.reindex(df_temp.index()).fillna(True)
        # Make a mask with False values where data needs to be filtered
        if arange == None:
            mask = bound_mask
        else:
            try:
                range_mask = (self.index() < arange[0]) | (arange[1] < self.index())
                mask = bound_mask + range_mask
            except TypeError:
                raise TypeError("Slicing not possible for index type " + \
                                str(type(self.data.index[0])) + " and arange "+\
                                "argument type " + str(type(arange[0])) + " or " +\
                                str(type(arange[1])) + ". Try changing the type "+\
                                "of the arange values to one compatible with " + \
                                str(type(self.data.index[0])) + " slicing.")

        # Update the index of self.meta_valid
        if clear:
            self._reset_meta_valid(data_name)
        self.meta_valid = self.meta_valid.reindex(self.index(),fill_value='!!')

        # Do the actual filtering, based on the mask
        df_temp.data[data_name] = df_temp.data[data_name].drop(df_temp.data[mask==False].index)
        len_new = df_temp.data[data_name].count()
        
        if log_file == None:
            _print_removed_output(len_orig,len_new,'double value tagging')
        elif type(log_file) == str:
            _log_removed_output(log_file,len_orig,len_new,'filtered')
        else:
            raise TypeError('Provide the location of the log file \
                            as a string type, or drop the argument if \
                            no log file is needed.')

        self.meta_valid[data_name][mask==False] = 'filtered'

        # Create new temporary object, where the dropped datapoints are replaced
        # by nan values (by assigning a new column to the original dataframe)
        #df_temp_2 = self.__class__(self.data.copy(),timedata_column=self.timename,
        #                           experiment_tag=self.tag,time_unit=self.time_unit)
        #df_temp_2.data[data_name] = df_temp.data[data_name]
        #df_temp_2._update_time()
        # Update the self.meta_valid dataframe, to contain False values for dropped
        # datapoints. This is done by tracking the nan values in df_temp_2
        #if data_name in self.meta_valid.columns:
        #    temp_1 = self.meta_valid[data_name].isin(['filtered'])
        #    temp_2 = pd.DataFrame(np.where(np.isnan(df_temp_2.data[data_name]),True,False))
        #    temp_3 = temp_1 | temp_2
        #    self.meta_valid[data_name] = np.where(temp_3,'filtered','original')
        #else:
        #    self.meta_valid[data_name] = np.isnan(df_temp_2.data[data_name])
        #    self.meta_valid[data_name] = np.where(self.meta_valid[data_name],'filtered','original')

        if plot == True:
            self.plot_analysed(data_name)

        if final:
            if inplace:
                self.data[data_name] = df_temp.data[data_name]
                self._update_time()
            elif not inplace:
                return df_temp

        if not final:
            return None
        
        
    def tag_extremes(self,data_name,arange=None,limit=0,method='below',
                     clear=False,plot=False):
        """
        Tags values above or below a given limit.
        
        Parameters
        ----------
        data_name : str
            name of the column containing the data to be tagged
        arange : array of two values
            the range within which extreme values need to be tagged 
        limit : int/float
            limit below or above which values need to be tagged
        method : 'below' or 'above'
            below tags all the values below the given limit, above tags
            the values above the limit
        clear : bool
            if True, the tags added before will be removed and put
            back to 'original'.
        plot : bool
             whether or not to make a plot of the newly tagged data points
        
        Returns
        -------
        None; 
        """
        if clear:
            self._reset_meta_valid(data_name)
        self.meta_valid = self.meta_valid.reindex(self.index(),fill_value='!!')

        if not data_name in self.meta_valid.columns:
            # if the data_name column doesn't exist yet in the meta_valid dataset,
            # add it
            self.add_to_meta_valid([data_name])

        if arange == None:
            len_orig = len(self.data[data_name])
            mask_valid = np.where(self.meta_valid[data_name] == 'filtered',True,False)
            if method == 'below':
                mask_tagging = np.where(self.data[data_name]<limit,True,False)
                mask = pd.DataFrame(np.transpose([mask_tagging,mask_valid])).any(axis=1)
                self.meta_valid[data_name] = np.where(mask,'filtered','original')
            elif method == 'above':
                mask_tagging = np.where(self.data[data_name]>limit,True,False)
                mask = pd.DataFrame(np.transpose([mask_tagging,mask_valid])).any(axis=1)
                self.meta_valid[data_name] = np.where(mask,'filtered','original')

        else:
            # check if arange has the right type
            try:
                len_orig = len(self.data[data_name][arange[0]:arange[1]])
                mask_valid = np.where(self.meta_valid[data_name][arange[0]:arange[1]] == 'filtered',True,False)
            except TypeError:
                raise TypeError("Slicing not possible for index type " + \
                                str(type(self.data.index[0])) + " and arange "+\
                                "argument type " + str(type(arange[0])) + " or " +\
                                str(type(arange[1])) + ". Try changing the type "+\
                                "of the arange values to one compatible with " + \
                                str(type(self.data.index[0])) + " slicing.") 
            if method == 'below':
                mask_tagging = np.where(self.data[data_name][arange[0]:arange[1]]<limit,True,False)
                mask = pd.DataFrame(np.transpose([mask_tagging,mask_valid])).any(axis=1)
                self.meta_valid[data_name][arange[0]:arange[1]] = np.where(mask,'filtered','original')
            elif method == 'above':
                mask_tagging = np.where(self.data[data_name][arange[0]:arange[1]]>limit,True,False)
                mask = pd.DataFrame(np.transpose([mask_tagging,mask_valid])).any(axis=1)
                self.meta_valid[data_name][arange[0]:arange[1]] = np.where(mask,'filtered','original')

        len_new = mask_tagging.sum()

        _print_removed_output(len_orig,len_new,'tagging of extremes ('+method+')')
        
        if plot == True:
            self.plot_analysed(data_name)

    def calc_slopes(self,xdata,ydata,time_unit=None,slope_range=None):
        """
        Calculates slopes for given xdata and data_name; if a time unit is given as
        an argument, the time values (xdata) will first be converted to this
        unit, which will then be used to calculate the slopes with.

        Parameters
        ----------
        xdata : str
            name of the column containing the xdata for slope calculation
            (e.g. time). If 'index', the index is used as xdata. If datetime
            objects, a time_unit is expected to calculate the slopes.
        data_name : str
            name of the column containing the data_name for slope calculation
        time_unit : str
            time unit to be used for the slope calculation (in case this is
            based on time); if None, slopes are simply calculated based on the
            values given
            !! This value has no impact if the xdata column is the index and is
            not a datetime type. If that is the case, it is assumed that the
            user knows the unit of the xdata !!

        Returns
        -------
        pd.Series
            pandas Series object containing the slopes calculated for the
            chosen variable
        """
        slopes = pd.DataFrame()

        if xdata == 'index':
            self.data[xdata] = self.data.index

        date_time = isinstance(self.data[xdata][0],np.datetime64) or \
                    isinstance(self.data[xdata][0],dt.datetime) or \
                    isinstance(self.data[xdata][0],pd.tslib.Timestamp)

        if time_unit == None or date_time == False:
            try:
                slopes = self.data[ydata].diff() / self.data[xdata].diff()
                self.time_unit = time_unit
            except TypeError:
                raise TypeError('Slope calculation cannot be executed, probably due to a \
                non-handlable datatype. Either use the time_unit argument or \
                use timedata of type np.datetime64, dt.datetime or pd.tslib.Timestamp.')
                return None
        elif time_unit == 'sec':
            slopes = self.data[ydata].diff()/ \
                     (self.data[xdata].diff().dt.seconds)
        elif time_unit == 'min':
            slopes = self.data[ydata].diff()/ \
                     (self.data[xdata].diff().dt.seconds / 60)
        elif time_unit == 'hr':
            slopes = self.data[ydata].diff()/ \
                     (self.data[xdata].diff().dt.seconds / 3600)
        elif time_unit == 'd':
            slopes = self.data[ydata].diff()/ \
                     (self.data[xdata].diff().dt.days + \
                     self.data[xdata].diff().dt.seconds / 3600 / 24)
        else :
            raise ValueError('Could not calculate slopes. If you are using \
            time-units to calculate slopes, please make sure you entered a \
            valid time unit for slope calculation (sec, min, hr or d)')

        if xdata == 'index':
            self.data.drop(xdata,axis=1,inplace=True)

        return slopes

    def moving_slope_filter(self,xdata,data_name,cutoff,arange,time_unit=None,
                            clear=False,inplace=False,log_file=None,plot=False,
                            final=False):
        """CONFIRMED
        Filters out datapoints based on the difference between the slope in one
        point and the next (sudden changes like noise get filtered out), based
        on a given cut off value. Replaces the dropped values with NaN values.

        Parameters
        ----------
        xdata : str
            name of the column containing the xdata for slope calculation
            (e.g. time). If 'index', the index is used as xdata. If datetime
            objects, a time_unit is expected to calculate the slopes.
        data_name : str
            name of the column containing the data that needs to be filtered
        cutoff: int
            the cutoff value to compare the slopes with to apply the filtering.
        arange : array of two values
            the range within which the moving slope filter needs to be applied 
        time_unit : str
            time unit to be used for the slope calculation (in case this is
            based on time); if None, slopes are calculated based on the values
            given
        clear : bool
            if True, the tags added to datapoints before will be removed and put
            back to 'original'.
        inplace : bool
            indicates whether a new dataframe is created and returned or whether
            the operations are executed on the existing dataframe (nothing is
            returned)
        log_file : str
            string containing the directory to a log file to be written out
            when using this function
        plot : bool
            if true, a plot is made, comparing the original dataset with the
            new, filtered dataset
        final : bool
            if true, the values are actually replaced with nan values (either
            inplace or in a new hp object)

        Returns
        -------
        HydroData object (if inplace=False)
            the dataframe from which the double values of 'data' are removed
        None (if inplace=True)

        Creates
        -------
        A new column in the self.meta_valid dataframe, containing a mask indicating
        what values are filtered
        """
        self._plot = 'valid'
        try:
            len_orig = self.data[data_name][arange[0]:arange[1]].count()
        except TypeError:
            raise TypeError("Slicing not possible for index type " + \
            str(type(self.data.index[0])) + " and arange argument type " + \
            str(type(arange[0])) + ". Try changing the type of the arange " + \
            "values to one compatible with " + str(type(self.data.index[0])) + \
            " slicing.")
        
        #if plot == True:
        #    original = self.__class__(self.data.copy(),timedata_column=self.timename,
        #                              experiment_tag=self.tag,time_unit=self.time_unit)
        # Make temporary object for operations
        df_temp = self.__class__(self.data[arange[0]:arange[1]].copy(),
                                 timedata_column=self.timename,experiment_tag=self.tag,
                                 time_unit=self.time_unit)
        # Update the index of self.meta_valid
        if clear:
            self._reset_meta_valid(data_name)
        self.meta_valid = self.meta_valid.reindex(self.index(),fill_value='!!')

        # Calculate slopes and drop values in temporary object
        slopes = df_temp.calc_slopes(xdata,data_name,time_unit=time_unit)
        if slopes is None:
            return None
        while abs(slopes).max() > cutoff:
            df_temp.data[data_name] = df_temp.data[data_name].drop(slopes[abs(slopes) > cutoff].index)
            slopes = df_temp.calc_slopes(xdata,data_name,time_unit=time_unit)
        len_new = df_temp.data[data_name].count()
        if log_file == None:
            _print_removed_output(len_orig,len_new,'moving slope filter')
        elif type(log_file) == str:
            _log_removed_output(log_file,len_orig,len_new,'filtered')
        else :
            raise TypeError('Please provide the location of the log file as '+ \
                            'a string type, or leave the argument if no log '+ \
                            'file is needed.')
        # Create new temporary object, where the dropped datapoints are replaced
        # by nan values
        df_temp_2 = self.__class__(self.data.copy(),
                                   timedata_column=self.timename,experiment_tag=self.tag,
                                   time_unit=self.time_unit)
        df_temp_2.data[data_name] = df_temp.data[data_name]
        df_temp_2._update_time()
        # Update the self.meta_valid dataframe, to contain False values for dropped
        # datapoints and for datapoints already filtered. This is done by
        # tracking the nan values in df_temp_2
        if data_name in self.meta_valid.columns:
            temp_1 = self.meta_valid[data_name].isin(['filtered'])
            temp_2 = np.where(np.isnan(df_temp_2.data[data_name]),True,False)
            temp_3 = temp_1 | temp_2
            self.meta_valid[data_name] = np.where(temp_3,'filtered','original')
        else:
            self.meta_valid[data_name] = np.isnan(df_temp_2.data[data_name])
            self.meta_valid[data_name] = np.where(self.meta_valid[data_name],'filtered','original')

        if plot == True:
            self.plot_analysed(data_name)

        if final:
            if inplace:
                self.data[data_name] = df_temp_2.data[data_name]
                self._update_time()
            elif not inplace:
                return df_temp_2

        if not final:
            return None

    def simple_moving_average(self,arange,window,data_name=None,inplace=False,
                              plot=True):
        """CONFIRMED
        Calculate the Simple Moving Average of a dataseries from a dataframe,
        using a window within which the datavalues are averaged.

        Parameters
        ----------
        arange : array of two values
            the range within which the moving average needs to be calculated
        window : int
            the number of values from the dataset that are used to take the
            average at the current point. Defaults to 10
        data_name : str or array of str
            name of the column(s) containing the data that needs to be
            smoothened. If None, smoothened data is computed for the whole
            dataframe. Defaults to None
        inplace : bool
            indicates whether a new dataframe is created and returned or whether
            the operations are executed on the existing dataframe (nothing is
            returned)
        plot : bool
            if True, a plot is given for comparison between original and smooth
            data

        Returns
        -------
        HydroData (or subclass) object
            either a new object (inplace=False) or an adjusted object, con-
            taining the smoothened data values
        """
        try:
            original = self.data[arange[0]:arange[1]].copy()
        except TypeError:
            raise TypeError("Slicing not possible for index type " + \
            str(type(self.data.index[0])) + " and arange argument type " + \
            str(type(arange[0])) + ". Try changing the type of the arange " + \
            "values to one compatible with " + str(type(self.data.index[0])) + \
            " slicing.")
        
        if len(original) < window:
            raise ValueError("Window width exceeds number of datapoints!")

        if plot == True:
            original = self.__class__(self.data[arange[0]:arange[1]].copy(),
                                      timedata_column=self.timename,experiment_tag=self.tag,
                                      time_unit=self.time_unit)

        if inplace == False:
            df_temp = self.__class__(self.data[arange[0]:arange[1]].copy(),
                                     timedata_column=self.timename, experiment_tag=self.tag,
                                     time_unit=self.time_unit)
            if data_name == None:
                df_temp = self.data.rolling(window=window,center=True).mean()
            elif isinstance(data_name,str):
                df_temp.data[data_name] = self.data[data_name].interpolate().\
                                        rolling(window=window,center=True).mean()
            else:
                for name in data_name:
                    df_temp.data[name] = self.data[name].interpolate().\
                                        rolling(window=window,center=True).mean()

        elif inplace == True:
            if data_name == None:
                self.data = self.data.rolling(window=window,center=True).mean()
            elif isinstance(data_name,str):
                self.data[data_name] = self.data[data_name].interpolate().\
                                        rolling(window=window,center=True).mean()
            else:
                for name in data_name:
                    self.data[name] = self.data[name].interpolate().\
                                        rolling(window=window,center=True).mean()
        if plot == True:
            fig = plt.figure(figsize=(16,6))
            ax = fig.add_subplot(111)
            ax.plot(original.time,original.data[data_name],'r--',label='original data')
            if inplace == False:
                ax.plot(df_temp.time,df_temp.data[data_name],'b-',label='averaged data')
            elif inplace is True:
                ax.plot(self.time,self.data[data_name],'b-',label='averaged data')
            ax.legend(fontsize=16)
            ax.set_xlabel(self.timename,fontsize=14)
            ax.set_ylabel(data_name,fontsize=14)
            ax.tick_params(labelsize=15)

        if inplace == False:
            return df_temp

    def moving_average_filter(self,data_name,window,cutoff_frac,arange,clear=False,
                              inplace=False,log_file=None,plot=False,final=False):
        """
        Filters out the peaks/outliers in a dataset by comparing its values to a
        smoothened representation of the dataset (Moving Average Filtering). The
        filtered values are replaced by NaN values.

        Parameters
        ----------
        data_name : str
            name of the column containing the data that needs to be filtered
        window : int
            the number of values from the dataset that are used to take the
            average at the current point.
        cutoff_frac: float
            the cutoff value (in fraction 0-1) to compare the data and smoothened
            data: a deviation higher than a certain percentage drops the data-
            point.
        arange : array of two values
            the range within which the moving average filter needs to be applied
        clear : bool
            if True, the tags added to datapoints before will be removed and put
            back to 'original'.
        inplace : bool
            indicates whether a new dataframe is created and returned or whether
            the operations are executed on the existing dataframe (nothing is
            returned)
        log_file : str
            string containing the directory to a log file to be written out
            when using this function
        plot : bool
            if true, a plot is made, comparing the original dataset with the
            new, filtered dataset
        final : bool
            if true, the values are actually replaced with nan values (either
            inplace or in a new hp object)

        Returns
        -------
        HydroData object (if inplace=False)
            the dataframe from which the double values of 'data' are removed
        None (if inplace=True)
        """
        self._plot = 'valid'
        try:
            len_orig = self.data[data_name][arange[0]:arange[1]].count()
        except TypeError:
            raise TypeError("Slicing not possible for index type " + \
            str(type(self.data.index[0])) + " and arange argument type " + \
            str(type(arange[0])) + ". Try changing the type of the arange " + \
            "values to one compatible with " + str(type(self.data.index[0])) + \
            " slicing.")

        #if plot == True:
        #    original = self.__class__(self.data.copy(),timedata_column=self.timename,
        #                              experiment_tag=self.tag,time_unit=self.time_unit)
        # Make temporary object for operations
        df_temp = self.__class__(self.data[arange[0]:arange[1]].copy(),
                                 timedata_column=self.timename,experiment_tag=self.tag,
                                 time_unit=self.time_unit)
        # Make a hydropy object with the smoothened data
        smooth_data = self.simple_moving_average(arange,window,data_name,inplace=False,
                                                 plot=False)
        # Make a mask by comparing smooth and original data, using the given
        # cut-off percentage
        mask = (abs(smooth_data.data[data_name] - self.data[data_name])/\
                smooth_data.data[data_name]) < cutoff_frac
        
        # Update the index of self.meta_valid
        if clear:
            self._reset_meta_valid(data_name)
        self.meta_valid = self.meta_valid.reindex(self.index(),fill_value=True)

        # Do the actual filtering, based on the mask
        df_temp.data[data_name] = df_temp.data[data_name].drop(df_temp.data[mask==False].index)
        len_new = df_temp.data[data_name].count()
        if log_file == None:
            _print_removed_output(len_orig,len_new,'moving average filter')
        elif type(log_file) == str:
            _log_removed_output(log_file,len_orig,len_new,'filtered')
        else :
            raise TypeError('Please provide the location of the log file as \
                            a string type, or leave the argument if no log \
                            file is needed.')
        # Create new temporary object, where the dropped datapoints are replaced
        # by nan values (by assigning a new column to the original dataframe)
        df_temp_2 = self.__class__(self.data.copy(),timedata_column=self.timename,
                                   experiment_tag=self.tag,time_unit=self.time_unit)
        df_temp_2.data[data_name] = df_temp.data[data_name]
        df_temp_2._update_time()
        # Update the self.meta_valid dataframe, to contain False values for dropped
        # datapoints. This is done by tracking the nan values in df_temp_2
        if data_name in self.meta_valid.columns:
            temp_1 = self.meta_valid[data_name].isin(['filtered'])
            temp_2 = np.where(np.isnan(df_temp_2.data[data_name]),True,False)
            temp_3 = temp_1 | temp_2
            self.meta_valid[data_name] = np.where(temp_3,'filtered','original')
        else:
            self.meta_valid[data_name] = np.isnan(df_temp_2.data[data_name])
            self.meta_valid[data_name] = np.where(self.meta_valid[data_name],'filtered','original')

        if plot:
            self.plot_analysed(data_name)

        if final:
            if inplace:
                self.data[data_name] = df_temp_2.data[data_name]
                self._update_time()
            elif not inplace:
                return df_temp_2

        if not final:
            return None

    def savgol(self,data_name,window=55,polyorder=2,plot=False,inplace=False):
        """
        Uses the scipy.signal Savitzky-Golay filter to smoothen the data of a column;
        The values are either replaced or a new dataframe is returned.

        Parameters
        ----------
        data_name : str
            name of the column containing the data that needs to be filtered
        window : int
            the length of the filter window; default to 55
        polyorder : int
            The order of the polynomial used to fit the samples.
            polyorder must be less than window. default to 1
        plot : bool
            if true, a plot is made, comparing the original dataset with the
            new, filtered dataset
        inplace : bool
            indicates whether a new dataframe is created and returned or whether
            the operations are executed on the existing dataframe (nothing is
            returned)

        Returns
        -------
        HydroData object (if inplace=False)
        None (if inplace=True)
        """
        from scipy import signal

        df_temp = self.__class__(self.data.copy(),timedata_column=self.timename,
                                   experiment_tag=self.tag,time_unit=self.time_unit)

        df_temp.data[data_name] = sp.signal.savgol_filter(self.data[data_name]\
                                    ,window,polyorder)

        if plot:
            fig = plt.figure(figsize=(16,6))
            ax = fig.add_subplot(111)
            ax.plot(self.time,self.data[data_name],'g--',label='original data')
            ax.plot(self.time,df_temp.data[data_name],'b-',label='filtered data')
            ax.legend(fontsize=16)
            ax.set_xlabel(self.timename,fontsize=20)
            ax.set_ylabel(data_name,fontsize=20)
            ax.tick_params(labelsize=15)

        if inplace:
            self.data[data_name] = df_temp.data[data_name]
        else:
            return df_temp

#==============================================================================
# DATA (COR)RELATION
#==============================================================================
    def calc_ratio(self,data_1,data_2,arange,only_checked=False):
        """
        Given two datasets or -columns, calculates the average ratio between
        the first and second dataset, within the given range. Also the standard
        deviation on this is calculated

        Parameters
        ----------
        data_1 : str
            name of the data column containing the data to be in the numerator
            of the ratio calculation
        data_2 : str
            name of the data column containing the data to be in the denominator
            of the ratio calculation
        arange : array of two values
            the range within which the ratio needs to be calculated
        only_checked : bool
            if 'True', filtered values are excluded; default to 'False'

        Returns
        -------
        The average ratio of the first data column over the second one within
        the given range and including the standard deviation
        """
        # If indexes are in datetime format, convert the arange array to date-
        # time values
        #if isinstance(self.data.index[0],pd.tslib.Timestamp):
        #    arange = [(self.data.index[0] + dt.timedelta(arange[0]-1)),
        #              (self.data.index[0] + dt.timedelta(arange[1]-1))]

        try:
            self.data.loc[arange[0]:arange[1]]
        except TypeError:
            raise TypeError("Slicing not possible for index type " + \
            str(type(self.data.index[0])) + " and arange argument type " + \
            str(type(arange[0])) + ". Try changing the type of the arange " + \
            "values to one compatible with " + str(type(self.data.index[0])) + \
            " slicing.")

        mean = (self.data[data_1]/self.data[data_2])[arange[0]:arange[1]]\

        if arange[0] < self.index()[0] or arange[1] > self.index()[-1]:
            raise IndexError('Index out of bounds. Check whether the values of ' + \
            '"arange" are within the index range of the data.')

        if only_checked == True:
            #create new pd.Dataframes for original values in range,
            #merge only rows in which both values are original
            data_1_checked = pd.DataFrame(self.data[arange[0]:arange[1]][data_1][self.meta_valid[data_1]=='original'].values,
                    index=self.data[arange[0]:arange[1]][data_1][self.meta_valid[data_1]=='original'].index)
            data_2_checked = pd.DataFrame(self.data[arange[0]:arange[1]][data_2][self.meta_valid[data_2]=='original'].values, \
                    index=self.data[data_2][arange[0]:arange[1]][self.meta_valid[data_2]=='original'].index)
            ratio_data = pd.merge(data_1_checked,data_2_checked,left_index=True, right_index=True, how = 'inner')
            ratio_data.columns = data_1,data_2

            mean = (ratio_data[data_1]/ratio_data[data_2])\
                    .replace(np.inf,np.nan).mean()
            std = (ratio_data[data_1]/ratio_data[data_2])\
                    .replace(np.inf,np.nan).std()

        else:
            mean = (self.data[arange[0]:arange[1]][data_1]/self.data[arange[0]:arange[1]][data_2])\
                                                .replace(np.inf,np.nan).mean()
            std = (self.data[arange[0]:arange[1]][data_1]/self.data[arange[0]:arange[1]][data_2])\
                                                .replace(np.inf,np.nan).std()

        #print('mean : '+str(mean)+ '\n' +'standard deviation : '+str(std))
        return mean,std

    def compare_ratio(self,data_1,data_2,arange,only_checked=False):
        """
        Compares the average ratios of two datasets in multiple different ranges
        and returns the most reliable one, based on the standard deviation on
        the ratio values

        Parameters
        ----------
        data_1 : str
            name of the data column containing the data to be in the numerator
            of the ratio calculation
        data_2 : str
            name of the data column containing the data to be in the denominator
            of the ratio calculation
        arange : int
            the range (in days) for which the ratios need to be calculated and
            compared
        only_checked : bool
            if 'True', filtered values are excluded; default to 'False'

        Returns
        -------
        The average ratio within the range that has been found to be the most
        reliable one
        """
        # Make the array with ranges within which to compute ratios, based on
        # arange, indicating what the interval should be.
        if isinstance(self.data.index[0],pd.tslib.Timestamp):
            days = [self.index()[0] + dt.timedelta(arange) * x for x in \
                    range(0, int((self.index()[-1]-self.index()[0]).days/arange))]
            starts = [[y] for y in days]
            ends = [[x + dt.timedelta(arange)] for x in days]
            #end = (self.data.index[-1] - self.data.index[0]).days+1

        elif isinstance(self.data.index[0],float):
            end = int(self.index()[-1]+1) # +1 because int rounds downwards
            starts = [[y] for y in range(0,end)]
            ends = [[x] for x in range(arange,end+arange)]

        ranges = np.append(starts,ends,1)
        rel_std = np.inf

        for r in range(0,len(ranges)):
            average,stdev = self.calc_ratio(data_1,data_2,ranges[r],only_checked)
            try:
                relative_std = stdev/average
                if relative_std < rel_std:
                    std = stdev
                    avg = average
                    index = r
                    rel_std = std/avg
            except (ZeroDivisionError):
                pass

        print('Best ratio (' + str(avg) + ' ± ' + str(std) + \
        ') was found in the range: ' + str(ranges[index]))

        return avg,std

    def get_correlation(self,data_1,data_2,arange,zero_intercept=False,
                        only_checked=False,plot=False):
        """
        Calculates the linear regression coefficients that relate data_1 to
        data_2

        Parameters
        ----------
        data_1 and data_2 : str
            names of the data columns containing the data between which the
            correlation will be calculated.
        arange : array
            array containing the beginning and end value between which the
            correlation needs to be calculated
        zero_intercept : bool
            indicates whether or not to assume a zero-intercept
        only_checked: bool
            if 'True', filtered values are excluded from calculation and plotting;
            default to 'False'
            if a value in one column is filtered, the corresponding value in the second
            column also gets excluded!

        Returns
        -------
        the linear regression coefficients of the correlation, as well as the
        r-squared -value
        """
        # If indexes are in datetime format, and arange values are not,
        # convert the arange array to datetime values
        if isinstance(self.data.index[0],pd.tslib.Timestamp) and \
        isinstance(arange[0],int) or isinstance(arange[0],float):
            wn.warn('Replacing arange values, assumed to be relative time' + \
            ' values, with absolute values of type dt.datetime')
            arange = [(self.data.index[0] + dt.timedelta(arange[0]-1)),
                      (self.data.index[0] + dt.timedelta(arange[1]-1))]

        #if arange[0] < self.time[0] or arange[1] > self.time[-1]:
        #    raise IndexError('Index out of bounds. Check whether the values of '+ \
        #    '"arange" are within the index range of the data.')

        self.data = self.data.sort_index()
        if only_checked:
            #create new pd.Dataframes for original values in range,
            #merge only rows in which both values are original
            data_1_checked = pd.DataFrame(self.data[data_1][arange[0]:arange[1]][self.meta_valid[data_1]=='original'].values,
                    index=self.data[data_1][arange[0]:arange[1]][self.meta_valid[data_1]=='original'].index)
            data_2_checked = pd.DataFrame(self.data[data_2][arange[0]:arange[1]][self.meta_valid[data_2]=='original'].values,
                    index=self.data[data_2][arange[0]:arange[1]][self.meta_valid[data_2]=='original'].index)
            corr_data = pd.merge(data_1_checked,data_2_checked,left_index=True, right_index=True, how = 'inner')

        else:
            corr_data = pd.DataFrame(self.data[arange[0]:arange[1]][[data_1,data_2]].values)
            
        corr_data.columns = data_1,data_2
        corr_data = corr_data[[data_1,data_2]].dropna()

        if zero_intercept == True:
            import statsmodels.api as sm
            model = sm.OLS(corr_data[data_1],corr_data[data_2])
            results = model.fit()
            slope = results.params[data_2]
            intercept = 0.0
            r_sq = results.rsquared

        else:
            regres = self.data[[data_1,data_2]][arange[0]:arange[1]].dropna()
            slope, intercept, r_value, p_value, std_err = sp.stats.linregress(regres)
            r_sq = r_value**2

        if plot:
            x = np.arange(self.data[data_1][arange[0]:arange[1]].min(),
                          self.data[data_1][arange[0]:arange[1]].max())
            y = slope * x + intercept
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(111)
            ax.plot(self.data[data_2][arange[0]:arange[1]],
                    self.data[data_1][arange[0]:arange[1]],'bo',markersize=4,
                   label='Data')
            ax.plot(y,x,label='Linear fit')
            ax.legend(fontsize=15)
            ax.tick_params(labelsize=15)
            ax.set_ylabel(data_1,size=17)
            ax.set_xlabel(data_2,size=17)
            #fig.text(1,0.9,'Slope: '+str(slope) + '\nIntercept: '+str(intercept)+'\nR$^2$: '+str(r_sq),color='black',verticalalignment='bottom', bbox={'edgecolor':'black','pad':10,'fill':False}, horizontalalignment='left',fontsize=17)
            fig.tight_layout()
            print('slope: ' + str(slope) + ' intercept: ' + str(intercept) + ' R2: ' + str(r_sq))
            
        return slope,intercept,r_sq

#==============================================================================
# DAILY PROFILE CALCULATION
#==============================================================================

    def calc_daily_profile(self,column_name,arange,quantile=0.9,plot=False,
                           plot_method='quantile',clear=False,only_checked=False):
        """
        Calculates a typical daily profile based on data from the indicated
        consecutive days. Also saves this average day, along with standard
        deviation and lower and upper percentiles as given in the arguments.
        Plotting is possible.

        Parameters
        ----------
        column_name : str
            name of the column containing the data to calculate an average day
            for
        arange : 2-element array of ints
            contains the beginning and end day of the period to use for average
            day calculation
        quantile : float between 0 and 1
            value to use for the calculation of the quantiles
        plot : bool
            plot or not
        plot_method : str
            method to use for plotting. Available: "quantile" or "stdev"
        clear : bool
            wether or not to clear the key in the self.daily_profile dictionary
            that is already present

        Returns
        ------
        None; creates a dictionary self.daily_profile containing information
        on the average day as calculated.
        """
        # several checks to make sure the right types, columns... are used
        try:
            if not isinstance(self.daily_profile,dict):
                self.daily_profile = {}
        except AttributeError:
            self.daily_profile = {}

        if clear:
            try:
                self.daily_profile.pop(column_name, None)
            except KeyError:
                pass

        if column_name in self.daily_profile.keys():
            raise KeyError('self.daily_profile dictionary already contains a ' +\
            'key ' + column_name + '. Set argument "clear" to True to erase the ' + \
            'key and create a new one.')

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
        except AttributeError:
            raise AttributeError('OnlineSensorBased instance has no attribute "highs". '+\
            'run .get_highs to tag the peaks in the dataset.')

        if rain :
            wn.warn('Data points obtained during a rain event will be used for' + \
            ' the calculation of an average day. This might lead to a not-' + \
            'representative average day and/or high standard deviations.')

        daily_profile = pd.DataFrame()

        if not isinstance(arange[0],int) and not isinstance(arange[0],dt.datetime):
            raise TypeError('The values of arange must be of type int or dt.datetime')

        if isinstance(self.data.index[0],dt.datetime):
            range_days = pd.date_range(arange[0],arange[1])
            indexes = [self.data.index[0],self.data.index[0]+dt.timedelta(1)]
        else :
            range_days = range(arange[0],arange[1])
            indexes = [0,1]
        #if isinstance(arange[0],dt.datetime):
        #    range_days = pd.date_range(arange[0],arange[1])


        #if only_checked:
        #    for i in range_days:
        #        daily_profile = pd.merge(daily_profile,
        #                        pd.DataFrame(self.data[column_name][i:i+1]\
        #                        [self.meta_valid[column_name]=='original'].values),
        #                        left_index=True, right_index=True,how='outer')
        #    mean_day = pd.DataFrame(index=daily_profile.index)
        #    self.data.loc[indexes[0]:indexes[1]].index)#\
        #    [self.meta_valid[column_name]=='original'].index)
        #    if isinstance(self.data.index[0],dt.datetime):
        #        mean_day.index = mean_day.index.time
        #else:


        if only_checked and column_name in self.meta_valid:
            for i in range_days:
                if isinstance(i,dt.datetime) or isinstance(i,np.datetime64) or isinstance(i,pd.tslib.Timestamp):
                    name = str(i.month) + '-' + str(i.day)
                else:
                    name = str(i)
                mask_valid = pd.DataFrame((self.meta_valid[column_name][i:i+1] == 'original').values,columns=[name])
                daily_profile = pd.merge(daily_profile,
                                         pd.DataFrame(self.data[column_name][i:i+1].values,
                                                      columns=[name]).where(mask_valid),
                                         left_index=True, right_index=True,how='outer')
        else:
            if only_checked:
                wn.warn('No values of selected column were filtered yet. All values '+ \
                'will be displayed.')
            for i in range_days:
                if isinstance(i,dt.datetime) or isinstance(i,np.datetime64) or isinstance(i,pd.tslib.Timestamp):
                    name = str(i.month) + '-' + str(i.day)
                else:
                    name = str(i)
                daily_profile = pd.merge(daily_profile,
                                         pd.DataFrame(self.data[column_name][i:i+1].values,
                                                      columns=[name]),
                                         left_index=True, right_index=True,how='outer')

        daily_profile['index'] = self.data.loc[indexes[0]:indexes[1]].index.time
        daily_profile = daily_profile.drop_duplicates(subset='index', keep='first')\
                                     .set_index('index').sort_index()

        mean_day = pd.DataFrame(index=daily_profile.index.values)
        mean_day['avg'] = daily_profile.mean(axis=1).values
        mean_day['std'] = daily_profile.std(axis=1).values
        mean_day['Qupper'] = daily_profile.quantile(quantile,axis=1).values
        mean_day['Qlower'] = daily_profile.quantile(1-quantile,axis=1).values

        self.daily_profile[column_name] = mean_day

        if plot:
            fig = plt.figure(figsize=(10,6))
            ax = fig.add_subplot(111)
            ax.plot(mean_day.index,mean_day['avg'],'g')
            if plot_method == 'quantile':
                ax.plot(mean_day.index,mean_day['Qupper'],'b',alpha=0.5)
                ax.plot(mean_day.index,mean_day['Qlower'],'b',alpha=0.5)
                ax.fill_between(mean_day.index,mean_day['avg'],mean_day['Qupper'],
                            color='grey', alpha=0.3)
                ax.fill_between(mean_day.index,mean_day['avg'],mean_day['Qlower'],
                            color='grey', alpha=0.3)
            elif plot_method == 'stdev':
                ax.plot(mean_day.index,mean_day['avg']+mean_day['std'],'b',alpha=0.5)
                ax.plot(mean_day.index,mean_day['avg']-mean_day['std'],'b',alpha=0.5)
                ax.fill_between(mean_day.index,mean_day['avg'],
                                mean_day['avg']+mean_day['std'],
                                color='grey', alpha=0.3)
                ax.fill_between(mean_day.index,mean_day['avg'],
                                mean_day['avg']-mean_day['std'],
                                color='grey', alpha=0.3)
            ax.tick_params(labelsize=15)
            ax.set_xlim(mean_day.index[0],mean_day.index[-1])
            ax.set_ylabel(column_name,size=17)
            ax.set_xlabel('Time',size=17)
            return fig,ax

    ##############
    ### PLOTTING
    ##############
    def plot_analysed(self,data_name,time_range='default',only_checked = False):

        """
        plots the values and their types (original, filtered, filled) \
        of a given column in the given time range.

        Parameters
        ----------
        data_name : str
            name of the column containing the data to plot
        time_range : array of two values
            the range within which the values are plotted; default is all
        only_checked : bool
            if 'True', filtered values are excluded; default to 'False'

        Returns
        -------
        Plot
        """

        # time range settings
        if time_range == 'default':
            if isinstance(self.time[0],float):
                time_range = [int(self.time[0]),int(self.time[-1])+1]
            elif isinstance(self.time[0],dt.datetime):
                time_range = [self.time[0],self.time[-1]]
        else:
            if not isinstance(time_range[0],type(self.time[0])) or not \
            isinstance(time_range[1],type(self.time[-1])):
                raise TypeError('The value type of the values in time_range must ' + \
                'be the same as the value type of index values')

            if time_range[0] < self.time[0] or time_range[1] > int(self.time[-1]):
                raise IndexError('Index out of bounds. Check whether the values of '+\
                '"time_range" are within the index range of the data.')

        fig = plt.figure(figsize=(16,6))
        ax = fig.add_subplot(111)

        #create new object with only the values within the given time range
        df = self.__class__(self.data[time_range[0]:time_range[1]].copy(),timedata_column=self.timename,
                            experiment_tag=self.tag,time_unit=self.time_unit)

        if self._plot == 'filled':
            df.meta_filled = self.meta_filled[time_range[0]:time_range[1]].copy()
            df.filled = self.filled[time_range[0]:time_range[1]].copy()
            ax.plot(df.time[df.meta_filled[data_name]=='original'],
                df.data[data_name][df.meta_filled[data_name]=='original'],
                '.g',label='original')
            if only_checked == False:
                if (df.meta_filled[data_name]=='filtered').any():
                    ax.plot(df.time[df.meta_filled[data_name]=='filtered'],
                            df.data[data_name][df.meta_filled[data_name]=='filtered'],
                            '.r',label='filtered')
            if (df.meta_filled[data_name]=='filled_interpol').any():
                ax.plot(df.time[df.meta_filled[data_name]=='filled_interpol'],
                        df.filled[data_name][df.meta_filled[data_name]=='filled_interpol'],
                        '.b',label='filled (interpolation)')
            if (df.meta_filled[data_name]=='filled_ratio').any():
                ax.plot(df.time[df.meta_filled[data_name]=='filled_ratio'],
                        df.filled[data_name][df.meta_filled[data_name]=='filled_ratio'],
                        '.m',label='filled (ratio-based)')
            if (df.meta_filled[data_name]=='filled_correlation').any():
                ax.plot(df.time[df.meta_filled[data_name]=='filled_correlation'],
                        df.filled[data_name][df.meta_filled[data_name]=='filled_correlation'],
                        '.k',label='filled (correlation-based)')
            if (df.meta_filled[data_name]=='filled_average_profile').any():
                ax.plot(df.time[df.meta_filled[data_name]=='filled_average_profile'],
                        df.filled[data_name][df.meta_filled[data_name]=='filled_average_profile'],
                        '.y',label='filled (typical day)')
            if (df.meta_filled[data_name]=='filled_infl_model').any():
                ax.plot(df.time[df.meta_filled[data_name]=='filled_infl_model'],
                        df.filled[data_name][df.meta_filled[data_name]=='filled_infl_model'],
                        '.c',label='filled (influent model)')
            if (df.meta_filled[data_name]=='filled_profile_day_before').any():
                ax.plot(df.time[df.meta_filled[data_name]=='filled_profile_day_before'],
                        df.filled[data_name][df.meta_filled[data_name]=='filled_profile_day_before'],
                        '.',label='filled (previous day)')
            #if (df.meta_filled[data_name]=='filled_savitzky_golay').any():
            #    ax.plot(df.time[df.meta_filled[data_name]=='filled_savitzky_golay'],
            #            df.filled[data_name][df.meta_filled[data_name]=='filled_savitzky_golay'],
            #            '.m',label='filled (Savitzky-Golay filter)')

        elif self._plot == 'valid':
            df.meta_valid = self.meta_valid[time_range[0]:time_range[1]].copy()
            ax.plot(df.time[self.meta_valid[data_name]=='original'],
                df.data[data_name][df.meta_valid[data_name]=='original'],
                '.g',label='original')
            if only_checked == False:
                if (df.meta_valid[data_name]=='filtered').any():
                    if data_name in df.filled.columns:
                        ax.plot(df.time[df.meta_valid[data_name]=='filtered'],
                                df.filled[data_name][df.meta_valid[data_name]=='filtered'],
                            '.r',label='filtered')
                    else:
                        ax.plot(df.time[df.meta_valid[data_name]=='filtered'],
                                df.data[data_name][df.meta_valid[data_name]=='filtered'],
                            '.r',label='filtered')
            print (str(float(df.meta_valid.groupby(data_name).size()['original']*100)/ \
                float(df.meta_valid[data_name].count())) + \
                '% datapoints are left over from the original ' + \
                str(float(df.meta_valid[data_name].count())))

        ax.legend(bbox_to_anchor=(1.05,1),loc=2,fontsize=16)
        ax.set_xlabel(self.timename,fontsize=20)
        ax.set_ylabel(data_name,fontsize=20)
        ax.tick_params(labelsize=14)


        return fig, ax

#    def plot_analysed(self,data_name):
#        """
#
#        """
#        fig = plt.figure(figsize=(16,6))
#        ax = fig.add_subplot(111)
#
#        if not self._plot == 'filled' or self._plot == 'valid':
#            ValueError('No filtering or filling of the current dataset has been done.\
#                        Run any filter or filling function to start the data analysis.')
#
#        if self._plot == 'filled':
#            ax.plot(self.time[self.meta_filled[data_name]=='original'],
#                self.data[data_name][self.meta_filled[data_name]=='original'],
#                '.g',label='original')
#            if (self.meta_filled[data_name]=='filtered').any():
#                ax.plot(self.time[self.meta_filled[data_name]=='filtered'],
#                        self.data[data_name][self.meta_filled[data_name]=='filtered'],
#                        '.r',label='filtered')
#            if (self.meta_filled[data_name]=='filled_interpol').any():
#                ax.plot(self.time[self.meta_filled[data_name]=='filled_interpol'],
#                        self.filled[data_name][self.meta_filled[data_name]=='filled_interpol'],
#                        '.b',label='filled (interpolation)')
#            if (self.meta_filled[data_name]=='filled_ratio').any():
#                ax.plot(self.time[self.meta_filled[data_name]=='filled_ratio'],
#                        self.filled[data_name][self.meta_filled[data_name]=='filled_ratio'],
#                        '.m',label='filled (ratio-based)')
#            if (self.meta_filled[data_name]=='filled_correlation').any():
#                ax.plot(self.time[self.meta_filled[data_name]=='filled_correlation'],
#                        self.filled[data_name][self.meta_filled[data_name]=='filled_correlation'],
#                        '.k',label='filled (correlation-based)')
#            if (self.meta_filled[data_name]=='filled_average_profile').any():
#                ax.plot(self.time[self.meta_filled[data_name]=='filled_average_profile'],
#                        self.filled[data_name][self.meta_filled[data_name]=='filled_average_profile'],
#                        '.y',label='filled (typical day)')
#            if (self.meta_filled[data_name]=='filled_infl_model').any():
#                ax.plot(self.time[self.meta_filled[data_name]=='filled_infl_model'],
#                        self.filled[data_name][self.meta_filled[data_name]=='filled_infl_model'],
#                        '.c',label='filled (influent model)')
#
#        elif self._plot == 'valid':
#            ax.plot(self.time[self.meta_valid[data_name]=='original'],
#                self.data[data_name][self.meta_valid[data_name]=='original'],
#                '.g',label='original')
#            if (self.meta_valid[data_name]=='filtered').any():
#                if data_name in self.filled.columns:
#                    ax.plot(self.time[self.meta_valid[data_name]=='filtered'],
#                        self.filled[data_name][self.meta_valid[data_name]=='filtered'],
#                        '.r',label='filtered')
#                else:
#                    ax.plot(self.time[self.meta_valid[data_name]=='filtered'],
#                        self.data[data_name][self.meta_valid[data_name]=='filtered'],
#                        '.r',label='filtered')
#
#        ax.legend(fontsize=16)
#        ax.set_xlabel(self.timename,fontsize=14)
#        ax.set_ylabel(data_name,fontsize=14)
#        ax.tick_params(labelsize=14)
#
#        print str(float(self.meta_valid.groupby(data_name).size()['original']*100)/ \
#                float(self.meta_valid[data_name].count())) + \
#                '% datapoints are left over from the original ' + \
#                str(float(self.meta_valid[data_name].count()))
#        return fig, ax


##############################
###   NON-CLASS FUNCTIONS  ###
##############################

def total_seconds(timedelta_value):
    return timedelta_value.total_seconds()

def _print_removed_output(original,new,function):
    """
    function printing the output of functions that tag datapoints.

    Parameters
    ----------
    original : int
        original length of the dataset
    new : int
        length of the new dataset
    function : str
        info on the function used to filter the data

    """
    print(str(original-new) + ' values detected and tagged as filtered by function ' + function)

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

# Prepends a WEST-header to read-in text files, to make them WEST compatible
def _prepend_WEST_header(filepath,sep,column_names,outputfilename,
                         comment='no comments'):
        """
        """
        f = open(filepath,'r')
        columns = f.readlines()
        temp = f.readlines()[1:]
        f.close()

        f = open(outputfilename, 'w')
        #f.write("%%Version3.3\ %%BeginComment\ ")
        #f.write(comment)
        #f.write("%%EndComment\ %%BeginHeader\ ")
        #f.write(str())#write the names
        #f.write(str())#write the units
        f.write(temp)
        f.close()
