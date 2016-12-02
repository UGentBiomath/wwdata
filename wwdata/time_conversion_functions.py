# -*- coding: utf-8 -*-
"""
    time_conversion_functions provides functionalities for converting certain types of time data to other types, in the context of the wwdata package.
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

import pandas as pd
import datetime as dt


def make_month_day_array():
    """
    makes a dataframe containing two columns, one with the number of the month,
    one with the day of the month. Useful in creating datetime objects based on
    e.g. date serial numbers from the Window Date System 
    (http://excelsemipro.com/2010/08/date-and-time-calculation-in-excel/)
    
    Returns
    -------
    pd.DataFrame : 
        dataframe with number of the month and number of the day of the month 
        for a whole year
    """
    days_in_months = [31,28,31,30,31,30,31,31,30,31,30,31]
    days = []
    months = []
    month = 1
    for i in days_in_months:
        for j in range(1,i+1):
            days.append(j)
            months.append(month)
        month += 1
    
    month_day_array = pd.DataFrame()
    month_day_array['month'] = months
    month_day_array['day'] = days
    
    return month_day_array        

def get_absolute_time(value,date_type='WindowsDateSystem'):
    """
    Converts a coded time to the absolute date at which the experiment was 
    conducted.
    
    Parameters
    ----------
    value : int or float
        a code for a certain point in time
    date_type : str
        type of coding used for the time point, probably depending on the 
        software which was used to acquire the data, e.g. the Windows Date 
        System (here as default, see also:
        http://excelsemipro.com/2010/08/date-and-time-calculation-in-excel/)
    
    Returns
    -------
    datetime.datetime
        python datetime object
    """
    if date_type == 'WindowsDateSystem':
        year_from_1900 = value / 365.25
        year_current = int(1900 + year_from_1900)
        decimals = year_from_1900 - int(year_from_1900)
        day_in_year = int(365.25*decimals)
        months_days = make_month_day_array()
        month = months_days['month'][day_in_year]
        day_in_month = months_days['day'][day_in_year]
        decimals = 365.25*decimals - day_in_year
        hour = int(24*decimals)
        decimals = 24*decimals - hour
        minutes = int(60*decimals)
        decimals = 60*decimals - minutes
        seconds = int(60*decimals)
    
        timestamp = dt.datetime(year_current,month,day_in_month,hour,minutes,
                                seconds)
    return timestamp
   
def to_days(timedelta):
    """
    timedelta : timedelta value
    """
    return float(timedelta.days) + float(timedelta.seconds)/(24*60*60)

def to_hours(timedelta):
    """
    timedelta : timedelta value
    """
    return float(timedelta.days) * 24 + float(timedelta.seconds) / (60*60)

def to_minutes(timedelta):
    """
    timedelta : timedelta value
    """
    return float(timedelta.days) * 24 * 60 + float(timedelta.seconds) / 60

def to_seconds(timedelta):
    """
    timedelta : timedelta value
    """
    return float(timedelta.days) * 24 * 60 * 60 + float(timedelta.seconds)


def timedelta_to_abs(timedelta,unit='d'):
    """
    timedelta : array of timedelta values
    """
    if unit == 'd':
        return map(to_days,timedelta) 
    elif unit == 'hr':
        return map(to_hours,timedelta) 
    elif unit == 'min':
        return map(to_minutes,timedelta) 
    elif unit == 'sec':
        return map(to_seconds,timedelta) 
    
def _get_datetime_info(string):
    """
    
    parameter
    --------
    string containing date and time info (format as received from EHV)
    """
    array = string.split()
    date = array[0].split("-")
    time = array[1].split(":")
    return date + time

def make_datetime(array):
    """
    parameter
    --------
    array with elements
        0: year (yy)
        1: month (mm)
        2: day in month (dd)
        3: hour (h or hh)
        4: minutes (minmin)
    """
    array[2] = '20' + array[2]
    return dt.datetime(int(array[2]),int(array[1]),int(array[0]),
                       int(array[3]),int(array[4]))

def to_datetime_singlevalue(time):
    """
    In case timedata is in a string format, to convert it to a datetime object,
    it needs to be in the right format, e.g. dd-mm-yyyy hh:mm:ss (so two of each)
    This function takes care of that, to a certain extent.

    """
    time_info = _get_datetime_info(time) 
    return make_datetime(time_info)
    
