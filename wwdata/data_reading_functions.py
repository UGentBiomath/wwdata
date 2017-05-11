# -*- coding: utf-8 -*-
"""
    data_reading_functions provides functionalities for data reading in the context of the wwdata package.
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
import xlrd

def list_files(path,ext):
    """
    
    """
    if ext == 'excel':
        files = [f for f in listdir(path) if '.xls' in f]
    elif ext == 'text':
        files = [f for f in listdir(path) if f.endswith('.txt')]
    elif ext == 'csv':
        files = [f for f in listdir(path) if f.endswith('.csv')]
    else:
        print('No files with',ext,'extension found in directory',path,'Please \
        choose one of the following: text, excel, csv')
        
        return None
        
    return files

def remove_empty_lines(path,ext):
    """
    """
    files = list_files(path,ext)
    if not files:
        print('Please provide a directory that contains '+ext+' files.')
        return None
    for filename in files:
        filepath = os.path.join(path,filename)
        data = pd.read_csv(filepath,sep='\t')
        data.dropna(axis=0,inplace=True)
        #filepath_new = os.path.join(path,filename+'_')
        data.to_csv(filepath,sep='\t',index=False,index_label=False)
    return None

def find_and_replace(path,ext,replace):
    """
    Finds the files with a certain extension in a directory and applies a find-
    replace action to those files. Removes the old files and produces files with
    a prefix stating the replacing value.
    
    Parameters
    ----------
    path : str
        the path name of the directory to apply the function to
    ext : str
        the extension of the files to be searched (excel, text or csv)
    replace : array of str
        the first value of replace is the string to be replaced by the second 
        value of replace.
    """
    files = list_files(path,ext)
    if not files:
        print('Please provide a directory that contains '+ext+' files.')
        return None
    for filename in files:
        filepath = os.path.join(path,filename)
        filedata = None
        with open(filepath, 'r') as file :
            filedata = file.read()

        # Replace the target string
        filedata = filedata.replace(replace[0], replace[1])

        # Write the file out again
        with open(filepath, 'w') as file:
            file.write(filedata)
        
        #data = pd.read_csv(filepath,sep='\t')
        #data.replace(to_replace=replace[0],value=replace[1],inplace=True)
        #data.to_csv(filepath,sep='\t',index=False,index_label=False)
    
    return None
    
def sort_data(data,based_on,reset_index=[False,'new_index_name'],
              convert_to_timestamp=[True,'time_name','%d.%m.%Y %H:%M:%S']):
    """
    Sorts a dataset based on values in one of the columns and splits them in 
    different dataframes, returned in the form of one dictionary
    
    Parameters
    ----------
    data : pd.dataframe
        the dataframe containing the data that needs to be sorted
    based_on : str
        the name of the column that contains the names or values the sorting 
        should be based on
    reset_index : [bool,str]
        array indicating if the index of the sorted datasets should be reset to
        a new one; if first element is true, the second element is the title of
        the column to use as new index; default: False
        
    Returns
    -------
    dict : 
        A dictionary of pandas dataframes with as labels those acquired from the 
        based_on column        
    """
    dictionary = {}
    measurement_codes = pd.Series(data[based_on].ravel()).unique()
    for i in measurement_codes:
        dictionary[i] = data[data[based_on]==i].drop(based_on,axis=1)
        if convert_to_timestamp[0] == True & reset_index[0] == True:
            dictionary[i][convert_to_timestamp[1]] = \
            pd.to_datetime(dictionary[i][convert_to_timestamp[1]],
                        format=convert_to_timestamp[2])
            dictionary[i].set_index(reset_index[1],inplace=True)
        elif convert_to_timestamp[0] == True & reset_index[0] == False:
            dictionary[i][convert_to_timestamp[1]] = \
            pd.to_datetime(dictionary[i][convert_to_timestamp[1]],
                        format=convert_to_timestamp[2])
        elif reset_index[0] == True & convert_to_timestamp[0] == False:
            dictionary[i].set_index(reset_index[1],inplace=True)
        print('Sorting',i,'...')
    
    return dictionary
    
def _get_header_length(read_file,ext='text',comment='#'):
    """
    Determines the amount of rows that are part of the header in a file that is
    already opened and readable    
    
    Parameters
    ----------
    read_file : opened file
        an opened file object that is readable
    ext : str
        the extension (in words) of the file the headerlength needs to be found
        for
    comment : str
        comment symbol used in the files 

    Returns
    ------- 
    headerlength : int
        the amount of rows that are part of the header in the read file
    
    """        
    
    headerlength = 0
    header_test = comment
    counter = 0
    if ext == 'excel' or ext == 'zrx':
        while header_test == comment:
            header_test = str(read_file.sheet_by_index(0).cell_value(counter,0))[0]
            headerlength += 1
            counter +=1
            
    elif ext == 'text' or ext == 'csv':
        while header_test == comment:
            header_test = read_file.readline()[0]
            headerlength += 1
     
    return headerlength-1


def read_mat(path):
    """
    Reads in .mat datafiles and returns them as pd.DataFrame
    http://stackoverflow.com/questions/24762122/read-matlab-data-file-into-python-need-to-export-to-csv
    
    """
    
    #Also write separate script for converting all .mat files in one dir to .csv files

def _get_header_length(read_file,ext='text',comment='#'):
    """
    Determines the amount of rows that are part of the header in a file that is
    already opened and readable    
    
    Parameters
    ----------
    read_file : opened file
        an opened file object that is readable
    ext : str
        the extension (in words) of the file the headerlength needs to be found
        for
    comment : str
        comment symbol used in the files 

    Returns
    ------- 
    headerlength : int
        the amount of rows that are part of the header in the read file
    
    """        
    
    headerlength = 0
    header_test = comment
    counter = 0
    if ext == 'excel' or ext == 'zrx':
        while header_test == comment:
            header_test = str(read_file.sheet_by_index(0).cell_value(counter,0))[0]
            headerlength += 1
            counter +=1
            
    elif ext == 'text':
        while header_test == comment:
            header_test = read_file.readline()[0]
            headerlength += 1
     
    return headerlength-1

def _open_file(filepath,ext='text'):
    """
    Opens file of a given extension in readable mode   
    
    Parameters
    ----------
    filepath : str
        the complete path to the file to be opened in read mode
    ext : str
        the extension (in words) of the file that needs to be opened in read 
        mode
    
    Returns
    ------- 
    The opened file in read mode
    
    """        
    if ext == 'text' or ext == 'zrx' or ext == 'csv':
        return open(filepath, 'r')
    elif ext == 'excel':
        return xlrd.open_workbook(filepath)

def _read_file(filepath,ext='text',skiprows=0,sep='\t',encoding='utf8',decimal='.'):
    """
    Read a file of given extension and save it as a pandas dataframe   
    
    Parameters
    ----------
    filepath : str
        the complete path to the file to be read and saved as dataframe
    ext : str
        the extension (in words) of the file that needs to be read and saved
    skiprows : int
        number of rows to skip when reading a file
    
    Returns
    ------- 
    A pandas dataframe containing the data from the given file
    
    """   
    if ext == 'text':
        return pd.read_table(filepath,skiprows=skiprows,decimal='.',low_memory=False,index_col=None)
    elif ext == 'excel':
        return pd.read_excel(filepath,skiprows=skiprows,low_memory=False,index_col=None)
    elif ext == 'csv':
        return pd.read_csv(filepath,sep=sep,skiprows=skiprows,encoding=encoding,
                           error_bad_lines=False,low_memory=False,index_col=None)
        
def join_files(path,files,ext='text',sep=',',comment='#',encoding='utf8',decimal='.'):
    """
    Reads all files in a given directory, joins them and returns one pd.dataframe
    
    Parameters
    ----------
    path : str
	path to the folder that contains the files to be joined
    files : list
        list of files to be joined, must be the same extension
    ext : str
        extention of the files to read; possible: excel, text, csv
    sep : str
        the separating element (e.g. , or \t) necessary when reading csv-files
    comment : str
        comment symbol used in the files
    sort : array of bool and str
        if first element is true, apply the sort function to sort the data 
        based on the tags in the column mentioned in the second element of the 
        sort array
    
    Returns
    -------
    pd.dataframe: 
        pandas dataframe containin concatenated files in the given directory
    """
    #Initialisations
    data = pd.DataFrame()
  
    #Select files based on extension and sort files alphabetically to make sure
    #they are added to each other in the correct order
    #files = list_files(path,ext)
    files.sort()
    print('joining',len(files),'files...')
    #Read files
    for file_name in files:
        dir_file_path = os.path.join(path,file_name)
        with _open_file(dir_file_path,ext) as read_file:
            headerlength = _get_header_length(read_file,ext,comment)
            data = data.append(_read_file(dir_file_path,ext=ext,sep=sep,
                                          skiprows=headerlength,
                                          decimal=decimal,encoding=encoding),
                                ignore_index=True)
        print('Adding file',file_name,'to dataframe')
    data.to_csv('joined_files',sep=sep)
    
    return data

def write_to_WEST(df,file_normal,file_west,units,filepath=os.getcwd(),fillna=True):
        """
        writes a text-file that is compatible with WEST. Adds the units as 
        they are given in the 'units' argument.
        
        Parameters
        ----------
        df : pd.DataFrame
            the dataframe to write to WEST
        file_normal : str
            name of the original file to write, not yet compatible with WEST
        file_west : str
            name of the file that needs to be WEST compatible
        units : array of strings
            array containing the units for the respective columns in df
        filepath : str
            directory to save the files in; defaults to the current one
        fillna : bool
            when True, replaces nan values with 0 values (this might avoid 
            WEST problems later one).
        
        Returns
        -------
        None; writes files
        """
        if fillna:
            df = df.fillna(0)
        df.to_csv(os.path.join(filepath,file_normal),sep='\t')
        
        f = open(os.path.join(filepath,file_normal),'r')
        columns = f.readline()
        temp = f.read()
        f.close()
        
        f = open(os.path.join(filepath,file_west), 'w')
        f.write('#.t' + columns)
        unit_line = '#d\t'
        for i in range(0,len(units)-1):
            unit_line = unit_line + '{}\t'.format(units[i])
        unit_line = unit_line + '{}\n'.format(units[-1])
        f.write(unit_line)
        f.write(temp)
        f.close()
