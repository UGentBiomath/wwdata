=====
Usage
=====

To use wwdata in a project::

    import wwdata as ww

As the package was developed with the help of Jupyter Notebook (version 4.4.1) and visualisation is an important part of it, it is highly recommended to make use of a Jupyter Notebook when using the package.

Once the package is imported, the suggested workflow is as follows:

1. Read data and convert it to a pandas DataFrame (see [the pandas documentation](https://pandas.pydata.org) for more information); format it in the way you like.

2. Create an object of any of the three classes in the wwdata package, e.g. the OnlineSensorBased class::

    data = ww.OnlineSensorBased(dataframe,timedata_column="time",data_type="WWTP",experiment_tag="Data 2017")

3. Explore and format the data (convert to datetime, make absolute time index, make some plots...), e.g.::

    data.to_datetime(time_column="time",time_format="%dd-%mm-%yy")
    data.get_avg()

4. Tag non-valid data points. The way to do this depends on the data you are working with, but the general approach would be::

    data.tag_nan()
    data.moving_slope_filter(xdata="time",data_name="series1",cutoff=3,arange=['1/1/2017','1/2/2017'])

i.e. to simply apply any of the filtering functions to the class object.

5. Apply any other functionalities to the object. In the below example, this is the filling of the gaps introduced by filtering data in the previous step (for details on the meaning of the arguments, please refer to the documentation provided within the source code)::

    data.fill_missing_interpolation("series1",range_=12,arange=['1/1/2017','1/2/2017'])
    data.fill_missing_model("series1","model_data_series",arange=['1/1/2017','1/2/2017'])
