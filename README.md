# wwdata

The wwdata (wastewater data) package is meant to make data analysis, validation and filling of data gaps more streamlined. It also provides simple visualisations of the whole procedure.

The package contains one class and three subclasses, all in separate .py files. division in subclasses is based on the type of data: online data from full scale plants (OnlineSensorBased), online data from lab experiments (LabSensorBased) and offline data obtained from lab experiments (LabExperimentBased). Jupyter notbeook files (.ipynb) illustrate the use of the available functions.

The workflow of the package is as follows: a dataset is read in as a pandas DataFrame and made into a Class object, after which all below functions become available to use on the data. Generally speaking, several steps are taken:  
* Formatting of the data: converting strings to floats, time units to the right values, setting the index the way you want it...
* Validate/filter the data: by use of several filter functions, data points are given a tag (kept in a separate DataFrame, self.meta_valid) that indicates whether the user deems them valid or not to continue using.
* Filling gaps in the data: for some purposes (e.g. the running of models based on the data), data at every point in time is needed, so missing data needs to be filled. This happens by making use of several filling functions (only available in the OnlineSensorBased Subclass). A new DataFrame (self.filled) is created, containing both the original datapoints and the datapoints that were filled in by one or more of the filling algorithms. Which filling function was used to fill which data point is kept track of by the tags in the self.meta_filled DataFrame.
* Simple writing functions are available to write the resulting dataset and metadata to .txt files


