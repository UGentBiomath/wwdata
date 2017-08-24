# The wwdata package
*GNU AGPL Licensed*

## Premise and background
The wwdata (wastewater data) package is meant to make data analysis, validation and filling of data gaps more streamlined. It contains code to do all this, while also providing simple visualisations of the whole procedure. 

The package was (and is) developed in the framework of PhD research, involving the modelling of a full scale wastewater treatment plant (WWTP). Online measurements at the plant are available, but as with all data, is not perfect and therefor needs validation. The gap filling originated from the need to have high-frequency influent data available to run the WWTP model with.

## Content and structure
The package contains one class and three subclasses, all in separate .py files. Division in subclasses is based on the type of data: online data from full scale wastewater treatment plants (OnlineSensorBased), online data from lab experiments (LabSensorBased) and offline data obtained from lab experiments (LabExperimentBased). Jupyter notbeook files (.ipynb) illustrate the use of the available functions.

The workflow of the package is as follows: a dataset is read in as a pandas DataFrame and made into the relevant Class object, after which all below functions become available to use on the data. Generally speaking, several steps are taken: 
* Formatting of the data: converting strings to floats, time units to the right values, setting the index the way you want it...<br>

![](https://github.com/cdemulde/wwdata/blob/master/figs/data01.png?raw=true)

* Validate/filter the data: by use of several filter functions, data points are given a tag (kept in a separate DataFrame, self.meta_valid) that indicates whether the user deems them valid or not to continue using.<br>

![](https://github.com/cdemulde/wwdata/blob/master/figs/data02.png?raw=true)

* Filling gaps in the data: for some purposes (e.g. the running of models based on data input), data at every point in time is needed, so missing data needs to be filled. This happens by making use of several filling functions (only available in the OnlineSensorBased subclass, as not relevant in the other cases). A new DataFrame (self.filled) is created, containing both the original and validated datapoints and the datapoints that were filled in by one or more of the filling algorithms. Which filling function was used to fill which data point is kept track of by the tags in the self.meta_filled DataFrame.<br>

![](https://github.com/cdemulde/wwdata/blob/master/figs/data03.png?raw=true)

*The shown Figure clearly already indicates that ratio- and correlation-based filling of data-gaps is not a good procedure in this case; more details in the accompanying Showcase jupyter notebook*

* Simple writing functions are available to write the resulting dataset and metadata to .txt files

For the workflow with code and more specific examples included, check out the Showcase Jupyter Notebook(s) included as documentation of the package.

## Dependencies and installation
To check if you have all necessary Python packages installed, you can used the dependencies.yml file, which lists all packages of the environment wwdata was developed in. You can also use this file to create a [conda environment](http://conda.pydata.org/docs/using/envs.html#managing-environments) (assuming you're using [conda](http://conda.pydata.org/docs/index.html)), in which the wwdata package will work:<br>
`conda create env -f dependencies.yml`<br>

To install the wwdata package, execute the following with you command line tool:<br>
`python setup.py install`<br>

## Development and future plans
This wwdata package is currently in development, so if you have ideas on how to improve the package content, or suggestions for things to include, feel free to contact me:<br>
<a href='mailto:chaim.demulder@ugent.be'>chaim.demulder@ugent.be</a><br>
<a href='https://github.com/cdemulde'>github.com/cdemulde</a><br>
<a href='https://twitter.com/ChaimDM'>@ChaimDM</a>

To be included/extended in the future:
* The LabExperimentBased Class. For people not into coding or modelling, the LabExperimentBased class is supposed to provide some really basic code. Together with the use of Jupyter notebooks, the goal is to enhance the reproducibility of data analysis happening on a lot of data coming from lab scale experiments.
* Improved validation functions
* Improved visualisation