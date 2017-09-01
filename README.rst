======
wwdata
======


.. image:: https://img.shields.io/pypi/v/wwdata.svg
        :target: https://pypi.python.org/pypi/wwdata

.. image:: https://img.shields.io/travis/cdemulde/wwdata.svg
        :target: https://travis-ci.org/cdemulde/wwdata

.. image:: https://readthedocs.org/projects/wwdata/badge/?version=latest
        :target: https://wwdata.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://pyup.io/repos/github/cdemulde/wwdata/shield.svg
     :target: https://pyup.io/repos/github/cdemulde/wwdata/
     :alt: Updates


Data analysis package aimed at data obtained in the context of (waste)water


* Free software: GNU General Public License v3
* Documentation: https://wwdata.readthedocs.io.

The package contains one class and three subclasses, all in separate .py files. Division in subclasses is based on the type of data: online data from full scale wastewater treatment plants (OnlineSensorBased), online data from lab experiments (LabSensorBased) and offline data obtained from lab experiments (LabExperimentBased). Jupyter notbeook files (.ipynb) illustrate the use of the available functions.

The workflow of the package is as follows: a dataset is read in as a pandas DataFrame and made into the relevant Class object, after which all below functions become available to use on the data. Generally speaking, several steps are taken:
* Formatting of the data: converting strings to floats, time units to the right values, setting the index the way you want it...<br>

![](https://github.com/cdemulde/wwdata/blob/master/figs/data01.png?raw=true)

* Validate/filter the data: by use of several filter functions, data points are given a tag (kept in a separate DataFrame, self.meta_valid) that indicates whether the user deems them valid or not to continue using.<br>

![](https://github.com/cdemulde/wwdata/blob/master/figs/data02.png?raw=true)

* Filling gaps in the data: for some purposes (e.g. the running of models based on data input), data at every point in time is needed, so missing data needs to be filled. This happens by making use of several filling functions (only available in the OnlineSensorBased subclass, as not relevant in the other cases). A new DataFrame (self.filled) is created, containing both the original and validated datapoints and the datapoints that were filled in by one or more of the filling algorithms. Which filling function was used to fill which data point is kept track of by the tags in the self.meta_filled DataFrame.<br>

![](https://github.com/cdemulde/wwdata/blob/master/figs/data03.png?raw=true)

*The shown Figure clearly already indicates that ratio- and correlation-based filled of data-gaps is not a good procedure in this case; more details in the accompanying Showcase jupyter notebook)*

* Simple writing functions are available to write the resulting dataset and metadata to .txt files

For the workflow with code and more specific examples included, check out the Showcase Jupyter Notebook(s) included as documentation of the package.



Features
--------

* TODO

Credits
---------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
