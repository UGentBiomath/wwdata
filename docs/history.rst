=======
History
=======

0.1.0 (2017-10-23)
------------------

First release on PyPI.

The wwdata (wastewater data) package is meant to make data analysis, validation and filling of data gaps more streamlined. It contains code to do all this, while also providing simple visualisations of the whole procedure.

The package was (and is) developed in the framework of PhD research, involving the modelling of a full scale wastewater treatment plant (WWTP). Online measurements at the plant are available, but as with all data, is not perfect and therefor needs validation. The gap filling originated from the need to have high-frequency influent data available to run the WWTP model with.

0.2.0 (2018-06-12)
------------------

Second release on PyPI.

The wwdata (wastewater data) package is meant to make data analysis, validation and filling of data gaps more streamlined. It contains code to do all this, while also providing simple visualisations of the whole procedure.

The package was (and is) developed in the framework of PhD research, involving the modelling of a full scale wastewater treatment plant (WWTP). Online measurements at the plant are available, but as with all data, is not perfect and therefor needs validation. The gap filling originated from the need to have high-frequency influent data available to run the WWTP model with.

New in version 0.2.0:

- Bug fixes
- Addition of an ``only_checked`` argument to multiple functions to allow application of the function to only the validated data points ('original' in self.meta_valid).
- Extended, improved and customized documentation website (generated with sphinx).
- Extended and improved Jupyter Notebook for documentation.
- Improved visualisation for *get_correlation*: a prediction band based on the obtained correlation is now included in the produced scatter plot.

Known bugs:

- See [open issues on Github](https://github.com/UGentBiomath/wwdata/issues)
