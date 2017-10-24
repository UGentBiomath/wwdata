======
wwdata
======


.. image:: https://img.shields.io/pypi/v/wwdata.svg
        :target: https://pypi.python.org/pypi/wwdata

.. image:: https://img.shields.io/travis/cdemulde/wwdata.svg
        :target: https://travis-ci.org/UGentBiomath/wwdata

.. image:: https://readthedocs.org/projects/wwdata/badge/?version=latest
        :target: https://wwdata.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://pyup.io/repos/github/UGentBiomath/wwdata/shield.svg
     :target: https://pyup.io/repos/github/UGentBiomath/wwdata/
     :alt: Updates


Data analysis package aimed at data obtained in the context of (waste)water

* Free software: GNU General Public License v3
* Documentation: https://wwdata.readthedocs.io.

The package contains one class and three subclasses, all in separate .py files. Division in subclasses is based on the type of data:

* online data from full scale wastewater treatment plants (OnlineSensorBased)
* online data from lab experiments (LabSensorBased)
* offline data obtained from lab experiments (LabExperimentBased).

Jupyter notbeook files (.ipynb) illustrate the use of the available functions. The most developed class is the OnlineSensorBased one. The workflow of this class is shown in below Figure, where OSB represents an OnlineSensorBased object. Main premises are to never delete data but to tag it and to be able to check the reliability when gaps in datasets are filled.

.. image:: ./figs/packagestructure_rel.png
    :width: 200px
    :scale: 50 %

For the workflow with code and more specific examples included, check out the Showcase Jupyter Notebook(s) included as documentation of the package.


Credits
---------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
