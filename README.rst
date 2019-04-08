======
wwdata
======

.. image:: https://badge.fury.io/py/wwdata.svg
    :target: https://badge.fury.io/py/wwdata

.. image:: https://travis-ci.org/UGentBiomath/wwdata.svg?branch=master
        :target: https://travis-ci.org/UGentBiomath/wwdata

.. image:: https://readthedocs.org/projects/wwdata-docs/badge/
        :target: https://wwdata-docs.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://pyup.io/repos/github/UGentBiomath/wwdata/shield.svg
     :target: https://pyup.io/repos/github/UGentBiomath/wwdata/
     :alt: Updates

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1288581.svg
  :target: https://doi.org/10.5281/zenodo.1288581


Data analysis package aimed at data obtained in the context of (waste)water

* Free software: GNU General Public License v3
* Documentation: https://ugentbiomath.github.io/wwdata-docs/
* Funding: Waterboard De Dommel
* Context: PhD research at BIOMATH, Ghent University

Structure
---------

The package contains one class and three subclasses, all in separate .py files. Division in subclasses is based on the type of data:

* online data from full scale installations (OnlineSensorBased)
* online data from lab experiments (LabSensorBased)
* offline data obtained from lab experiments (LabExperimentBased).

Jupyter notbeook files (.ipynb) illustrate the use of the available functions. The most developed class is the OnlineSensorBased one. The workflow of this class is shown in below Figure, where OSB represents an OnlineSensorBased object. Main premises are to never delete data but to tag it and to be able to check the reliability when gaps in datasets are filled.

.. image:: ./figs/packagestructure_rel.png
    :align: center


Examples
--------

For the workflow with code and more specific examples, check out the Showcase Jupyter Notebook(s) included as documentation of the package.


Credits
---------

This package was created with support from Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template, as well as this `GitHub page`_, provided by Daler_ and explaining how to use sphinx documentation generation in combination with GitHub Pages.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _`GitHub page`: http://daler.github.io/sphinxdoc-test/includeme.html
.. _`Daler`: https://github.com/daler
