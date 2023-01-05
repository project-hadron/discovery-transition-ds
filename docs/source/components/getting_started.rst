Getting Started
===============
The ``discovery-transition-ds`` package is a set of python components that are focussed on Data Science. They
are a concrete implementation of the Project Hadron abstract core. It is build to be very light weight
in terms of package dependencies requiring nothing beyond what would be found in an basic Data Science environment.
Its designed to be used easily within multiple python based interfaces such as Jupyter, IDE or terminal python.

Package Installation
--------------------
The best way to install the component packages is directly from the Python Package Index repository using pip.

The component package is ``discovery-transition-ds`` and pip installed with:

.. code-block:: bash

    python -m pip install discovery-transition-ds

if you want to upgrade your current version then using pip install upgrade with:

.. code-block:: bash

    python -m pip install -U discovery-transition-ds

This will also install or update dependent third party packages. The dependencies are
limited to python and related Data Science tooling such as pandas, numpy, scipy,
scikit-learn and visual packages matplotlib and seaborn, and thus have a limited
footprint and non-disruptive in a machine learning environment.

Get the Source Code
-------------------

``discovery-transition-ds`` is actively developed on GitHub, where the code is
`always available <https://github.com/project-hadron/discovery-transition-ds>`_.

You can clone the public repository with:

.. code-block:: bash

    $ git clone git@github.com:project-hadron/discovery-transition-ds.git

Once you have a copy of the source, you can embed it in your own Python
package, or install it into your site-packages easily running:

.. code-block:: bash

    $ cd discovery-transition-ds
    $ python -m pip install .

