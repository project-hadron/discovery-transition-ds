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

Release Process and Rules
-------------------------

Versions to be released after ``3.5.27``, the following rules will govern
and describe how the ``discovery-transition-ds`` produces a new release.

To find the current version of ``discovery-transition-ds``, from your
terminal run:

.. code-block:: bash

    $ python -c "import ds_discovery; print(ds_discovery.__version__)"

Major Releases
**************

A major release will include breaking changes. When it is versioned, it will
be versioned as ``vX.0.0``. For example, if the previous release was
``v10.2.7`` the next version will be ``v11.0.0``.

Breaking changes are changes that break backwards compatibility with prior
versions. If the project were to change an existing methods signature or
alter a class or method name, that would only happen in a Major release.
The majority of changes to the dependant core abstraction will result in a
major release. Major releases may also include miscellaneous bug fixes that
have significant implications.

Project Hadron is committed to providing a good user experience
and as such, committed to preserving backwards compatibility as much as possible.
Major releases will be infrequent and will need strong justifications before they
are considered.

Minor Releases
**************

A minor release will include addition methods, or noticeable changes to
code in a backward-compatable manner and miscellaneous bug fixes. If the previous
version released was ``v10.2.7`` a minor release would be versioned as
``v10.3.0``.

Minor releases will be backwards compatible with releases that have the same
major version number. In other words, all versions that would start with
``v10.`` should be compatible with each other.

Patch Releases
**************

A patch release include small and encapsulated code changes that do
not directly effect a Major or Minor release, for example changing
``round(...`` to ``np.around(...``, and bug fixes that were missed
when the project released the previous version. If the previous
version released ``v10.2.7`` the patch release would be versioned
as ``v10.2.8``.

