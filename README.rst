Project Hadron Data Science Tools and Synthetic Feature Builder
###############################################################

.. class:: no-web no-pdf

.. contents:: Table of Contents

Filling the Gap - Project Hadron
================================
Project Hadron has been built to bridge the gap between data scientists and data engineers. More specifically between
machine learning business outcomes and the final product.  It translates the work of data scientists into meaningful,
production ready solutions that can be easily managed by product engineers.

Project Hadron is a core set of abstractions that are the foundation of the three key elements that represent data
science, those being: (1) feature engineering, (2) the construction of synthetic data with simulators, and generators
(3) and statistics and machine learning algorithms for discovery and creating models. Project Hadron uniquely sees
data as ‘all the same’ (lazyprogrammer (2020) https://lazyprogrammer.me/all-data-is-the-same/) , by which we mean
its origin, shape and size stay independent throughout the disciplines so its content, form and structure can be
removed as a factor in the design and implementation of the components built.

Project Hadron has been designed to place data scientists in the familiar environment of machine learning and
statistical tools, extracting their ideas and translating them automagicially into production ready solutions
familiar to data engineers and Subject Matter Experts (SME’s).

Project Hadron provides a clear separation of concerns, whilst maintaining the original intentions of the data
scientist, that can be passed to a production team. It offers trust between the data scientists teams and product
teams. It brings with it transparency and traceability, dealing with bias, fairness, and knowledge. The resulting
outcome provides the product engineers with adaptability, robustness, and reuse; fitting seamlessly into a
microservices solution that can be language agnostic.

Project Hadron is designed using Microservices. Microservices - also known as the microservice architecture - is an
architectural pattern that structures an application as a collection of component services that are:

* Highly maintainable and testable
* Loosely coupled
* Independently deployable
* Highly reusable
* Resilient
* Technically independent

Component services are built for business capabilities and each service performs a single function. Because they are
independently run, each service can be updated, deployed, and scaled to meet demand for specific functions of an
application. Project Hadron microservices enable the rapid, frequent and reliable delivery of large, complex
applications. It also enables an organization to evolve its data science stack and experiment with innovative ideas.

At the heart of Project Hadron is a multi-tenant, NoSQL, singleton, in memory data store that has minimal code and
functionality and has been custom built specifically for Hadron tasks in  mind. Abstracted from this is the component
store which allows us to build a reusable set of methods that define each tenanted component that sits separately
from the store itself. In addition, a dynamic key value class provides labeling so that each tenant is not tied to
a fixed set of reference values unless by specificity. Each of the classes, the data store, the component property
manager, and the key value pairs that make up the component are all independent, giving complete flexibility and
minimum code footprint to the build process of new components.

This is what gives us the Domain Contract for each tennant which sits at the heart of what makes the contracts
reusable, translatable, transferable and brings the data scientist closer to the production engineer along with
building a production ready component solution.

Main features
-------------

* Data Preparation
* Feature Selection
* Feature Engineering
* Feature Cataloguing
* Augmented Knowledge
* Synthetic Feature Build

Feature transformers
--------------------

Project Hadron is a Python library with multiple transformers to engineer and select features to use
across a synthetic build, statistics and machine learning.

* Missing data imputation
* Categorical encoding
* Variable Discretisation
* Outlier capping or removal
* Numerical transformation
* Redundant feature removal
* Synthetic variable creation
* Synthetic multivariate
* Synthetic model distributions
* Datetime features
* Time series

Project Hadron allows one to present optimal parameters associated with each transformer, allowing
different engineering procedures to be applied to different variables and feature subsets.

Background
----------
Born out of the frustration of time constraints and the inability to show business value
within a business expectation, this project aims to provide a set of tools to quickly build production ready
data science disciplines within a component based solution demonstrating coupling and cohesion between each
disipline, providing a separation of concerns between components.

It also aims to improve the communication outputs needed by ML delivery to talk to Pre-Sales, Stakholders,
Business SME's, Data SME's product coders and tooling engineers while still remaining within familiar code
paradigms.

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
version released ``v10.2.7`` the hotfix release would be versioned
as ``v10.2.8``.

Reference
=========

Python version
--------------

Python 3.7 or less is not supported. Although it is recommended to install ``discovery-transition-ds`` against the
latest Python version or greater whenever possible.

Pandas version
--------------

Pandas 1.0.x and above are supported but It is highly recommended to use the latest 1.0.x release as the first
major release of Pandas.

GitHub Project
--------------

discovery-transition-ds: `<https://github.com/project-hadron/discovery-transition-ds>`_.

Change log
----------

See `CHANGELOG <https://github.com/project-hadron/discovery-transition-ds/blob/master/CHANGELOG.rst>`_.


License
-------
This project uses the following license:
MIT License: `<https://opensource.org/license/mit/>`_.



Authors
-------

`Gigas64`_  (`@gigas64`_) created discovery-transition-ds.


.. _pip: https://pip.pypa.io/en/stable/installing/
.. _Github API: http://developer.github.com/v3/issues/comments/#create-a-comment
.. _Gigas64: http://opengrass.io
.. _@gigas64: https://twitter.com/gigas64


