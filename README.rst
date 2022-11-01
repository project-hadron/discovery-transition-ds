Project Hadron Data Science and Synthetic Feature Builder
#########################################################

.. class:: no-web no-pdf

.. contents::

.. section-numbering::

Filling the Gap - Project Hadron
================================
Project Hadron has been built to bridge the gap between data scientists and data engineers. More specifically between
machine learning business outcomes and the final product.

Project Hadron is a core set of abstractions that are the foundation of the three key elements that represent data
science, those being: (1) feature engineering, (2) the construction of synthetic data with simulators, and generators
(3) and statistics and machine learning algorithms for discovery and creating models. Project Hadron uniquely sees
data as ‘all the same’ (lazyprogrammer (2020) https://lazyprogrammer.me/all-data-is-the-same/) , by which we mean
its origin, shape and size stay independent throughout the disciplines so its content, form and structure can be
removed as a factor in the design and implementation of the components built.

Project Hadron has been designed to place data scientists in the familiar environment of machine learning and
statistical tools, extracting their ideas and translating them automagicially into production ready solutions
familiar to data engineers and Subject Matter Experts (SME’s).

Project Hadron provides a clear Separation of Concerns, whilst maintaining the original intentions of the data
scientist, that can be passed to a production team. It offers trust between the data scientists teams and product
teams. It brings with it transparency and traceability, dealing with bias, fairness, and knowledge. The resulting
outcome provides the product engineers with adaptability, robustness, and reuse; fitting seamlessly into a
microservices solution that can be language agnostic.

At the heart of Project Hardon is a multi-tenant, NoSQL, singleton, in memory data store that has minimal code and
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
=============

* Data Preparation
* Feature Selection
* Feature Engineering
* Feature Cataloguing
* Augmented Knowledge
* Synthetic Feature Build

Background
==========
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
Its designed to be used easily within multiple python based interfaces such as Jupyter, IDE or command-line python.

Installation
============

package install
---------------
The best way to install AI-STAC component packages is directly from the Python Package Index repository using pip.
All AI-STAC components are based on a pure python foundation package ``aistac-foundation``

.. code-block:: bash

    $ pip install aistac-foundation

The AI-STAC component package for the Transition is ``discovery-transition-ds`` and pip installed with:

.. code-block:: bash

    $ pip install discovery-transition-ds

if you want to upgrade your current version then using pip install upgrade with:

.. code-block:: bash

    $ pip install --upgrade discovery-transition-ds

Building a Component
====================

This tutorial shows the fundamentals of how to run a basic Project
Hadron component. It is the simpliest form of running a task
demonstrating the input, throughput and output of a dataset. Each
instance of the component is given a unique reference name whereby the
Domain Contract uses that name as its unique identifier and thus can be
used to reference the said Domain Contract for the purposes of
referencing and reloading. Though this may seem complicated at this
early stage it is important to understand the relationship between a
named component and its Domain Contract.

Firstly we have imported a component from the Project Hadron library for
this demonstration. It should be noted, the choice of component is
arbritary for this demonstration, as even though each component has its
own unique set of tasks it also has methods shared across all
components. In this demonstration we only use these common tasks, this
is why our choice of component is arbitrary.

.. code:: ipython3

    from ds_discovery import Transition

To create a Domain Contract instance of the component we have used the
Factory method ``from_env`` and given it a referenceable name
``hello_comp``, and as this is the first instantiation, we have used the
one off parameter call ``has_contract`` that by default is set to True
and is used to avoid the accidential loading of a Domain Contract
instance of the same task name. As common practice we capture the
instance of this specific componant ``transition`` as ``tr``.

.. code:: ipython3

    tr = Transition.from_env('hello_comp', has_contract=False)

We have set where the data is coming from and where the resulting data
is going to. The source identifies a URI (URL) from which the data will
be collected and in this case persistance uses the default settings,
more on this later.

.. code:: ipython3

    tr.set_source_uri('https://www.openml.org/data/get_csv/16826755/phpMYEkMl.csv')
    tr.set_persist()

Run Component Pipeline
----------------------

To run a component we use the common method ``run_component_pipeline``
which loads the source data, executes the component task then persists
the results. This is the only method you can use to run the tasks of a
component and produce its results and should be a familiarized method.

.. code:: ipython3

    tr.run_component_pipeline()

This concludes building a component and though the component doesn’t
change the throughput, it shows the core steps to building any
component.

Reloading and Extending our Component
-------------------------------------

Though this is a single notebook, one of the powers of Project Hadron is
the ability to reload componant state across new notebooks, not just
locally but even across locations and teams. To load our componant state
we use the same factory method ``from_env`` passing the unique component
name ``hello_comp`` which reloads the Domain Contract. We have now
reinstated our origional component state and can continue to work on
this component.

.. code:: ipython3

    tr = Transition.from_env('hello_comp')

Lets look at a sample of some commonly used features that allow us to
peek inside our components. These features are extremely useful to
navigate the component and should become familiar.

The first and probably most useful method call is to be able to retrieve
the results of ``run_component_pipeline``. We do this using the
component method ``load_persist_canonical``. Because of the retained
state the component already knows the location of the results, and in
this instance returns a report.

Note: All the components from a package internally work with a canonical
data set. With this package of components, because they are data science
based, use Pandas Dataframes as their canonical, therefore wherever you
see the word canonical this will relate to a Pandas Dataframe.

.. code:: ipython3

    df = tr.load_persist_canonical()

The second most used feature is the reporting tool for the canonical. It
allows us to look at the results of the run as an informative
dictionary, this gives a deeper insight into the canonical results.
Though unlike other reports it requests the canonical of interest, this
means it can be used on a wider trajectory of circumstances such as
looking at source or other data that is being injested by the task.

Below we have an example of the processed canonical where we can see the
results of the pipeline that was persisted. The report has a wealth of
information and is worth taking time to explore as it is likely to speed
up your data discovery and the understanding of the dataset.

.. code:: ipython3

    tr.canonical_report(df)


When we set up the source and persist we use something called Connector
contracts, these act like brokers between external data and the internal
canonical. These are powerful tools that we will talk more about in a
dedicated tutorial but for now consider them as the means to talk data
to different data storage solutions. In this instance we are only using
a local connection and thus a Connector contract that manages this type
of connectivity.

In order to report on where the source and persist are located, along
with any other data we have connected to, we can use
``report_connectors`` which gives us, in part, the name of the connector
and the location of the data.

.. code:: ipython3

    tr.report_connectors()

This gives a flavour of the tools available to look inside a component
and time should be taken viewing the different reports a component
offers.

--------------

Environment Variables
---------------------

To this point we have using the default settings of where to store the
Domain Contract and the persisted dataset. These are in general local
and within your working directory. The use of environment variables
frees us up to use an extensive list of connector contracts to store the
data to a location of the choice or requirements.

Hadron provides an extensive list of environment variables to tailor how
your components retrieve and persist their information, this is beyond
the scope of this tutorial and tend to be for specialist use, therefore
we are going to focus on the two most commonly used for the majority of
projects.

We initially import Python’s ``os`` package.

.. code:: ipython3

    import os

In general and as good practice, most notebooks would ``run`` a set up
file that contains imports and environment variables that are common
across all notebooks. In this case, for visibility, because this is a
tutorial, we will import the packages and set up the two environment
variables within each notebook.

The first environment variable we set up is for the location of the
Domain Contract, this is critical to the components and the other
components that rely on it (more of this later). In this case we are
setting the Domain Contract location to be in a common local directory
of our naming.

.. code:: ipython3

    os.environ['HADRON_PM_PATH'] = '0_hello_meta/demo/contracts'

The second environment variable is for the location of where the data is
to be persisted. This allows us to place data away from the working
files and have a common directory where data can be sourced or
persisted. This is also used internally within the component to avoid
having to remember where data is located.

.. code:: ipython3

    os.environ['HADRON_DEFAULT_PATH'] = '0_hello_meta/demo/data'

As a tip we can see where the default path environment variable is set
by using ``report_connectors``. By passing the parameter
``inc_template=True`` to the ``report_connectors`` method, showing us
the connector names. By each name is the location path (uri) where, by
default, the component will source or persist the data set, this is
taken from the environment variable set. Likewise we can see where the
Domain Contract is being persisted by including the parameter ``inc_pm``
giving the location path (uri) given by the environment variable.

.. code:: ipython3

    tr.report_connectors(inc_template=True)


Because we have now changed the location of where the Domain Contract
can be found we need to reset things from the start giving the source
location and using the default persist location which we now know has
been set by the environment variable.

.. code:: ipython3

    tr = Transition.from_env('hello_tr,', has_contract=False)

.. code:: ipython3

    tr.set_source_uri('https://www.openml.org/data/get_csv/16826755/phpMYEkMl.csv')
    tr.set_persist()

Finally we run the pipeline with the new environemt variables in place
and check everything runs okay.

.. code:: ipython3

    tr.run_component_pipeline()

And we are there! We now know how to build a component and set its
environment variables. The next step is to build a real pipeline and
join that with other pipelines to construct our complete master Domain
Contract.

Reference
=========

Python version
--------------

Python 3.7 or less is not supported. Although it is recommended to install ``discovery-transition-ds`` against the
latest Python version or greater whenever possible.

Pandas version
--------------

Pandas 0.25.x and above are supported but It is highly recommended to use the latest 1.0.x release as the first
major release of Pandas.

GitHub Project
--------------
discovery-transition-ds: `<https://github.com/Gigas64/discovery-transition-ds>`_.

Change log
----------

See `CHANGELOG <https://github.com/doatridge-cs/discovery-transition-ds/blob/master/CHANGELOG.rst>`_.


Licence
-------

BSD-3-Clause: `LICENSE <https://github.com/doatridge-cs/discovery-transition-ds/blob/master/LICENSE.txt>`_.


Authors
-------

`Gigas64`_  (`@gigas64`_) created discovery-transition-ds.


.. _pip: https://pip.pypa.io/en/stable/installing/
.. _Github API: http://developer.github.com/v3/issues/comments/#create-a-comment
.. _Gigas64: http://opengrass.io
.. _@gigas64: https://twitter.com/gigas64


