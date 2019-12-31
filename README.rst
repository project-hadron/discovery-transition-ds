Accelerated Machine Learning
#############################

Accelerated Machine Learning is a unique approach around machine learning that innovates the data science discovery
vertical and productization of the data science delivery model. More specifically, it is an incubator project that
shadowed a team of Ph.D. data scientists in connection with the development and delivery of machine learning
initiatives to define measurable benefit propositions for customer success. To accomplish this, the project developed
specific and unique knowledge regarding transition and preparation of data sets for algorithmic execution and
augmented knowledge, which is at the core of the projects services offerings. From this the project developed a new
approach to data science discovery and productization dubbed “Accelerated Machine Learning”.

.. class:: no-web no-pdf

|pypi| |rdt| |license| |wheel|


.. contents::

.. section-numbering::

Main features
=============

* Machine Learning Capability Mapping
* Parametrized Intent
* Discovery Transitioning
* Machine Learning Feature Cataloguing
* Augmented Knowledge

Installation
============

package install
---------------

The best way to install this package is directly from the Python Package Index repository using pip

.. code-block:: bash

    $ pip install discovery-transition-ds

if you want to upgrade your current version then using pip

.. code-block:: bash

    $ pip install --upgrade discovery-transition-ds

Overview
========
The Accelerated Machine Learning project is a change of approach in terms of improving productivity of the data
scientists. This approach deconstructs the machine learning discovery vertical into a set of capabilities, ideas and
knowledge.  It presents a completely novel approach to the traditional process automation and model wrapping that is
broadly offered as a solution to solve the considerable challenges that currently restrict the effectiveness of
machine learning in the enterprise business.

To achieve this, the project offers advanced and specialized programming methods that are unique in approach and novel
while maintaining familiarity within common tooling can be identified in four constructs.

1. Machine Learning Capability Mapping - Separation of capabilities, breaking the machine learning vertical into a set
of decoupled and targeted layers of discrete and refined actions that collectively present a human lead (ethical AI)
base truth to the next set of capabilities. This not only allows improved transparency of, what is, a messy and
sometimes confusing set of discovery orientated coded ideas but also loosely couples and targets activities that are,
generally, complex and specialized into identifiable and discrete capabilities that can be chained as separately
allocated activities.

2. Parametrized Intent - A unique technique extracting the ideas and thinking of the data scientist from their
discovery code and capturing it as intent with parameters that can be replayed against productionized code and data.
This decoupling and Separation of Concern between data, code and the intent of actions from that code on that data,
considerably improves time to market, code reuse, transparency of actions and the communication of ideas between data
scientists and product delivery specialists.

3. Discovery Transitioning - Discovery Transitioning - is a foundation of the sepatation of concerns between data
provisioning and feature selection. As part of the Accelerated ML discovery Vertical, Transitioning is a foundation
base truth facilitating a transparent transition of the raw canonical dataset to a fit-for-purpose canonical dataset
to enable the optimisation of discovery analysis and the identification of features-of-interest, for the data scientist
and created boundary separation of capabilities decoupling the Data Scientist for the Data Engineer. As output it also
provides 'intelligent Communication', not only to the Data Scientist through canonical fit-for-purpose datasets, but
more generally offers powerful visual discovery tools and artefact generation for production architects, data and
business SME's, Stakeholders and is the initiator of Augmented Knowledge for an enriched and transparent shared view of
the extended data knowledge.

4. Machine Learning Feature Cataloguing – With cross over skills within machine learning and advanced data heuristics,
investigation identified commonality and separation across customer engagements that particularly challenged our
Ph.D data scientists in their effective delivery of customer success. As a result the project designed and developed
Feature Cataloguing, a machine learning technique of extracting and engineering features and their characteristics
appropriately parameterized for model selection.  This technique implements a juxta view of how features are
characterized and presented to the modelling layer. Traditionally features are directly mapped as a representation
of the underlying data set. Feature Cataloguing treats each individual feature as its own individual set of
characteristics as its representation. The resulting outcome considerably improves experimentation, cross feature
association, even when unrelated in the original data sets, and the reuse of identified features-of-interest across
use case and business domains.

5. Augmented Knowledge - This the ability to capture information on data, activities and the rich stream of subject
matter expertise, injected into the machine learning discovery vertical to provide an Augmented n-view of the model
build. This includes security, sensitivity, data value scaling, dictionary, observations, performance, optimization,
bias, etc. This enriched view of data allows, amongst other things, improved knowledge share, AI explainability,
feature transparency, and accountability that feeds into AI ethics, and insight analysis.

Background
==========
Born out of the frustration of time constraints and the inability to show business value
within a business expectation, this project aims to provide a set of tools to quickly
produce visual and observational results. It also aims to improve the communication
outputs needed by ML delivery to talk to Pre-Sales, Stakholders, Business SME's, Data SME's
product coders and tooling engineers while still remaining within familiar code paragigms.

The package looks to build a set of outputs as part of standard data wrangling and ML exploration
that, by their nature, are familiar tools to the various reliant people and processes. For example
Data dictionaries for SME's, Visual representations for clients and stakeholders and configuration
contracts for architects, tool builders and data ingestion.

ML Discovery
------------
ML Discovery is first and key part of an end to end process of discovery, productization and tooling. It defines
the ‘intelligence’ and business differentiators of everything downstream.

To become effective in the ML discovery phase, the ability to be able to micro-iterate within distinct layers
enables the needed adaptive delivery and quicker returns on ML use case.

The building and discovery of an ML model can be broken down into three Separation of Concerns (SoC)
or Scope of Responsibility (SoR) for the ML engineer and ML model builder.

- Data Preparation
- Feature Engineering
- Model selection and optimisation

with a forth discipline of insight, interpretation and profiling as an outcome. these three SoC's can be perceived as
eight distinct disciplines

Conceptuasl Eight stages of Model preparation
---------------------------------------------

#. Data Loading (fit-for-purpose, quality, quantity, veracity, connectivity)
#. Data Preparation (predictor selection, typing, cleaning, valuing, validating)
#. Augmented Knowledge (observation, visualisation, knowledge, value scale)
#. Feature Attribution (attribute mapping, quantitative attribute characterisation. predictor selection)
#. Feature Engineering (feature modelling, dirty clustering, time series, qualitative feature characterisation)
#. Feature Framing (hypothesis function, specialisation, custom model framing, model/feature selection)
#. Modelling (selection, optimisation, testing, training)
#. Training (learning, feedback loops, opacity testing, insight, profiling, stabilization)

Though conceptual they do represent a set of needed disciplines and the complexity of the journey to quality output.

Layered approach to ML
----------------------

The idea behind the conceptual eight stages of Machine Learning is to layer the preparation and reuse of the activities
undertaken by the ML Data Engineer and ML Modeller. To provide a platform for micro iterations rather than a
constant repetition of repeatable tasks through the stack. It also facilitates contractual definitions between
the different disciplines that allows loose coupling and automated regeneration of the different stages of model
build. Finally it reduces the cross discipline commitments by creating a 'by-design' set of contracts targeted
at, and written in, the language of the consumer.

The concept of being able to quickly run over a single aspect of the ML discovery and then present a stable base for
the next layer to iterate against. this micro-iteration approach allows for quick to market adaptive delivery.

Getting Started
===============

First Time Env Setup
--------------------

When you create a new project, or set up your default master notebook you import the Transition class

.. code-block:: python

    ...
    from ds_discovery import Transition


Within my master notebook, just as a fail-safe, as it costs nothing, I also set up the environment variable
``os.environ['TR_CONTRACT_PATH']`` with your root working path. In this example using the environmnt variable of ``PWD``

.. code-block:: python

    # set environment variables
    os.environ['TR_CONTRACT_PATH'] = Path(os.environ['PWD']).as_posix()

Setting ``TR_CONTRACT_PATH`` allows you to use the init factory pattern ``TransitionAgent.from_env(contract_name)``.

We now have all the appropriate imports and environment variables.

Transitioning: Data Sourcing
----------------------------

As part of the Accelerated ML Discovery Vertical, Transitioning is a
foundation base truth, facilitating a **transparent** transition of the
raw canonical dataset, to a **fit-for-purpose** canonical dataset, to
enable the optimisation of discovery analysis and the identification of
**features-of-interest**. The meaning of cononical is to convert formats
into common data language, not just bringing over the dataset but
bringing the construct of that dataset ie: type, format, structure, and
functionally, in our case because we are Python centric we use Pandas
Data Frames as our canonical.

With reference to the diagram, this notebook deals with the Sourcing
Contract and the raw canonical dataset as a prerequisite of the Sourcing
Contract: 1. Sourcing Notebooks 2. Sourcing Contract 3. Source
Connectivity and the Raw Canonical

.. figure:: ../98_images/AccML-Transition.png
   :alt: transition

   transition

Creating a Transitioning Contract Pipeline
------------------------------------------

-  Creating an instance of the Transitioning Class, passing a unique
   reference name. when wishing to reference this in other Juptyer
   Notebooks.
-  The reference name identifies the unique transitioning contract
   pipeline.

.. code-block:: python

    tr = TransitionAgent.from_env('synthetic_customer')

Reset the Source Contract
~~~~~~~~~~~~~~~~~~~~~~~~~

Reset the source contract so we start afresh. Printing the source report
validates that our values are empty.

.. code-block:: python

    # reset the contract and set the source contract
    tr.reset_contract()
    tr.report_source()


Find the files
~~~~~~~~~~~~~~

-  Use the discovery ``find_file(...)`` to explore the names of the raw
   files
-  Note, we use the file ‘property manager’ ``file_pm`` to get the
   data_path
-  Because this is a canonical, we can manipulate it as we would our
   source file

.. code-block:: python

    files = tr.discover.find_file('.csv', root_dir=tr.file_pm.data_path).iloc[:,[0,4]].sort_values('name', axis=0)
    files


Build the Source Contract
~~~~~~~~~~~~~~~~~~~~~~~~~

Source Contract is a set of attributes that define the resource, its
type and its location for retrieval and convertion to the raw canonical
for transitioning. The Source Contract additionally defines the module
and handler that is dynamically loaded at runtime.

By default the source contract requires
- resource: a local file, connector, URI or URL
- source_type a reference to the type ofresource. if None then extension of resource assumed
- location: a path, region or uri reference that can be used to identify location of resource
- module_name: a module name with full package path e.g ``ds_discovery.handlers.pandas_handlers``
- handler: the name os the handler class
- kwargs: additional arguments the handler might require

In this example, because we are using the standard Pandas data frame,
file handlers and the localized Transitioning default path locations, as
such we only need to provide the resource name and any other Key Word
Argument that the specific file handler may need. As our file is csv we
have defined the file separator and encoding.

.. code-block:: python

    tr.set_source_contract(uri='synthetic_customer.csv', module_name='ds_discovery.handlers.pandas_handlers',
                           handler='PandasSourceHandler', sep=',', encoding='latin1', load=False)

Other Connectivity
^^^^^^^^^^^^^^^^^^

As a comparison, in the following example we utilize the vast array of
other connectivity options. Here we are looking to connect to an AWS S3
containing csv files .

.. code-block:: python

    tr.set_source_contract(uri="s3://eu-west-1.amazonaws.com/synthetic/sftp/data/repo/synthetic_customer.csv",
                           module_name='ds_connectivity.handlers.aws_handlers', handler='S3SourceHandler',
                           sep=',', encoding='latin1', load=False)


Source Separation of Concerns
-----------------------------

The source details have now been recoreded in the contract pipeline

This Source separation of concerns means: \* New Notebooks are no longer
tied to the name or location of the data source \* File governance and
naming convention is managed automatically \* Connectivity can be
updated or reallocated independantly of the data science activities \*
Data location and infrastructure, through the delivery lifecycle, can be
hardened without effecting the Machine Learning discovery process

Loading the Canonical
~~~~~~~~~~~~~~~~~~~~~

Now we have recored the file information, we no longer need to reference
these details again To load the contract data we use the transitioning
method ``load_source_canonical()``\  and then we can use the canonical
dictionary report to examine the data set.

.. code-block:: python

    df = tr.load_source_canonical()
    tr.canonical_report(df)


Observations
~~~~~~~~~~~~

The report presents our attribute summary as a stylised data frame,
highlighting data points of interest. We will see more of this in the
next tutorial.

Next Steps
~~~~~~~~~~

Now we have our raw canonical data extracted and convereted to the
canonical from the source we can start the transitioning…

Python version
--------------

Python 2.6 and 2.7 are not supported. Although Python 3.5 is supported, it is recommended to install
``discovery-transition-ds`` against the latest Python 3.7.x whenever possible.
Python 3 is the default for Homebrew installations starting with version 0.9.4.

GitHub Project
--------------
Discovery-Transitioning-Utils: `<https://github.com/Gigas64/discovery-transition-ds>`_.

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


.. |pypi| image:: https://img.shields.io/pypi/pyversions/Django.svg
    :alt: PyPI - Python Version

.. |rdt| image:: https://readthedocs.org/projects/discovery-transition-ds/badge/?version=latest
    :target: http://discovery-transition-ds.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |license| image:: https://img.shields.io/pypi/l/Django.svg
    :target: https://github.com/Gigas64/discovery-transition-ds/blob/master/LICENSE.txt
    :alt: PyPI - License

.. |wheel| image:: https://img.shields.io/pypi/wheel/Django.svg
    :alt: PyPI - Wheel

