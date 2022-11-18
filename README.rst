Project Hadron Data Science Tools and Synthetic Feature Builder
###############################################################

.. class:: no-web no-pdf

.. contents:: Table of Contents

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
