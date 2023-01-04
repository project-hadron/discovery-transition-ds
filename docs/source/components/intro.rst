Filling the Gap - Project Hadron
================================

Introduction
------------

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

This is what gives us the Domain Contract for each tenant which sits at the heart of what makes the contracts
reusable, translatable, transferable and brings the data scientist closer to the production engineer along with
building a production ready component solution.

Main features
-------------

* Data Preparation
* Feature Selection
* Feature Engineering
* Feature Cataloguing
* Model Capture
* Custom Build
* Augmented Knowledge
* Synthetic Feature Build

Component capabilities
----------------------

The Project Hadron package comes with a number of component capabilities some of which are listed below
as the component name. Each capability represents a separation of concerns across the stakeholders and
data science teams model build workflow.

* SyntheticBuild - Synthetic data through Sampling, Subject Matter Expertise, artifacts and insight
* Transition - Selection through dimensionality reduction
* Wrangle - Feature Engineering through variable transformation
* FeatureCatalog - Feature cataloging through label optimisation
* ModelsBuilder - Model predict once the algorithm is trained and optimised

The diagram illustrates a typical workflow for stakeholders and data science teams looking to
implement business objectives. Highlighted within the diagram are where the capability components
sit within the workflow.

.. image:: /images/hello_hadron/0_img01.png
  :align: center
  :width: 800

The rectangles with a dotted outline box, that surround the processes, represent the components used at that
point within the workflow. Found within the rectangle is the name of the component used and in brackets its use.
This may not fit every workflow but when building a model, be it for production or as a proof of concept, each
of these capabilities are at the core of any model build and allow bridging the gap between data science and
production engineering.-

Feature transformers
--------------------

Project Hadron is a Python library with multiple transformers to engineer and select features to use
across a synthetic build, statistics and machine learning.

* Missing data imputation
* Categorical encoding
* Variable Discretization
* Outlier capping or removal
* Numerical transformation
* Redundant feature removal
* Synthetic variables creation
* Synthetic variables engineer
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

It also aims to improve the communication outputs needed by ML delivery to talk to Pre-Sales, Stakeholders,
Business SME's, Data SME's product coders and tooling engineers while still remaining within familiar code
paradigms.

