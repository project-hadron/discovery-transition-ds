Darryl Oatridge, August 2022

Reports: Data Analysis
======================
A deep dive look at patterns and relationships within a dataset to provide variable
analysis


.. code:: ipython3

    from ds_discovery import Transition

Schema
------

A Schema is a representation of our dataset as a set of statisticial and
probablistic values that are semantically common across all schemas. The
schema separates each data element into four parts:

-  Intent: shows how the data content is being discretionised and its
   type.
-  Params: the parameters used to specialise the Intent such as
   granularity, value limits etc.
-  Patterns: probabilstic values of how the datas relative frequency is
   distributed, along with a number of other values, related to the data
   type.
-  Stats: a broad set of statisticial analysis of the data dependant
   upon the data type including distribution indicators, limits and
   observations.

A schema can be fully or partially stored or represented as a relational
tree, through naming. One can build a semantic and contexualised view of
its data that can be distributed as a machine readable set of
comparitives or as part of some other outcome.

.. code:: ipython3

    tr = Transition.from_env('demo_schema', has_contract=False)

Set File Source
^^^^^^^^^^^^^^^

Initially we set the file source for the data of interest and run the
component.

.. code:: ipython3

    ## Set the file source location
    tr.set_source_uri('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv')
    tr.set_persist()
    tr.set_description("Titanic Dataset used by Seaborn")

.. code:: ipython3

    tr.run_component_pipeline()

Creating and Presenting Schema
------------------------------

By default the primary schema is generated using default values taking a
flat view of the data or feature set and producing a schema that is
either distributable through a given connector contract or, as in our
case, displayed within the notebook.

.. code:: ipython3

    tr.save_canonical_schema()

.. code:: ipython3

    tr.report_canonical_schema()

Report
------

As with all reports one can redistribute our schema to interested
parties or systems where the data can be observed or schematically
examined to produce decision making outcomes. For example with the
observation of concept drift.

.. code:: ipython3

    schema = tr.report_canonical_schema(tr.load_persist_canonical(), stylise=False)
    tr.save_report_canonical(reports=tr.REPORT_SCHEMA, report_canonical=schema)

Filter the Schema
-----------------

In the following example we taylor the view of the schema without
changing the underlying schema’s content. In this instance we have
filtered on:

-  root, with our interests in the data features ‘survived’ and ‘fare’
   and
-  section, where our interest is particulary the pattern subset.

This provides quick and easy visualisation of complex schemas and can
help to identify individuals or groups of elements of interest within
that schema.

.. code:: ipython3

    tr.report_canonical_schema(roots=['survived', 'fare'], sections='patterns')

Semantic Schema
---------------

Beyond the basic schema lies a complex but accessable set of
paramatization that allows for the creation of relational comparisions
between the data type.

In our demonstration below, when creating the schema, we have given it a
name and then provide the relational tree we are interested in. In this
case we take ‘survived’ as our root, being the target feature of
interest. We next relate this to ‘age’ to understand how age is
distributed both by ‘survived’ and ‘gender’.

.. code:: ipython3

    tr.save_canonical_schema(schema_name='survived', schema_tree=[
        {'survived': {'dtype': 'bool'}},
        {'age': {'granularity': [(0, 18), (18, 30), (30, 50), (50, 100)]}}])

.. code:: ipython3

    tr.report_canonical_schema(schema='survived')

Distrubutable Reporting
-----------------------

With this done one can now further investigate distributions and
discover a view of the data. In this case, as a simple example, one can
see the age range percentage of those that ‘survived’.

From this simple example one can see how schemas can be captured over a
period of time or fixed at a moment in time then distributed and
compared to provide monitoring and insight into data as it flows through
your system.

.. code:: ipython3

    result = tr.report_canonical_schema(schema='survived', roots='survived.1.age', elements=['relative_freq'], stylise=False)
    result['value'].to_list()

