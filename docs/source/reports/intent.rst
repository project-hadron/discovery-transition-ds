Darryl Oatridge, August 2022

Component Intent
================

-  Adding citations to support appropriate attribution by third-party
   users in order to formally incorporate data reuse

Validation of Data
------------------

-  The review of a data set by an expert with similar credentials and
   subject knowledge as the data creator to validate the accuracy of the
   data

.. code:: ipython3

    from ds_discovery import Transition, SyntheticBuilder
    import pandas as pd

Intent
------

Intent is a core concept that provides a set of intended actions
relating directly to the components core task. In this instance we are
using the Transitioning component that provides selection engineering of
a provided dataset.

As a core concept, Intent and its Parameterisation is captured in full
giving it transparency and traceability to an expert observer. It
provides direct editability of each Intent, with each Intent a seperate
concern. This means minimal rewrites, adaptability, clarity of change
and reduced testing.

.. code:: ipython3

    tr = Transition.from_env('demo_intent', has_contract=False)

Set File Source
^^^^^^^^^^^^^^^

Initially set the file source for the data of interest and runs the
component.

.. code:: ipython3

    ## Set the file source location
    data = 'https://www.openml.org/data/get_csv/16826755/phpMYEkMl.csv'
    tr.set_source_uri(data)
    tr.set_persist()
    tr.set_description("Original Titanic Dataset")

Parameterised Intent
--------------------

Through observations one identifies a number of selection engineering
that needs to be done with the provided dataset. We are therefore
looking to: - automaticially clean the header to remove spaces and
hidden characters in the header names. In addition note that ‘home.dest’
is seperated with a dot and best practice is to replace that with an
underscore. - reinstate nulls that have been obfuscated with ‘question
marks’ in order for us to clarify data quality and make better feature
engineering decisions. - identity selected data columns of no interest
and remove them. - apply logic that identifies potential catagoricals
and appropriately ‘type’ them. - insure the appropriate’typing’ of
indentifed numeric features. - turn our target boulian into a 0 and 1
integer type for better feature engineering, observability and decision
making.

Then run the pipeline to apply the Intent to the dataset.

.. code:: ipython3

    df = tr.load_source_canonical()

.. code:: ipython3

    df = tr.tools.auto_clean_header(df, rename_map={'home.dest': 'home_dest'}, intent_level='clean_header')
    df = tr.tools.auto_reinstate_nulls(df, nulls_list=['?'], intent_level='reinstate_nulls')
    df = tr.tools.to_remove(df, headers=['body', 'name', 'ticket', 'boat'], intent_level='to_remove')
    df = tr.tools.auto_to_category(df, intent_level='auto_categorize')
    df = tr.tools.to_numeric_type(df, headers=['age', 'fare'], intent_level='to_numeric')
    df = tr.tools.to_int_type(df, headers='survived', intent_level='to_int')
    
    tr.run_component_pipeline()

Report
------

The Intent, once applied, can now be observed through the Intent’s
report which outlines each activity which displays each line of the
Intent. So it is worth observing that the Intent report is presented in
alphabetical order and not the order in which it will run.

From the report one can clearly see each Intent and its Parameterisation
that can be modified by applying either a new Intent or a replacement of
the already existing line of code.

.. code:: ipython3

    tr.report_intent()

Intent Metadata
---------------

To enhance the readability and understanding of each intended action one
can also add metadata to help explain ones thinking. This can be used in
conjunction with the Intent report to provided a full picture of the
actions that were taken and their changes and those actions changes to
the outgoing dataset.

.. code:: ipython3

    tr.add_intent_level_description(level='clean_header', text="clean_header")
    tr.add_intent_level_description(level='reinstate_nulls', text="replace in question marks with nulls so its data can be properly typed")
    tr.add_intent_level_description(level='to_remove', text="Selective engineering to remove features of no interest")
    tr.add_intent_level_description(level='auto_categorize', text="categorise feature object types ")
    tr.add_intent_level_description(level='to_numeric', text="with nulls reinstated we can now reset the feature type")
    tr.add_intent_level_description(level='to_int', text="make the target type int rather than bool passing decision making down to the feature engineering")


.. code:: ipython3

    tr.report_column_catalog()

Run Book
--------

If not provided, the actions of the Intent will be aligned in the order
given but if one wishes to change this order it has the ability to
taylor the sequence using a Run Book. A Run Book provides the facility
to define run order to insure actions are run appropriate to the
Sequence they were intended. This is particulary useful when editing an
existing Intent pipeline or where changes effect other actions.

Run books can also be used to create multiple pipelines whereby a
sequence of Intent is created with multiple outcomes available for a
particular dataset. This is an advanced topic and not covered here.

As usual the Run Book comes with its own reporting tool for easy
visualisation.

.. code:: ipython3

    tr.add_run_book(run_levels=['clean_header', 'to_remove', 'reinstate_nulls', 'auto_categorize', 'to_numeric', 'to_int'])

.. code:: ipython3

    tr.report_run_book()
