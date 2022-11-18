Darryl Oatridge, August 2022

Validation of Data
==================

-  The review of a dataset by an expert with similar credentials and
   subject knowledge as the data creator to validate the accuracy of the
   data.

.. code:: ipython3

    from ds_discovery import Transition

Quality Assurance
-----------------

Quality assurance provides an immediate insight into the quality,
quantity, verasity and availability of the dataset being provided. This
is a critical step to the success of any machine learning or product
outcome.

Observational immediacy to the content of the dataset allows quick
decision making at the earliest stage of the process. It also provides
output for discussion for SME’s and data architects to share common
reports that are based on best practice and familiar to both parties.

Finially it provides observational tools presenting a broad-set of
information in a compacted and common display format.

.. code:: ipython3

    tr = Transition.from_env('demo_quality', has_contract=False)

Set File Source
^^^^^^^^^^^^^^^

Initially we set the file source for our data of interest and run the
component.

.. code:: ipython3

    ## Set the file source location
    data = 'https://www.openml.org/data/get_csv/16826755/phpMYEkMl.csv'
    tr.set_source_uri(data)
    tr.set_persist()
    tr.set_description("Original Titanic Dataset")

Data Dictionary
---------------

The data dictionary is a go to tool that gives both a visual and
shareable summary of the dataset provided. In this case one looks at the
raw source so as to assess its visual suitability.

In this instance, taking the original Titanic dataset, data elements
such as nulls have been masked and in some cases inappropriately ‘typed’
the data. There are also multiple features that are not required, all of
which need to be dealt with before one can get a better view of the data
presented.

.. code:: ipython3

    df = tr.load_source_canonical()
    tr.canonical_report (df)

Engineering Selection
^^^^^^^^^^^^^^^^^^^^^

The canonical is tidied up through engineering selection where one
adjusts the features of interest, whilst removing the data columns that
are of no interest and making sure the data is correctly typed.

.. code:: ipython3

    df = tr.tools.auto_reinstate_nulls(df, nulls_list=['?'])
    df = tr.tools.to_remove(df, headers=['body', 'name', 'ticket', 'boat', 'home.dest'])
    df = tr.tools.auto_to_category(df)
    df = tr.tools.to_numeric_type(df, headers=['age', 'fare'])
    df = tr.tools.to_int_type(df, headers='survived')
    
    tr.run_component_pipeline()

Validation
----------

Now our selection engineering has been applied to the dataset one has a
clearer view of the value of the data provided.

The canonical report provides an enhancment of already existing data
science tools to give a clear single view of our data set that is
familuar to a broader audience.

.. code:: ipython3

    tr.canonical_report(df)

Reporting
---------

As well as its visual display the enhanced dictionary can be distributed
to any connecting service, such as an XL spreadsheet and its graphical
tooling.

.. code:: ipython3

    dictionary = tr.canonical_report(tr.load_persist_canonical(), stylise=False)
    tr.save_report_canonical(reports=tr.REPORT_DICTIONARY, report_canonical=dictionary)

Report Tailoring
----------------

By default reports are given their own name and data type, though this
can be tailored to suit a targeted system with options of name,
versioning, timestamp and the data type of the data to be reported.

.. code:: ipython3

    reports = [tr.report2dict(report=tr.REPORT_DICTIONARY, prefix='titanic_', file_type='csv', stamped='days')]
    tr.save_report_canonical(reports=reports, report_canonical=dictionary)

Quality Summary
---------------

When looking at the data as well as the detail in the dictionary one can
also produce a summary overview of the dataset as a whole. The quality
report provides a subset view of quality score, data shape, data types,
usability summary and cost, if applicable.

.. code:: ipython3

    tr.report_quality_summary()

Report
------

As with the dictionary the quality report can be saved and redistributed
to interested parties.

.. code:: ipython3

    quality = tr.report_quality_summary(stylise=False)
    tr.save_report_canonical(reports=tr.REPORT_SUMMARY, report_canonical=quality)


