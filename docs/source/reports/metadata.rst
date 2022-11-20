
Reports: Adding Information of Interest
=======================================

Adding extended information of interest to guide and inform about appropriate
attribution and decision making by the creator or editors to promote information
share and assist data reuse.

.. code:: ipython3

    from ds_discovery import Transition, Wrangle

Adding Metadata
---------------

During the process of development multiple experts add value to our
understanding of the dataset. Project Hadron captures this knowledge as
part of its metadata and provides easy access tools to retain this
knowledge at real or near real time as well as adding it retrospectively
through automated processes.

Knowledge capture is placed under a tree structure of: - catalogue:
provides an encompassing group identifier such as attributes or
observations. - label: a subset of categories identifying the individual
set of text such as attribute name or observation type. - text: a brief
or descriptive narrative of the catalogue and label. Text is immutable
thus new text with the same catalogue and label will be added to the
existing content.

.. code:: ipython3

    tr = Transition.from_env('demo_metadata', has_contract=False)

Set File Source
^^^^^^^^^^^^^^^

Initially we set the file source for our data of interest and run the
component.

.. code:: ipython3

    ## Set the file source location
    tr.set_source_uri('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv', template_aligned=False)
    tr.set_persist()
    tr.set_description("Titanic Dataset used by Seaborn")

Adding Attributes
-----------------

A vital part of understanding one’s dataset is to describe the
attributes provided. In this instance we name our catalogue group
‘attributes’. The attributes are labeled with the name of the attribute
and given a description.

.. code:: ipython3

    ## Add some attribute descriptions
    tr.add_notes(catalog='attributes', label='age', text='The age of the passenger has limited null values')
    tr.add_notes(catalog='attributes', label='deck', text='cabin has already been split into deck from the originals')
    tr.add_notes(catalog='attributes', label='fare', text='the price of the fair')
    tr.add_notes(catalog='attributes', label='pclass', text='The class of the passenger')
    tr.add_notes(catalog='attributes', label='sex', text='The gender of the passenger')
    tr.add_notes(catalog='attributes', label='survived', text='If the passenger survived or not as the target')
    tr.add_notes(catalog='attributes', label='embarked', text='The code for the port the passengered embarked')

Adding Observations
-------------------

In addition we can capture feedback from an SME or data owner, for
example. In this case we capture ‘observations’ as our catalogue and
‘describe’ as our label which we maintain for both descriptions.

One can now use the reporting tool to visually present the knowledge
added. It is worth noting that with observations each description has
been captured.

.. code:: ipython3

    tr.add_notes(catalog='observations', label='describe', 
                 text='The original Titanic dataset has been engineered to fit Seaborn functionality')
    tr.add_notes(catalog='observations', label='describe', 
                 text='The age and deck attributes still maintain their null values')


.. code:: ipython3

    tr.report_notes(drop_dates=True)

.. image:: /images/reports/met_img01.png
  :align: center
  :width: 500


Bulk Notes
----------

In addition to adding individual notes one also has the ability to
upload bulk notes from an external data source. In our next example we
take an order book and from an already existing description catalogue
extract that knowledge and add it to our attributes.

.. code:: ipython3

    tr = Transition.from_env('cs_orders', has_contract=False)

Set File Source
^^^^^^^^^^^^^^^

Initially set the file source for the data of interest and run the
component.

.. code:: ipython3

    tr.set_source_uri(uri='data/CS_ORDERS.txt', sep='\t', error_bad_lines=False, low_memory=True, encoding='Latin1')
    tr.set_persist()
    tr.set_description("Consumer Notebook Orders for Q4 FY20")

Connect the Bulk Upload
^^^^^^^^^^^^^^^^^^^^^^^

First create a connector to the information source.

.. code:: ipython3

    tr.add_connector_uri(connector_name='bulk_notes', uri='data/cs_orders_dictionary.csv')

Upload the Descriptions
^^^^^^^^^^^^^^^^^^^^^^^

With our connector in place one can now load that data and specify the
columns of interest that provide both the label and the text.

Using our reporting tool one can now observe that attribute descriptions
have been uploaded.

.. code:: ipython3

    notes = tr.load_canonical(connector_name='bulk_notes')
    tr.upload_notes(canonical=notes, catalog='attributes', label_key='Attribute', text_key='Description')

.. code:: ipython3

    tr.report_notes(drop_dates=True)

.. figure:: /images/reports/met_img02.png
  :align: center
  :width: 500

  not all attributes are displayed

Report Filtering
^^^^^^^^^^^^^^^^

Sometimes bulk uploads can result in a large amount of added
information. Our reporting tool has the ability to filter what we
visualize giving us a clean summery of items of interest. In our example
we are filtering on ‘label’ across all sections, or catalogues.

.. code:: ipython3

    tr.report_notes(labels=['ORD_DTS', 'INV_DTS', 'HOLD_DTS'], drop_dates=True)

.. image:: /images/reports/met_img03.png
  :align: center
  :width: 250

