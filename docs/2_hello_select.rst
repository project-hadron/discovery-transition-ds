Darryl Oatridge, August 2022

Building a Pipeline
-------------------

Now we know what a component looks like we can start to build the
pipeline adding in actions that gives the component purpose.

The first component we will build as part of the pipeline is the data
selection component with the class name Transition. This component
provides a set of actions that focuses on tidying raw data by removing
data columns that are not useful to the final feature set. These may
include null columns, single value columns, duplicate columns and noise
etc. We can also ensure the data is properly canonicalised through
enforcing data typing.

Project Hadron Canonicalizes data following the canonical model pattern
so that every component speaks the same data language. In this case and
with this package all components use Pandas DataFrame format. This is
common format used by data scientists and statisticians to manipulate
and visualise large data sets.

Before we do that, and as shown in the previous section, we now use the
environment variables to define the location of the Domain Contract and
datastore.

.. code:: ipython3

    import os 

.. code:: ipython3

    os.environ['HADRON_PM_PATH'] = '0_hello_meta/demo/contracts'
    os.environ['HADRON_DEFAULT_PATH'] = '0_hello_meta/demo/data'

For the feature selection we are using the Transition component with the
ability to select the correct columns from raw data, potentially
reducing the column count. In addition the Transistioning component
extends the common reporting tools and provides additional functionality
for identifying quality, quantity, veracity and availability.

It should be worth noting we are creating a new component and as such
must set up the input and the output of the component.

.. code:: ipython3

    from ds_discovery import Transition

.. code:: ipython3

    # get the instance
    tr = Transition.from_env('hello_tr', has_contract=False)

.. code:: ipython3

    tr.set_source_uri('https://www.openml.org/data/get_csv/16826755/phpMYEkMl.csv')
    tr.set_persist()

Adding Select Actions
~~~~~~~~~~~~~~~~~~~~~

At the core of a component is its tasks, in other words how it changes
incoming data into a different data outcome. To achieve this we use the
actions that are set up specificially for this Component. These actions
are the intensions of the specific component also know as the components
intent. The components intent is a finate set of methods, unique to each
component, that can be applied to the raw data in order to change it in
a way that is useful to the outcome of the task.

In order to get a list of a component’s intent, in this case feature
selection, you can use the Python method ``__dir__()``. In this case
with the transition component ``tr`` we would use the comand
``tr.tools.__dir__()``\ to produce the directory of the components
select intent. Remember this method call can be used in any components
intent tools.

Now we have added where the raw data is situated we can load the
canonical, called, ``df``\ …

.. code:: ipython3

    df = tr.load_source_canonical()

…and produce the report on the raw data so we can observe the features
of interest.

.. code:: ipython3

    tr.canonical_report(df)




.. raw:: html

    <style type="text/css">
    #T_61705 th {
      font-size: 120%;
      text-align: center;
    }
    #T_61705 .row_heading {
      display: none;;
    }
    #T_61705  .blank {
      display: none;;
    }
    #T_61705_row0_col0, #T_61705_row1_col0, #T_61705_row2_col0, #T_61705_row3_col0, #T_61705_row4_col0, #T_61705_row5_col0, #T_61705_row6_col0, #T_61705_row7_col0, #T_61705_row8_col0, #T_61705_row9_col0, #T_61705_row10_col0, #T_61705_row11_col0, #T_61705_row12_col0, #T_61705_row13_col0 {
      font-weight: bold;
      font-size: 120%;
    }
    #T_61705_row0_col2, #T_61705_row0_col3, #T_61705_row1_col2, #T_61705_row1_col3, #T_61705_row2_col2, #T_61705_row2_col5, #T_61705_row3_col2, #T_61705_row3_col5, #T_61705_row4_col2, #T_61705_row5_col2, #T_61705_row5_col3, #T_61705_row5_col5, #T_61705_row6_col2, #T_61705_row6_col3, #T_61705_row6_col5, #T_61705_row7_col2, #T_61705_row7_col3, #T_61705_row7_col5, #T_61705_row8_col2, #T_61705_row9_col2, #T_61705_row9_col3, #T_61705_row10_col2, #T_61705_row10_col3, #T_61705_row11_col2, #T_61705_row12_col2, #T_61705_row12_col3, #T_61705_row13_col2, #T_61705_row13_col3, #T_61705_row13_col5 {
      color: black;
    }
    #T_61705_row0_col5 {
      background-color: #f0f9ed;
      color: black;
    }
    #T_61705_row1_col5 {
      background-color: #e5f5e0;
      color: black;
    }
    #T_61705_row2_col3 {
      background-color: #fcb499;
      color: black;
    }
    #T_61705_row3_col3, #T_61705_row4_col3, #T_61705_row8_col3, #T_61705_row11_col3 {
      background-color: #ffede5;
      color: black;
    }
    #T_61705_row4_col5, #T_61705_row9_col5 {
      background-color: #84cc83;
      color: black;
    }
    #T_61705_row8_col1, #T_61705_row9_col1, #T_61705_row11_col1, #T_61705_row12_col1 {
      color: #0f398a;
    }
    #T_61705_row8_col5, #T_61705_row11_col5 {
      background-color: #a4da9e;
      color: black;
    }
    #T_61705_row10_col5, #T_61705_row12_col5 {
      background-color: #a1cbe2;
      color: black;
    }
    </style>
    <table id="T_61705">
      <caption>%_Dom: The % most dominant element </caption>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_61705_level0_col0" class="col_heading level0 col0" >Attributes (14)</th>
          <th id="T_61705_level0_col1" class="col_heading level0 col1" >dType</th>
          <th id="T_61705_level0_col2" class="col_heading level0 col2" >%_Null</th>
          <th id="T_61705_level0_col3" class="col_heading level0 col3" >%_Dom</th>
          <th id="T_61705_level0_col4" class="col_heading level0 col4" >Count</th>
          <th id="T_61705_level0_col5" class="col_heading level0 col5" >Unique</th>
          <th id="T_61705_level0_col6" class="col_heading level0 col6" >Observations</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_61705_level0_row0" class="row_heading level0 row0" >0</th>
          <td id="T_61705_row0_col0" class="data row0 col0" >age</td>
          <td id="T_61705_row0_col1" class="data row0 col1" >object</td>
          <td id="T_61705_row0_col2" class="data row0 col2" >0.0%</td>
          <td id="T_61705_row0_col3" class="data row0 col3" >20.1%</td>
          <td id="T_61705_row0_col4" class="data row0 col4" >1309</td>
          <td id="T_61705_row0_col5" class="data row0 col5" >99</td>
          <td id="T_61705_row0_col6" class="data row0 col6" >Sample: ? | 24 | 22 | 21 | 30</td>
        </tr>
        <tr>
          <th id="T_61705_level0_row1" class="row_heading level0 row1" >1</th>
          <td id="T_61705_row1_col0" class="data row1 col0" >boat</td>
          <td id="T_61705_row1_col1" class="data row1 col1" >object</td>
          <td id="T_61705_row1_col2" class="data row1 col2" >0.0%</td>
          <td id="T_61705_row1_col3" class="data row1 col3" >62.9%</td>
          <td id="T_61705_row1_col4" class="data row1 col4" >1309</td>
          <td id="T_61705_row1_col5" class="data row1 col5" >28</td>
          <td id="T_61705_row1_col6" class="data row1 col6" >Sample: ? | 13 | C | 15 | 14</td>
        </tr>
        <tr>
          <th id="T_61705_level0_row2" class="row_heading level0 row2" >2</th>
          <td id="T_61705_row2_col0" class="data row2 col0" >body</td>
          <td id="T_61705_row2_col1" class="data row2 col1" >object</td>
          <td id="T_61705_row2_col2" class="data row2 col2" >0.0%</td>
          <td id="T_61705_row2_col3" class="data row2 col3" >90.8%</td>
          <td id="T_61705_row2_col4" class="data row2 col4" >1309</td>
          <td id="T_61705_row2_col5" class="data row2 col5" >122</td>
          <td id="T_61705_row2_col6" class="data row2 col6" >Sample: ? | 58 | 285 | 156 | 143</td>
        </tr>
        <tr>
          <th id="T_61705_level0_row3" class="row_heading level0 row3" >3</th>
          <td id="T_61705_row3_col0" class="data row3 col0" >cabin</td>
          <td id="T_61705_row3_col1" class="data row3 col1" >object</td>
          <td id="T_61705_row3_col2" class="data row3 col2" >0.0%</td>
          <td id="T_61705_row3_col3" class="data row3 col3" >77.5%</td>
          <td id="T_61705_row3_col4" class="data row3 col4" >1309</td>
          <td id="T_61705_row3_col5" class="data row3 col5" >187</td>
          <td id="T_61705_row3_col6" class="data row3 col6" >Sample: ? | C23 C25 C27 | G6 | B57 B59 B63 B66 | C22 C26</td>
        </tr>
        <tr>
          <th id="T_61705_level0_row4" class="row_heading level0 row4" >4</th>
          <td id="T_61705_row4_col0" class="data row4 col0" >embarked</td>
          <td id="T_61705_row4_col1" class="data row4 col1" >object</td>
          <td id="T_61705_row4_col2" class="data row4 col2" >0.0%</td>
          <td id="T_61705_row4_col3" class="data row4 col3" >69.8%</td>
          <td id="T_61705_row4_col4" class="data row4 col4" >1309</td>
          <td id="T_61705_row4_col5" class="data row4 col5" >4</td>
          <td id="T_61705_row4_col6" class="data row4 col6" >Sample: S | C | Q | ?</td>
        </tr>
        <tr>
          <th id="T_61705_level0_row5" class="row_heading level0 row5" >5</th>
          <td id="T_61705_row5_col0" class="data row5 col0" >fare</td>
          <td id="T_61705_row5_col1" class="data row5 col1" >object</td>
          <td id="T_61705_row5_col2" class="data row5 col2" >0.0%</td>
          <td id="T_61705_row5_col3" class="data row5 col3" >4.6%</td>
          <td id="T_61705_row5_col4" class="data row5 col4" >1309</td>
          <td id="T_61705_row5_col5" class="data row5 col5" >282</td>
          <td id="T_61705_row5_col6" class="data row5 col6" >Sample: 8.05 | 13 | 7.75 | 26 | 7.8958</td>
        </tr>
        <tr>
          <th id="T_61705_level0_row6" class="row_heading level0 row6" >6</th>
          <td id="T_61705_row6_col0" class="data row6 col0" >home.dest</td>
          <td id="T_61705_row6_col1" class="data row6 col1" >object</td>
          <td id="T_61705_row6_col2" class="data row6 col2" >0.0%</td>
          <td id="T_61705_row6_col3" class="data row6 col3" >43.1%</td>
          <td id="T_61705_row6_col4" class="data row6 col4" >1309</td>
          <td id="T_61705_row6_col5" class="data row6 col5" >370</td>
          <td id="T_61705_row6_col6" class="data row6 col6" >Sample: ? | New York, NY | London | Montreal, PQ | Paris, France</td>
        </tr>
        <tr>
          <th id="T_61705_level0_row7" class="row_heading level0 row7" >7</th>
          <td id="T_61705_row7_col0" class="data row7 col0" >name</td>
          <td id="T_61705_row7_col1" class="data row7 col1" >object</td>
          <td id="T_61705_row7_col2" class="data row7 col2" >0.0%</td>
          <td id="T_61705_row7_col3" class="data row7 col3" >0.2%</td>
          <td id="T_61705_row7_col4" class="data row7 col4" >1309</td>
          <td id="T_61705_row7_col5" class="data row7 col5" >1307</td>
          <td id="T_61705_row7_col6" class="data row7 col6" >Sample: Connolly, Miss. Kate | Kelly, Mr. James | Allen, Miss. Elisabeth Walton | Ilmakangas, Miss. ...</td>
        </tr>
        <tr>
          <th id="T_61705_level0_row8" class="row_heading level0 row8" >8</th>
          <td id="T_61705_row8_col0" class="data row8 col0" >parch</td>
          <td id="T_61705_row8_col1" class="data row8 col1" >int64</td>
          <td id="T_61705_row8_col2" class="data row8 col2" >0.0%</td>
          <td id="T_61705_row8_col3" class="data row8 col3" >76.5%</td>
          <td id="T_61705_row8_col4" class="data row8 col4" >1309</td>
          <td id="T_61705_row8_col5" class="data row8 col5" >8</td>
          <td id="T_61705_row8_col6" class="data row8 col6" >max=9 | min=0 | mean=0.39 | dominant=0</td>
        </tr>
        <tr>
          <th id="T_61705_level0_row9" class="row_heading level0 row9" >9</th>
          <td id="T_61705_row9_col0" class="data row9 col0" >pclass</td>
          <td id="T_61705_row9_col1" class="data row9 col1" >int64</td>
          <td id="T_61705_row9_col2" class="data row9 col2" >0.0%</td>
          <td id="T_61705_row9_col3" class="data row9 col3" >54.2%</td>
          <td id="T_61705_row9_col4" class="data row9 col4" >1309</td>
          <td id="T_61705_row9_col5" class="data row9 col5" >3</td>
          <td id="T_61705_row9_col6" class="data row9 col6" >max=3 | min=1 | mean=2.29 | dominant=3</td>
        </tr>
        <tr>
          <th id="T_61705_level0_row10" class="row_heading level0 row10" >10</th>
          <td id="T_61705_row10_col0" class="data row10 col0" >sex</td>
          <td id="T_61705_row10_col1" class="data row10 col1" >object</td>
          <td id="T_61705_row10_col2" class="data row10 col2" >0.0%</td>
          <td id="T_61705_row10_col3" class="data row10 col3" >64.4%</td>
          <td id="T_61705_row10_col4" class="data row10 col4" >1309</td>
          <td id="T_61705_row10_col5" class="data row10 col5" >2</td>
          <td id="T_61705_row10_col6" class="data row10 col6" >Sample: male | female</td>
        </tr>
        <tr>
          <th id="T_61705_level0_row11" class="row_heading level0 row11" >11</th>
          <td id="T_61705_row11_col0" class="data row11 col0" >sibsp</td>
          <td id="T_61705_row11_col1" class="data row11 col1" >int64</td>
          <td id="T_61705_row11_col2" class="data row11 col2" >0.0%</td>
          <td id="T_61705_row11_col3" class="data row11 col3" >68.1%</td>
          <td id="T_61705_row11_col4" class="data row11 col4" >1309</td>
          <td id="T_61705_row11_col5" class="data row11 col5" >7</td>
          <td id="T_61705_row11_col6" class="data row11 col6" >max=8 | min=0 | mean=0.5 | dominant=0</td>
        </tr>
        <tr>
          <th id="T_61705_level0_row12" class="row_heading level0 row12" >12</th>
          <td id="T_61705_row12_col0" class="data row12 col0" >survived</td>
          <td id="T_61705_row12_col1" class="data row12 col1" >int64</td>
          <td id="T_61705_row12_col2" class="data row12 col2" >0.0%</td>
          <td id="T_61705_row12_col3" class="data row12 col3" >61.8%</td>
          <td id="T_61705_row12_col4" class="data row12 col4" >1309</td>
          <td id="T_61705_row12_col5" class="data row12 col5" >2</td>
          <td id="T_61705_row12_col6" class="data row12 col6" >max=1 | min=0 | mean=0.38 | dominant=0</td>
        </tr>
        <tr>
          <th id="T_61705_level0_row13" class="row_heading level0 row13" >13</th>
          <td id="T_61705_row13_col0" class="data row13 col0" >ticket</td>
          <td id="T_61705_row13_col1" class="data row13 col1" >object</td>
          <td id="T_61705_row13_col2" class="data row13 col2" >0.0%</td>
          <td id="T_61705_row13_col3" class="data row13 col3" >0.8%</td>
          <td id="T_61705_row13_col4" class="data row13 col4" >1309</td>
          <td id="T_61705_row13_col5" class="data row13 col5" >929</td>
          <td id="T_61705_row13_col6" class="data row13 col6" >Sample: CA. 2343 | 1601 | CA 2144 | PC 17608 | 347077</td>
        </tr>
      </tbody>
    </table>




Featutres of Interest
~~~~~~~~~~~~~~~~~~~~~

The components intent methods are not first class methods but part of
the ``intent_model_class``. Therefore to access the intent specify the
controller instance name, in this case ``tr``, and then reference the
``intent_model_class`` to access the components intent. To make this
easier to remember with an abbreviated form we have overloaded the
``intent_model`` name with the name ``tools``. You can see with all
reference to the intent actions they start with ``tr.tools.``

When looking for features of interest, through observation, it appears,
within some columns ``space`` has been repalaced by a question mark
``?``. In this instance we would use the ``auto_reinstate_nulls`` to
replace all the obfusacted cells with nulls. In addition we can
immediately observe columns that are inappropriate for our needs. In
this case we do not need the column **name** and it is removed using
``to_remove`` passing the name of the attribute.

.. code:: ipython3

    # returns obfusacted nulls
    df = tr.tools.auto_reinstate_nulls(df, nulls_list=['?'])
    # removes data columns of no interest
    df = tr.tools.to_remove(df, headers=['name'])

Run Component Pipeline
~~~~~~~~~~~~~~~~~~~~~~

To run a component we use the common method ``run_component_pipeline``
which loads the source data, executes the component task then persists
the results. This is the only method you can use to run the tasks of a
component and produce its results and should be a familiarized method.

We can now run the ``run_component_pipeline`` and use the canonical
report to observe the outcome. From it we can see the nulls column now
indicates the number of nulls in each column correctly so we can deal
with them later. We have also removed the column **name**.

.. code:: ipython3

    tr.run_component_pipeline()
    tr.canonical_report(tr.load_persist_canonical())




.. raw:: html

    <style type="text/css">
    #T_976f2 th {
      font-size: 120%;
      text-align: center;
    }
    #T_976f2 .row_heading {
      display: none;;
    }
    #T_976f2  .blank {
      display: none;;
    }
    #T_976f2_row0_col0, #T_976f2_row1_col0, #T_976f2_row2_col0, #T_976f2_row3_col0, #T_976f2_row4_col0, #T_976f2_row5_col0, #T_976f2_row6_col0, #T_976f2_row7_col0, #T_976f2_row8_col0, #T_976f2_row9_col0, #T_976f2_row10_col0, #T_976f2_row11_col0, #T_976f2_row12_col0 {
      font-weight: bold;
      font-size: 120%;
    }
    #T_976f2_row0_col2, #T_976f2_row0_col3, #T_976f2_row1_col2, #T_976f2_row1_col3, #T_976f2_row2_col5, #T_976f2_row3_col5, #T_976f2_row4_col2, #T_976f2_row5_col2, #T_976f2_row5_col3, #T_976f2_row5_col5, #T_976f2_row6_col2, #T_976f2_row6_col3, #T_976f2_row6_col5, #T_976f2_row7_col2, #T_976f2_row8_col2, #T_976f2_row8_col3, #T_976f2_row9_col2, #T_976f2_row9_col3, #T_976f2_row10_col2, #T_976f2_row11_col2, #T_976f2_row11_col3, #T_976f2_row12_col2, #T_976f2_row12_col3, #T_976f2_row12_col5 {
      color: black;
    }
    #T_976f2_row0_col5 {
      background-color: #f0f9ed;
      color: black;
    }
    #T_976f2_row1_col5 {
      background-color: #e5f5e0;
      color: black;
    }
    #T_976f2_row2_col2, #T_976f2_row2_col3 {
      background-color: #fcb499;
      color: black;
    }
    #T_976f2_row3_col2, #T_976f2_row3_col3, #T_976f2_row4_col3, #T_976f2_row7_col3, #T_976f2_row10_col3 {
      background-color: #ffede5;
      color: black;
    }
    #T_976f2_row4_col5, #T_976f2_row8_col5 {
      background-color: #84cc83;
      color: black;
    }
    #T_976f2_row7_col1, #T_976f2_row8_col1, #T_976f2_row10_col1, #T_976f2_row11_col1 {
      color: #0f398a;
    }
    #T_976f2_row7_col5, #T_976f2_row10_col5 {
      background-color: #a4da9e;
      color: black;
    }
    #T_976f2_row9_col5, #T_976f2_row11_col5 {
      background-color: #a1cbe2;
      color: black;
    }
    </style>
    <table id="T_976f2">
      <caption>%_Dom: The % most dominant element </caption>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_976f2_level0_col0" class="col_heading level0 col0" >Attributes (13)</th>
          <th id="T_976f2_level0_col1" class="col_heading level0 col1" >dType</th>
          <th id="T_976f2_level0_col2" class="col_heading level0 col2" >%_Null</th>
          <th id="T_976f2_level0_col3" class="col_heading level0 col3" >%_Dom</th>
          <th id="T_976f2_level0_col4" class="col_heading level0 col4" >Count</th>
          <th id="T_976f2_level0_col5" class="col_heading level0 col5" >Unique</th>
          <th id="T_976f2_level0_col6" class="col_heading level0 col6" >Observations</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_976f2_level0_row0" class="row_heading level0 row0" >0</th>
          <td id="T_976f2_row0_col0" class="data row0 col0" >age</td>
          <td id="T_976f2_row0_col1" class="data row0 col1" >object</td>
          <td id="T_976f2_row0_col2" class="data row0 col2" >20.1%</td>
          <td id="T_976f2_row0_col3" class="data row0 col3" >20.1%</td>
          <td id="T_976f2_row0_col4" class="data row0 col4" >1309</td>
          <td id="T_976f2_row0_col5" class="data row0 col5" >99</td>
          <td id="T_976f2_row0_col6" class="data row0 col6" >Sample: 24 | 22 | 21 | 30 | 18</td>
        </tr>
        <tr>
          <th id="T_976f2_level0_row1" class="row_heading level0 row1" >1</th>
          <td id="T_976f2_row1_col0" class="data row1 col0" >boat</td>
          <td id="T_976f2_row1_col1" class="data row1 col1" >object</td>
          <td id="T_976f2_row1_col2" class="data row1 col2" >62.9%</td>
          <td id="T_976f2_row1_col3" class="data row1 col3" >62.9%</td>
          <td id="T_976f2_row1_col4" class="data row1 col4" >1309</td>
          <td id="T_976f2_row1_col5" class="data row1 col5" >28</td>
          <td id="T_976f2_row1_col6" class="data row1 col6" >Sample: 13 | C | 15 | 14 | 4</td>
        </tr>
        <tr>
          <th id="T_976f2_level0_row2" class="row_heading level0 row2" >2</th>
          <td id="T_976f2_row2_col0" class="data row2 col0" >body</td>
          <td id="T_976f2_row2_col1" class="data row2 col1" >object</td>
          <td id="T_976f2_row2_col2" class="data row2 col2" >90.8%</td>
          <td id="T_976f2_row2_col3" class="data row2 col3" >90.8%</td>
          <td id="T_976f2_row2_col4" class="data row2 col4" >1309</td>
          <td id="T_976f2_row2_col5" class="data row2 col5" >122</td>
          <td id="T_976f2_row2_col6" class="data row2 col6" >Sample: 135 | 101 | 37 | 285 | 156</td>
        </tr>
        <tr>
          <th id="T_976f2_level0_row3" class="row_heading level0 row3" >3</th>
          <td id="T_976f2_row3_col0" class="data row3 col0" >cabin</td>
          <td id="T_976f2_row3_col1" class="data row3 col1" >object</td>
          <td id="T_976f2_row3_col2" class="data row3 col2" >77.5%</td>
          <td id="T_976f2_row3_col3" class="data row3 col3" >77.5%</td>
          <td id="T_976f2_row3_col4" class="data row3 col4" >1309</td>
          <td id="T_976f2_row3_col5" class="data row3 col5" >187</td>
          <td id="T_976f2_row3_col6" class="data row3 col6" >Sample: C23 C25 C27 | G6 | B57 B59 B63 B66 | F4 | F33</td>
        </tr>
        <tr>
          <th id="T_976f2_level0_row4" class="row_heading level0 row4" >4</th>
          <td id="T_976f2_row4_col0" class="data row4 col0" >embarked</td>
          <td id="T_976f2_row4_col1" class="data row4 col1" >object</td>
          <td id="T_976f2_row4_col2" class="data row4 col2" >0.2%</td>
          <td id="T_976f2_row4_col3" class="data row4 col3" >69.8%</td>
          <td id="T_976f2_row4_col4" class="data row4 col4" >1309</td>
          <td id="T_976f2_row4_col5" class="data row4 col5" >4</td>
          <td id="T_976f2_row4_col6" class="data row4 col6" >Sample: S | C | Q</td>
        </tr>
        <tr>
          <th id="T_976f2_level0_row5" class="row_heading level0 row5" >5</th>
          <td id="T_976f2_row5_col0" class="data row5 col0" >fare</td>
          <td id="T_976f2_row5_col1" class="data row5 col1" >object</td>
          <td id="T_976f2_row5_col2" class="data row5 col2" >0.1%</td>
          <td id="T_976f2_row5_col3" class="data row5 col3" >4.6%</td>
          <td id="T_976f2_row5_col4" class="data row5 col4" >1309</td>
          <td id="T_976f2_row5_col5" class="data row5 col5" >282</td>
          <td id="T_976f2_row5_col6" class="data row5 col6" >Sample: 8.05 | 13 | 7.75 | 26 | 7.8958</td>
        </tr>
        <tr>
          <th id="T_976f2_level0_row6" class="row_heading level0 row6" >6</th>
          <td id="T_976f2_row6_col0" class="data row6 col0" >home.dest</td>
          <td id="T_976f2_row6_col1" class="data row6 col1" >object</td>
          <td id="T_976f2_row6_col2" class="data row6 col2" >43.1%</td>
          <td id="T_976f2_row6_col3" class="data row6 col3" >43.1%</td>
          <td id="T_976f2_row6_col4" class="data row6 col4" >1309</td>
          <td id="T_976f2_row6_col5" class="data row6 col5" >370</td>
          <td id="T_976f2_row6_col6" class="data row6 col6" >Sample: New York, NY | London | Montreal, PQ | Paris, France | Cornwall / Akron, OH</td>
        </tr>
        <tr>
          <th id="T_976f2_level0_row7" class="row_heading level0 row7" >7</th>
          <td id="T_976f2_row7_col0" class="data row7 col0" >parch</td>
          <td id="T_976f2_row7_col1" class="data row7 col1" >int64</td>
          <td id="T_976f2_row7_col2" class="data row7 col2" >0.0%</td>
          <td id="T_976f2_row7_col3" class="data row7 col3" >76.5%</td>
          <td id="T_976f2_row7_col4" class="data row7 col4" >1309</td>
          <td id="T_976f2_row7_col5" class="data row7 col5" >8</td>
          <td id="T_976f2_row7_col6" class="data row7 col6" >max=9 | min=0 | mean=0.39 | dominant=0</td>
        </tr>
        <tr>
          <th id="T_976f2_level0_row8" class="row_heading level0 row8" >8</th>
          <td id="T_976f2_row8_col0" class="data row8 col0" >pclass</td>
          <td id="T_976f2_row8_col1" class="data row8 col1" >int64</td>
          <td id="T_976f2_row8_col2" class="data row8 col2" >0.0%</td>
          <td id="T_976f2_row8_col3" class="data row8 col3" >54.2%</td>
          <td id="T_976f2_row8_col4" class="data row8 col4" >1309</td>
          <td id="T_976f2_row8_col5" class="data row8 col5" >3</td>
          <td id="T_976f2_row8_col6" class="data row8 col6" >max=3 | min=1 | mean=2.29 | dominant=3</td>
        </tr>
        <tr>
          <th id="T_976f2_level0_row9" class="row_heading level0 row9" >9</th>
          <td id="T_976f2_row9_col0" class="data row9 col0" >sex</td>
          <td id="T_976f2_row9_col1" class="data row9 col1" >object</td>
          <td id="T_976f2_row9_col2" class="data row9 col2" >0.0%</td>
          <td id="T_976f2_row9_col3" class="data row9 col3" >64.4%</td>
          <td id="T_976f2_row9_col4" class="data row9 col4" >1309</td>
          <td id="T_976f2_row9_col5" class="data row9 col5" >2</td>
          <td id="T_976f2_row9_col6" class="data row9 col6" >Sample: male | female</td>
        </tr>
        <tr>
          <th id="T_976f2_level0_row10" class="row_heading level0 row10" >10</th>
          <td id="T_976f2_row10_col0" class="data row10 col0" >sibsp</td>
          <td id="T_976f2_row10_col1" class="data row10 col1" >int64</td>
          <td id="T_976f2_row10_col2" class="data row10 col2" >0.0%</td>
          <td id="T_976f2_row10_col3" class="data row10 col3" >68.1%</td>
          <td id="T_976f2_row10_col4" class="data row10 col4" >1309</td>
          <td id="T_976f2_row10_col5" class="data row10 col5" >7</td>
          <td id="T_976f2_row10_col6" class="data row10 col6" >max=8 | min=0 | mean=0.5 | dominant=0</td>
        </tr>
        <tr>
          <th id="T_976f2_level0_row11" class="row_heading level0 row11" >11</th>
          <td id="T_976f2_row11_col0" class="data row11 col0" >survived</td>
          <td id="T_976f2_row11_col1" class="data row11 col1" >int64</td>
          <td id="T_976f2_row11_col2" class="data row11 col2" >0.0%</td>
          <td id="T_976f2_row11_col3" class="data row11 col3" >61.8%</td>
          <td id="T_976f2_row11_col4" class="data row11 col4" >1309</td>
          <td id="T_976f2_row11_col5" class="data row11 col5" >2</td>
          <td id="T_976f2_row11_col6" class="data row11 col6" >max=1 | min=0 | mean=0.38 | dominant=0</td>
        </tr>
        <tr>
          <th id="T_976f2_level0_row12" class="row_heading level0 row12" >12</th>
          <td id="T_976f2_row12_col0" class="data row12 col0" >ticket</td>
          <td id="T_976f2_row12_col1" class="data row12 col1" >object</td>
          <td id="T_976f2_row12_col2" class="data row12 col2" >0.0%</td>
          <td id="T_976f2_row12_col3" class="data row12 col3" >0.8%</td>
          <td id="T_976f2_row12_col4" class="data row12 col4" >1309</td>
          <td id="T_976f2_row12_col5" class="data row12 col5" >929</td>
          <td id="T_976f2_row12_col6" class="data row12 col6" >Sample: CA. 2343 | 1601 | CA 2144 | PC 17608 | 347077</td>
        </tr>
      </tbody>
    </table>




As we continue the observations we see more columns that are of limited
interest and need to be removed as part of the selection process.
Because the components intent action is mutable we can re-implement the
``to_remove`` including the new headers within the list. As this
overwrites the original component intent we must make sure to include
the **name** Column.

.. code:: ipython3

    df = tr.tools.to_remove(df, headers=['name', 'boat', 'body', 'home.dest'])

As the target is a cluster algorithm we can use the ``auto_to_category``
to ensure the data **typing** is appropriate to the column type.

.. code:: ipython3

    df = tr.tools.auto_to_category(df, unique_max=20)

Finally we ensure the two contigious columns are set to numeric type. It
is worth noting though age is an interger, Python does not recognise
nulls within an interger type and automaticially choses it as a float
type.

.. code:: ipython3

    df = tr.tools.to_numeric_type(df, headers=['age', 'fare'])

Using the Intent reporting tool to check the work and see what the
Intent currently looks like all together.

.. code:: ipython3

    tr.report_intent()




.. raw:: html

    <style type="text/css">
    #T_64277 th {
      font-size: 120%;
      text-align: center;
    }
    #T_64277 .row_heading {
      display: none;;
    }
    #T_64277  .blank {
      display: none;;
    }
    #T_64277_row0_col0, #T_64277_row1_col0, #T_64277_row2_col0, #T_64277_row3_col0 {
      text-align: left;
      font-weight: bold;
      font-size: 120%;
    }
    #T_64277_row0_col1, #T_64277_row0_col2, #T_64277_row0_col3, #T_64277_row0_col4, #T_64277_row1_col1, #T_64277_row1_col2, #T_64277_row1_col3, #T_64277_row1_col4, #T_64277_row2_col1, #T_64277_row2_col2, #T_64277_row2_col3, #T_64277_row2_col4, #T_64277_row3_col1, #T_64277_row3_col2, #T_64277_row3_col3, #T_64277_row3_col4 {
      text-align: left;
    }
    </style>
    <table id="T_64277">
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_64277_level0_col0" class="col_heading level0 col0" >level</th>
          <th id="T_64277_level0_col1" class="col_heading level0 col1" >order</th>
          <th id="T_64277_level0_col2" class="col_heading level0 col2" >intent</th>
          <th id="T_64277_level0_col3" class="col_heading level0 col3" >parameters</th>
          <th id="T_64277_level0_col4" class="col_heading level0 col4" >creator</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_64277_level0_row0" class="row_heading level0 row0" >0</th>
          <td id="T_64277_row0_col0" class="data row0 col0" >base</td>
          <td id="T_64277_row0_col1" class="data row0 col1" >0</td>
          <td id="T_64277_row0_col2" class="data row0 col2" >auto_reinstate_nulls</td>
          <td id="T_64277_row0_col3" class="data row0 col3" >["nulls_list=['?']"]</td>
          <td id="T_64277_row0_col4" class="data row0 col4" >doatridge</td>
        </tr>
        <tr>
          <th id="T_64277_level0_row1" class="row_heading level0 row1" >1</th>
          <td id="T_64277_row1_col0" class="data row1 col0" ></td>
          <td id="T_64277_row1_col1" class="data row1 col1" >0</td>
          <td id="T_64277_row1_col2" class="data row1 col2" >auto_to_category</td>
          <td id="T_64277_row1_col3" class="data row1 col3" >['unique_max=20']</td>
          <td id="T_64277_row1_col4" class="data row1 col4" >doatridge</td>
        </tr>
        <tr>
          <th id="T_64277_level0_row2" class="row_heading level0 row2" >2</th>
          <td id="T_64277_row2_col0" class="data row2 col0" ></td>
          <td id="T_64277_row2_col1" class="data row2 col1" >0</td>
          <td id="T_64277_row2_col2" class="data row2 col2" >to_numeric_type</td>
          <td id="T_64277_row2_col3" class="data row2 col3" >["headers=['age', 'fare']"]</td>
          <td id="T_64277_row2_col4" class="data row2 col4" >doatridge</td>
        </tr>
        <tr>
          <th id="T_64277_level0_row3" class="row_heading level0 row3" >3</th>
          <td id="T_64277_row3_col0" class="data row3 col0" ></td>
          <td id="T_64277_row3_col1" class="data row3 col1" >0</td>
          <td id="T_64277_row3_col2" class="data row3 col2" >to_remove</td>
          <td id="T_64277_row3_col3" class="data row3 col3" >["headers=['name', 'boat', 'body', 'home.dest']"]</td>
          <td id="T_64277_row3_col4" class="data row3 col4" >doatridge</td>
        </tr>
      </tbody>
    </table>




Adding these actions or the components intent is a process of looking at
the raw data and the observer making decisions on the selection of the
features of interest. Therefore component selection is potentially an
iterative task where we would add component intent, observe the changes
and then repeat until the process is complete.

--------------

Ordering the Actions of a Component
-----------------------------------

With the component intent now defined the run pipeline does its best to
guess the best order of that Intent but sometimes we want to ensure
things run in a certain order due to dependancies or other challenges.
Though not necessary, we will clear the previous Intent and write it
again, this time in order.

.. code:: ipython3

    tr.remove_intent()




.. parsed-literal::

    True



.. code:: ipython3

    tr.report_intent()




.. raw:: html

    <style type="text/css">
    #T_e3dec th {
      font-size: 120%;
      text-align: center;
    }
    #T_e3dec .row_heading {
      display: none;;
    }
    #T_e3dec  .blank {
      display: none;;
    }
    </style>
    <table id="T_e3dec">
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_e3dec_level0_col0" class="col_heading level0 col0" >level</th>
          <th id="T_e3dec_level0_col1" class="col_heading level0 col1" >order</th>
          <th id="T_e3dec_level0_col2" class="col_heading level0 col2" >intent</th>
          <th id="T_e3dec_level0_col3" class="col_heading level0 col3" >parameters</th>
          <th id="T_e3dec_level0_col4" class="col_heading level0 col4" >creator</th>
        </tr>
      </thead>
      <tbody>
      </tbody>
    </table>




This time when we add the Intent we include the parameter
``intent_level`` to indicate the different order or level of execution.

We load the source canonical and repeat the Intent, this time including
the new intent level.

.. code:: ipython3

    df = tr.load_source_canonical()

.. code:: ipython3

    df = tr.tools.auto_reinstate_nulls(df, nulls_list=['?'], intent_level='reinstate')
    df = tr.tools.to_remove(df, headers=['name', 'boat', 'body', 'home.dest'], intent_level='remove')
    df = tr.tools.auto_to_category(df, unique_max=20, intent_level='auto_category')
    df = tr.tools.to_numeric_type(df, headers=['age', 'fare'], intent_level='to_dtype')
    df = tr.tools.to_str_type(df, headers=['cabin', 'ticket'],use_string_type=True , intent_level='to_dtype')

In addition, and as an introduction to a new feature, we will add in the
column description that describes the reasoning behind why an Intent was
added.

.. code:: ipython3

    tr.add_column_description('reinstate', description="reinstate nulls that where obfuscated with '?'")
    tr.add_column_description('remove', description="remove column of no value")
    tr.add_column_description('auto_category', description="auto fit features to categories where their uniqueness is 20 or less")
    tr.add_column_description('to_dtype', description="ensure all other columns of interest are appropriately typed")


Using the report we can see the addition of the numbers, in the level
column, which helps the run component run the tasks in the order given.
It is worth noting that the tasks can be given the same level if the
order is not important and the run component will deal with it using its
ordering algorithm.

.. code:: ipython3

    tr.report_intent()




.. raw:: html

    <style type="text/css">
    #T_9034a th {
      font-size: 120%;
      text-align: center;
    }
    #T_9034a .row_heading {
      display: none;;
    }
    #T_9034a  .blank {
      display: none;;
    }
    #T_9034a_row0_col0, #T_9034a_row1_col0, #T_9034a_row2_col0, #T_9034a_row3_col0, #T_9034a_row4_col0 {
      text-align: left;
      font-weight: bold;
      font-size: 120%;
    }
    #T_9034a_row0_col1, #T_9034a_row0_col2, #T_9034a_row0_col3, #T_9034a_row0_col4, #T_9034a_row1_col1, #T_9034a_row1_col2, #T_9034a_row1_col3, #T_9034a_row1_col4, #T_9034a_row2_col1, #T_9034a_row2_col2, #T_9034a_row2_col3, #T_9034a_row2_col4, #T_9034a_row3_col1, #T_9034a_row3_col2, #T_9034a_row3_col3, #T_9034a_row3_col4, #T_9034a_row4_col1, #T_9034a_row4_col2, #T_9034a_row4_col3, #T_9034a_row4_col4 {
      text-align: left;
    }
    </style>
    <table id="T_9034a">
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_9034a_level0_col0" class="col_heading level0 col0" >level</th>
          <th id="T_9034a_level0_col1" class="col_heading level0 col1" >order</th>
          <th id="T_9034a_level0_col2" class="col_heading level0 col2" >intent</th>
          <th id="T_9034a_level0_col3" class="col_heading level0 col3" >parameters</th>
          <th id="T_9034a_level0_col4" class="col_heading level0 col4" >creator</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_9034a_level0_row0" class="row_heading level0 row0" >0</th>
          <td id="T_9034a_row0_col0" class="data row0 col0" >auto_category</td>
          <td id="T_9034a_row0_col1" class="data row0 col1" >0</td>
          <td id="T_9034a_row0_col2" class="data row0 col2" >auto_to_category</td>
          <td id="T_9034a_row0_col3" class="data row0 col3" >['unique_max=20']</td>
          <td id="T_9034a_row0_col4" class="data row0 col4" >doatridge</td>
        </tr>
        <tr>
          <th id="T_9034a_level0_row1" class="row_heading level0 row1" >1</th>
          <td id="T_9034a_row1_col0" class="data row1 col0" >reinstate</td>
          <td id="T_9034a_row1_col1" class="data row1 col1" >0</td>
          <td id="T_9034a_row1_col2" class="data row1 col2" >auto_reinstate_nulls</td>
          <td id="T_9034a_row1_col3" class="data row1 col3" >["nulls_list=['?']"]</td>
          <td id="T_9034a_row1_col4" class="data row1 col4" >doatridge</td>
        </tr>
        <tr>
          <th id="T_9034a_level0_row2" class="row_heading level0 row2" >2</th>
          <td id="T_9034a_row2_col0" class="data row2 col0" >remove</td>
          <td id="T_9034a_row2_col1" class="data row2 col1" >0</td>
          <td id="T_9034a_row2_col2" class="data row2 col2" >to_remove</td>
          <td id="T_9034a_row2_col3" class="data row2 col3" >["headers=['name', 'boat', 'body', 'home.dest']"]</td>
          <td id="T_9034a_row2_col4" class="data row2 col4" >doatridge</td>
        </tr>
        <tr>
          <th id="T_9034a_level0_row3" class="row_heading level0 row3" >3</th>
          <td id="T_9034a_row3_col0" class="data row3 col0" >to_dtype</td>
          <td id="T_9034a_row3_col1" class="data row3 col1" >0</td>
          <td id="T_9034a_row3_col2" class="data row3 col2" >to_numeric_type</td>
          <td id="T_9034a_row3_col3" class="data row3 col3" >["headers=['age', 'fare']"]</td>
          <td id="T_9034a_row3_col4" class="data row3 col4" >doatridge</td>
        </tr>
        <tr>
          <th id="T_9034a_level0_row4" class="row_heading level0 row4" >4</th>
          <td id="T_9034a_row4_col0" class="data row4 col0" ></td>
          <td id="T_9034a_row4_col1" class="data row4 col1" >0</td>
          <td id="T_9034a_row4_col2" class="data row4 col2" >to_str_type</td>
          <td id="T_9034a_row4_col3" class="data row4 col3" >["headers=['cabin', 'ticket']", 'use_string_type=True']</td>
          <td id="T_9034a_row4_col4" class="data row4 col4" >doatridge</td>
        </tr>
      </tbody>
    </table>




As we have taken the time to capture the reasoning to include the
compoment Intent we can use the reports to produce a view of the Intent
column comments that are invaluable when interrogating a component and
understanding why decisions were made.

.. code:: ipython3

    tr.report_column_catalog()




.. raw:: html

    <style type="text/css">
    #T_bf327 th {
      font-size: 120%;
      text-align: center;
    }
    #T_bf327 .row_heading {
      display: none;;
    }
    #T_bf327  .blank {
      display: none;;
    }
    #T_bf327_row0_col0, #T_bf327_row1_col0, #T_bf327_row2_col0, #T_bf327_row3_col0 {
      text-align: left;
      font-weight: bold;
    }
    #T_bf327_row0_col1, #T_bf327_row1_col1, #T_bf327_row2_col1, #T_bf327_row3_col1 {
      text-align: left;
    }
    </style>
    <table id="T_bf327">
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_bf327_level0_col0" class="col_heading level0 col0" >column_name</th>
          <th id="T_bf327_level0_col1" class="col_heading level0 col1" >description</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_bf327_level0_row0" class="row_heading level0 row0" >0</th>
          <td id="T_bf327_row0_col0" class="data row0 col0" >auto_category</td>
          <td id="T_bf327_row0_col1" class="data row0 col1" >auto fit features to categories where their uniqueness is 20 or less</td>
        </tr>
        <tr>
          <th id="T_bf327_level0_row1" class="row_heading level0 row1" >1</th>
          <td id="T_bf327_row1_col0" class="data row1 col0" >reinstate</td>
          <td id="T_bf327_row1_col1" class="data row1 col1" >reinstate nulls that where obfuscated with '?'</td>
        </tr>
        <tr>
          <th id="T_bf327_level0_row2" class="row_heading level0 row2" >2</th>
          <td id="T_bf327_row2_col0" class="data row2 col0" >remove</td>
          <td id="T_bf327_row2_col1" class="data row2 col1" >remove column of no value</td>
        </tr>
        <tr>
          <th id="T_bf327_level0_row3" class="row_heading level0 row3" >3</th>
          <td id="T_bf327_row3_col0" class="data row3 col0" >to_dtype</td>
          <td id="T_bf327_row3_col1" class="data row3 col1" >ensure all other columns of interest are appropriately typed</td>
        </tr>
      </tbody>
    </table>




Component Pipeline
------------------

As usual we can now run the Compant pipeline to apply the components
tasks.

.. code:: ipython3

    tr.run_component_pipeline()

As an extension of the default, ``run_component_pipeline`` provides
useful tools to help manage the outcome. In this case we’ve
specificially defined the Intent order we wanted to run.

.. code:: ipython3

    tr.run_component_pipeline(intent_levels=['remove', 'reinstate', 'auto_category', 'to_dtype'])

--------------

Run Books
---------

A challenge faced with the component intent is its order, as you have
seen. The solution thus far only applies at run time and is therefore
not repeatable. We introduced the idea of Run Books as a repeatable set
of instructions which contain the order in which to run the components
intent. Run Books also provide the ability to particially implement
component intent actions, meaning we can replay subsets of a fuller list
of a components intent. For example through experimentation we have
created a number of additional component intents, that are not pertinent
to a production ready selection. By setting up two Run Books we can
select which component intent is appropriate to their objectives and
``run_component_pipeline`` to produce the appropriate outcome.

In the example we add our list of intent to a book in the order needed.
In this case we have not specified a book name so this book is allocated
to the primary Run Book. Now each time we run pipeline, it is set to run
the primary Run Book.

.. code:: ipython3

    tr.add_run_book(run_levels=['remove', 'reinstate', 'auto_category', 'to_dtype'])

Here we had a book by name where we select only the intent that cleans
the raw data. The Run book report Now what are shows us the two run
books;

.. code:: ipython3

    tr.add_run_book(book_name='cleaner', run_levels=['remove', 'reinstate'])

.. code:: ipython3

    tr.report_run_book()




.. raw:: html

    <style type="text/css">
    #T_22ae8 th {
      font-size: 120%;
      text-align: center;
    }
    #T_22ae8 .row_heading {
      display: none;;
    }
    #T_22ae8  .blank {
      display: none;;
    }
    #T_22ae8_row0_col0, #T_22ae8_row1_col0 {
      text-align: left;
      font-weight: bold;
      font-size: 120%;
    }
    #T_22ae8_row0_col1, #T_22ae8_row1_col1 {
      text-align: left;
    }
    </style>
    <table id="T_22ae8">
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_22ae8_level0_col0" class="col_heading level0 col0" >name</th>
          <th id="T_22ae8_level0_col1" class="col_heading level0 col1" >run_book</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_22ae8_level0_row0" class="row_heading level0 row0" >0</th>
          <td id="T_22ae8_row0_col0" class="data row0 col0" >primary_run_book</td>
          <td id="T_22ae8_row0_col1" class="data row0 col1" >['remove', 'reinstate', 'auto_category', 'to_dtype']</td>
        </tr>
        <tr>
          <th id="T_22ae8_level0_row1" class="row_heading level0 row1" >1</th>
          <td id="T_22ae8_row1_col0" class="data row1 col0" >cleaner</td>
          <td id="T_22ae8_row1_col1" class="data row1 col1" >['remove', 'reinstate']</td>
        </tr>
      </tbody>
    </table>




In this next example we add an additional Run Book that is a subset of
the tasks to only clean the data. By passing this named Run Book to the
run pipeline it is obliged to only run this subset and only clean the
data. We can see the results of this in our canonical report below.

.. code:: ipython3

    tr.run_component_pipeline(run_book='cleaner')

.. code:: ipython3

    tr.canonical_report(tr.load_persist_canonical())




.. raw:: html

    <style type="text/css">
    #T_f4d48 th {
      font-size: 120%;
      text-align: center;
    }
    #T_f4d48 .row_heading {
      display: none;;
    }
    #T_f4d48  .blank {
      display: none;;
    }
    #T_f4d48_row0_col0, #T_f4d48_row1_col0, #T_f4d48_row2_col0, #T_f4d48_row3_col0, #T_f4d48_row4_col0, #T_f4d48_row5_col0, #T_f4d48_row6_col0, #T_f4d48_row7_col0, #T_f4d48_row8_col0, #T_f4d48_row9_col0 {
      font-weight: bold;
      font-size: 120%;
    }
    #T_f4d48_row0_col2, #T_f4d48_row0_col3, #T_f4d48_row1_col5, #T_f4d48_row2_col2, #T_f4d48_row3_col2, #T_f4d48_row3_col3, #T_f4d48_row3_col5, #T_f4d48_row4_col2, #T_f4d48_row5_col2, #T_f4d48_row5_col3, #T_f4d48_row6_col2, #T_f4d48_row6_col3, #T_f4d48_row7_col2, #T_f4d48_row8_col2, #T_f4d48_row8_col3, #T_f4d48_row9_col2, #T_f4d48_row9_col3, #T_f4d48_row9_col5 {
      color: black;
    }
    #T_f4d48_row0_col5 {
      background-color: #f0f9ed;
      color: black;
    }
    #T_f4d48_row1_col2, #T_f4d48_row1_col3, #T_f4d48_row2_col3, #T_f4d48_row4_col3, #T_f4d48_row7_col3 {
      background-color: #ffede5;
      color: black;
    }
    #T_f4d48_row2_col5, #T_f4d48_row5_col5 {
      background-color: #84cc83;
      color: black;
    }
    #T_f4d48_row4_col1, #T_f4d48_row5_col1, #T_f4d48_row7_col1, #T_f4d48_row8_col1 {
      color: #0f398a;
    }
    #T_f4d48_row4_col5, #T_f4d48_row7_col5 {
      background-color: #a4da9e;
      color: black;
    }
    #T_f4d48_row6_col5, #T_f4d48_row8_col5 {
      background-color: #a1cbe2;
      color: black;
    }
    </style>
    <table id="T_f4d48">
      <caption>%_Dom: The % most dominant element </caption>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_f4d48_level0_col0" class="col_heading level0 col0" >Attributes (10)</th>
          <th id="T_f4d48_level0_col1" class="col_heading level0 col1" >dType</th>
          <th id="T_f4d48_level0_col2" class="col_heading level0 col2" >%_Null</th>
          <th id="T_f4d48_level0_col3" class="col_heading level0 col3" >%_Dom</th>
          <th id="T_f4d48_level0_col4" class="col_heading level0 col4" >Count</th>
          <th id="T_f4d48_level0_col5" class="col_heading level0 col5" >Unique</th>
          <th id="T_f4d48_level0_col6" class="col_heading level0 col6" >Observations</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_f4d48_level0_row0" class="row_heading level0 row0" >0</th>
          <td id="T_f4d48_row0_col0" class="data row0 col0" >age</td>
          <td id="T_f4d48_row0_col1" class="data row0 col1" >object</td>
          <td id="T_f4d48_row0_col2" class="data row0 col2" >20.1%</td>
          <td id="T_f4d48_row0_col3" class="data row0 col3" >20.1%</td>
          <td id="T_f4d48_row0_col4" class="data row0 col4" >1309</td>
          <td id="T_f4d48_row0_col5" class="data row0 col5" >99</td>
          <td id="T_f4d48_row0_col6" class="data row0 col6" >Sample: 24 | 22 | 21 | 30 | 18</td>
        </tr>
        <tr>
          <th id="T_f4d48_level0_row1" class="row_heading level0 row1" >1</th>
          <td id="T_f4d48_row1_col0" class="data row1 col0" >cabin</td>
          <td id="T_f4d48_row1_col1" class="data row1 col1" >object</td>
          <td id="T_f4d48_row1_col2" class="data row1 col2" >77.5%</td>
          <td id="T_f4d48_row1_col3" class="data row1 col3" >77.5%</td>
          <td id="T_f4d48_row1_col4" class="data row1 col4" >1309</td>
          <td id="T_f4d48_row1_col5" class="data row1 col5" >187</td>
          <td id="T_f4d48_row1_col6" class="data row1 col6" >Sample: C23 C25 C27 | G6 | B57 B59 B63 B66 | F4 | F33</td>
        </tr>
        <tr>
          <th id="T_f4d48_level0_row2" class="row_heading level0 row2" >2</th>
          <td id="T_f4d48_row2_col0" class="data row2 col0" >embarked</td>
          <td id="T_f4d48_row2_col1" class="data row2 col1" >object</td>
          <td id="T_f4d48_row2_col2" class="data row2 col2" >0.2%</td>
          <td id="T_f4d48_row2_col3" class="data row2 col3" >69.8%</td>
          <td id="T_f4d48_row2_col4" class="data row2 col4" >1309</td>
          <td id="T_f4d48_row2_col5" class="data row2 col5" >4</td>
          <td id="T_f4d48_row2_col6" class="data row2 col6" >Sample: S | C | Q</td>
        </tr>
        <tr>
          <th id="T_f4d48_level0_row3" class="row_heading level0 row3" >3</th>
          <td id="T_f4d48_row3_col0" class="data row3 col0" >fare</td>
          <td id="T_f4d48_row3_col1" class="data row3 col1" >object</td>
          <td id="T_f4d48_row3_col2" class="data row3 col2" >0.1%</td>
          <td id="T_f4d48_row3_col3" class="data row3 col3" >4.6%</td>
          <td id="T_f4d48_row3_col4" class="data row3 col4" >1309</td>
          <td id="T_f4d48_row3_col5" class="data row3 col5" >282</td>
          <td id="T_f4d48_row3_col6" class="data row3 col6" >Sample: 8.05 | 13 | 7.75 | 26 | 7.8958</td>
        </tr>
        <tr>
          <th id="T_f4d48_level0_row4" class="row_heading level0 row4" >4</th>
          <td id="T_f4d48_row4_col0" class="data row4 col0" >parch</td>
          <td id="T_f4d48_row4_col1" class="data row4 col1" >int64</td>
          <td id="T_f4d48_row4_col2" class="data row4 col2" >0.0%</td>
          <td id="T_f4d48_row4_col3" class="data row4 col3" >76.5%</td>
          <td id="T_f4d48_row4_col4" class="data row4 col4" >1309</td>
          <td id="T_f4d48_row4_col5" class="data row4 col5" >8</td>
          <td id="T_f4d48_row4_col6" class="data row4 col6" >max=9 | min=0 | mean=0.39 | dominant=0</td>
        </tr>
        <tr>
          <th id="T_f4d48_level0_row5" class="row_heading level0 row5" >5</th>
          <td id="T_f4d48_row5_col0" class="data row5 col0" >pclass</td>
          <td id="T_f4d48_row5_col1" class="data row5 col1" >int64</td>
          <td id="T_f4d48_row5_col2" class="data row5 col2" >0.0%</td>
          <td id="T_f4d48_row5_col3" class="data row5 col3" >54.2%</td>
          <td id="T_f4d48_row5_col4" class="data row5 col4" >1309</td>
          <td id="T_f4d48_row5_col5" class="data row5 col5" >3</td>
          <td id="T_f4d48_row5_col6" class="data row5 col6" >max=3 | min=1 | mean=2.29 | dominant=3</td>
        </tr>
        <tr>
          <th id="T_f4d48_level0_row6" class="row_heading level0 row6" >6</th>
          <td id="T_f4d48_row6_col0" class="data row6 col0" >sex</td>
          <td id="T_f4d48_row6_col1" class="data row6 col1" >object</td>
          <td id="T_f4d48_row6_col2" class="data row6 col2" >0.0%</td>
          <td id="T_f4d48_row6_col3" class="data row6 col3" >64.4%</td>
          <td id="T_f4d48_row6_col4" class="data row6 col4" >1309</td>
          <td id="T_f4d48_row6_col5" class="data row6 col5" >2</td>
          <td id="T_f4d48_row6_col6" class="data row6 col6" >Sample: male | female</td>
        </tr>
        <tr>
          <th id="T_f4d48_level0_row7" class="row_heading level0 row7" >7</th>
          <td id="T_f4d48_row7_col0" class="data row7 col0" >sibsp</td>
          <td id="T_f4d48_row7_col1" class="data row7 col1" >int64</td>
          <td id="T_f4d48_row7_col2" class="data row7 col2" >0.0%</td>
          <td id="T_f4d48_row7_col3" class="data row7 col3" >68.1%</td>
          <td id="T_f4d48_row7_col4" class="data row7 col4" >1309</td>
          <td id="T_f4d48_row7_col5" class="data row7 col5" >7</td>
          <td id="T_f4d48_row7_col6" class="data row7 col6" >max=8 | min=0 | mean=0.5 | dominant=0</td>
        </tr>
        <tr>
          <th id="T_f4d48_level0_row8" class="row_heading level0 row8" >8</th>
          <td id="T_f4d48_row8_col0" class="data row8 col0" >survived</td>
          <td id="T_f4d48_row8_col1" class="data row8 col1" >int64</td>
          <td id="T_f4d48_row8_col2" class="data row8 col2" >0.0%</td>
          <td id="T_f4d48_row8_col3" class="data row8 col3" >61.8%</td>
          <td id="T_f4d48_row8_col4" class="data row8 col4" >1309</td>
          <td id="T_f4d48_row8_col5" class="data row8 col5" >2</td>
          <td id="T_f4d48_row8_col6" class="data row8 col6" >max=1 | min=0 | mean=0.38 | dominant=0</td>
        </tr>
        <tr>
          <th id="T_f4d48_level0_row9" class="row_heading level0 row9" >9</th>
          <td id="T_f4d48_row9_col0" class="data row9 col0" >ticket</td>
          <td id="T_f4d48_row9_col1" class="data row9 col1" >object</td>
          <td id="T_f4d48_row9_col2" class="data row9 col2" >0.0%</td>
          <td id="T_f4d48_row9_col3" class="data row9 col3" >0.8%</td>
          <td id="T_f4d48_row9_col4" class="data row9 col4" >1309</td>
          <td id="T_f4d48_row9_col5" class="data row9 col5" >929</td>
          <td id="T_f4d48_row9_col6" class="data row9 col6" >Sample: CA. 2343 | 1601 | CA 2144 | PC 17608 | 347077</td>
        </tr>
      </tbody>
    </table>




As a contrast to the above we can run the pipeline without providing a
Run Book name and it will automatically default to the primary run book,
assuming this has been set up. In this case running the full component
Intent the resulting outcome is shown below in the canonical report.

.. code:: ipython3

    tr.run_component_pipeline()

.. code:: ipython3

    tr.canonical_report(tr.load_persist_canonical())




.. raw:: html

    <style type="text/css">
    #T_c1e31 th {
      font-size: 120%;
      text-align: center;
    }
    #T_c1e31 .row_heading {
      display: none;;
    }
    #T_c1e31  .blank {
      display: none;;
    }
    #T_c1e31_row0_col0, #T_c1e31_row1_col0, #T_c1e31_row2_col0, #T_c1e31_row3_col0, #T_c1e31_row4_col0, #T_c1e31_row5_col0, #T_c1e31_row6_col0, #T_c1e31_row7_col0, #T_c1e31_row8_col0, #T_c1e31_row9_col0 {
      font-weight: bold;
      font-size: 120%;
    }
    #T_c1e31_row0_col1, #T_c1e31_row3_col1 {
      color: #2f0f8a;
    }
    #T_c1e31_row0_col2, #T_c1e31_row0_col3, #T_c1e31_row1_col5, #T_c1e31_row2_col2, #T_c1e31_row3_col2, #T_c1e31_row3_col3, #T_c1e31_row3_col5, #T_c1e31_row4_col2, #T_c1e31_row5_col2, #T_c1e31_row5_col3, #T_c1e31_row6_col2, #T_c1e31_row6_col3, #T_c1e31_row7_col2, #T_c1e31_row8_col2, #T_c1e31_row8_col3, #T_c1e31_row9_col2, #T_c1e31_row9_col3, #T_c1e31_row9_col5 {
      color: black;
    }
    #T_c1e31_row0_col5 {
      background-color: #f0f9ed;
      color: black;
    }
    #T_c1e31_row1_col1, #T_c1e31_row9_col1 {
      color: #761d38;
    }
    #T_c1e31_row1_col2, #T_c1e31_row1_col3, #T_c1e31_row2_col3, #T_c1e31_row4_col3, #T_c1e31_row7_col3 {
      background-color: #ffede5;
      color: black;
    }
    #T_c1e31_row2_col1, #T_c1e31_row4_col1, #T_c1e31_row5_col1, #T_c1e31_row6_col1, #T_c1e31_row7_col1, #T_c1e31_row8_col1 {
      color: #208a0f;
    }
    #T_c1e31_row2_col5, #T_c1e31_row5_col5 {
      background-color: #84cc83;
      color: black;
    }
    #T_c1e31_row4_col5, #T_c1e31_row7_col5 {
      background-color: #a4da9e;
      color: black;
    }
    #T_c1e31_row6_col5, #T_c1e31_row8_col5 {
      background-color: #a1cbe2;
      color: black;
    }
    </style>
    <table id="T_c1e31">
      <caption>%_Dom: The % most dominant element </caption>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_c1e31_level0_col0" class="col_heading level0 col0" >Attributes (10)</th>
          <th id="T_c1e31_level0_col1" class="col_heading level0 col1" >dType</th>
          <th id="T_c1e31_level0_col2" class="col_heading level0 col2" >%_Null</th>
          <th id="T_c1e31_level0_col3" class="col_heading level0 col3" >%_Dom</th>
          <th id="T_c1e31_level0_col4" class="col_heading level0 col4" >Count</th>
          <th id="T_c1e31_level0_col5" class="col_heading level0 col5" >Unique</th>
          <th id="T_c1e31_level0_col6" class="col_heading level0 col6" >Observations</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_c1e31_level0_row0" class="row_heading level0 row0" >0</th>
          <td id="T_c1e31_row0_col0" class="data row0 col0" >age</td>
          <td id="T_c1e31_row0_col1" class="data row0 col1" >float64</td>
          <td id="T_c1e31_row0_col2" class="data row0 col2" >20.1%</td>
          <td id="T_c1e31_row0_col3" class="data row0 col3" >20.1%</td>
          <td id="T_c1e31_row0_col4" class="data row0 col4" >1309</td>
          <td id="T_c1e31_row0_col5" class="data row0 col5" >99</td>
          <td id="T_c1e31_row0_col6" class="data row0 col6" >max=80.0 | min=0.1667 | mean=29.88 | dominant=24.0</td>
        </tr>
        <tr>
          <th id="T_c1e31_level0_row1" class="row_heading level0 row1" >1</th>
          <td id="T_c1e31_row1_col0" class="data row1 col0" >cabin</td>
          <td id="T_c1e31_row1_col1" class="data row1 col1" >string</td>
          <td id="T_c1e31_row1_col2" class="data row1 col2" >77.5%</td>
          <td id="T_c1e31_row1_col3" class="data row1 col3" >77.5%</td>
          <td id="T_c1e31_row1_col4" class="data row1 col4" >1309</td>
          <td id="T_c1e31_row1_col5" class="data row1 col5" >187</td>
          <td id="T_c1e31_row1_col6" class="data row1 col6" >Sample: C23 C25 C27 | G6 | B57 B59 B63 B66 | F4 | F33</td>
        </tr>
        <tr>
          <th id="T_c1e31_level0_row2" class="row_heading level0 row2" >2</th>
          <td id="T_c1e31_row2_col0" class="data row2 col0" >embarked</td>
          <td id="T_c1e31_row2_col1" class="data row2 col1" >category</td>
          <td id="T_c1e31_row2_col2" class="data row2 col2" >0.0%</td>
          <td id="T_c1e31_row2_col3" class="data row2 col3" >69.8%</td>
          <td id="T_c1e31_row2_col4" class="data row2 col4" >1309</td>
          <td id="T_c1e31_row2_col5" class="data row2 col5" >4</td>
          <td id="T_c1e31_row2_col6" class="data row2 col6" >Sample: S | C | Q | nan</td>
        </tr>
        <tr>
          <th id="T_c1e31_level0_row3" class="row_heading level0 row3" >3</th>
          <td id="T_c1e31_row3_col0" class="data row3 col0" >fare</td>
          <td id="T_c1e31_row3_col1" class="data row3 col1" >float64</td>
          <td id="T_c1e31_row3_col2" class="data row3 col2" >0.1%</td>
          <td id="T_c1e31_row3_col3" class="data row3 col3" >4.6%</td>
          <td id="T_c1e31_row3_col4" class="data row3 col4" >1309</td>
          <td id="T_c1e31_row3_col5" class="data row3 col5" >282</td>
          <td id="T_c1e31_row3_col6" class="data row3 col6" >max=512.3292 | min=0.0 | mean=33.3 | dominant=8.05</td>
        </tr>
        <tr>
          <th id="T_c1e31_level0_row4" class="row_heading level0 row4" >4</th>
          <td id="T_c1e31_row4_col0" class="data row4 col0" >parch</td>
          <td id="T_c1e31_row4_col1" class="data row4 col1" >category</td>
          <td id="T_c1e31_row4_col2" class="data row4 col2" >0.0%</td>
          <td id="T_c1e31_row4_col3" class="data row4 col3" >76.5%</td>
          <td id="T_c1e31_row4_col4" class="data row4 col4" >1309</td>
          <td id="T_c1e31_row4_col5" class="data row4 col5" >8</td>
          <td id="T_c1e31_row4_col6" class="data row4 col6" >Sample: 0 | 1 | 2 | 3 | 4</td>
        </tr>
        <tr>
          <th id="T_c1e31_level0_row5" class="row_heading level0 row5" >5</th>
          <td id="T_c1e31_row5_col0" class="data row5 col0" >pclass</td>
          <td id="T_c1e31_row5_col1" class="data row5 col1" >category</td>
          <td id="T_c1e31_row5_col2" class="data row5 col2" >0.0%</td>
          <td id="T_c1e31_row5_col3" class="data row5 col3" >54.2%</td>
          <td id="T_c1e31_row5_col4" class="data row5 col4" >1309</td>
          <td id="T_c1e31_row5_col5" class="data row5 col5" >3</td>
          <td id="T_c1e31_row5_col6" class="data row5 col6" >Sample: 3 | 1 | 2</td>
        </tr>
        <tr>
          <th id="T_c1e31_level0_row6" class="row_heading level0 row6" >6</th>
          <td id="T_c1e31_row6_col0" class="data row6 col0" >sex</td>
          <td id="T_c1e31_row6_col1" class="data row6 col1" >category</td>
          <td id="T_c1e31_row6_col2" class="data row6 col2" >0.0%</td>
          <td id="T_c1e31_row6_col3" class="data row6 col3" >64.4%</td>
          <td id="T_c1e31_row6_col4" class="data row6 col4" >1309</td>
          <td id="T_c1e31_row6_col5" class="data row6 col5" >2</td>
          <td id="T_c1e31_row6_col6" class="data row6 col6" >Sample: male | female</td>
        </tr>
        <tr>
          <th id="T_c1e31_level0_row7" class="row_heading level0 row7" >7</th>
          <td id="T_c1e31_row7_col0" class="data row7 col0" >sibsp</td>
          <td id="T_c1e31_row7_col1" class="data row7 col1" >category</td>
          <td id="T_c1e31_row7_col2" class="data row7 col2" >0.0%</td>
          <td id="T_c1e31_row7_col3" class="data row7 col3" >68.1%</td>
          <td id="T_c1e31_row7_col4" class="data row7 col4" >1309</td>
          <td id="T_c1e31_row7_col5" class="data row7 col5" >7</td>
          <td id="T_c1e31_row7_col6" class="data row7 col6" >Sample: 0 | 1 | 2 | 4 | 3</td>
        </tr>
        <tr>
          <th id="T_c1e31_level0_row8" class="row_heading level0 row8" >8</th>
          <td id="T_c1e31_row8_col0" class="data row8 col0" >survived</td>
          <td id="T_c1e31_row8_col1" class="data row8 col1" >category</td>
          <td id="T_c1e31_row8_col2" class="data row8 col2" >0.0%</td>
          <td id="T_c1e31_row8_col3" class="data row8 col3" >61.8%</td>
          <td id="T_c1e31_row8_col4" class="data row8 col4" >1309</td>
          <td id="T_c1e31_row8_col5" class="data row8 col5" >2</td>
          <td id="T_c1e31_row8_col6" class="data row8 col6" >Sample: 0 | 1</td>
        </tr>
        <tr>
          <th id="T_c1e31_level0_row9" class="row_heading level0 row9" >9</th>
          <td id="T_c1e31_row9_col0" class="data row9 col0" >ticket</td>
          <td id="T_c1e31_row9_col1" class="data row9 col1" >string</td>
          <td id="T_c1e31_row9_col2" class="data row9 col2" >0.0%</td>
          <td id="T_c1e31_row9_col3" class="data row9 col3" >0.8%</td>
          <td id="T_c1e31_row9_col4" class="data row9 col4" >1309</td>
          <td id="T_c1e31_row9_col5" class="data row9 col5" >929</td>
          <td id="T_c1e31_row9_col6" class="data row9 col6" >Sample: CA. 2343 | 1601 | CA 2144 | PC 17608 | 347077</td>
        </tr>
      </tbody>
    </table>




