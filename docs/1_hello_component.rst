Darryl Oatridge, August 2022

Building a Component
--------------------

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
~~~~~~~~~~~~~~~~~~~~~~

To run a component we use the common method ``run_component_pipeline``
which loads the source data, executes the component task then persists
the results. This is the only method you can use to run the tasks of a
component and produce its results and should be a familiarized method.

.. code:: ipython3

    tr.run_component_pipeline()

This concludes building a component and though the component doesn’t
change the throughput, it shows the core steps to building any
component.

--------------

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




.. raw:: html

    <style type="text/css">
    #T_7377f th {
      font-size: 120%;
      text-align: center;
    }
    #T_7377f .row_heading {
      display: none;;
    }
    #T_7377f  .blank {
      display: none;;
    }
    #T_7377f_row0_col0, #T_7377f_row1_col0, #T_7377f_row2_col0, #T_7377f_row3_col0, #T_7377f_row4_col0, #T_7377f_row5_col0, #T_7377f_row6_col0, #T_7377f_row7_col0, #T_7377f_row8_col0, #T_7377f_row9_col0, #T_7377f_row10_col0, #T_7377f_row11_col0, #T_7377f_row12_col0, #T_7377f_row13_col0 {
      font-weight: bold;
      font-size: 120%;
    }
    #T_7377f_row0_col2, #T_7377f_row0_col3, #T_7377f_row1_col2, #T_7377f_row1_col3, #T_7377f_row2_col2, #T_7377f_row2_col5, #T_7377f_row3_col2, #T_7377f_row3_col5, #T_7377f_row4_col2, #T_7377f_row5_col2, #T_7377f_row5_col3, #T_7377f_row5_col5, #T_7377f_row6_col2, #T_7377f_row6_col3, #T_7377f_row6_col5, #T_7377f_row7_col2, #T_7377f_row7_col3, #T_7377f_row7_col5, #T_7377f_row8_col2, #T_7377f_row9_col2, #T_7377f_row9_col3, #T_7377f_row10_col2, #T_7377f_row10_col3, #T_7377f_row11_col2, #T_7377f_row12_col2, #T_7377f_row12_col3, #T_7377f_row13_col2, #T_7377f_row13_col3, #T_7377f_row13_col5 {
      color: black;
    }
    #T_7377f_row0_col5 {
      background-color: #f0f9ed;
      color: black;
    }
    #T_7377f_row1_col5 {
      background-color: #e5f5e0;
      color: black;
    }
    #T_7377f_row2_col3 {
      background-color: #fcb499;
      color: black;
    }
    #T_7377f_row3_col3, #T_7377f_row4_col3, #T_7377f_row8_col3, #T_7377f_row11_col3 {
      background-color: #ffede5;
      color: black;
    }
    #T_7377f_row4_col5, #T_7377f_row9_col5 {
      background-color: #84cc83;
      color: black;
    }
    #T_7377f_row8_col1, #T_7377f_row9_col1, #T_7377f_row11_col1, #T_7377f_row12_col1 {
      color: #0f398a;
    }
    #T_7377f_row8_col5, #T_7377f_row11_col5 {
      background-color: #a4da9e;
      color: black;
    }
    #T_7377f_row10_col5, #T_7377f_row12_col5 {
      background-color: #a1cbe2;
      color: black;
    }
    </style>
    <table id="T_7377f">
      <caption>%_Dom: The % most dominant element </caption>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_7377f_level0_col0" class="col_heading level0 col0" >Attributes (14)</th>
          <th id="T_7377f_level0_col1" class="col_heading level0 col1" >dType</th>
          <th id="T_7377f_level0_col2" class="col_heading level0 col2" >%_Null</th>
          <th id="T_7377f_level0_col3" class="col_heading level0 col3" >%_Dom</th>
          <th id="T_7377f_level0_col4" class="col_heading level0 col4" >Count</th>
          <th id="T_7377f_level0_col5" class="col_heading level0 col5" >Unique</th>
          <th id="T_7377f_level0_col6" class="col_heading level0 col6" >Observations</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_7377f_level0_row0" class="row_heading level0 row0" >0</th>
          <td id="T_7377f_row0_col0" class="data row0 col0" >age</td>
          <td id="T_7377f_row0_col1" class="data row0 col1" >object</td>
          <td id="T_7377f_row0_col2" class="data row0 col2" >0.0%</td>
          <td id="T_7377f_row0_col3" class="data row0 col3" >20.1%</td>
          <td id="T_7377f_row0_col4" class="data row0 col4" >1309</td>
          <td id="T_7377f_row0_col5" class="data row0 col5" >99</td>
          <td id="T_7377f_row0_col6" class="data row0 col6" >Sample: ? | 24 | 22 | 21 | 30</td>
        </tr>
        <tr>
          <th id="T_7377f_level0_row1" class="row_heading level0 row1" >1</th>
          <td id="T_7377f_row1_col0" class="data row1 col0" >boat</td>
          <td id="T_7377f_row1_col1" class="data row1 col1" >object</td>
          <td id="T_7377f_row1_col2" class="data row1 col2" >0.0%</td>
          <td id="T_7377f_row1_col3" class="data row1 col3" >62.9%</td>
          <td id="T_7377f_row1_col4" class="data row1 col4" >1309</td>
          <td id="T_7377f_row1_col5" class="data row1 col5" >28</td>
          <td id="T_7377f_row1_col6" class="data row1 col6" >Sample: ? | 13 | C | 15 | 14</td>
        </tr>
        <tr>
          <th id="T_7377f_level0_row2" class="row_heading level0 row2" >2</th>
          <td id="T_7377f_row2_col0" class="data row2 col0" >body</td>
          <td id="T_7377f_row2_col1" class="data row2 col1" >object</td>
          <td id="T_7377f_row2_col2" class="data row2 col2" >0.0%</td>
          <td id="T_7377f_row2_col3" class="data row2 col3" >90.8%</td>
          <td id="T_7377f_row2_col4" class="data row2 col4" >1309</td>
          <td id="T_7377f_row2_col5" class="data row2 col5" >122</td>
          <td id="T_7377f_row2_col6" class="data row2 col6" >Sample: ? | 58 | 285 | 156 | 143</td>
        </tr>
        <tr>
          <th id="T_7377f_level0_row3" class="row_heading level0 row3" >3</th>
          <td id="T_7377f_row3_col0" class="data row3 col0" >cabin</td>
          <td id="T_7377f_row3_col1" class="data row3 col1" >object</td>
          <td id="T_7377f_row3_col2" class="data row3 col2" >0.0%</td>
          <td id="T_7377f_row3_col3" class="data row3 col3" >77.5%</td>
          <td id="T_7377f_row3_col4" class="data row3 col4" >1309</td>
          <td id="T_7377f_row3_col5" class="data row3 col5" >187</td>
          <td id="T_7377f_row3_col6" class="data row3 col6" >Sample: ? | C23 C25 C27 | G6 | B57 B59 B63 B66 | C22 C26</td>
        </tr>
        <tr>
          <th id="T_7377f_level0_row4" class="row_heading level0 row4" >4</th>
          <td id="T_7377f_row4_col0" class="data row4 col0" >embarked</td>
          <td id="T_7377f_row4_col1" class="data row4 col1" >object</td>
          <td id="T_7377f_row4_col2" class="data row4 col2" >0.0%</td>
          <td id="T_7377f_row4_col3" class="data row4 col3" >69.8%</td>
          <td id="T_7377f_row4_col4" class="data row4 col4" >1309</td>
          <td id="T_7377f_row4_col5" class="data row4 col5" >4</td>
          <td id="T_7377f_row4_col6" class="data row4 col6" >Sample: S | C | Q | ?</td>
        </tr>
        <tr>
          <th id="T_7377f_level0_row5" class="row_heading level0 row5" >5</th>
          <td id="T_7377f_row5_col0" class="data row5 col0" >fare</td>
          <td id="T_7377f_row5_col1" class="data row5 col1" >object</td>
          <td id="T_7377f_row5_col2" class="data row5 col2" >0.0%</td>
          <td id="T_7377f_row5_col3" class="data row5 col3" >4.6%</td>
          <td id="T_7377f_row5_col4" class="data row5 col4" >1309</td>
          <td id="T_7377f_row5_col5" class="data row5 col5" >282</td>
          <td id="T_7377f_row5_col6" class="data row5 col6" >Sample: 8.05 | 13 | 7.75 | 26 | 7.8958</td>
        </tr>
        <tr>
          <th id="T_7377f_level0_row6" class="row_heading level0 row6" >6</th>
          <td id="T_7377f_row6_col0" class="data row6 col0" >home.dest</td>
          <td id="T_7377f_row6_col1" class="data row6 col1" >object</td>
          <td id="T_7377f_row6_col2" class="data row6 col2" >0.0%</td>
          <td id="T_7377f_row6_col3" class="data row6 col3" >43.1%</td>
          <td id="T_7377f_row6_col4" class="data row6 col4" >1309</td>
          <td id="T_7377f_row6_col5" class="data row6 col5" >370</td>
          <td id="T_7377f_row6_col6" class="data row6 col6" >Sample: ? | New York, NY | London | Montreal, PQ | Paris, France</td>
        </tr>
        <tr>
          <th id="T_7377f_level0_row7" class="row_heading level0 row7" >7</th>
          <td id="T_7377f_row7_col0" class="data row7 col0" >name</td>
          <td id="T_7377f_row7_col1" class="data row7 col1" >object</td>
          <td id="T_7377f_row7_col2" class="data row7 col2" >0.0%</td>
          <td id="T_7377f_row7_col3" class="data row7 col3" >0.2%</td>
          <td id="T_7377f_row7_col4" class="data row7 col4" >1309</td>
          <td id="T_7377f_row7_col5" class="data row7 col5" >1307</td>
          <td id="T_7377f_row7_col6" class="data row7 col6" >Sample: Connolly, Miss. Kate | Kelly, Mr. James | Allen, Miss. Elisabeth Walton | Ilmakangas, Miss. ...</td>
        </tr>
        <tr>
          <th id="T_7377f_level0_row8" class="row_heading level0 row8" >8</th>
          <td id="T_7377f_row8_col0" class="data row8 col0" >parch</td>
          <td id="T_7377f_row8_col1" class="data row8 col1" >int64</td>
          <td id="T_7377f_row8_col2" class="data row8 col2" >0.0%</td>
          <td id="T_7377f_row8_col3" class="data row8 col3" >76.5%</td>
          <td id="T_7377f_row8_col4" class="data row8 col4" >1309</td>
          <td id="T_7377f_row8_col5" class="data row8 col5" >8</td>
          <td id="T_7377f_row8_col6" class="data row8 col6" >max=9 | min=0 | mean=0.39 | dominant=0</td>
        </tr>
        <tr>
          <th id="T_7377f_level0_row9" class="row_heading level0 row9" >9</th>
          <td id="T_7377f_row9_col0" class="data row9 col0" >pclass</td>
          <td id="T_7377f_row9_col1" class="data row9 col1" >int64</td>
          <td id="T_7377f_row9_col2" class="data row9 col2" >0.0%</td>
          <td id="T_7377f_row9_col3" class="data row9 col3" >54.2%</td>
          <td id="T_7377f_row9_col4" class="data row9 col4" >1309</td>
          <td id="T_7377f_row9_col5" class="data row9 col5" >3</td>
          <td id="T_7377f_row9_col6" class="data row9 col6" >max=3 | min=1 | mean=2.29 | dominant=3</td>
        </tr>
        <tr>
          <th id="T_7377f_level0_row10" class="row_heading level0 row10" >10</th>
          <td id="T_7377f_row10_col0" class="data row10 col0" >sex</td>
          <td id="T_7377f_row10_col1" class="data row10 col1" >object</td>
          <td id="T_7377f_row10_col2" class="data row10 col2" >0.0%</td>
          <td id="T_7377f_row10_col3" class="data row10 col3" >64.4%</td>
          <td id="T_7377f_row10_col4" class="data row10 col4" >1309</td>
          <td id="T_7377f_row10_col5" class="data row10 col5" >2</td>
          <td id="T_7377f_row10_col6" class="data row10 col6" >Sample: male | female</td>
        </tr>
        <tr>
          <th id="T_7377f_level0_row11" class="row_heading level0 row11" >11</th>
          <td id="T_7377f_row11_col0" class="data row11 col0" >sibsp</td>
          <td id="T_7377f_row11_col1" class="data row11 col1" >int64</td>
          <td id="T_7377f_row11_col2" class="data row11 col2" >0.0%</td>
          <td id="T_7377f_row11_col3" class="data row11 col3" >68.1%</td>
          <td id="T_7377f_row11_col4" class="data row11 col4" >1309</td>
          <td id="T_7377f_row11_col5" class="data row11 col5" >7</td>
          <td id="T_7377f_row11_col6" class="data row11 col6" >max=8 | min=0 | mean=0.5 | dominant=0</td>
        </tr>
        <tr>
          <th id="T_7377f_level0_row12" class="row_heading level0 row12" >12</th>
          <td id="T_7377f_row12_col0" class="data row12 col0" >survived</td>
          <td id="T_7377f_row12_col1" class="data row12 col1" >int64</td>
          <td id="T_7377f_row12_col2" class="data row12 col2" >0.0%</td>
          <td id="T_7377f_row12_col3" class="data row12 col3" >61.8%</td>
          <td id="T_7377f_row12_col4" class="data row12 col4" >1309</td>
          <td id="T_7377f_row12_col5" class="data row12 col5" >2</td>
          <td id="T_7377f_row12_col6" class="data row12 col6" >max=1 | min=0 | mean=0.38 | dominant=0</td>
        </tr>
        <tr>
          <th id="T_7377f_level0_row13" class="row_heading level0 row13" >13</th>
          <td id="T_7377f_row13_col0" class="data row13 col0" >ticket</td>
          <td id="T_7377f_row13_col1" class="data row13 col1" >object</td>
          <td id="T_7377f_row13_col2" class="data row13 col2" >0.0%</td>
          <td id="T_7377f_row13_col3" class="data row13 col3" >0.8%</td>
          <td id="T_7377f_row13_col4" class="data row13 col4" >1309</td>
          <td id="T_7377f_row13_col5" class="data row13 col5" >929</td>
          <td id="T_7377f_row13_col6" class="data row13 col6" >Sample: CA. 2343 | 1601 | CA 2144 | PC 17608 | 347077</td>
        </tr>
      </tbody>
    </table>




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




.. raw:: html

    <style type="text/css">
    #T_903ab th {
      font-size: 120%;
      text-align: center;
    }
    #T_903ab .row_heading {
      display: none;;
    }
    #T_903ab  .blank {
      display: none;;
    }
    #T_903ab_row0_col0, #T_903ab_row1_col0 {
      text-align: left;
      font-weight: bold;
      font-size: 120%;
    }
    #T_903ab_row0_col1, #T_903ab_row0_col2, #T_903ab_row0_col3, #T_903ab_row0_col4, #T_903ab_row0_col5, #T_903ab_row0_col6, #T_903ab_row0_col7, #T_903ab_row1_col1, #T_903ab_row1_col2, #T_903ab_row1_col3, #T_903ab_row1_col4, #T_903ab_row1_col5, #T_903ab_row1_col6, #T_903ab_row1_col7 {
      text-align: left;
    }
    </style>
    <table id="T_903ab">
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_903ab_level0_col0" class="col_heading level0 col0" >connector_name</th>
          <th id="T_903ab_level0_col1" class="col_heading level0 col1" >uri</th>
          <th id="T_903ab_level0_col2" class="col_heading level0 col2" >module_name</th>
          <th id="T_903ab_level0_col3" class="col_heading level0 col3" >handler</th>
          <th id="T_903ab_level0_col4" class="col_heading level0 col4" >version</th>
          <th id="T_903ab_level0_col5" class="col_heading level0 col5" >kwargs</th>
          <th id="T_903ab_level0_col6" class="col_heading level0 col6" >query</th>
          <th id="T_903ab_level0_col7" class="col_heading level0 col7" >aligned</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_903ab_level0_row0" class="row_heading level0 row0" >0</th>
          <td id="T_903ab_row0_col0" class="data row0 col0" >primary_source</td>
          <td id="T_903ab_row0_col1" class="data row0 col1" >https://www.openml.org/data/get_csv/16826755/phpMYEkMl.csv</td>
          <td id="T_903ab_row0_col2" class="data row0 col2" >ds_discovery.handlers.pandas_handlers</td>
          <td id="T_903ab_row0_col3" class="data row0 col3" >PandasPersistHandler</td>
          <td id="T_903ab_row0_col4" class="data row0 col4" >v0.00</td>
          <td id="T_903ab_row0_col5" class="data row0 col5" ></td>
          <td id="T_903ab_row0_col6" class="data row0 col6" ></td>
          <td id="T_903ab_row0_col7" class="data row0 col7" >False</td>
        </tr>
        <tr>
          <th id="T_903ab_level0_row1" class="row_heading level0 row1" >1</th>
          <td id="T_903ab_row1_col0" class="data row1 col0" >primary_persist</td>
          <td id="T_903ab_row1_col1" class="data row1 col1" >0_hello_meta/demo/data/hadron_transition_hello_comp_primary_persist.pickle</td>
          <td id="T_903ab_row1_col2" class="data row1 col2" >ds_discovery.handlers.pandas_handlers</td>
          <td id="T_903ab_row1_col3" class="data row1 col3" >PandasPersistHandler</td>
          <td id="T_903ab_row1_col4" class="data row1 col4" >v0.00</td>
          <td id="T_903ab_row1_col5" class="data row1 col5" ></td>
          <td id="T_903ab_row1_col6" class="data row1 col6" ></td>
          <td id="T_903ab_row1_col7" class="data row1 col7" >True</td>
        </tr>
      </tbody>
    </table>




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




.. raw:: html

    <style type="text/css">
    #T_b91b6 th {
      font-size: 120%;
      text-align: center;
    }
    #T_b91b6 .row_heading {
      display: none;;
    }
    #T_b91b6  .blank {
      display: none;;
    }
    #T_b91b6_row0_col0, #T_b91b6_row1_col0, #T_b91b6_row2_col0, #T_b91b6_row3_col0 {
      text-align: left;
      font-weight: bold;
      font-size: 120%;
    }
    #T_b91b6_row0_col1, #T_b91b6_row0_col2, #T_b91b6_row0_col3, #T_b91b6_row0_col4, #T_b91b6_row0_col5, #T_b91b6_row0_col6, #T_b91b6_row0_col7, #T_b91b6_row1_col1, #T_b91b6_row1_col2, #T_b91b6_row1_col3, #T_b91b6_row1_col4, #T_b91b6_row1_col5, #T_b91b6_row1_col6, #T_b91b6_row1_col7, #T_b91b6_row2_col1, #T_b91b6_row2_col2, #T_b91b6_row2_col3, #T_b91b6_row2_col4, #T_b91b6_row2_col5, #T_b91b6_row2_col6, #T_b91b6_row2_col7, #T_b91b6_row3_col1, #T_b91b6_row3_col2, #T_b91b6_row3_col3, #T_b91b6_row3_col4, #T_b91b6_row3_col5, #T_b91b6_row3_col6, #T_b91b6_row3_col7 {
      text-align: left;
    }
    </style>
    <table id="T_b91b6">
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_b91b6_level0_col0" class="col_heading level0 col0" >connector_name</th>
          <th id="T_b91b6_level0_col1" class="col_heading level0 col1" >uri</th>
          <th id="T_b91b6_level0_col2" class="col_heading level0 col2" >module_name</th>
          <th id="T_b91b6_level0_col3" class="col_heading level0 col3" >handler</th>
          <th id="T_b91b6_level0_col4" class="col_heading level0 col4" >version</th>
          <th id="T_b91b6_level0_col5" class="col_heading level0 col5" >kwargs</th>
          <th id="T_b91b6_level0_col6" class="col_heading level0 col6" >query</th>
          <th id="T_b91b6_level0_col7" class="col_heading level0 col7" >aligned</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_b91b6_level0_row0" class="row_heading level0 row0" >0</th>
          <td id="T_b91b6_row0_col0" class="data row0 col0" >primary_source</td>
          <td id="T_b91b6_row0_col1" class="data row0 col1" >https://www.openml.org/data/get_csv/16826755/phpMYEkMl.csv</td>
          <td id="T_b91b6_row0_col2" class="data row0 col2" >ds_discovery.handlers.pandas_handlers</td>
          <td id="T_b91b6_row0_col3" class="data row0 col3" >PandasPersistHandler</td>
          <td id="T_b91b6_row0_col4" class="data row0 col4" >v0.00</td>
          <td id="T_b91b6_row0_col5" class="data row0 col5" ></td>
          <td id="T_b91b6_row0_col6" class="data row0 col6" ></td>
          <td id="T_b91b6_row0_col7" class="data row0 col7" >False</td>
        </tr>
        <tr>
          <th id="T_b91b6_level0_row1" class="row_heading level0 row1" >1</th>
          <td id="T_b91b6_row1_col0" class="data row1 col0" >primary_persist</td>
          <td id="T_b91b6_row1_col1" class="data row1 col1" >0_hello_meta/demo/data/hadron_transition_hello_comp_primary_persist.pickle</td>
          <td id="T_b91b6_row1_col2" class="data row1 col2" >ds_discovery.handlers.pandas_handlers</td>
          <td id="T_b91b6_row1_col3" class="data row1 col3" >PandasPersistHandler</td>
          <td id="T_b91b6_row1_col4" class="data row1 col4" >v0.00</td>
          <td id="T_b91b6_row1_col5" class="data row1 col5" ></td>
          <td id="T_b91b6_row1_col6" class="data row1 col6" ></td>
          <td id="T_b91b6_row1_col7" class="data row1 col7" >True</td>
        </tr>
        <tr>
          <th id="T_b91b6_level0_row2" class="row_heading level0 row2" >2</th>
          <td id="T_b91b6_row2_col0" class="data row2 col0" >template_source</td>
          <td id="T_b91b6_row2_col1" class="data row2 col1" >0_hello_meta/demo/data</td>
          <td id="T_b91b6_row2_col2" class="data row2 col2" >ds_discovery.handlers.pandas_handlers</td>
          <td id="T_b91b6_row2_col3" class="data row2 col3" >PandasSourceHandler</td>
          <td id="T_b91b6_row2_col4" class="data row2 col4" >v0.00</td>
          <td id="T_b91b6_row2_col5" class="data row2 col5" ></td>
          <td id="T_b91b6_row2_col6" class="data row2 col6" ></td>
          <td id="T_b91b6_row2_col7" class="data row2 col7" >False</td>
        </tr>
        <tr>
          <th id="T_b91b6_level0_row3" class="row_heading level0 row3" >3</th>
          <td id="T_b91b6_row3_col0" class="data row3 col0" >template_persist</td>
          <td id="T_b91b6_row3_col1" class="data row3 col1" >0_hello_meta/demo/data</td>
          <td id="T_b91b6_row3_col2" class="data row3 col2" >ds_discovery.handlers.pandas_handlers</td>
          <td id="T_b91b6_row3_col3" class="data row3 col3" >PandasPersistHandler</td>
          <td id="T_b91b6_row3_col4" class="data row3 col4" >v0.00</td>
          <td id="T_b91b6_row3_col5" class="data row3 col5" ></td>
          <td id="T_b91b6_row3_col6" class="data row3 col6" ></td>
          <td id="T_b91b6_row3_col7" class="data row3 col7" >False</td>
        </tr>
      </tbody>
    </table>




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

