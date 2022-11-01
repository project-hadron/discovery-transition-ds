Darryl Oatridge, August 2022

.. code:: ipython3

    import os

.. code:: ipython3

    os.environ['HADRON_PM_PATH'] = '0_hello_meta/demo/contracts'
    os.environ['HADRON_DEFAULT_PATH'] = '0_hello_meta/demo/data'

Feature Engineering
-------------------

This new component works in exactly the same way as the selection
component, whereby we create the instance pertinent to our intentions,
give it a location to retrieve data from, the source, and where to
persist the results. Then we add the component intent, which in this
case is to engineer the features we have selected and make them
appropriate for a machine learning model or for further investigation.

For feature engineering the component we will use, that contains the
feature engineering intent, is called ``wrangle``.

.. code:: ipython3

    from ds_discovery import Wrangle, Transition

.. code:: ipython3

    # get the instance
    wr = Wrangle.from_env('hello_wr', has_contract=False)

With the source we want to be able to retrieve the outcome of the
previous select component as this contains the selected features of
interest. In order to retrieve this information we need to access the
select components Domain Contract, remember this holds all the knowledge
for any component. As this is a common thing to do there is a First
class method call ``get_persist_contract`` that can be called directly.

To retrieve the name of the source we are interested in we reload the
previous component ``Transition`` giving it the unique name we used when
creating the select component, in this case ``hello_wr``, this loads the
select components Domain Contract and then ``get_persist_contract``
which returns the string value of the outcome of that select component.

.. code:: ipython3

    source = Transition.from_env('hello_tr').get_persist_contract()
    wr.set_source_contract(source)
    wr.set_persist()

As a check we can run the canonical report and see that we have loaded
the output of the previous component (Transition component) into the
current source.

.. code:: ipython3

    df = wr.load_source_canonical()

.. code:: ipython3

    wr.canonical_report(df)




.. raw:: html

    <style type="text/css">
    #T_d8f4e th {
      font-size: 120%;
      text-align: center;
    }
    #T_d8f4e .row_heading {
      display: none;;
    }
    #T_d8f4e  .blank {
      display: none;;
    }
    #T_d8f4e_row0_col0, #T_d8f4e_row1_col0, #T_d8f4e_row2_col0, #T_d8f4e_row3_col0, #T_d8f4e_row4_col0, #T_d8f4e_row5_col0, #T_d8f4e_row6_col0, #T_d8f4e_row7_col0, #T_d8f4e_row8_col0, #T_d8f4e_row9_col0 {
      font-weight: bold;
      font-size: 120%;
    }
    #T_d8f4e_row0_col1, #T_d8f4e_row3_col1 {
      color: #2f0f8a;
    }
    #T_d8f4e_row0_col2, #T_d8f4e_row0_col3, #T_d8f4e_row1_col5, #T_d8f4e_row2_col2, #T_d8f4e_row3_col2, #T_d8f4e_row3_col3, #T_d8f4e_row3_col5, #T_d8f4e_row4_col2, #T_d8f4e_row5_col2, #T_d8f4e_row5_col3, #T_d8f4e_row6_col2, #T_d8f4e_row6_col3, #T_d8f4e_row7_col2, #T_d8f4e_row8_col2, #T_d8f4e_row8_col3, #T_d8f4e_row9_col2, #T_d8f4e_row9_col3, #T_d8f4e_row9_col5 {
      color: black;
    }
    #T_d8f4e_row0_col5 {
      background-color: #f0f9ed;
      color: black;
    }
    #T_d8f4e_row1_col1, #T_d8f4e_row9_col1 {
      color: #761d38;
    }
    #T_d8f4e_row1_col2, #T_d8f4e_row1_col3, #T_d8f4e_row2_col3, #T_d8f4e_row4_col3, #T_d8f4e_row7_col3 {
      background-color: #ffede5;
      color: black;
    }
    #T_d8f4e_row2_col1, #T_d8f4e_row4_col1, #T_d8f4e_row5_col1, #T_d8f4e_row6_col1, #T_d8f4e_row7_col1, #T_d8f4e_row8_col1 {
      color: #208a0f;
    }
    #T_d8f4e_row2_col5, #T_d8f4e_row5_col5 {
      background-color: #84cc83;
      color: black;
    }
    #T_d8f4e_row4_col5, #T_d8f4e_row7_col5 {
      background-color: #a4da9e;
      color: black;
    }
    #T_d8f4e_row6_col5, #T_d8f4e_row8_col5 {
      background-color: #a1cbe2;
      color: black;
    }
    </style>
    <table id="T_d8f4e">
      <caption>%_Dom: The % most dominant element </caption>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_d8f4e_level0_col0" class="col_heading level0 col0" >Attributes (10)</th>
          <th id="T_d8f4e_level0_col1" class="col_heading level0 col1" >dType</th>
          <th id="T_d8f4e_level0_col2" class="col_heading level0 col2" >%_Null</th>
          <th id="T_d8f4e_level0_col3" class="col_heading level0 col3" >%_Dom</th>
          <th id="T_d8f4e_level0_col4" class="col_heading level0 col4" >Count</th>
          <th id="T_d8f4e_level0_col5" class="col_heading level0 col5" >Unique</th>
          <th id="T_d8f4e_level0_col6" class="col_heading level0 col6" >Observations</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_d8f4e_level0_row0" class="row_heading level0 row0" >0</th>
          <td id="T_d8f4e_row0_col0" class="data row0 col0" >age</td>
          <td id="T_d8f4e_row0_col1" class="data row0 col1" >float64</td>
          <td id="T_d8f4e_row0_col2" class="data row0 col2" >20.1%</td>
          <td id="T_d8f4e_row0_col3" class="data row0 col3" >20.1%</td>
          <td id="T_d8f4e_row0_col4" class="data row0 col4" >1309</td>
          <td id="T_d8f4e_row0_col5" class="data row0 col5" >99</td>
          <td id="T_d8f4e_row0_col6" class="data row0 col6" >max=80.0 | min=0.1667 | mean=29.88 | dominant=24.0</td>
        </tr>
        <tr>
          <th id="T_d8f4e_level0_row1" class="row_heading level0 row1" >1</th>
          <td id="T_d8f4e_row1_col0" class="data row1 col0" >cabin</td>
          <td id="T_d8f4e_row1_col1" class="data row1 col1" >string</td>
          <td id="T_d8f4e_row1_col2" class="data row1 col2" >77.5%</td>
          <td id="T_d8f4e_row1_col3" class="data row1 col3" >77.5%</td>
          <td id="T_d8f4e_row1_col4" class="data row1 col4" >1309</td>
          <td id="T_d8f4e_row1_col5" class="data row1 col5" >187</td>
          <td id="T_d8f4e_row1_col6" class="data row1 col6" >Sample: C23 C25 C27 | G6 | B57 B59 B63 B66 | F4 | F33</td>
        </tr>
        <tr>
          <th id="T_d8f4e_level0_row2" class="row_heading level0 row2" >2</th>
          <td id="T_d8f4e_row2_col0" class="data row2 col0" >embarked</td>
          <td id="T_d8f4e_row2_col1" class="data row2 col1" >category</td>
          <td id="T_d8f4e_row2_col2" class="data row2 col2" >0.0%</td>
          <td id="T_d8f4e_row2_col3" class="data row2 col3" >69.8%</td>
          <td id="T_d8f4e_row2_col4" class="data row2 col4" >1309</td>
          <td id="T_d8f4e_row2_col5" class="data row2 col5" >4</td>
          <td id="T_d8f4e_row2_col6" class="data row2 col6" >Sample: S | C | Q | nan</td>
        </tr>
        <tr>
          <th id="T_d8f4e_level0_row3" class="row_heading level0 row3" >3</th>
          <td id="T_d8f4e_row3_col0" class="data row3 col0" >fare</td>
          <td id="T_d8f4e_row3_col1" class="data row3 col1" >float64</td>
          <td id="T_d8f4e_row3_col2" class="data row3 col2" >0.1%</td>
          <td id="T_d8f4e_row3_col3" class="data row3 col3" >4.6%</td>
          <td id="T_d8f4e_row3_col4" class="data row3 col4" >1309</td>
          <td id="T_d8f4e_row3_col5" class="data row3 col5" >282</td>
          <td id="T_d8f4e_row3_col6" class="data row3 col6" >max=512.3292 | min=0.0 | mean=33.3 | dominant=8.05</td>
        </tr>
        <tr>
          <th id="T_d8f4e_level0_row4" class="row_heading level0 row4" >4</th>
          <td id="T_d8f4e_row4_col0" class="data row4 col0" >parch</td>
          <td id="T_d8f4e_row4_col1" class="data row4 col1" >category</td>
          <td id="T_d8f4e_row4_col2" class="data row4 col2" >0.0%</td>
          <td id="T_d8f4e_row4_col3" class="data row4 col3" >76.5%</td>
          <td id="T_d8f4e_row4_col4" class="data row4 col4" >1309</td>
          <td id="T_d8f4e_row4_col5" class="data row4 col5" >8</td>
          <td id="T_d8f4e_row4_col6" class="data row4 col6" >Sample: 0 | 1 | 2 | 3 | 4</td>
        </tr>
        <tr>
          <th id="T_d8f4e_level0_row5" class="row_heading level0 row5" >5</th>
          <td id="T_d8f4e_row5_col0" class="data row5 col0" >pclass</td>
          <td id="T_d8f4e_row5_col1" class="data row5 col1" >category</td>
          <td id="T_d8f4e_row5_col2" class="data row5 col2" >0.0%</td>
          <td id="T_d8f4e_row5_col3" class="data row5 col3" >54.2%</td>
          <td id="T_d8f4e_row5_col4" class="data row5 col4" >1309</td>
          <td id="T_d8f4e_row5_col5" class="data row5 col5" >3</td>
          <td id="T_d8f4e_row5_col6" class="data row5 col6" >Sample: 3 | 1 | 2</td>
        </tr>
        <tr>
          <th id="T_d8f4e_level0_row6" class="row_heading level0 row6" >6</th>
          <td id="T_d8f4e_row6_col0" class="data row6 col0" >sex</td>
          <td id="T_d8f4e_row6_col1" class="data row6 col1" >category</td>
          <td id="T_d8f4e_row6_col2" class="data row6 col2" >0.0%</td>
          <td id="T_d8f4e_row6_col3" class="data row6 col3" >64.4%</td>
          <td id="T_d8f4e_row6_col4" class="data row6 col4" >1309</td>
          <td id="T_d8f4e_row6_col5" class="data row6 col5" >2</td>
          <td id="T_d8f4e_row6_col6" class="data row6 col6" >Sample: male | female</td>
        </tr>
        <tr>
          <th id="T_d8f4e_level0_row7" class="row_heading level0 row7" >7</th>
          <td id="T_d8f4e_row7_col0" class="data row7 col0" >sibsp</td>
          <td id="T_d8f4e_row7_col1" class="data row7 col1" >category</td>
          <td id="T_d8f4e_row7_col2" class="data row7 col2" >0.0%</td>
          <td id="T_d8f4e_row7_col3" class="data row7 col3" >68.1%</td>
          <td id="T_d8f4e_row7_col4" class="data row7 col4" >1309</td>
          <td id="T_d8f4e_row7_col5" class="data row7 col5" >7</td>
          <td id="T_d8f4e_row7_col6" class="data row7 col6" >Sample: 0 | 1 | 2 | 4 | 3</td>
        </tr>
        <tr>
          <th id="T_d8f4e_level0_row8" class="row_heading level0 row8" >8</th>
          <td id="T_d8f4e_row8_col0" class="data row8 col0" >survived</td>
          <td id="T_d8f4e_row8_col1" class="data row8 col1" >category</td>
          <td id="T_d8f4e_row8_col2" class="data row8 col2" >0.0%</td>
          <td id="T_d8f4e_row8_col3" class="data row8 col3" >61.8%</td>
          <td id="T_d8f4e_row8_col4" class="data row8 col4" >1309</td>
          <td id="T_d8f4e_row8_col5" class="data row8 col5" >2</td>
          <td id="T_d8f4e_row8_col6" class="data row8 col6" >Sample: 0 | 1</td>
        </tr>
        <tr>
          <th id="T_d8f4e_level0_row9" class="row_heading level0 row9" >9</th>
          <td id="T_d8f4e_row9_col0" class="data row9 col0" >ticket</td>
          <td id="T_d8f4e_row9_col1" class="data row9 col1" >string</td>
          <td id="T_d8f4e_row9_col2" class="data row9 col2" >0.0%</td>
          <td id="T_d8f4e_row9_col3" class="data row9 col3" >0.8%</td>
          <td id="T_d8f4e_row9_col4" class="data row9 col4" >1309</td>
          <td id="T_d8f4e_row9_col5" class="data row9 col5" >929</td>
          <td id="T_d8f4e_row9_col6" class="data row9 col6" >Sample: CA. 2343 | 1601 | CA 2144 | PC 17608 | 347077</td>
        </tr>
      </tbody>
    </table>




Engineering the Features
~~~~~~~~~~~~~~~~~~~~~~~~

As mentioned in the previous component demo, the components intent
methods are not first class methods but part of the intent_model_class.
Therefore to access the intent specify the controller instance name, in
this case tr, and then reference the intent_model_class to access the
components intent. To make this easier to remember with an abbreviated
form we have overloaded the intent_model name with the name tools. You
can see with all reference to the intent actions they start with
tr.tools.

Now we have the source we can deal with the feature Engineering. As this
is for the purpose of demonstration we are only sampling a small
selection of Intent methods. It is well worth looking through the other
Intent methods to get to know the full extent of the feature engineering
package.

To get started, the column name ``sibsip``, the number of siblings or
the spouse of a person onboard, and ``parch``, the number of parents or
children each passenger was touring with, added together provide a new
value that provides the size of each family.

.. code:: ipython3

    df['family'] = wr.tools.correlate_aggregate(df, headers=['parch', 'sibsp'], agg='sum', column_name='family')

The column name ``cabin`` provides us with a record of the cabin each
passenger was allocated. Taking the first letter from each cabin gives
us the deck the passenger was on. This provides us with a useful
catagorical.

.. code:: ipython3

    df['deck'] = wr.tools.correlate_custom(df, code_str="@['cabin'].str[0]", column_name='deck')

We also note that a passenger travelling alone seems to have an improved
survival rate. By selecting ``family``, who’s value is one and giving
all other values a zero we can create a new column ``is_alone`` that
indicates passengers travelling on their own.

.. code:: ipython3

    selection = [wr.tools.select2dict(column='family', condition='@==0')]
    df['is_alone'] = wr.tools.correlate_selection(df, selection=selection, action=1, default_action=0, column_name='is_alone')

Finally we ensure each of our new features are appropriately ``typed``
as a category. We also want to ensure the change to catagory runs after
the newly created columns so we add the parameter ``intent_order`` with
a value of one.

.. code:: ipython3

    df = wr.tools.model_to_category(df, headers=['family','deck','is_alone'], intent_order=1, column_name='to_category')

By running the Intent report we can observe the change of order of the
intent level.

.. code:: ipython3

    wr.report_intent()




.. raw:: html

    <style type="text/css">
    #T_36eae th {
      font-size: 120%;
      text-align: center;
    }
    #T_36eae .row_heading {
      display: none;;
    }
    #T_36eae  .blank {
      display: none;;
    }
    #T_36eae_row0_col0, #T_36eae_row1_col0, #T_36eae_row2_col0, #T_36eae_row3_col0 {
      text-align: left;
      font-weight: bold;
      font-size: 120%;
    }
    #T_36eae_row0_col1, #T_36eae_row0_col2, #T_36eae_row0_col3, #T_36eae_row0_col4, #T_36eae_row1_col1, #T_36eae_row1_col2, #T_36eae_row1_col3, #T_36eae_row1_col4, #T_36eae_row2_col1, #T_36eae_row2_col2, #T_36eae_row2_col3, #T_36eae_row2_col4, #T_36eae_row3_col1, #T_36eae_row3_col2, #T_36eae_row3_col3, #T_36eae_row3_col4 {
      text-align: left;
    }
    </style>
    <table id="T_36eae">
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_36eae_level0_col0" class="col_heading level0 col0" >level</th>
          <th id="T_36eae_level0_col1" class="col_heading level0 col1" >order</th>
          <th id="T_36eae_level0_col2" class="col_heading level0 col2" >intent</th>
          <th id="T_36eae_level0_col3" class="col_heading level0 col3" >parameters</th>
          <th id="T_36eae_level0_col4" class="col_heading level0 col4" >creator</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_36eae_level0_row0" class="row_heading level0 row0" >0</th>
          <td id="T_36eae_row0_col0" class="data row0 col0" >deck</td>
          <td id="T_36eae_row0_col1" class="data row0 col1" >0</td>
          <td id="T_36eae_row0_col2" class="data row0 col2" >correlate_custom</td>
          <td id="T_36eae_row0_col3" class="data row0 col3" >["code_str='@['cabin'].str[0]'", "column_name='deck'", 'kwargs={}']</td>
          <td id="T_36eae_row0_col4" class="data row0 col4" >doatridge</td>
        </tr>
        <tr>
          <th id="T_36eae_level0_row1" class="row_heading level0 row1" >1</th>
          <td id="T_36eae_row1_col0" class="data row1 col0" >family</td>
          <td id="T_36eae_row1_col1" class="data row1 col1" >0</td>
          <td id="T_36eae_row1_col2" class="data row1 col2" >correlate_aggregate</td>
          <td id="T_36eae_row1_col3" class="data row1 col3" >["headers=['parch', 'sibsp']", "agg='sum'", "column_name='family'"]</td>
          <td id="T_36eae_row1_col4" class="data row1 col4" >doatridge</td>
        </tr>
        <tr>
          <th id="T_36eae_level0_row2" class="row_heading level0 row2" >2</th>
          <td id="T_36eae_row2_col0" class="data row2 col0" >is_alone</td>
          <td id="T_36eae_row2_col1" class="data row2 col1" >0</td>
          <td id="T_36eae_row2_col2" class="data row2 col2" >correlate_selection</td>
          <td id="T_36eae_row2_col3" class="data row2 col3" >["selection=[{'column': 'family', 'condition': '@==0'}]", 'action=1', 'default_action=0', "column_name='is_alone'"]</td>
          <td id="T_36eae_row2_col4" class="data row2 col4" >doatridge</td>
        </tr>
        <tr>
          <th id="T_36eae_level0_row3" class="row_heading level0 row3" >3</th>
          <td id="T_36eae_row3_col0" class="data row3 col0" >to_category</td>
          <td id="T_36eae_row3_col1" class="data row3 col1" >1</td>
          <td id="T_36eae_row3_col2" class="data row3 col2" >model_to_category</td>
          <td id="T_36eae_row3_col3" class="data row3 col3" >["headers=['family', 'deck', 'is_alone']", "column_name='to_category'"]</td>
          <td id="T_36eae_row3_col4" class="data row3 col4" >doatridge</td>
        </tr>
      </tbody>
    </table>




Run Component Pipeline
----------------------

To run a component we use the common method ``run_component_pipeline``
which loads the source data, executes the component task , in this case
components intent, then persists the results. This is the only method
you can use to run the tasks of a component and produce its results and
should be a familiarized method.

At this point we can run the pipeline and see the results of the new
features.

.. code:: ipython3

    wr.run_component_pipeline()

.. code:: ipython3

    wr.canonical_report(df)




.. raw:: html

    <style type="text/css">
    #T_43afa th {
      font-size: 120%;
      text-align: center;
    }
    #T_43afa .row_heading {
      display: none;;
    }
    #T_43afa  .blank {
      display: none;;
    }
    #T_43afa_row0_col0, #T_43afa_row1_col0, #T_43afa_row2_col0, #T_43afa_row3_col0, #T_43afa_row4_col0, #T_43afa_row5_col0, #T_43afa_row6_col0, #T_43afa_row7_col0, #T_43afa_row8_col0, #T_43afa_row9_col0, #T_43afa_row10_col0, #T_43afa_row11_col0, #T_43afa_row12_col0 {
      font-weight: bold;
      font-size: 120%;
    }
    #T_43afa_row0_col1, #T_43afa_row5_col1 {
      color: #2f0f8a;
    }
    #T_43afa_row0_col2, #T_43afa_row0_col3, #T_43afa_row1_col5, #T_43afa_row2_col2, #T_43afa_row3_col2, #T_43afa_row4_col2, #T_43afa_row4_col3, #T_43afa_row5_col2, #T_43afa_row5_col3, #T_43afa_row5_col5, #T_43afa_row6_col2, #T_43afa_row6_col3, #T_43afa_row7_col2, #T_43afa_row8_col2, #T_43afa_row8_col3, #T_43afa_row9_col2, #T_43afa_row9_col3, #T_43afa_row10_col2, #T_43afa_row11_col2, #T_43afa_row11_col3, #T_43afa_row12_col2, #T_43afa_row12_col3, #T_43afa_row12_col5 {
      color: black;
    }
    #T_43afa_row0_col5 {
      background-color: #f0f9ed;
      color: black;
    }
    #T_43afa_row1_col1, #T_43afa_row12_col1 {
      color: #761d38;
    }
    #T_43afa_row1_col2, #T_43afa_row1_col3, #T_43afa_row2_col3, #T_43afa_row3_col3, #T_43afa_row7_col3, #T_43afa_row10_col3 {
      background-color: #ffede5;
      color: black;
    }
    #T_43afa_row2_col1, #T_43afa_row3_col1, #T_43afa_row4_col1, #T_43afa_row6_col1, #T_43afa_row7_col1, #T_43afa_row8_col1, #T_43afa_row9_col1, #T_43afa_row10_col1, #T_43afa_row11_col1 {
      color: #208a0f;
    }
    #T_43afa_row2_col5, #T_43afa_row4_col5, #T_43afa_row7_col5, #T_43afa_row10_col5 {
      background-color: #a4da9e;
      color: black;
    }
    #T_43afa_row3_col5, #T_43afa_row8_col5 {
      background-color: #84cc83;
      color: black;
    }
    #T_43afa_row6_col5, #T_43afa_row9_col5, #T_43afa_row11_col5 {
      background-color: #a1cbe2;
      color: black;
    }
    </style>
    <table id="T_43afa">
      <caption>%_Dom: The % most dominant element </caption>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_43afa_level0_col0" class="col_heading level0 col0" >Attributes (13)</th>
          <th id="T_43afa_level0_col1" class="col_heading level0 col1" >dType</th>
          <th id="T_43afa_level0_col2" class="col_heading level0 col2" >%_Null</th>
          <th id="T_43afa_level0_col3" class="col_heading level0 col3" >%_Dom</th>
          <th id="T_43afa_level0_col4" class="col_heading level0 col4" >Count</th>
          <th id="T_43afa_level0_col5" class="col_heading level0 col5" >Unique</th>
          <th id="T_43afa_level0_col6" class="col_heading level0 col6" >Observations</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_43afa_level0_row0" class="row_heading level0 row0" >0</th>
          <td id="T_43afa_row0_col0" class="data row0 col0" >age</td>
          <td id="T_43afa_row0_col1" class="data row0 col1" >float64</td>
          <td id="T_43afa_row0_col2" class="data row0 col2" >20.1%</td>
          <td id="T_43afa_row0_col3" class="data row0 col3" >20.1%</td>
          <td id="T_43afa_row0_col4" class="data row0 col4" >1309</td>
          <td id="T_43afa_row0_col5" class="data row0 col5" >99</td>
          <td id="T_43afa_row0_col6" class="data row0 col6" >max=80.0 | min=0.1667 | mean=29.88 | dominant=24.0</td>
        </tr>
        <tr>
          <th id="T_43afa_level0_row1" class="row_heading level0 row1" >1</th>
          <td id="T_43afa_row1_col0" class="data row1 col0" >cabin</td>
          <td id="T_43afa_row1_col1" class="data row1 col1" >string</td>
          <td id="T_43afa_row1_col2" class="data row1 col2" >77.5%</td>
          <td id="T_43afa_row1_col3" class="data row1 col3" >77.5%</td>
          <td id="T_43afa_row1_col4" class="data row1 col4" >1309</td>
          <td id="T_43afa_row1_col5" class="data row1 col5" >187</td>
          <td id="T_43afa_row1_col6" class="data row1 col6" >Sample: C23 C25 C27 | G6 | B57 B59 B63 B66 | F4 | F33</td>
        </tr>
        <tr>
          <th id="T_43afa_level0_row2" class="row_heading level0 row2" >2</th>
          <td id="T_43afa_row2_col0" class="data row2 col0" >deck</td>
          <td id="T_43afa_row2_col1" class="data row2 col1" >category</td>
          <td id="T_43afa_row2_col2" class="data row2 col2" >0.0%</td>
          <td id="T_43afa_row2_col3" class="data row2 col3" >77.5%</td>
          <td id="T_43afa_row2_col4" class="data row2 col4" >1309</td>
          <td id="T_43afa_row2_col5" class="data row2 col5" >9</td>
          <td id="T_43afa_row2_col6" class="data row2 col6" >Sample: <NA> | C | B | D | E</td>
        </tr>
        <tr>
          <th id="T_43afa_level0_row3" class="row_heading level0 row3" >3</th>
          <td id="T_43afa_row3_col0" class="data row3 col0" >embarked</td>
          <td id="T_43afa_row3_col1" class="data row3 col1" >category</td>
          <td id="T_43afa_row3_col2" class="data row3 col2" >0.0%</td>
          <td id="T_43afa_row3_col3" class="data row3 col3" >69.8%</td>
          <td id="T_43afa_row3_col4" class="data row3 col4" >1309</td>
          <td id="T_43afa_row3_col5" class="data row3 col5" >4</td>
          <td id="T_43afa_row3_col6" class="data row3 col6" >Sample: S | C | Q | nan</td>
        </tr>
        <tr>
          <th id="T_43afa_level0_row4" class="row_heading level0 row4" >4</th>
          <td id="T_43afa_row4_col0" class="data row4 col0" >family</td>
          <td id="T_43afa_row4_col1" class="data row4 col1" >category</td>
          <td id="T_43afa_row4_col2" class="data row4 col2" >0.0%</td>
          <td id="T_43afa_row4_col3" class="data row4 col3" >60.4%</td>
          <td id="T_43afa_row4_col4" class="data row4 col4" >1309</td>
          <td id="T_43afa_row4_col5" class="data row4 col5" >9</td>
          <td id="T_43afa_row4_col6" class="data row4 col6" >Sample: 0 | 1 | 2 | 3 | 5</td>
        </tr>
        <tr>
          <th id="T_43afa_level0_row5" class="row_heading level0 row5" >5</th>
          <td id="T_43afa_row5_col0" class="data row5 col0" >fare</td>
          <td id="T_43afa_row5_col1" class="data row5 col1" >float64</td>
          <td id="T_43afa_row5_col2" class="data row5 col2" >0.1%</td>
          <td id="T_43afa_row5_col3" class="data row5 col3" >4.6%</td>
          <td id="T_43afa_row5_col4" class="data row5 col4" >1309</td>
          <td id="T_43afa_row5_col5" class="data row5 col5" >282</td>
          <td id="T_43afa_row5_col6" class="data row5 col6" >max=512.3292 | min=0.0 | mean=33.3 | dominant=8.05</td>
        </tr>
        <tr>
          <th id="T_43afa_level0_row6" class="row_heading level0 row6" >6</th>
          <td id="T_43afa_row6_col0" class="data row6 col0" >is_alone</td>
          <td id="T_43afa_row6_col1" class="data row6 col1" >category</td>
          <td id="T_43afa_row6_col2" class="data row6 col2" >0.0%</td>
          <td id="T_43afa_row6_col3" class="data row6 col3" >60.4%</td>
          <td id="T_43afa_row6_col4" class="data row6 col4" >1309</td>
          <td id="T_43afa_row6_col5" class="data row6 col5" >2</td>
          <td id="T_43afa_row6_col6" class="data row6 col6" >Sample: 1 | 0</td>
        </tr>
        <tr>
          <th id="T_43afa_level0_row7" class="row_heading level0 row7" >7</th>
          <td id="T_43afa_row7_col0" class="data row7 col0" >parch</td>
          <td id="T_43afa_row7_col1" class="data row7 col1" >category</td>
          <td id="T_43afa_row7_col2" class="data row7 col2" >0.0%</td>
          <td id="T_43afa_row7_col3" class="data row7 col3" >76.5%</td>
          <td id="T_43afa_row7_col4" class="data row7 col4" >1309</td>
          <td id="T_43afa_row7_col5" class="data row7 col5" >8</td>
          <td id="T_43afa_row7_col6" class="data row7 col6" >Sample: 0 | 1 | 2 | 3 | 4</td>
        </tr>
        <tr>
          <th id="T_43afa_level0_row8" class="row_heading level0 row8" >8</th>
          <td id="T_43afa_row8_col0" class="data row8 col0" >pclass</td>
          <td id="T_43afa_row8_col1" class="data row8 col1" >category</td>
          <td id="T_43afa_row8_col2" class="data row8 col2" >0.0%</td>
          <td id="T_43afa_row8_col3" class="data row8 col3" >54.2%</td>
          <td id="T_43afa_row8_col4" class="data row8 col4" >1309</td>
          <td id="T_43afa_row8_col5" class="data row8 col5" >3</td>
          <td id="T_43afa_row8_col6" class="data row8 col6" >Sample: 3 | 1 | 2</td>
        </tr>
        <tr>
          <th id="T_43afa_level0_row9" class="row_heading level0 row9" >9</th>
          <td id="T_43afa_row9_col0" class="data row9 col0" >sex</td>
          <td id="T_43afa_row9_col1" class="data row9 col1" >category</td>
          <td id="T_43afa_row9_col2" class="data row9 col2" >0.0%</td>
          <td id="T_43afa_row9_col3" class="data row9 col3" >64.4%</td>
          <td id="T_43afa_row9_col4" class="data row9 col4" >1309</td>
          <td id="T_43afa_row9_col5" class="data row9 col5" >2</td>
          <td id="T_43afa_row9_col6" class="data row9 col6" >Sample: male | female</td>
        </tr>
        <tr>
          <th id="T_43afa_level0_row10" class="row_heading level0 row10" >10</th>
          <td id="T_43afa_row10_col0" class="data row10 col0" >sibsp</td>
          <td id="T_43afa_row10_col1" class="data row10 col1" >category</td>
          <td id="T_43afa_row10_col2" class="data row10 col2" >0.0%</td>
          <td id="T_43afa_row10_col3" class="data row10 col3" >68.1%</td>
          <td id="T_43afa_row10_col4" class="data row10 col4" >1309</td>
          <td id="T_43afa_row10_col5" class="data row10 col5" >7</td>
          <td id="T_43afa_row10_col6" class="data row10 col6" >Sample: 0 | 1 | 2 | 4 | 3</td>
        </tr>
        <tr>
          <th id="T_43afa_level0_row11" class="row_heading level0 row11" >11</th>
          <td id="T_43afa_row11_col0" class="data row11 col0" >survived</td>
          <td id="T_43afa_row11_col1" class="data row11 col1" >category</td>
          <td id="T_43afa_row11_col2" class="data row11 col2" >0.0%</td>
          <td id="T_43afa_row11_col3" class="data row11 col3" >61.8%</td>
          <td id="T_43afa_row11_col4" class="data row11 col4" >1309</td>
          <td id="T_43afa_row11_col5" class="data row11 col5" >2</td>
          <td id="T_43afa_row11_col6" class="data row11 col6" >Sample: 0 | 1</td>
        </tr>
        <tr>
          <th id="T_43afa_level0_row12" class="row_heading level0 row12" >12</th>
          <td id="T_43afa_row12_col0" class="data row12 col0" >ticket</td>
          <td id="T_43afa_row12_col1" class="data row12 col1" >string</td>
          <td id="T_43afa_row12_col2" class="data row12 col2" >0.0%</td>
          <td id="T_43afa_row12_col3" class="data row12 col3" >0.8%</td>
          <td id="T_43afa_row12_col4" class="data row12 col4" >1309</td>
          <td id="T_43afa_row12_col5" class="data row12 col5" >929</td>
          <td id="T_43afa_row12_col6" class="data row12 col6" >Sample: CA. 2343 | 1601 | CA 2144 | PC 17608 | 347077</td>
        </tr>
      </tbody>
    </table>




Imputation
----------

Imputation is the act of replacing missing data with statistical
estimates of the missing values. The goal of any imputation technique is
to produce a complete dataset that can be used to train machine learning
models. There are three types of missing data: - Missing Completely at
Random (MCAR); where the missing data has nothing to do with another
feature(s) - Missing at Random (MAR); where missing data can be
interpreted from another feature(s) - Missing not at Random (MNAR);
where missing data is not random and can be interpreted from another
feature(s)

With ``deck`` and ``fair`` we can assume MCAR but with ``age`` it
appears to have association with other features. But for the purposes of
the demo we are going to assume it to also be MCAR.

With ``deck`` the conversion to catagorical has already imputed the
nulls with the new catagorical value therefore we do not need to do
anything.

.. code:: ipython3

    df['deck'].value_counts()




.. parsed-literal::

    <NA>    1014
    C         94
    B         65
    D         46
    E         41
    A         22
    F         21
    G          5
    T          1
    Name: deck, dtype: int64



With ``fare`` we chose a random number whereby this number is more
likely to fall within a populated area and preserves the distribution of
the data. This works particulary well with the small amount of missing
data.

.. code:: ipython3

    df['fare'] = wr.tools.correlate_missing(df, header='fare', method='random', column_name='fare')

Age is slightly more tricky as its null values are quite large. In this
instance we will use probability frequency, which like random values
preserves the distribution of the data. Quite often, in these cases, we
can add an additional boulean column that tells us which values were
generated to replace nulls.

.. code:: ipython3

    df['age'] = wr.tools.correlate_missing_weighted(df, header='age', granularity=5.0, column_name='age')

Using the Intent report we can check on the additional intent added.

.. code:: ipython3

    wr.report_intent()




.. raw:: html

    <style type="text/css">
    #T_ca135 th {
      font-size: 120%;
      text-align: center;
    }
    #T_ca135 .row_heading {
      display: none;;
    }
    #T_ca135  .blank {
      display: none;;
    }
    #T_ca135_row0_col0, #T_ca135_row1_col0, #T_ca135_row2_col0, #T_ca135_row3_col0, #T_ca135_row4_col0, #T_ca135_row5_col0 {
      text-align: left;
      font-weight: bold;
      font-size: 120%;
    }
    #T_ca135_row0_col1, #T_ca135_row0_col2, #T_ca135_row0_col3, #T_ca135_row0_col4, #T_ca135_row1_col1, #T_ca135_row1_col2, #T_ca135_row1_col3, #T_ca135_row1_col4, #T_ca135_row2_col1, #T_ca135_row2_col2, #T_ca135_row2_col3, #T_ca135_row2_col4, #T_ca135_row3_col1, #T_ca135_row3_col2, #T_ca135_row3_col3, #T_ca135_row3_col4, #T_ca135_row4_col1, #T_ca135_row4_col2, #T_ca135_row4_col3, #T_ca135_row4_col4, #T_ca135_row5_col1, #T_ca135_row5_col2, #T_ca135_row5_col3, #T_ca135_row5_col4 {
      text-align: left;
    }
    </style>
    <table id="T_ca135">
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_ca135_level0_col0" class="col_heading level0 col0" >level</th>
          <th id="T_ca135_level0_col1" class="col_heading level0 col1" >order</th>
          <th id="T_ca135_level0_col2" class="col_heading level0 col2" >intent</th>
          <th id="T_ca135_level0_col3" class="col_heading level0 col3" >parameters</th>
          <th id="T_ca135_level0_col4" class="col_heading level0 col4" >creator</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_ca135_level0_row0" class="row_heading level0 row0" >0</th>
          <td id="T_ca135_row0_col0" class="data row0 col0" >age</td>
          <td id="T_ca135_row0_col1" class="data row0 col1" >0</td>
          <td id="T_ca135_row0_col2" class="data row0 col2" >correlate_missing_weighted</td>
          <td id="T_ca135_row0_col3" class="data row0 col3" >["header='age'", 'granularity=5.0', "column_name='age'"]</td>
          <td id="T_ca135_row0_col4" class="data row0 col4" >doatridge</td>
        </tr>
        <tr>
          <th id="T_ca135_level0_row1" class="row_heading level0 row1" >1</th>
          <td id="T_ca135_row1_col0" class="data row1 col0" >deck</td>
          <td id="T_ca135_row1_col1" class="data row1 col1" >0</td>
          <td id="T_ca135_row1_col2" class="data row1 col2" >correlate_custom</td>
          <td id="T_ca135_row1_col3" class="data row1 col3" >["code_str='@['cabin'].str[0]'", "column_name='deck'", 'kwargs={}']</td>
          <td id="T_ca135_row1_col4" class="data row1 col4" >doatridge</td>
        </tr>
        <tr>
          <th id="T_ca135_level0_row2" class="row_heading level0 row2" >2</th>
          <td id="T_ca135_row2_col0" class="data row2 col0" >family</td>
          <td id="T_ca135_row2_col1" class="data row2 col1" >0</td>
          <td id="T_ca135_row2_col2" class="data row2 col2" >correlate_aggregate</td>
          <td id="T_ca135_row2_col3" class="data row2 col3" >["headers=['parch', 'sibsp']", "agg='sum'", "column_name='family'"]</td>
          <td id="T_ca135_row2_col4" class="data row2 col4" >doatridge</td>
        </tr>
        <tr>
          <th id="T_ca135_level0_row3" class="row_heading level0 row3" >3</th>
          <td id="T_ca135_row3_col0" class="data row3 col0" >fare</td>
          <td id="T_ca135_row3_col1" class="data row3 col1" >0</td>
          <td id="T_ca135_row3_col2" class="data row3 col2" >correlate_missing</td>
          <td id="T_ca135_row3_col3" class="data row3 col3" >["header='fare'", "method='random'", "column_name='fare'"]</td>
          <td id="T_ca135_row3_col4" class="data row3 col4" >doatridge</td>
        </tr>
        <tr>
          <th id="T_ca135_level0_row4" class="row_heading level0 row4" >4</th>
          <td id="T_ca135_row4_col0" class="data row4 col0" >is_alone</td>
          <td id="T_ca135_row4_col1" class="data row4 col1" >0</td>
          <td id="T_ca135_row4_col2" class="data row4 col2" >correlate_selection</td>
          <td id="T_ca135_row4_col3" class="data row4 col3" >["selection=[{'column': 'family', 'condition': '@==0'}]", 'action=1', 'default_action=0', "column_name='is_alone'"]</td>
          <td id="T_ca135_row4_col4" class="data row4 col4" >doatridge</td>
        </tr>
        <tr>
          <th id="T_ca135_level0_row5" class="row_heading level0 row5" >5</th>
          <td id="T_ca135_row5_col0" class="data row5 col0" >to_category</td>
          <td id="T_ca135_row5_col1" class="data row5 col1" >1</td>
          <td id="T_ca135_row5_col2" class="data row5 col2" >model_to_category</td>
          <td id="T_ca135_row5_col3" class="data row5 col3" >["headers=['family', 'deck', 'is_alone']", "column_name='to_category'"]</td>
          <td id="T_ca135_row5_col4" class="data row5 col4" >doatridge</td>
        </tr>
      </tbody>
    </table>




Run Book
~~~~~~~~

We have touched on Run Book before where by the Run Book allows us to
define a run order that is preserved longer term. With the need for
``to_category`` to run as the final intent the Run Book fulfills this
perfectly.

Adding a Run Book is a simple task of listing the intent in the order in
which you wish it to run. As discussed before we are using the default
Run Book which will automatically be picked up by the run component as
its run order.

.. code:: ipython3

    wr.add_run_book(run_levels=['age','deck','family','fare','is_alone','to_category'])

.. code:: ipython3

    wr.run_component_pipeline()

Finially we can finish off by checking the Run Book with the Run Book
report and produce the Canonical Report to see the changes the feature
engineering has made.

.. code:: ipython3

    wr.report_run_book()




.. raw:: html

    <style type="text/css">
    #T_55c5e th {
      font-size: 120%;
      text-align: center;
    }
    #T_55c5e .row_heading {
      display: none;;
    }
    #T_55c5e  .blank {
      display: none;;
    }
    #T_55c5e_row0_col0 {
      text-align: left;
      font-weight: bold;
      font-size: 120%;
    }
    #T_55c5e_row0_col1 {
      text-align: left;
    }
    </style>
    <table id="T_55c5e">
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_55c5e_level0_col0" class="col_heading level0 col0" >name</th>
          <th id="T_55c5e_level0_col1" class="col_heading level0 col1" >run_book</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_55c5e_level0_row0" class="row_heading level0 row0" >0</th>
          <td id="T_55c5e_row0_col0" class="data row0 col0" >primary_run_book</td>
          <td id="T_55c5e_row0_col1" class="data row0 col1" >['age', 'deck', 'family', 'fare', 'is_alone', 'to_category']</td>
        </tr>
      </tbody>
    </table>




.. code:: ipython3

    wr.canonical_report(wr.load_persist_canonical())




.. raw:: html

    <style type="text/css">
    #T_2fba7 th {
      font-size: 120%;
      text-align: center;
    }
    #T_2fba7 .row_heading {
      display: none;;
    }
    #T_2fba7  .blank {
      display: none;;
    }
    #T_2fba7_row0_col0, #T_2fba7_row1_col0, #T_2fba7_row2_col0, #T_2fba7_row3_col0, #T_2fba7_row4_col0, #T_2fba7_row5_col0, #T_2fba7_row6_col0, #T_2fba7_row7_col0, #T_2fba7_row8_col0, #T_2fba7_row9_col0, #T_2fba7_row10_col0, #T_2fba7_row11_col0, #T_2fba7_row12_col0 {
      font-weight: bold;
      font-size: 120%;
    }
    #T_2fba7_row0_col1, #T_2fba7_row5_col1 {
      color: #2f0f8a;
    }
    #T_2fba7_row0_col2, #T_2fba7_row0_col3, #T_2fba7_row0_col5, #T_2fba7_row1_col5, #T_2fba7_row2_col2, #T_2fba7_row3_col2, #T_2fba7_row4_col2, #T_2fba7_row4_col3, #T_2fba7_row5_col2, #T_2fba7_row5_col3, #T_2fba7_row5_col5, #T_2fba7_row6_col2, #T_2fba7_row6_col3, #T_2fba7_row7_col2, #T_2fba7_row8_col2, #T_2fba7_row8_col3, #T_2fba7_row9_col2, #T_2fba7_row9_col3, #T_2fba7_row10_col2, #T_2fba7_row11_col2, #T_2fba7_row11_col3, #T_2fba7_row12_col2, #T_2fba7_row12_col3, #T_2fba7_row12_col5 {
      color: black;
    }
    #T_2fba7_row1_col1, #T_2fba7_row12_col1 {
      color: #761d38;
    }
    #T_2fba7_row1_col2, #T_2fba7_row1_col3, #T_2fba7_row2_col3, #T_2fba7_row3_col3, #T_2fba7_row7_col3, #T_2fba7_row10_col3 {
      background-color: #ffede5;
      color: black;
    }
    #T_2fba7_row2_col1, #T_2fba7_row3_col1, #T_2fba7_row4_col1, #T_2fba7_row6_col1, #T_2fba7_row7_col1, #T_2fba7_row8_col1, #T_2fba7_row9_col1, #T_2fba7_row10_col1, #T_2fba7_row11_col1 {
      color: #208a0f;
    }
    #T_2fba7_row2_col5, #T_2fba7_row4_col5, #T_2fba7_row7_col5, #T_2fba7_row10_col5 {
      background-color: #a4da9e;
      color: black;
    }
    #T_2fba7_row3_col5, #T_2fba7_row8_col5 {
      background-color: #84cc83;
      color: black;
    }
    #T_2fba7_row6_col5, #T_2fba7_row9_col5, #T_2fba7_row11_col5 {
      background-color: #a1cbe2;
      color: black;
    }
    </style>
    <table id="T_2fba7">
      <caption>%_Dom: The % most dominant element </caption>
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_2fba7_level0_col0" class="col_heading level0 col0" >Attributes (13)</th>
          <th id="T_2fba7_level0_col1" class="col_heading level0 col1" >dType</th>
          <th id="T_2fba7_level0_col2" class="col_heading level0 col2" >%_Null</th>
          <th id="T_2fba7_level0_col3" class="col_heading level0 col3" >%_Dom</th>
          <th id="T_2fba7_level0_col4" class="col_heading level0 col4" >Count</th>
          <th id="T_2fba7_level0_col5" class="col_heading level0 col5" >Unique</th>
          <th id="T_2fba7_level0_col6" class="col_heading level0 col6" >Observations</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_2fba7_level0_row0" class="row_heading level0 row0" >0</th>
          <td id="T_2fba7_row0_col0" class="data row0 col0" >age</td>
          <td id="T_2fba7_row0_col1" class="data row0 col1" >float64</td>
          <td id="T_2fba7_row0_col2" class="data row0 col2" >0.0%</td>
          <td id="T_2fba7_row0_col3" class="data row0 col3" >3.6%</td>
          <td id="T_2fba7_row0_col4" class="data row0 col4" >1309</td>
          <td id="T_2fba7_row0_col5" class="data row0 col5" >361</td>
          <td id="T_2fba7_row0_col6" class="data row0 col6" >max=80.0 | min=0.1667 | mean=29.91 | dominant=24.0</td>
        </tr>
        <tr>
          <th id="T_2fba7_level0_row1" class="row_heading level0 row1" >1</th>
          <td id="T_2fba7_row1_col0" class="data row1 col0" >cabin</td>
          <td id="T_2fba7_row1_col1" class="data row1 col1" >string</td>
          <td id="T_2fba7_row1_col2" class="data row1 col2" >77.5%</td>
          <td id="T_2fba7_row1_col3" class="data row1 col3" >77.5%</td>
          <td id="T_2fba7_row1_col4" class="data row1 col4" >1309</td>
          <td id="T_2fba7_row1_col5" class="data row1 col5" >187</td>
          <td id="T_2fba7_row1_col6" class="data row1 col6" >Sample: C23 C25 C27 | G6 | B57 B59 B63 B66 | F4 | F33</td>
        </tr>
        <tr>
          <th id="T_2fba7_level0_row2" class="row_heading level0 row2" >2</th>
          <td id="T_2fba7_row2_col0" class="data row2 col0" >deck</td>
          <td id="T_2fba7_row2_col1" class="data row2 col1" >category</td>
          <td id="T_2fba7_row2_col2" class="data row2 col2" >0.0%</td>
          <td id="T_2fba7_row2_col3" class="data row2 col3" >77.5%</td>
          <td id="T_2fba7_row2_col4" class="data row2 col4" >1309</td>
          <td id="T_2fba7_row2_col5" class="data row2 col5" >9</td>
          <td id="T_2fba7_row2_col6" class="data row2 col6" >Sample: <NA> | C | B | D | E</td>
        </tr>
        <tr>
          <th id="T_2fba7_level0_row3" class="row_heading level0 row3" >3</th>
          <td id="T_2fba7_row3_col0" class="data row3 col0" >embarked</td>
          <td id="T_2fba7_row3_col1" class="data row3 col1" >category</td>
          <td id="T_2fba7_row3_col2" class="data row3 col2" >0.0%</td>
          <td id="T_2fba7_row3_col3" class="data row3 col3" >69.8%</td>
          <td id="T_2fba7_row3_col4" class="data row3 col4" >1309</td>
          <td id="T_2fba7_row3_col5" class="data row3 col5" >4</td>
          <td id="T_2fba7_row3_col6" class="data row3 col6" >Sample: S | C | Q | nan</td>
        </tr>
        <tr>
          <th id="T_2fba7_level0_row4" class="row_heading level0 row4" >4</th>
          <td id="T_2fba7_row4_col0" class="data row4 col0" >family</td>
          <td id="T_2fba7_row4_col1" class="data row4 col1" >category</td>
          <td id="T_2fba7_row4_col2" class="data row4 col2" >0.0%</td>
          <td id="T_2fba7_row4_col3" class="data row4 col3" >60.4%</td>
          <td id="T_2fba7_row4_col4" class="data row4 col4" >1309</td>
          <td id="T_2fba7_row4_col5" class="data row4 col5" >9</td>
          <td id="T_2fba7_row4_col6" class="data row4 col6" >Sample: 0 | 1 | 2 | 3 | 5</td>
        </tr>
        <tr>
          <th id="T_2fba7_level0_row5" class="row_heading level0 row5" >5</th>
          <td id="T_2fba7_row5_col0" class="data row5 col0" >fare</td>
          <td id="T_2fba7_row5_col1" class="data row5 col1" >float64</td>
          <td id="T_2fba7_row5_col2" class="data row5 col2" >0.0%</td>
          <td id="T_2fba7_row5_col3" class="data row5 col3" >4.6%</td>
          <td id="T_2fba7_row5_col4" class="data row5 col4" >1309</td>
          <td id="T_2fba7_row5_col5" class="data row5 col5" >281</td>
          <td id="T_2fba7_row5_col6" class="data row5 col6" >max=512.3292 | min=0.0 | mean=33.28 | dominant=8.05</td>
        </tr>
        <tr>
          <th id="T_2fba7_level0_row6" class="row_heading level0 row6" >6</th>
          <td id="T_2fba7_row6_col0" class="data row6 col0" >is_alone</td>
          <td id="T_2fba7_row6_col1" class="data row6 col1" >category</td>
          <td id="T_2fba7_row6_col2" class="data row6 col2" >0.0%</td>
          <td id="T_2fba7_row6_col3" class="data row6 col3" >60.4%</td>
          <td id="T_2fba7_row6_col4" class="data row6 col4" >1309</td>
          <td id="T_2fba7_row6_col5" class="data row6 col5" >2</td>
          <td id="T_2fba7_row6_col6" class="data row6 col6" >Sample: 1 | 0</td>
        </tr>
        <tr>
          <th id="T_2fba7_level0_row7" class="row_heading level0 row7" >7</th>
          <td id="T_2fba7_row7_col0" class="data row7 col0" >parch</td>
          <td id="T_2fba7_row7_col1" class="data row7 col1" >category</td>
          <td id="T_2fba7_row7_col2" class="data row7 col2" >0.0%</td>
          <td id="T_2fba7_row7_col3" class="data row7 col3" >76.5%</td>
          <td id="T_2fba7_row7_col4" class="data row7 col4" >1309</td>
          <td id="T_2fba7_row7_col5" class="data row7 col5" >8</td>
          <td id="T_2fba7_row7_col6" class="data row7 col6" >Sample: 0 | 1 | 2 | 3 | 4</td>
        </tr>
        <tr>
          <th id="T_2fba7_level0_row8" class="row_heading level0 row8" >8</th>
          <td id="T_2fba7_row8_col0" class="data row8 col0" >pclass</td>
          <td id="T_2fba7_row8_col1" class="data row8 col1" >category</td>
          <td id="T_2fba7_row8_col2" class="data row8 col2" >0.0%</td>
          <td id="T_2fba7_row8_col3" class="data row8 col3" >54.2%</td>
          <td id="T_2fba7_row8_col4" class="data row8 col4" >1309</td>
          <td id="T_2fba7_row8_col5" class="data row8 col5" >3</td>
          <td id="T_2fba7_row8_col6" class="data row8 col6" >Sample: 3 | 1 | 2</td>
        </tr>
        <tr>
          <th id="T_2fba7_level0_row9" class="row_heading level0 row9" >9</th>
          <td id="T_2fba7_row9_col0" class="data row9 col0" >sex</td>
          <td id="T_2fba7_row9_col1" class="data row9 col1" >category</td>
          <td id="T_2fba7_row9_col2" class="data row9 col2" >0.0%</td>
          <td id="T_2fba7_row9_col3" class="data row9 col3" >64.4%</td>
          <td id="T_2fba7_row9_col4" class="data row9 col4" >1309</td>
          <td id="T_2fba7_row9_col5" class="data row9 col5" >2</td>
          <td id="T_2fba7_row9_col6" class="data row9 col6" >Sample: male | female</td>
        </tr>
        <tr>
          <th id="T_2fba7_level0_row10" class="row_heading level0 row10" >10</th>
          <td id="T_2fba7_row10_col0" class="data row10 col0" >sibsp</td>
          <td id="T_2fba7_row10_col1" class="data row10 col1" >category</td>
          <td id="T_2fba7_row10_col2" class="data row10 col2" >0.0%</td>
          <td id="T_2fba7_row10_col3" class="data row10 col3" >68.1%</td>
          <td id="T_2fba7_row10_col4" class="data row10 col4" >1309</td>
          <td id="T_2fba7_row10_col5" class="data row10 col5" >7</td>
          <td id="T_2fba7_row10_col6" class="data row10 col6" >Sample: 0 | 1 | 2 | 4 | 3</td>
        </tr>
        <tr>
          <th id="T_2fba7_level0_row11" class="row_heading level0 row11" >11</th>
          <td id="T_2fba7_row11_col0" class="data row11 col0" >survived</td>
          <td id="T_2fba7_row11_col1" class="data row11 col1" >category</td>
          <td id="T_2fba7_row11_col2" class="data row11 col2" >0.0%</td>
          <td id="T_2fba7_row11_col3" class="data row11 col3" >61.8%</td>
          <td id="T_2fba7_row11_col4" class="data row11 col4" >1309</td>
          <td id="T_2fba7_row11_col5" class="data row11 col5" >2</td>
          <td id="T_2fba7_row11_col6" class="data row11 col6" >Sample: 0 | 1</td>
        </tr>
        <tr>
          <th id="T_2fba7_level0_row12" class="row_heading level0 row12" >12</th>
          <td id="T_2fba7_row12_col0" class="data row12 col0" >ticket</td>
          <td id="T_2fba7_row12_col1" class="data row12 col1" >string</td>
          <td id="T_2fba7_row12_col2" class="data row12 col2" >0.0%</td>
          <td id="T_2fba7_row12_col3" class="data row12 col3" >0.8%</td>
          <td id="T_2fba7_row12_col4" class="data row12 col4" >1309</td>
          <td id="T_2fba7_row12_col5" class="data row12 col5" >929</td>
          <td id="T_2fba7_row12_col6" class="data row12 col6" >Sample: CA. 2343 | 1601 | CA 2144 | PC 17608 | 347077</td>
        </tr>
      </tbody>
    </table>




