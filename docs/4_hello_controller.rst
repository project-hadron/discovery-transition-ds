Darryl Oatridge, August 2022

.. code:: ipython3

    import os

.. code:: ipython3

    os.environ['HADRON_PM_PATH'] = '0_hello_meta/demo/contracts'
    os.environ['HADRON_DEFAULT_PATH'] = '0_hello_meta/demo/data'

Controller
----------

The Controller is a unique component that independantly orchestrates the
components registered to it. It executes the components Domain Contract
and not its code. Domain Contracts belonging to a Controller should be
in the same path location as the Controllers Domain Contract. The
Controller executes the registered Controllers Domain Contracts in
accordance to the instructions given to it when the ``run_components``
is executed. The Controller orchestrates how those components should run
with the components being independant in their actions and therefore a
separation of concerns. With Controller you do not need to give it a
name as this is assumed in each folder containing Domain Contracts for
this set of components, known as a Domain Contract Cluster. This allows
us the entry point to interogate the Controller and its components.

.. code:: ipython3

    from ds_discovery import Controller

.. code:: ipython3

    controller = Controller.from_env(has_contract=False)

Add Components
~~~~~~~~~~~~~~

Now we have the empty Controller we need to register or add which
components make up this Controller, it should be noted that the Domain
Contracts for each component must be in the same folder of the
Controller Domain Contract.

To add a component we use the intent method specific for that component
type in this case ``model_transition`` for ``hello_tr`` and
``model_wrangle`` for ``hello_wr``.

.. code:: ipython3

    controller.intent_model.transition(canonical=0, task_name='hello_tr', intent_level='hw_transition')

.. code:: ipython3

    controller.intent_model.wrangle(canonical=0, task_name='hello_wr', intent_level='hw_wrangle')

Report
~~~~~~

Using the Task report we can check the components have been added.

.. code:: ipython3

    controller.report_tasks()




.. raw:: html

    <style type="text/css">
    #T_e39ca th {
      font-size: 120%;
      text-align: center;
    }
    #T_e39ca .row_heading {
      display: none;;
    }
    #T_e39ca  .blank {
      display: none;;
    }
    #T_e39ca_row0_col0, #T_e39ca_row1_col0 {
      text-align: left;
      font-weight: bold;
      font-size: 120%;
    }
    #T_e39ca_row0_col1, #T_e39ca_row0_col2, #T_e39ca_row0_col3, #T_e39ca_row0_col4, #T_e39ca_row0_col5, #T_e39ca_row1_col1, #T_e39ca_row1_col2, #T_e39ca_row1_col3, #T_e39ca_row1_col4, #T_e39ca_row1_col5 {
      text-align: left;
    }
    </style>
    <table id="T_e39ca">
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th id="T_e39ca_level0_col0" class="col_heading level0 col0" >level</th>
          <th id="T_e39ca_level0_col1" class="col_heading level0 col1" >order</th>
          <th id="T_e39ca_level0_col2" class="col_heading level0 col2" >component</th>
          <th id="T_e39ca_level0_col3" class="col_heading level0 col3" >task</th>
          <th id="T_e39ca_level0_col4" class="col_heading level0 col4" >parameters</th>
          <th id="T_e39ca_level0_col5" class="col_heading level0 col5" >creator</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_e39ca_level0_row0" class="row_heading level0 row0" >0</th>
          <td id="T_e39ca_row0_col0" class="data row0 col0" >hw_transition</td>
          <td id="T_e39ca_row0_col1" class="data row0 col1" >0</td>
          <td id="T_e39ca_row0_col2" class="data row0 col2" >Transition</td>
          <td id="T_e39ca_row0_col3" class="data row0 col3" >'hello_tr'</td>
          <td id="T_e39ca_row0_col4" class="data row0 col4" >[]</td>
          <td id="T_e39ca_row0_col5" class="data row0 col5" >doatridge</td>
        </tr>
        <tr>
          <th id="T_e39ca_level0_row1" class="row_heading level0 row1" >1</th>
          <td id="T_e39ca_row1_col0" class="data row1 col0" >hw_wrangle</td>
          <td id="T_e39ca_row1_col1" class="data row1 col1" >0</td>
          <td id="T_e39ca_row1_col2" class="data row1 col2" >Wrangle</td>
          <td id="T_e39ca_row1_col3" class="data row1 col3" >'hello_wr'</td>
          <td id="T_e39ca_row1_col4" class="data row1 col4" >[]</td>
          <td id="T_e39ca_row1_col5" class="data row1 col5" >doatridge</td>
        </tr>
      </tbody>
    </table>




As with all components the Controller executes the components in the
order given. By using the Controller’s special Run Book we are given
considerabily more flexability in the order and behaviour of each
component and how it interacts with others.

As good practice a Run Book should always be created for each Controller
as this provides better transparency into how the components run.

.. code:: ipython3

    run_book = [
        controller.runbook2dict(task='hw_transition'),
        controller.runbook2dict(task='hw_wrangle'),
    ]
    controller.add_run_book(run_levels=run_book)

Run Controller Pipeline
~~~~~~~~~~~~~~~~~~~~~~~

To run the controller we execute ``run_controller`` this is a special
method and replaces ``run_component_pipeline``, common to other
components, adding extra features to enable the control of the
registared components. This is the only method you can use to run the
Controller and execute its registared components. It is worth noting it
is the components that produce the outcome of their collective
objectives or tasks and not the Controller. The Controller orchestrates
how those components should run with the components being independant in
their actions and therefore a separation of concerns.

.. code:: ipython3

    controller.run_controller()

The Controller is a powerful tool and should be investigated further to
understand all its options. The Run Book can be used to provide a set of
instructions on how each component recieves its source and persists, be
it to another component or as an external data set. The
``run_controller`` has useful tools to monitor changes in incoming data
and provide a run report of how all the components ran.

--------------

In the section below we will demonstrate a couple of these features.

One of the most useful parameters that comes with the ``run_controller``
is the ``run_cycle_report`` that saves off a run report, that provides
the run time of the controller and the components there in.

.. code:: ipython3

    controller.run_controller(run_cycle_report='cycle_report.csv')
    controller.load_canonical(connector_name='run_cycle_report')




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>time</th>
          <th>text</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>2022-10-31 11:29:58.246804</td>
          <td>start run-cycle 0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2022-10-31 11:29:58.248586</td>
          <td>start task cycle 0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2022-10-31 11:29:58.250208</td>
          <td>running hw_transition</td>
        </tr>
        <tr>
          <th>3</th>
          <td>2022-10-31 11:29:59.745230</td>
          <td>canonical shape is (1309, 10)</td>
        </tr>
        <tr>
          <th>4</th>
          <td>2022-10-31 11:29:59.747102</td>
          <td>running hw_wrangle</td>
        </tr>
        <tr>
          <th>5</th>
          <td>2022-10-31 11:29:59.782014</td>
          <td>canonical shape is (1309, 13)</td>
        </tr>
        <tr>
          <th>6</th>
          <td>2022-10-31 11:29:59.783219</td>
          <td>tasks complete</td>
        </tr>
        <tr>
          <th>7</th>
          <td>2022-10-31 11:29:59.784356</td>
          <td>end of report</td>
        </tr>
      </tbody>
    </table>
    </div>



Now we have the ``run_cycle_report`` we can observe the other
parameters. In this case we are adding the ``run_time`` parameter that
runs the controllers components for a time period of three seconds

.. code:: ipython3

    controller.run_controller(run_time=3, run_cycle_report='cycle_report.csv')
    controller.load_canonical(connector_name='run_cycle_report')




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>time</th>
          <th>text</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>2022-10-31 11:29:59.804318</td>
          <td>start run-cycle 0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2022-10-31 11:29:59.805766</td>
          <td>start task cycle 0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2022-10-31 11:29:59.807148</td>
          <td>running hw_transition</td>
        </tr>
        <tr>
          <th>3</th>
          <td>2022-10-31 11:30:01.393013</td>
          <td>canonical shape is (1309, 10)</td>
        </tr>
        <tr>
          <th>4</th>
          <td>2022-10-31 11:30:01.396067</td>
          <td>running hw_wrangle</td>
        </tr>
        <tr>
          <th>5</th>
          <td>2022-10-31 11:30:01.444290</td>
          <td>canonical shape is (1309, 13)</td>
        </tr>
        <tr>
          <th>6</th>
          <td>2022-10-31 11:30:01.445913</td>
          <td>tasks complete</td>
        </tr>
        <tr>
          <th>7</th>
          <td>2022-10-31 11:30:01.447497</td>
          <td>sleep for 1 seconds</td>
        </tr>
        <tr>
          <th>8</th>
          <td>2022-10-31 11:30:02.450195</td>
          <td>start run-cycle 1</td>
        </tr>
        <tr>
          <th>9</th>
          <td>2022-10-31 11:30:02.453278</td>
          <td>start task cycle 0</td>
        </tr>
        <tr>
          <th>10</th>
          <td>2022-10-31 11:30:02.455826</td>
          <td>running hw_transition</td>
        </tr>
        <tr>
          <th>11</th>
          <td>2022-10-31 11:30:04.005457</td>
          <td>canonical shape is (1309, 10)</td>
        </tr>
        <tr>
          <th>12</th>
          <td>2022-10-31 11:30:04.008056</td>
          <td>running hw_wrangle</td>
        </tr>
        <tr>
          <th>13</th>
          <td>2022-10-31 11:30:04.052639</td>
          <td>canonical shape is (1309, 13)</td>
        </tr>
        <tr>
          <th>14</th>
          <td>2022-10-31 11:30:04.054194</td>
          <td>tasks complete</td>
        </tr>
        <tr>
          <th>15</th>
          <td>2022-10-31 11:30:04.055488</td>
          <td>end of report</td>
        </tr>
      </tbody>
    </table>
    </div>



In this example we had the parameters ``repeat`` and ``sleep`` where the
first defines the number of times to repeat the component cycleand the
second, and the number of seconds to pause between each cycle.

.. code:: ipython3

    controller.run_controller(repeat=2, sleep=3, run_cycle_report='cycle_report.csv')
    controller.load_canonical(connector_name='run_cycle_report')




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>time</th>
          <th>text</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>2022-10-31 11:30:04.074245</td>
          <td>start run-cycle 0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2022-10-31 11:30:04.076095</td>
          <td>start task cycle 0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2022-10-31 11:30:04.079772</td>
          <td>running hw_transition</td>
        </tr>
        <tr>
          <th>3</th>
          <td>2022-10-31 11:30:06.011824</td>
          <td>canonical shape is (1309, 10)</td>
        </tr>
        <tr>
          <th>4</th>
          <td>2022-10-31 11:30:06.014329</td>
          <td>running hw_wrangle</td>
        </tr>
        <tr>
          <th>5</th>
          <td>2022-10-31 11:30:06.058910</td>
          <td>canonical shape is (1309, 13)</td>
        </tr>
        <tr>
          <th>6</th>
          <td>2022-10-31 11:30:06.060444</td>
          <td>tasks complete</td>
        </tr>
        <tr>
          <th>7</th>
          <td>2022-10-31 11:30:06.061820</td>
          <td>sleep for 3 seconds</td>
        </tr>
        <tr>
          <th>8</th>
          <td>2022-10-31 11:30:09.064581</td>
          <td>start task cycle 1</td>
        </tr>
        <tr>
          <th>9</th>
          <td>2022-10-31 11:30:09.066524</td>
          <td>running hw_transition</td>
        </tr>
        <tr>
          <th>10</th>
          <td>2022-10-31 11:30:10.626389</td>
          <td>canonical shape is (1309, 10)</td>
        </tr>
        <tr>
          <th>11</th>
          <td>2022-10-31 11:30:10.630010</td>
          <td>running hw_wrangle</td>
        </tr>
        <tr>
          <th>12</th>
          <td>2022-10-31 11:30:10.681425</td>
          <td>canonical shape is (1309, 13)</td>
        </tr>
        <tr>
          <th>13</th>
          <td>2022-10-31 11:30:10.683384</td>
          <td>tasks complete</td>
        </tr>
        <tr>
          <th>14</th>
          <td>2022-10-31 11:30:10.684870</td>
          <td>end of report</td>
        </tr>
      </tbody>
    </table>
    </div>



Finally we use the ``source_check_uri`` parameter as a pointer to and
input source to watch for changes.

.. code:: ipython3

    controller.run_controller(repeat=3, source_check_uri='https://www.openml.org/data/get_csv/16826755/phpMYEkMl.csv', run_cycle_report='cycle_report.csv')
    controller.load_canonical(connector_name='run_cycle_report')




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>time</th>
          <th>text</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>2022-10-31 11:30:10.708617</td>
          <td>start run-cycle 0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2022-10-31 11:30:10.709621</td>
          <td>start task cycle 0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2022-10-31 11:30:15.584539</td>
          <td>running hw_transition</td>
        </tr>
        <tr>
          <th>3</th>
          <td>2022-10-31 11:30:17.837519</td>
          <td>canonical shape is (1309, 10)</td>
        </tr>
        <tr>
          <th>4</th>
          <td>2022-10-31 11:30:17.839412</td>
          <td>running hw_wrangle</td>
        </tr>
        <tr>
          <th>5</th>
          <td>2022-10-31 11:30:17.880574</td>
          <td>canonical shape is (1309, 13)</td>
        </tr>
        <tr>
          <th>6</th>
          <td>2022-10-31 11:30:17.882085</td>
          <td>tasks complete</td>
        </tr>
        <tr>
          <th>7</th>
          <td>2022-10-31 11:30:17.883416</td>
          <td>start task cycle 1</td>
        </tr>
        <tr>
          <th>8</th>
          <td>2022-10-31 11:30:22.413937</td>
          <td>Source has not changed</td>
        </tr>
        <tr>
          <th>9</th>
          <td>2022-10-31 11:30:22.415728</td>
          <td>start task cycle 2</td>
        </tr>
        <tr>
          <th>10</th>
          <td>2022-10-31 11:30:27.528509</td>
          <td>Source has not changed</td>
        </tr>
        <tr>
          <th>11</th>
          <td>2022-10-31 11:30:27.531813</td>
          <td>end of report</td>
        </tr>
      </tbody>
    </table>
    </div>



