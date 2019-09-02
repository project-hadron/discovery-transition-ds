
.. code:: ipython3

    # saves you having to use print as all exposed variables are printed in the cell
    from IPython.core.interactiveshell import InteractiveShell
    InteractiveShell.ast_node_interactivity = "all"
    
    # core libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    
    # matpolitlib config
    %matplotlib inline
    plt.rcParams['figure.figsize'] = 12,4
    
    # suppress warning message
    import warnings
    warnings.filterwarnings('ignore')
    
    import ds_discovery
    print('DTU: {}'.format(ds_discovery.__version__))


.. parsed-literal::

    DTU: 1.04.044


Association
===========

Associations allows one to build results through the assocation of
relationships across a dataset.

let us consider a dataset of 'Gender' and 'Age'. What we are looking to
achive is - any male over 24 is a Dad, though there is a 20% chance he
is called Papa - any female over 24 is called Mum, though here there is
a 40% chance they are call Mother. - any male or female under the age of
25 is a student - Finally for anything else put 'Unknown'

.. code:: ipython3

    from ds_discovery.simulators.data_builder import DataBuilder
    from ds_discovery.transition.cleaners import ColumnCleaners as cleaner
    from ds_discovery.transition.discovery import DataDiscovery as discovery

.. code:: ipython3

    builder = DataBuilder('association')
    tools = builder.tools

Build the dataset
^^^^^^^^^^^^^^^^^

.. code:: ipython3

    rows = 100
    
    dataset = pd.DataFrame()
    dataset['age'] = tools.get_number(from_value=18, to_value=40, weight_pattern=[3,1], size=rows)
    dataset['gender'] = tools.get_category(selection=['M', 'F', 'U'], weight_pattern=[2,2,1], size=rows)
    
    dataset.head(5)




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
          <th>age</th>
          <th>gender</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>21</td>
          <td>F</td>
        </tr>
        <tr>
          <th>1</th>
          <td>38</td>
          <td>F</td>
        </tr>
        <tr>
          <th>2</th>
          <td>27</td>
          <td>U</td>
        </tr>
        <tr>
          <th>3</th>
          <td>33</td>
          <td>F</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20</td>
          <td>M</td>
        </tr>
      </tbody>
    </table>
    </div>



Build the correlation and actions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

to get the right format use the context help to cut and paste the format
and then adapt

.. code:: ipython3

    help(tools.associate_dataset)


.. parsed-literal::

    Help on function associate_dataset in module ds_discovery.simulators.data_builder:
    
    associate_dataset(dataset: Any, associations: list, actions: dict, default_value: Any = None, default_header: str = None, day_first: bool = True, quantity: float = None, seed: int = None)
        Associates a a set of criteria of an input values to a set of actions
            The association dictionary takes the form of a set of dictionaries in a list with each item in the list
            representing an index key for the action dictionary. Each dictionary are to associated relationship.
            In this example for the first index the associated values should be header1 is within a date range
            and header2 has a value of 'M'
                association = [{'header1': {'expect': 'date',
                                            'value': ['12/01/1984', '14/01/2014']},
                                'header2': {'expect': 'category',
                                            'value': ['M']}},
                                {...}]
        
            if the dataset is not a DataFrame then the header should be omitted. in this example the association is
            a range comparison between 2 and 7 inclusive.
                association= [{'expect': 'number', 'value': [2, 7]},
                              {...}]
        
            The actions dictionary takes the form of an index referenced dictionary of actions, where the key value
            of the dictionary corresponds to the index of the association list. In other words, if a match is found
            in the association, that list index is used as reference to the action to execute.
                {0: {'action': '', 'kwargs' : {}},
                 1: {...}}
            you can also use the action to specify a specific value:
                {0: {'action': ''},
                 1: {'action': ''}}
        
        :param dataset: the dataset to map against, this can be a str, int, float, list, Series or DataFrame
        :param associations: a list of categories (can also contain lists for multiple references.
        :param actions: the correlated set of categories that should map to the index
        :param default_header: (optional) if no association, the default column header to take the value from.
                    if None then the default_value is taken.
                    Note for non-DataFrame datasets the default header is '_default'
        :param default_value: (optional) if no default header then this value is taken if no association
        :param day_first: (optional) if expected type is date, indicates if the day is first. Default to true
        :param quantity: (optional) a number between 0 and 1 presenting the percentage quantity of the data
        :param seed: (optional) a seed value for the random function: default to None
        :return: a list of equal length to the one passed
    


.. code:: ipython3

    associations = [{'age': {'expect': 'n', 'value': [25, 100]},
                     'gender': {'expect': 'category','value': ['M']}},
                    {'age': {'expect': 'n', 'value': [25, 100]},
                     'gender': {'expect': 'category', 'value': ['F']}},
                    {'age': {'expect': 'n', 'value': [0, 24]}},
                    ]
    
    actions = {0: {'action': 'get_category', 'kwargs' : {'selection': ['Dad', 'Papa'], 'weight_pattern': [4,1]}},
               1: {'action': 'get_category','kwargs' : {'selection': ['Mum', 'Mother'], 'weight_pattern': [3,2]}},
               2: {'action': 'Student'}}


Create the new Column
^^^^^^^^^^^^^^^^^^^^^

Notice we have set a default\_value of Unknown for anything that doesn't
fit the rules.

.. code:: ipython3

    dataset['status'] = tools.associate_dataset(dataset, associations=associations, actions=actions, default_value='Unknown')


.. code:: ipython3

    ax = sns.boxplot(x="age", y="status", hue="gender", data=dataset)



.. image:: img/output_11_0.png


Time association
----------------

Time has always been challenging when creating behavioural datasets with
time dependancies and constraints across attribute sets.

In this next example we consider a staff data subset of account creation
and online setup. in the dataset we have: - Staff Id - Staff type
(contractors, part-time, full-time) - when they joined - when they
registered online - status P(ending), A(ctive), S(uspended)

We know from our SME that the following constraints apply: - staff
ration is 10% contractors, 30% part-time, the rest full time -
contractor id starts ith a 'CT-', part-timne with 'PE-' and full-time
with 'FE-' and it is an 8 didget number. - contractors can't register
on-line - Part-time staff and Full-time staff have to have an
online-account - staff records go back 10 years - online accounts only
started 5 years ago - when people join it takes between 5 and 10 days to
set up registration

Create the initial rows
^^^^^^^^^^^^^^^^^^^^^^^

Start with the rows that can be created with ``get_``. Note we are
creating the staff id so we can get unique numbers. we will modify this
next

.. code:: ipython3

    rows = 100
    df_staff = pd.DataFrame()
    df_staff['sid'] = tools.unique_identifiers(from_value=10000000, to_value=99999999, size=rows)
    df_staff['staff_type'] = tools.get_category(selection=['contractor', 'part-time', 'full-time'], weight_pattern=[1,3,6], size=rows)
    df_staff['joined'] = tools.get_datetime(start='01/01/2008', until='07/01/2019', date_format='%d-%m-%Y', size=rows)


.. code:: ipython3

    df_ax = cleaner.to_date_type(df_staff, headers='joined')
    df_ax = cleaner.auto_to_category(df_ax)
    df_ax = cleaner.to_int_type(df_ax, headers='sid')
    discovery.data_dictionary(df_ax)




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
          <th>Attribute</th>
          <th>Type</th>
          <th>% Nulls</th>
          <th>Count</th>
          <th>Unique</th>
          <th>Observations</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>joined</td>
          <td>datetime64[ns]</td>
          <td>0.0</td>
          <td>100</td>
          <td>98</td>
          <td>max=2018-11-16 00:00:00 | min=2008-03-14 00:00:00 | yr mean= 2013</td>
        </tr>
        <tr>
          <th>1</th>
          <td>sid</td>
          <td>int64</td>
          <td>0.0</td>
          <td>100</td>
          <td>100</td>
          <td>max=98989433 | min=10180116 | mean=55971197.5</td>
        </tr>
        <tr>
          <th>2</th>
          <td>staff_type</td>
          <td>category</td>
          <td>0.0</td>
          <td>100</td>
          <td>3</td>
          <td>contractor|full-time|part-time</td>
        </tr>
      </tbody>
    </table>
    </div>



modify the staff id
^^^^^^^^^^^^^^^^^^^

to modify the staff id we can use ``associate_dataset`` to find the
staff\_type. then in the action to prefix the sid.

to prefix the sid we use ``get_custom``

.. code:: ipython3

    tools.get_custom(code_str="f'CU_{x}'", x=1000)




.. parsed-literal::

    ['CU_1000']



To help us get the correct format for the associate and action
parameters use the contextual help and cut and paste the format

.. code:: ipython3

    help(tools.associate_dataset)


.. parsed-literal::

    Help on function associate_dataset in module ds_discovery.simulators.data_builder:
    
    associate_dataset(dataset: Any, associations: list, actions: dict, default_value: Any = None, default_header: str = None, day_first: bool = True, quantity: float = None, seed: int = None)
        Associates a a set of criteria of an input values to a set of actions
            The association dictionary takes the form of a set of dictionaries in a list with each item in the list
            representing an index key for the action dictionary. Each dictionary are to associated relationship.
            In this example for the first index the associated values should be header1 is within a date range
            and header2 has a value of 'M'
                association = [{'header1': {'expect': 'date',
                                            'value': ['12/01/1984', '14/01/2014']},
                                'header2': {'expect': 'category',
                                            'value': ['M']}},
                                {...}]
        
            if the dataset is not a DataFrame then the header should be omitted. in this example the association is
            a range comparison between 2 and 7 inclusive.
                association= [{'expect': 'number', 'value': [2, 7]},
                              {...}]
        
            The actions dictionary takes the form of an index referenced dictionary of actions, where the key value
            of the dictionary corresponds to the index of the association list. In other words, if a match is found
            in the association, that list index is used as reference to the action to execute.
                {0: {'action': '', 'kwargs' : {}},
                 1: {...}}
            you can also use the action to specify a specific value:
                {0: {'action': ''},
                 1: {'action': ''}}
        
        :param dataset: the dataset to map against, this can be a str, int, float, list, Series or DataFrame
        :param associations: a list of categories (can also contain lists for multiple references.
        :param actions: the correlated set of categories that should map to the index
        :param default_header: (optional) if no association, the default column header to take the value from.
                    if None then the default_value is taken.
                    Note for non-DataFrame datasets the default header is '_default'
        :param default_value: (optional) if no default header then this value is taken if no association
        :param day_first: (optional) if expected type is date, indicates if the day is first. Default to true
        :param quantity: (optional) a number between 0 and 1 presenting the percentage quantity of the data
        :param seed: (optional) a seed value for the random function: default to None
        :return: a list of equal length to the one passed
    


Associate Staff Id
^^^^^^^^^^^^^^^^^^

We now can associate the staff type with a modification of the staff id

.. code:: ipython3

    associations = [{'staff_type': {'expect': 'category', 'value': ['contractor']}},
                   {'staff_type': {'expect': 'category', 'value': ['part-time']}},
                   {'staff_type': {'expect': 'category', 'value': ['full-time']}}]
    
    actions = {0: {'action': 'get_custom', 'kwargs' : {'code_str': "f'CT-{sid}'", 'sid': {'header': 'sid'}}}, 
               1: {'action': 'get_custom', 'kwargs' : {'code_str': "f'PT-{sid}'", 'sid': {'header': 'sid'}}}, 
               2: {'action': 'get_custom', 'kwargs' : {'code_str': "f'FT-{sid}'", 'sid': {'header': 'sid'}}}}
    
    
    df_staff['sid'] = tools.associate_dataset(df_staff, associations=associations, actions=actions)

.. code:: ipython3

    df_staff.head(3)




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
          <th>sid</th>
          <th>staff_type</th>
          <th>joined</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>FT-{'header': 'sid'}</td>
          <td>full-time</td>
          <td>04-07-2015</td>
        </tr>
        <tr>
          <th>1</th>
          <td>PT-{'header': 'sid'}</td>
          <td>part-time</td>
          <td>27-10-2016</td>
        </tr>
        <tr>
          <th>2</th>
          <td>FT-{'header': 'sid'}</td>
          <td>full-time</td>
          <td>26-06-2011</td>
        </tr>
      </tbody>
    </table>
    </div>



Associate online registration with start date
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

without the other constraints we could easily use ``correlate_date``
method that correlates one date with a base date giving offset and
spread

.. code:: ipython3

    result = tools.correlate_dates(df_staff['joined'], offset={'days': 7}, lower_spread=4)

We can see from the offset limits max and min from the origional date we
have a difference of between 3 and 7 days giving us our

.. code:: ipython3

    def offset_limits(control, result):
        diff_list = []
        for index in range(rows):
            c_time = pd.to_datetime(control[index], errors='coerce', infer_datetime_format=True, dayfirst=True)
            r_time = pd.to_datetime(result[index], errors='coerce', infer_datetime_format=True, dayfirst=True)
            diff_list.append(r_time - c_time)
        max_diff = max(diff_list)
        min_diff = min(diff_list)
        mean_diff = np.mean(diff_list)
        return min_diff, mean_diff, max_diff
    
    mn, me, mx = offset_limits(df_staff['joined'], result)
    print("min: {}, mean: {}, max: {}".format(mn.days, me.days, mx.days))


.. parsed-literal::

    min: 3, mean: 5, max: 7


But registration only started 5 years ago so we need to set up some
association rules. - if more than 5 years previously then generate a
random date around the 10 days 5 years ago - if within the 5 years,
associate the registration with the join

.. code:: ipython3

    associations = [{'joined': {'expect': 'date', 'value': ['01/01/2000', '31/12/2013']},
                     'staff_type': {'expect': 'category', 'value': ['full-time', 'part-time']}},
                   {'joined': {'expect': 'date', 'value': ['31/12/2013', '31/12/2100']},
                     'staff_type': {'expect': 'category', 'value': ['full-time', 'part-time']}}]
    
    actions = {0: {'action': 'get_datetime', 'kwargs' : {'start': "05/01/2014", 'until': "16/01/2014"}}, 
               1: {'action': 'correlate_dates', 'kwargs' : {'dates': {'_header': 'joined'}, 'offset': {'days': 9}, 'lower_spread': 4}}}
    
    df_staff['registered'] = tools.associate_dataset(df_staff, associations=associations, actions=actions, default_value=None)

Looking at how the registrations match the joined we can clearly see the
characeristics of mass registration at the mid point.

.. code:: ipython3

    df_staff = cleaner.to_date_type(df_staff, headers=['joined', 'registered'], as_num=True)
    
    fig = plt.figure(figsize=(10,4))
    sns.set(style="whitegrid")
    ax = sns.kdeplot(df_staff['registered'], shade=True)
    ax = sns.kdeplot(df_staff['joined'], shade=True)



.. image:: img/output_30_0.png


