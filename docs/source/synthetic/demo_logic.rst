Use Case: Logic Test
====================

Set by a senior data scientist and researcher to test the logic of
correlation observed across a distributions. This applies to both the preparation of
features for machine learning and distribution techniques for synthetic
data. These logic tests were considered challenges to the data science
team in the preparation of their data for consumption into their models.

.. code:: ipython3

    import pandas as pd
    from ds_discovery import Wrangle

.. code:: ipython3

    wr = Wrangle.from_memory()
    tools = wr.tools

Logic Tests
***********

1. (A AND B) OR C
2. !A AND B
3. !(A AND B)
4. A AND !B
5. (A OR B) AND (C OR D)

.. code:: ipython3

    df = pd.DataFrame()

.. code:: ipython3

    df['s1'] = pd.Series(list('AAAABBBBCCCCDDDD'))
    df['s2'] = pd.Series(list('ABCDABCDABCDABCD'))
    df['s3'] = pd.Series([1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8])


(A AND B) OR C
**************

Single column
-------------

.. code:: ipython3

    A = tools.select2dict(column='s3', condition="(@ > 2)", logic='AND')
    B = tools.select2dict(column='s3', condition="(@ < 5)", logic='AND')
    C = tools.select2dict(column='s3', condition="@ == 8", logic='OR')
    
    selection = [[A, B], C]
    
    df['l1'] = tools.correlate_selection(df, selection=selection, action=1, default_action=0)

.. code:: ipython3

    df[df['l1'] == 1].loc[:,['s3']]

.. image:: /images/demo/log_img01.png
  :align: center
  :width: 55

Multi column
------------

.. code:: ipython3

    A = tools.select2dict(column='s1', condition="@ == 'A'")
    B = tools.select2dict(column='s2', condition="@ == 'B'", logic='AND')
    C = tools.select2dict(column='s1', condition="@ == 'C'", logic='OR')
    
    selection = [[A, B], C]
    
    df['l1'] = tools.correlate_selection(df, selection=selection, action=1, default_action=0)

.. code:: ipython3

    df[df['l1'] == 1].loc[:,['s1','s2']]

.. image:: /images/demo/log_img02.png
  :align: center
  :width: 85


!A AND B
********

Single column
-------------

.. code:: ipython3

    A = tools.select2dict(column='s3', condition="@ == 7", logic='NOT')
    B = tools.select2dict(column='s3', condition="@ > 4", logic='AND')
    
    selection = [A, B]
    
    df['l2'] = tools.correlate_selection(df, selection=selection, action=1, default_action=0)

.. code:: ipython3

    df[df['l2'] == 1].loc[:,['s3']]

.. image:: /images/demo/log_img03.png
  :align: center
  :width: 55

Multi column
------------

.. code:: ipython3

    A = tools.select2dict(column='s1', condition="@ == 'A'", logic='NOT')
    B = tools.select2dict(column='s2', condition="@ == 'B'", logic='AND')
    
    selection = [A, B]
    
    df['l2'] = tools.correlate_selection(df, selection=selection, action=1, default_action=0)

.. code:: ipython3

    df[df['l2'] == 1].loc[:,['s1', 's2']]

.. image:: /images/demo/log_img04.png
  :align: center
  :width: 75


!(A AND B)
**********

Single column
-------------

.. code:: ipython3

    A = tools.select2dict(column='s3', condition="@ < 8")
    B = tools.select2dict(column='s3', condition="@ > 3", logic='AND')
    
    selection = [[A, B], 'NOT']
    
    df['l1'] = tools.correlate_selection(df, selection=selection, action=1, default_action=0)

.. code:: ipython3

    df[df['l1'] == 1].loc[:,['s3']]

.. image:: /images/demo/log_img05.png
  :align: center
  :width: 55

Multi column
------------

.. code:: ipython3

    A = tools.select2dict(column='s1', condition="@ == 'A'")
    B = tools.select2dict(column='s2', condition="@ == 'B'", logic='AND')
    
    selection = selection = [[A, B], 'NOT']
    
    df['l3'] = tools.correlate_selection(df, selection=selection, action=1, default_action=0)

.. code:: ipython3

    df[df['l3'] == 1].loc[:,['s1','s2']]

.. image:: /images/demo/log_img06.png
  :align: center
  :width: 80



A AND !B
********

Single column
-------------

.. code:: ipython3

    A = tools.select2dict(column='s3', condition="@ > 5")
    B = tools.select2dict(column='s3', condition="@ == 7", logic='NOT')
    
    selection = [A, B]
    
    df['l1'] = tools.correlate_selection(df, selection=selection, action=1, default_action=0)

.. code:: ipython3

    df[df['l1'] == 1].loc[:,['s3']]

.. image:: /images/demo/log_img07.png
  :align: center
  :width: 55

Multi column
------------

.. code:: ipython3

    A = tools.select2dict(column='s1', condition="@ == 'A'")
    B = tools.select2dict(column='s2', condition="@ == 'B'", logic='NOT')
    
    selection = [A, B]
    
    df['l4'] = tools.correlate_selection(df, selection=selection, action=1, default_action=0)

.. code:: ipython3

    df[df['l4'] == 1].loc[:,['s1','s2']]

.. image:: /images/demo/log_img08.png
  :align: center
  :width: 75



(A OR B) AND (C OR D)
*********************

Single column
-------------

.. code:: ipython3

    A = tools.select2dict(column='s3', condition="(@ < 3)")
    B = tools.select2dict(column='s3', condition="(@ > 5)", logic='OR')
    C = tools.select2dict(column='s3', condition="@ == 2")
    D = tools.select2dict(column='s3', condition="@ > 7", logic='OR')
    
    selection = [[A, B], 'AND', [C, D]]
    
    df['l1'] = tools.correlate_selection(df, selection=selection, action=1, default_action=0)

.. code:: ipython3

    df[df['l1'] == 1].loc[:,['s3']]

.. image:: /images/demo/log_img09.png
  :align: center
  :width: 55

Multi column
------------

.. code:: ipython3

    A = tools.select2dict(column='s1', condition="@ == 'A'")
    B = tools.select2dict(column='s2', condition="@ == 'B'", logic='OR')
    C = tools.select2dict(column='s1', condition="@ == 'C'")
    D = tools.select2dict(column='s2', condition="@ == 'D'", logic='OR')
    
    selection = [[A, B], 'AND', [C, D]]
    
    df['l4'] = tools.correlate_selection(df, selection=selection, action=1, default_action=0)

.. code:: ipython3

    df[df['l4'] == 1].loc[:,['s1','s2']]

.. image:: /images/demo/log_img10.png
  :align: center
  :width: 75

