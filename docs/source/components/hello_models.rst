Building for Models
===================

Building a model prediction component works in exactly the same way as
the previous component builds, whereby we create the instance pertinent
to the intentions, give it a location to retrieve data from, the source,
and where to persist the results. Then we add the component intent,
which in this case is to register a trained model and run its
predictions.

In order to run the component you need an appropriate classification
dataset that has already been preprocessed and optimized for the model.
To do this we use the synthetic builder to create this unlabeled
optimised set of features.

Setting Up
----------

.. code:: python

    import os

.. code:: python

    os.environ['HADRON_PM_PATH'] = '0_hello_meta/models_log/contracts'
    os.environ['HADRON_DEFAULT_PATH'] = '0_hello_meta/models_log/data'

Synthetic Binary Classification
-------------------------------

For the synthetic binary classifier the component we use is called
``syntheticbuilder``. In here we have the intent to create the optimized
classification dataset ready for the trained model.

.. code:: python

    import numpy as np
    import pandas as pd
    from ds_discovery import SyntheticBuilder, ModelsBuilder, Commons
    from sklearn.model_selection import train_test_split
    
    %matplotlib inline

.. code:: python

    sb = SyntheticBuilder.from_env('ml_syn', has_contract=False)

.. code:: python

    sb.set_persist()

.. code:: python

    # build a sample dataframe
    sample = 1_000
    df = sb.tools.frame_starter(sample, column_name='frame_shape')
    df['ref_id'] = sb.tools.get_number(from_value=100_000, to_value=999_900, at_most=1, size=df.shape[0], seed=31, column_name='ref_id')
    
    # build classification features optimised for model predict
    df = sb.tools.model_synthetic_classification(canonical=df, n_features=3, n_informative=3, n_redundant=0, seed=42, column_name='classification')


Run Component
~~~~~~~~~~~~~

To run a component we use the common method ``run_component_pipeline``
which loads the source data, executes the component task then persists
the results. This is the only method you can use to run the tasks of a
component and produce its results and should be a familiarized method.

.. code:: python

    # run pipeline
    sb.run_component_pipeline(1_000)

Models Predict
--------------

This mimics the discovery phase of a model error test ultimately
producing the trained model. Discovery is part of the process of
identifying, selecting the features for, and optimizing the algorithm to
produce the predictive model.

.. code:: python

    from ds_discovery import ModelsBuilder
    from sklearn.linear_model import LogisticRegression

.. code:: python

    # get the instance
    ml = ModelsBuilder.from_env('ml_logreg', has_contract=False)

.. code:: python

    ml.set_source_uri(SyntheticBuilder.from_env('ml_syn').get_persist_contract().uri)

Split (test, train)
~~~~~~~~~~~~~~~~~~~

.. code:: python

    # select X, Y 
    X = df.drop(['target', 'ref_id'], axis=1)
    # X = df.drop(['target'], axis=1)
    y = df['target']

.. code:: python

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

Probabiliy of Y given X
~~~~~~~~~~~~~~~~~~~~~~~

For this example we use a simple logistic regression algorithm from
Scikit-learn, though this will apply to any model fit that has a predict
method. The following formula is applied.

.. math::  \hat y = \sigma( w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}} 

.. code:: python

    log_reg = LogisticRegression(solver='liblinear')
    log_reg.fit(X_train.values, y_train.values)




.. raw:: html

    <style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression(solver=&#x27;liblinear&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(solver=&#x27;liblinear&#x27;)</pre></div></div></div></div></div>



Trained Model
-------------

After all the preparation we now get to the component build. To this
point we have created the trained model as part of discovery.

Register Model
~~~~~~~~~~~~~~

With this train model we add it to the trained model registry using
either the singular default name or an optional unique name. This is
used when selecting an appropriate model predict against a given data
set.

.. code:: python

    ml.add_trained_model(trained_model=log_reg)

Predict Model
~~~~~~~~~~~~~

We are now ready to receive unlabeled data to predict its
classification. Each run of the pipeline will produce an ordered set of
predictions relating to the features given.

.. code:: python

    y_pred = ml.intent_model.label_predict(X_test)

.. code:: python

    # classification rate
    np.around(np.mean(y_test.to_numpy()==y_pred['predict'].to_numpy()),3)

.. code:: python

    0.897

Predict Model with Reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition a unique reference can be passed so that each prediction
aligns with that unique reference identifier.

.. code:: python

    # add the reference id to the predict frame
    df_ref = df['ref_id'].iloc[X_test.index].to_frame()
    X_test = pd.concat([df_ref, X_test], axis=1)

.. code:: python

    y_pred = ml.intent_model.label_predict(X_test, id_header='ref_id')

.. code:: python

    y_pred.head()


.. image:: /images/hello_hadron/6_img01.png
  :align: center
  :width: 150

\


