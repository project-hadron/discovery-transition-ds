Building for Models
===================

Building a model prediction component works in exactly the same way as
the previous component builds, whereby we create the instance pertinent
to the intentions, give it a location to retrieve data from, the source,
and where to persist the results. Then we add the component intent,
which in this case is to register a trained model and run its
predictions.

.. image:: /images/hello_hadron/component_pipeline.png
  :align: center
  :width: 700

\

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


To run a component we use the common method ``run_component_pipeline``
which loads the source data, executes the component task then persists
the results. This is the only method you can use to run the tasks of a
component and produce its results and should be a familiarized method.

.. code:: python

    # run pipeline
    sb.run_component_pipeline(1_000)

Discovery
---------

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

Split
~~~~~

.. code:: python

    # select X, Y 
    X = df.drop(['target', 'ref_id'], axis=1)
    # X = df.drop(['target'], axis=1)
    y = df['target']

.. code:: python

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

Test Train
~~~~~~~~~~

For this example we use a simple logistic regression algorithm from
Scikit-learn, though this will apply to any model fit that has a predict
method. The following formula is applied.

.. math::  \hat y = \sigma( w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}} 

.. code:: python

    log_reg = LogisticRegression(solver='liblinear')
    log_reg.fit(X_train.values, y_train.values)


Prediction
----------

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

Predict Classification
~~~~~~~~~~~~~~~~~~~~~~

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

Predict Classification with Reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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



