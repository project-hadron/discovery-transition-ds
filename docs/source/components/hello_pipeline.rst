Running a Component in Python
=============================

Project Hadron is designed using Microservices. The components services
are represented as a single Domain Contract or as a Domain Contract
Ensemble that contains a Controller to orchestrate this ensemple of
components. In both instances we need to point to the repo where the
Domain Contracts are, in this case GitHub.

.. code:: ipython3

    repo = "https://raw.githubusercontent.com/project-hadron/hadron-asset-bank/master/contracts/hello_hadron/hello_ensemble"

.. code:: ipython3

    from ds_discovery import Transition, Controller

Run Component Pipeline
----------------------

Generally pipelines are run as an ensemble, but occasionlly when
exploring an individual component’s details, one might want to run it on
its own to observe its outcome. Components are selected by their
``class``, their ``task_name`` and the repo where the Domain Contract
can be found.

.. code:: ipython3

    tr = Transition.from_env('hello_tr', uri_pm_repo=repo)

Now we have loaded the instance of the component we can observe its
details. In this case we are looking at its actions of intent.

.. code:: ipython3

    tr.report_intent()

.. image:: /images/hello_hadron/5_img01.png
  :align: center
  :width: 550

To run our sampled component we use the familiar method
``run_component_pipeline`` which loads the source data, executes the
component task then persists the results. This is the only method you
can use to run the tasks of a component and produce its results and
should be a familiarized method.

.. code:: ipython3

    tr.run_component_pipeline()

Run Ensemble Pipeline
---------------------

More commonly components are run as an ensemble of Domain Contracts with
a controller component orchestrating a single or set of other
components. Creating and using a controller component gives us far more
command over our individual components and allows us a better view of
the run. An ensemble of components, orchestrated by a controller is
considered akin to a microservice.

As we have seen before, as there is only one controller, we don’t need
to give it a ``task_name`` but we do need to point to the repo where the
domain contracts are, including the controller.

.. code:: ipython3

    controller = Controller.from_env(uri_pm_repo=repo)

We can now observe the tasks the controller is orchestrating, and their
details, allowing us a view of the ensemble or access to the details to
dive deeper into each component.

.. code:: ipython3

    controller.report_tasks()

.. image:: /images/hello_hadron/5_img02.png
  :align: center
  :width: 420

Finally we run the controller, passing parameters that help us observe
the run.

.. code:: ipython3

    controller.run_controller(run_cycle_report='cycle_report.csv')

.. code:: ipython3

    controller.load_canonical(connector_name='run_cycle_report')

.. image:: /images/hello_hadron/5_img03.png
  :align: center
  :width: 350

