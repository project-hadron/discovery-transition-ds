Reports: Citing the Data
========================

Adding citations to support appropriate attribution by third-party users
in order to formally incorporate data reuse

.. code:: ipython3

    from ds_discovery import Transition
    from aistac.handlers.abstract_handlers import ConnectorContract

Dataset Citation
----------------

As part of the set-up process and as best practice, the component is
cited through added knowledge from the component’s creator or SME
feedback. In addition the data location of the source and persist is
also captured.

This is extended with the Project Hadron transistion component,
considered the data entry point reporting tool, which includes a special
method call to add provenance. Provenance sites a number of origin
indicators that guide the user to the data’s provenance, its
restrictions such as cost and license, its provider and the data’s
author.

Additional knowledge can be added beyond the set provenance (see other
sections).

.. code:: ipython3

    tr = Transition.from_env('demo_citation', has_contract=False)

Adding Citation
---------------

As part of the set-up process, or at anytime during the component
development cycle, information can be gathered and added to the
component as part of its information store.

It is worth noting, method calls allow partial completion with
additional information added at a later date as knowledge is gained or
changed.

.. code:: ipython3

    tr.set_description("Every arrest effected in NYC by the NYPD from 2006 to the end of the previous calendar year")
    tr.set_version('0.0.1')
    tr.set_status('discovery')
    tr.set_source_uri(uri="https://data.cityofnewyork.us/api/views/8h9b-rp9u/rows.csv")
    tr.set_persist(uri_file='NYPD_Arrest_Historic.parquet')
    tr.setup_bootstrap(domain='Public Safty', 
                       project_name='arrest_reduction', 
                       description="Validate datasets quality, quantity, verasity and completeness", 
                       file_type='parquet')

.. code:: ipython3

    tr.set_provenance(title='NYPD Histroic Arrest Data',
                      domain='Public Safty', 
                      license_type='Public Consuption',
                      description="List of every arrest in NYC going back to 2006 through the end of the previous calendar year.",
                      provider_name='Police Department (NYPD)', 
                      provider_uri="https://data.cityofnewyork.us/Public-Safety/NYPD-Arrests-Data-Historic-/8h9b-rp9u",
                      provider_note="This data is manually extracted every quarter and reviewed by the Office of Management Analysis and Planning before being posted on the NYPD website.",
                      cost_price="$0.00",
                      cost_type="batch")

Reports
-------

Once information is added it can easily be accessed, either visually
through reporting or remotely through predefined connector contracts. In
our case we are visually displaying the reports for the purpose of
demonstration but would normally be connected to a reporting tool for
information capture.

Component Reporting
^^^^^^^^^^^^^^^^^^^

Our initial report shows information capture about our component.

.. code:: ipython3

    tr.report_task()

Connectivity Reporting
^^^^^^^^^^^^^^^^^^^^^^

As part of all components one can also interrogate where data is coming
from and going to, which connector contracts have been set up and what
they look like. In this case we only require our primary source and
persist connectors from which we can identify the data’s location and
how we retrived it.

.. code:: ipython3

    tr.report_connectors()

Provenance Reporting
^^^^^^^^^^^^^^^^^^^^

Finially and specificially to the transistioning component, we citate
the provider of our data and that citation can be added to as knowledge
is gained.

This information not only shows us the domain and description of the
provider but also the providers details, the datas author and
restrictions on that data through license and costs. This information
can easily be passed to a separate component that could for example
monitor cost/spend on data throughput or collate common provider
sourcing for data reuse.

.. code:: ipython3

    tr.report_provenance()

