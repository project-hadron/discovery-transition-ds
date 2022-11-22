Component Executor
==================
Project Hadron is designed using Microservices. These components services are represented as a
`Domain Contract Ensemble`, the metadata files of a Project Hadron component build. To run a docker
container we need to set the run parameters, primarily the location of the `Domain Contract Ensemble`,
then build and run.

Requirements
************
Python 3.8.x

Docker run parameters
*********************
The `Domain Contract Ensemble` location is identified through the environment variable `HADRON_PM_REPO` where
`<domain_contract_ensemble_repo>` is the location of the contract metadata files:

.. code:: console

   -e HADRON_PM_REPO=<domain_contract_ensemble_repo>


in this example we assume the `Domain Contract Ensemple` is locally persisting the data, though the component
can directly use a remote repository such as S3, blob storage or some form of Database. Working with dockers
instructions, we need to specify where the data will go in the docker container so we can later map the outcome.
At this point you can add any other Hadron environment variable specific for the component.

.. code:: console

   -e HADRON_DEFAULT_PATH=/root/hadron/data

now we know where the data will be put in our docker container we can map the volumes to '<your_local_path>' in
order to view the resulting outcomes.

.. code:: console

   -v <your_local_path>:/root/hadron/data

Docker Build and Run
********************

To build the container ensure you are in the root `executor` directory and run

.. code:: console

   docker build -t hadron/domain_executor:latest ./executor


Once complete, to run the `domain contract ensemble`, execute the `docker run` command with the docker parameters
above the `hadron/domain_executor` container. (remember to change `<domain_contract_ensemble_repo>` to the location
of the domain contract ensemble and `<your_local_path>` for your local path)

.. code:: console

   docker run --rm -it
      -e HADRON_DEFAULT_PATH=/root/hadron/data
      -e HADRON_PM_REPO=<domain_contract_ensemble_repo>
      -v <your_local_path>:/root/hadron/data
      hadron/domain_executor


Docker Compose
**************
Alternatively you can run the docker compose yaml file that can be found in the `executor` directory by running
the command:

.. code:: console

   docker-compose up

Next Steps
**********

Try different `Domain Contract Ensemble` components with additional environment variables and use the docker
documentation to learn about Hadron containers with remote or differing data locations
