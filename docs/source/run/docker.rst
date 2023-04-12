Running in Docker
=================
Docker is a platform for developing, shipping, and running applications in containers. Containers are
lightweight, portable, and self-contained environments that encapsulate all the necessary dependencies
and configurations needed to run an application.

Docker provides a way to package an application with its dependencies into a container, which can then
be run on any machine with the Docker runtime installed. This allows developers to create consistent
environments for their applications across different systems, which can simplify the process of deploying
and scaling applications.

Project Hadron is designed using Microservices. These components services are represented as a
`Domain Contract Ensemble`, the metadata files of a Project Hadron component build, and can be run as Docker
Containers.

To us Project Hadron to run a Docker container we need to set the run parameters, primarily the location
of the `Domain Contract Ensemble`, then build and run.

Setting up
**********
Ensure Docker is installed and if not Docker will help you get started

.. code:: console

   https://docs.docker.com/get-docker/

As part of Project Hadron is a reusable Docker container that can be used with any `Domain Contract Ensemble`.
The Docker executor code bundle can be obtained from the Project Hadron repo

.. code:: console

   https://github.com/project-hadron/discovery-transition-ds/tree/master/ds_discovery/engines/executor

into a folder called executor.

Docker run parameters
*********************
The `Domain Contract Ensemble` location is identified through the environment variable `HADRON_PM_REPO` where
`<domain_contract_ensemble_repo>` is the location of the contract metadata files:

.. code:: console

   -e HADRON_PM_REPO=<domain_contract_ensemble_repo>


in this example we assume the `Domain Contract Ensemple` is locally persisting the data, though the component
can directly use a remote repository such as S3, blob storage or some form of Database. Working with Dockers
instructions, we need to specify where the data will go in the Docker container so we can later map the outcome.
At this point you can add any other Hadron environment variable specific for the component.

.. code:: console

   -e HADRON_DEFAULT_PATH=/root/hadron/data

now we know where the data will be put in our Docker container we can map the volumes to '<your_local_path>' in
order to view the resulting outcomes.

.. code:: console

   -v <your_local_path>:/root/hadron/data

Docker Build and Run
********************

To build the container ensure you are in the root `executor` directory and run

.. code:: console

   docker build -t hadron/domain_executor:latest ./executor


Once complete, to run the `domain contract ensemble`, execute the `docker run` command with the Docker parameters
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
We can use the command line interface (CLI) to create and manage Docker containers. However, the CLI can be
tedious when dealing with multiple containers and configurations.

Docker Compose is a tool that allows you to define and run multi-container Docker applications. With Compose,
you can use a YAML file to configure your application's services, networks, and volumes, and then use a single
command to create and start all the containers that make up your application.

We run the Docker compose yaml file that can be found in the `executor` directory by running
the command:

.. code:: console

   docker-compose up

Next Steps
**********

Try different `Domain Contract Ensemble` components with additional environment variables and use the Docker
documentation to learn about Hadron containers with remote or differing data locations

Project Hadron has been built as a component model to fit seamlessly into an orchestration
engine such as Kubernetes