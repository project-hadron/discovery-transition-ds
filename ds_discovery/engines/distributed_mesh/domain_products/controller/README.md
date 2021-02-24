# Distributed Data Science Mesh (DDSM)
An example of using the DDSM in the form of a single 'Domain Product' run from docker container. In this example
we are creating a `Domain Product` generates a factory healthcare members dataset

## Requirements
Python 3.8.x

## Docker run parameters
The remote `Domain Contract` are identified through an environment variable point to where the root path of the 
contracts can be found, in this case we are pointing at a Repository directory and thus using `HADRON_PM_REPO`:

```
-e HADRON_PM_REPO=https://raw.githubusercontent.com/project-hadron/hadron-asset-bank/master/bundles/hadron/hk_income/contracts/
``` 

in this example the `Domain Contracts` locally persisting the data, rather than using a remote repository such as 
blob storage or a Database so we need to specify where the data will go in the container so we can map the outcome.

```
-e HADRON_DEFAULT_PATH=/root/hadron/data
```

now we know where the data will be put in our container we can map the volumes to '<your_local_path>' in order to 
view the resulting outcomes

```
-v <your_local_path>:/root/hadron/data
```

## Docker Build and Run
To build the container ensure you are in the root `domain_products` directory and run
```
docker build -t hadron/discovery-swarm:latest ./domain_products
```
Once complete, to run the `swarm`, execute the `docker run` command with the docker parameters above the
`hadron/discovery-swarm` container. (remember to change `<your_local_path>` for your local path)
```
docker run --rm -it 
            -e HADRON_DEFAULT_PATH=/root/hadron/data
            -e HADRON_PM_REPO=https://raw.githubusercontent.com/project-hadron/hadron-asset-bank/master/bundles/hadron/hk_income/contracts/
            -v <your_local_path>:/root/hadron/data
            hadron/discovery-swarm
```

## Docker Compose
Alternatively you can run the docker compose yaml file that can be found in the `domain_products` directory by running 
the command:

```
docker-compose up
```

## What you will see
The container runs the `Domain Product` producing the dataset as a comma separated file along with 4 other 
reporting datasets all in `.json` format
+ Data Dictionary
+ Data Quality Report
+ Data Quality Summary Report
+ Data Analytics Report

The reports produced are in addition to the outcome of the `Swarm` run and specified as part of the `Domain Contracts`