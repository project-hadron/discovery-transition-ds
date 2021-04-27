# Action file
import json
import sys
from cortex import Cortex
from cortex.utils import generate_token
from ds_discovery import Controller
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

__author__ = 'Darryl Oatridge'


def domain_controller(params: dict):
    # initialise the Cortex client
    api_endpoint = params.get('apiEndpoint')
    token = params.get('token')
    project_id = params.get('projectId')
    client = Cortex.client(api_endpoint=api_endpoint, token=token, project=project_id)

    # # get secrets keys
    # os.environ["AWS_SECRET_ACCESS_KEY"] = str(client.get_secret("awssecretkey"))
    # os.environ["AWS_ACCESS_KEY_ID"] = str(client.get_secret("awspublickey"))

    # just in case there are old environment variables for hadron
    for key in os.environ.keys():
        if key.startswith('HADRON'):
            del os.environ[key]

    # run through the payload and set environment variables, and remove from the params
    payload = params.get('payload')
    hadron_kwargs = payload.get('hadron_kwargs', {})
    uri_pm_repo = payload.get('domain_contract_repo')
    if not isinstance(uri_pm_repo, str):
        raise KeyError("The message parameters passed do not have the mandatory 'domain_contract_repo' payload key")
    for key in hadron_kwargs.copy().keys():
        if str(key).isupper():
            os.environ[key] = hadron_kwargs.pop(key)

    # Controller
    controller = Controller.from_env(uri_pm_repo=uri_pm_repo, default_save=False, has_contract=True, **hadron_kwargs)
    run_book = os.environ.get('HADRON_CONTROLLER_RUNBOOK', None)
    repeat = os.environ.get('HADRON_CONTROLLER_REPEAT', None)
    sleep = os.environ.get('HADRON_CONTROLLER_WAIT', None)
    controller.run_controller(run_book=run_book, repeat=repeat, sleep=sleep)


if __name__ == '__main__':
    # if len(sys.argv) < 2:
    #     print("Message/payload commandline is required")
    #     exit(1)
    # domain_controller(json.loads(sys.argv[-1]))
    # Local test
    asset_bank_uri = "https://raw.githubusercontent.com/project-hadron/hadron-asset-bank/master/contracts"
    contract = "factory/healthcare"
    PAT = {"jwk": {"crv": "Ed25519",
                   "x": "OXdyU11SG10iRiaYststIz5sSt7Dk0qWd-AEdVW-CA0",
                   "d": "HgCri9Xw33SC3qCQMG5Q1dcixv7OgU9lf91OCeiK7-g",
                   "kty": "OKP",
                   "kid": "tpmSp9SwgNUlZQC7xIe3wqwFPW6EcjTrn2hpfCnHjZ4"},
           "issuer": "cognitivescale.com",
           "audience": "cortex",
           "username": "f42d74b1-d67b-48dc-ab09-5d0771d6b7c0",
           "url": "https://api.dci-dev.dev-eks.insights.ai"}
    params = {
        "token": generate_token(PAT),
        "payload": {
            "domain_contract_repo": os.path.join(asset_bank_uri, contract),
            "hadron_kwargs": {
                "HADRON_DEFAULT_PATH": "s3://project-hadron-cs-repo/datalake_gen/healthcare/members",
                "HADRON_DEFAULT_MODULE": "ds_discovery.handlers.s3_handlers",
                "HADRON_DEFAULT_HANDLER": "S3PersistHandler",
            },
        },
        "apiEndpoint": PAT.get("url"),
        "projectId": "helloworld-new-3832b"
    }
    domain_controller(params)
