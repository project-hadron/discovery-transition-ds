# Action file
import json
import sys
from cortex import Cortex
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

    # get secrets keys
    os.environ["AWS_SECRET_ACCESS_KEY"] = str(client.get_secret("awssecretkey")["value"])
    os.environ["AWS_ACCESS_KEY_ID"] = str(client.get_secret("awspublickey")["value"])

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
    wait = os.environ.get('HADRON_CONTROLLER_WAIT', None)
    controller.run_controller(run_book=run_book, repeat=repeat, wait=wait)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Message/payload commandline is required")
        exit(1)
    domain_controller(json.loads(sys.argv[-1]))


"""
params = {
    "token": "eyJraWQiOiJ0cG1TcDlTd2dOVWxaUUM3eEllM3dxd0ZQVzZFY2pUcm4yaHBmQ25Ialo0IiwiYWxnIjoiRWREU0EifQ.eyJzdWIiOiJmNDJkNzRiMS1kNjdiLTQ4ZGMtYWIwOS01ZDA3NzFkNmI3YzAiLCJhdWQiOiJjb3J0ZXgiLCJpc3MiOiJjb2duaXRpdmVzY2FsZS5jb20iLCJpYXQiOjE2MTU5OTk4MjcsImV4cCI6MTYxNjA4NjIyN30.rKSt1INBAtNgB2o8KA011EBuesIZo8x8LhybkYvDeuqOI-RumgZuk6IpGbJs5AR2Or65UZAwzdBF9us0dEJ5AA",
    "payload": {
    "domain_contract_repo": "https://raw.githubusercontent.com/project-hadron/hadron-asset-bank/master/contracts/healthcare/factory/members/",
    "hadron_kwargs": {
        "HADRON_DEFAULT_PATH": "s3://project-hadron-cs-repo/datalake_gen/healthcare/members-test",
        "HADRON_DEFAULT_MODULE": "ds_discovery.handlers.s3_handlers",
        "HADRON_DEFAULT_HANDLER": "S3PersistHandler",
        },
    },
    "apiEndpoint": "https://api.dci-dev.dev-eks.insights.ai",
    "projectId": "helloworld-new-3832b"
}
domain_controller(params)
"""