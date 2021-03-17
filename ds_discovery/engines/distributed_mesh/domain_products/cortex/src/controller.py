# Action file
import json
import sys
from cortex import Cortex, Message
from ds_discovery import Controller
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

__author__ = 'Darryl Oatridge'


def domain_controller(prams: dict):
    # getting payload
    msg = Message(prams)
    # initialise the Cortex client
    # api_endpoint = msg.get('apiEndpoint')
    # token = msg.get('token')
    # project_id = msg.get('projectId')
    # client = Cortex.client(api_endpoint=api_endpoint, token=token, project=project_id)
    # just in case there are old environment variables for hadron
    for key in os.environ.keys():
        if key.startswith('HADRON'):
            del os.environ[key]
    # run through the hadron params and set environment variables, deleting them from the params
    hadron_kwargs = msg.payload.get('hadron_kwargs', {})
    uri_pm_repo = msg.payload.get('domain_contract_repo')
    if not isinstance(uri_pm_repo, str):
        raise KeyError("The message parameters passed do not have the mandatory 'domain_contract_repo' payload key")
    for key in hadron_kwargs.copy().keys():
        if str(key).isupper():
            os.environ[key] = hadron_kwargs.pop(key)
    # Controller
    controller = Controller.from_env(uri_pm_repo=uri_pm_repo, default_save=False, has_contract=True, **hadron_kwargs)
    run_book = os.environ.get('HADRON_CONTROLLER_RUNBOOK', None)
    synthetic_size_map = dict([(k[23:].lower(), int(v))
                               for k, v in os.environ.items()
                               if k.startswith('HADRON_CONTROLLER_SIZE_')])
    intent_levels = None
    if isinstance(run_book, str) and controller.pm.has_run_book(run_book):
        intent_levels = controller.pm.get_run_book(run_book)
    synthetic_size_map = synthetic_size_map if isinstance(synthetic_size_map, dict) else None
    controller.run_controller(intent_levels=intent_levels, synthetic_sizes=synthetic_size_map)


# if __name__ == '__main__':
#     if len(sys.argv) < 2:
#         print("Message/payload commandline is required")
#         exit(1)
#     domain_controller(json.loads(sys.argv[-1]))

if __name__ == '__main__':
    params = {
        "token": "qe4aef556yegsfdwf4356yheyhy7867",
        "payload": {
            'domain_contract_repo': 'https://raw.githubusercontent.com/project-hadron/hadron-asset-bank/master/'
                                    'contracts/healthcare/factory/members/',
            'hadron_kwargs': {
                'HADRON_DEFAULT_PATH': "s3://project-hadron-cs-repo/datalake_gen/healthcare/members",
                'HADRON_DEFAULT_MODULE': 'ds_discovery.handlers.s3_handlers',
                'HADRON_DEFAULT_HANDLER': 'S3PersistHandler',
                # 'HADRON_CONTROLLER_SIZE_MEMBERS_GEN': '1000',
                },
            },
        "apiEndpoint": "http://someendpoint.com",
        "projectId": "123435"
    }
    domain_controller(params)

