## Action file
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
    default_path = msg.payload.get('HADRON_DEFAULT_PATH', '')
    pm_repo = msg.payload.get('HADRON_PM_REPO', '')
    sample_size = msg.payload.get('HADRON_CONTROLLER_SIZE_MEMBERS_GEN', '')
    # Getting Secrets
    api_endpoint = msg.payload.get('apiEndpoint')
    token = msg.payload.get('token')
    project_id = msg.payload.get('projectId')
    client = Cortex.client(api_endpoint=api_endpoint, token=token, project=project_id)
    aws_key = client.get_secret('AWS_ACCESS_KEY_ID')['value']
    aws_secret = client.get_secret('AWS_SECRET_ACCESS_KEY')['value']
    # Controller
    controller = Controller.from_env(default_save=False, has_contract=True)
    run_book = os.environ.get('HADRON_CONTROLLER_RUNBOOK', None)
    synthetic_size_map = dict([(k[23:].lower(), int(v))
                               for k, v in os.environ.items()
                               if k.startswith('HADRON_CONTROLLER_SIZE_')])
    intent_levels = None
    if isinstance(run_book, str) and controller.pm.has_run_book(run_book):
        intent_levels = controller.pm.get_run_book(run_book)
    synthetic_size_map = synthetic_size_map if isinstance(synthetic_size_map, dict) else None
    controller.run_controller(intent_levels=intent_levels, synthetic_sizes=synthetic_size_map)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Message/payload commandline is required")
        exit(1)
    domain_controller(json.loads(sys.argv[-1]))
