from ds_discovery import Controller
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

__author__ = 'Darryl Oatridge'


def domain_controller():
    # Controller
    uri_pm_repo = os.environ.get('HADRON_PM_REPO', None)
    controller = Controller.from_env(uri_pm_repo=uri_pm_repo, default_save=False, has_contract=True)
    run_book = os.environ.get('HADRON_CONTROLLER_RUNBOOK', None)
    repeat = os.environ.get('HADRON_CONTROLLER_REPEAT', None)
    wait = os.environ.get('HADRON_CONTROLLER_WAIT', None)
    controller.run_controller(run_book=run_book, repeat=repeat, wait=wait)


if __name__ == '__main__':
    domain_controller()
