from ds_discovery import Controller
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

__author__ = 'Darryl Oatridge'


def domain_controller(params: dict):
    # extract the payload
    payload = params.get('payload', {})

    # get the domain contract repo from the payload
    uri_pm_repo = payload.get('domain_contract_repo')
    if not isinstance(uri_pm_repo, str):
        raise KeyError("The message parameters passed do not have the mandatory 'domain_contract_repo' payload key")

    # extract any extra kwargs
    hadron_kwargs = payload.get('hadron_kwargs', {})
    # export and pop any environment variable from the kwargs
    for key in hadron_kwargs.copy().keys():
        if str(key).isupper():
            os.environ[key] = hadron_kwargs.pop(key)
    # pop the run_controller attributes from the kwargs
    run_book = hadron_kwargs.pop('runbook', None)
    mod_tasks = hadron_kwargs.pop('mod_tasks', None)
    repeat = hadron_kwargs.pop('repeat', None)
    sleep = hadron_kwargs.pop('sleep', None)
    run_time = hadron_kwargs.pop('run_time', None)
    run_cycle_report = hadron_kwargs.pop('run_cycle_report', None)
    source_check_uri = hadron_kwargs.pop('source_check_uri', None)

    # instantiate the Controller passing any remaining kwargs
    controller = Controller.from_env(uri_pm_repo=uri_pm_repo, default_save=False, has_contract=True, **hadron_kwargs)
    # run the controller nano services.
    controller.run_controller(run_book=run_book, mod_tasks=mod_tasks, repeat=repeat, sleep=sleep, run_time=run_time,
                              source_check_uri=source_check_uri, run_cycle_report=run_cycle_report)


if __name__ == '__main__':
    domain_controller()
