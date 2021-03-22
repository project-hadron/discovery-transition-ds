from ds_discovery import Controller
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

__author__ = 'Darryl Oatridge'


def domain_controller():
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
    domain_controller()
