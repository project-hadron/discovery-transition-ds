from aistac.components.abstract_ledger_component import AbstractLedger
from aistac.properties.ledger_property_manager import LedgerPropertyManager

class Ledger(AbstractLedger):

    def __init__(self, property_manager: LedgerPropertyManager, intent_model: LedgerIntentModel,
                 default_save=None, reset_templates: bool = None, align_connectors: bool = None):
        super().__init__(property_manager=property_manager, intent_model=intent_model, default_save=default_save,
                         reset_templates=reset_templates, align_connectors=align_connectors)

    @classmethod
    def from_uri(cls, task_name: str, uri_pm_path: str, pm_file_type: str = None, pm_module: str = None,
                 pm_handler: str = None, pm_kwargs: dict = None, default_save=None, reset_templates: bool = None,
                 align_connectors: bool = None, default_save_intent: bool = None, default_intent_level: bool = None,
                 order_next_available: bool = None, default_replace_intent: bool = None):
        pm_file_type = pm_file_type if isinstance(pm_file_type, str) else 'json'
        pm_module = pm_module if isinstance(pm_module, str) else 'aistac.handlers.python_handlers'
        pm_handler = pm_handler if isinstance(pm_handler, str) else 'PythonPersistHandler'
        _pm = ExamplePropertyManager(task_name=task_name)
        _intent_model = ExampleIntentModel(property_manager=_pm, default_save_intent=default_save_intent,
                                           default_intent_level=default_intent_level,
                                           order_next_available=order_next_available,
                                           default_replace_intent=default_replace_intent)
        super()._init_properties(property_manager=_pm, uri_pm_path=uri_pm_path, pm_file_type=pm_file_type,
                                 pm_module=pm_module, pm_handler=pm_handler, pm_kwargs=pm_kwargs)
        return cls(property_manager=_pm, intent_model=_intent_model, default_save=default_save,
                   reset_templates=reset_templates, align_connectors=align_connectors)
