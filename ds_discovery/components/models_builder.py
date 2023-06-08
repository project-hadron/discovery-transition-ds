from __future__ import annotations

import os
from typing import Any

from aistac import ConnectorContract

from ds_discovery.components.abstract_common_component import AbstractCommonComponent
from ds_discovery.intent.models_intent import ModelsIntentModel
from ds_discovery.managers.models_property_manager import ModelsPropertyManager

__author__ = 'Darryl Oatridge'


class ModelsBuilder(AbstractCommonComponent):

    @classmethod
    def from_uri(cls, task_name: str, uri_pm_path: str, creator: str, uri_pm_repo: str=None, pm_file_type: str=None,
                 pm_module: str=None, pm_handler: str=None, pm_kwargs: dict=None, default_save=None,
                 reset_templates: bool=None, template_path: str=None, template_module: str=None,
                 template_source_handler: str=None, template_persist_handler: str=None, align_connectors: bool=None,
                 default_save_intent: bool=None, default_intent_level: bool=None, order_next_available: bool=None,
                 default_replace_intent: bool=None, has_contract: bool=None) -> ModelsBuilder:
        """ Class Factory Method to instantiates the components application. The Factory Method handles the
        instantiation of the Properties Manager, the Intent Model and the persistence of the uploaded properties.
        See class inline docs for an example method

         :param task_name: The reference name that uniquely identifies a task or subset of the property manager
         :param uri_pm_path: A URI that identifies the resource path for the property manager.
         :param creator: A user name for this task activity.
         :param uri_pm_repo: (optional) A repository URI to initially load the property manager but not save to.
         :param pm_file_type: (optional) defines a specific file type for the property manager
         :param pm_module: (optional) the module or package name where the handler can be found
         :param pm_handler: (optional) the handler for retrieving the resource
         :param pm_kwargs: (optional) a dictionary of kwargs to pass to the property manager
         :param default_save: (optional) if the configuration should be persisted. default to 'True'
         :param reset_templates: (optional) reset connector templates from environ variables. Default True
                                (see `report_environ()`)
         :param template_path: (optional) a template path to use if the environment variable does not exist
         :param template_module: (optional) a template module to use if the environment variable does not exist
         :param template_source_handler: (optional) a template source handler to use if no environment variable
         :param template_persist_handler: (optional) a template persist handler to use if no environment variable
         :param align_connectors: (optional) resets aligned connectors to the template. default Default True
         :param default_save_intent: (optional) The default action for saving intent in the property manager
         :param default_intent_level: (optional) the default level intent should be saved at
         :param order_next_available: (optional) if the default behaviour for the order should be next available order
         :param default_replace_intent: (optional) the default replace existing intent behaviour
         :param has_contract: (optional) indicates the instance should have a property manager domain contract
         :return: the initialised class instance
         """
        pm_file_type = pm_file_type if isinstance(pm_file_type, str) else 'parquet'
        pm_module = pm_module if isinstance(pm_module, str) else 'ds_discovery.handlers.pandas_handlers'
        pm_handler = pm_handler if isinstance(pm_handler, str) else 'PandasPersistHandler'
        creator = creator if isinstance(creator, str) else 'Unknown'
        _pm = ModelsPropertyManager(task_name=task_name, creator=creator)
        _intent_model = ModelsIntentModel(property_manager=_pm, default_save_intent=default_save_intent,
                                          default_intent_level=default_intent_level,
                                          order_next_available=order_next_available,
                                          default_replace_intent=default_replace_intent)
        super()._init_properties(property_manager=_pm, uri_pm_path=uri_pm_path, default_save=default_save,
                                 uri_pm_repo=uri_pm_repo, pm_file_type=pm_file_type, pm_module=pm_module,
                                 pm_handler=pm_handler, pm_kwargs=pm_kwargs, has_contract=has_contract)
        return cls(property_manager=_pm, intent_model=_intent_model, default_save=default_save,
                   reset_templates=reset_templates, template_path=template_path, template_module=template_module,
                   template_source_handler=template_source_handler, template_persist_handler=template_persist_handler,
                   align_connectors=align_connectors)

    @property
    def pm(self) -> ModelsPropertyManager:
        return self._component_pm

    @property
    def intent_model(self) -> ModelsIntentModel:
        return self._intent_model

    @property
    def models(self) -> ModelsIntentModel:
        return self._intent_model

    def add_trained_model(self, trained_model: Any, model_name: str=None, save: bool=None):
        """ A utility method to save the trained model ready for prediction.

        :param trained_model: model object that has been trained
        :param model_name: (optional) a unique name for the model
        :param save: (optional) override of the default save action set at initialisation.
        """
        connector_name = model_name if isinstance(model_name, str) else self.pm.CONNECTOR_ML_TRAINED
        uri_file =  self.pm.file_pattern(name=connector_name, file_type='pickle', versioned=True)
        template = self.pm.get_connector_contract(connector_name=self.pm.TEMPLATE_PERSIST)
        # uri = ConnectorContract.parse_environ(os.path.join(template.raw_uri, uri_file))
        uri = os.path.join(template.raw_uri, uri_file)
        cc = ConnectorContract(uri=uri, module_name=template.raw_module_name, handler=template.raw_handler,
                               version=self.pm.version)
        self.add_connector_contract(connector_name=connector_name, connector_contract=cc,
                                    template_aligned=True, save=save)
        self.persist_canonical(connector_name=connector_name, canonical=trained_model)
        return

    def run_component_pipeline(self, intent_levels: [str, int, list]=None, run_book: str=None, use_default: bool=None,
                               reset_changed: bool=None, has_changed: bool=None):
        """ Runs the component's pipeline from source to persist

        :param intent_levels: a single or list of intent levels to run
        :param run_book: a saved runbook to run
        :param use_default: if the default runbook should be used if it exists
        :param reset_changed: (optional) resets the has_changed boolean to True
        :param has_changed: (optional) tests if the underline canonical has changed since last load else error returned
        :return:
        """
        canonical = self.load_source_canonical(reset_changed=reset_changed, has_changed=has_changed)
        use_default = use_default if isinstance(use_default, bool) else True
        if not isinstance(run_book, str) and use_default:
            if self.pm.has_run_book(book_name=self.pm.PRIMARY_RUN_BOOK):
                run_book = self.pm.PRIMARY_RUN_BOOK
        result = self.intent_model.run_intent_pipeline(canonical, intent_levels=intent_levels, run_book=run_book)
        self.save_persist_canonical(result)
