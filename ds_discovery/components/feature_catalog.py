from __future__ import annotations

import pandas as pd
from ds_discovery.components.abstract_common_component import AbstractCommonComponent
from aistac.handlers.abstract_handlers import ConnectorContract

from ds_discovery.components.commons import Commons
from ds_discovery.intent.feature_catalog_intent import FeatureCatalogIntentModel
from ds_discovery.managers.feature_catalog_property_manager import FeatureCatalogPropertyManager


class FeatureCatalog(AbstractCommonComponent):

    REPORT_DICTIONARY = 'dictionary'

    @classmethod
    def from_uri(cls, task_name: str, uri_pm_path: str, username: str, uri_pm_repo: str=None, pm_file_type: str=None,
                 pm_module: str=None, pm_handler: str=None, pm_kwargs: dict=None, default_save=None,
                 reset_templates: bool=None, template_path: str=None, template_module: str=None,
                 template_source_handler: str=None, template_persist_handler: str=None, align_connectors: bool=None,
                 default_save_intent: bool=None, default_intent_level: bool=None, order_next_available: bool=None,
                 default_replace_intent: bool=None, has_contract: bool=None) -> FeatureCatalog:
        """ Class Factory Method to instantiates the components application. The Factory Method handles the
        instantiation of the Properties Manager, the Intent Model and the persistence of the uploaded properties.
        See class inline docs for an example method

         :param task_name: The reference name that uniquely identifies a task or subset of the property manager
         :param uri_pm_path: A URI that identifies the resource path for the property manager.
         :param username: A user name for this task activity.
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
        pm_file_type = pm_file_type if isinstance(pm_file_type, str) else 'json'
        pm_module = pm_module if isinstance(pm_module, str) else 'ds_discovery.handlers.pandas_handlers'
        pm_handler = pm_handler if isinstance(pm_handler, str) else 'PandasPersistHandler'
        username = username if isinstance(username, str) else 'Unknown'
        _pm = FeatureCatalogPropertyManager(task_name=task_name, username=username)
        _intent_model = FeatureCatalogIntentModel(property_manager=_pm, default_save_intent=default_save_intent,
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

    @classmethod
    def scratch_pad(cls) -> FeatureCatalogIntentModel:
        """ A class method to use the Components intent methods as a scratch pad"""
        return super().scratch_pad()

    @property
    def intent_model(self) -> FeatureCatalogIntentModel:
        """The intent model instance"""
        return self._intent_model

    @property
    def tools(self) -> FeatureCatalogIntentModel:
        """The intent model instance"""
        return self._intent_model

    @property
    def pm(self) -> FeatureCatalogPropertyManager:
        """The properties manager instance"""
        return self._component_pm

    def get_feature_contract(self, feature_name: str=None) -> ConnectorContract:
        """ gets the feature connector contract

        :param feature_name: The unique name of the feature
        :return: connector contract
        """
        return self.pm.get_connector_contract(connector_name=feature_name)

    def set_feature_bootstrap(self, feature_name: str, description: str=None, versioned: bool=None, stamped: bool=None,
                              file_type: str=None, save: bool=None):
        """sets a feature bootstrap setting description name and using the TEMPLATE_PERSIST connector contract to
        create a connector for that feature

        :param feature_name: the unique name of the feature
        :param description: an optional description for the feature
        :param versioned: (optional) if the component version should be included as part of the pattern
        :param stamped: (optional) A string of the timestamp options ['days', 'hours', 'minutes', 'seconds', 'ns']
        :param file_type: (optional) a connector supported file extension type different from the default e.g. 'csv'
        :param save: (optional) if True, save to file. Default is True
        """
        versioned = versioned if isinstance(versioned, bool) else True
        uri_file = self.pm.file_pattern(name=feature_name, versioned=versioned, stamped=stamped,
                                        file_type=file_type)
        self.add_connector_from_template(connector_name=feature_name, uri_file=uri_file,
                                         template_name=self.TEMPLATE_PERSIST, save=save)
        if isinstance(description, str):
            self.pm.set_intent_description(level=feature_name, text=description)
        return

    def load_catalog_feature(self, feature_name: str, reset_index: bool=None) -> pd.DataFrame:
        """returns the feature data as a DataFrame

        :param feature_name: a unique feature name
        :param reset_index: if the index should be reset bringing the key into the columns
        :return: pandas DataFrame
        """
        canonical = self.load_canonical(connector_name=feature_name)
        if isinstance(reset_index, bool) and reset_index:
            canonical.reset_index(inplace=True)
        return canonical

    def save_catalog_feature(self, canonical, feature_name: str):
        """Saves the pandas.DataFrame to the feature catalog

        :param canonical: the pandas DataFrame
        :param feature_name: a unique feature name
        """
        self.persist_canonical(connector_name=feature_name, canonical=canonical)

    def add_feature_description(self, feature_name: str, description: str, save: bool=None):
        """ adds a description note that is included in with the 'report_features'"""
        if isinstance(description, str) and description:
            self.pm.set_intent_description(level=feature_name, text=description)
            self.pm_persist(save)
        return

    def remove_feature(self, feature_name: str, save: bool=None):
        """completely removes a feature including connector, intent and description"""
        if self.pm.has_connector(connector_name=feature_name):
            self.remove_connector_contract(connector_name=feature_name, save=save)
        if self.pm.has_intent(level=feature_name):
            self.remove_intent(level=feature_name)
        return

    def run_component_pipeline(self, canonical: pd.DataFrame=None, feature_names: [str, list]=None,
                               auto_connectors: bool=None, save: bool=None, reset_changed: bool=None,
                               has_changed: bool=None):
        """runs all features within the feature catalog or an optional set of features

        :param canonical: (optional) A canonical if the source canonical isn't to be used
        :param feature_names: (optional) a single or list of features to run
        :param auto_connectors: (optional) Adds a versioned feature connector if not yet added. Default to True
        :param save: (optional) if True, persist changes to property manager. Default is True
        :param reset_changed: (optional) resets the has_changed boolean to True
        :param has_changed: (optional) tests if the underline canonical has changed since last load else error returned
        """
        auto_connectors = auto_connectors if isinstance(auto_connectors, bool) else True
        if isinstance(feature_names, (str, list)):
            feature_names = Commons.list_formatter(feature_names)
        else:
            feature_names = Commons.list_formatter(self.pm.get_intent())
        if not isinstance(canonical, (pd.DataFrame, str)):
            canonical = self.load_source_canonical(reset_changed=reset_changed, has_changed=has_changed)
        for feature in feature_names:
            if not self.pm.has_connector(feature):
                if not auto_connectors:
                    continue
                self.set_feature_bootstrap(feature_name=feature, versioned=True, save=save)
            result = self.intent_model.run_intent_pipeline(canonical, feature)
            self.save_catalog_feature(feature_name=feature, canonical=result)
        return
