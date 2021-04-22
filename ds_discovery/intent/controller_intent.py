import inspect
from typing import Any
import numpy as np
import pandas as pd
from aistac.intent.abstract_intent import AbstractIntentModel
from ds_discovery.components.abstract_common_component import AbstractCommonComponent

from ds_discovery import FeatureCatalog, Transition, DataDrift, Wrangle, SyntheticBuilder
from ds_discovery.managers.controller_property_manager import ControllerPropertyManager

__author__ = 'Darryl Oatridge'


class ControllerIntentModel(AbstractIntentModel):

    def __init__(self, property_manager: ControllerPropertyManager, default_save_intent: bool=None,
                 default_intent_level: [str, int, float]=None, order_next_available: bool=None,
                 default_replace_intent: bool=None):
        """initialisation of the Intent class.

        :param property_manager: the property manager class that references the intent contract.
        :param default_save_intent: (optional) The default action for saving intent in the property manager
        :param default_intent_level: (optional) the default level intent should be saved at
        :param order_next_available: (optional) if the default behaviour for the order should be next available order
        :param default_replace_intent: (optional) the default replace existing intent behaviour
        """
        default_save_intent = default_save_intent if isinstance(default_save_intent, bool) else True
        default_replace_intent = default_replace_intent if isinstance(default_replace_intent, bool) else True
        default_intent_level = default_intent_level if isinstance(default_intent_level, (str, int, float)) else 'base'
        default_intent_order = -1 if isinstance(order_next_available, bool) and order_next_available else 0
        intent_param_exclude = ['canonical']
        intent_type_additions = [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64,
                                 pd.Timestamp]
        super().__init__(property_manager=property_manager, default_save_intent=default_save_intent,
                         intent_param_exclude=intent_param_exclude, default_intent_level=default_intent_level,
                         default_intent_order=default_intent_order, default_replace_intent=default_replace_intent,
                         intent_type_additions=intent_type_additions)

    def run_intent_pipeline(self, canonical: pd.DataFrame, intent_level: [int, str]=None, synthetic_size: int=None,
                            controller_repo: str=None, persist_result: bool=None, **kwargs):
        """ Collectively runs all parameterised intent taken from the property manager against the code base as
        defined by the intent_contract.

        It is expected that all intent methods have the 'canonical' as the first parameter of the method signature
        and will contain 'save_intent' as parameters.

        :param canonical: this is the iterative value all intent are applied to and returned.
        :param intent_level: (optional) The intent_level to run. if none then assume pm constant DEFAULT_INTENT_LEVEL
        :param synthetic_size: (optional) a size to pass to any synthetic intent
        :param controller_repo: (optional) the controller repo to use if no uri_pm_repo is within the intent parameters
        :param persist_result: (optional) if the intent results should be persisted as well as returned in memory
        :param kwargs: additional kwargs to add to the parameterised intent, these will replace any that already exist
        :return Canonical with parameterised intent applied
        """
        # test if there is any intent to run
        if self._pm.has_intent():
            # get the list of levels to run
            intent_level = intent_level if isinstance(intent_level, (int, str)) else self._pm.DEFAULT_INTENT_LEVEL
            level_key = self._pm.join(self._pm.KEY.intent_key, intent_level)
            for order in sorted(self._pm.get(level_key, {})):
                for method, params in self._pm.get(self._pm.join(level_key, order), {}).items():
                    if method in self.__dir__():
                        # fail safe in case kwargs was sored as the reference
                        params.update(params.pop('kwargs', {}))
                        # add method kwargs to the params
                        if isinstance(kwargs, dict):
                            params.update(kwargs)
                        # remove the creator param
                        _ = params.pop('intent_creator', 'Unknown')
                        # add excluded params and set to False
                        params.update({'run_task': True, 'save_intent': False})
                        # add the controller_repo if given
                        if isinstance(controller_repo, str) and 'uri_pm_repo' not in params.keys():
                            params.update({'uri_pm_repo': controller_repo})
                        if isinstance(persist_result, bool):
                            params.update({'persist_result': persist_result})
                        if method == 'synthetic_builder':
                            if isinstance(synthetic_size, int):
                                canonical = synthetic_size
                        canonical = eval(f"self.{method}(canonical, **{params})", globals(), locals())
        return canonical

    def synthetic_builder(self, canonical: Any, task_name: str, columns: [str, list]=None, uri_pm_repo: str=None,
                          run_task: bool=None, persist_result: bool=None, size: int=None, save_intent: bool=None,
                          intent_order: int=None, intent_level: [int, str]=None, replace_intent: bool=None,
                          remove_duplicates: bool=None):
        """ register a synthetic component task pipeline

        :param canonical: this can be a size integer or a starting canonical size is based upon
        :param task_name: the task_name reference for this component
        :param columns: (optional) a single or list of intent_level to run, if list, run in order given
        :param uri_pm_repo: (optional) A repository URI to initially load the property manager but not save to.
        :param run_task: (optional) if when adding the task it should also be run returning the canonical outcome
        :param persist_result: (optional) if the resulting canonical should be persisted.
        :param size: (optional) legacy size parameter now replaced by passing an int as the canonical
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
       """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # create the event book
        if isinstance(run_task, bool) and run_task:
            persist_result = persist_result if isinstance(persist_result, bool) else False
            params = {'uri_pm_repo': uri_pm_repo} if isinstance(uri_pm_repo, str) else {}
            builder: SyntheticBuilder = eval(f"SyntheticBuilder.from_env(task_name=task_name, default_save=False, "
                                             f"has_contract=True, **{params})", globals(), locals())
            canonical = builder.intent_model.run_intent_pipeline(canonical=canonical, intent_levels=columns)
            # persist the canonical
            if persist_result and builder.pm.has_connector(builder.CONNECTOR_PERSIST):
                builder.save_persist_canonical(canonical=canonical)
            # create reports
            self._common_reports(builder)
            # customer reports
            if builder.pm.has_connector(builder.REPORT_SCHEMA):
                builder.save_report_canonical(reports=builder.REPORT_SCHEMA,
                                              report_canonical=builder.report_canonical_schema(stylise=False))
            if builder.pm.has_connector(builder.REPORT_CATALOG):
                builder.save_report_canonical(reports=builder.REPORT_CATALOG,
                                              report_canonical=builder.report_column_catalog(stylise=False))
            return canonical
        return

    def transition(self, canonical: Any, task_name: str, uri_pm_repo: str=None, run_task: bool=None,
                   transition_intent: [int, str, list]=None, persist_result: bool=None, save_intent: bool=None,
                   intent_order: int=None, intent_level: [int, str]=None, replace_intent: bool=None,
                   remove_duplicates: bool=None):
        """ register a Transition component task pipeline

        :param canonical: the canonical to run through the component pipeline
        :param task_name: the task_name reference for this component
        :param uri_pm_repo: (optional) A repository URI to initially load the property manager but not save to.
        :param run_task: (optional) if when adding the task it should also be run returning the canonical outcome
        :param transition_intent: (optional) a single or list of components levels to run, if list, run in order given
        :param persist_result: (optional) if the resulting canonical should be persisted.
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
       """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        if isinstance(run_task, bool) and run_task:
            persist_result = persist_result if isinstance(persist_result, bool) else False
            params = {'uri_pm_repo': uri_pm_repo} if isinstance(uri_pm_repo, str) else {}
            tr: Transition = eval(f"Transition.from_env(task_name=task_name, default_save=False, has_contract=True, "
                                  f"**{params})", globals(), locals())
            if canonical.shape == (0, 0):
                canonical = tr.load_source_canonical()
            canonical = tr.intent_model.run_intent_pipeline(canonical=canonical, intent_levels=intent_level,
                                                            inplace=False)
            # persist the canonical
            if persist_result and tr.pm.has_connector(tr.CONNECTOR_PERSIST):
                tr.save_persist_canonical(canonical=canonical)

            # create reports
            self._common_reports(tr)
            # customer reports
            if tr.pm.has_connector(tr.REPORT_SCHEMA):
                tr.save_report_canonical(reports=tr.REPORT_SCHEMA,
                                         report_canonical=tr.report_canonical_schema(stylise=False))
            if tr.pm.has_connector(tr.REPORT_SUMMARY):
                tr.save_report_canonical(reports=tr.REPORT_SUMMARY,
                                         report_canonical=tr.report_quality_summary(stylise=False))
            if tr.pm.has_connector(tr.REPORT_PROVENANCE):
                tr.save_report_canonical(reports=tr.REPORT_PROVENANCE,
                                         report_canonical=tr.report_provenance(stylise=False))
            if tr.pm.has_connector(tr.REPORT_FIELDS):
                tr.save_report_canonical(reports=tr.REPORT_FIELDS,
                                         report_canonical=tr.report_attributes(canonical=canonical, stylise=False))
            if tr.pm.has_connector(tr.REPORT_DICTIONARY):
                tr.save_report_canonical(reports=tr.REPORT_DICTIONARY,
                                         report_canonical=tr.canonical_report(canonical=canonical, stylise=False))
            if tr.pm.has_connector(tr.REPORT_QUALITY):
                tr.save_report_canonical(reports=tr.REPORT_QUALITY,
                                         report_canonical=tr.report_quality(canonical=canonical))
            return canonical
        return

    def wrangle(self, canonical: Any, task_name: str, uri_pm_repo: str=None, run_task: bool=None,
                wrangled_intent: [int, str, list]=None, persist_result: bool=None, save_intent: bool=None,
                intent_order: int=None, intent_level: [int, str]=None, replace_intent: bool=None,
                remove_duplicates: bool=None):
        """ register a Transition component task pipeline

        :param canonical: the canonical to run through the component pipeline
        :param task_name: the task_name reference for this component
        :param uri_pm_repo: (optional) A repository URI to initially load the property manager but not save to.
        :param run_task: (optional) if when adding the task it should also be run returning the canonical outcome
        :param wrangled_intent: (optional) an single or list of wrangled levels to run, if list, run in order given
        :param persist_result: (optional) if the resulting canonical should be persisted.
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
       """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # create the event book
        if isinstance(run_task, bool) and run_task:
            persist_result = persist_result if isinstance(persist_result, bool) else False
            params = {'uri_pm_repo': uri_pm_repo} if isinstance(uri_pm_repo, str) else {}
            wr: Wrangle = eval(f"Wrangle.from_env(task_name=task_name, default_save=False, has_contract=True, "
                               f"**{params})", globals(), locals())
            if canonical.shape == (0, 0):
                canonical = wr.load_source_canonical()
            canonical = wr.intent_model.run_intent_pipeline(canonical=canonical, intent_levels=intent_level,
                                                            inplace=False)
            # persist the canonical
            if persist_result and wr.pm.has_connector(wr.CONNECTOR_PERSIST):
                wr.save_persist_canonical(canonical=canonical)
            # create reports
            self._common_reports(wr)
            # customer reports

            return canonical
        return

    def feature_catalog(self, canonical: Any, task_name: str, feature_name: [int, str]=None, uri_pm_repo: str=None,
                        run_task: bool=None, train_size: [float, int]=None, seed: int=None, shuffle: bool=None,
                        persist_result: bool=None, save_intent: bool=None, intent_level: [int, str]=None,
                        intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None):
        """ register a Feature Catalog component task pipeline

        :param canonical: the canonical to run through the component pipeline
        :param task_name: the task_name reference for this component
        :param feature_name: feature to run
        :param uri_pm_repo: (optional) A repository URI to initially load the property manager but not save to.
        :param train_size: (optional) If float, should be between 0.0 and 1.0 and represent the proportion of the
                            dataset to include in the train split. If int, represents the absolute number of train
                            samples. If None, then not used
        :param seed: (optional) if shuffle is True a seed value for the choice
        :param shuffle: (optional) Whether or not to shuffle the data before splitting or just split on train size.
        :param run_task: (optional) if when adding the task it should also be run returning the canonical outcome
        :param persist_result: (optional) if the resulting canonical should be persisted.
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
       """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # create the event book
        if isinstance(run_task, bool) and run_task:
            persist_result = persist_result if isinstance(persist_result, bool) else False
            params = {'uri_pm_repo': uri_pm_repo} if isinstance(uri_pm_repo, str) else {}
            fc: FeatureCatalog = eval(f"FeatureCatalog.from_env(task_name=task_name, default_save=False, "
                                      f"has_contract=True, **{params})", globals(), locals())
            if canonical.shape == (0, 0):
                canonical = fc.load_source_canonical()
            canonical = fc.intent_model.run_intent_pipeline(canonical=canonical, feature_name=feature_name,
                                                            train_size=train_size, seed=seed, shuffle=shuffle)
            if persist_result and fc.pm.has_connector(feature_name):
                fc.save_catalog_feature(feature_name=feature_name, canonical=canonical)
            # create reports
            self._common_reports(fc)
            # customer reports

            return canonical
        return

    def data_drift(self, canonical: Any, task_name: str, measure: [int, str], uri_pm_repo: str=None,
                   run_task: bool=None, persist_result: bool=None, save_intent: bool=None, intent_order: int=None,
                   intent_level: [int, str]=None, replace_intent: bool=None, remove_duplicates: bool=None):
        """ register a data tolerance component task pipeline

        :param canonical: the canonical to run through the component pipeline
        :param task_name: the task_name reference for this component
        :param measure: a single measure to fun
        :param uri_pm_repo: (optional) A repository URI to initially load the property manager but not save to.
        :param run_task: (optional) if when adding the task it should also be run returning the canonical outcome
        :param persist_result: (optional) if the resulting canonical should be persisted.
        :param save_intent: (optional) if the intent contract should be saved to the property manager
        :param intent_level: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
       """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   intent_level=intent_level, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # create the event book
        if isinstance(run_task, bool) and run_task:
            persist_result = persist_result if isinstance(persist_result, bool) else False
            params = {'uri_pm_repo': uri_pm_repo} if isinstance(uri_pm_repo, str) else {}
            ct: DataDrift = eval(f"DataTolerance.from_env(task_name=task_name, default_save=False, "
                                 f"has_contract=True, **{params})", globals(), locals())
            if canonical.shape == (0, 0):
                canonical = ct.load_source_canonical()
            canonical = ct.intent_model.run_intent_pipeline(canonical=canonical, measure=measure)
            if persist_result and ct.pm.has_connector(measure):
                ct.save_persist_canonical(feature_name=measure, canonical=canonical)
            # create reports
            self._common_reports(ct)
            # customer reports

            return canonical
        return

    def _set_intend_signature(self, intent_params: dict, intent_level: [int, str]=None, intent_order: int=None,
                              replace_intent: bool=None, remove_duplicates: bool=None, save_intent: bool=None):
        """ sets the intent section in the configuration file. Note: by default any identical intent, e.g.
        intent with the same intent (name) and the same parameter values, are removed from any level.

        :param intent_params: a dictionary type set of configuration representing a intent section contract
        :param intent_level: (optional) ta level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :param save_intent (optional) if the intent contract should be saved to the property manager
        """
        intent_level = intent_level if isinstance(intent_level, (str, int)) else self._pm.DEFAULT_INTENT_LEVEL
        if save_intent or (not isinstance(save_intent, bool) and self._default_save_intent):
            if not isinstance(intent_order, int) or intent_order == -1:
                if self._pm.get_intent():
                    intent_order = 0
                    while True:
                        if not self._pm.is_key(self._pm.join(self._pm.KEY.intent_key, intent_level, intent_order)):
                            break
                        intent_order += 1
        super()._set_intend_signature(intent_params=intent_params, intent_level=intent_level, intent_order=intent_order,
                                      replace_intent=replace_intent, remove_duplicates=remove_duplicates,
                                      save_intent=save_intent)
        return

    @staticmethod
    def _common_reports(inst: AbstractCommonComponent):
        """all the common reports to all components"""
        if inst.pm.has_connector(inst.REPORT_SCHEMA):
            inst.save_report_canonical(reports=inst.REPORT_SCHEMA,
                                       report_canonical=inst.report_canonical_schema(stylise=False))
            inst.save_report_canonical(reports=inst.REPORT_INTENT,
                                       report_canonical=inst.report_intent(stylise=False))
            inst.save_report_canonical(reports=inst.REPORT_NOTES,
                                       report_canonical=inst.report_notes(stylise=False))
            inst.save_report_canonical(reports=inst.REPORT_RUNBOOK,
                                       report_canonical=inst.report_run_book(stylise=False))
            inst.save_report_canonical(reports=inst.REPORT_CONNECTORS,
                                       report_canonical=inst.report_connectors(stylise=False))
            inst.save_report_canonical(reports=inst.REPORT_ENVIRON,
                                       report_canonical=inst.report_environ(stylise=False))
