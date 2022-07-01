import inspect
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ds_discovery.intent.abstract_common_intent import AbstractCommonsIntentModel
from ds_discovery.managers.feature_catalog_property_manager import FeatureCatalogPropertyManager
from ds_discovery.components.commons import Commons
from aistac.components.aistac_commons import DataAnalytics
from ds_discovery.components.discovery import DataDiscovery

__author__ = 'Darryl Oatridge'


class FeatureCatalogIntentModel(AbstractCommonsIntentModel):
    """A set of methods to help build features as pandas.Dataframe"""

    def __init__(self, property_manager: FeatureCatalogPropertyManager, default_save_intent: bool=None,
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
        intent_param_exclude = []
        intent_type_additions = [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64,
                                 pd.Timestamp]
        super().__init__(property_manager=property_manager, default_save_intent=default_save_intent,
                         intent_param_exclude=intent_param_exclude, default_intent_level=default_intent_level,
                         default_intent_order=default_intent_order, default_replace_intent=default_replace_intent,
                         intent_type_additions=intent_type_additions)

    def run_intent_pipeline(self, canonical: Any, feature_name: [int, str], train_size: [float, int]=None,
                            seed: int=None, shuffle: bool=None, **kwargs) -> [pd.DataFrame, pd.Series]:
        """ Collectively runs all parameterised intent taken from the property manager against the code base as
        defined by the intent_contract.

        It is expected that all intent methods have the 'canonical' as the first parameter of the method signature
        and will contain 'save_intent' as parameters. It is also assumed that all features have a feature contract to
        save the feature outcome to

        :param canonical: this is the iterative value all intent are applied to and returned.
        :param feature_name: feature to run
        :param train_size: (optional) If float, should be between 0.0 and 1.0 and represent the proportion of the
                            dataset to include in the train split. If int, represents the absolute number of train
                            samples. If None, then not used
        :param seed: (optional) if shuffle is True a seed value for the choice
        :param shuffle: (optional) Whether or not to shuffle the data before splitting or just split on train size.
        :param kwargs: additional kwargs to add to the parameterised intent, these will replace any that already exist
        :return
        """
        # test if there is any intent to run
        if self._pm.has_intent(level=feature_name):
            canonical = self._get_canonical(canonical)
            if isinstance(train_size, (float, int)):
                canonical = self.canonical_sampler(canonical, sample_size=train_size, shuffle=shuffle, seed=seed)
            # run the feature
            level_key = self._pm.join(self._pm.KEY.intent_key, feature_name)
            df_feature = None
            for order in sorted(self._pm.get(level_key, {})):
                for method, params in self._pm.get(self._pm.join(level_key, order), {}).items():
                    if method in self.__dir__():
                        if 'canonical' in params.keys():
                            df_feature = params.pop('canonical')
                        elif df_feature is None:
                            df_feature = canonical
                        # fail safe in case kwargs was sored as the reference
                        params.update(params.pop('kwargs', {}))
                        # add method kwargs to the params
                        if isinstance(kwargs, dict):
                            params.update(kwargs)
                        # remove the creator param
                        _ = params.pop('intent_creator', 'Unknown')
                        # add excluded params and set to False
                        params.update({'save_intent': False})
                        df_feature = eval(f"self.{method}(df_feature, **{params})", globals(), locals())
            if df_feature is None:
                raise ValueError(f"The feature '{feature_name}' pipeline did not run. ")
            return df_feature
        raise ValueError(f"The feature '{feature_name}, can't be found in the feature catalog")

    def _template(self, canonical: Any, target: Any, seed: int=None,
                  save_intent: bool=None, feature_name: [int, str]=None, intent_order: int=None,
                  replace_intent: bool=None, remove_duplicates: bool=None) -> pd.DataFrame:
        """ used for feature attribution allowing columns to be selected directly from the canonical attributes

        :param canonical: the Pandas.DataFrame to get the selection from
        :param target: the key column to index on
        :param seed: (optional) a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param feature_name: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: selected list of headers indexed on key
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   feature_name=feature_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        canonical = self._get_canonical(canonical)
        seed = seed if isinstance(seed, int) else self._seed()

        return canonical

    def select_correlate(self, canonical: Any, target: Any, threshold: float=None, train_size: float=None,
                         seed: int=None, save_intent: bool=None, feature_name: [int, str]=None, intent_order: int=None,
                         replace_intent: bool=None, remove_duplicates: bool=None) -> pd.DataFrame:
        """ used for feature attribution allowing columns to be selected directly from the canonical attributes.
        Correlation Feature Selection evaluates subsets of features on the basis of the following hypothesis:
            "Good feature subsets contain features highly correlated with the target, yet uncorrelated to each other".
        References:
            M. Hall 1999, "Correlation-based Feature Selection for Machine Learning"
            Senliol, Baris, et al. "Fast Correlation Based Filter (FCBF) with a different search strategy."

        Using the Pearson's correlation method, finds groups of correlated features, which we can then explore using
        a Decision Tree to decide which one we keep and which ones we discard.

        :param canonical: the Pandas.DataFrame to get the selection from
        :param target: the key column to index on
        :param threshold: the correlation threshold to evaluate subsets of features set between 0 and 1. Default 0.998
        :param train_size: (optional) The size of the training data subset to avoid over-fitting set between 0 and 1.
        :param seed: (optional) a seed value for the random function: default to None
        :param save_intent (optional) if the intent contract should be saved to the property manager
        :param feature_name: (optional) the level name that groups intent by a reference name
        :param intent_order: (optional) the order in which each intent should run.
                        If None: default's to -1
                        if -1: added to a level above any current instance of the intent section, level 0 if not found
                        if int: added to the level specified, overwriting any that already exist
        :param replace_intent: (optional) if the intent method exists at the level, or default level
                        True - replaces the current intent method with the new
                        False - leaves it untouched, disregarding the new intent
        :param remove_duplicates: (optional) removes any duplicate intent in any level that is identical
        :return: selected list of headers indexed on key
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   feature_name=feature_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        canonical = self._get_canonical(canonical)
        seed = seed if isinstance(seed, int) else self._seed()
        threshold = threshold if isinstance(threshold, float) and 0 < threshold < 1 else 0.998
        train_size = train_size if isinstance(train_size, float) and 0 < train_size < 1 else 0.3
        # to avoid over-fitting
        X_train, X_test, y_train, y_test = train_test_split(canonical.drop(labels=[target], axis=1), canonical[target],
                                                            test_size=1-train_size, random_state=seed)
        corrmat = X_train.corr()
        corrmat = corrmat.abs().unstack().sort_values(ascending=False)
        corrmat = corrmat[corrmat >= threshold]
        corrmat = corrmat[corrmat < 1]
        corrmat = pd.DataFrame(corrmat).reset_index()
        corrmat.columns = ['feature1', 'feature2', 'corr']
        grouped_feature_ls = []
        correlated_groups = []
        for feature in corrmat.feature1.unique():
            if feature not in grouped_feature_ls:
                # find all features correlated to a single feature
                correlated_block = corrmat[corrmat.feature1 == feature]
                grouped_feature_ls = grouped_feature_ls + list(correlated_block.feature2.unique()) + [feature]
                # append the block of features to the list
                correlated_groups.append(correlated_block)
        to_drop = []
        for group in correlated_groups:
            # add all features of the group to a list
            features = list(group['feature2'].unique()) + list(group['feature1'].unique())
            # train a random forest
            rf = RandomForestClassifier(n_estimators=200, random_state=seed, max_depth=4)
            rf.fit(X_train[features].fillna(0), y_train)
            # get the feature importance attributed by the
            importance = pd.concat([pd.Series(features), pd.Series(rf.feature_importances_)], axis=1)
            importance.columns = ['feature', 'importance']
            # sort features by importance, most important first
            importance = importance.sort_values(by='importance', ascending=False)
            to_drop += (importance['feature'][1:].to_list())
        return canonical.drop(labels=to_drop, axis=1)
        # return Commons.filter_columns(canonical, headers=to_drop, drop=True)

    """
        PRIVATE METHODS SECTION
    """
    def _intent_builder(self, method: str, params: dict, exclude: list=None) -> dict:
        """builds the intent_params. Pass the method name and local() parameters
            Example:
                self._intent_builder(inspect.currentframe().f_code.co_name, **locals())

        :param method: the name of the method (intent). can use 'inspect.currentframe().f_code.co_name'
        :param params: the parameters passed to the method. use `locals()` in the caller method
        :param exclude: (optional) convenience parameter identifying param keys to exclude.
        :return: dict of the intent
        """
        if not isinstance(params.get('canonical', None), (str, dict)):
            exclude = ['canonical']
        return super()._intent_builder(method=method, params=params, exclude=exclude)

    def _set_intend_signature(self, intent_params: dict, feature_name: [int, str]=None, intent_order: int=None,
                              replace_intent: bool=None, remove_duplicates: bool=None, save_intent: bool=None):
        """ sets the intent section in the configuration file. Note: by default any identical intent, e.g.
        intent with the same intent (name) and the same parameter values, are removed from any level.

        :param intent_params: a dictionary type set of configuration representing a intent section contract
        :param feature_name: (optional) the feature name that groups intent by a reference name
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
        if save_intent or (not isinstance(save_intent, bool) and self._default_save_intent):
            if not isinstance(feature_name, (str, int)) or not feature_name:
                raise ValueError(f"if the intent is to be saved then a feature name must be provided")
        super()._set_intend_signature(intent_params=intent_params, intent_level=feature_name, intent_order=intent_order,
                                      replace_intent=replace_intent, remove_duplicates=remove_duplicates,
                                      save_intent=save_intent)
        return
