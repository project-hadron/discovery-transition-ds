import inspect
from typing import Any
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, \
    GradientBoostingRegressor

from ds_discovery.intent.abstract_common_intent import AbstractCommonsIntentModel
from ds_discovery.managers.feature_catalog_property_manager import FeatureCatalogPropertyManager
from ds_discovery.components.commons import Commons

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

    def run_intent_pipeline(self, canonical: Any, feature_name: [int, str], seed: int=None, shuffle: bool=None,
                            **kwargs) -> [pd.DataFrame, pd.Series]:
        """ Collectively runs all parameterised intent taken from the property manager against the code base as
        defined by the intent_contract.

        It is expected that all intent methods have the 'canonical' as the first parameter of the method signature
        and will contain 'save_intent' as parameters. It is also assumed that all features have a feature contract to
        save the feature outcome to

        :param canonical: this is the iterative value all intent are applied to and returned.
        :param feature_name: feature to run
        :param seed: (optional) if shuffle is True a seed value for the choice
        :param shuffle: (optional) Whether or not to shuffle the data before splitting or just split on train size.
        :param kwargs: additional kwargs to add to the parameterised intent, these will replace any that already exist
        :return
        """
        # test if there is any intent to run
        if self._pm.has_intent(level=feature_name):
            canonical = self._get_canonical(canonical)
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

    def select_regression_coefficient(self, canonical: Any, target: Any, seed: int=None, train_size: float=None,
                                      save_intent: bool=None, feature_name: [int, str]=None, intent_order: int=None,
                                      replace_intent: bool=None, remove_duplicates: bool=None, **kwargs):
        """ Uses a Random Forest regressor tree to measure the magnitude of coefficient against a target. It tries to 
        predict which features have the most relevance to the target. This is an embedded method.

        Random forests is one the most popular machine learning algorithms. It is so successful because it provides
        good predictive performance, low overfitting and easy interpretability. This interpretability is given by the
        fact that it is straightforward to derive the importance of each variable on the tree decision. In other words,
        it is easy to compute how much each variable is contributing to the decision.

        In general, features that are selected at the top of the trees are more important than features that are
        selected at the end nodes of the trees, as generally the top splits lead to bigger information gains.

        It should be considered:
        - Random Forests and decision trees in general give preference to features with high cardinality
        - Correlated features will be given equal or similar importance, but overall reduced importance compared to
          the same tree built without correlated counterparts.

        :param canonical: the Pandas.DataFrame to get the selection from
        :param target: the key column to index on
        :param seed: (optional) a seed value for the random function: default to None
        :param train_size: (optional) The size of the training data subset to avoid over-fitting set between 0 and 1.
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
        :param kwargs: (optional) additional arguments to pass to the model
        :return: selected list of headers indexed on key
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   feature_name=feature_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        canonical = self._get_canonical(canonical)
        df_numbers = Commons.filter_columns(canonical, dtype=['int', 'float']).dropna()
        seed = seed if isinstance(seed, int) else self._seed()
        params = {}
        # add method kwargs to the params
        if isinstance(kwargs, dict):
            params.update(kwargs)
        n_estimators = params.pop('n_estimators', 10)
        seed = params.pop('random_state', seed)
        train_size = train_size if isinstance(train_size, float) and 0 < train_size < 1 else 0.3
        # to avoid over-fitting
        X_train, _, y_train, _ = train_test_split(df_numbers.drop(labels=[target], axis=1), df_numbers[target],
                                                  test_size=1-train_size, random_state=seed)
        scaler = StandardScaler()
        scaler.fit(X_train)
        sel_ = SelectFromModel(RandomForestRegressor(random_state=seed, **kwargs))
        sel_.fit(scaler.transform(X_train), y_train)
        selected_feat = X_train.columns[(sel_.get_support())]
        return Commons.filter_columns(canonical, headers=selected_feat.to_list() + [target])

    def select_linear_coefficient(self, canonical: Any, target: Any, penalty: float=None, seed: int=None,
                                  train_size: float=None, save_intent: bool=None, feature_name: [int, str]=None,
                                  intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None):
        """ Uses linear regression model to measure the magnitude of coefficient against a target. It tries to predict
        which features have the most relevance to the target. This is an embedded method

        Regularisation consists in adding a penalty to the different parameters of the machine learning model to reduce
        the freedom of the model and avoid overfitting. In linear model regularization, the penalty is applied to the
        coefficients that multiply each of the predictors. The Lasso regularization or l1 has the property that is able
        to shrink some coefficients to zero. Therefore, those features can be removed from the model.

        Linear Regression makes the following assumptions:
        - There is a linear relationship between the predictors Xs and the outcome Y
        - The residuals follow a normal distribution centered at 0
        - There is little or no multicollinearity among predictors (Xs should not be linearly related to one another)
        - Homoscedasticity or homogeneity of variance where the variance should be the same
        in addition the result may be penalised by regularisation so setting this value high reduces its influence

        :param canonical: the Pandas.DataFrame to get the selection from
        :param target: the key column to index on
        :param penalty: (optional) penalty against the coefficient, large values less regularisation
        :param seed: (optional) a seed value for the random function: default to None
        :param train_size: (optional) The size of the training data subset to avoid over-fitting set between 0 and 1.
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
        df_numbers = Commons.filter_columns(canonical, dtype=['int', 'float']).dropna()
        seed = seed if isinstance(seed, int) else self._seed()
        penalty = penalty if isinstance(penalty, (int, float)) else 0.001
        train_size = train_size if isinstance(train_size, float) and 0 < train_size < 1 else 0.3
        # to avoid over-fitting
        X_train, _, y_train, _ = train_test_split(df_numbers.drop(labels=[target], axis=1), df_numbers[target],
                                                  test_size=1-train_size, random_state=seed)
        scaler = StandardScaler()
        scaler.fit(X_train)
        sel_ = SelectFromModel(Lasso(alpha=penalty, random_state=seed))
        sel_.fit(scaler.transform(X_train), y_train)
        selected_feat = X_train.columns[(sel_.get_support())]
        return Commons.filter_columns(canonical, headers=selected_feat.to_list() + [target])

    def select_classifier_coefficient(self, canonical: Any, target: Any, seed: int=None, train_size: float=None,
                                      save_intent: bool=None, feature_name: [int, str]=None, intent_order: int=None,
                                      replace_intent: bool=None, remove_duplicates: bool=None, **kwargs):
        """ Uses Random Forest classifier tree to measure the magnitude of coefficient against a target.It tries to
        predict which feature has the most relevance to the target. This is an embedded method

        Random forests is one the most popular machine learning algorithms. It is so successful because it provides
        good predictive performance, low overfitting and easy interpretability. This interpretability is given by the
        fact that it is straightforward to derive the importance of each variable on the tree decision. In other words,
        it is easy to compute how much each variable is contributing to the decision.

        In general, features that are selected at the top of the trees are more important than features that are
        selected at the end nodes of the trees, as generally the top splits lead to bigger information gains.

        It should be considered:
        - Random Forests and decision trees in general give preference to features with high cardinality
        - Correlated features will be given equal or similar importance, but overall reduced importance compared to
          the same tree built without correlated counterparts.

        :param canonical: the Pandas.DataFrame to get the selection from
        :param target: the key column to index on
        :param seed: (optional) a seed value for the random function: default to None
        :param train_size: (optional) The size of the training data subset to avoid over-fitting set between 0 and 1.
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
        :param kwargs: (optional) additional arguments to pass to the model
        :return: selected list of headers indexed on key
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   feature_name=feature_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        canonical = self._get_canonical(canonical)
        seed = seed if isinstance(seed, int) else self._seed()
        train_size = train_size if isinstance(train_size, float) and 0 < train_size < 1 else 0.3
        params = {}
        # add method kwargs to the params
        if isinstance(kwargs, dict):
            params.update(kwargs)
        n_estimators = params.pop('n_estimators', 10)
        seed = params.pop('random_state', seed)
       # to avoid over-fitting
        X_train, _, y_train, _ = train_test_split(canonical.drop(labels=[target], axis=1), canonical[target],
                                                  test_size=1-train_size, random_state=seed)
        scaler = StandardScaler()
        scaler.fit(X_train)
        sel_ = SelectFromModel(RandomForestClassifier(n_estimators=n_estimators, random_state=seed, **params))
        sel_.fit(scaler.transform(X_train), y_train)
        selected_feat = X_train.columns[(sel_.get_support())]
        return Commons.filter_columns(canonical, headers=selected_feat.to_list() + [target])

    def select_logistic_coefficient(self, canonical: Any, target: Any, regularisation: float=None, seed: int=None,
                                    train_size: float=None, save_intent: bool=None, feature_name: [int, str]=None,
                                    intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None):
        """ Uses logistic regression model to measure the magnitude of coefficient against a target.It tries to predict
        which feature has the most relevance to the target. This is an embedded method

        Regularisation consists in adding a penalty to the different parameters of the machine learning model to reduce
        the freedom of the model and avoid overfitting. In linear model regularization, the penalty is applied to the
        coefficients that multiply each of the predictors.

        :param canonical: the Pandas.DataFrame to get the selection from
        :param target: the key column to index on
        :param regularisation: (optional) regularisation against the coefficient, large values less regularisation
        :param seed: (optional) a seed value for the random function: default to None
        :param train_size: (optional) The size of the training data subset to avoid over-fitting set between 0 and 1.
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
        regularisation = regularisation if isinstance(regularisation, (int, float)) else 0.1
        train_size = train_size if isinstance(train_size, float) and 0 < train_size < 1 else 0.3
        # to avoid over-fitting
        X_train, _, y_train, _ = train_test_split(canonical.drop(labels=[target], axis=1), canonical[target],
                                                  test_size=1-train_size, random_state=seed)
        scaler = StandardScaler()
        scaler.fit(X_train)
        sel_ = SelectFromModel(LogisticRegression(C=regularisation, random_state=seed, solver='liblinear',
                                                  penalty='l1'))
        sel_.fit(scaler.transform(X_train), y_train)
        selected_feat = X_train.columns[(sel_.get_support())]
        return Commons.filter_columns(canonical, headers=selected_feat.to_list() + [target])

    def select_classifier_shuffled(self, canonical: Any, target: Any, threshold: int=None, seed: int=None,
                                   train_size: float=None, save_intent: bool=None, feature_name: [int, str]=None,
                                   intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None,
                                   **kwargs):
        """ Uses Random Forest classifier tree to measure the magnitude of coefficient against a target.It tries to
        predict which feature has the most relevance to the target. This is an embedded method

        Random forests is one the most popular machine learning algorithms. It is so successful because it provides
        good predictive performance, low overfitting and easy interpretability. This interpretability is given by the
        fact that it is straightforward to derive the importance of each variable on the tree decision. In other words,
        it is easy to compute how much each variable is contributing to the decision.

        A popular method of feature selection consists of randomly shuffling the values of a specific variable and
        determining how that permutation affects the performance metric of the machine learning algorithm. In other
        words, the idea is to shuffle the values of each feature, one feature at a time, and measure how much the
        permutation (or shuffling of its values) decreases the accuracy, or the roc_auc, or the mse of the machine
        learning model (or any other performance metric!). If the variables are important, a random permutation of
        their values will dramatically decrease any of these metrics. Contrarily, the permutation or shuffling of
        values should have little to no effect on the model performance metric we are assessing.

        :param canonical: the Pandas.DataFrame to get the selection from
        :param target: the key column to index on
        :param threshold:
        :param seed: (optional) a seed value for the random function: default to None
        :param train_size: (optional) The size of the training data subset to avoid over-fitting set between 0 and 1.
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
        :param kwargs: (optional) additional arguments to pass to the model
        :return: selected list of headers indexed on key
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   feature_name=feature_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        canonical = self._get_canonical(canonical)
        seed = seed if isinstance(seed, int) else self._seed()
        threshold = threshold if isinstance(threshold, int) else 0
        train_size = train_size if isinstance(train_size, float) and 0 < train_size < 1 else 0.3
        params = {}
        # add method kwargs to the params
        if isinstance(kwargs, dict):
            params.update(kwargs)
        n_estimators = params.pop('n_estimators', 10)
        seed = params.pop('random_state', seed)
       # to avoid over-fitting
        X_train, _, y_train, _ = train_test_split(canonical.drop(labels=[target], axis=1), canonical[target],
                                                  test_size=1-train_size, random_state=seed)
        X_train.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        scaler = StandardScaler()
        scaler.fit(X_train)
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=2, n_jobs=4, random_state=seed, **params)
        rf.fit(X_train, y_train)
        train_roc = roc_auc_score(y_train, (rf.predict_proba(X_train))[:, 1])
        performance_shift = []
        for feature in X_train.columns:
            X_train_c = X_train.copy()
            # shuffle individual feature
            X_train_c[feature] = X_train_c[feature].sample(frac=1, random_state=seed).reset_index(drop=True)
            # make prediction with shuffled feature and calculate roc-auc
            shuff_roc = roc_auc_score(y_train, rf.predict_proba(X_train_c)[:, 1])
            drift = train_roc - shuff_roc
            performance_shift.append(drift)
        feature_importance = pd.Series(performance_shift)
        feature_importance.index = X_train.columns
        feature_importance.sort_values(ascending=False)
        feature_importance = feature_importance[feature_importance > threshold]
        selected_feat = feature_importance.index
        return Commons.filter_columns(canonical, headers=selected_feat.to_list() + [target])

    def select_regressor_shuffled(self, canonical: Any, target: Any, threshold: int=None, seed: int=None,
                                  train_size: float=None, save_intent: bool=None, feature_name: [int, str]=None,
                                  intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None,
                                  **kwargs):
        """ Uses Random Forest regressor tree to measure the magnitude of coefficient against a target.It tries to
        predict which feature has the most relevance to the target. This is an embedded method

        Random forests is one the most popular machine learning algorithms. It is so successful because it provides
        good predictive performance, low overfitting and easy interpretability. This interpretability is given by the
        fact that it is straightforward to derive the importance of each variable on the tree decision. In other words,
        it is easy to compute how much each variable is contributing to the decision.

        A popular method of feature selection consists of randomly shuffling the values of a specific variable and
        determining how that permutation affects the performance metric of the machine learning algorithm. In other
        words, the idea is to shuffle the values of each feature, one feature at a time, and measure how much the
        permutation (or shuffling of its values) decreases the accuracy, or the roc_auc, or the mse of the machine
        learning model (or any other performance metric!). If the variables are important, a random permutation of
        their values will dramatically decrease any of these metrics. Contrarily, the permutation or shuffling of
        values should have little to no effect on the model performance metric we are assessing.

        :param canonical: the Pandas.DataFrame to get the selection from
        :param target: the key column to index on
        :param threshold:
        :param seed: (optional) a seed value for the random function: default to None
        :param train_size: (optional) The size of the training data subset to avoid over-fitting set between 0 and 1.
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
        :param kwargs: (optional) additional arguments to pass to the model
        :return: selected list of headers indexed on key
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   feature_name=feature_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        canonical = self._get_canonical(canonical)
        seed = seed if isinstance(seed, int) else self._seed()
        threshold = threshold if isinstance(threshold, int) else 0
        train_size = train_size if isinstance(train_size, float) and 0 < train_size < 1 else 0.3
        params = {}
        # add method kwargs to the params
        if isinstance(kwargs, dict):
            params.update(kwargs)
        n_estimators = params.pop('n_estimators', 10)
        seed = params.pop('random_state', seed)
        # to avoid over-fitting
        X_train, _, y_train, _ = train_test_split(canonical.drop(labels=[target], axis=1), canonical[target],
                                                  test_size=1 - train_size, random_state=seed)
        X_train.reset_index(drop=True, inplace=True)
        scaler = StandardScaler()
        scaler.fit(X_train)
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=2, n_jobs=4, random_state=seed, **params)
        rf.fit(X_train, y_train)
        train_rmse = mean_squared_error(y_train, rf.predict(X_train), squared=False)
        performance_shift = []
        for feature in X_train.columns:
            X_train_c = X_train.copy()
            # shuffle individual feature
            X_train_c[feature] = X_train_c[feature].sample(frac=1, random_state=seed).reset_index(drop=True)
            # make prediction with shuffled feature and calculate rmse
            shuff_rmse = mean_squared_error(y_train, rf.predict(X_train_c), squared=False)
            drift = train_rmse - shuff_rmse
            performance_shift.append(drift)
        feature_importance = pd.Series(performance_shift)
        feature_importance.index = X_train.columns
        feature_importance.sort_values(ascending=False)
        feature_importance = feature_importance[feature_importance < threshold]
        selected_feat = feature_importance.index
        return Commons.filter_columns(canonical, headers=selected_feat.to_list() + [target])

    def select_classifier_elimination(self, canonical: Any, target: Any, threshold: int=None, seed: int=None,
                                      train_size: float=None, save_intent: bool=None, feature_name: [int, str]=None,
                                      intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None,
                                  **kwargs):
        """ Uses Gradient Boosting Classifier to recursively eliminate features based on a target.It tries to
        predict which feature has the most relevance to the target. This is an hybrid method.

        Dependent on the number of features this may take some time.

        :param canonical: the Pandas.DataFrame to get the selection from
        :param target: the key column to index on
        :param threshold:
        :param seed: (optional) a seed value for the random function: default to None
        :param train_size: (optional) The size of the training data subset to avoid over-fitting set between 0 and 1.
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
        :param kwargs: (optional) additional arguments to pass to the model
        :return: selected list of headers indexed on key
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   feature_name=feature_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        canonical = self._get_canonical(canonical)
        seed = seed if isinstance(seed, int) else self._seed()
        threshold = threshold if isinstance(threshold, int) else 0.0005
        train_size = train_size if isinstance(train_size, float) and 0 < train_size < 1 else 0.3
        params = {}
        # add method kwargs to the params
        if isinstance(kwargs, dict):
            params.update(kwargs)
        seed = params.pop('random_state', seed)
        # to avoid over-fitting
        X_train, X_test, y_train, y_test = train_test_split(canonical.drop(labels=[target], axis=1), canonical[target],
                                                  test_size=1 - train_size, random_state=seed)
        # build initial model using all the features
        model_full = GradientBoostingClassifier(n_estimators=10, max_depth=4, random_state=seed)
        model_full.fit(X_train, y_train)
        y_pred_test = model_full.predict_proba(X_test)[:, 1]
        roc_full = roc_auc_score(y_test, y_pred_test)
        features = pd.Series(model_full.feature_importances_)
        features.index = X_train.columns
        features = features.index.to_list()
        features_to_remove = []
        for feature in features:
            # initialise model
            model_int = GradientBoostingClassifier(n_estimators=10, max_depth=4, random_state=seed)
            # fit model with all variables, minus the feature to be evaluated
            # and also minus all features that were deemed to be removed
            model_int.fit(X_train.drop(features_to_remove + [feature], axis=1), y_train)
            # make a prediction using the test set
            y_pred_test = model_int.predict_proba(X_test.drop(features_to_remove + [feature], axis=1))[:, 1]
            # calculate the new roc-auc
            roc_int = roc_auc_score(y_test, y_pred_test)
            # determine the drop in the roc-auc
            diff_roc = roc_full - roc_int
            # compare the drop in roc-auc with the tolerance we set previously
            if diff_roc < threshold:
                # if the drop in the roc is small and we remove the
                # feature, we need to set the new roc to the one based on
                # the remaining features
                roc_full = roc_int
                # and append the feature to remove to the collecting list
                features_to_remove.append(feature)
        selected_feat = [x for x in features if x not in features_to_remove]
        return Commons.filter_columns(canonical, headers=selected_feat + [target])

    def select_regressor_elimination(self, canonical: Any, target: Any, threshold: int=None, seed: int=None,
                                     train_size: float=None, save_intent: bool=None, feature_name: [int, str]=None,
                                     intent_order: int=None, replace_intent: bool=None, remove_duplicates: bool=None,
                                     **kwargs):
        """ Uses Gradient Boosting Regressor to recursively eliminate features based on a target.It tries to
        predict which feature has the most relevance to the target. This is an hybrid method.

        Dependent on the number of features this may take some time.

        :param canonical: the Pandas.DataFrame to get the selection from
        :param target: the key column to index on
        :param threshold:
        :param seed: (optional) a seed value for the random function: default to None
        :param train_size: (optional) The size of the training data subset to avoid over-fitting set between 0 and 1.
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
        :param kwargs: (optional) additional arguments to pass to the model
        :return: selected list of headers indexed on key
        """
        # resolve intent persist options
        self._set_intend_signature(self._intent_builder(method=inspect.currentframe().f_code.co_name, params=locals()),
                                   feature_name=feature_name, intent_order=intent_order, replace_intent=replace_intent,
                                   remove_duplicates=remove_duplicates, save_intent=save_intent)
        # Code block for intent
        canonical = self._get_canonical(canonical)
        seed = seed if isinstance(seed, int) else self._seed()
        threshold = threshold if isinstance(threshold, int) else 0.0005
        train_size = train_size if isinstance(train_size, float) and 0 < train_size < 1 else 0.3
        params = {}
        # add method kwargs to the params
        if isinstance(kwargs, dict):
            params.update(kwargs)
        seed = params.pop('random_state', seed)
        # to avoid over-fitting
        X_train, X_test, y_train, y_test = train_test_split(canonical.drop(labels=[target], axis=1), canonical[target],
                                                  test_size=1 - train_size, random_state=seed)
        # build initial model using all the features
        model_full = GradientBoostingRegressor(n_estimators=10, max_depth=4, random_state=seed)
        model_full.fit(X_train, y_train)
        # calculate r2 in the test set
        y_pred_test = model_full.predict(X_test)
        r2_full = r2_score(y_test, y_pred_test)
        features = pd.Series(model_full.feature_importances_)
        features.index = X_train.columns
        features = features.index.to_list()
        features_to_remove = []
        for feature in features:
            # initialise model
            model_int = GradientBoostingClassifier(n_estimators=10, max_depth=4, random_state=seed)
            # fit model with all variables, minus the feature to be evaluated
            # and also minus all features that were deemed to be removed
            model_int.fit(X_train.drop(features_to_remove + [feature], axis=1), y_train)
            # make a prediction using the test set
            y_pred_test = model_int.predict_proba(X_test.drop(features_to_remove + [feature], axis=1))[:, 1]
            # calculate the new roc-auc
            # calculate the new r2
            r2_int = r2_score(y_test, y_pred_test)
            diff_r2 = r2_full - r2_int
            if diff_r2 < threshold:
                # if the drop in the roc is small and we remove the
                # feature, we need to set the new roc to the one based on
                # the remaining features
                r2_full = r2_int
                # and append the feature to remove to the collecting list
                features_to_remove.append(feature)
        selected_feat = [x for x in features if x not in features_to_remove]
        return Commons.filter_columns(canonical, headers=selected_feat + [target])

    def select_correlated(self, canonical: Any, target: Any, threshold: float=None, train_size: float=None,
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
