import catboost
from catboost import CatBoostClassifier, sum_models
from sklearn.model_selection import StratifiedKFold
from hyperopt import fmin, tpe, STATUS_OK, Trials
import numpy as np


class CatboostModel:

    def __init__(self, label):
        self.label = label
        self.CAT_COLUMNS = []
        self.model = None

    def get_trained_model(self):
        return self.model

    def train(
            self,
            train_df,
            train_y_df,
            test_df,
            test_y_df,
            cat_columns,
            params=None) -> dict:
        self.CAT_COLUMNS = cat_columns

        default_params = {
            'iterations': 3000,
            'depth': 5,
            'loss_function': 'Logloss',
            'logging_level': None,
            'random_strength': 2.0,
            'l2_leaf_reg': 3,
            'bagging_temperature': 0.244,
            'eval_metric': 'AUC',
            'scale_pos_weight': 75,
            'use_best_model': True,
            'early_stopping_rounds': 300
        }

        if params is None:
            params = default_params
        else:
            for key in default_params:
                if key not in params:
                    params[key] = default_params[key]

        model = CatBoostClassifier(**params)
        cat_cols = list(set(self.CAT_COLUMNS) & set(train_df.columns))
        model.fit(train_df, train_y_df, cat_features=cat_cols,
                  eval_set=[(test_df, test_y_df)])

        self.model = model

        train_metrics = predict_and_score(model, train_df, train_y_df)
        test_metrics = predict_and_score(model, test_df, test_y_df)

        ret = {
            "train": train_metrics,
            "test": test_metrics,
            "model": save_model_pickle(model, list(train_df.columns))
        }

        return ret

    def predict(self, data_df, pkl, prediction_type=None):

        dct = load_model_pickle(pkl)
        if dct:
            model = dct['model']
            columns = dct['columns']
            clean_data = reformat_dataframe(data_df, columns)
            if prediction_type is None:
                preds = model.predict_proba(clean_data)
            else:
                preds = model.predict(clean_data, prediction_type=prediction_type)
            return preds
        else:
            raise Exception("Could not read model from provided pickle.")

    def _get_objective(self, data_df, num_rounds=200, folds=5):

        skf = StratifiedKFold(n_splits=folds)
        y_df = data_df[self.label]
        data_df = data_df.drop(columns=[self.label])
        cat_cols = list(set(self.CAT_COLUMNS) & set(data_df.columns))

        indexes = list(skf.split(data_df.index, y_df))

        def objective(space):

            params = {
                'iterations': num_rounds,
                'depth': int(space['depth']),
                'loss_function': 'Logloss',
                'verbose': False,
                'random_strength': space['random_strength'],
                'l2_leaf_reg': space['l2_leaf_reg'],
                'bagging_temperature': space['bagging_temp'],
                'eval_metric': 'Logloss',
                'scale_pos_weight': space['scale_pos_weight'],
                'use_best_model': True,
                'early_stopping_rounds': 20
            }

            train_auc = []
            train_f1 = []
            test_auc = []
            test_f1 = []
            for index in indexes:

                train_df = data_df.loc[index[0]]
                train_y_df = y_df.loc[index[0]]
                test_df = data_df.loc[index[1]]
                test_y_df = y_df.loc[index[1]]

                model = CatBoostClassifier(**params)
                model.fit(
                    train_df, train_y_df, cat_features=cat_cols, eval_set=[
                        (test_df, test_y_df)])

                train_metrics = predict_and_score(model, train_df, train_y_df)
                train_auc.append(train_metrics[0])
                train_f1.append(train_metrics[1])

                test_metrics = predict_and_score(model, test_df, test_y_df)
                test_auc.append(test_metrics[0])
                test_f1.append(test_metrics[1])

            ret = {
                'status': STATUS_OK,
                'loss': 1 - np.mean(test_auc),
                'AUC': np.mean(test_auc),
                'F1': np.mean(test_f1),
                'AUC-train': np.mean(train_auc),
                'F1-train': np.mean(train_f1)
            }
            return ret

        return objective

    def optimize_hyperparameters(
            self,
            data_df,
            space,
            cat_columns,
            max_evals=100,
            num_rounds=200,
            folds=5,
            algo=tpe.suggest):

        self.CAT_COLUMNS = cat_columns

        trials = Trials()
        best = fmin(
            fn=self._get_objective(
                data_df,
                num_rounds=num_rounds,
                folds=folds),
            space=space,
            algo=algo,
            max_evals=max_evals,
            trials=trials)

        return best, trials

    def retrain(self, model_bytes_arr, train_df, train_y_df, test_df, test_y_df):
        models = []
        for model_bytes in model_bytes_arr:
           models.append(load_model_pickle(model_bytes)['model'])

        model = sum_models(models)
        train_metrics = predict_and_score(model, train_df, train_y_df, 'Probability')
        test_metrics = predict_and_score(model, test_df, test_y_df, 'Probability')

        ret = {
            "train": train_metrics,
            "test": test_metrics,
            "model": save_model_pickle(model, list(train_df.columns))
        }

        return ret

    def is_of_type_cb_classifier(self, model_obj):
        return isinstance(model_obj, catboost.core.CatBoostClassifier)
