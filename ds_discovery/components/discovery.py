import itertools
import math
import os
import random
import time
from builtins import staticmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
from numpy.polynomial.polynomial import Polynomial
from matplotlib.colors import LogNorm
from scipy.stats import shapiro, normaltest, anderson
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from aistac.handlers.abstract_handlers import HandlerFactory
from ds_discovery.components.commons import Commons

__author__ = 'Darryl Oatridge'


class Visualisation(object):
    """ a set of data components methods to Visualise pandas.Dataframe"""

    @staticmethod
    def show_fisher_score(df, target: [str, list]):
        """displays the f-score for all applicable columns."""
        target = Commons.list_formatter(target)
        for name in target:
            if name not in df.columns:
                raise ValueError(f"The target header '{name}' was not found in the DataFrame")
        df_numbers = Commons.filter_columns(df, dtype=['number'], exclude=False)
        for col in Commons.filter_columns(df, dtype=['category'], exclude=False):
            df_numbers[col] = df[col].cat.codes
        df_headers = Commons.filter_columns(df_numbers, headers=target, drop=True).fillna(0).abs()
        if df_headers.shape[1] == 0:
            raise ValueError(f"There were no suitable columns to apply the score too")
        df_target = Commons.filter_columns(df, headers=target, drop=False)
        f_score = chi2(df_headers, df_target)
        pvalues = pd.Series(f_score[1])
        pvalues.index = df_headers.columns
        _ = pvalues[pvalues.values > 0].sort_values(ascending=True).plot.bar(figsize=(20, 8))
        plt.tight_layout()
        plt.show()
        plt.clf()

    @staticmethod
    def show_ecdf(df: pd.DataFrame, header: str, x_label: str=None):
        """Empirical cumulative distribution function"""
        x = np.sort(df[header])
        sns.set()
        y = np.arange(1, len(x) + 1) / len(x)
        _ = plt.plot(x, y, marker='.', linestyle='none')
        _ = plt.xlabel(x_label if isinstance(x_label, str) else header)
        _ = plt.ylabel('ECDF')
        plt.margins(0.02)  # Keeps data off plot edges
        plt.tight_layout()
        plt.show()
        plt.clf()

    @staticmethod
    def show_bootstrap_confidence_interval(df: pd.DataFrame, header: str, p_percent: float=None, replicates: int=None,
                                           func: Any=None, x_label: str=None):
        """ displays a Bootstrap Confidence Interval as a histogram with the 95% confidence interval shaded

        :param df: the DataFrame to visualise
        :param header: the header from the DataFrame to bootstrap
        :param p_percent: the p percent confidence interval. Default t0 0.95 for 95% confidence interval
        :param replicates: the number of replicates to run. default 10,000
        :param func: the function that computes the statistic of interest, np.mean, np.mode etc. Default np.mean
        :param x_label: a label to give to the x axis. Default to header name
        :return:
        """
        p_percent = p_percent if isinstance(p_percent, float) and 0 < p_percent < 1 else 0.95
        replicates = replicates if isinstance(replicates, int) else 10000
        func = np.mean if func is None else func
        result = []
        for i in range(replicates):
            bs_sample = np.random.choice(df[header].dropna(), len(df[header].dropna()))
            result.append(func(bs_sample))
        low_percentile = round((1 - p_percent) / 2, 5)
        high_percentile = 1 - low_percentile
        low_interval, high_interval = tuple(np.percentile(result, [low_percentile, high_percentile]))
        sns.set()
        plt.rcParams['figure.figsize'] = (8, 5)
        _ = plt.hist(result, bins=35, normed=True)
        _ = plt.xlabel(x_label if isinstance(x_label, str) else header)
        _ = plt.ylabel('PDF')
        _ = plt.axvspan(low_interval, high_interval, alpha=0.2)
        _ = plt.axvline(x=low_interval)
        _ = plt.axvline(x=high_interval)
        plt.show()

    @staticmethod
    def show_num_density(df: pd.DataFrame, category: bool=False, filename: str=None, **kwargs):
        """ shows the number density across all the numberic attributes

        :param df: the dataframe to visualise
        :param category: if True then convert Categoricals to numeric codes and include
        :param filename: and optional name of a file to print to
        :param kwargs:
        """
        selection = ['number', 'category'] if category else 'number'
        num_cols = Commons.filter_headers(df, dtype=selection)
        if len(num_cols) > 0:
            depth = int(round(len(num_cols) / 2, 0) + len(num_cols) % 2)
            _figsize = (20, 5 * depth)
            fig = plt.figure(figsize=_figsize)

            right = False
            line = 0
            for c in num_cols:
                col = df[c]
                if col.dtype.name == 'category':
                    col = col.cat.codes
                #     print("{}, {}, {}, {}".format(c, depth, line, right))
                ax = plt.subplot2grid((depth, 2), (line, int(right)))
                g = col.dropna().plot.kde(ax=ax, title=str.title(c), **kwargs)
                g.get_xaxis().tick_bottom()
                g.get_yaxis().tick_left()

                if right:
                    line += 1
                right = not right
            plt.tight_layout()

            if filename is None:
                plt.show()
            else:
                fig.savefig(filename, dpi=300)
            plt.clf()
        else:
            raise LookupError("No numeric columns found in the dataframe")

    @staticmethod
    def show_cat_count(df, category=None, top=None, filename=None):
        if isinstance(category, (str, list)):
            cat_headers = Commons.filter_headers(df, headers=category)
        else:
            cat_headers = Commons.filter_headers(df, dtype=['category'])
        if len(cat_headers) > 0:
            wide_col, thin_col = [], []
            for c in cat_headers:
                if len(df[c].cat.categories) > 10:
                    wide_col += [c]
                else:
                    thin_col += [c]
            depth = len(wide_col) + int(round(len(thin_col) / 2, 0))

            _figsize = (20, 5 * depth)
            fig = plt.figure(figsize=_figsize)
            sns.set(style='darkgrid', color_codes=True)

            for c, i in zip(wide_col, range(len(wide_col))):
                ax = plt.subplot2grid((depth, 2), (i, 0), colspan=2)
                order = list(df[c].value_counts().index.values)
                if isinstance(top, int):
                    order = order[:top]
                _ = sns.countplot(x=c, data=df, ax=ax, order=order, palette="summer")
                _ = plt.xticks(rotation=-90)
                _ = plt.xlabel(str.title(c))
                _ = plt.ylabel('Count')
                title = "{} Categories".format(str.title(c))
                _ = plt.title(title, fontsize=16)

            right = False
            line = len(wide_col)
            for c in thin_col:
                ax = plt.subplot2grid((depth, 2), (line, int(right)))
                order = list(df[c].value_counts().index.values)
                _ = sns.countplot(x=c, data=df, ax=ax, order=order, palette="summer")
                _ = plt.xticks(rotation=-90)
                _ = plt.xlabel(str.title(c))
                _ = plt.ylabel('Count')
                _ = title = "{} Categories".format(str.title(c))
                _ = plt.title(title, fontsize=16)
                if right:
                    line += 1
                right = not right

            plt.tight_layout()

            if filename is None:
                plt.show()
            else:
                fig.savefig(filename, dpi=300)
            plt.clf()
        else:
            raise LookupError("No category columns found in the dataframe")

    @staticmethod
    def show_corr(df, filename=None, figsize=None, **kwargs):
        if figsize is None or not isinstance(figsize, tuple):
            _figsize = (12, 4)
        else:
            _figsize = figsize
        fig = plt.figure(figsize=_figsize)
        sns.heatmap(df.corr(), annot=True, cmap='BuGn', robust=True, **kwargs)
        if filename is None:
            plt.show()
        else:
            fig.savefig(filename)
        plt.clf()

    @staticmethod
    def show_missing(df, filename=None, figsize=None, **kwargs):
        if figsize is None or not isinstance(figsize, tuple):
            _figsize = (12, 4)
        else:
            _figsize = figsize
        fig = plt.figure(figsize=_figsize)
        sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis', **kwargs)
        if filename is None:
            plt.show()
        else:
            fig.savefig(filename)
        plt.clf()

    @staticmethod
    def show_cat_time_index(df, col_index, category=None, col_exclude=None, filename=None, logscale=False,
                            subplot_h=2, subplot_w=15, param_scale=8, rotation=360, hspace=0.35):
        """ creates the frequencies (colors of heatmap) of the elements (y axis) of the categorical columns
        over time (x axis)

        :param df: the data frame
        :param col_index: the names of the column with the date
        :param category: the name of the column to show or the list of columns
        :param col_exclude: the name of the column to exclude or the list of columns to exclude
        :param filename: output file name
        :param logscale: bool, apply log10 transform?
        :param subplot_h: the height of the block which corresponds to showing 'param_scale'
                        categories on the y axis of the heatmap
        :param subplot_w: the width of the figure
        :param param_scale: the parameter which controls the height of a single subplot
        :param rotation: rotation of the y-axis labels
        :param hspace: horizontal space between subplots
        """
        dates = pd.date_range(start=df[col_index].min(), end=df[col_index].max())
        if not isinstance(col_exclude, (np.ndarray, list)):
            col_exclude = [col_exclude]
        if category is None:
            col_names = Commons.filter_headers(df, dtype=['category'], headers=col_exclude, drop=True)
        else:
            col_names = category
            if not isinstance(col_names, (np.ndarray, list)):
                col_names = [col_names]
        n_categories = len(col_names)
        cbar_kws = {'orientation': 'horizontal', 'shrink': 0.5}
        n_subplot_rows = np.ceil(df[col_names].nunique(dropna=True).divide(param_scale))
        n_subplot_rows[-1] += 1
        n_rows = int(n_subplot_rows.sum())
        grid_weights = {'height_ratios': n_subplot_rows.values}
        cmap = 'rocket_r'
        # cmap = sns.cm.rocket_r
        fig, axes = plt.subplots(n_categories, 1, gridspec_kw=grid_weights, sharex='col',
                                 figsize=(subplot_w, n_rows * subplot_h))
        if n_categories == 1:
            axes = [axes]
        for ii in range(n_categories):
            cc = col_names[ii]
            df_single_cat = df[[col_index, cc]]
            df_single_cat = df_single_cat.loc[df_single_cat[col_index].notnull(), ]
            df_single_cat['Index'] = df_single_cat[col_index].dt.date
            df_pivot = df_single_cat.pivot_table(index='Index', columns=cc, aggfunc=len, dropna=True)
            df_pivot.index = pd.to_datetime(df_pivot.index)
            toplot = df_pivot.reindex(dates.date).T

            v_min = toplot.min().min()
            v_max = toplot.max().max()
            toplot.reset_index(level=0, drop=True, inplace=True)
            if logscale:
                cbar_ticks = [math.pow(10, i) for i in range(int(math.floor(math.log10(v_min))),
                                                             int(1 + math.ceil(math.log10(v_max))))]
                log_norm = LogNorm(vmin=v_min, vmax=v_max)
            else:
                cbar_ticks = list(range(int(v_min), int(v_max + 1)))
                if len(cbar_ticks) > 5:
                    v_step = int(math.ceil((v_max - v_min) / 4))
                    cbar_ticks = list(range(int(v_min), int(v_max + 1), v_step))
                log_norm = None
            cbar_kws['ticks'] = cbar_ticks
            if ii < (n_categories - 1):
                cbar_kws['pad'] = 0.05
            else:
                cbar_kws['pad'] = 0.25
            sns.heatmap(toplot, cmap=cmap, ax=axes[ii], norm=log_norm, cbar_kws=cbar_kws, yticklabels=True)
            axes[ii].set_ylabel('')
            axes[ii].set_xlabel('')
            axes[ii].set_title(cc)
            axes[ii].set_yticklabels(axes[ii].get_yticklabels(), rotation=rotation)
            for _, spine in axes[ii].spines.items():
                spine.set_visible(True)
        axes[-1].set_xlabel(col_index)
        plt.subplots_adjust(bottom=0.05, hspace=hspace)
        if filename is None:
            plt.show()
        else:
            fig.savefig(filename)
        plt.clf()
        return

    @staticmethod
    def show_percent_cat_time_index(df, col_index, category=None, col_exclude=None, filename=None, subplot_h=6,
                                    subplot_w=10, rotation=360):
        """ creates the proportion (as percentages) (colors of heatmap) of the apearing elements (y axis)
        of the categorical columns over time (x axis)

        :param df: the data frame
        :param col_index: the names of the column with the date
        :param category: the name of the column to show or the list of columns
        :param col_exclude: the name of the column to exclude or the list of columns to exclude
        :param filename: output file name
        :param subplot_h: the height of the figure
        :param subplot_w: the width of the figure
        :param subplot_w: the width of the figure
        :param rotation: rotation of the y-axis labels
        """
        dates = pd.date_range(start=df[col_index].min(), end=df[col_index].max())
        if not isinstance(col_exclude, (np.ndarray, list)):
            col_exclude = [col_exclude]
        if category is None:
            col_names = Commons.filter_headers(df, dtype=['category'], headers=col_exclude, drop=True)
        else:
            col_names = category
            if not isinstance(col_names, (np.ndarray, list)):
                col_names = [col_names]
        cmap = 'rocket_r'
        # cmap = sns.cm.rocket_r
        df0 = df[col_names + [col_index]]
        df0['Index'] = df0[col_index].dt.date
        df_unique = df0[col_names].nunique(dropna=True)
        df_agg = df0.groupby('Index').nunique(dropna=True).drop('Index', axis=1)
        df_frac = df_agg[col_names].divide(df_unique, axis=1)
        df_frac.index = pd.to_datetime(df_frac.index)
        toplot = df_frac.reindex(dates.date).T
        new_labels = df_unique.index.values + '\n(' + pd.Series(df_unique.values).apply(str) + ')'
        fig = plt.figure(figsize=(subplot_w, subplot_h))
        ax = sns.heatmap(toplot, cmap=cmap, vmin=0, vmax=1, cbar_kws={'shrink': 0.75})
        ax.set_yticklabels(new_labels, rotation=rotation)
        ax.set_ylabel('')
        ax.set_xlabel(col_index)
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
        plt.tight_layout()
        if filename is None:
            plt.show()
        else:
            fig.savefig(filename)
        plt.clf()

    @staticmethod
    def show_roc_curve(model, train, test, train_labels, test_labels, ):
        """Compare machine learning model to baseline performance.
        Computes statistics and shows ROC curve."""
        # n_nodes = []
        # max_depths = []
        #
        # # Stats about the trees in random forest
        # for ind_tree in model.estimators_:
        #     n_nodes.append(ind_tree.tree_.node_count)
        #     max_depths.append(ind_tree.tree_.max_depth)
        #
        # print(f'Average number of nodes {int(np.mean(n_nodes))}')
        # print(f'Average maximum depth {int(np.mean(max_depths))}')

        # Training predictions (to demonstrate overfitting)
        train_predictions = model.predict(train)
        train_probs = model.predict_proba(train)[:, 1]

        # Testing predictions (to determine performance)
        predictions = model.predict(test)
        probs = model.predict_proba(test)[:, 1]

        baseline = {'recall': recall_score(test_labels, [1 for _ in range(len(test_labels))]),
                    'precision': precision_score(test_labels, [1 for _ in range(len(test_labels))]),
                    'roc': 0.5}

        results = {'recall': recall_score(test_labels, predictions),
                   'precision': precision_score(test_labels, predictions),
                   'roc': roc_auc_score(test_labels, probs)}

        train_results = {'recall': recall_score(train_labels, train_predictions),
                         'precision': precision_score(train_labels, train_predictions),
                         'roc': roc_auc_score(train_labels, train_probs)}
        print('\n')
        for metric in ['recall', 'precision', 'roc']:
            print(
                f'{metric.capitalize()} '
                f'-> Baseline: {round(baseline[metric], 2)} '
                f'Test: {round(results[metric], 2)} '
                f'Train: {round(train_results[metric], 2)}')
        print('\n')

        # Calculate false positive rates and true positive rates
        base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
        model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

        # Plot formatting
        plt.style.use('fivethirtyeight')
        plt.rcParams['font.size'] = 18

        plt.figure(figsize=(8, 6))
        plt.rcParams['font.size'] = 16

        # Plot both curves
        plt.plot(base_fpr, base_tpr, 'b', label='baseline')
        plt.plot(model_fpr, model_tpr, 'r', label='model')
        plt.legend()
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.show()
        plt.clf()

    @staticmethod
    def show_confusion_matrix(test_labels, predictions, classes, normalize=False, title=None, cmap=None):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
        """
        title = title if isinstance(title, str) else 'Confusion matrix'
        cmap = plt.cm.Oranges if cmap is None else cmap
        cm = confusion_matrix(test_labels, predictions)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        # Plot the confusion matrix
        plt.figure(figsize=(15, 15))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, size=24)
        plt.colorbar(aspect=4)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, size=14)
        plt.yticks(tick_marks, classes, size=14)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.

        # Labeling the plot
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), fontsize=20,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.grid(None)
        plt.tight_layout()
        plt.ylabel('True label', size=18)
        plt.xlabel('Predicted label', size=18)
        plt.show()


class DataDiscovery(object):
    """ a set of data components methods to view manipulate a pandas.Dataframe"""

    def __dir__(self):
        rtn_list = []
        for m in dir(DataDiscovery):
            if not m.startswith('_'):
                rtn_list.append(m)
        return rtn_list

    @staticmethod
    def interquartile_outliers(values: pd.Series, k_factor: float=None) -> Tuple[list, list]:
        """ The interquartile range (IQR), also called the midspread, middle 50%, or H‑spread, is a measure of
        statistical dispersion, being equal to the difference between 75th and 25th percentiles, or between upper
        and lower quartiles

        The IQR can be used to identify outliers by defining limits on the sample values that are a factor k of the
        IQR below the 25th percentile or above the 75th percentile. The common value for the factor k is the value 1.5.
        A factor k of 3 or more can be used to identify values that are extreme outliers.

        :param values: the values to be assessed
        :param k_factor: the k factor to apply beyond the 25th and 75th percentile, defaults to 1.5
        :return: a tuple of the lower and upper outliers as a list of index of the values
        """
        k_factor = k_factor if isinstance(k_factor, float) and 0 < k_factor <= 10 else 1.5
        values = values.copy().dropna()
        # calculate interquartile range
        q25, q75 = np.percentile(values, 25), np.percentile(values, 75)
        iqr = q75 - q25
        # calculate the outlier cutoff
        cut_off = iqr * k_factor
        lower, upper = q25 - cut_off, q75 + cut_off
        # identify outliers
        lower_outliers = values.loc[values.apply(lambda x: x < lower)].index.to_list()
        upper_outliers = values.loc[values.apply(lambda x: x > upper)].index.to_list()
        return lower_outliers, upper_outliers

    @staticmethod
    def empirical_outliers(values: pd.Series, std_width: int=None) -> Tuple[list, list]:
        """ The empirical rule states that for a normal distribution, nearly all of the data will fall within three
        standard deviations of the mean.

        Given mu and sigma, a simple way to identify outliers is to compute a z-score for every value, which is
        defined as the number of standard deviations away value is from the mean. The 68–95–99.7 rule, also known as
        the empirical rule, is a shorthand used to remember the percentage of values that lie within a band around
        the mean in a normal distribution with a width of two, four and six standard deviations, respectively

        :param values: the values to be assessed
        :param std_width: (optional) A standard deviation away from the mean. Default is 3
        :return: a list of index of outliers
        """
        values = values.copy().dropna()
        std_width = std_width if isinstance(std_width, int) and 2 <= std_width <= 6 else 3
        data_mean, data_std = np.mean(values), np.std(values)
        # identify outliers
        cut_off = data_std * std_width
        lower, upper = data_mean - cut_off, data_mean + cut_off
        # identify outliers
        lower_outliers = values.loc[values.apply(lambda x: x < lower)].index.to_list()
        upper_outliers = values.loc[values.apply(lambda x: x > upper)].index.to_list()
        return lower_outliers, upper_outliers

    @staticmethod
    def pairs_bootstrap_with_linear_regression(x: pd.Series, y: pd.Series, p_percent: float=None, replicates: int=None,
                                               precision: int=None) -> (float, float):
        """"""
        # TODO: Test and finish off
        if x.size != y.size:
            raise ValueError(f"The length of x ({x.size}) does not match the length of y ({y.size})")
        precision = precision if isinstance(precision, int) else 4
        p_percent = p_percent if isinstance(p_percent, float) and 0 < p_percent < 1 else 0.95
        replicates = replicates if isinstance(replicates, int) else 1000
        inds = np.arange(x.size)
        result = []
        for i in range(replicates):
            bs_inds = np.random.choice(inds, len(inds), replace=True)
            bs_x = x.iloc[bs_inds].to_numpy()
            bs_y = y.iloc[bs_inds].to_numpy()

            bs_sample = Polynomial.fit(bs_x, bs_y, 1)
            result.append(bs_sample)
        low_percentile = round((1 - p_percent) / 2, 5)
        high_percentile = 1 - low_percentile
        low_ci, high_ci = np.percentile(result, [low_percentile*100, high_percentile*100])
        return np.round(low_ci, precision), np.round(high_ci, precision)

    @staticmethod
    def bootstrap_confidence_interval(values: pd.Series, p_percent: float=None, replicates: int=None,
                                      func: Any=None, precision: int=None) -> (float, float):
        """ returns a Bootstrap Confidence Interval as a tuple of the low and high confidence interval.

        The function takes a bootstrap sample through sampling with replacement and creates a bootstrap replicate, a
        summary of the statistic taken from the bootstrap sample. This is repeated to create a list of replicates

        :param values: the values to bootstrap
        :param p_percent: the p percent confidence intervals. Default to 0.95 for 95% confidence interval
        :param replicates: the number of replicates to run. default 1,000
        :param func: the function that computes the statistic of interest, np.mean, np.mode etc. Default np.mean
        :param precision: a precision for the output
        :return: a tuple of the low and high confidence interval
        """
        precision = precision if isinstance(precision, int) else 4
        p_percent = p_percent if isinstance(p_percent, float) and 0 < p_percent < 1 else 0.95
        replicates = replicates if isinstance(replicates, int) else 1000
        func = np.mean if func is None else func
        result = []
        for i in range(replicates):
            sample = values.dropna()
            bs_sample = np.random.choice(sample, len(sample), replace=True)
            result.append(func(bs_sample))
        low_percentile = round((1 - p_percent) / 2, 5)
        high_percentile = 1 - low_percentile
        low_ci, high_ci = np.percentile(result, [low_percentile*100, high_percentile*100])
        return np.round(low_ci, precision), np.round(high_ci, precision)

    @staticmethod
    def shapiro_wilk_normality(values: pd.Series, precision: int=None) -> (float, float):
        """The Shapiro-Wilk test evaluates a data sample and quantifies how likely it is that the data was drawn from
        a Gaussian distribution, named for Samuel Shapiro and Martin Wilk.

        In practice, the Shapiro-Wilk test is believed to be a reliable test of normality, although there is some
        suggestion that the test may be suitable for smaller samples of data, e.g. thousands of observations or fewer.

        :param values: the values to consider
        :param precision: a precision for the output
        :return: the Statistics and p-value
        """
        if values.size < 8:
            raise ValueError(f"Shapiro Wilk Normality test requires 8 or more samples; {values.size} samples given")
        precision = precision if isinstance(precision, int) else 4
        stats, p_values = shapiro(values)

        stats = round(stats, precision)
        p_values = round(p_values, precision)
        return stats, p_values

    @staticmethod
    def dagostinos_k2_normality(values: pd.Series, precision: int=None) -> (float, float):
        """The D’Agostino’s K^2 test calculates summary statistics from the data, namely kurtosis and skewness,
        to determine if the data distribution departs from the normal distribution, named for Ralph D’Agostino.

            - Skew: is a quantification of how much a distribution is pushed left or right, a measure of asymmetry
                    in the distribution.
            - Kurtosis: quantifies how much of the distribution is in the tail. It is a simple and commonly used
                    statistical test for normality.

        :param values: the values to consider
        :param precision: a precision for the output
        :return: the Statistics and p-value
        """
        if values.size < 8:
            raise ValueError(f"K2 Normality test requires 8 or more samples; {values.size} samples given")
        precision = precision if isinstance(precision, int) else 4
        stats, p_values = normaltest(values, nan_policy='omit')
        stats = round(stats, precision)
        p_values = round(p_values, precision)
        return stats, p_values

    @staticmethod
    def anderson_darling_tests(values: pd.Series, dist: str=None, precision: int=None) -> (float, list, list):
        """Anderson-Darling Test is a statistical test that can be used to evaluate whether a data sample comes from
        one of among many known data samples, named for Theodore Anderson and Donald Darling.

        It can be used to check whether a data sample is normal. The test is a modified version of a more
        sophisticated nonparametric goodness-of-fit statistical test called the Kolmogorov-Smirnov test.

        A feature of the Anderson-Darling test is that it returns a list of critical values rather than a single
        p-value. This can provide the basis for a more thorough interpretation of the result.

        :param values:
        :param dist: (optional) The type of distribution to test against. default is 'norm'
                      options are {'norm', 'expon', 'logistic', 'gumbel', 'gumbel_l', 'gumbel_r', 'extreme1'}

        :param (optional) precision: the precision of the return
        :return: the Statistics, the significance level and the critical values
        """
        if values.size < 8:
            raise ValueError(f"Anderson Darling Normality test requires 8 or more samples; {values.size} samples given")
        dist = dist if isinstance(dist, str) and dist in ['norm', 'expon', 'logistic', 'gumbel', 'gumbel_l',
                                                          'gumbel_r', 'extreme1'] else 'norm'
        precision = precision if isinstance(precision, int) else 4
        statistic, significance_level, critical_values = anderson(values, dist=dist)
        statistic = round(statistic, precision)
        critical_values = [round(x, precision) for x in critical_values]
        return statistic, significance_level, critical_values

    @staticmethod
    def _distance_validator(p: pd.Series, q: pd.Series, precision: int=None):
        """Distance validator as a preprocess"""
        precision = precision if isinstance(precision, int) else 4
        p = pd.Series(p)
        q = pd.Series(q)
        if round(p.sum(), 3) != 1 or round(q.sum(), 3) != 1:
            raise ValueError(f"the probability scores must add up to 1. "
                             f"sum(p)={round(p.sum(), 3)}, sum(q)={round(q.sum(), 3)}")
        if p.size > q.size:
            q = q.append(pd.Series([0] * (p.size - q.size)), ignore_index=True)
        if p.size < q.size:
            p = p.append(pd.Series([0] * (q.size - p.size)), ignore_index=True)
        return p, q, precision

    @staticmethod
    def hellinger_distance(p: pd.Series, q: pd.Series, precision: int=None):
        """Hellinger distance between distributions (Hoens et al, 2011)

        Hellinger distance is a metric to measure the difference between two probability distributions.
        It is the probabilistic analog of Euclidean distance.
                [1,0] -> [1,0] => 0.0
                [1,0] -> [0.9999, 0.0001] => 0.0
                [1,0] -> [0.99, 0.01] => 0.007
                [1,0] -> [0.98, 0.02] => 0.014
                ...
                [1,0] -> [0.02, 0.98] => 1.214
                [1,0] -> [0.01, 0.99] => 1.273
                [1,0] -> [0.001, 0.999] => 1.369
                [1,0] -> [0, 1] => 1.414
        """
        p, q, precision = DataDiscovery._distance_validator(p, q, precision)
        r = sum([(np.sqrt(t[0]) - np.sqrt(t[1])) * (np.sqrt(t[0]) - np.sqrt(t[1])) for t in zip(p, q)])/np.sqrt(2.)
        return np.round(r, precision)

    @staticmethod
    def total_variation_distance(p: pd.Series, q: pd.Series, precision: int=None):
        """Total Variation Distance (Levin et al, 2008)

        Total Variation Distance is a distance measure for probability distributions. It is an example of a
        statistical distance metric, and is sometimes called the statistical distance or variational distance
                [1,0] -> [1,0] => 0.0
                [1,0] -> [0.9999, 0.0001] => 0.0
                [1,0] -> [0.99, 0.01] => 0.01
                [1,0] -> [0.98, 0.02] => 0.02
                ...
                [1,0] -> [0.02, 0.98] => 0.98
                [1,0] -> [0.01, 0.99] => 0.99
                [1,0] -> [0.001, 0.999] => 0.999
                [1,0] -> [0, 1] => 1.0
         """
        p, q, precision = DataDiscovery._distance_validator(p, q, precision)
        return np.round((sum(abs(p - q)) / 2), precision)

    @staticmethod
    def jensen_shannon_distance(p: pd.Series, q: pd.Series, precision: int=None):
        """Jensen-Shannon Divergence. Distance between two probability distributions

        the Jensen–Shannon divergence is a method of measuring the similarity between two probability distributions.
        It is based on the Kullback–Leibler divergence, with some notable (and useful) differences, including that
        it is symmetric and it always has a finite value. The square root of the Jensen–Shannon divergence is a metric
        often referred to as Jensen-Shannon distance
                [1,0] -> [1,0] => 0.0
                [1,0] -> [0.999, 0.001] => 0.019
                [1,0] -> [0.99, 0.01] => 0.059
                [1,0] -> [0.98, 0.02] => 0.084
                ...
                [1,0] -> [0.02, 0.98] => 0.802
                [1,0] -> [0.01, 0.99] => 0.816
                [1,0] -> [0.0001, 0.9999] => 0.832
                [1,0] -> [0, 1] => 0.833
        """
        p, q, precision = DataDiscovery._distance_validator(p, q, precision)
        # calculate m
        m = (p + q) / 2
        # compute Jensen Shannon Divergence
        divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2
        # compute the Jensen Shannon Distance
        distance = np.sqrt(divergence)
        return np.round(distance, precision)

    @staticmethod
    def filter_univariate_roc_auc(df: pd.DataFrame, target: [str, list], package: str=None, model: str=None,
                                  inc_category: bool=None, threshold: float=None, as_series: bool=None,
                                  train_split: float=None, random_state: int=None, fit_kwargs: dict=None,
                                  classifier_kwargs: dict=None):
        """

        :param df: the DataFrame to analyse
        :param target: the target header(s)
        :param package:
        :param model:
        :param inc_category:
        :param threshold:
        :param as_series: returns by default returns an ordered list, if set to True, returns an ordered pandas Series
        :param train_split: the split percentage as a number between 0 and 1
        :param random_state: a random state constant
        :param fit_kwargs:
        :param classifier_kwargs:
        :return:
        """
        model = model if isinstance(model, str) else "DecisionTreeClassifier"
        package = package if isinstance(package, str) else "sklearn.tree"
        threshold = threshold if isinstance(threshold, float) and 0.5 < threshold < 1 else 0.5
        inc_category = inc_category if isinstance(inc_category, bool) else False
        as_series = as_series if isinstance(as_series, bool) else False
        train_split = train_split if isinstance(train_split, float) and 0 < train_split < 1 else 0.25
        fit_kwargs = fit_kwargs if isinstance(fit_kwargs, dict) else {}
        classifier_kwargs = classifier_kwargs if isinstance(classifier_kwargs, dict) else {}

        df_numbers = Commons.filter_columns(df, dtype=['number'], exclude=False)
        if inc_category:
            for col in Commons.filter_columns(df, dtype=['category'], exclude=False):
                df_numbers[col] = df[col].cat.codes
        df_headers = Commons.filter_columns(df_numbers, headers=target, drop=True).fillna(0).abs()
        if df_headers.shape[1] == 0:
            raise ValueError(f"There were no suitable columns to apply the score too")
        df_target = Commons.filter_columns(df, headers=target, drop=False)
        df_headers, test_headers, df_target, test_target = train_test_split(df_headers, df_target,
                                                                            test_size=1-train_split,
                                                                            random_state=random_state)
        roc_values = []
        for feature in df_headers.columns:
            exec(f"from {package} import {model}")
            clf = eval(f"{model}(**{classifier_kwargs})")
            clf.fit(df_headers[feature].fillna(0).to_frame(), df_target, **fit_kwargs)
            y_scored = clf.predict_proba(test_headers[feature].fillna(0).to_frame())
            roc_values.append(roc_auc_score(test_target, y_scored[:, 1]))
        roc_values = pd.Series(roc_values)
        roc_values.index = df_headers.columns
        roc_values = roc_values[roc_values.values > threshold].sort_values(ascending=False)
        if as_series:
            return roc_values
        return list(roc_values.index)

    @staticmethod
    def filter_univariate_mse(df: pd.DataFrame, target: [str, list], top: [int, float]=None, package: str=None,
                              model: str=None, inc_category: bool=None, as_series: bool=None, train_split: float=None,
                              random_state: int=None, fit_kwargs: dict=None, regressor_kwargs: dict=None):
        """

        :param df: the DataFrame to analyse
        :param target: the target header(s)
        :param top:
        :param package:
        :param model:
        :param inc_category:
        :param as_series: returns by default returns an ordered list, if set to True, returns an ordered pandas Series
        :param train_split: the split percentage as a number between 0 and 1
        :param random_state: a random state constant
        :param fit_kwargs:
        :param regressor_kwargs:
        :return:
        """
        model = model if isinstance(model, str) else "DecisionTreeRegressor"
        package = package if isinstance(package, str) else "sklearn.tree"
        top = top if isinstance(top, (int, float)) else df.shape[1]
        inc_category = inc_category if isinstance(inc_category, bool) else False
        as_series = as_series if isinstance(as_series, bool) else False
        train_split = train_split if isinstance(train_split, float) and 0 < train_split < 1 else 0.25
        fit_kwargs = fit_kwargs if isinstance(fit_kwargs, dict) else {}
        regressor_kwargs = regressor_kwargs if isinstance(regressor_kwargs, dict) else {}

        df_numbers = Commons.filter_columns(df, dtype=['number'], exclude=False)
        if inc_category:
            for col in Commons.filter_columns(df, dtype=['category'], exclude=False):
                df_numbers[col] = df[col].cat.codes
        df_headers = Commons.filter_columns(df_numbers, headers=target, drop=True).fillna(0).abs()
        if df_headers.shape[1] == 0:
            raise ValueError(f"There were no suitable columns to apply the score too")
        df_target = Commons.filter_columns(df, headers=target, drop=False)
        df_headers, test_headers, df_target, test_target = train_test_split(df_headers, df_target,
                                                                            test_size=1-train_split,
                                                                            random_state=random_state)
        if HandlerFactory.check_module(module_name=package):
            module = HandlerFactory.get_module(module_name=package)
        else:
            raise ModuleNotFoundError(f"The required module {package} has not been installed. "
                                      f"Please pip install the appropriate package in order to complete this action")
        mse_values = []
        for feature in df_headers.columns:
            clf = eval(f"module.{model}(**{regressor_kwargs})", globals(), locals())
            clf.fit(df_headers[feature].fillna(0).to_frame(), df_target, **fit_kwargs)
            y_scored = clf.predict(test_headers[feature].fillna(0).to_frame())
            mse_values.append(mean_squared_error(test_target, y_scored))
        mse_values = pd.Series(mse_values)
        mse_values.index = df_headers.columns

        mse_values = mse_values[mse_values.values > 0].sort_values(ascending=True).iloc[:int(top)]
        if 0 < top < 1:
            mse_values = mse_values[mse_values.values < mse_values.quantile(q=top)].sort_values(ascending=True)
        elif top >= 1:
            mse_values = mse_values.iloc[:int(top)]
        if as_series:
            return mse_values
        return list(mse_values.index)

    @staticmethod
    def filter_fisher_score(df: pd.DataFrame, target: [str, list], top: [int, float]=None, inc_zero_score: bool=None,
                            as_series: bool=None, train_split: float=None, random_state: int=None) -> [pd.Series, list]:
        """ Ranking columns to returns a list of headers in order of p-value from lowest to highest.
        Measured the dependence of 2 variables. This is suited to categorical variables where the target is binary.
        Variable values should be non-negative, and typically Boolean, frequencies or counts.

        :param df: the DataFrame to analyse
        :param target: the target header(s)
        :param top: used if the value passed is greater than zero. NOTE zero p-scores are excluded
                    1 <= value => n returns that number of ranked columns, top=3 returns the top 3
                    0 < value > 1 returns the percentile of ranked columns, top=0.1 would give the top 10 percentile
        :param inc_zero_score: if a p-value of zero should be included in the list. only included if 'top' is 0 or None
        :param as_series: returns by default returns an ordered list, if set to True, returns an ordered pandas Series
        :param train_split: the split percentage as a number between 0 and 1
        :param random_state: a random state constant
        :return: a list of headers in order of lowest p-score to highest or a pandas series
        """
        target = Commons.list_formatter(target)
        for name in target:
            if name not in df.columns:
                raise ValueError(f"The target header '{name}' was not found in the DataFrame")
        train_split = train_split if isinstance(train_split, float) else 0.25
        if not 0 < train_split < 1:
            raise ValueError(f"The train_split value must be between 0 and 1, '{train_split}' was passed")
        inc_zero_score = inc_zero_score if isinstance(inc_zero_score, bool) else False
        as_series = as_series if isinstance(as_series, bool) else False
        top = top if isinstance(top, (int, float)) else 0
        df_numbers = Commons.filter_columns(df, dtype=['number'], exclude=False)
        for col in Commons.filter_columns(df, dtype=['category'], exclude=False):
            df_numbers[col] = df[col].cat.codes
        df_headers = Commons.filter_columns(df_numbers, headers=target, drop=True).fillna(0).abs()
        if df_headers.shape[1] == 0:
            raise ValueError(f"There were no suitable columns to apply the score too")
        df_target = Commons.filter_columns(df, headers=target, drop=False)
        if isinstance(train_split, float) and 0 < train_split < 1:
            df_headers, _, df_target, _ = train_test_split(df_headers, df_target, test_size=1-train_split,
                                                           random_state=random_state)
        f_score = chi2(df_headers, df_target)
        pvalues = pd.Series(f_score[1])
        pvalues.index = df_headers.columns
        pvalues = pvalues.round(8)
        zero_scores = pvalues[pvalues.values == 0]
        plus_score = pvalues[pvalues.values > 0]
        if 0 < top < 1:
            rtn_values = plus_score[plus_score.values < plus_score.quantile(q=top)].sort_values(ascending=True)
        else:
            rtn_values = plus_score.sort_values(ascending=True)
            if inc_zero_score:
                rtn_values = rtn_values.append(zero_scores)
            if top >= 1:
                rtn_values = rtn_values.iloc[:int(top)]
        if as_series:
            return rtn_values
        return list(rtn_values.index)

    @staticmethod
    def filter_correlated(df: pd.DataFrame, target: str, threshold: float=None, inc_category: bool=None,
                          train_split: float=None, random_state: int=None,  **classifier_kwargs) -> list:
        """ Using scikit-learn RandomForestClassifier, identifies groups of highly correlated columns based on the
        threshold, returning the column names of the most important of the correlated group to the target.
        ref:  Senliol, Baris, et al (2008). 'Fast Correlation Based Filter (FCBF) .

        :param df: the Canonical data to drop correlated collumns from
        :param target: a target column to relate importance to
        :param threshold: (optional) threshold correlation between columns. default 0.998
        :param inc_category: (optional) if category type columns should be converted to numeric representations
        :param train_split: a train percentage split from the df to avoid over-fitting.
        :param random_state: a random state should be applied to the test train split.
        :param classifier_kwargs: kwargs to send to the classifier.
        :return: if inplace, returns a formatted cleaner contract for this method, else a deep copy Canonical,.
        """
        # Code block for intent
        train_split = train_split if isinstance(train_split, float) else 0.25
        if not 0 < train_split < 1:
            raise ValueError(f"The train_split value must be between 0 and 1, '{train_split}' was passed")
        inc_category = inc_category if isinstance(inc_category, bool) else False
        threshold = threshold if isinstance(threshold, float) and 0 < threshold < 1 else 0.998
        df_numbers = Commons.filter_columns(df, dtype=['number'], exclude=False)
        if inc_category:
            for col in Commons.filter_columns(df, dtype=['category'], exclude=False):
                df_numbers[col] = df[col].cat.codes
        df_headers = Commons.filter_columns(df, headers=target, drop=True)
        df_target = df[target]
        if isinstance(train_split, float) and 0 < train_split < 1:
            df_headers, _, df_target, _ = train_test_split(df_headers, df_target, test_size=1-train_split,
                                                           random_state=random_state)
        # build a dataframe with the correlation between features
        corr_features = df_headers.corr()
        corr_features = corr_features.abs().unstack()  # absolute value of corr coef
        corr_features = corr_features.sort_values(ascending=False)
        corr_features = corr_features[corr_features >= threshold]
        corr_features = corr_features[corr_features < 1]
        corr_features = pd.DataFrame(corr_features).reset_index()
        corr_features.columns = ['feature1', 'feature2', 'corr']
        # find groups of correlated features
        grouped_feature_ls = []
        correlated_groups = []
        for feature in corr_features.feature1.unique():
            if feature not in grouped_feature_ls:
                # find all features correlated to a single feature
                correlated_block = corr_features[corr_features.feature1 == feature]
                grouped_feature_ls = grouped_feature_ls + list(
                    correlated_block.feature2.unique()) + [feature]
                # append the block of features to the list
                correlated_groups.append(correlated_block)
        # use random forest classifier to select features of importance
        n_estimators = classifier_kwargs.pop('n_estimators', 200)
        random_state = classifier_kwargs.pop('random_state', random_state)
        max_depth = classifier_kwargs.pop('max_depth', 4)
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth,
                                    **classifier_kwargs)
        importance_set = set()
        for group in correlated_groups:
            features = list(group.feature2.unique()) + list(group.feature1.unique())
            _ = df_headers[features].fillna(0)
            _ = rf.fit(X=_, y=df_target)
            importance = pd.concat([pd.Series(features), pd.Series(rf.feature_importances_)], axis=1)
            importance_set.add(importance.iloc[0, 0])
        return list(importance_set)

    @staticmethod
    def analyse_category(categories: Any, lower: [int, float]=None, upper: [int, float]=None, top: int=None,
                         nulls_list: list=None, replace_zero: [int, float]=None, freq_precision: int=None):
        """Analyses a set of categories and returns a dictionary of intent and patterns and statitics.

        :param categories: the categories to analyse
        :param lower: outliers lower limit and below to remove. (optional)
                         int represents the category count, (removed before the weighting pattern)
                         float between 0 and 1 represents normalised value (removed from weighting pattern)
        :param upper: outliers upper limit and above to remove. (optional
                         integer represents category count, (removed before the weighting pattern)
                         float between 0 and 1 represents normalised value (removed from weighting pattern)
        :param top: (optional) only select the top n from a selection, regardless of weighting equivalence.
        :param replace_zero: (optional) if zero what to replace the weighting value with to avoid zero probability
        :param nulls_list: (optional) a list of nulls if more than the default empty string
        :param freq_precision: (optional) The precision of the relative freq values. by default set to 2.
        :return: a dictionary of results
        """
        categories = pd.Series(categories)
        freq_precision = 2 if not isinstance(freq_precision, int) else freq_precision
        nulls_list = ['<NA>', '', ' ', 'NaN', 'nan', None] if not isinstance(nulls_list, list) else nulls_list
        replace_zero = 0 if not isinstance(replace_zero, (int, float)) else replace_zero
        lower = lower if isinstance(lower, (int, float)) else None
        upper = upper if isinstance(upper, (int, float)) else None
        param_lower = lower
        param_upper = upper
        param_top = top if isinstance(top, int) else 0
        _original_size = categories.size
        categories.replace(nulls_list, np.nan, inplace=True, regex=True)
        categories = categories.dropna()
        nulls_percent = round(((_original_size - categories.size) / _original_size) * 100,
                              freq_precision) if _original_size > 0 else 0
        _categories_size = categories.size
        value_count = categories.value_counts(sort=True, normalize=False, dropna=True)
        _granularity = value_count.nunique()
        # if integer filter out the value counts
        if isinstance(lower, int):
            value_count = value_count[value_count >= lower]
        if isinstance(upper, int):
            value_count = value_count[value_count <= upper]
        sub_categories = categories[categories.isin(value_count.index)]
        value_count = sub_categories.value_counts(sort=True, normalize=True, dropna=True)
        _granularity = value_count.nunique()
        # if integer filter out the value counts
        if isinstance(lower, float) and 0 <= lower <= 1:
            value_count = value_count[value_count >= lower]
        if isinstance(upper, float) and 0 <= upper <= 1:
            value_count = value_count[value_count <= upper]
        if isinstance(top, int) and top > 0:
            value_count = value_count.iloc[:top]
        _granularity = value_count.nunique()
        sub_categories = sub_categories[sub_categories.isin(value_count.index)]
        _outlier_percent = round(((_categories_size - sub_categories.size) / _categories_size) * 100,
                                   freq_precision) if _categories_size > 0 else 0
        _outlier_count = len(set(set(categories).symmetric_difference(set(sub_categories))))
        _sample_dist = sub_categories.value_counts(sort=True, normalize=False, dropna=True).to_list()
        if isinstance(top, int) and top > 0:
            _sample_dist = _sample_dist[:top]
        value_count = value_count.replace(np.nan, 0).replace(0, replace_zero)*100
        _weighting = [round(x, freq_precision) for x in value_count.to_list()]
        if len(_weighting) == 0:
            _weighting = [0]
        _lower = lower if isinstance(lower, (int, float)) else _weighting[-1]
        _upper = upper if isinstance(upper, (int, float)) else _weighting[0]
        rtn_dict = {'intent': {'categories': value_count.index.to_list(), 'dtype': 'category'},
                    'params': {'freq_precision': freq_precision},
                    'patterns': {'relative_freq': _weighting, 'sample_distribution': _sample_dist},
                    'stats': {'category_count': _granularity, 'highest_unique': _upper, 'lowest_unique': _lower,
                              'nulls_percent': nulls_percent,  'sample_size': sub_categories.size,
                              'excluded_percent': _outlier_percent}}
        if isinstance(param_top, int) and param_top > 0:
            rtn_dict.get('params')['top'] = top
        if isinstance(param_lower, (int, float)):
            rtn_dict.get('params')['lower'] = param_lower
        if isinstance(param_upper, (int, float)):
            rtn_dict.get('params')['upper'] = param_upper
        if replace_zero != 0:
            rtn_dict.get('params')['replace_zero'] = replace_zero
        if _outlier_count != 0:
            rtn_dict.get('stats')['excluded_categories'] = _outlier_count
            rtn_dict.get('stats')['excluded_sample'] = categories.size - sub_categories.size
        return rtn_dict

    @staticmethod
    def analyse_number(values: Any, granularity: [int, float, list]=None, lower: [int, float]=None,
                       upper: [int, float]=None, precision: int=None, freq_precision: int=None,
                       dominant: [int, float, list]=None, exclude_dominant: bool=None, detail_stats: bool=None,
                       p_percent: float=None, replicates: int=None):
        """Analyses a set of values and returns a dictionary of analytical statistics. Unless zero is not common,
        to avoid zero values skewing the weighting you should always consider 0 as a dominant value

        :param values: the values to analyse
        :param granularity: (optional) the granularity of the analysis across the range. Default is 1
                int passed - represents the number of periods
                float passed - the length of each interval
                list[tuple] - specific interval periods e.g []
                list[float] - the percentile or quantities, All should fall between 0 and 1
        :param lower: (optional) the lower limit of the number value. Default min()
        :param upper: (optional) the upper limit of the number value. Default max()
        :param precision: (optional) The precision of the range and boundary values. by default set to 3.
        :param freq_precision: (optional) The precision of the relative freq values. by default set to 2.
        :param dominant: (optional) identify dominant value or list of values, can be empty if None, mode is dominant
        :param exclude_dominant: (optional) if the dominant values should be excluded from the weighting. Default False
        :param detail_stats: (optional) statistics use boostrap confidence intervals and include outliers algorithms
        :param p_percent: the confidence interval p percent confidence intervals. Default to 0.95 for 95% CI
        :param replicates: confidence interval the number of replicates to run. default 1,000
        :return: a dictionary of results
        """
        values = pd.Series(values, dtype=np.number)
        _original_size = values.size
        precision = 3 if not isinstance(precision, int) else precision
        freq_precision = 2 if not isinstance(freq_precision, int) else freq_precision
        granularity = 3 if not isinstance(granularity, (int, float, list)) or granularity == 0 else granularity
        detail_stats = detail_stats if isinstance(detail_stats, bool) else True
        replicates = replicates if isinstance(replicates, int) else 1000
        p_percent = p_percent if isinstance(p_percent, float) and 0 <= p_percent <= 1 else 0.95
        _intervals = granularity
        exclude_dominant = False if not isinstance(exclude_dominant, bool) else exclude_dominant
        param_lower = None if not isinstance(lower, (int, float)) else lower
        param_upper = None if not isinstance(upper, (int, float)) else upper
        # nulls
        _values_size = values.size
        values = values.dropna()
        _nulls_size = _original_size - values.size
        _nulls_percent = round(((_values_size - values.size) / _values_size) * 100,
                               freq_precision) if _values_size > 0 else 0
        # limits
        if values.size > 0:
            lower = values.min() if not isinstance(lower, (int, float)) else lower
            upper = values.max() if not isinstance(upper, (int, float)) else upper
            if lower >= upper:
                upper = lower
                _intervals = [(lower, upper, 'both')]
        else:
            lower = 0
            upper = 0
            _intervals = [(lower, upper, 'both')]
        # outliers
        _values_size = values.size
        values = values.loc[values.between(lower, upper, inclusive='both').values]
        _excluded_size = _original_size - _nulls_size - _values_size
        _excluded_percent = round(((_values_size - values.size) / _values_size) * 100,
                                  freq_precision) if _values_size > 0 else 0
        # dominance
        dominant = values.mode(dropna=True).to_list()[:10] if not isinstance(dominant, (int, float, list)) else dominant
        dominant = Commons.list_formatter(dominant)
        _dominant_values = values[values.isin(dominant)]
        _dominant_count = _dominant_values.value_counts(normalize=False, dropna=True)
        _dominance_freq = _dominant_values.value_counts(normalize=True, dropna=True).round(freq_precision)*100
        _dominance_freq = [0] if _dominance_freq.size == 0 else _dominance_freq.to_list()

        _values_size = values.size
        if exclude_dominant:
            values = values[~values.isin(dominant)]
        _dominant_size = _dominant_count.sum()
        if _dominant_size == _values_size:
            _dominant_percent = 1
        else:
            _dominant_percent = round((_dominant_size / _values_size) * 100, freq_precision) if _values_size > 0 else 0
        # if there are no samples remaining
        if values.size == 0:
            _intervals = [(lower, upper, 'both')]
            return {'intent': {'intervals': _intervals, 'dtype': 'number'},
                    'params': {'precision': precision, 'freq_precision': freq_precision, 'granularity': granularity,
                               'detail_stats': detail_stats, 'lower': param_lower, 'upper': param_upper},
                    'patterns': {'relative_freq': [1], 'freq_mean': [0], 'freq_std': [0], 'sample_distribution': [0]},
                    'stats': {'lowest': round(lower, precision), 'highest': round(upper, precision),
                              'nulls_percent': _nulls_percent, 'sample_size': 0, 'excluded_sample': _excluded_size,
                              'excluded_percent': _excluded_percent, 'mean': 0, 'std': 0}}
        # granularity
        if isinstance(_intervals, (int, float)):
            # if granularity float then convert frequency to intervals
            if isinstance(_intervals, float):
                # make sure frequency goes beyond the upper
                _end = upper + _intervals - (upper % _intervals)
                periods = pd.interval_range(start=lower, end=_end, freq=_intervals).drop_duplicates()
                periods = periods.to_tuples().to_list()
                _intervals = []
                while len(periods) > 0:
                    period = periods.pop(0)
                    if len(periods) == 0:
                        _intervals += [(period[0], period[1], 'both')]
                    else:
                        _intervals += [(period[0], period[1], 'left')]
            # if granularity int then convert periods to intervals
            else:
                periods = pd.interval_range(start=lower, end=upper, periods=_intervals).drop_duplicates()
                _intervals = periods.to_tuples().to_list()
        if isinstance(_intervals, list):
            if all(isinstance(value, tuple) for value in _intervals):
                if len(_intervals[0]) == 2:
                    _intervals[0] = (_intervals[0][0], _intervals[0][1], 'both')
                _intervals = [(t[0], t[1], 'right') if len(t) == 2 else t for t in _intervals]
            elif all(isinstance(value, float) and 0 < value < 1 for value in _intervals):
                quantiles = list(set(_intervals + [0, 1.0]))
                boundaries = values.quantile(quantiles).values
                boundaries.sort()
                _intervals = [(boundaries[0], boundaries[1], 'both')]
                _intervals += [(boundaries[i - 1], boundaries[i], 'right') for i in range(2, boundaries.size)]
            else:
                _intervals = (lower, upper, 'both')

        # interval weighting
        _sample_dist = []
        _values_weights = []
        _mean_weights = []
        _std_weights = []
        for interval in _intervals:
            low, high, closed = interval
            if str.lower(closed) == 'neither':
                interval_values = values.loc[(values > low) & (values < high)]
            elif str.lower(closed) == 'left':
                interval_values = values.loc[(values >= low) & (values < high)]
            elif str.lower(closed) == 'both':
                interval_values = values.loc[(values >= low) & (values <= high)]
            else:  # default right
                interval_values = values.loc[(values > low) & (values <= high)]
            _values_weights.append(interval_values.size)
            _sample_dist.append(interval_values.size)
            if interval_values.size == 0:
                _mean_weights.append(0.0)
                _std_weights.append(0.0)
            else:
                # interal patterns
                if detail_stats:
                    _mean_weights.append(
                        DataDiscovery.bootstrap_confidence_interval(interval_values, func=np.mean, p_percent=p_percent,
                                                                    precision=freq_precision, replicates=replicates))
                    _std_weights.append(
                        DataDiscovery.bootstrap_confidence_interval(interval_values, func=np.std, p_percent=p_percent,
                                                                    precision=freq_precision, replicates=replicates))
                else:
                    _mean_weights.append(np.mean(interval_values))
                    _std_weights.append(np.std(interval_values))
        if len(_values_weights) == 0:
            _values_weights = [0]
        _values_weights = pd.Series(_values_weights)
        if values.size > 0:
            _values_weights = _values_weights.apply(lambda x: round((x / values.size) * 100, freq_precision))
        _intervals = [(round(p[0], precision), round(p[1], precision), p[2]) for p in _intervals]
        rtn_dict = {'intent': {'intervals': _intervals, 'dtype': 'number'},
                    'params': {'precision': precision, 'freq_precision': freq_precision, 'granularity': granularity,
                               'detail_stats': detail_stats},
                    'patterns': {'relative_freq': _values_weights.to_list(), 'sample_distribution': _sample_dist},
                    'stats': {'lowest': round(lower, precision), 'highest': round(upper, precision),
                              'nulls_percent': _nulls_percent, 'sample_size': values.size,
                              'excluded_sample': _excluded_size,'excluded_percent': _excluded_percent}}
        # params
        if isinstance(param_lower, (int, float)):
            rtn_dict.get('params')['lower'] = round(param_lower, freq_precision)
        if isinstance(param_upper, (int, float)):
            rtn_dict.get('params')['upper'] = round(param_upper, freq_precision)
        # statistics
        if detail_stats:
            rtn_dict.get('params')['bci_replicates'] = replicates
            rtn_dict.get('params')['bci_p_percent'] = p_percent
            rtn_dict.get('patterns')['freq_mean_bci'] = _mean_weights
            rtn_dict.get('patterns')['freq_std_bci'] = _std_weights
            _mean = DataDiscovery.bootstrap_confidence_interval(values, func=np.mean, precision=freq_precision,
                                                                replicates=replicates, p_percent=p_percent)
            rtn_dict.get('stats')['bci_mean'] = _mean
            _std = DataDiscovery.bootstrap_confidence_interval(values, func=np.std, precision=freq_precision,
                                                               replicates=replicates, p_percent=p_percent)
            rtn_dict.get('stats')['bci_std'] = _std
            _o_low, _o_high = DataDiscovery.interquartile_outliers(values)
            rtn_dict.get('stats')['outliers_iqr'] = (len(_o_low), len(_o_high))
            _o_low, _o_high = DataDiscovery.empirical_outliers(values)
            rtn_dict.get('stats')['outliers_emp'] = (len(_o_low), len(_o_high))
        else:
            rtn_dict.get('stats')['mean'] = np.round(np.mean(values), freq_precision)
            rtn_dict.get('stats')['std'] = np.round(np.std(values), freq_precision)
            rtn_dict.get('patterns')['freq_mean'] = _mean_weights
            rtn_dict.get('patterns')['freq_std'] = _std_weights
        # normality
        if values.size >= 10 and detail_stats:
            if values.size < 5000:
                _, _p_value = DataDiscovery.shapiro_wilk_normality(values, precision=freq_precision)
                rtn_dict.get('stats')['p_value_sharpo'] = _p_value
            _, _p_value = DataDiscovery.dagostinos_k2_normality(values, precision=freq_precision)
            rtn_dict.get('stats')['p_value_k2'] = _p_value
            _stats, _level, _ = DataDiscovery.anderson_darling_tests(values, dist='norm')
            rtn_dict.get('stats')['anderson_norm'] = [round(x - _stats, freq_precision) for x in _level]
            _stats, _level, _ = DataDiscovery.anderson_darling_tests(values, dist='expon')
            rtn_dict.get('stats')['anderson_expon'] = [round(x - _stats, freq_precision) for x in _level]
            _stats, _level, _ = DataDiscovery.anderson_darling_tests(values, dist='logistic')
            rtn_dict.get('stats')['anderson_logistic'] = [round(x - _stats, freq_precision) for x in _level]
            _stats, _level, _ = DataDiscovery.anderson_darling_tests(values, dist='gumbel')
            rtn_dict.get('stats')['anderson_gumbel'] = [round(x - _stats, freq_precision) for x in _level]
        # dominance
        if exclude_dominant:
            rtn_dict.get('patterns')['dominant_excluded'] = dominant
        else:
            rtn_dict.get('patterns')['dominant_values'] = dominant
        rtn_dict.get('patterns')['dominance_freq'] = _dominance_freq
        rtn_dict.get('patterns')['dominant_percent'] = _dominant_percent
        return rtn_dict

    @staticmethod
    def analyse_date(values: Any, granularity: [int, float, pd.Timedelta]=None, lower: Any=None, upper: Any=None,
                     day_first: bool=None, year_first: bool=None, date_format: str=None, freq_precision: int=None):
        """Analyses a set of dates and returns a dictionary of selection and weighting

        :param values: the values to analyse
        :param granularity: (optional) the granularity of the analysis across the range.
                int passed - the number of sections to break the value range into
                pd.Timedelta passed - a frequency time delta
        :param lower: (optional) the lower limit of the number value. Takes min() if not set
        :param upper: (optional) the upper limit of the number value. Takes max() if not set
        :param day_first: if the date provided has day first
        :param year_first: if the date provided has year first
        :param date_format: the format of the output dates, if None then pd.Timestamp
        :param freq_precision: (optional) The precision of the weighting values. by default set to 2.
        :return: a dictionary of results
        """
        values = pd.to_datetime(values, errors='coerce', infer_datetime_format=True, dayfirst=day_first,
                                yearfirst=year_first)
        lower = pd.to_datetime(lower, errors='coerce', infer_datetime_format=True, dayfirst=day_first,
                               yearfirst=year_first)
        upper = pd.to_datetime(upper, errors='coerce', infer_datetime_format=True, dayfirst=day_first,
                               yearfirst=year_first)
        params_lower = lower if isinstance(lower, pd.Timestamp) else None
        params_upper = upper if isinstance(upper, pd.Timestamp) else None
        values = mdates.date2num(values)
        values = pd.Series(values)
        lower = values.min() if not isinstance(lower, pd.Timestamp) else mdates.date2num(lower)
        upper = values.max() if not isinstance(upper, pd.Timestamp) else mdates.date2num(upper)
        if isinstance(granularity, pd.Timedelta):
            granularity = mdates.date2num(mdates.num2date(lower) + granularity) - lower
        rtn_dict = DataDiscovery.analyse_number(values, granularity=granularity, lower=lower, upper=upper,
                                                precision=10, freq_precision=freq_precision, detail_stats=False)
        # add the specific data
        rtn_dict.get('params')['day_first'] = False if not isinstance(day_first, bool) else day_first
        rtn_dict.get('params')['year_first'] = False if not isinstance(year_first, bool) else year_first
        if isinstance(date_format, str):
            rtn_dict.get('params')['date_format'] = date_format
        if isinstance(year_first, bool):
            rtn_dict.get('params')['year_first'] = year_first
        if isinstance(day_first, bool):
            rtn_dict.get('params')['day_first'] = day_first
        # tidy back all the dates
        rtn_dict.get('intent')['intervals'] = [(pd.Timestamp(mdates.num2date(p[0])),
                                                pd.Timestamp(mdates.num2date(p[1])),
                                                p[2]) for p in rtn_dict.get('intent')['intervals']]
        if params_lower:
            rtn_dict.get('params')['lower'] = params_lower
        if params_upper:
            rtn_dict.get('params')['upper'] = params_upper
        rtn_dict.get('stats')['lowest'] = pd.Timestamp(mdates.num2date(rtn_dict.get('stats')['lowest']))
        rtn_dict.get('stats')['highest'] = pd.Timestamp(mdates.num2date(rtn_dict.get('stats')['highest']))
        rtn_dict.get('stats')['mean'] = pd.Timestamp(mdates.num2date(rtn_dict.get('stats')['mean']))
        if isinstance(date_format, str):
            rtn_dict.get('intent')['intervals'] = [(p[0].strftime(date_format), p[1].strftime(date_format),
                                                    p[2]) for p in rtn_dict.get('intent')['intervals']]
            if params_lower:
                rtn_dict.get('params')['lower'] = rtn_dict.get('params')['lower'].strftime(date_format)
            if params_upper:
                rtn_dict.get('params')['upper'] = rtn_dict.get('params')['upper'].strftime(date_format)
            rtn_dict.get('stats')['lowest'] = rtn_dict.get('stats')['lowest'].strftime(date_format)
            rtn_dict.get('stats')['highest'] = rtn_dict.get('stats')['highest'].strftime(date_format)
            rtn_dict.get('stats')['mean'] = rtn_dict.get('stats')['mean'].strftime(date_format)
        rtn_dict.get('intent')['dtype'] = 'date'
        # remove things that don't make sense to dates
        rtn_dict.get('params').pop('precision', None)
        for label in ['dominant_excluded', 'dominant_values', 'dominance_freq', 'dominant_percent',
                      'freq_std', 'freq_mean']:
            rtn_dict.get('patterns').pop(label, None)
        return rtn_dict

    @staticmethod
    def analyse_association(df: pd.DataFrame, columns_list: list, exclude_associate: list=None, 
                            detail_numeric: bool=None, strict_typing: bool=None, category_limit: int=None):
        """ Analyses the association of Category against Values and returns a dictionary of resulting weighting
        the structure of the columns_list is a list of dictionaries with the key words
            - label: the label or name of the header in the DataFrame
            - dtype: one of category|number|date indicating the origin of the data
            - chunk_size: if the weighting pattern is over the size of the data the number of chunks
            - replace_zero: if a zero reference is returned it can optionally be replaced with a low probability
        and example structure might look like:
            [{'label1': {'granularity': 5}},
             {'label2': {'dtype': 'category', 'top': 10, 'replace_zero': 0.001}}]

        :param df: the DataFrame to take the columns from
        :param columns_list: a list structure of columns to select for association
        :param exclude_associate: (optional) a list of dot separated tree of items to exclude from iteration 
                (e.g. ['age.gender.salary']
        :param detail_numeric: (optional) as a default, if numeric columns should have detail stats, slowing analysis
        :param strict_typing: (optional) stops objects and string types being seen as categories
        :param category_limit: (optional) a global cap on categories captured. zero value returns no limits
        :return: an analytics model dictionary
        """
        tools = DataDiscovery
        if not isinstance(columns_list, list):
            raise ValueError("The columns list must be a list of dictionaries")
        if all(isinstance(x, str) for x in columns_list):
            analysis_dict = {}
            for item in columns_list:
                if df[item].dtype.name.startswith('float'):
                    analysis_dict[item] = {'dtype': 'number', 'granularity': 5,
                                           'exclude_dominant': True}
                elif df[item].dtype.name.startswith('int'):
                    analysis_dict[item] = {'dtype': 'number', 'granularity': 5,
                                           'exclude_dominant': True, 'precision': 0}
                elif df[item].dtype.name == 'category':
                    analysis_dict[item] = {'dtype': 'category', 'top': 10}
                elif df[item].dtype.name.startswith('date'):
                    analysis_dict[item] = {'dtype': 'date', 'granularity': 5}
                elif df[item].dtype.name.startswith('bool'):
                    analysis_dict[item] = {'dtype': 'bool'}
                else:
                    analysis_dict[item] = {}
            columns_list = [analysis_dict]

        def _get_weights(_df: pd.DataFrame, columns: list, index: int, weighting: dict, parent: list):
            for label, kwargs in columns[index].items():
                tree = parent.copy()
                tree.append(label)
                if '.'.join(tree) in exclude_associate:
                    continue
                section = {'branch': {'label': label, 'root': str('.'.join(tree))}}
                if label not in _df.columns:
                    raise ValueError("header '{}' not found in the DataFrame".format(label))
                dtype = kwargs.get('dtype')
                if not dtype:
                    dtype = df[label].dtype.name
                    dtype = 'number' if dtype.startswith('int') or dtype.startswith('float') else dtype
                dtype = dtype.lower()
                if not strict_typing and (dtype.startswith('object') or dtype.startswith('string')):
                    dtype = 'category'
                lower = kwargs.get('lower')
                upper = kwargs.get('upper')
                granularity = kwargs.get('granularity')
                freq_precision = kwargs.get('freq_precision')
                if dtype.startswith('int') or dtype.startswith('float') or dtype.startswith('number'):
                    precision = kwargs.get('precision')
                    dominant = kwargs.get('dominant')
                    exclude_dominant = kwargs.get('exclude_dominant')
                    detail_stats = kwargs.get('detail_stats', detail_numeric)
                    selection = 'intervals'
                    section['insight'] = tools.analyse_number(_df[label], granularity=granularity, lower=lower,
                                                              upper=upper, detail_stats=detail_stats, 
                                                              precision=precision, freq_precision=freq_precision,
                                                              dominant=dominant, exclude_dominant=exclude_dominant)

                elif dtype.startswith('date'):
                    day_first = kwargs.get('day_first')
                    year_first = kwargs.get('year_first')
                    date_format = kwargs.get('date_format')
                    selection = 'intervals'
                    section['insight'] = tools.analyse_date(_df[label], granularity=granularity, lower=lower,
                                                            upper=upper, day_first=day_first, year_first=year_first,
                                                            freq_precision=freq_precision, date_format=date_format)
                elif dtype.startswith('category') or dtype.startswith('bool'):
                    top = kwargs.get('top', category_limit)
                    replace_zero = kwargs.get('replace_zero')
                    nulls_list = kwargs.get('nulls_list')
                    selection = 'categories'
                    section['insight'] = tools.analyse_category(_df[label], lower=lower, upper=upper, top=top,
                                                                replace_zero=replace_zero, nulls_list=nulls_list,
                                                                freq_precision=freq_precision)
                else:
                    continue
                # check if we trim the branch
                if kwargs.get('trim', False):
                    continue
                # iterate the sub categories
                _leaves = section.get('insight', {}).get('intent', {}).get(selection, [])
                _leaves = [str(x) for x in _leaves]
                section['branch'].update({'leaves': _leaves})
                for idx in range(len(_leaves)):
                    _leaf = _leaves[idx]
                    if section.get('sub_category') is None:
                        section['sub_category'] = {}
                    section.get('sub_category').update({_leaf: {}})
                    sub_category = section.get('sub_category', {}).get(_leaf, {})
                    if index < len(columns) - 1:
                        _selection = section.get('insight', {}).get('intent', {}).get(selection, [])[idx]
                        if isinstance(_selection, tuple):
                            interval = pd.Interval(left=_selection[0], right=_selection[1], closed=_selection[2])
                            df_filter = _df.loc[_df[label].apply(lambda x: x in interval)]
                        else:
                            df_filter = _df[_df[label] == _selection]
                        _get_weights(df_filter, columns=columns, index=index + 1, weighting=sub_category,
                                     parent=tree+[str(idx)])
                    # tidy empty sub categories
                    if section.get('sub_category').get(_leaf) == {}:
                        section.pop('sub_category')
                weighting[label] = section
            return

        category_limit = category_limit if isinstance(category_limit, int) else 0
        strict_typing = strict_typing if isinstance(strict_typing, bool) else False
        exclude_associate = list() if not isinstance(exclude_associate, list) else exclude_associate
        rtn_dict = {}
        _get_weights(df, columns=columns_list, index=0, weighting=rtn_dict, parent=list())
        return rtn_dict

    @staticmethod
    def to_sample_num(df, sample_num=10000, is_random=True, file_name=None, sep=None) -> pd.DataFrame:
        """ Creates a sample_num of a pandas dataframe.
        This is used to reduce the size of large files when investigating and experimenting.
        the rows are selected from the start the middle and the end of the file

        :param df: The dataframe to sub file
        :param sample_num: the positive sample_num of rows to extract. Default to 10000
        :param is_random: how to extract the rows. Default to True
                True: will select sample_num random values from the df
                False: will take sample_num from top, mid and tail of the df
        :param file_name: the name of the csv file to save. Default is None
                if no file name is provided the file is NOT saved to persistence
        :param sep: the csv file separator. Default to ',' [Comma]
        :return: pandas.Dataframe
        """
        if df is None or len(df) < 1:
            return pd.DataFrame()
        if is_random:
            n = len(df)
            if n < sample_num:
                sample_num = n
            index_list = sorted(random.sample(range(n), k=sample_num))
            df_sub = df.iloc[index_list]
        else:
            diff = sample_num % 3
            sample_num = int(sample_num / 3)
            df_sub = df.iloc[:sample_num]
            mid = int(len(df) / 2)
            df_sub = df_sub.append(df.iloc[mid:mid + sample_num + diff])
            df_sub = df_sub.append(df.iloc[-sample_num:])

        if file_name is not None:
            if sep is None:
                sep = ','
            df_sub.to_csv(file_name, sep=sep, encoding='utf-8')

        return df_sub

    @staticmethod
    def find_file(find_name=None, root_dir=None, ignorecase=True, extensions=None):
        """ find file(s) under the root path with the extension types given
        find_name can be full or part and will return a pandas.DatafFrame of
        matching names with the following headings:

        ["name", "parent", "stem", "suffix", "created", "search"]

        :param find_name: the name of the item to find. Defualt to None
        :param root_dir: the root directory to seach from. Default is cwd
        :param ignorecase: if the search should ignore the name case. Default to True
        :param extensions: a list of extensions to look for (should start with a .)
            Default are ['csv', 'xlsx', 'json', 'p', 'yaml', 'tsv']
        :return:
            a pandas.DataFrame of files found that match the find_name
        """
        pd.set_option('max_colwidth', 80)
        if root_dir is None:
            root_dir = os.getcwd()
        if not os.path.exists(root_dir):
            raise ValueError('The root path {} does not exist'.format(root_dir))
        if extensions is None or not extensions:
            extensions = ['csv', 'tsv', 'txt', 'xlsx', 'json', 'pickle', 'p', 'yaml']
        extensions = Commons.list_formatter(extensions)
        all_files = []
        # Get all the files in the whole directory tree and create the dataframe
        for i in Path(root_dir).rglob('*'):
            if i.is_file() and i.suffix[1:] in extensions:
                all_files.append((i.name, i.parent, i.stem, i.suffix[1:],
                                  time.ctime(i.stat().st_ctime), i.name.lower()))
        columns = ["name", "parent", "stem", "suffix", "created", "search"]
        pd.set_option('display.max_colwidth', -1)
        df = pd.DataFrame.from_records(all_files, columns=columns)
        if find_name is None:
            return df
        if ignorecase is True:
            return df[df['search'].str.contains(find_name.lower())]
        return df[df['name'].str.contains(find_name)]

    @staticmethod
    def data_dictionary(df, stylise: bool=None, inc_next_dom: bool=None, report_header: str=None,
                        condition: str=None):
        """ returns a DataFrame of a data dictionary showing 'Attribute', 'Type', '% Nulls', 'Count',
        'Unique', 'Observations' where attribute is the column names in the df
        Note that the subject_matter, if used, should be in the form:
            { subject_ref, { column_name : text_str}}
        the subject reference will be the header of the column and the text_str put in next to each attribute row

        :param df: (optional) the pandas.DataFrame to get the dictionary from
        :param stylise: (optional) returns a stylised dataframe with formatting
        :param inc_next_dom: (optional) if to include the next dominate element column
        :param report_header: (optional) filter on a header where the condition is true. Condition must exist
        :param condition: (optional) the condition to apply to the header. Header must exist. examples:
                ' > 0.95', ".str.contains('shed')"
        :return: a pandas.DataFrame
        """
        stylise = stylise if isinstance(stylise, bool) else True
        inc_next_dom = inc_next_dom if isinstance(inc_next_dom, bool) else False
        style = [{'selector': 'th', 'props': [('font-size', "120%"), ("text-align", "center")]},
                 {'selector': '.row_heading, .blank', 'props': [('display', 'none;')]}]
        pd.set_option('max_colwidth', 200)
        df_len = len(df)
        file = []
        labels = [f'Attributes ({len(df.columns)})', 'dType', '%_Null', '%_Dom', '%_Nxt', 'Count', 'Unique',
                  'Observations']
        for c in df.columns.sort_values().values:
            line = [c,
                    str(df[c].dtype),
                    round(df[c].replace('', np.nan).isnull().sum() / df_len, 3)]
            # Predominant Difference
            col = deepcopy(df[c])
            if len(col.dropna()) > 0:
                result = (col.apply(str).value_counts() /
                          np.float64(len(col.apply(str).dropna()))).sort_values(ascending=False).values
                line.append(round(result[0], 3))
                if len(result) > 1:
                    line.append(round(result[1], 3))
                else:
                    line.append(0)
            else:
                line.append(0)
                line.append(0)
            # value count
            line.append(col.apply(str).notnull().sum())
            # unique
            line.append(col.apply(str).nunique())
            # Observations
            if col.dtype.name == 'category' or col.dtype.name == 'object' or col.dtype.name == 'string':
                value_set = list(col.dropna().apply(str).value_counts().index)
                if len(value_set) > 0:
                    sample_num = 5 if len(value_set) >= 5 else len(value_set)
                    sample = str(' | '.join(value_set[:sample_num]))
                else:
                    sample = 'Null Values'
                line_str = 'Sample: {}'.format(sample)
                line.append('{}...'.format(line_str[:100]) if len(line_str) > 100 else line_str)
            elif col.dtype.name == 'bool':
                line.append(str(' | '.join(col.map({True: 'True', False: 'False'}).unique())))
            elif col.dtype.name.startswith('int') \
                    or col.dtype.name.startswith('float') \
                    or col.dtype.name.startswith('date'):
                my_str = 'max=' + str(col.max()) + ' | min=' + str(col.min())
                if col.dtype.name.startswith('date'):
                    my_str += ' | yr mean= ' + str(round(col.dt.year.mean(), 0)).partition('.')[0]
                else:
                    my_str += ' | mean=' + str(round(col.mean(), 2))
                    dominant = col.mode(dropna=True).to_list()[:2]
                    if len(dominant) == 1:
                        dominant = dominant[0]
                    my_str += ' | dominant=' + str(dominant)
                line.append(my_str)
            else:
                line.append('')
            file.append(line)
        df_dd = pd.DataFrame(file, columns=labels)
        if isinstance(report_header, str) and report_header in labels and isinstance(condition, str):
            str_value = "df_dd['{}']{}".format(report_header, condition)
            try:
                df_dd = df_dd.where(eval(str_value)).dropna()
            except(SyntaxError, ValueError):
                pass
        if not inc_next_dom:
            df_dd.drop('%_Nxt', axis='columns', inplace=True)
        if stylise:
            df_style = df_dd.style.set_table_styles(style)
            _ = df_style.applymap(DataDiscovery._highlight_null_dom, subset=['%_Null', '%_Dom'])
            _ = df_style.applymap(lambda x: 'color: white' if x > 0.98 else 'color: black', subset=['%_Null', '%_Dom'])
            _ = df_style.applymap(DataDiscovery._highlight_next, subset=['%_Nxt'])
            _ = df_style.applymap(lambda x: 'color: white' if x < 0.02 else 'color: black', subset=['%_Nxt'])
            _ = df_style.applymap(DataDiscovery._dtype_color, subset=['dType'])
            _ = df_style.applymap(DataDiscovery._color_unique, subset=['Unique'])
            _ = df_style.applymap(lambda x: 'color: white' if x < 2 else 'color: black', subset=['Unique'])
            _ = df_style.format({'%_Null': "{:.1%}", '%_Dom': '{:.1%}'})
            _ = df_style.set_caption('%_Dom: The % most dominant element ')
            _ = df_style.set_properties(subset=[f'Attributes ({len(df.columns)})'],  **{'font-weight': 'bold',
                                                                                        'font-size': "120%"})
            if inc_next_dom:
                _ = df_style.format({'%_Null': "{:.1%}", '%_Dom': '{:.1%}', '%_Nxt': '{:.1%}'})
                _ = df_style.set_caption('%_Dom: The % most dominant element - %_Nxt: The % next most dominant element')
            return df_style
        return df_dd

    @staticmethod
    def analysis2dict(header: str, dtype: str=None, lower: [int, float]=None, upper: [int, float]=None,
                      freq_precision: int=None, precision: int=None, **kwargs) -> dict:
        """ a utility method to help build analytics conditions by aligning method parameters with dictionary format.

        :param header:the header name of the value column
        :param dtype: the data type of the values
        :param lower: (optional) the lower limit of category count or numeric boundary
        :param upper: (optional) the upper limit of category count or numeric boundary
        :param freq_precision: (optional) The precision of the relative freq values. by default set to 2.
        :param precision: (optional) if numeric, sets the precision
        :param kwargs: name value pairs associated with the method that are specific to categories, numbers or dates
        :return: a dictionary for an individual element
        """
        outcome = Commons.param2dict(**locals())
        header = outcome.pop('header')
        outcome.update(outcome.pop('kwargs'))
        return {header: outcome}

    @staticmethod
    def _category_dictionary(df) -> list:
        rtn_list = []
        for c in Commons.filter_columns(df, dtype=['category']):
            df_stat = df[c].value_counts().reset_index()
            df_stat.columns = [c, 'Count']
            rtn_list.append(df_stat)
        return rtn_list

    @staticmethod
    def _dictionary_format(df, writer, sheet):
        # First set the workbook up
        workbook = writer.book

        number_fmt = workbook.add_format({'num_format': '#,##0', 'align': 'right', 'valign': 'top'})
        decimal_fmt = workbook.add_format({'num_format': '#,##0.00', 'align': 'right', 'valign': 'top'})
        percent_fmt = workbook.add_format({'num_format': '0%', 'align': 'right', 'valign': 'top'})
        attr_format = workbook.add_format({'bold': True, 'align': 'left', 'text_wrap': True, 'valign': 'top'})
        text_fmt = workbook.add_format({'align': 'left', 'text_wrap': True, 'valign': 'top'})

        header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'top', 'font_size': '12',
                                             'fg_color': '#8B0000', 'font_color': '#FFFFFF', 'border': 1})

        # The the worksheets
        worksheet = writer.sheets[sheet]

        worksheet.set_zoom(100)

        if sheet.endswith('_stat'):
            worksheet.set_column('A:A', 15, attr_format)
            letter_count = 0
            pre_letter_count = 0
            pre_letter = ''
            for index in range(1, len(df.columns)):
                letter = chr(letter_count + 65)  # A-Z => 65-91
                column = df.columns[index]
                width, _ = Commons.col_width(df, column)
                fmt = decimal_fmt if df[column].iloc[0].startswith('float') else number_fmt
                worksheet.set_column('{pre}{letter}:{pre}{letter}'.format(pre=pre_letter, letter=letter),
                                     width + 4, fmt)
                letter_count += 1
                if letter == 'Z':
                    pre_letter = chr(65 + pre_letter_count)
                    pre_letter_count += 1
                    letter_count = 0
        elif sheet.endswith('_cat'):
            for letter in range(65, 91, 3):
                worksheet.set_column('{}:{}'.format(chr(letter), chr(letter)), 20, text_fmt)
                worksheet.set_column('{}:{}'.format(chr(letter + 1), chr(letter + 1)), 8, number_fmt)
                worksheet.conditional_format('{}1:{}40'.format(chr(letter + 1), chr(letter + 1)),
                                             {'type': '3_color_scale', 'min_color': "#ecf9ec",
                                              'mid_color': "#c6ecc6", 'max_color': "#8cd98c"})
        else:
            worksheet.set_column('A:A', 20, attr_format)
            worksheet.set_column('B:B', 12, text_fmt)
            worksheet.set_column('C:C', 8, percent_fmt)
            worksheet.set_column('D:E', 10, number_fmt)
            worksheet.set_column('F:G', 80, text_fmt)
        if not sheet.endswith('_cat'):
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
        return

    @staticmethod
    def _dtype_color(dtype: str):
        """Apply color to types"""
        if str(dtype).startswith('cat'):
            color = '#208a0f'
        elif str(dtype).startswith('int'):
            color = '#0f398a'
        elif str(dtype).startswith('float'):
            color = '#2f0f8a'
        elif str(dtype).startswith('date'):
            color = '#790f8a'
        elif str(dtype).startswith('bool'):
            color = '#08488e'
        elif str(dtype).startswith('str'):
            color = '#761d38'
        else:
            return ''
        return 'color: %s' % color

    @staticmethod
    def _highlight_null_dom(x: str):
        x = float(x)
        if not isinstance(x, float) or x < 0.65:
            return ''
        elif x < 0.85:
            color = '#ffede5'
        elif x < 0.90:
            color = '#fdcdb9'
        elif x < 0.95:
            color = '#fcb499'
        elif x < 0.98:
            color = '#fc9576'
        elif x < 0.99:
            color = '#fb7858'
        elif x < 0.997:
            color = '#f7593f'
        else:
            color = '#ec382b'
        return 'background-color: %s' % color

    @staticmethod
    def _highlight_next(x: str):
        x = float(x)
        if not isinstance(x, float):
            return ''
        elif x < 0.01:
            color = '#ec382b'
        elif x < 0.02:
            color = '#f7593f'
        elif x < 0.03:
            color = '#fb7858'
        elif x < 0.05:
            color = '#fc9576'
        elif x < 0.08:
            color = '#fcb499'
        elif x < 0.12:
            color = '#fdcdb9'
        elif x < 0.18:
            color = '#ffede5'
        else:
            return ''
        return 'background-color: %s' % color

    @staticmethod
    def _color_unique(x: str):
        x = int(x)
        if not isinstance(x, int):
            return ''
        elif x < 2:
            color = '#ec382b'
        elif x < 3:
            color = '#a1cbe2'
        elif x < 5:
            color = '#84cc83'
        elif x < 10:
            color = '#a4da9e'
        elif x < 20:
            color = '#c1e6ba'
        elif x < 50:
            color = '#e5f5e0'
        elif x < 100:
            color = '#f0f9ed'
        else:
            return ''
        return 'background-color: %s' % color
