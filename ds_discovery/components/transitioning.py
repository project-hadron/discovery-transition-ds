from __future__ import annotations

import uuid
import numpy as np
import pandas as pd
from aistac.handlers.abstract_handlers import ConnectorContract
from ds_discovery.components.abstract_common_component import AbstractCommonComponent
from ds_discovery.components.commons import Commons
from ds_discovery.intent.transition_intent import TransitionIntentModel
from ds_discovery.managers.transition_property_manager import TransitionPropertyManager

__author__ = 'Darryl Oatridge'


class Transition(AbstractCommonComponent):

    REPORT_DICTIONARY = 'dictionary'
    REPORT_ANALYSIS = 'analysis'
    REPORT_FIELDS = 'field_description'
    REPORT_QUALITY = 'data_quality'
    REPORT_SUMMARY = 'data_quality_summary'
    REPORT_PROVENANCE = 'provenance'

    @classmethod
    def from_uri(cls, task_name: str, uri_pm_path: str, username: str, uri_pm_repo: str=None, pm_file_type: str=None,
                 pm_module: str=None, pm_handler: str=None, pm_kwargs: dict=None, default_save=None,
                 reset_templates: bool=None, template_path: str=None, template_module: str=None,
                 template_source_handler: str=None, template_persist_handler: str=None, align_connectors: bool=None,
                 default_save_intent: bool=None, default_intent_level: bool=None, order_next_available: bool=None,
                 default_replace_intent: bool=None, has_contract: bool=None) -> Transition:
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
        pm_module = pm_module if isinstance(pm_module, str) else cls.DEFAULT_MODULE
        pm_handler = pm_handler if isinstance(pm_handler, str) else cls.DEFAULT_PERSIST_HANDLER
        _pm = TransitionPropertyManager(task_name=task_name, username=username)
        _intent_model = TransitionIntentModel(property_manager=_pm, default_save_intent=default_save_intent,
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
    def scratch_pad(cls) -> TransitionIntentModel:
        """ A class method to use the Components intent methods as a scratch pad"""
        return super().scratch_pad()

    @property
    def intent_model(self) -> TransitionIntentModel:
        """The intent model instance"""
        return self._intent_model

    @property
    def cleaners(self) -> TransitionIntentModel:
        """The intent model instance"""
        return self._intent_model

    @property
    def pm(self) -> TransitionPropertyManager:
        """The properties manager instance"""
        return self._component_pm

    def set_provenance(self, title: str=None, domain: str=None, description: str=None, license_type: str=None,
                       license_name: str=None, license_uri: str=None, cost_price: str = None, cost_code: str = None,
                       cost_type: str = None, provider_name: str=None, provider_uri: str=None, provider_note: str=None,
                       author_name: str=None, author_uri: str=None, author_contact: str=None, save: bool=None):
        """sets the provenance values. Only sets those passed

        :param title: (optional) the title of the provenance
        :param domain: (optional) the domain it sits within
        :param description: (optional) a description of the provenance
        :param license_type: (optional) The type of the license. Default 'ODC-By'
        :param license_name: (optional) The full name of the license. Default 'Open Data Commons Attribution License'
        :param license_uri: (optional) The license uri. Default https://opendatacommons.org/licenses/by/
        :param cost_price: (optional) a cost price associated with this provenance
        :param cost_code: (optional) a cost centre code or reference code
        :param cost_type: (optional) the cost type or description
        :param provider_name: (optional) the provider system or institution name or title
        :param provider_uri: (optional) a uri reference that helps identify the provider
        :param provider_note: (optional) any notes that might be useful
        :param author_name: (optional) the author of the data
        :param author_uri: (optional) the author uri
        :param author_contact: (optional)the the author contact information
        :param save: (optional) if True, save to file. Default is True
        """
        license_type = license_type if license_type else 'PDDL'
        license_name = license_name if license_name else 'Open Data Commons Attribution License'
        license_uri = license_uri if license_uri else 'https://opendatacommons.org/licenses/pddl/summary'

        self.pm.set_provenance(title=title, domain=domain, description=description, license_type=license_type,
                               license_name=license_name, license_uri=license_uri, cost_price=cost_price,
                               cost_code=cost_code, cost_type=cost_type, provider_name=provider_name,
                               provider_uri=provider_uri, provider_note=provider_note, author_name=author_name,
                               author_uri=author_uri, author_contact=author_contact)
        self.pm_persist(save=save)

    def reset_provenance(self, save: bool=None):
        """resets the provenance back to its default values"""
        self.pm.reset_provenance()
        self.pm_persist(save)

    def save_quality_report(self, canonical: pd.DataFrame=None, file_type: str=None, versioned: bool=None,
                            stamped: str=None):
        """ Generates and persists the data quality

        :param canonical: the canonical to base the report on
        :param file_type: (optional) an alternative file extension to the default 'json' format
        :param versioned: (optional) if the component version should be included as part of the pattern
        :param stamped: (optional) A string of the timestamp options ['days', 'hours', 'minutes', 'seconds', 'ns']
        :return:
        """
        if isinstance(file_type, str) or isinstance(versioned, bool) or isinstance(stamped, str):
            file_pattern = self.pm.file_pattern(self.REPORT_QUALITY, file_type=file_type, versioned=versioned,
                                                stamped=stamped)
            self.set_report_persist(self.REPORT_QUALITY, uri_file=file_pattern)
        report = self.report_quality(canonical=canonical)
        self.save_report_canonical(reports=self.REPORT_QUALITY, report_canonical=report, auto_connectors=True)
        return

    def run_component_pipeline(self, intent_levels: [str, int, list]=None, run_book: str=None):
        """Runs the components pipeline from source to persist"""
        canonical = self.load_source_canonical()
        result = self.intent_model.run_intent_pipeline(canonical, intent_levels=intent_levels, run_book=run_book,
                                                       inplace=False)
        self.save_persist_canonical(result)

    def report_attributes(self, canonical, stylise: bool=True):
        """ generates a report on the attributes and any description provided

        :param canonical: the canonical to report on
        :param stylise: if True present the report stylised.
        :return: pd.DataFrame
        """
        labels = [f'Attributes ({len(canonical.columns)})', 'dType', 'Description']
        file = []
        for c in canonical.columns.sort_values().values:
            line = [c, str(canonical[c].dtype),
                    ". ".join(self.pm.report_notes(catalog='attributes', labels=c, drop_dates=True).get('text', []))]
            file.append(line)
        df_dd = pd.DataFrame(file, columns=labels)
        if stylise:
            style = [{'selector': 'th', 'props': [('font-size', "120%"), ("text-align", "center")]},
                     {'selector': '.row_heading, .blank', 'props': [('display', 'none;')]}]
            df_style = df_dd.style.set_table_styles(style)
            _ = df_style.applymap(self._dtype_color, subset=['dType'])
            _ = df_style.set_properties(subset=['Description'],  **{"text-align": "left"})
            _ = df_style.set_properties(subset=[f'Attributes ({len(canonical.columns)})'], **{'font-weight': 'bold',
                                                                                              'font-size': "120%"})
            return df_style
        return df_dd

    def report_provenance(self, as_dict: bool=None, stylise: bool=None):
        """ a report on the provenance set as part of the domain contract

        :param as_dict: (optional) if the result should be a dictionary. Default is False
        :param stylise: (optional) if as_dict is False, if the return dataFrame should be stylised
        :return:
        """
        as_dict = as_dict if isinstance(as_dict, bool) else False
        stylise = stylise if isinstance(stylise, bool) else True
        report = self.pm.report_provenance()
        if as_dict:
            return report
        df = pd.DataFrame(report, index=['values'])
        df = df.transpose().reset_index()
        df.columns = ['provenance', 'values']
        if stylise:
            return Commons.report(df, index_header='provenance')
        return df

    def report_quality_summary(self, canonical: pd.DataFrame=None, as_dict: bool=None, stylise: bool=None):
        """ a summary quality report of the canonical

        :param canonical: (optional) the canonical to be sumarised. If not passed then loads the canonical source
        :param as_dict: (optional) if the result should be a dictionary. Default is False
        :param stylise: (optional) if as_dict is False, if the return dataFrame should be stylised
        :return: a dict or pd.DataFrame
        """
        as_dict = as_dict if isinstance(as_dict, bool) else False
        stylise = stylise if isinstance(stylise, bool) else True
        if not isinstance(canonical, pd.DataFrame):
            canonical = self._auto_transition()
        # provinance
        _provenance_headers = ['title', 'license', 'domain', 'description', 'provider',  'author', 'cost']
        _provenance_count = len(list(filter(lambda x: x in _provenance_headers, self.pm.provenance.keys())))
        _provenance_cost = self.pm.provenance.get('cost', {}).get('price', 'NA')
        # descibed
        _descibed_keys = self.pm.get_knowledge(catalog='attributes').keys()
        _descibed_count = len(list(filter(lambda x: x in canonical.columns, _descibed_keys)))
        # dictionary
        _dictionary = self.canonical_report(canonical, stylise=False)
        _total_fields = _dictionary.shape[0]
        _null_total = _dictionary['%_Null'].sum()
        _dom_fields = _dictionary['%_Dom'].sum()

        _null_columns = _dictionary['%_Null'].where(_dictionary['%_Null'] > 0.98).dropna()
        _dom_columns = _dictionary['%_Dom'].where(_dictionary['%_Dom'] > 0.98).dropna()
        _usable_fields = set(_null_columns)
        _usable_fields.update(_dom_columns)
        _numeric_fields = len(Commons.filter_headers(canonical, dtype='number'))
        _category_fields = len(Commons.filter_headers(canonical, dtype='category'))
        _date_fields = len(Commons.filter_headers(canonical, dtype='datetime'))
        _bool_fields = len(Commons.filter_headers(canonical, dtype='bool'))
        _other_fields = len(Commons.filter_headers(canonical, dtype=['category', 'datetime', 'bool',
                                                                     'number'],  exclude=True))
        _null_avg = _null_total / canonical.shape[1]
        _dom_avg = _dom_fields / canonical.shape[1]
        _quality_avg = int(round(100 - (((_null_avg + _dom_avg)/2)*100), 0))
        _correlated = self._correlated_columns(canonical)
        _usable = int(round(100 - (len(_usable_fields) / canonical.columns.size) * 100, 2))
        _field_avg = int(round(_descibed_count / canonical.shape[1] * 100, 0))
        _prov_avg = int(round(_provenance_count/len(_provenance_headers)*100, 0))
        _adjustments = self.report_intent(stylise=False).intent.size
        report = {'score': {'quality_avg': f"{_quality_avg}%", 'usability_avg': f"{_usable}%",
                            'provenance_complete': f"{_prov_avg}%", 'data_described': f"{_field_avg}%"},
                  'data_shape': {'rows': canonical.shape[0], 'columns': canonical.shape[1],
                                 'memory': Commons.bytes2human(canonical.memory_usage(deep=True).sum())},
                  'data_type': {'numeric': _numeric_fields, 'category': _category_fields,
                                'datetime': _date_fields, 'bool': _bool_fields,
                                'others': _other_fields},
                  'usability': {'mostly_null': len(_null_columns),
                                'predominance': len(_dom_columns),
                                'correlated': len(_correlated),
                                'adjustments': _adjustments},
                  'cost': {'price': _provenance_cost}}
        if as_dict:
            return report
        df = pd.DataFrame(columns=['report', 'summary', 'result'])
        counter = 0
        for index, values in report.items():
            for summary, result in values.items():
                df.loc[counter] = [index, summary, result]
                counter += 1
        if stylise:
            Commons.report(df, index_header='report', bold='summary')
        return df

    def report_quality(self, canonical: pd.DataFrame=None) -> dict:
        """A complete report of the components"""
        if not isinstance(canonical, pd.DataFrame):
            canonical = self._auto_transition()
        # meta
        report = {'meta-data': {'uid': str(uuid.uuid4()),
                                'created': str(pd.Timestamp.now()),
                                'creator': self.pm.username},
                  'description': self.pm.description,
                  'summary': self.report_quality_summary(canonical, as_dict=True)}
        # connectors
        _connectors = {}
        for connector in self.pm.connector_contract_list:
            if connector.startswith('pm_transition') or connector.startswith('template_'):
                continue
            _connector = self.pm.get_connector_contract(connector_name=connector)
            _connector_dict = {}
            if isinstance(_connector, ConnectorContract):
                kwargs = ''
                if isinstance(_connector.raw_kwargs, dict):
                    for k, v in _connector.raw_kwargs.items():
                        if len(kwargs) > 0:
                            kwargs += "  "
                        kwargs += f"{k}='{v}'"
                query = ''
                if isinstance(_connector.query, dict):
                    for k, v in _connector.query.items():
                        if len(query) > 0:
                            query += "  "
                        query += f"{k}='{v}'"
                _connector_dict['uri'] = _connector.raw_uri
                _connector_dict['version'] = _connector.version
                if len(kwargs) > 0:
                    _connector_dict['kwargs'] = kwargs
                if len(query) > 0:
                    _connector_dict['query'] = query
            _connectors[connector] = _connector_dict
        report['connectors'] = _connectors
        # provenance
        report['provenance'] = self.pm.provenance
        _provenance_headers = ['title', 'license', 'domain', 'description', 'provider',  'author', 'cost']
        _provenance_count = len(list(filter(lambda x: x in _provenance_headers, self.pm.provenance.keys())))
        # fields
        _field_count = 0
        _fields = {}
        for label, items in self.pm.get_knowledge(catalog='attributes').items():
            _fields[label] = Commons.list_formatter(items.values())
            if label in canonical.columns:
                _field_count += 1
        report['attributes'] = _fields
        # dictionary
        _data_dict = {}
        for _, row in self.canonical_report(canonical, stylise=False).iterrows():
            _data_dict[row.iloc[0]] = {}
            _att_name = None
            for index in row.index:
                if index.startswith('Attribute'):
                    _att_name = row.loc[index]
                    continue
                _data_dict[row.iloc[0]].update({index: row.loc[index]})
        report['dictionary'] = _data_dict
        # notes
        _observations = {}
        for label, items in self.pm.get_knowledge(catalog='observations').items():
            _observations[label] = Commons.list_formatter(items.values())
        _actions = {}
        for label, items in self.pm.get_knowledge(catalog='actions').items():
            _actions[label] = Commons.list_formatter(items.values())
        report['notes'] = {'observations': _observations,
                           'actions': _actions}
        return report

    def upload_attributes(self, canonical: pd.DataFrame, label_key: str, text_key: str, constraints: list=None,
                          save=None):
        """ Allows bulk upload of notes. Assumes a dictionary of key value pairs where the key is the
        label and the value the text

        :param canonical: a DataFrame of where the key is the label and value is the text
        :param label_key: the dictionary key name for the labels
        :param text_key: the dictionary key name for the text
        :param constraints: (optional) the limited list of acceptable labels. If not in list then ignored
        :param save: if True, save to file. Default is True
        """
        super().upload_notes(canonical=canonical.to_dict(orient='list'), catalog='attributes', label_key=label_key,
                             text_key=text_key, constraints=constraints, save=save)

    def upload_notes(self, canonical: pd.DataFrame, catalog: str, label_key: str, text_key: str, constraints: list=None,
                     save=None):
        """ Allows bulk upload of notes. Assumes a dictionary of key value pairs where the key is the
        label and the value the text

        :param canonical: a DataFrame of where the key is the label and value is the text
        :param catalog: the section these notes should be put in
        :param label_key: the dictionary key name for the labels
        :param text_key: the dictionary key name for the text
        :param constraints: (optional) the limited list of acceptable labels. If not in list then ignored
        :param save: if True, save to file. Default is True
        """
        super().upload_notes(canonical=canonical.to_dict(orient='list'), catalog=catalog, label_key=label_key,
                             text_key=text_key, constraints=constraints, save=save)

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

    def _correlated_columns(self, canonical: pd.DataFrame):
        """returns th percentage of useful colums"""
        threshold = 0.98
        pad = self.scratch_pad()
        canonical = pad.auto_to_category(canonical, unique_max=1000, inplace=False)
        canonical = pad.to_category_type(canonical, dtype='category', as_num=True)
        for c in canonical.columns:
            if all(Commons.valid_date(x) for x in canonical[c].dropna()):
                canonical = pad.to_date_type(canonical, dtype='datetime', as_num=True)
        canonical = Commons.filter_columns(canonical, dtype=['number'], exclude=False)
        for c in canonical.columns:
            canonical[c] = Commons.fillna(canonical[c])
        col_corr = set()
        corr_matrix = canonical.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:  # we are interested in absolute coeff value
                    colname = corr_matrix.columns[i]  # getting the name of column
                    col_corr.add(colname)
        return col_corr

    def _auto_transition(self) -> pd.DataFrame:
        """ attempts auto components on a canonical """
        pad = self.scratch_pad()
        if not self.pm.has_connector(self.CONNECTOR_SOURCE):
            raise ConnectionError("Unable to load Source canonical as the Source Connector has not been set")
        canonical = self.load_source_canonical()
        unique_max = np.log2(canonical.shape[0]) ** 2 if canonical.shape[0] > 50000 else np.sqrt(canonical.shape[0])
        return pad.auto_transition(canonical, unique_max=unique_max)
