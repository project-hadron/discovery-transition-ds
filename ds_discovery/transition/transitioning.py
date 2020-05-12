import pandas as pd
import uuid
import ds_discovery
from aistac.handlers.abstract_handlers import ConnectorContract
from aistac.components.abstract_component import AbstractComponent
from ds_discovery.intent.transition_intent import TransitionIntentModel
from ds_discovery.managers.transition_property_manager import TransitionPropertyManager
from ds_discovery.transition.commons import Commons
from ds_discovery.transition.discovery import DataDiscovery, Visualisation

__author__ = 'Darryl Oatridge'


class Transition(AbstractComponent):

    CONNECTOR_SOURCE = 'primary_source'
    CONNECTOR_PERSIST = 'primary_persist'
    CONNECTOR_DICTIONARY = 'dictionary'
    CONNECTOR_INSIGHT = 'insight'
    CONNECTOR_INTENT = 'intent'
    CONNECTOR_FIELDS = 'field_description'
    CONNECTOR_NUTRITION = 'nutrition'

    def __init__(self, property_manager: TransitionPropertyManager, intent_model: TransitionIntentModel,
                 default_save=None, reset_templates: bool=None, align_connectors: bool=None):
        """ Encapsulation class for the transition set of classes

        :param property_manager: The contract property manager instance for this component
        :param intent_model: the model codebase containing the parameterizable intent
        :param default_save: The default behaviour of persisting the contracts:
                    if False: The connector contracts are kept in memory (useful for restricted file systems)
        :param reset_templates: (optional) reset connector templates from environ variables (see `report_environ()`)
        :param align_connectors: (optional) resets aligned connectors to the template
        """
        super().__init__(property_manager=property_manager, intent_model=intent_model, default_save=default_save,
                         reset_templates=reset_templates, align_connectors=align_connectors)
        self._raw_attribute_list = []

    @classmethod
    def from_uri(cls, task_name: str, uri_pm_path: str, username: str, pm_file_type: str=None, pm_module: str=None,
                 pm_handler: str=None, pm_kwargs: dict=None, default_save=None, reset_templates: bool=None,
                 align_connectors: bool=None, default_save_intent: bool=None, default_intent_level: bool=None,
                 order_next_available: bool=None, default_replace_intent: bool=None):
        """ Class Factory Method to instantiates the components application. The Factory Method handles the
        instantiation of the Properties Manager, the Intent Model and the persistence of the uploaded properties.
        See class inline docs for an example method

         :param task_name: The reference name that uniquely identifies a task or subset of the property manager
         :param uri_pm_path: A URI that identifies the resource path for the property manager.
         :param username: A user name for this task activity.
         :param pm_file_type: (optional) defines a specific file type for the property manager
         :param pm_module: (optional) the module or package name where the handler can be found
         :param pm_handler: (optional) the handler for retrieving the resource
         :param pm_kwargs: (optional) a dictionary of kwargs to pass to the property manager
         :param default_save: (optional) if the configuration should be persisted. default to 'True'
         :param reset_templates: (optional) reset connector templates from environ variables. Default True
                                (see `report_environ()`)
         :param align_connectors: (optional) resets aligned connectors to the template. default Default True
         :param default_save_intent: (optional) The default action for saving intent in the property manager
         :param default_intent_level: (optional) the default level intent should be saved at
         :param order_next_available: (optional) if the default behaviour for the order should be next available order
         :param default_replace_intent: (optional) the default replace existing intent behaviour
         :return: the initialised class instance
         """
        pm_file_type = pm_file_type if isinstance(pm_file_type, str) else 'json'
        pm_module = pm_module if isinstance(pm_module, str) else 'ds_discovery.handlers.pandas_handlers'
        pm_handler = pm_handler if isinstance(pm_handler, str) else 'PandasPersistHandler'
        _pm = TransitionPropertyManager(task_name=task_name, username=username)
        _intent_model = TransitionIntentModel(property_manager=_pm, default_save_intent=default_save_intent,
                                              default_intent_level=default_intent_level,
                                              order_next_available=order_next_available,
                                              default_replace_intent=default_replace_intent)
        super()._init_properties(property_manager=_pm, uri_pm_path=uri_pm_path, pm_file_type=pm_file_type,
                                 pm_module=pm_module, pm_handler=pm_handler, pm_kwargs=pm_kwargs)
        return cls(property_manager=_pm, intent_model=_intent_model, default_save=default_save,
                   reset_templates=reset_templates, align_connectors=align_connectors)

    @classmethod
    def _from_remote_s3(cls) -> (str, str):
        """ Class Factory Method that builds the connector handlers for Amazon AWS S3 """
        _module_name = 'ds_connectors.handlers.aws_s3_handlers'
        _handler = 'AwsS3PersistHandler'
        return _module_name, _handler

    @classmethod
    def _from_remote_redis(cls) -> (str, str):
        """ Class Factory Method that builds the connector handlers for Redis."""
        _module_name = 'ds_connectors.handlers.redis_handlers'
        _handler = 'RedisPersistHandler'
        return _module_name, _handler

    @classmethod
    def scratch_pad(cls) -> TransitionIntentModel:
        """ A class method to use the Components intent methods as a scratch pad"""
        return super().scratch_pad()

    @classmethod
    def discovery_pad(cls) -> DataDiscovery:
        """ A class method to use the Components discovery methods as a scratch pad"""
        return DataDiscovery()

    @classmethod
    def visual_pad(cls) -> Visualisation:
        """ A class method to use the Components visualisation methods as a scratch pad"""
        return Visualisation()

    @property
    def intent_model(self) -> TransitionIntentModel:
        """The intent model instance"""
        return self._intent_model

    @property
    def pm(self) -> TransitionPropertyManager:
        """The properties manager instance"""
        return self._component_pm

    @property
    def discover(self) -> DataDiscovery:
        """The transition instance"""
        return DataDiscovery()

    @property
    def visual(self) -> Visualisation:
        """The visualisation instance"""
        return Visualisation()

    def set_provenance(self, title: str=None, domain: str=None, description: str=None, usage_license: str=None,
                       provider_name: str=None, provider_uri: str=None, provider_note: str=None,
                       author_name: str=None, author_uri: str=None, author_contact: str=None,
                       save: bool=None):
        """sets the provenance values. Only sets those passed

        :param title: (optional) the title of the provenance
        :param domain: (optional) the domain it sits within
        :param description: (optional) a description of the provenance
        :param usage_license: (optional) any associated usage licensing
        :param provider_name: (optional) the provider system or institution name or title
        :param provider_uri: (optional) a uri reference that helps identify the provider
        :param provider_note: (optional) any notes that might be useful
        :param author_name: (optional) the author of the data
        :param author_uri: (optional) the author uri
        :param author_contact: (optional)the the author contact information
        :param save: (optional) if True, save to file. Default is True
        """
        self.pm.set_provenance(title=title, domain=domain, description=description, usage_license=usage_license,
                               provider_name=provider_name, provider_uri=provider_uri, provider_note=provider_note,
                               author_name=author_name, author_uri=author_uri, author_contact=author_contact)
        self.pm_persist(save=save)

    def reset_provenance(self, save: bool=None):
        """resets the provenance back to its default values"""
        self.pm.reset_provenance()
        self.pm_persist(save)

    def set_insight(self, blueprint: dict, endpoints: list=None, save: bool=None):
        """sets the insight analysis parameters

        :param blueprint: the analysis blueprint of what to analysis and how
        :param endpoints: (optional) the endpoints, if any, where the tree ends
        :param save: (optional) if True, save to file. Default is True
        """
        self.pm.set_insight(blueprint=blueprint, endpoints=endpoints)
        self.pm_persist(save)

    def reset_insight(self, save: bool=None):
        """resets the insight back to its default"""
        self.pm.reset_insight()
        self.pm_persist(save)

    def is_source_modified(self):
        """Test if the source file is modified since last load"""
        return self.pm.is_connector_modified(self.CONNECTOR_SOURCE)

    def get_persist_contract(self):
        """ gets the persist connector contract that can be used as the next chain source"""
        return self.pm.get_connector_contract(self.CONNECTOR_PERSIST)

    def set_source_contract(self, connector_contract: ConnectorContract, save: bool=None):
        """ Sets the source contract

        :param connector_contract: a Connector Contract for the source data
        :param save: (optional) if True, save to file. Default is True
        """
        self.add_connector_contract(self.CONNECTOR_SOURCE, connector_contract=connector_contract, save=save)
        return

    def set_persist_contract(self, connector_contract: ConnectorContract, save: bool=None):
        """ Sets the persist contract.

        :param connector_contract: a Connector Contract for the persisted data
        :param save: (optional) if True, save to file. Default is True
        """
        self.add_connector_contract(self.CONNECTOR_PERSIST, connector_contract=connector_contract, save=save)
        return

    def set_source(self, uri_file: str, save: bool=None, **kwargs):
        """sets the source contract CONNECTOR_SOURCE using the TEMPLATE_SOURCE connector contract,

        :param uri_file: the uri_file is appended to the template path
        :param save: (optional) if True, save to file. Default is True
        """
        self.add_connector_from_template(connector_name=self.CONNECTOR_SOURCE, uri_file=uri_file,
                                         template_name=self.TEMPLATE_SOURCE, save=save, **kwargs)

    def set_persist(self, uri_file: str=None, save: bool=None, **kwargs):
        """sets the persist contract CONNECTOR_PERSIST using the TEMPLATE_PERSIST connector contract

        :param uri_file: (optional) the uri_file is appended to the template path
        :param save: (optional) if True, save to file. Default is True
        """
        file_pattern = self.pm.file_pattern(connector_name=self.CONNECTOR_PERSIST)
        uri_file = uri_file if isinstance(uri_file, str) else file_pattern
        self.add_connector_from_template(connector_name=self.CONNECTOR_PERSIST, uri_file=uri_file,
                                         template_name=self.TEMPLATE_PERSIST, save=save, **kwargs)

    def set_report_persist(self, report_connector_name, uri_file: str=None, save: bool=None, **kwargs):
        """sets the report persist the TEMPLATE_PERSIST connector contract, there are preset constants that should be
        used

        :param report_connector_name: the name of the report connector to set (see class CONNECTOR constants)
        :param uri_file: (optional) the uri_file is appended to the template path
        :param save: (optional) if True, save to file. Default is True
        """
        if report_connector_name not in [self.CONNECTOR_DICTIONARY, self.CONNECTOR_INSIGHT, self.CONNECTOR_INTENT,
                                         self.CONNECTOR_NUTRITION, self.CONNECTOR_FIELDS]:
            raise ValueError("Report name must be one of the class report constants")
        file_pattern = self.pm.file_pattern(connector_name=report_connector_name, file_type='json', versioned=True)
        uri_file = uri_file if isinstance(uri_file, str) else file_pattern
        self.add_connector_from_template(connector_name=report_connector_name, uri_file=uri_file,
                                         template_name=self.TEMPLATE_PERSIST, save=save, **kwargs)

    def load_source_canonical(self, **kwargs) -> pd.DataFrame:
        """returns the contracted source data as a DataFrame """
        return self.load_canonical(self.CONNECTOR_SOURCE, **kwargs)

    def load_clean_canonical(self, **kwargs) -> pd.DataFrame:
        """loads the clean pandas.DataFrame from the clean folder for this contract"""
        return self.load_canonical(self.CONNECTOR_PERSIST, **kwargs)

    def load_canonical(self, connector_name: str, **kwargs) -> pd.DataFrame:
        """returns the canonical of the referenced connector

        :param connector_name: the name or label to identify and reference the connector
        """
        canonical = super().load_canonical(connector_name=connector_name, **kwargs)
        if isinstance(canonical, dict):
            canonical = pd.DataFrame.from_dict(data=canonical, orient='columns')
        return canonical

    def save_clean_canonical(self, df, **kwargs):
        """Saves the pandas.DataFrame to the clean files folder"""
        self.persist_canonical(connector_name=self.CONNECTOR_PERSIST, canonical=df, **kwargs)

    def save_report_canonical(self, report_connector_name, report: dict, **kwargs):
        """Saves the pandas.DataFrame to the nutrition folder"""
        if report_connector_name not in [self.CONNECTOR_DICTIONARY, self.CONNECTOR_INSIGHT, self.CONNECTOR_INTENT,
                                         self.CONNECTOR_NUTRITION, self.CONNECTOR_FIELDS]:
            raise ValueError("Report name must be one of the class report constants")
        self.persist_canonical(connector_name=report_connector_name, canonical=report, **kwargs)

    def run_transition_pipeline(self, intent_levels: [str, int, list]=None):
        """Runs the transition pipeline from source to persist"""
        canonical = self.load_source_canonical()
        result = self.intent_model.run_intent_pipeline(canonical, intent_levels=intent_levels, inplace=False)
        self.save_clean_canonical(result)

    def canonical_report(self, df, stylise: bool=True, inc_next_dom: bool=False, report_header: str=None,
                         condition: str=None):
        """The Canonical Report is a data dictionary of the canonical providing a reference view of the dataset's
        attribute properties

        :param df: the DataFrame to view
        :param stylise: if True present the report stylised.
        :param inc_next_dom: (optional) if to include the next dominate element column
        :param report_header: (optional) filter on a header where the condition is true. Condition must exist
        :param condition: (optional) the condition to apply to the header. Header must exist. examples:
                ' > 0.95', ".str.contains('shed')"
        :return:
        """
        return self.discover.data_dictionary(df=df, stylise=stylise, inc_next_dom=inc_next_dom,
                                             report_header=report_header, condition=condition)

    def report_attributes(self, df, stylise: bool=True):
        labels = [f'Attributes ({len(df.columns)})', 'dType', 'Description']
        file = []
        for c in df.columns.sort_values().values:
            line = [c, str(df[c].dtype),
                    ". ".join(self.pm.report_notes(catalog='attributes', labels=c, drop_dates=True).get('text', []))]
            file.append(line)
        df_dd = pd.DataFrame(file, columns=labels)
        if stylise:
            style = [{'selector': 'th', 'props': [('font-size', "120%"), ("text-align", "center")]},
                     {'selector': '.row_heading, .blank', 'props': [('display', 'none;')]}]
            df_style = df_dd.style.set_table_styles(style)
            _ = df_style.applymap(self._dtype_color, subset=['dType'])
            _ = df_style.set_properties(subset=['Description'],  **{"text-align": "left"})
            _ = df_style.set_properties(subset=[f'Attributes ({len(df.columns)})'],  **{'font-weight': 'bold',
                                                                                        'font-size': "120%"})
            return df_style
        df_dd.set_index(keys=f'Attributes ({len(df.columns)})', inplace=True)
        return df_dd

    def report_connectors(self, connector_filter: [str, list]=None, stylise: bool=True):
        """ generates a report on the source contract

        :param connector_filter: (optional) filters on the connector name.
        :param stylise: (optional) returns a stylised DataFrame with formatting
        :return: pd.DataFrame
        """
        df = pd.DataFrame.from_dict(data=self.pm.report_connectors(connector_filter=connector_filter), orient='columns')
        if stylise:
            Commons.report(df, index_header='connector_name')
        df.set_index(keys='connector_name', inplace=True)
        return df

    def report_run_book(self, stylise: bool=True):
        """ generates a report on all the intent

        :param stylise: returns a stylised dataframe with formatting
        :return: pd.Dataframe
        """
        df = pd.DataFrame.from_dict(data=self.pm.report_run_book(), orient='columns')
        if stylise:
            Commons.report(df, index_header='name')
        df.set_index(keys='name', inplace=True)
        return df

    def report_intent(self, stylise: bool=True):
        """ generates a report on all the intent

        :param stylise: returns a stylised dataframe with formatting
        :return: pd.Dataframe
        """
        df = pd.DataFrame.from_dict(data=self.pm.report_intent(), orient='columns')
        if stylise:
            Commons.report(df, index_header='level')
        df.set_index(keys='level', inplace=True)
        return df

    def report_notes(self, catalog: [str, list]=None, labels: [str, list]=None, regex: [str, list]=None,
                     re_ignore_case: bool=False, stylise: bool=True, drop_dates: bool=False):
        """ generates a report on the notes

        :param catalog: (optional) the catalog to filter on
        :param labels: (optional) s label or list of labels to filter on
        :param regex: (optional) a regular expression on the notes
        :param re_ignore_case: (optional) if the regular expression should be case sensitive
        :param stylise: (optional) returns a stylised dataframe with formatting
        :param drop_dates: (optional) excludes the 'date' column from the report
        :return: pd.Dataframe
        """
        report = self.pm.report_notes(catalog=catalog, labels=labels, regex=regex, re_ignore_case=re_ignore_case,
                                      drop_dates=drop_dates)
        df = pd.DataFrame.from_dict(data=report, orient='columns')
        if stylise:
            Commons.report(df, index_header='section', bold='label')
        df.set_index(keys='section', inplace=True)
        return df

    def report_provenance(self, stylise: bool=True):
        report = self.pm.report_provenance()
        df = pd.DataFrame(report, index=['values'])
        df = df.transpose().reset_index()
        df.columns = ['provenance', 'values']
        if stylise:
            Commons.report(df, index_header='provenance')
        df.set_index(keys='provenance', inplace=True)
        return df

    def report_nutrition(self, df: pd.DataFrame=None, granularity: [int, float, list]=None, top: int=None) -> dict:
        """A complete report of the transition"""
        if not isinstance(df, pd.DataFrame):
            df = self.load_source_canonical()
            pad: TransitionIntentModel = self.scratch_pad()
            _date_headers = []
            _bool_headers = []
            for c in Commons.filter_headers(df, dtype=object):
                if all(Commons.valid_date(x) for x in df[c].dropna()):
                    _date_headers.append(c)
                if df[c].nunique() == 2:
                    values = df[c].value_counts().index.to_list()
                    if any(x in ['True', 1, 'Y', 'y', 'Yes', 'yes'] for x in values):
                        _bool_headers.append(c)
            if len(_bool_headers) > 0:
                bool_map = {1: True}
                df = pad.to_bool_type(df, headers=_bool_headers, inplace=False, bool_map=bool_map)
            if len(_date_headers) > 0:
                df = pad.to_date_type(df, headers=_date_headers, inplace=False)
            df = pad.to_numeric_type(df, dtype='number', inplace=False)
            df = pad.to_category_type(df, dtype='object', inplace=False)
        # meta
        report = {'_meta-data': {'uid': str(uuid.uuid4()),
                                 'created': str(pd.Timestamp.now())}}
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
        _provenance_count = 0
        for item in self.pm.provenance.keys():
            if item in ['title', 'license', 'domain', 'description', 'provider', 'author']:
                _provenance_count += 1
        # fields
        _field_count = 0
        _fields = {}
        for label, items in self.pm.get_knowledge(catalog='attributes').items():
            _fields[label] = Commons.list_formatter(items.values())
            if label in df.columns:
                _field_count += 1
        report['fields'] = _fields
        # dictionary
        _null_total = 0
        _dom_fields = 0
        _numeric_fields = 0
        _category_fields = 0
        _date_fields = 0
        _bool_fields = 0
        _other_fields = 0
        _data_dict = {}
        _null_columns = set()
        _dom_columns = set()
        _date_columns = set()
        _numeric_columns = set()
        for _, row in self.canonical_report(df, stylise=False).iterrows():
            _data_dict[row.iloc[0]] = {}
            _att_name = None
            for index in row.index:
                if index.startswith('Attribute'):
                    _att_name = row.loc[index]
                    continue
                if index.startswith('%_Null'):
                    _null_total += row.loc[index]
                    if row.loc[index] > 0.98:
                        _null_columns.add(_att_name)
                if index.startswith('%_Dom') and row.loc[index] > 0.98:
                    _dom_columns.add(_att_name)
                    _dom_fields += 1
                if index.startswith('dType'):
                    if row.loc[index].startswith('int') or row.loc[index].startswith('float'):
                        _numeric_fields += 1
                        _numeric_columns.add(_att_name)
                    elif row.loc[index].startswith('category'):
                        _category_fields += 1
                    elif row.loc[index].startswith('date'):
                        _date_fields += 1
                        _date_columns.add(_att_name)
                    elif row.loc[index].startswith('bool'):
                        _bool_fields += 1
                    else:
                        if any(Commons.valid_date(x) for x in df[_att_name]):
                            _date_fields += 1
                            _date_columns.add(_att_name)
                        elif df[_att_name].nunique() < 100 and df[df[_att_name].isnull()].size < (df.size * 0.8):
                            _category_fields += 1
                        else:
                            _other_fields += 1
                _data_dict[row.iloc[0]].update({index: row.loc[index]})
        report['dictionary'] = _data_dict
        # notes
        _notes = {}
        for label, items in self.pm.get_knowledge(catalog='transition').items():
            _notes[label] = Commons.list_formatter(items.values())

        report['transition'] = {'description': self.pm.description,
                                'notes': _notes}
        _null_avg = _null_total / len(df.columns)
        _dom_avg = _dom_fields / len(df.columns)
        _quality_avg = int(round(100 - (((_null_avg + _dom_avg)/2)*100), 0))
        _correlated = self._correlated_columns(df)
        _usable = set(_null_columns).union(_dom_columns).union(_correlated)
        _usable = int(round(100 - (len(_usable)/df.columns.size) * 100, 2))
        _field_avg = int(round(_field_count/len(df.columns)*100, 0))
        _prov_avg = int(round(_provenance_count/6*100, 0))
        report['Scores %'] = {'quality avg': _quality_avg, 'usability avg': _usable,
                              'provenance complete': _prov_avg, 'data described': _field_avg}
        report['Versions'] = {'component': {'name': self.pm.manager_name(),
                                            'version': ds_discovery.__version__},
                              'task': {'name': self.pm.task_name,
                                       'version': self.pm.version},
                              'source': {'name': self.CONNECTOR_SOURCE,
                                         'version': self.pm.get_connector_contract(self.CONNECTOR_SOURCE).version}}
        # Raw Measures
        report['Raw Measures'] = {'data_shape': {'rows': df.shape[0], 'columns': df.shape[1],
                                                 'memory': Commons.bytes2human(df.memory_usage(deep=True).sum())},
                                  'data_type': {'numeric': _numeric_fields, 'category': _category_fields,
                                                'datetime': _date_fields, 'bool': _bool_fields,
                                                'others': _other_fields},
                                  'usability': {'mostly null': len(_null_columns),
                                                'predominance': len(_dom_columns),
                                                'correlated': len(_correlated)}}
        # analysis
        _analysis_dict = {}
        granularity = granularity if isinstance(granularity, (int, float, list)) else [0.9, 0.75, 0.5, 0.25, 0.1]
        col_kwargs = {}
        for col in _date_columns:
            col_kwargs.update({col: {'date_format': '%Y-%m-%dT%H:%M:%S'}})
        for col in _numeric_columns:
            col_kwargs.update({col: {'dominant': 0, 'exclude_dominant': True}})
        top = top if isinstance(top, int) else 6
        for _, row in self.discover.analysis_dictionary(df, granularity=granularity, top=top,
                                                        col_kwargs=col_kwargs).iterrows():
            _analysis_dict[row.iloc[0]] = {}
            for index in row.index:
                if index.startswith('Attribute') or index.startswith('Unique') or index.startswith('%_Nulls') \
                        or index.startswith('Sample'):
                    continue
                if index.startswith('Upper') or index.startswith('Lower'):
                    if _analysis_dict[row.iloc[0]].get('Limits') is None:
                        _analysis_dict[row.iloc[0]].update({'Limits': {index: str(row.loc[index])}})
                    else:
                        _analysis_dict[row.iloc[0]].get('Limits').update({index: str(row.loc[index])})
                    continue
                _value = row.loc[index]
                if index.startswith('Selection') and \
                        isinstance(_value, list) and all(isinstance(x, tuple) for x in _value):
                    for i in range(len(_value)):
                        _value[i] = (str(_value[i][0]), str(_value[i][1]), _value[i][2])
                    _value = [str(x) for x in _value]
                _analysis_dict[row.iloc[0]].update({index: _value})
        report['insight'] = _analysis_dict
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

    def _correlated_columns(self, df: pd.DataFrame):
        """returns th percentage of useful colums"""
        threshold = 0.98
        pad: TransitionIntentModel = self.scratch_pad()
        df = pad.auto_to_category(df, unique_max=1000, inplace=False)
        df = pad.to_category_type(df, dtype='category', as_num=True)
        for c in df.columns:
            if all(Commons.valid_date(x) for x in df[c].dropna()):
                df = pad.to_date_type(df, dtype='datetime', as_num=True)
        df = Commons.filter_columns(df, dtype=['number'], exclude=False)
        for c in df.columns:
            df[c] = Commons.fillna(df[c])
        col_corr = set()
        corr_matrix = df.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:  # we are interested in absolute coeff value
                    colname = corr_matrix.columns[i]  # getting the name of column
                    col_corr.add(colname)
        return col_corr
