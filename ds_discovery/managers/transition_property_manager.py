from aistac.properties.abstract_properties import AbstractPropertyManager

__author__ = 'Darryl Oatridge'

from ds_discovery.transition.commons import Commons


class TransitionPropertyManager(AbstractPropertyManager):

    def __init__(self, task_name: str, username: str):
        # set additional keys
        root_keys = [{'provenance': ['title', 'domain', 'description', 'license', 'provider', 'author']},
                     {'insight': ['blueprint', 'endpoints']}]
        knowledge_keys = ['transition', 'observations', 'actions', 'attributes']
        super().__init__(task_name=task_name, root_keys=root_keys, knowledge_keys=knowledge_keys, username=username)

    @property
    def provenance(self) -> dict:
        """Return the provenance report"""
        return self.get(self.KEY.provenance_key, {})

    def report_provenance(self) -> dict:
        """Return the provenance report"""
        report = dict()
        for catalog in self.get(self.KEY.provenance_key, {}).keys():
            _key = self.join(self.KEY.provenance_key, catalog)
            if catalog in ['provider', 'author']:
                for name, value in self.get(_key, {}).items():
                    report[f"{catalog}_{name}"] = value
            else:
                report[catalog] = self.get(_key, '')
        return report

    def set_provenance(self, title: str=None, domain: str=None, description: str=None, usage_license: str=None,
                       provider_name: str=None, provider_uri: str=None, provider_note: str=None,
                       author_name: str=None, author_uri: str=None, author_contact: str=None):
        """ sets the provenance values. Only sets those passed

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
        """
        for name, value in locals().items():
            if value is None:
                continue
            if name in ['title', 'domain', 'description', 'license']:
                self.set(self.join(self.KEY.provenance_key, name), value)
            if name.startswith('provider') or name.startswith('author'):
                key, item = name.split(sep='_')
                self.set(self.join(self.KEY.provenance_key, key, item), value)
        return

    def reset_provenance(self):
        """resets provenance back to its default values"""
        self._base_pm.remove(self.KEY.provenance_key)
        self.set(self.KEY.provenance_key, {})
        return

    @property
    def insight(self) -> (dict, list):
        """The insight analysis parameters, returning a tuple of blueprint and endpoints """
        blueprint: dict = self.get(self.KEY.insight.blueprint_key, {})
        endpoints: list = self.get(self.KEY.insight.endpoints_key, [])
        return blueprint, endpoints

    def set_insight(self, blueprint: dict, endpoints: list=None):
        """sets the insight analysis parameters. This removes any current insight values

        :param blueprint: the analysis blueprint of what to analysis and how
        :param endpoints: the endpoints, if any, where the tree ends
        """
        self.reset_insight()
        self.set(self.KEY.insight.blueprint_key, blueprint)
        if isinstance(endpoints, list):
            self.set(self.KEY.insight.endpoints_key, endpoints)
        return

    def reset_insight(self):
        """resets the insights back to default"""
        self._base_pm.remove(self.KEY.insight_key)
        self.set(self.KEY.insight_key, {})
        return

    @staticmethod
    def list_formatter(value) -> list:
        """override of the list_formatter to include Pandas types"""
        return Commons.list_formatter(value=value)
