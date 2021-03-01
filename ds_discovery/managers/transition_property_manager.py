from ds_discovery.components.commons import Commons
from aistac.properties.abstract_properties import AbstractPropertyManager

__author__ = 'Darryl Oatridge'


class TransitionPropertyManager(AbstractPropertyManager):

    def __init__(self, task_name: str, username: str):
        # set additional keys
        root_keys = [{'provenance': ['title', 'domain', 'description', 'license', 'provider', 'author', 'cost']}]
        knowledge_keys = ['observations', 'actions', 'attributes']
        super().__init__(task_name=task_name, root_keys=root_keys, knowledge_keys=knowledge_keys, username=username)

    @property
    def provenance(self) -> dict:
        """Return the provenance report_canonical"""
        return self.get(self.KEY.provenance_key, {})

    def report_provenance(self) -> dict:
        """Return the provenance report_canonical"""
        report = dict()
        for catalog in self.get(self.KEY.provenance_key, {}).keys():
            _key = self.join(self.KEY.provenance_key, catalog)
            if catalog in ['provider', 'author', 'cost', 'license']:
                for name, value in self.get(_key, {}).items():
                    report[f"{catalog}_{name}"] = value
            else:
                report[catalog] = self.get(_key, '')
        return report

    def set_provenance(self, title: str=None, domain: str=None, description: str=None, license_type: str=None,
                       license_name: str=None, license_uri: str=None, cost_price: str=None, cost_code: str=None,
                       cost_type: str=None, provider_name: str=None, provider_uri: str=None, provider_note: str=None,
                       author_name: str=None, author_uri: str=None, author_contact: str=None):
        """ sets the provenance values. Only sets those passed

        :param title: (optional) the title of the provenance
        :param domain: (optional) the domain it sits within
        :param description: (optional) a description of the provenance
        :param license_type: (optional) The type of the license such as PDDL, ODC-By, ODC-ODbL
        :param license_name: (optional) The full name of the license e.g. 'Open Data Commons Attribution License'
        :param license_uri: (optional) The link to the license e.g. 'https://opendatacommons.org/licenses/by/summary/'
        :param cost_price: (optional) a cost price associated with this provenance
        :param cost_code: (optional) a cost centre code or reference code
        :param cost_type: (optional) the cost type or description
        :param provider_name: (optional) the provider system or institution name or title
        :param provider_uri: (optional) a uri reference that helps identify the provider
        :param provider_note: (optional) any notes that might be useful
        :param author_name: (optional) the author of the data
        :param author_uri: (optional) the author uri
        :param author_contact: (optional)the the author contact information
        """
        params = locals().copy()
        params.pop('self', None)
        for name, value in params.items():
            if value is None:
                continue
            if name in ['title', 'domain', 'description']:
                self.set(self.join(self.KEY.provenance_key, name), value)
            if name.startswith('provider') or name.startswith('author') or name.startswith('cost') or name.startswith(
                    'license'):
                key, item = name.split(sep='_')
                self.set(self.join(self.KEY.provenance_key, key, item), value)
        return

    def reset_provenance(self):
        """resets provenance back to its default values"""
        self._base_pm.remove(self.KEY.provenance_key)
        self.set(self.KEY.provenance_key, {})
        return

    @staticmethod
    def list_formatter(value) -> list:
        """override of the list_formatter to include Pandas types"""
        return Commons.list_formatter(value=value)
