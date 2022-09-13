from aistac.properties.abstract_properties import AbstractPropertyManager

__author__ = 'Darryl Oatridge'


class ControllerPropertyManager(AbstractPropertyManager):

    DEFAULT_INTENT_LEVEL = 'primary_intent'

    def __init__(self, task_name: str, creator: str):
        """Abstract Class for the Master Properties"""
        root_keys = [{'use_case': ['title', 'domain', 'overview', 'scope', 'situation', 'opportunity',
                                   'actions', 'owner', 'stakeholder']}]
        knowledge_keys = []
        super().__init__(task_name=task_name, root_keys=root_keys, knowledge_keys=knowledge_keys, creator=creator)

    @property
    def use_case(self) -> dict:
        """Return the use case"""
        return self.get(self.KEY.use_case_key, {})

    def report_use_case(self) -> dict:
        """Return the use case report_canonical"""
        report = dict()
        for catalog in self.get(self.KEY.use_case_key, {}).keys():
            _key = self.join(self.KEY.use_case_key, catalog)
            report[catalog] = self.get(_key, '')
        return report

    def set_use_case(self, title: str=None, domain: str=None, overview: str=None, scope: str=None,
                     situation: str=None, opportunity: str=None, actions: str=None, project_name: str=None,
                     project_lead: str=None, project_contact: str=None, stakeholder_domain: str=None,
                     stakeholder_group: str=None, stakeholder_lead: str=None, stakeholder_contact: str=None):
        """ sets the use_case values. Only sets those passed

        :param title: (optional) the title of the use_case
        :param domain: (optional) the domain it sits within
        :param overview: (optional) a overview of the use case
        :param scope: (optional) the scope of responsibility
        :param situation: (optional) The inferred 'Why', 'What' or 'How' and predicted 'therefore can we'
        :param opportunity: (optional) The opportunity of the situation
        :param actions: (optional) the actions to fulfil the opportunity
        :param project_name: (optional) the name of the project this use case is for
        :param project_lead: (optional) the person who is project lead
        :param project_contact: (optional) the conact information for the project lead
        :param stakeholder_domain: (optional) the domain of the stakeholders
        :param stakeholder_group: (optional) the stakeholder group name
        :param stakeholder_lead: (optional) the stakeholder lead
        :param stakeholder_contact: (optional) contact information for the stakeholder lead
        """
        params = locals().copy()
        params.pop('self', None)
        for name, value in params.items():
            if value is None:
                continue
            if name in ['title', 'domain', 'description', 'situation', 'opportunity', 'actions']:
                self.set(self.join(self.KEY.use_case_key, name), value)
            if name.startswith('project') or name.startswith('stakeholder'):
                key, item = name.split(sep='_')
                self.set(self.join(self.KEY.use_case_key, key, item), value)
        return

    def reset_use_case(self):
        """resets use case back to its default values"""
        self._base_pm.remove(self.KEY.use_case_key)
        self.set(self.KEY.use_case_key, {})
        return
