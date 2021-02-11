import unittest
from ds_discovery import Transition, FeatureCatalog
from ds_discovery import SyntheticBuilder


class MyTestCase(unittest.TestCase):

    @staticmethod
    def synthetic_agent(agent_name: str, size: int, remote_uri: str):
        SyntheticBuilder.from_env(agent_name, uri_pm_repo=remote_uri).run_synthetic_pipeline(size=size)
        Transition.from_env(agent_name, uri_pm_repo=remote_uri).run_transition_pipeline()
        FeatureCatalog.from_env(agent_name, uri_pm_repo=remote_uri).run_feature_pipeline()

    def test_run_pipeline(self):
        agent_name = 'hk_income'
        repo_uri = "https://raw.githubusercontent.com/project-hadron/hadron-asset-bank/master/bundles/samples/hk_income_sample/contracts/"
        self.synthetic_agent(agent_name=agent_name, size=1000, remote_uri=repo_uri)


if __name__ == '__main__':
    unittest.main()
