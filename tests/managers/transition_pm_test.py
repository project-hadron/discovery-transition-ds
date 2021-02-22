import os
import shutil
import unittest

from ds_discovery.managers.transition_property_manager import TransitionPropertyManager


class TransitionPMTest(unittest.TestCase):

    def setUp(self):
        os.environ['HADRON_PM_PATH'] = os.path.join(os.environ['PWD'], 'work')
        pass

    def tearDown(self):
        try:
            shutil.rmtree(os.path.join(os.environ['PWD'], 'work'))
        except:
            pass

    def test_runs(self):
        """Basic smoke test"""
        TransitionPropertyManager('test', username='TestUser')

    def test_catalog(self):
        pm = TransitionPropertyManager('test', username='TestUser')
        catalog = pm.knowledge_catalog
        self.assertCountEqual(['observations', 'actions', 'schema', 'intent', 'attributes'], catalog)

    def test_manager_name(self):
        pm = TransitionPropertyManager('test', username='TestUser')
        result = pm.manager_name()
        self.assertEqual('components', result)

    def test_provenance_catalog(self):
        pm = TransitionPropertyManager('test', username='TestUser')
        pm.set(pm.KEY.provenance.title_key, 'my_title')
        self.assertEqual('my_title', pm.get(pm.KEY.provenance.title_key))
        pm.set(pm.KEY.provenance.domain_key, 'my_domain')
        self.assertEqual('my_domain', pm.get(pm.KEY.provenance.domain_key))
        result = list(pm.provenance)
        self.assertCountEqual(['title', 'domain'], result)
        # using methods
        pm.set_provenance(title='new_title')
        self.assertEqual('new_title', pm.get(pm.KEY.provenance.title_key))
        self.assertEqual('my_domain', pm.get(pm.KEY.provenance.domain_key))
        pm.set_provenance(author_name='joe bloggs')
        result = pm.report_provenance()
        control = {'author_name': 'joe bloggs', 'domain': 'my_domain', 'title': 'new_title'}
        self.assertEqual(control, result)
        pm.set_provenance(cost_price='$0.04', cost_code='32412')
        result = pm.get(pm.KEY.provenance.cost_key)
        control = {'price': '$0.04', 'code': '32412'}
        self.assertEqual(control, result)


if __name__ == '__main__':
    unittest.main()
