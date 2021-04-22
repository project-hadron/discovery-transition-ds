import unittest

from tests.intent.controller_intent_test import ControllerIntentTest
from tests.intent.event_book_intent_test import EventBookIntentModelTest
from tests.intent.feature_catalog_intent_model_test import FeatureCatalogIntentTest
from tests.intent.synthetic_get_canonical_test import SyntheticGetCanonicalTest
from tests.intent.synthetic_intent_get_test import SyntheticIntentGetTest
from tests.intent.synthetic_intent_analysis_test import SyntheticIntentAnalysisTest
from tests.intent.synthetic_intent_correlate_selection_test import SyntheticIntentCorrelateSelectionTest
from tests.intent.synthetic_intent_model_test import SyntheticIntentModelTest
from tests.intent.synthetic_pipeline_test import SyntheticPipelineTest
from tests.intent.synthetic_weighting_test import SyntheticWeightingTest
from tests.intent.transition_intent_model_test import TransitionIntentModelTest
from tests.intent.wrangle_intent_correlate_test import WrangleIntentCorrelateTest

# initialize the test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

# add tests to the test suite
suite.addTests(loader.loadTestsFromTestCase(ControllerIntentTest))
suite.addTests(loader.loadTestsFromTestCase(EventBookIntentModelTest))
suite.addTests(loader.loadTestsFromTestCase(FeatureCatalogIntentTest))
suite.addTests(loader.loadTestsFromTestCase(SyntheticGetCanonicalTest))
suite.addTests(loader.loadTestsFromTestCase(SyntheticIntentGetTest))
suite.addTests(loader.loadTestsFromTestCase(SyntheticIntentAnalysisTest))
suite.addTests(loader.loadTestsFromTestCase(SyntheticIntentCorrelateSelectionTest))
suite.addTests(loader.loadTestsFromTestCase(SyntheticIntentModelTest))
suite.addTests(loader.loadTestsFromTestCase(SyntheticPipelineTest))
suite.addTests(loader.loadTestsFromTestCase(SyntheticWeightingTest))
suite.addTests(loader.loadTestsFromTestCase(TransitionIntentModelTest))
suite.addTests(loader.loadTestsFromTestCase(WrangleIntentCorrelateTest))

# initialize a runner, pass it your suite and run it
runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)
