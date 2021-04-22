import unittest

from tests.components.abstract_common_component_test import AbstractCommonComponentTest
from tests.components.commons_test import CommonsTest
from tests.components.controller_test import ControllerTest
from tests.components.discovery_analyse_test import DiscoveryAnalysisTest
from tests.components.discovery_test import DiscoveryTest
from tests.components.event_book_portfolio_test import EventBookPortfolioTest
from tests.components.event_book_test import EventBookTest
from tests.components.feature_catalog_test import FeatureCatalog
from tests.components.synthetic_builder_test import SyntheticBuilderTest
from tests.components.transition_test import TransitionTest
from tests.components.visual_test import VisualTest

# initialize the test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

# add tests to the test suite
suite.addTests(loader.loadTestsFromTestCase(AbstractCommonComponentTest))
suite.addTests(loader.loadTestsFromTestCase(CommonsTest))
suite.addTests(loader.loadTestsFromTestCase(ControllerTest))
suite.addTests(loader.loadTestsFromTestCase(DiscoveryAnalysisTest))
suite.addTests(loader.loadTestsFromTestCase(DiscoveryTest))
suite.addTests(loader.loadTestsFromTestCase(EventBookPortfolioTest))
suite.addTests(loader.loadTestsFromTestCase(EventBookTest))
suite.addTests(loader.loadTestsFromTestCase(FeatureCatalog))
suite.addTests(loader.loadTestsFromTestCase(SyntheticBuilderTest))
suite.addTests(loader.loadTestsFromTestCase(TransitionTest))
suite.addTests(loader.loadTestsFromTestCase(VisualTest))

# initialize a runner, pass it your suite and run it
runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)
