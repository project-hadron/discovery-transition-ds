import unittest

import pandas as pd

class ScratchTest(unittest.TestCase):
    """Test: """

    def setUp(self):
        pass

    def tearDown(self):
        pass


    def test_scratch(self):
        pd.Categorical([])







class Atts(object):
    def __init__(self, attributes: dict):
        self._create_method(attributes, path='')

    def _create_method(self, attributes: [dict, list, str], path: str):
        attr_dict = {}
        if isinstance(attributes, str):
            attr_dict[attributes] = attributes
        elif isinstance(attributes, list):
            for i in attributes:
                if isinstance(i, dict):
                    attr_dict.update(i)
                else:
                    attr_dict[i] = i
        else:
            attr_dict = attributes
        for k, v in attr_dict.items():
            if isinstance(v, (dict, list)):
                _next_level = Atts(v)
                self._add_method(k, _next_level)
                _next_level._create_method(v, ".".join([path, k]))
            else:
                self._add_method(k, path)
        return

    def _add_method(self, name, rtn_value):
        _method = self._make_method(rtn_value)
        setattr(self, name, _method)

    def _make_method(self, rtn_value):
        @property
        def _method():
            return rtn_value

        return _method.fget()


if __name__ == '__main__':
    unittest.main()
