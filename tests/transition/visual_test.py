import unittest

import numpy as np
import pandas as pd

from ds_discovery.components.discovery import Visualisation


class VisualTest(unittest.TestCase):

    def setup(self):
        pass

    def tearDown(self):
        pass

    def test_runs(self):
        """Basic smoke test"""
        Visualisation()

    def test_show_cat(self):
        v = Visualisation()
        df = self.control_frame()
        v.show_cat_count(df)

    @staticmethod
    def control_frame():
        data = {'A' : [1, 10, 'Fred', 'M'],
                'B' : [2, 11, 'Jim', 'M'],
                'C' : [3, '', 'Bob', 'S'],
                'D' : [4, '', 'Fred', 'S'],
                'E' : [5, 14, 'Fred', np.nan],
                'F' : [6, 15, 'Bob', 'S'],
                'G' : [7, 16, 'Fred', 'S'],
                'H' : [8, '', 'Jim', np.nan],
                'I' : [9, 17, 'Bob', np.nan],
                'J' : [0, 19, 'Fred', np.nan],
                }
        df = pd.DataFrame(data).transpose()
        df.columns = ['count', 'num', 'name', 'gender']
        df['count'] = df['count'].astype(int)
        df['name'] = df['name'].astype('category')
        df['gender'] = df['gender'].astype('category')
        return df


if __name__ == '__main__':
    unittest.main()
