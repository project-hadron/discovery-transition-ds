import pandas as pd
import numpy as np

__author__ = 'Darryl Oatridge'


class FrameBuild():

    @staticmethod
    def related_xyz(size=1000, type='chain') -> pd.DataFrame:
        if type == 'fork': # y->x, y->z
            y = np.random.normal(size=size)
            x = y + np.random.normal(size=size)
            z = y + np.random.normal(size=size)
        elif type == 'collide': # x->y, z->y
            x = np.random.normal(size=size)
            z = np.random.normal(size=size)
            y = x + z + np.random.normal(size=size)
        else: # x->y->z
            x = np.random.normal(size=size)
            y = x + np.random.normal(size=size)
            z = y + np.random.normal(size=size)

        return pd.DataFrame({'x': x, 'y': y, 'z': z})

