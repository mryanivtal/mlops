import unittest
import pandas as pd
import numpy as np

from helpers.data_helper import change_int_values
from drift_detection.drift_testers.chi_drift_tester import ChiDriftTester


class MyTestCase(unittest.TestCase):
    def test_something(self):
        aa = pd.DataFrame(np.ones([100, 3]), columns=['a', 'b', 'c'])
        bb = pd.DataFrame(np.ones([100, 3]), columns=['a', 'b', 'c'])

        aa = change_int_values(aa, 'a', 1, 0, 0.5)
        bb = change_int_values(bb, 'a', 1, 0, 0.5)

        aa = change_int_values(aa, 'b', 1, 0, 0.2)
        bb = change_int_values(bb, 'b', 1, 0, 0.6)

        test_same = ChiDriftTester('tester_same', 'a', 0.005)
        test_same.fit(aa)

        test_diff = ChiDriftTester('tester_different', 'b', 0.005)
        test_diff.fit(aa)

        res_same = test_same.test_drift(bb)
        res_diff = test_diff.test_drift(bb)

        print(res_same)
        print(res_diff)

        self.assertEqual(res_same['drift_found'], False)  # add assertion here
        self.assertEqual(res_diff['drift_found'], True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
