from typing import Dict


class AbstractDriftTester:
    def test_drift(self, data: object):
        raise NotImplementedError
