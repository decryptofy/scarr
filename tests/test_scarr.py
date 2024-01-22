import unittest
import numpy as np

from src.scarr.engines.snr import SNR
from src.scarr.engines.cpa import CPA
from src.scarr.engines.mia import MIA

from src.scarr.models.subBytes_weight import SubBytes_weight

from devtools.data_creation.correlation_data import CorrelationData

class TestSNR(unittest.TestCase):
    def test_snr(self):

        cd = CorrelationData(5000, 50)

        cd.generate_data()

        bytes_pos = np.arange(16)

        cd.configure(0,0,bytes_pos)

        instance = SNR()

        instance.run(cd)

        results = instance.get_result()

        for byte in range(16):
            for pos, snr in enumerate(results[0][byte]):
                if byte in (pos - 4, pos - 24):
                    continue
                self.assertTrue(snr < 1.2)


            self.assertTrue(results[0][byte][byte + 4] > 5000)
            self.assertTrue(results[0][byte][byte + 24] > 5000)

class TestCPA(unittest.TestCase):
    def test_cpa(self):

        cd = CorrelationData(5000, 50)

        cd.generate_data()

        model = SubBytes_weight()

        bytes_pos = np.arange(16)

        cd.configure(0,0,bytes_pos)

        instance = CPA(model=model)

        instance.run(cd)

        instance.get_result()
        results = np.squeeze(instance.get_candidate())

        print("TRUE KEYS:\n", cd.get_key())
        print("KEY CANDIDATES:\n", results)

        self.assertTrue(np.array_equal(results, cd.get_key()))

class TestMIA(unittest.TestCase):
    def test_mia(self):

        cd = CorrelationData(5000, 50)

        cd.generate_data()

        model = SubBytes_weight()

        bytes_pos = np.arange(16)

        cd.configure(0,0,bytes_pos)

        instance = MIA(model=model, bin_num=9)

        instance.run(cd)

        instance.get_result()
        results = np.squeeze(instance.get_candidate())

        print("TRUE KEYS:\n", cd.get_key())
        print("KEY CANDIDATES:\n", results)

        self.assertTrue(np.array_equal(results, cd.get_key()))


if __name__ == "__main__":
    unittest.main()
