import unittest
import numpy as np

from src.scarr.engines.snr import SNR
from src.scarr.engines.cpa import CPA
from src.scarr.engines.mia import MIA

from src.scarr.model_values.sbox_weight import SboxWeight

from devtools.data_creation.correlation_data import CorrelationData

class TestSNR(unittest.TestCase):
    def test_snr(self):

        cd = CorrelationData(5000, 50)

        cd.generate_data()

        model_positions = np.arange(16)

        cd.configure(0,0,model_positions)

        instance = SNR()

        instance.run(cd)

        results = instance.get_result()

        for model_pos in range(16):
            for pos, snr in enumerate(results[0][model_pos]):
                if model_pos in (pos - 4, pos - 24):
                    continue
                self.assertTrue(snr < 1.2)


            self.assertTrue(results[0][model_pos][model_pos + 4] > 5000)
            self.assertTrue(results[0][model_pos][model_pos + 24] > 5000)

class TestCPA(unittest.TestCase):
    def test_cpa(self):

        cd = CorrelationData(5000, 50)

        cd.generate_data()

        model_value = SboxWeight()

        model_positions = np.arange(16)

        cd.configure(0,0,model_positions)

        instance = CPA(model_value=model_value)

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

        model_value = SboxWeight()

        model_positions = np.arange(16)

        cd.configure(0,0,model_positions)

        instance = MIA(model_value=model_value, bin_num=9)

        instance.run(cd)

        instance.get_result()
        results = np.squeeze(instance.get_candidate())

        print("TRUE KEYS:\n", cd.get_key())
        print("KEY CANDIDATES:\n", results)

        self.assertTrue(np.array_equal(results, cd.get_key()))


if __name__ == "__main__":
    unittest.main()
