import unittest
from radiative_forcing_aggregator import aggregate_radiative_forcing

class TestRadiativeForcingAggregator(unittest.TestCase):

    def test_valid_input(self):
        self.assertAlmostEqual(aggregate_radiative_forcing({'CO2': 1.68, 'CH4': 0.97}), 2.65)

    def test_edge_case_zero(self):
        self.assertAlmostEqual(aggregate_radiative_forcing({'CO2': 0.0, 'CH4': 0.0}), 0.0)

    def test_negative_value(self):
        with self.assertRaises(ValueError):
            aggregate_radiative_forcing({'CO2': -1.0})

    def test_invalid_value(self):
        with self.assertRaises(ValueError):
            aggregate_radiative_forcing({'CO2': 'invalid'})

    def test_empty_input(self):
        self.assertAlmostEqual(aggregate_radiative_forcing({}), 0.0)

if __name__ == '__main__':
    unittest.main()