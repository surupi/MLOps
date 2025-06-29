import unittest
import pandas as pd

class TestIrisDataset(unittest.TestCase):
    def test_dataset_row_count(self):
        df = pd.read_csv('iris.csv')
        expected_rows = 150
        self.assertEqual(df.shape[0], expected_rows,
                         f"Dataset should have {expected_rows} rows, found {df.shape[0]}")

if __name__ == '__main__':
    unittest.main()
