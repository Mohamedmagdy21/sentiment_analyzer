import unittest


class TestDatasetTruncation(unittest.TestCase):

    def test_take_160_samples(self):
        import pandas as pd
        from datasets import Dataset

        df = pd.DataFrame({
            "text": [f"sample {i}" for i in range(1000)],
            "label": [i % 3 for i in range(1000)],
        })
        ds = Dataset.from_pandas(df, preserve_index=False)

        take = 160
        truncated = ds.select(range(min(take, len(ds))))

        self.assertEqual(len(truncated), 160)

    def test_take_less_than_available(self):
        import pandas as pd
        from datasets import Dataset

        df = pd.DataFrame({
            "text": [f"sample {i}" for i in range(50)],
            "label": [i % 3 for i in range(50)],
        })
        ds = Dataset.from_pandas(df, preserve_index=False)

        take = 160
        truncated = ds.select(range(min(take, len(ds))))

        self.assertEqual(len(truncated), 50)


if __name__ == "__main__":
    unittest.main()
