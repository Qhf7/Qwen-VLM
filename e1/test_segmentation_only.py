"""CPU regression tests for anomalous-view sampling and mask alignment."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image

from e1.train_segmentation_only import load_mask, make_schedule, relative_path


class SegmentationDataTests(unittest.TestCase):
    def test_category_balanced_reproducible_schedule(self):
        data = pd.DataFrame([
            {'class_name': c, 'group_path': f'{c}/{g}', 'image_path': f'{c}/{g}/{v}.png'}
            for c in ('a', 'b', 'c') for g in range(2) for v in range(g+1)
        ])
        schedule = make_schedule(data, 25, 8, 20260901)
        pd.testing.assert_frame_equal(schedule, make_schedule(data,25,8,20260901))
        self.assertTrue(schedule.groupby('step').size().eq(8).all())
        counts = schedule.class_name.value_counts()
        self.assertLessEqual(int(counts.max()-counts.min()),1)
        self.assertEqual(set(schedule.head(8).class_name), {'a','b','c'})
        for row in schedule.itertuples():
            self.assertEqual(data.iloc[row.dataset_index].image_path,row.image_path)

    def test_mask_nearest_alignment_and_empty_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask = np.zeros((896,896), np.uint8)
            mask[100:300,200:400] = 255
            Image.fromarray(mask).save(root/'mask.png')
            row = SimpleNamespace(mask_path='mask.png')
            result = load_mask(row,root)
            self.assertEqual(result.shape,(448,448))
            expected = np.zeros((448,448),bool)
            expected[50:150,100:200] = True
            np.testing.assert_array_equal(result,expected)
            Image.fromarray(np.zeros((448,448),np.uint8)).save(root/'mask.png')
            with self.assertRaisesRegex(ValueError,'Empty original'):
                load_mask(row,root)

    def test_relative_paths(self):
        with self.assertRaises(ValueError):
            relative_path(Path('Dataset'),'../mask.png')
        with self.assertRaises(ValueError):
            relative_path(Path('Dataset'),'/tmp/mask.png')


if __name__ == '__main__':
    unittest.main()
