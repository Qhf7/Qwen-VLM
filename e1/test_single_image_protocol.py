import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
import pandas as pd
import torch
from e1.evaluate_classification_fold1 import summarize
from e1.model import QwenE1


class SingleImageProtocolTests(unittest.TestCase):
    def test_qwen3_patch_spatial_order(self):
        raster=torch.arange(16,dtype=torch.float32).reshape(1,4,4,1)
        packed=raster.reshape(1,2,2,2,2,1).permute(0,1,3,2,4,5).reshape(16,1)
        stub=SimpleNamespace(visual=SimpleNamespace(spatial_merge_size=2))
        restored=QwenE1._restore_spatial(stub,packed,torch.tensor([[1,4,4]]))
        torch.testing.assert_close(restored,raster.permute(0,3,1,2))

    def test_metrics_use_view_labels(self):
        views=pd.DataFrame(dict(image_path=['a','b','c','d'],class_name=['part']*4,
            group_path=['same_part']*4,sample_label=[1]*4,view_label=[0,1,0,1],
            defect_ratio=[0,.005,0,.02],p_yes=[.1,.9,.2,.8]))
        report,rows,classes=summarize(views)
        self.assertEqual(len(rows),4)
        self.assertEqual(report['overall']['accuracy'],1)
        self.assertEqual(report['overall']['tn'],2)
        self.assertEqual(report['overall']['tp'],2)
        self.assertEqual(report['small_defect_recall'],1)
        self.assertEqual(classes.n_views.iloc[0],4)

    def test_schedule_and_missing_cache_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); data=[]
            for c in range(18):
                for label in (0,1):
                    for n in range(8):
                        name=f'{c}_{label}_{n}.png'
                        (root/name).touch()
                        data.append(dict(class_name=str(c),view_label=label,image_path=name))
            frame=pd.DataFrame(data)
            frame.to_parquet(root/'train.parquet'); frame.to_csv(root/'manifest.csv',index=False)
            cmd=[sys.executable,'-m','e1.make_real_classification_schedule','--train',str(root/'train.parquet'),
                 '--cache-root',str(root),'--cache-manifest',str(root/'manifest.csv'),
                 '--out-dir',str(root/'out')]
            subprocess.run(cmd,check=True,capture_output=True)
            schedule=pd.read_parquet(root/'out/real_classification_schedule.parquet')
            self.assertEqual(len(schedule),8000)
            self.assertTrue(schedule.groupby(['step','source_type']).size().eq(4).all())
            lookup=frame.set_index('image_path')
            first=schedule[schedule.step<=36].copy()
            first['class_name']=first.image_path.map(lookup.class_name)
            counts=first.groupby(['source_type','class_name']).size()
            self.assertTrue(counts.eq(8).all())
            self.assertEqual(first.image_path.nunique(),len(frame))
            (root/frame.image_path.iloc[0]).unlink()
            self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)


if __name__=='__main__':
    unittest.main()
