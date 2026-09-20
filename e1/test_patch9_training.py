"""CPU gates for real-data pools, empty-mask loss, and actual evaluation."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from e1.model import segmentation_loss_components
from e1.patch9_training_data import check_splits, audit_frame, make_schedule, example, KINDS
from e1.patch9_validation import evaluate

class PatchTrainingTests(unittest.TestCase):
    def data(self, root, split):
        rows=[]
        for cls in ('a','b'):
            for label in (0,1):
                name=f'{split}_{cls}_{label}'
                rgb=np.zeros((160,200,3),np.uint8)
                mask=np.zeros((160,200),np.uint8)
                if label:
                    mask[50:65,60:75]=255
                    rgb[mask>0]=255
                Image.fromarray(rgb).save(root/f'{name}.png')
                Image.fromarray(mask).save(root/f'{name}_mask.png')
                rows.append(dict(image_path=f'{name}.png',mask_path=f'{name}_mask.png',view_label=label,
                                 class_name=cls,group_path=name,visual_group_id=name))
        return pd.DataFrame(rows)

    def test_pools_schedule_and_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); train=self.data(root,'train'); dev=self.data(root,'dev')
            check_splits(train,dev)
            records=audit_frame(train,root,root)
            schedule=make_schedule(train,records,12,42)
            pd.testing.assert_frame_equal(schedule,make_schedule(train,records,12,42))
            for _,batch in schedule.groupby('step'):
                self.assertEqual(sorted(batch.kind),sorted(KINDS))
            for _,group in schedule.groupby('kind'):
                counts=group.class_name.value_counts()
                self.assertLessEqual(counts.max()-counts.min(),1)
            for row in schedule.head(8).itertuples(index=False):
                image,mask=example(train,row,root,root)
                self.assertEqual(image.shape,(448,448,3))
                self.assertEqual(bool(mask.any()),row.kind in ('anomaly_full','positive_patch'))
            with self.assertRaisesRegex(ValueError,'both normal and anomaly'):
                check_splits(train[train.view_label.eq(1)],dev)
            with self.assertRaisesRegex(ValueError,'overlap'):
                check_splits(train,train)

    def test_empty_loss_and_positive_compatibility(self):
        logits=torch.zeros((1,1,16,16),requires_grad=True)
        zero=torch.zeros_like(logits)
        loss,focal,dice,n,e=segmentation_loss_components(logits,zero,dice_nonempty_only=True)
        self.assertEqual(float(dice),0)
        self.assertEqual((n,e),(0,1))
        torch.testing.assert_close(loss,focal)
        loss.backward()
        self.assertGreater(float(logits.grad.abs().sum()),0)
        target=zero.clone(); target[:,:,3:8,3:8]=1
        new=segmentation_loss_components(logits,target,True)
        old=segmentation_loss_components(logits,target,False)
        for a,b in zip(new[:3],old[:3]):
            torch.testing.assert_close(a,b)

    def test_real_evaluation_outputs_and_no_gt_gating(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); dev=self.data(root,'dev'); calls=[]
            def predict(image):
                calls.append(image.shape)
                return image[:,:,0].astype(np.float32)/255
            result=evaluate(predict,dev,root,root,root/'eval',250)
            self.assertEqual(len(calls),len(dev)*10)
            self.assertEqual(set(result['variants']),{'full','mean_w0.5'})
            for metrics in result['variants'].values():
                self.assertEqual(metrics['normal_n'],2)
                self.assertEqual(metrics['anomaly_n'],2)
                self.assertEqual(metrics['normal_nonempty_rate'],0)
            for suffix in ('metrics.json','views.csv','per_class.csv','per_area.csv'):
                self.assertTrue((root/'eval'/f'step00250_{suffix}').is_file())

if __name__=='__main__':
    unittest.main()
