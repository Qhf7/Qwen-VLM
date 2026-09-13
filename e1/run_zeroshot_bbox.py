from pathlib import Path
import argparse,json,re,time
import numpy as np,pandas as pd,torch
from PIL import Image
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from e1.zeroshot_bbox_protocol import PROMPT,parse,rasterize

def main():
 p=argparse.ArgumentParser();p.add_argument('--model-id',default='models/Qwen3-VL-2B-Instruct');p.add_argument('--dev',type=Path,default=Path('artifacts/splits/fold1/dev.parquet'));p.add_argument('--root',type=Path,default=Path('Dataset448'));p.add_argument('--gt-root',type=Path,default=Path('Dataset'));p.add_argument('--out',type=Path,default=Path('artifacts/segmentation_zeroshot_qwen3'));a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 d=pd.read_parquet(a.dev); device='cuda'; proc=AutoProcessor.from_pretrained(a.model_id,min_pixels=448**2,max_pixels=448**2); model=Qwen3VLForConditionalGeneration.from_pretrained(a.model_id,dtype=torch.bfloat16,device_map='auto').eval(); rows=[];t=time.time()
 for i,r in d.iterrows():
  ip=a.root/str(r.image_path); im=Image.open(ip).convert('RGB'); msg=[{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':PROMPT}]}]; x=proc.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt'); x={k:v.to(device) for k,v in x.items()}; out=model.generate(**x,max_new_tokens=80); txt=proc.batch_decode(out[:,x['input_ids'].shape[1]:],skip_special_tokens=True)[0]; q=parse(txt); pred=rasterize(q); mp=r.get('mask_path'); gt=np.zeros((448,448),bool)
  if isinstance(mp,str) and mp:
   gp=a.gt_root/mp
   if gp.exists():
    gm=Image.open(gp).convert('L')
    if gm.size!=(448,448): gm=gm.resize((448,448),Image.Resampling.NEAREST)
    gt=np.asarray(gm)>0
  inter=np.logical_and(pred>0,gt).sum(); ps=(pred>0).sum(); gs=gt.sum(); dice=float(2*inter/(ps+gs)) if ps+gs else 1.; iou=float(inter/(ps+gs-inter)) if ps+gs-inter else 1.; rows.append({'image_path':r.image_path,'class_name':r.get('class_name',''),'label':int(r.get('view_label',0)),'valid':q.get('valid',False),'defect':q.get('defect',False),'dice':dice,'iou':iou,'gt_area':int(gs),'pred_area':int(ps),'pixel_recall':float(inter/gs) if gs else 0.});
  if (i+1)%100==0: print(i+1,'/',len(d),'elapsed',round(time.time()-t,1),flush=True)
 pd.DataFrame(rows).to_csv(a.out/'views.csv',index=False); z=pd.DataFrame(rows); summary={'n':len(z),'valid_rate':float(z.valid.mean()),'dice_mean':float(z.dice.mean()),'dice_nonempty':float(z.loc[z.gt_area>0,'dice'].mean()),'iou_mean':float(z.iou.mean()),'pixel_recall_nonempty':float(z.loc[z.gt_area>0,'pixel_recall'].mean()),'ok_pred_area_mean':float(z.loc[z.label==0,'pred_area'].mean()/448**2),'elapsed_s':time.time()-t};(a.out/'metrics.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
