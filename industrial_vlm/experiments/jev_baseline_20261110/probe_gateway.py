#!/usr/bin/env python3
"""Image transport smoke test only; synthetic pictures are never evaluation data."""
import argparse, base64, getpass, hashlib, io, json, os, time, uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
import requests
from PIL import Image, ImageDraw

def main():
    p=argparse.ArgumentParser();p.add_argument('--shots',nargs='+',type=int,choices=[0,1,4],default=[0,1,4]);p.add_argument('--negative',action='store_true');p.add_argument('--endpoint',default='https://api.modelgo.ai/v1/systemone');p.add_argument('--output',type=Path,default=Path(__file__).with_name('gateway_probe.json'));a=p.parse_args()
    u=urlsplit(a.endpoint)
    if u.scheme!='https' or u.path!='/v1/systemone' or u.username or u.password or u.query or u.fragment:raise SystemExit('Use the HTTPS /v1/systemone endpoint without credentials in URL')
    key=os.environ.get('TYPESAFE_API_KEY') or os.environ.get('MODELGO_API_KEY') or getpass.getpass('API key (not saved): ')
    if not key:raise SystemExit('Missing key')
    def img(color):
        im=Image.new('RGB',(448,448),'white');ImageDraw.Draw(im).rectangle((128,128,320,320),fill=color);b=io.BytesIO();im.save(b,format='PNG');return b.getvalue()
    query=img('blue' if a.negative else 'red');normal=img('blue');records=[]
    for shots in a.shots:
        q=str(uuid.uuid4());refs=[str(uuid.uuid4()) for _ in range(shots)]
        assets={q:{'type':'image','format':'png','data_base64':base64.b64encode(query).decode()}}
        for k in refs:assets[k]={'type':'image','format':'png','data_base64':base64.b64encode(normal).decode()}
        body={'model':'jev-1.14.0','state':{'query':q,'normal_references':refs},'assets':assets,'questions':{'anomaly':{'type':'noul','instructions':'Does the query image contain a red square? Evaluate only query. normal_references, if present, are separate blue-square reference images.'}}}
        t=time.perf_counter();record={'shots':shots,'purpose':'synthetic_image_transport_smoke_only','query_color':'blue' if a.negative else 'red','request_model':'jev-1.14.0','image_size':[448,448],'assets_count':len(assets),'request_sha256':hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()}
        try:
            response=requests.post(a.endpoint,json=body,headers={'Authorization':'Bearer '+key,'TypeSafe-Danger-Zone-Images':'true'},timeout=(15,90),allow_redirects=False)
            record['http_status']=response.status_code
            record['request_id']=response.headers.get('x-request-id') or response.headers.get('request-id')
            try:data=response.json()
            except ValueError:data={'error':response.text[:1000]}
            if response.status_code==200:
                record['response']={k:data[k] for k in ('model','answers','usage','id') if k in data}
                answer=data.get('answers',{}).get('anomaly',{});prob=answer.get('noul');record['valid_noul']=answer.get('type')=='noul' and type(prob) in (float,int) and 0<=prob<=1
                record['model_name_matches']=data.get('model')=='jev-1.14.0'
            else:record['error']=str(data).replace(key,'[REDACTED]')[:1200]
        except requests.RequestException as e:record['error']=type(e).__name__
        record['seconds']=time.perf_counter()-t;records.append(record)
        report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'endpoint':a.endpoint,'request_model':'jev-1.14.0','no_benchmark_predictions':True,'model_version_note':'ModelGo documents model as logical request-name echo; not independent upstream version proof.','records':records}
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2).replace(key,'[REDACTED]')+'\n')
        print(json.dumps(record,ensure_ascii=False),flush=True)
        if not record.get('valid_noul') or not record.get('model_name_matches'):break
    key=None
if __name__=='__main__':main()
