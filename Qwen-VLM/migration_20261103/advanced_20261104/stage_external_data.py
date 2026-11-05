"""Stage only frozen-manifest external files; verify every image/mask SHA256."""
import argparse,concurrent.futures,hashlib,json,os,struct,time,urllib.request,zlib
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--zip-inventory',type=Path);ap.add_argument('--workers',type=int,default=8);ap.add_argument('--base-url',default='https://huggingface.co/datasets/foersben/mvtec-ad/resolve/main/');args=ap.parse_args()
 records=[json.loads(x) for x in args.manifest.read_text().splitlines() if x];files={r[k]:r[h] for r in records for k,h in [('path','image_sha256'),('mask_path','mask_sha256')] if r.get(k)}
 args.out.mkdir(parents=True,exist_ok=True);began=time.time();recovered=0;failed=[];zipdata=None
 if args.zip_inventory:
  zipdata=json.loads(args.zip_inventory.read_text());entries={e['name'].split('/',1)[1]:e for e in zipdata['entries'] if '/' in e['name']}
  with open(zipdata['archive'],'rb') as archive:
   for name,expected in files.items():
    target=args.out/name
    if target.exists() and sha(target)==expected:continue
    if name not in entries:continue
    entry=entries[name];archive.seek(entry['begin']);packed=archive.read(entry['compressed']);data=packed if entry['method']==0 else zlib.decompress(packed,-15) if entry['method']==8 else None
    if data is None or hashlib.sha256(data).hexdigest()!=expected:continue
    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);recovered+=1
 print(json.dumps(dict(stage='archive_recovery',recovered=recovered,total=len(files))),flush=True)
 def fetch(pair):
  name,expected=pair;target=args.out/name
  if target.exists() and sha(target)==expected:return 'existing'
  url=args.base_url.rstrip('/')+'/'+name
  temporary=target.with_suffix(target.suffix+'.partial');target.parent.mkdir(parents=True,exist_ok=True)
  for attempt in range(4):
   try:
    request=urllib.request.Request(url,headers={'User-Agent':'VLM_Ali frozen-manifest research audit'})
    with urllib.request.urlopen(request,timeout=40) as source,open(temporary,'wb') as dest:
     for chunk in iter(lambda:source.read(1<<16),b''):dest.write(chunk)
    if sha(temporary)!=expected:raise RuntimeError('frozen SHA mismatch')
    temporary.replace(target);return 'downloaded'
   except Exception as e:
    if attempt==3:return dict(path=name,error=str(e))
    time.sleep(1+attempt)
 counts={}
 with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
  for n,result in enumerate(pool.map(fetch,files.items()),1):
   if isinstance(result,dict):failed.append(result)
   else:counts[result]=counts.get(result,0)+1
   if n%128==0 or n==len(files):print(json.dumps(dict(stage='stage',done=n,total=len(files),counts=counts,failed=len(failed),elapsed=time.time()-began)),flush=True)
 summary=dict(state='failed' if failed else 'complete',manifest_sha256=sha(args.manifest),files=len(files),image_count=len(records),recovered=recovered,counts=counts,failed=failed,elapsed=time.time()-began,model_queries=0)
 (args.out/'staging_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
 if failed:raise RuntimeError('Missing or mismatched external files: '+str(len(failed)))
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
