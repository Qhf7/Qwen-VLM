import json,os,subprocess,time
from pathlib import Path
r=Path("/root/autodl-tmp/AnomalyGPT/experiments/strict_qwen_bridge_v0");f=r/"followup_20261026"
py="/root/autodl-tmp/conda-envs/qwen3vl2/bin/python"
state={"phase":"waiting_for_main","failed":None,"completed":False}
def save(): (f/"after_main_state.json").write_text(json.dumps(state,indent=2)+"\n")
save()
try:
 while True:
  s=json.loads((f/"safe_state.json").read_text())
  if s.get("failed"): raise RuntimeError("Main pipeline failed: "+str(s["failed"]))
  if not s.get("running") and s.get("finished_at") and len(s["completed"])==32: break
  time.sleep(5)
 state["phase"]="main_complete_summarize";save()
 commands=[[py,str(f/"summarize_followup.py"),"--root",str(f),"--out",str(f/"final_report")],[py,str(f/"diagnose_generation.py"),"--root",str(f),"--out",str(f/"generation_diagnostics.json")],[py,str(f/"sensitivity_previous_test8.py"),"--root",str(f),"--out",str(f/"final_report/sensitivity_previous_test8.json")]]
 for command in commands: subprocess.run(command,check=True,cwd=r/"code")
 state["phase"]="map_prompt_diagnostic";save()
 subprocess.run([py,str(f/"prompt_map_ablation.py"),"--experiment",str(r),"--root",str(f),"--out",str(f/"map_prompt_diagnostic")],check=True,cwd=r/"code",env={**os.environ,"PYTHONUNBUFFERED":"1","OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4","TOKENIZERS_PARALLELISM":"false"})
 state.update(phase="complete",completed=True);save()
 print("AFTER_MAIN_COMPLETE",flush=True)
except BaseException as error:
 state.update(failed={"type":type(error).__name__,"message":str(error)},completed=False);save();raise
