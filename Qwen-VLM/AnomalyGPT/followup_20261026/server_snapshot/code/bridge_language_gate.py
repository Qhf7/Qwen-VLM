import json, time, torch
from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
M='/root/autodl-tmp/industrial_vlm/models/Qwen3-VL-8B-Instruct'
t0=time.time(); bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
tok=AutoTokenizer.from_pretrained(M, trust_remote_code=True)
model=Qwen3VLForConditionalGeneration.from_pretrained(M,quantization_config=bnb,device_map={'':0},dtype=torch.bfloat16,low_cpu_mem_usage=True).eval()
lm=model.model.language_model; head=model.lm_head
ids=tok('### Human: <Img> anomaly inspection. </Img> Is there any anomaly? ### Assistant:',return_tensors='pt',add_special_tokens=False).input_ids.cuda()
emb=lm.embed_tokens(ids).detach().clone().requires_grad_(True)
B,S=emb.shape[:2]; pos=torch.arange(S,device=emb.device).view(1,1,-1).expand(3,B,-1)
mask=torch.ones(B,S,dtype=torch.long,device=emb.device)
out=lm(inputs_embeds=emb,attention_mask=mask,position_ids=pos,return_dict=True,use_cache=False)
logits=head(out.last_hidden_state)
target=torch.full((B,S),-100,dtype=torch.long,device=emb.device); target[:, -1]=ids[:, -1]
loss=torch.nn.functional.cross_entropy(logits[:,:-1].float().reshape(-1,logits.size(-1)),target[:,1:].reshape(-1),ignore_index=-100)
loss.backward()
print(json.dumps({'loss':float(loss),'grad_mean':float(emb.grad.abs().mean()),'seq':S,'hidden':emb.shape[-1],'vram_gb':torch.cuda.max_memory_allocated()/1e9,'seconds':time.time()-t0}),flush=True)
