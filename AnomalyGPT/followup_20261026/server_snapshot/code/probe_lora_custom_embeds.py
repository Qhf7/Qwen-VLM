import json, torch
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
CKPT='/root/autodl-tmp/industrial_vlm/models/Qwen3-VL-8B-Instruct'
def main():
 bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
 full=Qwen3VLForConditionalGeneration.from_pretrained(CKPT,quantization_config=bnb,device_map={'':'cuda:0'},dtype=torch.bfloat16,low_cpu_mem_usage=True)
 lm=full.model.language_model
 for p in lm.parameters(): p.requires_grad=False
 if not hasattr(lm,'prepare_inputs_for_generation'): lm.prepare_inputs_for_generation=lambda input_ids=None,past_key_values=None,**kwargs: kwargs
 lora=get_peft_model(lm,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj']))
 tok=AutoTokenizer.from_pretrained(CKPT,trust_remote_code=True); tok.pad_token=tok.pad_token or tok.eos_token
 t=tok('There is an anomaly.',return_tensors='pt',add_special_tokens=False).to('cuda:0')
 tail=lora.get_input_embeddings()(t.input_ids)
 prefix=torch.randn(1,19,tail.shape[-1],device='cuda:0',dtype=tail.dtype)*0.01
 emb=torch.cat([prefix,tail],dim=1); B,S=emb.shape[:2]
 attn=torch.ones(B,S,device='cuda:0',dtype=torch.long); pos=torch.arange(S,device='cuda:0').view(1,1,-1).expand(3,B,-1)
 out=lora(inputs_embeds=emb,attention_mask=attn,position_ids=pos,use_cache=False,return_dict=True)
 logits=full.lm_head(out.last_hidden_state)
 labels=torch.full((B,S),-100,device='cuda:0',dtype=torch.long); labels[:,19:]=t.input_ids
 loss=torch.nn.functional.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),labels[:,1:].reshape(-1),ignore_index=-100)
 loss.backward(); grads=[p.grad for p in lora.parameters() if p.requires_grad and p.grad is not None]
 result={'custom_prefix_tokens':19,'tail_tokens':int(tail.shape[1]),'sequence_length':int(S),'loss':float(loss),'nonzero_grad_tensors':sum(bool(g.abs().sum()>0) for g in grads),'grad_tensors':len(grads),'peak_vram_gb':torch.cuda.max_memory_allocated()/1e9}
 json.dump(result,open('lora_custom_embeds_result.json','w'),indent=2); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
