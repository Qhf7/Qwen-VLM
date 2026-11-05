import json, os, time
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

CKPT='/root/autodl-tmp/industrial_vlm/models/Qwen3-VL-8B-Instruct'

def main():
    bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
    print('[load] start',flush=True)
    full=Qwen3VLForConditionalGeneration.from_pretrained(CKPT,quantization_config=bnb,device_map={'':'cuda:0'},dtype=torch.bfloat16,low_cpu_mem_usage=True)
    lm=full.model.language_model
    for p in lm.parameters(): p.requires_grad=False
    # PEFT 0.7.1 expects generation hooks on causal models; the extracted
    # Qwen text backbone is intentionally used through inputs_embeds only.
    if not hasattr(lm, 'prepare_inputs_for_generation'):
        lm.prepare_inputs_for_generation = lambda input_ids=None, past_key_values=None, **kwargs: kwargs
    print('[lm] type',type(lm).__name__,'hidden',getattr(lm.config,'hidden_size',None),flush=True)
    # Adapt only the text-side language model. Do not call get_peft_model on full multimodal model.
    cfg=LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj'])
    try:
        lm_lora=get_peft_model(lm,cfg)
    except Exception as e:
        print('[peft_error]',repr(e),flush=True); raise
    lm_lora.print_trainable_parameters()
    trainable=[(n,p.numel()) for n,p in lm_lora.named_parameters() if p.requires_grad]
    print('[trainable_count]',len(trainable),'first',trainable[:8],flush=True)
    tok=AutoTokenizer.from_pretrained(CKPT,trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    x=tok('### Human: inspect this industrial image.\n### Assistant: There is an anomaly.',return_tensors='pt',add_special_tokens=False).to('cuda:0')
    emb=lm_lora.get_input_embeddings()(x.input_ids)
    B,S=emb.shape[:2]
    pos=torch.arange(S,device='cuda:0').view(1,1,-1).expand(3,B,-1)
    out=lm_lora(inputs_embeds=emb,attention_mask=x.attention_mask,position_ids=pos,use_cache=False,return_dict=True)
    logits=full.lm_head(out.last_hidden_state)
    labels=x.input_ids.clone(); loss=torch.nn.functional.cross_entropy(logits[:,:-1].float().reshape(-1,logits.size(-1)),labels[:,1:].reshape(-1))
    print('[forward] seq',S,'loss',float(loss),'logits',tuple(logits.shape),flush=True)
    loss.backward()
    grads=[(n,float(p.grad.detach().abs().mean())) for n,p in lm_lora.named_parameters() if p.requires_grad and p.grad is not None]
    print('[backward] grad_count',len(grads),'first',grads[:8],flush=True)
    result={'peft_version':'0.7.1','target_modules':['q_proj','k_proj','v_proj','o_proj'],'trainable_parameter_count':sum(n for _,n in trainable),'trainable_tensor_count':len(trainable),'sequence_length':S,'hidden_size':int(lm.config.hidden_size),'loss':float(loss),'gradient_tensor_count':len(grads),'mean_first_grad':float(sum(g for _,g in grads[:8])/max(1,len(grads[:8]))),'peak_vram_gb':torch.cuda.max_memory_allocated()/1e9}
    out_path='lora_language_gate_result.json'; json.dump(result,open(out_path,'w'),indent=2); print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__': main()
