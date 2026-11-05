"""Qwen3-VL-8B 原生视觉 + AnomalyGPT 18-token 注入：前向与冒烟训练。
用法: python native_bridge.py --smoke
"""
import argparse, glob, os, sys, time, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

REPO = "/root/autodl-tmp/AnomalyGPT"
sys.path.insert(0, os.path.join(REPO, "code/code/model"))   # 绕开 model/__init__ 的副作用
from ImageBind import imagebind_model, data                            # noqa: E402
from ImageBind.models.imagebind_model import ModalityType              # noqa: E402
from AnomalyGPT_models import LinearLayer, PromptLearner               # noqa: E402
from visa_pipeline import PIPELINE_VERSION, load_image_tensor, load_image_pil

QWEN = "/root/autodl-tmp/industrial_vlm/models/Qwen3-VL-8B-Instruct"
IB   = os.path.join(REPO, "pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth")
CKPT = os.path.join(REPO, "ckpt/train_visa/pytorch_model.pt")
BPE  = os.path.join(REPO, "code/code/model/ImageBind/bpe/bpe_simple_vocab_16e6.txt.gz")
data.BPE_PATH = BPE

PROMPT_NORMAL   = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
PROMPT_ABNORMAL = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
TEMPLATES       = ['a photo of a {}.', 'a photo of the {}.']

def load_module_checkpoint(module, checkpoint, name, aliases=()):
    """Load one complete module from nested or prefixed checkpoint keys."""
    for key in (name, *aliases):
        nested = checkpoint.get(key)
        if isinstance(nested, dict):
            module.load_state_dict(nested, strict=True)
            return key
        prefix = key + "."
        state = {k[len(prefix):]: v for k, v in checkpoint.items()
                 if k.startswith(prefix)}
        if state:
            module.load_state_dict(state, strict=True)
            return prefix
    raise KeyError("Checkpoint has no complete %s module" % name)

def text_anchors(encoder, cls, device):
    def build(states):
        sents = [t.format(s.format(cls)) for s in states for t in TEMPLATES]
        toks  = data.load_and_transform_text(sents, device)
        with torch.no_grad():
            emb = encoder({ModalityType.TEXT: toks})[ModalityType.TEXT][0]
        emb = emb.reshape(1, len(sents), 1024).mean(1, keepdim=True)
        return emb / emb.norm(dim=-1, keepdim=True)
    return torch.cat([build(PROMPT_NORMAL), build(PROMPT_ABNORMAL)], dim=1)   # [1,2,1024]

@torch.no_grad()
def eq1_map_with_cls(encoder, decoder, pixel_224, anchors):
    """同 eq1_map，但额外返回 ImageBind CLS 特征 [1,1024]"""
    emb = encoder({ModalityType.VISION: pixel_224})
    cls = emb['vision'][0].float()
    patches = decoder([x.float() for x in emb['vision'][1]])
    maps = []
    for p in patches:
        p = p / p.norm(dim=-1, keepdim=True)
        lg = 100.0 * p @ anchors.transpose(-2, -1).float()
        B, L, C = lg.shape; H = int(L ** 0.5)
        up = F.interpolate(lg.permute(0, 2, 1).view(B, C, H, H), size=224,
                           mode='bilinear', align_corners=True)
        maps.append(torch.softmax(up, dim=1)[:, 1])
    return torch.stack(maps).mean(0), cls

@torch.no_grad()
def eq1_map(encoder, decoder, pixel_224, anchors):
    """论文式(1)：投影后归一化 → ×100 → 插值 → softmax → 异常通道 → 4 层平均"""
    return eq1_map_with_cls(encoder, decoder, pixel_224, anchors)[0]

class NativeBridge(nn.Module):
    def __init__(self, load_lora=False, lora_r=8):
        super().__init__()
        from transformers import (AutoProcessor, Qwen3VLForConditionalGeneration,
                                  BitsAndBytesConfig)
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        self.processor = AutoProcessor.from_pretrained(QWEN)
        self.qwen = Qwen3VLForConditionalGeneration.from_pretrained(
            QWEN, quantization_config=bnb, device_map={"": 0},
            dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
        for p in self.qwen.parameters():
            p.requires_grad = False
        self.lm = self.qwen.model.language_model
        self.lm_head = self.qwen.lm_head
        self.hidden = self.qwen.config.text_config.hidden_size
        self.image_token_id = self.qwen.config.image_token_id

        self.encoder, _ = imagebind_model.imagebind_huge({"layers": [7, 15, 23, 31]})
        self.encoder.load_state_dict(torch.load(IB, map_location="cpu"), strict=True)
        self.encoder.half().cuda().eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.decoder = LinearLayer(1280, 1024, 4).float().cuda()
        self.prompt_learner = PromptLearner(1, self.hidden).cuda()
        sd = torch.load(CKPT, map_location="cpu")
        decoder_source = load_module_checkpoint(self.decoder, sd, "image_decoder", ("decoder",))
        prompt_source = load_module_checkpoint(self.prompt_learner, sd, "prompt_learner")
        print("[load] decoder=%s prompt_learner=%s (strict); pipeline=%s"
              % (decoder_source, prompt_source, PIPELINE_VERSION), flush=True)
        for p in self.decoder.parameters():
            p.requires_grad = False
        self.anomaly_proj = nn.Identity()
        self.vis_adapter = None      # 由训练脚本注入；默认恒等
        self.llama_proj = nn.Linear(1024, self.hidden).cuda()   # ImageBind CLS -> LLM 空间          # 4096 -> 4096

        if load_lora:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                             task_type="CAUSAL_LM")
            self.lm = get_peft_model(self.lm, cfg)

    def build_batch(self, image_pil, question, answer, anomaly_tokens, img_emb=None):
        """prompt(含64个视觉token) + 18个异常token + answer，教师强制对齐"""
        proc = self.processor
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        q = proc(text=[prompt], images=[image_pil], return_tensors="pt")
        ans = proc.tokenizer([answer + proc.tokenizer.eos_token], return_tensors="pt",
                             add_special_tokens=False)["input_ids"].cuda()
        ids_p = q["input_ids"].cuda()
        ids = torch.cat([ids_p, ans], dim=1)                     # 完整序列（含答案）
        pv, grid = q["pixel_values"].cuda(), q["image_grid_thw"].cuda()

        with torch.no_grad():
            emb = self.qwen.get_input_embeddings()(ids)
            vis, deep = self.qwen.model.get_image_features(pixel_values=pv, image_grid_thw=grid)
        vis = vis[0] if isinstance(vis, (list, tuple)) else vis
        if getattr(self, "vis_adapter", None) is not None:
            vis = vis + self.vis_adapter(vis.float()).to(vis.dtype)
        pos = (ids_p[0] == self.image_token_id).nonzero(as_tuple=True)[0]
        assert len(pos) == vis.shape[0], (len(pos), vis.shape)
        emb[0, pos] = vis.to(emb.dtype)

        ins = int(pos[-1]) + 1  # Existing protocol: before <vision_end>.
        Lp = ids_p.shape[1]
        tok = anomaly_tokens.to(emb.dtype)
        if img_emb is not None:
            tok = torch.cat([img_emb.to(emb.dtype).view(1, 1, -1), tok], dim=1)
        n = tok.shape[1]
        emb = torch.cat([emb[:, :ins], tok, emb[:, ins:]], dim=1)

        vpm = torch.zeros(1, ids.shape[1], dtype=torch.bool, device=ids.device)
        vpm[0, pos] = True
        vpm = torch.cat([vpm[:, :ins], torch.zeros(1, n, dtype=torch.bool, device=ids.device),
                         vpm[:, ins:]], dim=1)
        labels = torch.cat([torch.full((1, Lp + n), -100, dtype=torch.long, device=ids.device),
                            ans], dim=1)
        attn = torch.ones(emb.shape[:2], dtype=torch.long, device=ids.device)
        return dict(inputs_embeds=emb, attention_mask=attn, labels=labels,
                    visual_pos_masks=vpm, deepstack_visual_embeds=deep)

    def forward(self, batch):
        out = self.lm(inputs_embeds=batch["inputs_embeds"],
                      attention_mask=batch["attention_mask"],
                      visual_pos_masks=batch["visual_pos_masks"],
                      deepstack_visual_embeds=batch["deepstack_visual_embeds"])
        logits = self.lm_head(out.last_hidden_state)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                               batch["labels"][:, 1:].reshape(-1), ignore_index=-100)
        return loss, logits

def find_image(cls_dir):
    fs = sorted(glob.glob(os.path.join(cls_dir, "Data/Images/Normal/*")))
    return fs[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--lora", action="store_true")
    args = ap.parse_args()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    m = NativeBridge(load_lora=args.lora)
    print("[load] %.0fs  vram %.2fGB" % (time.time() - t0, torch.cuda.memory_allocated() / 1e9), flush=True)

    visa = "/root/autodl-tmp/industrial_vlm/datasets/VisA"
    img_path = find_image(os.path.join(visa, "candle"))
    pil = load_image_pil(img_path)
    x = load_image_tensor(img_path, device="cuda")
    anchors = text_anchors(m.encoder, "candle", "cuda")
    with torch.no_grad():
        M = eq1_map(m.encoder, m.decoder, x.half(), anchors)          # [1,224,224]
    tok = m.prompt_learner(M.unsqueeze(1).float())                    # [1,18,4096] 需要梯度
    print("[map] range %.4f..%.4f  tokens %s" % (M.min(), M.max(), tuple(tok.shape)), flush=True)

    params = [p for p in m.prompt_learner.parameters() if p.requires_grad]
    if args.lora:
        params += [p for p in m.lm.parameters() if p.requires_grad]
    print("[train] params %.2fM" % (sum(p.numel() for p in params) / 1e6), flush=True)
    opt = torch.optim.AdamW(params, lr=1e-4)
    q = "This is a photo of a candle for anomaly detection. Is there any anomaly in the image?"
    a = "Yes, there is an anomaly in the image."
    for step in range(args.steps):
        tok = m.prompt_learner(M.unsqueeze(1).float())   # 每步重算，避免复用计算图
        batch = m.build_batch(pil, q, a, tok)
        loss, logits = m.forward(batch)
        opt.zero_grad(); loss.backward(); opt.step()
        g = sum(float(p.grad.norm()) for p in params if p.grad is not None)
        print("[step %d] loss %.4f  seq %d  grad_norm %.3f  vram %.2fGB (peak %.2f)"
              % (step, loss.item(), batch["inputs_embeds"].shape[1], g,
                 torch.cuda.memory_allocated() / 1e9, torch.cuda.max_memory_allocated() / 1e9), flush=True)
    print("[done] peak vram %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))

if __name__ == "__main__":
    main()
