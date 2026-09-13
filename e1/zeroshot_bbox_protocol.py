"""Prompt and strict parser for native Qwen3 zero-shot box-to-mask baseline."""
import json, re
from typing import Any

PROMPT = '''Inspect this single image of an industrial component for a visible defect. A defect is a localized scratch, crack, deformation, discoloration, stain, missing or broken part, pit, burn mark, coating problem, contaminant, or other abnormality. Normal edges, holes, reflections, shadows, text and regular texture are not defects.
Return exactly one JSON object and no other text. Use this schema: {"defect":true,"bbox":[x1,y1,x2,y2]} when a defect is visible, with integer coordinates in the 0..448 image coordinate system (x1< x2, y1< y2). Use {"defect":false,"bbox":null} for a normal image. Do not guess a box for a normal image.'''

def parse(text: str) -> dict[str, Any]:
    m=re.search(r'\{.*?\}',text,re.S)
    if not m: return {'valid':False,'reason':'no_json','raw':text}
    try: x=json.loads(m.group(0))
    except Exception: return {'valid':False,'reason':'invalid_json','raw':text}
    if not isinstance(x.get('defect'),bool): return {'valid':False,'reason':'bad_defect','raw':text}
    b=x.get('bbox')
    if not x['defect'] and b is None: return {'valid':True,'defect':False,'bbox':None}
    if not isinstance(b,list) or len(b)!=4: return {'valid':False,'reason':'bad_bbox','raw':text}
    try: b=[max(0,min(448,int(v))) for v in b]
    except Exception: return {'valid':False,'reason':'non_integer_bbox','raw':text}
    if not (b[0]<b[2] and b[1]<b[3]): return {'valid':False,'reason':'degenerate_bbox','raw':text}
    return {'valid':True,'defect':True,'bbox':b}

def rasterize(parsed: dict[str,Any], side=448):
    import numpy as np
    mask=np.zeros((side,side),dtype=np.uint8)
    if parsed.get('valid') and parsed.get('defect'):
        x1,y1,x2,y2=parsed['bbox']; mask[y1:y2,x1:x2]=1
    return mask
