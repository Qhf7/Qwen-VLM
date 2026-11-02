# -*- coding: utf-8 -*-
import re, sys
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.enum.text import WD_ALIGN_PARAGRAPH

src, dst = sys.argv[1], sys.argv[2]
lines = open(src, encoding='utf-8').read().split('\n')

doc = Document()
st = doc.styles['Normal']
st.font.name = 'Calibri'; st.font.size = Pt(10.5)
st.element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
st.paragraph_format.space_after = Pt(4)
st.paragraph_format.line_spacing = 1.12
for lvl, sz in [(1, 16), (2, 13), (3, 11.5)]:
    h = doc.styles[f'Heading {lvl}']
    h.font.name = 'Calibri'; h.font.size = Pt(sz)
    h.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
    h.element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
    h.paragraph_format.space_before = Pt(10); h.paragraph_format.space_after = Pt(4)
sec = doc.sections[0]
sec.left_margin = sec.right_margin = Cm(2.2)
sec.top_margin = sec.bottom_margin = Cm(2.0)

def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto'); shd.set(qn('w:fill'), hexcolor)
    tcPr.append(shd)

def add_table(rows, font=8.5):
    ncol = max(len(r) for r in rows)
    t = doc.add_table(rows=0, cols=ncol)
    t.style = 'Table Grid'
    for i, row in enumerate(rows):
        cells = t.add_row().cells
        for j in range(ncol):
            v = row[j] if j < len(row) else ''
            v = re.sub(r'\*\*(.+?)\*\*', r'\1', v)
            cells[j].text = ''
            p = cells[j].paragraphs[0]
            r = p.add_run(v); r.font.size = Pt(font)
            if i == 0:
                r.bold = True; r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                shade(cells[j], '1F3A5F')
    doc.add_paragraph().paragraph_format.space_after = Pt(2)

i = 0
while i < len(lines):
    line = lines[i].rstrip()
    if not line.strip():
        i += 1; continue
    # code fence
    if line.startswith('```'):
        buf = []; i += 1
        while i < len(lines) and not lines[i].startswith('```'):
            buf.append(lines[i]); i += 1
        i += 1
        p = doc.add_paragraph()
        r = p.add_run('\n'.join(buf)); r.font.name = 'Menlo'; r.font.size = Pt(8.5)
        p.paragraph_format.left_indent = Cm(0.4)
        p.paragraph_format.space_after = Pt(6)
        continue
    # table
    if line.startswith('|') and i + 1 < len(lines) and re.match(r'^\|[\s:\-|]+\|$', lines[i+1].strip()):
        rows = []
        hdr = [c.strip() for c in line.strip('|').split('|')]
        rows.append(hdr); i += 2
        while i < len(lines) and lines[i].startswith('|'):
            rows.append([c.strip() for c in lines[i].strip('|').split('|')]); i += 1
        add_table(rows); continue
    # headings
    m = re.match(r'^(#{1,4})\s+(.*)$', line)
    if m:
        lvl = len(m.group(1)); txt = re.sub(r'\*\*(.+?)\*\*', r'\1', m.group(2))
        if lvl == 1:
            p = doc.add_paragraph(); r = p.add_run(txt)
            r.bold = True; r.font.size = Pt(20); r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
            p.paragraph_format.space_after = Pt(10)
        else:
            doc.add_heading(txt, level=min(lvl - 1, 3))
        i += 1; continue
    # hr
    if line.strip() == '---':
        doc.add_paragraph('─' * 40).alignment = WD_ALIGN_PARAGRAPH.CENTER
        i += 1; continue
    # list
    m = re.match(r'^(\s*)[-*]\s+(.*)$', line)
    if m:
        doc.add_paragraph(re.sub(r'\*\*(.+?)\*\*', r'\1', m.group(2)), style='List Bullet')
        i += 1; continue
    m = re.match(r'^(\s*)\d+\.\s+(.*)$', line)
    if m:
        doc.add_paragraph(re.sub(r'\*\*(.+?)\*\*', r'\1', m.group(2)), style='List Number')
        i += 1; continue
    # paragraph (strip bold markers, handle inline code)
    txt = re.sub(r'\*\*(.+?)\*\*', r'\1', line)
    p = doc.add_paragraph()
    for part in re.split(r'(`[^`]+`)', txt):
        r = p.add_run(part.strip('`'))
        r.font.size = Pt(10.5)
        if part.startswith('`'):
            r.font.name = 'Menlo'; r.font.size = Pt(9)
    i += 1

doc.save(dst)
print('saved', dst)
