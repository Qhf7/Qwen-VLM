#!/usr/bin/env python3
"""Server-only aggregate completion report; no raw image, legacy or label reads."""
from __future__ import annotations
import argparse,hashlib,json,math,os,socket
from datetime import datetime,timezone,timedelta
from pathlib import Path
HOST='autodl-container-132444827f-9c594038'
SCHEMA='jev-continuous-research-completion-report-v1'
def require(ok,msg):
    if not ok:raise ValueError(msg)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()
def publish(p,value,plain=False):
    p=Path(p);raw=value.encode() if plain else json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False).encode()+b'\n'
    if p.exists():require(p.read_bytes()==raw,'Frozen report differs: '+p.name);return
    temp=p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
    with temp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    try:os.link(temp,p)
    finally:temp.unlink(missing_ok=True)
def pct(v):return f'{100*v:.2f}%'
def num(v):return f'{v:.6f}'
def diff(v,pp=False):
    k='delta' if 'delta' in v else 'value';scale=100 if pp else 1;unit='个百分点' if pp else ''
    return f"{scale*v[k]:+.4f}{unit}，95%区间[{scale*v['ci95'][0]:+.4f}, {scale*v['ci95'][1]:+.4f}]"
def table_metrics(lines,title,rows):
    lines.extend(['',title,'','| 版本 | AUROC | AP | MacroF1 | Recall | FPR | Brier | ECE10 |','|---|---:|---:|---:|---:|---:|---:|---:|'])
    for name,m in rows:lines.append('| '+name+' | '+' | '.join(pct(m[k]) for k in ('auroc','ap','macro_f1','recall','fpr'))+' | '+num(m['brier'])+' | '+num(m['ece_10_equal_width'])+' |')
def table_uncertainty(lines,methods):
    lines.extend(['','| 版本 | TP/FP/TN/FN | Recall计数与Wilson95% | FPR计数与Wilson95% | Binary log loss |',
        '|---|---|---|---|---:|'])
    for name,m in methods.items():
        pieces=[]
        for key in ('recall','fpr'):
            v=m[key];pieces.append(f"{v['count']}/{v['denominator']}；[{pct(v['ci95'][0])}, {pct(v['ci95'][1])}]")
        confusion='/'.join(str(m['confusion'][k]) for k in ('TP','FP','TN','FN'))
        lines.append('| '+name+' | '+confusion+' | '+' | '.join(pieces)+' | '+num(m['binary_logloss'])+' |')
def build(root):
    required=['stages/11_summary.json','stages/11_summary_status.json','stage12/run/metrics.json','stages/12_postexecution_audit.json',
        'stage13_confirmation/metrics.json','stage13_confirmation/completion_certificate.json','stages/13_postexecution_audit.json',
        'stage12/run/STATISTICAL_SUPPLEMENT.json','stage13_confirmation/STATISTICAL_SUPPLEMENT.json','stages/13_crossstage_source_audit.json']
    missing=[n for n in required if not(root/n).is_file()]
    if missing:
        print(json.dumps(dict(status='pending',missing=missing,partial_final_report_produced=False),ensure_ascii=False),flush=True);return 0
    s=read(root/'stages/11_summary.json');ss=read(root/'stages/11_summary_status.json');m12=read(root/'stage12/run/metrics.json')
    m13=read(root/'stage13_confirmation/metrics.json');p12=read(root/'stages/12_postexecution_audit.json');p13=read(root/'stages/13_postexecution_audit.json')
    cert=read(root/'stage13_confirmation/completion_certificate.json');bundle=read(root/'stage13_confirmation/final_system_bundle.json')
    sup12=read(root/'stage12/run/STATISTICAL_SUPPLEMENT.json');sup13=read(root/'stage13_confirmation/STATISTICAL_SUPPLEMENT.json')
    sources=read(root/'stages/13_crossstage_source_audit.json')
    for n,v in [('11summary',s),('11status',ss),('12metrics',m12),('13metrics',m13),('13certificate',cert),('12audit',p12),('13audit',p13)]:require(v['status']=='complete','Incomplete '+n)
    for v in (p12,p13):require(v['failed']==0 and v['passed']>0 and all(x['passed'] for x in v['checks']),'Independent audit failed')
    require(ss['summary_sha256']==sha(root/'stages/11_summary.json') and ss['report_sha256']==sha(root/'FINAL_FOLLOWUP_REPORT.md'),'Frozen11 changed')
    for field,name in [('metrics_sha256','metrics.json'),('report_sha256','REPORT.md'),('prediction_freeze_sha256','prediction_freeze.json'),('bundle_sha256','final_system_bundle.json')]:
        require(cert[field]==sha(root/'stage13_confirmation'/name),'Final certificate changed: '+name)
    require(p12['metrics_sha256']==sha(root/'stage12/run/metrics.json') and p13['metrics_sha256']==sha(root/'stage13_confirmation/metrics.json'),'Post audit metrics identity differs')
    require(cert['completed']==cert['expected']==m13['denominator']==240 and m13['primary']=='primary_calibrated' and m13['secondary']=='secondary_raw','Final primary/population differs')
    require(m12['heldout_used_for_fit'] is False and m13['no_refit'] is True and m13['no_threshold_optimization'] is True and m13['selection_independent_of_confirmation'] is True,'Final isolation differs')
    require(bundle['primary']=='calibrated' and bundle['secondary']=='raw' and bundle['threshold']==m13['threshold']==.5 and bundle['calibrator']==m12['calibrator'],'Final mapping differs')
    require(m12['confirmation_opened'] is False and m13['confirmation_opened'] is True and cert['masks_opened'] is False,'Stage role flags differ')
    # This source-union certificate is produced by the input-only independent auditor.
    require(sources['status']=='complete' and sources['union_unique_source_ids']==600 and sources['union_unique_original_sha256']==600
        and sources['source_counts']==dict(development120=120,calibration_heldout240=240,confirmation240=240)
        and all(x['source_overlap']==x['original_sha256_overlap']==0 for x in sources['pairwise_intersections'].values()),
        'Need independently verified600 query source/byte union')
    op=s['execution_audits'];q=s['matched448'];su=s['supports'];res=s['resolution'];roi=s['roi'];routing=s['routing']
    main=m13['results']['primary_calibrated']['metrics'];raw=m13['results']['secondary_raw']['metrics'];fit=bundle['calibrator'];a=fit['coefficient'][0][0];b=fit['intercept'][0]
    require(a>0 and math.isfinite(a) and math.isfinite(b),'Expected positive finite mapping')
    formal=[('固定ROI/duplicate',op['roi']),('支持0/1/4-shot',op['support']),('新320分辨率',op['resolution']),('calibration120＋heldout120',m12['operations']),('confirmation240',m13['operations'])]
    count=lambda v:v['explicit_failed'] if 'explicit_failed' in v else v['failed']
    accepted=sum(v['accepted'] for _,v in formal);started=sum(v['started'] for _,v in formal);failed=sum(count(v) for _,v in formal)
    require(accepted==1440 and started==accepted+failed and m12['operations']['unknown']==m13['operations']['unknown']==0,'Formal execution count mismatch')
    date=datetime.fromisoformat(m13['timestamp_utc']).astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
    L=['# Jev与Qwen工业异常检测：连续研究完成报告','',
        '本文件汇总服务器阶段01—13的完整冻结结果，补充原FINAL_FOLLOWUP_REPORT.md；历史合同、预测、数据划分与旧报告保持原样。',
        f'最终确认评价完成时间：{date}（北京时间）。后续推理、付费API评测、验证和统计均在指定AutoDL服务器完成。','',
        '## 1. 完成情况与主要结论','',
        f"预声明主系统为Jev 1.14.0＋4张同类正常参考图（seed43）＋448输入＋固定Logistic概率映射，阈值p≥0.5；完整240张确认图AUROC为 **{pct(main['auroc'])}**，Recall **{pct(main['recall'])}**，FPR **{pct(main['fpr'])}**，Brier **{num(main['brier'])}**。",
        '阶段13只评估这一固定Jev系统及同次raw概率诊断，没有在confirmation上测试Qwen或其他候选，因此不作确认集跨模型优越性结论。',
        '- 共同448开发输入下，监督Qwen LoRA相对量化底座的pooled排序改善明确；逐类别宏平均区间跨零。',
        '- 两组固定支持bank中的few-shot正常参考改善开发排序；它是推理上下文，不更新Jev模型权重。',
        '- ROI复核尚未证明定位净收益；路由仅为离线回放，未证明在线Agent或省费收益。',
        '- 独立校准与heldout开发评价、最终一次性confirmation有不同角色，不能把各阶段分数混为同一测试。','',
        '## 2. 阶段01—13工作总表','','| 阶段 | 工作 | 范围 | 完成结果与边界 | 主要证据 |','|---|---|---|---|---|']
    stages=[
        ('01','来源与隔离审计','旧开发120、原生训练768','核验source/SHA；字节隔离不证明物理零件独立','stages/01_source_audit.json'),
        ('02','Teacher对齐','旧query120、支持48','query训练暴露96、验证9、测试15；未见24不是新盲测','stages/02_teacher_alignment.json'),
        ('03','固定候选提取','旧开发120','60/120为整图框，候选几何质量是瓶颈','stages/03_candidates.json'),
        ('04','ROI预测冻结','ROI120＋duplicate120','预测完整持久化后评价；不据结果重挑框','stages/04_prediction_freeze.json'),
        ('05','候选区域评价','全60异常＋60正常','尚未证明ROI定位净收益，保留负结果','stages/05_evaluation.json'),
        ('06','顺序路由回放','已采集240响应','逻辑调用预算回放，不等于在线实际省费','stages/06_routing_results.json'),
        ('07','开发OOF校准/拒答','旧120、5折、25个拟合','开发交叉拟合诊断，不替代独立校准或确认','stages/07_calibration_results.json'),
        ('08','支持种子稳健性','同120×5条件＝600判断','共享0shot一次；43/44各1/4shot；独立运维救援补齐3条','stage08_rescue/metrics.json'),
        ('09','共同448输入Qwen','base/DD各120','同PNG、两路native tensor一致；480候选forward','stage09_qwen_matched/final_audit.json'),
        ('10','320/448分辨率','新320120＋复用448120','只复用shared0，排除480支持响应；调用时间未交错','stage10_resolution/metrics.json'),
        ('11','完整研究汇总','阶段01—10冻结统计','保留负结果、区间及审计，原报告不覆盖','stages/11_summary.json'),
        ('12','独立校准与heldout','cal120＋heldout120','只用cal120拟合；系数保存后评价heldout，不再择主系统','stage12/run/metrics.json'),
        ('13','一次性最终确认','原冻结test240','最终bundle后一次开封；calibrated主/raw次；全240阈值评价','stage13_confirmation/metrics.json')]
    for row in stages:L.append('| '+' | '.join(row)+' |')
    L+=['','## 3. 数据角色和唯一来源','',
        '| 角色 | 来源数 | 用途 | 评价定位 |','|---|---:|---|---|',
        '| 历史原生监督训练 | 768 | Qwen LoRA训练；支持从训练正常来源固定抽取 | 训练 |',
        '| 旧开发query | 120 | 基线、ROI、OOF、支持和分辨率 | 已多次查看的开发数据 |',
        '| 独立calibration | 120 | 固定一维Logistic映射拟合 | 校准数据 |',
        '| 来源互斥heldout | 120 | 映射固定后检验可靠性和阈值 | 开发评价，不是confirmation |',
        '| 原冻结confirmation | 240 | 最终系统后一次开启，全240主评价 | 本轮最终确认 |','',
        '输入独立审计已核：旧开发120＋阶段12的240＋阶段13的240，唯一query来源与原图SHA并集均为600，三组两两source/byte交集为0。训练768不计入这600个query。',
        '阶段12来源虽然与旧120和训练来源隔离，但native验证剩余池可能曾参与历史验证，不能称全新盲测。阶段13历史构建使用类别/标签元数据分层，本轮预测与评价标签在最终系统前封存；“盲”仅指本轮操作顺序，不证明所有历史模型完全没见过图像。',
        'Teacher划分不同于原生Qwen：旧query96在Teacher训练划分，未见24只有22异常/2正常；Teacher未见子集不是新盲测。公开VisA预训练污染、物理零件与近重复独立性仍未知。','',
        '## 4. zero-shot和few-shot正常参考','',
        '固定模型、问题/criteria、query和448渲染。0-shot每请求只给query；1-shot增加1张同类正常图；4-shot增加4张。1-shot嵌套4-shot首张，seed与顺序预先固定，不按query结果挑图，不训练Jev参数。','',
        '| 条件 | AUROC | AP | Recall（p≥0.5） | FPR |','|---|---:|---:|---:|---:|']
    for name,v in su['arms'].items():m=v['metrics'];L.append('| '+name+' | '+' | '.join(pct(m[k]) for k in ('auroc','ap','recall','fpr'))+' |')
    for shot,v in su['seed_point_summary'].items():
        lo,hi=v['descriptive_range']['auroc'];L.append(f"- {shot}-shot两bank均值{pct(v['descriptive_mean']['auroc'])}，范围[{pct(lo)}, {pct(hi)}]；是两固定bank点估计，非支持总体方差区间。")
    for pair in ('s43_4-minus-shared0','s44_4-minus-shared0'):L.append('- '+pair+' AUROC：'+diff(su['paired_bootstrap'][pair]['pooled']['auroc'],True)+'。')
    L+=['384张训练正常候选每类32张经原图SHA核验，与query来源/字节无交。4-shot仍有23.33% FPR，ECE不保证单调改善。旧seed42历史0/1/4shot AUROC82.54%/87.19%/88.85%；renderer不同，只保留历史对照，不纯归因seed。','',
        '## 5. 共同448输入的监督Qwen LoRA对照','',
        '| 路线 | AUROC | AP | Recall（logodds>0） | FPR | TP/FP/TN/FN |','|---|---:|---:|---:|---:|---|']
    for name in ('base','decision_decision'):
        m=q['arms'][name];L.append('| '+name+' | '+' | '.join(pct(m[k]) for k in ('auroc','ap','recall','fpr'))+' | '+'/'.join(str(m[k]) for k in ('tp','fp','tn','fn'))+' |')
    L+=['- DD−base pooled AUROC：'+diff(q['paired_bootstrap']['decision_decision-minus-base']['pooled']['auroc'],True)+'。',
        '- 逐类别宏平均AUROC：'+diff(q['paired_bootstrap']['decision_decision-minus-base']['category_macro']['auroc'],True)+'，区间跨零，pcb3/capsules等仍有弱项。',
        '- base/DD均252语言NF4 double-quant层、BF16 compute、视觉BF16、SDPA；DD有144个q/k/v/o注意力LoRA模块，r16/alpha32/dropout0.05。本轮只推理，无新训练。',
        '- 同query两路PNG与native像素tensor一致，grid=[1,28,28]、196个merged视觉token；每图true/false各1个候选forward，240判断＝480forward，无额外预热。',
        '- 历史原图82.39%→94.56%（+12.17个百分点）与新共同448的82.44%→93.49%属于不同输入预算，不能混写。',
        '- Jev/Qwen只匹配PNG/图数，问题、输出协议、监督训练预算与内部算力不同；不能把Qwen监督LoRA和Jev推理参考数当等训练成本。','',
        '## 6. 分辨率和ROI负结果','',
        '| 0shot输入 | AUROC | AP | Recall | FPR |','|---|---:|---:|---:|---:|']
    for name,v in res['arms'].items():m=v['metrics'];L.append('| '+name+' | '+' | '.join(pct(m[k]) for k in ('auroc','ap','recall','fpr'))+' |')
    L+=['320−448 AUROC：'+diff(res['paired_bootstrap']['pooled']['auroc'],True)+'；区间跨零。下采样同时改变细节/预算，调用时间不交错，不独立归因算力或分辨率。','',
        '| 候选门控 | 全60异常meanIoU | IoU≥0.5 | 保留召回 | 全60正常出框率 |','|---|---:|---:|---:|---:|']
    for name,m in roi['methods'].items():L.append('| '+name+' | '+num(m['anomaly_bbox_iou_mean_all'])+' | '+' | '.join(pct(m[k]) for k in ('anomaly_iou_at_05_all','anomaly_retained_recall','normal_positive_box_rate'))+' |')
    d=roi['roi_minus_duplicate']['pooled'];rd=routing['uncertain_minus_duplicate']['pooled']
    L+=['ROI−duplicate meanIoU：'+diff(d['anomaly_bbox_iou_mean_all'])+'；保留召回：'+diff(d['anomaly_retained_recall'],True)+'。',
        '60/120候选为整图框；门控只能保留/删除，不能修复几何。ROI定位净收益未被证明，负结果保留。',
        '不确定路由离线162逻辑请求/42升级，Recall70.00%、正常出框率23.33%；相对duplicate召回'+diff(rd['anomaly_retained_recall'],True)+'，正常出框率'+diff(rd['normal_positive_box_rate'],True)+'。已实际采集的响应不因回放而省费，也不是在线Agent部署。','',
        '## 7. 阶段12：独立校准与heldout开发评价','',
        f'固定映射：p_cal = sigmoid({a:.14f} × logit(clip(p_raw,1e-6,1−1e-6)) {b:+.14f})。只用calibration120拟合，固定C1/L2/lbfgs/random_state42；系数保存后才评价heldout120。',
        '系数为正使映射严格保序，AUROC/AP不变是数学性质，不是重新训练Jev；概率值与固定0.5阈值的决策可以改变。']
    table_metrics(L,'heldout120固定结果',[(k,v['metrics']) for k,v in m12['heldout'].items()])
    table_uncertainty(L,sup12['methods'])
    L+=['- calibrated−raw Brier：'+diff(m12['paired_bootstrap']['pooled']['brier'])+'；负值代表平方误差改善。',
        '- calibrated−raw ECE10：'+diff(m12['paired_bootstrap']['pooled']['ece_10_equal_width'])+'，区间跨零。',
        '- raw FPR11/60→calibrated6/60；不能由10%点估计声称生产FPR≤10%。',
        '- 主calibrated/rawsecondary在12结果前固定，未依heldout结果切换主系统。旧120上的OOF25拟合只是历史诊断，不等同本次独立cal120映射。']
    L.append('固定映射−raw binary log loss：'+diff(sup12['paired_logloss_difference'])+'；这与Brier一起描述概率质量，ECE仍受分箱与小样本影响。')
    table_metrics(L,'## 8. 阶段13：固定系统的一次性确认240',[(k,v['metrics']) for k,v in m13['results'].items()])
    table_uncertainty(L,sup13['methods'])
    L.append('主映射−raw binary log loss：'+diff(sup13['paired_logloss_difference']['pooled'])+'。Wilson区间对应独立二项率描述，bootstrap区间固定本次来源/校准器，二者估计对象不同。')
    L+=['','| 主系统指标 | 点估计 | 95%固定映射分层bootstrap区间 |','|---|---:|---|']
    for k,v in m13['bootstrap']['intervals']['primary_calibrated'].items():
        fmt=num if k in ('brier','ece_10_equal_width') else pct;L.append('| '+k+' | '+fmt(v['value'])+' | ['+fmt(v['ci95'][0])+', '+fmt(v['ci95'][1])+'] |')
    L+=['- 主映射−raw Brier：'+diff(m13['bootstrap']['primary_minus_secondary']['brier'])+'。',
        '- 主映射−raw ECE10：'+diff(m13['bootstrap']['primary_minus_secondary']['ece_10_equal_width'])+'。',
        '- 主映射−raw Recall：'+diff(m13['bootstrap']['primary_minus_secondary']['recall'],True)+'；FPR：'+diff(m13['bootstrap']['primary_minus_secondary']['fpr'],True)+'。',
        '- 本次映射减少5个误报，同时少检出4个异常；Recall损失的配对区间不跨零，不能写成校准全面提高分类表现。MacroF1变化'+diff(m13['bootstrap']['primary_minus_secondary']['macro_f1'],True)+'，区间跨零。',
        '- 不论方向好坏均不再选择模型、问题、seed、阈值或映射；240主评价全部保留，不拒答。以下仅预声明回顾覆盖率风险诊断，不等于部署覆盖保证。','',
        '| 版本 | 覆盖率 | 接收/240 | 接收异常/正常 | 接收错误率 |','|---|---:|---:|---:|---:|']
    for name,v in m13['results'].items():
        for coverage,item in v['retrospective_coverage'].items():
            m=item['metrics'];L.append('| '+name+' | '+pct(item['actual_coverage'])+f" | {item['accepted']}/240 | {m['anomalous']}/{m['normal']} | "+pct(item['error_rate'])+' |')
    L+=['最终确认仅有固定Jev及其raw secondary，不能声称确认集优于未跑的Qwen、AnomalyGPT或其他系统。teacher历史暴露需另列，不宣称对所有既往模型完全未知。','',
        '## 9. 正式API执行量与恢复过程','',
        '| 正式实验 | 接受logical targets | Started | 显式失败 | 未决 |','|---|---:|---:|---:|---:|']
    for name,v in formal:L.append('| '+name+f" | {v['accepted']} | {v['started']} | {count(v)} | "+str(v.get('unknown','0（完整冻结审计）'))+' |')
    L += [f'| 正式合计 | {accepted} | {started} | {failed} | 0 |','',
        '1440＝ROI240＋支持600＋新320120＋stage12 240＋stage13 240，是正式API目标数，不是1440张唯一图。独立source/byte审计核实query并集为600；同development120的多臂比较不能当新增独立query。',
        '独立单目标预检另列在其预检目录，不混入正式合计；旧360条历史基线也不混入。原生Qwen另列240图像判断＝480候选forward。',
        'stage08原597接受后3个目标均三次明确502耗尽cap3，原科学合同/seed/PNG/问题/请求UUID/body/日志保持不变；显式单并发运维修订补3条。旧status仍保留未完整，独立completion证书诚实标识合并完整。',
        '最终确认单目标最多3次、全局started≤720；累计失败30停止新提交，已reserved inflight最多drain到32。timeout/transport/HTTP200 typed解析异常永久unknown，不得当明确502重发。',
        '费用只依据实际返回usage，缺失/失败usage未知、不补0；没有可靠货币账单，不从token编造费用。服务model回显也不是上游权重不可变的证明。','',
        '## 10. 审计与解释边界','',
        f"阶段12独立审核{p12['passed']}通过/0失败；阶段13独立审核{p13['passed']}通过/0失败。最终bundle另42项核验及7组合成ledger边界通过；审计项可能重复，不相加当独立证据数。",
        '表中率的Wilson区间为二项比例区间；其余所列bootstrap区间采用2000次、seed42、category×label分层，固定当前query/支持/拟合校准器/服务响应，不含训练seed、calibration拟合、provider重复与支持总体方差，未做多重比较校正。',
        '类均衡50%异常先验不等于生产先验；误报/漏检分母和区间不可省略。来源/byte隔离不等于近重复或物理零件独立，公开基准预训练污染未知。阶段01—11“未打开confirmation”是历史阶段状态；阶段13现已按冻结流程完成最终评价。',
        '本报告只读取完整冻结聚合，不打开legacy确认成员、保护labels、RGB或GT mask，不新增模型/API/拟合。','',
        '## 11. 后续工作','',
        '本轮确认应作为不可回调终点。任何新问题、支持、模型、阈值、候选器或校准器修改都建立新开发和独立测试协议，不再把已打开240称新盲测。',
        '1. 以训练数据冻结更好的候选器，验证meanIoU、IoU≥0.5、整图框比例与Teacher监督暴露。',
        '2. 预声明更多支持bank和时间交错provider重复，测支持/服务方差、实际usage与账单。',
        '3. 在新来源与生产异常先验上验证固定系统的概率可靠性、FPR上界和业务效用。',
        '4. 预测收益已建立后，再做在线路由/Agent端到端时延、真实费用与误报/漏检效用。','',
        '## 12. 冻结证据索引','']
    figures=[('最终固定系统确认240','confirmation_fixed_system_analysis'),('独立calibration/heldout','calibration_heldout_analysis'),
        ('支持bank与共同448Qwen开发对照','support_qwen_matched_comparison'),('320/448零参考分辨率','resolution_tradeoff')]
    L+=['### 12.1 结果图索引','', '| 图 | PNG | 来源/审计 |','|---|---|---|']
    for title,name in figures:L.append('| '+title+' | figures/'+name+'.png | figures/'+name+'.provenance.json |')
    L+=['','确认图与heldout图对应不同来源，不横向拼成一份测试；Qwen图仅是旧开发120的共同输入对照。各图PNG/SVG/provenance均按下列SHA索引留存，独立visual QA结果由图形审计单独记录。','', '### 12.2 文件SHA256','']
    figure_files=['figures/'+name+suffix for _,name in figures for suffix in ('.png','.svg','.provenance.json')]
    files=required+['FINAL_FOLLOWUP_REPORT.md','stage12/run/calibrator.json','stage12/run/prediction_freeze.json',
        'stage13_confirmation/final_system_bundle.json','stage13_confirmation/prediction_freeze.json',
        'stages/13_runner_preopening_audit.json','stages/13_bundle_preopening_audit.json','code/compile_continuous_completion.py']+figure_files
    for name in files:L.append('- '+name+'：'+sha(root/name))
    L+=['','各阶段原始结果保留，新增报告与stages/14_completion_report_manifest.json同步本地；不拷legacy确认元数据或保护标签。','']
    # Keep paragraphs/lists separated for CommonMark renderers.
    spaced=[]
    for line in L:
        is_list=line.startswith('- ') or (len(line)>2 and line[0].isdigit() and line[1:3]=='. ')
        previous=spaced[-1] if spaced else ''
        previous_list=previous.startswith('- ') or (len(previous)>2 and previous[0].isdigit() and previous[1:3]=='. ')
        if is_list and previous and not previous_list:spaced.append('')
        if line and not is_list and previous_list:spaced.append('')
        spaced.append(line)
    report='\n'.join(spaced)
    # Publication copy: the original guard also matched a server-credential prefix; removed here.
    require(not any(x in report for x in ('mgk_live_','Bearer ')),'Credential-like string in report')
    manifest=dict(schema=SCHEMA,status='complete',hostname=HOST,source_files_sha256={n:sha(root/n) for n in files},script_sha256=sha(__file__),
        report_sha256=hashlib.sha256(report.encode()).hexdigest(),final_evaluation_timestamp_utc=m13['timestamp_utc'],
        formal_api=dict(accepted_logical_targets=accepted,started=started,failed=failed,unknown=0),unique_query_sources=600,unique_query_bytes=600,
        qwen=dict(image_decisions=240,boolean_candidate_forwards=480),new_api_requests=0,model_calls=0,new_fits=0,
        protected_labels_opened_by_report=False,legacy_confirmation_members_opened_by_report=False,images_or_masks_opened_by_report=False,previous_reports_unchanged=True)
    publish(root/'CONTINUOUS_RESEARCH_COMPLETION.md',report,True);publish(root/'stages/14_completion_report_manifest.json',manifest)
    print(json.dumps(dict(status='complete',report_sha256=sha(root/'CONTINUOUS_RESEARCH_COMPLETION.md'),manifest_sha256=sha(root/'stages/14_completion_report_manifest.json'),formal_api=manifest['formal_api'],protected_labels_opened_by_report=False)),flush=True)
    return 0
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    require(socket.gethostname()==HOST,'Compile only on designated AutoDL server');return build(a.root.resolve())
if __name__=='__main__':raise SystemExit(main())

