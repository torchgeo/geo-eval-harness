"""Render :mod:`experiments.seg_runs.curves` runs as one self-contained interactive HTML page."""

import json
from datetime import UTC, datetime
from pathlib import Path

from experiments.seg_runs.curves import ACTIONS, Run

_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Segmentation curves</title>
<style>
body{font:13px system-ui,sans-serif;margin:12px}
#filters{display:flex;gap:10px;flex-wrap:wrap;align-items:flex-start}
#filters label{display:flex;flex-direction:column;font-weight:600}
select[multiple]{min-width:150px;height:110px}
#tablewrap{max-height:38vh;overflow:auto;border:1px solid #ccc;margin-top:8px}
table{border-collapse:collapse;width:100%}
th,td{padding:2px 6px;border-bottom:1px solid #eee;white-space:nowrap;text-align:left}
th{position:sticky;top:0;background:#f4f4f4;cursor:pointer}
tr.sel{background:#e8f0ff}
.high{color:#b00020;font-weight:600}.medium{color:#b26a00}.low{color:#555}
#plots{height:62vh}
.muted{color:#666}
</style>
<script>__PLOTLY__</script></head><body>
<div><b>Segmentation training curves</b> <span class="muted">generated __GENERATED__ from __ROOTS__ -
__NRUNS__ runs</span></div>
<div id="filters">
 <label>campaign<select id="f_campaign" multiple></select></label>
 <label>dataset<select id="f_dataset" multiple></select></label>
 <label>model<select id="f_model" multiple></select></label>
 <label>regime<select id="f_regime" multiple></select></label>
 <label>status<select id="f_status" multiple></select></label>
 <label>flag rule<select id="f_rule" multiple></select></label>
 <label>job contains<input id="f_text" placeholder="substring, space = AND"></label>
 <div><button id="b_plot">plot all matching (max 40)</button><br><button id="b_clear">clear plot</button>
 <br><label style="flex-direction:row;font-weight:400"><input type="checkbox" id="f_flagged"> flagged only</label>
 <br><span class="muted">Ctrl/Cmd-click: multi-select. Click a row: toggle in plot.</span></div>
</div>
<div id="tablewrap"><table><thead><tr id="thead"></tr></thead><tbody id="tbody"></tbody></table></div>
<div id="plots"></div>
<details><summary>Actions</summary><pre>__ACTIONS__</pre></details>
<script>
const RUNS = __DATA__;
const COLS = [["campaign","campaign"],["job","job"],["dataset","dataset"],["model","model"],["regime","regime"],
 ["status","status"],["n","ep"],["best_epoch","best ep"],["best_val_miou","best val"],["test_miou","test"],["flags","flags"]];
RUNS.forEach((r,i)=>{r.i=i;r.n=r.epochs.length;
 r.flags=r.findings.map(f=>f.rule).join(", ");
 r.sev=r.findings.length?Math.min(...r.findings.map(f=>({high:0,medium:1,low:2})[f.severity])):3;});
const selected = new Set(); let sortKey="campaign", sortDir=1;
function fill(id,key,list){const s=document.getElementById(id);
 const vals=[...new Set(list?RUNS.flatMap(r=>r.findings.map(f=>f.rule)):RUNS.map(r=>String(r[key])))].sort();
 vals.forEach(v=>{const o=document.createElement("option");o.textContent=v;s.appendChild(o)});
 s.onchange=render;}
fill("f_campaign","campaign");fill("f_dataset","dataset");fill("f_model","model");fill("f_regime","regime");
fill("f_status","status");fill("f_rule",null,true);
["f_text","f_flagged"].forEach(id=>document.getElementById(id).oninput=render);
const picked=id=>[...document.getElementById(id).selectedOptions].map(o=>o.value);
function matching(){
 const c=picked("f_campaign"),d=picked("f_dataset"),m=picked("f_model"),g=picked("f_regime"),s=picked("f_status"),
  rules=picked("f_rule"),words=document.getElementById("f_text").value.toLowerCase().split(/\\s+/).filter(Boolean),
  flagged=document.getElementById("f_flagged").checked;
 return RUNS.filter(r=>(!c.length||c.includes(r.campaign))&&(!d.length||d.includes(r.dataset))&&
  (!m.length||m.includes(r.model))&&(!g.length||g.includes(r.regime))&&(!s.length||s.includes(r.status))&&
  (!rules.length||r.findings.some(f=>rules.includes(f.rule)))&&(!flagged||r.findings.length)&&
  words.every(w=>(r.campaign+"/"+r.job).toLowerCase().includes(w)));}
const fmt=v=>v==null?"-":(typeof v==="number"&&!Number.isInteger(v)?v.toFixed(4):v);
function render(){
 const th=document.getElementById("thead");th.innerHTML="";
 COLS.forEach(([k,label])=>{const e=document.createElement("th");e.textContent=label+(k===sortKey?(sortDir>0?" ▲":" ▼"):"");
  e.onclick=()=>{sortDir=k===sortKey?-sortDir:1;sortKey=k;render()};th.appendChild(e)});
 const rows=matching().sort((a,b)=>{const x=sortKey==="flags"?a.sev:a[sortKey],y=sortKey==="flags"?b.sev:b[sortKey];
  return (x>y?1:x<y?-1:0)*sortDir;});
 const tb=document.getElementById("tbody");tb.innerHTML="";
 rows.forEach(r=>{const tr=document.createElement("tr");if(selected.has(r.i))tr.className="sel";
  COLS.forEach(([k])=>{const td=document.createElement("td");
   if(k==="flags"){r.findings.forEach(f=>{const s=document.createElement("span");s.className=f.severity;
     s.textContent=f.rule+" ";s.title=f.detail+" -> "+f.action;td.appendChild(s)});}
   else td.textContent=fmt(r[k]);tr.appendChild(td)});
  tr.onclick=()=>{selected.has(r.i)?selected.delete(r.i):selected.add(r.i);render();plot()};tb.appendChild(tr)});}
const PANELS=[["train_loss","train loss"],["val_loss","val loss"],["train_miou","train mIoU"],["val_miou","val mIoU"]];
const COLORS=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"];
function plot(){
 const traces=[];[...selected].forEach((i,k)=>{const r=RUNS[i],color=COLORS[k%COLORS.length],
  name=r.campaign+"/"+r.job, dash=r.regime==="frozen"?"dash":"solid";
  PANELS.forEach(([key],p)=>{traces.push({x:r.epochs.map(e=>e.epoch),y:r.epochs.map(e=>e[key]),mode:"lines+markers",
   marker:{size:4},line:{color,dash},name,legendgroup:name,showlegend:p===0,xaxis:"x"+(p+1),yaxis:"y"+(p+1),
   hovertemplate:name+"<br>epoch %{x}: %{y:.4f}<extra>"+PANELS[p][1]+"</extra>"})});
  if(r.best_epoch!=null){const e=r.epochs.find(e=>e.epoch===r.best_epoch);
   if(e)traces.push({x:[e.epoch],y:[e.val_miou],mode:"markers",marker:{symbol:"star",size:12,color},xaxis:"x4",yaxis:"y4",
    legendgroup:name,showlegend:false,hovertemplate:name+"<br>selected epoch %{x}: %{y:.4f}<extra></extra>"})}});
 const layout={grid:{rows:2,columns:2,pattern:"independent"},margin:{t:30,l:50,r:10,b:30},
  legend:{orientation:"h",y:-0.12,font:{size:10}},hovermode:"closest",
  annotations:PANELS.map(([,t],p)=>({text:t,xref:"x"+(p+1)+" domain",yref:"y"+(p+1)+" domain",x:0.5,y:1.08,showarrow:false}))};
 Plotly.react("plots",traces,layout,{responsive:true});}
document.getElementById("b_plot").onclick=()=>{matching().slice(0,40).forEach(r=>selected.add(r.i));render();plot()};
document.getElementById("b_clear").onclick=()=>{selected.clear();render();plot()};
render();plot();
</script></body></html>
"""


def write_html(runs: list[Run], path: Path, roots: list[Path]) -> None:
    """Write the viewer with plotly.js inlined, so it opens offline from any browser."""
    from plotly.offline import get_plotlyjs

    data = json.dumps([run.to_json() for run in runs], allow_nan=False).replace("</", "<\\/")
    actions = "\n".join(f"{k:16} {v}" for k, v in ACTIONS.items())
    html = (
        _TEMPLATE.replace("__PLOTLY__", get_plotlyjs())
        .replace("__GENERATED__", datetime.now(UTC).strftime("%Y-%m-%d %H:%MZ"))
        .replace("__ROOTS__", ", ".join(str(r) for r in roots))
        .replace("__NRUNS__", str(len(runs)))
        .replace("__ACTIONS__", actions)
        .replace("__DATA__", data)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
