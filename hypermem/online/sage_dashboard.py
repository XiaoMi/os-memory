# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Self-contained, read-only dashboard for the HyperOS-Mem layered graph."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import argparse
import json
import html
import re
import sqlite3


TEMPLATE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>:root{--paper:#f5f5f5;--white:#fff;--ink:#2d3142;--muted:#4f5d75;--soft:#7a8399;--rule:rgba(45,49,66,.13);--accent:#eb6c36;--tint:rgba(235,108,54,.08);--link:#2e5aa8;--green:#56816b;--gold:#9b7628}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.55 Geist,"Microsoft YaHei",sans-serif}header{padding:24px 28px 18px;background:#fff;border-bottom:1px solid var(--rule)}.eyebrow{font:500 8px Geist Mono,monospace;letter-spacing:.18em;color:var(--accent)}h1{font:400 32px Instrument Serif,"Microsoft YaHei",serif;margin:4px 0}.sub{color:var(--muted)}.stats{padding:12px 28px;background:#fff;border-bottom:1px solid var(--rule);display:flex;gap:8px;flex-wrap:wrap}.chip{border:1px solid var(--rule);border-radius:4px;padding:4px 9px;background:var(--paper);font-size:12px}.chip b{color:var(--accent);font:600 15px Geist}.tabs{display:flex;padding:0 28px;background:#fff;border-bottom:1px solid var(--rule);overflow:auto}.tab{padding:12px 16px;border:0;border-bottom:3px solid transparent;background:none;color:var(--muted);white-space:nowrap;cursor:pointer}.tab.on{color:var(--ink);border-bottom-color:var(--accent);font-weight:600}.pane{display:none;padding:20px 28px}.pane.on{display:block}.desc{margin:0 0 14px;padding:12px 14px;border-left:3px solid var(--accent);background:var(--tint);color:var(--muted)}.toolbar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;position:sticky;top:0;background:rgba(245,245,245,.96);padding:8px 0;z-index:3}input,select{border:1px solid #bfc0c0;border-radius:4px;background:#fff;padding:8px 10px;color:var(--ink)}input{min-width:320px}.event-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:12px}.card{background:#fff;border:1px solid var(--rule);border-radius:6px;padding:14px}.card.pending{border-style:dashed}.card h3{margin:0 0 4px;font-size:15px}.meta,.mono{color:var(--soft);font:11px/1.5 Geist Mono,Consolas,monospace}.summary{margin:9px 0}.tags{display:flex;gap:5px;flex-wrap:wrap}.tag{padding:2px 6px;border-radius:3px;background:rgba(46,90,168,.08);color:var(--link);font-size:11px}.tag.topic{background:rgba(155,118,40,.10);color:var(--gold)}details{margin-top:10px;border-top:1px solid var(--rule);padding-top:8px}summary{cursor:pointer;color:var(--link)}.fact,.obs{margin:7px 0;padding-left:10px;border-left:2px solid rgba(86,129,107,.35)}.obs{border-color:rgba(46,90,168,.3)}table{border-collapse:collapse;width:100%;background:#fff;font-size:12px}th,td{border:1px solid var(--rule);padding:6px 8px;text-align:left;vertical-align:top}th{position:sticky;top:0;background:#ececec;z-index:1}.scroll{overflow:auto;max-height:72vh}.layers{display:grid;grid-template-columns:1.15fr 1fr .9fr;gap:12px}.layer h2{font-size:14px;margin:0 0 9px}.memory{border-left:3px solid var(--accent)}.count{font:12px Geist Mono;color:var(--soft)}#graph-wrap{background:#fff;border:1px solid var(--rule);border-radius:6px;overflow:auto}#graph{display:block;min-width:1040px;width:100%;height:680px}.node rect{fill:#fff;stroke:var(--ink)}.node text{font-family:Geist,"Microsoft YaHei"}.edge{fill:none;stroke:var(--muted);stroke-width:1}.edge.support{stroke:var(--accent)}.edge.anchor{stroke:var(--gold)}.relation{font:11px Geist Mono;color:var(--soft)}.badge{display:inline-block;padding:1px 6px;border-radius:3px;color:#fff;background:var(--muted);font-size:10px}.badge.continuation{background:var(--link)}.badge.part-of{background:var(--gold)}.badge.cause{background:#9b4c45}.badge.precondition{background:var(--green)}.empty{text-align:center;color:var(--soft);padding:30px}footer{margin:24px 28px;padding:12px 0;border-top:1px solid var(--rule);font:10px Geist Mono;color:var(--soft)}@media(max-width:850px){.layers{grid-template-columns:1fr}.pane{padding:14px}.tabs,.stats,header{padding-left:16px;padding-right:16px}input{min-width:180px;flex:1}}button{cursor:pointer;padding:6px 10px;background:#fff;border:1px solid var(--rule);border-radius:4px;color:var(--ink)}button:disabled{opacity:.45;cursor:default}.node{cursor:pointer}.node:focus rect{stroke:#eb6c36;stroke-width:2}.card{overflow-wrap:anywhere}#graph-event{max-width:100%}.layers>*{min-width:0}@media(max-width:450px){.event-grid{grid-template-columns:1fr}.toolbar input{min-width:0;max-width:100%}.tabs{max-width:100vw}} </style></head><body>
<header><div class=eyebrow>HYPER-MEM · DRED + SAGE · SNAPSHOT</div><h1>__TITLE__</h1><div class=sub>原子观测 → Event → Anchor / Fact → LongMemory；本页是可独立打开的当前构建快照。</div></header><div class=stats id=stats></div><nav class=tabs><button class="tab on" data-t=0>① Event</button><button class=tab data-t=1>② 原子观测</button><button class=tab data-t=2>③ 语义层</button><button class=tab data-t=3>④ 局部记忆图</button><button class=tab data-t=4>⑤ 关系边</button></nav>
<section class="pane on" data-p=0><div class=desc>每张卡就是一个 Event。可查看事件摘要、锚点、事实和原始观测；“待语义化”表示当前快照生成时该批次仍在构建。</div><div class=toolbar><input id=event-q placeholder="搜索标题、摘要、锚点、事实或观测"><select id=event-type><option value="">全部类型</option></select><select id=event-status><option value="">全部状态</option><option value=ready>已语义化</option><option value=pending>待语义化</option></select></div><div class=event-grid id=events></div></section>
<section class=pane data-p=1><div class=desc>保留的原子观测及其 Event 归属。每页显示 100 条，可翻页查看全部观测，或搜索应用、内容和所属事件。</div><div class=toolbar><input id=obs-q placeholder="搜索应用、内容或 Event"><span class=count id=obs-count></span></div><div class=scroll><table><thead><tr><th>时间</th><th>应用</th><th>观测内容</th><th>所属 Event</th></tr></thead><tbody id=obs-body></tbody></table></div></section>
<section class=pane data-p=2><div class=desc>SAGE 将 Event 转写为可检索的主题锚点与带证据 Fact，再从多个 Event 中巩固高置信 LongMemory。结构自洽不等同于人工语义验收。</div><div class=layers><div class=layer><h2>LongMemory <span class=count id=lm-count></span></h2><div id=longs></div></div><div class=layer><h2>Anchor <span class=count id=anchor-count></span></h2><div id=anchors></div></div><div class=layer><h2>Fact <span class=count id=fact-count></span></h2><div id=latest-facts></div></div></div></section>
<section class=pane data-p=3><div class=desc>选择一个 Event，按“一跳证据”展开其 Observation、Anchor、Fact 与 LongMemory。切换关系层可查看全部邻接节点，每页最多 8 个；点击节点查看原文与支撑。</div><div class=toolbar><select id=graph-event></select></div><div id=graph-wrap><svg id=graph viewBox="0 0 1040 680" role=img aria-labelledby="hm-graph-title hm-graph-desc"><title id=hm-graph-title>Hyper-Mem 局部层级记忆图</title><desc id=hm-graph-desc>展示一个事件与其原子观测、主题锚点、事实和长期记忆之间的可追溯关系。</desc><g id=graph-content></g></svg></div></section>
<section class=pane data-p=4><div class=desc>本页包含跨层关系和事件间关系：part-of 表示 Event → 主题 Anchor；continuation 为共享锚点与时间窗的规则推导。cause / precondition 当前未抽取。</div><div class=stats id=rel-stats style="padding:0 0 12px;background:none"></div><div class=scroll><table><thead><tr><th>from</th><th>to</th><th>关系</th><th>依据</th></tr></thead><tbody id=rel-body></tbody></table></div></section><footer>Hyper-Mem snapshot · generated from the local SQLite evidence store · no raw data is fetched from the network</footer>
<script>const G=__GRAPH__;
G.edges.forEach(e=>{if(e.type==='part_of')e.type='part-of'});
G.summary.enriched_events??=G.nodes.filter(n=>n.type==='event'&&n.enrichment_status!=='pending').length;
G.nodes.forEach(n=>{if(n.type==='event')n.enrichment_status??='ready';if(n.type==='long_memory')n.support_events??=G.edges.filter(e=>e.type==='supports'&&e.target===n.id).map(e=>e.source)});
const N=new Map(G.nodes.map(n=>[n.id,n]));const ins=new Map(),outs=new Map();for(const e of G.edges){if(!outs.has(e.source))outs.set(e.source,[]);outs.get(e.source).push(e);if(!ins.has(e.target))ins.set(e.target,[]);ins.get(e.target).push(e)}const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const trunc=(s,n=120)=>String(s??'').length>n?String(s).slice(0,n)+'…':String(s??'');const events=G.nodes.filter(n=>n.type==='event'),obs=G.nodes.filter(n=>n.type==='observation'),facts=G.nodes.filter(n=>n.type==='fact'),anchors=G.nodes.filter(n=>n.type==='anchor'),longs=G.nodes.filter(n=>n.type==='long_memory');
document.querySelector('#stats').innerHTML=[['原子观测',G.summary.observations],['Event',G.summary.events],['已语义化',G.summary.enriched_events],['Anchor',G.summary.anchors],['Fact',G.summary.facts],['LongMemory',G.summary.long_memories]].map(x=>`<span class=chip><b>${x[1]}</b> ${x[0]}</span>`).join('');document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tab,.pane').forEach(x=>x.classList.remove('on'));b.classList.add('on');document.querySelector(`.pane[data-p="${b.dataset.t}"]`).classList.add('on');if(b.dataset.t==='3')drawGraph()});function rel(id,type){return [...(ins.get(id)||[]),...(outs.get(id)||[])].filter(e=>e.type===type).map(e=>N.get(e.source===id?e.target:e.source)).filter(Boolean)}const types=[...new Set(events.map(e=>e.event_type))].sort();document.querySelector('#event-type').innerHTML+=types.map(t=>`<option>${esc(t)}</option>`).join('');
function renderEvents(){const q=document.querySelector('#event-q').value.trim().toLowerCase(),type=document.querySelector('#event-type').value,status=document.querySelector('#event-status').value;let out='';for(const e of [...events].sort((a,b)=>b.start_time.localeCompare(a.start_time))){const aa=[...rel(e.id,'anchored_by'),...rel(e.id,'part-of')],ff=rel(e.id,'fact_of'),oo=rel(e.id,'member_of');const hay=[e.label,e.summary,...aa.map(x=>x.label),...ff.map(x=>x.label),...oo.map(x=>x.label)].join(' ').toLowerCase();if((q&&!hay.includes(q))||(type&&e.event_type!==type)||(status&&e.enrichment_status!==status))continue;out+=`<article class="card ${e.enrichment_status==='pending'?'pending':''}"><h3>${esc(e.label)}</h3><div class=meta>${esc(e.start_time)} → ${esc(e.end_time)} · ${esc(e.event_type)} · ${e.member_count} 条观测 · ${e.enrichment_status==='ready'?'已语义化':'待语义化'}</div><p class=summary>${esc(e.summary)}</p><div class=tags>${aa.map(a=>`<span class="tag ${a.anchor_kind==='topic'?'topic':''}">${a.anchor_kind==='topic'?'主题':'实体'} · ${esc(a.label)}</span>`).join('')}</div><details><summary>${ff.length} 条 Fact · ${oo.length} 条 Observation</summary>${ff.map(f=>`<div class=fact>${esc(f.label)} <span class=mono>${f.support.length} 条支撑</span></div>`).join('')}${oo.map(o=>`<div class=obs><span class=mono>${esc(o.time)} · ${esc(o.app)}</span><br>${esc(o.label)}</div>`).join('')}</details></article>`}document.querySelector('#events').innerHTML=out||'<div class=empty>没有匹配的 Event</div>'}['event-q','event-type','event-status'].forEach(id=>document.querySelector('#'+id).addEventListener(id==='event-q'?'input':'change',renderEvents));
function renderObs(){const q=document.querySelector('#obs-q').value.trim().toLowerCase();const selected=obs.filter(o=>[o.label,o.app,o.owner].join(' ').toLowerCase().includes(q));document.querySelector('#obs-count').textContent=`匹配 ${selected.length} 条，显示前 ${Math.min(300,selected.length)} 条`;document.querySelector('#obs-body').innerHTML=selected.slice(0,300).map(o=>`<tr><td class=mono>${esc(o.time)}</td><td>${esc(o.app)}</td><td>${esc(o.label)}</td><td class=mono>${esc(o.owner)}</td></tr>`).join('')};document.querySelector('#obs-q').oninput=renderObs;document.querySelector('#lm-count').textContent=`${longs.length} 条`;document.querySelector('#longs').innerHTML=longs.slice(0,200).map(m=>`<div class="card memory"><h3>${esc(m.label)}</h3><div class=meta>${m.support_events.length} 个 Event 支撑 · revision ${m.revision}</div></div>`).join('')||'<div class=empty>暂无达到门槛的长期记忆</div>';document.querySelector('#anchor-count').textContent=`${anchors.length} 个`;document.querySelector('#anchors').innerHTML=[...anchors].sort((a,b)=>b.degree-a.degree).slice(0,120).map(a=>`<div class=card><h3>${esc(a.label)}</h3><div class=meta>${esc(a.anchor_kind)} · degree ${a.degree}</div></div>`).join('');document.querySelector('#fact-count').textContent=`${facts.length} 条`;document.querySelector('#latest-facts').innerHTML=facts.slice(-120).reverse().map(f=>`<div class=card><p>${esc(f.label)}</p><div class=meta>${f.support.length} 条观测支撑</div></div>`).join('');

const graphSelect=document.querySelector('#graph-event');
graphSelect.innerHTML=[...events].sort((a,b)=>b.member_count-a.member_count).map(e=>'<option value="'+esc(e.id)+'">'+esc(trunc(e.label,52))+' · '+e.member_count+'</option>').join('');
let graphPage=0;
graphSelect.insertAdjacentHTML('afterend','<select id="graph-kind" aria-label="关系层"><option value="member_of">原子观测</option><option value="fact_of">事实 Fact</option><option value="anchors">主题 / 实体 Anchor</option><option value="supports">长期记忆 LongMemory</option><option value="continuation">事件续接 continuation</option></select><button id="graph-prev">上一页</button><span id="graph-count" class="count"></span><button id="graph-next">下一页</button>');
const graphKind=document.querySelector('#graph-kind');
graphSelect.onchange=graphKind.onchange=()=>{graphPage=0;drawGraph()};
document.querySelector('#graph-prev').onclick=()=>{graphPage=Math.max(0,graphPage-1);drawGraph()};
document.querySelector('#graph-next').onclick=()=>{graphPage++;drawGraph()};
function node(x,y,w,h,n,subtitle,focus=false){
 const lines=Array.from(String(n.label||'')).reduce((a,c,i)=>{if(i%20===0)a.push('');a[a.length-1]+=c;return a},[]).slice(0,focus?7:3);
 const text=lines.map((s,i)=>'<text x="16" y="'+(32+i*20)+'" font-size="16">'+esc(s)+(i===2&&!focus&&String(n.label).length>60?'…':'')+'</text>').join('');
 return '<g class="node" data-node="'+esc(n.id)+'" role="button" tabindex="0" aria-label="'+esc(n.label)+'" transform="translate('+x+' '+y+')"><title>'+esc(n.label)+'</title><rect width="'+w+'" height="'+h+'" rx="6" style="fill:'+(focus?'#fff6ef':'#fff')+';stroke:'+(focus?'#eb6c36':'#bfc0c0')+'"/>'+text+'<text x="16" y="'+(h-16)+'" font-size="12" fill="#4f5d75">'+esc(subtitle)+'</text></g>';
}
function drawGraph(){
 const e=N.get(graphSelect.value||events[0]?.id);if(!e)return;
 const k=graphKind.value,all=k==='anchors'?[...rel(e.id,'anchored_by'),...rel(e.id,'part-of')]:rel(e.id,k);
 const max=Math.max(0,Math.ceil(all.length/8)-1);graphPage=Math.min(graphPage,max);
 document.querySelector('#graph-prev').disabled=graphPage===0;document.querySelector('#graph-next').disabled=graphPage===max;
 document.querySelector('#graph-count').textContent=all.length+' 个节点 · 第 '+(graphPage+1)+' / '+(max+1)+' 页';
 const svg=document.querySelector('#graph');svg.setAttribute('viewBox','0 0 1200 688');svg.style.minWidth='960px';
 let edges='<defs><marker id="hm-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto-start-reverse"><path d="M 0 0 L 8 4 L 0 8 Z" fill="#4f5d75"/></marker></defs>';
 let nodes=node(440,48,320,576,e,'EVENT · '+e.member_count+' 条观测',true);
 all.slice(graphPage*8,graphPage*8+8).forEach((n,i)=>{
   const left=i<4,row=i%4,x=left?16:864,y=64+144*row,cy=y+56;
   nodes+=node(x,y,320,112,n,n.type==='anchor'?(n.anchor_kind==='topic'?'主题锚点':'实体锚点'):n.type);
   const incoming=k==='member_of'||k==='fact_of';
   const actual=k==='continuation'?G.edges.find(r=>r.type===k&&((r.source===e.id&&r.target===n.id)||(r.source===n.id&&r.target===e.id))):null;
   const neighborIsSource=actual?actual.source===n.id:incoming;
   const nx=left?336:864,ex=left?440:760;
   edges+='<path class="edge" marker-end="url(#hm-arrow)" d="M '+(neighborIsSource?nx:ex)+' '+cy+' H '+(neighborIsSource?ex:nx)+'"/>';
   const label=k==='anchors'?(n.anchor_kind==='topic'?'part-of':'anchored_by'):k;
   edges+='<text x="'+(left?388:812)+'" y="'+(cy-12)+'" text-anchor="middle" font-size="8" fill="#4f5d75">'+label+'</text>';
 });
 if(!all.length)nodes+='<text x="16" y="96" font-size="16" fill="#4f5d75">本事件在此层暂无节点</text>';
 svg.querySelector('#graph-content').innerHTML=edges+nodes;
 svg.querySelectorAll('[data-node]').forEach(el=>{
  el.onclick=()=>showNode(el.dataset.node);
  el.onkeydown=ev=>{if(ev.key==='Enter'||ev.key===' '){ev.preventDefault();showNode(el.dataset.node)}};
 });
}
document.body.insertAdjacentHTML('beforeend','<dialog id="node-dialog" style="width:min(720px,90vw);max-height:85vh;border:1px solid #bfc0c0;border-radius:6px;padding:24px"><button id="node-close" style="float:right">关闭</button><div id="node-detail"></div></dialog>');
document.querySelector('#node-close').onclick=()=>document.querySelector('#node-dialog').close();
function showNode(id){
 const n=N.get(id);if(!n)return;
 let body='<h2>'+esc(n.label)+'</h2><p class="mono">'+esc(n.id)+' · '+esc(n.type)+'</p>';
 if(n.summary)body+='<p>'+esc(n.summary)+'</p>';
 const supported=n.support||n.support_events||[];
 body+=supported.map(oid=>'<p><button data-open-node="'+esc(oid)+'">查看支撑</button> '+esc(N.get(oid)?.label||oid)+'</p>').join('');
 if(n.type==='observation')body+='<p>'+esc(n.app)+' · '+esc(n.time)+'</p>';
 if(n.type==='event')body+='<p>'+n.member_count+' 条观测</p>';
 document.querySelector('#node-detail').innerHTML=body;
 document.querySelectorAll('[data-open-node]').forEach(b=>b.onclick=()=>showNode(b.dataset.openNode));
 const d=document.querySelector('#node-dialog');if(!d.open)d.showModal();
}

const relationTypes=['part-of','continuation','cause','precondition'];const relationEdges=G.edges.filter(e=>relationTypes.includes(e.type));document.querySelector('#rel-stats').innerHTML=relationTypes.map(t=>`<span class=chip><b>${relationEdges.filter(e=>e.type===t).length}</b> ${t}</span>`).join('');document.querySelector('#rel-body').innerHTML=relationEdges.map(e=>`<tr><td>${esc(N.get(e.source)?.label||e.source)}</td><td>${esc(N.get(e.target)?.label||e.target)}</td><td><span class="badge ${e.type}">${e.type}</span></td><td class=relation>${esc(e.evidence||(e.derived?'由共享锚点与时间窗确定':'显式关系'))}</td></tr>`).join('')||'<tr><td colspan=4 class=empty>当前快照没有事件关系边</td></tr>';
let obsPage=0;
document.querySelector('#obs-count').insertAdjacentHTML('afterend','<button id="obs-prev">上一页</button><button id="obs-next">下一页</button>');
renderObs=function(){
 const q=document.querySelector('#obs-q').value.trim().toLowerCase();
 const selected=obs.filter(o=>[o.label,o.app,o.owner,N.get(o.owner)?.label].join(' ').toLowerCase().includes(q));
 const max=Math.max(0,Math.ceil(selected.length/100)-1);obsPage=Math.min(obsPage,max);
 document.querySelector('#obs-count').textContent='共 '+selected.length+' 条 · 第 '+(obsPage+1)+' / '+(max+1)+' 页';
 document.querySelector('#obs-prev').disabled=obsPage===0;document.querySelector('#obs-next').disabled=obsPage===max;
 document.querySelector('#obs-body').innerHTML=selected.slice(obsPage*100,obsPage*100+100).map(o=>'<tr><td class="mono">'+esc(o.time)+'</td><td>'+esc(o.app)+'</td><td>'+esc(o.label)+'</td><td>'+esc(N.get(o.owner)?.label||o.owner)+'</td></tr>').join('');
};
document.querySelector('#obs-q').oninput=()=>{obsPage=0;renderObs()};
document.querySelector('#obs-prev').onclick=()=>{obsPage=Math.max(0,obsPage-1);renderObs()};
document.querySelector('#obs-next').onclick=()=>{obsPage++;renderObs()};
const layerPage={longs:0,anchors:0,'latest-facts':0};
const layerItems={longs,anchors:[...anchors].sort((a,b)=>b.degree-a.degree),'latest-facts':facts};
for(const id of Object.keys(layerItems)){
 const root=document.querySelector('#'+id);
 root.insertAdjacentHTML('beforebegin','<div class="toolbar"><input id="'+id+'-q" placeholder="搜索此层" style="min-width:0;width:100%"><button id="'+id+'-prev">上一页</button><span id="'+id+'-page" class="count"></span><button id="'+id+'-next">下一页</button></div>');
 const render=()=>{
  const q=document.querySelector('#'+id+'-q').value.toLowerCase(),all=layerItems[id].filter(n=>n.label.toLowerCase().includes(q));
  const max=Math.max(0,Math.ceil(all.length/30)-1);layerPage[id]=Math.min(layerPage[id],max);
  document.querySelector('#'+id+'-page').textContent=all.length+' 条 · '+(layerPage[id]+1)+'/'+(max+1);
  document.querySelector('#'+id+'-prev').disabled=layerPage[id]===0;document.querySelector('#'+id+'-next').disabled=layerPage[id]===max;
  root.innerHTML=all.slice(layerPage[id]*30,layerPage[id]*30+30).map(n=>'<article class="card"><h3>'+esc(n.label)+'</h3><p class="meta">'+(n.type==='anchor'?'连接 '+n.degree+' 个事件':n.type==='fact'?n.support.length+' 条观测支撑':n.support_events.length+' 个事件支撑')+'</p><button data-detail="'+esc(n.id)+'">查看详情与支撑</button></article>').join('')||'<p class="empty">没有匹配结果</p>';
  root.querySelectorAll('[data-detail]').forEach(b=>b.onclick=()=>showNode(b.dataset.detail));
 };
 document.querySelector('#'+id+'-q').oninput=()=>{layerPage[id]=0;render()};
 document.querySelector('#'+id+'-prev').onclick=()=>{layerPage[id]--;render()};
 document.querySelector('#'+id+'-next').onclick=()=>{layerPage[id]++;render()};
 render();
}
renderEvents();renderObs();drawGraph();</script></body></html>'''


def _records(conn, table, key_column):
    return {key: json.loads(value) for key, value in conn.execute(f"SELECT {key_column},record_json FROM {table}")}


def snapshot_graph(database):
    """Project a consistent read-only snapshot, even while SAGE is running."""
    conn = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        conn.execute("BEGIN")
        runtime = {eid: json.loads(raw) for eid, raw in conn.execute("SELECT event_id,runtime_json FROM on_events")}
        current = {eid for eid, value in runtime.items() if value.get("lifecycle") != "superseded"}
        base = {eid: (start, end) for eid, start, end in conn.execute("SELECT event_id,start_time,end_time FROM events")}
        wiki = _records(conn, "on_event_wiki", "event_id"); semantic = _records(conn, "sage_events", "event_id")
        grounding = {oid: json.loads(raw) for oid, raw in conn.execute("SELECT observation_id,grounding_json FROM on_vectors")}
        observations = {oid: (start, json.loads(raw)) for oid, start, raw in conn.execute("SELECT observation_id,start_time,observation_json FROM observations")}
        owners = {oid: eid for eid, oid in conn.execute("SELECT event_id,observation_id FROM event_observations") if eid in current}
        members = defaultdict(list)
        for oid, eid in owners.items(): members[eid].append(oid)
        nodes, edges = [], []
        for eid in sorted(current):
            sem, card = semantic.get(eid), wiki.get(eid, {}); start, end = base[eid]
            nodes.append({"id": eid, "type": "event", "label": (sem or card).get("title", eid), "summary": (sem or card).get("summary", "语义巩固仍在构建"), "event_type": sem.get("type", "pending") if sem else "pending", "enrichment_status": "ready" if sem else "pending", "start_time": start, "end_time": end, "member_count": len(members[eid])})
        for oid, eid in owners.items():
            start, raw = observations[oid]; ground = grounding.get(oid, {}); text = ground.get("text") or raw.get("content") or ""; app = ground.get("app") or raw.get("source") or "unknown"
            nodes.append({"id": oid, "type": "observation", "label": str(text), "time": start, "app": app, "owner": eid}); edges.append({"source": oid, "target": eid, "type": "member_of"})
        anchors = _records(conn, "sage_anchors", "anchor_id")
        links = [(eid, aid, json.loads(raw)) for eid, aid, raw in conn.execute("SELECT event_id,anchor_id,record_json FROM sage_event_anchors") if eid in current and aid in anchors]
        used = {aid for _, aid, _ in links}
        for aid in sorted(used):
            a = anchors[aid]; nodes.append({"id": aid, "type": "anchor", "label": a.get("key", aid), "anchor_kind": a.get("kind", "topic"), "degree": sum(x[1] == aid for x in links)})
        event_anchors, event_topics = defaultdict(set), defaultdict(set)
        for eid, aid, link in links:
            kind = link.get("kind", anchors[aid].get("kind", "topic")); event_anchors[eid].add(aid)
            if kind == "topic": event_topics[eid].add(aid)
            edges.append({"source": eid, "target": aid, "type": "part-of" if kind == "topic" else "anchored_by"})
        facts = _records(conn, "sage_facts", "fact_id")
        for fid, fact in facts.items():
            if fact.get("event_id") in current:
                nodes.append({"id": fid, "type": "fact", "label": fact.get("text", ""), "tags": fact.get("tags", []), "support": fact.get("support", [])}); edges.append({"source": fid, "target": fact["event_id"], "type": "fact_of"})
        memories = _records(conn, "sage_long_memory", "memory_id")
        for mid, memory in memories.items():
            support = [eid for eid in memory.get("support_events", []) if eid in current]
            if support:
                nodes.append({"id": mid, "type": "long_memory", "label": memory.get("text", ""), "confidence": memory.get("confidence"), "revision": memory.get("revision", 0), "support_events": support}); edges.extend({"source": eid, "target": mid, "type": "supports"} for eid in support)
        event_nodes = {n["id"]: n for n in nodes if n["type"] == "event"}; entity = {eid: {aid for aid in aids if anchors[aid].get("kind") == "entity"} for eid, aids in event_anchors.items()}
        ordered = sorted(current, key=lambda eid: (base[eid][0], eid)); from datetime import datetime
        for i, left in enumerate(ordered):
            left_end = datetime.fromisoformat(base[left][1])
            for right in ordered[i + 1:]:
                gap = (datetime.fromisoformat(base[right][0]) - left_end).total_seconds()
                if gap > 7 * 86400: break
                shared = entity.get(left, set()) & entity.get(right, set())
                if shared and (event_nodes[left]["event_type"] == event_nodes[right]["event_type"] or event_topics[left] & event_topics[right]):
                    edges.append({"source": left, "target": right, "type": "continuation", "derived": True, "evidence": "共享实体锚点 + 七日时间窗 + 同类型/主题"})
        counts = Counter(n["type"] for n in nodes)
        graph = {"schema": "hyperos-mem-layered-graph-snapshot-v1", "nodes": nodes, "edges": edges, "summary": {"observations": counts["observation"], "events": counts["event"], "enriched_events": len(semantic.keys() & current), "anchors": counts["anchor"], "facts": counts["fact"], "long_memories": counts["long_memory"], "relations": dict(Counter(e["type"] for e in edges)), "snapshot": True}}
        from hypermem.online.sage_tools import add_tool_parents
        return add_tool_parents(graph,conn)
    finally:
        conn.close()


def export_dashboard(graph, path):
    payload = json.dumps(graph, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    values={"__GRAPH__":payload,"__TITLE__":html.escape(graph.get("title","OS-Memory · 层级记忆图"))}
    Path(path).write_text(re.sub(r"__(?:GRAPH|TITLE)__",lambda match:values[match.group()],TEMPLATE), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Export a read-only Hyper-Mem dashboard snapshot"); parser.add_argument("--database", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    graph = snapshot_graph(args.database); args.output.parent.mkdir(parents=True, exist_ok=True); export_dashboard(graph, args.output); graph_path = args.output.with_suffix(".json"); graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"html": str(args.output), "graph": str(graph_path), "summary": graph["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
