// Reuse the repository's local Chromium DevTools transport.
const fs = require('node:fs');
const path = require('node:path');
const transport = fs.readFileSync(path.join(__dirname, 'check_dashboard_browser.cjs'), 'utf8');
const boundary = transport.indexOf('(async()=>{try{');
if (boundary < 0) throw new Error('Browser transport boundary changed');
const task = `(async()=>{try{
  fs.mkdirSync(out,{recursive:true});
  const target=await send('Target.createTarget',{url:'about:blank'},null);
  session=(await send('Target.attachToTarget',{targetId:target.targetId,flatten:true},null)).sessionId;
  await send('Page.enable');await send('Runtime.enable');await send('Network.enable');
  await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await send('Page.navigate',{url:pathToFileURL(html).href});
  for(let i=0;i<100;i++){if(await evaluate("document.querySelectorAll('#events article').length>0"))break;await new Promise(r=>setTimeout(r,100));}
  assert.equal(await evaluate("document.querySelectorAll('#events article').length"),await evaluate('G.summary.events'));
  await evaluate("document.querySelector('#events summary').click()");
  assert.equal(await evaluate("document.querySelector('#events details').open"),true);
  await screenshot('events.png');
  await evaluate("document.querySelector('#event-status').value='pending';renderEvents()");
  assert.equal(await evaluate("document.querySelectorAll('#events article').length"),await evaluate('G.summary.events-G.summary.enriched_events'));
  await evaluate("document.querySelector('#event-q').value='NO_MATCH_X78';renderEvents()");
  assert.equal(await evaluate("document.querySelectorAll('#events article').length"),0);
  await evaluate("document.querySelector('#event-q').value='';document.querySelector('#event-status').value='';renderEvents()");
  checks.push('all Event cards, evidence expansion, pending and search filters');
  await evaluate("document.querySelector('[data-t=\\\"1\\\"]').click()");
  assert.equal(await evaluate("document.querySelectorAll('#obs-body tr').length"),100);
  const first=await evaluate("document.querySelector('#obs-body').textContent");
  await evaluate("document.querySelector('#obs-next').click()");
  assert.notEqual(await evaluate("document.querySelector('#obs-body').textContent"),first);
  await evaluate("obsPage=999;renderObs()");
  assert.equal(await evaluate("document.querySelector('#obs-next').disabled"),true);
  checks.push('observation pagination includes final page');
  await evaluate("document.querySelector('[data-t=\\\"2\\\"]').click()");
  await screenshot('semantic_layers.png');
  await evaluate("document.querySelector('#longs [data-detail]').click()");
  assert.equal(await evaluate("document.querySelector('#node-dialog').open"),true);
  assert.equal(await evaluate("document.querySelectorAll('#node-detail [data-open-node]').length>0"),true);
  await evaluate("document.querySelector('#node-close').click()");
  await evaluate("document.querySelector('#anchors-next').click()");
  assert.equal(await evaluate('layerPage.anchors'),1);
  checks.push('semantic pagination and LongMemory support drilldown');
  await evaluate("document.querySelector('[data-t=\\\"3\\\"]').click()");
  assert.equal(await evaluate("document.querySelectorAll('#graph .node').length"),9);
  const bounds=await evaluate("Array.from(document.querySelectorAll('#graph .node')).map(n=>{const r=n.getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height}})");
  for(let i=0;i<bounds.length;i++)for(let j=i+1;j<bounds.length;j++){
    const a=bounds[i],b=bounds[j];assert.ok(a.x+a.w<=b.x||b.x+b.w<=a.x||a.y+a.h<=b.y||b.y+b.h<=a.y,'graph nodes overlap');
  }
  await screenshot('local_graph.png');
  await evaluate("document.querySelector('#graph-next').click()");
  assert.equal(await evaluate('graphPage'),1);
  await evaluate("document.querySelector('#graph .node').dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}))");
  assert.equal(await evaluate("document.querySelector('#node-dialog').open"),true);
  await evaluate("document.querySelector('#node-close').click()");
  for(const k of ['fact_of','anchors','supports','continuation']){
    await evaluate("graphKind.value="+JSON.stringify(k)+";graphPage=0;drawGraph()");
    assert.ok(await evaluate("document.querySelectorAll('#graph .node').length<=9"));
  }
  checks.push('non-overlapping graph, bounded pages, all layers and keyboard detail');
  await evaluate("document.querySelector('[data-t=\\\"4\\\"]').click()");
  assert.equal(await evaluate("document.querySelectorAll('#rel-body tr').length"),await evaluate('relationEdges.length'));
  await send('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  await evaluate("document.querySelector('[data-t=\\\"0\\\"]').click()");
  assert.equal(await evaluate('document.documentElement.scrollWidth<=innerWidth'),true);
  await screenshot('mobile.png');
  assert.deepEqual(errors,[]);assert.deepEqual(remote,[]);
  checks.push('relations, mobile layout, no JavaScript errors or network dependencies');
  fs.writeFileSync(path.join(out,'browser_check.json'),JSON.stringify({status:'passed',html,checks},null,2));
  console.log(JSON.stringify({status:'passed',checks},null,2));
}finally{if(child.exitCode===null){try{await send('Browser.close',{},null)}catch{}child.kill()}}})().catch(e=>{console.error(e);process.exitCode=1});`;
new Function('require','process',transport.slice(0,boundary)+task)(require,process);
