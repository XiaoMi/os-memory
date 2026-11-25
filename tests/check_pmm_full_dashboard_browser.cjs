// Full-user browser acceptance check. Requires the completed graph next to HTML.
// Use HYPERMEM_BROWSER_PROFILE_ROOT and XDG_CACHE_HOME on D: for profiles/cache;
// leave TMPDIR on a native filesystem for Chromium's small IPC files.
const fs = require('node:fs');
const path = require('node:path');
const transport = fs.readFileSync(path.join(__dirname, 'check_dashboard_browser.cjs'), 'utf8');
const boundary = transport.indexOf('(async()=>{try{');
if (boundary < 0) throw Error('Browser transport boundary changed');

async function verifyFullDashboard() {
  try {
    fs.mkdirSync(out, {recursive: true});
    const expected = JSON.parse(fs.readFileSync(path.join(path.dirname(html), 'hypermem_dashboard.json'), 'utf8'));
    assert.ok(expected.user_id);
    const expectedEvents = expected.nodes.filter(n => n.type === 'event');
    const expectedObservations = expected.nodes.filter(n => n.type === 'observation');
    assert.ok(expectedEvents.length && expectedObservations.length);
    assert.equal(expectedObservations.length, expected.summary.observations);
    const target = await send('Target.createTarget', {url: 'about:blank'}, null);
    session = (await send('Target.attachToTarget', {targetId: target.targetId, flatten: true}, null)).sessionId;
    await send('Page.enable'); await send('Runtime.enable'); await send('Network.enable');
    await send('Emulation.setDeviceMetricsOverride', {width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false});
    await send('Page.navigate', {url: pathToFileURL(html).href});
    for (let i = 0; i < 100; i++) {
      if (await evaluate("document.querySelectorAll('#events article').length>0")) break;
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    assert.equal(await evaluate('G.user_id'), expected.user_id);
    assert.equal(await evaluate('G.nodes.length'), expected.nodes.length);
    assert.equal(await evaluate('G.edges.length'), expected.edges.length);
    const identity = graph => ({ids: graph.nodes.map(n => n.id).sort(),
      ownership: graph.edges.filter(e => e.type === 'member_of').map(e => [e.source, e.target]).sort()});
    const hash = require('node:crypto').createHash('sha256').update(JSON.stringify(identity(expected))).digest('hex');
    const browserHash = await evaluate(`(async()=>{
      const value=(${identity.toString()})(G);
      const bytes=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(JSON.stringify(value)));
      return Array.from(new Uint8Array(bytes),b=>b.toString(16).padStart(2,'0')).join('');
    })()`);
    assert.equal(browserHash, hash);
    assert.equal(await evaluate("document.querySelectorAll('#events article').length"), expectedEvents.length);
    await evaluate("document.querySelector('#events summary').click()");
    assert.equal(await evaluate("document.querySelector('#events details').open"), true);
    await screenshot('full_events.png');
    await evaluate("document.querySelector('#event-status').value='pending';renderEvents()");
    assert.equal(await evaluate("document.querySelectorAll('#events article').length"), 0);
    await evaluate("document.querySelector('#event-status').value='';document.querySelector('#event-q').value='NO_MATCH_FULL_USER_937149';renderEvents()");
    assert.equal(await evaluate("document.querySelectorAll('#events article').length"), 0);
    await evaluate("document.querySelector('#event-q').value='';renderEvents()");
    checks.push('full-user identity, every node and ownership edge, ready Event cards, evidence expansion and filters');

    await evaluate("document.querySelector('[data-t=\"1\"]').click()");
    assert.equal(await evaluate("document.querySelectorAll('#obs-body tr').length"), Math.min(100, expectedObservations.length));
    await evaluate('obsPage=Number.MAX_SAFE_INTEGER;renderObs()');
    assert.equal(await evaluate('obsPage'), Math.ceil(expectedObservations.length / 100) - 1);
    assert.equal(await evaluate("document.querySelectorAll('#obs-body tr').length"), ((expectedObservations.length - 1) % 100) + 1);
    assert.equal(await evaluate("document.querySelector('#obs-next').disabled"), true);
    checks.push('observation pagination reaches the actual final page');

    await evaluate("document.querySelector('[data-t=\"2\"]').click()");
    for (const [id, type] of [['longs','long_memory'], ['anchors','anchor'], ['latest-facts','fact']]) {
      const total = expected.nodes.filter(n => n.type === type).length;
      assert.equal(await evaluate(`document.querySelectorAll('#${id} article').length`), Math.min(30, total));
      if (total) {
        await evaluate(`document.querySelector('#${id} [data-detail]').click()`);
        assert.equal(await evaluate("document.querySelector('#node-dialog').open"), true);
        await evaluate("document.querySelector('#node-close').click()");
      }
    }
    await screenshot('full_semantic_layers.png');
    checks.push('semantic layer counts and detail dialogs, including legitimately empty layers');

    await evaluate("document.querySelector('[data-t=\"3\"]').click()");
    for (const kind of ['member_of', 'fact_of', 'anchors', 'supports', 'continuation']) {
      await evaluate(`graphKind.value=${JSON.stringify(kind)};graphPage=0;drawGraph()`);
      const count = await evaluate("document.querySelectorAll('#graph .node').length");
      assert.ok(count >= 1 && count <= 9);
    }
    await evaluate("graphKind.value='member_of';graphPage=0;drawGraph()");
    await screenshot('full_local_graph.png');
    await evaluate("document.querySelector('#graph .node').dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}))");
    assert.equal(await evaluate("document.querySelector('#node-dialog').open"), true);
    await evaluate("document.querySelector('#node-close').click()");
    checks.push('bounded evidence graph across every layer and keyboard details');
    await send('Emulation.setDeviceMetricsOverride', {width: 390, height: 844, deviceScaleFactor: 1, mobile: true});
    await evaluate("document.querySelector('[data-t=\"0\"]').click()");
    assert.equal(await evaluate('document.documentElement.scrollWidth<=innerWidth'), true);
    await screenshot('full_mobile.png');
    assert.deepEqual(errors, []); assert.deepEqual(remote, []);
    checks.push('mobile width, no script exceptions, no external requests');
    const result = {status: 'passed', html, user_id: expected.user_id, observations: expectedObservations.length,
      events: expectedEvents.length, graph_identity_sha256: hash, checks, errors, remote};
    fs.writeFileSync(path.join(out, 'browser_check.json'), JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result));
  } finally {
    if (child.exitCode === null) {
      try { await send('Browser.close', {}, null); } catch {}
      child.kill();
    }
  }
}

new Function('require', 'process', transport.slice(0, boundary) + '(' + verifyFullDashboard.toString() +
  ')().catch(e=>{console.error(e);process.exitCode=1});')(require, process);
