# Copyright (C) 2026 Xiaomi Corporation.
#
# OS-Memory is an open source project by Xiaomi Corporation,
# supported by Beijing Tongming Lake Information Technology Application
# Innovation Center (TLAIC).
#
# This product is licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License in the LICENSE file or at
# http://www.apache.org/licenses/LICENSE-2.0

"""Emit a local browser probe for platforms without the DevTools pipe transport."""
import sys
from pathlib import Path

PROBE = r'''
try {
 const check=(ok,label)=>{if(!ok)throw Error(label)};
 const tab=i=>document.querySelector('.tab[data-t="'+i+'"]').click();
 check(document.querySelectorAll('#events article').length===G.summary.events,'event coverage');
 document.querySelector('#events summary').click();
 check(document.querySelector('#events details').open,'event evidence expansion');
 document.querySelector('#event-status').value='pending';renderEvents();
 check(document.querySelectorAll('#events article').length===G.summary.events-G.summary.enriched_events,'pending events');
 document.querySelector('#event-status').value='';renderEvents();
 tab(1);
 check(document.querySelectorAll('#obs-body tr').length===100,'observation page size');
 document.querySelector('#obs-next').click();check(obsPage===1,'observation next page');
 obsPage=999;renderObs();check(document.querySelector('#obs-next').disabled,'observation final page');
 tab(2);
 document.querySelector('#longs [data-detail]').click();
 check(document.querySelector('#node-dialog').open,'LongMemory details');
 check(document.querySelectorAll('#node-detail [data-open-node]').length>0,'LongMemory evidence');
 document.querySelector('#node-close').click();
 document.querySelector('#anchors-next').click();check(layerPage.anchors===1,'anchor pagination');
 tab(4);check(document.querySelectorAll('#rel-body tr').length===relationEdges.length,'relationship coverage');
 tab(3);
 check(document.querySelectorAll('#graph .node').length===9,'graph budget');
 const boxes=Array.from(document.querySelectorAll('#graph .node')).map(n=>n.getBoundingClientRect());
 for(let i=0;i<boxes.length;i++)for(let j=i+1;j<boxes.length;j++){
  const a=boxes[i],b=boxes[j];check(a.right<=b.left||b.right<=a.left||a.bottom<=b.top||b.bottom<=a.top,'graph overlap');
 }
 document.querySelector('#graph-next').click();check(graphPage===1,'graph pagination');
 document.querySelector('#graph .node').dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));
 check(document.querySelector('#node-dialog').open,'keyboard graph details');document.querySelector('#node-close').click();
 for(const k of ['fact_of','anchors','supports','continuation']){graphKind.value=k;graphPage=0;drawGraph();check(document.querySelectorAll('#graph .node').length<=9,'graph layer budget')}
 graphKind.value='member_of';graphPage=0;drawGraph();
 document.body.dataset.browserCheck='passed';
} catch (e) { document.body.dataset.browserCheck='failed';document.body.dataset.browserError=e.message; }
'''

source, target = map(Path, sys.argv[1:3])
html = source.read_text(encoding='utf-8')
target.write_text(html.replace('</script>', PROBE + '\n</script>'), encoding='utf-8')
