"""Interactive multi-version DAG viewer.

Replaces the single-version ``to_html_3d`` page in :mod:`export`, which had
three defects found by driving it in a real browser:

1. ``three-spritetext`` needs a global ``THREE``; ``3d-force-graph`` bundles
   its own copy and does not expose it, so ``SpriteText`` was undefined and no
   node labels rendered at all — five coloured spheres with nothing to tell
   them apart.
2. Loading a second ``three`` to satisfy that produced two instances with
   different colour-management defaults (r152 changed sRGB handling), which
   visibly corrupted node colours — and colour is the primary encoding of node
   type, so it broke the thing it exists to communicate.
3. A free force layout scatters a causal chain into a blob, hiding the one
   property that matters most in a causal graph: the direction of flow.

This module fixes all three. Labels are plain HTML elements positioned each
frame from ``graph2ScreenCoords``, so there is exactly one ``three`` on the
page and text is crisp at any zoom. The default layout is layered
(``dagMode``), so causes sit upstream of effects and the graph reads left to
right. Every version of a network is embedded in one page with a selector, so
switching hypotheses does not mean regenerating a file.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .models.dag_version import DAGVersion

# Node type -> colour. Chosen to stay distinguishable under the common forms
# of colour-blindness: blue/orange/red carry different luminance as well as
# different hue, so they remain separable in greyscale.
NODE_COLOURS = {
    "Exposure": "#4f9ff0",
    "Outcome": "#f2545b",
    "Confounder": "#f0a500",
    "Mediator": "#3ecf8e",
    "Collider": "#b57bee",
}
DEFAULT_COLOUR = "#8b949e"

# Measurability -> node radius. A second, redundant channel so two nodes of
# the same type are still distinguishable.
SIZE_MAP = {
    "Hypothetical": 3.0,
    "Identified": 5.0,
    "Proxied": 7.5,
    "Validated": 10.0,
}

_JS_CDN = (
    '<script src="https://unpkg.com/3d-force-graph@1.73.4"></script>'
)


def _version_payload(version: "DAGVersion") -> dict:
    """Serialise one version into the shape the page consumes."""
    nodes = []
    for node in version.nodes:
        node_type = getattr(node.node_type, "value", str(node.node_type))
        state = getattr(
            node.measurability_state, "value", str(node.measurability_state)
        )
        proxies = list((node.adapter_metadata or {}).get("proxy_variables", []))
        nodes.append(
            {
                "id": node.id,
                "label": node.label,
                "type": node_type,
                "state": state,
                "description": node.description or "",
                "proxies": proxies,
                "color": NODE_COLOURS.get(node_type, DEFAULT_COLOUR),
                "val": SIZE_MAP.get(state, 4.0),
            }
        )

    known = {n.id for n in version.nodes}
    links = [
        {
            "source": edge.source_node_id,
            "target": edge.target_node_id,
            "label": edge.label or "",
            "description": edge.description or "",
        }
        for edge in version.edges
        if edge.source_node_id in known and edge.target_node_id in known
    ]

    result = version.backtest_result
    contributions = {}
    if result is not None:
        for item in getattr(result, "contributions", []) or []:
            contributions[item.node_id] = {
                "value": item.contribution,
                "low": item.ci_low,
                "high": item.ci_high,
                "verdict": item.verdict,
            }

    return {
        "versionId": version.version_id,
        "shortId": version.version_id[:8],
        "status": getattr(version.status, "value", str(version.status)),
        "createdAt": str(version.created_at),
        "parentId": version.parent_version_id,
        "rationale": version.modification_rationale or "",
        "nodes": nodes,
        "links": links,
        "contributions": contributions,
        "warnings": version.structural_warnings(),
    }


def to_html(
    versions: Sequence["DAGVersion"],
    network_name: str = "",
    open_index: int = 0,
) -> str:
    """Render one self-navigating HTML page containing every version given.

    Versions are embedded in the order supplied — chronological, so the v1/v2
    labels match real lineage — and *open_index* selects which one is shown
    first. The selector switches between them without a reload, so a network's
    history is one artefact rather than N files.
    """
    if not versions:
        raise ValueError(
            "Problem: No versions to render.\n"
            "  Cause: to_html was called with an empty sequence.\n"
            "  Fix: Pass at least one DAGVersion."
        )

    payload = [_version_payload(v) for v in versions]
    # json.dumps does not escape "<", so a label containing "</script>" would
    # break out of the script block. Labels are model-authored, so this is a
    # real injection surface, not a hypothetical one.
    data_json = (
        json.dumps({"network": network_name, "versions": payload})
        .replace("</", "<\\/")
        .replace("<!--", "<\\!--")
    )

    legend_rows = "\n".join(
        f'<div class="legend-row"><span class="swatch" style="background:{colour}">'
        f"</span>{name}</div>"
        for name, colour in NODE_COLOURS.items()
    )
    size_rows = "\n".join(
        f'<div class="legend-row"><span class="dot" style="width:{4 + i * 3}px;'
        f'height:{4 + i * 3}px"></span>{state}</div>'
        for i, state in enumerate(SIZE_MAP)
    )

    open_at = max(0, min(int(open_index), len(payload) - 1))
    return (
        _TEMPLATE.replace("%%CDN%%", _JS_CDN)
        .replace("%%OPEN%%", str(open_at))
        .replace("%%DATA%%", data_json)
        .replace("%%LEGEND%%", legend_rows)
        .replace("%%SIZES%%", size_rows)
        .replace("%%TITLE%%", network_name or "Causal hypothesis network")
    )


_TEMPLATE = r"""<meta charset="utf-8">
<title>%%TITLE%% — causal-engine</title>
<style>
  :root {
    --bg: #0d1117; --panel: rgba(22,27,34,.94); --border: #30363d;
    --text: #e6edf3; --dim: #8b949e; --accent: #4f9ff0;
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:13px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         overflow:hidden; }
  #graph { position:fixed; inset:0; }

  /* Labels are HTML, not sprites: crisp at any zoom, styleable, and no second
     three.js on the page. Positioned each frame from graph2ScreenCoords. */
  #labels { position:fixed; inset:0; pointer-events:none; z-index:2; }
  .lbl { position:absolute; transform:translate(-50%,-50%);
         white-space:nowrap; font-weight:600; font-size:12px;
         text-shadow:0 1px 3px #000,0 0 8px #000; transition:opacity .12s; }

  .bar { position:fixed; top:0; left:0; right:0; height:46px; z-index:5;
         display:flex; align-items:center; gap:14px; padding:0 14px;
         background:var(--panel); border-bottom:1px solid var(--border);
         backdrop-filter:blur(8px); }
  .bar h1 { font-size:14px; margin:0; font-weight:600; }
  .meta { color:var(--dim); font-size:12px; }
  select, button, input {
    background:#21262d; color:var(--text); border:1px solid var(--border);
    border-radius:6px; padding:5px 9px; font-size:12px; font-family:inherit; }
  button { cursor:pointer; }
  button:hover, select:hover { border-color:var(--accent); }
  button.on { background:var(--accent); border-color:var(--accent); color:#05121f; }
  .spacer { flex:1; }

  .panel { position:fixed; background:var(--panel); border:1px solid var(--border);
           border-radius:10px; padding:12px 14px; z-index:5; backdrop-filter:blur(8px); }
  #legend { left:14px; bottom:14px; width:186px; }
  #legend h2 { font-size:10px; letter-spacing:.09em; text-transform:uppercase;
               color:var(--dim); margin:0 0 7px; font-weight:600; }
  .legend-row { display:flex; align-items:center; gap:8px; margin:3px 0; font-size:12px; }
  .swatch { width:11px; height:11px; border-radius:3px; flex:none; }
  .dot { background:var(--dim); border-radius:50%; flex:none; }
  hr { border:0; border-top:1px solid var(--border); margin:10px 0; }

  #inspector { right:14px; top:60px; width:290px; max-height:calc(100vh - 150px);
               overflow:auto; display:none; }
  #inspector h2 { margin:0 0 3px; font-size:15px; }
  .chip { display:inline-block; padding:2px 8px; border-radius:999px;
          font-size:11px; font-weight:600; margin:3px 4px 3px 0; }
  .kv { margin:9px 0 0; font-size:12px; }
  .kv .k { color:var(--dim); text-transform:uppercase; font-size:10px;
           letter-spacing:.07em; }
  code { background:#161b22; padding:1px 5px; border-radius:4px; font-size:11px; }
  .verdict-positive { color:#3ecf8e; } .verdict-negative { color:#f2545b; }
  .verdict-inconclusive { color:#f0a500; }
  #hint { position:fixed; bottom:14px; left:50%; transform:translateX(-50%);
          color:var(--dim); font-size:11px; z-index:4; }
  /* 3d-force-graph injects its own nav blurb bottom-centre, which collides
     with ours. */
  .scene-nav-info { display:none !important; }
  #warn { position:fixed; top:56px; left:14px; max-width:380px; z-index:4;
          color:#f0a500; font-size:11px; }
</style>

<div class="bar">
  <h1>%%TITLE%%</h1>
  <select id="versionSel" title="Switch version"></select>
  <span class="meta" id="counts"></span>
  <span class="spacer"></span>
  <input id="search" placeholder="Search nodes…" style="width:150px">
  <button id="btnLayout" class="on" title="Layered shows causal order (L)">Layered</button>
  <button id="btnLabels" class="on" title="Toggle labels (T)">Labels</button>
  <button id="btnFit" title="Fit to view (F)">Fit</button>
</div>

<div id="graph"></div>
<div id="labels"></div>
<div id="warn"></div>

<div class="panel" id="legend">
  <h2>Node type</h2>
  %%LEGEND%%
  <hr>
  <h2>Size = measurability</h2>
  %%SIZES%%
</div>

<div class="panel" id="inspector"></div>
<div id="hint">drag to rotate · scroll to zoom · right-drag to pan · click a node to inspect</div>

%%CDN%%
<script>
const DATA = %%DATA%%;
const el = id => document.getElementById(id);

let current = 0, showLabels = true, layered = true, selected = null, query = '';

// ---- version selector -----------------------------------------------------
const sel = el('versionSel');
DATA.versions.forEach((v, i) => {
  const o = document.createElement('option');
  o.value = i;
  o.textContent = `v${i + 1} · ${v.shortId} · ${v.status}`;
  sel.appendChild(o);
});
sel.onchange = () => load(+sel.value);

const graph = ForceGraph3D()(el('graph'))
  .backgroundColor('#0d1117')
  .nodeRelSize(4)
  .nodeVal(n => n.val)
  .nodeColor(n => nodeColour(n))
  .nodeOpacity(0.95)
  .linkColor(l => linkColour(l))
  .linkWidth(l => isAdjacent(l) ? 1.6 : 0.6)
  .linkOpacity(0.55)
  // Arrows sized generously: on a causal graph the direction IS the claim,
  // and the previous 3px arrowheads were unreadable at default zoom.
  .linkDirectionalArrowLength(5.5)
  .linkDirectionalArrowRelPos(0.92)
  .linkDirectionalArrowColor(l => linkColour(l))
  .linkLabel(l => l.label ? `<b>${esc(l.label)}</b>` : '')
  .onNodeHover(onHover)
  .onNodeClick(onClick)
  .onBackgroundClick(() => { selected = null; render(); })
  .cooldownTicks(120);

// A layered layout is the default because a free force layout renders a causal
// chain as a blob. 'lr' puts causes left of effects, which is how these graphs
// are read and drawn on paper.
function applyLayout() {
  graph.dagMode(layered ? 'lr' : null).dagLevelDistance(110);
  // Nodes sharing a layer are pinned to the same X under 'lr', so they only
  // separate if repulsion is strong enough. At -140 the two root Confounders
  // overlapped their own labels.
  graph.d3Force('charge').strength(layered ? -320 : -190);
  const link = graph.d3Force('link');
  if (link) link.distance(layered ? 60 : 45);
}

// ---- highlight state ------------------------------------------------------
let hovered = null, neighbours = new Set(), incident = new Set();

function recomputeNeighbours() {
  neighbours = new Set(); incident = new Set();
  const focus = hovered || selected;
  if (!focus) return;
  neighbours.add(focus.id);
  for (const l of cur().links) {
    const s = idOf(l.source), t = idOf(l.target);
    if (s === focus.id) { neighbours.add(t); incident.add(l); }
    if (t === focus.id) { neighbours.add(s); incident.add(l); }
  }
}
const idOf = x => (typeof x === 'object' ? x.id : x);
const isAdjacent = l => incident.has(l);

function matchesQuery(n) {
  if (!query) return true;
  return n.label.toLowerCase().includes(query)
      || n.type.toLowerCase().includes(query);
}

// Dimming rather than hiding: context is preserved, which is what makes
// Obsidian's hover behaviour legible instead of disorienting.
function nodeColour(n) {
  const faded = '#2b3138';
  if (query && !matchesQuery(n)) return faded;
  const focus = hovered || selected;
  if (focus && !neighbours.has(n.id)) return faded;
  return n.color;
}
function linkColour(l) {
  const focus = hovered || selected;
  if (focus) return incident.has(l) ? '#e6edf3' : '#20262d';
  return '#5b6673';
}

function onHover(n) { hovered = n; recomputeNeighbours(); render();
  document.body.style.cursor = n ? 'pointer' : 'default'; }
function onClick(n) { selected = n; recomputeNeighbours(); render();
  graph.cameraPosition(offsetFrom(n), n, 700); }
function offsetFrom(n) {
  const d = 110, r = Math.hypot(n.x, n.y, n.z) || 1;
  return { x: n.x * (1 + d / r), y: n.y * (1 + d / r), z: n.z * (1 + d / r) };
}

function render() { graph.nodeColor(graph.nodeColor()).linkColor(graph.linkColor());
  drawInspector(); }

// ---- HTML labels ----------------------------------------------------------
// Positioned per frame from graph2ScreenCoords. This is why there is no
// three-spritetext and no second three.js: the previous approach crashed
// outright, and the workaround corrupted every node colour.
const layer = el('labels');
let labelEls = new Map();

function rebuildLabels() {
  layer.innerHTML = ''; labelEls = new Map();
  for (const n of cur().nodes) {
    const d = document.createElement('div');
    d.className = 'lbl'; d.textContent = n.label;
    layer.appendChild(d); labelEls.set(n.id, d);
  }
}

function positionLabels() {
  if (!showLabels) return;
  const cam = graph.camera();
  for (const n of cur().nodes) {
    const d = labelEls.get(n.id);
    if (!d || n.x === undefined) continue;
    const p = graph.graph2ScreenCoords(n.x, n.y, n.z);
    const dist = Math.hypot(cam.position.x - n.x, cam.position.y - n.y,
                            cam.position.z - n.z);
    // Fade distant labels instead of letting them pile up — the same trick
    // Obsidian uses to keep a dense graph readable.
    let op = dist > 900 ? 0 : dist > 500 ? (900 - dist) / 400 : 1;
    const focus = hovered || selected;
    if (focus && !neighbours.has(n.id)) op *= 0.22;
    if (query && !matchesQuery(n)) op *= 0.15;
    // Clear the sphere by its true screen-space radius. Offsetting along
    // world X is wrong: under the 'lr' layout that axis is the layer axis and
    // is foreshortened, which threw labels far above their nodes. Offsetting
    // along the camera's up vector projects to a genuine vertical distance.
    const up = cam.up;
    const r = n.val * 4;
    const q = graph.graph2ScreenCoords(n.x + up.x * r, n.y + up.y * r,
                                       n.z + up.z * r);
    const px = Math.min(60, Math.max(9, Math.hypot(q.x - p.x, q.y - p.y)));
    d.style.left = p.x + 'px';
    d.style.top = (p.y - px - 8) + 'px';
    d.style.opacity = op;
    d.style.color = (focus && neighbours.has(n.id)) ? '#fff' : '#c9d1d9';
  }
}

// ---- inspector ------------------------------------------------------------
function drawInspector() {
  const box = el('inspector'), n = selected;
  if (!n) { box.style.display = 'none'; return; }
  const c = cur().contributions[n.id];
  const parents = cur().links.filter(l => idOf(l.target) === n.id);
  const children = cur().links.filter(l => idOf(l.source) === n.id);
  const list = ls => ls.length
    ? ls.map(l => `<div>${esc(nameOf(l, ls === parents ? 'source' : 'target'))}`
        + (l.label ? ` <span class="meta">(${esc(l.label)})</span>` : '') + '</div>').join('')
    : '<div class="meta">none</div>';

  box.innerHTML = `
    <h2>${esc(n.label)}</h2>
    <span class="chip" style="background:${n.color};color:#05121f">${esc(n.type)}</span>
    <span class="chip" style="background:#21262d;color:var(--dim)">${esc(n.state)}</span>
    ${n.description ? `<div class="kv">${esc(n.description)}</div>` : ''}
    ${n.proxies.length ? `<div class="kv"><div class="k">Bound columns</div>
      ${n.proxies.map(p => `<code>${esc(p)}</code>`).join(' ')}</div>` : ''}
    ${c ? `<div class="kv"><div class="k">Backtest contribution</div>
      <span class="verdict-${c.verdict}">${c.value >= 0 ? '+' : ''}${c.value.toFixed(4)}
      — ${c.verdict}</span>
      ${c.low !== null ? `<div class="meta">95% CI ${c.low.toFixed(4)} … ${c.high.toFixed(4)}</div>` : ''}
      </div>` : ''}
    <div class="kv"><div class="k">Caused by (${parents.length})</div>${list(parents)}</div>
    <div class="kv"><div class="k">Causes (${children.length})</div>${list(children)}</div>`;
  box.style.display = 'block';
}
function nameOf(l, which) {
  const id = idOf(l[which]);
  const n = cur().nodes.find(x => x.id === id);
  return n ? n.label : id.slice(0, 8);
}
function esc(s) { const d = document.createElement('div'); d.textContent = s;
  return d.innerHTML; }

// ---- load a version -------------------------------------------------------
const cur = () => graphData;
let graphData = { nodes: [], links: [] };

function load(i) {
  current = i; selected = null; hovered = null;
  const v = DATA.versions[i];
  // Deep copy: 3d-force-graph mutates nodes with x/y/z, and we switch back
  // and forth between versions.
  graphData = JSON.parse(JSON.stringify({ nodes: v.nodes, links: v.links }));
  graphData.contributions = v.contributions;
  graph.graphData(graphData);
  applyLayout();
  rebuildLabels();
  recomputeNeighbours();
  el('counts').textContent =
    `${v.nodes.length} nodes · ${v.links.length} edges · ${v.status}`
    + (v.parentId ? ` · child of ${v.parentId.slice(0, 8)}` : '');
  el('warn').innerHTML = (v.warnings || []).map(w => '⚠ ' + esc(w)).join('<br>');
  sel.value = i;
  drawInspector();
  setTimeout(() => graph.zoomToFit(600, 90), 350);
}

// ---- controls -------------------------------------------------------------
el('btnFit').onclick = () => graph.zoomToFit(600, 90);
el('btnLabels').onclick = e => {
  showLabels = !showLabels; e.target.classList.toggle('on', showLabels);
  layer.style.display = showLabels ? 'block' : 'none';
};
el('btnLayout').onclick = e => {
  layered = !layered; e.target.classList.toggle('on', layered);
  e.target.textContent = layered ? 'Layered' : 'Free 3D';
  applyLayout(); graph.d3ReheatSimulation();
  setTimeout(() => graph.zoomToFit(600, 90), 700);
};
el('search').oninput = e => { query = e.target.value.trim().toLowerCase(); render(); };

addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'f') graph.zoomToFit(600, 90);
  if (e.key === 't') el('btnLabels').click();
  if (e.key === 'l') el('btnLayout').click();
  if (e.key === 'Escape') { selected = null; el('search').value = ''; query = ''; render(); }
  if (e.key === 'ArrowRight' && current < DATA.versions.length - 1) load(current + 1);
  if (e.key === 'ArrowLeft' && current > 0) load(current - 1);
});
addEventListener('resize', () => graph.width(innerWidth).height(innerHeight));

(function tick() { positionLabels(); requestAnimationFrame(tick); })();
load(%%OPEN%%);
</script>
"""
