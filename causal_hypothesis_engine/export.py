"""Export DAGVersion to various formats.

Supported formats:
  mermaid  — Mermaid flowchart syntax (renders in GitHub, Notion, etc.)
  dot      — Graphviz DOT format (render with: dot -Tsvg file.dot > file.svg)
  json     — Clean JSON (model_dump)
  html     — 3D interactive graph (opens in browser; needs network access —
             the two JS libraries are loaded from a pinned CDN)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models.dag_version import DAGVersion

# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

# node_type → Mermaid style class
_MERMAID_STYLES = {
    "Exposure":   "fill:#4f86c6,stroke:#2c5f8a,color:#fff",
    "Outcome":    "fill:#e05c5c,stroke:#a83232,color:#fff",
    "Confounder": "fill:#f0a500,stroke:#c47d00,color:#fff",
    "Mediator":   "fill:#5cb85c,stroke:#3a7a3a,color:#fff",
    "Collider":   "fill:#9b59b6,stroke:#6c3483,color:#fff",
}

# node_type → hex colour for DOT and 3D HTML
_NODE_COLOURS = {
    "Exposure":   "#4f86c6",
    "Outcome":    "#e05c5c",
    "Confounder": "#f0a500",
    "Mediator":   "#5cb85c",
    "Collider":   "#9b59b6",
}

_DEFAULT_COLOUR = "#aaaaaa"

# measurability_state → node size multiplier for 3D viewer
_SIZE_MAP = {
    "Hypothetical": 6,
    "Identified":   8,
    "Proxied":      10,
    "Validated":    13,
}


# ---------------------------------------------------------------------------
# Mermaid
# ---------------------------------------------------------------------------


def to_mermaid(version: "DAGVersion") -> str:
    """Return a Mermaid flowchart string for *version*."""
    # Build id-safe aliases (Mermaid node IDs must be alphanumeric)
    node_alias: dict[str, str] = {}
    for i, node in enumerate(version.nodes):
        node_alias[node.id] = f"n{i}"

    lines = ["flowchart TD"]

    # Style class definitions
    for ntype, style in _MERMAID_STYLES.items():
        lines.append(f"    classDef {ntype.lower()} {style}")
    lines.append("")

    # Node declarations
    for node in version.nodes:
        alias = node_alias[node.id]
        ntype = node.node_type.value if hasattr(node.node_type, "value") else str(node.node_type)
        mstate = (
            node.measurability_state.value
            if hasattr(node.measurability_state, "value")
            else str(node.measurability_state)
        )
        # Escape quotes in label
        label_safe = node.label.replace('"', "'")
        lines.append(f'    {alias}["{label_safe}\\n[{ntype} · {mstate}]"]')
        lines.append(f"    class {alias} {ntype.lower()}")

    lines.append("")

    # Edge declarations
    node_by_id = {n.id: n for n in version.nodes}
    for edge in version.edges:
        src = node_alias.get(edge.source_node_id)
        tgt = node_alias.get(edge.target_node_id)
        if src is None or tgt is None:
            continue
        if edge.label:
            label_safe = edge.label.replace('"', "'")
            lines.append(f"    {src} -->|{label_safe}| {tgt}")
        else:
            lines.append(f"    {src} --> {tgt}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DOT (Graphviz)
# ---------------------------------------------------------------------------


def to_dot(version: "DAGVersion") -> str:
    """Return a Graphviz DOT string for *version*."""
    lines = [
        "digraph causal_dag {",
        "    rankdir=LR;",
        '    node [fontname="Helvetica", fontsize=11, style=filled, shape=box, '
        'rounded=true, margin="0.3,0.15"];',
        '    edge [fontname="Helvetica", fontsize=9, color="#555555"];',
        "",
    ]

    for node in version.nodes:
        ntype = node.node_type.value if hasattr(node.node_type, "value") else str(node.node_type)
        mstate = (
            node.measurability_state.value
            if hasattr(node.measurability_state, "value")
            else str(node.measurability_state)
        )
        colour = _NODE_COLOURS.get(ntype, _DEFAULT_COLOUR)
        label_safe = node.label.replace('"', '\\"')
        desc = (
            node.description.replace('"', '\\"')[:60] + "..."
            if len(node.description) > 60
            else node.description.replace('"', '\\"')
        )
        tooltip = f"{ntype} · {mstate}"
        if desc:
            tooltip += f"\\n{desc}"
        lines.append(
            f'    "{node.id}" ['
            f'label="{label_safe}\\n[{ntype}]", '
            f'fillcolor="{colour}", '
            f'fontcolor="white", '
            f'tooltip="{tooltip}"'
            f"];"
        )

    lines.append("")

    for edge in version.edges:
        label_part = f' [label="{edge.label}"]' if edge.label else ""
        lines.append(f'    "{edge.source_node_id}" -> "{edge.target_node_id}"{label_part};')

    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def to_json(version: "DAGVersion") -> str:
    """Return a clean JSON string for *version*."""
    return json.dumps(version.model_dump(mode="json"), indent=2)


# ---------------------------------------------------------------------------
# 3D interactive HTML viewer
# ---------------------------------------------------------------------------


def to_html_3d(version: "DAGVersion") -> str:
    """Return an HTML page with a 3D force-directed graph.

    No pip dependencies, but *not* self-contained: 3d-force-graph and
    three-spritetext are fetched from unpkg at pinned versions, so the page is
    blank offline. Open in any browser.

    Visual encoding:
      Colour  → node_type  (blue=Exposure, red=Outcome, orange=Confounder,
                             green=Mediator, purple=Collider)
      Size    → measurability_state  (Hypothetical=6 … Validated=13)
      Label   → shown on hover and as floating text
    """
    nodes_data = []
    for node in version.nodes:
        ntype = node.node_type.value if hasattr(node.node_type, "value") else str(node.node_type)
        mstate = (
            node.measurability_state.value
            if hasattr(node.measurability_state, "value")
            else str(node.measurability_state)
        )
        nodes_data.append({
            "id": node.id,
            "label": node.label,
            "type": ntype,
            "measurability": mstate,
            "description": node.description,
            "color": _NODE_COLOURS.get(ntype, _DEFAULT_COLOUR),
            "val": _SIZE_MAP.get(mstate, 6),
        })

    links_data = []
    for edge in version.edges:
        links_data.append({
            "source": edge.source_node_id,
            "target": edge.target_node_id,
            "label": edge.label,
            "description": edge.description,
        })

    # json.dumps does not escape "<", ">" or "/", so a node label containing
    # "</script>" would terminate the script block and inject arbitrary HTML.
    # Labels come from LLM tool output, so they are attacker-influenceable via
    # prompt injection in whatever source material the agent read.
    graph_data = (
        json.dumps({"nodes": nodes_data, "links": links_data}, indent=2)
        .replace("</", "<\\/")
        .replace("<!--", "<\\!--")
    )

    network_name = version.version_id[:8]
    node_count = len(version.nodes)
    edge_count = len(version.edges)

    # Legend entries
    legend_items = "\n".join(
        f'<div class="legend-item">'
        f'<span class="legend-dot" style="background:{colour}"></span>'
        f'<span>{ntype}</span></div>'
        for ntype, colour in _NODE_COLOURS.items()
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DAG Viewer — {network_name}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #0d1117; color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; overflow: hidden; }}

    #header {{
      position: fixed; top: 0; left: 0; right: 0; z-index: 100;
      padding: 12px 20px;
      background: rgba(13,17,23,0.85);
      backdrop-filter: blur(8px);
      border-bottom: 1px solid #30363d;
      display: flex; align-items: center; gap: 16px;
    }}
    #header h1 {{ font-size: 14px; font-weight: 600; color: #e6edf3; }}
    #header .meta {{ font-size: 12px; color: #8b949e; }}

    #legend {{
      position: fixed; bottom: 20px; left: 20px; z-index: 100;
      background: rgba(13,17,23,0.85);
      backdrop-filter: blur(8px);
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 12px 16px;
    }}
    #legend h3 {{ font-size: 11px; font-weight: 600; color: #8b949e; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }}
    .legend-item {{ display: flex; align-items: center; gap: 8px; font-size: 12px; margin-bottom: 5px; }}
    .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}

    #tooltip {{
      position: fixed; z-index: 200;
      background: rgba(22,27,34,0.95);
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 10px 14px;
      font-size: 12px;
      max-width: 260px;
      pointer-events: none;
      display: none;
    }}
    #tooltip .tt-label {{ font-weight: 600; font-size: 13px; margin-bottom: 4px; }}
    #tooltip .tt-type {{ color: #8b949e; margin-bottom: 2px; }}
    #tooltip .tt-desc {{ color: #c9d1d9; margin-top: 6px; line-height: 1.4; }}

    #controls {{
      position: fixed; top: 56px; right: 20px; z-index: 100;
      display: flex; flex-direction: column; gap: 8px;
    }}
    #controls button {{
      background: rgba(22,27,34,0.9);
      border: 1px solid #30363d;
      border-radius: 6px;
      color: #e6edf3;
      font-size: 12px;
      padding: 6px 12px;
      cursor: pointer;
      backdrop-filter: blur(8px);
    }}
    #controls button:hover {{ background: rgba(48,54,61,0.9); }}

    #graph {{ width: 100vw; height: 100vh; }}
  </style>
</head>
<body>

<div id="header">
  <h1>DAG Viewer</h1>
  <span class="meta">version {network_name} &nbsp;·&nbsp; {node_count} nodes &nbsp;·&nbsp; {edge_count} edges</span>
</div>

<div id="legend">
  <h3>Node Type</h3>
  {legend_items}
  <div style="margin-top:10px; border-top:1px solid #30363d; padding-top:8px;">
    <h3 style="margin-bottom:6px;">Size = Measurability</h3>
    <div class="legend-item"><span style="font-size:10px">● small</span><span>Hypothetical</span></div>
    <div class="legend-item"><span style="font-size:12px">● large</span><span>Validated</span></div>
  </div>
</div>

<div id="controls">
  <button onclick="graph.zoomToFit(400)">Fit to screen</button>
  <button onclick="toggleLabels()">Toggle labels</button>
</div>

<div id="tooltip"></div>
<div id="graph"></div>

<!-- Both libraries load BEFORE the script that uses them. three-spritetext was
     previously loaded after this block, so SpriteText was undefined when
     nodeThreeObject first ran. Versions are pinned: an unpinned CDN tag is a
     live third-party dependency executing against your research data. -->
<script src="https://unpkg.com/3d-force-graph@1.73.4"></script>
<script src="https://unpkg.com/three-spritetext@1.8.2"></script>
<script>
const graphData = {graph_data};

let showLabels = true;

const graph = ForceGraph3D()(document.getElementById('graph'))
  .backgroundColor('#0d1117')
  .graphData(graphData)
  .nodeId('id')
  .nodeLabel(node => '')
  .nodeColor(node => node.color)
  .nodeVal(node => node.val)
  .nodeResolution(16)
  .nodeThreeObjectExtend(true)
  .nodeThreeObject(node => {{
    if (!showLabels) return null;
    const sprite = new SpriteText(node.label);
    sprite.color = '#e6edf3';
    sprite.textHeight = 3.5;
    sprite.backgroundColor = 'rgba(13,17,23,0.75)';
    sprite.padding = 2;
    sprite.borderRadius = 3;
    sprite.position.y = node.val + 6;
    return sprite;
  }})
  .linkColor(() => '#58a6ff')
  .linkOpacity(0.6)
  .linkWidth(1.5)
  .linkDirectionalArrowLength(6)
  .linkDirectionalArrowRelPos(1)
  .linkDirectionalParticles(1)
  .linkDirectionalParticleSpeed(0.004)
  .linkDirectionalParticleWidth(2)
  .onNodeHover(node => {{
    const tt = document.getElementById('tooltip');
    if (node) {{
      tt.style.display = 'block';
      // textContent, not innerHTML: labels and descriptions are model-authored
      // and must never be parsed as HTML.
      tt.replaceChildren();
      const labelEl = document.createElement('div');
      labelEl.className = 'tt-label';
      labelEl.style.color = node.color;
      labelEl.textContent = node.label;
      tt.appendChild(labelEl);
      const typeEl = document.createElement('div');
      typeEl.className = 'tt-type';
      typeEl.textContent = node.type + ' · ' + node.measurability;
      tt.appendChild(typeEl);
      if (node.description) {{
        const descEl = document.createElement('div');
        descEl.className = 'tt-desc';
        descEl.textContent = node.description;
        tt.appendChild(descEl);
      }}
    }} else {{
      tt.style.display = 'none';
    }}
  }})
  .onNodeClick(node => {{
    graph.cameraPosition(
      {{ x: node.x * 1.5, y: node.y * 1.5, z: node.z * 1.5 }},
      {{ x: node.x, y: node.y, z: node.z }},
      800
    );
  }});

document.addEventListener('mousemove', e => {{
  const tt = document.getElementById('tooltip');
  tt.style.left = (e.clientX + 16) + 'px';
  tt.style.top  = (e.clientY - 10) + 'px';
}});

function toggleLabels() {{
  showLabels = !showLabels;
  graph.nodeThreeObject(node => {{
    if (!showLabels) return null;
    const sprite = new SpriteText(node.label);
    sprite.color = '#e6edf3';
    sprite.textHeight = 3.5;
    sprite.backgroundColor = 'rgba(13,17,23,0.75)';
    sprite.padding = 2;
    sprite.borderRadius = 3;
    sprite.position.y = node.val + 6;
    return sprite;
  }});
}}

</script>
</body>
</html>"""
