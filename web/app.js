// Obsidian-style viewer. Node kinds, relations and roles all come from
// data/graph.js (written by src/build_subgraph.py); nothing about the
// biology is fixed here.

(function () {
  const data = window.GRAPH_DATA;
  const message = document.getElementById("message");

  function showMessage(html) {
    message.innerHTML = html;
    message.hidden = false;
  }

  if (!data) {
    showMessage("No graph data found. Run <code>python src/build_subgraph.py</code>, then reload this page.");
    return;
  }
  if (typeof ForceGraph === "undefined") {
    showMessage("Could not load the <code>force-graph</code> library. Check your internet connection and reload.");
    return;
  }

  const meta = data.meta;
  const roles = meta.roles;
  const predRel = meta.prediction_relation;
  const labelRels = new Set(meta.label_relations);

  // ---------- colors: evenly spaced hues, one per node kind in the data ----------
  const kinds = [...new Set(data.nodes.map((n) => n.kind))].sort();
  const kindColor = {};
  kinds.forEach((k, i) => {
    kindColor[k] = `hsl(${Math.round((360 * i) / kinds.length + 200) % 360}, 70%, 62%)`;
  });
  const css = getComputedStyle(document.documentElement);
  const GOLD = css.getPropertyValue("--gold").trim();
  const TEXT = css.getPropertyValue("--text").trim();

  // ---------- adjacency ----------
  const byId = new Map(data.nodes.map((n) => [n.id, n]));
  // Explanation paths name their edges by (source, target, relation) ids.
  const edgeKey = (source, target, rel) => JSON.stringify([source, target, rel]);
  const linkByKey = new Map();
  data.nodes.forEach((n) => {
    n.neighbors = new Set();
    n.links = [];
  });
  data.links.forEach((l) => {
    const a = byId.get(l.source);
    const b = byId.get(l.target);
    linkByKey.set(edgeKey(l.source, l.target, l.rel), l);
    a.neighbors.add(b);
    b.neighbors.add(a);
    a.links.push(l);
    b.links.push(l);
  });
  const symmetric = new Set((meta.symmetric_edges || []).map((e) => e.join("|")));
  const maxDegree = Math.max(...data.nodes.map((n) => n.neighbors.size));

  function radius(n) {
    const base = 2 + 10 * Math.sqrt(n.neighbors.size / maxDegree);
    return n.role === roles.seed ? base + 3 : base;
  }

  // ---------- state ----------
  const hiddenKinds = new Set();
  const highlightNodes = new Set();
  const highlightLinks = new Set();
  let hoverNode = null;
  let selectedNode = null;
  // Explanation path shown in the graph: { key, nodes: Set, links: Set }. While it
  // is active, hovering does not change the highlight.
  let activePath = null;

  function setHighlight(node) {
    highlightNodes.clear();
    highlightLinks.clear();
    if (node) {
      highlightNodes.add(node);
      node.neighbors.forEach((n) => highlightNodes.add(n));
      node.links.forEach((l) => highlightLinks.add(l));
    }
    refreshParticles();
  }

  // force-graph counts particles per link only when the accessor is set, so set it
  // again whenever the highlight changes.
  function refreshParticles() {
    graph.linkDirectionalParticles(graph.linkDirectionalParticles());
  }

  const isVisible = (n) => !hiddenKinds.has(n.kind);
  const endpoint = (x) => (typeof x === "object" ? x : byId.get(x));

  // ---------- graph ----------
  const graph = ForceGraph()(document.getElementById("graph"))
    .graphData({ nodes: data.nodes, links: data.links })
    .backgroundColor(css.getPropertyValue("--bg").trim())
    .nodeId("id")
    .nodeLabel(() => "")
    .autoPauseRedraw(false)
    .cooldownTicks(300)
    .nodeCanvasObject(drawNode)
    .nodePointerAreaPaint((n, color, ctx) => {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(n.x, n.y, radius(n) + 2, 0, 2 * Math.PI);
      ctx.fill();
    })
    .linkColor(linkColor)
    .linkWidth((l) => (highlightLinks.has(l) ? (activePath ? 2.6 : 1.6) : l.rel === predRel ? 1.2 : 0.4))
    .linkLineDash((l) => (l.rel === predRel ? [4, 3] : null))
    .linkDirectionalParticles((l) => {
      if (activePath) return highlightLinks.has(l) ? 3 : 0;
      return l.rel === predRel ? 3 : highlightLinks.has(l) ? 2 : 0;
    })
    .linkDirectionalParticleWidth((l) => (activePath && highlightLinks.has(l) ? 3 : 2))
    .linkDirectionalParticleColor((l) => (l.rel === predRel || activePath ? GOLD : TEXT))
    .onNodeHover((node) => {
      hoverNode = node;
      if (!activePath) setHighlight(node || selectedNode);
      document.body.style.cursor = node ? "pointer" : "";
    })
    .onNodeClick(selectNode)
    .onBackgroundClick(() => selectNode(null))
    // A click on a dimmed (near-invisible) link counts as a background click.
    .onLinkClick((l) => {
      if (!highlightLinks.has(l)) selectNode(null);
    })
    .onNodeDragEnd((n) => {
      n.fx = n.x;
      n.fy = n.y;
    });

  graph.d3Force("charge").strength(-45);
  graph.d3Force("link").distance(28);
  applyVisibility();

  let fitted = false;
  graph.onEngineStop(() => {
    if (!fitted) {
      graph.zoomToFit(600, 60);
      fitted = true;
    }
  });

  function drawNode(n, ctx, scale) {
    const r = radius(n);
    const color = kindColor[n.kind];
    const active = highlightNodes.has(n);
    const dimmed = highlightNodes.size > 0 && !active;

    ctx.globalAlpha = dimmed ? 0.12 : 1;
    if (!dimmed && (active || n.role === roles.seed)) {
      ctx.shadowColor = color;
      ctx.shadowBlur = 18;
    }
    ctx.beginPath();
    ctx.arc(n.x, n.y, r, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.shadowBlur = 0;

    if (n.role === roles.candidate || n.role === roles.predicted) {
      ctx.lineWidth = 2 / scale;
      ctx.strokeStyle = GOLD;
      ctx.stroke();
    }
    if (n === selectedNode) {
      ctx.lineWidth = 2.5 / scale;
      ctx.strokeStyle = TEXT;
      ctx.stroke();
    }

    const showLabel = n.role === roles.seed || active || scale > 2.5;
    if (showLabel && !dimmed) {
      const size = Math.max(11 / scale, 1.5);
      ctx.font = `${n.role === roles.seed ? "600 " : ""}${size}px Inter, Segoe UI, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillStyle = TEXT;
      ctx.fillText(n.name, n.x, n.y + r + 2 / scale);
    }
    ctx.globalAlpha = 1;
  }

  function linkColor(l) {
    if (highlightLinks.size) {
      if (!highlightLinks.has(l)) return "rgba(255,255,255,0.03)";
      return l.rel === predRel || activePath ? GOLD : "rgba(255,255,255,0.75)";
    }
    if (l.rel === predRel) return GOLD;
    if (labelRels.has(l.rel)) return "rgba(255,255,255,0.22)";
    return "rgba(255,255,255,0.09)";
  }

  function applyVisibility() {
    graph
      .nodeVisibility((n) => isVisible(n))
      .linkVisibility((l) => isVisible(endpoint(l.source)) && isVisible(endpoint(l.target)));
  }

  // ---------- selection + side panel ----------
  const panel = document.getElementById("panel");
  const panelBody = document.getElementById("panel-body");

  function selectNode(node) {
    activePath = null;
    markActiveItem();
    selectedNode = node;
    setHighlight(hoverNode || node);
    if (!node) {
      panel.hidden = true;
      return;
    }
    graph.centerAt(node.x, node.y, 700);
    graph.zoom(Math.max(graph.zoom(), 2.5), 700);
    renderPanel(node);
  }

  function el(tag, attrs = {}, ...children) {
    const e = document.createElement(tag);
    Object.entries(attrs).forEach(([k, v]) => {
      if (k === "style") Object.assign(e.style, v);
      else e[k] = v;
    });
    children.forEach((c) => e.append(c));
    return e;
  }

  function nodeItem(other, extra) {
    const li = el("li", {}, el("span", { className: "dot", style: { background: kindColor[other.kind] } }), other.name);
    if (extra) li.append(el("span", { className: "score" }, extra));
    li.onclick = () => selectNode(other);
    return li;
  }

  function renderPanel(n) {
    panelBody.replaceChildren();
    panelBody.append(
      el("h2", {}, n.name),
      el(
        "div",
        { className: "meta" },
        el("span", { className: "badge", style: { background: kindColor[n.kind] } }, n.kind),
        `${n.neighbors.size} connections`
      ),
      el("div", { className: "meta" }, n.id)
    );

    if (n.candidate_for) {
      n.candidate_for.forEach((c) => {
        const hint = el(
          "div",
          { className: "hint" },
          `Repurposing hint: binds ${c.shared_targets} ${meta.kinds.target.toLowerCase()}s associated with `,
          el("b", {}, c.disease),
          ` but is not a known treatment for it.`
        );
        hint.style.cursor = "pointer";
        hint.onclick = () => selectNode(byId.get(c.disease_id));
        panelBody.append(hint);
      });
    }

    renderExplanations(n);

    // Group neighbors by relation and direction, e.g. "binds →" or "← treats".
    const groups = new Map();
    n.links.forEach((l) => {
      const src = endpoint(l.source);
      const other = src === n ? endpoint(l.target) : src;
      if (!isVisible(other)) return;
      const key = src === n ? `${l.rel} →` : `← ${l.rel}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push({ other, link: l });
    });

    [...groups.entries()]
      .sort((a, b) => (a[0].includes(predRel) ? -1 : b[0].includes(predRel) ? 1 : b[1].length - a[1].length))
      .forEach(([key, items]) => {
        items.sort((a, b) => (b.link.score ?? 0) - (a.link.score ?? 0) || a.other.name.localeCompare(b.other.name));
        panelBody.append(
          el("h3", {}, `${key.replaceAll("_", " ")} (${items.length})`),
          el("ul", {}, ...items.map(({ other, link }) => nodeItem(other, link.score != null ? link.score.toFixed(3) : "")))
        );
      });
    panel.hidden = false;
  }

  // ---------- explanation paths ("Why predicted?") ----------
  const pct = (x) => `${Math.round(100 * x)}%`;
  const label = (s) => s.replaceAll("_", " ");

  function nodeChip(node, onclick) {
    const chip = el("span", { className: "chip" }, el("span", { className: "dot", style: { background: kindColor[node.kind] } }), node.name);
    if (onclick) {
      chip.classList.add("clickable");
      chip.onclick = (e) => {
        e.preventDefault();
        e.stopPropagation();
        onclick();
      };
    }
    return chip;
  }

  // Relation between path positions i and i+1, with an arrow in the stored direction.
  function relationText(edge, fromId) {
    const kindsKey = [byId.get(edge.source).kind, edge.rel, byId.get(edge.target).kind].join("|");
    if (symmetric.has(kindsKey)) return `—${label(edge.rel)}—`;
    return edge.source === fromId ? `—${label(edge.rel)}→` : `←${label(edge.rel)}—`;
  }

  function pathItem(link, path, key) {
    const ok = path.nodes.every((id) => byId.has(id)) && path.edges.every((e) => byId.has(e.source) && byId.has(e.target));
    const chain = el("div", { className: "chain" });
    if (ok) {
      path.nodes.forEach((id, i) => {
        if (i > 0) chain.append(el("span", { className: "rel" }, relationText(path.edges[i - 1], path.nodes[i - 1])));
        chain.append(nodeChip(byId.get(id)));
      });
    } else {
      chain.append("Path nodes are missing from the graph data.");
    }
    const item = el("div", { className: "path", title: "Show this path in the graph" }, chain);
    if (path.share != null) {
      item.append(el("div", { className: "share", title: "Share of this metapath's degree-weighted path count" }, `${pct(path.share)} of signal`));
    }
    item.dataset.key = key;
    if (activePath && activePath.key === key) item.classList.add("active");
    if (ok) item.onclick = () => togglePath(link, path, key);
    return item;
  }

  function renderExplanations(n) {
    const preds = n.links
      .filter((l) => l.rel === predRel && Array.isArray(l.explanations))
      .sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
    if (!preds.length) return;
    panelBody.append(el("h3", {}, `Why predicted? (${preds.length})`));

    preds.forEach((link, li) => {
      const src = endpoint(link.source);
      const tgt = endpoint(link.target);
      const summary = el("summary", {}, nodeChip(src, src === n ? null : () => selectNode(src)), " → ",
        nodeChip(tgt, tgt === n ? null : () => selectNode(tgt)));
      if (link.score != null) summary.append(el("span", { className: "score" }, link.score.toFixed(3)));
      const box = el("details", { className: "why", open: li === 0 }, summary);

      if (!link.explanations.length) {
        box.append(el("div", { className: "meta" }, `No supporting paths found between ${src.name} and ${tgt.name}.`));
      }
      // Group paths by metapath, in the order given (strongest contribution first).
      const groups = new Map();
      link.explanations.forEach((p, pi) => {
        if (!groups.has(p.metapath)) groups.set(p.metapath, []);
        groups.get(p.metapath).push([p, pi]);
      });
      groups.forEach((items, metapath) => {
        const contribution = items[0][0].contribution;
        const head = el("div", { className: "metapath" }, el("b", {}, metapath));
        if (meta.metapaths && meta.metapaths[metapath]) head.append(" ", el("span", { className: "desc" }, meta.metapaths[metapath]));
        if (contribution != null) {
          head.append(el("span", { className: "score", title: "Contribution to the model's log-odds" }, `+${contribution.toFixed(2)}`));
        }
        box.append(head, ...items.map(([p, pi]) => pathItem(link, p, JSON.stringify([src.id, tgt.id, pi]))));
      });
      if (box.querySelector(".path.active")) box.open = true;
      panelBody.append(box);
    });
  }

  function markActiveItem() {
    panelBody.querySelectorAll(".path").forEach((item) => {
      item.classList.toggle("active", !!activePath && item.dataset.key === activePath.key);
    });
  }

  function togglePath(link, path, key) {
    if (activePath && activePath.key === key) {
      clearPath();
      return;
    }
    const nodes = new Set(path.nodes.map((id) => byId.get(id)));
    const links = new Set(path.edges.map((e) => linkByKey.get(edgeKey(e.source, e.target, e.rel))).filter(Boolean));
    links.add(link);
    activePath = { key, nodes, links };
    highlightNodes.clear();
    highlightLinks.clear();
    nodes.forEach((x) => highlightNodes.add(x));
    links.forEach((x) => highlightLinks.add(x));
    refreshParticles();

    let unhid = false;
    nodes.forEach((x) => {
      if (hiddenKinds.delete(x.kind)) unhid = true;
    });
    if (unhid) {
      syncLegend();
      if (selectedNode) renderPanel(selectedNode);
    }
    markActiveItem();
    fitNodes(nodes);
  }

  function clearPath() {
    if (!activePath) return;
    activePath = null;
    setHighlight(hoverNode || selectedNode);
    markActiveItem();
  }

  // Zoom to the given nodes, keeping them clear of the side panel.
  function fitNodes(nodes, ms = 700, padding = 70) {
    const box = graph.getGraphBbox((x) => nodes.has(x));
    if (!box) return;
    const covered = panel.hidden ? 0 : panel.getBoundingClientRect().width + 14;
    const w = Math.max(window.innerWidth - covered - 2 * padding, 50);
    const h = Math.max(window.innerHeight - 2 * padding, 50);
    const k = Math.min(w / Math.max(box.x[1] - box.x[0], 1), h / Math.max(box.y[1] - box.y[0], 1), 8);
    const cx = (box.x[0] + box.x[1]) / 2 + covered / 2 / k;
    const cy = (box.y[0] + box.y[1]) / 2;
    graph.centerAt(cx, cy, ms);
    graph.zoom(k, ms);
  }

  document.getElementById("panel-close").onclick = () => selectNode(null);
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (activePath) clearPath();
    else selectNode(null);
  });

  // ---------- toolbar ----------
  const countBy = (key) => data.nodes.reduce((m, n) => ((m[n[key]] = (m[n[key]] || 0) + 1), m), {});
  const kindCounts = countBy("kind");
  const roleCounts = countBy("role");

  document.getElementById("stats").textContent =
    `${data.nodes.length.toLocaleString()} nodes · ${data.links.length.toLocaleString()} links · ` +
    `${meta.seed_diseases.length} seed ${meta.kinds.disease.toLowerCase()}s`;

  const datalist = document.getElementById("node-names");
  [...data.nodes]
    .sort((a, b) => a.name.localeCompare(b.name))
    .forEach((n) => datalist.append(el("option", { value: n.name, label: n.kind })));

  const search = document.getElementById("search");
  function runSearch() {
    const q = search.value.trim().toLowerCase();
    if (!q) return;
    const match =
      data.nodes.find((n) => n.name.toLowerCase() === q) || data.nodes.find((n) => n.name.toLowerCase().includes(q));
    if (match) {
      hiddenKinds.delete(match.kind);
      syncLegend();
      selectNode(match);
    }
  }
  search.addEventListener("change", runSearch);
  search.addEventListener("keydown", (e) => e.key === "Enter" && runSearch());

  document.getElementById("fit").onclick = () => graph.zoomToFit(600, 60);

  // ---------- legend ----------
  const legend = document.getElementById("legend");
  const checkboxes = {};

  legend.append(el("h2", {}, "Node types"));
  kinds.forEach((k) => {
    const box = el("input", { type: "checkbox", checked: true });
    box.onchange = () => {
      box.checked ? hiddenKinds.delete(k) : hiddenKinds.add(k);
      applyVisibility();
      if (selectedNode) renderPanel(selectedNode);
    };
    checkboxes[k] = box;
    legend.append(
      el("label", {}, box, el("span", { className: "dot", style: { background: kindColor[k] } }), k,
        el("span", { className: "count" }, kindCounts[k]))
    );
  });

  function syncLegend() {
    kinds.forEach((k) => (checkboxes[k].checked = !hiddenKinds.has(k)));
    applyVisibility();
  }

  legend.append(el("h2", {}, "Markers"));
  if (roleCounts[roles.candidate]) {
    legend.append(
      el("div", { className: "static" }, el("span", { className: "ring" }),
        `${meta.kinds.drug} candidate (heuristic)`, el("span", { className: "count" }, roleCounts[roles.candidate]))
    );
  }
  const predCount = data.links.filter((l) => l.rel === predRel).length;
  if (predCount) {
    legend.append(
      el("div", { className: "static" }, el("span", { className: "dash" }),
        predRel.replaceAll("_", " "), el("span", { className: "count" }, predCount))
    );
  }
  legend.append(el("div", { className: "static meta" }, "Bigger node = more connections"));

  window.addEventListener("resize", () => graph.width(window.innerWidth).height(window.innerHeight));
})();
