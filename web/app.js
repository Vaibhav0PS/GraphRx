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
  data.nodes.forEach((n) => {
    n.neighbors = new Set();
    n.links = [];
  });
  data.links.forEach((l) => {
    const a = byId.get(l.source);
    const b = byId.get(l.target);
    a.neighbors.add(b);
    b.neighbors.add(a);
    a.links.push(l);
    b.links.push(l);
  });
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

  function setHighlight(node) {
    highlightNodes.clear();
    highlightLinks.clear();
    if (!node) return;
    highlightNodes.add(node);
    node.neighbors.forEach((n) => highlightNodes.add(n));
    node.links.forEach((l) => highlightLinks.add(l));
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
    .linkWidth((l) => (highlightLinks.has(l) ? 1.6 : l.rel === predRel ? 1.2 : 0.4))
    .linkLineDash((l) => (l.rel === predRel ? [4, 3] : null))
    .linkDirectionalParticles((l) => (l.rel === predRel ? 3 : highlightLinks.has(l) ? 2 : 0))
    .linkDirectionalParticleWidth(2)
    .linkDirectionalParticleColor((l) => (l.rel === predRel ? GOLD : TEXT))
    .onNodeHover((node) => {
      hoverNode = node;
      setHighlight(node || selectedNode);
      document.body.style.cursor = node ? "pointer" : "";
    })
    .onNodeClick(selectNode)
    .onBackgroundClick(() => selectNode(null))
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
      return l.rel === predRel ? GOLD : "rgba(255,255,255,0.75)";
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

  document.getElementById("panel-close").onclick = () => selectNode(null);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") selectNode(null);
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
