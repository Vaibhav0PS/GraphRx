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

  // ---------- colors: one hue per node kind; yellow-orange is kept free for predictions ----------
  const css = getComputedStyle(document.documentElement);
  const GOLD = css.getPropertyValue("--gold").trim();
  const TEXT = css.getPropertyValue("--text").trim();
  const BG = css.getPropertyValue("--bg").trim();
  const GOLD_SOFT = css.getPropertyValue("--gold-soft").trim(); // predicted links in the overview
  const kinds = [...new Set(data.nodes.map((n) => n.kind))].sort();
  // The kinds that play the main roles (from the data's meta) get the most distinct
  // hues; any other kinds take the remaining ones in alphabetical order.
  const ROLE_HUES = { drug: 205, disease: 275, target: 335, context: 160 };
  const SPARE_HUES = [95, 5, 240, 185, 305, 135, 225, 355];
  const hueOf = {};
  Object.entries(ROLE_HUES).forEach(([role, hue]) => {
    const kind = meta.kinds[role];
    if (kind && kinds.includes(kind)) hueOf[kind] = hue;
  });
  kinds.filter((k) => !(k in hueOf)).forEach((k, i) => {
    // Past the spare list, spread evenly over the hues outside the gold band (25-80).
    hueOf[k] = i < SPARE_HUES.length ? SPARE_HUES[i] : (80 + (i * 137.5) % 305) % 360;
  });
  const kindColor = Object.fromEntries(kinds.map((k) => [k, `hsl(${hueOf[k]}, 68%, 64%)`]));

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
  const isCandidate = (n) => Array.isArray(n.candidate_for) && n.candidate_for.length > 0;
  const endpoint = (x) => (typeof x === "object" ? x : byId.get(x));
  const predLinks = data.links.filter((l) => l.rel === predRel);
  // Drugs with at least one model prediction (the source end of predicted links).
  const predictedDrugs = new Set(predLinks.map((l) => byId.get(l.source)));

  // ---------- graph ----------
  const graph = ForceGraph()(document.getElementById("graph"))
    .graphData({ nodes: data.nodes, links: data.links })
    .backgroundColor(BG)
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

  // Seed diseases are hubs that otherwise pile up in the middle with their labels on
  // top of each other; a much stronger repulsion keeps them apart.
  graph.d3Force("charge").strength((n) => (n.role === roles.seed ? -900 : -40));
  graph.d3Force("link").distance((l) => (l.rel === predRel ? 60 : 28));
  applyVisibility();

  let fitted = false;
  graph.onEngineStop(() => {
    if (!fitted) {
      fitAll();
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

    // Model predictions get a solid gold ring; heuristic candidates a dashed one. A drug
    // can be a known treatment for one disease and a candidate for another, so the
    // candidate marker follows candidate_for, not the node's single role.
    if (predictedDrugs.has(n) || isCandidate(n)) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, r + 1.2 / scale, 0, 2 * Math.PI);
      ctx.lineWidth = 2 / scale;
      ctx.strokeStyle = GOLD;
      ctx.setLineDash(predictedDrugs.has(n) ? [] : [2.5 / scale, 2 / scale]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    if (n === selectedNode) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, r + 3 / scale, 0, 2 * Math.PI);
      ctx.lineWidth = 2 / scale;
      ctx.strokeStyle = TEXT;
      ctx.stroke();
    }

    const showLabel = n.role === roles.seed || active || scale > 2.5;
    if (showLabel && !dimmed) {
      const seed = n.role === roles.seed;
      const size = Math.max((seed ? 13 : 11) / scale, 1.5);
      ctx.font = `${seed ? "600 " : ""}${size}px Inter, Segoe UI, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      const y = n.y + r + 3 / scale;
      // Dark outline so labels stay readable on top of edges and other nodes.
      ctx.lineJoin = "round";
      ctx.lineWidth = 3.5 / scale;
      ctx.strokeStyle = BG;
      ctx.strokeText(n.name, n.x, y);
      ctx.fillStyle = TEXT;
      ctx.fillText(n.name, n.x, y);
    }
    ctx.globalAlpha = 1;
  }

  function linkColor(l) {
    if (highlightLinks.size) {
      if (!highlightLinks.has(l)) return "rgba(255,255,255,0.03)";
      return l.rel === predRel || activePath ? GOLD : "rgba(255,255,255,0.75)";
    }
    if (l.rel === predRel) return GOLD_SOFT;
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

  // node = null goes back to the overview (top predictions) in the panel.
  function selectNode(node) {
    activePath = null;
    markActiveItem();
    selectedNode = node;
    setHighlight(hoverNode || node);
    if (!node) {
      renderOverview();
      return;
    }
    renderPanel(node);
    // Fit the node and all its visible neighbors in the part of the view not under the panel.
    fitNodes(new Set([node, ...[...node.neighbors].filter(isVisible)]));
  }

  function closePanel() {
    selectNode(null);
    panel.hidden = true;
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

  // ---------- overview: what the panel shows when nothing is selected ----------
  function renderOverview() {
    panelBody.replaceChildren();
    const drugWord = meta.kinds.drug.toLowerCase();
    const diseaseWord = meta.kinds.disease.toLowerCase();
    panelBody.append(el("h2", {}, "Top predictions"));
    if (!predLinks.length) {
      panelBody.append(el("div", { className: "meta" },
        `No model predictions in this graph. Run the training scripts, then rebuild it. ` +
        `Click any node to explore its connections.`));
      panel.hidden = false;
      return;
    }
    panelBody.append(el("div", { className: "meta intro" },
      `New ${drugWord} → ${diseaseWord} links the model predicts, grouped by ${diseaseWord}. ` +
      `Click a ${drugWord} to see why it was predicted.`));

    // Diseases ordered by their best prediction; drugs inside by score.
    const byDisease = new Map();
    predLinks.forEach((l) => {
      const d = endpoint(l.target);
      if (!byDisease.has(d)) byDisease.set(d, []);
      byDisease.get(d).push(l);
    });
    const best = (ls) => Math.max(...ls.map((l) => l.score ?? 0));
    [...byDisease.entries()]
      .sort((a, b) => best(b[1]) - best(a[1]) || a[0].name.localeCompare(b[0].name))
      .forEach(([disease, ls]) => {
        const head = el("h3", { className: "group" }, nodeChip(disease, () => selectNode(disease)));
        const items = ls
          .sort((a, b) => (b.score ?? 0) - (a.score ?? 0))
          .map((l) => nodeItem(endpoint(l.source), l.score != null ? l.score.toFixed(3) : ""));
        panelBody.append(head, el("ul", {}, ...items));
      });
    panel.hidden = false;
  }

  // ---------- explanation paths ("Why predicted?") ----------
  const pct = (x) => (x > 0 && x < 0.01 ? "<1%" : `${Math.round(100 * x)}%`);
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
      // Each relation is kept on one line with the node it leads to, so a line break
      // never separates "—causes→" from what is caused.
      path.nodes.forEach((id, i) => {
        if (i === 0) {
          chain.append(nodeChip(byId.get(id)));
          return;
        }
        chain.append(el("span", { className: "step" },
          el("span", { className: "rel" }, relationText(path.edges[i - 1], path.nodes[i - 1])), nodeChip(byId.get(id))));
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
        // Readable pattern first; the metapath code is kept as a small tag for the report.
        const desc = (meta.metapaths && meta.metapaths[metapath]) || metapath;
        const head = el("div", { className: "metapath" }, el("span", { className: "desc" }, desc),
          el("span", { className: "code", title: "Metapath code" }, metapath));
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

  function fitAll(ms = 700) {
    fitNodes(new Set(data.nodes.filter(isVisible)), ms, 40);
  }

  // Zoom to the given nodes, keeping them clear of the side panel on the right and
  // the toolbar/legend column on the left.
  function fitNodes(nodes, ms = 700, padding = 60) {
    const box = graph.getGraphBbox((x) => nodes.has(x));
    if (!box) return;
    const right = panel.hidden ? 0 : window.innerWidth - panel.getBoundingClientRect().left;
    const left = Math.max(...["toolbar", "legend"].map((id) => document.getElementById(id).getBoundingClientRect().right));
    const w = Math.max(window.innerWidth - left - right - 2 * padding, 50);
    const h = Math.max(window.innerHeight - 2 * padding, 50);
    const k = Math.min(w / Math.max(box.x[1] - box.x[0], 1), h / Math.max(box.y[1] - box.y[0], 1), 8);
    // Put the box centre at the centre of the free area instead of the screen centre.
    const shift = (left - right) / 2 / k;
    graph.centerAt((box.x[0] + box.x[1]) / 2 - shift, (box.y[0] + box.y[1]) / 2, ms);
    graph.zoom(k, ms);
  }

  // × on a node goes back to the overview; × on the overview hides the panel.
  document.getElementById("panel-close").onclick = () => (selectedNode ? selectNode(null) : closePanel());
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (activePath) clearPath();
    else if (selectedNode) selectNode(null);
    else closePanel();
  });

  // ---------- toolbar ----------
  const countBy = (key) => data.nodes.reduce((m, n) => ((m[n[key]] = (m[n[key]] || 0) + 1), m), {});
  const kindCounts = countBy("kind");

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

  document.getElementById("fit").onclick = () => fitAll();

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
  if (predLinks.length) {
    legend.append(
      el("div", { className: "static" }, el("span", { className: "dash" }),
        label(predRel), el("span", { className: "count" }, predLinks.length)),
      el("div", { className: "static" }, el("span", { className: "ring" }),
        `${meta.kinds.drug} with a prediction`, el("span", { className: "count" }, predictedDrugs.size))
    );
  }
  const candidateCount = data.nodes.filter(isCandidate).length;
  if (candidateCount) {
    legend.append(
      el("div", { className: "static" }, el("span", { className: "ring dashed" }),
        `${meta.kinds.drug} candidate (heuristic)`, el("span", { className: "count" }, candidateCount))
    );
  }
  legend.append(el("div", { className: "static meta" }, "Bigger node = more connections"));

  window.addEventListener("resize", () => graph.width(window.innerWidth).height(window.innerHeight));
  renderOverview();
})();
