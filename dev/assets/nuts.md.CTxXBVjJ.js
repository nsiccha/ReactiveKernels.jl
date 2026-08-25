import{_ as t,C as n,ap as l,o as r,c as p,ao as a,E as d,w as k,a2 as h,j as o}from"./chunks/framework.DF1ZdG05.js";const b=JSON.parse('{"title":"Graph-backed NUTS sampling","description":"","frontmatter":{},"headers":[],"relativePath":"nuts.md","filePath":"nuts.md","lastUpdated":null}'),c={name:"nuts.md"},g={class:"vp-raw-html",innerHTML:`<div class="rk-dag" aria-label="Interactive ReactiveKernels plan · total cost 5.0"><style>
.rk-dag{container-type:inline-size;color:#0f172a;background:#fff;border:1px solid #cbd5e1;border-radius:12px;overflow:hidden;font:13px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;box-shadow:0 1px 2px rgba(15,23,42,.06)}
.rk-dag *{box-sizing:border-box}.rk-dag-toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:9px 12px;border-bottom:1px solid #e2e8f0;background:#f8fafc}.rk-dag-toolbar button{appearance:none;border:1px solid #94a3b8;border-radius:6px;background:#fff;color:#0f172a;padding:5px 10px;font:600 12px/1.2 inherit;cursor:pointer}.rk-dag-toolbar button:hover{background:#eff6ff;border-color:#2563eb}.rk-dag-toolbar button:focus-visible{outline:3px solid rgba(37,99,235,.3);outline-offset:1px}.rk-dag-hint{margin-left:auto;color:#475569;font-size:12px}
.rk-dag-workspace{display:grid;grid-template-columns:minmax(0,1fr) minmax(190px,24%);min-height:360px;max-height:min(720px,72vh)}.rk-dag-canvas{min-width:0;min-height:360px;overflow:hidden;touch-action:none;cursor:grab;background:#fff}.rk-dag-canvas:active{cursor:grabbing}.rk-dag-canvas:focus-visible{outline:3px solid rgba(37,99,235,.3);outline-offset:-3px}.rk-dag-canvas svg{display:block;width:100%;height:100%;max-height:none}.rk-dag-canvas .rk-node{cursor:pointer}
.rk-dag-inspector{min-width:0;overflow:auto;border-left:1px solid #e2e8f0;background:#f8fafc;padding:16px}.rk-dag-inspector h3{font-size:14px;line-height:1.3;margin:0 0 12px;overflow-wrap:anywhere}.rk-dag-inspector dl{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:7px 10px;margin:0}.rk-dag-inspector dt{font-weight:650;color:#475569}.rk-dag-inspector dd{margin:0;overflow-wrap:anywhere;white-space:normal}.rk-dag-inspector .rk-muted{color:#64748b}
@media(max-width:700px){.rk-dag-hint{width:100%;margin-left:0}.rk-dag-workspace{grid-template-columns:1fr;max-height:none}.rk-dag-inspector{border-left:0;border-top:1px solid #e2e8f0}.rk-dag-canvas{min-height:320px}}
@container(max-width:700px){.rk-dag-hint{width:100%;margin-left:0}.rk-dag-workspace{grid-template-columns:1fr;max-height:none}.rk-dag-inspector{border-left:0;border-top:1px solid #e2e8f0}.rk-dag-canvas{min-height:320px}}
@media(prefers-color-scheme:dark){.rk-dag{color:#e2e8f0;background:#0f172a;border-color:#475569}.rk-dag-toolbar,.rk-dag-inspector{background:#1e293b;border-color:#475569}.rk-dag-toolbar button{background:#0f172a;color:#e2e8f0;border-color:#64748b}.rk-dag-toolbar button:hover{background:#1e3a5f}.rk-dag-hint,.rk-dag-inspector dt,.rk-dag-inspector .rk-muted{color:#cbd5e1}}
</style>
<div class="rk-dag-toolbar" role="toolbar" aria-label="DAG navigation"><button type="button" data-rk-fit>Fit</button><button type="button" data-rk-zoom-in>Zoom in</button><button type="button" data-rk-zoom-out>Zoom out</button><span class="rk-dag-hint">Drag to pan · scroll to zoom · select a node to inspect</span></div><div class="rk-dag-workspace"><div class="rk-dag-canvas" data-rk-canvas tabindex="0" aria-label="DAG canvas; use plus, minus, and zero keys to zoom and fit"><svg xmlns="http://www.w3.org/2000/svg" width="100%" viewBox="0 0 2388.7999999999997 374.0" role="img" aria-label="ReactiveKernels plan · total cost 5.0" data-rk-svg>
<style>
.rk-bg{fill:#fff}.rk-title{font:600 15px system-ui,sans-serif;fill:#0f172a}.rk-node text{font:12px system-ui,sans-serif;fill:#0f172a;pointer-events:none}.rk-node .detail{font-size:10px;fill:#475569}.rk-node rect,.rk-node ellipse{stroke-width:1.5}.rk-node.value ellipse{fill:#fff;stroke:#475569}.rk-node.recipe rect{fill:#f1f5f9;stroke:#475569}.rk-node.selected rect{fill:#dbeafe;stroke:#2563eb}.rk-node.have ellipse{fill:#dcfce7;stroke:#15803d;stroke-width:2.5}.rk-node.want ellipse{fill:#ffedd5;stroke:#c2410c;stroke-width:2.5}.rk-node.havewant ellipse{fill:#fef3c7;stroke:#a16207;stroke-width:2.5}.rk-node.alternative rect{fill:#f8fafc;stroke:#94a3b8;stroke-dasharray:6 4}.rk-node.effectful rect{fill:#fee2e2;stroke:#b91c1c;stroke-dasharray:6 4}.rk-node.is-inspected rect,.rk-node.is-inspected ellipse{stroke:#7c3aed;stroke-width:4}.rk-node:focus{outline:none}.rk-node:focus rect,.rk-node:focus ellipse{stroke:#7c3aed;stroke-width:4}.rk-edge{fill:none;stroke:#64748b;stroke-width:1.5}.rk-edge.alternative{stroke:#94a3b8;stroke-dasharray:6 4}
</style>
<defs><marker id="rk-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/></marker></defs>
<rect class="rk-bg" x="0" y="0" width="2388.7999999999997" height="374.0"/>
<text class="rk-title" x="20" y="25">ReactiveKernels plan · total cost 5.0</text>
<path class="rk-edge selected" d="M 304.8 206.0 C 350.79999999999995 206.0, 350.79999999999995 165.0, 396.79999999999995 165.0" marker-end="url(#rk-arrow)" data-rk-src="v_81" data-rk-dst="r_1"/>
<path class="rk-edge selected" d="M 640.8 165.0 C 686.8 165.0, 686.8 115.0, 732.8 115.0" marker-end="url(#rk-arrow)" data-rk-src="r_1" data-rk-dst="v_82"/>
<path class="rk-edge selected" d="M 304.8 306.0 C 350.79999999999995 306.0, 350.79999999999995 247.0, 396.79999999999995 247.0" marker-end="url(#rk-arrow)" data-rk-src="v_79" data-rk-dst="r_3"/>
<path class="rk-edge selected" d="M 640.8 247.0 C 699.3999999999999 247.0, 699.3999999999999 206.0, 757.9999999999999 206.0" marker-end="url(#rk-arrow)" data-rk-src="r_3" data-rk-dst="v_83"/>
<path class="rk-edge selected" d="M 640.8 247.0 C 686.8 247.0, 686.8 297.0, 732.8 297.0" marker-end="url(#rk-arrow)" data-rk-src="r_3" data-rk-dst="v_84"/>
<path class="rk-edge selected" d="M 1005.5999999999999 115.0 C 1051.6 115.0, 1051.6 165.0, 1097.6 165.0" marker-end="url(#rk-arrow)" data-rk-src="v_82" data-rk-dst="r_4"/>
<path class="rk-edge selected" d="M 304.8 106.0 C 701.1999999999999 106.0, 701.1999999999999 165.0, 1097.6 165.0" marker-end="url(#rk-arrow)" data-rk-src="v_80" data-rk-dst="r_4"/>
<path class="rk-edge selected" d="M 1341.6 165.0 C 1400.1999999999998 165.0, 1400.1999999999998 106.0, 1458.8 106.0" marker-end="url(#rk-arrow)" data-rk-src="r_4" data-rk-dst="v_85"/>
<path class="rk-edge selected" d="M 1341.6 165.0 C 1387.6 165.0, 1387.6 197.0, 1433.6 197.0" marker-end="url(#rk-arrow)" data-rk-src="r_4" data-rk-dst="v_86"/>
<path class="rk-edge selected" d="M 980.4 206.0 C 1389.3999999999999 206.0, 1389.3999999999999 206.0, 1798.3999999999999 206.0" marker-end="url(#rk-arrow)" data-rk-src="v_83" data-rk-dst="r_5"/>
<path class="rk-edge selected" d="M 1681.2 106.0 C 1739.8 106.0, 1739.8 206.0, 1798.3999999999999 206.0" marker-end="url(#rk-arrow)" data-rk-src="v_85" data-rk-dst="r_5"/>
<path class="rk-edge selected" d="M 2042.3999999999999 206.0 C 2088.3999999999996 206.0, 2088.3999999999996 206.0, 2134.3999999999996 206.0" marker-end="url(#rk-arrow)" data-rk-src="r_5" data-rk-dst="v_87"/>
<path class="rk-edge selected" d="M 1005.5999999999999 297.0 C 1051.6 297.0, 1051.6 247.0, 1097.6 247.0" marker-end="url(#rk-arrow)" data-rk-src="v_84" data-rk-dst="r_6"/>
<path class="rk-edge selected" d="M 1341.6 247.0 C 1387.6 247.0, 1387.6 297.0, 1433.6 297.0" marker-end="url(#rk-arrow)" data-rk-src="r_6" data-rk-dst="v_88"/>
<g class="rk-node value have" role="button" tabindex="0" aria-label="pos — HAVE · ::Vector{Float64} · value 79" data-rk-id="v_79" data-rk-kind="value" data-rk-state="have" data-rk-label="pos" data-rk-detail="HAVE · ::Vector{Float64} · value 79">
<title>pos — HAVE · ::Vector{Float64} · value 79</title>
<ellipse cx="168.4" cy="306.0" rx="136.4" ry="36.0"/>
<text text-anchor="middle">
<tspan x="168.4" y="291.0">pos</tspan>
<tspan class="detail" x="168.4" y="306.0">HAVE · ::Vector{Float64} · value 7</tspan>
<tspan class="detail" x="168.4" y="321.0">9</tspan>
</text></g>
<g class="rk-node value have" role="button" tabindex="0" aria-label="mom — HAVE · ::Vector{Float64} · value 80" data-rk-id="v_80" data-rk-kind="value" data-rk-state="have" data-rk-label="mom" data-rk-detail="HAVE · ::Vector{Float64} · value 80">
<title>mom — HAVE · ::Vector{Float64} · value 80</title>
<ellipse cx="168.4" cy="106.0" rx="136.4" ry="36.0"/>
<text text-anchor="middle">
<tspan x="168.4" y="91.0">mom</tspan>
<tspan class="detail" x="168.4" y="106.0">HAVE · ::Vector{Float64} · value 8</tspan>
<tspan class="detail" x="168.4" y="121.0">0</tspan>
</text></g>
<g class="rk-node value have" role="button" tabindex="0" aria-label="metric — HAVE · ::Matrix{Float64} · value 81" data-rk-id="v_81" data-rk-kind="value" data-rk-state="have" data-rk-label="metric" data-rk-detail="HAVE · ::Matrix{Float64} · value 81">
<title>metric — HAVE · ::Matrix{Float64} · value 81</title>
<ellipse cx="168.4" cy="206.0" rx="136.4" ry="36.0"/>
<text text-anchor="middle">
<tspan x="168.4" y="191.0">metric</tspan>
<tspan class="detail" x="168.4" y="206.0">HAVE · ::Matrix{Float64} · value 8</tspan>
<tspan class="detail" x="168.4" y="221.0">1</tspan>
</text></g>
<g class="rk-node value want" role="button" tabindex="0" aria-label="chol_metric — WANT · ::LinearAlgebra.Cholesky{Float64, Matrix{Float64}} · value 82" data-rk-id="v_82" data-rk-kind="value" data-rk-state="want" data-rk-label="chol_metric" data-rk-detail="WANT · ::LinearAlgebra.Cholesky{Float64, Matrix{Float64}} · value 82">
<title>chol_metric — WANT · ::LinearAlgebra.Cholesky{Float64, Matrix{Float64}} · value 82</title>
<ellipse cx="869.1999999999999" cy="115.0" rx="136.4" ry="36.0"/>
<text text-anchor="middle">
<tspan x="869.1999999999999" y="100.0">chol_metric</tspan>
<tspan class="detail" x="869.1999999999999" y="115.0">WANT · ::LinearAlgebra.Cholesky{Fl</tspan>
<tspan class="detail" x="869.1999999999999" y="130.0">oat64, Matrix{Float64}} · value 82</tspan>
</text></g>
<g class="rk-node value want" role="button" tabindex="0" aria-label="pot — WANT · ::Float64 · value 83" data-rk-id="v_83" data-rk-kind="value" data-rk-state="want" data-rk-label="pot" data-rk-detail="WANT · ::Float64 · value 83">
<title>pot — WANT · ::Float64 · value 83</title>
<ellipse cx="869.1999999999999" cy="206.0" rx="111.2" ry="27.0"/>
<text text-anchor="middle">
<tspan x="869.1999999999999" y="198.5">pot</tspan>
<tspan class="detail" x="869.1999999999999" y="213.5">WANT · ::Float64 · value 83</tspan>
</text></g>
<g class="rk-node value want" role="button" tabindex="0" aria-label="dpot_dpos — WANT · ::Vector{Float64} · value 84" data-rk-id="v_84" data-rk-kind="value" data-rk-state="want" data-rk-label="dpot_dpos" data-rk-detail="WANT · ::Vector{Float64} · value 84">
<title>dpot_dpos — WANT · ::Vector{Float64} · value 84</title>
<ellipse cx="869.1999999999999" cy="297.0" rx="136.4" ry="36.0"/>
<text text-anchor="middle">
<tspan x="869.1999999999999" y="282.0">dpot_dpos</tspan>
<tspan class="detail" x="869.1999999999999" y="297.0">WANT · ::Vector{Float64} · value 8</tspan>
<tspan class="detail" x="869.1999999999999" y="312.0">4</tspan>
</text></g>
<g class="rk-node value want" role="button" tabindex="0" aria-label="kin — WANT · ::Float64 · value 85" data-rk-id="v_85" data-rk-kind="value" data-rk-state="want" data-rk-label="kin" data-rk-detail="WANT · ::Float64 · value 85">
<title>kin — WANT · ::Float64 · value 85</title>
<ellipse cx="1570.0" cy="106.0" rx="111.2" ry="27.0"/>
<text text-anchor="middle">
<tspan x="1570.0" y="98.5">kin</tspan>
<tspan class="detail" x="1570.0" y="113.5">WANT · ::Float64 · value 85</tspan>
</text></g>
<g class="rk-node value want" role="button" tabindex="0" aria-label="dham_dmom — WANT · ::Vector{Float64} · value 86" data-rk-id="v_86" data-rk-kind="value" data-rk-state="want" data-rk-label="dham_dmom" data-rk-detail="WANT · ::Vector{Float64} · value 86">
<title>dham_dmom — WANT · ::Vector{Float64} · value 86</title>
<ellipse cx="1570.0" cy="197.0" rx="136.4" ry="36.0"/>
<text text-anchor="middle">
<tspan x="1570.0" y="182.0">dham_dmom</tspan>
<tspan class="detail" x="1570.0" y="197.0">WANT · ::Vector{Float64} · value 8</tspan>
<tspan class="detail" x="1570.0" y="212.0">6</tspan>
</text></g>
<g class="rk-node value want" role="button" tabindex="0" aria-label="ham — WANT · ::Float64 · value 87" data-rk-id="v_87" data-rk-kind="value" data-rk-state="want" data-rk-label="ham" data-rk-detail="WANT · ::Float64 · value 87">
<title>ham — WANT · ::Float64 · value 87</title>
<ellipse cx="2245.5999999999995" cy="206.0" rx="111.2" ry="27.0"/>
<text text-anchor="middle">
<tspan x="2245.5999999999995" y="198.5">ham</tspan>
<tspan class="detail" x="2245.5999999999995" y="213.5">WANT · ::Float64 · value 87</tspan>
</text></g>
<g class="rk-node value want" role="button" tabindex="0" aria-label="dham_dpos — WANT · ::Vector{Float64} · value 88" data-rk-id="v_88" data-rk-kind="value" data-rk-state="want" data-rk-label="dham_dpos" data-rk-detail="WANT · ::Vector{Float64} · value 88">
<title>dham_dpos — WANT · ::Vector{Float64} · value 88</title>
<ellipse cx="1570.0" cy="297.0" rx="136.4" ry="36.0"/>
<text text-anchor="middle">
<tspan x="1570.0" y="282.0">dham_dpos</tspan>
<tspan class="detail" x="1570.0" y="297.0">WANT · ::Vector{Float64} · value 8</tspan>
<tspan class="detail" x="1570.0" y="312.0">8</tspan>
</text></g>
<g class="rk-node recipe selected" role="button" tabindex="0" aria-label="cholesky — selected · cost 1.0 · recipe 1" data-rk-id="r_1" data-rk-kind="recipe" data-rk-state="selected" data-rk-label="cholesky" data-rk-detail="selected · cost 1.0 · recipe 1">
<title>cholesky — selected · cost 1.0 · recipe 1</title>
<rect x="396.79999999999995" y="138.0" width="244.0" height="54.0" rx="8"/>
<text text-anchor="middle">
<tspan x="518.8" y="157.5">cholesky</tspan>
<tspan class="detail" x="518.8" y="172.5">selected · cost 1.0 · recipe 1</tspan>
</text></g>
<g class="rk-node recipe selected" role="button" tabindex="0" aria-label="potential_gradient — selected · cost 1.0 · recipe 3" data-rk-id="r_3" data-rk-kind="recipe" data-rk-state="selected" data-rk-label="potential_gradient" data-rk-detail="selected · cost 1.0 · recipe 3">
<title>potential_gradient — selected · cost 1.0 · recipe 3</title>
<rect x="396.79999999999995" y="220.0" width="244.0" height="54.0" rx="8"/>
<text text-anchor="middle">
<tspan x="518.8" y="239.5">potential_gradient</tspan>
<tspan class="detail" x="518.8" y="254.5">selected · cost 1.0 · recipe 3</tspan>
</text></g>
<g class="rk-node recipe selected" role="button" tabindex="0" aria-label="#3 — selected · cost 1.0 · recipe 4" data-rk-id="r_4" data-rk-kind="recipe" data-rk-state="selected" data-rk-label="#3" data-rk-detail="selected · cost 1.0 · recipe 4">
<title>#3 — selected · cost 1.0 · recipe 4</title>
<rect x="1097.6" y="138.0" width="244.0" height="54.0" rx="8"/>
<text text-anchor="middle">
<tspan x="1219.6" y="157.5">#3</tspan>
<tspan class="detail" x="1219.6" y="172.5">selected · cost 1.0 · recipe 4</tspan>
</text></g>
<g class="rk-node recipe selected" role="button" tabindex="0" aria-label="+ — selected · cost 1.0 · recipe 5" data-rk-id="r_5" data-rk-kind="recipe" data-rk-state="selected" data-rk-label="+" data-rk-detail="selected · cost 1.0 · recipe 5">
<title>+ — selected · cost 1.0 · recipe 5</title>
<rect x="1798.3999999999999" y="179.0" width="244.0" height="54.0" rx="8"/>
<text text-anchor="middle">
<tspan x="1920.3999999999999" y="198.5">+</tspan>
<tspan class="detail" x="1920.3999999999999" y="213.5">selected · cost 1.0 · recipe 5</tspan>
</text></g>
<g class="rk-node recipe selected" role="button" tabindex="0" aria-label="#5 — selected · cost 1.0 · recipe 6" data-rk-id="r_6" data-rk-kind="recipe" data-rk-state="selected" data-rk-label="#5" data-rk-detail="selected · cost 1.0 · recipe 6">
<title>#5 — selected · cost 1.0 · recipe 6</title>
<rect x="1097.6" y="220.0" width="244.0" height="54.0" rx="8"/>
<text text-anchor="middle">
<tspan x="1219.6" y="239.5">#5</tspan>
<tspan class="detail" x="1219.6" y="254.5">selected · cost 1.0 · recipe 6</tspan>
</text></g>
</svg></div><aside class="rk-dag-inspector" aria-live="polite" aria-label="Node inspector"><h3 data-rk-inspect-title>Select a value or recipe</h3><dl><dt>Kind</dt><dd data-rk-inspect-kind class="rk-muted">—</dd><dt>State</dt><dd data-rk-inspect-state class="rk-muted">—</dd><dt>Details</dt><dd data-rk-inspect-detail class="rk-muted">—</dd><dt>Incoming</dt><dd data-rk-inspect-incoming class="rk-muted">—</dd><dt>Outgoing</dt><dd data-rk-inspect-outgoing class="rk-muted">—</dd></dl></aside></div><script>
(() => {
  const script = document.currentScript;
  const root = script && script.closest('.rk-dag');
  if (!root || root.dataset.rkReady === 'true') return;
  root.dataset.rkReady = 'true';
  const canvas = root.querySelector('[data-rk-canvas]');
  const svg = root.querySelector('[data-rk-svg]');
  if (!canvas || !svg) return;

  const rawBox = svg.getAttribute('viewBox').trim().split(/s+/).map(Number);
  const original = {x: rawBox[0], y: rawBox[1], w: rawBox[2], h: rawBox[3]};
  let box = {...original};
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  const applyBox = () => svg.setAttribute('viewBox', \`\${box.x} \${box.y} \${box.w} \${box.h}\`);
  const fit = () => { box = {...original}; applyBox(); };
  const zoom = (scale, clientX, clientY) => {
    const rect = canvas.getBoundingClientRect();
    const px = clientX == null ? .5 : clamp((clientX - rect.left) / rect.width, 0, 1);
    const py = clientY == null ? .5 : clamp((clientY - rect.top) / rect.height, 0, 1);
    const nextW = clamp(box.w * scale, original.w * .12, original.w * 8);
    const nextH = clamp(box.h * scale, original.h * .12, original.h * 8);
    box.x += (box.w - nextW) * px;
    box.y += (box.h - nextH) * py;
    box.w = nextW;
    box.h = nextH;
    applyBox();
  };

  root.querySelector('[data-rk-fit]').addEventListener('click', fit);
  root.querySelector('[data-rk-zoom-in]').addEventListener('click', () => zoom(.8));
  root.querySelector('[data-rk-zoom-out]').addEventListener('click', () => zoom(1.25));
  canvas.addEventListener('wheel', event => {
    event.preventDefault();
    zoom(event.deltaY < 0 ? .82 : 1.22, event.clientX, event.clientY);
  }, {passive: false});

  let drag = null;
  canvas.addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    drag = {x: event.clientX, y: event.clientY, box: {...box}, moved: false};
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointermove', event => {
    if (!drag) return;
    const rect = canvas.getBoundingClientRect();
    const dx = (event.clientX - drag.x) * drag.box.w / rect.width;
    const dy = (event.clientY - drag.y) * drag.box.h / rect.height;
    drag.moved = drag.moved || Math.abs(event.clientX - drag.x) + Math.abs(event.clientY - drag.y) > 3;
    box.x = drag.box.x - dx;
    box.y = drag.box.y - dy;
    applyBox();
  });
  const stopDrag = event => {
    if (!drag) return;
    root.dataset.rkDragged = drag.moved ? 'true' : 'false';
    drag = null;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  };
  canvas.addEventListener('pointerup', stopDrag);
  canvas.addEventListener('pointercancel', stopDrag);

  const nodes = Array.from(root.querySelectorAll('[data-rk-id]'));
  const byId = new Map(nodes.map(node => [node.dataset.rkId, node]));
  const edges = Array.from(root.querySelectorAll('[data-rk-src][data-rk-dst]'));
  const field = name => root.querySelector(\`[data-rk-inspect-\${name}]\`);
  const nodeName = id => {
    const node = byId.get(id);
    return node ? node.dataset.rkLabel : id;
  };
  const inspect = node => {
    nodes.forEach(candidate => candidate.classList.toggle('is-inspected', candidate === node));
    field('title').textContent = node.dataset.rkLabel;
    field('kind').textContent = node.dataset.rkKind;
    field('state').textContent = node.dataset.rkState;
    field('detail').textContent = node.dataset.rkDetail;
    const incoming = edges.filter(edge => edge.dataset.rkDst === node.dataset.rkId)
                          .map(edge => nodeName(edge.dataset.rkSrc));
    const outgoing = edges.filter(edge => edge.dataset.rkSrc === node.dataset.rkId)
                          .map(edge => nodeName(edge.dataset.rkDst));
    field('incoming').textContent = incoming.length ? incoming.join(' · ') : 'None';
    field('outgoing').textContent = outgoing.length ? outgoing.join(' · ') : 'None';
  };
  nodes.forEach(node => {
    node.addEventListener('click', () => {
      if (root.dataset.rkDragged === 'true') {
        root.dataset.rkDragged = 'false';
        return;
      }
      inspect(node);
    });
    node.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        inspect(node);
      }
    });
  });
  canvas.addEventListener('keydown', event => {
    if (event.target !== canvas) return;
    if (event.key === '+' || event.key === '=') { event.preventDefault(); zoom(.8); }
    else if (event.key === '-') { event.preventDefault(); zoom(1.25); }
    else if (event.key === '0') { event.preventDefault(); fit(); }
  });

  root.rkDag = Object.freeze({fit, zoomIn: () => zoom(.8), zoomOut: () => zoom(1.25),
    inspect: id => { const node = byId.get(id); if (node) inspect(node); }});
})();
<\/script>
</div>`};function y(u,s,v,m,E,_){const e=n("ClientOnly"),i=l("exec-scripts");return r(),p("div",null,[s[0]||(s[0]=a(`<h1 id="Graph-backed-NUTS-sampling" tabindex="-1">Graph-backed NUTS sampling <a class="header-anchor" href="#Graph-backed-NUTS-sampling" aria-label="Permalink to &quot;Graph-backed NUTS sampling {#Graph-backed-NUTS-sampling}&quot;">​</a></h1><p><code>ReactiveKernels</code> includes the multinomial NUTS transition and warmup utilities ported from ReactiveHMC.jl. The transition follows Hoffman and Gelman&#39;s <a href="https://jmlr.org/papers/v15/hoffman14a.html" target="_blank" rel="noreferrer">Algorithm 3</a>; Hamiltonian fields come from a compiled reactive program, so mutating <code>pos</code>, <code>mom</code>, or <code>metric</code> invalidates and lazily recomputes only their downstream recipes.</p><p>The complete runnable source is <a href="https://github.com/nsiccha/ReactiveKernels.jl/blob/main/examples/nuts.jl" target="_blank" rel="noreferrer"><code>examples/nuts.jl</code></a>. Its model and Hamiltonian are ordinary declarative recipes:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> LinearAlgebra, Random, ReactiveKernels</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">potential</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(position) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(abs2, position) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">potential_gradient</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(position) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">potential</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(position), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">copy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(position))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @kernel</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> begin</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    pos</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Vector{Float64}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    mom</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Vector{Float64}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    metric</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Matrix{Float64}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    chol_metric</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Cholesky{Float64,Matrix{Float64}}</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cholesky</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(metric)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    pot</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Float64</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> potential</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(pos)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (pot, dpot_dpos</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Vector{Float64}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> potential_gradient</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(pos)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (kin</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Float64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, dham_dmom</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Vector{Float64}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> begin</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        velocity </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> chol_metric </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">\\</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> mom</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.5</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> *</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">logdet</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(chol_metric) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dot</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(mom, velocity)), velocity)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ham</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Float64</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pot </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> kin</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dham_dpos</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Vector{Float64}</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dpot_dpos</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (pot, dpot_dpos, chol_metric, kin, ham, dham_dpos, dham_dmom)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>The same public spec exposes the selected plan, generated Hamiltonian getter, and colored compute DAG before sampling:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dimension </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 4</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">point </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> euclidean_phasepoint</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, (</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    pos </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> zeros</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dimension),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    mom </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> zeros</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dimension),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    metric </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Matrix{Float64}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(I, dimension, dimension),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">selected_plan </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> explain</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">plan</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(point))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">generated_hamiltonian </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> code_expr</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(point, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:ham</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dag </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> visualize</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">plan</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(point))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(selected_plan, generated_hamiltonian)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">(&quot;Have:\\n  pos, mom, metric\\nWant:\\n  pot, dpot_dpos, chol_metric, kin, ham, dham_dpos, dham_dmom\\nSelected recipes:\\n  chol_metric = cholesky(metric)              cost 1.0\\n  (pot, dpot_dpos) = potential_gradient(pos)  cost 1.0\\n  (kin, dham_dmom) = #3(chol_metric, mom)     cost 1.0\\n  ham = +(pot, kin)                           cost 1.0\\n  dham_dpos = #5(dpot_dpos)                   cost 1.0\\nAlternatives not selected:\\n  pot = potential(pos)  (cost 1.0)\\nTotal graph cost: 5.0&quot;, :(function (__ops__, __slots__, __valid__)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">      begin</span></span>
<span class="line"><span style="--shiki-light:#959da5;--shiki-dark:#959da5;">          #= /home/runner/work/ReactiveKernels.jl/ReactiveKernels.jl/src/stateful.jl:163 =#</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">          if !(__valid__[9])</span></span>
<span class="line"><span style="--shiki-light:#959da5;--shiki-dark:#959da5;">              #= /home/runner/work/ReactiveKernels.jl/ReactiveKernels.jl/src/stateful.jl:164 =#</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">              begin</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                  begin</span></span>
<span class="line"><span style="--shiki-light:#959da5;--shiki-dark:#959da5;">                      #= /home/runner/work/ReactiveKernels.jl/ReactiveKernels.jl/src/stateful.jl:163 =#</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                      if !(__valid__[5])</span></span>
<span class="line"><span style="--shiki-light:#959da5;--shiki-dark:#959da5;">                          #= /home/runner/work/ReactiveKernels.jl/ReactiveKernels.jl/src/stateful.jl:164 =#</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                          begin</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                              begin</span></span>
<span class="line"><span style="--shiki-light:#959da5;--shiki-dark:#959da5;">                                  #= /home/runner/work/ReactiveKernels.jl/ReactiveKernels.jl/src/stateful.jl:125 =#</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                  __valid__[1] || error(&quot;compiled reactive source slot $(slot_index) is invalid&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                              end</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                              (var&quot;##recipe_result#447&quot;, var&quot;##recipe_result#448&quot;) = (__ops__[2])((__slots__[1])[])</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                              begin</span></span>
<span class="line"><span style="--shiki-light:#959da5;--shiki-dark:#959da5;">                                  #= /home/runner/work/ReactiveKernels.jl/ReactiveKernels.jl/src/stateful.jl:155 =#</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                  if !(__valid__[5])</span></span>
<span class="line"><span style="--shiki-light:#959da5;--shiki-dark:#959da5;">                                      #= /home/runner/work/ReactiveKernels.jl/ReactiveKernels.jl/src/stateful.jl:156 =#</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                      (__slots__[5])[] = var&quot;##recipe_result#447&quot;</span></span>
<span class="line"><span style="--shiki-light:#959da5;--shiki-dark:#959da5;">                                      #= /home/runner/work/ReactiveKernels.jl/ReactiveKernels.jl/src/stateful.jl:157 =#</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                      __valid__[5] = true</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                  end</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                              end</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                              begin</span></span>
<span class="line"><span style="--shiki-light:#959da5;--shiki-dark:#959da5;">                                  #= /home/runner/work/ReactiveKernels.jl/ReactiveKernels.jl/src/stateful.jl:155 =#</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                  if !(__valid__[6])</span></span>
<span class="line"><span style="--shiki-light:#959da5;--shiki-dark:#959da5;">                                      #= /home/runner/work/ReactiveKernels.jl/ReactiveKernels.jl/src/stateful.jl:156 =#</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                      (__slots__[6])[] = var&quot;##recipe_result#448&quot;</span></span>
<span class="line"><span style="--shiki-light:#959da5;--shiki-dark:#959da5;">                                      #= /home/runner/work/ReactiveKernels.jl/ReactiveKernels.jl/src/stateful.jl:157 =#</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                      __valid__[6] = true</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                  end</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                              end</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                          end</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                      end</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                  end</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                  begin</span></span>
<span class="line"><span style="--shiki-light:#959da5;--shiki-dark:#959da5;">                      #= /home/runner/work/ReactiveKernels.jl/ReactiveKernels.jl/src/stateful.jl:163 =#</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                      if !(__valid__[7])</span></span>
<span class="line"><span style="--shiki-light:#959da5;--shiki-dark:#959da5;">                          #= /home/runner/work/ReactiveKernels.jl/ReactiveKernels.jl/src/stateful.jl:164 =#</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                          begin</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                              begin</span></span>
<span class="line"><span style="--shiki-light:#959da5;--shiki-dark:#959da5;">                                  #= /home/runner/work/ReactiveKernels.jl/ReactiveKernels.jl/src/stateful.jl:163 =#</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                  if !(__valid__[4])</span></span>
<span class="line"><span style="--shiki-light:#959da5;--shiki-dark:#959da5;">                                      #= /home/runner/work/ReactiveKernels.jl/ReactiveKernels.jl/src/stateful.jl:164 =#</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                      begin</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                          begin</span></span>
<span class="line"><span style="--shiki-light:#959da5;--shiki-dark:#959da5;">                                              #= /home/runner/work/ReactiveKernels.jl/ReactiveKernels.jl/src/stateful.jl:125 =#</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                              __valid__[3] || error(&quot;compiled reactive source slot $(slot_index) is invalid&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                          end</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                          var&quot;##recipe_result#449&quot; = (__ops__[1])((__slots__[3])[])</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                          begin</span></span>
<span class="line"><span style="--shiki-light:#959da5;--shiki-dark:#959da5;">                                              #= /home/runner/work/ReactiveKernels.jl/ReactiveKernels.jl/src/stateful.jl:155 =#</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                              if !(__valid__[4])</span></span>
<span class="line"><span style="--shiki-light:#959da5;--shiki-dark:#959da5;">                                                  #= /home/runner/work/ReactiveKernels.jl/ReactiveKernels.jl/src/stateful.jl:156 =#</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                                  (__slots__[4])[] = var&quot;##recipe_result#449&quot;</span></span>
<span class="line"><span style="--shiki-light:#959da5;--shiki-dark:#959da5;">                                                  #= /home/runner/work/ReactiveKernels.jl/ReactiveKernels.jl/src/stateful.jl:157 =#</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                                  __valid__[4] = true</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                              end</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                          end</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                      end</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                  end</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                              end</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                              begin</span></span>
<span class="line"><span style="--shiki-light:#959da5;--shiki-dark:#959da5;">                                  #= /home/runner/work/ReactiveKernels.jl/ReactiveKernels.jl/src/stateful.jl:125 =#</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                  __valid__[2] || error(&quot;compiled reactive source slot $(slot_index) is invalid&quot;)</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                              end</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                              (var&quot;##recipe_result#450&quot;, var&quot;##recipe_result#451&quot;) = (__ops__[3])((__slots__[4])[], (__slots__[2])[])</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                              begin</span></span>
<span class="line"><span style="--shiki-light:#959da5;--shiki-dark:#959da5;">                                  #= /home/runner/work/ReactiveKernels.jl/ReactiveKernels.jl/src/stateful.jl:155 =#</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                  if !(__valid__[7])</span></span>
<span class="line"><span style="--shiki-light:#959da5;--shiki-dark:#959da5;">                                      #= /home/runner/work/ReactiveKernels.jl/ReactiveKernels.jl/src/stateful.jl:156 =#</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                      (__slots__[7])[] = var&quot;##recipe_result#450&quot;</span></span>
<span class="line"><span style="--shiki-light:#959da5;--shiki-dark:#959da5;">                                      #= /home/runner/work/ReactiveKernels.jl/ReactiveKernels.jl/src/stateful.jl:157 =#</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                      __valid__[7] = true</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                  end</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                              end</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                              begin</span></span>
<span class="line"><span style="--shiki-light:#959da5;--shiki-dark:#959da5;">                                  #= /home/runner/work/ReactiveKernels.jl/ReactiveKernels.jl/src/stateful.jl:155 =#</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                  if !(__valid__[8])</span></span>
<span class="line"><span style="--shiki-light:#959da5;--shiki-dark:#959da5;">                                      #= /home/runner/work/ReactiveKernels.jl/ReactiveKernels.jl/src/stateful.jl:156 =#</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                      (__slots__[8])[] = var&quot;##recipe_result#451&quot;</span></span>
<span class="line"><span style="--shiki-light:#959da5;--shiki-dark:#959da5;">                                      #= /home/runner/work/ReactiveKernels.jl/ReactiveKernels.jl/src/stateful.jl:157 =#</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                      __valid__[8] = true</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                                  end</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                              end</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                          end</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                      end</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                  end</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                  var&quot;##recipe_result#452&quot; = (__ops__[4])((__slots__[5])[], (__slots__[7])[])</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                  begin</span></span>
<span class="line"><span style="--shiki-light:#959da5;--shiki-dark:#959da5;">                      #= /home/runner/work/ReactiveKernels.jl/ReactiveKernels.jl/src/stateful.jl:155 =#</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                      if !(__valid__[9])</span></span>
<span class="line"><span style="--shiki-light:#959da5;--shiki-dark:#959da5;">                          #= /home/runner/work/ReactiveKernels.jl/ReactiveKernels.jl/src/stateful.jl:156 =#</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                          (__slots__[9])[] = var&quot;##recipe_result#452&quot;</span></span>
<span class="line"><span style="--shiki-light:#959da5;--shiki-dark:#959da5;">                          #= /home/runner/work/ReactiveKernels.jl/ReactiveKernels.jl/src/stateful.jl:157 =#</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                          __valid__[9] = true</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                      end</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">                  end</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">              end</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">          end</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">      end</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">      return (__slots__[9])[]</span></span>
<span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">  end))</span></span></code></pre></div><p>The selected plan&#39;s HTML-showable DAG is rendered directly, preserving its fit, zoom, pan, and node-inspection controls:</p>`,8)),d(e,null,{default:k(()=>[h(o("div",g,null,512),[[i]])]),_:1}),s[1]||(s[1]=a(`<p>The same compiled reactive program then drives warmup and sampling:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">sampler </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> nuts_state</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(point;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    rng </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Xoshiro</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">20260825</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    step_f </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> partial</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(leapfrog!; stepsize </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.35</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    max_depth </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 7</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">warmup </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> warmup!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(sampler, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">chain </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sample!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(sampler, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">100</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">count</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(diagnostic </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> diagnostic</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">diverged, chain</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">diagnostics)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">0</span></span></code></pre></div><p><code>warmup!</code> performs initial step-size search, dual averaging, and windowed diagonal metric adaptation. <code>sample!</code> returns samples and per-transition diagnostics, including acceptance, tree depth, leapfrog count, energy error, and divergence status.</p><p>For a reproducible comparison under identical four-chain settings, run <code>julia --startup-file=no benchmark/nuts_comparison.jl</code>. That script creates a temporary environment and pins AdvancedHMC and DynamicHMC outside the package&#39;s dependencies. Treat its setup time, sampling time, gradient efficiency, divergences, ESS, and R-hat as separate measurements; it is not evidence of blanket sampler superiority.</p>`,5))])}const f=t(c,[["render",y]]);export{b as __pageData,f as default};
