/* ============================================================================
   低空之眼 · 控制台前端
   免构建 免依赖 断网可跑：只用 fetch + DOM 事件 图表用手写进度条
   动效约定：所有出现的元素都有进入动画 所有离开的元素都先播退场动画再从 DOM 移除
   ========================================================================== */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const state = {
  view: 'console',
  assets: [],
  result: null,
  orders: [],
  orderFilter: 'all',
  traces: [],
  health: null,
  kbStats: null,
  busy: false,
};

/* ---------------------------------------------------------------- 基础设施 */

async function apiGet(path) {
  const res = await fetch(path, { headers: { Accept: 'application/json' } });
  if (!res.ok) throw new Error(`${res.status} ${await shortError(res)}`);
  return res.json();
}

async function apiPost(path, body) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  if (!res.ok) throw new Error(`${res.status} ${await shortError(res)}`);
  return res.json();
}

async function shortError(res) {
  try {
    const data = await res.json();
    if (typeof data.detail === 'string') return data.detail;
    if (Array.isArray(data.detail)) return data.detail.map((d) => `${d.loc?.join('.')} ${d.msg}`).join('；');
    return JSON.stringify(data).slice(0, 160);
  } catch (_) {
    return res.statusText;
  }
}

function icon(name, cls = 'ico') {
  return `<svg class="${cls}"><use href="vendor/icons.svg#i-${name}"></use></svg>`;
}

/* 进入动画：把 stagger 子元素逐个点亮 */
function stagger(container, selector, step = 55) {
  $$(selector, container).forEach((el, i) => {
    el.style.animationDelay = REDUCED ? '0ms' : `${i * step}ms`;
  });
}

/* 退场：播完动画再从 DOM 摘掉 保证「消失」也有动画 */
function exitAndRemove(el, done) {
  if (!el) return;
  if (REDUCED) { el.remove(); if (done) done(); return; }
  el.classList.add('is-out');
  let finished = false;
  const finish = () => {
    if (finished) return;
    finished = true;
    el.remove();
    if (done) done();
  };
  el.addEventListener('animationend', finish, { once: true });
  setTimeout(finish, 480);
}

function clearWithExit(container, selector) {
  const nodes = $$(selector, container);
  nodes.forEach((el, i) => setTimeout(() => exitAndRemove(el), REDUCED ? 0 : i * 35));
  return nodes.length;
}

/* 数字滚动 让统计值出现时有过渡而不是硬跳 */
function countUp(el, target, ms = 620) {
  const to = Number(target) || 0;
  if (REDUCED || !to) { el.textContent = fmtNum(to); return; }
  const from = 0;
  const t0 = performance.now();
  const tick = (t) => {
    const p = Math.min(1, (t - t0) / ms);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = fmtNum(from + (to - from) * eased);
    if (p < 1) requestAnimationFrame(tick);
    else el.textContent = fmtNum(to);
  };
  requestAnimationFrame(tick);
}

const fmtNum = (v) => (Number.isInteger(v) ? String(v) : v.toFixed(1));
const fmtMs = (v) => (v >= 100 ? v.toFixed(0) : v.toFixed(1));
const pct = (v) => `${Math.max(0, Math.min(1, Number(v) || 0)) * 100}%`;

/* ---------------------------------------------------------------- 提示条 */

function toast(title, message = '', kind = 'info', ttl = 4200) {
  const box = $('#toasts');
  const el = document.createElement('div');
  el.className = `toast${kind === 'ok' ? ' is-ok' : kind === 'warn' ? ' is-warn' : ''}`;
  const ic = kind === 'ok' ? 'badge-check' : kind === 'warn' ? 'triangle-alert' : 'sparkles';
  el.innerHTML = `${icon(ic)}<div><b></b><p></p></div>`;
  $('b', el).textContent = title;
  const p = $('p', el);
  if (message) p.textContent = message; else p.remove();
  box.appendChild(el);
  setTimeout(() => exitAndRemove(el), ttl);
}

/* ---------------------------------------------------------------- 视图切换 */

function switchView(name) {
  if (state.view === name) return;
  const from = $(`#view-${state.view}`);
  const to = $(`#view-${name}`);
  state.view = name;

  $$('.tab').forEach((t) => t.classList.toggle('is-active', t.dataset.view === name));
  moveInk();
  location.hash = name === 'console' ? '' : name;

  const apply = () => {
    if (from) { from.classList.remove('is-active'); from.classList.add('is-leaving'); }
    to.classList.add('is-active');
    setTimeout(() => from && from.classList.remove('is-leaving'), 320);
    if (name === 'orders') renderOrders();
    if (name === 'traces') loadTraces();
  };

  if (document.startViewTransition && !REDUCED) document.startViewTransition(apply);
  else apply();
}

function moveInk() {
  const active = $('.tab.is-active');
  const ink = $('#tab-ink');
  if (!active || !ink) return;
  const box = active.getBoundingClientRect();
  const bar = $('#tabs').getBoundingClientRect();
  ink.style.width = `${box.width - 20}px`;
  ink.style.transform = `translateX(${box.left - bar.left + 10}px)`;
}

/* ---------------------------------------------------------------- 健康状态 */

async function loadHealth() {
  const pill = $('#health-pill');
  try {
    const h = await apiGet('/api/health');
    state.health = h;
    pill.classList.remove('is-loading');
    pill.classList.toggle('is-ok', h.status === 'ok');
    pill.classList.toggle('is-warn', h.status !== 'ok');
    const back = h.backends || {};
    $('#health-text').textContent =
      h.status === 'ok'
        ? `${h.kb_docs} 份文档 ${h.kb_chunks} 条款 · 检测 ${back.detector || '—'} · 生成 ${back.generator || '—'}`
        : '系统降级运行';
    $('#health-pill').title = (h.warnings || []).join('\n') || '全链路正常';

    const list = $('#backend-list');
    list.innerHTML = '';
    const rows = [
      ['检测后端', back.detector || '—'],
      ['生成后端', back.generator || '—'],
      ['BM25', back.bm25 || '—'],
      ['向量', back.embedder || '—'],
      ['重排', back.reranker || '—'],
    ];
    rows.forEach(([k, v], i) => {
      const dt = document.createElement('dt'); dt.textContent = k;
      const dd = document.createElement('dd'); dd.innerHTML = '<b></b>'; $('b', dd).textContent = v;
      dt.style.animation = `meta-in 320ms var(--ease-out) ${i * 45}ms both`;
      dd.style.animation = `meta-in 320ms var(--ease-out) ${i * 45}ms both`;
      list.append(dt, dd);
    });
    const note = $('#backend-note');
    note.textContent = (h.warnings || []).length
      ? `降级项 ${h.warnings.length} 条：${h.warnings[0]}`
      : '全部后端为本地实现 无外网依赖 断网可跑';
  } catch (err) {
    pill.classList.remove('is-loading');
    pill.classList.add('is-warn');
    $('#health-text').textContent = '服务未就绪';
    toast('连不上服务', String(err.message || err), 'warn');
  }
}

/* ---------------------------------------------------------------- 图库 */

async function loadAssets() {
  try {
    const data = await apiGet('/api/assets');
    state.assets = data.items || [];
    renderAssets();
  } catch (err) {
    toast('图库读取失败', String(err.message || err), 'warn');
  }
}

function renderAssets() {
  const grid = $('#asset-grid');
  clearWithExit(grid, '.asset-card');
  setTimeout(() => {
    state.assets.forEach((a, i) => {
      const card = document.createElement('button');
      card.className = 'asset-card';
      card.type = 'button';
      card.dataset.id = a.image_id;
      card.style.animationDelay = REDUCED ? '0ms' : `${i * 60}ms`;
      card.innerHTML = `
        <img alt="" loading="lazy">
        <figcaption><span class="asset-name"></span><span class="tag">${a.has_annotation ? '含标注' : '无标注'}</span></figcaption>`;
      $('img', card).src = a.url;
      $('.asset-name', card).textContent = a.image_id.replace(/^demo_/, '');
      card.addEventListener('click', () => analyzeAsset(a));
      grid.appendChild(card);
    });
  }, REDUCED ? 0 : 160);
}

/* ---------------------------------------------------------------- 分析主流程 */

function setPipeline(stage, ms) {
  $$('.pnode').forEach((n) => {
    const idx = ['detect', 'diagnose', 'grade_dispatch', 'archive'];
    const cur = idx.indexOf(stage);
    const mine = idx.indexOf(n.dataset.stage);
    n.classList.toggle('is-active', mine === cur);
    n.classList.toggle('is-done', mine < cur);
    if (n.dataset.stage === stage && ms != null) $('.pms', n).textContent = `${fmtMs(ms)}ms`;
  });
  const fill = $('#ptrack-fill');
  const order = ['detect', 'diagnose', 'grade_dispatch', 'archive'];
  fill.style.width = `${((order.indexOf(stage) + 1) / order.length) * 100}%`;
}

function resetPipeline() {
  $$('.pnode').forEach((n) => {
    n.classList.remove('is-active', 'is-done');
    $('.pms', n).textContent = '—';
  });
  $('#ptrack-fill').style.width = '0%';
}

async function analyzeAsset(asset) {
  if (state.busy) return;
  state.busy = true;
  markBusy(asset.image_id);
  toast('开始分析', `${asset.image_id} 走 识别 → 诊断 → 分级 → 派单 → 归档`, 'info', 2600);
  resetPipeline();
  setPipeline('detect');
  clearWithExit($('#overlay'), '.det-box');
  const emptyFrame = $('#frame-empty');
  if (emptyFrame && !emptyFrame.classList.contains('is-gone')) {
    emptyFrame.classList.add('is-gone');
    setTimeout(() => emptyFrame.remove(), 260);
  }
  try {
    const result = await apiPost('/api/analyze', { image_id: asset.image_id });
    await playPipeline(result.timings || []);
    showResult(result);
    await loadOrders();
    toast('分析完成', `${result.diagnoses.length} 条诊断 · ${result.work_orders.length} 张工单 · 端到端 ${fmtMs(result.total_ms)}ms`, 'ok');
  } catch (err) {
    toast('分析失败', String(err.message || err), 'warn');
    resetPipeline();
  } finally {
    state.busy = false;
    markBusy(null);
  }
}

function markBusy(imageId) {
  $$('.asset-card').forEach((c) => {
    c.classList.toggle('is-busy', c.dataset.id === imageId);
    c.classList.toggle('is-active', c.dataset.id === imageId);
    c.disabled = imageId !== null && c.dataset.id !== imageId ? false : false;
  });
  const btn = $('#btn-report-pdf');
  if (imageId) btn.disabled = true;
}

/* 流水线节点按真实耗时依次点亮 让「链路在跑」这件事看得见 */
async function playPipeline(timings) {
  const order = ['detect', 'diagnose', 'grade_dispatch', 'archive'];
  const map = Object.fromEntries(timings.map((t) => [t.stage, t.ms]));
  for (const stage of order) {
    setPipeline(stage, map[stage]);
    await sleep(REDUCED ? 0 : 190);
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function showResult(result) {
  state.result = result;
  const img = $('#stage-img');
  img.hidden = false;
  img.src = `${result.image_url}?t=${Date.now()}`;
  $('#chip-scene').textContent = `场景 ${sceneLabel(result.scene)}`;
  const chipTrace = $('#chip-trace');
  chipTrace.textContent = `追溯号 ${result.trace_id}`;
  chipTrace.classList.remove('chip-fly');
  void chipTrace.offsetWidth;
  chipTrace.classList.add('chip-fly');
  $('#btn-report-pdf').disabled = false;

  const severities = {};
  result.work_orders.forEach((o) => { severities[o.severity] = (severities[o.severity] || 0) + 1; });
  const worst = Object.keys(severities).sort()[0] || '—';

  const meta = $('#stage-meta');
  meta.innerHTML = '';
  const cells = [
    ['检出目标', String(result.detect.detections.length)],
    ['切片块数', String(result.detect.slice_count)],
    ['检测耗时', `${fmtMs(result.detect.elapsed_ms)}ms`],
    ['端到端', `${fmtMs(result.total_ms)}ms`],
    ['最高等级', worst, worst === 'P1' ? 'is-sev-p1' : 'is-sev'],
    ['降级项', String(result.degradations.length)],
  ];
  cells.forEach(([k, v, cls], i) => {
    const cell = document.createElement('div');
    cell.className = `meta-cell${cls ? ' ' + cls : ''}`;
    cell.style.animationDelay = REDUCED ? '0ms' : `${i * 55}ms`;
    cell.innerHTML = '<b></b><span></span>';
    $('b', cell).textContent = v;
    $('span', cell).textContent = k;
    meta.appendChild(cell);
  });

  const box = $('#overlay');
  clearWithExit(box, '.det-box');
  setTimeout(() => drawBoxes(result), REDUCED ? 0 : 220);
  renderFindings(result);
  renderOrderStrip(result.work_orders);
}

function sceneLabel(scene) {
  return { power: '输电通道', agriculture: '农田植保', emergency: '应急消防' }[scene] || scene;
}

/* 检测框：位置用百分比 跟随图片实际渲染尺寸 不写死像素 */
function drawBoxes(result) {
  const img = $('#stage-img');
  const overlay = $('#overlay');
  const place = () => {
    const nw = img.naturalWidth || 1;
    const nh = img.naturalHeight || 1;
    result.detect.detections.forEach((det, i) => {
      const sev = (result.work_orders[i] || {}).severity || 'P2';
      const el = document.createElement('div');
      el.className = `det-box sev-${sev}`;
      el.style.left = pct(det.bbox.x1 / nw);
      el.style.top = pct(det.bbox.y1 / nh);
      el.style.width = pct((det.bbox.x2 - det.bbox.x1) / nw);
      el.style.height = pct((det.bbox.y2 - det.bbox.y1) / nh);
      el.style.animationDelay = REDUCED ? '0ms' : `${i * 130}ms`;
      const label = document.createElement('span');
      label.className = 'det-label';
      label.textContent = `${det.cls} ${det.score.toFixed(2)} · ${sev}`;
      el.appendChild(label);
      overlay.appendChild(el);
    });
  };
  if (img.complete && img.naturalWidth) place();
  else img.addEventListener('load', place, { once: true });
}

/* ---------------------------------------------------------------- 诊断卡 */

function renderFindings(result) {
  const host = $('#findings');
  const empty = $('#findings-empty');
  if (empty && result.diagnoses.length) {
    empty.classList.add('is-gone');
    setTimeout(() => empty.remove(), 260);
  }
  clearWithExit(host, '.finding-card');
  const ordersByIndex = result.work_orders;

  setTimeout(() => {
    result.diagnoses.forEach((d, i) => {
      const order = ordersByIndex[i] || {};
      const card = document.createElement('article');
      card.className = `finding-card sev-${d.severity}${d.grounded ? '' : ' is-ungrounded'}`;
      card.style.animationDelay = REDUCED ? '0ms' : `${i * 90}ms`;

      const head = document.createElement('div');
      head.className = 'finding-head';
      head.innerHTML = `
        <span class="finding-title"></span>
        <span class="finding-cls"></span>
        <span class="sev-badge sev-${d.severity}">${d.severity}</span>
        ${order.route === 'human_review' ? `<span class="route-tag is-hr">${icon('user-check')}待人工确认</span>` : `<span class="route-tag is-auto">${icon('route')}自动派单</span>`}
        <span class="conf-wrap"><span class="conf-track"><span class="conf-fill"></span></span><span class="conf-num"></span></span>`;
      $('.finding-title', head).textContent = d.label;
      $('.finding-cls', head).textContent = d.cls;
      $('.conf-num', head).textContent = d.confidence.toFixed(2);

      const finding = document.createElement('p');
      finding.className = 'finding-finding';
      finding.textContent = d.finding;

      card.append(head, finding);

      if (d.actions.length) {
        const ul = document.createElement('ul');
        ul.className = 'action-list';
        d.actions.forEach((a, k) => {
          const li = document.createElement('li');
          li.style.animationDelay = REDUCED ? '0ms' : `${i * 90 + 160 + k * 60}ms`;
          li.innerHTML = `<span class="action-idx">${k + 1}</span><span class="action-text"></span>`;
          $('.action-text', li).textContent = a;
          ul.appendChild(li);
        });
        card.appendChild(ul);
      }

      if (!d.grounded) {
        const note = document.createElement('div');
        note.className = 'ungrounded-note';
        note.textContent = '知识库未覆盖该缺陷类型 系统只报检出结果 不给处置动作';
        card.appendChild(note);
      }

      if (d.citations.length) {
        // 依据条款与参考条款分两行显示：参考条款是检索命中但没成为处置动作的
        // （判定句、响应分级），它支撑结论但不支撑动作，混在一行会让人误以为都是依据。
        const basis = d.citations.filter((c) => (c.role || 'basis') === 'basis');
        const refs = d.citations.filter((c) => (c.role || 'basis') !== 'basis');
        [['依据条款', basis, false], ['参考条款', refs, true]].forEach(([label, list, isRef]) => {
          if (!list.length) return;
          const row = document.createElement('div');
          row.className = 'cite-row' + (isRef ? ' is-ref' : '');
          const tag = document.createElement('span');
          tag.textContent = label;
          row.appendChild(tag);
          list.forEach((c) => {
            const chip = document.createElement('button');
            chip.type = 'button';
            chip.className = 'cite-chip' + (isRef ? ' is-ref' : '');
            chip.textContent = `${c.clause_no} ${c.doc_title.slice(0, 12)}…`;
            chip.title = `${c.doc_title} ${c.clause_no} · ${c.quote}`;
            chip.addEventListener('click', () => openClause(c));
            row.appendChild(chip);
          });
          card.appendChild(row);
        });
      }

      host.appendChild(card);
      requestAnimationFrame(() => { $('.conf-fill', card).style.width = pct(d.confidence); });
    });
    stagger(host, '.finding-card', 90);
  }, REDUCED ? 0 : 180);
}

function openClause(c) {
  $('#modal-title').textContent = `${c.clause_no} · ${c.doc_title}`;
  const body = $('#modal-body');
  body.innerHTML = '<p class="quote"></p><div class="meta"></div>';
  $('.quote', body).textContent = c.quote;
  $('.meta', body).textContent = `条款 ${c.clause_no}｜相关度 ${Number(c.score).toFixed(3)}｜chunk ${c.chunk_id}`;
  openModal();
}

function openModal() {
  const m = $('#modal');
  m.hidden = false;
  m.classList.remove('is-closing');
}

function closeModal() {
  const m = $('#modal');
  if (m.hidden) return;
  m.classList.add('is-closing');
  setTimeout(() => { m.hidden = true; m.classList.remove('is-closing'); }, REDUCED ? 0 : 240);
}

/* ---------------------------------------------------------------- 工单 */

async function loadOrders() {
  try {
    const data = await apiGet('/api/orders');
    state.orders = data.items || [];
    const badge = $('#orders-badge');
    badge.textContent = String(state.orders.length);
    badge.classList.toggle('is-hot', state.orders.some((o) => o.route === 'human_review'));
    if (state.view === 'orders') renderOrders();
  } catch (err) {
    toast('工单读取失败', String(err.message || err), 'warn');
  }
}

function renderOrderStrip(orders) {
  const host = $('#orders-strip');
  clearWithExit(host, '.order-card');
  setTimeout(() => {
    orders.forEach((o, i) => {
      const el = document.createElement('div');
      el.className = 'order-card';
      el.style.animationDelay = REDUCED ? '0ms' : `${i * 80}ms`;
      el.innerHTML = `
        <span class="sev-badge sev-${o.severity}">${o.severity}</span>
        <div>
          <div class="order-title"></div>
          <div class="order-sub"></div>
        </div>
        <span class="route-tag ${o.route === 'human_review' ? 'is-hr' : 'is-auto'}"></span>`;
      $('.order-title', el).textContent = o.title;
      $('.order-sub', el).textContent = `${o.order_id} · ${o.assignee_role} · SLA ${o.sla_hours}h · ${statusLabel(o.status)} · 通知 ${o.notify_status}`;
      $('.route-tag', el).textContent = o.route === 'human_review' ? '待人工确认' : '自动派单';
      host.appendChild(el);
    });
  }, REDUCED ? 0 : 120);
}

const statusLabel = (s) => ({ created: '已建单', pending_review: '待复核', dispatched: '已派单', closed: '已关闭' }[s] || s);

function renderOrders() {
  const host = $('#order-list');
  const f = state.orderFilter;
  const rows = state.orders.filter((o) =>
    f === 'all' ? true : f === 'human_review' ? o.route === 'human_review' : o.severity === f
  );
  clearWithExit(host, '.order-row');
  if (!rows.length) {
    setTimeout(() => {
      const empty = document.createElement('div');
      empty.className = 'panel empty-panel order-row';
      empty.innerHTML = `${icon('list-checks', 'ico ico-lg')}<p>这个筛选下暂无工单</p><em>先去分析台跑一张图</em>`;
      host.appendChild(empty);
    }, REDUCED ? 0 : 160);
    return;
  }
  setTimeout(() => {
    rows.forEach((o, i) => {
      const el = document.createElement('div');
      el.className = `order-row${o.status === 'closed' ? ' is-done' : ''}`;
      el.style.animationDelay = REDUCED ? '0ms' : `${i * 50}ms`;
      el.innerHTML = `
        <span class="sev-badge sev-${o.severity}">${o.severity}</span>
        <div>
          <div class="order-title"></div>
          <div class="order-sub"></div>
          ${o.actions && o.actions.length ? '<ol class="reason-list"></ol>' : ''}
        </div>
        <div class="row-actions"></div>`;
      $('.order-title', el).textContent = o.title;
      $('.order-sub', el).textContent = `${o.order_id} · ${o.assignee_role} · SLA ${o.sla_hours}h · ${statusLabel(o.status)}`;
      if (o.actions && o.actions.length) {
        const ol = $('.reason-list', el);
        o.actions.slice(0, 3).forEach((a) => {
          const li = document.createElement('li');
          li.textContent = a;
          ol.appendChild(li);
        });
      }
      const acts = $('.row-actions', el);
      if (o.route === 'human_review' && o.status !== 'closed') {
        acts.innerHTML = `
          <button class="btn btn-sm btn-primary" data-act="approve">${icon('check')}<span>确认派单</span></button>
          <button class="btn btn-sm btn-danger" data-act="reject">${icon('x')}<span>退回</span></button>`;
      } else {
        acts.innerHTML = `<span class="chip">${statusLabel(o.status)}</span>`;
      }
      acts.addEventListener('click', (ev) => {
        const btn = ev.target.closest('button[data-act]');
        if (btn) reviewOrder(o, btn.dataset.act, el, btn);
      });
      host.appendChild(el);
    });
  }, REDUCED ? 0 : 140);
}

async function reviewOrder(order, decision, el, btn) {
  btn.classList.add('is-busy');
  const first = $('.ico', btn).outerHTML;
  $('.ico', btn).outerHTML = icon('loader-circle');
  try {
    const res = await apiPost(`/api/orders/${order.order_id}/review`, {
      decision,
      reviewer: '控制台值守',
      note: decision === 'approve' ? '控制台确认' : '控制台退回重新核实',
    });
    toast(decision === 'approve' ? '已确认派单' : '已退回', `${order.order_id} → ${statusLabel(res.status)}`, 'ok');
    exitAndRemove(el, () => {});
    await loadOrders();
    renderOrders();
  } catch (err) {
    $('.ico', btn).outerHTML = first;
    btn.classList.remove('is-busy');
    toast('操作失败', String(err.message || err), 'warn');
  }
}

/* ---------------------------------------------------------------- 知识库 */

function renderKbStats() {
  const dl = $('#kb-stats');
  const s = state.kbStats;
  if (!s) return;
  const kv = dl || document.createElement('dl');
  kv.innerHTML = '';
  const rows = [
    ['条款块', String(s.chunks)],
    ['文档数', String(s.docs)],
    ['BM25 后端', s.bm25_backend],
    ['向量后端', `${s.embedder} · ${s.dim} 维`],
    ['重排后端', s.reranker],
  ];
  rows.forEach(([k, v], i) => {
    const dt = document.createElement('dt'); dt.textContent = k;
    const dd = document.createElement('dd'); dd.innerHTML = '<b></b>'; $('b', dd).textContent = v;
    dt.style.animation = `meta-in 320ms var(--ease-out) ${i * 45}ms both`;
    dd.style.animation = `meta-in 320ms var(--ease-out) ${i * 45}ms both`;
    kv.append(dt, dd);
  });
  const chip = $('#kb-chip');
  chip.textContent = `${s.chunks} 条款 · ${s.docs} 文档`;
  chip.classList.add('chip-fly');
}

async function loadKbStats() {
  try {
    const data = await apiGet('/api/kb/stats');
    state.kbStats = data.stats;
    state.minRelevance = data.min_relevance;
    state.lexiconTerms = data.lexicon_terms;
    renderKbStats();
  } catch (_) { /* 健康接口已经会提示 这里静默 */ }
}

async function doSearch(q) {
  const list = $('#hit-list');
  if (!q.trim()) return;
  clearWithExit(list, '.hit, .refusal, .listen, .hitnote');
  const form = $('#search-form');
  const btn = $('button', form);
  btn.classList.add('is-busy');
  try {
    const res = await apiPost('/api/search', { query: q, top_k: 5 });
    setTimeout(() => {
      // 先说清楚「系统把这句话听成了什么」——术语表会把口语词对齐到规程术语，
      // 这一层是可见的，不是黑箱。
      const terms = res.matched_terms || [];
      if (terms.length || (res.search_query && res.search_query !== q)) {
        const strip = document.createElement('div');
        strip.className = 'listen';
        strip.innerHTML = `<span class="listen-k">${icon('audio-lines', 'ico')}识别到的领域术语</span>
          <span class="listen-terms"></span>
          <span class="listen-q"></span>`;
        $('.listen-terms', strip).textContent = terms.length ? terms.join(' · ') : '未识别到术语表内的词';
        $('.listen-q', strip).textContent = res.search_query === q ? '' : `实际检索用查询：${res.search_query}`;
        list.appendChild(strip);
      }
      if (!res.covered) {
        const box = document.createElement('div');
        box.className = 'refusal';
        if (res.scope_excluded) {
          // 两类拒答必须分开说：一类是知识库明确声明「本范围不适用」，
          // 一类只是分数不够。前者是能力边界，后者是证据不足。
          box.innerHTML = `${icon('shield-off', 'ico ico-lg')}<div><b>超出知识库覆盖范围 系统拒答</b><p></p><p class="refusal-sub"></p></div>`;
          $$('p', box)[0].textContent =
            '最相似的条款本身就是一条「不适用 / 不涵盖」声明 说明该问题落在文档明确排除的范围里 系统不给出任何处置结论';
          $('.refusal-sub', box).textContent = `最近条款 ${res.hits[0] ? res.hits[0].clause.doc_title + ' ' + res.hits[0].clause.clause_no : '无'} · 相似度 ${Number(res.top_score).toFixed(3)}`;
        } else {
          box.innerHTML = `${icon('circle-slash', 'ico ico-lg')}<div><b>知识库未覆盖 系统拒答</b><p></p><p class="refusal-sub"></p></div>`;
          $$('p', box)[0].textContent = '最高相关度低于覆盖阈值 系统不给出任何处置结论 这是设计要求 不是故障';
          $('.refusal-sub', box).textContent = `最高相关度 ${Number(res.top_score).toFixed(3)} · 阈值 ${Number(state.minRelevance != null ? state.minRelevance : 0.18).toFixed(2)}`;
        }
        list.appendChild(box);
      }
      if (res.hits.length && !res.covered) {
        // 拒答时下面仍列出最近的条款，但必须标明它们不是依据。
        const note = document.createElement('p');
        note.className = 'hitnote';
        note.textContent = '以下条款只是「最相似」 全部低于覆盖阈值 不作为处置依据 仅供人工判断参考';
        list.appendChild(note);
      }
      res.hits.forEach((h, i) => {
        const el = document.createElement('article');
        el.className = 'hit';
        if (!res.covered) el.classList.add('is-below');
        el.style.animationDelay = REDUCED ? '0ms' : `${i * 70}ms`;
        el.innerHTML = `
          <div class="hit-head">
            <span class="hit-no"></span>
            <span class="hit-doc"></span>
          </div>
          <p class="hit-text"></p>
          <p class="hit-path"></p>
          <div class="score-bars">
            <div class="sbar sbar-bm25"><i><span>BM25</span><b></b></i><span class="sbar-track"><span class="sbar-fill"></span></span></div>
            <div class="sbar sbar-vec"><i><span>向量</span><b></b></i><span class="sbar-track"><span class="sbar-fill"></span></span></div>
            <div class="sbar sbar-rrf"><i><span>RRF</span><b></b></i><span class="sbar-track"><span class="sbar-fill"></span></span></div>
            <div class="sbar sbar-re"><i><span>重排</span><b></b></i><span class="sbar-track"><span class="sbar-fill"></span></span></div>
          </div>`;
        $('.hit-no', el).textContent = h.clause.clause_no || '无编号';
        $('.hit-doc', el).textContent = h.clause.doc_title;
        $('.hit-text', el).textContent = h.clause.text;
        $('.hit-path', el).textContent = `${h.clause.hierarchy_path}｜${h.clause.source_version}`;
        const vals = [h.bm25, h.vector, h.rrf, h.rerank];
        $$('.sbar', el).forEach((bar, k) => {
          $('b', bar).textContent = Number(vals[k]).toFixed(3);
          $('.sbar-fill', bar).style.width = pct(vals[k]);
        });
        list.appendChild(el);
      });
      const verdict = res.covered ? '检索命中' : (res.scope_excluded ? '超出覆盖范围 拒答' : '按设计拒答');
      toast(verdict, `${res.hits.length} 条命中 · 耗时 ${fmtMs(res.elapsed_ms)}ms`, res.covered ? 'ok' : 'info', 2800);
    }, REDUCED ? 0 : 150);
  } catch (err) {
    toast('检索失败', String(err.message || err), 'warn');
  } finally {
    btn.classList.remove('is-busy');
  }
}

/* ---------------------------------------------------------------- 追溯 */

async function loadTraces() {
  try {
    const data = await apiGet('/api/traces');
    state.traces = data.items || [];
    const host = $('#trace-list');
    clearWithExit(host, '.trace-row');
    if (!state.traces.length) {
      setTimeout(() => {
        const empty = document.createElement('div');
        empty.className = 'panel empty-panel trace-row';
        empty.innerHTML = `${icon('history', 'ico ico-lg')}<p>还没有分析记录</p><em>分析台跑一张图 这里就有了</em>`;
        host.appendChild(empty);
      }, REDUCED ? 0 : 140);
      return;
    }
    setTimeout(() => {
      state.traces.forEach((t, i) => {
        const el = document.createElement('div');
        el.className = 'trace-row';
        el.style.animationDelay = REDUCED ? '0ms' : `${i * 55}ms`;
        el.innerHTML = `
          <div>
            <div class="trace-id"></div>
            <div class="trace-meta"></div>
          </div>
          <div class="row-actions">
            <button class="btn btn-sm" data-fmt="html">${icon('file-text')}<span>HTML</span></button>
            <button class="btn btn-sm" data-fmt="pdf">${icon('download')}<span>PDF</span></button>
          </div>`;
        $('.trace-id', el).textContent = `${t.trace_id} · ${t.image_id}`;
        $('.trace-meta', el).innerHTML = `
          <span>${t.created_at}</span><span>${sceneLabel(t.scene)}</span>
          <span>检出 ${t.n_detections}</span><span>工单 ${t.n_orders}</span>
          <span>待人工 ${t.n_human_review}</span><span>最高 ${t.max_severity}</span>
          <span>${fmtMs(t.total_ms)}ms</span>`;
        $('.row-actions', el).addEventListener('click', (ev) => {
          const btn = ev.target.closest('button[data-fmt]');
          if (btn) window.open(`/api/report/${t.trace_id}?fmt=${btn.dataset.fmt}`, '_blank');
        });
        host.appendChild(el);
      });
    }, REDUCED ? 0 : 140);
  } catch (err) {
    toast('追溯读取失败', String(err.message || err), 'warn');
  }
}

/* ---------------------------------------------------------------- 自检 */

async function runSelfcheck() {
  const btn = $('#btn-selfcheck');
  btn.classList.add('is-busy');
  btn.disabled = true;
  toast('开始自检', '配置 → 建流水线 → 知识库 → 跑图 → 出报告', 'info', 2600);
  try {
    const res = await apiPost('/api/selfcheck');
    const bad = (res.steps || []).filter((s) => !s.ok);
    if (res.ok) {
      toast('自检通过', `${res.steps.length} 步全通 · 总耗时 ${fmtMs(res.total_ms)}ms`, 'ok', 5200);
    } else {
      toast('自检未通过', bad.map((s) => s.step).join('、') || '未知步骤失败', 'warn', 6000);
    }
    await loadHealth();
    await loadTraces();
  } catch (err) {
    toast('自检请求失败', String(err.message || err), 'warn');
  } finally {
    btn.classList.remove('is-busy');
    btn.disabled = false;
  }
}

/* ---------------------------------------------------------------- 上传 */

async function uploadFile(file) {
  const drop = $('#upload-drop');
  drop.classList.add('is-busy');
  const form = new FormData();
  form.append('file', file);
  try {
    const res = await fetch('/api/analyze/upload', { method: 'POST', body: form });
    if (!res.ok) throw new Error(`${res.status} ${await shortError(res)}`);
    const result = await res.json();
    resetPipeline();
    await playPipeline(result.timings || []);
    showResult(result);
    await loadOrders();
    toast('上传分析完成', `${result.diagnoses.length} 条诊断 · ${result.work_orders.length} 张工单`, 'ok');
  } catch (err) {
    toast('上传分析失败', String(err.message || err), 'warn');
  } finally {
    drop.classList.remove('is-busy');
  }
}

/* ---------------------------------------------------------------- 事件绑定 */

function bind() {
  $$('.tab').forEach((t) => t.addEventListener('click', () => switchView(t.dataset.view)));

  $('#btn-selfcheck').addEventListener('click', runSelfcheck);
  $('#btn-refresh-assets').addEventListener('click', loadAssets);
  $('#btn-refresh-traces').addEventListener('click', loadTraces);
  $('#modal-close').addEventListener('click', closeModal);
  $('#modal').addEventListener('click', (e) => { if (e.target.id === 'modal') closeModal(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });

  $('#btn-report-pdf').addEventListener('click', () => {
    if (state.result) window.open(`/api/report/${state.result.trace_id}?fmt=pdf`, '_blank');
  });

  $('#search-form').addEventListener('submit', (e) => {
    e.preventDefault();
    doSearch($('#search-input').value);
  });
  $$('.hint').forEach((h) => h.addEventListener('click', () => {
    $('#search-input').value = h.dataset.q;
    doSearch(h.dataset.q);
  }));

  $('#order-filter').addEventListener('click', (e) => {
    const seg = e.target.closest('.seg');
    if (!seg) return;
    $$('.seg').forEach((s) => s.classList.toggle('is-active', s === seg));
    state.orderFilter = seg.dataset.filter;
    renderOrders();
  });

  const drop = $('#upload-drop');
  const input = $('#file-input');
  input.addEventListener('change', () => { if (input.files[0]) uploadFile(input.files[0]); });
  ['dragenter', 'dragover'].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('is-over'); }));
  ['dragleave', 'drop'].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('is-over'); }));
  drop.addEventListener('drop', (e) => {
    const f = e.dataTransfer.files[0];
    if (f) uploadFile(f);
  });

  window.addEventListener('resize', moveInk);
  window.addEventListener('scroll', () => {
    $('#topbar').classList.toggle('is-scrolled', window.scrollY > 6);
  }, { passive: true });
}

/* ---------------------------------------------------------------- 启动 */

async function boot() {
  bind();
  const params = new URLSearchParams(location.search);
  const hash = location.hash.replace('#', '');
  const wantView = params.get('view') || hash;
  if (['orders', 'kb', 'traces'].includes(wantView)) switchView(wantView);
  requestAnimationFrame(moveInk);
  await Promise.all([loadHealth(), loadKbStats(), loadAssets(), loadOrders()]);
  await loadTraces();
  const t = $('#tab-ink');
  if (t) t.style.opacity = '1';

  /* 深链：?image=demo_tower_01 直接跑一张图 ?view=kb&q=绝缘子自爆 直接检索
     演示与录屏用得上 也方便把某次结果作为链接发给别人 */
  const autoQuery = params.get('q');
  if (autoQuery) {
    if (state.view !== 'kb') switchView('kb');
    $('#search-input').value = autoQuery;
    doSearch(autoQuery);
  }
  const autoImage = params.get('image');
  if (autoImage) {
    if (state.view !== 'console') switchView('console');
    const asset = state.assets.find((a) => a.image_id === autoImage);
    if (asset) analyzeAsset(asset);
    else toast('没找到这张演示图', autoImage, 'warn');
  }
}

boot();
