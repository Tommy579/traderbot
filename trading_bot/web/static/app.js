/* Logique du tableau de bord TradingBot.
   Extraite de index.html ; chargée avec defer, donc le DOM est prêt. */
"use strict";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function toast(text, kind = 'info') {
  const container = $('#toast-container');
  const el = document.createElement('div');
  el.className = 'toast ' + kind;
  el.textContent = text;
  container.appendChild(el);
  setTimeout(() => {
    el.style.opacity = '0';
    el.style.transform = 'translateX(20px)';
    el.style.transition = 'all 0.3s ease';
    setTimeout(() => el.remove(), 300);
  }, 4000);
}

function setLoading(btnId, isLoading, text = null) {
  const btn = $('#' + btnId);
  if (!btn) return;
  btn.disabled = isLoading;
  if (text && isLoading) {
    btn.dataset.oldText = btn.textContent;
    btn.textContent = text;
  } else if (btn.dataset.oldText) {
    btn.textContent = btn.dataset.oldText;
    delete btn.dataset.oldText;
  }
}

const STRATEGIES = {
  donchian_breakout: {
    name: "Cassure de tendance", code: "donchian_breakout",
    desc: "Achète quand une action franchit son plus haut des 20 derniers jours, seulement si le marché va bien. Vends quand elle casse son plus bas des 10 derniers jours. La méthode des traders « Tortue »."
  },
  relative_momentum: {
    name: "Force relative", code: "relative_momentum",
    desc: "N'achète que les actions qui montent plus fort que le marché (comparé à SPY), et les lâche quand elles faiblissent."
  },
  ema_cross: {
    name: "Croisement de moyennes", code: "ema_cross",
    desc: "La méthode d'origine du robot : achète quand la tendance courte dépasse la tendance longue. Simple mais réagit tard."
  },
  shield_sword: {
    name: "Bouclier + Épée", code: "shield_sword",
    desc: "80 % en actifs ultra-défensifs (cash/safe haven) pour sanctuariser le capital, 20 % sur le Top 3 des actions en plus forte tendance avec rebalancement mensuel (20 jours)."
  }
};
function esc(v) {
  if (v == null) return "";
  return String(v)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
const money = (v) => (v == null) ? "—" :
  new Intl.NumberFormat("fr-FR", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(v);
const money2 = (v) => (v == null) ? "—" :
  new Intl.NumberFormat("fr-FR", { style: "currency", currency: "USD" }).format(v);
const pct = (v) => (v == null) ? "—" : ((v > 0 ? "+" : "") + (v * 100).toFixed(2).replace(".", ",") + " %");
const cls = (v) => v > 0 ? "pos" : v < 0 ? "neg" : "";
const frDate = (iso) => {
  if (!iso) return "—";
  const parts = String(iso).split("T")[0].split("-");
  if (parts.length === 3) {
    const d = new Date(parseInt(parts[0], 10), parseInt(parts[1], 10) - 1, parseInt(parts[2], 10));
    return d.toLocaleDateString("fr-FR", { day: "numeric", month: "short", year: "numeric" });
  }
  const d = new Date(iso);
  return isNaN(d) ? String(iso) : d.toLocaleDateString("fr-FR", { day: "numeric", month: "short", year: "numeric" });
};
/** « 6 nov. » : jour et mois, sans l'année (synthèse compacte). */
const frDayMonth = (iso) => {
  if (!iso) return "—";
  const d = new Date(String(iso).slice(0, 10) + "T00:00:00");
  return isNaN(d) ? String(iso) : d.toLocaleDateString("fr-FR", { day: "numeric", month: "short" });
};
/** « 9 nov. · 16:00 » : échéance courte d'un réveil planifié. */
const frWhen = (iso) => {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d)) return String(iso);
  const day = d.toLocaleDateString("fr-FR", { day: "numeric", month: "short" });
  const time = d.toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" });
  return `${day} · ${time}`;
};
const frDateTime = (iso) => {
  if (!iso) return "bientôt";
  const d = new Date(iso);
  return isNaN(d) ? String(iso) : d.toLocaleString("fr-FR");
};

function msg(id, text, kind, { silent = false } = {}) {
  if (!silent && (kind === 'ok' || kind === 'err')) {
    toast(text, kind);
  }
  const el = $("#" + id);
  if (!el) return;
  el.className = "msg " + kind;
  el.textContent = text;
}
async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body)
  });
  const data = await res.json().catch(() => ({ ok: false, error: "Réponse illisible du serveur." }));
  if (!res.ok || data.ok === false) throw new Error(data.error || "Erreur serveur.");
  return data;
}
async function pollJob(jobId, onTick) {
  for (;;) {
    const snap = await api("/api/job/" + jobId);
    if (snap.status === "failed") throw new Error(snap.error || "La tâche a échoué.");
    if (snap.status === "done") return snap.result;
    if (onTick) onTick();
    await new Promise(r => setTimeout(r, 900));
  }
}

/* ---------------- Onglets ---------------- */
const TAB_BUTTONS = Array.from(document.querySelectorAll('nav [role="tab"]'));

function activateTab(btn, { focus = false } = {}) {
  if (!btn) return;
  const panelId = btn.dataset.tab;
  for (const other of TAB_BUTTONS) {
    const selected = other === btn;
    other.classList.toggle("active", selected);
    other.setAttribute("aria-selected", selected ? "true" : "false");
    // Un seul onglet reste dans l'ordre de tabulation : les flèches font le reste.
    other.tabIndex = selected ? 0 : -1;
  }
  document.querySelectorAll("section[role='tabpanel']").forEach(section => {
    section.classList.toggle("active", section.id === panelId);
  });
  if (panelId === "history") loadHistory(true);
  if (focus) btn.focus();
}

TAB_BUTTONS.forEach((btn, index) => {
  btn.addEventListener("click", () => activateTab(btn));
  btn.addEventListener("keydown", event => {
    const last = TAB_BUTTONS.length - 1;
    let target = null;
    if (event.key === "ArrowRight") target = TAB_BUTTONS[index === last ? 0 : index + 1];
    else if (event.key === "ArrowLeft") target = TAB_BUTTONS[index === 0 ? last : index - 1];
    else if (event.key === "Home") target = TAB_BUTTONS[0];
    else if (event.key === "End") target = TAB_BUTTONS[last];
    if (target) {
      event.preventDefault();
      activateTab(target, { focus: true });
    }
  });
});

/* ---------------- Sélecteurs de stratégie ---------------- */
for (const id of ["#check-strategy", "#auto-strategy"]) {
  $(id).innerHTML = Object.values(STRATEGIES).map(s =>
    `<option value="${s.code}">${esc(s.name)}</option>`).join("");
  $(id).value = "donchian_breakout";
}
$("#bt-strategies").innerHTML = Object.values(STRATEGIES).map(s =>
  `<button type="button" role="radio" aria-checked="false" class="strategy-card" data-code="${s.code}"><b>${esc(s.name)}</b><span>${esc(s.desc)}</span></button>`).join("");
let btStrategy = "donchian_breakout";

/** Sélectionne une méthode de backtest (clic, clavier, ou état initial). */
function selectBacktestStrategy(card, { focus = false } = {}) {
  document.querySelectorAll(".strategy-card").forEach(c => {
    const selected = c === card;
    c.classList.toggle("selected", selected);
    c.setAttribute("aria-checked", selected ? "true" : "false");
    c.tabIndex = selected ? 0 : -1;
  });
  btStrategy = card.dataset.code;
  if (focus) card.focus();
}

document.querySelectorAll(".strategy-card").forEach(card => {
  card.tabIndex = -1;
  if (card.dataset.code === btStrategy) selectBacktestStrategy(card);
  card.addEventListener("click", () => selectBacktestStrategy(card));
  card.addEventListener("keydown", event => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      selectBacktestStrategy(card);
    }
  });
});

/* ---------------- Tableau de bord ---------------- */
function renderHalt(state) {
  const el = $("#halt-banner");
  el.innerHTML = "";
  if (!state || !state.halted) return;
  const permanent = state.halt_scope === "permanent";
  el.innerHTML = `<div class="banner">
    <b>Coupe-circuit déclenché — le robot s'est mis en sécurité.</b><br>
    ${esc(state.halt_reason || "")}
    ${permanent ? `<div class="actions"><button class="secondary" id="btn-halt-reset">
      J'ai vérifié, relancer le robot</button></div>` :
    `<div class="small" style="margin-top:6px">Pause automatique : il reprendra tout seul au prochain jour de marché.</div>`}
  </div>`;
  if (permanent) {
    $("#btn-halt-reset").addEventListener("click", async () => {
      const reason = prompt("En une phrase : qu'avez-vous vérifié avant de relancer ?");
      if (!reason) return;
      try {
        const out = await api("/api/halt/reset", { confirm: true, reason });
        msg("dashboard-msg", out.message, "ok");
        refreshStatus();
      } catch (e) { msg("dashboard-msg", e.message, "err"); }
    });
  }
}
function renderPositions(rows, equity = 0) {
  const el = $("#positions");
  if (!rows.length) { el.innerHTML = `<p class="skel">Aucune position pour l'instant. Le robot n'achète que lorsque sa méthode lui donne un signal d'achat.</p>`; return; }
  el.innerHTML = `<div class="table-wrap"><table><thead><tr>
    <th scope="col">Action</th><th class="num" scope="col">Quantité</th><th class="num" scope="col">Prix d'achat</th><th class="num" scope="col">Prix actuel</th>
    <th class="num" scope="col">Gains / pertes</th><th class="num" scope="col">Filet (stop)</th><th scope="col">Statut</th></tr></thead><tbody>
    ${rows.map(p => {
      const weightPct = equity > 0 && p.value ? ((p.value / equity) * 100).toFixed(1).replace(".", ",") + " % du compte" : "";
      const stopDist = (p.stop_price && p.current_price > 0)
        ? `<br><small class="small">${(((p.stop_price - p.current_price) / p.current_price) * 100).toFixed(1).replace(".", ",")} %</small>`
        : "";
      // Une position sans stop n'est pas un détail d'affichage : sans ordre chez
      // le courtier, rien ne la protège si le marché plonge sans le robot.
      const bare = !p.is_shield && !p.stop_price;
      const statut = p.is_shield
        ? `<span class="badge ok">Bouclier</span>`
        : bare
          ? `<span class="badge warn">Sans filet</span>`
          : (p.partial_tp_taken ? `<span class="badge ok">TP 50 % pris (Break-Even)</span>` : `<span class="badge">Actif</span>`);
      return `<tr><td><b>${esc(p.symbol)}</b>${weightPct ? `<br><small class="small">${weightPct}</small>` : ""}</td>
      <td class="num">${p.qty}<br><small class="small">${money(p.value)}</small></td><td class="num">${money2(p.entry_price)}</td><td class="num">${money2(p.current_price)}</td>
      <td class="num ${cls(p.pnl)}">${money2(p.pnl)}<br><small>${pct(p.pnl_pct)}</small></td>
      <td class="num">${p.is_shield ? "—<br><small>sans stop</small>" : (p.stop_price ? money2(p.stop_price) + stopDist : "<span class=\"neg\">aucun</span>")}</td>
      <td>${statut}</td></tr>`;
    }).join("")}
    </tbody></table></div>`;
}
function renderAllocation(account, positions, settings) {
  const el = $("#allocation-bar");
  if (!el) return;
  if (!account || !account.equity || account.equity <= 0) {
    el.innerHTML = "";
    return;
  }
  const equity = account.equity;
  const invested = Math.max(0, (positions || []).reduce((acc, p) => acc + (p.value || 0), 0));
  const floor = Math.min(account.cash || 0, Math.max(0, settings?.withdrawable_cash_floor_usd || 0));
  const freeCash = Math.max(0, (account.cash || 0) - floor);
  const total = Math.max(equity, invested + freeCash + floor, 1);

  const wInvested = Math.min(100, (invested / total) * 100);
  const wFree = Math.min(100 - wInvested, (freeCash / total) * 100);
  const wFloor = Math.max(0, Math.min(100 - wInvested - wFree, (floor / total) * 100));

  const hardFloor = settings?.equity_hard_floor_usd || 0;
  const hardFloorNote = hardFloor > 0
    ? `<span>Seuil coupe-circuit global : <b>${money(hardFloor)}</b></span>`
    : "";

  el.innerHTML = `<div class="alloc-wrap">
    <div class="alloc-header">
      <span><b>Répartition du portefeuille</b></span>
      ${hardFloorNote}
    </div>
    <div class="alloc-track" role="img" aria-label="Répartition du capital entre actions, cash libre et cash sanctuarisé">
      <div class="alloc-seg invested" style="width:${wInvested.toFixed(1)}%" title="Actions : ${money(invested)}"></div>
      <div class="alloc-seg free-cash" style="width:${wFree.toFixed(1)}%" title="Cash libre : ${money(freeCash)}"></div>
      ${floor > 0 ? `<div class="alloc-seg floor-cash" style="width:${wFloor.toFixed(1)}%" title="Cash sanctuarisé : ${money(floor)}"></div>` : ""}
    </div>
    <div class="alloc-legend">
      <span><i class="alloc-dot invested"></i>Actions : <b>${money(invested)}</b> (${wInvested.toFixed(0)} %)</span>
      <span><i class="alloc-dot free-cash"></i>Cash libre : <b>${money(freeCash)}</b> (${wFree.toFixed(0)} %)</span>
      ${floor > 0 ? `<span><i class="alloc-dot floor-cash"></i>Cash sanctuarisé : <b>${money(floor)}</b> (${wFloor.toFixed(0)} %)</span>` : ""}
    </div>
  </div>`;
}
function renderAutomation(a) {
  if (!a) return;
  const nextText = a.next_run ? frWhen(a.next_run) : "séance US ouvrée";
  const stratText = STRATEGIES[a.strategy]?.name || a.strategy;
  const timerBadge = a.timer_active ? `<span class="badge ok">Cycle complet (15h55 NY / 21h55 Paris)</span>` : `<span class="badge warn">Cycle complet inactif</span>`;
  const watchBadge = a.watch_timer_active ? `<span class="badge ok">Vigie horaire (10h-15h NY / 16h-21h Paris)</span>` : `<span class="badge warn">Vigie inactif</span>`;
  const content = (a.timer_active || a.watch_timer_active)
    ? `<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:6px">${timerBadge} ${watchBadge}</div>Prochain réveil : <b>${esc(nextText)}</b> — méthode : <b>${esc(stratText)}</b>`
    : `<span class="badge warn">Inactif</span> Le robot ne se réveille pas tout seul. Ouvrez l'onglet « Piloter » pour l'activer.`;
  $("#automation-mini").innerHTML = content;
  $("#automation-state").innerHTML = content;
}
let lastKnownInitialEquity = 100000;
let refreshInFlight = false;

const FORM_FIELDS = [
  "#set-symbols",
  "#bt-symbols",
  "#set-webhook-format",
  "#set-webhook-chat",
  "#set-cash-floor",
  "#set-equity-floor",
  "#auto-strategy"
];
for (const sel of FORM_FIELDS) {
  const el = $(sel);
  if (!el) continue;
  const markDirty = () => { el.dataset.dirty = "1"; };
  el.addEventListener("input", markDirty);
  el.addEventListener("change", markDirty);
}
function setFieldIfClean(sel, val) {
  const el = $(sel);
  if (!el) return;
  if (document.activeElement === el || el.dataset.dirty === "1") return;
  el.value = val;
}
function clearDirtyFields(selectors) {
  for (const sel of selectors) {
    const el = $(sel);
    if (el) delete el.dataset.dirty;
  }
}

function setPill(id, text, state) {
  const el = $(id);
  if (!el) return;
  el.textContent = text;
  el.dataset.state = state;
}

/** Synthèse en tête de page : l'état du robot en quatre réponses. */
function renderSummary(s) {
  const state = s.state || {};
  const automation = s.automation || {};

  if (state.halted) {
    const permanent = state.halt_scope === "permanent";
    setPill("#s-robot", permanent ? "Arrêté (coupe-circuit)" : "En pause aujourd'hui", "bad");
  } else if (automation.timer_active || automation.watch_timer_active) {
    setPill("#s-robot", "Actif", "ok");
  } else if (automation.timer_installed) {
    setPill("#s-robot", "Installé mais inactif", "warn");
  } else {
    setPill("#s-robot", "Manuel (pas de réveil auto)", "warn");
  }

  if (s.market_open === true) setPill("#s-market", "Ouvert", "ok");
  else if (s.market_open === false) setPill("#s-market", "Fermé", "warn");
  else setPill("#s-market", "Indisponible", "warn");

  const last = s.last_cycle;
  if (last) {
    const when = last.timestamp || last.recorded_at;
    const day = frDayMonth(last.session_date || when);
    const time = when ? new Date(when).toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" }) : "";
    const label = last.mode === "watch_only" ? "vigie" : "complet";
    setPill("#s-last", `${day}${time ? " · " + time : ""} (${label})`, "ok");
  } else {
    setPill("#s-last", "Aucun cycle enregistré", "warn");
  }

  if (automation.next_run) {
    setPill("#s-next", frWhen(automation.next_run), "ok");
  } else if (automation.timer_active || automation.watch_timer_active) {
    setPill("#s-next", "séance US ouvrée", "ok");
  } else {
    setPill("#s-next", "Non planifié", "warn");
  }

  const hint = $("#summary-hint");
  if (hint) {
    if (s.error) {
      hint.textContent = `Compte indisponible : ${s.error}`;
    } else {
      const positions = s.positions || [];
      const pnl = positions.reduce((acc, p) => acc + (p.pnl || 0), 0);
      hint.textContent = `${positions.length} position(s) ouverte(s), gain/perte latent ${money2(pnl)}. Détails ci-dessous.`;
    }
  }
}

async function refreshStatus({ silent = false } = {}) {
  if (refreshInFlight) return;
  refreshInFlight = true;
  try {
    const s = await api("/api/status");
    if (s.error) {
      msg("dashboard-msg", s.error, "err", { silent });
      $("#market-state").textContent = "indisponible (erreur de connexion)";
    } else {
      const dm = $("#dashboard-msg");
      if (dm) dm.style.display = "none";
      $("#market-state").textContent = s.market_open ? "OUVERT — le robot peut agir" : "fermé — le robot attend";
    }
    if (s.state) {
      if (s.state.day_start_equity && s.state.day_start_equity > 0) {
        lastKnownInitialEquity = s.state.day_start_equity;
      } else if (s.state.peak_equity && s.state.peak_equity > 0) {
        lastKnownInitialEquity = s.state.peak_equity;
      }
    }
    if (s.account) {
      if (!s.state?.day_start_equity && s.account.equity > 0) {
        lastKnownInitialEquity = s.account.equity;
      }
      $("#t-equity").textContent = money(s.account.equity);
      const cashText = money(s.account.cash);
      const floor = s.settings && s.settings.withdrawable_cash_floor_usd ? s.settings.withdrawable_cash_floor_usd : 0;
      $("#t-cash").innerHTML = floor > 0
        ? `${cashText}<br><span class="small hint" style="font-size:11px;font-weight:normal">dont ${money(floor)} sanctuarisé</span>`
        : cashText;
      $("#t-count").textContent = s.positions.length;
      const pnl = s.positions.reduce((acc, p) => acc + p.pnl, 0);
      const t = $("#t-pnl");
      t.textContent = (pnl > 0 ? "+" : "") + money(pnl);
      t.className = "value " + cls(pnl);
    }
    renderHalt(s.state);
    renderPositions(s.positions, s.account ? s.account.equity : 0);
    renderAllocation(s.account, s.positions, s.settings);
    renderAutomation(s.automation);
    renderSummary(s);

    const selWithdraw = $("#withdraw-symbol");
    if (selWithdraw && document.activeElement !== selWithdraw) {
      const prevVal = selWithdraw.value;
      if (!s.positions || !s.positions.length) {
        selWithdraw.innerHTML = `<option value="">(aucune position ouverte)</option>`;
        selWithdraw.disabled = true;
      } else {
        selWithdraw.disabled = false;
        selWithdraw.innerHTML = `<option value="">Sélectionner une action…</option>` +
          s.positions.map(p => `<option value="${esc(p.symbol)}">${esc(p.symbol)} (${p.qty} actions ~ ${money2(p.value)})</option>`).join("");
        if (prevVal && s.positions.some(p => p.symbol === prevVal)) {
          selWithdraw.value = prevVal;
        }
      }
    }

    if (s.automation && s.automation.strategy && STRATEGIES[s.automation.strategy]) {
      setFieldIfClean("#auto-strategy", s.automation.strategy);
    }
    if (s.settings) {
      setFieldIfClean("#set-symbols", s.settings.symbols.join(", "));
      setFieldIfClean("#bt-symbols", s.settings.symbols.join(", "));
      setFieldIfClean("#set-webhook-format", s.settings.webhook_format || "discord");
      if (s.settings.webhook_chat_id) {
        setFieldIfClean("#set-webhook-chat", s.settings.webhook_chat_id);
      }
      setFieldIfClean("#set-cash-floor", s.settings.withdrawable_cash_floor_usd || "");
      setFieldIfClean("#set-equity-floor", s.settings.equity_hard_floor_usd || "");
      if ($("#s-cash-floor-display")) $("#s-cash-floor-display").textContent = money(s.settings.withdrawable_cash_floor_usd || 0);
      if ($("#s-equity-floor-display")) $("#s-equity-floor-display").textContent = money(s.settings.equity_hard_floor_usd || 0);
      const badge = $("#webhook-badge");
      if (badge) {
        if (s.settings.webhook_enabled) {
          badge.className = "badge ok";
          badge.textContent = "Actif";
          $("#set-webhook-url").placeholder = "(enregistré — laissez vide pour désactiver)";
        } else {
          badge.className = "badge warn";
          badge.textContent = "Inactif";
          $("#set-webhook-url").placeholder = "https://discord.com/api/webhooks/…";
        }
      }
      const tokenField = $("#set-webhook-token");
      if (tokenField) {
        tokenField.placeholder = s.settings.webhook_token_set
          ? "jeton enregistré — laissez vide pour le conserver"
          : "obligatoire si l'adresse n'est ni Discord ni Telegram";
      }
      $("#s-daily").textContent = Math.round(s.settings.daily_loss_limit_pct * 100);
      $("#s-dd").textContent = Math.round(s.settings.drawdown_limit_pct * 100);
    }
  } catch (e) {
    msg("dashboard-msg", e.message, "err", { silent });
  } finally {
    refreshInFlight = false;
  }
}

/* ---------------- Historique ---------------- */
const HISTORY_SOURCE = {
  timer:      { label: "Réveil automatique", cls: "ok" },
  gui:        { label: "Lancé depuis cette page", cls: "warn" },
  cli:        { label: "Ligne de commande", cls: "hold" },
  watch_only: { label: "Vigie horaire", cls: "hold" },
};
const SIGNAL_CLS = { BUY: "buy", SELL: "sell", HOLD: "hold" };
const sideLabel = (s) => s === "buy" ? "achat" : s === "sell" ? "vente" : s;

async function loadHistory(force) {
  const el = $("#history-list");
  if (!force && el.dataset.loaded === "1") return;
  el.innerHTML = `<div class="card"><p class="skel">Chargement de l'historique…</p></div>`;
  try {
    const data = await api("/api/history");
    renderHistory(data.entries || []);
  } catch (e) {
    el.innerHTML = `<div class="card"><p class="skel">${esc(e.message)}</p></div>`;
  }
}

function renderHistory(entries) {
  drawHistoryChart(entries);
  const el = $("#history-list");
  el.dataset.loaded = "1";
  if (!entries.length) {
    el.innerHTML = `<div class="card"><p class="skel">Aucun cycle enregistré pour l'instant. L'historique se remplit au premier cycle : bouton « Cycle complet (compte fictif) », ou réveil automatique du prochain jour de marché.</p></div>`;
    return;
  }

  // Regroupement par séance de marché (du plus récent au plus ancien)
  const sessions = new Map();
  for (const entry of entries) {
    const sDate = entry.session_date || "date-inconnue";
    if (!sessions.has(sDate)) sessions.set(sDate, []);
    sessions.get(sDate).push(entry);
  }

  el.innerHTML = Array.from(sessions.entries()).map(([sessionDate, dayEntries]) => {
    // Au sein de la journée, tri chronologique de l'ouverture à la clôture
    const sortedDay = dayEntries.slice().sort((a, b) => {
      const ta = a.timestamp || a.recorded_at || "";
      const tb = b.timestamp || b.recorded_at || "";
      return ta.localeCompare(tb);
    });

    const vigies = sortedDay.filter(e => e.mode === "watch_only").length;
    const complets = sortedDay.length - vigies;
    const summaryBadges = [];
    if (vigies > 0) summaryBadges.push(`<span class="badge hold">${vigies} vigie(s)</span>`);
    if (complets > 0) summaryBadges.push(`<span class="badge ok">${complets} complet</span>`);

    const latestCycle = sortedDay[sortedDay.length - 1];
    if (latestCycle.equity != null) {
      summaryBadges.push(`<span class="badge ok">Solde : ${money2(latestCycle.equity)}</span>`);
    }

    const subdivisions = sortedDay.map(entry => {
      const isWatch = entry.mode === "watch_only";
      const src = HISTORY_SOURCE[entry.source] || { label: entry.source || "origine inconnue", cls: "hold" };
      const timeRaw = entry.timestamp || entry.recorded_at;
      const timeStr = timeRaw ? new Date(timeRaw).toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" }) : "";

      const badges = [
        `<span class="badge ${isWatch ? "hold" : "ok"}">${isWatch ? "Vigie horaire" : "Cycle complet"}</span>`,
        `<span class="badge ${src.cls}">${esc(src.label)}</span>`
      ];
      if (entry.dry_run) badges.push(`<span class="badge hold">Répétition — aucun ordre</span>`);
      if (entry.skipped) badges.push(`<span class="badge hold">Ignoré</span>`);

      const signals = Object.entries(entry.signals || {});
      const orders = entry.orders || [];
      const approved = (entry.decisions || []).filter(d => d.approved);
      const halts = entry.halts || [];
      const errors = entry.errors || [];
      const lines = [];

      if (entry.skipped) {
        lines.push(`<p class="hint">Ignoré : ${esc(entry.skip_reason || "séance non ouverte")}.</p>`);
      } else {
        const hasAccountDetails = entry.cash != null || entry.positions_count != null || entry.unrealized_pnl != null;
        if (hasAccountDetails) {
          const pnl = entry.unrealized_pnl || 0;
          lines.push(`<div class="tiles" style="margin-bottom:10px">
            <div class="tile"><div class="label">Valeur totale</div><div class="value">${money2(entry.equity)}</div></div>
            <div class="tile"><div class="label">Cash disponible</div><div class="value">${money2(entry.cash)}</div></div>
            <div class="tile"><div class="label">Actions détenues</div><div class="value">${entry.positions_count ?? 0}</div></div>
            <div class="tile"><div class="label">Gains / pertes latents</div><div class="value ${cls(pnl)}">${(pnl > 0 ? "+" : "") + money2(pnl)}</div></div>
          </div>`);
        } else {
          lines.push(`<p class="hint">Valeur du compte : <b>${money2(entry.equity)}</b>.</p>`);
        }

        if (signals.length) {
          lines.push(`<p class="small"><b>Signaux lus :</b> ${signals.map(([sym, act]) =>
            `<span class="badge ${SIGNAL_CLS[act] || "hold"}">${esc(sym)} : ${esc(act)}</span>`).join(" ")}</p>`);
        }
        if (orders.length) {
          lines.push(`<p class="small"><b>Ordres transmis :</b></p><ul class="small">${orders.map(o =>
            `<li>${esc(sideLabel(o.side))} de ${esc(String(o.qty))} ${esc(o.symbol)} — statut : ${esc(o.status)}${o.avg_fill_price ? ` à ${money2(o.avg_fill_price)}` : ""}</li>`).join("")}</ul>`);
        } else if (isWatch) {
          lines.push(`<p class="small" style="color:var(--muted)">Surveillance en direct : stops ATR et alertes d'actualité vérifiés, aucune coupure nécessaire.</p>`);
        } else {
          lines.push(`<p class="small" style="color:var(--muted)">Aucun ordre transmis : la méthode n'a rien demandé ce jour-là.</p>`);
        }
        if (approved.length) {
          lines.push(`<p class="small"><b>Décisions validées :</b></p><ul class="small">${approved.map(d =>
            `<li>${esc(sideLabel(d.side))} de ${esc(String(d.qty))} ${esc(d.symbol)}${d.stop_price ? ` — filet à ${money2(d.stop_price)}` : ""} (${esc(d.reason)})</li>`).join("")}</ul>`);
        }
      }
      if (halts.length) {
        lines.push(`<div class="banner"><b>Coupe-circuit :</b><ul class="small" style="margin:6px 0 0">${halts.map(h =>
          `<li>${esc(h.scope === "permanent" ? "arrêt complet" : "pause du jour")} — ${esc(h.reason)}</li>`).join("")}</ul></div>`);
      }
      if (errors.length) {
        lines.push(`<p class="small" style="color:var(--bad)"><b>Alertes :</b></p><ul class="small">${errors.map(e => `<li>${esc(e)}</li>`).join("")}</ul>`);
      }

      return `<div class="cycle-subdivision">
        <div class="cycle-subdivision-header">
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
            <span class="cycle-time">${esc(timeStr || "—")}</span>
            ${badges.join("")}
          </div>
          <div style="font-weight:600;font-size:13.5px">${money2(entry.equity)}</div>
        </div>
        ${lines.join("")}
      </div>`;
    }).join("");

    return `<div class="card">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:10px;flex-wrap:wrap;margin-bottom:6px">
        <h2 style="margin:0">${frDate(sessionDate)}</h2>
        <div style="display:flex;gap:6px;flex-wrap:wrap">${summaryBadges.join("")}</div>
      </div>
      ${subdivisions}
    </div>`;
  }).join("");
}

function drawHistoryChart(entries) {
  const wrap = $("#history-svg");
  const card = $("#history-chart-card");
  const tip = $("#history-chart-tip");
  if (!wrap || !card || !tip) return;

  if (!entries || !entries.length) {
    card.style.display = "none";
    return;
  }

  const sorted = entries.slice().sort((a, b) => {
    const ta = a.timestamp || a.session_date || "";
    const tb = b.timestamp || b.session_date || "";
    return ta.localeCompare(tb);
  });

  const firstWithEquity = sorted.find(e => e.equity && e.equity > 0);
  let lastEquity = firstWithEquity ? firstWithEquity.equity : lastKnownInitialEquity;
  const points = [];
  for (const e of sorted) {
    if (e.equity && e.equity > 0) {
      lastEquity = e.equity;
    }
    if (e.session_date) {
      const tRaw = e.timestamp || e.recorded_at;
      const tStr = tRaw ? new Date(tRaw).toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" }) : "";
      points.push({ date: e.session_date, time: tStr, equity: lastEquity });
    }
  }

  if (!points.length) {
    card.style.display = "none";
    return;
  }

  if (points.length === 1) {
    const d = new Date(points[0].date + "T00:00:00");
    d.setDate(d.getDate() - 1);
    points.unshift({ date: d.toISOString().split("T")[0], equity: points[0].equity || lastKnownInitialEquity });
  }

  card.style.display = "block";
  const initial = points[0].equity || lastKnownInitialEquity;

  const W = 760, H = 260, padL = 62, padR = 16, padT = 14, padB = 30;
  const values = points.map(p => p.equity);
  let vmin = Math.min(...values, initial), vmax = Math.max(...values, initial);
  const span = vmax - vmin || 100;
  vmin -= span * 0.08; vmax += span * 0.08;

  const x = i => padL + (i / (points.length - 1)) * (W - padL - padR);
  const y = v => padT + (1 - (v - vmin) / (vmax - vmin)) * (H - padT - padB);
  const chartAria = `du ${frDate(points[0].date)} (${money2(points[0].equity)}) au ${frDate(points[points.length - 1].date)} (${money2(points[points.length - 1].equity)})`;
  const path = points.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.equity).toFixed(1)}`).join("");

  const gridY = [0, 0.25, 0.5, 0.75, 1].map(f => {
    const gy = padT + f * (H - padT - padB), gv = vmax - f * (vmax - vmin);
    return `<line x1="${padL}" y1="${gy}" x2="${W - padR}" y2="${gy}" stroke="var(--card-border)" stroke-width="1"/>
      <text x="${padL - 8}" y="${gy + 4}" text-anchor="end" font-size="11" fill="var(--muted)">${money(gv)}</text>`;
  }).join("");

  const step = Math.max(1, Math.floor(points.length / 6));
  const dates = points.map((p, i) => {
    if (i % step !== 0 && i !== points.length - 1) return "";
    const dt = new Date(p.date + "T00:00:00");
    const label = dt.toLocaleDateString("fr-FR", { day: "numeric", month: "short" });
    return `<text x="${x(i)}" y="${H - 8}" text-anchor="middle" font-size="11" fill="var(--muted)">${label}</text>`;
  }).join("");
  const baseY = y(initial);
  const areaPath = `${path} L${x(points.length - 1).toFixed(1)},${(H - padB).toFixed(1)} L${x(0).toFixed(1)},${(H - padB).toFixed(1)} Z`;

  wrap.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="display:block" role="img" aria-label="Courbe de la valeur du compte : ${esc(chartAria)}">
    <defs>
      <linearGradient id="hist-grad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="var(--series-1)" stop-opacity="0.22"/>
        <stop offset="100%" stop-color="var(--series-1)" stop-opacity="0.0"/>
      </linearGradient>
    </defs>
    ${gridY}${dates}
    <path d="${areaPath}" fill="url(#hist-grad)"/>
    <line x1="${padL}" y1="${baseY}" x2="${W - padR}" y2="${baseY}" stroke="var(--muted)" stroke-width="1" stroke-dasharray="4 4" opacity="0.7"/>
    <text x="${W - padR}" y="${baseY - 5}" text-anchor="end" font-size="10.5" fill="var(--muted)">départ ${money(initial)}</text>
    <path d="${path}" fill="none" stroke="var(--series-1)" stroke-width="2.5" stroke-linejoin="round"/>
    <line id="hist-cross" x1="0" y1="${padT}" x2="0" y2="${H - padB}" stroke="var(--muted)" stroke-width="1" visibility="hidden"/>
    <circle id="hist-dot" r="4.5" fill="var(--series-1)" stroke="var(--surface-1)" stroke-width="2" visibility="hidden"/>
  </svg>`;

  const svg = wrap.querySelector("svg");
  function onMove(clientX) {
    const rect = svg.getBoundingClientRect();
    if (!rect.width) return;
    const fx = ((clientX - rect.left) / rect.width) * W;
    const i = Math.round(Math.min(points.length - 1, Math.max(0, ((fx - padL) / (W - padL - padR)) * (points.length - 1))));
    const p = points[i];
    if (!p) return;
    const cross = svg.querySelector("#hist-cross"), dot = svg.querySelector("#hist-dot");
    cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
    dot.setAttribute("cx", x(i)); dot.setAttribute("cy", y(p.equity)); dot.setAttribute("visibility", "visible");
    tip.style.display = "block";
    const diff = p.equity - initial;
    const pctDiff = initial > 0 ? (diff / initial) : 0;
    tip.innerHTML = `<b>${frDate(p.date)}${p.time ? ` à ${p.time}` : ""}</b><br>Valeur : ${money2(p.equity)}<br>Évolution : ${(diff >= 0 ? "+" : "") + money2(diff)} (${pct(pctDiff)})`;
    const px = (x(i) / W) * rect.width;
    tip.style.left = Math.min(px + 12, Math.max(8, rect.width - 200)) + "px";
    tip.style.top = "6px";
  }
  function onLeave() {
    tip.style.display = "none";
    const cross = svg.querySelector("#hist-cross"), dot = svg.querySelector("#hist-dot");
    if (cross) cross.setAttribute("visibility", "hidden");
    if (dot) dot.setAttribute("visibility", "hidden");
  }
  svg.addEventListener("mousemove", ev => onMove(ev.clientX));
  svg.addEventListener("mouseleave", onLeave);
  svg.addEventListener("touchstart", ev => {
    if (ev.touches && ev.touches.length > 0) onMove(ev.touches[0].clientX);
  });
}

$("#btn-history-refresh").addEventListener("click", () => loadHistory(true));

/* ---------------- Diagnostic ---------------- */
$("#btn-check").addEventListener("click", async () => {
  const btn = $("#btn-check");
  setLoading("btn-check", true, "Analyse...");
  msg("check-msg", "Analyse en cours (quelques secondes)…", "info");
  try {
    const { job } = await api("/api/check", { strategy: $("#check-strategy").value });
    const result = await pollJob(job.id, () => msg("check-msg", "Analyse en cours (quelques secondes)…", "info"));
    const badge = { BUY: "buy", SELL: "sell", HOLD: "hold" };
    const label = { BUY: "ACHETER", SELL: "VENDRE", HOLD: "NE RIEN FAIRE" };
    /* Note d'actualités : c'est elle qui décide si un achat passera la nuit
       venue. Afficher le signal brut sans elle promettrait un achat qui
       n'aura pas lieu. */
    const newsCell = (n) => {
      if (!n) return `<td class="num small">—</td>`;
      const cls = n.blocks_entry ? "err" : (n.score <= -0.3 ? "warn" : "ok");
      const sign = n.score >= 0 ? "+" : "";
      const flag = n.critical ? `<br><span class="small">menace critique</span>` : "";
      return `<td class="num small"><span class="badge ${cls}">${sign}${n.score.toFixed(2)}</span>${flag}</td>`;
    };
    $("#check-result").innerHTML = `
      <p class="small" style="margin-top:14px">Décision proposée pour chaque action suivie, actualités prises en compte :</p>
      <div class="table-wrap"><table><thead><tr><th scope="col">Action</th><th scope="col">Décision</th><th scope="col">Actualités</th><th scope="col">Prix actuel</th><th scope="col">Filet proposé</th><th scope="col">Pourquoi</th></tr></thead><tbody>
      ${result.signals.map(r => {
        const blocked = !!(r.news && r.news.blocks_entry && r.action === "BUY");
        const cls = blocked ? "hold" : (badge[r.action] || "hold");
        const txt = blocked ? "ACHAT REFUSÉ (actualité)" : (label[r.action] || esc(r.action));
        const why = blocked && r.news ? `${esc(r.news.effect)} — ${esc(r.news.rationale)}` : esc(r.reason);
        return `<tr><td><b>${esc(r.symbol)}</b></td>
        <td><span class="badge ${cls}">${txt}</span></td>
        ${newsCell(r.news)}
        <td class="num">${money2(r.price)}</td>
        <td class="num">${r.proposed_stop ? money2(r.proposed_stop) : "—"}</td>
        <td class="small">${why}</td></tr>`;
      }).join("")}
      </tbody></table></div>
      ${result.news && result.news.consulted
        ? `<p class="small">Filtre d'actualités : ${esc(result.news.model)} · veto d'achat ≤ ${result.news.veto_threshold} · sortie d'urgence ≤ ${result.news.panic_threshold}</p>`
        : `<p class="small">Filtre d'actualités inactif (aucune analyse) : la décision ci-dessus ne tient compte que de la stratégie.</p>`}
      ${(result.errors || []).length ? `<div class="banner"><b>Symboles écartés :</b><ul class="small" style="margin:6px 0 0">${(result.errors || []).map(x => `<li>${esc(x)}</li>`).join("")}</ul></div>` : ""}`;
    msg("check-msg", "Diagnostic terminé. Rien n'a été acheté ni vendu.", "ok");
  } catch (e) { msg("check-msg", e.message, "err"); }
  setLoading("btn-check", false);
});

/* ---------------- Backtest ---------------- */
$("#btn-backtest").addEventListener("click", async () => {
  setLoading("btn-backtest", true, "Simulation…");
  $("#bt-result").innerHTML = "";
  msg("bt-msg", "Test en cours : téléchargement des prix puis simulation, environ 30 secondes…", "info");
  const symbols = $("#bt-symbols").value.split(",").map(s => s.trim()).filter(Boolean);
  try {
    const { job } = await api("/api/backtest", {
      strategy: btStrategy, symbols, start: $("#bt-start").value
    });
    const r = await pollJob(job.id, () => msg("bt-msg", "Simulation en cours…", "info"));
    const m = r.metrics;
    const gain = m.total_return, dd = m.max_drawdown;
    $("#bt-result").innerHTML = `
      <div class="card">
        <h2>Résultat — ${esc(STRATEGIES[r.strategy]?.name || r.strategy)} sur ${esc(r.symbols.join(", "))}</h2>
        <div class="tiles">
          <div class="tile"><div class="label">Gain total</div><div class="value ${cls(gain)}">${pct(gain)}<br><small>soit ${money2(m.final_equity - m.initial_equity)}</small></div></div>
          <div class="tile"><div class="label">Pire chute encaissée</div><div class="value neg">${(dd * 100).toFixed(1).replace(".", ",")} %</div></div>
          <div class="tile"><div class="label">Gains par an (moyenne)</div><div class="value ${cls(m.cagr)}">${pct(m.cagr)}</div></div>
          <div class="tile"><div class="label">Paris gagnants</div><div class="value">${m.win_rate ? (m.win_rate * 100).toFixed(0) + " %" : "—"}</div></div>
          <div class="tile"><div class="label">Frais payés</div><div class="value">${money2(m.total_fees)}</div></div>
          <div class="tile"><div class="label">Temps investi</div><div class="value">${(m.exposure_pct * 100).toFixed(0)} %<br><small>du temps en marché</small></div></div>
        </div>
        <p class="hint" style="margin-top:12px">Référence de marché (SPY acheté et conservé sur la même période) :
          <b>${pct(m.benchmark_buy_hold)}</b> — c'est elle qu'il faut battre. Panier équipondéré des actions testées,
          acheté et conservé : <b>${pct(m.basket_buy_hold)}</b>.</p>
      </div>
      <div class="card">
        <h2>Valeur du compte dans le temps</h2>
        <p class="hint">Survolez la courbe : date, valeur, et chute à ce moment-là.</p>
        <div class="chart-wrap" id="chart-wrap">
          <div class="chart-tip" id="chart-tip"></div>
          <div id="chart-svg"></div>
        </div>
      </div>
      ${r.trades.length ? `<div class="card">
        <h2>Les ${r.trades.length} allers-retours simulés (les 10 derniers)</h2>
        <div class="table-wrap"><table><thead><tr><th scope="col">Action</th><th scope="col">Acheté le</th><th scope="col">Vendu le</th><th scope="col">Résultat</th><th scope="col">Motif de sortie</th></tr></thead><tbody>
        ${r.trades.slice(-10).map(t => `<tr><td><b>${esc(t.symbol)}</b></td><td>${frDate(t.entry_date)}</td><td>${frDate(t.exit_date)}</td>
          <td class="num ${cls(t.pnl)}">${money2(t.pnl)} <small>(${pct(t.return_pct)})</small></td>
          <td class="small">${esc({ stop_loss: "filet de sécurité", trailing_stop: "filet remonté", strategy_exit: "règle de sortie", circuit_breaker: "coupe-circuit", stop_gap: "trou d'ouverture sous le filet" }[t.reason] || t.reason)}</td></tr>`).join("")}
        </tbody></table></div></div>` : ""}
      ${r.halts.length ? `<div class="banner"><b>Coupe-circuits déclenchés pendant le test :</b><ul class="small" style="margin:6px 0 0">
        ${r.halts.map(h => `<li>${frDate(h.date)} — ${esc(h.reason)}</li>`).join("")}</ul></div>` : ""}
    `;
    drawEquityChart(r.equity_points, m.initial_equity);
    msg("bt-msg", "Test terminé.", "ok");
  } catch (e) { msg("bt-msg", e.message, "err"); }
  setLoading("btn-backtest", false);
});

function drawEquityChart(points, initial) {
  const W = 760, H = 280, padL = 62, padR = 16, padT = 14, padB = 30;
  const wrap = $("#chart-svg");
  if (!points || points.length < 2) { wrap.innerHTML = "<p class='skel'>Pas assez de données.</p>"; return; }
  const values = points.map(p => p.equity);
  let vmin = Math.min(...values, initial), vmax = Math.max(...values, initial);
  const span = vmax - vmin || 1;
  vmin -= span * 0.05; vmax += span * 0.05;
  const x = i => padL + (i / (points.length - 1)) * (W - padL - padR);
  const y = v => padT + (1 - (v - vmin) / (vmax - vmin)) * (H - padT - padB);
  const chartAria = `${money2(points[0].equity)} → ${money2(points[points.length - 1].equity)}`;
  const path = points.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.equity).toFixed(1)}`).join("");
  const gridY = [0, 0.25, 0.5, 0.75, 1].map(f => {
    const gy = padT + f * (H - padT - padB), gv = vmax - f * (vmax - vmin);
    return `<line x1="${padL}" y1="${gy}" x2="${W - padR}" y2="${gy}" stroke="var(--card-border)" stroke-width="1"/>
      <text x="${padL - 8}" y="${gy + 4}" text-anchor="end" font-size="11" fill="var(--muted)">${money(gv)}</text>`;
  }).join("");
  const step = Math.max(1, Math.floor(points.length / 6));
  const dates = points.map((p, i) => i % step === 0 || i === points.length - 1
    ? `<text x="${x(i)}" y="${H - 8}" text-anchor="middle" font-size="11" fill="var(--muted)">${new Date(p.date).toLocaleDateString("fr-FR", { month: "short", year: "2-digit" })}</text>` : "").join("");
  const baseY = y(initial);
  const areaPath = `${path} L${x(points.length - 1).toFixed(1)},${(H - padB).toFixed(1)} L${x(0).toFixed(1)},${(H - padB).toFixed(1)} Z`;
  wrap.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="display:block" role="img" aria-label="Courbe de performance du test : ${esc(chartAria)}">
    <defs>
      <linearGradient id="bt-grad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="var(--series-1)" stop-opacity="0.22"/>
        <stop offset="100%" stop-color="var(--series-1)" stop-opacity="0.0"/>
      </linearGradient>
    </defs>
    ${gridY}${dates}
    <path d="${areaPath}" fill="url(#bt-grad)"/>
    <line x1="${padL}" y1="${baseY}" x2="${W - padR}" y2="${baseY}" stroke="var(--muted)" stroke-width="1" stroke-dasharray="4 4" opacity="0.7"/>
    <text x="${W - padR}" y="${baseY - 5}" text-anchor="end" font-size="10.5" fill="var(--muted)">départ ${money(initial)}</text>
    <path d="${path}" fill="none" stroke="var(--series-1)" stroke-width="2.5" stroke-linejoin="round"/>
    <line id="ch-cross" x1="0" y1="${padT}" x2="0" y2="${H - padB}" stroke="var(--muted)" stroke-width="1" visibility="hidden"/>
    <circle id="ch-dot" r="4.5" fill="var(--series-1)" stroke="var(--surface-1)" stroke-width="2" visibility="hidden"/>
  </svg>`;
  const svg = wrap.querySelector("svg");
  const tip = $("#chart-tip");
  function onMove(clientX) {
    const rect = svg.getBoundingClientRect();
    if (!rect.width) return;
    const fx = ((clientX - rect.left) / rect.width) * W;
    const i = Math.round(Math.min(points.length - 1, Math.max(0, ((fx - padL) / (W - padL - padR)) * (points.length - 1))));
    const p = points[i];
    if (!p) return;
    const cross = svg.querySelector("#ch-cross"), dot = svg.querySelector("#ch-dot");
    cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
    dot.setAttribute("cx", x(i)); dot.setAttribute("cy", y(p.equity)); dot.setAttribute("visibility", "visible");
    tip.style.display = "block";
    tip.innerHTML = `<b>${frDate(p.date)}</b><br>Valeur : ${money2(p.equity)}<br>Chute depuis le sommet : ${(p.drawdown_pct * 100).toFixed(1).replace(".", ",")} %`;
    const px = (x(i) / W) * rect.width;
    tip.style.left = Math.min(px + 12, Math.max(8, rect.width - 180)) + "px";
    tip.style.top = "6px";
  }
  function onLeave() {
    tip.style.display = "none";
    const cross = svg.querySelector("#ch-cross"), dot = svg.querySelector("#ch-dot");
    if (cross) cross.setAttribute("visibility", "hidden");
    if (dot) dot.setAttribute("visibility", "hidden");
  }
  svg.addEventListener("mousemove", ev => onMove(ev.clientX));
  svg.addEventListener("mouseleave", onLeave);
  svg.addEventListener("touchstart", ev => {
    if (ev.touches && ev.touches.length > 0) onMove(ev.touches[0].clientX);
  }, { passive: true });
  svg.addEventListener("touchmove", ev => {
    if (ev.touches && ev.touches.length > 0) onMove(ev.touches[0].clientX);
  }, { passive: true });
  svg.addEventListener("touchend", onLeave);
}

/* ---------------- Pilotage ---------------- */
$("#btn-automation").addEventListener("click", async () => {
  setLoading("btn-automation", true, "Activation…");
  msg("automation-msg", "Installation en cours…", "info");
  try {
    const out = await api("/api/automation", { strategy: $("#auto-strategy").value });
    clearDirtyFields(["#auto-strategy"]);
    msg("automation-msg", out.message, "ok");
    refreshStatus();
  } catch (e) { msg("automation-msg", e.message, "err"); }
  setLoading("btn-automation", false);
});
function renderCycle(r) {
  const label = { buy: "achat", sell: "vente" };
  const modeLabel = r.mode === "watch_only" ? "vigie horaire" : "cycle complet";
  const orders = (r.orders || []).map(o => `<li>${label[o.side] || esc(o.side)} de ${o.qty} ${esc(o.symbol)} — statut : ${esc(o.status)}</li>`).join("");
  const dryOrders = (r.decisions || []).filter(d => d.approved && d.symbol).map(d =>
    `<li>Ordre simulé : <b>${label[d.side] || esc(d.side)}</b> de ${d.qty} <b>${esc(d.symbol)}</b> (motif : ${esc(d.reason)})</li>`
  ).join("");
  const errors = (r.errors || []).map(e => `<li>${esc(e)}</li>`).join("");
  return `<div class="card" style="margin-top:12px">
    <h2>Cycle ${r.skipped ? "ignoré" : "terminé"} (${modeLabel}) — ${frDate(r.session_date)}</h2>
    ${r.skipped ? `<p class="hint">${esc(r.skip_reason)}.</p>` : `
    <p class="hint">Valeur du compte après le cycle : <b>${money2(r.equity)}</b>.</p>
    ${r.dry_run ? (dryOrders ? `<p class="small"><b>Ordres qui auraient été transmis :</b></p><ul class="small">${dryOrders}</ul>` : "<p class='small'>Répétition générale : aucun ordre d'achat ou de vente n'aurait été nécessaire aujourd'hui.</p>") :
      (orders ? `<p class="small"><b>Ordres transmis :</b></p><ul class="small">${orders}</ul>` : `<p class='small'>${r.mode === "watch_only" ? "Vigie horaire : positions saines, aucun ordre de coupure nécessaire." : "Aucun ordre nécessaire aujourd'hui."}</p>`)}
    ${errors ? `<p class="small" style="color:var(--bad)"><b>Alertes :</b></p><ul class="small">${errors}</ul>` : ""}`}
  </div>`;
}
async function runCycle({ dryRun = false, watchOnly = false } = {}) {
  const btnWatch = $("#btn-cycle-watch");
  const btnDry = $("#btn-cycle-dry");
  const btnLive = $("#btn-cycle-live");
  if (btnWatch) btnWatch.disabled = true;
  btnDry.disabled = true;
  btnLive.disabled = true;

  let msgText = "Cycle en cours…";
  if (watchOnly) msgText = "Vigie horaire en cours (surveillance stops & news)…";
  else if (dryRun) msgText = "Répétition générale en cours…";
  else msgText = "Cycle complet réel (compte fictif) en cours…";

  msg("cycle-msg", msgText, "info");
  try {
    const { job } = await api("/api/cycle", {
      strategy: $("#auto-strategy").value,
      dry_run: dryRun,
      watch_only: watchOnly,
    });
    const result = await pollJob(job.id);
    $("#cycle-result").innerHTML = renderCycle(result);
    msg("cycle-msg", dryRun ? "Répétition générale terminée : aucune somme engagée." : "Cycle terminé avec succès.", "ok");
    refreshStatus();
    loadHistory(true);
  } catch (e) { msg("cycle-msg", e.message, "err"); }
  finally {
    if (btnWatch) btnWatch.disabled = false;
    btnDry.disabled = false;
    btnLive.disabled = false;
  }
}
$("#btn-cycle-watch")?.addEventListener("click", () => runCycle({ watchOnly: true }));
$("#btn-cycle-dry").addEventListener("click", () => runCycle({ dryRun: true }));
$("#btn-cycle-live").addEventListener("click", () => {
  if (!confirm("Le robot va analyser le marché et placer de vrais ordres — sur votre compte D'ENTRAÎNEMENT uniquement (argent fictif). Confirmer ?")) return;
  runCycle({ dryRun: false });
});

/* ---------------- Réglages ---------------- */
const ASSET_LABEL = {
  us_equity: "action américaine",
  crypto: "crypto (non géré par ce bot)"
};

function renderResolved(rows, warning) {
  const el = $("#settings-resolved");
  if (!el) return;
  const note = warning
    ? `<p class="small" style="color:var(--bad);margin-top:8px">${esc(warning)}</p>`
    : "";
  if (!rows || !rows.length) {
    el.innerHTML = note;
    return;
  }
  el.innerHTML = `
    <p class="small" style="margin-top:12px"><b>Ce que le robot suivra réellement :</b></p>
    <div class="table-wrap"><table><thead><tr>
      <th scope="col">Symbole saisi</th><th scope="col">Instrument chez Alpaca</th><th scope="col">Type</th>
    </tr></thead><tbody>
      ${rows.map(r => `<tr><td><b>${esc(r.symbol)}</b></td>
        <td>${esc(r.name)}</td>
        <td class="small">${esc(ASSET_LABEL[r.asset_class] || r.asset_class)}</td></tr>`).join("")}
    </tbody></table></div>${note}`;
}

$("#btn-settings").addEventListener("click", async () => {
  setLoading("btn-settings", true, "Vérification...");
  msg("settings-msg", "Vérification des symboles auprès d'Alpaca…", "info");
  try {
    const body = {
      symbols: $("#set-symbols").value.split(",").map(s => s.trim()).filter(Boolean),
      webhook_format: $("#set-webhook-format").value,
      webhook_url: $("#set-webhook-url").value.trim(),
      webhook_chat_id: $("#set-webhook-chat").value.trim()
    };
    // Le jeton n'est transmis que s'il a été saisi : un champ vide conserve
    // celui déjà enregistré plutôt que de le supprimer.
    const token = $("#set-webhook-token").value.trim();
    if (token) body.webhook_token = token;
    const cashFloor = $("#set-cash-floor") ? $("#set-cash-floor").value.trim() : "";
    if (cashFloor !== "") body.withdrawable_cash_floor_usd = cashFloor;
    const eqFloor = $("#set-equity-floor") ? $("#set-equity-floor").value.trim() : "";
    if (eqFloor !== "") body.equity_hard_floor_usd = eqFloor;
    const out = await api("/api/config", body);
    clearDirtyFields([
      "#set-symbols",
      "#set-webhook-format",
      "#set-webhook-chat",
      "#set-cash-floor",
      "#set-equity-floor"
    ]);
    msg("settings-msg", out.message, "ok");
    $("#set-webhook-token").value = "";
    renderResolved(out.resolved, out.warning);
    refreshStatus();
  } catch (e) {
    msg("settings-msg", e.message, "err");
    renderResolved(null, null);
  }
  setLoading("btn-settings", false);
});

/* ---------------- Retraits et Sanctuarisation ---------------- */
function renderWithdrawResult(out) {
  const el = $("#withdraw-result");
  if (!el) return;
  if (!out.orders || !out.orders.length) {
    el.innerHTML = `<p class="small hint">Aucun ordre de vente nécessaire : le cash disponible couvrait déjà le montant demandé.</p>`;
    return;
  }
  el.innerHTML = `
    <p class="small" style="font-weight:600;margin-bottom:6px">Ordres exécutés via le moteur de risque :</p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th scope="col">Action</th><th class="num" scope="col">Quantité</th><th class="num" scope="col">Prix d'exécution</th><th class="num" scope="col">Cash libéré</th><th scope="col">Motif</th></tr>
        </thead>
        <tbody>
          ${out.orders.map(o => `<tr>
            <td><b>${esc(o.symbol)}</b></td>
            <td class="num">${o.qty}</td>
            <td class="num">${money2(o.price)}</td>
            <td class="num pos">+${money2(o.cash_freed)}</td>
            <td class="small">${esc(o.reason)}</td>
          </tr>`).join("")}
        </tbody>
      </table>
    </div>
  `;
}

$("#btn-withdraw-pos")?.addEventListener("click", async () => {
  const symbol = $("#withdraw-symbol").value;
  const amount = $("#withdraw-amount").value.trim();
  const pctVal = $("#withdraw-pct").value.trim();
  if (!symbol) {
    msg("withdraw-msg", "Veuillez sélectionner une action en portefeuille.", "err");
    return;
  }
  if (!amount && !pctVal) {
    msg("withdraw-msg", "Veuillez renseigner un montant ($) ou un pourcentage (%) à retirer.", "err");
    return;
  }
  setLoading("btn-withdraw-pos", true, "Exécution...");
  msg("withdraw-msg", "Validation du risque et transmission de l'ordre…", "info");
  try {
    const payload = { symbol, async: true };
    if (amount) payload.amount = parseFloat(amount);
    else if (pctVal) payload.pct = parseFloat(pctVal);
    const res = await api("/api/withdraw", payload);
    const out = res.job ? await pollJob(res.job.id) : res;
    msg("withdraw-msg", out.message, "ok");
    $("#withdraw-amount").value = "";
    $("#withdraw-pct").value = "";
    renderWithdrawResult(out);
    refreshStatus();
  } catch (e) {
    msg("withdraw-msg", e.message, "err");
  } finally {
    setLoading("btn-withdraw-pos", false);
  }
});

$("#btn-withdraw-free")?.addEventListener("click", async () => {
  const freeCash = $("#withdraw-free-cash").value.trim();
  if (!freeCash) {
    msg("withdraw-msg", "Veuillez indiquer un montant de cash cible à sanctuariser.", "err");
    return;
  }
  setLoading("btn-withdraw-free", true, "Exécution...");
  msg("withdraw-msg", "Vérification des poches défensives et gagnantes…", "info");
  try {
    const res = await api("/api/withdraw", { free_cash: parseFloat(freeCash), async: true });
    const out = res.job ? await pollJob(res.job.id) : res;
    clearDirtyFields(["#set-cash-floor"]);
    msg("withdraw-msg", out.message, "ok");
    $("#withdraw-free-cash").value = "";
    renderWithdrawResult(out);
    refreshStatus();
  } catch (e) {
    msg("withdraw-msg", e.message, "err");
  } finally {
    setLoading("btn-withdraw-free", false);
  }
});

/* Dépôt ou retrait fait directement chez Alpaca : recaler la référence d'équité. */
$("#btn-rebase")?.addEventListener("click", async () => {
  const reason = ($("#rebase-reason").value || "").trim();
  if (!reason) {
    msg("withdraw-msg", "Décrivez le mouvement de cash (ex. « retrait de 5 000 $ »).", "err");
    return;
  }
  if (!confirm("Recaler la référence d'équité sur la valeur actuelle ? Le pic précédent sera oublié.")) return;
  setLoading("btn-rebase", true, "Recalage...");
  try {
    const out = await api("/api/rebase", { reason });
    msg("withdraw-msg", out.message, "ok");
    $("#rebase-reason").value = "";
    refreshStatus();
  } catch (e) {
    msg("withdraw-msg", e.message, "err");
  } finally {
    setLoading("btn-rebase", false);
  }
});

$("#btn-refresh").addEventListener("click", () => {
  refreshStatus({ silent: false });
});

/* ---------------- Thème Clair / Sombre ---------------- */
const themeBtn = $("#toggle-theme");
function applyTheme(theme) {
  const root = document.documentElement;
  if (theme === "dark" || theme === "light") {
    root.dataset.theme = theme;
  } else {
    delete root.dataset.theme;
  }
  if (themeBtn) {
    const isDark = theme === "dark" || (!theme && window.matchMedia?.("(prefers-color-scheme: dark)").matches);
    themeBtn.textContent = isDark ? "☀" : "☾";
    themeBtn.title = isDark ? "Passer en thème clair" : "Passer en thème sombre";
  }
}
try {
  applyTheme(localStorage.getItem("tb-theme"));
} catch (_) {
  applyTheme(null);
}
themeBtn?.addEventListener("click", () => {
  const root = document.documentElement;
  const sysDark = window.matchMedia?.("(prefers-color-scheme: dark)").matches;
  const current = root.dataset.theme || (sysDark ? "dark" : "light");
  const next = current === "dark" ? "light" : "dark";
  try { localStorage.setItem("tb-theme", next); } catch (_) {}
  applyTheme(next);
});

let pollTimer = null;
let liveActive = false;
const liveBtn = $("#toggle-live");

function schedulePolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  const intervalMs = liveActive ? 5000 : 30000;
  pollTimer = setInterval(() => {
    if (!document.hidden) {
      refreshStatus({ silent: true });
    }
  }, intervalMs);
}

function setLive(active, { initial = false } = {}) {
  liveActive = Boolean(active);
  liveBtn.setAttribute("aria-checked", liveActive ? "true" : "false");
  if (liveActive) {
    liveBtn.classList.add("active");
  } else {
    liveBtn.classList.remove("active");
  }
  schedulePolling();
  if (!initial) {
    refreshStatus({ silent: false });
  }
}

liveBtn.addEventListener("click", () => {
  setLive(!liveActive);
});

document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    refreshStatus({ silent: true });
  }
});

refreshStatus({ silent: true });

/* ---------------- Lien direct vers un onglet (#pilot, #backtest&run=1…) ---------------- */
(function () {
  const parts = location.hash.replace("#", "").split("&");
  const tab = (parts[0] || "").replace("tab-", "");
  if (tab) {
    const btn = document.querySelector(`nav button[data-tab="${tab}"]`);
    if (btn) btn.click();
  }
  const params = new URLSearchParams(parts.slice(1).join("&"));
  if (params.get("check") === "1") document.querySelector("#btn-check").click();
  if (params.get("run") === "1") document.querySelector("#btn-backtest").click();
})();
setLive(true, { initial: true });
