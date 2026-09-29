// 대화 기록 검토 화면 (web/review.html) - 2026-09-29
//
// 목록(필터·검색) -> 한 건 클릭 -> 상세 패널에서 질문·답변·근거 공고를 보고 "좋음 / 나쁨" + 메모를 남김.
// 판정은 TB_CHAT_REVIEW에 저장되고(src/api/chat_log.py), 원본 대화 기록(TB_CHAT_LOG)은 건드리지 않습니다.
// 답변 표시는 챗봇 화면과 같은 renderAnswer를 씀 - XSS 방어(모든 글자를 먼저 이스케이프)가 테스트된 함수라서.

import { errorMessage, escapeHtml, formatDuration, renderAnswer } from "./chat-format.js";

const PAGE_SIZE = 30;

// 상태 이름과 "왜 이 상태가 됐는지" 설명. 검토하는 사람이 코드(pipeline.py)를 몰라도 판단할 수 있게.
const STATUS = {
  ok: { label: "정상 답변", explain: "근거 공고를 [번호]로 인용한 답변이 나갔어요. 인용한 공고가 질문에 맞는지, 기간·금액이 원문과 같은지 확인해 주세요." },
  no_evidence: { label: "근거 없음", explain: "관련 공고를 못 찾아 답하지 않았어요. 정말 데이터에 없는 질문인지(예: 개인 복지), 아니면 검색이 놓친 건지 봐 주세요." },
  ungrounded: { label: "규칙 위반", explain: "LLM이 답을 만들었지만 출처 번호가 없어서 가드레일이 버렸어요. 원래 답은 저장되지 않아요." },
  llm_error: { label: "LLM 오류", explain: "답변 생성 중 오류·시간 초과가 나서 공고 목록만 보여줬어요. 서버 로그(logs/backend.log)를 같이 보세요." },
  llm_unavailable: { label: "LLM 꺼짐", explain: "LLM이 꺼져 있어 공고 목록만 보여줬어요." },
  unknown: { label: "알 수 없음", explain: "상태가 기록되지 않은 옛 대화예요." },
};
const FAIL_STATUSES = ["ungrounded", "llm_error", "llm_unavailable"];

const state = { offset: 0, total: 0, items: new Map(), current: null };
const $ = (s) => document.querySelector(s);

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(path, { headers: options.body ? { "Content-Type": "application/json" } : undefined, ...options });
  } catch {
    throw new Error(errorMessage(0));
  }
  if (response.status === 204) return null;
  const data = await response.json().catch(() => null);
  if (!response.ok) throw new Error(errorMessage(response.status, data));
  return data;
}

const statusKey = (s) => (STATUS[s] ? s : "unknown");
const when = (iso) => new Date(iso).toLocaleString("ko-KR", { month: "numeric", day: "numeric", hour: "numeric", minute: "2-digit" });
const pct = (n, d) => (d ? `${Math.round((n / d) * 100)}%` : "-");

// ── 요약 ────────────────────────────────────────────────────────────
async function loadSummary() {
  const s = await api("/review/summary");
  const count = (k) => s.by_status[k] || 0;
  const fails = FAIL_STATUSES.reduce((sum, k) => sum + count(k), 0);
  $("#t-total").textContent = s.total.toLocaleString("ko-KR");
  // "2026-09-28" -> "09-28" (타일 폭에 한 줄로)
  $("#t-days").textContent = s.by_day.length ? `${s.by_day[0].date.slice(5)} ~ ${s.by_day.at(-1).date.slice(5)}` : "아직 대화 없음";
  $("#t-ok").textContent = count("ok");
  $("#t-ok-rate").textContent = `전체의 ${pct(count("ok"), s.total)}`;
  $("#t-noev").textContent = count("no_evidence");
  $("#t-fail").textContent = fails;
  $("#t-reviewed").textContent = `${s.reviewed} / ${s.total}`;
  $("#t-verdicts").textContent = `좋음 ${s.good} · 나쁨 ${s.bad}`;
  $("#t-avg").textContent = s.avg_response_ms == null ? "-" : formatDuration(s.avg_response_ms);
  $("#t-p95").textContent = s.p95_response_ms == null ? " " : `느린 5% 경계 ${formatDuration(s.p95_response_ms)}`;

  const order = ["ok", "no_evidence", "ungrounded", "llm_error", "llm_unavailable", "unknown"];
  const present = order.filter((k) => count(k));
  $("#status-bar").replaceChildren(...present.map((k) => {
    const seg = document.createElement("span");
    seg.className = `s-${k}`;
    seg.style.width = `${(count(k) / s.total) * 100}%`;
    seg.title = `${STATUS[k].label} ${count(k)}건`;
    return seg;
  }));
  $("#status-legend").innerHTML = present
    .map((k) => `<span><i class="s-${k}"></i>${STATUS[k].label} ${count(k)}건 (${pct(count(k), s.total)})</span>`).join("");
}

// ── 목록 ────────────────────────────────────────────────────────────
function filters() {
  const params = new URLSearchParams({ limit: PAGE_SIZE, offset: state.offset });
  for (const [key, el] of [["status", "#f-status"], ["verdict", "#f-verdict"], ["q", "#f-query"]]) {
    const value = $(el).value.trim();
    if (value) params.set(key, value);
  }
  return params;
}

function verdictChip(verdict) {
  const [cls, label] = verdict === "good" ? ["good", "좋음"] : verdict === "bad" ? ["bad", "나쁨"] : ["none", "판정 전"];
  return `<span class="verdict-chip ${cls}">${label}</span>`;
}

function rowHtml(item) {
  const key = statusKey(item.status);
  return `<span class="when">${escapeHtml(when(item.created_at))}</span>
    <span class="status-chip ${key}">${STATUS[key].label}</span>
    <span class="q">${escapeHtml(item.question)}</span>
    ${verdictChip(item.verdict)}
    <span class="ms">${escapeHtml(formatDuration(item.response_time_ms))}</span>`;
}

async function loadList({ append = false } = {}) {
  if (!append) state.offset = 0;
  const list = $("#chat-list");
  let data;
  try {
    data = await api(`/review/chats?${filters()}`);
  } catch (e) {
    list.innerHTML = `<div class="empty">${escapeHtml(e.message)}</div>`;
    return;
  }
  state.total = data.total;
  if (!append) { list.replaceChildren(); state.items.clear(); }
  for (const item of data.items) {
    state.items.set(item.chat_id, item);
    const row = document.createElement("button");
    row.type = "button";
    row.className = "chat-row";
    row.setAttribute("role", "listitem");
    row.dataset.id = item.chat_id;
    row.innerHTML = rowHtml(item);
    row.addEventListener("click", () => openDetail(item.chat_id));
    list.append(row);
  }
  state.offset += data.items.length;
  if (!state.total) list.innerHTML = '<div class="empty">조건에 맞는 대화가 없어요.</div>';
  $("#list-count").textContent = state.total ? `${state.total}건 중 ${state.offset}건 표시` : "";
  $("#more").hidden = state.offset >= state.total;
}

// ── 상세 + 판정 ─────────────────────────────────────────────────────
async function openDetail(chatId) {
  const item = state.items.get(chatId);
  state.current = chatId;
  const key = statusKey(item.status);
  const body = $("#detail-body");
  body.innerHTML = `
    <div class="detail-meta"><span class="status-chip ${key}">${STATUS[key].label}</span>
      <span>#${item.chat_id}</span><span>${escapeHtml(when(item.created_at))}</span>
      <span>${escapeHtml(formatDuration(item.response_time_ms))}</span></div>
    <div class="detail-block"><h4>질문</h4><div class="qbox"></div></div>
    <div class="detail-block"><h4>답변</h4><div class="abox">${key === "ok" ? renderAnswer(item.answer) : `<p>${escapeHtml(item.answer)}</p>`}</div></div>
    <div class="detail-block"><p class="explain">${STATUS[key].explain}</p></div>
    <div class="detail-block"><h4>${key === "ok" ? "인용한 공고" : "함께 보여준 공고"} (${item.cited_program_ids.length})</h4>
      <ul class="sources" id="detail-sources">${item.cited_program_ids.length ? "" : '<li class="muted">없음</li>'}</ul></div>
    <div class="detail-block"><h4>판정</h4>
      <div class="verdict-buttons">
        <button type="button" class="button secondary good" data-verdict="good" aria-pressed="${item.verdict === "good"}">좋음</button>
        <button type="button" class="button secondary bad" data-verdict="bad" aria-pressed="${item.verdict === "bad"}">나쁨</button>
      </div>
      <textarea id="note" class="note" maxlength="500" placeholder="메모 (예: 공고는 맞는데 접수기간이 원문과 다름)"></textarea>
      <div class="save-row"><span class="muted" id="save-state">${item.reviewed_at ? `판정 ${escapeHtml(when(item.reviewed_at))}` : "아직 판정하지 않았어요"}</span>
        <button type="button" class="button secondary" id="clear" ${item.verdict ? "" : "hidden"}>판정 취소</button>
        <button type="button" class="button primary" id="save">저장</button></div>
    </div>`;
  body.querySelector(".qbox").textContent = item.question;
  body.querySelector("#note").value = item.note;
  let chosen = item.verdict;
  body.querySelectorAll("[data-verdict]").forEach((b) => b.addEventListener("click", () => {
    chosen = b.dataset.verdict;
    body.querySelectorAll("[data-verdict]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  }));
  body.querySelector("#save").addEventListener("click", () => saveVerdict(chatId, chosen));
  body.querySelector("#clear").addEventListener("click", () => clearVerdict(chatId));
  $("#scrim").hidden = false;
  $("#detail").hidden = false;
  body.querySelector("[data-verdict]").focus();
  loadSources(item.cited_program_ids);
}

async function loadSources(ids) {
  // 대화 기록에는 공고 ID만 저장돼 있어서(TB_CHAT_LOG) 이름·링크는 /programs/{id}로 다시 조회
  const list = $("#detail-sources");
  for (const id of ids) {
    const li = document.createElement("li");
    li.textContent = id;
    list.append(li);
    try {
      const p = await api(`/programs/${encodeURIComponent(id)}?max_chunks=1`);
      li.innerHTML = `<span>${escapeHtml(p.program_name)} <span class="muted">· ${escapeHtml(p.apply_period)}</span></span>
        <a target="_blank" rel="noopener noreferrer">원문 ↗</a>`;
      li.querySelector("a").href = p.source_url;
    } catch {
      li.innerHTML = `<span>${escapeHtml(id)} <span class="muted">· 색인에서 찾을 수 없음</span></span>`;
    }
  }
}

async function saveVerdict(chatId, verdict) {
  const stateEl = $("#save-state");
  if (!verdict) { stateEl.textContent = "좋음 또는 나쁨을 먼저 골라 주세요."; return; }
  try {
    const updated = await api(`/review/chats/${chatId}`, {
      method: "PUT", body: JSON.stringify({ verdict, note: $("#note").value }),
    });
    state.items.set(chatId, updated);
    stateEl.textContent = "저장했어요.";
    $("#clear").hidden = false;
    refreshRow(chatId);
    loadSummary();
  } catch (e) {
    stateEl.textContent = e.message;
  }
}

async function clearVerdict(chatId) {
  try {
    await api(`/review/chats/${chatId}`, { method: "DELETE" });
    const item = { ...state.items.get(chatId), verdict: null, note: "", reviewed_at: null };
    state.items.set(chatId, item);
    closeDetail();
    refreshRow(chatId);
    loadSummary();
  } catch (e) {
    $("#save-state").textContent = e.message;
  }
}

function refreshRow(chatId) {
  const row = document.querySelector(`.chat-row[data-id="${chatId}"]`);
  if (row) row.innerHTML = rowHtml(state.items.get(chatId));
}

function closeDetail() {
  $("#scrim").hidden = true;
  $("#detail").hidden = true;
  const row = document.querySelector(`.chat-row[data-id="${state.current}"]`);
  row?.focus();
}

// ── 시작 ────────────────────────────────────────────────────────────
let searchTimer;
$("#f-status").addEventListener("change", () => loadList());
$("#f-verdict").addEventListener("change", () => loadList());
$("#f-query").addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => loadList(), 300); });
$("#more").addEventListener("click", () => loadList({ append: true }));
$("#scrim").addEventListener("click", closeDetail);
document.querySelector("[data-close]").addEventListener("click", closeDetail);
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("#detail").hidden) closeDetail(); });

loadSummary().catch((e) => { $("#summary-note").textContent = e.message; });
loadList();
