// [WBS 9] 챗봇 웹 화면 - 동작
//
// FastAPI가 이 폴더(web/)를 "/"에서 그대로 서빙합니다(src/api/main.py). 같은 주소에서 API(/chat 등)를 부르므로
// CORS 설정이 필요 없습니다. 빌드 도구 없이 브라우저가 바로 읽는 ES 모듈이라, 파일을 고치고 새로고침하면 끝입니다.
//
// [Streamlit 화면(ui/app.py)과 다른 점]
//   Streamlit은 입력할 때마다 파이썬 스크립트 전체를 다시 실행해서 화면을 새로 그립니다. 그래서 "답변을 기다리는 동안
//   점 3개가 움직이는 말풍선"이나 "인용 번호를 누르면 카드로 이동" 같은 챗봇다운 동작을 넣기 어려웠습니다.
//   여기서는 필요한 부분만 DOM에 추가/수정합니다.

import { deadlineInfo, describeConditions, errorMessage, escapeHtml, formatDuration, groupSources, renderAnswer } from "./chat-format.js";

// 첫 화면 추천 질문. ui/app.py의 EXAMPLE_QUESTIONS와 같은 기준으로 고름 - 실제 LLM으로 정상 답변(ok)을 확인한 질문만.
const EXAMPLE_QUESTIONS = [
  "서울 소상공인 온라인 판로 지원사업 알려줘",
  "여성기업 대상 수출 지원사업 있어?",
  "인력 채용하면 받을 수 있는 지원금 알려줘",
  "AI 기술개발 지원사업 찾아줘",
];

// 백엔드의 최악 대기 시간(로컬 LLM 120초, src/rag/llm_client.py)보다 길게. ui/api_client.py CHAT_TIMEOUT_SEC와 같은 값.
const CHAT_TIMEOUT_MS = 150_000;
const DETAIL_FIRST_CHUNKS = 3;
const DETAIL_MORE_CHUNKS = 5;
const SESSION_KEY = "lgsrag.session";

// status별 안내 (src/rag/pipeline.py의 5가지 결과). ok가 아니면 답변 대신 이 문구를 말풍선으로 보여줍니다.
const STATUS_NOTICE = {
  no_evidence: { tone: "info", text: "관련 공고를 찾지 못했어요. 질문을 조금 바꾸거나 필터를 풀고 다시 물어봐 주세요. (기업 대상 지원사업만 안내하고 있어요)" },
  ungrounded: { tone: "warning", text: "근거를 확인할 수 없는 답변이라 보여드리지 않았어요. 질문을 조금 더 구체적으로 바꿔 주세요." },
  llm_unavailable: { tone: "warning", text: "지금은 답변 생성 기능이 꺼져 있어서, 질문과 관련된 공고만 보여드려요." },
  llm_error: { tone: "error", text: "답변을 만드는 중 문제가 생겨서, 질문과 관련된 공고만 보여드려요." },
};

// 기다리는 동안 바뀌는 안내. 로컬 LLM은 평균 6초, 모델이 GPU에서 내려간 뒤 첫 질문은 20~30초 걸려서
// 아무 말 없이 기다리게 하면 "멈췄나?" 싶어짐 - 지금 무엇을 하는지 단계별로 알려줌.
const WAITING_STEPS = [
  [0, "관련 공고를 찾고 있어요"],
  [2500, "공고 내용을 읽고 답변을 쓰고 있어요"],
  [12000, "조금만 기다려 주세요. 처음 질문은 답변 모델을 준비하느라 더 걸려요"],
];

const ICONS = {
  info: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/></svg>',
  warning: '<svg viewBox="0 0 24 24"><path d="M12 4l9 16H3z"/><path d="M12 10v4M12 17h.01"/></svg>',
  error: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M9 9l6 6M15 9l-6 6"/></svg>',
  external: '<svg viewBox="0 0 24 24"><path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/></svg>',
  close: '<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg>',
  bot: '<svg viewBox="0 0 24 24"><path d="M4 10l8-5 8 5M6 10v8M12 10v8M18 10v8M3 20h18"/></svg>',
};

// ── 상태 ───────────────────────────────────────────────────────────
const state = {
  sessionId: loadSession(),
  busy: false,
  filters: { region: "", target: "", categories: [] },
  filterOptions: { regions: [], targets: [], categories: [] },
  answerCount: 0,     // 답변마다 카드 id를 겹치지 않게 (같은 사업이 여러 답변에 나올 수 있음)
  detail: null,       // 상세 패널에 열린 사업 { programId, chunks }
};

const $ = (selector) => document.querySelector(selector);
const messagesEl = $("#messages");
const questionEl = $("#question");
const sendButton = $("#send-button");

// localStorage는 사생활 보호 모드 등에서 막혀 있을 수 있어서 항상 try로 감쌈 (막혀도 대화는 되고, 새로고침 복원만 안 됨)
function loadSession() {
  try { return localStorage.getItem(SESSION_KEY); } catch { return null; }
}
function saveSession(id) {
  try { id ? localStorage.setItem(SESSION_KEY, id) : localStorage.removeItem(SESSION_KEY); } catch { /* 무시 */ }
}

// ── API ────────────────────────────────────────────────────────────
class ApiError extends Error {}

async function api(path, { method = "GET", body, timeout = 10_000 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  let response;
  try {
    response = await fetch(path, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
  } catch (error) {
    throw new ApiError(error.name === "AbortError"
      ? "응답이 너무 오래 걸려서 중단했어요. 잠시 후 다시 시도해 주세요."
      : errorMessage(0));
  } finally {
    clearTimeout(timer);
  }
  const data = await response.json().catch(() => null);
  if (!response.ok) throw new ApiError(errorMessage(response.status, data));
  return data;
}

// ── 말풍선 ─────────────────────────────────────────────────────────
function timeLabel(date = new Date()) {
  return date.toLocaleTimeString("ko-KR", { hour: "numeric", minute: "2-digit" });
}

function scrollToBottom() {
  requestAnimationFrame(() => { messagesEl.scrollTop = messagesEl.scrollHeight; });
}

/** 사용자 말풍선. 질문은 textContent로 넣어서 HTML로 해석되지 않게 함. */
function addUserMessage(text, when) {
  const row = document.createElement("div");
  row.className = "row user";
  row.innerHTML = '<div class="stack"><div class="bubble"></div><span class="time"></span></div>';
  row.querySelector(".bubble").textContent = text;
  row.querySelector(".time").textContent = timeLabel(when);
  messagesEl.append(row);
  scrollToBottom();
}

/** 챗봇 말풍선 틀. stack 안에 말풍선·조건·카드를 차례로 붙임. */
function addBotRow() {
  const row = document.createElement("div");
  row.className = "row bot";
  row.innerHTML = `<div class="avatar avatar-bot" aria-hidden="true">${ICONS.bot}</div><div class="stack"></div>`;
  messagesEl.append(row);
  return row.querySelector(".stack");
}

function bubble(html, extraClass = "") {
  const el = document.createElement("div");
  el.className = `bubble ${extraClass}`.trim();
  el.innerHTML = html;
  return el;
}

function noticeHtml(tone, text) {
  return `<div class="notice ${tone}">${ICONS[tone]}<div>${escapeHtml(text)}</div></div>`;
}

function showWelcome() {
  const stack = addBotRow();
  stack.append(bubble(
    "<p><strong>안녕하세요! 지원사업 도우미예요.</strong></p>" +
    "<p>기업마당에 올라온 지자체·중앙부처 지원사업 공고를 찾아서, 어느 공고에 나온 내용인지 번호로 출처를 달아 알려드려요.</p>" +
    // 범위를 먼저 밝히는 이유: 데이터가 기업마당(중소기업·소상공인·창업 대상)뿐이라, "청년 월세 지원"처럼 개인 복지를
    // 물으면 근거가 없어 "찾지 못했어요"가 됨 (실제 대화 로그에 있던 질문). 미리 알려서 헛걸음을 줄임.
    '<p class="closing">중소기업·소상공인·창업기업 대상 지원사업을 안내해요. 개인 복지(주거·월세 등)는 아직 다루지 않아요.</p>' +
    "<p>상황을 편하게 말씀해 주세요. 예를 들면 이렇게요.</p>"));
  const chips = document.createElement("div");
  chips.className = "chip-group";
  for (const question of EXAMPLE_QUESTIONS) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = question;
    chip.addEventListener("click", () => ask(question));
    chips.append(chip);
  }
  stack.append(chips);
}

function showTyping() {
  const stack = addBotRow();
  stack.parentElement.classList.add("typing-row");
  const el = bubble('<div class="typing"><span class="dots"><span></span><span></span><span></span></span><span class="typing-text"></span></div>');
  stack.append(el);
  const textEl = el.querySelector(".typing-text");
  const timers = WAITING_STEPS.map(([delay, text]) => setTimeout(() => { textEl.textContent = text; }, delay));
  scrollToBottom();
  return () => { timers.forEach(clearTimeout); stack.parentElement.remove(); };
}

/** 답변 1건 그리기: 말풍선(답변 또는 안내) + 검색 조건 + 출처 카드 */
function renderResult(result) {
  const answerId = ++state.answerCount;
  const stack = addBotRow();
  const citations = result.sources.map((s) => s.citation_index);
  const notice = STATUS_NOTICE[result.status];

  if (notice) {
    stack.append(bubble(noticeHtml(notice.tone, notice.text), `tone-${notice.tone}`));
  } else {
    const answer = bubble(renderAnswer(result.answer, citations));
    // 인용 번호를 누르면 아래 카드 중 그 번호가 들어 있는 카드로 스크롤하고 잠깐 강조
    answer.addEventListener("click", (event) => {
      const cite = event.target.closest("button.cite");
      if (cite) focusCard(answerId, Number(cite.dataset.cite));
    });
    stack.append(answer);
  }

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.innerHTML = describeConditions(result.slots).map((c) => `<span class="tag">${escapeHtml(c)}</span>`).join("") +
    `<span>${escapeHtml(formatDuration(result.response_time_ms))}</span><span>· ${escapeHtml(timeLabel())}</span>`;
  meta.title = "이 조건으로 공고를 찾았어요";
  stack.append(meta);

  if (result.sources.length) stack.append(...renderCards(result.sources, answerId, Boolean(notice)));
  scrollToBottom();
}

function renderCards(sources, answerId, isFallback) {
  const label = document.createElement("div");
  label.className = "cards-label";
  label.textContent = isFallback ? `관련 공고 ${groupSources(sources).length}건` : "답변 근거 공고";
  const list = document.createElement("div");
  list.className = "cards";
  list.setAttribute("role", "list");
  for (const { source, citations } of groupSources(sources)) {
    const card = document.createElement("article");
    card.className = "card";
    card.setAttribute("role", "listitem");
    card.dataset.citations = citations.join(",");
    card.id = `card-${answerId}-${source.program_id}`;
    const badges = citations.map((n) => `<span class="cite cite-static">${n}</span>`).join("");
    const deadline = deadlineInfo(source.apply_period);
    const deadlineBadge = !deadline ? ""
      : deadline.closed ? '<span class="deadline closed">마감</span>'
      : deadline.daysLeft <= 7 ? `<span class="deadline soon">${deadline.daysLeft === 0 ? "오늘 마감" : `D-${deadline.daysLeft}`}</span>`
      : '<span class="deadline open">접수중</span>';
    if (deadline?.closed) card.classList.add("is-closed");
    card.innerHTML = `
      <div class="card-top">${badges}${source.category ? `<span class="category">${escapeHtml(source.category)}</span>` : ""}${deadlineBadge}</div>
      <h3>${escapeHtml(source.program_name)}</h3>
      <dl>
        <dt>대상</dt><dd>${escapeHtml(source.target || "명시 없음")}</dd>
        <dt>접수</dt><dd>${escapeHtml(source.apply_period || "명시 없음")}</dd>
        <dt>금액</dt><dd>${escapeHtml(source.amount || "명시 없음")}</dd>
      </dl>
      <div class="card-actions">
        <button type="button" class="button secondary" data-detail>자세히</button>
        <a class="button primary" target="_blank" rel="noopener noreferrer">원문 ${ICONS.external}</a>
      </div>`;
    card.querySelector("a").href = source.url;
    card.querySelector("[data-detail]").addEventListener("click", () => openDetail(source.program_id));
    list.append(card);
  }
  return [label, list];
}

function focusCard(answerId, citation) {
  const card = [...document.querySelectorAll(`[id^="card-${answerId}-"]`)]
    .find((el) => el.dataset.citations.split(",").map(Number).includes(citation));
  if (!card) return;
  card.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
  card.classList.add("highlight");
  setTimeout(() => card.classList.remove("highlight"), 1600);
}

function renderError(message, retryQuestion) {
  const stack = addBotRow();
  const el = bubble(noticeHtml("error", message), "tone-error");
  if (retryQuestion) {
    const retry = document.createElement("button");
    retry.type = "button";
    retry.className = "chip";
    retry.textContent = "다시 시도";
    retry.addEventListener("click", () => { stack.parentElement.remove(); ask(retryQuestion, { echo: false }); });
    stack.append(el, retry);
  } else {
    stack.append(el);
  }
  scrollToBottom();
}

// ── 질문 보내기 ─────────────────────────────────────────────────────
async function ask(question, { echo = true } = {}) {
  question = question.trim();
  if (!question || state.busy) return;
  setBusy(true);
  if (echo) addUserMessage(question);
  const stopTyping = showTyping();
  try {
    const result = await api("/chat", {
      method: "POST",
      timeout: CHAT_TIMEOUT_MS,
      body: {
        question,
        session_id: state.sessionId,
        region: state.filters.region || null,
        categories: state.filters.categories,
        targets: state.filters.target ? [state.filters.target] : [],
      },
    });
    stopTyping();
    state.sessionId = result.session_id;
    saveSession(result.session_id);
    renderResult(result);
  } catch (error) {
    stopTyping();
    renderError(error.message, question);
  } finally {
    setBusy(false);
  }
}

function setBusy(busy) {
  state.busy = busy;
  updateSendButton();
}

function updateSendButton() {
  sendButton.disabled = state.busy || !questionEl.value.trim();
}

// ── 대화 이력 복원 (새로고침해도 이어지게, SFR-008) ─────────────────
async function restoreHistory() {
  if (!state.sessionId) return false;
  let items;
  try {
    ({ items } = await api(`/chat/history/${encodeURIComponent(state.sessionId)}`));
  } catch {
    return false;
  }
  if (!items.length) return false;
  for (const item of items) {
    const when = new Date(item.created_at);
    addUserMessage(item.question, when);
    const stack = addBotRow();
    // 이력에는 카드에 필요한 정보(사업명·기간)가 저장되지 않아서(TB_CHAT_LOG) 답변 글만 복원합니다.
    stack.append(item.status === "ok"
      ? bubble(renderAnswer(item.answer))
      : bubble(noticeHtml("info", item.answer), "tone-info"));
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `이전 대화 · ${timeLabel(when)} · 근거 공고 ${item.cited_program_ids.length}건`;
    stack.append(meta);
  }
  scrollToBottom();
  return true;
}

function newConversation() {
  if (state.busy) return;
  state.sessionId = null;
  saveSession(null);
  state.answerCount = 0;
  messagesEl.replaceChildren();
  showWelcome();
  questionEl.focus();
}

// ── 상세 패널 (SCR-02) ──────────────────────────────────────────────
async function openDetail(programId, chunkCount = DETAIL_FIRST_CHUNKS) {
  const body = $("#detail-body");
  if (state.detail?.programId !== programId) {
    body.innerHTML = '<div class="skeleton" style="width:80%"></div><div class="skeleton"></div><div class="skeleton" style="width:60%"></div>';
  }
  openSheet("#detail-panel");
  state.detail = { programId, chunks: chunkCount };
  let program;
  try {
    program = await api(`/programs/${encodeURIComponent(programId)}?max_chunks=${chunkCount}`);
  } catch (error) {
    body.innerHTML = noticeHtml("error", error.message);
    return;
  }
  if (state.detail?.programId !== programId) return; // 기다리는 사이 다른 공고를 열었으면 버림

  const info = [
    ["지자체", program.region_name], ["분야", program.category],
    ["접수기간", program.apply_period, "wide"], ["지원대상", program.target, "wide"], ["지원금액", program.amount, "wide"],
  ];
  body.innerHTML = `
    <h3 class="detail-title">${escapeHtml(program.program_name)}</h3>
    <dl class="info-grid">${info.map(([label, value, cls]) =>
      `<div class="${cls || ""}"><dt>${label}</dt><dd>${escapeHtml(value || "명시 없음")}</dd></div>`).join("")}</dl>
    <a class="button primary" style="width:100%;height:42px" target="_blank" rel="noopener noreferrer" id="detail-source">
      기업마당 원문 공고 · 첨부파일 보기 ${ICONS.external}</a>
    <h4 class="section-title">공고 본문</h4>
    <p class="muted">전체 ${program.chunk_count}개 구간 중 앞 ${program.chunks.length}개 · PDF/HWP에서 자동 추출한 글이라 표나 서식이 깨져 보일 수 있어요.</p>
    ${program.chunks.map((c) => `<div class="chunk">${escapeHtml(c.text)}</div>`).join("")}
    ${program.chunks.length < program.chunk_count
      ? '<button type="button" class="button secondary" id="detail-more" style="width:100%;margin-top:12px">본문 더 보기</button>' : ""}`;
  body.querySelector("#detail-source").href = program.source_url;
  body.querySelector("#detail-more")?.addEventListener("click", () => openDetail(programId, chunkCount + DETAIL_MORE_CHUNKS));
}

// ── 필터 패널 ───────────────────────────────────────────────────────
function fillFilterOptions() {
  const { regions, targets, categories } = state.filterOptions;
  const fill = (select, values) => {
    select.replaceChildren(new Option("전체", ""), ...values.map((v) => new Option(v, v)));
  };
  fill($("#filter-region"), regions);
  fill($("#filter-target"), targets);
  const group = $("#filter-categories");
  group.replaceChildren(...categories.map((value) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = value;
    chip.dataset.value = value;
    chip.setAttribute("aria-pressed", "false");
    chip.addEventListener("click", () => chip.setAttribute("aria-pressed", String(chip.getAttribute("aria-pressed") !== "true")));
    return chip;
  }));
}

function openFilters() {
  $("#filter-region").value = state.filters.region;
  $("#filter-target").value = state.filters.target;
  for (const chip of document.querySelectorAll("#filter-categories .chip")) {
    chip.setAttribute("aria-pressed", String(state.filters.categories.includes(chip.dataset.value)));
  }
  openSheet("#filter-panel");
}

function applyFilters() {
  state.filters = {
    region: $("#filter-region").value,
    target: $("#filter-target").value,
    categories: [...document.querySelectorAll('#filter-categories .chip[aria-pressed="true"]')].map((c) => c.dataset.value),
  };
  renderActiveFilters();
  closeSheets();
}

/** 입력창 위에 "지금 걸린 필터"를 칩으로. x를 누르면 그 필터만 해제. 필터가 걸린 줄 모르고 결과가 적다고 느끼지 않게. */
function renderActiveFilters() {
  const { region, target, categories } = state.filters;
  const items = [
    ...(region ? [["region", region, `지역 ${region}`]] : []),
    ...(target ? [["target", target, `대상 ${target}`]] : []),
    ...categories.map((c) => ["category", c, `분야 ${c}`]),
  ];
  const box = $("#active-filters");
  box.hidden = !items.length;
  box.replaceChildren(...items.map(([kind, value, label]) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "filter-chip";
    chip.innerHTML = `${escapeHtml(label)}${ICONS.close}`;
    chip.setAttribute("aria-label", `${label} 필터 해제`);
    chip.addEventListener("click", () => {
      if (kind === "region") state.filters.region = "";
      if (kind === "target") state.filters.target = "";
      if (kind === "category") state.filters.categories = state.filters.categories.filter((c) => c !== value);
      renderActiveFilters();
    });
    return chip;
  }));
  const badge = $("#filter-count");
  badge.hidden = !items.length;
  badge.textContent = String(items.length);
}

// ── 겹쳐 뜨는 창 공통 ───────────────────────────────────────────────
function openSheet(selector) {
  closeSheets();
  $("#scrim").hidden = false;
  const sheet = $(selector);
  sheet.hidden = false;
  sheet.querySelector("[data-close]")?.focus();
}

function closeSheets() {
  $("#scrim").hidden = true;
  document.querySelectorAll(".sheet").forEach((s) => { s.hidden = true; });
  state.detail = null;
}

// ── 헤더 상태 ───────────────────────────────────────────────────────
async function loadStatus() {
  const status = $("#status");
  const text = $("#status-text");
  try {
    const health = await api("/health");
    if (!health.opensearch) {
      status.dataset.state = "down";
      text.textContent = "검색 서버 꺼짐";
      status.title = "OpenSearch에 연결되지 않아 질문에 답할 수 없어요.";
    } else if (health.llm_enabled) {
      status.dataset.state = "ok";
      text.textContent = "답변 가능";
      status.title = `답변 모델: ${health.llm_provider} / ${health.llm_model} · 색인 청크 ${health.index_docs?.toLocaleString("ko-KR")}개`;
    } else {
      status.dataset.state = "search-only";
      text.textContent = "공고 검색만";
      status.title = "답변 생성(LLM)이 꺼져 있어 관련 공고만 안내해요.";
    }
  } catch (error) {
    status.dataset.state = "down";
    text.textContent = "서버 연결 안 됨";
    status.title = error.message;
  }
  try {
    const { total } = await api("/programs?size=1");
    $("#service-meta").textContent = `기업마당 공고 ${total.toLocaleString("ko-KR")}건 · 서울`;
  } catch { /* 헤더 부가 정보라 실패해도 무시 */ }
  try {
    // 데이터가 "언제 기준"인지 보여줌 - 자동 갱신(src/sync)이 멈추면 사용자가 알아챌 수 있게
    const sync = await api("/sync/status");
    const done = sync.last_success?.finished_at;
    if (done) {
      const when = new Date(done);
      const label = when.toLocaleString("ko-KR", { month: "numeric", day: "numeric", hour: "numeric", minute: "2-digit" });
      $("#service-meta").textContent += ` · ${label} 갱신`;
      $("#service-meta").title = `최근 자동 갱신: 새 공고 ${sync.last_success.added_programs ?? 0}건 반영`;
    }
  } catch { /* 갱신 기록이 없어도 화면은 정상 */ }
  try {
    state.filterOptions = await api("/programs/filters");
    fillFilterOptions();
  } catch { /* 필터 없이도 질문은 가능 */ }
}

// ── 이벤트 연결 & 시작 ─────────────────────────────────────────────
function autoGrow() {
  questionEl.style.height = "auto";
  questionEl.style.height = `${Math.min(questionEl.scrollHeight, 140)}px`;
}

$("#chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const question = questionEl.value;
  if (!question.trim() || state.busy) return;
  questionEl.value = "";
  autoGrow();
  updateSendButton();
  ask(question);
});
questionEl.addEventListener("input", () => { autoGrow(); updateSendButton(); });
questionEl.addEventListener("keydown", (event) => {
  // Enter = 보내기, Shift+Enter = 줄바꿈. 한글 입력 중(조합 중) Enter는 글자 확정이라 보내면 안 됨 (마지막 글자가 두 번 보내지는 문제)
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("#chat-form").requestSubmit();
  }
});
$("#new-chat-button").addEventListener("click", newConversation);
$("#filter-button").addEventListener("click", openFilters);
$("#filter-apply").addEventListener("click", applyFilters);
$("#filter-reset").addEventListener("click", () => {
  $("#filter-region").value = "";
  $("#filter-target").value = "";
  document.querySelectorAll("#filter-categories .chip").forEach((c) => c.setAttribute("aria-pressed", "false"));
});
$("#scrim").addEventListener("click", closeSheets);
document.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", closeSheets));
document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeSheets(); });

async function start() {
  loadStatus();
  const restored = await restoreHistory();
  if (!restored) showWelcome();
  // 주소에 ?q=질문 을 붙이면 바로 물어봄 - 시연 링크나 화면 테스트(tests/test_web_ui.py)에서 씀
  const preset = new URLSearchParams(location.search).get("q");
  if (preset) ask(preset);
  else questionEl.focus();
}

start();
