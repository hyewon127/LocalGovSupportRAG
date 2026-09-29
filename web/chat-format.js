// [WBS 9] 챗봇 웹 화면 - 화면(DOM)과 상관없는 순수 로직
//
// 왜 app.js와 나눴는가: 답변 문자열을 HTML로 바꾸는 부분은 버그가 나면 "LLM 답변 속 <script>가 실행되는" 보안 문제로
// 이어질 수 있어서, 브라우저 없이 Node로 바로 테스트할 수 있게 떼어냈습니다 (tests/web_format.test.mjs).
// 이 파일은 document/window를 쓰지 않습니다.

/** HTML 특수문자를 글자 그대로 보이게 바꿈. LLM 답변과 공고 원문은 전부 이걸 거친 뒤에만 화면에 넣습니다. */
export function escapeHtml(text) {
  return String(text ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

// LLM이 마지막 줄에 붙이는 고정 문구 (src/rag/prompt_template.py 규칙 6). 본문과 구분해 흐리게 표시합니다.
const CLOSING_NOTE = "정확한 내용은 원문 공고를 확인하세요.";

/**
 * 한 줄 안의 꾸밈(굵게, 인용 번호)을 HTML로 바꿈. 입력은 이미 escapeHtml을 거친 문자열이어야 합니다.
 * 인용 번호 [n]은 실제 근거 번호(validCitations)에 있을 때만 버튼으로 만듭니다 - 백엔드 가드레일이 없는 번호를
 * 지우지만(guardrail.py), 화면에서도 "누르면 아무 일도 없는 버튼"이 생기지 않게 한 번 더 막습니다.
 */
function renderInline(escaped, validCitations) {
  return escaped
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\[(\d{1,2})\]/g, (match, n) =>
      validCitations.has(Number(n))
        ? `<button type="button" class="cite" data-cite="${n}" aria-label="근거 ${n}번 공고 보기">${n}</button>`
        : match);
}

/**
 * LLM 답변(마크다운 일부)을 안전한 HTML로 바꿈.
 * 마크다운 라이브러리를 쓰지 않은 이유: 이 챗봇의 답변은 프롬프트로 형식을 정해 둬서(굵은 사업명, "- " 목록, 인용 번호)
 * 필요한 문법이 몇 개뿐이고, 외부 라이브러리 없이 오프라인에서도 동작하게 하려고.
 *
 * @param {string} text - 답변 원문
 * @param {Iterable<number>} citations - 유효한 인용 번호들 (sources의 citation_index)
 */
export function renderAnswer(text, citations = []) {
  const valid = new Set(citations);
  const html = [];
  let list = null; // 지금 열려 있는 목록: { tag: "ul"|"ol", items: [] }

  const closeList = () => {
    if (list) html.push(`<${list.tag}>${list.items.join("")}</${list.tag}>`);
    list = null;
  };

  for (const rawLine of String(text ?? "").split(/\r?\n/)) {
    const line = rawLine.trimEnd();
    const bullet = line.match(/^(\s*)[-*•]\s+(.*)$/);
    const numbered = line.match(/^(\s*)\d+[.)]\s+(.*)$/);
    const item = bullet || numbered;
    if (item) {
      const tag = bullet ? "ul" : "ol";
      if (!list || list.tag !== tag) {
        closeList();
        list = { tag, items: [] };
      }
      if (item[2].trim() === CLOSING_NOTE) { // 모델이 마지막 안내 문구까지 목록 항목으로 쓴 경우 (실제 답변에서 확인)
        closeList();
        html.push(`<p class="closing">${escapeHtml(CLOSING_NOTE)}</p>`);
        continue;
      }
      const nested = item[1].length >= 2 ? ' class="nested"' : ""; // "  - " 들여쓴 하위 항목
      list.items.push(`<li${nested}>${renderInline(escapeHtml(item[2]), valid)}</li>`);
      continue;
    }
    closeList();
    if (!line.trim()) continue;
    const heading = line.match(/^#{1,6}\s+(.*)$/); // "### 제목"은 굵은 문단으로 (말풍선 안에서 큰 제목은 어색함)
    if (heading) {
      html.push(`<p><strong>${renderInline(escapeHtml(heading[1]), valid)}</strong></p>`);
    } else if (line.trim() === CLOSING_NOTE) {
      html.push(`<p class="closing">${escapeHtml(line.trim())}</p>`);
    } else {
      html.push(`<p>${renderInline(escapeHtml(line), valid)}</p>`);
    }
  }
  closeList();
  return html.join("");
}

/**
 * 같은 사업이 근거에 2번 들어오면(hybrid_search.py가 사업당 청크 최대 2개 허용) 카드 1장으로 합침.
 * 인용 번호는 [1,2]처럼 모아서 답변 본문의 번호와 계속 대응되게 합니다. 순서는 처음 나온 순서(= 검색 점수 순).
 */
export function groupSources(sources = []) {
  const byProgram = new Map();
  for (const source of sources) {
    if (!byProgram.has(source.program_id)) byProgram.set(source.program_id, { source, citations: [] });
    byProgram.get(source.program_id).citations.push(source.citation_index);
  }
  return [...byProgram.values()];
}

/** 답변 아래 작은 글씨로 보여줄 "이 조건으로 찾았어요" 목록. 결과가 이상할 때 필터 탓인지 바로 보이게 (투명성). */
export function describeConditions(slots = {}) {
  const parts = [];
  if (slots.region) parts.push(`지역 ${slots.region}`);
  if (slots.categories?.length) parts.push(`분야 ${slots.categories.join(", ")}`);
  if (slots.target_keywords?.length) parts.push(`대상 ${slots.target_keywords.join(", ")}`);
  return parts.length ? parts : ["조건 없음 (전체 검색)"];
}

/**
 * 접수기간 문자열("2026-08-05 ~ 2026-09-02")로 마감 여부와 남은 날을 계산. 카드에 "마감" / "D-3" 표시용.
 * 날짜가 아닌 값("예산 소진시까지", "세부사업별 상이")은 판단할 수 없으므로 null.
 * 왜 필요한가: 8월에 받은 공고는 9월 말 기준 대부분 마감이라, 표시가 없으면 이미 끝난 사업을 신청할 수 있는 것처럼 보임.
 *
 * @param {string} period - Source.apply_period
 * @param {Date} today - 테스트에서 날짜를 고정하려고 받음
 * @returns {{closed: boolean, daysLeft: number} | null}
 */
export function deadlineInfo(period, today = new Date()) {
  const dates = String(period ?? "").match(/\d{4}-\d{2}-\d{2}/g);
  if (!dates) return null;
  const end = new Date(`${dates[dates.length - 1]}T23:59:59`);
  if (Number.isNaN(end.getTime())) return null;
  const startOfToday = new Date(today.getFullYear(), today.getMonth(), today.getDate());
  const daysLeft = Math.floor((end - startOfToday) / 86_400_000);
  return { closed: daysLeft < 0, daysLeft };
}

/** 응답 시간(ms)을 "3.2초"처럼. 1초 미만은 ms 그대로 - 검색만 한 경우(근거 없음) 수십 ms라서. */
export function formatDuration(ms) {
  if (!Number.isFinite(ms)) return "";
  return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}초`;
}

/**
 * API 에러 응답을 사람이 읽을 문장으로. 우리 서버의 에러는 {"detail": "문장"}인데(src/api/main.py),
 * 입력 검증 실패(422)는 FastAPI 기본 형식이라 detail이 목록으로 옵니다.
 */
export function errorMessage(status, body) {
  const detail = body?.detail;
  if (typeof detail === "string" && detail) return detail;
  if (status === 422) {
    const first = Array.isArray(detail) ? detail[0] : null;
    return first?.msg ? `입력값을 확인해 주세요: ${first.msg}` : "입력값을 확인해 주세요.";
  }
  if (status === 0) return "서버에 연결할 수 없어요. 백엔드가 켜져 있는지 확인해 주세요.";
  return `서버 오류가 발생했어요 (${status}). 잠시 후 다시 시도해 주세요.`;
}
