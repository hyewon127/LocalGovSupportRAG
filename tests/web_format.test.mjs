// [WBS 9] 웹 화면 순수 로직(web/chat-format.js) 테스트 - 브라우저 없이 Node로 실행
//
// 실행: node tests/web_format.test.mjs   (tests/test_web_ui.py가 같이 돌림)
// 가장 중요한 건 1번: LLM 답변과 공고 원문은 외부에서 온 글이라, 그 안의 <script>나 <img onerror>가 화면에서 실행되면
// 안 됩니다(XSS). 답변을 HTML로 바꾸는 renderAnswer가 모든 글자를 이스케이프한 뒤에만 꾸밈을 붙이는지 확인합니다.

import { deadlineInfo, describeConditions, errorMessage, escapeHtml, formatDuration, groupSources, renderAnswer } from "../web/chat-format.js";

const results = [];
function check(name, ok, detail) {
  results.push(Boolean(ok));
  console.log(`[${ok ? "PASS" : "FAIL"}] ${name}${ok || detail === undefined ? "" : `  (${detail})`}`);
}

// 1. 보안: 답변 속 HTML은 글자로만
const evil = renderAnswer('<img src=x onerror="alert(1)"> **굵게** <script>alert(2)</script> [1]', [1]);
check("XSS: <img onerror>가 태그로 살아남지 않음", !evil.includes("<img") && evil.includes("&lt;img"), evil);
check("XSS: <script>가 태그로 살아남지 않음", !evil.includes("<script") && evil.includes("&lt;script&gt;"), evil);
check("XSS: escapeHtml이 따옴표까지 처리", escapeHtml(`"'<>&`) === "&quot;&#39;&lt;&gt;&amp;");

// 2. 형식: 실제 LLM 답변 모양 (prompt_template.py의 답변 형식)
const answer = [
  "**2026년 소상공인 온라인판로 지원사업** [1]",
  "- 지원대상: 소상공인 [1]",
  "- 접수기간: 2026-07-20 ~ 2026-08-24 [1][3]",
  "",
  "정확한 내용은 원문 공고를 확인하세요.",
].join("\n");
const html = renderAnswer(answer, [1, 3]);
check("굵게 -> <strong>", html.includes("<strong>2026년 소상공인 온라인판로 지원사업</strong>"), html);
check("'- ' 줄 -> <ul><li> 목록 하나로 묶음", (html.match(/<ul>/g) || []).length === 1 && (html.match(/<li>/g) || []).length === 2, html);
check("인용 번호 -> 누를 수 있는 버튼", html.includes('data-cite="1"') && html.includes('data-cite="3"'), html);
check("마지막 안내 문구는 흐린 문단(closing)", html.includes('<p class="closing">정확한 내용은 원문 공고를 확인하세요.</p>'), html);

// 3. 근거에 없는 번호는 버튼으로 만들지 않음 (눌러도 갈 카드가 없음)
const invalid = renderAnswer("A 사업입니다 [7]", [1, 2]);
check("없는 번호 [7]은 글자 그대로", !invalid.includes("data-cite") && invalid.includes("[7]"), invalid);

// 4. 모델이 안내 문구까지 목록 항목으로 쓴 경우 (실제 답변에서 확인)
const listClosing = renderAnswer("- 지원대상: 소상공인 [1]\n- 정확한 내용은 원문 공고를 확인하세요.", [1]);
check("목록 안의 안내 문구도 closing으로 분리", listClosing.includes('class="closing"') && (listClosing.match(/<li>/g) || []).length === 1, listClosing);

// 5. 같은 사업의 청크 2개 -> 카드 1장, 인용 번호는 모음
const groups = groupSources([
  { program_id: "P1", citation_index: 1 }, { program_id: "P2", citation_index: 2 }, { program_id: "P1", citation_index: 3 },
]);
check("같은 사업은 카드 1장으로", groups.length === 2 && groups[0].citations.join() === "1,3" && groups[1].citations.join() === "2",
  JSON.stringify(groups));

// 6. 검색 조건 / 시간 / 에러 문구
check("검색 조건: 슬롯이 있으면 나열", describeConditions({ region: "서울", categories: ["수출"], target_keywords: ["여성기업"] }).join("|")
  === "지역 서울|분야 수출|대상 여성기업");
check("검색 조건: 없으면 '전체 검색'", describeConditions({ region: null, categories: [], target_keywords: [] })[0].includes("전체 검색"));
check("시간: 1초 미만은 ms, 이상은 초", formatDuration(49) === "49ms" && formatDuration(6430) === "6.4초");
check("에러: 서버의 detail 문장을 그대로", errorMessage(503, { detail: "검색 서버에 연결할 수 없습니다." }) === "검색 서버에 연결할 수 없습니다.");
check("에러: 422 목록 형식도 문장으로", errorMessage(422, { detail: [{ msg: "String should have at most 500 characters" }] }).startsWith("입력값을 확인해 주세요"));

// 7. 마감 표시 (오늘 = 2026-09-29로 고정)
const today = new Date(2026, 8, 29);
check("마감: 종료일이 지난 공고", deadlineInfo("2026-08-05 ~ 2026-09-02", today)?.closed === true);
check("마감: 오늘이 종료일이면 D-0 (아직 접수 가능)", deadlineInfo("2026-09-01 ~ 2026-09-29", today)?.daysLeft === 0
  && deadlineInfo("2026-09-01 ~ 2026-09-29", today)?.closed === false);
check("마감: 남은 날 계산", deadlineInfo("2026-09-28 ~ 2026-10-07", today)?.daysLeft === 8);
check("마감: 날짜가 아닌 기간은 판단 안 함", deadlineInfo("예산 소진시까지", today) === null && deadlineInfo(null, today) === null);

const passed = results.filter(Boolean).length;
console.log(`\n${passed}/${results.length} passed`);
process.exit(passed === results.length ? 0 : 1);
