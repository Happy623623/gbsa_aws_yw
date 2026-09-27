/**
 * The review route must render criterion names, not raw UUIDs.
 *
 * The ReviewWorkspace component tests hand it an already-readable `criterionName`,
 * so they pass even when the adapter feeds the component a UUID. Only a test that
 * goes through `ReviewRoute` with a real API payload catches that.
 */
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ReviewRoute } from "../routeAdapters";

const SESSION_ID = "00000000-0000-7000-8000-000000000001";
const INVITATION_ID = "00000000-0000-7000-8000-000000000002";
const REPORT_ID = "00000000-0000-7000-8000-000000000003";
const REPORT_ITEM_ID = "00000000-0000-7000-8000-000000000004";
const CRITERION_ID = "00000000-0000-7000-8000-000000000005";

function reportPayload(criterionName: string) {
  return {
    report_id: REPORT_ID,
    report_version: 1,
    status: "ready",
    summary: "최종 답변 Evidence에 기반한 AI 원본 리포트",
    ai_original_immutable: true,
    items: [
      {
        report_item_id: REPORT_ITEM_ID,
        criterion_id: CRITERION_ID,
        criterion_name: criterionName,
        assessment_state: "confirmed",
        observation: "장애 대응에서 큐 도입 근거를 설명함",
        rationale: "실제 최종 답변만 평가 근거로 사용",
        uncertainty: "AI 원본이며 사람 검토 필요",
        follow_up_question: null,
        evidence: [],
      },
    ],
    human_reviews: [],
  };
}

const TIMELINE_PAYLOAD = {
  entries: [],
  playback: { url: null, expires_at: null, status: "unavailable" },
};

function stubApi(criterionName: string) {
  vi.stubGlobal(
    "fetch",
    vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      const body = url.endsWith("/timeline")
        ? TIMELINE_PAYLOAD
        : url.endsWith("/recruiting-state")
          ? {
              invitation_id: INVITATION_ID,
              position_id: "00000000-0000-7000-8000-000000000006",
              recruiting_stage_id: "00000000-0000-7000-8000-000000000007",
              pipeline_row_version: 2,
              stages: [
                {
                  recruiting_stage_id: "00000000-0000-7000-8000-000000000007",
                  position_id: "00000000-0000-7000-8000-000000000006",
                  name: "1차 합격",
                  sort_order: 2,
                  row_version: 1,
                },
              ],
            }
          : reportPayload(criterionName);
      return Promise.resolve(
        new Response(JSON.stringify(body), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      );
    }),
  );
}

function renderReview(search = `invitationId=${INVITATION_ID}`) {
  render(
    <MemoryRouter initialEntries={[`/review/${SESSION_ID}?${search}`]}>
      <Routes>
        <Route path="/review/:sessionId" element={<ReviewRoute />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("ReviewRoute", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("shows the criterion name the report captured", async () => {
    stubApi("장애 대응");
    renderReview();

    // The name is a row header on the 종합평가 sheet's criterion table.
    expect(
      await screen.findByRole("rowheader", { name: "장애 대응" }),
    ).toBeTruthy();
    expect(await screen.findByText("현재 단계 · 1차 합격")).toBeTruthy();
    expect(screen.queryByText(CRITERION_ID)).toBeNull();
  });

  it("falls back to the id for reports generated before names were captured", async () => {
    stubApi("");
    renderReview();

    expect(
      await screen.findByRole("rowheader", { name: CRITERION_ID }),
    ).toBeTruthy();
  });

  it("reads a report generated before scoring existed without inventing zeroes", async () => {
    // This payload has no overall_score, average_score or axis_assessments at all,
    // which is what every report written before the scoring migration looks like.
    stubApi("장애 대응");
    renderReview();

    await screen.findByRole("rowheader", { name: "장애 대응" });
    expect(screen.getByText("점수화된 기준 없음")).toBeTruthy();
    expect(screen.getAllByText("판단 근거 없음").length).toBeGreaterThan(0);
    expect(screen.queryByText("0점")).toBeNull();
  });

  it("saves the selected recruiting stage with the live pipeline version", async () => {
    const finalStageId = "00000000-0000-7000-8000-000000000008";
    const requests: RequestInit[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/final-decisions")) {
          requests.push(init ?? {});
          return Promise.resolve(
            new Response(
              JSON.stringify({
                human_review: {},
                invitation_id: INVITATION_ID,
                position_id: "00000000-0000-7000-8000-000000000006",
                recruiting_stage_id: finalStageId,
                recruiting_stage_name: "최종 합격",
                pipeline_row_version: 3,
                invitation_state: "reviewed",
              }),
              { status: 201, headers: { "Content-Type": "application/json" } },
            ),
          );
        }
        const body = url.endsWith("/timeline")
          ? TIMELINE_PAYLOAD
          : url.endsWith("/recruiting-state")
            ? {
                invitation_id: INVITATION_ID,
                position_id: "00000000-0000-7000-8000-000000000006",
                recruiting_stage_id: "00000000-0000-7000-8000-000000000007",
                pipeline_row_version: 2,
                stages: [
                  {
                    recruiting_stage_id: "00000000-0000-7000-8000-000000000007",
                    position_id: "00000000-0000-7000-8000-000000000006",
                    name: "1차 합격",
                    sort_order: 0,
                    row_version: 1,
                  },
                  {
                    recruiting_stage_id: finalStageId,
                    position_id: "00000000-0000-7000-8000-000000000006",
                    name: "최종 합격",
                    sort_order: 1,
                    row_version: 1,
                  },
                ],
              }
            : reportPayload("역량 기준");
        return Promise.resolve(
          new Response(JSON.stringify(body), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }),
        );
      }),
    );
    renderReview();

    const option = await screen.findByRole("option", { name: "최종 합격" });
    const select = option.parentElement as HTMLSelectElement;
    fireEvent.change(select, { target: { value: finalStageId } });
    const section = select.closest("section");
    fireEvent.click(section?.querySelector("button") as HTMLButtonElement);

    await waitFor(() => expect(requests).toHaveLength(1));
    expect(JSON.parse(String(requests[0]?.body))).toEqual({
      recruiting_stage_id: finalStageId,
      expected_pipeline_version: 2,
    });
    expect(await screen.findByText("현재 단계 · 최종 합격")).toBeTruthy();
  });

  it("refreshes the report stage after a kanban change when focus returns", async () => {
    const firstStageId = "00000000-0000-7000-8000-000000000007";
    const movedStageId = "00000000-0000-7000-8000-000000000008";
    let recruitingReads = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        const url = String(input);
        let body: object;
        if (url.endsWith("/timeline")) {
          body = TIMELINE_PAYLOAD;
        } else if (url.endsWith("/recruiting-state")) {
          recruitingReads += 1;
          const moved = recruitingReads > 1;
          body = {
            invitation_id: INVITATION_ID,
            position_id: "00000000-0000-7000-8000-000000000006",
            recruiting_stage_id: moved ? movedStageId : firstStageId,
            pipeline_row_version: moved ? 3 : 2,
            stages: [
              {
                recruiting_stage_id: firstStageId,
                position_id: "00000000-0000-7000-8000-000000000006",
                name: "1차 합격",
                sort_order: 0,
                row_version: 1,
              },
              {
                recruiting_stage_id: movedStageId,
                position_id: "00000000-0000-7000-8000-000000000006",
                name: "최종 합격",
                sort_order: 1,
                row_version: 1,
              },
            ],
          };
        } else {
          body = reportPayload("역량 기준");
        }
        return Promise.resolve(
          new Response(JSON.stringify(body), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }),
        );
      }),
    );
    renderReview();

    expect(await screen.findByText("현재 단계 · 1차 합격")).toBeTruthy();
    await act(async () => window.dispatchEvent(new Event("focus")));

    expect(await screen.findByText("현재 단계 · 최종 합격")).toBeTruthy();
    expect(recruitingReads).toBe(2);
  });

  it("reloads the latest stage after an optimistic-lock conflict", async () => {
    const firstStageId = "00000000-0000-7000-8000-000000000007";
    const movedStageId = "00000000-0000-7000-8000-000000000008";
    let recruitingReads = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("/final-decisions")) {
          return Promise.resolve(
            new Response(
              JSON.stringify({ detail: "stale applicant pipeline version" }),
              {
                status: 409,
                headers: { "Content-Type": "application/json" },
              },
            ),
          );
        }
        const body = url.endsWith("/timeline")
          ? TIMELINE_PAYLOAD
          : url.endsWith("/recruiting-state")
            ? (() => {
                recruitingReads += 1;
                const refreshed = recruitingReads > 1;
                return {
                  invitation_id: INVITATION_ID,
                  position_id: "00000000-0000-7000-8000-000000000006",
                  recruiting_stage_id: refreshed ? movedStageId : firstStageId,
                  pipeline_row_version: refreshed ? 3 : 2,
                  stages: [
                    {
                      recruiting_stage_id: firstStageId,
                      position_id: "00000000-0000-7000-8000-000000000006",
                      name: "1차 합격",
                      sort_order: 0,
                      row_version: 1,
                    },
                    {
                      recruiting_stage_id: movedStageId,
                      position_id: "00000000-0000-7000-8000-000000000006",
                      name: "최종 합격",
                      sort_order: 1,
                      row_version: 1,
                    },
                  ],
                };
              })()
            : reportPayload("역량 기준");
        return Promise.resolve(
          new Response(JSON.stringify(body), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }),
        );
      }),
    );
    renderReview();

    const option = await screen.findByRole("option", { name: "최종 합격" });
    const select = option.parentElement as HTMLSelectElement;
    fireEvent.change(select, { target: { value: movedStageId } });
    fireEvent.click(
      select.closest("section")?.querySelector("button") as HTMLButtonElement,
    );

    expect(await screen.findByText("현재 단계 · 최종 합격")).toBeTruthy();
    expect(recruitingReads).toBe(2);
    expect(await screen.findByRole("alert")).toBeTruthy();
  });

  it.each([
    ["auto=1", "자동 면접이 끝났습니다. 최종 리포트를 생성하고 있습니다."],
    [
      `invitationId=${INVITATION_ID}`,
      "면접이 끝났습니다. 최종 리포트를 생성하고 있습니다.",
    ],
  ])(
    "polls until a completed interview report is ready",
    async (search, message) => {
      vi.useFakeTimers();
      try {
        let reportRequests = 0;
        vi.stubGlobal(
          "fetch",
          vi.fn((input: RequestInfo | URL) => {
            const url = String(input);
            if (url.endsWith("/timeline")) {
              return Promise.resolve(
                new Response(JSON.stringify(TIMELINE_PAYLOAD), {
                  status: 200,
                  headers: { "Content-Type": "application/json" },
                }),
              );
            }
            if (url.endsWith("/recruiting-state")) {
              return Promise.resolve(
                new Response(
                  JSON.stringify({
                    invitation_id: INVITATION_ID,
                    position_id: "00000000-0000-7000-8000-000000000006",
                    recruiting_stage_id: "00000000-0000-7000-8000-000000000007",
                    pipeline_row_version: 1,
                    stages: [],
                  }),
                  {
                    status: 200,
                    headers: { "Content-Type": "application/json" },
                  },
                ),
              );
            }
            reportRequests += 1;
            const pending = reportRequests === 1;
            return Promise.resolve(
              new Response(
                JSON.stringify(
                  pending
                    ? { status: "queued", retryable: true, message: null }
                    : reportPayload("자동 면접 검증"),
                ),
                {
                  status: pending ? 202 : 200,
                  headers: { "Content-Type": "application/json" },
                },
              ),
            );
          }),
        );

        renderReview(search);
        await act(async () => {
          await Promise.resolve();
          await Promise.resolve();
        });
        expect(screen.getByText(message)).toBeTruthy();

        await act(async () => {
          await vi.advanceTimersByTimeAsync(2000);
        });

        expect(
          screen.getByRole("rowheader", { name: "자동 면접 검증" }),
        ).toBeTruthy();
        expect(reportRequests).toBe(2);
      } finally {
        vi.useRealTimers();
      }
    },
  );

  it("shows an explicit delay warning while a report remains queued", async () => {
    vi.useFakeTimers();
    try {
      vi.stubGlobal(
        "fetch",
        vi.fn((input: RequestInfo | URL) => {
          const url = String(input);
          const body = url.endsWith("/timeline")
            ? TIMELINE_PAYLOAD
            : url.endsWith("/recruiting-state")
              ? {
                  invitation_id: INVITATION_ID,
                  position_id: "00000000-0000-7000-8000-000000000006",
                  recruiting_stage_id: "00000000-0000-7000-8000-000000000007",
                  pipeline_row_version: 1,
                  stages: [],
                }
              : { status: "queued", retryable: true, message: null };
          return Promise.resolve(
            new Response(JSON.stringify(body), {
              status: url.endsWith("/report") ? 202 : 200,
              headers: { "Content-Type": "application/json" },
            }),
          );
        }),
      );

      renderReview();
      await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
      });
      expect(screen.getByRole("status")).toBeTruthy();

      await act(async () => {
        await vi.advanceTimersByTimeAsync(8000);
      });

      const warning = screen.getByRole("alert");
      expect(warning.textContent).toContain(
        "최종 리포트 생성이 지연되고 있습니다",
      );
      expect(warning.textContent).toContain(
        "최종 채용 결정을 확정할 수 없습니다",
      );
    } finally {
      vi.useRealTimers();
    }
  });
});
