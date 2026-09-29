import { useCallback, useEffect, useState } from "react";
import {
  Navigate,
  useNavigate,
  useParams,
  useSearchParams,
} from "react-router-dom";

import {
  activateDemoCompanyAccess,
  beginCompanyLogin,
  beginCompanySignup,
  completeCompanyLogin,
  getCompanyAccessToken,
} from "../features/company/cognitoAuth";
import {
  CompanyAuthStatusView,
  CompanyAuthView,
} from "../features/company/CompanyAuthView";
import { AiRecruitingAssistant } from "../features/assistant";
import {
  ApplicantDetail,
  ApplicantManagement,
  CompanyOverview,
  CompanyPositions,
  PositionOperations,
  type CompanyApplicantRecruitingState,
  type CompanyApplicantInsight,
  type CompanyApplicantReport,
  type CompanyDeletionStatus,
  type CompanyOperationsApi,
} from "../features/company";
import {
  assessmentAxisKeys,
  type AxisWeightDraft,
  defaultAxisWeights,
  type InvitationStatus,
  HiringWorkspace,
  InvitationEmailSettings,
  type InvitationEmailTemplate,
  type InvitationEmailTemplateApi,
  type InvitationEmailTemplateState,
  type PositionInvitationApi,
  type HiringWorkspaceApi,
} from "../features/hiring";
import {
  ReviewWorkspace,
  type ReviewApi,
  type ReviewReport,
  type ReviewTimeline,
  type ScoreBreakdown,
} from "../features/review";
import {
  mockCompanyOperationsApi,
  mockInvitationEmailTemplateApi,
  mockPositionInvitationApi,
  mockRecruitingAssistantApi,
} from "../mocks/recruitingApi";
import type { components } from "@iep/contracts/generated/typescript/openapi";

import {
  CompanyRequestError,
  companyAuthConfig as AUTH_CONFIG,
  companyRequest,
  companyWorkspaceApi,
  idempotencyKey,
} from "./api/companyClient";
import {
  ASYNC_STATE,
  PAGE_EYEBROW_IN_HEADER,
  PAGE_HEADER,
  PAGE_HEADER_TEXT,
  PAGE_HEADER_TITLE,
} from "./styles/primitives";

const DEMO_COMPANY_EMAIL = import.meta.env.VITE_DEMO_COMPANY_EMAIL?.trim();
const DEMO_COMPANY_TOKEN = import.meta.env.VITE_DEMO_COMPANY_TOKEN?.trim();
const AUTOMATED_INTERVIEW_ENABLED =
  import.meta.env.DEV ||
  import.meta.env.VITE_AUTOMATED_INTERVIEW_ENABLED === "true";
const REPORT_DELAY_WARNING_MS = 8000;

function toPositionRequest(
  input: Parameters<HiringWorkspaceApi["createPosition"]>[0],
) {
  return {
    title: input.title,
    description: input.description,
    role_type: input.roleType,
    headcount: input.headcount,
    interview_capacity: input.interviewCapacity,
    interview_at: input.interviewAt
      ? new Date(input.interviewAt).toISOString()
      : undefined,
    recruitment_start_at: input.recruitmentStartAt,
    recruitment_end_at: input.recruitmentEndAt,
    submission_requirements: input.submissionRequirements.map(
      (requirement) => ({
        material_type: requirement.materialType,
        required: requirement.required,
        enabled: requirement.enabled,
        instructions: null,
      }),
    ),
  };
}

const hiringApi: HiringWorkspaceApi = {
  async createPosition(input) {
    const result = await companyRequest<components["schemas"]["Position"]>(
      "/v1/positions",
      {
        method: "POST",
        headers: { "Idempotency-Key": idempotencyKey("position") },
        body: JSON.stringify(toPositionRequest(input)),
      },
    );
    return {
      positionId: result.position_id,
      rowVersion: result.row_version,
    };
  },
  async publishCriteria(positionId, input) {
    const draft = await companyRequest<
      components["schemas"]["CompetencyModelVersion"]
    >(`/v1/positions/${positionId}/competency-model-versions`, {
      method: "POST",
      headers: { "Idempotency-Key": idempotencyKey("criteria") },
      body: JSON.stringify({
        job_requirements: input.jobRequirements.map((requirement) => ({
          requirement_type: requirement.requirementType,
          statement: requirement.statement,
          priority: requirement.priority,
          criterion_code: requirement.criterionCode,
        })),
        criteria: input.criteria.map((criterion) => ({
          code: criterion.code,
          name: criterion.name,
          description: criterion.description,
          weight: criterion.weight,
          verification_guide: {
            observable_dimensions:
              criterion.verificationGuide.observableDimensions,
            strong_answer_signals:
              criterion.verificationGuide.strongAnswerSignals,
            weak_answer_signals: criterion.verificationGuide.weakAnswerSignals,
            follow_up_directions:
              criterion.verificationGuide.followUpDirections,
            max_follow_ups: criterion.verificationGuide.maxFollowUps,
            time_budget_seconds: criterion.verificationGuide.timeBudgetSeconds,
          },
          abstain_guidance: criterion.abstainGuidance,
          common_questions: criterion.commonQuestions,
          required: criterion.required,
        })),
        prohibited_topics: input.prohibitedTopics,
        interview_duration_minutes: 30,
        interview_level: input.interviewLevel,
        // All five axes, always. The API accepts an empty mapping as equal weight for versions
        // published before weights existed, but refuses a partial one — no reading of the
        // absent keys is anything but a silently wrong score.
        axis_weights: input.axisWeights,
        persona_definition: {
          name: input.personaDefinition.name,
          tone: input.personaDefinition.tone,
          voice_id: input.personaDefinition.voiceId,
        },
      }),
    });
    const published = await companyRequest<
      components["schemas"]["CompetencyModelVersion"]
    >(
      `/v1/competency-model-versions/${draft.competency_model_version_id}/publish`,
      {
        method: "POST",
        headers: {
          "Idempotency-Key": idempotencyKey("criteria-publish"),
          "If-Match-Version": String(draft.row_version),
        },
      },
    );
    return { versionId: published.competency_model_version_id };
  },
  async activatePosition(positionId, rowVersion, input) {
    await companyRequest<components["schemas"]["Position"]>(
      `/v1/positions/${positionId}`,
      {
        method: "PATCH",
        headers: { "If-Match-Version": String(rowVersion) },
        body: JSON.stringify({
          ...toPositionRequest(input),
          status: "active",
        }),
      },
    );
  },
};

const positionInvitationApi: PositionInvitationApi = {
  async listInvitations(positionId) {
    const items: components["schemas"]["InvitationView"][] = [];
    let cursor: string | null = null;
    do {
      const query = new URLSearchParams({ limit: "500" });
      if (cursor) query.set("cursor", cursor);
      const page = await companyRequest<
        components["schemas"]["InvitationPage"]
      >(`/v1/positions/${positionId}/invitations?${query}`);
      items.push(...page.items);
      cursor = page.next_cursor ?? null;
    } while (cursor);
    return items.map(toCompanyInvitation);
  },
  async createInvitations(
    positionId,
    applicants,
    expiresInDays,
    deliveryMethod = "email",
  ) {
    const result = await companyRequest<
      components["schemas"]["InvitationBatchResult"]
    >(`/v1/positions/${positionId}/invitations`, {
      method: "POST",
      headers: { "Idempotency-Key": idempotencyKey("invitation-batch") },
      body: JSON.stringify({
        applicants: applicants.map((applicant) => ({
          email: applicant.email,
          display_name: applicant.displayName,
        })),
        expires_at: new Date(
          Date.now() + expiresInDays * 86_400_000,
        ).toISOString(),
        delivery_method: deliveryMethod,
      }),
    });
    return {
      acceptedCount: result.accepted_count,
      rejectedCount: result.rejected_count,
      accessLinks: result.access_links.map((link) => ({
        invitationId: link.invitation_id,
        applicantEmail: link.applicant_email,
        applicantDisplayName: link.applicant_display_name,
        accessUrl: link.access_url,
        expiresAt: link.expires_at,
      })),
      invitations: result.invitations.map((invitation) => ({
        invitationId: invitation.invitation_id,
        positionId: invitation.position_id,
        competencyModelVersionId: invitation.competency_model_version_id,
        applicantEmail: invitation.applicant_email,
        applicantDisplayName: invitation.applicant_display_name,
        status: invitation.status as InvitationStatus,
        expiresAt: invitation.expires_at,
        rowVersion: invitation.row_version,
        recruitingStageId: invitation.recruiting_stage_id,
        pipelineRowVersion: invitation.pipeline_row_version,
      })),
    };
  },
};

type TemplateResponse = components["schemas"]["InvitationEmailTemplateView"];

function toTemplateState(
  response: TemplateResponse,
): InvitationEmailTemplateState {
  return {
    subject: response.subject,
    headline: response.headline,
    intro: response.intro,
    guides: response.guides,
    ctaLabel: response.cta_label,
    outro: response.outro,
    footer: response.footer,
    brandColor: response.brand_color,
    useApplicantName: response.use_applicant_name,
    emphasizeDeadline: response.emphasize_deadline,
    showSecurityNotice: response.show_security_notice,
    logoUrl: response.logo_url ?? null,
    isPositionOverride: response.is_position_override,
  };
}

function toTemplateBody(template: InvitationEmailTemplate) {
  // logo_url is deliberately not sent: the server derives it from the uploaded logo.
  return JSON.stringify({
    subject: template.subject,
    headline: template.headline,
    intro: template.intro,
    guides: template.guides,
    cta_label: template.ctaLabel,
    outro: template.outro,
    footer: template.footer,
    brand_color: template.brandColor,
    use_applicant_name: template.useApplicantName,
    emphasize_deadline: template.emphasizeDeadline,
    show_security_notice: template.showSecurityNotice,
  });
}

const invitationEmailTemplateApi: InvitationEmailTemplateApi = {
  async getCompanyTemplate() {
    return toTemplateState(
      await companyRequest<TemplateResponse>("/v1/invitation-email-template"),
    );
  },
  async saveCompanyTemplate(template) {
    return toTemplateState(
      await companyRequest<TemplateResponse>("/v1/invitation-email-template", {
        method: "PUT",
        body: toTemplateBody(template),
      }),
    );
  },
  async resetCompanyTemplate() {
    return toTemplateState(
      await companyRequest<TemplateResponse>("/v1/invitation-email-template", {
        method: "DELETE",
      }),
    );
  },
  async getPositionTemplate(positionId) {
    return toTemplateState(
      await companyRequest<TemplateResponse>(
        `/v1/positions/${positionId}/invitation-email-template`,
      ),
    );
  },
  async savePositionTemplate(positionId, template) {
    return toTemplateState(
      await companyRequest<TemplateResponse>(
        `/v1/positions/${positionId}/invitation-email-template`,
        { method: "PUT", body: toTemplateBody(template) },
      ),
    );
  },
  async resetPositionTemplate(positionId) {
    return toTemplateState(
      await companyRequest<TemplateResponse>(
        `/v1/positions/${positionId}/invitation-email-template`,
        { method: "DELETE" },
      ),
    );
  },
  async previewTemplate(template) {
    const result = await companyRequest<
      components["schemas"]["InvitationEmailPreview"]
    >("/v1/invitation-email-template/preview", {
      method: "POST",
      body: toTemplateBody(template),
    });
    return { subject: result.subject, htmlBody: result.html_body };
  },
  async uploadLogo(file) {
    const result = await companyRequest<
      components["schemas"]["CompanyLogoView"]
    >("/v1/invitation-email-template/logo", {
      method: "PUT",
      headers: { "Content-Type": file.type },
      body: file,
    });
    return {
      logoUrl: result.logo_url,
      contentType: result.content_type,
      byteSize: result.byte_size,
    };
  },
  async deleteLogo() {
    await companyRequest("/v1/invitation-email-template/logo", {
      method: "DELETE",
    });
  },
};

const companyOperationsApi: CompanyOperationsApi = {
  ...companyWorkspaceApi,
  listInvitations: positionInvitationApi.listInvitations,
  async listRecruitingStages(positionId) {
    const query = positionId
      ? `?${new URLSearchParams({ position_id: positionId })}`
      : "";
    const result = await companyRequest<
      components["schemas"]["RecruitingStagePage"]
    >(`/v1/recruiting-stages${query}`);
    return result.items.map(toCompanyRecruitingStage);
  },
  async getApplicantRecruitingState(invitationId) {
    const result = await companyRequest<
      components["schemas"]["ApplicantRecruitingState"]
    >(`/v1/invitations/${invitationId}/recruiting-state`);
    return {
      invitationId: result.invitation_id,
      positionId: result.position_id,
      recruitingStageId: result.recruiting_stage_id,
      pipelineRowVersion: result.pipeline_row_version,
      stages: result.stages.map(toCompanyRecruitingStage),
    };
  },
  async createRecruitingStage(positionId, name) {
    const result = await companyRequest<
      components["schemas"]["RecruitingStage"]
    >(`/v1/positions/${positionId}/recruiting-stages`, {
      method: "POST",
      body: JSON.stringify({ name }),
    });
    return toCompanyRecruitingStage(result);
  },
  async updateRecruitingStage(positionId, stageId, name, rowVersion) {
    const result = await companyRequest<
      components["schemas"]["RecruitingStage"]
    >(`/v1/positions/${positionId}/recruiting-stages/${stageId}`, {
      method: "PATCH",
      headers: { "If-Match-Version": String(rowVersion) },
      body: JSON.stringify({ name }),
    });
    return toCompanyRecruitingStage(result);
  },
  async reorderRecruitingStages(positionId, orderedStageIds) {
    const result = await companyRequest<
      components["schemas"]["RecruitingStagePage"]
    >(`/v1/positions/${positionId}/recruiting-stages/reorder`, {
      method: "POST",
      body: JSON.stringify({ ordered_stage_ids: orderedStageIds }),
    });
    return result.items.map(toCompanyRecruitingStage);
  },
  async deleteRecruitingStage(positionId, stageId, replacementStageId) {
    const result = await companyRequest<
      components["schemas"]["RecruitingStagePage"]
    >(`/v1/positions/${positionId}/recruiting-stages/${stageId}/delete`, {
      method: "POST",
      body: JSON.stringify({ replacement_stage_id: replacementStageId }),
    });
    return result.items.map(toCompanyRecruitingStage);
  },
  async moveApplicantsToRecruitingStage(positionId, targetStageId, applicants) {
    const result = await companyRequest<
      components["schemas"]["ApplicantPipelineAssignmentPage"]
    >(`/v1/positions/${positionId}/invitations/recruiting-stage`, {
      method: "PATCH",
      body: JSON.stringify({
        target_stage_id: targetStageId,
        applicants: applicants.map((applicant) => ({
          invitation_id: applicant.invitationId,
          expected_version: applicant.expectedVersion,
        })),
      }),
    });
    return result.items.map((assignment) => ({
      invitationId: assignment.invitation_id,
      recruitingStageId: assignment.recruiting_stage_id,
      pipelineRowVersion: assignment.pipeline_row_version,
    }));
  },
  async requestApplicantDeletion(invitationId) {
    try {
      const result = await companyRequest<
        components["schemas"]["DeletionStatus"]
      >("/v1/privacy/deletion-requests", {
        method: "POST",
        headers: { "Idempotency-Key": idempotencyKey("applicant-deletion") },
        body: JSON.stringify({
          scope_type: "invitation",
          scope_id: invitationId,
          reason: "company_user_requested_applicant_deletion",
        }),
      });
      return toCompanyDeletionStatus(result);
    } catch (error) {
      if (error instanceof CompanyRequestError && error.status === 404) {
        return {
          deletionRequestId: `already-deleted-${invitationId}`,
          status: "completed",
          expectedTargets: 0,
          verifiedTargets: 0,
        };
      }
      throw error;
    }
  },
  async getApplicantDeletion(deletionRequestId) {
    try {
      const result = await companyRequest<
        components["schemas"]["DeletionStatus"]
      >(`/v1/privacy/deletion-requests/${deletionRequestId}`);
      return toCompanyDeletionStatus(result);
    } catch (error) {
      if (error instanceof CompanyRequestError && error.status === 404) {
        return {
          deletionRequestId,
          status: "completed",
          expectedTargets: 0,
          verifiedTargets: 0,
        };
      }
      throw error;
    }
  },
  async updatePosition(input) {
    const result = await companyRequest<components["schemas"]["Position"]>(
      `/v1/positions/${input.positionId}`,
      {
        method: "PATCH",
        headers: { "If-Match-Version": String(input.rowVersion) },
        body: JSON.stringify({
          title: input.title,
          description: input.description,
          role_type: input.roleType ?? null,
          headcount: input.headcount ?? null,
          applicant_capacity: input.applicantCapacity ?? null,
          interview_capacity: input.interviewCapacity ?? null,
          interview_at: input.interviewAt
            ? new Date(input.interviewAt).toISOString()
            : null,
          recruitment_start_at: input.recruitmentStartAt ?? null,
          recruitment_end_at: input.recruitmentEndAt ?? null,
          submission_requirements: input.submissionRequirements.map(
            (requirement) => ({
              material_type: requirement.materialType,
              required: requirement.required,
              enabled: requirement.enabled,
              instructions: requirement.instructions ?? null,
            }),
          ),
          status: input.status,
        }),
      },
    );
    return {
      positionId: result.position_id,
      title: result.title,
      description: result.description,
      roleType: result.role_type,
      headcount: result.headcount,
      applicantCapacity: result.applicant_capacity,
      interviewCapacity: result.interview_capacity,
      interviewAt: result.interview_at,
      recruitmentStartAt: result.recruitment_start_at,
      recruitmentEndAt: result.recruitment_end_at,
      submissionRequirements: (
        result as typeof result & {
          submission_requirements: Array<{
            material_type:
              | "resume"
              | "cover_letter"
              | "career_description"
              | "projects"
              | "portfolio";
            required: boolean;
            enabled: boolean;
            instructions?: string | null;
          }>;
        }
      ).submission_requirements.map((requirement) => ({
        materialType: requirement.material_type,
        required: requirement.required,
        enabled: requirement.enabled,
        instructions: requirement.instructions,
      })),
      status: result.status,
      rowVersion: result.row_version,
      createdAt: result.created_at,
    };
  },
  async listCriterionVersions(positionId) {
    const result = await companyRequest<
      components["schemas"]["CompetencyModelVersionPage"]
    >(`/v1/positions/${positionId}/competency-model-versions?limit=100`);
    return result.items.map((version) => ({
      versionId: version.competency_model_version_id,
      positionId: version.position_id,
      versionNumber: version.version_number,
      status: version.status,
      rowVersion: version.row_version,
      publishedAt: version.published_at,
      jobRequirements: version.job_requirements.map((requirement) => ({
        requirementType: requirement.requirement_type,
        statement: requirement.statement,
        priority: requirement.priority,
        criterionCode: requirement.criterion_code,
      })),
      criteria: version.criteria.map((criterion) => ({
        code: criterion.code,
        name: criterion.name,
        description: criterion.description,
        weight: criterion.weight,
        required: criterion.required,
        verificationGuide: {
          observableDimensions:
            criterion.verification_guide.observable_dimensions,
          strongAnswerSignals:
            criterion.verification_guide.strong_answer_signals,
          weakAnswerSignals: criterion.verification_guide.weak_answer_signals,
          followUpDirections: criterion.verification_guide.follow_up_directions,
          maxFollowUps: criterion.verification_guide.max_follow_ups,
          timeBudgetSeconds: criterion.verification_guide.time_budget_seconds,
        },
        abstainGuidance: criterion.abstain_guidance,
        commonQuestions: criterion.common_questions ?? [],
      })),
      prohibitedTopics: version.prohibited_topics,
      interviewDurationMinutes: 30,
      // Versions published before the difficulty toggle existed omit the field.
      interviewLevel: version.interview_level ?? "junior",
      // Versions published before axis weights existed carry no mapping, and were scored with
      // every axis counting the same. Filling the equal defaults in here shows the recruiter the
      // weighting those reports actually used rather than a set of blank fields.
      axisWeights: readAxisWeights(version.axis_weights),
      personaDefinition: toCompanyPersona(version.persona_definition),
    }));
  },
  async listSubmissions(invitationId) {
    const result = await companyRequest<
      Array<{
        submission_id: string;
        material_type:
          | "resume"
          | "cover_letter"
          | "career_description"
          | "projects"
          | "portfolio";
        source_type: string;
        original_filename: string | null;
        source_url: string | null;
        status: string;
        failure_code: string | null;
        impact_summary: string | null;
        created_at: string;
      }>
    >(`/v1/company/invitations/${invitationId}/submissions`);
    return result.map((submission) => ({
      submissionId: submission.submission_id,
      materialType: submission.material_type,
      sourceType: submission.source_type,
      originalFilename: submission.original_filename,
      sourceUrl: submission.source_url,
      status: submission.status,
      failureCode: submission.failure_code,
      impactSummary: submission.impact_summary,
      createdAt: submission.created_at,
    }));
  },
  async listApplicantInsights(positionId) {
    const invitations = await positionInvitationApi.listInvitations(positionId);
    const completed = invitations.filter(
      (invitation) => invitation.interviewSessionId,
    );
    const reports = await Promise.allSettled(
      completed.map(async (invitation) => {
        const sessionId = invitation.interviewSessionId;
        if (!sessionId) return null;
        const report = await companyRequest<ReportResponse>(
          `/v1/interview-sessions/${sessionId}/report`,
        );
        return toApplicantInsight(
          invitation.invitationId,
          sessionId,
          invitation.competencyModelVersionId,
          toReviewReport(report),
        );
      }),
    );
    return reports.flatMap((result) =>
      result.status === "fulfilled" && result.value ? [result.value] : [],
    );
  },
  async getApplicantReport(
    interviewSessionId,
    invitationId,
    competencyModelVersionId,
  ) {
    const [report, timeline] = await Promise.all([
      companyRequest<ReportResponse>(
        `/v1/interview-sessions/${interviewSessionId}/report`,
      ),
      companyRequest<TimelineResponse>(
        `/v1/interview-sessions/${interviewSessionId}/timeline`,
      ),
    ]);
    const mappedReport = toReviewReport(report);
    return {
      insight: toApplicantInsight(
        invitationId,
        interviewSessionId,
        competencyModelVersionId,
        mappedReport,
      ),
      report: mappedReport,
      timeline: toReviewTimeline(timeline),
    } satisfies CompanyApplicantReport;
  },
  publishCriteria: hiringApi.publishCriteria,
};

function toCompanyInvitation(
  invitation: components["schemas"]["InvitationView"],
) {
  return {
    invitationId: invitation.invitation_id,
    positionId: invitation.position_id,
    competencyModelVersionId: invitation.competency_model_version_id,
    applicantEmail: invitation.applicant_email,
    applicantDisplayName: invitation.applicant_display_name,
    status: invitation.status as InvitationStatus,
    expiresAt: invitation.expires_at,
    rowVersion: invitation.row_version,
    recruitingStageId: invitation.recruiting_stage_id,
    pipelineRowVersion: invitation.pipeline_row_version,
    analysisStatus: invitation.analysis_status,
    interviewStatus: invitation.interview_status,
    reportStatus: invitation.report_status,
    interviewSessionId: invitation.interview_session_id,
    overallScore: invitation.overall_score ?? null,
    scoredCriteriaCount: invitation.scored_criteria_count ?? null,
    totalCriteriaCount: invitation.total_criteria_count ?? null,
  };
}

function toCompanyRecruitingStage(
  stage: components["schemas"]["RecruitingStage"],
) {
  return {
    recruitingStageId: stage.recruiting_stage_id,
    positionId: stage.position_id,
    name: stage.name,
    sortOrder: stage.sort_order,
    rowVersion: stage.row_version,
  };
}

function toCompanyDeletionStatus(
  result: components["schemas"]["DeletionStatus"],
): CompanyDeletionStatus {
  return {
    deletionRequestId: result.deletion_request_id,
    status: result.status,
    expectedTargets: result.expected_targets,
    verifiedTargets: result.verified_targets,
  };
}

/**
 * Read a published version's axis weights into the shape the wizard edits.
 *
 * An absent or empty mapping is a version published before weights existed, and those interviews
 * were scored with every axis counting the same — so the equal defaults are what that version
 * actually used, not a placeholder. A key the server somehow omitted falls back the same way
 * rather than rendering an empty field the recruiter would have to guess at.
 */
function readAxisWeights(
  weights: Readonly<Partial<Record<string, number>>> | undefined,
): AxisWeightDraft {
  const draft = { ...defaultAxisWeights };
  if (!weights) return draft;
  for (const key of assessmentAxisKeys) {
    const value = weights[key];
    if (typeof value === "number" && Number.isFinite(value) && value >= 0) {
      draft[key] = value;
    }
  }
  return draft;
}

function toCompanyPersona(value: unknown) {
  if (!value || typeof value !== "object") return undefined;
  const persona = value as Record<string, unknown>;
  const tone = persona.tone;
  if (
    typeof persona.name !== "string" ||
    typeof persona.voice_id !== "string" ||
    !["calm", "friendly", "analytical", "concise"].includes(String(tone))
  ) {
    return undefined;
  }
  return {
    name: persona.name,
    tone: tone as "calm" | "friendly" | "analytical" | "concise",
    voiceId: persona.voice_id,
  };
}

/**
 * Read a score's arithmetic, or null when the report predates it.
 *
 * Null rather than a zeroed breakdown: an empty calculator claiming `0 ÷ 0` would read as a
 * finding, when the truth is that this report was generated before the arithmetic was recorded
 * and its score is a plain mean.
 */
function toScoreBreakdown(
  breakdown: components["schemas"]["ScoreBreakdown"] | undefined,
): ScoreBreakdown | null {
  if (!breakdown) return null;
  return {
    numerator: breakdown.numerator,
    denominator: breakdown.denominator,
    contributions: breakdown.contributions.map((contribution) => ({
      key: contribution.key,
      score: contribution.score,
      weight: contribution.weight,
      normalizedWeight: contribution.normalized_weight,
      contribution: contribution.contribution,
      criterionName: contribution.criterion_name ?? null,
      assessmentState: contribution.assessment_state ?? null,
      reason: contribution.reason ?? null,
    })),
    exclusions: breakdown.exclusions.map((exclusion) => ({
      key: exclusion.key,
      weight: exclusion.weight,
      normalizedWeight: exclusion.normalized_weight,
      criterionName: exclusion.criterion_name ?? null,
      assessmentState: exclusion.assessment_state ?? null,
      reason: exclusion.reason ?? null,
    })),
  };
}

function toReviewReport(report: ReportResponse): ReviewReport {
  return {
    summary: report.summary,
    status: report.status,
    overallScore: report.overall_score ?? null,
    communicationScore: report.communication_score ?? null,
    communicationScoredCriteriaCount:
      report.communication_scored_criteria_count ?? 0,
    unscoredCriteriaCount: report.unscored_criteria_count ?? 0,
    scoringBreakdown: toScoreBreakdown(report.scoring_breakdown),
    requirementAssessments: (report.requirement_assessments ?? []).map(
      (item) => ({
        requirementAssessmentId: item.requirement_assessment_id,
        jobRequirementId: item.job_requirement_id,
        requirementType: item.requirement_type,
        statement: item.statement,
        status: item.status,
        rationale: item.rationale,
        confidence: item.confidence,
        evidence: item.evidence.map((evidence) => ({
          evidenceId: evidence.evidence_id,
          sourceKind: evidence.source_kind,
          sourceType: evidence.source_type,
          excerpt: evidence.excerpt,
          locator: evidence.locator,
          relation: evidence.relation,
          explanation: evidence.explanation,
        })),
        humanOverride: item.human_override
          ? {
              status: item.human_override.status,
              reason: item.human_override.reason ?? null,
              createdAt: item.human_override.created_at,
            }
          : null,
      }),
    ),
    items: report.items.map((item) => ({
      reportItemId: item.report_item_id,
      criterionId: item.criterion_id,
      criterionName: item.criterion_name || item.criterion_id,
      assessmentState: item.assessment_state,
      observation: item.observation,
      followUpQuestion: item.follow_up_question ?? null,
      averageScore: item.average_score ?? null,
      // 1, not 0: a report from before weights existed counted every criterion equally, and
      // zero would drop it out of any arithmetic the console does with these.
      criterionWeight: item.criterion_weight ?? 1,
      axisBreakdown: toScoreBreakdown(item.axis_breakdown),
      axisAssessments: (item.axis_assessments ?? []).map((axis) => ({
        axis: axis.axis,
        label: axis.label,
        score: axis.score,
        rationale: axis.rationale,
        quotedEvidenceIds: [...axis.quoted_evidence_ids],
        weight: axis.weight ?? null,
      })),
      evidence: item.evidence.map((evidence) => ({
        evidenceId: evidence.evidence_id,
        answerTurnId: evidence.answer_turn_id,
        transcriptSegmentId: evidence.transcript_segment_id,
        startMs: evidence.video_start_ms,
        endMs: evidence.video_end_ms,
        observation: evidence.observation,
        rationale: evidence.rationale,
        sufficiency: evidence.sufficiency,
      })),
    })),
  };
}

function toReviewTimeline(timeline: TimelineResponse): ReviewTimeline {
  return {
    entries: timeline.entries.map((entry) => ({
      entryId: entry.entry_id,
      type: entry.entry_type,
      startMs: entry.start_ms,
      endMs: entry.end_ms,
      text: entry.text ?? null,
      questionRationale: entry.question_rationale
        ? {
            criterionId: entry.question_rationale.criterion_id,
            interviewStage: entry.question_rationale.interview_stage,
            verificationTargetType:
              entry.question_rationale.verification_target_type,
            objective: entry.question_rationale.objective,
            questionType: entry.question_rationale.question_type,
            policyResult: entry.question_rationale.policy_result,
            sourceReferences: entry.question_rationale.source_references.map(
              (source) => ({
                sourceId: source.source_id,
                sourceType: source.source_type,
                locator: source.locator,
                excerpt: source.excerpt,
              }),
            ),
          }
        : null,
    })),
    playback: {
      status: timeline.playback.status,
      url: timeline.playback.url ?? undefined,
    },
  };
}

function toApplicantInsight(
  invitationId: string,
  interviewSessionId: string,
  competencyModelVersionId: string,
  report: ReviewReport,
): CompanyApplicantInsight {
  const criteria = report.items.map((item) => ({
    criterionId: item.criterionId,
    criterionName: item.criterionName,
    score: item.averageScore,
    assessmentState: item.assessmentState,
    evidenceCount: item.evidence.length,
  }));
  return {
    invitationId,
    interviewSessionId,
    competencyModelVersionId,
    overallScore: report.overallScore,
    unscoredCriteriaCount: report.unscoredCriteriaCount,
    evidenceCoverage: criteria.length
      ? Math.round(
          (criteria.filter((criterion) => criterion.evidenceCount > 0).length /
            criteria.length) *
            100,
        )
      : 0,
    summary: report.summary,
    criteria,
  };
}

const useMockRecruitingData =
  import.meta.env.DEV && import.meta.env.VITE_USE_MOCK_DATA === "true";
const recruitingOperationsApi = useMockRecruitingData
  ? mockCompanyOperationsApi
  : companyOperationsApi;
const recruitingInvitationApi = useMockRecruitingData
  ? mockPositionInvitationApi
  : positionInvitationApi;
const recruitingTemplateApi = useMockRecruitingData
  ? mockInvitationEmailTemplateApi
  : invitationEmailTemplateApi;

type ReportResponse = components["schemas"]["ReportView"];

type PendingReportResponse = Readonly<{
  status: "queued";
  retryable: boolean;
  message: string | null;
}>;

type FailedReportResponse = Readonly<{
  status: "failed";
  retryable: false;
  message: string;
}>;

type TimelineResponse = components["schemas"]["TimelineView"];

type FinalDecisionResponse = components["schemas"]["FinalDecisionView"];

export function CompanyHomeRoute() {
  if (AUTH_CONFIG && !getCompanyAccessToken(localStorage)) {
    return <Navigate replace to="/auth/login" />;
  }
  return <CompanyOverview api={recruitingOperationsApi} />;
}

export function CompanyPositionsRoute() {
  if (AUTH_CONFIG && !getCompanyAccessToken(localStorage)) {
    return <Navigate replace to="/auth/login" />;
  }
  return <CompanyPositions api={recruitingOperationsApi} />;
}

export function PositionOperationsRoute() {
  const { positionId = "" } = useParams();
  if (AUTH_CONFIG && !getCompanyAccessToken(localStorage)) {
    return <Navigate replace to="/auth/login" />;
  }
  if (!positionId) {
    return <Navigate replace to="/positions" />;
  }
  return (
    <PositionOperations
      positionId={positionId}
      api={recruitingOperationsApi}
      invitationApi={recruitingInvitationApi}
      templateApi={recruitingTemplateApi}
    />
  );
}

export function InvitationEmailSettingsRoute() {
  if (AUTH_CONFIG && !getCompanyAccessToken(localStorage)) {
    return <Navigate replace to="/auth/login" />;
  }
  return <InvitationEmailSettings api={invitationEmailTemplateApi} />;
}

export function ApplicantManagementRoute() {
  if (AUTH_CONFIG && !getCompanyAccessToken(localStorage)) {
    return <Navigate replace to="/auth/login" />;
  }
  return <ApplicantManagement api={recruitingOperationsApi} />;
}

export function AiRecruitingAssistantRoute() {
  if (AUTH_CONFIG && !getCompanyAccessToken(localStorage)) {
    return <Navigate replace to="/auth/login" />;
  }
  return (
    <AiRecruitingAssistant
      api={recruitingOperationsApi}
      assistantApi={
        useMockRecruitingData ? mockRecruitingAssistantApi : undefined
      }
    />
  );
}

export function ApplicantDetailRoute() {
  const { positionId = "", invitationId = "" } = useParams();
  if (AUTH_CONFIG && !getCompanyAccessToken(localStorage)) {
    return <Navigate replace to="/auth/login" />;
  }
  if (!positionId || !invitationId) {
    return <Navigate replace to="/applicants" />;
  }
  return (
    <ApplicantDetail
      positionId={positionId}
      invitationId={invitationId}
      api={recruitingOperationsApi}
    />
  );
}

export function HiringRoute() {
  const navigate = useNavigate();
  if (AUTH_CONFIG && !getCompanyAccessToken(localStorage)) {
    return <Navigate replace to="/auth/login" />;
  }
  return (
    <HiringWorkspace
      api={hiringApi}
      onOpenPosition={(positionId) => navigate(`/positions/${positionId}`)}
    />
  );
}

export function ReviewRoute() {
  const { sessionId = "" } = useParams();
  const [search] = useSearchParams();
  const invitationId = search.get("invitationId") ?? "";
  const automatedReview =
    AUTOMATED_INTERVIEW_ENABLED && search.get("auto") === "1";
  const [report, setReport] = useState<ReportResponse | null>(null);
  const [timeline, setTimeline] = useState<TimelineResponse | null>(null);
  const [recruitingState, setRecruitingState] =
    useState<CompanyApplicantRecruitingState | null>(null);
  const [reportPending, setReportPending] = useState(false);
  const [reportDelayed, setReportDelayed] = useState(false);
  const [reportFailure, setReportFailure] = useState<string | null>(null);
  const [error, setError] = useState(false);
  const authenticated =
    !AUTH_CONFIG || Boolean(getCompanyAccessToken(localStorage));

  const fetchRecruitingState = useCallback(async () => {
    if (!authenticated || !invitationId) return null;
    const getRecruitingState = companyOperationsApi.getApplicantRecruitingState;
    if (!getRecruitingState) return null;
    return getRecruitingState(invitationId);
  }, [authenticated, invitationId]);

  useEffect(() => {
    if (!authenticated) return;
    let active = true;
    let retryTimer: number | undefined;
    let delayTimer: number | undefined;

    function scheduleDelayWarning() {
      if (delayTimer !== undefined) return;
      delayTimer = window.setTimeout(() => {
        if (active) setReportDelayed(true);
      }, REPORT_DELAY_WARNING_MS);
    }

    async function loadReview() {
      try {
        const [nextReport, nextTimeline] = await Promise.all([
          companyRequest<ReportResponse | PendingReportResponse | FailedReportResponse>(
            `/v1/interview-sessions/${sessionId}/report`,
          ),
          companyRequest<TimelineResponse>(
            `/v1/interview-sessions/${sessionId}/timeline`,
          ),
        ]);
        if (!active) return;
        if (isFailedReport(nextReport)) {
          setReport(null);
          setTimeline(nextTimeline);
          setReportPending(false);
          setReportDelayed(false);
          setReportFailure(nextReport.message);
          setError(false);
          if (delayTimer !== undefined) window.clearTimeout(delayTimer);
          return;
        }
        if (!isPendingReport(nextReport)) {
          setReport(nextReport);
          setTimeline(nextTimeline);
          setReportPending(false);
          setReportDelayed(false);
          setReportFailure(null);
          setError(false);
          if (delayTimer !== undefined) window.clearTimeout(delayTimer);
          return;
        }
        setReportFailure(null);
        setReportPending(true);
        scheduleDelayWarning();
        retryTimer = window.setTimeout(() => void loadReview(), 2000);
      } catch {
        if (!active) return;
        if (automatedReview) {
          setReportPending(true);
          scheduleDelayWarning();
          retryTimer = window.setTimeout(() => void loadReview(), 2000);
          return;
        }
        setReportPending(false);
        setReportDelayed(false);
        setReportFailure(null);
        setError(true);
      }
    }

    void loadReview();
    return () => {
      active = false;
      if (retryTimer !== undefined) window.clearTimeout(retryTimer);
      if (delayTimer !== undefined) window.clearTimeout(delayTimer);
    };
  }, [authenticated, automatedReview, sessionId]);

  useEffect(() => {
    let active = true;

    async function refresh() {
      try {
        const state = await fetchRecruitingState();
        if (active && state) setRecruitingState(state);
      } catch {
        if (active) setRecruitingState(null);
      }
    }

    function refreshWhenVisible() {
      if (document.visibilityState === "visible") void refresh();
    }

    void refresh();
    window.addEventListener("focus", refresh);
    document.addEventListener("visibilitychange", refreshWhenVisible);
    return () => {
      active = false;
      window.removeEventListener("focus", refresh);
      document.removeEventListener("visibilitychange", refreshWhenVisible);
    };
  }, [fetchRecruitingState]);

  if (!authenticated) {
    return <Navigate replace to="/auth/login" />;
  }

  const reviewApi: ReviewApi = {
    async overrideAssessment(reportItemId, assessmentState, reason) {
      if (!report) return;
      await companyRequest(
        `/v1/reports/${report.report_id}/items/${reportItemId}/reviews`,
        {
          method: "POST",
          headers: { "Idempotency-Key": idempotencyKey("override") },
          body: JSON.stringify({
            assessment_state: assessmentState,
            reason,
          }),
        },
      );
    },
    async overrideRequirement(
      requirementAssessmentId,
      requirementStatus,
      reason,
    ) {
      if (!report) return;
      await companyRequest(
        `/v1/reports/${report.report_id}/requirements/${requirementAssessmentId}/reviews`,
        {
          method: "POST",
          headers: {
            "Idempotency-Key": idempotencyKey("requirement-override"),
          },
          body: JSON.stringify({
            requirement_status: requirementStatus,
            reason,
          }),
        },
      );
    },
    async addNote(targetId, value) {
      await companyRequest(
        `/v1/interview-sessions/${sessionId}/review-artifacts`,
        {
          method: "POST",
          headers: { "Idempotency-Key": idempotencyKey("note") },
          body: JSON.stringify({
            review_type: "note",
            target_id: targetId,
            value,
          }),
        },
      );
    },
    async saveFinalDecisionStage(stageId) {
      if (!invitationId || !recruitingState) {
        throw new Error("recruiting state is not ready");
      }
      let result: FinalDecisionResponse;
      try {
        result = await companyRequest<FinalDecisionResponse>(
          `/v1/invitations/${invitationId}/final-decisions`,
          {
            method: "POST",
            headers: { "Idempotency-Key": idempotencyKey("final-decision") },
            body: JSON.stringify({
              recruiting_stage_id: stageId,
              expected_pipeline_version: recruitingState.pipelineRowVersion,
            }),
          },
        );
      } catch (cause) {
        if (cause instanceof CompanyRequestError && cause.status === 409) {
          try {
            const latest = await fetchRecruitingState();
            if (latest) setRecruitingState(latest);
          } catch {
            // Preserve the actionable conflict from the write; a later focus retries the read.
          }
        }
        throw cause;
      }
      setRecruitingState((current) =>
        current
          ? {
              ...current,
              recruitingStageId: result.recruiting_stage_id,
              pipelineRowVersion: result.pipeline_row_version,
            }
          : current,
      );
    },
  };

  return (
    <>
      {!report || !timeline ? (
        <section className="min-w-0">
          {/* `.review-workspace .page-header { display: none }` under `@media print`. */}
          <header className={`${PAGE_HEADER} print:hidden`}>
            <div>
              <p className={PAGE_EYEBROW_IN_HEADER}>Interview evidence</p>
              <h1 className={PAGE_HEADER_TITLE}>지원자 검토</h1>
              <p className={PAGE_HEADER_TEXT}>
                AI 분석과 실제 답변 구간을 불러오고 있습니다.
              </p>
            </div>
          </header>
          <div
            className={ASYNC_STATE}
            role={error || reportDelayed || reportFailure ? "alert" : "status"}
            data-report-state={
              reportFailure
                ? "failed"
                : error
                ? "error"
                : reportDelayed
                  ? "delayed"
                  : reportPending
                    ? "pending"
                    : "loading"
            }
          >
            <p className="text-[12px]">
              {reportFailure
                ? reportFailure
                : error
                ? "리포트를 불러올 수 없습니다. 잠시 후 다시 시도해 주세요."
                : reportDelayed
                  ? "최종 리포트 생성이 지연되고 있습니다. 리포트가 준비될 때까지 최종 채용 결정을 확정할 수 없습니다."
                  : automatedReview || reportPending
                    ? `${automatedReview ? "자동 면접" : "면접"}이 끝났습니다. 최종 리포트를 생성하고 있습니다.`
                    : "리포트와 영상 타임라인을 불러오는 중입니다."}
            </p>
          </div>
        </section>
      ) : (
        <ReviewWorkspace
          sessionId={sessionId}
          invitationId={invitationId}
          api={reviewApi}
          report={toReviewReport(report)}
          timeline={toReviewTimeline(timeline)}
          recruitingState={recruitingState}
        />
      )}
    </>
  );
}

function isPendingReport(
  report: ReportResponse | PendingReportResponse | FailedReportResponse,
): report is PendingReportResponse {
  return (
    "retryable" in report && report.status === "queued" && report.retryable
  );
}

function isFailedReport(
  report: ReportResponse | PendingReportResponse | FailedReportResponse,
): report is FailedReportResponse {
  return "retryable" in report && report.status === "failed" && !report.retryable;
}

export function CompanyLoginRoute() {
  const navigate = useNavigate();
  const [error, setError] = useState(false);

  async function login() {
    if (!AUTH_CONFIG) {
      navigate("/company", { replace: true });
      return;
    }
    try {
      await beginCompanyLogin(AUTH_CONFIG, {
        sessionStorage,
        navigate: (location) => window.location.assign(location),
      });
    } catch {
      setError(true);
    }
  }

  async function demoLogin() {
    if (DEMO_COMPANY_TOKEN) {
      activateDemoCompanyAccess(localStorage, DEMO_COMPANY_TOKEN);
      navigate("/company", { replace: true });
      return;
    }
    if (!AUTH_CONFIG) {
      navigate("/company", { replace: true });
      return;
    }
    try {
      await beginCompanyLogin(
        AUTH_CONFIG,
        {
          sessionStorage,
          navigate: (location) => window.location.assign(location),
        },
        {
          loginHint: DEMO_COMPANY_EMAIL || undefined,
          prompt: "login",
        },
      );
    } catch {
      setError(true);
    }
  }

  return (
    <CompanyAuthView
      mode="login"
      cognitoEnabled={Boolean(AUTH_CONFIG)}
      error={error}
      onPrimary={() => void login()}
      onDemo={() => void demoLogin()}
    />
  );
}

export function CompanySignupRoute() {
  const navigate = useNavigate();
  const [error, setError] = useState(false);

  async function signup() {
    if (!AUTH_CONFIG) return;
    try {
      await beginCompanySignup(AUTH_CONFIG, {
        sessionStorage,
        navigate: (location) => window.location.assign(location),
      });
    } catch {
      setError(true);
    }
  }

  async function demoLogin() {
    if (DEMO_COMPANY_TOKEN) {
      activateDemoCompanyAccess(localStorage, DEMO_COMPANY_TOKEN);
      navigate("/company", { replace: true });
      return;
    }
    if (!AUTH_CONFIG) {
      navigate("/company", { replace: true });
      return;
    }
    try {
      await beginCompanyLogin(
        AUTH_CONFIG,
        {
          sessionStorage,
          navigate: (location) => window.location.assign(location),
        },
        {
          loginHint: DEMO_COMPANY_EMAIL || undefined,
          prompt: "login",
        },
      );
    } catch {
      setError(true);
    }
  }

  return (
    <CompanyAuthView
      mode="signup"
      cognitoEnabled={Boolean(AUTH_CONFIG)}
      error={error}
      onPrimary={() => void signup()}
      onDemo={() => void demoLogin()}
    />
  );
}

export function CompanyAuthCallbackRoute() {
  const [search] = useSearchParams();
  const navigate = useNavigate();
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!AUTH_CONFIG) {
      navigate("/hiring", { replace: true });
      return;
    }
    void completeCompanyLogin(AUTH_CONFIG, search, {
      sessionStorage,
      localStorage,
      // Bound, because `completeCompanyLogin` calls this as `dependencies.fetcher(...)`,
      // which would set `this` to the dependencies object -- and `fetch` throws
      // `TypeError: Illegal invocation` unless `this` is the window. Passing it bare made
      // every hosted login fail at the token exchange with "로그인 응답을 확인할 수 없습니다",
      // indistinguishable from a rejected credential. The unit suite injects a mock, which
      // ignores `this`, so only a browser can catch it.
      fetcher: (...args) => fetch(...args),
    })
      .then(() => navigate("/hiring", { replace: true }))
      .catch(() => setError(true));
  }, [navigate, search]);

  return <CompanyAuthStatusView error={error} />;
}
