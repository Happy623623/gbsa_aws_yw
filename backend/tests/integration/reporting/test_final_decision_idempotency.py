from uuid import UUID

from backend.tests.integration.reporting.test_recruiting_stage_decision import (
    INVITATION_ID,
    REPORT_ID,
    STAGE_ID,
    DecisionWriter,
    client,
    context,
)


def test_same_key_final_decision_replays_one_complete_effect_set() -> None:
    writer = DecisionWriter()
    http, repository, session = client(writer)
    headers = {
        "Authorization": "Bearer company-token",
        "Idempotency-Key": "same-final-decision-key",
    }
    body = {"recruiting_stage_id": str(STAGE_ID), "expected_pipeline_version": 3}

    first = http.post(
        f"/v1/invitations/{INVITATION_ID}/final-decisions",
        headers=headers,
        json=body,
    )
    replay = http.post(
        f"/v1/invitations/{INVITATION_ID}/final-decisions",
        headers=headers,
        json=body,
    )

    assert first.status_code == 201
    assert replay.status_code == 201
    assert replay.json() == first.json()
    assert writer.moves == [(INVITATION_ID, STAGE_ID, 3)]
    assert len(writer.advances) == 1
    assert len(repository.list_reviews(context(), REPORT_ID)) == 1
    session.close()


def test_same_key_with_a_different_stage_is_rejected_without_new_effects() -> None:
    writer = DecisionWriter()
    http, repository, session = client(writer)
    headers = {
        "Authorization": "Bearer company-token",
        "Idempotency-Key": "conflicting-final-decision-key",
    }
    first = http.post(
        f"/v1/invitations/{INVITATION_ID}/final-decisions",
        headers=headers,
        json={"recruiting_stage_id": str(STAGE_ID), "expected_pipeline_version": 3},
    )
    conflict = http.post(
        f"/v1/invitations/{INVITATION_ID}/final-decisions",
        headers=headers,
        json={
            "recruiting_stage_id": str(
                UUID("00000000-0000-7000-8000-000000000099")
            ),
            "expected_pipeline_version": 3,
        },
    )

    assert first.status_code == 201
    assert conflict.status_code == 409
    assert writer.moves == [(INVITATION_ID, STAGE_ID, 3)]
    assert len(writer.advances) == 1
    assert len(repository.list_reviews(context(), REPORT_ID)) == 1
    session.close()


def test_same_key_with_a_different_expected_version_is_rejected() -> None:
    writer = DecisionWriter()
    http, repository, session = client(writer)
    headers = {
        "Authorization": "Bearer company-token",
        "Idempotency-Key": "conflicting-final-decision-version-key",
    }
    first = http.post(
        f"/v1/invitations/{INVITATION_ID}/final-decisions",
        headers=headers,
        json={"recruiting_stage_id": str(STAGE_ID), "expected_pipeline_version": 3},
    )
    conflict = http.post(
        f"/v1/invitations/{INVITATION_ID}/final-decisions",
        headers=headers,
        json={"recruiting_stage_id": str(STAGE_ID), "expected_pipeline_version": 4},
    )

    assert first.status_code == 201
    assert conflict.status_code == 409
    assert writer.moves == [(INVITATION_ID, STAGE_ID, 3)]
    assert len(writer.advances) == 1
    assert len(repository.list_reviews(context(), REPORT_ID)) == 1
    session.close()
