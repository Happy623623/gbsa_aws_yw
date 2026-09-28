# Created: 2026-08-23 22:52
"""The queue a local run creates has to say what the runtime consuming it already assumes."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
from interview_evidence.runtime.local_infra import LocalSqsClient, _ensure_workflow_queue
from interview_evidence.runtime.queue_topology import (
    DEAD_LETTER_MESSAGE_RETENTION_SECONDS,
    DEFAULT_VISIBILITY_TIMEOUT_SECONDS,
    MEDIA_VISIBILITY_TIMEOUT_SECONDS,
    QUEUE_MAX_RECEIVE_COUNT,
    QUEUE_MESSAGE_RETENTION_SECONDS,
    REPORTING_QUEUE_MAX_RECEIVE_COUNT,
    WORKFLOW_QUEUE_NAMES,
    queue_max_receive_count,
    queue_visibility_timeout_seconds,
)

TERRAFORM_MODULE = (
    Path(__file__).resolve().parents[3].parent / "infra/modules/async-workflow/main.tf"
)


class RecordingSqsClient:
    def __init__(self) -> None:
        self.attributes: dict[str, dict[str, str]] = {}

    def create_queue(self, **kwargs: object) -> Mapping[str, object]:
        name = str(kwargs["QueueName"])
        return {"QueueUrl": f"http://localhost:4566/000000000000/{name}"}

    def get_queue_attributes(self, **kwargs: object) -> Mapping[str, object]:
        url = str(kwargs["QueueUrl"])
        return {"Attributes": {"QueueArn": f"arn:aws:sqs:ap-northeast-2:000000000000:{url[-40:]}"}}

    def set_queue_attributes(self, **kwargs: object) -> object:
        url = str(kwargs["QueueUrl"])
        self.attributes.setdefault(url, {}).update(
            cast(dict[str, str], kwargs["Attributes"]),
        )
        return None


def _provision(name: str, environment: Mapping[str, str] | None = None) -> RecordingSqsClient:
    client = RecordingSqsClient()
    _ensure_workflow_queue(
        cast(LocalSqsClient, client),
        name=name,
        queue_name=f"iep-{name}",
        environment=environment or {},
    )
    return client


def _work_queue_attributes(client: RecordingSqsClient, name: str) -> dict[str, str]:
    return client.attributes[f"http://localhost:4566/000000000000/iep-{name}"]


@pytest.mark.parametrize("name", WORKFLOW_QUEUE_NAMES)
def test_the_created_queue_states_the_lease_the_consumer_paces_against(name: str) -> None:
    """The regression this guards: the runtime heartbeat extends the lease at a third of the
    configured timeout, so a queue that expires sooner redelivers work that is still running and
    a second worker repeats it. The two numbers have to come from the same place."""
    attributes = _work_queue_attributes(_provision(name), name)

    assert attributes["VisibilityTimeout"] == str(queue_visibility_timeout_seconds(name, {}))
    assert int(attributes["VisibilityTimeout"]) >= DEFAULT_VISIBILITY_TIMEOUT_SECONDS


def test_media_keeps_the_longer_lease_and_the_rest_share_the_default() -> None:
    assert queue_visibility_timeout_seconds("media", {}) == MEDIA_VISIBILITY_TIMEOUT_SECONDS
    for name in WORKFLOW_QUEUE_NAMES:
        if name == "media":
            continue
        assert queue_visibility_timeout_seconds(name, {}) == DEFAULT_VISIBILITY_TIMEOUT_SECONDS


def test_an_environment_override_reaches_both_the_queue_and_the_consumer() -> None:
    environment = {"SQS_ANALYSIS_VISIBILITY_TIMEOUT_SECONDS": "600"}

    attributes = _work_queue_attributes(_provision("analysis", environment), "analysis")

    assert attributes["VisibilityTimeout"] == "600"
    assert queue_visibility_timeout_seconds("analysis", environment) == 600


def test_work_queue_redrives_to_its_dead_letter_queue() -> None:
    """Without this a message that always fails is retried until retention expires and then
    vanishes, so a local failure leaves nothing to inspect."""
    client = _provision("analysis")

    policy = json.loads(_work_queue_attributes(client, "analysis")["RedrivePolicy"])
    assert policy["maxReceiveCount"] == QUEUE_MAX_RECEIVE_COUNT
    assert policy["deadLetterTargetArn"].endswith("iep-analysis-dlq")

    dead_letter = client.attributes["http://localhost:4566/000000000000/iep-analysis-dlq"]
    assert dead_letter["MessageRetentionPeriod"] == str(DEAD_LETTER_MESSAGE_RETENTION_SECONDS)
    assert int(dead_letter["MessageRetentionPeriod"]) > QUEUE_MESSAGE_RETENTION_SECONDS


def test_reporting_queue_gets_a_longer_retry_window() -> None:
    client = _provision("reporting")

    policy = json.loads(_work_queue_attributes(client, "reporting")["RedrivePolicy"])

    assert policy["maxReceiveCount"] == REPORTING_QUEUE_MAX_RECEIVE_COUNT
    assert queue_max_receive_count("analysis", {}) == QUEUE_MAX_RECEIVE_COUNT
    assert queue_max_receive_count("reporting", {}) == REPORTING_QUEUE_MAX_RECEIVE_COUNT


def test_reporting_retry_count_can_be_overridden_locally() -> None:
    environment = {"SQS_REPORTING_MAX_RECEIVE_COUNT": "20"}
    client = _provision("reporting", environment)

    policy = json.loads(_work_queue_attributes(client, "reporting")["RedrivePolicy"])

    assert policy["maxReceiveCount"] == 20


def test_controlproof_reporting_topology_uses_exact_local_demo_contract() -> None:
    environment = {
        "SQS_REPORTING_MAX_RECEIVE_COUNT": "3",
        "SQS_REPORTING_VISIBILITY_TIMEOUT_SECONDS": "5",
    }
    client = _provision("reporting", environment)
    work = _work_queue_attributes(client, "reporting")
    policy = json.loads(work["RedrivePolicy"])
    dead_letter = client.attributes[
        "http://localhost:4566/000000000000/iep-reporting-dlq"
    ]

    assert work["VisibilityTimeout"] == "5"
    assert work["MessageRetentionPeriod"] == str(QUEUE_MESSAGE_RETENTION_SECONDS)
    assert policy == {
        "deadLetterTargetArn": policy["deadLetterTargetArn"],
        "maxReceiveCount": 3,
    }
    assert policy["deadLetterTargetArn"].endswith("iep-reporting-dlq")
    assert dead_letter["MessageRetentionPeriod"] == str(
        DEAD_LETTER_MESSAGE_RETENTION_SECONDS
    )
    assert int(dead_letter["MessageRetentionPeriod"]) > int(
        work["MessageRetentionPeriod"]
    )


def test_attributes_are_asserted_rather_than_passed_to_create_queue() -> None:
    """`create_queue` leaves an existing queue as it is, so a queue created before these
    attributes existed would keep the SQS defaults forever. Setting them afterwards repairs it."""
    client = _provision("reporting")

    assert set(client.attributes) == {
        "http://localhost:4566/000000000000/iep-reporting",
        "http://localhost:4566/000000000000/iep-reporting-dlq",
    }


def test_the_shared_values_match_the_terraform_module() -> None:
    """Terraform owns the deployed queues. If these drift, local runs stop rehearsing production
    and the disagreement is invisible until something is redelivered in the wrong environment."""
    module = TERRAFORM_MODULE.read_text(encoding="utf-8")

    visibility = re.search(
        r"visibility_timeout_seconds\s*=\s*each\.key == \"media\" \? (\d+) : (\d+)", module
    )
    assert visibility is not None, "terraform no longer sets visibility per workflow"
    assert int(visibility.group(1)) == MEDIA_VISIBILITY_TIMEOUT_SECONDS
    assert int(visibility.group(2)) == DEFAULT_VISIBILITY_TIMEOUT_SECONDS

    retentions = [
        int(value) for value in re.findall(r"message_retention_seconds\s*=\s*(\d+)", module)
    ]
    assert QUEUE_MESSAGE_RETENTION_SECONDS in retentions
    assert DEAD_LETTER_MESSAGE_RETENTION_SECONDS in retentions

    receive_count = re.search(r"variable \"max_receive_count\"[^}]*default\s*=\s*(\d+)", module)
    assert receive_count is not None, "terraform no longer defaults max_receive_count"
    assert int(receive_count.group(1)) == QUEUE_MAX_RECEIVE_COUNT
    assert "reporting = 12" in module
    assert "lookup(var.max_receive_count_by_workflow" in module


def test_the_workflow_list_matches_terraform() -> None:
    module = TERRAFORM_MODULE.read_text(encoding="utf-8")
    declared = re.search(r"workflows\s*=\s*toset\(\[([^\]]+)\]\)", module)
    assert declared is not None
    names = tuple(value.strip().strip('"') for value in declared.group(1).split(","))

    assert set(names) == set(WORKFLOW_QUEUE_NAMES)
