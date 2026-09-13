"""Exact cron provenance connects stored output to selective native erasure."""

from concurrent.futures import ThreadPoolExecutor
import json
import threading
from types import SimpleNamespace

import pytest

from cron import delivery_queue, executions, incidents, jobs, owned_output
from cron.scheduler_delivery import (
    _maybe_mirror_cron_delivery,
    _seed_cron_channel_session,
    _seed_cron_thread_session,
)
from gateway.config import GatewayConfig
from gateway.session import SessionStore
import hermes_state
from hermes_state_registry import acquire, release_or_close


@pytest.fixture
def profile(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # The global fixture pins DEFAULT_DB_PATH. Exercise native profile resolution
    # here, including a cron-store override in a process serving another profile.
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    db = acquire(home / "state.db")
    try:
        with jobs.use_cron_store(home):
            yield home, db
    finally:
        release_or_close(db)


def _job():
    return jobs.create_job("Read the selected local source", "every 1h", name="Shared label",
                           deliver="local", script="reader.py", no_agent=True)


def _rows(db):
    return {row["id"]: dict(row) for row in db._conn.execute("SELECT * FROM messages ORDER BY id")}


def _failed_execution(job, error):
    run = executions.create_execution(job["id"], source="manual")
    return executions.finish_execution(run["id"], success=False, error=error)


def test_native_mirror_provenance_selects_only_its_job_and_preserves_identical_user_text(profile):
    home, db = profile
    owned, other = _job(), _job()
    db.create_session("conversation", source="telegram", session_key="route:owner")
    db.record_gateway_session_peer("conversation", source="telegram", session_key="route:owner",
                                   user_id="owner", chat_id="100", chat_type="dm")
    text = "Selected source says orchard badge cobalt-716."
    for job, execution_id in ((owned, "owned-first"), (other, "unrelated-run"),
                              (owned, "owned-second")):
        _maybe_mirror_cron_delivery(job | {"execution_id": execution_id}, "telegram", "100",
                                    text, user_id="owner", enabled=True)
    mirrors = _rows(db)
    assert len(mirrors) == 3
    selected = []
    for row in mirrors.values():
        metadata = json.loads(row["display_metadata"])
        assert row["role"] == "user" and metadata["mirror_source"] == "cron"
        assert metadata["cron_execution_id"] in {"owned-first", "unrelated-run", "owned-second"}
        if metadata["cron_job_id"] == owned["id"]:
            selected.append(row["id"])
    assert len(selected) == 2
    identical = db.append_message("conversation", "user", mirrors[selected[0]]["content"])
    session_meta = db.append_message("conversation", "session_meta", "Retain session routing",
        display_metadata={"mirror_source": "cron", "cron_job_id": owned["id"]})
    db.create_session("other-conversation", source="cli")
    untouched_session = db.append_message("other-conversation", "user", text)
    # Archived output remains an owned copy even when ordinary replay omits it.
    db._execute_write(lambda conn: conn.execute(
        "UPDATE messages SET active=0,compacted=1 WHERE id=?", (selected[0],)))
    before = _rows(db)
    session_before = db.get_session("conversation")

    snapshot = owned_output.snapshot(owned["id"])
    assert snapshot["native_home"] == str(home)
    assert snapshot["native_db"] == str(db.db_path)
    assert "cobalt-716" not in json.dumps(snapshot)
    assert len(snapshot["sessions"]) == 1
    selection = snapshot["sessions"][0]
    assert selection["session_id"] == "conversation"
    assert [row["id"] for row in selection["messages"]] == selected
    result = db.redact_message_payloads(selection["session_id"], selection["messages"],
        expected_message_watermark=selection["message_watermark"])
    assert result["redacted_ids"] == selected
    after = _rows(db)
    assert set(after) == set(before)
    assert {key: row for key, row in after.items() if key not in selected} == {
        key: row for key, row in before.items() if key not in selected}
    assert after[identical] == before[identical]
    assert after[session_meta] == before[session_meta]
    assert after[untouched_session] == before[untouched_session]
    assert db.get_session("conversation") == session_before
    for row_id in selected:
        assert "cobalt-716" not in json.dumps(after[row_id])
        assert after[row_id]["role"] == before[row_id]["role"]
        assert after[row_id]["active"] == before[row_id]["active"]
    assert owned_output.snapshot(owned["id"])["sessions"] == []
    assert db.redact_message_payloads(selection["session_id"], selection["messages"])["redacted_ids"] == []


@pytest.mark.parametrize("threaded", [False, True], ids=["channel", "thread"])
def test_native_seed_persists_job_and_execution_provenance_through_real_session_store(profile, threaded):
    home, db = profile
    owned, other = _job(), _job()
    store = SessionStore(home / "sessions", GatewayConfig())
    adapter = SimpleNamespace(_session_store=store)
    try:
        for job, execution_id in ((owned, "owned-seed"), (other, "other-seed")):
            delivered = job | {"execution_id": execution_id}
            if threaded:
                _seed_cron_thread_session(delivered, adapter, "telegram", "100", "20",
                                          "Same source brief", is_dm=True)
            else:
                assert _seed_cron_channel_session(delivered, adapter, "telegram", "100",
                    "Same source brief", is_dm=True, user_id="owner")
        before = _rows(db)
        snapshot = owned_output.snapshot(owned["id"])
        assert len(snapshot["sessions"]) == 1
        selection = snapshot["sessions"][0]
        assert len(selection["messages"]) == 1
        row_id = selection["messages"][0]["id"]
        row = before[row_id]
        metadata = json.loads(row["display_metadata"])
        assert row["role"] == "user"
        assert metadata["mirror_source"] == "cron"
        assert metadata["cron_job_id"] == owned["id"]
        assert metadata["cron_execution_id"] == "owned-seed"
        session = db.get_session(selection["session_id"])
        assert str(session["chat_id"]) == "100"
        assert session["thread_id"] == ("20" if threaded else None)
        result = db.redact_message_payloads(selection["session_id"], selection["messages"],
            expected_message_watermark=selection["message_watermark"])
        assert result["redacted_ids"] == [row_id]
        after = _rows(db)
        assert {key: value for key, value in after.items() if key != row_id} == {
            key: value for key, value in before.items() if key != row_id}
        assert owned_output.snapshot(owned["id"])["sessions"] == []
    finally:
        store.close_all_db_handles()


def test_erase_clears_files_errors_and_delivery_copies_without_changing_another_job(profile):
    _, db = profile
    owned, other = _job(), _job()
    content = "Source-derived orchard badge cobalt-716"
    paths = {job["id"]: jobs.save_job_output(job["id"], content) for job in (owned, other)}
    runs = {job["id"]: _failed_execution(job, content) for job in (owned, other)}
    incident_ids = {job["id"]: incidents.upsert_incident(job["id"], content,
        output_file=paths[job["id"]])[0] for job in (owned, other)}
    errors = {key: content for key in (
        "last_error", "last_delivery_error", "last_delivery_unverified", "last_fire_error")}
    for job in (owned, other):
        jobs.update_job(job["id"], errors)
    delivery_queue.enqueue("owned-failed", owned, content)
    assert delivery_queue.drain(lambda *_: content) == 1
    delivery_queue.enqueue("owned-pending", owned | {"prompt": content}, content)
    delivery_queue.enqueue("other-pending", other | {"prompt": content}, content)
    other_before = {"job": jobs.get_job(other["id"]),
        "run": executions.get_execution(runs[other["id"]]["id"]),
        "incident": incidents.get_incident(incident_ids[other["id"]]),
        "delivery": delivery_queue.get_status("other-pending")}
    snapshot = owned_output.snapshot(owned["id"])
    assert snapshot["files"] and snapshot["retained_errors"] and snapshot["queue_payloads"]
    assert "cobalt-716" not in json.dumps(snapshot)

    result = owned_output.erase(owned["id"], expected_payload_fingerprint=snapshot["payload_fingerprint"])
    assert result["status"] == "erased" and result["cancelled_deliveries"] == 1
    assert result["messages_erased"] is False
    assert not paths[owned["id"]].exists()
    assert paths[other["id"]].read_text() == content
    assert executions.get_execution(runs[owned["id"]]["id"])["error"] is None
    incident = incidents.get_incident(incident_ids[owned["id"]])
    assert content not in incident["error"] and incident["output_file"] is None
    retained_job = jobs.get_job(owned["id"])
    assert retained_job["enabled"] is False and retained_job["state"] == "paused"
    assert all(retained_job[key] is None for key in errors)
    for execution_id in ("owned-pending", "owned-failed"):
        queue_row = delivery_queue.get_status(execution_id)
        assert queue_row["status"] == "failed"
        assert queue_row["job_json"] == "{}" and queue_row["content"] == ""
        assert content not in str(queue_row["error"])
    assert jobs.get_job(other["id"]) == other_before["job"]
    assert executions.get_execution(runs[other["id"]]["id"]) == other_before["run"]
    assert incidents.get_incident(incident_ids[other["id"]]) == other_before["incident"]
    assert delivery_queue.get_status("other-pending") == other_before["delivery"]
    fresh = owned_output.snapshot(owned["id"])
    assert not fresh["files"] and not fresh["retained_errors"] and not fresh["queue_payloads"]
    assert fresh["non_message_artifacts"] is False
    assert owned_output.erase(owned["id"], expected_payload_fingerprint=snapshot["payload_fingerprint"])["status"] == "erased"


@pytest.mark.parametrize("running", [False, True], ids=["claimed", "running"])
def test_active_execution_defers_erasure_until_the_real_writer_settles(profile, running):
    owned = _job()
    output = jobs.save_job_output(owned["id"], "Original source output")
    run = executions.create_execution(owned["id"], source="manual")
    if running:
        assert executions.mark_execution_running(run["id"])["status"] == "running"
    delivery_queue.enqueue("waiting-output", owned, "Unsent source output")
    before = executions.get_execution(run["id"])
    snapshot = owned_output.snapshot(owned["id"])
    assert snapshot["active_executions"] == [run["id"]]

    pending = owned_output.erase(owned["id"], expected_payload_fingerprint=snapshot["payload_fingerprint"])
    assert pending["status"] == "pending" and pending["active_executions"] == [run["id"]]
    assert executions.get_execution(run["id"]) == before
    assert output.read_text() == "Original source output"
    assert jobs.get_job(owned["id"])["enabled"] is False
    assert delivery_queue.get_status("waiting-output")["status"] == "failed"
    assert executions.finish_execution(run["id"], success=False, error="Late source output")
    result = owned_output.erase(owned["id"], expected_payload_fingerprint=snapshot["payload_fingerprint"])
    assert result["status"] == "erased" and not output.exists()
    assert executions.get_execution(run["id"])["status"] == "failed"
    assert executions.get_execution(run["id"])["error"] is None
    assert owned_output.snapshot(owned["id"])["non_message_artifacts"] is False


def test_delivery_in_progress_defers_erasure_and_is_never_retried(profile, monkeypatch):
    owned = _job()
    output = jobs.save_job_output(owned["id"], "Source output waiting for its sender")
    delivery_queue.enqueue("sending-output", owned, "Source output")
    entered, settle = threading.Event(), threading.Event()
    sends = []

    def send(job, content, for_failure):
        sends.append((job, content, for_failure))
        entered.set()
        assert settle.wait(10), "Test delivery was not released"
        return "Source-derived delivery error"

    with ThreadPoolExecutor(max_workers=1) as executor:
        worker = executor.submit(delivery_queue.drain, send)
        try:
            assert entered.wait(10), "Native delivery never entered its send"
            row = delivery_queue.get_status("sending-output")
            snapshot = owned_output.snapshot(owned["id"])
            assert snapshot["delivering"] == 1
            pending = owned_output.erase(owned["id"], expected_payload_fingerprint=snapshot["payload_fingerprint"])
            assert pending["status"] == "pending" and pending["delivering"] == 1
            assert delivery_queue.get_status("sending-output") == row
            assert output.exists()
            # A native waiter timing out does not stop a send already in progress.
            monkeypatch.setattr(delivery_queue, "MAX_TERMINAL_DELIVERIES", 0)
            delivery_queue._terminalize_wait_timeout("sending-output")
            uncertain = delivery_queue.get_status("sending-output")
            assert uncertain["status"] == "unknown"
            pending = owned_output.erase(owned["id"], expected_payload_fingerprint=snapshot["payload_fingerprint"])
            assert pending["status"] == "pending"
            assert delivery_queue.get_status("sending-output") == uncertain
            assert output.exists()
        finally:
            settle.set()
        assert worker.result(timeout=10) == 1
    assert owned_output.erase(owned["id"], expected_payload_fingerprint=snapshot["payload_fingerprint"])["status"] == "erased"
    assert delivery_queue.drain(send) == 0 and len(sends) == 1
    assert not output.exists()
    assert owned_output.snapshot(owned["id"])["non_message_artifacts"] is False


@pytest.mark.parametrize("updates", [{"prompt": "Read a newly selected source"},
                                     {"deliver": "telegram:200"}])
def test_changed_job_payload_cannot_be_paused_or_erased_with_an_old_fingerprint(profile, updates):
    owned = _job()
    output = jobs.save_job_output(owned["id"], "Retained output")
    delivery_queue.enqueue("retained-output", owned, "Retained output")
    snapshot = owned_output.snapshot(owned["id"])
    # Schedule advancement and outcome recording do not alter execution ownership.
    jobs.update_job(owned["id"], {"schedule": "every 2h", "last_error": "Retained error"})
    assert owned_output.snapshot(owned["id"])["payload_fingerprint"] == snapshot["payload_fingerprint"]
    jobs.update_job(owned["id"], updates)
    changed_job = jobs.get_job(owned["id"])
    queue_before = delivery_queue.get_status("retained-output")
    result = owned_output.erase(owned["id"], expected_payload_fingerprint=snapshot["payload_fingerprint"])
    assert result == {"status": "pending", "reason": "cron_job_changed"}
    assert jobs.get_job(owned["id"]) == changed_job
    assert delivery_queue.get_status("retained-output") == queue_before
    assert output.read_text() == "Retained output"


def test_cron_store_override_erases_only_its_profile_when_process_home_differs(profile, tmp_path, monkeypatch):
    owner_home, _ = profile
    owned = _job()
    owned_file = jobs.save_job_output(owned["id"], "Selected profile output")
    delivery_queue.enqueue("owner-output", owned, "Selected profile output")
    process_home = tmp_path / "another-profile"
    monkeypatch.setenv("HERMES_HOME", str(process_home))
    with jobs.use_cron_store(process_home):
        other = _job()
        other_file = jobs.save_job_output(other["id"], "Other profile output")
        other_queue = delivery_queue.enqueue("other-profile-output", other, "Other profile output")
    snapshot = owned_output.snapshot(owned["id"])
    assert snapshot["native_home"] == str(owner_home)
    assert snapshot["native_db"] == str(owner_home / "state.db")
    result = owned_output.erase(owned["id"], expected_payload_fingerprint=snapshot["payload_fingerprint"])
    assert result["status"] == "erased" and not owned_file.exists()
    assert other_file.read_text() == "Other profile output"
    with jobs.use_cron_store(process_home):
        assert jobs.get_job(other["id"])["enabled"] is True
        assert delivery_queue.get_status("other-profile-output") == other_queue
