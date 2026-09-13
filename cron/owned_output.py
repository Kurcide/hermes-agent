"""Exact-job output ownership for source-dependent integrations.

This is a native store operation, not an authorization or delivery service. Callers
attest the job's ownership and keep their source references. Native transcript
writers separately erase the returned message snapshots and evict live caches.
"""
from contextlib import closing
import hashlib
from itertools import islice
import json
import sqlite3

from cron import delivery_queue, executions, incidents, jobs
from cron.scheduler_provider import _profile_cron_scope

_LIMIT = 512
_ERASED = "[Cron output erased]"
_PAYLOAD_FIELDS = ("prompt", "script", "no_agent", "origin", "deliver")


def payload_fingerprint(job: dict) -> str:
    """Pin execution/delivery content while allowing its schedule and outcome to advance."""
    value = {key: job.get(key) for key in _PAYLOAD_FIELDS}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def _bounded(rows):
    values = list(islice(rows, _LIMIT + 1))
    if len(values) > _LIMIT:
        raise ValueError("cron_output_batch_limit")
    return values


def _files(job_id):
    directory = jobs._job_output_dir(job_id)
    if not directory.exists():
        return []
    values = _bounded(directory.iterdir())
    if directory.is_symlink() or any(not path.is_file() or path.is_symlink() for path in values):
        raise ValueError("cron_output_directory_changed")
    return values


def _execution_rows(job_id):
    with executions._transaction() as conn:
        return [dict(row) for row in _bounded(conn.execute(
            "SELECT id,status,error FROM executions WHERE job_id=? LIMIT ?", (job_id, _LIMIT + 1)))]


def _sessions(native_db, job_id):
    if not native_db.exists():
        return []
    from hermes_state import SessionDB
    with closing(sqlite3.connect(native_db.as_uri() + "?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN")
        rows = _bounded(conn.execute("""SELECT * FROM messages WHERE role!='session_meta'
            AND json_valid(display_metadata)
            AND json_extract(display_metadata,'$.mirror_source')='cron'
            AND json_extract(display_metadata,'$.cron_job_id')=? ORDER BY id LIMIT ?""",
            (job_id, _LIMIT + 1)))
        grouped = {}
        for row in rows:
            session = row["session_id"]
            if session not in grouped:
                watermark = conn.execute("SELECT coalesce(MAX(id),0) FROM messages WHERE session_id=?",
                                         (session,)).fetchone()[0]
                grouped[session] = {"session_id": session, "message_watermark": watermark, "messages": []}
            grouped[session]["messages"].append(SessionDB.message_redaction_snapshot(row, mode="payload"))
        return list(grouped.values())


def snapshot(job_id: str) -> dict:
    """Bounded content-free observations of one job's output and native mirror rows."""
    home = jobs._current_cron_store().cron_dir.parent.resolve()
    with _profile_cron_scope(home):
        files = _files(job_id)  # Also validates the native job-id path component.
        job = jobs.get_job(job_id)
        from hermes_state import _default_db_path
        native_db = _default_db_path().resolve()
        execution_rows = _execution_rows(job_id)
        with incidents._transaction() as conn:
            incident_rows = _bounded(conn.execute(
                "SELECT error,output_file FROM cron_incidents WHERE job_id=? LIMIT ?", (job_id, _LIMIT + 1)))
        with delivery_queue._transaction() as conn:
            queued = _bounded(conn.execute("SELECT * FROM deliveries WHERE job_id=? LIMIT ?",
                                           (job_id, _LIMIT + 1)))
            delivering = sum(delivery_queue._delivery_is_inflight(row) for row in queued)
        fields = ("last_error", "last_delivery_error", "last_delivery_unverified", "last_fire_error")
        retained_errors = sum(bool(row["error"] and row["error"] != _ERASED) for row in execution_rows)
        retained_errors += sum(bool(row["error"] and row["error"] != _ERASED) or bool(row["output_file"])
                               for row in incident_rows)
        retained_errors += sum(bool((job or {}).get(key)) for key in fields)
        queue_payloads = sum(bool(row["content"] or row["job_json"] != "{}"
                                  or (row["error"] and row["error"] != _ERASED)) for row in queued)
        active = [row["id"] for row in execution_rows if row["status"] in {"claimed", "running"}]
        return {"job_id": job_id, "native_home": str(home), "native_db": str(native_db),
                "payload_fingerprint": payload_fingerprint(job) if job is not None else None,
                "sessions": _sessions(native_db, job_id),
                "files": [{"path": str(path), "size": path.stat().st_size} for path in files],
                "retained_errors": retained_errors, "queue_payloads": queue_payloads,
                "active_executions": active, "delivering": delivering,
                "non_message_artifacts": bool(files or retained_errors or queue_payloads)}


def erase(job_id: str, *, expected_payload_fingerprint: str | None = None) -> dict:
    """Pause and erase one job's non-message output; never claim an active writer stopped.

    The caller must subsequently take a fresh snapshot and pass its message
    selections to the native/gateway transcript erasure API. Retrying this
    operation is safe after native execution or delivery settles.
    """
    home = jobs._current_cron_store().cron_dir.parent.resolve()
    with _profile_cron_scope(home), jobs._fire_job_lock(job_id) as locked:
        if not locked:
            return {"status": "pending", "reason": "cron_output_busy"}
        with jobs._jobs_lock():
            job = jobs.get_job(job_id)
            if (job is not None and expected_payload_fingerprint is not None
                    and payload_fingerprint(job) != expected_payload_fingerprint):
                return {"status": "pending", "reason": "cron_job_changed"}
            if job is not None and not jobs.is_terminal_job(job):
                jobs.pause_job(job_id, reason="Cron output erased")
        # Cancelling an unsent queued result lets its still-running native
        # waiter settle. A delivering row owns its outcome and is never stopped.
        queue = delivery_queue.erase_job_payloads(job_id)
        active = [row["id"] for row in _execution_rows(job_id) if row["status"] in {"claimed", "running"}]
        current = jobs.get_job(job_id) or {}
        now = jobs._hermes_now()
        claimed = any(jobs._claim_is_live(current.get(key), now, ttl) for key, ttl in (
            ("fire_claim", jobs.FIRE_CLAIM_TTL_SECONDS),
            ("run_claim", jobs._oneshot_run_claim_ttl_seconds())))
        if active or claimed or queue["status"] == "pending":
            return {"status": "pending", "reason": "cron_output_active",
                    "active_executions": active, "claimed": claimed, "delivering": queue["delivering"]}
        try:
            files = _files(job_id)
        except ValueError as error:
            return {"status": "pending", "reason": str(error)}
        with jobs._jobs_lock():
            fresh = jobs.get_job(job_id)
            if (fresh is not None and expected_payload_fingerprint is not None
                    and payload_fingerprint(fresh) != expected_payload_fingerprint):
                return {"status": "pending", "reason": "cron_job_changed"}
            if fresh is not None:
                jobs.update_job(job_id, {"last_error": None, "last_delivery_error": None,
                    "last_delivery_unverified": None, "last_fire_error": None, "last_delivery_queued": None})
        for path in files:
            path.unlink(missing_ok=True)
        with executions._transaction() as conn:
            conn.execute("UPDATE executions SET error=NULL WHERE job_id=? AND status IN ('completed','failed','unknown')",
                         (job_id,))
        with incidents._transaction() as conn:
            conn.execute("UPDATE cron_incidents SET error=?,output_file=NULL WHERE job_id=?", (_ERASED, job_id))
        return {"status": "erased", "job_id": job_id, "files_removed": len(files),
                "cancelled_deliveries": queue["cancelled"], "messages_erased": False}
