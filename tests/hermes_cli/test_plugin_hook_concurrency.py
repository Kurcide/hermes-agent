"""Healthy overlap must retain each invocation's callback result and context."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
import threading

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _overlapping_results(monkeypatch, hook, sessions):
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 5.0)
    identity = ContextVar("hook_test_identity", default=None)
    first_entered = threading.Event()
    release_first = threading.Event()

    class AdmissionLock:
        """Release the holder only after the peer has inspected the running slot.

        The existing skip path releases on return; serialization releases when
        waiting. This handshake forces real overlap without a scheduling sleep.
        """
        def __init__(self):
            self.lock = threading.Lock()

        def acquire(self, *args, **kwargs):
            return self.lock.acquire(*args, **kwargs)

        def release(self):
            if identity.get() == "second":
                release_first.set()
            self.lock.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, *_args):
            self.release()

    manager = PluginManager()
    lock = AdmissionLock()
    manager._hook_timeout_lock = lock
    manager._hook_timeout_running_cond = threading.Condition(lock)
    context = PluginContext(PluginManifest(name="context-observer"), manager)
    seen = []

    def callback(session_id, tool_call_id):
        current = identity.get()
        seen.append((session_id, tool_call_id, current))
        if current == "first":
            first_entered.set()
            assert release_first.wait(5.0), "Peer never reached hook admission"
        return {"context": current, "session_id": session_id, "tool_call_id": tool_call_id}

    registration = context.register_hook(hook, callback)

    def fire(label, session):
        token = identity.set(label)
        try:
            return manager.invoke_hook(hook, session_id=session, tool_call_id=label)
        finally:
            identity.reset(token)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(fire, "first", sessions[0])
            assert first_entered.wait(5.0)
            second = pool.submit(fire, "second", sessions[1])
            results = [first.result(timeout=10.0), second.result(timeout=10.0)]
    finally:
        release_first.set()
        registration.dispose()
    assert seen == [(sessions[0], "first", "first"), (sessions[1], "second", "second")]
    assert results == [
        [{"context": "first", "session_id": sessions[0], "tool_call_id": "first"}],
        [{"context": "second", "session_id": sessions[1], "tool_call_id": "second"}],
    ]


def test_independent_sessions_keep_their_hook_results_and_context(monkeypatch):
    _overlapping_results(monkeypatch, "pre_llm_call", ("session-one", "session-two"))


def test_parallel_tools_in_one_session_each_get_a_policy_decision(monkeypatch):
    _overlapping_results(monkeypatch, "pre_tool_call", ("session-one", "session-one"))
