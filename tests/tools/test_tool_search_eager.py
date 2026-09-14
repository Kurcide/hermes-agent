"""Named eager schemas retain native session selection and execution contracts."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml


@pytest.fixture
def configured_plugin(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    plugin = home / "plugins" / "disclosure_probe"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: disclosure_probe\n")
    (plugin / "__init__.py").write_text('''
import json

def register(ctx):
    for name, toolset, available in (
        ("eager_probe_read", "probe_allowed", True),
        ("deferred_probe_read", "probe_allowed", True),
        ("unavailable_probe_read", "probe_allowed", False),
        ("excluded_probe_read", "probe_excluded", True),
    ):
        ctx.register_tool(name=name, toolset=toolset,
            schema={"name": name, "description": "Read a probe value.", "parameters": {
                "type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}},
            check_fn=lambda available=available: available,
            handler=lambda args, name=name, **kw: json.dumps({"tool": name, **args}))

    def checked(tool_name, args, **kw):
        if tool_name in {"eager_probe_read", "deferred_probe_read"}:
            return {"args": {**args, "checked": True}, "source": "disclosure_probe"}
    ctx.register_middleware("tool_request", checked)
''')
    config = {
        "model": {"context_length": 256000},
        "plugins": {"enabled": ["disclosure_probe"]},
        "tools": {"tool_search": {"enabled": "on", "eager": [
            "eager_probe_read", "eager_probe_read", "unavailable_probe_read",
            "excluded_probe_read", "unknown_probe_read",
        ], "defer": ["eager_probe_read"]}},
    }
    (home / "config.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import plugins

    manager = plugins.PluginManager()
    manager.discover_and_load()
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    yield home
    manager.unload()


@pytest.fixture
def native_agent(configured_plugin):
    from run_agent import AIAgent

    # Only provider construction is replaced; no inference is performed.
    with patch("agent.process_bootstrap.OpenAI"):
        agent = AIAgent(
            api_key="test-key", base_url="https://example.invalid/v1", provider="openrouter",
            model="test-model", enabled_toolsets=["probe_allowed", "probe_excluded"],
            disabled_toolsets=["probe_excluded"], quiet_mode=True,
            skip_context_files=True, skip_memory=True,
        )
    try:
        yield agent
    finally:
        agent.close()


def test_named_eager_tools_use_real_native_selection_validation_and_middleware(native_agent):
    from agent.turn_tool_validation import validate_tool_calls
    import model_tools

    agent = native_agent
    named = {tool["function"]["name"]: tool["function"] for tool in agent.tools}
    assert "eager_probe_read" in named
    assert len(named) == len(agent.tools)
    assert agent.valid_tool_names == set(named)
    assert {"tool_search", "tool_describe", "tool_call"} <= named.keys()
    assert "deferred_probe_read" not in named
    assert "deferred_probe_read" in named["tool_search"]["description"]
    assert "eager_probe_read" not in named["tool_search"]["description"]
    for name in ("unavailable_probe_read", "excluded_probe_read", "unknown_probe_read"):
        assert name not in named and name not in named["tool_search"]["description"]

    for index, (wire_name, tool_name) in enumerate((
        ("eager_probe_read", "eager_probe_read"),
        ("tool_call", "deferred_probe_read"),
        ("tool_call", "eager_probe_read"),
    )):
        arguments = {"value": str(index)}
        if wire_name == "tool_call":
            arguments = {"calls": [{"name": tool_name, "arguments": arguments}]}
        call = SimpleNamespace(id=f"call-{index}", type="function",
            function=SimpleNamespace(name=wire_name, arguments=json.dumps(arguments)))
        message = SimpleNamespace(content="", tool_calls=[call])
        messages = []
        verdict = validate_tool_calls(agent, message, "tool_calls", messages=messages,
            conversation_history=[], api_call_count=1, effective_task_id="probe-task")
        assert verdict.action == "ok"
        agent._execute_tool_calls_sequential(message, messages, "probe-task")
        result = json.loads(messages[-1]["content"])
        assert result == {"tool": tool_name, "value": str(index), "checked": True}

    # Visibility preference cannot admit a tool outside this session's toolsets.
    excluded = model_tools.get_tool_definitions(enabled_toolsets=[], quiet_mode=True)
    assert excluded == []
    denied = json.loads(model_tools.handle_function_call("tool_call", {
        "calls": [{"name": "excluded_probe_read", "arguments": {"value": "no"}}]},
        enabled_toolsets=["probe_allowed"]))
    assert "not available in this session" in json.dumps(denied)


def test_empty_or_malformed_eager_setting_preserves_default_disclosure(configured_plugin):
    import model_tools
    from tools.tool_search import ToolSearchConfig, assemble_tool_defs

    incoming = model_tools.get_tool_definitions(enabled_toolsets=["probe_allowed"],
        quiet_mode=True, skip_tool_search_assembly=True)
    baseline = assemble_tool_defs(incoming, config=ToolSearchConfig.from_raw({"enabled": "on"}))
    for value in ([], None, "eager_probe_read", [None, {}, 7, " "]):
        candidate = assemble_tool_defs(incoming,
            config=ToolSearchConfig.from_raw({"enabled": "on", "eager": value}))
        assert candidate == baseline
    assert {tool["function"]["name"] for tool in baseline.tool_defs} == {
        "tool_search", "tool_describe", "tool_call"}
