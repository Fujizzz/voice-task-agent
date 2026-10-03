import asyncio

import pytest
from conftest import Empty, ScriptedModel

from voice_task_agent import Tool, ToolRegistry
from voice_task_agent.contracts import Decision
from voice_task_agent.errors import (
    InvalidModelOutput,
    ModelError,
    RequestConflict,
    SessionBusy,
    SessionFinished,
)


def tool(name="lookup", **arguments):
    return Decision(action="tool", tool=name, arguments=arguments)


async def test_tool_observation_reaches_model_and_turn_commits(factory):
    model = ScriptedModel(tool(), Decision(action="respond", text="The value is 42"))
    runtime, store, _ = factory(model)
    session = runtime.create_session({"value": 42})
    result = await runtime.run(session, "Read value", request_id="one")
    assert model.calls[-1][1]["observations"][0]["result"] == {"value": 42}
    assert result.state_version == 1
    assert store.get(session).history[0].assistant == "The value is 42"


async def test_replay_survives_process_restart_without_model_call(factory, tmp_path):
    path = tmp_path / "state.sqlite"
    first, _, _ = factory(path=path)
    session = first.create_session()
    result = await first.run(session, "Hello", request_id="one")
    resumed, _, model = factory(path=path)
    assert await resumed.run(session, "Hello", request_id="one") == result
    assert model.calls == []
    with pytest.raises(RequestConflict):
        await resumed.run(session, "Different input", request_id="one")


async def test_independent_store_connections_reject_concurrent_turn(factory, tmp_path):
    path = tmp_path / "state.sqlite"
    slow = ScriptedModel(delay=10)
    first, store, _ = factory(slow, path=path)
    second, _, other = factory(path=path)
    session = first.create_session()
    running = asyncio.create_task(first.run(session, "hello", request_id="one"))
    await slow.started.wait()
    try:
        with pytest.raises(SessionBusy):
            await second.run(session, "second", request_id="two")
        assert other.calls == []
    finally:
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
    assert store.get(session).version == 0
    assert store.get(session).history == []
    result = await second.run(session, "hello", request_id="one")
    assert result.state_version == 1


async def test_failed_commit_does_not_publish_and_releases_reservation(factory, monkeypatch):
    runtime, store, _ = factory()
    session = runtime.create_session()
    original = store.commit

    def fail(*args):
        raise RuntimeError("Simulated disk failure")

    monkeypatch.setattr(store, "commit", fail)
    with pytest.raises(RuntimeError):
        await runtime.run(session, "hello", request_id="one")
    assert store.get(session).version == 0
    monkeypatch.setattr(store, "commit", original)
    assert (await runtime.run(session, "hello", request_id="one")).state_version == 1


@pytest.mark.parametrize(
    "decision,error",
    [
        (tool("unknown"), "TOOL_NOT_ALLOWED"),
        (tool(unexpected=1), "INVALID_TOOL_ARGUMENTS"),
    ],
)
async def test_bad_tool_is_not_executed(factory, decision, error):
    called = []

    async def handler(arguments, context):
        called.append(1)

    model = ScriptedModel(decision)
    runtime, _, _ = factory(model, handler=handler)
    result = await runtime.run(runtime.create_session(), "test")
    assert called == []
    assert model.calls[-1][1]["observations"][-1]["result"] == {"error": error}
    assert any(e.status == error for e in result.trace)


async def test_repeated_tool_is_executed_once_and_loop_is_bounded(factory):
    calls = []

    async def handler(arguments, context):
        calls.append(1)
        return {"ok": True}

    model = ScriptedModel(tool(), tool(), tool())
    runtime, _, _ = factory(model, handler=handler, budget={"max_tool_calls": 2})
    result = await runtime.run(runtime.create_session(), "test")
    assert calls == [1]
    assert result.stop_reason == "TOOL_LIMIT"
    assert result.fallback_used
    assert any(e.status == "REPEATED_TOOL_CALL" for e in result.trace)


async def test_invalid_output_repairs_then_succeeds(factory):
    model = ScriptedModel(
        {"action": "execute_arbitrary_code"}, Decision(action="ask", text="Which file?")
    )
    runtime, _, _ = factory(model)
    result = await runtime.run(runtime.create_session(), "test")
    assert not result.fallback_used
    assert model.calls[-1][1]["repair_errors"]


async def test_repairs_exhaust_without_unbounded_provider_retries(factory):
    model = ScriptedModel(InvalidModelOutput("bad"), InvalidModelOutput("bad"))
    runtime, _, _ = factory(model, budget={"max_repairs": 1})
    result = await runtime.run(runtime.create_session(), "test")
    assert result.stop_reason == "INVALID_OUTPUT"
    assert len(model.calls) == 3  # one planning call, two execution calls


async def test_whole_turn_deadline_commits_honest_fallback(factory):
    runtime, store, _ = factory(ScriptedModel(delay=0.2), budget={"turn_seconds": 0.03})
    session = runtime.create_session()
    result = await runtime.run(session, "test")
    assert result.stop_reason == "TIMEOUT" and result.fallback_used
    assert store.get(session).version == 1


async def test_tool_timeout_is_observed_and_execution_can_continue(factory):
    async def slow(arguments, context):
        await asyncio.sleep(1)

    model = ScriptedModel(tool())
    runtime, _, _ = factory(model, handler=slow, budget={"tool_seconds": 0.01})
    result = await runtime.run(runtime.create_session(), "test")
    assert any(e.status == "TOOL_TIMEOUT" for e in result.trace)
    assert model.calls[-1][1]["observations"][0]["result"]["error"] == "TOOL_TIMEOUT"


async def test_context_isolation_and_observation_limit(factory):
    async def mutate(arguments, context):
        context["value"] = "changed"
        return {"data": "x" * 1000}

    model = ScriptedModel(tool())
    runtime, store, _ = factory(model, handler=mutate, budget={"max_observation_chars": 100})
    first = runtime.create_session({"value": "first"})
    second = runtime.create_session({"value": "second"})
    await runtime.run(first, "test")
    assert store.get(first).context["value"] == "first"
    assert store.get(second).version == 0
    assert model.calls[-1][1]["observations"][0]["result"]["truncated"]


async def test_completed_session_can_replay_but_not_take_new_requests(factory):
    runtime, _, _ = factory(ScriptedModel(Decision(action="finish", text="Done")))
    session = runtime.create_session()
    result = await runtime.run(session, "finish", request_id="one")
    assert await runtime.run(session, "finish", request_id="one") == result
    with pytest.raises(SessionFinished):
        await runtime.run(session, "new input")


async def test_history_is_bounded_and_total_turn_counter_is_not_truncated(factory):
    runtime, store, model = factory(budget={"history_turns": 1, "max_turns": 2})
    session = runtime.create_session()
    await runtime.run(session, "one")
    await runtime.run(session, "two")
    assert len(store.get(session).history) == 1
    before = len(model.calls)
    result = await runtime.run(session, "three")
    assert result.stop_reason == "TURN_LIMIT" and result.action == "finish"
    assert len(model.calls) == before


async def test_configuration_change_cannot_reuse_an_existing_session(factory, tmp_path):
    path = tmp_path / "state.sqlite"
    first, _, _ = factory(path=path)
    session = first.create_session()
    different, _, _ = factory(path=path, name="different")
    with pytest.raises(RequestConflict):
        await different.run(session, "test")


async def test_crashed_request_requires_explicit_recovery(factory):
    runtime, store, _ = factory()
    session = runtime.create_session()
    store.begin(session, "one", "test", runtime.task_hash)
    with pytest.raises(SessionBusy):
        await runtime.run(session, "test", request_id="one")
    assert store.recover(session) == 1
    assert (await runtime.run(session, "test", request_id="one")).state_version == 1


@pytest.mark.parametrize("text", ["", "   ", "x" * 8001])
async def test_bad_input_does_not_reserve_or_call_model(factory, text):
    runtime, store, model = factory()
    session = runtime.create_session()
    with pytest.raises(ValueError):
        await runtime.run(session, text)
    assert store.get(session).version == 0 and model.calls == []


def test_write_tools_rejected_until_durable_side_effect_protocol_exists():
    async def handler(arguments, context):
        return None

    with pytest.raises(ValueError, match="read-only"):
        ToolRegistry([Tool("write", "Write", Empty, handler, read_only=False)])


async def test_model_transport_failure_is_redacted_in_persisted_result(factory):
    model = ScriptedModel(ModelError("private provider response"))
    runtime, _, _ = factory(model)
    result = await runtime.run(runtime.create_session(), "test")
    assert result.stop_reason == "MODEL_UNAVAILABLE"
    assert "private provider response" not in result.model_dump_json()
