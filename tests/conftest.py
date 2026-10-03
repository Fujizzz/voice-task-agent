import asyncio

import pytest

from voice_task_agent import AgentRuntime, SQLiteStore, TaskSpec, Tool, ToolRegistry
from voice_task_agent.contracts import Contract, Decision, Plan


class Empty(Contract):
    pass


class ScriptedModel:
    def __init__(self, *decisions, delay=0):
        self.decisions = list(decisions)
        self.calls = []
        self.delay = delay
        self.started = asyncio.Event()

    async def generate_structured(self, *, prompt, payload, schema):
        self.calls.append((schema, payload))
        self.started.set()
        if self.delay:
            await asyncio.sleep(self.delay)
        if schema is Plan:
            return Plan(steps=["Inspect context", "Answer the user"])
        result = (
            self.decisions.pop(0) if self.decisions else Decision(action="respond", text="Answer")
        )
        if isinstance(result, Exception):
            raise result
        return schema.model_validate(result)


@pytest.fixture
def factory():
    stores = []

    def create(model=None, *, path=":memory:", budget=None, handler=None, name="task"):
        async def lookup(arguments, context):
            return {"value": context.get("value")}

        store = SQLiteStore(str(path))
        stores.append(store)
        task = TaskSpec(
            name=name,
            instructions="Use only supplied evidence.",
            goal="Answer the user",
            allowed_tools=["lookup"],
            budget=budget or {},
        )
        tools = ToolRegistry([Tool("lookup", "Read context", Empty, handler or lookup)])
        model = model or ScriptedModel()
        return AgentRuntime(model, store, task, tools), store, model

    yield create
    for store in stores:
        store.close()
