"""A third application assembled entirely through public interfaces."""

import asyncio

from voice_task_agent import AgentRuntime, SQLiteStore, TaskSpec, Tool, ToolRegistry
from voice_task_agent.contracts import Contract
from voice_task_agent.providers import DemoModel


class NoArguments(Contract):
    pass


async def read_checklist(arguments, context):
    return {"items": context["checklist"]}


async def main():
    task = TaskSpec(
        name="checklist",
        instructions="Help the user review a checklist.",
        goal="Explain remaining checklist items.",
        allowed_tools=["read_checklist"],
    )
    registry = ToolRegistry(
        [Tool("read_checklist", "Read a local checklist", NoArguments, read_checklist)]
    )
    store = SQLiteStore()
    try:
        agent = AgentRuntime(DemoModel(), store, task, registry)
        session = agent.create_session({"checklist": ["Review dependencies", "Run tests"]})
        result = await agent.run(session, "What should I check?")
        print(result.model_dump_json(indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    asyncio.run(main())
