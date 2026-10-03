"""Two application definitions sharing the same runtime without core modifications."""

import re

from pydantic import Field

from .contracts import Contract, TaskSpec
from .tools import Tool, ToolRegistry


class NoArguments(Contract):
    pass


class SearchArguments(Contract):
    query: str = Field(min_length=1, max_length=1000)


async def read_profile(arguments, context):
    return {"profile": context.get("profile", "No profile provided")}


async def search_notes(arguments: SearchArguments, context):
    """Small transparent lexical retriever, not an embedding/vector RAG pipeline."""
    terms = set(re.findall(r"\w+", arguments.query.casefold()))
    ranked = []
    for index, note in enumerate(context.get("notes", [])):
        content = str(note)
        score = sum(term in content.casefold() for term in terms)
        if score:
            ranked.append((score, index, content))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return {"matches": [{"source": f"note-{i}", "text": text} for _, i, text in ranked[:3]]}


def interviewer():
    task = TaskSpec(
        name="interviewer",
        instructions=(
            "You are a technical interview assistant. Ask one focused question per turn. "
            "Use the supplied profile and actual answers to choose follow-ups. "
            "Do not invent candidate experience or numeric scores. Finish when the user "
            "asks to end and summarize only observed answers. Reply in the user's language."
        ),
        goal="Understand the user's technical decisions through evidence-based follow-up questions.",
        allowed_tools=["read_profile"],
    )
    return task, ToolRegistry(
        [Tool("read_profile", "Read the supplied candidate profile", NoArguments, read_profile)]
    )


def knowledge_assistant():
    task = TaskSpec(
        name="knowledge",
        instructions=(
            "Answer questions using the supplied notes. Search before answering factual "
            "questions, cite returned note IDs, and say when evidence is missing. "
            "Ask for clarification when needed. Reply in the user's language."
        ),
        goal="Answer the user's question using traceable evidence from local notes.",
        allowed_tools=["search_notes"],
    )
    return task, ToolRegistry(
        [
            Tool(
                "search_notes",
                "Search local notes for matching words",
                SearchArguments,
                search_notes,
            )
        ]
    )


def application(name):
    return {"interviewer": interviewer, "knowledge": knowledge_assistant}[name]()
