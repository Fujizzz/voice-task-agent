"""Domain-neutral contracts; application-specific data stays in session context."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Budget(Contract):
    turn_seconds: float = Field(default=60, gt=0, le=600)
    model_seconds: float = Field(default=20, gt=0, le=300)
    tool_seconds: float = Field(default=5, gt=0, le=60)
    max_tool_calls: int = Field(default=3, ge=0, le=20)
    max_repairs: int = Field(default=2, ge=0, le=5)
    max_turns: int = Field(default=30, ge=1, le=1000)
    history_turns: int = Field(default=8, ge=1, le=50)
    max_input_chars: int = Field(default=8000, ge=1, le=64000)
    max_context_chars: int = Field(default=32000, ge=1, le=256000)
    max_observation_chars: int = Field(default=6000, ge=100, le=32000)


class TaskSpec(Contract):
    name: str = Field(min_length=1, max_length=100)
    instructions: str = Field(min_length=1, max_length=12000)
    goal: str = Field(min_length=1, max_length=2000)
    allowed_tools: list[str] = Field(default_factory=list, max_length=50)
    budget: Budget = Field(default_factory=Budget)


class Plan(Contract):
    steps: list[str] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def nonempty_steps(self):
        if any(not s.strip() or len(s) > 500 for s in self.steps):
            raise ValueError("Plan steps must contain 1 to 500 characters")
        return self


class Decision(Contract):
    action: Literal["tool", "ask", "respond", "finish"]
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    text: str | None = Field(default=None, max_length=12000)

    @model_validator(mode="after")
    def action_fields(self):
        if self.action == "tool":
            if not self.tool or self.text is not None:
                raise ValueError("Tool actions require tool and arguments, without text")
        elif not self.text or self.tool is not None or self.arguments:
            raise ValueError("Reply actions require nonempty text without tool arguments")
        return self


class TraceEvent(Contract):
    component: str
    status: str
    elapsed_ms: int = Field(default=0, ge=0)
    tool: str | None = None


class TurnResult(Contract):
    session_id: str
    request_id: str
    state_version: int
    action: Literal["ask", "respond", "finish"]
    text: str
    stop_reason: str
    fallback_used: bool = False
    plan: Plan
    trace: list[TraceEvent] = Field(default_factory=list)


class HistoryTurn(Contract):
    request_id: str
    user: str
    assistant: str


class Session(Contract):
    session_id: str
    task_hash: str
    context: dict[str, Any] = Field(default_factory=dict)
    version: int = 0
    turn_count: int = 0
    status: Literal["active", "finished"] = "active"
    plan: Plan | None = None
    history: list[HistoryTurn] = Field(default_factory=list)
