"""Typed failures exposed to applications without provider response bodies."""


class AgentError(Exception):
    pass


class SessionNotFound(AgentError):
    pass


class SessionBusy(AgentError):
    pass


class SessionFinished(AgentError):
    pass


class StateConflict(AgentError):
    pass


class RequestConflict(AgentError):
    pass


class ModelError(AgentError):
    pass


class InvalidModelOutput(AgentError):
    pass


class ToolError(AgentError):
    pass
