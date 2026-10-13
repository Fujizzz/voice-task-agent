"""SQLite sessions with atomic turn commits and cross-process request reservation."""

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

from .contracts import HistoryTurn, Session, TurnResult
from .errors import RequestConflict, SessionBusy, SessionFinished, SessionNotFound, StateConflict


def digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


class SQLiteStore:
    def __init__(self, path: str = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, timeout=5, check_same_thread=False)
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, version INTEGER NOT NULL, data TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS requests (
                session_id TEXT NOT NULL REFERENCES sessions(id),
                request_id TEXT NOT NULL, fingerprint TEXT NOT NULL,
                status TEXT NOT NULL, result TEXT,
                PRIMARY KEY(session_id, request_id)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS one_running_request
                ON requests(session_id) WHERE status='running';
        """)

    def create(self, session: Session):
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO sessions VALUES (?, ?, ?)",
                (session.session_id, session.version, session.model_dump_json()),
            )

    def get(self, session_id: str) -> Session:
        with self._lock:
            row = self._db.execute("SELECT data FROM sessions WHERE id=?", (session_id,)).fetchone()
            if row is None:
                raise SessionNotFound(session_id)
            return Session.model_validate_json(row[0])

    def begin(
        self, session_id: str, request_id: str, text: str, task_hash: str
    ) -> Session | TurnResult:
        fingerprint = digest({"text": text, "task": task_hash})
        with self._lock, self._db:
            self._db.execute("BEGIN IMMEDIATE")
            session = self.get(session_id)
            if session.task_hash != task_hash:
                raise RequestConflict("Session belongs to a different task configuration")
            row = self._db.execute(
                "SELECT fingerprint,status,result FROM requests WHERE session_id=? AND request_id=?",
                (session_id, request_id),
            ).fetchone()
            if row:
                if row[0] != fingerprint:
                    raise RequestConflict("Request ID was already used with different input")
                if row[1] == "done":
                    return TurnResult.model_validate_json(row[2])
            if session.status == "finished":
                raise SessionFinished(session_id)
            running = self._db.execute(
                "SELECT 1 FROM requests WHERE session_id=? AND status='running'", (session_id,)
            ).fetchone()
            if running:
                raise SessionBusy("A request is already running for this session")
            self._db.execute(
                "INSERT INTO requests VALUES (?,?,?,'running',NULL) "
                "ON CONFLICT(session_id,request_id) DO UPDATE SET status='running',result=NULL",
                (session_id, request_id, fingerprint),
            )
            return session

    def commit(self, previous: Session, text: str, result: TurnResult, history_limit: int):
        if result.session_id != previous.session_id or result.state_version != previous.version + 1:
            raise StateConflict("Result identity/version does not match the pending turn")
        updated = previous.model_copy(deep=True)
        updated.version = result.state_version
        updated.turn_count += 1
        updated.status = "finished" if result.action == "finish" else "active"
        updated.plan = result.plan
        updated.history = (
            updated.history
            + [HistoryTurn(request_id=result.request_id, user=text, assistant=result.text)]
        )[-history_limit:]
        with self._lock, self._db:
            self._db.execute("BEGIN IMMEDIATE")
            request = self._db.execute(
                "SELECT status FROM requests WHERE session_id=? AND request_id=?",
                (previous.session_id, result.request_id),
            ).fetchone()
            if request is None or request[0] != "running":
                raise StateConflict("Request is no longer reserved")
            cursor = self._db.execute(
                "UPDATE sessions SET version=?,data=? WHERE id=? AND version=?",
                (updated.version, updated.model_dump_json(), previous.session_id, previous.version),
            )
            if cursor.rowcount != 1:
                raise StateConflict("Session changed before commit")
            self._db.execute(
                "UPDATE requests SET status='done',result=? WHERE session_id=? AND request_id=?",
                (result.model_dump_json(), previous.session_id, result.request_id),
            )

    def abort(self, session_id: str, request_id: str):
        with self._lock, self._db:
            self._db.execute(
                "UPDATE requests SET status='aborted' WHERE session_id=? AND request_id=? AND status='running'",
                (session_id, request_id),
            )

    def recover(self, session_id: str) -> int:
        """Release a crashed request after its worker has stopped."""
        with self._lock, self._db:
            self.get(session_id)
            cursor = self._db.execute(
                "UPDATE requests SET status='aborted' WHERE session_id=? AND status='running'",
                (session_id,),
            )
            return cursor.rowcount

    def close(self):
        self._db.close()
