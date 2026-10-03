import sqlite3

import pytest


async def test_sql_failure_after_state_update_rolls_back_entire_turn(factory):
    runtime, store, _ = factory()
    session = runtime.create_session()
    store._db.executescript("""
        CREATE TRIGGER fail_result BEFORE UPDATE OF result ON requests
        WHEN NEW.status='done'
        BEGIN SELECT RAISE(ABORT, 'simulated commit failure'); END;
    """)
    with pytest.raises(sqlite3.IntegrityError):
        await runtime.run(session, "hello", request_id="one")
    assert store.get(session).version == 0
    assert store.get(session).history == []
    store._db.execute("DROP TRIGGER fail_result")
    result = await runtime.run(session, "hello", request_id="one")
    assert result.state_version == 1
