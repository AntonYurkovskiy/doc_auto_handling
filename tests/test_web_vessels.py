from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.models import Operation, OperationKind


@pytest.fixture
def client() -> Iterator[TestClient]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def override_get_db() -> Iterator[Session]:
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
        Base.metadata.drop_all(engine)
        engine.dispose()


def test_vessel_and_portcall_pages(client: TestClient) -> None:
    assert client.get("/vessels").status_code == 200
    assert client.get("/portcalls").status_code == 200

    vessel_response = client.post(
        "/vessels",
        data={
            "name": "Тестовое судно",
            "imo": "1234567",
            "loa_m": "100,5",
            "grt": "2500",
        },
        follow_redirects=False,
    )
    assert vessel_response.status_code == 303
    vessel_location = vessel_response.headers["location"]
    assert client.get(vessel_location).status_code == 200
    vessel_id = int(vessel_location.rsplit("/", 1)[-1])

    portcall_response = client.post(
        "/portcalls",
        data={
            "vessel_id": str(vessel_id),
            "direction": "вход",
            "status": "confirmed",
            "agent": "Тестовый агент",
            "eta": "2025-01-15T12:30",
        },
        follow_redirects=False,
    )
    assert portcall_response.status_code == 303
    portcall_location = portcall_response.headers["location"]
    assert client.get(portcall_location).status_code == 200
    assert client.get("/vessels/" + str(vessel_id)).status_code == 200

    portcall_id = int(portcall_location.rsplit("/", 1)[-1])
    operation_response = client.post(
        f"/portcalls/{portcall_id}/operations",
        data={
            "kind": "швартовка",
            "seq": "1",
            "draft_m": "7,4",
            "notes": "Тестовая операция",
        },
        follow_redirects=False,
    )
    assert operation_response.status_code == 303
    assert operation_response.headers["location"] == f"/portcalls/{portcall_id}"
    assert client.get(operation_response.headers["location"]).status_code == 200

    db_generator = app.dependency_overrides[get_db]()
    db = next(db_generator)
    try:
        operation = db.scalar(select(Operation))
        assert operation is not None
        assert operation.kind is OperationKind.mooring
        assert operation.draft_m == 7.4
        assert operation.notes == "Тестовая операция"
    finally:
        db.close()
