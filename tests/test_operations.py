from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.models import Operation, OperationTug, PortCall, Tug, Vessel
from app.services.operations import escort_likely, recommended_tug_count


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


def test_escort_likely_boundaries() -> None:
    assert escort_likely(9.2) is False
    assert escort_likely(9.3) is True
    assert escort_likely(9.6) is True
    assert escort_likely(9.7) is False
    assert escort_likely(None) is False


def test_recommended_tug_count_boundaries() -> None:
    assert recommended_tug_count(119) == 1
    assert recommended_tug_count(120) == 2
    assert recommended_tug_count(150) == 2
    assert recommended_tug_count(160) == 2
    assert recommended_tug_count(161) == 3
    assert recommended_tug_count(200) == 3
    assert recommended_tug_count(None) is None


def test_assigns_two_tugs_and_only_one_escort(client: TestClient) -> None:
    db_generator = app.dependency_overrides[get_db]()
    db = next(db_generator)
    try:
        vessel = Vessel(name="Test vessel", loa_m=150)
        db.add(vessel)
        db.flush()
        portcall = PortCall(vessel_id=vessel.id)
        db.add(portcall)
        db.flush()
        operation = Operation(portcall_id=portcall.id, draft_m=9.3)
        tug_one = Tug(name="Tug one")
        tug_two = Tug(name="Tug two")
        db.add_all([operation, tug_one, tug_two])
        db.flush()
        operation_id = operation.id
        portcall_id = portcall.id
        tug_one_id = tug_one.id
        tug_two_id = tug_two.id
        db.commit()
    finally:
        db.close()

    first = client.post(
        f"/operations/{operation_id}/tugs",
        data={"tug_id": str(tug_one_id), "escort": "on"},
        follow_redirects=False,
    )
    assert first.status_code == 303
    page = client.get(f"/portcalls/{portcall_id}")
    assert page.status_code == 200
    assert "назначено 1 из рекомендуемых 2" in page.text
    assert "нужно ещё" in page.text

    second = client.post(
        f"/operations/{operation_id}/tugs",
        data={"tug_id": str(tug_two_id), "escort": "on"},
        follow_redirects=False,
    )
    assert second.status_code == 303
    duplicate = client.post(
        f"/operations/{operation_id}/tugs",
        data={"tug_id": str(tug_two_id)},
        follow_redirects=False,
    )
    assert duplicate.status_code == 303

    db_generator = app.dependency_overrides[get_db]()
    db = next(db_generator)
    try:
        links = db.scalars(
            select(OperationTug)
            .where(OperationTug.operation_id == operation_id)
            .order_by(OperationTug.tug_id)
        ).all()
        assert len(links) == 2
        assert {link.tug.name for link in links} == {"Tug one", "Tug two"}
        assert sum(link.escort for link in links) == 1
        assert links[0].escort is True
        assert links[1].escort is False
    finally:
        db.close()
