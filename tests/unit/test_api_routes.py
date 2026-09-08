import pytest
from fastapi.testclient import TestClient

from match_engine.presentation import api_controller
from match_engine.presentation.api_controller import app

USER_HEADER = {"X-User-ID": "test-user"}


@pytest.fixture
def client():
    return TestClient(app)


class StoredResult:
    def __init__(self, data):
        self.result_data = data


@pytest.fixture
def stored_result(monkeypatch):
    data = {
        "people": ["Ada", "Grace", "Katherine"],
        "matches": {
            "0": [
                {"position": 1, "name": "Grace", "company": "Univac", "weight": 0.9},
                {"position": 2, "name": "Katherine", "company": "NACA", "weight": 0.4},
            ]
        },
    }
    monkeypatch.setattr(
        api_controller.job_service,
        "get_job_result",
        lambda job_id: StoredResult(data) if job_id == "job-1" else None,
    )
    return data


class TestJobRoutes:
    def test_stats_is_not_shadowed_by_the_job_id_route(self, client):
        response = client.get("/jobs/stats", headers=USER_HEADER)
        assert response.status_code == 200

    def test_cleanup_is_not_shadowed_by_the_job_id_route(self, client):
        response = client.delete("/jobs/cleanup")
        assert response.status_code == 200

    def test_an_unknown_job_still_returns_not_found(self, client):
        response = client.get("/jobs/no-such-job")
        assert response.status_code == 404

    def test_listing_jobs_accepts_the_filter_arguments(self, client):
        response = client.get("/jobs", params={"status": "completed", "limit": 5})
        assert response.status_code == 200

    def test_a_person_route_serves_the_stored_ranking(self, client, stored_result):
        response = client.get("/jobs/job-1/people/0/matches")

        assert response.status_code == 200
        body = response.json()
        assert body["name"] == "Ada"
        assert body["matches"][0]["name"] == "Grace"

    def test_top_caps_the_ranking(self, client, stored_result):
        response = client.get("/jobs/job-1/people/0/matches", params={"top": 1})

        assert len(response.json()["matches"]) == 1

    def test_a_position_nobody_occupies_returns_not_found(self, client, stored_result):
        response = client.get("/jobs/job-1/people/9/matches")

        assert response.status_code == 404

    def test_a_person_route_on_an_unknown_job_returns_not_found(self, client):
        response = client.get("/jobs/no-such-job/people/0/matches")

        assert response.status_code == 404

    def test_no_path_is_registered_twice_for_one_method(self):
        seen = [
            (route.path, method)
            for route in app.routes
            if hasattr(route, "methods")
            for method in route.methods
        ]
        assert len(seen) == len(set(seen))
