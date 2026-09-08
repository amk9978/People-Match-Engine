import pytest
from fastapi.testclient import TestClient

from match_engine.presentation.api_controller import app

USER_HEADER = {"X-User-ID": "test-user"}


@pytest.fixture
def client():
    return TestClient(app)


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

    def test_no_path_is_registered_twice_for_one_method(self):
        seen = [
            (route.path, method)
            for route in app.routes
            if hasattr(route, "methods")
            for method in route.methods
        ]
        assert len(seen) == len(set(seen))
