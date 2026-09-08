import pytest

from services.cache.cache import Cache, EmbeddingCache
from services.cache.factory import create_cache_backend
from services.cache.memory import InMemoryBackend
from services.cache.redis_backend import RedisUnavailable


@pytest.fixture
def backend():
    return InMemoryBackend()


class TestInMemoryBackend:
    def test_set_then_get_returns_the_value(self, backend):
        backend.set("k", "v")
        assert backend.get("k") == "v"

    def test_get_missing_key_returns_none(self, backend):
        assert backend.get("absent") is None

    def test_delete_reports_how_many_keys_went_away(self, backend):
        backend.set("a", "1")
        backend.set("b", "2")
        assert backend.delete(["a", "b", "c"]) == 2
        assert backend.get("a") is None

    def test_expired_key_reads_as_missing(self, backend, monkeypatch):
        clock = [1000.0]
        monkeypatch.setattr("services.cache.memory.time.monotonic", lambda: clock[0])
        backend.set("k", "v", ttl=10)
        assert backend.get("k") == "v"
        clock[0] += 11
        assert backend.get("k") is None

    def test_scan_matches_glob_patterns(self, backend):
        backend.set("person_embedding:role:aaa", "1")
        backend.set("person_embedding:market:bbb", "2")
        backend.set("job:1", "3")
        assert sorted(backend.scan("person_embedding:*")) == [
            "person_embedding:market:bbb",
            "person_embedding:role:aaa",
        ]

    def test_set_operations_round_trip(self, backend):
        assert backend.sadd("jobs", ["a", "b"]) == 2
        assert backend.sadd("jobs", ["a"]) == 0
        assert backend.smembers("jobs") == {"a", "b"}
        assert backend.sismember("jobs", "a") is True
        assert backend.srem("jobs", ["a"]) == 1
        assert backend.smembers("jobs") == {"b"}

    def test_non_string_value_fails_fast(self, backend):
        with pytest.raises(AssertionError):
            backend.set("k", ["not", "a", "string"])


class TestCacheKeyDerivation:
    def test_namespaced_write_can_be_deleted_by_the_same_name(self, backend):
        cache = Cache(backend, "graph_cache")
        cache.set("feature_embeddings_job1", "payload")

        assert cache.delete("feature_embeddings_job1") == 1
        assert cache.get("feature_embeddings_job1") is None

    def test_namespaced_write_can_be_probed_by_the_same_name(self, backend):
        cache = Cache(backend, "graph_cache")
        cache.set("networkx_graph_job1", "payload")
        assert cache.exists("networkx_graph_job1") is True

    def test_namespaces_do_not_collide(self, backend):
        graphs = Cache(backend, "graph_cache")
        results = Cache(backend, "job_results")

        graphs.set("job1", "graph")
        results.set("job1", "result")

        assert graphs.get("job1") == "graph"
        assert results.get("job1") == "result"

    def test_unnamespaced_cache_stores_keys_verbatim(self, backend):
        cache = Cache(backend)
        cache.set("job:abc", "payload")
        assert backend.get("job:abc") == "payload"

    def test_clear_removes_only_this_namespace(self, backend):
        graphs = Cache(backend, "graph_cache")
        results = Cache(backend, "job_results")
        graphs.set("job1", "graph")
        results.set("job1", "result")

        graphs.clear()

        assert graphs.get("job1") is None
        assert results.get("job1") == "result"

    def test_delete_by_pattern_rejects_a_namespaced_cache(self, backend):
        cache = Cache(backend, "graph_cache")
        with pytest.raises(AssertionError):
            cache.delete_by_pattern("graph_cache:*")

    def test_delete_by_pattern_removes_matching_raw_keys(self, backend):
        cache = Cache(backend)
        cache.set("person_embedding:role:a", "1")
        cache.set("person_embedding:role:b", "2")
        cache.set("job:1", "3")

        assert cache.delete_by_pattern("person_embedding:*") == 2
        assert cache.get("job:1") == "3"


class TestEmbeddingCache:
    def test_vector_round_trips(self, backend):
        cache = EmbeddingCache(backend)
        cache.set("chief technology officer", [0.1, 0.2, 0.3])
        assert cache.get("chief technology officer") == [0.1, 0.2, 0.3]

    def test_missing_text_returns_none(self, backend):
        assert EmbeddingCache(backend).get("never seen") is None

    def test_malformed_entry_is_discarded(self, backend):
        cache = EmbeddingCache(backend)
        backend.set(Cache(backend, "embeddings").storage_key("t"), "not json")

        assert cache.get("t") is None
        assert cache.get("t") is None


class TestBackendSelection:
    def test_missing_redis_url_selects_memory(self):
        assert isinstance(create_cache_backend(""), InMemoryBackend)

    def test_unreachable_redis_falls_back_to_memory(self, monkeypatch):
        def refuse(redis_url):
            raise RedisUnavailable(f"cannot reach Redis at {redis_url}")

        monkeypatch.setattr("services.cache.factory.RedisBackend", refuse)

        backend = create_cache_backend("redis://localhost:6379/0")
        assert isinstance(backend, InMemoryBackend)
