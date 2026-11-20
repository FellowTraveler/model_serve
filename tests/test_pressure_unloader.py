"""Tests for pressure_unloader.py LRU logic."""

import time
import unittest

from unittest import mock

from pressure_unloader import LoadedModel, ModelTracker, get_llama_swap_models


class TestModelTracker(unittest.TestCase):
    """Test LRU model selection."""

    def test_lru_selects_oldest_model(self):
        """Should select the model with oldest last_used time."""
        tracker = ModelTracker()
        now = time.time()

        # Add models with different ages
        tracker.models["model_old"] = LoadedModel(
            name="model_old",
            backend="llama-swap",
            last_used=now - 60,  # 60 seconds ago
        )
        tracker.models["model_new"] = LoadedModel(
            name="model_new",
            backend="ollama",
            last_used=now - 5,  # 5 seconds ago
        )
        tracker.models["model_medium"] = LoadedModel(
            name="model_medium",
            backend="llama-swap",
            last_used=now - 30,  # 30 seconds ago
        )

        # Should select the oldest (model_old)
        candidate = tracker.get_lru_candidate(min_age_seconds=0)
        self.assertEqual(candidate.name, "model_old")

    def test_lru_respects_min_age(self):
        """Should not select models that are too recently used."""
        tracker = ModelTracker()
        now = time.time()

        # Both models are recent
        tracker.models["model_a"] = LoadedModel(
            name="model_a",
            backend="llama-swap",
            last_used=now - 3,  # 3 seconds ago
        )
        tracker.models["model_b"] = LoadedModel(
            name="model_b",
            backend="ollama",
            last_used=now - 2,  # 2 seconds ago
        )

        # With min_age=5, neither should be selected
        candidate = tracker.get_lru_candidate(min_age_seconds=5)
        self.assertIsNone(candidate)

        # With min_age=2, model_a should be selected (oldest that qualifies)
        candidate = tracker.get_lru_candidate(min_age_seconds=2)
        self.assertEqual(candidate.name, "model_a")

    def test_update_from_llama_swap_tracks_new_models(self):
        """Should track newly appearing models."""
        tracker = ModelTracker()

        tracker.update_from_llama_swap([("model1", None), ("model2", None)])

        self.assertEqual(len(tracker.models), 2)
        self.assertEqual(tracker.models["model1"].backend, "llama-swap")
        self.assertEqual(tracker.models["model2"].backend, "llama-swap")

    def test_update_from_llama_swap_removes_unloaded(self):
        """Should remove models that are no longer running."""
        tracker = ModelTracker()

        tracker.update_from_llama_swap([("model1", None), ("model2", None)])
        self.assertEqual(len(tracker.models), 2)

        # model2 is unloaded
        tracker.update_from_llama_swap([("model1", None)])
        self.assertEqual(len(tracker.models), 1)
        self.assertIn("model1", tracker.models)
        self.assertNotIn("model2", tracker.models)

    def test_update_from_ollama_tracks_new_models(self):
        """Should track Ollama models with VRAM size."""
        tracker = ModelTracker()

        tracker.update_from_ollama([
            {"name": "ollama_model", "size_vram": 1000000000}
        ])

        self.assertEqual(len(tracker.models), 1)
        self.assertEqual(tracker.models["ollama_model"].backend, "ollama")
        self.assertEqual(tracker.models["ollama_model"].size_bytes, 1000000000)

    def test_mixed_backends_tracked_separately(self):
        """Should track llama-swap and Ollama models independently."""
        tracker = ModelTracker()

        tracker.update_from_llama_swap([("ls_model", None)])
        tracker.update_from_ollama([{"name": "ollama_model", "size_vram": 0}])

        self.assertEqual(len(tracker.models), 2)

        # Removing llama-swap model shouldn't affect Ollama model
        tracker.update_from_llama_swap([])
        self.assertEqual(len(tracker.models), 1)
        self.assertIn("ollama_model", tracker.models)



class TestIdleDetection(unittest.TestCase):
    """A model is idle only while its usage marker stays the same."""

    def test_polling_an_unused_model_does_not_refresh_last_used(self):
        tracker = ModelTracker()
        tracker.observe("m", "ollama", "expires-1", now=100.0)
        tracker.observe("m", "ollama", "expires-1", now=200.0)
        self.assertEqual(tracker.models["m"].last_used, 100.0)
        candidate = tracker.get_lru_candidate(min_age_seconds=5)
        self.assertEqual(candidate.name, "m")

    def test_changed_marker_means_the_model_was_used(self):
        tracker = ModelTracker()
        tracker.observe("m", "llama-swap", ("9", "3"), now=100.0)
        tracker.observe("m", "llama-swap", ("20", "8"), now=200.0)
        self.assertEqual(tracker.models["m"].last_used, 200.0)

    def test_unreadable_marker_counts_as_use(self):
        tracker = ModelTracker()
        tracker.observe("m", "llama-swap", None, now=100.0)
        tracker.observe("m", "llama-swap", None, now=200.0)
        self.assertEqual(tracker.models["m"].last_used, 200.0)

    def test_ollama_expires_at_is_the_marker(self):
        tracker = ModelTracker()
        with mock.patch("pressure_unloader.time.time", return_value=100.0):
            tracker.update_from_ollama([{"name": "g", "expires_at": "t1"}])
        with mock.patch("pressure_unloader.time.time", return_value=200.0):
            tracker.update_from_ollama([{"name": "g", "expires_at": "t1"}])
        self.assertEqual(tracker.models["g"].last_used, 100.0)


class TestLlamaSwapRunning(unittest.TestCase):
    """llama-swap /running lists objects, not names."""

    def test_parses_running_objects_and_reads_metrics(self):
        def fake_get(url, timeout):
            response = mock.Mock()
            response.raise_for_status = mock.Mock()
            if url.endswith("/running"):
                response.json.return_value = {"running": [
                    {"model": "ms/a", "state": "ready", "proxy": "http://127.0.0.1:5999"},
                ]}
            else:
                self.assertEqual(url, "http://127.0.0.1:5999/metrics")
                response.text = (
                    "# HELP x\nllamacpp:prompt_tokens_total 9\n"
                    "llamacpp:tokens_predicted_total 3\nllamacpp:n_decode_total 3\n"
                )
            return response

        with mock.patch("pressure_unloader.requests.get", side_effect=fake_get):
            running = get_llama_swap_models("http://127.0.0.1:5847")
        self.assertEqual(running, [("ms/a", ("9", "3"))])


if __name__ == "__main__":
    unittest.main()
