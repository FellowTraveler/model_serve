"""Tests for pressure_unloader.py LRU logic."""

import time
import unittest

from pressure_unloader import LoadedModel, ModelTracker


class TestModelTracker(unittest.TestCase):
    """Test LRU model selection."""

    def test_lru_selects_oldest_model(self):
        """Should select the model with oldest last_seen time."""
        tracker = ModelTracker()
        now = time.time()

        # Add models with different ages
        tracker.models["model_old"] = LoadedModel(
            name="model_old",
            backend="llama-swap",
            last_seen=now - 60,  # 60 seconds ago
        )
        tracker.models["model_new"] = LoadedModel(
            name="model_new",
            backend="ollama",
            last_seen=now - 5,  # 5 seconds ago
        )
        tracker.models["model_medium"] = LoadedModel(
            name="model_medium",
            backend="llama-swap",
            last_seen=now - 30,  # 30 seconds ago
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
            last_seen=now - 3,  # 3 seconds ago
        )
        tracker.models["model_b"] = LoadedModel(
            name="model_b",
            backend="ollama",
            last_seen=now - 2,  # 2 seconds ago
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

        tracker.update_from_llama_swap(["model1", "model2"])

        self.assertEqual(len(tracker.models), 2)
        self.assertEqual(tracker.models["model1"].backend, "llama-swap")
        self.assertEqual(tracker.models["model2"].backend, "llama-swap")

    def test_update_from_llama_swap_removes_unloaded(self):
        """Should remove models that are no longer running."""
        tracker = ModelTracker()

        tracker.update_from_llama_swap(["model1", "model2"])
        self.assertEqual(len(tracker.models), 2)

        # model2 is unloaded
        tracker.update_from_llama_swap(["model1"])
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

        tracker.update_from_llama_swap(["ls_model"])
        tracker.update_from_ollama([{"name": "ollama_model", "size_vram": 0}])

        self.assertEqual(len(tracker.models), 2)

        # Removing llama-swap model shouldn't affect Ollama model
        tracker.update_from_llama_swap([])
        self.assertEqual(len(tracker.models), 1)
        self.assertIn("ollama_model", tracker.models)


if __name__ == "__main__":
    unittest.main()
