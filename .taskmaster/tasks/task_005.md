# Task ID: 5

**Title:** Create FastAPI proxy server with routing logic

**Status:** pending

**Dependencies:** 4

**Priority:** medium

**Description:** Implement the main FastAPI application with model-based routing between Harmony and passthrough modes

**Details:**

Create FastAPI app with:
- Configuration loading from `harmony_models.yaml`
- `is_harmony_model(model)` function for routing decisions
- Environment variable `LLAMA_SWAP_BASE` (default: http://127.0.0.1:8000)
- Basic error handling and logging
- Health check endpoint

Implement routing logic that checks if the requested model is in `HARMONY_MODELS` set and routes accordingly. Include proper HTTP status code handling and error responses.

**Test Strategy:**

Test configuration loading, verify routing logic with different model names, test error handling for invalid requests, verify environment variable handling
