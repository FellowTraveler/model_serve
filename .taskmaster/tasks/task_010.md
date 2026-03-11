# Task ID: 10

**Title:** Create integration documentation and model_serve integration

**Status:** pending

**Dependencies:** 9

**Priority:** low

**Description:** Document the proxy setup and create integration scripts for model_serve

**Details:**

Create documentation covering:
- How to configure `harmony_models.yaml`
- Environment variables and configuration options
- Integration with existing `model_serve` setup
- Client configuration (pointing to proxy instead of llama-swap)

Create a startup function for `model_serve` similar to `start_llama_swap()`:
```bash
start_harmony_proxy() {
    echo "Starting Harmony proxy on port ${HARMONY_PROXY_PORT}..."
    uvicorn harmony_proxy:app --host 0.0.0.0 --port ${HARMONY_PROXY_PORT} &
    HARMONY_PROXY_PID=$!
    echo "Harmony proxy PID: $HARMONY_PROXY_PID"
}
```

Include troubleshooting guide and examples of client configuration changes.

**Test Strategy:**

Manual testing of integration with model_serve, verify startup scripts work correctly, test client connectivity through proxy, validate documentation accuracy
