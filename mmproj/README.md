# Vision projectors (mmproj)

Multimodal projector GGUFs referenced by `mmproj:` keys in `custom_models.yaml`.
They enable vision on llama-server via `--mmproj`. Ollama's `hf.co/...` pulls
only fetch the main model GGUF, so projectors are downloaded here manually.

The `.gguf` files are gitignored; re-download as needed:

```bash
# Qwen3.8-27B (unsloth/Qwen3.8-27B-GGUF)
curl -sL -o mmproj/Qwen3.8-27B-mmproj-F16.gguf \
  "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/mmproj-F16.gguf"

# Qwen3.8-27B-heretic-ara (mradermacher/Qwen3.8-27B-heretic-ara-GGUF)
curl -sL -o mmproj/Qwen3.8-27B-heretic-ara-mmproj-F16.gguf \
  "https://huggingface.co/mradermacher/Qwen3.8-27B-heretic-ara-GGUF/resolve/main/Qwen3.8-27B-heretic-ara.mmproj-f16.gguf"
```
