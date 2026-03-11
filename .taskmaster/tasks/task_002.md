# Task ID: 2

**Title:** Set up Python dependencies and project structure

**Status:** pending

**Dependencies:** 1

**Priority:** high

**Description:** Install required Python packages and create the basic project structure for the Harmony proxy

**Details:**

Install dependencies:
```bash
pip install openai-harmony httpx fastapi uvicorn pyyaml
```
Create the main proxy file `harmony_proxy.py` with imports:
```python
from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
    ToolDescription,
    StreamableParser,
)
```
Load the GPT-OSS encoding at startup: `ENC = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)`

**Test Strategy:**

Verify all imports work correctly and the Harmony encoding loads without errors. Test that the encoding can be used to create basic Harmony structures.
