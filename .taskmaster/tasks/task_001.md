# Task ID: 1

**Title:** Create Harmony models configuration file

**Status:** pending

**Dependencies:** None

**Priority:** high

**Description:** Create a YAML configuration file to define which models should use Harmony encoding/decoding

**Details:**

Create a `harmony_models.yaml` file with a structure like:
```yaml
harmony_models:
  - gpt-oss-20b
  - gpt-oss-120b
```
This file will be read by the proxy to determine which models require Harmony processing. The model names must match the model IDs used in llama-swap's config.yaml. Include validation to ensure the file exists and is properly formatted.

**Test Strategy:**

Create unit tests to verify YAML parsing, handle missing files gracefully, and validate that model names are correctly loaded into a set for fast lookup
