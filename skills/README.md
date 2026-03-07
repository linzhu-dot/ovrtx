<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
-->
# ovrtx Skills Directory

This directory contains structured skill files for AI coding agents (Cursor, Copilot, etc.) working on the ovrtx codebase. Each skill is a self-contained reference for a specific ovrtx task, with code snippets for both the Python and C APIs.

## Structure

Each subdirectory contains a single `SKILLS.md` file with YAML frontmatter:

```
skills/
  renderer-creation/SKILLS.md
  loading-usd/SKILLS.md
  stepping-and-rendering/SKILLS.md
  ...
```

## SKILLS.md Format

```markdown
---
name: skill-name
description: What this skill covers. Use when user asks to [trigger phrases].
---

# Skill Title

## Overview
Brief explanation of when/why you'd use this.

## Python
Step-by-step with code snippets.

## C
Step-by-step with code snippets.

## Key Types / Functions
Quick reference of the API surface involved.

## Common Pitfalls
Gotchas and things to watch out for.
```

## Code Snippet References

Skills reference live code in test and example files instead of duplicating snippets inline. This keeps code in skills accurate as the API evolves.

### Marker format in source files

Python (`tests/test_ovrtx.py`, `examples/python/*/main.py`):
```python
# [snippet:write-attribute-cpu]
source_np = np.eye(4, dtype=np.float64)
renderer.write_attribute(...)
# [/snippet:write-attribute-cpu]
```

C/C++ (`examples/c/*/main.cpp`):
```cpp
// [snippet:create-renderer]
ovrtx_config_t config {};
ovrtx_result_t result = ovrtx_create_renderer(&config, &renderer);
// [/snippet:create-renderer]
```

Names are kebab-case and unique within each file.

### Reference format in SKILLS.md

Replace inline code blocks with a blockquote directive:

```markdown
> **Source:** `tests/test_ovrtx.py` snippet `write-attribute-cpu`
```

Agents read the referenced file between the `# [snippet:name]` and `# [/snippet:name]` markers to get the current code.

### Reference format in RST docs

Use Sphinx `literalinclude` with marker-based selectors:

```rst
.. literalinclude:: ../../examples/c/minimal/main.cpp
   :language: cpp
   :start-after: // [snippet:create-renderer]
   :end-before: // [/snippet:create-renderer]
```

## Adding a New Skill

1. **Write tests first.** Add focused test functions to `tests/test_ovrtx.py` (or a new test file) that demonstrate each code path the skill will reference. Wrap every illustrative section in `# [snippet:name]` / `# [/snippet:name]` markers.
2. Create a new directory under `skills/` named after the skill (use kebab-case).
3. Add a `SKILLS.md` file inside it following the format above.
4. **Do not write inline code blocks for API usage.** Reference the test/example snippets using `> **Source:** ...` blockquotes. This ensures code in skills always matches real, tested code.
5. For C API patterns, add markers to an existing example under `examples/c/` or create a new example if needed.

## Updating Skills

When you make changes to the ovrtx API surface, examples, or conventions that affect an existing skill, update the corresponding `SKILLS.md` to keep it accurate.

## Modifying Tests or Examples

- **Preserve snippet markers.** If you move or restructure marked code, update the markers to stay around the illustrative section.
- **Do not remove markers** without also removing or updating every `> **Source:**` reference in `skills/` and every `literalinclude` in `docs/` that points to them.
- **Add markers to new tests.** Every new test function that demonstrates an API workflow should have snippet markers around its illustrative code. If the workflow maps to an existing skill, add a reference there. If not, consider creating a new skill.
