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

## Adding a New Skill

1. Create a new directory under `skills/` named after the skill (use kebab-case).
2. Add a `SKILLS.md` file inside it following the format above.
3. Include working code snippets for both Python and C where applicable.

## Updating Skills

When you make changes to the ovrtx API surface, examples, or conventions that affect an existing skill, update the corresponding `SKILLS.md` to keep it accurate.
