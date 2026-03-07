# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Sphinx extension: filtered-literalinclude directive.

Extends ``literalinclude`` with an ``:exclude-pattern:`` option that strips
lines matching a regular expression from the included source.  The primary
use-case is removing ``# [snippet:...]`` / ``// [snippet:...]`` markers that
are present in example files for skill references but should not appear in
rendered documentation.
"""

import re

from sphinx.directives.code import LiteralInclude


class FilteredLiteralInclude(LiteralInclude):
    """A ``literalinclude`` variant that can exclude lines by regex."""

    option_spec = {
        **LiteralInclude.option_spec,
        "exclude-pattern": str,
    }

    def run(self):
        nodes = super().run()

        pattern = self.options.get("exclude-pattern")
        if not pattern:
            return nodes

        regex = re.compile(pattern)

        for node in nodes:
            if node.rawsource or hasattr(node, "astext"):
                text = node.astext()
                filtered = "\n".join(
                    line for line in text.splitlines() if not regex.search(line)
                )
                # Update both rawsource (used by Pygments for highlighting)
                # and the child Text node (used for plain-text output)
                node.rawsource = filtered
                node.children[0] = node.children[0].__class__(filtered)

        return nodes


def setup(app):
    app.add_directive("filtered-literalinclude", FilteredLiteralInclude)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
