# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Output stream filtering to scrub physics warnings."""

import re
from typing import List, Pattern


class NoiseFilter:
    """Wraps a text stream and drops lines whose content matches any of
    a fixed set of regex patterns. Used to scrub cosmetic PhysX
    joint warnings (and similar noise) from stderr/stdout so they don't
    show up in screen recordings.
    """

    def __init__(self, wrapped, drop_patterns: List[str]):
        """Initialize the noise filter.
        
        Args:
            wrapped: The wrapped stream (sys.stderr, sys.stdout, etc.)
            drop_patterns: List of regex patterns (as strings) to filter
        """
        self._wrapped = wrapped
        self._patterns: List[Pattern] = [re.compile(p) for p in drop_patterns]
        self._buf = ""

    def write(self, s: str) -> None:
        """Write string to stream, filtering lines matching drop patterns."""
        if not s:
            return
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if not any(p.search(line) for p in self._patterns):
                self._wrapped.write(line + "\n")

    def flush(self) -> None:
        """Flush any remaining buffer and underlying stream."""
        if self._buf:
            if not any(p.search(self._buf) for p in self._patterns):
                self._wrapped.write(self._buf)
            self._buf = ""
        try:
            self._wrapped.flush()
        except Exception:
            pass

    def __getattr__(self, name: str):
        """Delegate unknown attributes to the wrapped stream."""
        return getattr(self._wrapped, name)
