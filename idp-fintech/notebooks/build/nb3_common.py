"""Shared helpers for building Notebook 3 from separately-owned parts.

Every part module exposes `cells() -> list[nbformat cell]`.  The assembler is `build_nb3.py`.
"""
import nbformat as nbf


def M(src, tags=None):
    c = nbf.v4.new_markdown_cell(src.strip("\n") + "\n")
    if tags:
        c.metadata["tags"] = tags
    return c


def C(src, tags=None):
    c = nbf.v4.new_code_cell(src.strip("\n") + "\n")
    if tags:
        c.metadata["tags"] = tags
    return c
