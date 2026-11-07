"""Offline regression checks for official reference passage extraction."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("bs4")
spec = importlib.util.spec_from_file_location("prepare_faa_training",
    Path(__file__).parents[1] / "scripts/prepare_faa_training.py")
faa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(faa)


def test_aim_paragraphs_exclude_navigation_and_next_heading():
    text = "A clearance instruction contains operational requirements for the pilot. " * 5
    html = ('<nav>Navigation noise</nav><main><h4 class="paragraph-title" id="4-4-1">'
            '4-4-1. Clearance</h4><p>' + text + '</p><h4 class="paragraph-title" '
            'id="4-4-2">4-4-2. Prefix</h4><p>' + text.replace("clearance", "prefix") + '</p></main>')
    rows = faa.aim_sections(html)
    assert len(rows) == 2
    assert rows[0][0] == "4-4-1"
    assert "Navigation" not in rows[0][2]
    assert "Prefix" not in rows[0][2]


def test_cfr_extraction_preserves_paragraphs_and_limits_selected_parts():
    text = "This paragraph describes an operational instruction and its conditions. " * 5
    xml = ('<ROOT><SECTION><SECTNO>§ 91.123</SECTNO><SUBJECT>Clearances</SUBJECT>'
           '<P>' + text + '</P><P>Additional <E>conditions</E> apply.</P></SECTION>'
           '<SECTION><SECTNO>§ 91.1701</SECTNO><SUBJECT>Excluded</SUBJECT><P>' + text +
           '</P></SECTION></ROOT>')
    rows = faa.cfr_sections(xml)
    assert len(rows) == 1
    assert rows[0][0] == "91.123"
    assert "Additional conditions apply." in rows[0][2]
