"""Tests for the Notion block-to-markdown renderer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.notion_renderer import render_block, render_blocks, render_rich_text

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    """Load a named block from the blocks_raw.json fixture."""
    path = FIXTURES_DIR / "blocks_raw.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data[name]


# ------------------------------------------------------------------
# Rich text rendering
# ------------------------------------------------------------------


class TestRenderRichText:
    def test_plain_text(self):
        rt = [_load_fixture("rich_text_bold")]
        rt[0]["annotations"]["bold"] = False
        result = render_rich_text(rt)
        assert result == "bold text"

    def test_bold(self):
        rt = [_load_fixture("rich_text_bold")]
        result = render_rich_text(rt)
        assert result == "**bold text**"

    def test_italic(self):
        rt = [_load_fixture("rich_text_italic")]
        result = render_rich_text(rt)
        assert result == "*italic text*"

    def test_code(self):
        rt = [_load_fixture("rich_text_code")]
        result = render_rich_text(rt)
        assert result == "`code text`"

    def test_strikethrough(self):
        rt = [_load_fixture("rich_text_strikethrough")]
        result = render_rich_text(rt)
        assert result == "~struck~"

    def test_underline(self):
        rt = [_load_fixture("rich_text_underline")]
        result = render_rich_text(rt)
        assert result == "<u>underlined</u>"

    def test_bold_italic(self):
        rt = [_load_fixture("rich_text_bold_italic")]
        result = render_rich_text(rt)
        assert "**" in result
        assert "*" in result
        assert "both" in result

    def test_link(self):
        rt = [_load_fixture("rich_text_link")]
        result = render_rich_text(rt)
        assert result == "[click here](https://example.com)"

    def test_mention_user(self):
        rt = [_load_fixture("rich_text_mention_user")]
        result = render_rich_text(rt)
        assert '<mention-user url="user://1d2d872b-594c-813d-9bbb-00025fbef218"/>' in result

    def test_mention_page(self):
        rt = [_load_fixture("rich_text_mention_page")]
        result = render_rich_text(rt)
        assert '<mention-page url="https://www.notion.so/aabb1122ccdd3344eeff556677889900"/>' in result

    def test_mention_date(self):
        rt = [_load_fixture("rich_text_mention_date")]
        result = render_rich_text(rt)
        assert '<mention-date start="2026-01-08"/>' in result

    def test_equation(self):
        rt = [_load_fixture("rich_text_equation")]
        result = render_rich_text(rt)
        assert result == "$x^2 + y^2 = z^2$"

    def test_empty_array(self):
        result = render_rich_text([])
        assert result == ""

    def test_multiple_segments(self):
        plain = {
            "type": "text",
            "text": {"content": "Hello ", "link": None},
            "annotations": {"bold": False, "italic": False, "strikethrough": False, "code": False, "underline": False, "color": "default"},
        }
        bold = _load_fixture("rich_text_bold")
        result = render_rich_text([plain, bold])
        assert result == "Hello **bold text**"


# ------------------------------------------------------------------
# Block rendering
# ------------------------------------------------------------------


class TestRenderBlock:
    def test_paragraph(self):
        block = _load_fixture("paragraph_basic")
        result = render_block(block)
        assert result == "Hello world"

    def test_paragraph_empty(self):
        block = _load_fixture("paragraph_empty")
        result = render_block(block)
        assert result == "<empty-block/>"

    def test_paragraph_with_color(self):
        block = _load_fixture("paragraph_with_color")
        result = render_block(block)
        assert 'Colored text' in result
        assert '{color="red"}' in result

    def test_heading_1(self):
        block = _load_fixture("heading_1")
        result = render_block(block)
        assert result == "# Main Title"

    def test_heading_2(self):
        block = _load_fixture("heading_2")
        result = render_block(block)
        assert result == "## Section Title"

    def test_heading_3(self):
        block = _load_fixture("heading_3")
        result = render_block(block)
        assert result == "### Subsection"

    def test_heading_with_color(self):
        block = _load_fixture("heading_with_color")
        result = render_block(block)
        assert result == '# Red Heading {color="red"}'

    def test_bulleted_list(self):
        block = _load_fixture("bulleted_list_item")
        result = render_block(block)
        assert result == "- Bullet point"

    def test_numbered_list(self):
        block = _load_fixture("numbered_list_item")
        result = render_block(block)
        assert result == "1. First item"

    def test_todo_unchecked(self):
        block = _load_fixture("to_do_unchecked")
        result = render_block(block)
        assert result == "- [ ] Task not done"

    def test_todo_checked(self):
        block = _load_fixture("to_do_checked")
        result = render_block(block)
        assert result == "- [x] Task completed"

    def test_toggle(self):
        block = _load_fixture("toggle")
        result = render_block(block)
        assert "<details>" in result
        assert "<summary>Click to expand</summary>" in result
        assert "Hidden content" in result
        assert "</details>" in result

    def test_callout(self):
        block = _load_fixture("callout")
        result = render_block(block)
        assert '<callout color="blue_bg">' in result
        assert "**Important note**" in result
        assert "Details here" in result
        assert "</callout>" in result

    def test_quote(self):
        block = _load_fixture("quote")
        result = render_block(block)
        assert result == "> A wise quote"

    def test_code_block(self):
        block = _load_fixture("code")
        result = render_block(block)
        assert "```python" in result
        assert "print('hello')" in result
        assert result.rstrip().endswith("```")

    def test_divider(self):
        block = _load_fixture("divider")
        result = render_block(block)
        assert result == "---"

    def test_image_external(self):
        block = _load_fixture("image_external")
        result = render_block(block)
        assert result == "![A diagram](https://example.com/image.png)"

    def test_image_file(self):
        block = _load_fixture("image_file")
        result = render_block(block)
        assert "![](https://prod-files-secure.s3.us-west-2.amazonaws.com/image.png)" in result

    def test_bookmark(self):
        block = _load_fixture("bookmark")
        result = render_block(block)
        assert result == "[Useful article](https://example.com/article)"

    def test_table(self):
        block = _load_fixture("table")
        result = render_block(block)
        assert '<table header-row="true">' in result
        assert "<tr>" in result
        assert "<td>Header 1</td>" in result
        assert "<td>Cell A</td>" in result
        assert "</table>" in result

    def test_columns(self):
        block = _load_fixture("column_list")
        result = render_block(block)
        assert "<columns>" in result
        assert "<column>" in result
        assert "Left column" in result
        assert "Right column" in result
        assert "</columns>" in result

    def test_child_page(self):
        block = _load_fixture("child_page")
        result = render_block(block)
        assert '<page url="https://www.notion.so/aaaa1111bbbb2222cccc333344445555">' in result
        assert "Sub Page Title" in result
        assert "</page>" in result

    def test_child_database(self):
        block = _load_fixture("child_database")
        result = render_block(block)
        assert '<database url="https://www.notion.so/dddd6666eeee7777ffff888899990000">' in result
        assert "Embedded DB" in result

    def test_equation_display(self):
        block = _load_fixture("equation_display")
        result = render_block(block)
        assert "$$" in result
        assert "E = mc^2" in result

    def test_embed(self):
        block = _load_fixture("embed")
        result = render_block(block)
        assert "[Embed](https://example.com/embed)" in result

    def test_link_preview(self):
        block = _load_fixture("link_preview")
        result = render_block(block)
        assert "[https://example.com/preview](https://example.com/preview)" in result

    def test_synced_block(self):
        block = _load_fixture("synced_block")
        result = render_block(block)
        assert "Synced content" in result

    def test_file_block(self):
        block = _load_fixture("file_block")
        result = render_block(block)
        assert "[Important doc](https://example.com/doc.pdf)" in result

    def test_unsupported_type(self):
        block = {"type": "unknown_widget", "id": "xxx"}
        result = render_block(block)
        assert "<!-- Unsupported block type: unknown_widget -->" in result


# ------------------------------------------------------------------
# Multi-block rendering
# ------------------------------------------------------------------


class TestRenderBlocks:
    def test_empty_list(self):
        result = render_blocks([])
        assert result == ""

    def test_multiple_blocks(self):
        blocks = [
            _load_fixture("heading_1"),
            _load_fixture("paragraph_basic"),
        ]
        result = render_blocks(blocks)
        assert "# Main Title" in result
        assert "Hello world" in result
        lines = result.split("\n")
        assert len(lines) == 2

    def test_nested_blocks(self):
        blocks = [_load_fixture("callout")]
        result = render_blocks(blocks)
        assert '<callout color="blue_bg">' in result
        assert "Details here" in result
        assert "</callout>" in result

    def test_mixed_block_types(self):
        blocks = [
            _load_fixture("heading_2"),
            _load_fixture("paragraph_basic"),
            _load_fixture("bulleted_list_item"),
            _load_fixture("divider"),
            _load_fixture("code"),
        ]
        result = render_blocks(blocks)
        assert "## Section Title" in result
        assert "Hello world" in result
        assert "- Bullet point" in result
        assert "---" in result
        assert "```python" in result

    def test_indented_at_depth(self):
        blocks = [_load_fixture("paragraph_basic")]
        result = render_blocks(blocks, depth=1)
        assert result.startswith("\t")
        assert "Hello world" in result

    def test_bulleted_list_with_children(self):
        block = _load_fixture("bulleted_list_item")
        block = {**block, "has_children": True, "children": [
            {
                "id": "child-1",
                "type": "bulleted_list_item",
                "has_children": False,
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": "Sub-item", "link": None}, "annotations": {"bold": False, "italic": False, "strikethrough": False, "code": False, "underline": False, "color": "default"}}],
                    "color": "default",
                },
            }
        ]}
        result = render_block(block)
        assert "- Bullet point" in result
        assert "\t- Sub-item" in result
