"""Notion block-to-markdown renderer.

Converts raw Notion API block JSON (from blocks.children.list) into
Notion-flavored Markdown matching the format used by the Notion MCP tools.
"""

from __future__ import annotations

from typing import Any


def render_blocks(blocks: list[dict], depth: int = 0) -> str:
    """Render a list of blocks to Notion-flavored Markdown.

    Args:
        blocks: List of Notion API block dicts (with children populated).
        depth: Current indentation depth (0 = top level).

    Returns:
        Markdown string. Empty string if no blocks.
    """
    if not blocks:
        return ""

    parts: list[str] = []
    for block in blocks:
        rendered = render_block(block, depth)
        if rendered is not None:
            parts.append(rendered)

    return "\n".join(parts)


def render_block(block: dict, depth: int = 0) -> str | None:
    """Render a single block to Markdown.

    Args:
        block: Notion API block dict.
        depth: Current indentation depth.

    Returns:
        Markdown string, or None if block should be skipped.
    """
    block_type = block.get("type", "")
    indent = "\t" * depth

    handler = _BLOCK_HANDLERS.get(block_type)
    if handler:
        return handler(block, depth, indent)

    # Unsupported block type
    return f"{indent}<!-- Unsupported block type: {block_type} -->"


def render_rich_text(rich_text_array: list[dict]) -> str:
    """Render an array of Notion rich text objects to inline Markdown.

    Args:
        rich_text_array: List of Notion rich text objects.

    Returns:
        Markdown string with inline formatting.
    """
    if not rich_text_array:
        return ""

    parts: list[str] = []
    for rt in rich_text_array:
        parts.append(_render_rich_text_segment(rt))

    return "".join(parts)


# ------------------------------------------------------------------
# Rich text segment rendering
# ------------------------------------------------------------------


def _render_rich_text_segment(rt: dict) -> str:
    """Render a single rich text segment."""
    rt_type = rt.get("type", "text")

    if rt_type == "mention":
        return _render_mention(rt)
    if rt_type == "equation":
        expr = rt.get("equation", {}).get("expression", "")
        return f"${expr}$"

    # Default: text type
    text_obj = rt.get("text", {})
    content = text_obj.get("content", "")
    link = text_obj.get("link")

    # Apply annotations
    annotations = rt.get("annotations", {})
    content = _apply_annotations(content, annotations)

    # Wrap in link if present
    if link and link.get("url"):
        content = f"[{content}]({link['url']})"

    return content


def _render_mention(rt: dict) -> str:
    """Render a mention rich text object."""
    mention = rt.get("mention", {})
    mention_type = mention.get("type", "")

    if mention_type == "user":
        user_id = mention.get("user", {}).get("id", "")
        return f'<mention-user url="user://{user_id}"/>'

    if mention_type == "page":
        page_id = mention.get("page", {}).get("id", "")
        url = f"https://www.notion.so/{page_id.replace('-', '')}"
        return f'<mention-page url="{url}"/>'

    if mention_type == "date":
        date_obj = mention.get("date", {})
        start = date_obj.get("start", "")
        end = date_obj.get("end")
        if end:
            return f'<mention-date start="{start}" end="{end}"/>'
        return f'<mention-date start="{start}"/>'

    if mention_type == "database":
        db_id = mention.get("database", {}).get("id", "")
        url = f"https://www.notion.so/{db_id.replace('-', '')}"
        return f'<mention-page url="{url}"/>'

    if mention_type == "link_preview":
        url = mention.get("link_preview", {}).get("url", "")
        return f"[{url}]({url})"

    # Template mentions (@today, @now)
    if mention_type == "template_mention":
        tmpl = mention.get("template_mention", {})
        tmpl_type = tmpl.get("type", "")
        if tmpl_type == "template_mention_date":
            return tmpl.get("template_mention_date", "today")
        if tmpl_type == "template_mention_user":
            return tmpl.get("template_mention_user", "me")

    return rt.get("plain_text", "")


def _apply_annotations(text: str, annotations: dict) -> str:
    """Apply rich text annotations (bold, italic, etc.) to text."""
    if not text:
        return text

    if annotations.get("code"):
        text = f"`{text}`"
    if annotations.get("bold"):
        text = f"**{text}**"
    if annotations.get("italic"):
        text = f"*{text}*"
    if annotations.get("strikethrough"):
        text = f"~{text}~"
    if annotations.get("underline"):
        text = f"<u>{text}</u>"

    return text


# ------------------------------------------------------------------
# Block type handlers
# ------------------------------------------------------------------


def _render_paragraph(block: dict, depth: int, indent: str) -> str:
    data = block.get("paragraph", {})
    text = render_rich_text(data.get("rich_text", []))
    if not text:
        return f"{indent}<empty-block/>"

    color = data.get("color", "default")
    result = f"{indent}{text}"
    if color != "default":
        result += f' {{color="{color}"}}'

    children = _render_children(block, depth)
    if children:
        result += "\n" + children

    return result


def _render_heading(level: int):
    def handler(block: dict, depth: int, indent: str) -> str:
        type_key = f"heading_{level}"
        data = block.get(type_key, {})
        text = render_rich_text(data.get("rich_text", []))
        prefix = "#" * level
        color = data.get("color", "default")

        result = f"{indent}{prefix} {text}"
        if color != "default":
            result += f' {{color="{color}"}}'

        # Headings can be toggleable (has_children)
        children = _render_children(block, depth)
        if children:
            result += "\n" + children

        return result

    return handler


def _render_bulleted_list_item(block: dict, depth: int, indent: str) -> str:
    data = block.get("bulleted_list_item", {})
    text = render_rich_text(data.get("rich_text", []))
    color = data.get("color", "default")

    result = f"{indent}- {text}"
    if color != "default":
        result += f' {{color="{color}"}}'

    children = _render_children(block, depth + 1)
    if children:
        result += "\n" + children

    return result


def _render_numbered_list_item(block: dict, depth: int, indent: str) -> str:
    data = block.get("numbered_list_item", {})
    text = render_rich_text(data.get("rich_text", []))
    color = data.get("color", "default")

    result = f"{indent}1. {text}"
    if color != "default":
        result += f' {{color="{color}"}}'

    children = _render_children(block, depth + 1)
    if children:
        result += "\n" + children

    return result


def _render_to_do(block: dict, depth: int, indent: str) -> str:
    data = block.get("to_do", {})
    text = render_rich_text(data.get("rich_text", []))
    checked = data.get("checked", False)
    marker = "[x]" if checked else "[ ]"
    color = data.get("color", "default")

    result = f"{indent}- {marker} {text}"
    if color != "default":
        result += f' {{color="{color}"}}'

    children = _render_children(block, depth + 1)
    if children:
        result += "\n" + children

    return result


def _render_toggle(block: dict, depth: int, indent: str) -> str:
    data = block.get("toggle", {})
    text = render_rich_text(data.get("rich_text", []))

    parts = [f"{indent}<details>"]
    parts.append(f"{indent}<summary>{text}</summary>")

    children = _render_children(block, depth + 1)
    if children:
        parts.append(children)

    parts.append(f"{indent}</details>")
    return "\n".join(parts)


def _render_callout(block: dict, depth: int, indent: str) -> str:
    data = block.get("callout", {})
    text = render_rich_text(data.get("rich_text", []))
    color = data.get("color", "default")

    parts = [f'{indent}<callout color="{color}">']
    if text:
        parts.append(f"{indent}\t{text}")

    children = _render_children(block, depth + 1)
    if children:
        parts.append(children)

    parts.append(f"{indent}</callout>")
    return "\n".join(parts)


def _render_quote(block: dict, depth: int, indent: str) -> str:
    data = block.get("quote", {})
    text = render_rich_text(data.get("rich_text", []))
    color = data.get("color", "default")

    lines = text.split("\n") if text else [""]
    result = "\n".join(f"{indent}> {line}" for line in lines)

    if color != "default":
        result += f' {{color="{color}"}}'

    children = _render_children(block, depth + 1)
    if children:
        result += "\n" + children

    return result


def _render_code(block: dict, depth: int, indent: str) -> str:
    data = block.get("code", {})
    text = render_rich_text(data.get("rich_text", []))
    language = data.get("language", "")

    parts = [f"{indent}```{language}"]
    parts.append(f"{indent}{text}")
    parts.append(f"{indent}```")
    return "\n".join(parts)


def _render_divider(block: dict, depth: int, indent: str) -> str:
    return f"{indent}---"


def _render_image(block: dict, depth: int, indent: str) -> str:
    data = block.get("image", {})
    caption = render_rich_text(data.get("caption", []))
    url = _get_media_url(data)
    return f"{indent}![{caption}]({url})"


def _render_file_block(block_type: str):
    def handler(block: dict, depth: int, indent: str) -> str:
        data = block.get(block_type, {})
        caption = render_rich_text(data.get("caption", []))
        url = _get_media_url(data)
        label = caption or block_type.title()
        return f"{indent}[{label}]({url})"

    return handler


def _render_bookmark(block: dict, depth: int, indent: str) -> str:
    data = block.get("bookmark", {})
    caption = render_rich_text(data.get("caption", []))
    url = data.get("url", "")
    label = caption or url
    return f"{indent}[{label}]({url})"


def _render_table(block: dict, depth: int, indent: str) -> str:
    data = block.get("table", {})
    has_column_header = data.get("has_column_header", False)
    children = block.get("children", [])

    parts = [f'{indent}<table header-row="{str(has_column_header).lower()}">']
    for row_block in children:
        row_data = row_block.get("table_row", {})
        cells = row_data.get("cells", [])
        parts.append(f"{indent}<tr>")
        for cell in cells:
            cell_text = render_rich_text(cell)
            parts.append(f"{indent}<td>{cell_text}</td>")
        parts.append(f"{indent}</tr>")
    parts.append(f"{indent}</table>")

    return "\n".join(parts)


def _render_column_list(block: dict, depth: int, indent: str) -> str:
    children = block.get("children", [])

    parts = [f"{indent}<columns>"]
    for col_block in children:
        parts.append(f"{indent}\t<column>")
        col_children = col_block.get("children", [])
        if col_children:
            col_content = render_blocks(col_children, depth + 2)
            parts.append(col_content)
        parts.append(f"{indent}\t</column>")
    parts.append(f"{indent}</columns>")

    return "\n".join(parts)


def _render_child_page(block: dict, depth: int, indent: str) -> str:
    data = block.get("child_page", {})
    title = data.get("title", "")
    page_id = block.get("id", "").replace("-", "")
    url = f"https://www.notion.so/{page_id}"
    return f'{indent}<page url="{url}">{title}</page>'


def _render_child_database(block: dict, depth: int, indent: str) -> str:
    data = block.get("child_database", {})
    title = data.get("title", "")
    db_id = block.get("id", "").replace("-", "")
    url = f"https://www.notion.so/{db_id}"
    return f'{indent}<database url="{url}">{title}</database>'


def _render_equation_display(block: dict, depth: int, indent: str) -> str:
    data = block.get("equation", {})
    expression = data.get("expression", "")
    return f"{indent}$$\n{indent}{expression}\n{indent}$$"


def _render_synced_block(block: dict, depth: int, indent: str) -> str:
    children = _render_children(block, depth)
    return children or ""


def _render_embed(block: dict, depth: int, indent: str) -> str:
    data = block.get("embed", {})
    url = data.get("url", "")
    caption = render_rich_text(data.get("caption", []))
    label = caption or "Embed"
    return f"{indent}[{label}]({url})"


def _render_link_preview(block: dict, depth: int, indent: str) -> str:
    data = block.get("link_preview", {})
    url = data.get("url", "")
    return f"{indent}[{url}]({url})"


def _render_table_of_contents(block: dict, depth: int, indent: str) -> str:
    return f"{indent}[TOC]"


def _render_breadcrumb(block: dict, depth: int, indent: str) -> str:
    return f"{indent}[Breadcrumb]"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _render_children(block: dict, depth: int) -> str:
    """Render a block's children if present."""
    children = block.get("children", [])
    if not children:
        return ""
    return render_blocks(children, depth + 1)


def _get_media_url(data: dict) -> str:
    """Extract URL from a media block (image, file, video, etc.)."""
    if "file" in data:
        return data["file"].get("url", "")
    if "external" in data:
        return data["external"].get("url", "")
    return ""


# ------------------------------------------------------------------
# Block handler dispatch table
# ------------------------------------------------------------------


_BLOCK_HANDLERS: dict[str, Any] = {
    "paragraph": _render_paragraph,
    "heading_1": _render_heading(1),
    "heading_2": _render_heading(2),
    "heading_3": _render_heading(3),
    "bulleted_list_item": _render_bulleted_list_item,
    "numbered_list_item": _render_numbered_list_item,
    "to_do": _render_to_do,
    "toggle": _render_toggle,
    "callout": _render_callout,
    "quote": _render_quote,
    "code": _render_code,
    "divider": _render_divider,
    "image": _render_image,
    "file": _render_file_block("file"),
    "pdf": _render_file_block("pdf"),
    "video": _render_file_block("video"),
    "audio": _render_file_block("audio"),
    "bookmark": _render_bookmark,
    "table": _render_table,
    "column_list": _render_column_list,
    "child_page": _render_child_page,
    "child_database": _render_child_database,
    "equation": _render_equation_display,
    "synced_block": _render_synced_block,
    "embed": _render_embed,
    "link_preview": _render_link_preview,
    "table_of_contents": _render_table_of_contents,
    "breadcrumb": _render_breadcrumb,
}
