"""Print a Notion page's blocks as indented text, following the page's content order."""
import json
import sys

data = json.load(open(sys.argv[1]))
page_id = sys.argv[2]
blocks = {}
for bid, rec in data.get("recordMap", {}).get("block", {}).items():
    val = rec.get("value", {})
    val = val.get("value", val)  # some responses nest value twice
    blocks[bid] = val


def text(rich) -> str:
    out = []
    for seg in rich or []:
        s = seg[0]
        for fmt in seg[1] if len(seg) > 1 else []:
            if fmt[0] == "a":
                s += f" <{fmt[1]}>"
        out.append(s)
    return "".join(out)


def walk(bid: str, depth: int) -> None:
    b = blocks.get(bid)
    if not b:
        print("  " * depth + f"[missing block {bid}]")
        return
    t = b.get("type")
    title = text(b.get("properties", {}).get("title"))
    marker = {"bulleted_list": "- ", "numbered_list": "1. ", "to_do": "[ ] ", "header": "# ", "sub_header": "## ",
              "sub_sub_header": "### ", "toggle": "> "}.get(t, "")
    if title or t not in ("page", "column_list", "column"):
        print("  " * depth + marker + (title or f"[{t}]"))
    for child in b.get("content", []) or []:
        walk(child, depth + 1 if t not in ("page", "column_list", "column") else depth)


walk(page_id, 0)
