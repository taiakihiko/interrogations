from pathlib import Path
from markdown_it import MarkdownIt

text = Path("README.md").read_text(encoding="utf-8")
tokens = MarkdownIt().parse(text)

for token in tokens:
    #print(token.type, repr(token.content))
    print(token, "\n")

print("---------------------")

def extract_section(text: str, heading: str) -> str | None:
    tokens = MarkdownIt().parse(text)
    lines = text.splitlines(keepends=True)

    start = None
    level = None

    for i, token in enumerate(tokens):
        if token.type != "heading_open":
            continue

        current_level = int(token.tag[1:])
        title = tokens[i + 1].content

        if start is not None:
            if current_level <= level:
                return "".join(lines[start:token.map[0]]).strip()
        elif title == heading:
            start = token.map[1]
            level = current_level

    if start is not None:
        return "".join(lines[start:]).strip()

    return None


print(extract_section(text, "インストール"))
