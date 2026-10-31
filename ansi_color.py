import re

GRAY = 90
RED = 91
GREEN = 92
YELLOW = 93
BLUE = 94
PURPLE = 95
CYAN = 96
WHITE = 97

DARK_GRAY = 30
DARK_RED = 31
DARK_GREEN = 32
DARK_YELLOW = 33
DARK_BLUE = 34
DARK_PURPLE = 35
DARK_CYAN = 36
DARK_WHITE = 37

KEEP = 0

def code(code: int):
    if code == KEEP:
        return
    print(s(code), end="")

def s(code: int) -> str:
    return f"\x1b[{code}m"

def origin():
    print(f"\x1b[0m", end="")

def strip_ansi(text: str) -> str:
    ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)
