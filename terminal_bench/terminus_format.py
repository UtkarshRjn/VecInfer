"""Make Llama 3.1 replies executable by Terminus-2.

Terminus-2 only types into the terminal when the model returns one JSON object
with ``analysis``, ``plan``, and ``commands[].keystrokes``. Llama 3.1 8B
usually answers with a markdown script instead, and Harbor then runs nothing.
This module turns that draft into the JSON object Terminus-2 already knows how
to execute.
"""

from __future__ import annotations

import json
import re
from typing import Any

TERMINUS_SYSTEM = (
    "You control a Linux terminal for Terminus-2. "
    "Reply with one JSON object and no markdown. "
    "Required keys: analysis (string), plan (string), "
    "commands (array of objects with keystrokes and duration). "
    "keystrokes is literal terminal input and must end with a newline. "
    "Set task_complete to true only after the requested files exist."
)

_FENCE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
_PROMPT = re.compile(r"^(?:[^\s#]+@[^\s#]+[#\$]|\$)\s?(.*)$")
# tee/cat with only a filename never sees EOF, so it reads every later keystroke.
_BARE_STDIN = re.compile(r"^(?:tee|cat)\s+\S+\s*$")
# tail -f never returns, so every later keystroke is typed into it.
_TAIL_F = re.compile(r"^tail\s+-[Ff]\b\s*(.*)$")
# A bare script with no arguments is how the model starts a server.
_FOREGROUND_SERVER = re.compile(r"^python3?\s+\S+\.py$")
_KEYSTROKE_OBJECT = re.compile(
    r'\{\s*"keystrokes"\s*:\s*"((?:\\.|[^"\\])*)"\s*,\s*"duration"\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)\s*\}'
)
_ANALYSIS = re.compile(r'"analysis"\s*:\s*"((?:\\.|[^"\\])*)"')
_PLAN = re.compile(r'"plan"\s*:\s*"((?:\\.|[^"\\])*)"')
_VALID_ESCAPE = set('"\\/bfnrt')


def is_terminus_chat(messages: list[dict[str, Any]]) -> bool:
    """True when this request is a Terminus-2 turn, not a summary subagent."""
    blob = "\n".join(_text(message.get("content")) for message in messages)
    return "keystrokes" in blob or "Format your response as JSON" in blob


def repair_terminus_response(text: str) -> str:
    """Return Terminus-2 JSON, using the model's own code when it did not."""
    native = _native_terminus(text)
    if native is not None:
        return json.dumps(native)
    commands = _commands_from_draft(text)
    if not commands:
        return text
    return json.dumps(
        {
            "analysis": "The draft answer was not Terminus JSON. Run its commands.",
            "plan": "Execute the shell or Python from the draft, then check the result.",
            "commands": [{"keystrokes": command, "duration": 10.0} for command in commands],
            "task_complete": False,
        }
    )


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return "" if content is None else str(content)


def _native_terminus(text: str) -> dict[str, Any] | None:
    for candidate in _json_objects(text):
        data = _loads_lenient(candidate)
        if _is_terminus(data):
            return _sanitize_commands(data)
    salvaged = _salvage_truncated(text)
    if salvaged is not None:
        return _sanitize_commands(salvaged)
    return None


def _is_terminus(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    commands = data.get("commands")
    if not isinstance(data.get("analysis"), str) or not isinstance(data.get("plan"), str):
        return False
    if not isinstance(commands, list):
        return False
    return not commands or all(
        isinstance(item, dict) and isinstance(item.get("keystrokes"), str) for item in commands
    )


def _loads_lenient(candidate: str) -> Any:
    """Parse Terminus JSON, repairing backslashes the model did not escape."""
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_fix_invalid_escapes(candidate))
    except json.JSONDecodeError:
        return None


def _fix_invalid_escapes(text: str) -> str:
    out: list[str] = []
    index = 0
    in_string = False
    while index < len(text):
        char = text[index]
        if not in_string:
            if char == '"':
                in_string = True
            out.append(char)
            index += 1
            continue
        if char == '"':
            in_string = False
            out.append(char)
            index += 1
            continue
        if char != "\\":
            out.append(char)
            index += 1
            continue
        nxt = text[index + 1] if index + 1 < len(text) else ""
        if nxt == "u" and re.fullmatch(r"[0-9a-fA-F]{4}", text[index + 2 : index + 6] or ""):
            out.append(text[index : index + 6])
            index += 6
            continue
        if nxt in _VALID_ESCAPE:
            out.append(char)
            out.append(nxt)
            index += 2
            continue
        out.append("\\\\")
        index += 1
    return "".join(out)


def _salvage_truncated(text: str) -> dict[str, Any] | None:
    """Keep complete commands when the token limit cuts the JSON object off."""
    analysis = _json_string(_ANALYSIS, text)
    plan = _json_string(_PLAN, text)
    if analysis is None or plan is None:
        return None
    commands = []
    for match in _KEYSTROKE_OBJECT.finditer(text):
        try:
            keystrokes = json.loads('"' + match.group(1) + '"')
        except json.JSONDecodeError:
            continue
        commands.append({"keystrokes": keystrokes, "duration": float(match.group(2))})
    if not commands:
        return None
    return {"analysis": analysis, "plan": plan, "commands": commands, "task_complete": False}


def _json_string(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    if match is None:
        return None
    try:
        return json.loads('"' + match.group(1) + '"')
    except json.JSONDecodeError:
        return None


def _sanitize_commands(data: dict[str, Any]) -> dict[str, Any]:
    """Drop commands that would swallow the rest of the terminal session."""
    prepared = []
    for item in data.get("commands") or []:
        keystrokes = _sanitize_keystrokes(item["keystrokes"])
        if keystrokes is None:
            continue
        prepared.append((item, keystrokes))
    cleaned = []
    for index, (item, keystrokes) in enumerate(prepared):
        more_after = index < len(prepared) - 1
        updated = dict(item)
        updated["keystrokes"] = _background_blocking(keystrokes, more_after)
        cleaned.append(updated)
    updated_data = dict(data)
    updated_data["commands"] = cleaned
    return updated_data


def _sanitize_keystrokes(text: str) -> str | None:
    if _BARE_STDIN.match(text.strip()):
        return None
    # "cd.." is not a command. The model means "cd ..".
    # "tail -f" never returns, so later keystrokes never reach the shell.
    lines = []
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        newline = "\n" if line.endswith("\n") else ""
        if stripped == "cd..":
            line = "cd .." + newline
        else:
            follow = _TAIL_F.match(stripped)
            if follow:
                target = follow.group(1).strip()
                line = ("tail -n 80 " + target).rstrip() + newline
        lines.append(line)
    text = "".join(lines)
    # The model often writes the two characters "\n" where it wants a newline.
    if "\\n" in text and ("<<<" in text or "printf " in text):
        text = text.replace("\\n", "\n")
        if not text.endswith("\n"):
            text += "\n"
    return text


def _background_blocking(text: str, more_after: bool) -> str:
    """Background a server when later keystrokes still need the shell."""
    lines = text.splitlines(keepends=True)
    rewritten = []
    for index, line in enumerate(lines):
        body = line.strip()
        followed = more_after or index < len(lines) - 1
        if followed and _FOREGROUND_SERVER.match(body):
            newline = "\n" if line.endswith("\n") else ""
            line = body + " &" + (newline or "\n")
        rewritten.append(line)
    return "".join(rewritten)


def _json_objects(text: str) -> list[str]:
    found = []
    start = -1
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                found.append(text[start : index + 1])
                start = -1
    return found


def _commands_from_draft(text: str) -> list[str]:
    commands = []
    for language, body in _FENCE.findall(text):
        command = _fence_to_command(language.strip().lower(), body)
        if command:
            commands.append(command)
    return commands


def _fence_to_command(language: str, body: str) -> str | None:
    stripped = body.strip("\n")
    if not stripped.strip():
        return None
    if language in {"bash", "sh", "shell", "console", "zsh"}:
        script = _strip_prompts(stripped)
        if not script.strip():
            return None
        return script if script.endswith("\n") else script + "\n"
    if language in {"python", "py", "python3"} or (
        language == "" and _looks_like_python(stripped)
    ):
        return _python_command(stripped)
    return None


def _looks_like_python(body: str) -> bool:
    head = body.lstrip()
    return head.startswith(("import ", "from ", "def ", "class ", "print(", "with ", "for "))


def _strip_prompts(script: str) -> str:
    lines = []
    for line in script.splitlines():
        match = _PROMPT.match(line.strip())
        lines.append(match.group(1) if match else line)
    return "\n".join(lines).strip("\n")


def _python_command(code: str) -> str:
    delimiter = "PY"
    while f"\n{delimiter}\n" in f"\n{code}\n" or code.strip() == delimiter:
        delimiter += "X"
    return f"python - << '{delimiter}'\n{code.rstrip()}\n{delimiter}\n"
