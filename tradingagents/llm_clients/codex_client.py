from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field

from .base_client import BaseLLMClient


_DEFAULT_CODEX_PATH = "/Applications/Codex.app/Contents/Resources/codex"
_CODEX_DEFAULT_MODEL = "codex-default"
_DEFAULT_DISABLED_FEATURES = (
    "plugins",
    "apps",
    "browser_use",
    "computer_use",
    "image_generation",
    "multi_agent",
    "shell_tool",
)

_CODEX_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "content": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    # Strict JSON schemas do not allow arbitrary object keys here,
                    # so Codex returns the tool arguments as an encoded JSON object.
                    "arguments": {"type": "string"},
                },
                "required": ["name", "arguments"],
            },
        },
    },
    "required": ["content", "tool_calls"],
}


class CodexChatModel(BaseChatModel):
    """LangChain chat model adapter for the local Codex desktop/CLI login.

    Codex is an agent CLI rather than a chat-completions API, so this adapter
    invokes `codex exec` with a strict output schema and maps the final JSON
    response back to LangChain's `AIMessage` shape.
    """

    model: str = _CODEX_DEFAULT_MODEL
    codex_path: Optional[str] = None
    cwd: Optional[str] = None
    timeout: int = 600
    sandbox: str = "read-only"
    profile: Optional[str] = None
    model_reasoning_effort: Optional[str] = None
    ephemeral: bool = True
    ignore_rules: bool = True
    ignore_user_config: bool = False
    disabled_features: Sequence[str] = Field(
        default_factory=lambda: list(_DEFAULT_DISABLED_FEATURES)
    )
    bound_tools: Sequence[Any] = Field(default_factory=list)
    tool_choice: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return "codex"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "codex_path": self._resolve_codex_path(),
            "sandbox": self.sandbox,
            "profile": self.profile,
        }

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Any | BaseTool],
        *,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ):
        updates = {"bound_tools": list(tools), "tool_choice": tool_choice}
        if hasattr(self, "model_copy"):
            return self.model_copy(update=updates)
        return self.copy(update=updates)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        output = self._run_codex(self._build_prompt(messages, stop=stop))
        message = AIMessage(
            content=output["content"],
            tool_calls=self._parse_tool_calls(output["tool_calls"]),
            response_metadata={
                "provider": "codex",
                "model": self.model,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _build_prompt(
        self, messages: list[BaseMessage], stop: Optional[list[str]] = None
    ) -> str:
        payload = {
            "messages": [self._message_to_dict(message) for message in messages],
            "tools": self._tool_specs(),
            "tool_choice": self.tool_choice,
            "stop": stop or [],
        }

        instructions = [
            "You are serving as a chat-model backend for TradingAgents.",
            "Do not edit files. Do not make network calls or run shell commands.",
            "Use only the conversation, tool schemas, and tool results provided below.",
            "Your final answer must satisfy the JSON schema supplied by the caller.",
            "Set `content` to your natural-language answer when no more tools are needed.",
            "Set `tool_calls` to [] unless you need TradingAgents to execute a provided tool.",
            "When calling a tool, set `content` to an empty string and include only tools from the provided list.",
            "Each tool call's `arguments` value must be a JSON object encoded as a string.",
            "Never invent tool results; request a tool call and wait for its result in a later message.",
        ]

        if not self.bound_tools:
            instructions.append("No tools are available, so `tool_calls` must be [].")

        return "\n".join(instructions) + "\n\n" + json.dumps(payload, indent=2)

    def _tool_specs(self) -> list[dict[str, Any]]:
        specs = []
        for tool in self.bound_tools:
            try:
                specs.append(convert_to_openai_tool(tool))
            except Exception:
                specs.append(
                    {
                        "type": "function",
                        "function": {
                            "name": getattr(tool, "name", tool.__class__.__name__),
                            "description": getattr(tool, "description", ""),
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "additionalProperties": True,
                            },
                        },
                    }
                )
        return specs

    def _message_to_dict(self, message: BaseMessage) -> dict[str, Any]:
        data: dict[str, Any] = {
            "role": getattr(message, "type", message.__class__.__name__),
            "content": self._safe_content(message.content),
        }

        name = getattr(message, "name", None)
        if name:
            data["name"] = name

        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id:
            data["tool_call_id"] = tool_call_id

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            data["tool_calls"] = tool_calls

        return data

    def _safe_content(self, content: Any) -> Any:
        if isinstance(content, (str, int, float, bool)) or content is None:
            return content
        try:
            json.dumps(content)
            return content
        except TypeError:
            return str(content)

    def _parse_tool_calls(self, raw_tool_calls: Any) -> list[dict[str, Any]]:
        tool_calls = []
        for raw_call in raw_tool_calls or []:
            name = raw_call.get("name", "")
            arguments = raw_call.get("arguments", "{}")
            try:
                args = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Codex returned invalid JSON arguments for tool '{name}': {arguments}"
                ) from exc

            if not isinstance(args, dict):
                raise ValueError(
                    f"Codex returned non-object arguments for tool '{name}': {arguments}"
                )

            tool_calls.append(
                {
                    "name": name,
                    "args": args,
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                }
            )

        return tool_calls

    def _run_codex(self, prompt: str) -> dict[str, Any]:
        codex_path = self._resolve_codex_path()
        with tempfile.TemporaryDirectory(prefix="tradingagents-codex-") as tmpdir:
            tmp_path = Path(tmpdir)
            schema_path = tmp_path / "schema.json"
            output_path = tmp_path / "output.json"
            schema_path.write_text(json.dumps(_CODEX_OUTPUT_SCHEMA), encoding="utf-8")

            cmd = [
                codex_path,
                "exec",
                "--output-schema",
                str(schema_path),
                "-o",
                str(output_path),
                "--color",
                "never",
                "--skip-git-repo-check",
                "-s",
                self.sandbox,
            ]

            if self.ephemeral:
                cmd.append("--ephemeral")
            if self.ignore_rules:
                cmd.append("--ignore-rules")
            if self.ignore_user_config:
                cmd.append("--ignore-user-config")
            if self.profile:
                cmd.extend(["--profile", self.profile])
            if self.model_reasoning_effort:
                cmd.extend([
                    "-c",
                    f'model_reasoning_effort="{self.model_reasoning_effort}"',
                ])
            for feature in self.disabled_features:
                cmd.extend(["--disable", feature])
            if self.cwd:
                cmd.extend(["--cd", self.cwd])
            if self.model and self.model != _CODEX_DEFAULT_MODEL:
                cmd.extend(["--model", self.model])

            cmd.append("-")

            result = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )

            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip()
                raise RuntimeError(
                    f"Codex exec failed with exit code {result.returncode}: {detail[-4000:]}"
                )

            if not output_path.exists():
                detail = (result.stderr or result.stdout).strip()
                raise RuntimeError(
                    f"Codex exec did not write an output file. Output: {detail[-4000:]}"
                )

            return json.loads(output_path.read_text(encoding="utf-8"))

    def _resolve_codex_path(self) -> str:
        candidates = [
            self.codex_path,
            os.getenv("CODEX_EXECUTABLE"),
            shutil.which("codex"),
            _DEFAULT_CODEX_PATH,
        ]
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return candidate
        raise FileNotFoundError(
            "Could not find the Codex CLI. Set CODEX_EXECUTABLE or config['codex_path']."
        )


class CodexClient(BaseLLMClient):
    """Client that routes TradingAgents LLM calls through the local Codex CLI."""

    def get_llm(self) -> Any:
        return CodexChatModel(
            model=self.model or _CODEX_DEFAULT_MODEL,
            codex_path=self.kwargs.get("codex_path"),
            cwd=self.kwargs.get("codex_cwd") or os.getcwd(),
            timeout=self.kwargs.get("codex_timeout", 600),
            sandbox=self.kwargs.get("codex_sandbox", "read-only"),
            profile=self.kwargs.get("codex_profile"),
            model_reasoning_effort=self.kwargs.get("codex_model_reasoning_effort"),
            ephemeral=self.kwargs.get("codex_ephemeral", True),
            ignore_rules=self.kwargs.get("codex_ignore_rules", True),
            ignore_user_config=self.kwargs.get("codex_ignore_user_config", False),
            disabled_features=self.kwargs.get(
                "codex_disable_features", list(_DEFAULT_DISABLED_FEATURES)
            ),
            callbacks=self.kwargs.get("callbacks"),
        )

    def validate_model(self) -> bool:
        return True
