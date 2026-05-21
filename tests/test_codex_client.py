import unittest
from unittest.mock import patch

from tradingagents.llm_clients import create_llm_client
from tradingagents.llm_clients.codex_client import CodexChatModel
import tradingagents.llm_clients.codex_client as codex_client


class FakeCodexChatModel(CodexChatModel):
    def _run_codex(self, prompt):
        return {
            "content": "analysis complete",
            "tool_calls": [
                {
                    "name": "lookup_price",
                    "arguments": '{"ticker": "NVDA"}',
                }
            ],
        }


class TestCodexClient(unittest.TestCase):
    def test_factory_creates_codex_client(self):
        client = create_llm_client(
            "codex",
            "codex-default",
            codex_timeout=60,
            codex_ignore_user_config=True,
        )

        llm = client.get_llm()

        self.assertIsInstance(llm, CodexChatModel)
        self.assertEqual(llm.model, "codex-default")

    def test_codex_response_maps_to_ai_message(self):
        message = FakeCodexChatModel().invoke("Analyze NVDA")

        self.assertEqual(message.content, "analysis complete")
        self.assertEqual(message.tool_calls[0]["name"], "lookup_price")
        self.assertEqual(message.tool_calls[0]["args"], {"ticker": "NVDA"})

    def test_codex_command_passes_model_and_xhigh_reasoning(self):
        captured = {}

        def fake_run(cmd, input, text, capture_output, timeout, check):
            captured["cmd"] = cmd
            output_path = cmd[cmd.index("-o") + 1]
            with open(output_path, "w", encoding="utf-8") as f:
                f.write('{"content":"ok","tool_calls":[]}')

            class Result:
                returncode = 0
                stdout = ""
                stderr = ""

            return Result()

        llm = CodexChatModel(
            model="gpt-5.5",
            codex_path="/tmp/codex",
            model_reasoning_effort="xhigh",
        )

        with patch.object(codex_client.subprocess, "run", side_effect=fake_run):
            llm.invoke("Analyze NVDA")

        self.assertIn("--model", captured["cmd"])
        self.assertEqual(captured["cmd"][captured["cmd"].index("--model") + 1], "gpt-5.5")
        self.assertIn("-c", captured["cmd"])
        self.assertIn('model_reasoning_effort="xhigh"', captured["cmd"])


if __name__ == "__main__":
    unittest.main()
