import unittest

from tradingagents.llm_clients import create_llm_client
from tradingagents.llm_clients.codex_client import CodexChatModel


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


if __name__ == "__main__":
    unittest.main()
