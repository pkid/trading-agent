import unittest

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.propagation import Propagator


class DebateDefaultTests(unittest.TestCase):
    def test_project_defaults_use_ten_round_deliberation(self):
        self.assertEqual(DEFAULT_CONFIG["max_debate_rounds"], 10)
        self.assertEqual(DEFAULT_CONFIG["max_risk_discuss_rounds"], 10)
        self.assertGreaterEqual(DEFAULT_CONFIG["max_recur_limit"], 200)

    def test_project_defaults_use_gpt55_xhigh_for_codex(self):
        self.assertEqual(DEFAULT_CONFIG["llm_provider"], "codex")
        self.assertEqual(DEFAULT_CONFIG["quick_think_llm"], "gpt-5.5")
        self.assertEqual(DEFAULT_CONFIG["deep_think_llm"], "gpt-5.5")
        self.assertEqual(DEFAULT_CONFIG["codex_model_reasoning_effort"], "xhigh")

    def test_conditional_logic_defaults_complete_full_investment_rounds(self):
        logic = ConditionalLogic()

        almost_done_state = {
            "investment_debate_state": {
                "count": 19,
                "current_response": "Bear Analyst: final rebuttal pending",
            }
        }
        self.assertEqual(logic.should_continue_debate(almost_done_state), "Bull Researcher")

        done_state = {
            "investment_debate_state": {
                "count": 20,
                "current_response": "Bear Analyst: debate complete",
            }
        }
        self.assertEqual(logic.should_continue_debate(done_state), "Research Manager")

    def test_conditional_logic_defaults_complete_full_risk_rounds(self):
        logic = ConditionalLogic()

        almost_done_state = {
            "risk_debate_state": {
                "count": 29,
                "latest_speaker": "Conservative Analyst",
            }
        }
        self.assertEqual(logic.should_continue_risk_analysis(almost_done_state), "Neutral Analyst")

        done_state = {
            "risk_debate_state": {
                "count": 30,
                "latest_speaker": "Neutral Analyst",
            }
        }
        self.assertEqual(logic.should_continue_risk_analysis(done_state), "Portfolio Manager")

    def test_propagator_default_recursion_limit_supports_full_deliberation(self):
        args = Propagator().get_graph_args()
        self.assertGreaterEqual(args["config"]["recursion_limit"], 200)


if __name__ == "__main__":
    unittest.main()
