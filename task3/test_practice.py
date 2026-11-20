"""真实框架继承＋预设模型响应；所有测试不发送网络请求。"""
import json
import os
import unittest
from unittest.mock import patch

from hello_agents import Config, HelloAgentsLLM, Message
from hello_agents.core.agent import Agent
from pydantic import ValidationError

from practice import GeminiLLM, ScoredReflectionAgent, TreeOfThoughtAgent, parse_review


def review(score, complete=False):
    return json.dumps({"score": score, "feedback": "检查任务约束", "complete": complete})


class ScriptedLLM:
    provider = "offline"

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return next(self.outputs)


class ExtensionTests(unittest.TestCase):
    @patch.object(HelloAgentsLLM, "_create_client", return_value=None)
    def test_gemini_detection_and_endpoint(self, _client):
        env = {"GEMINI_API_KEY": "fake-test-only", "GEMINI_MODEL_ID": "test-model",
               "LLM_BASE_URL": "http://localhost:11434/v1"}
        with patch.dict(os.environ, env, clear=True):
            llm = GeminiLLM()
        self.assertIsInstance(llm, HelloAgentsLLM)
        self.assertEqual(llm.provider, "gemini")
        self.assertEqual(llm.base_url, "https://generativelanguage.googleapis.com/v1beta/openai/")
        self.assertEqual(llm.model, "test-model")

    @patch.object(HelloAgentsLLM, "_create_client", return_value=None)
    def test_explicit_provider_and_parent_fallback(self, _client):
        env = {"GEMINI_API_KEY": "fake-gemini", "OPENAI_API_KEY": "fake-openai"}
        with patch.dict(os.environ, env, clear=True):
            llm = GeminiLLM(provider="openai", model="test-model")
        self.assertEqual(llm.provider, "openai")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-openai"}, clear=True):
            llm = GeminiLLM(model="test-model")
        self.assertEqual(llm.provider, "openai")

    def test_gemini_missing_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                GeminiLLM(provider="gemini")

    def test_actual_provider_priority(self):
        # 调用真实0.1.1检测方法，不创建客户端或打印密钥。
        llm = object.__new__(HelloAgentsLLM)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-test-only",
                                    "LLM_BASE_URL": "http://localhost:11434/v1"}, clear=True):
            self.assertEqual(llm._auto_detect_provider(None, None), "openai")

    def test_base_contract_and_config_identity(self):
        self.assertIn("run", Agent.__abstractmethods__)
        self.assertNotIn("_execute", Agent.__dict__)
        self.assertIsNot(Config(), Config())
        with self.assertRaises(ValidationError):
            Message(role="bad-role", content="test")

    def test_reflection_stops_at_threshold(self):
        llm = ScriptedLLM(["first", review(90, True)])
        agent = ScoredReflectionAgent("test", llm)
        self.assertEqual(agent.run("task"), "first")
        self.assertEqual(llm.calls, 2)
        self.assertEqual(agent.stop_reason, "threshold")

    def test_reflection_revision_and_history(self):
        llm = ScriptedLLM(["first", review(40), "improved", review(90, True)])
        agent = ScoredReflectionAgent("test", llm)
        self.assertEqual(agent.run("task"), "improved")
        self.assertEqual(llm.calls, 4)
        self.assertEqual([m.role for m in agent.get_history()], ["user", "assistant"])

    def test_reflection_retains_best_and_budget(self):
        llm = ScriptedLLM(["first", review(70), "worse", review(50)])
        agent = ScoredReflectionAgent("test", llm, max_revisions=1)
        self.assertEqual(agent.run("task"), "first")
        self.assertEqual(agent.stop_reason, "revision_budget")

    def test_conflicting_review_is_not_success(self):
        agent = ScoredReflectionAgent("test", ScriptedLLM(["partial", review(99)]))
        agent.run("task")
        self.assertEqual(agent.stop_reason, "review_conflict")

    def test_invalid_review_scores(self):
        for score in (True, -1, 101, float("nan"), float("inf"), "90"):
            with self.subTest(score=score):
                with self.assertRaises(ValueError):
                    parse_review(review(score))

    def test_tree_selects_best_candidate(self):
        llm = ScriptedLLM(['{"candidates":["A","B"]}', review(30), review(90, True)])
        agent = TreeOfThoughtAgent("test", llm)
        self.assertIsInstance(agent, Agent)
        self.assertEqual(agent.run("task"), "B")
        self.assertEqual(agent.stop_reason, "complete")
        self.assertEqual(agent.calls, 3)

    def test_tree_depth_and_call_budgets(self):
        llm = ScriptedLLM(['{"candidates":["A","B"]}', review(30), review(50)])
        agent = TreeOfThoughtAgent("test", llm, max_depth=1)
        self.assertEqual(agent.run("task"), "B")
        self.assertEqual(agent.stop_reason, "depth_budget")
        llm = ScriptedLLM(['{"candidates":["A","B"]}', review(30)])
        agent = TreeOfThoughtAgent("test", llm, max_calls=2)
        self.assertEqual(agent.run("task"), "A")
        self.assertEqual(agent.calls, 2)
        self.assertEqual(agent.stop_reason, "call_budget")

    def test_tree_rejects_duplicate_and_malformed_candidates(self):
        for raw in ['{"candidates":["A","A"]}', '[]', '{"candidates":["A",7]}']:
            with self.subTest(raw=raw):
                agent = TreeOfThoughtAgent("test", ScriptedLLM([raw]))
                with self.assertRaises(ValueError):
                    agent.run("task")


if __name__ == "__main__":
    unittest.main()
