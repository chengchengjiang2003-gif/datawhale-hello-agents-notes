"""离线验证协议、算术边界和纠错流程；不代表真实模型表现。"""
import json
import unittest
from chapter4_practice import calculator, parse_action, run_agent, ToolExecutor


class PracticeTests(unittest.TestCase):
    def test_arithmetic(self):
        for expression, expected in [("(123+456)×789/12", 38069.25),
                                     ("-2 + 3 * (4 - 1)", 7), ("2**10", 1024)]:
            with self.subTest(expression=expression):
                self.assertEqual(calculator(expression), expected)

    def test_reject_executable_or_unbounded_input(self):
        for expression in ["__import__('os')", "x+1", "[1][0]", "True+1", "2**1000000",
                           "1/0", "1e101", "9" * 301]:
            with self.subTest(expression=expression):
                with self.assertRaises((ValueError, SyntaxError, ZeroDivisionError)):
                    calculator(expression)

    def test_protocol(self):
        self.assertEqual(parse_action('{"type":"finish","answer":"a[b]"}')["answer"], "a[b]")
        for raw in ['```json\n{}\n```', '[]', '{"type":"finish","answer":3}',
                    '{"type":"tool","name":"Calculator","arguments":[]}',
                    '{"type":"finish","answer":"a","extra":1}']:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_action(raw)

    def test_failure_then_correction(self):
        executor = ToolExecutor()
        executor.registerTool("Calculator", "参数expression", calculator)
        outputs = iter([
            '{"type":"tool","name":"Oops","arguments":{}}',
            '{"type":"tool","name":"Calculator","arguments":{"expression":"2+3"}}',
            '{"type":"finish","answer":"5"}',
        ])
        observed = []

        def fake_llm(messages):
            if len(messages) > 2:
                observed.append(json.loads(messages[-1]["content"].removeprefix("Observation: ")))
            return next(outputs)

        self.assertEqual(run_agent(fake_llm, "2+3", executor), "5")
        self.assertFalse(observed[0]["ok"])
        self.assertEqual(observed[1]["result"], 5)

    def test_failure_and_step_limits(self):
        executor = ToolExecutor()
        calls = []

        def invalid(messages):
            calls.append(1)
            return '{}'

        self.assertIn("失败上限", run_agent(invalid, "test", executor))
        self.assertEqual(len(calls), 3)
        executor.registerTool("Echo", "无参数", lambda: "ok")
        output = run_agent(lambda _: '{"type":"tool","name":"Echo","arguments":{}}',
                           "test", executor, max_steps=2)
        self.assertIn("总步数上限", output)


if __name__ == "__main__":
    unittest.main()
