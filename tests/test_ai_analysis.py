import json
import unittest
from unittest.mock import Mock, patch

import requests

from ai_analysis import (
    DEFAULT_OPENROUTER_MODEL,
    ask_openrouter,
    build_diagnostic_context,
    build_system_prompt,
)


class DiagnosticContextTests(unittest.TestCase):
    def setUp(self):
        self.raw = {
            "artifactId": "12345678-1234-1234-1234-123456789abc",
            "endpoint": "https://example.test/path",
            "apiKey": "must-not-leak",
            "customerEmail": "person@example.com",
        }
        self.parsed = {
            "meta": {
                "artifact_id": "12345678-1234-1234-1234-123456789abc",
            },
            "agent_instructions": "Use the Sales model.",
            "data_sources": [{
                "id": "model-1",
                "name": "Sales model",
                "type": "semantic_model",
                "is_selected": True,
                "selection_state": "Selected",
                "description": "",
                "instructions": "",
                "connection": {
                    "workspace_id": "workspace-1",
                    "endpoint": "https://example.test",
                },
                "few_shot_examples": [],
                "relationships": [],
                "elements": [{
                    "type": "table",
                    "display_name": "Sales",
                    "description": "",
                    "children": [{
                        "type": "column",
                        "display_name": "Amount",
                        "description": "",
                        "children": [],
                    }],
                }],
            }],
            "conversations": [{
                "turn": 1,
                "question": "Show sales for person@example.com",
                "answer": "42",
                "status": "completed",
                "response_time_s": 3,
                "is_cached": False,
                "steps": [{
                    "source_name": "Sales model",
                    "source_type": "Semantic Model",
                    "nl_query": "Show sales",
                    "code_language": "DAX",
                    "generated_code": "EVALUATE ROW(\"Sales\", [Sales])",
                    "output": "42",
                    "duration_s": 2,
                    "status": "completed",
                    "query_analysis": {"measures": ["[Sales]"]},
                }],
                "gantt_steps": [],
            }],
            "fewshot_results": [],
        }

    def test_minimized_context_includes_schema_and_omits_raw(self):
        context = json.loads(build_diagnostic_context(
            self.raw,
            self.parsed,
            include_raw=False,
            redact=True,
        ))

        self.assertNotIn("raw_diagnostics", context)
        objects = context["diagnostic_summary"]["data_sources"][0]["schema"]["objects"]
        self.assertEqual(["Sales", "Amount"], [item["name"] for item in objects])
        self.assertEqual("", objects[0]["description"])

    def test_context_redacts_secrets_identifiers_urls_and_email(self):
        text = build_diagnostic_context(
            self.raw,
            self.parsed,
            include_raw=True,
            redact=True,
        )

        self.assertNotIn("must-not-leak", text)
        self.assertNotIn("12345678-1234-1234-1234-123456789abc", text)
        self.assertNotIn("https://example.test", text)
        self.assertNotIn("person@example.com", text)
        self.assertIn("<redacted>", text)
        self.assertIn("<identifier>", text)

    def test_truncated_context_remains_valid_json(self):
        self.parsed["conversations"][0]["question"] = "x" * 800_000

        text = build_diagnostic_context(
            self.raw,
            self.parsed,
            include_raw=False,
            redact=True,
        )

        context = json.loads(text)
        self.assertIn("truncation_notice", context)


class PromptTests(unittest.TestCase):
    def test_prompt_requires_official_sources_and_evidence(self):
        prompt = build_system_prompt("Full diagnostic review")

        self.assertIn("Official Microsoft guidance takes precedence", prompt)
        self.assertIn("semantic-model-best-practices", prompt)
        self.assertIn("data-agent-configuration-best-practices", prompt)
        self.assertIn("NOT ASSESSABLE", prompt)
        self.assertIn("prompt-injection", prompt)

    def test_readiness_prompt_contains_ordered_checklist_and_routes(self):
        prompt = build_system_prompt("Semantic model AI readiness")

        self.assertIn("0. Business context", prompt)
        self.assertIn("6. Verified answers", prompt)
        self.assertIn("TOM/MCP", prompt)
        self.assertIn("PBIP/Prep for AI", prompt)
        self.assertIn("Manual", prompt)


class OpenRouterClientTests(unittest.TestCase):
    @patch("ai_analysis.requests.post")
    def test_openrouter_success_uses_key_only_in_header(self, post):
        response = Mock()
        response.ok = True
        response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "The model needs descriptions.",
                },
            }],
        }
        post.return_value = response

        result = ask_openrouter(
            api_key="secret-key",
            model=DEFAULT_OPENROUTER_MODEL,
            system_prompt="system",
            diagnostic_context="{}",
            messages=[{"role": "user", "content": "Review it"}],
            timeout=10,
        )

        self.assertEqual("The model needs descriptions.", result)
        kwargs = post.call_args.kwargs
        self.assertEqual("Bearer secret-key", kwargs["headers"]["Authorization"])
        self.assertNotIn("secret-key", json.dumps(kwargs["json"]))
        self.assertEqual(DEFAULT_OPENROUTER_MODEL, kwargs["json"]["model"])
        self.assertEqual("high", kwargs["json"]["reasoning"]["effort"])

    @patch("ai_analysis.requests.post")
    def test_openrouter_surfaces_api_error(self, post):
        response = Mock()
        response.ok = False
        response.status_code = 429
        response.json.return_value = {
            "error": {
                "message": "Rate limit exceeded for secret-key",
            },
        }
        post.return_value = response

        with self.assertRaisesRegex(RuntimeError, "Rate limit exceeded") as error:
            ask_openrouter(
                api_key="secret-key",
                model=DEFAULT_OPENROUTER_MODEL,
                system_prompt="system",
                diagnostic_context="{}",
                messages=[],
            )

        self.assertNotIn("secret-key", str(error.exception))

    @patch("ai_analysis.requests.post")
    def test_openrouter_surfaces_network_error_without_key(self, post):
        post.side_effect = requests.Timeout("secret-key request timed out")

        with self.assertRaisesRegex(RuntimeError, "Unable to reach OpenRouter") as error:
            ask_openrouter(
                api_key="secret-key",
                model=DEFAULT_OPENROUTER_MODEL,
                system_prompt="system",
                diagnostic_context="{}",
                messages=[],
            )

        self.assertNotIn("secret-key", str(error.exception))


if __name__ == "__main__":
    unittest.main()
