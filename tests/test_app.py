import json
import unittest
from unittest.mock import patch

import app


class DiagnosticsCompatibilityTests(unittest.TestCase):
    def test_current_schema_is_valid_and_parsed(self):
        raw = {
            "diagnosticsSchemaVersion": 3,
            "artifactId": "agent-id",
            "workspaceId": "workspace-id",
            "rolloutEnvironment": "prod",
            "stage": "prod",
            "config": {
                "configuration": {
                    "additionalInstructions": "",
                    "dataSources": [{
                        "id": "model-id",
                        "artifactId": "model-id",
                        "workspaceId": "model-workspace-id",
                        "type": "semantic_model",
                        "isSelected": True,
                    }],
                },
            },
            "datasources": {
                "model-id": {
                    "schema": {
                        "dataSourceInfo": {
                            "id": "model-id",
                            "type": "semantic_model",
                            "display_name": "Sales model",
                            "workspace_id": "model-workspace-id",
                            "elements": [{
                                "id": "sales",
                                "type": "table",
                                "display_name": "Sales",
                                "is_selected": True,
                                "description": "",
                                "children": [],
                                "has_sub_elements": True,
                            }],
                        },
                    },
                },
            },
            "conversationItems": [
                {
                    "id": "user-1",
                    "role": "user",
                    "content": "Show sales",
                    "responseId": "response-1",
                },
                {
                    "id": "assistant-1",
                    "role": "assistant",
                    "content": "Sales are 42.",
                    "responseId": "response-1",
                },
            ],
            "responses": [{
                "id": "response-1",
                "status": "completed",
                "createdAt": 100,
                "completedAt": 112,
                "legacy": {"runId": "run-1"},
            }],
            "reasoningItemsByResponseId": {
                "response-1": [{
                    "id": "reasoning-1",
                    "kind": "tool-call",
                    "status": "completed",
                    "order": 1,
                    "title": "analyze.database.execute",
                    "input": {
                        "datasource_artifact_id": "model-id",
                        "datasource_name": "Sales model",
                        "datasource_type": "SemanticModel",
                        "natural_language_query": "Show sales",
                        "code": "```dax\nEVALUATE ROW(\"Sales\", 42)\n```",
                    },
                    "output": [{"Sales": 42}],
                }],
            },
            "latency": {
                "responses": [{
                    "response_id": "response-1",
                    "duration_seconds": 12,
                }],
                "reasoning_items": [{
                    "response_id": "response-1",
                    "reasoning_item_id": "reasoning-1",
                    "duration_seconds": 4,
                }],
            },
        }

        is_valid, error = app.validate_diagnostics(raw)
        parsed = app.parse_diagnostics(raw)

        self.assertTrue(is_valid, error)
        self.assertEqual(len(parsed["conversations"]), 1)
        self.assertEqual(parsed["conversations"][0]["response_time_s"], 12)
        self.assertEqual(parsed["conversations"][0]["steps"][0]["code_language"], "Dax")
        self.assertEqual(
            parsed["conversations"][0]["steps"][0]["generated_code"],
            'EVALUATE ROW("Sales", 42)',
        )
        self.assertEqual(
            parsed["data_sources"][0]["connection"]["semantic_model_name"],
            "Sales model",
        )
        self.assertEqual(
            parsed["conversations"][0]["steps"][0]["query_analysis"]["language"],
            "DAX",
        )

    def test_raw_json_scalar_serialization_is_valid_json(self):
        values = [
            "0333b170-84c4-493f-9a49-515071a0040a",
            3,
            True,
            None,
        ]

        for value in values:
            with self.subTest(value=value):
                self.assertEqual(json.loads(app._json_text(value)), value)

    def test_json_export_with_filename_header_is_supported(self):
        raw = app._parse_json_document(
            "diagnostics-example.json\n"
            '{"config": {}, "thread": {}}'
        )

        self.assertIn("config", raw)
        self.assertIn("thread", raw)

    def test_raw_json_tab_uses_code_renderer_for_scalar_sections(self):
        class StreamlitStub:
            def __init__(self):
                self.code_calls = []
                self.json_calls = []

            def markdown(self, *_args, **_kwargs):
                return None

            def caption(self, *_args, **_kwargs):
                return None

            def selectbox(self, *_args, **_kwargs):
                return "workspaceId"

            def code(self, *args, **kwargs):
                self.code_calls.append((args, kwargs))

            def json(self, *args, **kwargs):
                self.json_calls.append((args, kwargs))

        streamlit_stub = StreamlitStub()
        with patch.object(app, "st", streamlit_stub):
            app.render_raw_json_tab({
                "workspaceId": "0333b170-84c4-493f-9a49-515071a0040a",
            })

        self.assertEqual(len(streamlit_stub.code_calls), 1)
        self.assertEqual(len(streamlit_stub.json_calls), 0)
        rendered = streamlit_stub.code_calls[0][0][0]
        self.assertEqual(
            json.loads(rendered),
            "0333b170-84c4-493f-9a49-515071a0040a",
        )

    def test_schema_inventory_includes_missing_descriptions(self):
        elements = [{
            "type": "semantic_model.table",
            "display_name": "Sales",
            "is_selected": True,
            "description": None,
            "children": [
                {
                    "type": "semantic_model.column",
                    "display_name": "Amount",
                    "is_selected": True,
                    "description": "",
                    "children": [],
                },
                {
                    "type": "semantic_model.measure",
                    "display_name": "Total Sales",
                    "is_selected": True,
                    "description": "Sum of sales.",
                    "children": [],
                },
            ],
        }]

        rows = app._collect_schema_inventory(elements)

        self.assertEqual([row["type"] for row in rows], ["Table", "Column", "Measure"])
        self.assertEqual(
            [row["description_status"] for row in rows],
            ["Missing", "Missing", "Configured"],
        )
        self.assertEqual(app._count_descriptions(elements), {
            "total": 3,
            "with_desc": 1,
        })

    def test_grouping_nodes_are_not_counted_as_tables(self):
        elements = [{
            "type": "table_grouping",
            "display_name": "Tables",
            "children": [{
                "type": "table",
                "display_name": "Orders",
                "is_selected": True,
                "children": [],
            }],
        }]

        self.assertEqual(app._count_schema(elements), (1, 1, 0, 0))

    def test_kusto_execution_is_classified_as_kql(self):
        self.assertEqual(
            app._detect_language("analyze.database.execute", "Kusto"),
            ("KQL", "Kusto"),
        )

    def test_review_steps_hide_trace_operations(self):
        primary, traces = app._split_review_steps([
            {"source_type": "SemanticModel"},
            {"source_type": "Trace"},
        ])

        self.assertEqual(len(primary), 1)
        self.assertEqual(len(traces), 1)

    def test_failed_turn_uses_failure_indicator(self):
        self.assertEqual(app._turn_status_indicator("failed"), (" failed", "!"))
        self.assertEqual(app._turn_status_indicator("completed"), ("", "✓"))

    def test_gantt_uses_recorded_starts_for_parallel_steps(self):
        gantt = app._build_gantt_steps([
            {
                "step_id": "one",
                "func_name": "first",
                "created_at": 101,
                "duration_s": 5,
                "latency_duration_s": 5,
                "order": 1,
            },
            {
                "step_id": "two",
                "func_name": "second",
                "created_at": 101,
                "duration_s": 5,
                "latency_duration_s": 5,
                "order": 2,
            },
        ], 100)

        self.assertEqual(gantt[0]["start"], 101)
        self.assertEqual(gantt[1]["start"], 101)

    def test_ai_search_experimental_config_becomes_setup_source(self):
        raw = {
            "config": {
                "configuration": {
                    "dataSources": [],
                    "additionalInstructions": "",
                    "experimental": {
                        "azureAISearchConfigs": [{
                            "azureAiSearchIndexName": "support-transcripts",
                            "azureAiSearchEndpoint": "https://example.search.windows.net",
                            "azureAiSearchUserDescription": "",
                            "azureAiSearchSearchType": "semantic",
                            "azureAiSearchTopk": 10,
                        }],
                    },
                },
            },
            "datasources": {},
            "thread": {
                "messages": [],
                "runs": [],
                "run_steps": [],
            },
            "latency": {},
        }

        parsed = app.parse_diagnostics(raw)
        search_source = parsed["data_sources"][0]

        self.assertEqual(search_source["type"], "azure_ai_search")
        self.assertEqual(search_source["name"], "support-transcripts")
        self.assertEqual(search_source["connection"]["top_k"], 10)

    def test_welcome_page_explains_workflow_and_core_features(self):
        with patch("app.st.markdown") as markdown:
            app._render_welcome()

        content = markdown.call_args.args[0]
        self.assertIn("How it works", content)
        self.assertIn("Analyze Fabric Data Agent", content)
        self.assertIn("Conversation and queries", content)
        self.assertIn("Setup and schema", content)
        self.assertIn("Failures and performance", content)
        self.assertIn("AI-assisted review", content)
        self.assertTrue(markdown.call_args.kwargs["unsafe_allow_html"])


if __name__ == "__main__":
    unittest.main()
