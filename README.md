# Agent Inspector

A diagnostic analyzer for Microsoft Fabric Data Agent. Upload a diagnostics JSON file and visually inspect agent configuration, conversation turns, generated queries, latency, and automated issue detection.

Diagnostics are analyzed locally unless you explicitly use the experimental
OpenRouter feature. AI analysis sends the previewed context to OpenRouter and
the selected model provider. Always review the context and avoid submitting
secrets, personal data, or regulated data.

https://data-agent-inspector.streamlit.app/

## Features

- **Conversation Viewer** -- Browse user/assistant exchanges with generated DAX, SQL, KQL, and GQL queries, execution outputs, and step-by-step trace details.
- **Query Review** -- Deterministically extracts referenced objects, filters, joins, aggregations, and generated measures from DAX, SQL, and KQL without claiming business correctness.
- **Configuration Inspector** -- View agent instructions, data source descriptions, instructions, few-shot examples, and connection details per source.
- **Schema Explorer** -- Sidebar tree view of all data source elements (tables, columns, measures, entities) with selection status.
- **Schema Inventory** -- Shows every exported table, column, and measure, including objects with missing descriptions and exports that omit child metadata.
- **Analysis Dashboard** -- Configuration metrics, schema quality summary, response time chart, step breakdown table, and automated issue detection with severity levels.
- **ERD View** -- Shows semantic model relationships.
- **Multi-Source Support** -- Semantic Models, Lakehouse Tables, KQL Databases, Ontology (Graph), SQL Databases, and more.
- **Azure AI Search Setup** -- Surfaces experimental Azure AI Search index, endpoint, search mode, top-k, and description configuration.
- **Latency Analysis** -- Identify slow responses with detailed time breakdowns per step.
- **Issue Detection** -- Automated checks for missing descriptions, failed runs, slow queries, unselected sources, and other common configuration problems.
- **Analyze with AI (Experimental)** -- Use your own OpenRouter key for grounded diagnostic Q&A, full reviews, and semantic-model AI-readiness assessments. Context is minimized and redacted by default, previewed before submission, and grounded in Microsoft Fabric guidance.

## Supported Data Source Types

| Type | Query Language | Description/Instructions | Few-Shot Examples |
|------|---------------|--------------------------|-------------------|
| Semantic Model | DAX | Schema descriptions only | N/A |
| Lakehouse Tables | SQL | Yes | Yes |
| KQL Database | KQL | Yes | Yes |
| Ontology (Graph) | GQL | N/A | N/A |

## Getting Started

### Run Locally

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Launch the app:
   ```
   streamlit run app.py
   ```

3. Open `http://localhost:8501` in your browser.

4. Upload a diagnostics JSON file using the sidebar.

5. Optional: open **Analyze with AI (Experimental)**, enter an OpenRouter API
   key and model, review the exact context, and confirm consent before sending.

### Deploy to Streamlit Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click "Create app", select this repository, and set the main file to `app.py`.
4. Click "Deploy".

## Diagnostics JSON

Export diagnostics from the Fabric Data Agent UI. The JSON contains agent configuration, data source schemas, conversation history, run steps, and tool call details. Both legacy schema-v2 files and current schema-v3 files are supported.

Sample files are included in the `sample_diagnostics/` folder for testing.

## Resources

- [Semantic Model Best Practices](https://learn.microsoft.com/en-us/fabric/data-science/semantic-model-best-practices)
- [Data Agent Configurations](https://learn.microsoft.com/en-us/fabric/data-science/data-agent-configurations)
- [Data Agent Configuration Best Practices](https://learn.microsoft.com/en-us/fabric/data-science/data-agent-configuration-best-practices)

## Requirements

- Python 3.9+
- streamlit >= 1.30
- pandas >= 2.0
- plotly >= 5.18
- requests >= 2.31

## License

MIT License. See [LICENSE](LICENSE) for details.
