# Agent Inspector

A diagnostic analyzer for Microsoft Fabric Data Agent. Upload a diagnostics JSON file and visually inspect agent configuration, conversation turns, generated queries, latency, and automated issue detection.

## Features

- **Conversation Viewer** -- Browse user/assistant exchanges with generated DAX, SQL, KQL, and GQL queries, execution outputs, and step-by-step trace details.
- **Configuration Inspector** -- View agent instructions, data source descriptions, instructions, few-shot examples, and connection details per source.
- **Schema Explorer** -- Sidebar tree view of all data source elements (tables, columns, measures, entities) with selection status.
- **Analysis Dashboard** -- Configuration metrics, schema quality summary, response time chart, step breakdown table, and automated issue detection with severity levels.
- **Multi-Source Support** -- Semantic Models, Lakehouse Tables, KQL Databases, Ontology (Graph), SQL Databases, and more.
- **Latency Analysis** -- Identify slow responses with detailed time breakdowns per step.
- **Issue Detection** -- Automated checks for missing descriptions, failed runs, slow queries, unselected sources, and other common configuration problems.

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

### Deploy to Streamlit Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click "Create app", select this repository, and set the main file to `app.py`.
4. Click "Deploy".

## Diagnostics JSON

Export diagnostics from the Fabric Data Agent UI. The JSON contains agent configuration, data source schemas, conversation history, run steps, and tool call details.

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

## License

MIT License. See [LICENSE](LICENSE) for details.
