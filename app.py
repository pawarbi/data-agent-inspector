"""
Agent Inspector — Fabric Data Agent Diagnostics Analyzer — Streamlit UI
=====================================================
Mirrors the Fabric Data Agent UI for diagnostic debugging.
Upload a diagnostics JSON and visually inspect configuration,
conversation turns, generated queries, and step details.
"""

import streamlit as st
import json
import html as html_lib
import pandas as pd
from datetime import datetime, timezone

# ──────────────────────────────────────────────────────────
# Configurable Thresholds
# ──────────────────────────────────────────────────────────

THRESHOLDS = {
    "slow_turn_s": 20,         # Flag responses slower than this
    "slow_step_s": 10,         # Flag individual steps slower than this
    "very_slow_turn_s": 30,    # Issue detection threshold
    "desc_char_limit": 200,    # DA truncates descriptions beyond this
    "instr_warn_chars": 8000,  # Warn if agent instructions exceed this
    "max_file_mb": 50,         # Warn before parsing files larger than this
    "page_size": 25,           # Turns per page in conversation view
}

# ──────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Agent Inspector",
    page_icon="magnifying_glass",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
# Custom CSS — matches Fabric Data Agent UI styling
# ──────────────────────────────────────────────────────────

st.markdown("""
<style>
/* User question bubble — teal, right-aligned */
.user-bubble {
    display: flex;
    justify-content: flex-end;
    margin: 16px 0 8px 0;
}
.user-bubble-inner {
    background: #e6f7f0;
    color: #1a1a1a;
    padding: 12px 16px;
    border-radius: 8px;
    max-width: 80%;
    font-size: 14px;
    line-height: 1.6;
}

/* Step header — green left border for completed, red for failed */
.step-header {
    padding: 10px 14px;
    background: #f8f9fa;
    border-radius: 6px;
    margin-bottom: 8px;
    border-left: 3px solid #2ecc71;
    font-size: 14px;
    line-height: 1.5;
}
.step-header.failed {
    border-left-color: #e74c3c;
}

/* Response time — right-aligned gray */
.response-time {
    text-align: right;
    color: #666;
    font-size: 13px;
    padding-top: 4px;
}

/* Language badge on code blocks */
.lang-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: bold;
    color: white;
    margin-bottom: 8px;
}
.lang-dax  { background: #2ecc71; }
.lang-sql  { background: #3498db; }
.lang-kql  { background: #9b59b6; }
.lang-gql  { background: #e67e22; }
.lang-trace { background: #7f8c8d; }
.lang-code { background: #95a5a6; }

/* Metadata bar */
.meta-bar {
    display: flex;
    gap: 24px;
    padding: 10px 16px;
    background: #f8f9fa;
    border-radius: 6px;
    margin-bottom: 16px;
    font-size: 13px;
    color: #555;
    flex-wrap: wrap;
    border: 1px solid #e8e8e8;
}
.meta-bar span { white-space: nowrap; }

/* Top banner */
.top-banner {
    background: linear-gradient(135deg, #1a2744 0%, #2c3e6b 100%);
    color: #ffffff;
    padding: 18px 32px;
    border-radius: 8px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.top-banner .banner-title {
    font-size: 1.6em;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.top-banner .banner-subtitle {
    font-size: 0.85em;
    color: #b0bec5;
    margin-top: 2px;
}
.top-banner .banner-btn {
    background: rgba(255,255,255,0.12);
    color: #ffffff;
    border: 1px solid rgba(255,255,255,0.25);
    padding: 6px 18px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    text-decoration: none;
    transition: background 0.2s;
}
.top-banner .banner-btn:hover {
    background: rgba(255,255,255,0.22);
}
/* Resources dropdown */
.resources-wrap {
    position: relative;
    display: inline-block;
}
.resources-menu {
    display: none;
    position: absolute;
    right: 0;
    top: 110%;
    background: #ffffff;
    border-radius: 6px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.18);
    min-width: 320px;
    z-index: 1000;
    padding: 8px 0;
}
.resources-wrap:hover .resources-menu,
.resources-wrap:focus-within .resources-menu {
    display: block;
}
.resources-menu a {
    display: block;
    padding: 10px 18px;
    color: #1a2744;
    text-decoration: none;
    font-size: 13px;
    font-weight: 500;
    border-bottom: 1px solid #f0f0f0;
}
.resources-menu a:last-child { border-bottom: none; }
.resources-menu a:hover { background: #f4f6fa; }
.resources-menu .res-label {
    font-size: 11px;
    color: #888;
    margin-top: 2px;
}

/* Welcome screen */
.welcome {
    text-align: center;
    padding: 80px 20px;
    color: #555;
}
.welcome h1 { font-size: 2em; margin-bottom: 8px; }

/* Schema child indent */
.schema-child {
    padding-left: 16px;
    font-size: 13px;
    line-height: 1.8;
}

/* Tighter sidebar spacing */
section[data-testid="stSidebar"] .stExpander {
    margin-bottom: -8px;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def _s(val):
    """Coerce None to empty string."""
    return val if val is not None else ""


def _extract_text(content_blocks):
    """Extract display text from message content blocks."""
    parts = []
    for block in (content_blocks or []):
        if isinstance(block, dict):
            txt = block.get("text", {})
            if isinstance(txt, dict):
                parts.append(txt.get("value", ""))
            elif isinstance(txt, str):
                parts.append(txt)
    return "\n".join(parts).strip()


def _safe_json(s):
    """Parse a JSON string safely, handling double-encoded strings."""
    if not s:
        return {}
    try:
        result = json.loads(s)
        # Handle double-encoded JSON (string inside a string)
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                return {}
        if isinstance(result, dict):
            return result
        return {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _ts(epoch):
    """Convert epoch to datetime."""
    if not epoch:
        return None
    try:
        return datetime.fromtimestamp(epoch, tz=timezone.utc)
    except (OSError, ValueError, TypeError):
        return None


# ──────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────

def validate_diagnostics(raw):
    """Validate diagnostics JSON structure. Returns (is_valid, error_message)."""
    if not isinstance(raw, dict):
        return False, "File content is not a JSON object."

    missing = []
    if "config" not in raw:
        missing.append("config")
    if "thread" not in raw:
        missing.append("thread")

    if missing:
        return False, (
            f"Missing required sections: {', '.join(missing)}. "
            "This may not be a Fabric Data Agent diagnostics file."
        )
    return True, ""


# ──────────────────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────────────────

def parse_diagnostics(raw):
    """Parse diagnostics JSON into structured data for the UI."""
    config = raw.get("config", {}).get("configuration", {})
    datasources_raw = raw.get("datasources", {})
    thread = raw.get("thread", {})

    # Metadata
    meta = {
        "artifact_id": raw.get("artifactId", "N/A"),
        "workspace_id": raw.get("workspaceId", "N/A"),
        "environment": raw.get("rolloutEnvironment", "N/A"),
        "stage": raw.get("stage", "N/A"),
        "downloaded_at": raw.get("downloaded_at", ""),
    }

    # Agent instructions
    agent_instructions = config.get("additionalInstructions", "")

    # Data sources
    data_sources = _parse_data_sources(config, datasources_raw)

    # Few-shot loading results from run_steps
    fewshot_results = _parse_fewshot_loading(thread.get("run_steps", []))

    # Conversations
    conversations = _parse_conversations(thread)

    return {
        "meta": meta,
        "agent_instructions": agent_instructions,
        "data_sources": data_sources,
        "conversations": conversations,
        "fewshot_results": fewshot_results,
    }


def _parse_data_sources(config, datasources_raw):
    """Parse data source configs and merge with schema info."""
    ds_configs = config.get("dataSources", [])
    data_sources = []

    for ds_conf in ds_configs:
        ds_id = ds_conf.get("id", ds_conf.get("artifactId", ""))
        ds_schema = (
            datasources_raw
            .get(ds_id, {})
            .get("schema", {})
            .get("dataSourceInfo", {})
        )

        ds_type = ds_conf.get("type", ds_schema.get("type", "unknown"))

        # Connection details vary by source type
        connection = {}
        if ds_type == "kusto":
            connection = {
                "endpoint": ds_conf.get("endpoint", ""),
                "database": ds_conf.get("database_name", ""),
                "kusto_id": ds_conf.get("kusto_id", ""),
            }
        elif ds_type == "lakehouse_tables":
            connection = {
                "sql_endpoint": ds_conf.get("sql_endpoint", ""),
                "lakehouse_name": ds_conf.get("lakehouse_name", ""),
                "lakehouse_id": ds_conf.get("lakehouse_id", ""),
            }
        elif ds_type == "semantic_model":
            connection = {
                "semantic_model_name": ds_conf.get("semantic_model_name", ""),
                "semantic_model_id": ds_conf.get("semantic_model_id", ""),
            }
        elif ds_type == "ontology":
            connection = {
                "ontology_id": ds_conf.get("artifactId", ""),
            }

        # Few-shot examples from config
        few_shot_examples = ds_conf.get("few_shot_examples") or []

        data_sources.append({
            "id": ds_id,
            "name": (
                ds_schema.get("display_name")
                or ds_conf.get("displayName")
                or ds_id[:12]
            ),
            "type": ds_type,
            "is_selected": ds_conf.get("isSelected", True),
            "description": ds_schema.get("user_description") or "",
            "instructions": ds_schema.get("additional_instructions") or "",
            "elements": ds_schema.get("elements", []),
            "connection": connection,
            "few_shot_examples": few_shot_examples,
        })

    return data_sources


def _parse_conversations(thread):
    """Parse thread into conversation turns with associated steps."""
    messages = thread.get("messages", [])
    runs = thread.get("runs", [])
    run_steps = thread.get("run_steps", [])

    sorted_msgs = sorted(messages, key=lambda m: m.get("created_at", 0))
    user_msgs = [m for m in sorted_msgs if m.get("role") == "user"]
    asst_msgs = [m for m in sorted_msgs if m.get("role") == "assistant"]

    # Index runs by ID
    runs_map = {}
    for r in runs:
        rid = r.get("id", "")
        created = r.get("created_at") or 0
        completed = r.get("completed_at") or 0
        runs_map[rid] = {
            "status": r.get("status", ""),
            "model": r.get("model", ""),
            "total_time_s": completed - created if completed and created else 0,
        }

    # Index tool calls by run_id
    steps_by_run = _parse_run_steps(run_steps)

    # Pair user questions with assistant answers
    conversations = []
    for i, umsg in enumerate(user_msgs):
        question = _extract_text(umsg.get("content", []))
        answer = ""
        run_id = ""

        if i < len(asst_msgs):
            answer = _extract_text(asst_msgs[i].get("content", []))
            run_id = asst_msgs[i].get("run_id") or ""

        run_info = runs_map.get(run_id, {})
        raw_steps = steps_by_run.get(run_id, [])
        analyze_steps = _group_analyze_steps(raw_steps)

        # Detect cached responses:
        #  - Answer exists but no run was created (run_id is empty/null)
        #  - Or run exists but no tool call steps and completed very fast (<3s)
        is_cached = False
        if answer and not run_id:
            is_cached = True
        elif answer and run_id and run_id not in runs_map:
            is_cached = True
        elif (answer and run_id and not analyze_steps
              and run_info.get("total_time_s", 0) < 3):
            is_cached = True

        conversations.append({
            "turn": i + 1,
            "question": question,
            "answer": answer,
            "run_id": run_id,
            "response_time_s": run_info.get("total_time_s", 0),
            "status": run_info.get("status", "unknown"),
            "steps": analyze_steps,
            "is_cached": is_cached,
        })

    return conversations


def _parse_run_steps(run_steps):
    """Parse run_steps into tool call dicts indexed by run_id."""
    steps_by_run = {}
    for s in run_steps:
        run_id = s.get("run_id", "")
        if s.get("type") != "tool_calls":
            continue

        created = s.get("created_at", 0) or 0
        completed = s.get("completed_at", 0) or 0
        duration = completed - created if completed and created else 0

        for tc in s.get("step_details", {}).get("tool_calls", []):
            func = tc.get("function", {})
            args = _safe_json(func.get("arguments"))

            steps_by_run.setdefault(run_id, []).append({
                "func_name": func.get("name", "unknown"),
                "datasource_name": args.get("datasource_name", ""),
                "datasource_type": args.get("datasource_type", ""),
                "nl_query": args.get("natural_language_query", ""),
                "code": args.get("code", ""),
                "output": func.get("output", "") or "",
                "duration_s": duration,
                "status": s.get("status", ""),
            })

    return steps_by_run


def _group_analyze_steps(raw_steps):
    """Group nl2code + execute pairs into analyze operations, include trace steps."""
    nl2code = [s for s in raw_steps if "nl2code" in s["func_name"]]
    execute = [s for s in raw_steps if "execute" in s["func_name"]]
    trace = [s for s in raw_steps if "trace." in s["func_name"]]

    ops = []
    used_exec = set()

    for nl in nl2code:
        ds = nl["datasource_name"]
        lang, ds_label = _detect_language(nl["func_name"], nl["datasource_type"])

        # Find matching execute step for same datasource
        matched = None
        for j, ex in enumerate(execute):
            if j not in used_exec and ex["datasource_name"] == ds:
                matched = ex
                used_exec.add(j)
                break

        # Generated code: prefer execute args, then nl2code output, then nl2code args
        generated_code = ""
        if matched and matched["code"]:
            generated_code = matched["code"]
        elif nl["output"]:
            generated_code = nl["output"]
        elif nl["code"]:
            generated_code = nl["code"]

        query_output = matched["output"] if matched else ""
        total_duration = nl["duration_s"] + (matched["duration_s"] if matched else 0)

        status = "completed"
        if matched and matched["status"] != "completed":
            status = matched["status"]
        elif nl["status"] != "completed":
            status = nl["status"]

        ops.append({
            "source_name": ds,
            "source_type": ds_label,
            "nl_query": nl["nl_query"],
            "generated_code": generated_code,
            "code_language": lang,
            "output": query_output,
            "duration_s": total_duration,
            "status": status,
        })

    # HV-6: Orphan execute steps — label clearly
    for j, ex in enumerate(execute):
        if j not in used_exec:
            lang, ds_label = _detect_language(ex["func_name"], ex["datasource_type"])
            ops.append({
                "source_name": ex["datasource_name"],
                "source_type": ds_label,
                "nl_query": "(direct execution, no NL query)",
                "generated_code": ex["code"],
                "code_language": lang,
                "output": ex["output"],
                "duration_s": ex["duration_s"],
                "status": ex["status"],
            })

    # HV-2: Trace steps (debug info from trace.analyze_ontology, trace.analyze_lakehouse_tables, etc.)
    for t in trace:
        args = {"query": t.get("nl_query", "")}
        ops.append({
            "source_name": t["datasource_name"] or t["func_name"].split(".")[-1],
            "source_type": "Trace",
            "nl_query": t.get("nl_query", ""),
            "generated_code": "",
            "code_language": "Trace",
            "output": t["output"],
            "duration_s": t["duration_s"],
            "status": t["status"],
        })

    return ops


def _detect_language(func_name, datasource_type=""):
    """Determine query language and data source label from function name and type."""
    fn = func_name.lower()
    ds_type = datasource_type.lower() if datasource_type else ""

    # Ontology uses database.* functions but generates GQL
    if ds_type == "ontology":
        return "GQL", "Ontology"

    if "semanticmodel" in fn:
        return "Dax", "SemanticModel"
    elif "database" in fn:
        return "SQL", "LakehouseTable"
    elif "kusto" in fn or "kql" in fn:
        return "KQL", "Kusto"
    return "Code", ""


def _parse_fewshot_loading(run_steps):
    """Extract few-shot loading results from run_steps."""
    results = []
    for s in run_steps:
        if s.get("type") != "tool_calls":
            continue
        for tc in s.get("step_details", {}).get("tool_calls", []):
            func = tc.get("function", {})
            name = func.get("name", "")
            if "fewshot" not in name.lower():
                continue
            args = _safe_json(func.get("arguments"))
            output = func.get("output", "") or ""
            results.append({
                "datasource_name": args.get("datasource_name", ""),
                "datasource_type": args.get("datasource_type", ""),
                "output": output,
            })
    return results


def _count_schema(elements):
    """Recursively count tables/entities, selected, columns, measures."""
    tables = selected = columns = measures = 0
    for el in elements:
        el_type = el.get("type", "")
        if "table" in el_type or "entity" in el_type:
            tables += 1
            if el.get("is_selected"):
                selected += 1
            for child in el.get("children", []):
                ct = child.get("type", "")
                if "measure" in ct:
                    measures += 1
                elif "column" in ct:
                    columns += 1
        elif el.get("children"):
            t, s, c, m = _count_schema(el["children"])
            tables += t; selected += s; columns += c; measures += m
    return tables, selected, columns, measures


# ──────────────────────────────────────────────────────────
# Sidebar — Explorer
# ──────────────────────────────────────────────────────────


def render_sidebar(parsed):
    """Render the Explorer sidebar: agent instructions at top, then Data/Setup tabs."""
    st.markdown("### Explorer")

    # Agent instructions always visible at top (global config)
    instr = parsed["agent_instructions"]
    instr_label = f"Agent instructions ({len(instr):,} chars)" if instr else "Agent instructions"
    with st.expander(instr_label, expanded=False):
        if instr:
            st.markdown(instr[:5000])
        else:
            st.caption("No agent instructions configured")

    tab_data, tab_setup = st.tabs(["Data", "Setup"])

    with tab_data:
        _render_data_tab(parsed["data_sources"])

    with tab_setup:
        _render_setup_tab(parsed)


def _render_data_tab(data_sources):
    """Data tab: schema browser with selectable tables."""
    if not data_sources:
        st.caption("No data sources found")
        return

    for ds in data_sources:
        selected_mark = "✅" if ds["is_selected"] else "❌"
        st.markdown(f"**{ds['name']}** {selected_mark}")
        st.caption(f"Type: {ds['type']}")

        # Schema summary
        if ds["elements"]:
            t, s, c, m = _count_schema(ds["elements"])
            item_label = "entities" if ds["type"] == "ontology" else "tables"
            parts = []
            if t: parts.append(f"{s}/{t} {item_label} selected")
            if c: parts.append(f"{c} cols")
            if m: parts.append(f"{m} measures")
            st.caption(" · ".join(parts))

        _render_schema_tree(ds["elements"])
        st.divider()


def _render_schema_tree(elements):
    """Render schema elements with expandable tables."""
    if not elements:
        st.caption("No schema data available")
        return

    for el in elements:
        el_type = el.get("type", "")
        name = el.get("display_name", "unknown")
        selected = el.get("is_selected", False)
        children = el.get("children", [])
        check = "✅" if selected else "❌"

        if "table" in el_type or "entity" in el_type:
            if children:
                with st.expander(f"{check} {name}"):
                    for child in children:
                        _render_schema_child(child)
            else:
                st.markdown(f"{check} {name}")
        elif children:
            # Intermediate node (schema namespace like "dbo")
            st.markdown(f"**{name}**")
            _render_schema_tree(children)
        else:
            st.markdown(f"{check} {name}")


def _render_schema_child(child):
    """Render a column or measure inside a table expander."""
    name = child.get("display_name", "?")
    child_type = child.get("type", "")
    selected = child.get("is_selected", False)
    data_type = child.get("data_type", "")

    check = "✅" if selected else "❌"
    type_tag = "[M]" if "measure" in child_type else ""

    label = f"{check} {html_lib.escape(name)}"
    if type_tag:
        label += f" <code style='font-size:11px;color:#7c3aed;'>{type_tag}</code>"
    if data_type:
        label += f" <code style='font-size:11px;color:#888;'>{data_type}</code>"

    st.markdown(
        f'<div class="schema-child">{label}</div>',
        unsafe_allow_html=True,
    )


def _render_setup_tab(parsed):
    """Setup tab: per-source connection, description, instructions, few-shots, schema descriptions."""
    fewshot_results = parsed.get("fewshot_results", [])

    for ds in parsed["data_sources"]:
        st.markdown(f"**{ds['name']}** ({ds['type']})")

        # Connection details
        conn = ds.get("connection", {})
        if conn and any(v for v in conn.values()):
            with st.expander("Connection details"):
                for k, v in conn.items():
                    if v:
                        st.markdown(f"**{k}**: `{v}`")

        # Data source description (all types)
        is_ontology = ds["type"] == "ontology"
        if ds["description"]:
            desc_len = len(ds["description"])
            with st.expander(f"Data source description ({desc_len:,} chars)"):
                st.markdown(ds["description"])
        elif is_ontology:
            st.caption("Data source description: N/A (not supported for ontology)")

        # Data source instructions (all types)
        instr_text = ds["instructions"]
        if instr_text:
            instr_len_label = f" ({len(instr_text):,} chars)"
            with st.expander(f"Data source instructions{instr_len_label}"):
                st.markdown(instr_text)
        elif is_ontology:
            st.caption("Data source instructions: N/A (not supported for ontology)")
        else:
            with st.expander("Data source instructions"):
                st.caption("No data source instructions configured")

        # Schema descriptions (all types that have elements)
        if ds.get("elements"):
            _render_schema_descriptions(ds)

        # Few-shot examples from config
        fs_config = ds.get("few_shot_examples", [])
        if fs_config:
            with st.expander(f"Few-shot examples ({len(fs_config)})"):
                for i, ex in enumerate(fs_config):
                    st.markdown(f"**Example {i+1}**")
                    if isinstance(ex, dict):
                        st.json(ex)
                    else:
                        st.code(str(ex))

        # Few-shot loading results from run_steps
        if is_ontology:
            st.caption("Few-shot examples: N/A (not supported for ontology)")
        else:
            ds_fewshots = [f for f in fewshot_results if f["datasource_name"] == ds["name"]]
            if ds_fewshots:
                outputs = [f["output"] for f in ds_fewshots]
                unique_outputs = list(set(outputs))
                label = unique_outputs[0] if len(unique_outputs) == 1 else "; ".join(unique_outputs)
                st.caption(f"Few-shot status: {label}")
            else:
                st.caption("Few-shot status: Not loaded")

        st.divider()


def _render_schema_descriptions(ds):
    """Show table/column/measure descriptions for a semantic model data source."""
    elements = ds.get("elements", [])
    if not elements:
        st.caption("No schema data available")
        return

    desc_rows = _collect_descriptions(elements)

    if not desc_rows:
        st.caption("No descriptions configured on any table, column, or measure")
        return

    # Summary
    total = len(desc_rows)
    tables_with = sum(1 for r in desc_rows if r["type"] == "Table")
    cols_with = sum(1 for r in desc_rows if r["type"] == "Column")
    measures_with = sum(1 for r in desc_rows if r["type"] == "Measure")
    over_200 = sum(1 for r in desc_rows if r["chars"] > THRESHOLDS["desc_char_limit"])
    warning = f" — {over_200} over {THRESHOLDS['desc_char_limit']} chars (DA truncates)" if over_200 else ""
    st.caption(f"{total} descriptions: {tables_with} tables, {cols_with} columns, {measures_with} measures{warning}")

    with st.expander(f"Schema descriptions ({total})", expanded=False):
        df = pd.DataFrame(desc_rows)
        # Flag descriptions over 200 chars
        df["flag"] = df["chars"].apply(lambda x: "!" if x > THRESHOLDS["desc_char_limit"] else "")
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "table": st.column_config.TextColumn("Table", width="small"),
                "name": st.column_config.TextColumn("Name", width="small"),
                "type": st.column_config.TextColumn("Type", width="small"),
                "description": st.column_config.TextColumn("Description", width="large"),
                "chars": st.column_config.NumberColumn("Chars", width="small"),
                "flag": st.column_config.TextColumn("", width="small"),
            },
        )


def _collect_descriptions(elements, parent_table=""):
    """Recursively collect all elements that have descriptions."""
    rows = []
    for el in elements:
        el_type = el.get("type", "")
        name = el.get("display_name", "?")
        desc = el.get("description", "")
        children = el.get("children", [])

        if "table" in el_type or "entity" in el_type:
            type_label = "Entity" if "entity" in el_type else "Table"
            if desc:
                rows.append({
                    "table": name,
                    "name": name,
                    "type": type_label,
                    "description": desc,
                    "chars": len(desc),
                })
            # Recurse into children (columns/measures)
            for child in children:
                child_name = child.get("display_name", "?")
                child_type_raw = child.get("type", "")
                child_desc = child.get("description", "")
                if child_desc:
                    rows.append({
                        "table": name,
                        "name": child_name,
                        "type": "Measure" if "measure" in child_type_raw else "Column",
                        "description": child_desc,
                        "chars": len(child_desc),
                    })
        elif children:
            # Intermediate node (namespace like "dbo") — recurse
            rows.extend(_collect_descriptions(children, parent_table))

    return rows


# ──────────────────────────────────────────────────────────
# Main Content — Conversation View
# ──────────────────────────────────────────────────────────

def render_main(parsed):
    """Render the main conversation area matching Fabric UI."""
    meta = parsed["meta"]

    # Metadata bar
    st.markdown(f"""
    <div class="meta-bar">
        <span><strong>Artifact:</strong> {html_lib.escape(str(meta['artifact_id']))[:20]}...</span>
        <span><strong>Environment:</strong> {html_lib.escape(str(meta['environment']))}</span>
        <span><strong>Stage:</strong> {html_lib.escape(str(meta['stage']))}</span>
        <span><strong>Data Sources:</strong> {len(parsed['data_sources'])}</span>
        <span><strong>Turns:</strong> {len(parsed['conversations'])}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Test the agent's responses")

    if not parsed["conversations"]:
        st.info("No conversation data found in this diagnostics file.")
        return

    conversations = parsed["conversations"]

    # HV-1: Conversation filters
    filter_cols = st.columns([1, 1, 1, 1, 2])
    with filter_cols[0]:
        show_all = st.button("All", use_container_width=True)
    with filter_cols[1]:
        show_failed = st.button("Failed", use_container_width=True)
    with filter_cols[2]:
        show_slow = st.button(f"Slow (>{THRESHOLDS['slow_turn_s']}s)", use_container_width=True)
    with filter_cols[3]:
        show_cached = st.button("Cached", use_container_width=True)

    if show_failed:
        st.session_state["conv_filter"] = "failed"
    elif show_slow:
        st.session_state["conv_filter"] = "slow"
    elif show_cached:
        st.session_state["conv_filter"] = "cached"
    elif show_all:
        st.session_state["conv_filter"] = "all"

    active_filter = st.session_state.get("conv_filter", "all")

    if active_filter == "failed":
        conversations = [c for c in conversations if c["status"] not in ("completed", "unknown", "")]
        st.caption(f"Showing {len(conversations)} failed turn(s)")
    elif active_filter == "slow":
        conversations = [c for c in conversations if c["response_time_s"] > THRESHOLDS["slow_turn_s"]]
        st.caption(f"Showing {len(conversations)} slow turn(s) (>{THRESHOLDS['slow_turn_s']}s)")
    elif active_filter == "cached":
        conversations = [c for c in conversations if c.get("is_cached")]
        st.caption(f"Showing {len(conversations)} cached turn(s)")

    if not conversations:
        st.info("No turns match the selected filter.")
        return

    # HV-5: Pagination for large conversations
    total_turns = len(conversations)
    page_size = THRESHOLDS["page_size"]
    if total_turns > page_size:
        total_pages = (total_turns + page_size - 1) // page_size
        page = st.select_slider(
            "Page",
            options=list(range(1, total_pages + 1)),
            value=1,
            format_func=lambda p: f"Page {p} (turns {(p-1)*page_size+1}–{min(p*page_size, total_turns)} of {total_turns})",
        )
        start = (page - 1) * page_size
        conversations = conversations[start:start + page_size]

    for conv in conversations:
        _render_turn(conv)


def _render_turn(conv):
    """Render a single conversation turn matching Fabric Data Agent UI."""

    # ── User question — teal bubble, right-aligned ──
    st.markdown(f"""
    <div class="user-bubble">
        <div class="user-bubble-inner">
            {html_lib.escape(conv['question'])}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Assistant response ──
    if conv["answer"]:
        with st.container(border=True):
            st.markdown(conv["answer"])

    # ── Step count + response time ──
    num_steps = len(conv["steps"])
    resp_time = conv["response_time_s"]
    is_slow = resp_time > THRESHOLDS["slow_turn_s"]

    col1, col2 = st.columns([3, 1])

    with col2:
        if resp_time > 0:
            time_color = "#e74c3c" if is_slow else "#666"
            slow_label = "SLOW " if is_slow else ""
            st.markdown(
                f'<div class="response-time" style="color:{time_color};">'
                f'{slow_label}{resp_time:.0f}s</div>',
                unsafe_allow_html=True,
            )

    with col1:
        if conv.get("is_cached"):
            st.markdown(
                '<span style="background:#f0e6ff; color:#7c3aed; padding:4px 10px; '
                'border-radius:4px; font-size:13px;">Cached response</span>',
                unsafe_allow_html=True,
            )
        elif num_steps > 0:
            status_label = "completed" if conv["status"] == "completed" else conv["status"]
            label = f"{num_steps} step{'s' if num_steps != 1 else ''} — {status_label}"
            with st.expander(label, expanded=is_slow):
                # Show time breakdown for slow responses
                if is_slow and num_steps > 0:
                    _render_time_breakdown(conv)
                for step in conv["steps"]:
                    _render_step(step)
        elif conv["answer"]:
            st.caption("No data query steps — agent answered from LLM knowledge")

    st.markdown("---")


def _render_time_breakdown(conv):
    """Show a visual time breakdown for slow responses."""
    steps = conv["steps"]
    total = conv["response_time_s"]
    step_total = sum(s["duration_s"] for s in steps)
    orchestrator = max(0, total - step_total)

    # Build breakdown bars
    rows = []
    for s in steps:
        pct = (s["duration_s"] / total * 100) if total > 0 else 0
        label = s["source_name"] or s["source_type"] or "step"
        rows.append({"Component": f"{s['code_language']} · {label}", "Duration (s)": s["duration_s"], "% of Total": pct})
    if orchestrator > 0:
        rows.append({"Component": "Orchestrator (queue + response)", "Duration (s)": orchestrator,
                      "% of Total": (orchestrator / total * 100) if total > 0 else 0})

    df = pd.DataFrame(rows)
    st.markdown(
        '<div style="background:#fff3cd; border-left:3px solid #f39c12; padding:8px 12px; '
        'border-radius:4px; margin-bottom:10px; font-size:13px;">'
        f'<strong>Slow response ({total:.0f}s)</strong> — breakdown:</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(df, use_container_width=True, hide_index=True,
                 column_config={"% of Total": st.column_config.ProgressColumn(
                     "% of Total", min_value=0, max_value=100, format="%.0f%%")})
    st.markdown("")


def _render_step(step):
    """Render a single analyze step with query code and output."""

    # Step header
    status_class = "" if step["status"] == "completed" else " failed"
    is_trace = step["source_type"] == "Trace"

    if is_trace:
        header_parts = [f'<strong>Trace</strong> · {html_lib.escape(step["source_name"])}']
    else:
        header_parts = [f'Analyzed <strong>{html_lib.escape(step["source_name"])}</strong>']
        if step["source_type"]:
            header_parts.append(html_lib.escape(step["source_type"]))
    if step["nl_query"]:
        header_parts.append(f'for: &ldquo;<em>{html_lib.escape(step["nl_query"][:300])}</em>&rdquo;')

    st.markdown(
        f'<div class="step-header{status_class}">{" ".join(header_parts)}</div>',
        unsafe_allow_html=True,
    )

    # Query code
    if step["generated_code"]:
        with st.expander("Query code"):
            lang = step["code_language"]
            lang_class = f"lang-{lang.lower()}"
            st.markdown(
                f'<span class="lang-badge {lang_class}">{lang}</span>',
                unsafe_allow_html=True,
            )
            st.code(step["generated_code"], language="sql")

    # Query output / Trace output
    if step["output"]:
        label = "Trace output" if is_trace else "Query output"
        with st.expander(label):
            _render_output(step["output"])

    # Duration
    if step["duration_s"] > 0:
        st.caption(f"Duration: {step['duration_s']:.1f}s")


def _render_output(output_str):
    """Render query output as a table if structured, otherwise as text."""
    if not output_str:
        return

    # HV-4: Check for error patterns
    error_patterns = ["error", "exception", "failed", "timeout", "canceled",
                      "cancelled", "not found", "invalid", "unable to"]
    output_lower = output_str.lower()
    is_error = any(p in output_lower for p in error_patterns)

    if is_error:
        st.markdown(
            '<div style="background:#fde8e8; border-left:3px solid #e74c3c; '
            'padding:8px 12px; border-radius:4px; margin-bottom:8px; font-size:13px;">'
            '<strong>Error detected in output</strong></div>',
            unsafe_allow_html=True,
        )

    # Try JSON array → DataFrame
    try:
        data = json.loads(output_str)
        if isinstance(data, list) and data:
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
            return
        elif isinstance(data, dict):
            st.json(data)
            return
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Try tab-separated values
    if "\t" in output_str and "\n" in output_str:
        lines = output_str.strip().split("\n")
        if len(lines) > 1:
            try:
                rows = [line.split("\t") for line in lines]
                if all(len(r) == len(rows[0]) for r in rows):
                    df = pd.DataFrame(rows[1:], columns=rows[0])
                    st.dataframe(df, use_container_width=True)
                    return
            except Exception:
                pass

    # Fallback: plain text (red for errors)
    if is_error:
        st.code(output_str[:3000], language="text")
    else:
        st.text(output_str[:3000])



# ──────────────────────────────────────────────────────────
# Analysis Dashboard
# ──────────────────────────────────────────────────────────

def render_analysis(parsed):
    """Render the analysis/diagnostics dashboard."""
    st.markdown("### Analysis Dashboard")

    _render_config_metrics(parsed)
    st.markdown("---")

    col_left, col_right = st.columns(2)
    with col_left:
        _render_schema_quality(parsed["data_sources"])
    with col_right:
        _render_latency_analysis(parsed["conversations"])

    st.markdown("---")
    _render_step_breakdown(parsed["conversations"])
    st.markdown("---")
    _render_issue_detection(parsed)


# ── 1. Configuration Metrics ─────────────────────────────

def _render_config_metrics(parsed):
    """Top-level metric cards for quick overview."""
    data_sources = parsed["data_sources"]
    conversations = parsed["conversations"]

    # Aggregate schema stats across all sources
    total_tables = total_selected = total_cols = total_measures = 0
    total_with_desc = total_objects = 0
    for ds in data_sources:
        t, s, c, m = _count_schema(ds["elements"])
        total_tables += t
        total_selected += s
        total_cols += c
        total_measures += m
        # Description coverage
        desc_counts = _count_descriptions(ds["elements"])
        total_with_desc += desc_counts["with_desc"]
        total_objects += desc_counts["total"]

    desc_pct = (total_with_desc / total_objects * 100) if total_objects > 0 else 0

    # Response time stats
    resp_times = [c["response_time_s"] for c in conversations if c["response_time_s"] > 0]
    avg_resp = sum(resp_times) / len(resp_times) if resp_times else 0
    max_resp = max(resp_times) if resp_times else 0

    # Total steps across all turns
    total_steps = sum(len(c["steps"]) for c in conversations)
    failed_steps = sum(
        1 for c in conversations for s in c["steps"] if s["status"] != "completed"
    )

    # Few-shot stats
    fewshot_results = parsed.get("fewshot_results", [])
    fewshot_loaded = sum(1 for f in fewshot_results if "Loaded 0" not in (f["output"] or ""))

    # Metric row 1: Configuration
    instr_len = len(parsed["agent_instructions"] or "")
    st.markdown("#### Configuration")
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("Data Sources", len(data_sources))
    m2.metric("Tables", f"{total_selected}/{total_tables}",
              help="Selected / Total tables")
    m3.metric("Columns", total_cols)
    m4.metric("Measures", total_measures)
    m5.metric("Description Coverage", f"{desc_pct:.0f}%",
              delta="Good" if desc_pct >= 80 else "Low",
              delta_color="normal" if desc_pct >= 80 else "inverse")
    m6.metric("Instructions", f"{instr_len:,} chars",
              help="Agent instructions character length")
    m7.metric("Few-shots", f"{fewshot_loaded}/{len(fewshot_results)}",
              help="Data sources with loaded few-shot examples")

    # Metric row 2: Performance
    cached_count = sum(1 for c in conversations if c.get("is_cached"))
    st.markdown("#### Performance")
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    p1.metric("Turns", len(conversations))
    p2.metric("Cached", cached_count,
              help="Responses served from cache (no new run/query executed)")
    p3.metric("Total Steps", total_steps)
    p4.metric("Failed Steps", failed_steps,
              delta="None" if failed_steps == 0 else f"{failed_steps} failed",
              delta_color="normal" if failed_steps == 0 else "inverse")
    p5.metric("Avg Response", f"{avg_resp:.1f}s")
    p6.metric("Max Response", f"{max_resp:.1f}s")


def _count_descriptions(elements):
    """Recursively count objects with/without descriptions."""
    total = with_desc = 0
    for el in elements:
        el_type = el.get("type", "")
        if "table" in el_type or "entity" in el_type:
            total += 1
            if el.get("description"):
                with_desc += 1
            for child in el.get("children", []):
                total += 1
                if child.get("description"):
                    with_desc += 1
        elif el.get("children"):
            sub = _count_descriptions(el["children"])
            total += sub["total"]
            with_desc += sub["with_desc"]
    return {"total": total, "with_desc": with_desc}


# ── 2. Schema Quality ────────────────────────────────────

def _render_schema_quality(data_sources):
    """Per-table schema quality breakdown."""
    st.markdown("#### Schema Quality")

    for ds in data_sources:
        st.markdown(f"**{ds['name']}** ({ds['type']})")

        rows = _build_table_stats(ds["elements"])
        if not rows:
            st.caption("No schema data")
            continue

        df = pd.DataFrame(rows)
        # Color the coverage column
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Table": st.column_config.TextColumn("Table", width="medium"),
                "Selected": st.column_config.TextColumn("Selected", width="small"),
                "Columns": st.column_config.NumberColumn("Columns", width="small"),
                "Measures": st.column_config.NumberColumn("Measures", width="small"),
                "With Desc": st.column_config.NumberColumn("With Desc", width="small"),
                ">200 chars": st.column_config.NumberColumn(">200 chars", width="small",
                                                             help="Descriptions over 200 chars (DA truncates)"),
                "Coverage": st.column_config.ProgressColumn(
                    "Desc Coverage",
                    min_value=0,
                    max_value=100,
                    format="%d%%",
                ),
            },
        )

        # Flag issues
        no_desc = [r for r in rows if r["Coverage"] < 50]
        unselected = [r for r in rows if r["Selected"] == "❌"]
        over_200 = [r for r in rows if r[">200 chars"] > 0]
        if no_desc:
            st.warning(
                f"{len(no_desc)} table(s) have <50% description coverage: "
                + ", ".join(r["Table"] for r in no_desc[:5])
            )
        if over_200:
            total_over = sum(r[">200 chars"] for r in over_200)
            st.warning(
                f"{total_over} description(s) exceed {THRESHOLDS['desc_char_limit']} chars (DA truncates): "
                + ", ".join(r["Table"] for r in over_200[:5])
            )
        if unselected:
            st.info(
                f"{len(unselected)} table(s) not selected: "
                + ", ".join(r["Table"] for r in unselected[:5])
            )


def _build_table_stats(elements):
    """Build per-table/entity stats for schema quality view."""
    rows = []
    for el in elements:
        el_type = el.get("type", "")
        if "table" in el_type or "entity" in el_type:
            name = el.get("display_name", "?")
            selected = el.get("is_selected", False)
            children = el.get("children", [])

            cols = [c for c in children if "column" in c.get("type", "")]
            measures = [c for c in children if "measure" in c.get("type", "")]
            all_objects = [el] + children  # include table/entity itself
            with_desc = sum(1 for o in all_objects if o.get("description"))
            over_200 = sum(1 for o in all_objects if len(_s(o.get("description"))) > THRESHOLDS["desc_char_limit"])
            total = len(all_objects)
            coverage = (with_desc / total * 100) if total > 0 else 0

            rows.append({
                "Table": name,
                "Selected": "✅" if selected else "❌",
                "Columns": len(cols),
                "Measures": len(measures),
                "With Desc": with_desc,
                "Total": total,
                "Coverage": int(coverage),
                ">200 chars": over_200,
            })
        elif el.get("children"):
            rows.extend(_build_table_stats(el["children"]))
    return rows


# ── 3. Latency Analysis ──────────────────────────────────

def _render_latency_analysis(conversations):
    """Per-turn response time chart."""
    st.markdown("#### Response Time per Turn")

    if not conversations:
        st.caption("No conversation data")
        return

    rows = []
    for conv in conversations:
        if conv.get("is_cached"):
            status = "cached"
        else:
            status = conv["status"]
        rows.append({
            "Turn": f"T{conv['turn']}",
            "Question": conv["question"][:50] + ("..." if len(conv["question"]) > 50 else ""),
            "Response Time (s)": conv["response_time_s"],
            "Steps": len(conv["steps"]),
            "Status": status,
        })

    df = pd.DataFrame(rows)

    if len(df) > 0:
        try:
            import plotly.express as px
            fig = px.bar(
                df,
                x="Turn",
                y="Response Time (s)",
                color="Status",
                color_discrete_map={
                    "completed": "#2ecc71",
                    "failed": "#e74c3c",
                    "cached": "#a78bfa",
                    "unknown": "#95a5a6",
                },
                hover_data=["Question", "Steps"],
                text="Response Time (s)",
            )
            fig.update_traces(texttemplate="%{text:.1f}s", textposition="outside")
            fig.update_layout(
                height=350,
                margin=dict(t=10, b=40),
                showlegend=True,
                yaxis_title="Seconds",
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.dataframe(df, use_container_width=True, hide_index=True)

    # Flag slow turns
    slow = [r for r in rows if r["Response Time (s)"] > THRESHOLDS["very_slow_turn_s"]]
    if slow:
        st.warning(
            f"{len(slow)} turn(s) took >{THRESHOLDS['very_slow_turn_s']}s: "
            + ", ".join(f"{r['Turn']} ({r['Response Time (s)']:.0f}s)" for r in slow)
        )


# ── 4. Step Breakdown ────────────────────────────────────

def _render_step_breakdown(conversations):
    """Detailed step-by-step breakdown across all turns."""
    st.markdown("#### Query Step Breakdown")

    all_steps = []
    for conv in conversations:
        for step in conv["steps"]:
            all_steps.append({
                "Turn": f"T{conv['turn']}",
                "Question": conv["question"][:40],
                "Function": step["source_type"],
                "Data Source": step["source_name"],
                "Language": step["code_language"],
                "Duration (s)": step["duration_s"],
                "Status": step["status"],
                "Has Code": "✅" if step["generated_code"] else "—",
                "Has Output": "✅" if step["output"] else "—",
            })

    if not all_steps:
        st.caption("No query steps found across any conversation turn")
        return

    df = pd.DataFrame(all_steps)

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Steps", len(all_steps))
    c2.metric("Avg Step Duration", f"{df['Duration (s)'].mean():.1f}s")
    c3.metric("Max Step Duration", f"{df['Duration (s)'].max():.1f}s")
    failed = len(df[df["Status"] != "completed"])
    c4.metric("Failed", failed, delta_color="inverse" if failed > 0 else "normal",
              delta="None" if failed == 0 else f"{failed} errors")

    # Step duration chart by turn
    try:
        import plotly.express as px
        fig = px.bar(
            df,
            x="Turn",
            y="Duration (s)",
            color="Language",
            color_discrete_map={"Dax": "#2ecc71", "SQL": "#3498db", "KQL": "#9b59b6", "Code": "#95a5a6"},
            hover_data=["Data Source", "Question", "Status"],
            barmode="group",
        )
        fig.update_layout(
            height=300,
            margin=dict(t=10, b=40),
            yaxis_title="Seconds",
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        pass

    # Full table
    with st.expander("All steps detail", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Slowest steps
    slow_steps = df.nlargest(3, "Duration (s)")
    if not slow_steps.empty and slow_steps.iloc[0]["Duration (s)"] > 5:
        st.markdown("**Slowest steps:**")
        for _, row in slow_steps.iterrows():
            st.caption(
                f"  {row['Turn']} · {row['Data Source']} · {row['Language']} · "
                f"**{row['Duration (s)']:.1f}s** · {row['Status']}"
            )


# ── 5. Issue Detection ───────────────────────────────────

def _render_issue_detection(parsed):
    """Automated issue detection with severity levels."""
    st.markdown("#### Issue Detection")

    issues = _detect_issues(parsed)

    if not issues:
        st.success("No issues detected!")
        return

    # Count by severity
    critical = [i for i in issues if i["Severity"] == "Critical"]
    warnings = [i for i in issues if i["Severity"] == "Warning"]
    info = [i for i in issues if i["Severity"] == "Info"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Critical", len(critical))
    c2.metric("Warnings", len(warnings))
    c3.metric("Info", len(info))

    df = pd.DataFrame(issues)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Severity": st.column_config.TextColumn("Severity", width="small"),
            "Category": st.column_config.TextColumn("Category", width="small"),
            "Issue": st.column_config.TextColumn("Issue", width="large"),
            "Recommendation": st.column_config.TextColumn("Recommendation", width="large"),
        },
    )


def _detect_issues(parsed):
    """Run all diagnostic checks and return list of issues."""
    issues = []
    conversations = parsed["conversations"]
    data_sources = parsed["data_sources"]
    agent_instr = parsed["agent_instructions"]

    # ── Schema issues ──
    for ds in data_sources:
        desc_counts = _count_descriptions(ds["elements"])
        total = desc_counts["total"]
        with_desc = desc_counts["with_desc"]
        if total > 0:
            pct = with_desc / total * 100
            if pct == 0:
                issues.append({
                    "Severity": "Critical",
                    "Category": "Descriptions",
                    "Issue": f"'{ds['name']}' has NO descriptions on any table, column, or measure ({total} objects)",
                    "Recommendation": "Add descriptions with synonyms and sample values to improve NL matching accuracy",
                })
            elif pct < 80:
                issues.append({
                    "Severity": "Warning",
                    "Category": "Descriptions",
                    "Issue": f"'{ds['name']}' — only {pct:.0f}% description coverage ({with_desc}/{total})",
                    "Recommendation": "Add descriptions to columns/measures the agent uses most",
                })

        # Large schema
        t, _, c, m = _count_schema(ds["elements"])
        if c + m > 100:
            issues.append({
                "Severity": "Warning",
                "Category": "Schema Size",
                "Issue": f"'{ds['name']}' has {c} columns + {m} measures = {c+m} objects",
                "Recommendation": "Deselect unused tables/columns to reduce agent confusion and token cost",
            })

        # Unselected data source
        if not ds["is_selected"]:
            issues.append({
                "Severity": "Info",
                "Category": "Configuration",
                "Issue": f"Data source '{ds['name']}' is not selected",
                "Recommendation": "Remove unused data sources or enable them if needed",
            })

    # ── Agent instructions ──
    if not agent_instr:
        issues.append({
            "Severity": "Info",
            "Category": "Instructions",
            "Issue": "No agent instructions configured",
            "Recommendation": "Add instructions for response formatting, terminology, and business rules",
        })
    elif len(_s(agent_instr)) > THRESHOLDS["instr_warn_chars"]:
        issues.append({
            "Severity": "Warning",
            "Category": "Instructions",
            "Issue": f"Agent instructions are {len(_s(agent_instr))} chars — may approach context limits",
            "Recommendation": "Trim verbose instructions; prioritize high-impact guidance",
        })

    # ── Per-source description & instructions ──
    _NO_DESC_INSTR_TYPES = {"semantic_model", "ontology"}
    for ds in data_sources:
        if not ds["description"] and ds["type"] not in _NO_DESC_INSTR_TYPES:
            issues.append({
                "Severity": "Info",
                "Category": "Configuration",
                "Issue": f"'{ds['name']}' has no data source description",
                "Recommendation": "Add a description to help the agent understand the data source context",
            })
        if not ds["instructions"] and ds["type"] not in _NO_DESC_INSTR_TYPES:
            issues.append({
                "Severity": "Info",
                "Category": "Configuration",
                "Issue": f"'{ds['name']}' has no data source instructions",
                "Recommendation": "Add query hints, naming conventions, or join guidance for this source",
            })

    # ── Few-shot issues ──
    fewshot_results = parsed.get("fewshot_results", [])
    for fr in fewshot_results:
        # Ontology sources don't support few-shots
        if fr.get("datasource_type", "").lower() == "ontology":
            continue
        if "Loaded 0" in (fr["output"] or ""):
            issues.append({
                "Severity": "Info",
                "Category": "Few-shots",
                "Issue": f"'{fr['datasource_name']}' loaded 0 few-shot examples",
                "Recommendation": "Add few-shot examples (NL question → query pairs) to improve query accuracy",
            })

    # ── Conversation issues ──
    for conv in conversations:
        # Failed runs
        if conv["status"] not in ("completed", "unknown", ""):
            issues.append({
                "Severity": "Critical",
                "Category": "Failed Run",
                "Issue": f"Turn {conv['turn']}: Run ended with status '{conv['status']}'",
                "Recommendation": "Check data source connectivity, permissions, or query syntax",
            })

        # No data query (LLM-only answer) — skip cached responses
        if not conv["steps"] and conv["answer"] and not conv.get("is_cached"):
            issues.append({
                "Severity": "Warning",
                "Category": "No Data Query",
                "Issue": f"Turn {conv['turn']}: Agent responded without querying any data source",
                "Recommendation": "Agent may be using LLM knowledge. Check if the question is in scope.",
            })

        # Slow response
        if conv["response_time_s"] > THRESHOLDS["very_slow_turn_s"]:
            issues.append({
                "Severity": "Warning",
                "Category": "Latency",
                "Issue": f"Turn {conv['turn']}: Response took {conv['response_time_s']:.0f}s (>{THRESHOLDS['very_slow_turn_s']}s threshold)",
                "Recommendation": "Check DAX/SQL complexity, schema size, or model performance",
            })

        # Slow individual steps (>10s)
        for step in conv["steps"]:
            if step["duration_s"] > THRESHOLDS["slow_step_s"]:
                issues.append({
                    "Severity": "Warning",
                    "Category": "Slow Step",
                    "Issue": (
                        f"Turn {conv['turn']}: '{step['source_name']}' "
                        f"{step['code_language']} step took {step['duration_s']:.1f}s"
                    ),
                    "Recommendation": "Review generated query complexity; consider simplifying the schema",
                })

        # Failed steps
        for step in conv["steps"]:
            if step["status"] not in ("completed", ""):
                issues.append({
                    "Severity": "Critical",
                    "Category": "Failed Step",
                    "Issue": (
                        f"Turn {conv['turn']}: Step '{step['source_name']}' "
                        f"ended with status '{step['status']}'"
                    ),
                    "Recommendation": "Check the generated code in the Conversation tab for errors",
                })

        # Empty/short answer
        if conv["answer"] and len(conv["answer"]) < 20:
            issues.append({
                "Severity": "Critical",
                "Category": "Empty Response",
                "Issue": f"Turn {conv['turn']}: Very short response ({len(conv['answer'])} chars)",
                "Recommendation": "Check if the question is in scope or data source returned empty results",
            })

        # Agent asked for clarification
        if conv["answer"]:
            answer_lower = conv["answer"].lower()
            clarification_phrases = [
                "could you clarify", "can you specify", "more specific",
                "what do you mean", "please provide", "i need more details",
                "wasn't able to retrieve", "unable to", "i'm not sure what",
            ]
            if any(phrase in answer_lower for phrase in clarification_phrases):
                issues.append({
                    "Severity": "Warning",
                    "Category": "Clarification",
                    "Issue": f"Turn {conv['turn']}: Agent asked for clarification instead of answering",
                    "Recommendation": "Add terminology definitions or threshold values to agent instructions",
                })

        # Step generated code but no output (possible execution failure)
        for step in conv["steps"]:
            if step["generated_code"] and not step["output"] and step["status"] == "completed":
                issues.append({
                    "Severity": "Info",
                    "Category": "No Output",
                    "Issue": (
                        f"Turn {conv['turn']}: '{step['source_name']}' generated code "
                        f"but returned no output"
                    ),
                    "Recommendation": "Query may have returned empty results — verify the generated DAX/SQL",
                })

    # Sort by severity
    severity_order = {"Critical": 0, "Warning": 1, "Info": 2}
    issues.sort(key=lambda x: severity_order.get(x["Severity"], 3))

    return issues


# ──────────────────────────────────────────────────────────
# Raw JSON Viewer
# ──────────────────────────────────────────────────────────

def render_raw_json_tab(raw):
    """Read-only raw JSON viewer with syntax highlighting."""
    st.markdown("### Raw JSON")
    st.caption("Read-only view of the full diagnostics file")

    # Top-level section selector
    sections = list(raw.keys())
    selected = st.selectbox("Section", ["Full file"] + sections)

    if selected == "Full file":
        st.json(raw)
    else:
        st.json(raw.get(selected, {}))


# ──────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────

def _render_banner():
    st.markdown("""
    <div class="top-banner">
        <div>
            <div class="banner-title">Agent Inspector</div>
            <div class="banner-subtitle">Fabric Data Agent Diagnostics Analyzer</div>
        </div>
        <div class="resources-wrap">
            <a class="banner-btn" href="#" onclick="return false;">Resources</a>
            <div class="resources-menu">
                <a href="https://learn.microsoft.com/en-us/fabric/data-science/semantic-model-best-practices" target="_blank">
                    Semantic Model Best Practices
                    <div class="res-label">Best practices for semantic model configuration</div>
                </a>
                <a href="https://learn.microsoft.com/en-us/fabric/data-science/data-agent-configurations" target="_blank">
                    Data Agent Configurations
                    <div class="res-label">How to configure Fabric Data Agents</div>
                </a>
                <a href="https://learn.microsoft.com/en-us/fabric/data-science/data-agent-configuration-best-practices" target="_blank">
                    Configuration Best Practices
                    <div class="res-label">Recommended configuration patterns</div>
                </a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    _render_banner()
    # ── Sidebar ──
    with st.sidebar:
        uploaded = st.file_uploader("Upload diagnostics JSON", type=["json"])

        if uploaded:
            # BF-4: File size check
            file_mb = uploaded.size / (1024 * 1024)
            if file_mb > THRESHOLDS["max_file_mb"]:
                st.warning(f"Large file ({file_mb:.0f} MB). Parsing may be slow.")

            file_key = f"{uploaded.name}_{uploaded.size}"
            if st.session_state.get("_file_key") != file_key:
                try:
                    raw = json.load(uploaded)
                except json.JSONDecodeError:
                    st.error("Invalid JSON file. Please check the file format.")
                    return
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                    return

                ok, err = validate_diagnostics(raw)
                if not ok:
                    st.error(f"{err}")
                    return

                try:
                    parsed = parse_diagnostics(raw)
                except Exception as e:
                    st.error(f"Error parsing diagnostics: {e}")
                    return

                st.session_state["parsed"] = parsed
                st.session_state["raw"] = raw
                st.session_state["_file_key"] = file_key

            if "parsed" in st.session_state:
                render_sidebar(st.session_state["parsed"])
        else:
            # Clear state when file is removed
            for key in ["parsed", "raw", "_file_key"]:
                st.session_state.pop(key, None)

    # ── Main content ──
    if "parsed" in st.session_state:
        tab_conv, tab_analysis, tab_json = st.tabs(["Conversation", "Analysis", "Raw JSON"])

        with tab_conv:
            render_main(st.session_state["parsed"])

        with tab_analysis:
            render_analysis(st.session_state["parsed"])

        with tab_json:
            render_raw_json_tab(st.session_state["raw"])
    else:
        st.markdown("""
        <div class="welcome">
            <h1>Agent Inspector</h1>
            <p>Upload a diagnostics JSON file using the sidebar to begin analysis.</p>
            <p style="font-size: 14px; margin-top: 16px; color: #888;">
                Supports Semantic Models, Lakehouse Tables, KQL Databases, Ontology (Graph), and more.<br>
                Few-shot examples, per-source connection details, descriptions and instructions for all source types.<br>
                The Explorer panel shows data sources, schema, instructions, and few-shot status.
            </p>
        </div>
        """, unsafe_allow_html=True)


main()
