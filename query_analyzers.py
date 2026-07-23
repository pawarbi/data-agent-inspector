"""Deterministic structural analyzers for generated DAX, SQL, and KQL."""

import re


_DAX_FILTER_FUNCTIONS = {
    "ALL",
    "ALLEXCEPT",
    "ALLSELECTED",
    "CALCULATE",
    "CALCULATETABLE",
    "CROSSFILTER",
    "DATESBETWEEN",
    "DATESINPERIOD",
    "FILTER",
    "KEEPFILTERS",
    "REMOVEFILTERS",
    "TREATAS",
    "USERELATIONSHIP",
}

_DAX_AGGREGATION_FUNCTIONS = {
    "AVERAGE",
    "AVERAGEX",
    "COUNT",
    "COUNTA",
    "COUNTROWS",
    "DISTINCTCOUNT",
    "MAX",
    "MAXX",
    "MIN",
    "MINX",
    "SUM",
    "SUMMARIZE",
    "SUMMARIZECOLUMNS",
    "SUMX",
}

_SQL_AGGREGATION_FUNCTIONS = {
    "AVG",
    "COUNT",
    "MAX",
    "MIN",
    "STRING_AGG",
    "SUM",
}

_KQL_AGGREGATION_FUNCTIONS = {
    "arg_max",
    "arg_min",
    "avg",
    "count",
    "count_distinct",
    "dcount",
    "make_list",
    "make_set",
    "max",
    "min",
    "percentile",
    "sum",
}


def analyze_query(language, code, schema_elements=None):
    """Return normalized structural facts for a generated query."""
    normalized = (language or "").strip().lower()
    code = (code or "").strip()
    if not code:
        return _empty_result(language, "No query")

    if normalized == "dax":
        return _analyze_dax(code, schema_elements or [])
    if normalized == "sql":
        return _analyze_sql(code)
    if normalized == "kql":
        return _analyze_kql(code, schema_elements or [])

    result = _empty_result(language, "Not available")
    result["notes"].append(
        f"Structural analysis is not available for {language or 'this language'}."
    )
    return result


def _empty_result(language, parser):
    return {
        "language": language or "Unknown",
        "parser": parser,
        "confidence": "Not analyzed",
        "tables": [],
        "columns": [],
        "measures": [],
        "other_references": [],
        "generated_measures": [],
        "filters": [],
        "joins": [],
        "aggregations": [],
        "notes": [],
        "parse_error": "",
    }


def _dedupe(values):
    seen = set()
    result = []
    for value in values:
        value = str(value).strip()
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _strip_comments(code, line_marker="//", block_comments=True):
    result = []
    index = 0
    quote = ""
    while index < len(code):
        if quote:
            char = code[index]
            result.append(char)
            if char == "\\" and index + 1 < len(code):
                index += 1
                result.append(code[index])
            elif char == quote:
                if index + 1 < len(code) and code[index + 1] == quote:
                    index += 1
                    result.append(code[index])
                else:
                    quote = ""
            index += 1
            continue

        if code[index] in {"'", '"'}:
            quote = code[index]
            result.append(code[index])
            index += 1
            continue

        if block_comments and code.startswith("/*", index):
            end = code.find("*/", index + 2)
            index = len(code) if end < 0 else end + 2
            result.append(" ")
            continue

        if line_marker and code.startswith(line_marker, index):
            newline = code.find("\n", index + len(line_marker))
            if newline < 0:
                break
            result.append("\n")
            index = newline + 1
            continue

        result.append(code[index])
        index += 1

    return "".join(result)


def _element_kind(element_type):
    normalized = (element_type or "").lower()
    leaf_type = normalized.rsplit(".", 1)[-1]
    if normalized == "ontology.entity" or leaf_type in {
        "table",
        "view",
        "external_table",
        "materialized_view",
    }:
        return "table"
    if leaf_type == "column":
        return "column"
    if leaf_type == "measure":
        return "measure"
    return "group"


def _schema_catalog(elements):
    catalog = {
        "tables": {},
        "columns": {},
        "measures": {},
    }

    def visit(items, parent_table=""):
        for element in items or []:
            name = str(element.get("display_name") or element.get("name") or "")
            kind = _element_kind(element.get("type", ""))
            children = element.get("children") or []

            if kind == "table":
                catalog["tables"][name.lower()] = name
                visit(children, name)
            elif kind in {"column", "measure"}:
                qualified = f"{parent_table}[{name}]" if parent_table else f"[{name}]"
                catalog[f"{kind}s"][(parent_table.lower(), name.lower())] = qualified
                visit(children, parent_table)
            else:
                visit(children, parent_table)

    visit(elements)
    return catalog


def _quoted_spans(code, quote):
    """Return spans for quoted strings, accounting for doubled quote escapes."""
    spans = []
    start = None
    index = 0
    while index < len(code):
        if start is None:
            if code[index] == quote:
                start = index
        elif code[index] == quote:
            if index + 1 < len(code) and code[index + 1] == quote:
                index += 1
            else:
                spans.append((start, index + 1))
                start = None
        index += 1
    if start is not None:
        spans.append((start, len(code)))
    return spans


def _inside_span(start, end, spans):
    return any(span_start <= start and end <= span_end for span_start, span_end in spans)


def _analyze_dax(code, schema_elements):
    result = _empty_result("DAX", "DAX reference analyzer")
    result["confidence"] = "Medium"
    clean = _strip_comments(code)
    catalog = _schema_catalog(schema_elements)
    string_spans = _quoted_spans(clean, '"')

    generated_measures = []
    declaration_spans = []
    generated_pattern = re.compile(
        r"(?im)^\s*MEASURE\s+"
        r"(?:'((?:[^']|'')+)'|([A-Za-z_][A-Za-z0-9_]*))\s*"
        r"\[([^\]]+)\]\s*="
    )
    for match in generated_pattern.finditer(clean):
        table = (match.group(1) or match.group(2) or "").replace("''", "'")
        measure = match.group(3).strip()
        generated_measures.append(f"{table}[{measure}]")
        declaration_spans.append(match.span())

    qualified_pattern = re.compile(
        r"(?:'((?:[^']|'')+)'\s*\[|([A-Za-z_][A-Za-z0-9_]*)\[)([^\]]+)\]"
    )
    qualified_spans = []
    tables = []
    columns = []
    measures = []
    unknown = []

    for match in qualified_pattern.finditer(clean):
        if _inside_span(match.start(), match.end(), string_spans):
            continue
        if any(
            start <= match.start() and match.end() <= end
            for start, end in declaration_spans
        ):
            continue
        table = (match.group(1) or match.group(2) or "").replace("''", "'")
        artifact = match.group(3).strip()
        qualified_spans.append(match.span(3))
        tables.append(table)

        key = (table.lower(), artifact.lower())
        reference = f"{table}[{artifact}]"
        if key in catalog["measures"]:
            measures.append(reference)
        elif key in catalog["columns"]:
            columns.append(reference)
        else:
            unknown.append(reference)

    for generated_measure in generated_measures:
        table = generated_measure.split("[", 1)[0]
        tables.append(table)

    generated_names = {
        value.rsplit("[", 1)[-1].rstrip("]").lower()
        for value in generated_measures
    }
    for match in re.finditer(r"\[([^\]]+)\]", clean):
        if _inside_span(match.start(), match.end(), string_spans):
            continue
        if any(start <= match.start(1) < end for start, end in qualified_spans):
            continue
        if any(
            start <= match.start() and match.end() <= end
            for start, end in declaration_spans
        ):
            continue
        name = match.group(1).strip()
        if name.lower() in generated_names:
            measures.append(f"[{name}]")
        elif any(key[1] == name.lower() for key in catalog["measures"]):
            measures.append(f"[{name}]")
        else:
            unknown.append(f"[{name}]")

    present_functions = {
        name.upper()
        for name in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", clean)
    }
    filter_functions = sorted(present_functions & _DAX_FILTER_FUNCTIONS)
    aggregation_functions = sorted(
        present_functions & _DAX_AGGREGATION_FUNCTIONS
    )

    predicate_lines = []
    for line in clean.splitlines():
        stripped = line.strip().rstrip(",")
        if (
            stripped
            and len(stripped) <= 240
            and re.search(r"(?:>=|<=|<>|=|>|<|\bIN\b)", stripped, re.IGNORECASE)
            and not stripped.upper().startswith(
                ("DEFINE", "VAR ", "MEASURE ", "COLUMN ")
            )
        ):
            predicate_lines.append(stripped)

    result.update({
        "tables": _dedupe(tables),
        "columns": _dedupe(columns),
        "measures": _dedupe(measures),
        "other_references": _dedupe(unknown),
        "generated_measures": _dedupe(generated_measures),
        "filters": _dedupe(
            [f"{name}()" for name in filter_functions] + predicate_lines
        ),
        "aggregations": aggregation_functions,
    })

    if not schema_elements and result["other_references"]:
        result["notes"].append(
            "Model metadata was not exported, so some DAX references could not be "
            "classified as columns or measures."
        )
    if "IFERROR" in present_functions:
        result["notes"].append("IFERROR is present; DIVIDE or explicit error handling may be clearer.")
    return result


def _analyze_sql(code):
    clean = _strip_comments(code, line_marker="--").strip()
    if not clean:
        result = _empty_result("SQL", "No executable SQL")
        result["notes"].append(
            "The recorded query contains comments but no executable SQL statement."
        )
        return result

    try:
        import sqlglot
        from sqlglot import exp

        clean = re.sub(r"^;\s*(?=WITH\b)", "", clean, flags=re.IGNORECASE)
        tree = sqlglot.parse_one(clean, read="tsql")
        result = _empty_result("SQL", "sqlglot T-SQL AST")
        result["confidence"] = "High"

        tables = []
        for table in tree.find_all(exp.Table):
            tables.append(table.sql(dialect="tsql"))

        columns = []
        for column in tree.find_all(exp.Column):
            columns.append(column.sql(dialect="tsql"))

        filters = []
        for where in tree.find_all(exp.Where):
            filters.append(where.this.sql(dialect="tsql"))
        for having in tree.find_all(exp.Having):
            filters.append(f"HAVING {having.this.sql(dialect='tsql')}")

        joins = []
        for join in tree.find_all(exp.Join):
            joins.append(join.sql(dialect="tsql"))

        aggregations = []
        for aggregate in tree.find_all(exp.AggFunc):
            aggregations.append(aggregate.sql_name())

        result.update({
            "tables": _dedupe(tables),
            "columns": _dedupe(columns),
            "filters": _dedupe(filters),
            "joins": _dedupe(joins),
            "aggregations": _dedupe(aggregations),
        })
        return result
    except ImportError:
        result = _analyze_sql_fallback(code)
        result["notes"].append(
            "sqlglot is unavailable; results use a lower-confidence SQL scanner."
        )
        return result
    except Exception as exc:
        result = _analyze_sql_fallback(code)
        result["parse_error"] = str(exc)
        result["notes"].append(
            "The SQL AST parser could not parse the query; fallback extraction was used."
        )
        return result


def _analyze_sql_fallback(code):
    clean = _strip_comments(code, line_marker="--")
    result = _empty_result("SQL", "SQL fallback scanner")
    result["confidence"] = "Low"
    result["tables"] = _dedupe(re.findall(
        r"(?i)\b(?:FROM|JOIN|UPDATE|INTO)\s+"
        r"((?:\[[^\]]+\]|[A-Za-z_][\w$#]*)(?:\.(?:\[[^\]]+\]|[A-Za-z_][\w$#]*)){0,2})",
        clean,
    ))

    filters = re.findall(
        r"(?is)\bWHERE\s+(.*?)(?=\bGROUP\s+BY\b|\bHAVING\b|\bORDER\s+BY\b|;|$)",
        clean,
    )
    having = re.findall(
        r"(?is)\bHAVING\s+(.*?)(?=\bORDER\s+BY\b|;|$)",
        clean,
    )
    result["filters"] = _dedupe(
        [value.strip() for value in filters]
        + [f"HAVING {value.strip()}" for value in having]
    )
    result["joins"] = _dedupe(re.findall(
        r"(?is)\b(?:INNER|LEFT|RIGHT|FULL|CROSS)?\s*JOIN\s+"
        r"(.+?)(?=\b(?:INNER|LEFT|RIGHT|FULL|CROSS)?\s*JOIN\b|\bWHERE\b|\bGROUP\b|\bORDER\b|;|$)",
        clean,
    ))
    functions = {
        name.upper()
        for name in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", clean)
    }
    result["aggregations"] = sorted(functions & _SQL_AGGREGATION_FUNCTIONS)
    return result


def _split_kql_pipeline(code):
    parts = []
    current = []
    quote = ""
    depth = 0
    for char in code:
        if quote:
            current.append(char)
            if char == quote:
                quote = ""
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
        elif char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth = max(0, depth - 1)
            current.append(char)
        elif char == "|" and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    parts.append("".join(current).strip())
    return [part for part in parts if part]


def _split_statements(code):
    statements = []
    current = []
    quote = ""
    depth = 0
    for char in code:
        if quote:
            current.append(char)
            if char == quote:
                quote = ""
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
        elif char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth = max(0, depth - 1)
            current.append(char)
        elif char == ";" and depth == 0:
            statements.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    statements.append("".join(current).strip())
    return [statement for statement in statements if statement]


def _analyze_kql(code, schema_elements):
    result = _empty_result("KQL", "KQL pipeline analyzer")
    result["confidence"] = "Medium"
    clean = _strip_comments(code)
    catalog = _schema_catalog(schema_elements)

    tables = []
    columns = []
    filters = []
    joins = []
    aggregations = []

    pipelines = []
    for statement in _split_statements(clean):
        let_match = re.match(
            r"(?is)^\s*let\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*(.+)$",
            statement,
        )
        pipeline_text = let_match.group(1) if let_match else statement
        segments = _split_kql_pipeline(pipeline_text)
        if not segments:
            continue
        pipelines.append(segments)
        first_identifier = re.match(
            r"\s*([A-Za-z_][A-Za-z0-9_]*)",
            segments[0],
        )
        if (
            first_identifier
            and first_identifier.group(1).lower() not in {"let", "print"}
        ):
            tables.append(first_identifier.group(1))

    for table_name in catalog["tables"].values():
        if re.search(rf"(?<![\w]){re.escape(table_name)}(?![\w])", clean, re.IGNORECASE):
            tables.append(table_name)
    for (_, column_name), qualified in catalog["columns"].items():
        display_name = qualified.rsplit("[", 1)[-1].rstrip("]")
        if re.search(rf"(?<![\w]){re.escape(display_name)}(?![\w])", clean, re.IGNORECASE):
            columns.append(display_name)

    for segments in pipelines:
        for segment in segments[1:]:
            operator_match = re.match(
                r"\s*([A-Za-z_][A-Za-z0-9_-]*)",
                segment,
            )
            operator = operator_match.group(1).lower() if operator_match else ""
            if operator in {"where", "filter"}:
                filters.append(segment)
            elif operator in {"join", "lookup", "union"}:
                joins.append(segment)
                tables.extend(re.findall(
                    r"(?i)\b(?:join|lookup|union)(?:\s+kind\s*=\s*\w+)?\s*\(?\s*"
                    r"([A-Za-z_][A-Za-z0-9_]*)",
                    segment,
                ))
            elif operator == "summarize":
                aggregations.append(segment)

    function_names = {
        name.lower()
        for name in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", clean)
    }
    aggregate_functions = sorted(function_names & _KQL_AGGREGATION_FUNCTIONS)

    result.update({
        "tables": _dedupe(tables),
        "columns": _dedupe(columns),
        "filters": _dedupe(filters),
        "joins": _dedupe(joins),
        "aggregations": _dedupe(aggregate_functions + aggregations),
    })
    if not schema_elements:
        result["notes"].append(
            "KQL schema metadata was not exported; column extraction may be incomplete."
        )
    return result
