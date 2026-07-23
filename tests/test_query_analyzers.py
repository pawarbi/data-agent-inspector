import unittest

from query_analyzers import analyze_query


class QueryAnalyzerTests(unittest.TestCase):
    def test_dax_extracts_schema_objects_filters_and_generated_measures(self):
        schema = [{
            "type": "semantic_model.table",
            "display_name": "Sales",
            "children": [
                {
                    "type": "semantic_model.column",
                    "display_name": "Amount",
                    "children": [],
                },
                {
                    "type": "semantic_model.measure",
                    "display_name": "Total Sales",
                    "children": [],
                },
            ],
        }]
        code = """
        DEFINE
          MEASURE 'Sales'[Average Sale] = AVERAGE('Sales'[Amount])
        EVALUATE
          ADDCOLUMNS(
            CALCULATETABLE(
              ROW("Total", [Total Sales]),
              FILTER('Sales', 'Sales'[Amount] > 100)
            ),
            "Filter - Sales[Amount]",
            100
          )
        ORDER BY
          [Total Sales] DESC
        """

        result = analyze_query("Dax", code, schema)

        self.assertIn("Sales", result["tables"])
        self.assertIn("Sales[Amount]", result["columns"])
        self.assertIn("[Total Sales]", result["measures"])
        self.assertIn("Sales[Average Sale]", result["generated_measures"])
        self.assertNotIn(
            "Sales[Average Sale]",
            result["other_references"],
        )
        self.assertFalse(
            any(value.startswith("MEASURE ") for value in result["filters"])
        )
        self.assertNotIn("Sales", result["tables"][1:])
        self.assertNotIn("BY", result["tables"])
        self.assertIn("FILTER()", result["filters"])
        self.assertIn("AVERAGE", result["aggregations"])

    def test_sql_extracts_tables_columns_filters_joins_and_aggregations(self):
        code = """
        SELECT c.CustomerName, SUM(s.Amount) AS TotalAmount
        FROM dbo.Sales AS s
        INNER JOIN dbo.Customers AS c ON c.CustomerId = s.CustomerId
        WHERE s.OrderDate >= '2026-01-01'
        GROUP BY c.CustomerName
        HAVING SUM(s.Amount) > 1000
        """

        result = analyze_query("SQL", code)

        self.assertTrue(any("Sales" in table for table in result["tables"]))
        self.assertTrue(any("Customers" in table for table in result["tables"]))
        self.assertTrue(any("CustomerName" in column for column in result["columns"]))
        self.assertTrue(any("OrderDate" in value for value in result["filters"]))
        self.assertTrue(result["joins"])
        self.assertIn("SUM", result["aggregations"])

    def test_sql_accepts_tsql_leading_semicolon_cte(self):
        result = analyze_query(
            "SQL",
            """
            -- Latest rows
            ;WITH Latest AS (
                SELECT CustomerId, MAX(SnapshotId) AS SnapshotId
                FROM dbo.Sales
                GROUP BY CustomerId
            )
            SELECT COUNT(*) AS Customers
            FROM Latest;
            """,
        )

        self.assertEqual(result["parse_error"], "")
        self.assertTrue(any("Sales" in table for table in result["tables"]))

    def test_comment_markers_inside_strings_are_preserved(self):
        sql_result = analyze_query(
            "SQL",
            "SELECT * FROM dbo.Events WHERE Message = 'a--b'",
        )
        kql_result = analyze_query(
            "KQL",
            'Events | where Url == "https://example.test/a" | summarize count()',
        )

        self.assertTrue(any("Events" in table for table in sql_result["tables"]))
        self.assertTrue(kql_result["filters"])
        self.assertIn("count", kql_result["aggregations"])

    def test_kql_extracts_pipeline_structure_using_schema(self):
        schema = [{
            "type": "kusto.table",
            "display_name": "Events",
            "children": [
                {
                    "type": "kusto.column",
                    "display_name": "Timestamp",
                    "children": [],
                },
                {
                    "type": "kusto.column",
                    "display_name": "Region",
                    "children": [],
                },
                {
                    "type": "kusto.column",
                    "display_name": "Value",
                    "children": [],
                },
            ],
        }]
        code = """
        Events
        | where Timestamp >= ago(30d) and Region == "West"
        | summarize TotalValue=sum(Value) by Region
        | order by TotalValue desc
        """

        result = analyze_query("KQL", code, schema)

        self.assertIn("Events", result["tables"])
        self.assertIn("Timestamp", result["columns"])
        self.assertIn("Region", result["columns"])
        self.assertTrue(any(value.startswith("where ") for value in result["filters"]))
        self.assertTrue(any("summarize" in value for value in result["aggregations"]))
        self.assertIn("sum", result["aggregations"])

    def test_kql_extracts_sources_from_let_statements(self):
        result = analyze_query(
            "KQL",
            """
            let recent = Events | where Timestamp > ago(1d);
            recent | summarize count() by Region
            """,
        )

        self.assertIn("Events", result["tables"])
        self.assertIn("recent", [value.lower() for value in result["tables"]])
        self.assertTrue(result["filters"])


if __name__ == "__main__":
    unittest.main()
