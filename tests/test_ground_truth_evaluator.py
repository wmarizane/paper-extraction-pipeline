import csv
import tempfile
import unittest
from pathlib import Path

from pipeline.ground_truth_evaluator import (
    PolyCritRow,
    evaluate_rows,
    field_similarity,
    load_polycrit_csv,
    match_reference_group,
    parse_references,
)
from pipeline.standardized_csv_exporter import POLYCRIT_FIELDNAMES


def make_row(row_id, reference, values=None):
    row_values = {field: "" for field in POLYCRIT_FIELDNAMES}
    row_values["Reference"] = reference
    row_values.update(values or {})
    return PolyCritRow(
        row_id=row_id,
        values=row_values,
        references=parse_references(reference),
        source="synthetic",
    )


class GroundTruthEvaluatorTests(unittest.TestCase):
    def test_composite_reference_parser(self):
        self.assertEqual(parse_references("95, 112, 107"), ("95", "112", "107"))
        self.assertEqual(parse_references(124.0), ("124",))
        self.assertEqual(parse_references("[251] Trathnigg2005"), ("251",))
        self.assertEqual(parse_references("Im2008"), ("im2008",))

    def test_solvent_locants_and_aliases_are_preserved(self):
        self.assertEqual(
            field_similarity(
                "Solvents",
                "1,4-Dioxane, Hexane",
                "1,4-Dioxane, n-hexane",
            ),
            1.0,
        )
        self.assertEqual(
            field_similarity(
                "Solvents",
                "Tetrahydrofuran, Water",
                "THF, H2O",
            ),
            1.0,
        )
        self.assertEqual(
            field_similarity(
                "Solvents",
                "Water, Tetrahydrofuran",
                "THF, H2O",
            ),
            0.5,
        )
        self.assertEqual(
            field_similarity(
                "Polymer",
                "Poly[(adipic acid)-co-(1,2-ethanediol)]",
                "AA-1,2ED",
            ),
            1.0,
        )

    def test_repeated_reference_uses_one_to_one_matching(self):
        common = {
            "Polymer": "Poly(ethylene glycol)",
            "Stationary Phase": "C18",
            "Phase": "Reverse",
            "Temperature (Celsius)": "25",
        }
        gt_one = make_row(
            "gt:1",
            "124",
            {
                **common,
                "Solvents": "Acetonitrile, Water",
                "Solvent Ratio (vol%)": "47, 53",
            },
        )
        gt_two = make_row(
            "gt:2",
            "124",
            {
                **common,
                "Solvents": "Methanol, Water",
                "Solvent Ratio (vol%)": "80, 20",
            },
        )
        ex_two = make_row(
            "ex:2",
            "124",
            {
                **common,
                "Polymer": "PEG",
                "Solvents": "MeOH, H2O",
                "Solvent Ratio (vol%)": "80, 20",
            },
        )
        ex_one = make_row(
            "ex:1",
            "124",
            {
                **common,
                "Polymer": "PEO",
                "Solvents": "ACN, H2O",
                "Solvent Ratio (vol%)": "47, 53",
            },
        )
        extra = make_row(
            "ex:3",
            "124",
            {
                **common,
                "Solvents": "Tetrahydrofuran, Water",
                "Solvent Ratio (vol%)": "60, 40",
            },
        )

        matched, false_negatives, false_positives = match_reference_group(
            "124",
            [gt_one, gt_two],
            [ex_two, ex_one, extra],
        )
        self.assertEqual(len(matched), 2)
        self.assertEqual(false_negatives, [])
        self.assertEqual([row.row_id for row in false_positives], ["ex:3"])
        pair_ids = {
            (pair.ground_truth.row_id, pair.extracted.row_id) for pair in matched
        }
        self.assertEqual(pair_ids, {("gt:1", "ex:1"), ("gt:2", "ex:2")})

    def test_composite_reference_is_evaluated_per_membership(self):
        values = {
            "Polymer": "Poly(ethylene glycol)",
            "Solvents": "Acetonitrile, Water",
            "Solvent Ratio (vol%)": "44, 56",
            "Phase": "Reverse",
        }
        ground_truth = make_row("gt:1", "95, 112", values)
        extracted_95 = make_row("ex:95", "95", values)
        extracted_112 = make_row("ex:112", "112", values)

        report = evaluate_rows([ground_truth], [extracted_95, extracted_112])
        summary = report["summary"]
        self.assertEqual(summary["ground_truth_rows"], 1)
        self.assertEqual(summary["ground_truth_reference_memberships"], 2)
        self.assertEqual(summary["matched_pairs"], 2)
        self.assertEqual(summary["false_negatives"], 0)
        self.assertEqual(summary["false_positives"], 0)

    def test_unmatched_rows_are_reported(self):
        values = {
            "Polymer": "Poly(L-lactide)",
            "Solvents": "1,4-Dioxane, Hexane",
            "Phase": "Normal",
        }
        ground_truth = make_row("gt:1", "210", values)
        missing_ground_truth = make_row("gt:2", "211", values)
        extracted = make_row("ex:1", "210", values)
        extra = make_row(
            "ex:2",
            "210",
            {
                **values,
                "Solvents": "Acetonitrile, Water",
            },
        )
        out_of_scope = make_row("ex:3", "999", values)

        report = evaluate_rows(
            [ground_truth, missing_ground_truth],
            [extracted, extra, out_of_scope],
        )
        summary = report["summary"]
        self.assertEqual(summary["matched_pairs"], 1)
        self.assertEqual(summary["false_negatives"], 1)
        self.assertEqual(summary["false_positives"], 1)
        self.assertEqual(summary["out_of_scope_extracted_memberships"], 1)
        self.assertEqual(report["field_summary"]["Polymer"]["missing"], 1)

    def test_csv_loader_uses_shared_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.csv"
            values = {field: "" for field in POLYCRIT_FIELDNAMES}
            values.update(
                {
                    "Reference": "95, 112",
                    "Polymer": "Poly(ethylene glycol)",
                    "Solvents": "Acetonitrile, Water",
                }
            )
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=POLYCRIT_FIELDNAMES)
                writer.writeheader()
                writer.writerow(values)

            rows = load_polycrit_csv(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].references, ("95", "112"))
            self.assertEqual(rows[0].values["Solvents"], "Acetonitrile, Water")


if __name__ == "__main__":
    unittest.main()
