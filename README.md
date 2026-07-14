# paper-extraction-pipeline
Pipeline for extracting unstructured data from scientific papers into structured data.

# How to run GROBID
```sh
# On mac
open -a docker

# Start GROBID
docker run -d --rm -p 8070:8070 lfoppiano/grobid:0.8.0
```

# How to run the parser
```
python pipeline/pdf_parser.py <path_to_pdf>
```

# Intermediate TEI XML files
When running the full pipeline (`run_local.py` or `run_extraction.slurm`), raw GROBID TEI XML is saved to:

`./tei_xml/<pdf_name>_tei.xml`

## Standardize and evaluate against PolyCrit

Run the standardizer, export the PolyCrit shaped CSV, then evaluate it against
the human verified workbook:

```powershell
python pipeline/standardizer.py results/consensus results/standardized_latest
python pipeline/standardized_csv_exporter.py results/standardized_latest results/polycrit_standardized.csv
python -m pipeline.ground_truth_evaluator C:\path\to\Pharma_Polymers.xlsx results/polycrit_standardized.csv evaluation
```

The evaluator writes a JSON report, a one to one condition match report, and a
field score report. Repeated and composite references are retained. Unmatched
human rows and unmatched extracted rows are reported separately.

Use a fresh standardized output directory for each evaluation. This prevents
older JSON files from being included as extra extracted conditions.
