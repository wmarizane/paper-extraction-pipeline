# Copilot Instructions - Paper Extraction Pipeline

## Project Overview

**Objective:** Build a production-grade LLM-based pipeline to automatically extract structured scientific data from research PDFs with high reliability.

**Key Goals:**
- Use LLMs to convert unstructured text into structured datasets
- Improve reliability through verification and correction pipelines
- Scale to large PDF collections on GPU cluster infrastructure
- Implement automated evaluation and quality improvement mechanisms

## Project Scope & Responsibilities

This is a research infrastructure project with two team members:

**Balaji (High School Research Student):**
- Experimentation with PDF parsing, chunking, and local LLM extraction
- Prompt design and extraction quality improvements

**Wesley (Research Assistant - YOUR ROLE):**
- **Build robust infrastructure pipeline**
- **Cluster execution, logging, batching, and reproducibility**
- **Build evaluation, verification, and automation framework** ← Core responsibility

## System Architecture (7-Stage Pipeline)

```
1. PDF Documents
   ↓
2. PDF Parsing (GROBID)
   ↓
3. Text Chunking
   ↓
4. LLM Extraction (qwen/phi-mini)
   ↓
5. JSON Structuring
   ↓
6. Verification ← CRITICAL COMPONENT
   ↓
7. Correction
   ↓
Final Dataset
```

**Current Status:**
- ✅ Stage 1-2: PDF parsing via GROBID (basic implementation exists)
- ✅ Stage 5: JSON output (see `extracted_lccc_data.json`)
- ❌ Stage 3: Text chunking for LLM consumption
- ❌ Stage 4: Scalable LLM extraction pipeline
- ❌ Stage 6: Verification layer (YOUR KEY DELIVERABLE)
- ❌ Stage 7: Correction mechanisms
- ❌ Cluster execution & batch processing
- ❌ Logging & experiment tracking

## Prerequisites

### GPU Cluster Access

**Login:** `ssh login_node`  
**Working Directory:** `/project/wkmrzane/`

Pipeline must support cluster execution for batch processing at scale.

### GROBID Server

GROBID must be running locally before executing the parser:

```bash
# Start Docker (macOS)
open -a docker

# Start GROBID container
docker run -d --rm -p 8070:8070 lfoppiano/grobid:0.8.0
```

The parser expects GROBID at `http://localhost:8070` by default.

## Running the Pipeline

### Parse a PDF

```bash
python pipeline/pdf_parser.py <path_to_pdf>

# Example:
python pipeline/pdf_parser.py Inputs/polymerPaper1.pdf
```

The script will:
1. Check if GROBID server is running
2. Send the PDF to GROBID's `/api/processFulltextDocument` endpoint
3. Parse the returned TEI XML
4. Print a preview of extracted text

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Key Conventions

### TEI XML Processing

GROBID returns structured TEI XML. The parser extracts:
- **Title:** From `//tei:titleStmt/tei:title`
- **Abstract:** From `//tei:profileDesc/tei:abstract/tei:p`
- **Body:** From `//tei:text/tei:body//tei:p`

Namespace: `http://www.tei-c.org/ns/1.0`

### GROBID API Parameters

When calling `processFulltextDocument`, the pipeline uses:
- `generateIDs=true` - Generate XML IDs for elements
- `consolidateHeader=0` - Skip header consolidation with external services
- `consolidateCitations=0` - Skip citation consolidation
- `includeRawCitations=1` - Include raw citation strings

These minimize external API calls and processing time while retaining essential data.

### Output Structure

Extracted data (e.g., `extracted_lccc_data.json`) contains:
- **metadata:** Statistics about extraction (polymers found, processing time, LLM usage)
- **LCCC_conditions:** Extracted experimental conditions (domain-specific)
- **tables:** Structured data including polymers and conditions

The JSON uses a "compact" output mode with LLM-based extraction for domain-specific entities.

## Project Structure

```
.
├── Inputs/              # Input PDF files
├── pipeline/
│   └── pdf_parser.py    # Main parser script
├── requirements.txt     # Python dependencies
└── extracted_lccc_data.json  # Example output
```

## Development Notes

- The parser currently extracts plain text from TEI XML as a simplified representation
- Future enhancements may leverage more of TEI XML's structured metadata (authors, affiliations, citations, section headers)
- Error handling: The parser exits with helpful messages if GROBID is unavailable

## Key Technical Challenges

1. **LLM hallucinations** - LLM outputs may contain fabricated or missing fields
2. **Extraction inconsistency** - Results vary across documents and runs
3. **Scalability** - Need pipeline for large PDF collections on GPU cluster
4. **Quality assurance** - Automated evaluation and improvement mechanisms required

## Project Roadmap (3 Phases)

### Phase 1: Infrastructure Pipeline (Weeks 1-2) ← CURRENT FOCUS

**Goals:**
- Build reproducible PDF → JSON pipeline
- Batch execution on GPU cluster
- Logging and experiment tracking

**Deliverables:**
- Minimal end-to-end pipeline (PDF → text → chunking → LLM → JSON)
- Pipeline runs on GPU cluster
- Supports batch processing of multiple PDFs
- Logs inputs, prompts, and outputs for debugging
- Working pipeline + example outputs

**Status:**
- Basic GROBID parsing exists
- Need: chunking, cluster execution, logging, batch processing

### Phase 2: Verification Layer (Weeks 3-5) ← YOUR CORE RESPONSIBILITY

**Goals:**
- Schema validation
- Consistency checking across extractions
- LLM-based verification of outputs
- Automated quality assessment

**Critical Component:** This is the make-or-break feature that ensures reliability.

### Phase 3: Optimization (Weeks 6-9)

**Goals:**
- Prompt refinement based on verification results
- Error correction loops
- Model evaluation and tuning
- Continuous improvement based on logged data

## Future Enhancements (Post-Initial Phase)

- Multi-modal extraction: figures, charts, graphs, diagrams
- Database integration for querying extracted data
- Domain generalization (currently focused on polymer science)
- Advanced TEI XML metadata extraction (authors, affiliations, citations)
