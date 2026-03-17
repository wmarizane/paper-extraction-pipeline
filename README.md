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