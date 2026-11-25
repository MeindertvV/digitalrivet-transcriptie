# Transcription in the cloud (a minimal copy of DigitalRivet's transcription step)

The only job here: the works of `works.tsv` (scanned Latin books, 16th–17th c.) through the steps of `cloud/run.py`
(transcription, failed pages again, OCR line check, contents, English translation), and push the results. The owner's rules:

- **Haiku only.** Every step runs `claude -p --model haiku` (`--via claude`, the subscription). Never Sonnet or
  Opus, never `--model`, never the API (`ANTHROPIC_API_KEY` is removed from the run). You do not read or transcribe
  pages yourself: the script does.
- Do not change the prompt or the code in `backend/`. If something breaks, stop and report it.
- The printed verse sign is a plain `v` (the prompt says so).

## Run

```bash
pip install -r requirements.txt
python cloud/run.py --shard 0 --of 1 --workers 4 --base "$PDF_BASE"
```

Run it in the background and keep it going; it pushes each finished work to the branch `results-<shard>`.
It stops by itself when the subscription's usage limit is reached; report how many works and pages were done.
Check first with `--limit 1` that a work comes through (book.json in out/<work>/, pages ok in its transcribe.log).
More shards (`--shard i --of N`) or more `--workers` only while pages stay ok and nothing hits a rate limit.

## OCR layers (a second, separate job)

`cloud/ocr.py` makes Tesseract OCR layers (no model, no credit) for the works of `ocr_works.tsv`: invisible text
only, Latin, with `cloud/tessdata/lat.traineddata`. It needs only `pip install pymupdf` (PyMuPDF has Tesseract
built in). Do not change `scan_dpi` or `ocr_page`: they are the same as on the server.

```bash
pip install pymupdf
python cloud/ocr.py --shard 0 --of 4 --base "$PDF_BASE"
```

Run it in the background and keep it going; it pushes each finished layer to the branch `ocr-<shard>`. Check first
with `--limit 1` that a work comes through (a line with pages, words and s/p.). `--workers` defaults to the number
of cores. Report how many works and pages were done, and the seconds per page.
