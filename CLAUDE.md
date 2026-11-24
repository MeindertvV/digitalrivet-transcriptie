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
