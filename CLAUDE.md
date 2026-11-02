# CLAUDE.md

@AGENTS.md

## Claude Code notes

- Project guidance shared with other agents is in `AGENTS.md` (imported above). Edit it
  there, not here, so the two files don't drift apart.
- Start long `asrsbench run` / `prepare` jobs in the background and watch their output,
  rather than blocking a turn on them.
- Ask the user before deleting anything under `cache/` or `data/processed/`: rebuilding
  means re-downloading the dataset and re-encoding the corpus.
- The `Research/` PDFs (course guidelines, proposal) set the requirements for the
  report. Read the relevant one before drafting or restructuring report sections.
