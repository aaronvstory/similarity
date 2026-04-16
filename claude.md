# Claude / Anthropic Context (claude.md)

This file provides specific context for Claude.

### Project Origins
This project was initiated as an "Enterprise-Grade Local Face Similarity Application". The goal was to overcome the limitations of cloud-based KYC and facial similarity checks by building a robust, offline desktop application.

### Key Mathematical Decision (Cosine Distance Mapping)
ArcFace utilizes Cosine Distance. The threshold for ArcFace is officially `0.68`. Any distance <= 0.68 is considered the same person.
However, end-users requested an easy-to-understand 0-100% UI where an 80% represents the cutoff. 
Because of this, `engine.py` contains a dynamic mathematical curve that maps `distance 0.0 -> threshold 0.68` to `score 100.0% -> 80.0%`. 
Do **not** overwrite this math with a standard `(1 - distance) * 100` formula, as that will fail mathematically accurate matches.

### Directory Structure Requirements
- `main.py` serves as the sole entry point, passing control to `cli.py` or `gui.py`.
- Automated bash/batch scripts exist at the root level to circumvent python setup friction for non-technical users. 
