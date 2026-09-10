# OSF Preregistration Archival Bundle

**Purpose:** upload `docs/preregistration.md` (and companions) to OSF
(or AsPredicted) as third-party-timestamped archival of the
preregistered hypotheses, decision rules, and their committed-before-run
provenance.

## Files to upload

| File | Role |
|---|---|
| `preregistration.md` | The full preregistration (H1–H19 + R1–R5 with decision rules, machinery gates, and recorded outcomes) |
| `physSAE_reconciliation.md` | The concurrent-work reconciliation (con-found decomposition, level ladder, registered R1–R5 rationale) |
| `experiment_matrix.md` | Claim → hypothesis → command → artifact → figure map |
| `fresh_campaign_record.md` | The drift audit + full-rerun record (evidence of honest reporting) |
| `osf_manifest.json` | Machine-readable manifest with git SHAs binding each preregistration section to the commit that preceded the run |

## Provenance chain (the point of the upload)

Each stage's hypotheses + decision rules were committed to the public
repository BEFORE the stage ran. The chain to cite on OSF:

| Stage | Hypotheses | Registered at (commit) | Outcome recorded at (commit) |
|---|---|---|---|
| 5 (SAE battery) | H5 | pre-campaign baseline | 816bd2e line |
| 8 (PCA battery) | H8 | 437cced era | e9cbf4a (fresh rerun) |
| 9 (causal abstraction) | H6 | 437cced era | e9cbf4a |
| 14 (operator causal) | H14a/H14b | **f6eedf9** | **ab54b82** |

The decisive example: stage 14's H14a/H14b decision rules landed in
commit `f6eedf9` ("outcomes pending"), and the outcome — including the
rule firing AGAINST the interesting direction and the recorded design
flaw in condition (iii) — landed in `ab54b82`. GitHub timestamps make
this independently checkable.

## Upload steps (OSF)

1. Create an OSF project: title
   "When Sparse Features Become Mechanistic — Preregistration & Campaign Records".
2. Add a Wiki or README stating: preregistration is in-repo
   (git-committed before each run); this upload is third-party
   timestamping of the same documents.
3. Upload the four files above (from this directory).
4. Optional: register a DOI for the project; add it to the repo README
   and `CITATION.cff` after.
5. Set the license CC BY 4.0 to match the repo.

## After upload

- Add the OSF DOI/URL to: README.md (Open Science section),
  `docs/preregistration.md` header, and `PROJECT_STATUS.md`
  (move "OSF archival" from not-started to done).
- Note: outcomes are already recorded inside `preregistration.md` as the
  rules fired; OSF archival does not change any recorded claim.
