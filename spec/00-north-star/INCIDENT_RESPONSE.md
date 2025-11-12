# Incident Response Playbook

**Purpose:** Standard procedures for responding to build failures, quality violations, and protocol
deviations.

**Scope:** All roles, all loops, all artifact types

**Last Updated:** 2025-11-05

**Status:** Active (created in response to Adventure Bay Binder Breakdown)

---

## Incident Categories

### Category 1: Build Failures

**Symptoms:**

- EPUB/PDF generation fails
- Missing or incorrect assets in output
- Validation errors during export

**Severity:** High (blocks release)

### Category 2: Protocol Violations

**Symptoms:**

- Heuristic mappings used instead of manifest
- Cold SoT bypassed
- Missing SHA-256 hashes

**Severity:** Critical (corrupts determinism)

### Category 3: Quality Bar Failures

**Symptoms:**

- Gatekeeper rejects TU
- Dead references found
- Style inconsistencies

**Severity:** Medium (blocks Cold merge)

### Category 4: Asset Drift

**Symptoms:**

- "Latest" file ≠ "approved" file
- Stale images in build
- Filename confusion

**Severity:** High (corrupts content)

---

## General Response Protocol

### Step 1: STOP

**Immediately:**

- ❌ **Stop all builds**
- ❌ **Stop all asset updates**
- ❌ **Stop all file operations**

**Do NOT:**

- ❌ Attempt to "fix" with heuristics
- ❌ Guess which file is correct
- ❌ Rebuild without understanding root cause

### Step 2: FREEZE

**Preserve state:**

1. Snapshot current working directory: `tar -czf incident-$(date +%Y%m%d-%H%M%S).tar.gz .`
2. Record git status: `git status > incident-git-status.txt`
3. List all files with timestamps: `find . -type f -exec ls -lh {} \; > incident-file-list.txt`
4. Compute hashes for all assets: `find assets/ -type f -exec sha256sum {} \; > incident-hashes.txt`

### Step 3: ASSESS

**Determine what's wrong:**

| Question                            | Check                                                     |
| ----------------------------------- | --------------------------------------------------------- |
| Is Cold SoT intact?                 | Verify `cold/manifest.json` exists and validates          |
| Are all manifests present?          | Check `cold/book.json`, `cold/art_manifest.json`          |
| Do hashes match?                    | Compare `incident-hashes.txt` to `cold/art_manifest.json` |
| What was the last successful build? | Check `build.json` from previous output                   |
| What changed since then?            | `git diff` or `git log`                                   |

### Step 4: DIAGNOSE

**Identify root cause:**

Common root causes (from post-mortems):

1. **Missing hashes** — Assets approved without SHA-256
2. **Manifest drift** — Manifest out of sync with assets
3. **Environment reset** — Working directory changed
4. **Multiple build paths** — Mixed Pandoc + manual
5. **Heuristic fallback** — "Newest file wins" logic used
6. **Protocol bypass** — Cold SoT not used as source

### Step 5: REMEDIATE

**Fix the root cause:**

See incident-specific playbooks below.

### Step 6: VERIFY

**Confirm fix:**

1. Run `binder:verify` preflight
2. Check all hashes match manifest
3. Validate all schemas
4. Test build with strict mode

### Step 7: DOCUMENT

**Record findings:**

1. Create post-mortem in `docs/post_mortems/YYYY-MM-DD_<title>.md`
2. Update this playbook if new pattern identified
3. File improvement TUs for systemic fixes

---

## Incident-Specific Playbooks

### Playbook A: Build Failure (Missing Assets)

**Symptoms:** EPUB/PDF missing images; Kobo shows cover only

**Response:**

1. **STOP + FREEZE** (general protocol above)

2. **Check manifest:**

   ```bash
   # Does cold/art_manifest.json exist?
   cat cold/art_manifest.json

   # Does every anchor have an entry?
   jq '.assets | map(.anchor)' cold/art_manifest.json
   ```

3. **Check physical assets:**

   ```bash
   # Do all files exist?
   jq -r '.assets[].filename' cold/art_manifest.json | while read f; do
     if [ ! -f "assets/$f" ]; then
       echo "MISSING: assets/$f"
     fi
   done
   ```

4. **Check hashes:**

   ```bash
   # Do SHA-256 match?
   jq -r '.assets[] | "\(.filename) \(.sha256)"' cold/art_manifest.json | while read filename expected_hash; do
     actual_hash=$(sha256sum "assets/$filename" | cut -d' ' -f1)
     if [ "$actual_hash" != "$expected_hash" ]; then
       echo "HASH MISMATCH: $filename (expected: $expected_hash, got: $actual_hash)"
     fi
   done
   ```

5. **Remediation:**
   - If files missing → Restore from last approved snapshot
   - If hashes mismatch → Identify which file is correct (check approval time)
   - Update manifest with correct hashes
   - Re-run `binder:verify`

6. **Prevention:**
   - Ensure approval hook records SHA-256 immediately
   - Add preflight check before every build

---

### Playbook B: Protocol Violation (Heuristic Mapping)

**Symptoms:** Build used "newest file wins" or filename guessing

**Response:**

1. **STOP immediately** — This is a **critical** violation

2. **Identify deviation:**

   ```bash
   # Check build logs for keywords:
   grep -i "heuristic\|guessing\|newest\|fallback" build.log
   ```

3. **Freeze and snapshot** (general protocol)

4. **Reconstruct Cold SoT:**
   - Review approval history (git log, email, chat)
   - For each anchor, identify which asset was approved and when
   - Compute SHA-256 for approved assets
   - Create/update `cold/art_manifest.json` with authoritative mappings

5. **Verify reconstruction:**

   ```bash
   # Validate manifest schema
   qfspec-check-instance cold_art_manifest cold/art_manifest.json

   # Verify all files exist with correct hashes
   ./scripts/verify-cold-manifest.sh
   ```

6. **Rebuild from Cold:**
   - Delete all previous build outputs
   - Run `binder:build --strict --cold-only`
   - Verify output matches expectations

7. **Prevention:**
   - Update Gatekeeper prompt: **REJECT** builds without Cold manifest
   - Update Binder prompt: **FAIL** if any input not from Cold
   - Add `--strict` flag that disables all fallbacks

---

### Playbook C: Asset Drift (Stale Files)

**Symptoms:** Old version of image used; "latest" ≠ "approved"

**Response:**

1. **STOP + FREEZE**

2. **Identify versions:**

   ```bash
   # List all files matching anchor pattern
   ls -lt assets/anchor001__*

   # Compare to manifest
   jq '.assets[] | select(.anchor=="anchor001")' cold/art_manifest.json
   ```

3. **Determine approved version:**
   - Check `cold/art_manifest.json` for `approved_at` timestamp
   - Check `provenance.version` field
   - Cross-reference with approval records (git commits, messages)

4. **Remove stale files:**

   ```bash
   # Archive old versions (don't delete immediately)
   mkdir -p archive/$(date +%Y%m%d)
   mv assets/anchor001__plate__v1.png archive/$(date +%Y%m%d)/

   # Keep only approved version
   # (Verify hash matches manifest before deleting archive)
   ```

5. **Verify cleanup:**

   ```bash
   # Should be exactly one file per anchor
   jq -r '.assets[] | .anchor' cold/art_manifest.json | sort | uniq -d
   # (empty output = good)
   ```

6. **Prevention:**
   - Enforce deterministic filenames: `<anchor>__<role>__v<N>.png`
   - Approval hook should **rename** file, not create new
   - Add uniqueness constraint to manifest schema

---

### Playbook D: Environment Reset

**Symptoms:** Working directory changed; paths broken

**Response:**

1. **STOP + FREEZE**

2. **Restore canonical paths:**

   ```bash
   # Expected structure:
   project/
   ├── cold/
   │   ├── manifest.json
   │   ├── book.json
   │   └── art_manifest.json
   ├── assets/
   │   └── *.png
   └── sections/
       └── *.md
   ```

3. **Verify Cold integrity:**

   ```bash
   # All Cold files should validate
   qfspec-check-instance cold_manifest cold/manifest.json
   qfspec-check-instance cold_book cold/book.json
   qfspec-check-instance cold_art_manifest cold/art_manifest.json
   ```

4. **Re-establish relative paths:**
   - Binder should always use paths relative to project root
   - Never use absolute paths
   - Pandoc `--resource-path` should be `.:assets`

5. **Rebuild:**

   ```bash
   # From project root
   cd /path/to/project
   binder:build --cold-only
   ```

6. **Prevention:**
   - Pin working directory in `build.json`
   - Add environment check to `binder:verify`
   - Document canonical structure in README

---

### Playbook E: Manifest Drift

**Symptoms:** Manifest out of sync with assets; missing entries

**Response:**

1. **STOP + FREEZE**

2. **Audit:**

   ```bash
   # Files in assets/ not in manifest:
   comm -23 \
     <(ls assets/*.png | sort) \
     <(jq -r '.assets[].filename' cold/art_manifest.json | sed 's|^|assets/|' | sort)

   # Entries in manifest not in assets/:
   comm -13 \
     <(ls assets/*.png | sort) \
     <(jq -r '.assets[].filename' cold/art_manifest.json | sed 's|^|assets/|' | sort)
   ```

3. **Reconcile:**
   - For untracked files: Determine if they should be in manifest
   - For missing files: Restore from backup or mark as missing
   - Update `cold/art_manifest.json` to match reality

4. **Validation:**

   ```bash
   # Every section in book.json should have art in art_manifest.json
   join -v1 \
     <(jq -r '.sections[].anchor' cold/book.json | sort) \
     <(jq -r '.assets[].anchor' cold/art_manifest.json | sort)
   # (empty output = good)
   ```

5. **Prevention:**
   - Approval hook must update manifest atomically
   - Add `manifest:sync` command to reconcile
   - Gatekeeper checks manifest completeness before build

---

## Escalation

### When to Escalate to Showrunner

Escalate if:

- Root cause unclear after 30 minutes
- Multiple systems affected
- Data loss suspected
- Protocol violation intentional/repeated

### When to Escalate to Human

Escalate if:

- Showrunner cannot resolve
- Ethical/policy questions (e.g., child likeness)
- Spec ambiguity discovered
- Systemic fix requires ADR

---

## Post-Incident Review

After resolving incident:

### 1. Post-Mortem

Create `docs/post_mortems/YYYY-MM-DD_<title>.md` with:

- Executive summary
- Timeline
- Root cause analysis
- What went well / What went wrong
- Action items

### 2. Specification Updates

If spec gap identified:

- File ADR if architectural change needed
- Create TU for documentation updates
- Update relevant layers (0-5)

### 3. Automation

Add checks to prevent recurrence:

- New schema constraints
- New validation rules
- New preflight checks
- New CI tests

### 4. Communication

Inform stakeholders:

- User (if external)
- Team (if collaborative)
- Repository maintainers (via PR)

---

## Tooling

### Required Scripts

```bash
# Verify Cold manifest completeness and hashes
./scripts/verify-cold-manifest.sh

# Sync manifest with assets (reconcile drift)
./scripts/manifest-sync.sh

# Preflight check before build
./scripts/binder-preflight.sh

# Snapshot project state
./scripts/snapshot-project.sh
```

(Scripts to be created in `spec-tools/`)

---

## Related Documents

- **Post-Mortems:** `docs/post_mortems/`
- **Quality Bars:** `00-north-star/QUALITY_BARS.md`
- **Traceability:** `00-north-star/TRACEABILITY.md`
- **Cold SoT Format:** `00-north-star/COLD_SOT_FORMAT.md` (to be created)
- **Binder Charter:** `01-roles/charters/book_binder.md`

---

## Revision History

| Date       | Version | Changes                                       |
| ---------- | ------- | --------------------------------------------- |
| 2025-11-05 | 1.0.0   | Initial version (post Adventure Bay incident) |

---

**Remember: When in doubt, STOP and FREEZE. Heuristics destroy determinism.**
