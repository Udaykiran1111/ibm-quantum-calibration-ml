# Runbook - first day

0. **Keep the two original ZIP files untouched** (copy them to another drive or cloud). They are the master evidence copy.

1. **Freeze the current public repo** so nothing is lost:
   ```bash
   git clone https://github.com/Udaykiran1111/ibm-quantum-calibration-ml.git
   cd ibm-quantum-calibration-ml
   git tag v0-before-cleanup
   git push origin v0-before-cleanup
   ```

2. **Put the clean layout on a branch**:
   ```bash
   git checkout -b clean-v1
   git rm -r -q --ignore-unmatch .
   ```
   Copy everything from the unzipped package into this folder (including the hidden
   `.github` folder and `.gitignore`), then:
   ```bash
   git add -A
   git commit -m "Clean merged layout: ML + hardware in one repository"
   git push -u origin clean-v1
   ```

3. **GitHub settings** (repository -> Settings):
   - Secrets and variables -> Actions -> New repository secret: `IBM_QUANTUM_TOKEN`, and `IBM_INSTANCE` (the CRN of your Open Plan instance).
   - Actions -> General -> Workflow permissions -> **Read and write permissions**.

4. **Merge `clean-v1` into `main`** (pull request, or `git checkout main && git merge clean-v1 && git push`).
   Scheduled workflows only run from the default branch.

5. **First snapshot now**: Actions tab -> "Daily collect and rank" -> Run workflow.
   Expected: a new commit "data: daily snapshot YYYY-MM-DD" with `data/daily/calibration/<date>.csv`,
   `couplers/<date>.csv` and `meta/<date>.json`. The ranking step says "Not enough history" on days 1-2; that is correct.

6. **Check it worked**: the calibration CSV has about 460-470 rows covering 3 backends, and the meta
   JSON shows `"ok": true` for each backend. If a step is red, copy the error text and send it to me.

From day 3 the workflow also writes `predictions/daily/<date>_model_b.csv` and `_model_c.csv`
(ranked using only earlier days) and commits them **before** any hardware run.
