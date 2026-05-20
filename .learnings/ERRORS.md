# Errors

Command failures and integration errors.

---

## [ERR-20260506-001] powershell-heredoc

**Logged**: 2026-05-06T11:55:00+08:00
**Priority**: low
**Context**: Used Bash-style `python - <<'PY'` redirection in a PowerShell shell.
**Impact**: Verification command failed before running Python; no repo files were affected.
**Fix**: Use PowerShell here-string: `@' ... '@ | python -`.

## [ERR-20260506-002] local-config-bom

**Logged**: 2026-05-06T11:57:00+08:00
**Priority**: medium
**Context**: PowerShell-created `local_config.json` used UTF-8 with BOM, while Python opened it as plain UTF-8.
**Impact**: Runtime config loader returned an empty config.
**Fix**: Read local runtime config using `encoding="utf-8-sig"`.

## [ERR-20260506-003] powershell-search-and-python-invocation

**Logged**: 2026-05-06T12:20:00+08:00
**Priority**: low
**Context**: `rg.exe` was blocked with Access denied and Bash-style `python - <<'PY'` was used in PowerShell.
**Impact**: Inspection commands failed before reading application code; no source files were changed.
**Fix**: Use PowerShell-native `Select-String`/`Get-ChildItem` and pipe a here-string into Python.

## [ERR-20260508-001] patch-context-mojibake

**Logged**: 2026-05-08T10:39:00+08:00
**Priority**: low
**Context**: A broad `apply_patch` failed because Chinese comment context in `src/processor.py` did not match reliably in the terminal view.
**Impact**: No source files were changed by the failed patch.
**Fix**: Use smaller ASCII-only context anchors for patching files that contain mojibake-prone comments.

## [ERR-20260508-002] streamlit-process-filter-self-kill

**Logged**: 2026-05-08T18:00:00+08:00
**Priority**: medium
**Context**: A PowerShell cleanup command matched `*streamlit*app.py*8501*` against the current PowerShell command line and killed the tool process while trying to stop old Streamlit services.
**Impact**: The cleanup command aborted before restart verification; source files were not affected.
**Fix**: When stopping Streamlit, filter to Python processes only, e.g. `Where-Object { $_.Name -match 'python' -and $_.CommandLine -like '*streamlit*' -and $_.CommandLine -like '*app.py*' -and $_.CommandLine -like '*8501*' }`.

## [ERR-20260508-003] playwright-cli-chrome-spawn-unknown

**Logged**: 2026-05-08T18:05:00+08:00
**Priority**: low
**Context**: `npx --package @playwright/cli playwright-cli open http://localhost:8501` failed on Windows with Chrome launch error `spawn UNKNOWN`.
**Impact**: CLI browser automation could not start; Streamlit service was still reachable by HTTP.
**Fix**: Fall back to Python Playwright in the project environment for browser smoke tests.

## [ERR-20260508-004] powershell-python-stdin-unicode-path

**Logged**: 2026-05-08T18:10:00+08:00
**Priority**: medium
**Context**: A Python script piped from PowerShell here-string received a Chinese video path as `????`, causing `Input file does not exist` even though the file existed.
**Impact**: Backend E2E test failed before media processing.
**Fix**: For automated tests, copy/link media with non-ASCII paths to an ASCII path under `D:\video_clip` before passing it to Python/FFmpeg.

## [ERR-20260509-001] pytest-not-installed

**Logged**: 2026-05-09T10:31:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
`venv_gpu` does not have `pytest` installed.

### Error
```text
D:\video_clip\SongFormer_install\venv_gpu\Scripts\python.exe: No module named pytest
```

### Context
- Attempted to run `python -m pytest -q tests\test_segment_postprocess.py`.
- Switched to the existing stdlib test runner: `python -m unittest tests.test_segment_postprocess -v`.

### Suggested Fix
Use `unittest` for current project tests unless `pytest` is explicitly added to the environment.

### Metadata
- Reproducible: yes
- Related Files: tests/test_segment_postprocess.py

### Resolution
- **Resolved**: 2026-05-09T10:31:00+08:00
- **Notes**: Used `unittest` instead of installing dependencies.

## [ERR-20260518-001] powershell-empty-pipe-element

**Logged**: 2026-05-18T16:45:00+08:00
**Priority**: low
**Context**: A PowerShell one-liner attempted to emit objects from a `foreach` block and pipe directly after the block, which produced `An empty pipe element is not allowed`.
**Impact**: Directory size inspection failed; no files were changed.
**Fix**: Collect rows into an array first, then pipe the array to `Format-Table`.

## [ERR-20260518-002] powershell-null-raw-content

**Logged**: 2026-05-18T16:50:00+08:00
**Priority**: low
**Context**: `Get-Content -Raw` returned `$null` for an empty legacy file, then `.Replace()` was called on the null value.
**Impact**: Bulk path replacement stopped early; no source files were corrupted.
**Fix**: Skip zero-length files or coerce null raw content to an empty string before replacement.

## [ERR-20260520-001] start-service-timeout-but-service-listening

**Logged**: 2026-05-20T12:59:00+08:00
**Priority**: low
**Context**: `tools/start_service.ps1 -Port 8501` timed out in the tool after 124s, but port inspection showed `8501` listening and `Invoke-WebRequest http://localhost:8501` returned `200`.
**Impact**: The command looked failed from the runner, while the Streamlit app was actually available.
**Fix**: After a start-script timeout, verify port/process/HTTP status before retrying or killing processes.
