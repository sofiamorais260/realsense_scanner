# GitHub Setup Guide — realsense_scanner

Follow these steps **in order** in PowerShell or Git Bash, from inside your project folder.

---

## Prerequisites

- [Git for Windows](https://git-scm.com/download/win) installed
- A [GitHub](https://github.com) account

---

## Step 1 — Clean up the broken .git folder

A failed git init attempt left a broken `.git` folder in your project. Delete it first:

```powershell
# In PowerShell, from C:\Users\Sofia\Projects\realsense_scanner
Remove-Item -Recurse -Force .git
```

Or just delete the `.git` folder manually in File Explorer (make sure "Show hidden items" is on).

---

## Step 2 — Initialize the repository

```powershell
cd C:\Users\Sofia\Projects\realsense_scanner

git init
git branch -m main
git config user.name "Sofia Morais"
git config user.email "sofiagranadodemorais@gmail.com"
```

---

## Step 3 — Stage and verify your files

```powershell
git add .
git status
```

**Check the output carefully.** You should see your source code files (`src/`, `scripts/`, `main.py`, `requirements.txt`, `.gitignore`, etc.) but NOT:
- `.venv/`
- `scan_results/`
- `calibration_results/topography/`
- `codigo_joao_git/` or `joao25abril/`
- `__pycache__/`

If any of those appear, stop and check the `.gitignore`.

---

## Step 4 — Make the first commit

```powershell
git commit -m "Initial commit: 3D scanner platform for autofluorescence imaging"
```

---

## Step 5 — Create a private repository on GitHub

1. Go to [github.com/new](https://github.com/new)
2. Set **Repository name** to `realsense_scanner`
3. Set visibility to **Private**
4. **Do NOT** check "Add a README file" or "Add .gitignore" — you already have these
5. Click **Create repository**
6. Copy the HTTPS URL shown (it will look like `https://github.com/sofiamorais260/realsense_scanner.git`)

---

## Step 6 — Push to GitHub

```powershell
git remote add origin https://github.com/sofiamorais260/realsense_scanner.git
git push -u origin main
```

Git will ask for your GitHub username and password. Use a **personal access token** as the password (GitHub no longer accepts plain passwords):
- Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate new token
- Give it `repo` scope
- Copy it and paste it as your password when prompted

---

## Step 7 — Verify

Go to `https://github.com/sofiamorais260/realsense_scanner` in your browser. You should see all your code files there.

---

## What is and isn't on GitHub

| Included | Excluded |
|---|---|
| `src/` — all scanner source code | `scan_results/` — 12 GB of scan data |
| `scripts/` — analysis scripts | `calibration_results/topography/` — 151 MB of mesh outputs |
| `main.py`, `requirements.txt` | `.venv/` — Python environment |
| `calibration_results/machine_camera/` | `codigo_joao_git/`, `joao25abril/` — external dependency |
| `calibration_results/scan_space/` | `__pycache__/` — Python cache |
| `calibration_results/targets/` | |
| `docs/`, `settings.json`, `.gitignore` | |

**Scan data and large calibration outputs** should be kept backed up on your local machine or the Champalimaud network storage — they are too large for GitHub.

---

## External dependencies (not included in this repo)

### pyProbe and pyProbeAnalysis — Dr. João Lagarto

These libraries are developed and maintained by Dr. João Lagarto and live in his own GitHub repositories. They are **not** committed here.

To set them up locally, clone them from his GitHub into the project folder:

```powershell
cd C:\Users\Sofia\Projects\realsense_scanner
git clone https://github.com/joaolagarto/pyProbe.git          codigo_joao_git/pyProbe
git clone https://github.com/joaolagarto/pyProbeAnalysis.git  codigo_joao_git/pyProbeAnalysis
```

> **Note:** Confirm the exact repository URLs with Dr. Lagarto — the above are placeholders based on his GitHub account name.

Anyone cloning your repo will need to do this step manually before running the scanner.

---

## Adding supervisors as collaborators

Since the repo is private, your supervisors need to be invited to see the code:
- GitHub → your repo → Settings → Collaborators → Add people
- Add Dr. Lagarto's and Professor Hugo's GitHub usernames
