# CareMate — Setup Guide
## Getting real data into the database

This guide walks you through everything step by step.
No technical knowledge assumed.

---

## What you'll need

- Your laptop (Mac or Windows)
- About 20 minutes to set things up
- The STG PDF (you already have this)
- A credit card for Railway (free tier, no charge for what we're doing)

---

## Step 1 — Install Python

Python is the programming language the backend is written in.

**Mac:**
1. Go to https://www.python.org/downloads/
2. Click the big yellow "Download Python 3.12.x" button
3. Open the downloaded file and follow the installer
4. When done, open Terminal (press Cmd+Space, type "Terminal", press Enter)
5. Type: `python3 --version` and press Enter
6. You should see something like: `Python 3.12.3`

**Windows:**
1. Go to https://www.python.org/downloads/
2. Click the big yellow "Download Python 3.12.x" button
3. Open the downloaded file
4. **IMPORTANT:** Tick "Add Python to PATH" before clicking Install
5. Follow the installer
6. When done, open Command Prompt (press Windows key, type "cmd", press Enter)
7. Type: `python --version` and press Enter
8. You should see something like: `Python 3.12.3`

---

## Step 2 — Get your Anthropic API key

This lets the code talk to Claude.

1. Go to https://console.anthropic.com/settings/keys
2. Sign in (or create an account)
3. Click "Create Key"
4. Name it "CareMate"
5. Copy the key — it starts with `sk-ant-...`
6. **Save it somewhere safe** — you only see it once

---

## Step 3 — Set up the database on Railway

Railway gives you a free PostgreSQL database in the cloud.

1. Go to https://railway.app
2. Click "Start a New Project"
3. Sign in with GitHub (create a free GitHub account first if needed)
4. Click "Add a Service"
5. Choose "Database" → "PostgreSQL"
6. Wait about 30 seconds for it to provision
7. Click on the PostgreSQL service that appeared
8. Click the "Connect" tab
9. Copy the "Database URL" — it looks like:
   `postgresql://postgres:abc123@host.railway.app:5432/railway`
10. Save this — you'll need it in the next step

---

## Step 4 — Set up the project folder

1. Download the caremate-backend folder (from the files shared above)
2. Put it somewhere easy to find, like your Desktop
3. Inside the folder, find the file called `.env.example`
4. Make a copy of it and rename the copy to `.env` (just `.env`, no "example")
5. Open `.env` with any text editor (Notepad on Windows, TextEdit on Mac)
6. Replace `sk-ant-your-key-here` with your Anthropic API key from Step 2
7. Replace `postgresql://your-connection-string-here` with your Railway URL from Step 3
8. Save the file

It should look like this when done:
```
ANTHROPIC_API_KEY=sk-ant-api03-abc123...
DATABASE_URL=postgresql://postgres:xyz@host.railway.app:5432/railway
```

---

## Step 5 — Put the STG PDF in the project folder

Copy the STG PDF file into the `caremate-backend` folder.

The file is called:
`Primary-Healthcare-Standard-Treatment-Guidelines-and-Essential-Medicines-List-8th-Edition-2024-Updated-December-2025__1_.pdf`

---

## Step 6 — Open a terminal in the project folder

**Mac:**
1. Open Terminal (Cmd+Space → "Terminal")
2. Type: `cd ` (with a space after cd)
3. Drag the caremate-backend folder into the Terminal window
4. Press Enter

**Windows:**
1. Open the caremate-backend folder in File Explorer
2. Click in the address bar at the top
3. Type `cmd` and press Enter
4. A Command Prompt opens in the right folder

---

## Step 7 — Run the setup check

Type this and press Enter:

**Mac:**
```
python3 setup_and_test.py
```

**Windows:**
```
python setup_and_test.py
```

You should see green checkmarks for everything:
```
✅ Python 3.12.3
✅ All packages installed
✅ Anthropic API key: sk-ant-api03...
✅ Database URL: postgresql://postgres...
✅ Anthropic API working: OK
✅ Database connected, pgvector enabled
✅ PDF found: Primary-Healthcare...pdf
```

If anything shows ❌, fix that issue and run again.

---

## Step 8 — Run test ingestion (5 conditions)

Once everything is green, run this:

**Mac:**
```
python3 ingestion/pipeline.py --test
```

**Windows:**
```
python ingestion/pipeline.py --test
```

This processes 5 conditions from the STG and saves them to your database.
It takes about 2-3 minutes.

You'll see output like:
```
STEP 1: Segmenting PDF...
  Found 444 conditions

STEP 3: Extracting 5 conditions...
  [1/5] 1.2 CANDIDIASIS, ORAL (THRUSH)
    ✅ Saved (id=1, features=6, meds=1)
  [2/5] 2.9.1 DIARRHOEA, ACUTE IN CHILDREN
    ✅ Saved (id=2, features=9, meds=3)
  ...

PIPELINE COMPLETE
✅ Extracted: 5/5 conditions
```

---

## Step 9 — Run full ingestion (all 444 conditions)

If the test worked, run the full ingestion:

**Mac:**
```
python3 ingestion/pipeline.py
```

**Windows:**
```
python ingestion/pipeline.py
```

This takes about 45-60 minutes (444 conditions × ~6 seconds each).
You can leave it running and check back.

If it stops for any reason, run with `--resume` to pick up where it left off:
```
python3 ingestion/pipeline.py --resume
```

---

## Step 10 — Review the quality report

When done, open `ingestion_quality_report_summary.txt` in any text editor.

It will show you which conditions need your clinical review — things like:
```
⚠️  4.7.1 HYPERTENSION IN ADULTS
    Ambiguity score: 0.7
    Notes: Multiple treatment algorithms — first vs second line unclear
    Features: 8, Medicines: 12
```

For each flagged condition, check it against the STG PDF and confirm the extraction is correct.

---

## If something goes wrong

Just send me the error message and I'll fix it.
Everything is designed to be restartable — nothing gets corrupted.

---

## What's in the database when done

After full ingestion you'll have:
- 444 conditions with their clinical features
- ~2,000+ clinical relationships (symptom → condition links)
- ~150+ medicines with dosing information
- Knowledge chunks ready for vector search
- A quality report telling you exactly what needs human review
