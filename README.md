# Minute Writer — LangChain Single Agent Project

A LangChain single-agent project that turns raw, messy meeting notes into a clean, structured summary plus a numbered list of action items. Built on **LangChain + OpenAI** following the two-tool single-agent pattern.

## What It Does

You paste in raw meeting notes — bullet points, stream-of-consciousness, typos and all — and the agent returns:

1. A **structured meeting summary** with sections for *Attendees*, *Key Discussions*, and *Decisions Made*
2. A **numbered list of action items** in the format `[Owner] - [Task] - [Deadline]`

Owners default to *"Unassigned"* and deadlines default to *"TBD"* when not stated, so nothing is hallucinated.

## What You'll Learn

- How LangChain works (LLMs, prompts, tools, agents)
- How to build a two-tool single agent
- How to write tool docstrings that the agent reads to decide when to call each tool
- How `PromptTemplate` shapes LLM output
- How an agent's tool-calling loop works (think → act → observe → repeat)

## How It Works

```
                Raw meeting notes
                       |
                       v
   [Agent thinks: "I need to summarize first"]
                       |
                       v
   [Tool: summarize_meeting_notes]  -->  structured summary
                       |
                       v
   [Agent thinks: "Now extract action items"]
                       |
                       v
   [Tool: extract_action_items]  -->  numbered action items list
                       |
                       v
   Final output: BOTH the summary AND the action items
```

Both tools take the **raw notes** (not tool 1's output), so action-item extraction does not miss items the summary may have dropped.

## Prerequisites

- Python 3.10 or higher
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys))

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate
  ```
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

Copy the example env file and add your real key:

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your actual OpenAI API key:

```
OPENAI_API_KEY=sk-your-actual-key-here
```

## Run

```bash
python minute_writer.py
```

You'll see an interactive prompt:

```
============================================================
  MINUTE WRITER AGENT
  Powered by LangChain + OpenAI
============================================================

Paste your raw meeting notes and the agent will return
a structured summary plus a numbered list of action items.

Type 'quit' to exit.

Your meeting notes (or 'quit'):
```

Paste your meeting notes (single-line or multi-line) and the agent will return both a summary and an action items list. You'll also see detailed logs showing the agent's reasoning and tool calls.

## Example

**Input:**
```
Q4 planning, Tuesday Nov 12. Sarah, Mike, Priya, James joined.
Talked about Q1 roadmap. Sarah wants feature X by end Jan. Mike said engineering tight.
Decision: cut feature Y, ship X early Feb.
Mike to send revised timeline by Friday.
Priya talking to design about mockups.
James will draft customer comms.
```

**Output:**
```
## Meeting Summary

**Meeting:** Q4 planning
**Date:** Tuesday Nov 12

### Attendees
- Sarah
- Mike
- Priya
- James

### Key Discussions
- Q1 roadmap
- Feature X timeline (Sarah wants end of January)
- Engineering capacity constraints (raised by Mike)

### Decisions Made
- Cut feature Y from Q1
- Ship feature X in early February

## Action Items

1. Mike - Send revised timeline - Friday
2. Priya - Talk to design team about feature X mockups - TBD
3. James - Draft customer comms - TBD
```

## Project Structure

```
.
├── minute_writer.py     # Main agent code (fully commented)
├── requirements.txt     # Python dependencies
├── .env.example         # API key template
├── .gitignore           # Keeps secrets and venv out of git
└── README.md            # This file
```

## Tech Stack

- [LangChain](https://python.langchain.com/) — Framework for building LLM applications
- [OpenAI GPT-4.1-mini](https://platform.openai.com/) — The LLM powering the agent (temperature `0.3` for fidelity-focused summarization)
- [python-dotenv](https://pypi.org/project/python-dotenv/) — Environment variable management

## Design Notes

- **Two tools, both take raw notes.** The summarizer produces a structured summary; the action-item extractor scans the original notes directly. This prevents lost action items when the summary is concise.
- **Three layered prompts.** The system prompt orchestrates tool order; each tool's internal `PromptTemplate` handles its specific transformation.
- **Hallucination defenses.** Each tool has explicit *"do not invent"* rules and defined defaults (*"Not specified"*, *"No action items identified."*, *"Unassigned"*, *"TBD"*) for missing data.
- **Scope refusal.** The system prompt instructs the agent to refuse non-meeting-note input rather than fabricating content.
