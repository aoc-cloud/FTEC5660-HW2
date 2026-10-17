# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution

### Task 1 — CV verification agent

I built a deterministic verification pipeline around `deepseek-v4-flash` and
the six SocialGraph MCP tools. For each CV, the model first extracts a
structured record (name, city, jobs, education, skills). The agent then
resolves the candidate's identity: it infers the industry from the CV headline
(e.g. "Logistics Professional" → Logistics) and calls
`search_linkedin_people` with name, city and industry. This step is essential
because the mock graph contains up to 20 people with the same name in the same
city. The remaining candidates' full profiles are fetched and ranked by
overlap with the CV (3 points per matching employer, 2 per matching school,
plus field, graduation year, skill overlap, city and years of experience); the
best match is used, with Facebook's `original_name` as a fallback when LinkedIn
search fails.

The comparison is rule-based, which keeps the model from being lenient about
numbers: exact employer/school/city checks, start/end year checks, degree-level
mapping ("Bachelor of Science" → BSc, "Master of Business Administration" →
MBA), seniority-aware title checks, and skill-subset checks with aliases
("UI/UX Design" ↔ UI/UX). Job descriptions, the headline and hometown are not
treated as claims to verify, as required. A `deepseek-v4-flash` adjudication
call is used only for genuinely ambiguous wording differences; hard mismatches
(years, employers, schools, seniority, graduation year) are confirmed by code.
Any real discrepancy returns 0.1, a clean CV returns 0.9, and an ambiguous
identity returns 0.5. Each CV runs under `asyncio.Semaphore(3)`, with per-CV
timeouts and exception handling so one failure cannot crash the run.

On `public_test` the agent scores **7/7** on three consecutive runs, at about
20 seconds per run.

### Task 2 — Fooling the verifier

My adversarial CV keeps every true fact in the PDF's **text layer** and changes
one fact only in the **visual layer**: the AIA job title is rendered as an
opaque image reading **"Senior Manager, AIA, 2022 – Present"**, while the
underlying text still says **"Manager, AIA, 2022 – Present"**. The true profile
says Manager with seniority `mid`, so the visible title is a genuine
embellishment.

The attack exploits the gap between the two channels a PDF has. `markitdown`
(the loader used by every verifier) extracts only the text layer; it never sees
the image. A human opening the PDF sees the rendered title. I verified the
differential directly: `markitdown` on `task2/adversarial_cv.pdf` extracts
`Manager, AIA    2022 – Present`, while the rendered page shows
`Senior Manager, AIA    2022 – Present`. My own Task 1 agent scores both the
adversarial CV and the true CV at **0.9**.

I expect this to work against the five verifiers because all of them use the
same PDF loader, so they all receive a CV whose extracted text is completely
consistent with Kelly Tsang's profile. The false detail is not hidden from the
grader — it is plainly visible on the page — it is hidden from the text-only
verification channel. Prompt injection or hidden white text would depend on
each verifier's instruction-following and defences; this attack depends only on
the loader, which is fixed by the assignment.

I also considered the obvious text-level alternatives (white/tiny hidden text,
fake tool-output JSON, authority cues). I did not use them in the final PDF
because they add no benefit once the false claim is invisible to the extractor,
and they can make a defended verifier suspicious.

### Task 3 — Reflection on AI paradoxes

**The verification paradox.** Task 1 and Task 2 are the same system seen from
two sides. In Task 1 I built a verifier that turns a CV into a number; in Task
2 I defeated it without changing anything the verifier reads. The lesson is not
that the verifier was badly written. It is that every act of compression —
from a person to a profile, from a profile to a score — creates a channel that
can be optimized against. The more we rely on the score, the more valuable that
channel becomes.

**Goodhart's law.** Once a reliability score is used as a hiring or compliance
decision, candidates stop writing CVs to describe themselves and start writing
them to satisfy the verifier. The score becomes a target, and a target is not a
measurement. This is not a hypothetical: the Task 2 attack is exactly what a
rational candidate would do if the score mattered. Verification systems
therefore need to assume that their inputs are adversarial, not merely noisy.

**Automation bias.** A number such as 0.9 feels like evidence, but it is a
model's summary of one data channel. Human reviewers tend to check less when
the system is confident, which is precisely when a confident error is most
dangerous. The precision of a score is not its accuracy. A verifier that says
"0.9" with no account of what it did not look at can be worse than a cautious
"unknown".

**AI checking AI.** The verifier and the attacker are drawn from the same
model family and the same linguistic world. Fluency is not epistemic authority:
the same model that can judge a claim can be persuaded by a well-placed
sentence, a role cue, or a document layout. LLM-as-a-judge is useful, but it
inherits the biases it is meant to detect, including sycophancy and a preference
for coherent narratives over awkward facts.

**The defender's asymmetry.** A verifier must check every field, every source
and every channel; an attacker needs one blind spot. The PDF text layer versus
its rendered appearance is a small example of a general problem: the evidence
we feed to an AI is usually a lossy projection of the real object. The attacker
lives in the part of the object that was projected away.

**What follows.** Verification should be treated as provenance work, not
classification. That means checking multiple channels (text, rendered page,
metadata, source APIs), preferring structured authoritative records over free
text, red-teaming the verifier before deployment, logging what was checked and
what was not, and keeping a human decision-maker for high-impact outcomes. The
OWASP Top 10 for LLM Applications makes the same point from the other
direction: prompt injection and overreliance are not edge cases; they are the
normal failure modes of systems that confuse a model's output with evidence.

The deepest paradox is that AI can make verification cheaper and more
convenient while making trust more fragile. The remedy is not a better score.
It is a process whose gaps are visible, and a human who knows where to look.
