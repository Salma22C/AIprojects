# 🧠 TalentCheck AI — Evaluator–Optimizer Resume Screening System

An AI-powered resume screening pipeline that uses a **self-correcting multi-agent loop** to improve reliability and reduce hallucinations in LLM-based candidate evaluation systems.

Instead of relying on a single model response, TalentCheck AI uses a **two-agent architecture**:

- **Optimizer Agent** → generates structured candidate evaluations
- **Evaluator Agent** → audits responses and rejects unsupported reasoning

---

# ⚙️ Core Idea

Many LLM-based resume screening systems fail because they:

- hallucinate skills
- infer experience not explicitly mentioned
- overestimate candidate seniority
- produce inconsistent evaluations

TalentCheck AI addresses this problem using a **verification and correction loop**:

> **Generate → Critique → Fix → Repeat**

The system continuously validates outputs before accepting the final evaluation.

---

# 🧠 System Architecture

## 1. Optimizer Agent

The Optimizer Agent generates a structured candidate scorecard based on:

- resume content
- job description alignment
- technical strengths
- relevant experience

### Example Output

```json
{
  "match_score": 78,
  "key_skills": ["Python", "NLP", "AWS"],
  "summary": "Strong NLP background with relevant cloud deployment experience."
}
```

---

## 2. Evaluator Agent

The Evaluator Agent acts as a strict auditor that verifies whether the Optimizer’s output is grounded in the actual resume data.

It checks for:

- hallucinated or unsupported skills
- timeline inconsistencies
- exaggerated experience claims
- seniority mismatches
- weak justification reasoning

### Validation Output

#### PASS

```json
{
  "status": "PASS"
}
```

#### FAIL

```json
{
  "status": "FAIL",
  "issues": [
    "AWS experience was inferred but not explicitly stated.",
    "Candidate seniority overstated based on timeline."
  ]
}
```

---

# 🔁 Self-Correction Loop

If the Evaluator returns `FAIL`, the feedback is sent back to the Optimizer for correction.

The loop repeats for up to **3 iterations**:

```text
Optimizer → Evaluator → Fix → Re-evaluate
```

Only outputs that pass validation are accepted as final results.

---

# 🚀 Features

- 📄 PDF resume parsing using `pypdf`
- 🤖 LLM-based structured candidate evaluation
- 🧪 Multi-agent validation architecture
- 🔁 Iterative self-correction loop
- 📊 JSON-based scoring system
- 🧠 Hallucination detection layer
- ⚡ Configurable LLM backend support

---

# 🧰 Tech Stack

- Python
- OpenAI SDK (via OpenRouter)
- Qwen / GPT / Gemma models
- pypdf
- Google Colab

---

# 📌 Example Final Output

```json
{
  "verified_scorecard": {
    "match_score": 65,
    "key_skills": ["Python", "AI", "AWS"],
    "summary": "Strong AI background but limited ML framework depth."
  },
  "status": "FAILED_AFTER_RETRIES",
  "audit": {
    "status": "FAIL",
    "issues": [
      "Timeline mismatch detected.",
      "Skill inference unsupported by resume."
    ]
  }
}
```

---

# 🧪 Project Goals

This project explores how **multi-agent validation systems** can improve trust and reliability in AI decision-making pipelines.

Key focus areas include:

- LLM reliability
- hallucination reduction
- AI evaluation systems
- autonomous verification loops
- resume intelligence systems

---

# 👨‍💻 Author

Built as part of an AI Engineering portfolio project focused on:

- RAG systems
- LLM applications
- AI evaluation pipelines
- real-world AI decision systems
