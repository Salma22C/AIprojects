# AI + HITL LinkedIn Posting Workflow

A practice project demonstrating a **Human-in-the-Loop (HITL) AI workflow** using n8n. The workflow allows AI-generated LinkedIn posts to be reviewed by a human via Telegram before publishing to LinkedIn.

---

## 🚀 Overview

This project automates LinkedIn posting while keeping **human oversight** in the loop. It ensures that posts maintain **credibility, brand voice, and context**.  

**Workflow Summary:**
1. A chat message triggers the workflow.
2. The message topic is sent to the AI agent for content generation.
3. The generated post is sent to a **Human-in-the-Loop review via Telegram**.
4. The approved post is sent to an **HTTP Request node** to fetch the LinkedIn ID.
5. The post is published on LinkedIn.

---

## ⚙️ Workflow Steps

1. **Trigger:** Chat message initiates the workflow.
2. **AI Agent:** Generates draft content based on the topic.
3. **HITL Review:**  
   - Telegram node sends the draft to a human reviewer.  
   - Reviewer can approve, edit, or reject.
4. **LinkedIn Posting:**  
   - HTTP Request node fetches LinkedIn ID.  
   - LinkedIn node publishes the approved post.

---

## 🛠️ Tools & Integrations

- **n8n** — workflow automation  
- **OpenAI API** — AI content generation  
- **Telegram** — human-in-the-loop review  
- **HTTP Request** — fetch LinkedIn ID  
- **LinkedIn API** — post publication  

---

## 💡 Notes

- This is a **practice project**, designed to demonstrate HITL principles and AI automation in real-world workflows.  
- Human review ensures quality and reduces errors before posting.


