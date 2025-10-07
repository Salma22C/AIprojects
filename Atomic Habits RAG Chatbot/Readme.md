# 💬 Atomic Habits RAG Chatbot 

This project is a **Retrieval-Augmented Generation (RAG)** chatbot built entirely in **n8n**, inspired by James Clear’s *Atomic Habits*.

It helps users build and sustain good habits using the core principles of the book — turning theory into step-by-step, actionable coaching.

---

## 🚀 Overview

The chatbot connects:
- 🗂️ **Google Drive** → Watches your `Atomic Habits.pdf`
- 🧠 **OpenAI Embeddings** → Turns text into vector representations
- 🌲 **Pinecone** → Stores and retrieves knowledge contextually
- 🤖 **AI Agent (GPT-4.1-mini)** → Gives personalized habit advice in real time

---

## ⚙️ Architecture

```
📘 Atomic Habits.pdf
↓
🔔 Google Drive Trigger — detects file updates
↓
📥 Download → Split Text → Create Embeddings
↓
🌲 Pinecone Vector Store (insert mode) — indexes document
↓
💬 Chat Trigger — user sends a question
↓
🧠 AI Agent (with Pinecone retrieval) — contextual response
↓
🤖 OpenAI Chat Model — crafts final personalized message
↓
📤 Output — sent back to user
```

---

## 🧩 Key Features

✅ **Automatic Document Sync**
- Watches for updates in Google Drive and re-indexes them in Pinecone automatically.

✅ **Context-Aware Chatbot**
- Uses RAG (Retrieval-Augmented Generation) to answer based on Atomic Habits concepts.

✅ **Personalized Habit Plans**
- Provides practical, small, consistent actions (easy, attractive, rewarding).

✅ **No Code**
- Entire pipeline built visually inside n8n.

---

## 🧠 Example Interaction

**User:**  
> I want to build a gym habit.

**AI Coach:**  
> - Lay out your gym clothes the night before to reduce friction.  
> - Place your water bottle somewhere visible as a cue.  
> - Start with short 10-minute sessions.  
> - Stack it after an existing habit, e.g. “after brushing teeth, I put on gym clothes.”  
> - Reward yourself afterward to reinforce it.  

*(All principles grounded in “Atomic Habits.”)*

---


---

## 🧩 Workflow Components

| Section | Description |
|----------|--------------|
| 📂 **Document Indexing Flow** | Watches for changes in the Atomic Habits PDF, splits the document into chunks, generates embeddings, and updates Pinecone. |
| 💬 **Chatbot Flow (RAG Retrieval)** | Handles chat messages, retrieves relevant context from Pinecone, and generates personalized habit advice using OpenAI. |
| ⚙️ **Core Integrations** | Connects Google Drive, Pinecone, and OpenAI seamlessly in n8n. |

---

## 🧠 Example Interaction

**User:**  
> I want to build a gym habit.

**AI Coach:**  
> - Lay out your gym clothes the night before to reduce friction.  
> - Place your water bottle somewhere visible as a cue.  
> - Start with short 10-minute sessions.  
> - Stack it after an existing habit, e.g. “after brushing teeth, I put on gym clothes.”  
> - Reward yourself afterward to reinforce it.  

*(All principles grounded in “Atomic Habits.”)*

---



## 🧰 Requirements

- [n8n](https://n8n.io)
- [OpenAI API Key](https://platform.openai.com)
- [Pinecone Account](https://www.pinecone.io)
- Google Drive connected to n8n
- `Atomic Habits.pdf` (your reference document)

---

## ⚙️ Setup Instructions

1. Import the provided workflow JSON into n8n.
2. Connect your credentials:
   - Google Drive OAuth2
   - Pinecone API
   - OpenAI API
3. Activate the workflow (green toggle ON).
4. Upload `Atomic Habits.pdf` to your linked Google Drive folder.
5. Use the Chat Trigger URL or n8n’s chat interface to start chatting.

---

## 💡 Future Improvements

- 🧵 Add memory (conversation context)
- 🧩 Allow multiple documents or topics
- 🗣️ Connect to Telegram or Notion for front-end chat
- 📊 Log interactions for habit analytics

---

## 🧑‍💻 Author

Built by **Salma Kassem**  
💼 Sharing experiments in AI Automation, n8n, and RAG systems.





