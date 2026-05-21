# 📘 AI Career Advisor (RAG System)

An intelligent Arabic AI Career Advisor built using **Retrieval-Augmented Generation (RAG)** that recommends the best learning path from structured course data and explains *why* each choice fits the user.

---

## 🚀 Demo

The system takes a user question in Arabic or English and:

- Retrieves relevant courses using semantic search (FAISS)
- Selects the **single best course**
- Explains reasoning clearly
- Generates a full **career roadmap (Beginner → Advanced)**

> ⚠️ Demo screenshots included are AI-generated outputs for educational purposes only.

---

## 🧠 Problem Statement

Learners often struggle with:

- Too many course options 📚  
- No clear learning path ❌  
- No explanation behind recommendations 🤔  

This project solves this by combining:

**Search + Reasoning + Structured Decision Making**

---

## 💡 Solution

A **RAG-based AI system** that:

- Understands user intent in Arabic 🇪🇬  
- Uses semantic search (FAISS) 🔍  
- Retrieves relevant course chunks  
- Ranks and selects ONLY the best option 🥇  
- Generates explanation + career path 🚀  

---

## ⚙️ Tech Stack

- Python 🐍  
- FAISS (Vector Search)  
- SentenceTransformers (Embeddings)  
- OpenRouter LLM API (DeepSeek / other models)  
- Gradio (UI)  
- PyPDF (Data extraction)

---

## 🏗️ Architecture
User Query
↓
Embedding Model (SentenceTransformers)
↓
FAISS Vector Search
↓
Top-K Course Retrieval
↓
LLM (OpenRouter)
↓
Ranked Recommendation + Explanation + Career Path

---

## 🧪 Example Output

🎯 Recommended Course (BEST ONLY)  
📚 Why this is the best choice  
⚖️ Why other courses are weaker  
🧩 Skills you will learn  
🚀 Career roadmap (Beginner → Advanced)

---

## 🧪 Test Question

Try this:

> أنا مبتدئ في تحليل البيانات، ما أفضل دورة أبدأ بها ولماذا؟ وما هو المسار المهني الكامل حتى الاحتراف؟

---

##⚠️ Disclaimer

This is an independent research project built for educational purposes.

Course names are used as structured dataset inputs for retrieval testing
No affiliation or endorsement with any educational provider
Screenshots are AI-generated outputs for demonstration only
🔥 Key Learnings

### This project demonstrates:

How RAG systems combine retrieval + reasoning
How LLMs act as decision engines, not just chatbots
Real-world architecture of AI recommendation systems
###📈 Future Improvements
Multi-user memory system
Personalized learning paths
Skill gap analysis
Job-market alignment scoring
Deployment as SaaS AI Career Coach
###👨‍💻 Author

Built as part of an AI Engineering portfolio project focused on:

RAG systems
LLM applications
Real-world AI decision systems
