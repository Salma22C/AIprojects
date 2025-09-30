# AI-Powered Chat Assistant with n8n  

## 📌 Overview  
This project is my first hands-on build as an **AI Automation Engineer**.  
I used **n8n** to create an AI agent that responds to chat messages, remembers context, and automates tasks such as scheduling events, sending emails, and retrieving contacts.  

---

## 🎯 Problem Statement  
Most chatbots today only answer questions — they don’t take real actions.  
I wanted to build a chatbot that can not only chat but also **automate workflows** such as:  
- Scheduling meetings  
- Sending emails  
- Looking up contacts from a database  

---

## ⚙️ Workflow Description  
1. **Trigger** → A chat message is received  
2. **AI Agent** → The chatbot interprets the message using the **OpenAI Chat Model**  
3. **Memory** → Past conversations are stored with **Simple Memory** for context  
4. **Automations**:  
   - **Google Calendar** → Creates events when requested  
   - **Gmail** → Sends follow-up emails  
   - **Google Sheets** → Retrieves contact details from a contacts database  

![Workflow Screenshot](workflow.png)  

---

## 🛠️ Tools & Skills Used  
- **n8n**: Workflow automation and orchestration  
- **OpenAI API**: Natural language chat model  
- **Google Calendar API**: Event creation  
- **Gmail API**: Automated email sending  
- **Google Sheets API**: Database for contacts  
- **AI Agent + Memory**: Context-aware conversation  

---

## 🚀 Outcome  
✅ Built a chatbot that doesn’t just reply, but **automates real tasks**  
✅ Integrated multiple APIs into one seamless workflow  
✅ Demonstrated **AI + Automation + Orchestration** in a single project  


