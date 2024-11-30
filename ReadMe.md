# Shepherd AI

> ### "The Lord is my shepherd; I shall not want." — Psalm 23:1  
> **Empathy-Driven AI Inspired by the Good Shepherd**

## Inspiration
In the Bible, Jesus is often referred to as the Good Shepherd, guiding His flock with compassion, patience, and love. Inspired by this metaphor, **Shepherd AI** was created to emulate this care, providing empathetic and accessible mental health support to those who may feel lost or burdened. By bringing technology and faith together, Shepherd AI seeks to remind users that they are never alone, even in their darkest moments.

> _"Come to me, all you who are weary and burdened, and I will give you rest." — Matthew 11:28_

## What It Does
**Shepherd AI** is a virtual guide designed to offer compassionate, professional, and faith-anchored support for mental health and emotional well-being.

### Key Features:
- **Empathetic Guidance**: Provides tailored, caring responses to users’ concerns.  
- **Faith-Inspired Perspective**: Encourages hope, healing, and grace in alignment with Christian principles.  
- **Concise and Clear**: Keeps responses short but meaningful, reflecting thoughtful listening.  
- **Accessible Help**: Always available, offering a non-judgmental space for users to share their struggles.  

## How We Built It
Inspired by the principles of servant leadership and love, we utilized advanced AI technologies to create a tool rooted in compassion.

### Technologies Used:
- **AI Framework**: Trained to deliver thoughtful, supportive, and professional responses.  
- **Agent Prompting**: Modeled after a shepherd’s role—gentle, guiding, and protective.  
- **Real-Time Interaction**: WebSocket integration ensures smooth, real-time conversations.  
- **Ethical and Secure**: Prioritizes user privacy, following strict data protection standards.  

## Challenges We Ran Into
> _"Consider it pure joy, my brothers and sisters, whenever you face trials of many kinds." — James 1:2_

Building Shepherd AI reminded us that challenges are opportunities for growth:
- **Balancing Empathy and Professionalism**: Ensuring the AI maintains a caring yet professional tone.  
- **Respect for User Beliefs**: Designing interactions that are welcoming to all, regardless of faith background.  
- **Data Security**: Protecting the sensitive nature of user conversations.  

## Accomplishments That We're Proud Of
- **Faithful to Its Purpose**: Crafted a tool inspired by biblical principles of love and guidance.  
- **Accessible Support**: Lowered barriers to mental health care, ensuring more people feel seen and supported.  
- **Real Conversations**: Achieved a human-like conversational flow, bringing warmth and empathy to users.  

## What We Learned
> _"For I know the plans I have for you," declares the Lord, "plans to prosper you and not to harm you, plans to give you hope and a future." — Jeremiah 29:11_

Through Shepherd AI, we’ve learned:  
- The importance of active listening and creating a safe space for users.  
- The potential of technology to reflect God’s love and care in serving others.  
- How faith can inspire innovation, reminding us that our work is part of a higher calling.  

## What's Next for Shepherd AI
> _"Let us not become weary in doing good, for at the proper time we will reap a harvest if we do not give up." — Galatians 6:9_

Looking to the future, we plan to expand Shepherd AI's capabilities:  
- **Deeper Emotional Understanding**: Enhance the AI’s ability to respond to complex emotional states.  
- **Faith-Based Resources**: Integrate scripture readings, prayers, and devotionals tailored to users’ struggles.  
- **Journaling and Reflection Tools**: Help users process their thoughts and emotions in a meaningful way.  
- **Community Connection**: Provide options for users to connect with local churches or Christian counselors.  
- **Multilingual Support**: Reach a global audience by offering interactions in multiple languages.  

## Conclusion
**Shepherd AI** is more than a tool; it is a reflection of the care and love we are called to show one another. Just as the Good Shepherd leaves the ninety-nine to seek the one lost sheep, Shepherd AI strives to meet each individual where they are, offering guidance, hope, and healing.

> _"And surely I am with you always, to the very end of the age." — Matthew 28:20_

Join us in this mission to bring light and support to those in need through the transformative power of Shepherd AI.

## Technologies Used

  Backend built with: <br>
  <img src="https://img.shields.io/badge/JavaScript-323330?style=for-the-badge&logo=javascript&logoColor=F7DF1E" alt="Javascript">
  <img src="https://img.shields.io/badge/OpenAI-34a853?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI">
  <img src="https://img.shields.io/badge/Retell_AI-FF7F50?style=for-the-badge&logo=retell&logoColor=white" alt="Retell AI">
  <img src="https://img.shields.io/badge/FAISS-000000?style=for-the-badge&logo=faiss&logoColor=white" alt="FAISS">
  <img src="https://img.shields.io/badge/FastAPI-000000?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Uvicorn-000000?style=for-the-badge&logo=uvicorn&logoColor=white" alt="Uvicorn">
  <br>

### Prerequisites
- Python 3.8 or higher
- [ngrok](https://ngrok.com/) installed for public network exposure
- API keys for the LLM service (configured in `.env`)



### Setup

1. **Install Dependencies**:
   Run the following command to install required packages:
   ```bash
   pip3 install -r requirements.txt
   ```
2. **Configure Environment Variables**:
   Copy the `.env.example` file to `.env` and set the required variables.
   ```bash
   cp .env.example .env
   ```
3. **In another bash, use ngrok to expose this port to public network**
```bash
ngrok http 8080
```

4. **Start the websocket server**
```bash
uvicorn app.server:app --reload --port=8080
```

You should see a fowarding address like
`https://dc14-2601-645-c57f-8670-9986-5662-2c9a-adbd.ngrok-free.app`, and you
are going to take the hostname `dc14-2601-645-c57f-8670-9986-5662-2c9a-adbd.ngrok-free.app`, prepend it with `wss://`, postpend with
`/llm-websocket` (the route setup to handle LLM websocket connection in the code) to create the url to use in the [dashboard](https://beta.retellai.com/dashboard) to create a new agent. Now
the agent you created should connect with your localhost.

The custom LLM URL would look like
`wss://dc14-2601-645-c57f-8670-9986-5662-2c9a-adbd.ngrok-free.app/llm-websocket`