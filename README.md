# 🤖 YouTube AI Agent Dashboard 🎬  
> Built with LangChain, LangGraph, GPT, Streamlit, and the YouTube Data API

A multi-tool autonomous AI Agent that analyzes any YouTube channel based on a provided Channel ID and a new video idea. It generates catchy titles, descriptions, thumbnails, scheduling insights, interactive charts, and more — all in one streamlined dashboard!

---

## 📌 Project Features

✅ Multi-step agentic workflow (LangChain)  
✅ Human-style outputs powered by GPT-4o Mini  
✅ Integrated Q&A and regeneration tools  
✅ Two interactive visualizations (Plotly)  
✅ AI-generated title, description, and thumbnail  
✅ YouTube Data API-powered channel analysis  
✅ Clean Streamlit UI, emoji-rich and styled  
✅ Memory-enabled using `MemorySaver`  

---

## 🧠 Agent Architecture

- **Agent Type**: `create_react_agent()` from LangGraph
- **Model Used**: `gpt-4o-mini` for reasoning, tool routing, and general instructions
- **Q&A Fallback**: `gpt-3.5-turbo` used in Mini Q&A Tool
- **Memory**: Enabled with `MemorySaver()` to retain context across the run
- **Tool Routing**: Intelligent function calling using tool schema descriptions

---

## 🔧 Tech Stack & Tools

| Component     | Used Technology              |
|--------------|------------------------------|
| 🧠 LLM        | OpenAI GPT-4o, GPT-4o Mini, GPT-3.5 Turbo  |
| 🧰 Framework  | LangChain, LangGraph         |
| 🎛️ Agent Flow | create_react_agent()         |
| 📊 Viz        | Plotly, Matplotlib, HTML      |
| 📺 Data API   | YouTube Data API v3          |
| 🎙️ Audio      | Whisper (for voice input – optional) |
| 🌐 Frontend   | Streamlit                    |

---

## 📂 Project Structure

