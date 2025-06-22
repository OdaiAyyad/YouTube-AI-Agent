import streamlit as st
import sys
import os
from pathlib import Path
import tempfile
import base64
from io import BytesIO
import re
import time

# Import your main agent code
# Make sure your main code is in a file called 'youtube_agent.py'
try:
    from youtube_agent import agent, config, tools, mini_qa
    from langchain_core.messages import HumanMessage
    import whisper
    from gtts import gTTS
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error("Make sure your main code is saved as 'youtube_agent.py' in the same directory")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🎬 YouTube AI Agent",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #4facfe;
        font-size: 3rem;
        margin-bottom: 2rem;
    }
    .sub-header {
        text-align: center;
        color: #0a0a0a;
        margin-bottom: 2rem;
    }
    .result-container {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .qa-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4facfe;
    }
    .stButton > button {
        background: linear-gradient(45deg, #4facfe, #00f2fe);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(45deg, #00f2fe, #4facfe);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'main_results' not in st.session_state:
    st.session_state.main_results = ""
if 'channel_id' not in st.session_state:
    st.session_state.channel_id = ""
if 'video_idea' not in st.session_state:
    st.session_state.video_idea = ""
if 'title_style' not in st.session_state:
    st.session_state.title_style = ""

def clean_for_tts(text):
    """Enhanced text cleaning for text-to-speech - removes emojis and formatting"""
    import re
    
    # Remove emojis completely (all Unicode emoji ranges)
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        "]+", flags=re.UNICODE)
    
    text = emoji_pattern.sub('', text)
    
    # Remove markdown formatting
    text = re.sub(r"[#*•`_~\[\]]+", "", text)  # Remove markdown symbols
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # Bold text
    text = re.sub(r"\*(.*?)\*", r"\1", text)  # Italic text
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove special characters that cause TTS issues
    text = re.sub(r"[💰📊🎯📈📺⏰✅❌🚀🎨📅⚡🔄📝💡🎬🤖💬🎙️🗣️🔍🤔]", "", text)
    
    # Clean up extra punctuation and symbols
    text = re.sub(r"[|•▪▫◦‣⁃]", "", text)  # Remove bullet points
    text = re.sub(r"[-=]{3,}", "", text)  # Remove long dashes
    text = re.sub(r"_{3,}", "", text)  # Remove underscores
    
    # Fix spacing and line breaks
    text = re.sub(r"\s+", " ", text)  # Multiple spaces to single
    text = re.sub(r"\n+", ". ", text)  # Line breaks to periods
    
    # Clean up punctuation for better speech
    text = re.sub(r"\.{2,}", ".", text)  # Multiple periods
    text = re.sub(r",{2,}", ",", text)  # Multiple commas
    text = re.sub(r"!{2,}", "!", text)  # Multiple exclamations
    text = re.sub(r"\?{2,}", "?", text)  # Multiple questions
    
    # Remove parentheses content that might be noisy
    text = re.sub(r'\([^)]*\)', '', text)  # Remove content in parentheses
    
    # Final cleanup
    text = text.strip()
    
    # If text is too long, truncate for TTS (optional)
    if len(text) > 500:
        text = text[:500] + "..."
    
    return text

def run_clean_agent(video_prompt, config):
    """Clean agent execution with progress tracking"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🚀 Starting YouTube AI Agent Analysis...")
    progress_bar.progress(10)
    
    all_messages = []
    step_count = 0
    
    for step in agent.stream(
        {"messages": [HumanMessage(content=video_prompt)]}, 
        config, 
        stream_mode="values"
    ):
        all_messages.extend(step["messages"])
        step_count += 1
        
        # Update progress
        progress = min(90, 10 + (step_count * 10))
        progress_bar.progress(progress)
        
        if step_count == 1:
            status_text.text("📺 Fetching channel information...")
        elif step_count == 2:
            status_text.text("🎨 Analyzing content and creating visualizations...")
        elif step_count == 3:
            status_text.text("✍️ Generating titles and descriptions...")
        elif step_count == 4:
            status_text.text("⏰ Calculating best posting times...")
        elif step_count >= 5:
            status_text.text("🎯 Finalizing results...")
    
    # Find final response
    final_response = None
    for msg in reversed(all_messages):
        if hasattr(msg, 'type') and msg.type == 'ai' and hasattr(msg, 'content'):
            if not msg.tool_calls:
                final_response = msg.content
                break
    
    progress_bar.progress(100)
    status_text.text("✅ Analysis Complete!")
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()
    
    return final_response

def handle_qa_question(channel_id, question, video_idea=None, title_style=None):
    """Handle Q&A questions with context from main analysis"""
    
    # Build the QA prompt with context
    if video_idea and title_style:
        qa_prompt = f"""channel_id: {channel_id}
question: {question}
video_idea: {video_idea}
title_style: {title_style}"""
    else:
        qa_prompt = f"""channel_id: {channel_id}
question: {question}"""
    
    response = ""
    for step in agent.stream(
        {"messages": [HumanMessage(content=qa_prompt)]}, 
        config, 
        stream_mode="values"
    ):
        # Get the last message content
        if step["messages"] and hasattr(step["messages"][-1], 'content'):
            response = step["messages"][-1].content
    
    return response
# Main UI
st.markdown('''
<div class="result-container">
    <h1 class="main-header">🎬 YouTube AI Agent</h1>
    <p class="sub-header">Analyze your channel, generate content ideas, and optimize your YouTube strategy!</p>
</div>
''', unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.header("📝 Video Configuration")
    
    # Channel ID input
    channel_id = st.text_input(
        "🎯 YouTube Channel ID",
        placeholder="UCxxxxxxxxxxxxxxxxxx",
        help="Find your Channel ID in YouTube Studio > Settings > Channel"
    )
    
    # Video idea input
    video_idea = st.text_area(
        "💡 Video Idea",
        placeholder="Enter your next video idea...",
        height=100
    )
    
    # Style selection
    title_style = st.selectbox(
        "🎨 Title Style",
        ["Conservative", "Bold", "Trendy"],
        index=2
    )
    
    # Analyze button
    analyze_button = st.button("🚀 Analyze Channel & Generate Content", type="primary")

# Main content area
if analyze_button and channel_id and video_idea:
    # Store in session state
    st.session_state.channel_id = channel_id
    st.session_state.video_idea = video_idea
    st.session_state.title_style = title_style
    
    # Create the analysis prompt
    video_prompt = f"""I have a new video idea: "{video_idea}".
Please:
- First, show me the channel name for channel ID: {channel_id}
- Analyze my channel content and create comprehensive INTERACTIVE visualizations
- Suggest 3 catchy titles in {title_style} style
- Create a short YouTube description
- Recommend the BEST TIME to post this video based on my channel's performance
- Generate an intelligent, attractive thumbnail optimized for my content type

Use the tools you need, and show all results including the scheduling recommendation."""
    
    # Run analysis
    with st.container():
        results = run_clean_agent(video_prompt, config)
        
        if results:
            st.session_state.main_results = results
            st.session_state.analysis_done = True
            
            st.markdown("## 📊 Analysis Results")
            st.markdown(results)
            
            # Check if HTML files were created and display them
            if os.path.exists("interactive_youtube_analytics.html"):
                st.markdown("### 📈 Interactive Analytics Dashboard")
                with open("interactive_youtube_analytics.html", "r", encoding="utf-8") as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=800, scrolling=True)
            
            if os.path.exists("content_performance_analysis.html"):
                st.markdown("### 📊 Content Performance Analysis")
                with open("content_performance_analysis.html", "r", encoding="utf-8") as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=500, scrolling=True)

elif analyze_button:
    st.error("Please fill in both Channel ID and Video Idea!")

# Display previous results if available
elif st.session_state.analysis_done:
    st.markdown("## 📊 Analysis Results")
    st.markdown(st.session_state.main_results)
    
    # Display HTML files if they exist
    if os.path.exists("interactive_youtube_analytics.html"):
        st.markdown("### 📈 Interactive Analytics Dashboard")
        with open("interactive_youtube_analytics.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800, scrolling=True)

# Q&A Section
if st.session_state.analysis_done:
    st.markdown("---")
    st.markdown("## 🤖 Q&A Section")
    
    # Text-based Q&A
    st.markdown("### 💬 Ask a Question")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        qa_question = st.text_input(
            "Type your question:",
            placeholder="What's my best performing video?",
            key="qa_input",
            label_visibility="collapsed"  # Add this line
        )
    
    with col2:
        ask_button = st.button("Ask", type="secondary")
    
    if ask_button and qa_question:
        with st.spinner("🤔 Thinking..."):
            qa_response = handle_qa_question(
                st.session_state.channel_id, 
                qa_question,
                st.session_state.get('video_idea'),
                st.session_state.get('title_style')
            )
            st.markdown("**Answer:**")
            st.markdown(qa_response)
            
            # Generate voice response
            try:
                clean_text = clean_for_tts(qa_response)
                tts = gTTS(text=clean_text, lang='en')
                tts.save("qa_answer.mp3")
                
                # Play audio
                audio_file = open("qa_answer.mp3", "rb")
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3")
                audio_file.close()
                
                # Clean up
                if os.path.exists("qa_answer.mp3"):
                    os.remove("qa_answer.mp3")
                    
            except Exception as e:
                st.warning(f"Voice generation failed: {e}")
    
    # Voice-based Q&A
    st.markdown("### 🎙️ Voice Question")
    uploaded_audio = st.file_uploader(
        "Upload an audio question (MP3, WAV, M4A):",
        type=['mp3', 'wav', 'm4a'],
        key="voice_upload"
    )
    
    if uploaded_audio is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(uploaded_audio.read())
            tmp_file_path = tmp_file.name
        
        try:
            # Transcribe audio
            with st.spinner("🔍 Transcribing audio..."):
                model = whisper.load_model("base")
                transcription_result = model.transcribe(tmp_file_path)
                question_text = transcription_result["text"]
                
                st.markdown(f"**🗣️ Transcribed Question:** {question_text}")
                
                # Get answer
                with st.spinner("🤔 Processing your question..."):
                    qa_response = handle_qa_question(
                        st.session_state.channel_id, 
                        question_text,
                        st.session_state.get('video_idea'),
                        st.session_state.get('title_style')
                    )
                    st.markdown("**Answer:**")
                    st.markdown(qa_response)
                    
                    # Generate voice response
                    try:
                        clean_text = clean_for_tts(qa_response)
                        tts = gTTS(text=clean_text, lang='en')
                        tts.save("voice_qa_answer.mp3")
                        
                        # Play audio
                        audio_file = open("voice_qa_answer.mp3", "rb")
                        audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format="audio/mp3")
                        audio_file.close()
                        
                        # Clean up
                        if os.path.exists("voice_qa_answer.mp3"):
                            os.remove("voice_qa_answer.mp3")
                            
                    except Exception as e:
                        st.warning(f"Voice generation failed: {e}")
                        
        except Exception as e:
            st.error(f"Audio processing failed: {e}")
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Quick Actions
if st.session_state.analysis_done:
    st.markdown("---")
    st.markdown("## ⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Regenerate Titles"):
            with st.spinner("Generating new titles..."):
                response = handle_qa_question(
                    st.session_state.channel_id, 
                    "Regenerate Title",
                    st.session_state.get('video_idea'),
                    st.session_state.get('title_style')
                )
                st.markdown(response)

    with col2:
        if st.button("📝 Regenerate Description"):
            with st.spinner("Generating new description..."):
                response = handle_qa_question(
                    st.session_state.channel_id, 
                    "Regenerate Description",
                    st.session_state.get('video_idea'),
                    st.session_state.get('title_style')
                )
                st.markdown(response)

    with col3:
        if st.button("📊 Channel Summary"):
            with st.spinner("Getting channel summary..."):
                response = handle_qa_question(
                    st.session_state.channel_id, 
                    "Give me a channel summary"
                )
                st.markdown(response)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🎬 YouTube AI Agent | Built with LangChain, LangGraph & Streamlit</p>
    <p>Perfect for content creators who want to optimize their YouTube strategy!</p>
</div>
""", unsafe_allow_html=True)