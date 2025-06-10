import streamlit as st
import whisper
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import uuid
import tempfile
import re
from typing import List, Dict
import time
import numpy as np
import subprocess
import sys

# Load environment variables from .env file
load_dotenv()

def check_ffmpeg():
    """Check if FFmpeg is installed and accessible."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

# Set page config
st.set_page_config(
    page_title="AI Podcast Analyzer",
    page_icon="🎙️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
    .highlight-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #ffcdd2;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Streamlit app
st.title("🎙️ AI Podcast Analyzer")
st.markdown("""
    Upload a podcast audio file to get:
    - 📝 A concise summary
    - ✨ Key highlights
    - ❓ Interactive Q&A about the content
""")

# Check for FFmpeg
if not check_ffmpeg():
    st.markdown('<div class="error-box">', unsafe_allow_html=True)
    st.error("""
        FFmpeg is not installed or not found in PATH! Please:
        1. Download FFmpeg from https://github.com/BtbN/FFmpeg-Builds/releases (get the ffmpeg-master-latest-win64-gpl.zip)
        2. Extract the zip file
        3. Copy the contents of the 'bin' folder to a new folder (e.g., C:\\ffmpeg\\bin)
        4. Add this folder to your system PATH:
           - Open System Properties > Advanced > Environment Variables
           - Under System Variables, find and select 'Path'
           - Click Edit > New
           - Add the path to your FFmpeg bin folder (e.g., C:\\ffmpeg\\bin)
        5. Restart your terminal/IDE
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# Check for OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.markdown('<div class="error-box">', unsafe_allow_html=True)
    st.error("""
        OpenAI API key not found! Please ensure:
        1. You have a .env file in the project directory
        2. The .env file contains: OPENAI_API_KEY=your-api-key
        3. The API key is valid
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# Initialize session state
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = []

# Initialize models
@st.cache_resource
def load_models():
    with st.spinner("Loading models..."):
        whisper_model = whisper.load_model("base")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Initialize summarization model
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        # Initialize Q&A model
        qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
        return whisper_model, embedding_model, summarizer, qa_model

whisper_model, embedding_model, summarizer, qa_model = load_models()

def transcribe_audio(audio_file) -> str:
    """Transcribe audio file using Whisper with progress tracking."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_file.write(audio_file.read())
        temp_file_path = temp_file.name
    
    try:
        status_text.text("Starting transcription...")
        progress_bar.progress(10)
        
        result = whisper_model.transcribe(temp_file_path)
        progress_bar.progress(100)
        status_text.text("Transcription complete!")
        
        return result["text"]
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None
    finally:
        os.unlink(temp_file_path)
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()

def chunk_transcript(transcript: str, max_length: int = 1000) -> List[Dict]:
    """Enhanced chunking strategy with better sentence boundary detection."""
    # Split on sentence boundaries while preserving punctuation
    sentences = re.split(r'(?<=[.!?])\s+', transcript)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "content": current_chunk.strip(),
                    "meta": {"chunk_index": len(chunks)}
                })
            current_chunk = sentence
    
    if current_chunk:
        chunks.append({
            "id": str(uuid.uuid4()),
            "content": current_chunk.strip(),
            "meta": {"chunk_index": len(chunks)}
        })
    
    return chunks

def get_embeddings(chunks: List[Dict]) -> List[np.ndarray]:
    """Generate embeddings for text chunks."""
    texts = [chunk["content"] for chunk in chunks]
    return embedding_model.encode(texts)

def find_relevant_chunks(query: str, chunks: List[Dict], embeddings: List[np.ndarray], top_k: int = 3) -> List[str]:
    """Find most relevant chunks for a query using cosine similarity."""
    query_embedding = embedding_model.encode(query)
    similarities = [np.dot(query_embedding, chunk_embedding) / 
                   (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding))
                   for chunk_embedding in embeddings]
    
    # Get top k chunks
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i]["content"] for i in top_indices]

def process_podcast(audio_file):
    """Process the podcast with progress tracking."""
    # Transcribe
    transcript = transcribe_audio(audio_file)
    if not transcript:
        return
    
    st.session_state.transcript = transcript
    
    # Chunk transcript and generate embeddings
    chunks = chunk_transcript(transcript)
    embeddings = get_embeddings(chunks)
    
    st.session_state.chunks = chunks
    st.session_state.embeddings = embeddings
    
    # Generate summary
    with st.spinner("Generating summary and highlights..."):
        try:
            # Split transcript into chunks for summarization (BART has a max input length)
            max_chunk_length = 1024
            transcript_chunks = [transcript[i:i + max_chunk_length] for i in range(0, len(transcript), max_chunk_length)]
            
            # Generate summaries for each chunk
            chunk_summaries = []
            for chunk in transcript_chunks:
                summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
                chunk_summaries.append(summary[0]['summary_text'])
            
            # Combine summaries
            combined_summary = " ".join(chunk_summaries)
            
            # Generate final summary
            final_summary = summarizer(combined_summary, max_length=150, min_length=50, do_sample=False)
            
            # Format the summary with highlights
            formatted_summary = f"""SUMMARY:
{final_summary[0]['summary_text']}

HIGHLIGHTS:
• {chunk_summaries[0]}
• {chunk_summaries[1] if len(chunk_summaries) > 1 else ''}
• {chunk_summaries[2] if len(chunk_summaries) > 2 else ''}

TOPICS:
• Main discussion points extracted from the summary
• Key themes identified in the conversation
• Important takeaways from the podcast"""
            
            st.session_state.summary = formatted_summary
        except Exception as e:
            st.error(f"""
            ❌ Error generating summary: {str(e)}
            
            If this error persists:
            1. Check your internet connection
            2. Ensure you have sufficient disk space
            3. Try restarting the application
            """)
            return
    
    st.session_state.processing_complete = True

# Main UI
uploaded_file = st.file_uploader(
    "Upload Podcast Audio",
    type=["mp3", "wav", "m4a", "ogg"],
    help="Supported formats: MP3, WAV, M4A, OGG"
)

if uploaded_file and not st.session_state.processing_complete:
    process_podcast(uploaded_file)

# Display results
if st.session_state.processing_complete:
    # Summary and Highlights
    st.markdown("### 📝 Summary & Highlights")
    st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
    st.write(st.session_state.summary)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Transcript (collapsible)
    with st.expander("View Full Transcript"):
        st.text_area("Transcript", st.session_state.transcript, height=300)
    
    # Q&A Section
    st.markdown("### ❓ Ask Questions About the Podcast")
    user_question = st.text_input("Your Question:", placeholder="Type your question here...")
    
    if user_question:
        with st.spinner("Generating answer..."):
            # Find relevant chunks
            relevant_chunks = find_relevant_chunks(
                user_question,
                st.session_state.chunks,
                st.session_state.embeddings
            )
            context = "\n".join(relevant_chunks)
            
            # Generate answer using the Q&A model
            try:
                answer = qa_model(
                    question=user_question,
                    context=context
                )
                
                st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
                st.write("**Answer:**")
                st.write(answer['answer'])
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"""
                ❌ Error generating answer: {str(e)}
                
                If this error persists:
                1. Check your internet connection
                2. Try rephrasing your question
                3. Ensure the question is related to the podcast content
                """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit, Whisper, and Hugging Face Transformers</p>
    </div>
""", unsafe_allow_html=True)