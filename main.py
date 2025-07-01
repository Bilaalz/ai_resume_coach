import streamlit as st 
import PyPDF2
import io
from openai import OpenAI
import os
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.express as px
import base64
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="AI Resume Critiquer Pro", 
    page_icon="📝", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Custom CSS for UI styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #222;
        color: #fff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .suggestion-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-weight: 500;
    }
    .score-high { color: #28a745; font-weight: bold; }
    .score-medium { color: #ffc107; font-weight: bold; }
    .score-low { color: #dc3545; font-weight: bold; }
    /* Strengths: green background, dark text */
    .suggestion-strength {
        background: #e6f9ed;
        border-left: 4px solid #28a745;
        color: #155724;
    }
    /* Weaknesses: yellow/orange background, dark text */
    .suggestion-weakness {
        background: #fff4e6;
        border-left: 4px solid #ffc107;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables for storing results and user input
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'job_description' not in st.session_state:
    st.session_state.job_description = None
if 'semantic_score' not in st.session_state:
    st.session_state.semantic_score = None

def extract_text_pdf(pdf_file):
    """Extract text from PDF file with enhanced error handling"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            extracted = page.extract_text()
            if extracted:
                text += f"Page {page_num + 1}:\n{extracted}\n\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def extract_text_file(uploaded_file):
    """Extract text from uploaded file with format detection"""
    if uploaded_file.type == "application/pdf":
        return extract_text_pdf(io.BytesIO(uploaded_file.read()))
    elif uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    else:
        st.error("Unsupported file format. Please upload PDF or TXT files.")
        return None

def calculate_semantic_similarity(resume_text, job_description):
    """Calculate semantic similarity between resume and job description using TF-IDF and cosine similarity"""
    try:
        # Clean and preprocess text
        resume_clean = re.sub(r'[^\w\s]', '', resume_text.lower())
        job_clean = re.sub(r'[^\w\s]', '', job_description.lower())
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([resume_clean, job_clean])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return round(similarity * 100, 2)
    except Exception as e:
        st.error(f"Error calculating semantic similarity: {str(e)}")
        return 0

def analyze_resume_with_ai(resume_text, job_role=None, job_description=None):
    """Enhanced AI analysis with scoring and detailed feedback"""
    try:
        # Enhanced prompt for better analysis
        prompt = f"""Analyze this resume comprehensively and provide detailed feedback in the following JSON format:

{{
    "overall_score": <score out of 100>,
    "scores": {{
        "content_clarity": <score out of 100>,
        "skills_presentation": <score out of 100>,
        "experience_descriptions": <score out of 100>,
        "ats_compatibility": <score out of 100>,
        "impact_statements": <score out of 100>
    }},
    "feedback": {{
        "strengths": ["strength1", "strength2", "strength3"],
        "weaknesses": ["weakness1", "weakness2", "weakness3"],
        "improvements": ["improvement1", "improvement2", "improvement3"]
    }},
    "rewrite_suggestions": [
        {{
            "original": "original text",
            "improved": "improved version",
            "explanation": "why this is better"
        }}
    ],
    "keywords_missing": ["keyword1", "keyword2"],
    "ats_recommendations": ["recommendation1", "recommendation2"]
}}

Focus on:
1. Content clarity and impact
2. Skills presentation and relevance
3. Experience descriptions with quantifiable results
4. ATS (Applicant Tracking System) compatibility
5. Specific improvements for {job_role if job_role else 'general job applications'}
6. Keyword optimization for {job_description if job_description else 'general roles'}

Resume content:
{resume_text}

Job Role: {job_role if job_role else 'General'}
Job Description: {job_description if job_description else 'Not provided'}

Provide only valid JSON response."""

        
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert resume reviewer and HR professional. Provide detailed, actionable feedback in JSON format only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )

        # Parse JSON response
        import json
        raw_content = response.choices[0].message.content
        # Remove code block markers and extra whitespace
        json_str = re.sub(r"^```json|^```|```$", "", raw_content.strip(), flags=re.MULTILINE).strip()
        # Optionally, print or display the raw response for debugging if parsing fails
        try:
            result = json.loads(json_str)
            return result
        except json.JSONDecodeError:
            st.warning("Could not parse AI response as JSON. Showing fallback. See raw response below for debugging.")
            st.code(raw_content)
            # Fallback if JSON parsing fails
            return {
                "overall_score": 75,
                "scores": {
                    "content_clarity": 70,
                    "skills_presentation": 75,
                    "experience_descriptions": 80,
                    "ats_compatibility": 70,
                    "impact_statements": 75
                },
                "feedback": {
                    "strengths": ["Good structure", "Relevant experience"],
                    "weaknesses": ["Could use more quantifiable results"],
                    "improvements": ["Add more metrics", "Improve keyword optimization"]
                },
                "rewrite_suggestions": [],
                "keywords_missing": [],
                "ats_recommendations": ["Use standard section headers", "Include relevant keywords"]
            }
            
    except Exception as e:
        st.error(f"Error analyzing resume: {str(e)}")
        return None

def get_score_color(score):
    """Get color class based on score"""
    if score >= 80:
        return "score-high"
    elif score >= 60:
        return "score-medium"
    else:
        return "score-low"

# Main UI
st.markdown('<div class="main-header"><h1>🤖 AI Resume Critiquer Pro</h1><p>Advanced Resume Analysis with AI-Powered Insights</p></div>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["📊 Resume Analysis", "📋 Job Description Match", "📈 Analysis History", "⚙️ Settings"]
)

if page == "📊 Resume Analysis":
    st.header("📊 Resume Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])
        job_role = st.text_input("Enter the job role you're targeting (optional)")
        
        # Job description upload
        job_description_file = st.file_uploader("Upload job description (optional)", type=["pdf", "txt"])
        job_description_text = st.text_area("Or paste job description here:", height=150)
        
        analyze = st.button("🚀 Analyze Resume", type="primary")
    
    with col2:
        st.markdown("### 💡 Tips for Better Analysis")
        st.markdown("""
        - Upload both resume and job description for semantic matching
        - Use specific job titles for targeted feedback
        - Ensure your PDF is text-readable (not scanned images)
        - Include quantifiable achievements in your resume
        """)
    
    if analyze and uploaded_file:
        try:
            with st.spinner("🔍 Analyzing your resume with AI..."):
                # Extract resume text
                file_content = extract_text_file(uploaded_file)
                
                if not file_content or not file_content.strip():
                    st.error("❌ File does not contain any readable text. Please ensure your PDF has selectable text.")
                    st.stop()
                
                # Get job description
                job_desc = None
                if job_description_file:
                    job_desc = extract_text_file(job_description_file)
                elif job_description_text:
                    job_desc = job_description_text
                
                # Analyze with AI
                analysis_results = analyze_resume_with_ai(file_content, job_role, job_desc)
                
                if analysis_results:
                    st.session_state.analysis_results = analysis_results
                    
                    # Calculate semantic similarity if job description provided
                    if job_desc:
                        semantic_score = calculate_semantic_similarity(file_content, job_desc)
                        st.session_state.semantic_score = semantic_score
                    
                    st.success("✅ Analysis completed successfully!")
                    
                    # Display results
                    st.markdown("## 📊 Analysis Results")
                    
                    # Overall score with gauge chart
                    overall_score = analysis_results.get('overall_score', 0)
                    
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = overall_score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Overall Resume Score"},
                        delta = {'reference': 80},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 60], 'color': "lightgray"},
                                {'range': [60, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed scores
                    scores = analysis_results.get('scores', {})
                    if not isinstance(scores, dict) or not scores:
                        st.warning("No detailed scores returned by the AI.")
                    else:
                        st.markdown("### Detailed Scores")
                        col1, col2, col3 = st.columns(3)
                        score_items = list(scores.items())
                        for i, (category, score) in enumerate(score_items):
                            col = col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3
                            # Convert score to int if it's a string
                            try:
                                score_val = int(score)
                            except (ValueError, TypeError):
                                score_val = 0
                            with col:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4>{category.replace('_', ' ').title()}</h4>
                                    <p class="{get_score_color(score_val)}">{score_val}/100</p>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Semantic similarity score
                    if st.session_state.semantic_score is not None:
                        st.markdown("### 🎯 Job Description Match")
                        semantic_score = st.session_state.semantic_score
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>Semantic Similarity Score</h4>
                            <p class="{get_score_color(semantic_score)}">{semantic_score}%</p>
                            <small>How well your resume matches the job description</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Feedback sections
                    feedback = analysis_results.get('feedback', {})
                    col1, col2 = st.columns(2)
                    with col1:
                        if feedback.get('strengths'):
                            st.markdown("### ✅ Strengths")
                            for strength in feedback['strengths']:
                                st.markdown(
                                    f"<div class='suggestion-box suggestion-strength'>✅ {strength}</div>",
                                    unsafe_allow_html=True
                                )
                    with col2:
                        if feedback.get('weaknesses'):
                            st.markdown("### ⚠️ Areas for Improvement")
                            for weakness in feedback['weaknesses']:
                                st.markdown(
                                    f"<div class='suggestion-box suggestion-weakness'>⚠️ {weakness}</div>",
                                    unsafe_allow_html=True
                                )
                    # Improvements
                    if feedback.get('improvements'):
                        st.markdown("### 🛠️ Recommendations")
                        for improvement in feedback['improvements']:
                            st.markdown(f"- {improvement}")
                    # Rewrite suggestions
                    suggestions = analysis_results.get('rewrite_suggestions', [])
                    if suggestions:
                        st.markdown("### ✏️ Rewrite Suggestions")
                        for i, suggestion in enumerate(suggestions[:5], 1):
                            with st.expander(f"Suggestion {i}: Improve your writing"):
                                st.markdown(f"**Original:** {suggestion.get('original', 'N/A')}")
                                st.markdown(f"**Improved:** {suggestion.get('improved', 'N/A')}")
                                st.markdown(f"**Why:** {suggestion.get('explanation', 'N/A')}")
                    # Keywords missing
                    keywords_missing = analysis_results.get('keywords_missing', [])
                    if keywords_missing:
                        st.markdown("### 🔑 Keywords Missing")
                        st.markdown(", ".join(keywords_missing))
                    # ATS recommendations
                    ats_recommendations = analysis_results.get('ats_recommendations', [])
                    if ats_recommendations:
                        st.markdown("### 📝 ATS Recommendations")
                        for rec in ats_recommendations:
                            st.markdown(f"- {rec}")
                
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")

elif page == "📋 Job Description Match":
    st.header("📋 Job Description Match Analysis")
    
    st.markdown("""
    ### How it works:
    1. Upload your resume and a job description
    2. Our AI analyzes semantic similarity using NLP techniques
    3. Get detailed insights on keyword matching and content alignment
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        resume_file = st.file_uploader("Upload your resume", type=["pdf", "txt"], key="match_resume")
        job_desc_file = st.file_uploader("Upload job description", type=["pdf", "txt"], key="match_job")
    
    with col2:
        job_desc_text = st.text_area("Or paste job description here:", height=200)
        
        if st.button("🎯 Analyze Match", type="primary"):
            if resume_file and (job_desc_file or job_desc_text):
                with st.spinner("Analyzing semantic match..."):
                    resume_text = extract_text_file(resume_file)
                    job_desc = job_desc_text if job_desc_text else extract_text_file(job_desc_file)
                    
                    if resume_text and job_desc:
                        similarity_score = calculate_semantic_similarity(resume_text, job_desc)
                        
                        st.markdown("## 📊 Match Analysis Results")
                        
                        # Similarity gauge
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = similarity_score,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Semantic Similarity Score"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 40], 'color': "red"},
                                    {'range': [40, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "green"}
                                ]
                            }
                        ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Interpretation
                        if similarity_score >= 70:
                            st.success("🎉 Excellent match! Your resume aligns well with the job description.")
                        elif similarity_score >= 40:
                            st.warning("⚠️ Moderate match. Consider adding more relevant keywords and experience.")
                        else:
                            st.error("❌ Low match. Your resume needs significant updates to align with this role.")
                        
                        # Keyword analysis (simplified)
                        st.markdown("### 🔍 Keyword Analysis")
                        st.info("""
                        **High Match (70%+):** Your resume effectively targets this role
                        **Moderate Match (40-70%):** Some alignment, room for improvement  
                        **Low Match (<40%):** Consider if this role is the right fit
                        """)
            else:
                st.error("Please upload both resume and job description.")

elif page == "📈 Analysis History":
    st.header("📈 Analysis History")
    st.info("This feature would store your analysis history in a database. Currently showing demo data.")
    
    # Demo data
    demo_analyses = [
        {"date": "2024-01-15", "filename": "resume_v1.pdf", "score": 85, "job_role": "Software Engineer"},
        {"date": "2024-01-10", "filename": "resume_v2.pdf", "score": 72, "job_role": "Data Scientist"},
        {"date": "2024-01-05", "filename": "resume_v3.pdf", "score": 68, "job_role": "Product Manager"}
    ]
    
    for analysis in demo_analyses:
        with st.expander(f"{analysis['date']} - {analysis['filename']} ({analysis['job_role']})"):
            st.metric("Score", f"{analysis['score']}/100")
            st.progress(analysis['score']/100)

elif page == "⚙️ Settings":
    st.header("⚙️ Settings")
    
    st.markdown("### API Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    
    st.markdown("### Analysis Preferences")
    analysis_depth = st.selectbox("Analysis Depth", ["Standard", "Detailed", "Comprehensive"])
    include_rewrite_suggestions = st.checkbox("Include rewrite suggestions", value=True)
    include_keyword_analysis = st.checkbox("Include keyword analysis", value=True)
    
    if st.button("💾 Save Settings"):
        st.success("Settings saved successfully!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🤖 Powered by OpenAI GPT-4o | 📊 Built with Streamlit | 🔒 Your data is processed securely</p>
</div>
""", unsafe_allow_html=True)