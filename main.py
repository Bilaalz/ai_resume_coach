import streamlit as st 
import PyPDF2
import io
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="AI Resume Critiquer", page_icon="📝", layout="centered")
st.title("AI Resume Critiquer")
st.markdown("Upload your resume and get AI-powered feedback instantly!")

uploaded_file = st.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])
job_role = st.text_input("Enter the job role you're targeting (optional)")
analyze = st.button("Analyze Resume")

def extract_text_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def extract_text_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return extract_text_pdf(io.BytesIO(uploaded_file.read()))
    return uploaded_file.read().decode("utf-8")

if analyze and uploaded_file:
    try:
        with st.spinner("Analyzing your resume..."):
            file_content = extract_text_file(uploaded_file)

            if not file_content.strip():
                st.error("File does not have any content...")
                st.stop()
            
            prompt = f"""Please analyze this resume and provide constructive feedback.
            Focus on the following aspects:
            1. Content clarity and impact
            2. Skills presentation
            3. Experience Descriptions 
            4. Specific improvements for {job_role if job_role else 'general job applications'}

            Resume content:
            {file_content}

            Please provide your analysis in a clear, structured format with specific recommendations."""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # or "gpt-4" if you paid for GPT-4 access
                messages=[
                    {"role": "system", "content": "You are an expert resume reviewer with years of experience in HR and recruitment"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            st.markdown("### 📝 Analysis Results")
            st.markdown(response.choices[0].message.content)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")