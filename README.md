# AI Resume Critiquer Pro

## Overview
AI Resume Critiquer Pro is a web application that leverages advanced AI (OpenAI GPT-4o) to analyze resumes, provide actionable feedback, and generate professional reports. Users can upload their resume and a job description, receive detailed analysis, and view all feedback directly in the app.

## 🎥 Demo

[Click here to watch the video demo](https://youtu.be/BTZQ2M6rGJA)

## Features
- **Resume Upload:** Supports PDF and TXT files for resume analysis.
- **AI-Powered Analysis:** Uses OpenAI's GPT-4o to provide structured feedback, scores, and rewrite suggestions.
- **Job Description Match:** Calculates semantic similarity between your resume and a job description using NLP techniques.
- **Interactive UI:** Built with Streamlit for a modern, responsive user experience.
- **Customizable Settings:** Users can adjust analysis depth and preferences.
- **All Feedback In-App:** All analysis and recommendations are displayed directly in the app for easy review and copying.

## Technologies & Libraries Used
- [Streamlit](https://streamlit.io/) - For building the interactive web UI
- [OpenAI API](https://platform.openai.com/docs/api-reference) - For AI-powered resume analysis
- [PyPDF2](https://pypdf2.readthedocs.io/) - For extracting text from PDF resumes
- [scikit-learn](https://scikit-learn.org/) - For TF-IDF vectorization and semantic similarity
- [Plotly](https://plotly.com/python/) - For data visualization (gauge charts, etc.)
- [python-dotenv](https://pypi.org/project/python-dotenv/) - For environment variable management
- [pandas, numpy] - For data handling and manipulation

## What I Learned
- **Integrating with APIs:** How to securely use the OpenAI API for advanced text analysis.
- **PDF/Text Processing:** Extracting and handling text from user-uploaded files using PyPDF2 and robust error handling.
- **Building Modern UIs:** Creating interactive, user-friendly apps with Streamlit and custom CSS.
- **Session State Management:** Persisting data across reruns in Streamlit using `st.session_state`.
- **Error Handling:** Providing user-friendly error messages and robust exception handling.
- **Natural Language Processing:** Applying TF-IDF and cosine similarity for semantic matching.
- **Data Visualization:** Using Plotly to create engaging, informative charts.
- **Prompt Engineering:** Crafting prompts to elicit structured, reliable responses from LLMs.

## Getting Started

### Prerequisites
- Python 3.9+
- An OpenAI API key ([get one here](https://platform.openai.com/account/api-keys))

### Installation
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd resume_critiquer
   ```
2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up your `.env` file:**
   Create a `.env` file in the project root with:
   ```
   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

### Running the App
```bash
streamlit run main.py
```

### Usage
- Upload your resume (PDF or TXT).
- (Optional) Upload or paste a job description for targeted analysis.
- Click "Analyze Resume" to receive feedback and scores.
- All feedback and suggestions will be displayed directly in the app for you to review and copy.

## Future Improvements
- **User Authentication:** Allow users to create accounts and save their analysis history.
- **Analysis History:** Store and display past analyses for easy comparison and tracking progress.
- **Advanced NLP:** Integrate more advanced NLP models or custom keyword extraction for deeper analysis.
- **Support More File Types:** Add support for DOCX and other resume formats.
- **Improved UI/UX:** Enhance the interface with more visualizations, themes, and accessibility features.
- **Export Options:** Add options to export feedback as text or other formats (e.g., CSV, DOCX).
- **Customizable Feedback:** Let users select which aspects of their resume to focus on.

## Notes
- Your data is processed securely and never stored.
- For best results, use text-based (not scanned image) PDFs.
- If you encounter issues, check your `.env` file and ensure your OpenAI API key is valid.

## License
This project is for educational and personal use. See `LICENSE` for more details.

## 🚀 Features

### Core Analysis
- **AI-Powered Resume Analysis** using OpenAI GPT-4o
- **Multi-format Support** (PDF, TXT) with robust text extraction
- **Comprehensive Scoring System** with detailed metrics
- **ATS (Applicant Tracking System) Compatibility** analysis

### Advanced Features
- **Semantic Similarity Analysis** using TF-IDF vectorization and cosine similarity
- **Job Description Matching** with NLP techniques
- **Interactive Data Visualization** using Plotly
- **Professional PDF Report Generation** with ReportLab
- **Real-time Rewrite Suggestions** with contextual improvements

### Technical Stack
- **Natural Language Processing (NLP)** for semantic analysis
- **Machine Learning** algorithms for text similarity
- **Data Visualization** with interactive charts and gauges
- **Document Processing** with automated text extraction
- **API Integration** with OpenAI's GPT models
- **Responsive Web Interface** built with Streamlit

## 🛠️ Technologies Used

### Backend & AI
- **Python 3.9+** - Core programming language
- **OpenAI GPT-4o** - Natural language processing and analysis
- **scikit-learn** - Machine learning for semantic similarity
- **TF-IDF Vectorization** - Text feature extraction
- **Cosine Similarity** - Document similarity algorithms

### Data Processing & Visualization
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Plotly** - Interactive data visualization
- **ReportLab** - PDF generation and document creation

### Web Framework & UI
- **Streamlit** - Rapid web application development
- **Custom CSS** - Responsive design and modern UI
- **Session State Management** - User data persistence

### File Processing
- **PyPDF2** - PDF text extraction and processing
- **Multi-format Support** - PDF and TXT file handling
- **Error Handling** - Robust file processing with validation

## 📊 Technical Features

### 1. Semantic Analysis Engine
- **TF-IDF Vectorization** for text feature extraction
- **Cosine Similarity** algorithms for document comparison
- **N-gram Analysis** (1-2 grams) for better keyword matching
- **Stop Word Removal** for improved accuracy

### 2. AI-Powered Analysis
- **Structured JSON Response** parsing for consistent results
- **Multi-criteria Scoring** (content clarity, skills, experience, ATS compatibility)
- **Contextual Feedback** based on job role and description
- **Rewrite Suggestions** with explanations

### 3. Data Visualization
- **Interactive Gauge Charts** for score visualization
- **Progress Indicators** for analysis status
- **Color-coded Metrics** for quick assessment
- **Responsive Layout** for different screen sizes

### 4. Document Generation
- **Professional PDF Reports** with structured formatting
- **Custom Styling** with ReportLab
- **Automated Report Generation** with timestamps
- **Downloadable Analysis** for offline review

## 🔒 Security & Privacy

- **Local Processing** - Files processed locally, not stored
- **API Key Security** - Environment variable management
- **Data Privacy** - No personal data retention
- **Secure File Handling** - Input validation and sanitization

## 🚀 Future Enhancements

- **Database Integration** for analysis history
- **User Authentication** system
- **Advanced NLP Models** (BERT, GPT-4)
- **Resume Templates** and suggestions
- **Industry-specific Analysis** modules
- **Collaborative Features** for team reviews

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ using Python, Streamlit, and OpenAI**






