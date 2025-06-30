# 🤖 AI Resume Critiquer Pro

An advanced AI-powered resume analysis tool that provides comprehensive feedback, semantic matching, and professional PDF reports.

## 🚀 Features

### Core Analysis
- **AI-Powered Resume Analysis** using OpenAI GPT-3.5
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

## 🎯 Resume-Ready Technical Buzzwords

This project demonstrates proficiency in:

### AI & Machine Learning
- **Natural Language Processing (NLP)**
- **Semantic Similarity Analysis**
- **TF-IDF Vectorization**
- **Cosine Similarity Algorithms**
- **Machine Learning Integration**

### Data Science & Analytics
- **Data Visualization**
- **Interactive Dashboards**
- **Statistical Analysis**
- **Real-time Data Processing**

### Software Engineering
- **API Integration**
- **Web Application Development**
- **File Processing & Validation**
- **Error Handling & Exception Management**
- **Session State Management**

### Document Processing
- **PDF Text Extraction**
- **Multi-format File Support**
- **Automated Report Generation**
- **Document Analysis**

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd resume_critiquer
```

2. Install dependencies:
```bash
pip install -e .
```

3. Set up environment variables:
```bash
cp .env.example .env
# Add your OpenAI API key to .env
```

4. Run the application:
```bash
streamlit run main.py
```

## 📈 Usage

1. **Upload Resume**: Upload your resume in PDF or TXT format
2. **Add Job Description**: Optionally upload or paste a job description
3. **Get Analysis**: Receive comprehensive AI-powered feedback
4. **View Results**: Interactive visualizations and detailed scores
5. **Download Report**: Generate and download professional PDF reports

## 🔧 Architecture

```
resume_critiquer/
├── main.py                 # Main Streamlit application
├── pyproject.toml         # Project dependencies
├── README.md              # Project documentation
└── .env                   # Environment variables
```

### Key Components

- **Text Extraction Engine**: Handles PDF and TXT file processing
- **AI Analysis Module**: Integrates with OpenAI for comprehensive feedback
- **Semantic Matching Engine**: Uses NLP for job description matching
- **Visualization Module**: Creates interactive charts and metrics
- **Report Generator**: Produces professional PDF reports

## 🎨 UI/UX Features

- **Modern Gradient Design** with professional styling
- **Responsive Layout** that works on all devices
- **Interactive Navigation** with sidebar menu
- **Real-time Feedback** with loading states
- **Color-coded Metrics** for quick assessment
- **Expandable Sections** for detailed information

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

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Built with ❤️ using Python, Streamlit, and OpenAI**




