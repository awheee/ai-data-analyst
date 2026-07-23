# XMail – AI-Driven Email Categorization Engine

An AI-powered email classification system that automatically categorizes emails into **Spam, Promotions, Social, and Important** using Natural Language Processing (NLP) and supervised Machine Learning. The project combines a Python backend with a React frontend and integrates the Gmail API for secure email access.

## Features

- 📧 Automatic email classification into:
  - Spam
  - Promotions
  - Social
  - Important
- 🤖 Supervised Machine Learning pipeline for text classification
- 📝 Text preprocessing and vectorization using NLP techniques
- 📊 Model evaluation with cross-validation for reliable performance
- 🔐 Secure Gmail integration using OAuth 2.0 authentication
- ☁️ Google Cloud integration for Gmail API access
- 🌐 Full-stack web application with React frontend and FastAPI/Flask backend

## Tech Stack

### Backend
- Python
- FastAPI / Flask
- Scikit-learn
- Pandas
- NumPy

### Machine Learning & NLP
- Scikit-learn
- TF-IDF / Count Vectorization
- Text preprocessing
- Supervised Classification
- Cross-validation

### Frontend
- React
- HTML
- CSS
- JavaScript

### APIs & Cloud
- Gmail API
- OAuth 2.0
- Google Cloud Platform

## Machine Learning Pipeline

1. Email data collection and preprocessing
2. Text cleaning and normalization
3. Feature extraction using text vectorization
4. Model training using supervised learning algorithms
5. Performance evaluation using cross-validation
6. Prediction and categorization of incoming emails

## Email Categories

| Category | Description |
|----------|-------------|
| Spam | Unwanted or malicious emails |
| Promotions | Marketing and promotional content |
| Social | Notifications from social media platforms |
| Important | Personal or high-priority emails |

## Installation

### Clone the repository

```bash
git clone <repository-url>
cd xmail
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure Google Cloud & Gmail API

1. Create a Google Cloud project.
2. Enable the Gmail API.
3. Create OAuth 2.0 credentials.
4. Download the credentials file and place it in the project directory.
5. Configure the required environment variables.

### Run the Backend

```bash
python app.py
```

or

```bash
uvicorn main:app --reload
```

### Run the Frontend

```bash
cd frontend
npm install
npm run dev
```

## Project Structure

```
xmail/
│
├── backend/
│   ├── models/
│   ├── preprocessing/
│   ├── api/
│   └── app.py
│
├── frontend/
│   ├── src/
│   ├── components/
│   └── public/
│
├── requirements.txt
└── README.md
```

## Future Improvements

- Multi-label email classification
- Email summarization using LLMs
- Priority scoring system
- Custom user-defined categories
- Continuous model retraining
- Real-time email monitoring
- Multi-language email support

## License

This project is intended for educational and portfolio purposes.
