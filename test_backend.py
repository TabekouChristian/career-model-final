#!/usr/bin/env python3
"""
Test Backend for Final Career Model
Uses the working final_career_model.pkl
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Global model data
model_data = None

def load_model():
    """Load the final working model"""
    global model_data

    # Try to find the model file
    model_paths = [
        "improved_quick_career_model.pkl",
        "final_career_model.pkl"
    ]
    
    model_file = None
    for path in model_paths:
        if os.path.exists(path):
            model_file = path
            break

    if not model_file:
        print("❌ Model file not found in any location")
        return False

    try:
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        print("✅ Model loaded successfully!")
        print(f"📊 Careers: {len(model_data['career_names'])}")
        print(f"📊 Version: {model_data.get('model_version', 'unknown')}")
        print(f"📊 Accuracy: {model_data['performance']['test_accuracy']:.1%}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def create_features(subjects, interests):
    """Create comprehensive features that match the improved quick model exactly"""

    # All 32 Cameroon GCE subjects
    all_subjects = [
        "English", "French", "General Paper", "Religious Studies",
        "Philosophy", "Logic", "Mathematics", "Further Mathematics",
        "Physics", "Chemistry", "Biology", "Computer Science",
        "Ict", "Geology", "Technical Drawing", "Food Science",
        "Nutrition", "Agricultural Science", "Physical Education", "Environmental Management",
        "History", "Geography", "Literature", "Education",
        "Art", "Music", "Economics", "Accounting",
        "Business Mathematics", "Management", "Law", "Commerce"
    ]

    features = {}

    # Subject features (32 features) - exactly like improved model
    for subject in all_subjects:
        feature_name = f'subject_{subject.lower().replace(" ", "_")}'
        features[feature_name] = 1 if subject in subjects else 0

    # Interest categories (comprehensive mapping)
    interest_categories = [
        'analytical_thinking', 'problem_solving', 'helping_others', 'healthcare',
        'teaching', 'mentoring', 'communication', 'business', 'entrepreneurship',
        'leadership', 'technical_skills', 'engineering', 'creative_arts', 'design',
        'writing', 'literature', 'travel', 'law', 'justice', 'social_impact',
        'finance', 'outdoor_work', 'nature', 'management', 'organization',
        'biology', 'science', 'research', 'discovery', 'economics', 'trade',
        'technology', 'programming', 'media', 'entertainment', 'security'
    ]

    # Interest mapping
    interest_mapping = {
        1: "analytical_thinking,problem_solving",
        2: "helping_others,healthcare",
        3: "teaching,mentoring,communication",
        4: "business,entrepreneurship,leadership",
        5: "technical_skills,engineering",
        6: "creative_arts,design",
        7: "writing,communication,literature",
        8: "travel,cultural_awareness",
        9: "law,justice,social_impact",
        10: "finance,analytical_thinking",
        11: "outdoor_work,nature",
        12: "social_impact,community_service",
        13: "management,leadership,organization",
        14: "healthcare,biology,science",
        15: "engineering,construction,design",
        16: "security,law_enforcement",
        17: "technology,programming",
        18: "research,science,discovery",
        19: "economics,business,trade",
        20: "digital_media,content_creation",
        21: "animal_care,veterinary",
        22: "fashion,beauty,personal_care",
        23: "counseling,psychology,helping_others",
        24: "mathematics,analytical_thinking",
        25: "media,entertainment",
        26: "environmental_science,sustainability",
        27: "electronics,technology",
        28: "child_education,teaching",
        29: "aerospace,aviation,exploration",
        30: "artificial_intelligence,robotics"
    }

    # Calculate interest scores
    interest_scores = {cat: 0 for cat in interest_categories}

    for q_id, answer in interests.items():
        if answer and int(q_id) in interest_mapping:
            mapped_interests = interest_mapping[int(q_id)].split(',')
            for interest in mapped_interests:
                if interest in interest_scores:
                    interest_scores[interest] += 1

    # Normalize and add interest features
    max_score = max(interest_scores.values()) if max(interest_scores.values()) > 0 else 1
    for interest in interest_categories:
        feature_name = f'interest_{interest}'
        features[feature_name] = interest_scores[interest] / max_score

    # Advanced interaction features (key for accuracy)
    features['stem_analytical'] = (
        features['subject_mathematics'] + features['subject_physics'] + features['subject_chemistry']
    ) * features['interest_analytical_thinking']

    features['tech_programming'] = (
        features['subject_computer_science'] + features['subject_ict']
    ) * features['interest_technology']

    features['health_helping'] = (
        features['subject_biology'] + features['subject_chemistry']
    ) * features['interest_helping_others']

    features['business_leadership'] = (
        features['subject_economics'] + features['subject_management']
    ) * features['interest_business']

    return list(features.values())

@app.route('/')
def home():
    """Home page"""
    return """
    <h1>🚀 Career Recommendation API</h1>
    <p>✅ Final working model loaded!</p>
    <p>📊 Careers: {}</p>
    <p>📊 Accuracy: {:.1%}</p>
    <p>🔗 <a href="/test">Test Interface</a></p>
    """.format(
        len(model_data['career_names']) if model_data else 0,
        model_data['performance']['test_accuracy'] if model_data else 0
    )

@app.route('/test')
def test_interface():
    """Simple test interface"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Career Recommendation System</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <!-- jsPDF for PDF generation -->
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
        <style>
            :root {
                --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                --card-bg: #ffffff;
                --text-primary: #2d3748;
                --text-secondary: #718096;
                --border-color: #e2e8f0;
                --shadow: 0 10px 25px rgba(0,0,0,0.1);
                --shadow-hover: 0 20px 40px rgba(0,0,0,0.15);
            }

            body {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                min-height: 100vh;
                color: var(--text-primary);
            }

            .hero-section {
                background: var(--primary-gradient);
                color: white;
                padding: 4rem 0 2rem;
                margin-bottom: 3rem;
                position: relative;
                overflow: hidden;
            }

            .hero-section::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="1" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
                opacity: 0.3;
            }

            .hero-content {
                position: relative;
                z-index: 2;
            }

            .main-card {
                background: var(--card-bg);
                border-radius: 20px;
                box-shadow: var(--shadow);
                border: 1px solid var(--border-color);
                overflow: hidden;
                transition: all 0.3s ease;
            }

            .main-card:hover {
                box-shadow: var(--shadow-hover);
                transform: translateY(-2px);
            }

            .form-section {
                background: #f8fafc;
                border-radius: 15px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                border: 1px solid var(--border-color);
            }

            .form-section h4 {
                color: var(--text-primary);
                font-weight: 600;
                margin-bottom: 1rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }

            .checkbox-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 0.5rem;
                max-height: 300px;
                overflow-y: auto;
                padding: 0.5rem;
                border: 1px solid var(--border-color);
                border-radius: 10px;
                background: white;
            }

            .form-check {
                margin: 0;
                padding: 0.5rem;
                border-radius: 8px;
                transition: all 0.2s ease;
            }

            .form-check:hover {
                background: #f1f5f9;
            }

            .form-check-input:checked {
                background: var(--primary-gradient);
                border-color: transparent;
            }

            .form-check-label {
                font-size: 0.9rem;
                color: var(--text-primary);
                cursor: pointer;
                margin-left: 0.5rem;
            }

            .btn-predict {
                background: var(--secondary-gradient);
                border: none;
                padding: 1rem 3rem;
                border-radius: 50px;
                font-weight: 600;
                font-size: 1.1rem;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }

            .btn-predict:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.3);
            }

            .results-section {
                background: white;
                border-radius: 15px;
                padding: 2rem;
                margin-top: 2rem;
                box-shadow: var(--shadow);
                border: 1px solid var(--border-color);
            }

            .download-section {
                background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
                border-radius: 15px;
                padding: 1.5rem;
                margin-top: 1.5rem;
                border: 1px solid var(--border-color);
                text-align: center;
            }

            .download-buttons {
                display: flex;
                gap: 1rem;
                justify-content: center;
                flex-wrap: wrap;
                margin-top: 1rem;
            }

            .btn-download {
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                border: none;
                color: white;
                font-weight: 600;
                padding: 0.75rem 1.5rem;
                border-radius: 50px;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
                display: flex;
                align-items: center;
                gap: 0.5rem;
                text-decoration: none;
                min-width: 150px;
                justify-content: center;
                cursor: pointer;
            }

            .btn-download:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4);
                color: white;
            }

            .btn-download.pdf {
                background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
                box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
            }

            .btn-download.pdf:hover {
                box-shadow: 0 8px 25px rgba(239, 68, 68, 0.4);
            }

            .career-card {
                background: linear-gradient(135deg, #fff 0%, #f8fafc 100%);
                border: 1px solid var(--border-color);
                border-radius: 12px;
                padding: 1.5rem;
                margin-bottom: 1rem;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }

            .career-card::before {
                content: '';
                position: absolute;
                left: 0;
                top: 0;
                bottom: 0;
                width: 4px;
                background: var(--primary-gradient);
            }

            .career-card:hover {
                transform: translateX(5px);
                box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            }

            .career-rank {
                background: var(--primary-gradient);
                color: white;
                width: 30px;
                height: 30px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                font-size: 0.9rem;
            }

            .confidence-bar {
                height: 8px;
                background: #e2e8f0;
                border-radius: 4px;
                overflow: hidden;
                margin-top: 0.5rem;
            }

            .confidence-fill {
                height: 100%;
                background: var(--success-gradient);
                border-radius: 4px;
                transition: width 0.8s ease;
            }

            .loading-spinner {
                display: none;
                text-align: center;
                padding: 2rem;
            }

            .spinner-border-custom {
                width: 3rem;
                height: 3rem;
                border-width: 0.3rem;
                border-color: transparent;
                border-top-color: #667eea;
                animation: spin 1s linear infinite;
            }

            @keyframes spin {
                to { transform: rotate(360deg); }
            }

            .fade-in {
                animation: fadeIn 0.6s ease-in;
            }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }

            @media (max-width: 768px) {
                .checkbox-grid {
                    grid-template-columns: 1fr;
                }

                .hero-section {
                    padding: 2rem 0 1rem;
                }

                .btn-predict {
                    width: 100%;
                    padding: 1rem;
                }
            }
        </style>
    </head>
    <body>
        <!-- Hero Section -->
        <div class="hero-section">
            <div class="container">
                <div class="hero-content text-center">
                    <h1 class="display-4 fw-bold mb-3">
                        <i class="fas fa-brain me-3"></i>AI Career Recommendation System
                    </h1>
                    <p class="lead mb-0">Discover your perfect career path based on Cameroon GCE subjects and interests</p>
                </div>
            </div>
        </div>

        <!-- Main Content -->
        <div class="container">
            <div class="main-card">
                <div class="card-body p-4">
                    <!-- Subjects Section -->
                    <div class="form-section">
                        <h4>
                            <i class="fas fa-graduation-cap text-primary"></i>
                            Select Your GCE Subjects
                        </h4>
                        <p class="text-muted mb-3">Choose the subjects you are studying or have studied</p>
                        <div class="checkbox-grid">
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="English" id="subject_english">
                                <label class="form-check-label" for="subject_english">English</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="French" id="subject_french">
                                <label class="form-check-label" for="subject_french">French</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="General Paper" id="subject_general">
                                <label class="form-check-label" for="subject_general">General Paper</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="Religious Studies" id="subject_religious">
                                <label class="form-check-label" for="subject_religious">Religious Studies</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="Philosophy" id="subject_philosophy">
                                <label class="form-check-label" for="subject_philosophy">Philosophy</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="Logic" id="subject_logic">
                                <label class="form-check-label" for="subject_logic">Logic</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="Mathematics" id="subject_mathematics">
                                <label class="form-check-label" for="subject_mathematics">Mathematics</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="Further Mathematics" id="subject_further_math">
                                <label class="form-check-label" for="subject_further_math">Further Mathematics</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="Physics" id="subject_physics">
                                <label class="form-check-label" for="subject_physics">Physics</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="Chemistry" id="subject_chemistry">
                                <label class="form-check-label" for="subject_chemistry">Chemistry</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="Biology" id="subject_biology">
                                <label class="form-check-label" for="subject_biology">Biology</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="Computer Science" id="subject_cs">
                                <label class="form-check-label" for="subject_cs">Computer Science</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="Ict" id="subject_ict">
                                <label class="form-check-label" for="subject_ict">ICT</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="Geology" id="subject_geology">
                                <label class="form-check-label" for="subject_geology">Geology</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="Technical Drawing" id="subject_tech_drawing">
                                <label class="form-check-label" for="subject_tech_drawing">Technical Drawing</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="Food Science" id="subject_food_science">
                                <label class="form-check-label" for="subject_food_science">Food Science</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="Nutrition" id="subject_nutrition">
                                <label class="form-check-label" for="subject_nutrition">Nutrition</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="Agricultural Science" id="subject_agriculture">
                                <label class="form-check-label" for="subject_agriculture">Agricultural Science</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="Physical Education" id="subject_pe">
                                <label class="form-check-label" for="subject_pe">Physical Education</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="Environmental Management" id="subject_env">
                                <label class="form-check-label" for="subject_env">Environmental Management</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="History" id="subject_history">
                                <label class="form-check-label" for="subject_history">History</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="Geography" id="subject_geography">
                                <label class="form-check-label" for="subject_geography">Geography</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="Literature" id="subject_literature">
                                <label class="form-check-label" for="subject_literature">Literature</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="Education" id="subject_education">
                                <label class="form-check-label" for="subject_education">Education</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="Art" id="subject_art">
                                <label class="form-check-label" for="subject_art">Art</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="Music" id="subject_music">
                                <label class="form-check-label" for="subject_music">Music</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="Economics" id="subject_economics">
                                <label class="form-check-label" for="subject_economics">Economics</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="Accounting" id="subject_accounting">
                                <label class="form-check-label" for="subject_accounting">Accounting</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="Business Mathematics" id="subject_business_math">
                                <label class="form-check-label" for="subject_business_math">Business Mathematics</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="Management" id="subject_management">
                                <label class="form-check-label" for="subject_management">Management</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="Law" id="subject_law">
                                <label class="form-check-label" for="subject_law">Law</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="Commerce" id="subject_commerce">
                                <label class="form-check-label" for="subject_commerce">Commerce</label>
                            </div>
                        </div>
                    </div>

                    <!-- Interest Questions Section -->
                    <div class="form-section">
                        <h4>
                            <i class="fas fa-heart text-danger"></i>
                            Interest Questions
                        </h4>
                        <p class="text-muted mb-3">Answer honestly about your interests and preferences</p>
                        <div class="checkbox-grid" style="max-height: 400px;">
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="q1">
                                <label class="form-check-label" for="q1">1. Do you enjoy solving complex mathematical problems?</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="q2">
                                <label class="form-check-label" for="q2">2. Are you interested in helping people with their health problems?</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="q3">
                                <label class="form-check-label" for="q3">3. Do you like teaching or explaining concepts to others?</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="q4">
                                <label class="form-check-label" for="q4">4. Are you interested in starting your own business?</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="q5">
                                <label class="form-check-label" for="q5">5. Do you enjoy working with machines and technical equipment?</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="q6">
                                <label class="form-check-label" for="q6">6. Are you drawn to creative arts like painting, music, or design?</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="q7">
                                <label class="form-check-label" for="q7">7. Do you enjoy writing stories, articles, or reports?</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="q8">
                                <label class="form-check-label" for="q8">8. Are you interested in traveling and learning about different cultures?</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="q9">
                                <label class="form-check-label" for="q9">9. Are you interested in law, justice, and legal matters?</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="q10">
                                <label class="form-check-label" for="q10">10. Do you enjoy working with numbers and financial data?</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="q11">
                                <label class="form-check-label" for="q11">11. Do you prefer working outdoors in nature?</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="q12">
                                <label class="form-check-label" for="q12">12. Are you passionate about making a positive impact on society?</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="q13">
                                <label class="form-check-label" for="q13">13. Do you enjoy leading teams and managing projects?</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="q14">
                                <label class="form-check-label" for="q14">14. Are you interested in biology and how the human body works?</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="q15">
                                <label class="form-check-label" for="q15">15. Do you like building or constructing things?</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="q16">
                                <label class="form-check-label" for="q16">16. Are you interested in security and protecting others?</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="q17">
                                <label class="form-check-label" for="q17">17. Do you enjoy programming and working with computers?</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="q18">
                                <label class="form-check-label" for="q18">18. Are you curious about scientific research and discoveries?</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="q19">
                                <label class="form-check-label" for="q19">19. Are you interested in economics and how markets work?</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="q20">
                                <label class="form-check-label" for="q20">20. Do you enjoy creating digital content like videos or websites?</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="q21">
                                <label class="form-check-label" for="q21">21. Do you love working with animals?</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="q22">
                                <label class="form-check-label" for="q22">22. Are you interested in fashion, beauty, or personal styling?</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="q23">
                                <label class="form-check-label" for="q23">23. Do you enjoy counseling and helping people with personal problems?</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="q24">
                                <label class="form-check-label" for="q24">24. Are you passionate about mathematics and logical thinking?</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="q25">
                                <label class="form-check-label" for="q25">25. Are you interested in media, entertainment, or journalism?</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="q26">
                                <label class="form-check-label" for="q26">26. Are you concerned about environmental issues and sustainability?</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="q27">
                                <label class="form-check-label" for="q27">27. Do you enjoy working with electronic devices and circuits?</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="q28">
                                <label class="form-check-label" for="q28">28. Do you like working with children and education?</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="q29">
                                <label class="form-check-label" for="q29">29. Are you fascinated by space, aviation, or aerospace?</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="q30">
                                <label class="form-check-label" for="q30">30. Are you interested in artificial intelligence and robotics?</label>
                            </div>
                        </div>
                    </div>

                    <!-- Submit Button -->
                    <div class="text-center mb-4">
                        <button type="button" class="btn btn-predict text-white" onclick="getRecommendations()">
                            <i class="fas fa-magic me-2"></i>
                            Get My Career Recommendations
                        </button>
                    </div>

                    <!-- Loading Spinner -->
                    <div class="loading-spinner" id="loadingSpinner">
                        <div class="spinner-border spinner-border-custom" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <p class="mt-3 text-muted">Analyzing your profile...</p>
                    </div>

                    <!-- Results Section -->
                    <div class="results-section fade-in" id="results" style="display:none;">
                        <h4 class="mb-4">
                            <i class="fas fa-trophy text-warning me-2"></i>
                            Your Personalized Career Recommendations
                        </h4>
                        <div id="recommendations"></div>

                        <!-- Download Section -->
                        <div class="download-section" id="downloadSection">
                            <h5 class="mb-3">
                                <i class="fas fa-download me-2 text-success"></i>
                                Save Your Results
                            </h5>
                            <p class="text-muted mb-3">Download your career recommendations for future reference</p>
                            <div class="download-buttons">
                                <button onclick="downloadTXT()" class="btn-download txt">
                                    <i class="fas fa-file-alt me-2"></i>
                                    Download TXT
                                </button>
                                <button onclick="downloadPDF()" class="btn-download pdf">
                                    <i class="fas fa-file-pdf me-2"></i>
                                    Download PDF
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            // Global variables to store data
            let currentRecommendations = [];
            let currentSubjects = [];
            let currentInterests = {};

            function getRecommendations() {
                // Validation
                const subjects = [];
                document.querySelectorAll('input[type="checkbox"][value]').forEach(cb => {
                    if (cb.checked) subjects.push(cb.value);
                });

                if (subjects.length === 0) {
                    showAlert('Please select at least one subject!', 'warning');
                    return;
                }

                // Get all 30 interest answers
                const interests = {};
                for (let i = 1; i <= 30; i++) {
                    const checkbox = document.getElementById('q' + i);
                    if (checkbox) {
                        interests[i.toString()] = checkbox.checked;
                    }
                }

                // Store globally for download functions
                currentSubjects = subjects;
                currentInterests = interests;

                // Show loading
                document.getElementById('loadingSpinner').style.display = 'block';
                document.getElementById('results').style.display = 'none';

                // Scroll to loading area
                document.getElementById('loadingSpinner').scrollIntoView({
                    behavior: 'smooth',
                    block: 'center'
                });

                // Make API call
                fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({subjects: subjects, interests: interests})
                })
                .then(response => response.json())
                .then(data => {
                    // Hide loading
                    document.getElementById('loadingSpinner').style.display = 'none';

                    if (data.success) {
                        displayRecommendations(data.recommendations);
                    } else {
                        showAlert('Error: ' + data.error, 'danger');
                    }
                })
                .catch(error => {
                    document.getElementById('loadingSpinner').style.display = 'none';
                    showAlert('Network error: ' + error.message, 'danger');
                });
            }

            function displayRecommendations(recommendations) {
                // Store recommendations globally for download
                currentRecommendations = recommendations;

                let html = '';

                recommendations.forEach((rec, i) => {
                    const confidence = rec.match_percentage;
                    const confidenceColor = confidence >= 50 ? 'success' : confidence >= 30 ? 'warning' : 'info';

                    html += `
                        <div class="career-card" style="animation-delay: ${i * 0.1}s">
                            <div class="d-flex align-items-center mb-3">
                                <div class="career-rank me-3">${i + 1}</div>
                                <div class="flex-grow-1">
                                    <h5 class="mb-1 fw-bold text-primary">${rec.career}</h5>
                                    <div class="d-flex align-items-center">
                                        <span class="badge bg-${confidenceColor} me-2">${confidence.toFixed(1)}% Match</span>
                                        <small class="text-muted">Confidence Level</small>
                                    </div>
                                </div>
                                <div class="text-end">
                                    <i class="fas fa-star text-warning"></i>
                                </div>
                            </div>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${confidence}%"></div>
                            </div>
                        </div>
                    `;
                });

                document.getElementById('recommendations').innerHTML = html;
                document.getElementById('results').style.display = 'block';

                // Scroll to results
                setTimeout(() => {
                    document.getElementById('results').scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }, 100);

                // Animate confidence bars
                setTimeout(() => {
                    document.querySelectorAll('.confidence-fill').forEach(bar => {
                        bar.style.width = bar.style.width;
                    });
                }, 500);
            }

            function showAlert(message, type) {
                const alertDiv = document.createElement('div');
                alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
                alertDiv.innerHTML = `
                    ${message}
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                `;

                const container = document.querySelector('.main-card .card-body');
                container.insertBefore(alertDiv, container.firstChild);

                // Auto dismiss after 5 seconds
                setTimeout(() => {
                    if (alertDiv.parentNode) {
                        alertDiv.remove();
                    }
                }, 5000);
            }

            // Add some interactive effects
            document.addEventListener('DOMContentLoaded', function() {
                // Add hover effects to checkboxes
                document.querySelectorAll('.form-check').forEach(check => {
                    check.addEventListener('mouseenter', function() {
                        this.style.transform = 'translateX(5px)';
                    });

                    check.addEventListener('mouseleave', function() {
                        this.style.transform = 'translateX(0)';
                    });
                });

                // Add click animation to button
                document.querySelector('.btn-predict').addEventListener('click', function() {
                    this.style.transform = 'scale(0.95)';
                    setTimeout(() => {
                        this.style.transform = 'scale(1)';
                    }, 150);
                });
            });

            // Download functions
            function downloadTXT() {
                if (!currentRecommendations || currentRecommendations.length === 0) {
                    showAlert('No recommendations to download!', 'warning');
                    return;
                }

                const timestamp = new Date().toLocaleString();
                const selectedSubjects = currentSubjects.join(', ');
                const selectedInterests = Object.keys(currentInterests)
                    .filter(key => currentInterests[key])
                    .map(key => `Q${key}`)
                    .join(', ');

                let content = `CAREER RECOMMENDATIONS REPORT\\n`;
                content += `Generated on: ${timestamp}\\n`;
                content += `=`.repeat(50) + '\\n\\n';

                content += `STUDENT PROFILE:\\n`;
                content += `Selected Subjects: ${selectedSubjects}\\n`;
                content += `Answered Yes to Questions: ${selectedInterests}\\n\\n`;

                content += `CAREER RECOMMENDATIONS:\\n`;
                content += `=`.repeat(30) + '\\n\\n';

                currentRecommendations.forEach((rec, i) => {
                    content += `${i + 1}. ${rec.career}\\n`;
                    content += `   Match Percentage: ${rec.match_percentage.toFixed(1)}%\\n`;
                    content += `   Confidence Level: ${getConfidenceLevel(rec.match_percentage)}\\n\\n`;
                });

                content += `\\nRECOMMENDATIONS SUMMARY:\\n`;
                content += `=`.repeat(25) + '\\n';
                content += `Total Recommendations: ${currentRecommendations.length}\\n`;
                content += `Best Match: ${currentRecommendations[0].career} (${currentRecommendations[0].match_percentage.toFixed(1)}%)\\n`;
                content += `Average Match: ${(currentRecommendations.reduce((sum, rec) => sum + rec.match_percentage, 0) / currentRecommendations.length).toFixed(1)}%\\n\\n`;

                content += `Generated by AI Career Recommendation System\\n`;
                content += `Powered by Advanced Machine Learning\\n`;

                const blob = new Blob([content], { type: 'text/plain' });
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `career-recommendations-${new Date().toISOString().split('T')[0]}.txt`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);

                showAlert('TXT file downloaded successfully!', 'success');
            }

            function downloadPDF() {
                if (!currentRecommendations || currentRecommendations.length === 0) {
                    showAlert('No recommendations to download!', 'warning');
                    return;
                }

                const { jsPDF } = window.jspdf;
                const doc = new jsPDF();

                // Set up the document
                const pageWidth = doc.internal.pageSize.width;
                const margin = 20;
                let yPosition = 30;

                // Title
                doc.setFontSize(20);
                doc.setFont(undefined, 'bold');
                doc.text('CAREER RECOMMENDATIONS REPORT', pageWidth / 2, yPosition, { align: 'center' });

                yPosition += 20;
                doc.setFontSize(12);
                doc.setFont(undefined, 'normal');
                doc.text(`Generated on: ${new Date().toLocaleString()}`, pageWidth / 2, yPosition, { align: 'center' });

                yPosition += 30;

                // Student Profile Section
                doc.setFontSize(14);
                doc.setFont(undefined, 'bold');
                doc.text('STUDENT PROFILE', margin, yPosition);

                yPosition += 15;
                doc.setFontSize(10);
                doc.setFont(undefined, 'normal');

                const selectedSubjects = currentSubjects.join(', ');
                const selectedInterests = Object.keys(currentInterests)
                    .filter(key => currentInterests[key])
                    .map(key => `Q${key}`)
                    .join(', ');

                doc.text(`Selected Subjects: ${selectedSubjects}`, margin, yPosition);
                yPosition += 10;
                doc.text(`Answered Yes to Questions: ${selectedInterests}`, margin, yPosition);

                yPosition += 25;

                // Recommendations Section
                doc.setFontSize(14);
                doc.setFont(undefined, 'bold');
                doc.text('CAREER RECOMMENDATIONS', margin, yPosition);

                yPosition += 15;

                currentRecommendations.forEach((rec, i) => {
                    if (yPosition > 250) {
                        doc.addPage();
                        yPosition = 30;
                    }

                    doc.setFontSize(12);
                    doc.setFont(undefined, 'bold');
                    doc.text(`${i + 1}. ${rec.career}`, margin, yPosition);

                    yPosition += 10;
                    doc.setFontSize(10);
                    doc.setFont(undefined, 'normal');
                    doc.text(`Match Percentage: ${rec.match_percentage.toFixed(1)}%`, margin + 10, yPosition);

                    yPosition += 8;
                    doc.text(`Confidence Level: ${getConfidenceLevel(rec.match_percentage)}`, margin + 10, yPosition);

                    yPosition += 15;
                });

                // Summary Section
                if (yPosition > 200) {
                    doc.addPage();
                    yPosition = 30;
                }

                yPosition += 10;
                doc.setFontSize(14);
                doc.setFont(undefined, 'bold');
                doc.text('SUMMARY', margin, yPosition);

                yPosition += 15;
                doc.setFontSize(10);
                doc.setFont(undefined, 'normal');
                doc.text(`Total Recommendations: ${currentRecommendations.length}`, margin, yPosition);
                yPosition += 8;
                doc.text(`Best Match: ${currentRecommendations[0].career} (${currentRecommendations[0].match_percentage.toFixed(1)}%)`, margin, yPosition);
                yPosition += 8;
                doc.text(`Average Match: ${(currentRecommendations.reduce((sum, rec) => sum + rec.match_percentage, 0) / currentRecommendations.length).toFixed(1)}%`, margin, yPosition);

                // Footer
                yPosition = doc.internal.pageSize.height - 30;
                doc.setFontSize(8);
                doc.text('Generated by AI Career Recommendation System', pageWidth / 2, yPosition, { align: 'center' });
                doc.text('Powered by Advanced Machine Learning', pageWidth / 2, yPosition + 8, { align: 'center' });

                // Save the PDF
                doc.save(`career-recommendations-${new Date().toISOString().split('T')[0]}.pdf`);

                showAlert('PDF file downloaded successfully!', 'success');
            }

            function getConfidenceLevel(percentage) {
                if (percentage >= 60) return 'Very High';
                if (percentage >= 45) return 'High';
                if (percentage >= 30) return 'Medium';
                if (percentage >= 15) return 'Low';
                return 'Very Low';
            }
        </script>
    </body>
    </html>
    """

@app.route('/predict', methods=['POST'])
def predict():
    """Predict career recommendations"""
    
    if not model_data:
        return jsonify({'success': False, 'error': 'Model not loaded'})
    
    try:
        data = request.json
        subjects = data.get('subjects', [])
        interests = data.get('interests', {})
        
        # Create features
        features = create_features(subjects, interests)
        
        # Scale features
        features_scaled = model_data['scaler'].transform([features])
        
        # Get predictions
        probabilities = model_data['model'].predict_proba(features_scaled)[0]
        
        # Get top 5 recommendations
        top_indices = np.argsort(probabilities)[-5:][::-1]
        
        recommendations = []
        for idx in top_indices:
            career_name = model_data['label_encoder'].inverse_transform([idx])[0]
            confidence = probabilities[idx]
            
            recommendations.append({
                'career': career_name,
                'confidence': float(confidence),
                'match_percentage': float(confidence * 100)
            })
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'model_info': {
                'version': model_data.get('model_version', 'unknown'),
                'accuracy': model_data['performance']['test_accuracy'],
                'total_careers': len(model_data['career_names'])
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_data is not None,
        'careers': len(model_data['career_names']) if model_data else 0,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Starting Career Recommendation Backend")
    print("=" * 50)
    
    # Load the model
    if load_model():
        print("✅ Backend ready!")
        print("🌐 Test URL: http://localhost:5000/test")
        print("📡 API URL: http://localhost:5000/predict")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to start - model not loaded")
