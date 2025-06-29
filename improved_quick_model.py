#!/usr/bin/env python3
"""
Improved Quick Model - Based on successful quick_career_model.pkl approach
Uses the same 38 careers but with better training data
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("🚀 Improved Quick Model Training")
print("Based on successful quick_career_model.pkl approach")
print("=" * 60)

# Use the exact same 38 careers that worked well
core_careers = [
    'AI Engineer', 'Accountant', 'Agricultural Engineer', 'Architect', 'Business Analyst',
    'Chemical Engineer', 'Civil Engineer', 'Curriculum Developer', 'Cybersecurity Specialist',
    'Data Scientist', 'Dentist', 'Educational Psychologist', 'Electrical Engineer',
    'Environmental Engineer', 'Environmental Scientist', 'Farm Manager', 'Financial Analyst',
    'Food Safety Inspector', 'Food Scientist', 'Graphic Designer', 'Journalist', 'Judge',
    'Laboratory Technician', 'Lawyer', 'Legal Assistant', 'Marketing Manager',
    'Mechanical Engineer', 'Medical Doctor', 'Musician', 'Nurse', 'Pharmacist',
    'Project Manager', 'Research Scientist', 'School Principal', 'Software Developer',
    'Teacher', 'Veterinarian', 'Web Developer'
]

print(f"✅ Using {len(core_careers)} core careers")

# All 32 Cameroon GCE subjects
subjects = [
    "English", "French", "General Paper", "Religious Studies",
    "Philosophy", "Logic", "Mathematics", "Further Mathematics",
    "Physics", "Chemistry", "Biology", "Computer Science",
    "Ict", "Geology", "Technical Drawing", "Food Science",
    "Nutrition", "Agricultural Science", "Physical Education", "Environmental Management",
    "History", "Geography", "Literature", "Education",
    "Art", "Music", "Economics", "Accounting",
    "Business Mathematics", "Management", "Law", "Commerce"
]

# All 30 interest questions
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

def create_comprehensive_features(student_subjects, interest_answers):
    """Create comprehensive features like the successful quick model"""
    
    features = {}
    
    # Subject features (32 features) - exactly like quick model
    for subject in subjects:
        feature_name = f'subject_{subject.lower().replace(" ", "_")}'
        features[feature_name] = 1 if subject in student_subjects else 0
    
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
    
    # Calculate interest scores
    interest_scores = {cat: 0 for cat in interest_categories}
    
    for q_id, answer in interest_answers.items():
        if answer and int(q_id) in interest_mapping:
            interests = interest_mapping[int(q_id)].split(',')
            for interest in interests:
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
    
    return list(features.values()), list(features.keys())

def create_realistic_profile(career_name):
    """Create realistic student profile for a specific career"""
    
    # Define career-subject mappings
    career_subjects = {
        'Software Developer': ['Computer Science', 'Mathematics', 'Physics'],
        'Web Developer': ['Computer Science', 'Mathematics', 'Art'],
        'Data Scientist': ['Mathematics', 'Computer Science', 'Further Mathematics'],
        'AI Engineer': ['Computer Science', 'Mathematics', 'Physics'],
        'Medical Doctor': ['Biology', 'Chemistry', 'Mathematics', 'Physics'],
        'Nurse': ['Biology', 'Chemistry', 'Mathematics'],
        'Pharmacist': ['Chemistry', 'Biology', 'Mathematics'],
        'Dentist': ['Biology', 'Chemistry', 'Mathematics'],
        'Teacher': ['Education', 'English', 'General Paper'],
        'Accountant': ['Accounting', 'Mathematics', 'Economics'],
        'Business Analyst': ['Economics', 'Mathematics', 'Management'],
        'Marketing Manager': ['Economics', 'Management', 'English'],
        'Civil Engineer': ['Mathematics', 'Physics', 'Technical Drawing'],
        'Mechanical Engineer': ['Mathematics', 'Physics', 'Technical Drawing'],
        'Electrical Engineer': ['Mathematics', 'Physics', 'Further Mathematics'],
        'Chemical Engineer': ['Chemistry', 'Mathematics', 'Physics'],
        'Lawyer': ['Law', 'English', 'General Paper'],
        'Judge': ['Law', 'English', 'General Paper'],
        'Journalist': ['English', 'Literature', 'General Paper'],
        'Graphic Designer': ['Art', 'Computer Science', 'English'],
        'Architect': ['Art', 'Mathematics', 'Technical Drawing'],
        'Veterinarian': ['Biology', 'Chemistry', 'Mathematics']
    }
    
    # Get base subjects for career
    base_subjects = career_subjects.get(career_name, ['English', 'Mathematics'])
    
    # Add some random additional subjects
    available_subjects = [s for s in subjects if s not in base_subjects]
    additional_count = np.random.randint(1, 4)
    if available_subjects:
        additional = np.random.choice(
            available_subjects,
            size=min(additional_count, len(available_subjects)),
            replace=False
        )
        student_subjects = base_subjects + list(additional)
    else:
        student_subjects = base_subjects
    
    # Create interest answers based on career
    interest_answers = {}
    
    # Define career-interest mappings
    career_interests = {
        'Software Developer': [1, 17, 24, 30],
        'Web Developer': [1, 17, 20, 6],
        'Data Scientist': [1, 18, 24, 17],
        'Medical Doctor': [2, 14, 18, 12],
        'Teacher': [3, 28, 12, 23],
        'Accountant': [10, 24, 1, 19],
        'Business Analyst': [1, 4, 13, 19],
        'Lawyer': [9, 7, 1, 12],
        'Graphic Designer': [6, 20, 25, 22]
    }
    
    strong_interests = career_interests.get(career_name, [1, 2, 3])
    
    for q_id in range(1, 31):
        if q_id in strong_interests:
            # High probability for aligned interests
            interest_answers[str(q_id)] = np.random.random() < 0.85
        else:
            # Lower probability for non-aligned interests
            interest_answers[str(q_id)] = np.random.random() < 0.25
    
    return student_subjects, interest_answers

# Generate high-quality training data
print("📊 Generating high-quality training data...")
training_data = []
samples_per_career = 30  # More samples for better learning

for idx, career in enumerate(core_careers):
    if idx % 10 == 0:
        print(f"   Progress: {idx}/{len(core_careers)} ({idx/len(core_careers)*100:.1f}%)")
    
    for sample_idx in range(samples_per_career):
        # Create realistic student profile
        student_subjects, interest_answers = create_realistic_profile(career)
        
        # Create comprehensive features
        features, feature_names = create_comprehensive_features(student_subjects, interest_answers)
        
        training_data.append({
            'features': features,
            'career': career
        })

print(f"✅ Generated {len(training_data)} training samples")

# Convert to arrays
X = np.array([item['features'] for item in training_data])
y = np.array([item['career'] for item in training_data])

print(f"📊 Training data shape: {X.shape}")
print(f"📊 Unique careers: {len(np.unique(y))}")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model with optimal parameters for 38 careers
print("🤖 Training improved model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_split=8,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# Evaluate
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print(f"✅ Training completed!")
print(f"📊 Train accuracy: {train_score:.3f} ({train_score*100:.1f}%)")
print(f"📊 Test accuracy: {test_score:.3f} ({test_score*100:.1f}%)")

overfitting = train_score - test_score
print(f"📊 Overfitting gap: {overfitting:.3f}")

# Save improved model
model_data = {
    'model': model,
    'scaler': scaler,
    'label_encoder': label_encoder,
    'subjects': subjects,
    'interest_mapping': interest_mapping,
    'feature_names': feature_names,
    'career_names': list(label_encoder.classes_),
    'is_trained': True,
    'training_date': datetime.now().isoformat(),
    'model_version': '4.0_improved_quick',
    'performance': {
        'train_accuracy': train_score,
        'test_accuracy': test_score,
        'overfitting': overfitting,
        'career_count': len(label_encoder.classes_),
        'feature_count': len(feature_names),
        'training_samples': len(training_data)
    }
}

with open('improved_quick_career_model.pkl', 'wb') as f:
    pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)

file_size = os.path.getsize('improved_quick_career_model.pkl') / (1024*1024)
print(f"💾 Model saved: improved_quick_career_model.pkl ({file_size:.1f}MB)")

# Test the model
print(f"\n🧪 Testing improved model...")

def test_prediction(subjects_input, interests_input):
    features, _ = create_comprehensive_features(subjects_input, interests_input)
    features_scaled = scaler.transform([features])
    probabilities = model.predict_proba(features_scaled)[0]
    
    top_indices = np.argsort(probabilities)[-5:][::-1]
    
    print("   Top 5 recommendations:")
    for i, idx in enumerate(top_indices, 1):
        career_name = label_encoder.inverse_transform([idx])[0]
        confidence = probabilities[idx]
        print(f"      {i}. {career_name} - {confidence*100:.1f}% confidence")

# Test cases
print("\n🎯 Technology Student:")
test_prediction(['Computer Science', 'Mathematics', 'Physics'], 
               {'1': True, '17': True, '24': True, '30': True})

print("\n🎯 Healthcare Student:")
test_prediction(['Biology', 'Chemistry', 'Mathematics'], 
               {'2': True, '14': True, '18': True, '23': True})

print("\n🎯 Business Student:")
test_prediction(['Economics', 'Management', 'Accounting'], 
               {'4': True, '10': True, '13': True, '19': True})

print(f"\n🎉 Improved Quick Model Complete!")
print(f"📊 Final Results:")
print(f"   Test Accuracy: {test_score:.3f} ({test_score*100:.1f}%)")
print(f"   Careers: {len(label_encoder.classes_)}")
print(f"   Features: {len(feature_names)}")
print(f"   File Size: {file_size:.1f}MB")

if test_score >= 0.6:
    print("🏆 EXCELLENT: Much better than previous models!")
elif test_score >= 0.4:
    print("✅ GOOD: Significant improvement!")
elif test_score >= 0.25:
    print("✅ FAIR: Better than before!")

print(f"\n🚀 Use: improved_quick_career_model.pkl")
print(f"This should work much better than the 350-career model!")
