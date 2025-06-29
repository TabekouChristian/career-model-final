#!/usr/bin/env python3
"""
Test script to debug model loading issues in Docker/Render
"""

import os
import pickle
import sys

def test_model_loading():
    """Test if models can be loaded successfully"""
    
    print("🔍 Model Loading Debug Test")
    print("=" * 50)
    
    # Check current directory
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"📁 Directory contents:")
    for item in sorted(os.listdir('.')):
        if os.path.isfile(item):
            size = os.path.getsize(item) / (1024*1024)  # MB
            print(f"   📄 {item} ({size:.1f}MB)")
        else:
            print(f"   📁 {item}/")
    
    print()
    
    # Test model files
    model_files = [
        "improved_quick_career_model.pkl",
        "final_career_model.pkl"
    ]
    
    loaded_model = None
    
    for model_file in model_files:
        print(f"🧪 Testing {model_file}...")
        
        if not os.path.exists(model_file):
            print(f"   ❌ File not found: {model_file}")
            continue
            
        try:
            file_size = os.path.getsize(model_file) / (1024*1024)
            print(f"   📊 File size: {file_size:.1f}MB")
            
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            print(f"   ✅ Successfully loaded {model_file}")
            print(f"   📊 Careers: {len(model_data.get('career_names', []))}")
            print(f"   📊 Features: {len(model_data.get('feature_names', []))}")
            print(f"   📊 Version: {model_data.get('model_version', 'unknown')}")
            
            if 'performance' in model_data:
                perf = model_data['performance']
                print(f"   📊 Accuracy: {perf.get('test_accuracy', 'unknown')}")
            
            loaded_model = model_data
            print(f"   🎯 Using {model_file} as primary model")
            break
            
        except Exception as e:
            print(f"   ❌ Failed to load {model_file}: {e}")
            print(f"   🔍 Error type: {type(e).__name__}")
    
    print()
    
    if loaded_model:
        print("✅ Model loading test PASSED")
        print("🚀 Ready for deployment")
        return True
    else:
        print("❌ Model loading test FAILED")
        print("🔧 Check model files and dependencies")
        return False

def test_dependencies():
    """Test if all required dependencies are available"""
    
    print("🔍 Dependencies Test")
    print("=" * 30)
    
    required_packages = [
        'flask',
        'flask_cors', 
        'pickle',
        'numpy',
        'sklearn'
    ]
    
    for package in required_packages:
        try:
            if package == 'pickle':
                import pickle
            elif package == 'flask':
                import flask
            elif package == 'flask_cors':
                import flask_cors
            elif package == 'numpy':
                import numpy
            elif package == 'sklearn':
                import sklearn
                
            print(f"   ✅ {package}")
        except ImportError as e:
            print(f"   ❌ {package}: {e}")
            return False
    
    print("✅ All dependencies available")
    return True

if __name__ == "__main__":
    print("🧪 Docker/Render Model Loading Test")
    print("=" * 60)
    
    # Test dependencies first
    deps_ok = test_dependencies()
    print()
    
    # Test model loading
    model_ok = test_model_loading()
    print()
    
    if deps_ok and model_ok:
        print("🎉 ALL TESTS PASSED - Ready for deployment!")
        sys.exit(0)
    else:
        print("💥 TESTS FAILED - Check issues above")
        sys.exit(1)
