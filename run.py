#!/usr/bin/env python3
"""
Banknote Verification System - Streamlit UI Runner
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required dependencies are installed"""
    # Map of package names to their import names
    required_packages = {
        'streamlit': 'streamlit',
        'google-generativeai': 'google.generativeai',
        'opencv-python': 'cv2',
        'Pillow': 'PIL',
        'numpy': 'numpy',
        'plotly': 'plotly',
        'python-dotenv': 'dotenv'
    }
    
    missing_packages = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ Missing dependencies:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install requirements: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def check_env_file():
    """Check if environment file exists"""
    if not os.path.exists('.env'):
        print("⚠️  .env file not found. Creating from template...")
        if os.path.exists('.env.example'):
            with open('.env.example', 'r') as example:
                with open('.env', 'w') as env_file:
                    env_file.write(example.read())
            print("✅ .env file created from template")
            print("📝 Please edit .env file with your Gemini API key")
        else:
            print("❌ .env.example not found. Please create .env file manually")
            return False
    return True

def main():
    """Main entry point"""
    print("🪙 Starting Indian Banknote Verification System...")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment file
    if not check_env_file():
        sys.exit(1)
    
    # Run Streamlit app
    print("🚀 Launching Streamlit UI...")
    print("📱 Open your browser to http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Run streamlit app
        subprocess.run(["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"])
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

if __name__ == "__main__":
    main()