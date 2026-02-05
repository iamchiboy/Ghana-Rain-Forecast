"""
Quick start guide for Ghana Rain Forecast application

This script helps you set up and run the application quickly.
"""
import os
import sys
import subprocess


def run_command(command, description):
    """Run a shell command with error handling."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {command}\n")
    
    result = subprocess.run(command, shell=True)
    
    if result.returncode == 0:
        print(f"✅ {description} - SUCCESS")
        return True
    else:
        print(f"❌ {description} - FAILED")
        return False


def check_environment():
    """Check if all required files and directories exist."""
    print("\n" + "="*60)
    print("🔍 Checking environment...")
    print("="*60)
    
    required_files = [
        "openweather.env",
        "requirements.txt",
        "src/config.py",
        "src/collect_data.py",
        "src/preprocess.py",
        "src/train_model.py",
        "src/predict.py",
        "app/dashboard.py"
    ]
    
    required_dirs = ["data/raw", "data/processed", "logs"]
    
    # Check files
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
            print(f"❌ Missing: {file}")
        else:
            print(f"✅ Found: {file}")
    
    # Create directories
    for dir in required_dirs:
        os.makedirs(dir, exist_ok=True)
        print(f"✅ Directory ready: {dir}")
    
    # Check API key
    if not os.path.exists("openweather.env"):
        print("\n⚠️  WARNING: openweather.env not found!")
        print("   Please create it with: OPENWEATHER_API_KEY=your_key_here")
        return False
    
    with open("openweather.env", "r") as f:
        if "OPENWEATHER_API_KEY" not in f.read():
            print("\n⚠️  WARNING: OPENWEATHER_API_KEY not found in openweather.env!")
            return False
    
    return len(missing_files) == 0


def main():
    """Main quick start menu."""
    print("\n" + "="*60)
    print("🌧️  GHANA RAIN FORECAST - QUICK START")
    print("="*60)
    
    # Check environment
    if not check_environment():
        print("\n⚠️  Please fix the issues above before proceeding.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("📋 AVAILABLE COMMANDS")
    print("="*60)
    print("""
1. Collect Data
   python -m src.collect_data
   
2. Preprocess Data
   python -m src.preprocess
   
3. Train Model
   python -m src.train_model
   
4. Make Prediction
   python -m src.predict
   
5. Run Dashboard
   streamlit run app/dashboard.py
   
6. View Logs
   cat logs/app.log  (Linux/Mac)
   type logs\app.log (Windows)
   
7. Run Full Pipeline
   (Runs all steps: collect → preprocess → train → predict)
""")
    
    print("\n" + "="*60)
    print("⚡ QUICK SETUP (First Time Only)")
    print("="*60)
    
    response = input("\nRun full pipeline now? (y/n): ").strip().lower()
    
    if response == 'y':
        commands = [
            ("python -m src.collect_data", "Data Collection (Ctrl+C after 30 min)"),
            ("python -m src.preprocess", "Data Preprocessing"),
            ("python -m src.train_model", "Model Training"),
            ("python -m src.predict", "Make Prediction"),
        ]
        
        for i, (cmd, desc) in enumerate(commands, 1):
            print(f"\n[{i}/{len(commands)}]", end=" ")
            if not run_command(cmd, desc):
                print(f"\n⚠️  Pipeline stopped at step {i}")
                sys.exit(1)
        
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETE!")
        print("="*60)
        print("\nYour dashboard is ready. Run:")
        print("  streamlit run app/dashboard.py")
    else:
        print("\nRun commands manually as needed. See menu above.")


if __name__ == "__main__":
    main()
