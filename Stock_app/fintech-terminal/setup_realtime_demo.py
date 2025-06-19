#!/usr/bin/env python3
"""
Quick setup script to demonstrate real-time market updates
Run this to see your fintech terminal with live data updates
"""
import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running {cmd}: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Exception running {cmd}: {e}")
        return False

def main():
    print("🚀 Setting up Fintech Terminal for Real-time Updates Demo")
    print("=" * 60)
    
    # Get the current directory
    current_dir = Path(__file__).parent.absolute()
    frontend_dir = current_dir / "frontend"
    backend_dir = current_dir / "backend"
    
    print(f"Working directory: {current_dir}")
    
    # Step 1: Install frontend dependencies
    print("\n📦 Installing frontend dependencies...")
    if not run_command("npm install", cwd=frontend_dir):
        print("❌ Failed to install frontend dependencies")
        return
    print("✅ Frontend dependencies installed")
    
    # Step 2: Install backend dependencies
    print("\n🐍 Installing backend dependencies...")
    
    # Try minimal requirements first
    print("Trying minimal requirements first...")
    if not run_command("pip install -r requirements-minimal.txt", cwd=backend_dir):
        print("⚠️  Minimal requirements failed, trying individual packages...")
        
        # Install core packages individually
        essential_packages = [
            "fastapi==0.109.0",
            "uvicorn[standard]==0.27.0", 
            "pydantic==2.5.3",
            "python-dotenv==1.0.0",
            "yfinance==0.2.33",
            "aiohttp==3.9.1",
            "websockets==12.0"
        ]
        
        failed_packages = []
        for package in essential_packages:
            print(f"Installing {package}...")
            if not run_command(f"pip install {package}", cwd=backend_dir):
                failed_packages.append(package)
        
        if failed_packages:
            print(f"❌ Failed to install: {', '.join(failed_packages)}")
            print("💡 Try running QUICK_FIX_INSTALLATION.bat or see TROUBLESHOOT_INSTALLATION.md")
            return
    
    print("✅ Backend dependencies installed")
    
    # Step 3: Create environment file if it doesn't exist
    env_file = current_dir / ".env"
    if not env_file.exists():
        print("\n⚙️ Creating environment file...")
        env_content = """# Fintech Terminal Environment Variables
DATABASE_URL=postgresql://fintech:fintech123@localhost:5432/fintech_db
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your-secret-key-change-in-production
CORS_ORIGINS=http://localhost:3000
API_KEY=demo-key

# Frontend
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000
"""
        with open(env_file, "w") as f:
            f.write(env_content)
        print("✅ Environment file created")
    
    print("\n🎯 Setup complete! Now you can run the demo in two ways:")
    print("\nOption 1 - Quick Start (Frontend Only):")
    print("  1. cd frontend")
    print("  2. npm run dev")
    print("  3. Open http://localhost:3000")
    
    print("\nOption 2 - Full Stack (Recommended):")
    print("  1. Terminal 1: cd backend && uvicorn main:app --reload")
    print("  2. Terminal 2: cd frontend && npm run dev")
    print("  3. Open http://localhost:3000")
    
    print("\nOption 3 - One-Click Windows:")
    print("  Double-click QUICK_START.bat")
    print("\n🔥 Your terminal will show live stock updates every 5 seconds!")

if __name__ == "__main__":
    main()