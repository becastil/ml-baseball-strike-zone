# 🔧 Installation Troubleshooting Guide

## Common Issues and Solutions

### Issue 1: Python Environment Problems

**Check your Python version:**
```bash
python --version
# Should be Python 3.9+ (recommended 3.10 or 3.11)
```

**If Python version is wrong:**
```bash
# Windows
py -3.10 --version

# Use specific Python version
py -3.10 -m pip install -r requirements.txt
```

### Issue 2: Package Conflicts

The current requirements.txt has some problematic packages. Try this **simplified version first:**

```bash
# Create a minimal requirements file for testing
cd fintech-terminal/backend
```

### Issue 3: Windows-Specific Problems

**Common Windows issues:**
1. **Microsoft Visual C++ Build Tools missing** (for psycopg2, cryptography)
2. **Long path names** (Windows path limit)
3. **Permission issues**

**Solutions:**
```bash
# Option 1: Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Option 2: Use pre-compiled packages
pip install --only-binary=all psycopg2-binary cryptography

# Option 3: Run as Administrator
# Right-click Command Prompt → "Run as Administrator"
```

### Issue 4: Virtual Environment Issues

**Create a fresh virtual environment:**
```bash
# Navigate to project root
cd fintech-terminal

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
cd backend
pip install -r requirements.txt
```

### Issue 5: Network/Proxy Issues

**If behind corporate firewall:**
```bash
# Use trusted hosts
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt

# Or use specific index
pip install -i https://pypi.org/simple/ -r requirements.txt
```

## Quick Diagnostic Commands

**Run these to identify the specific issue:**

```bash
# 1. Check Python and pip
python --version
pip --version

# 2. Check if pip can install basic packages
pip install requests

# 3. Try installing problematic packages individually
pip install fastapi
pip install uvicorn
pip install yfinance

# 4. Check for specific error packages
pip install psycopg2-binary
pip install cryptography
```

## Step-by-Step Recovery Process

### Step 1: Clean Installation
```bash
# 1. Navigate to backend directory
cd fintech-terminal/backend

# 2. Create new virtual environment
python -m venv fresh_venv

# 3. Activate virtual environment
# Windows:
fresh_venv\Scripts\activate
# macOS/Linux:
source fresh_venv/bin/activate

# 4. Upgrade pip
python -m pip install --upgrade pip setuptools wheel
```

### Step 2: Install Core Dependencies First
```bash
# Install in order of importance
pip install fastapi
pip install uvicorn[standard]
pip install pydantic
pip install python-dotenv
pip install yfinance
pip install aiohttp
```

### Step 3: Test Basic Functionality
```bash
# Test if basic imports work
python -c "import fastapi; import uvicorn; import yfinance; print('✅ Core packages working')"
```

### Step 4: Add Additional Packages
```bash
# Only if core packages work
pip install pandas numpy
pip install sqlalchemy
pip install pytest
```

## Alternative: Docker Installation

**If Python installation keeps failing, use Docker:**

```bash
# Navigate to project root
cd fintech-terminal

# Build and run with Docker
docker-compose up --build

# This avoids all Python environment issues
```

## Minimal Demo Version

**Create a minimal version to test:**

```bash
# Create minimal_requirements.txt with just essentials:
echo "fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
python-dotenv==1.0.0
yfinance==0.2.33
aiohttp==3.9.1" > minimal_requirements.txt

# Install minimal version
pip install -r minimal_requirements.txt
```

## Common Error Messages and Solutions

### "Microsoft Visual C++ 14.0 is required"
```bash
# Solution 1: Install Visual Studio Build Tools
# Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Solution 2: Use conda instead
conda install psycopg2 cryptography

# Solution 3: Use pre-compiled wheels
pip install --only-binary=all -r requirements.txt
```

### "Failed building wheel for cryptography"
```bash
# Update pip and setuptools
python -m pip install --upgrade pip setuptools wheel

# Install Rust (cryptography dependency)
# Download: https://rustup.rs/

# Or use pre-compiled version
pip install --only-binary=cryptography cryptography
```

### "Could not find a version that satisfies the requirement"
```bash
# Check package name spelling
pip search fastapi

# Try different Python version
py -3.10 -m pip install -r requirements.txt

# Check if package exists
pip index versions fastapi
```

### "Permission denied" or "Access denied"
```bash
# Windows: Run as Administrator
# Or install for user only
pip install --user -r requirements.txt

# Or use virtual environment (recommended)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Last Resort: Manual Installation

**If all else fails, manually install each package:**

```bash
# Copy and paste each line individually
pip install fastapi==0.109.0
pip install uvicorn[standard]==0.27.0
pip install pydantic==2.5.3
pip install python-dotenv==1.0.0
pip install yfinance==0.2.33
# ... continue with each package

# Skip problematic packages and install later
# pip install psycopg2-binary  # Skip if database not needed immediately
# pip install redis            # Skip if Redis not needed immediately
```

## Verify Installation Success

```bash
# Test if everything works
python -c "
import fastapi
import uvicorn
import yfinance
import pydantic
print('✅ All core packages installed successfully!')

# Test yfinance
import yfinance as yf
ticker = yf.Ticker('AAPL')
print('✅ Yahoo Finance connection working!')
"
```

## Contact Information

If you're still having issues, please share:
1. **Operating System**: Windows 10/11, macOS, Linux
2. **Python Version**: `python --version`
3. **Error Message**: Full error text
4. **Installation Method**: pip, conda, virtual environment

Common working configurations:
- **Windows 10/11** + **Python 3.10** + **Virtual Environment** ✅
- **Windows 11** + **Python 3.11** + **Docker** ✅
- **macOS** + **Python 3.10** + **pip** ✅