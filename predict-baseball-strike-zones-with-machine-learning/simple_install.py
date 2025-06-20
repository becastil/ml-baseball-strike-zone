import subprocess
import sys

def install_package(package):
    """Install a single package"""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} installed successfully\n")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to install {package}\n")
        return False

def main():
    print("Python 3.13 Compatible Installation")
    print("=" * 50)
    
    # Essential packages for the project
    packages = [
        "numpy",  # Will get latest version compatible with Python 3.13
        "pandas",
        "matplotlib", 
        "scikit-learn",
        "pybaseball",
        "jupyter",
        "ipykernel"
    ]
    
    failed = []
    
    for package in packages:
        if not install_package(package):
            failed.append(package)
    
    print("\n" + "=" * 50)
    print("Installation Summary:")
    print(f"✓ Successfully installed: {len(packages) - len(failed)} packages")
    
    if failed:
        print(f"✗ Failed to install: {', '.join(failed)}")
        print("\nTry installing failed packages manually with:")
        for pkg in failed:
            print(f"  pip install {pkg}")
    else:
        print("\nAll packages installed successfully!")
        print("\nNow run: python download_player_data.py")

if __name__ == "__main__":
    main()