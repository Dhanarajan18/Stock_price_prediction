"""
Setup script for Stock Price Prediction System.
Run this script to set up the project and verify installation.
"""

import subprocess
import sys
import os
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def check_python_version():
    """Check if Python version is compatible."""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher required!")
        print("   Please upgrade your Python installation.")
        return False
    
    print("✅ Python version compatible")
    return True


def install_dependencies():
    """Install required dependencies."""
    print_header("Installing Dependencies")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    print("Installing packages from requirements.txt...")
    print("This may take several minutes...\n")
    
    try:
        subprocess.check_call([
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            "-r", 
            str(requirements_file),
            "--quiet"
        ])
        print("\n✅ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error installing dependencies: {e}")
        print("\nTry installing manually:")
        print(f"  pip install -r {requirements_file}")
        return False


def verify_imports():
    """Verify that key packages can be imported."""
    print_header("Verifying Package Installation")
    
    packages = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'statsmodels': 'Statsmodels',
        'tensorflow': 'TensorFlow',
    }
    
    all_ok = True
    
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"✅ {name:20s} - OK")
        except ImportError:
            print(f"❌ {name:20s} - NOT FOUND")
            all_ok = False
    
    # Optional packages
    optional = {
        'nsepython': 'NSEPython',
        'pmdarima': 'pmdarima',
    }
    
    print("\nOptional packages:")
    for module, name in optional.items():
        try:
            __import__(module)
            print(f"✅ {name:20s} - OK")
        except ImportError:
            print(f"⚠️  {name:20s} - NOT FOUND (optional)")
    
    if all_ok:
        print("\n✅ All required packages verified!")
    else:
        print("\n❌ Some required packages are missing")
        print("   Try running: pip install -r requirements.txt")
    
    return all_ok


def create_directories():
    """Create necessary directories."""
    print_header("Creating Project Directories")
    
    base_dir = Path(__file__).parent
    dirs = ['data', 'models']
    
    for dir_name in dirs:
        dir_path = base_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"✅ {dir_name}/ directory ready")
    
    return True


def test_modules():
    """Test if modules can be imported."""
    print_header("Testing Project Modules")
    
    src_dir = Path(__file__).parent / "src"
    
    if not src_dir.exists():
        print("❌ src/ directory not found!")
        return False
    
    # Add src to path
    sys.path.insert(0, str(src_dir))
    
    modules = [
        'config',
        'data_fetcher',
        'preprocess',
        'features',
        'train_model',
        'predict',
        'evaluate',
        'cli_interface',
    ]
    
    all_ok = True
    
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module:20s} - OK")
        except Exception as e:
            print(f"❌ {module:20s} - ERROR: {str(e)[:40]}")
            all_ok = False
    
    if all_ok:
        print("\n✅ All project modules loaded successfully!")
    else:
        print("\n⚠️  Some modules had errors (may be due to missing dependencies)")
    
    return all_ok


def display_next_steps():
    """Display next steps for the user."""
    print_header("Setup Complete!")
    
    print("🎉 Your Stock Price Prediction System is ready to use!\n")
    
    print("Next Steps:\n")
    print("1. Run the demo:")
    print("   python demo.py\n")
    
    print("2. Use the interactive CLI:")
    print("   python src/cli_interface.py -i\n")
    
    print("3. Make a quick prediction:")
    print("   python src/cli_interface.py RELIANCE --train --days 5\n")
    
    print("4. Read the documentation:")
    print("   - README.md for full documentation")
    
    print("Popular stock symbols to try:")
    print("   RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN\n")
    
    print("="*70)


def main():
    """Main setup function."""
    print("\n" + "*" + "="*68 + "*")
    print("  Stock Price Prediction System - Setup")
    print("*" + "="*68 + "*")
    
    # Check Python version
    if not check_python_version():
        return
    
    # Ask user if they want to install dependencies
    print("\nThis setup will:")
    print("  1. Install required Python packages (~2-3 GB)")
    print("  2. Create necessary directories")
    print("  3. Verify installation")
    print("  4. Test project modules")
    
    response = input("\nDo you want to continue? (y/n): ").strip().lower()
    
    if response != 'y':
        print("\nSetup cancelled. You can run this script again anytime.")
        return
    
    # Install dependencies
    install_ok = install_dependencies()
    
    # Create directories
    create_directories()
    
    # Verify imports
    if install_ok:
        verify_imports()
    
    # Test modules
    test_modules()
    
    # Display next steps
    display_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Setup error: {str(e)}")
        import traceback
        traceback.print_exc()
