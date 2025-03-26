import subprocess
from setuptools import setup
from Cython.Build import cythonize
import numpy as np
import os

def build_cython():
    """Build the Cython extension."""
    print("Building Cython extensions...")
    setup(
        ext_modules=cythonize(
            "lib/_ball_sort_game.pyx",
            compiler_directives={"language_level": "3"}
        ),
        include_dirs=[np.get_include()],  # Add NumPy include directory
        script_args=["build_ext", "--inplace"]
    )
    print("Build complete.")

def run_pytest():
    """Run pytest after building the Cython extension."""
    print("Running pytest...")
    result = subprocess.run(["pytest"], shell=True)
    if result.returncode == 0:
        print("All tests passed!")
    else:
        print("Some tests failed.")
    return result.returncode

if __name__ == "__main__":
    # Build the Cython extension
    build_cython()

    # Run pytest
    exit_code = run_pytest()

    # Exit with the pytest result code
    os._exit(exit_code)
