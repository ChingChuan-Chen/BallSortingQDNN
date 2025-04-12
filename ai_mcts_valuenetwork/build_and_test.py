import subprocess
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os
import sys


def build_cpp_extension():
    print("Building C++ extension...")
    ext_modules = [
        Pybind11Extension(
            "ball_sort_env",
            ["alpha_sort/lib/bindings.cpp", "alpha_sort/lib/ball_sort_env.cpp"],
            language="c++",
            include_dirs=[],
        ),
    ]

    build_dir = os.path.join("alpha_sort", "lib")
    os.makedirs(build_dir, exist_ok=True)

    setup(
        name="ball_sort_env",
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
        script_args=["build_ext", "--inplace", f"--build-lib={build_dir}"],
    )


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

    # Build C++ extension
    build_cpp_extension()

    # Run pytest
    exit_code = run_pytest()

    # Exit with the pytest result code
    os._exit(exit_code)
