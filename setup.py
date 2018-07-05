from cx_Freeze import setup, Executable
import os.path
PYTHON_INSTALL_DIR = os.path.dirname(os.path.dirname(os.__file__))
os.environ['TCL_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tcl8.6')
os.environ['TK_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tk8.6')
additional_mods = ['numpy.core._methods', 'numpy.lib.format']
setup(
    name = "Weather Prediction",
    version = "0.9",
    author = "Adam",
    author_email = "Omitted",
    options = {"build_exe": {
                        'includes':additional_mods,
                        "packages":["os"],
                        "include_files":[
                                os.path.join(PYTHON_INSTALL_DIR, 'DLLs', 'tk86t.dll'),
                                os.path.join(PYTHON_INSTALL_DIR, 'DLLs', 'tcl86t.dll')]
                    }
                },
    executables = [Executable("combine_optimize_xgboost_windows.py")],
    )