"""Exercise real ELF relocation and installed wheel loading without a GPU.

The tiny native fixtures model the GGML dependency graph, not HIP execution.
Hardware synthesis still needs the ROCm smoke test and system ROCm libraries.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from qwentts_cpp._binding import QWENTTS_NATIVE_REVISION
from scripts.build_native import copy_shared_libraries


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="ELF relocation requires Linux")
def test_hip_wheel_loads_without_source_or_build_directories(tmp_path, monkeypatch):
    compiler = shutil.which("cc")
    if not compiler or not shutil.which("patchelf"):
        pytest.skip("A C compiler and patchelf are required")
    root = Path(__file__).resolve().parents[1]
    project = tmp_path / "project"
    project.mkdir()
    for name in ("setup.py", "pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(root / name, project / name)
    shutil.copytree(root / "src" / "qwentts_cpp", project / "src" / "qwentts_cpp",
                    ignore=shutil.ignore_patterns("lib", "__pycache__"))
    build = tmp_path / "native-build"
    build.mkdir()

    def compile_library(name, source, dependencies=()):
        c_file = build / f"{name}.c"
        c_file.write_text(source)
        soname = f"lib{name}.so.0" if name != "qwen" else "libqwen.so"
        versioned = soname + ".25.3" if name != "qwen" else soname
        subprocess.run([compiler, "-shared", "-fPIC", str(c_file), "-o", str(build / versioned),
                        f"-Wl,-soname,{soname}", f"-Wl,-rpath,{build}", "-L", str(build),
                        *[f"-l{dep}" for dep in dependencies]], check=True)
        if versioned != soname:
            (build / soname).symlink_to(versioned)
            (build / f"lib{name}.so").symlink_to(soname)

    compile_library("ggml-base", "int base_marker(void) { return 20; }")
    compile_library("ggml-cpu", "extern int base_marker(void); int cpu_marker(void) { return base_marker(); }",
                    ("ggml-base",))
    compile_library("ggml-hip", "extern int base_marker(void); int hip_marker(void) { return base_marker() + 2; }",
                    ("ggml-base",))
    compile_library("ggml", "extern int hip_marker(void), cpu_marker(void); "
                    "int backend_marker(void) { return hip_marker() + cpu_marker(); }", ("ggml-hip", "ggml-cpu"))
    # Required symbols are bound but never called. Only qt_version and the
    # dependency marker run; this deliberately makes no synthesis claim.
    exports = ("qt_init_default_params", "qt_tts_default_params", "qt_init", "qt_free",
               "qt_synthesize", "qt_audio_free", "qt_log_set", "qt_duration_sec_to_tokens",
               "qt_n_languages", "qt_language_name", "qt_model_type")
    compile_library("qwen", f'const char *qt_version(void) {{ return "{QWENTTS_NATIVE_REVISION} (fixture)"; }}\n'
                    'const char *qt_last_error(void) { return ""; }\n'
                    "extern int backend_marker(void); int qt_test_marker(void) { return backend_marker(); }\n"
                    + "\n".join(f"void {symbol}(void) {{}}" for symbol in exports), ("ggml",))
    monkeypatch.setenv("QWENTTS_CPP_NO_STRIP", "1")
    package_lib = project / "src" / "qwentts_cpp" / "lib"
    copy_shared_libraries(build, package_lib)
    for library in package_lib.iterdir():
        assert subprocess.check_output(["patchelf", "--print-rpath", str(library)], text=True).strip() == "$ORIGIN"
    wheelhouse = tmp_path / "wheelhouse"
    env = os.environ.copy()
    for name in ("QWENTTS_CPP_LIBRARY", "QWEN_LIBRARY_PATH", "LD_LIBRARY_PATH", "PYTHONPATH",
                 "QWENTTS_CPP_WHEEL_BUILD_TAG"):
        env.pop(name, None)
    subprocess.run([sys.executable, "setup.py", "bdist_wheel", "--dist-dir", str(wheelhouse)],
                   cwd=project, env=env, check=True, text=True)
    wheel, = wheelhouse.glob("*.whl")
    installed = tmp_path / "installed"
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(installed), str(wheel)],
                   env=env, check=True, text=True)
    shutil.rmtree(project)
    shutil.rmtree(build)
    probe = (
        "import sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(installed)!r}); "
        "from qwentts_cpp import QwenLibrary; lib = QwenLibrary(); "
        f"assert lib.path.parent == Path({str(installed / 'qwentts_cpp' / 'lib')!r}); "
        "assert lib._lib.qt_test_marker() == 42; "
        "assert any(Path(dep._name).name == 'libggml-hip.so.0' for dep in lib._dependency_handles); "
        "print(lib.path, lib.version())"
    )
    subprocess.run([sys.executable, "-I", "-c", probe], cwd=tmp_path, env=env, check=True)
