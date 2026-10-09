"""Check backend configuration, packaging, and native build compatibility."""

from contextlib import nullcontext
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import Mock

import pytest

from scripts import build_native
from scripts.build_native import native_diagnostic_compatibility


@pytest.mark.parametrize("from_env", [False, True])
def test_hip_configuration(monkeypatch, tmp_path, from_env):
    monkeypatch.setattr(sys, "platform", "linux")
    for name in ("QWENTTS_CPP_CMAKE_ARGS", "CMAKE_HIP_COMPILER", "CMAKE_HIP_ARCHITECTURES", "ROCM_PATH"):
        monkeypatch.delenv(name, raising=False)
    argv = ["build_native.py", "--source", str(tmp_path), "--build-dir", str(tmp_path / "build")]
    if from_env:
        monkeypatch.setenv("QWENTTS_CPP_BACKEND", "hip")
        monkeypatch.setenv("CMAKE_HIP_COMPILER", "/custom/rocm/llvm/bin/clang++")
        monkeypatch.setenv("CMAKE_HIP_ARCHITECTURES", "gfx942;gfx950")
        monkeypatch.setenv("ROCM_PATH", "/custom/rocm")
        monkeypatch.setenv("QWENTTS_CPP_CMAKE_ARGS", "-DGGML_HIP_GRAPHS=OFF")
    else:
        argv += ["--backend", "hip", "--hip-compiler", "/custom/rocm/llvm/bin/clang++",
                 "--hip-architectures", "gfx942;gfx950", "--rocm-path", "/custom/rocm",
                 "--cmake-arg=-DGGML_HIP_GRAPHS=OFF"]
    monkeypatch.setattr(sys, "argv", argv)
    for name in ("native_logging_compatibility", "native_diagnostic_compatibility", "metal_shader_compatibility"):
        monkeypatch.setattr(build_native, name, lambda *args: nullcontext())
    run = Mock()
    monkeypatch.setattr(build_native, "run", run)
    copy = Mock()
    monkeypatch.setattr(build_native, "copy_shared_libraries", copy)
    assert build_native.main() == 0
    configure, build = [call.args[0] for call in run.call_args_list]
    for flag in ("-DBUILD_SHARED_LIBS=ON", "-DGGML_HIP=ON", "-DGGML_CUDA=OFF",
                 "-DGGML_BACKEND_DL=OFF",
                 "-DGGML_METAL=OFF", "-DGGML_BLAS=OFF", "-DGGML_NATIVE=OFF",
                 "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON", "-DCMAKE_INSTALL_RPATH=$ORIGIN",
                 "-DCMAKE_HIP_COMPILER=/custom/rocm/llvm/bin/clang++",
                 "-DCMAKE_HIP_ARCHITECTURES=gfx942;gfx950", "-DCMAKE_PREFIX_PATH=/custom/rocm",
                 "-DCMAKE_HIP_COMPILER_ROCM_ROOT=/custom/rocm"):
        assert flag in configure
    assert configure.index("-DGGML_HIP_GRAPHS=ON") < configure.index("-DGGML_HIP_GRAPHS=OFF")
    assert build[:2] == ["cmake", "--build"]
    copy.assert_called_once()


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_hip_rejects_unsupported_platforms(monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(sys, "argv", ["build_native.py", "--backend", "hip"])
    with pytest.raises(SystemExit) as exc:
        build_native.main()
    assert exc.value.code == 2


def test_copy_versioned_hip_library_and_relocate_siblings(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("QWENTTS_CPP_NO_STRIP", "1")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/patchelf" if name == "patchelf" else None)
    build = tmp_path / "build"
    build.mkdir()
    (build / "libqwen.so").write_bytes(b"qwen")
    versioned = build / "libggml-hip.so.0.25.3"
    versioned.write_bytes(b"hip")
    (build / "libggml-hip.so.0").symlink_to(versioned.name)
    (build / "libggml-hip.so").symlink_to("libggml-hip.so.0")
    run = Mock()
    monkeypatch.setattr(build_native, "run", run)
    package = tmp_path / "installed" / "lib"
    build_native.copy_shared_libraries(build, package)
    shutil.rmtree(build)
    assert (package / "libggml-hip.so.0").read_bytes() == b"hip"
    assert not (package / "libggml-hip.so.0").is_symlink()
    assert sorted(p.name for p in package.iterdir()) == ["libggml-hip.so.0", "libqwen.so"]
    for path in package.iterdir():
        assert any(call.args[0] == ["/usr/bin/patchelf", "--set-rpath", "$ORIGIN", str(path)]
                   for call in run.call_args_list)


@pytest.mark.parametrize("missing_tool", [False, True])
def test_hip_packaging_requires_successful_relocation(monkeypatch, tmp_path, missing_tool):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(shutil, "which", lambda name: None if missing_tool else "/usr/bin/patchelf")
    (tmp_path / "libqwen.so").touch()
    (tmp_path / "libggml-hip.so.0.25.3").touch()
    monkeypatch.setattr(build_native, "run", Mock(side_effect=subprocess.CalledProcessError(1, "patchelf")))
    with pytest.raises(SystemExit if missing_tool else subprocess.CalledProcessError):
        build_native.copy_shared_libraries(tmp_path, tmp_path / "package")


@pytest.mark.parametrize("from_env", [False, True])
def test_dynamic_hip_rejected_after_configure_before_build(monkeypatch, tmp_path, from_env):
    monkeypatch.setattr(sys, "platform", "linux")
    build = tmp_path / "build"
    argv = ["build_native.py", "--backend", "hip", "--source", str(tmp_path), "--build-dir", str(build)]
    monkeypatch.delenv("QWENTTS_CPP_CMAKE_ARGS", raising=False)
    if from_env:
        monkeypatch.setenv("QWENTTS_CPP_CMAKE_ARGS", "-DGGML_BACKEND_DL:BOOL=ON")
    else:
        argv.append("--cmake-arg=-DGGML_BACKEND_DL:BOOL=ON")
    monkeypatch.setattr(sys, "argv", argv)
    for name in ("native_logging_compatibility", "native_diagnostic_compatibility", "metal_shader_compatibility"):
        monkeypatch.setattr(build_native, name, lambda *args: nullcontext())

    def configure(command):
        assert command.index("-DGGML_BACKEND_DL=OFF") < command.index("-DGGML_BACKEND_DL:BOOL=ON")
        (build / "CMakeCache.txt").write_text("GGML_BACKEND_DL:BOOL=ON\n")

    run = Mock(side_effect=configure)
    copy = Mock()
    monkeypatch.setattr(build_native, "run", run)
    monkeypatch.setattr(build_native, "copy_shared_libraries", copy)
    with pytest.raises(SystemExit, match="Dynamic HIP backends.*not supported"):
        build_native.main()
    run.assert_called_once()
    copy.assert_not_called()


@pytest.mark.parametrize("stale_versioned", [False, True])
def test_unversioned_hip_module_rejected_without_replacing_package(monkeypatch, tmp_path, stale_versioned):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(build_native, "__file__", str(tmp_path / "project" / "scripts" / "build_native.py"))
    build = tmp_path / "build"
    build.mkdir()
    (build / "libqwen.so").write_bytes(b"new qwen")
    (build / "libggml-hip.so").write_bytes(b"HIP module")
    if stale_versioned:
        (build / "libggml-hip.so.0.25.3").write_bytes(b"old linked HIP")
    package = tmp_path / "project" / "src" / "qwentts_cpp" / "lib"
    package.mkdir(parents=True)
    (package / "libqwen.so").write_bytes(b"existing qwen")
    monkeypatch.setattr(sys, "argv", [
        "build_native.py", "--backend", "hip", "--skip-build", "--build-dir", str(build),
    ])
    run = Mock()
    monkeypatch.setattr(build_native, "run", run)
    with pytest.raises(SystemExit, match="Dynamic HIP backends.*not supported"):
        build_native.main()
    run.assert_not_called()
    assert (package / "libqwen.so").read_bytes() == b"existing qwen"
    assert [path.name for path in package.iterdir()] == ["libqwen.so"]


@pytest.mark.parametrize("value", ["ON", "YES", "1", "TRUE"])
def test_dynamic_hip_cache_rejected_even_with_versioned_library(monkeypatch, tmp_path, value):
    monkeypatch.setattr(sys, "platform", "linux")
    (tmp_path / "CMakeCache.txt").write_text(f"GGML_BACKEND_DL:BOOL={value}\n")
    (tmp_path / "libggml-hip.so.0.25.3").touch()
    with pytest.raises(SystemExit, match="Dynamic HIP backends.*not supported"):
        build_native.copy_shared_libraries(tmp_path, tmp_path / "package")
    assert not (tmp_path / "package").exists()


def test_missing_tokenizer_remains_an_error(tmp_path):
    source = os.environ.get("QWENTTS_CPP_SOURCE")
    if not source:
        pytest.skip("QWENTTS_CPP_SOURCE not set")

    shutil.copytree(Path(source).resolve() / "src", tmp_path / "src")
    with native_diagnostic_compatibility(tmp_path):
        bpe = (tmp_path / "src" / "bpe.h").read_text()
        assert 'qt_log(QT_LOG_ERROR, "[BPE] Tokenizer not found in %s", gguf_path);' in bpe
