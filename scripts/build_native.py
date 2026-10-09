#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
import shutil
import shlex
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def split_env_args(value: str | None) -> list[str]:
    return shlex.split(value or "")


@contextmanager
def metal_shader_compatibility(source: Path, enabled: bool):
    """Work around the pinned ggml's invalid float -> bfloat4 fill cast.

    TC is float for scalar kernels and float4 for vector kernels. Broadcast
    through TC before converting to T, as the other unary operations do.
    Restore the checkout afterwards; only the embedded shader needs this fix.
    """
    if not enabled:
        yield
        return
    shader = source / "ggml/src/ggml-metal/kernels/unary.metal"
    original = shader.read_text()
    old = "dst_ptr[i0] = (T) args.val;"
    new = "dst_ptr[i0] = (T) ((TC) args.val);"
    if new in original:
        yield
        return
    if original.count(old) != 1:
        raise SystemExit("Metal fill shader changed; review the BF16 compatibility fix for this native revision")
    try:
        shader.write_text(original.replace(old, new))
        yield
    finally:
        shader.write_text(original)


@contextmanager
def native_logging_compatibility(source: Path):
    """Preserve an embedding application's GGML callback during backend init.

    The pinned native source replaces it unconditionally on first init. Keep
    its existing deduplicating logger when GGML still has the default callback.
    Restore the checkout after building; no native source changes are committed.
    """
    header = source / "src/backend.h"
    original = header.read_text()
    old = "        ggml_log_set(qt_ggml_log, nullptr);"
    new = (
        "        // ggml's default callback is exported by this pinned revision.\n"
        "        ggml_log_callback callback = nullptr;\n"
        "        void * user_data = nullptr;\n"
        "        ggml_log_get(&callback, &user_data);\n"
        "        if (callback == ggml_log_callback_default) {\n"
        "            ggml_log_set(qt_ggml_log, nullptr);\n"
        "        }"
    )
    if original.count(old) != 1:
        raise SystemExit("Native backend logging changed; review the logging compatibility fix for this revision")
    declaration = "extern \"C\" void ggml_log_callback_default(enum ggml_log_level, const char *, void *);\n"
    marker = "static BackendPair backend_init(const char * label) {"
    if original.count(marker) != 1:
        raise SystemExit("Native backend layout changed; review the logging compatibility fix for this revision")
    try:
        header.write_text(original.replace(marker, declaration + marker).replace(old, new))
        yield
    finally:
        header.write_text(original)


@contextmanager
def native_diagnostic_compatibility(source: Path):
    """Keep a missing tokenizer visible at error log level.

    Upstream routes diagnostics through qt_log now, but reports this load
    failure at INFO. Restore the checkout after building.
    """
    path = source / "src/bpe.h"
    original = path.read_text()
    old = 'qt_log(QT_LOG_INFO, "[BPE] Tokenizer not found in %s", gguf_path);'
    new = 'qt_log(QT_LOG_ERROR, "[BPE] Tokenizer not found in %s", gguf_path);'
    if original.count(old) != 1:
        raise SystemExit("Native tokenizer diagnostic changed; review the logging compatibility fix")
    try:
        path.write_text(original.replace(old, new))
        yield
    finally:
        path.write_text(original)


def find_first(root: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(root.rglob(pattern))
        for match in matches:
            if match.is_file() or match.is_symlink():
                return match
    return None


def strip_shared_library(path: Path) -> None:
    if not sys.platform.startswith("linux"):
        return
    strip = shutil.which("strip")
    if not strip:
        return
    try:
        run([strip, "--strip-unneeded", str(path)])
    except subprocess.CalledProcessError:
        pass


def relocate_macos_libraries(copied: list[Path], build_dir: Path) -> None:
    # ggml uses versioned install names, while the package exposes unversioned
    # filenames. Resolve every alias before replacing the Mach-O load commands.
    destinations = {}
    for path in build_dir.rglob("*.dylib"):
        for dest in copied:
            original = find_first(build_dir, [dest.name])
            if original and path.resolve() == original.resolve():
                destinations[path.name] = dest.name
    for path in copied:
        dependencies = subprocess.check_output(["otool", "-L", str(path)], text=True)
        run(["install_name_tool", "-id", f"@rpath/{path.name}", str(path)])
        for line in dependencies.splitlines()[2:]:
            dependency = line.strip().split(" (", 1)[0]
            dest_name = destinations.get(Path(dependency).name)
            if dest_name:
                run(["install_name_tool", "-change", dependency,
                     f"@loader_path/{dest_name}", str(path)])
        # Changing load commands invalidates arm64's ad-hoc signature.
        run(["codesign", "--force", "--sign", "-", str(path)])


def validate_hip_build_mode(build_dir: Path) -> None:
    """Reject module backends; the package supports linked HIP backends only."""
    error = (
        "Dynamic HIP backends (GGML_BACKEND_DL=ON / libggml-hip.so modules) "
        "are not supported by this package. Rebuild with GGML_BACKEND_DL=OFF "
        "and --clean to produce the linked, versioned HIP runtime."
    )
    cache = build_dir / "CMakeCache.txt"
    if cache.is_file():
        for line in cache.read_text().splitlines():
            name, separator, value = line.partition("=")
            if separator and name.split(":", 1)[0] == "GGML_BACKEND_DL":
                value = value.upper()
                disabled = ("", "0", "OFF", "NO", "FALSE", "N", "IGNORE", "NOTFOUND")
                if value not in disabled and not value.endswith("-NOTFOUND"):
                    raise SystemExit(error)
    module = find_first(build_dir, ["libggml-hip.so"])
    versioned = find_first(build_dir, ["libggml-hip.so.0", "libggml-hip.so.*"])
    # A normal shared build also has an unversioned linker symlink. Accept
    # that alias only when it resolves to the versioned runtime we package.
    if module and (versioned is None or module.resolve() != versioned.resolve()):
        raise SystemExit(error)


def copy_shared_libraries(build_dir: Path, package_lib_dir: Path) -> None:
    hip_library = find_first(build_dir, ["libggml-hip.so.0", "libggml-hip.so.*"])
    if sys.platform.startswith("linux") and find_first(build_dir, ["libggml-hip.so", "libggml-hip.so.*"]):
        validate_hip_build_mode(build_dir)
    patchelf = shutil.which("patchelf")
    if sys.platform.startswith("linux") and hip_library and not patchelf:
        raise SystemExit("Packaging HIP libraries requires patchelf to set sibling-library RPATHs")
    package_lib_dir.mkdir(parents=True, exist_ok=True)
    for path in package_lib_dir.iterdir():
        if path.name == ".gitkeep":
            continue
        if path.is_file() or path.is_symlink():
            path.unlink()

    if sys.platform.startswith("linux"):
        libraries = [
            ("libqwen.so", ["libqwen.so", "libqwen.so.*"]),
            ("libggml-base.so.0", ["libggml-base.so.0", "libggml-base.so.*"]),
            ("libggml-cpu.so.0", ["libggml-cpu.so.0", "libggml-cpu.so.*"]),
            ("libggml-cuda.so.0", ["libggml-cuda.so.0", "libggml-cuda.so.*"]),
            ("libggml-hip.so.0", ["libggml-hip.so.0", "libggml-hip.so.*"]),
            ("libggml-vulkan.so.0", ["libggml-vulkan.so.0", "libggml-vulkan.so.*"]),
            ("libggml-sycl.so.0", ["libggml-sycl.so.0", "libggml-sycl.so.*"]),
            ("libggml.so.0", ["libggml.so.0", "libggml.so.*"]),
        ]
    elif sys.platform == "darwin":
        libraries = [
            ("libqwen.dylib", ["libqwen.dylib"]),
            ("libggml-base.dylib", ["libggml-base.dylib"]),
            ("libggml-cpu.dylib", ["libggml-cpu.dylib"]),
            ("libggml-metal.dylib", ["libggml-metal.dylib"]),
            ("libggml-blas.dylib", ["libggml-blas.dylib"]),
            ("libggml-cuda.dylib", ["libggml-cuda.dylib"]),
            ("libggml-vulkan.dylib", ["libggml-vulkan.dylib"]),
            ("libggml-sycl.dylib", ["libggml-sycl.dylib"]),
            ("libggml.dylib", ["libggml.dylib"]),
        ]
    else:
        libraries = [
            ("qwen.dll", ["qwen.dll"]),
            ("libqwen.dll", ["libqwen.dll"]),
            ("ggml-base.dll", ["ggml-base.dll"]),
            ("ggml-cpu.dll", ["ggml-cpu.dll"]),
            ("ggml-cuda.dll", ["ggml-cuda.dll"]),
            ("ggml-vulkan.dll", ["ggml-vulkan.dll"]),
            ("ggml-sycl.dll", ["ggml-sycl.dll"]),
            ("ggml.dll", ["ggml.dll"]),
        ]

    copied = []
    seen_targets = set()
    for dest_name, patterns in libraries:
        path = find_first(build_dir, patterns)
        if path is None:
            continue
        dest = package_lib_dir / dest_name
        resolved = path.resolve()
        target_key = (dest_name, resolved)
        if target_key in seen_targets:
            continue
        seen_targets.add(target_key)
        shutil.copy2(resolved, dest)
        copied.append(dest)

    if not any(p.name.startswith(("libqwen", "qwen")) for p in copied):
        raise SystemExit(f"No qwentts shared library found in {build_dir}")

    if sys.platform == "darwin":
        relocate_macos_libraries(copied, build_dir)

    if patchelf and sys.platform.startswith("linux"):
        for path in copied:
            if path.is_file() and ".so" in path.name:
                try:
                    run([patchelf, "--set-rpath", "$ORIGIN", str(path)])
                except subprocess.CalledProcessError:
                    if hip_library:
                        raise
                    pass

    if os.environ.get("QWENTTS_CPP_NO_STRIP") != "1":
        for path in copied:
            strip_shared_library(path)

    print("Copied:")
    for path in copied:
        print(f"  {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=os.environ.get("QWENTTS_CPP_SOURCE", "third_party/qwentts.cpp"))
    parser.add_argument("--build-dir", default=os.environ.get("QWENTTS_CPP_BUILD_DIR", "build/qwentts-cpp"))
    parser.add_argument(
        "--backend",
        choices=["cpu", "cuda", "hip", "metal", "vulkan", "sycl"],
        default=os.environ.get("QWENTTS_CPP_BACKEND", "metal" if sys.platform == "darwin" else "cuda"),
    )
    parser.add_argument("--cuda-compiler", default=os.environ.get("CMAKE_CUDA_COMPILER", "/usr/local/cuda/bin/nvcc"))
    parser.add_argument("--hip-compiler", default=os.environ.get("CMAKE_HIP_COMPILER"),
                        help="HIP clang++ compiler (otherwise CMake discovers it)")
    parser.add_argument("--hip-architectures", default=os.environ.get("CMAKE_HIP_ARCHITECTURES"),
                        help="GPU targets, e.g. gfx942 or a quoted semicolon-separated list")
    parser.add_argument("--rocm-path", default=os.environ.get("ROCM_PATH"),
                        help="ROCm installation prefix (otherwise upstream discovers it)")
    parser.add_argument("--cmake-arg", action="append", default=[], help="Extra CMake configure argument; repeatable")
    parser.add_argument("--target", default=os.environ.get("QWENTTS_CPP_CMAKE_TARGET", "qwen"))
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("QWENTTS_CPP_BUILD_JOBS", os.cpu_count() or 2)))
    parser.add_argument("--skip-build", action="store_true", help="Only copy shared libraries from --build-dir")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.backend == "metal" and sys.platform != "darwin":
        parser.error("The Metal backend requires macOS")
    if args.backend == "hip" and not sys.platform.startswith("linux"):
        parser.error("The HIP backend is supported on Linux")

    root = Path(__file__).resolve().parents[1]
    source = Path(args.source).resolve()
    build_dir = Path(args.build_dir).resolve()
    package_lib_dir = root / "src" / "qwentts_cpp" / "lib"

    if not args.skip_build and not source.is_dir():
        raise SystemExit(
            f"qwentts.cpp source checkout not found: {source}\n"
            "Clone it with --recurse-submodules or pass --source /path/to/qwentts.cpp."
        )

    if args.clean and build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_build:
        cmake_args = [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build_dir),
            "-DQWEN_SHARED=ON",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON",
            f"-DCMAKE_INSTALL_RPATH={'@loader_path' if sys.platform == 'darwin' else '$ORIGIN'}",
        ]
        if args.backend == "cpu":
            cmake_args.extend(["-DGGML_BLAS=OFF", "-DGGML_METAL=OFF"])
        elif args.backend == "cuda":
            cmake_args.extend(["-DGGML_CUDA=ON", f"-DCMAKE_CUDA_COMPILER={args.cuda_compiler}"])
        elif args.backend == "hip":
            cmake_args.extend([
                "-DBUILD_SHARED_LIBS=ON", "-DGGML_HIP=ON", "-DGGML_HIP_GRAPHS=ON",
                "-DGGML_BACKEND_DL=OFF",
                "-DGGML_CUDA=OFF", "-DGGML_METAL=OFF", "-DGGML_BLAS=OFF",
                "-DGGML_NATIVE=OFF", "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
            ])
            if args.hip_compiler:
                cmake_args.append(f"-DCMAKE_HIP_COMPILER={args.hip_compiler}")
            if args.hip_architectures:
                cmake_args.append(f"-DCMAKE_HIP_ARCHITECTURES={args.hip_architectures}")
            if args.rocm_path:
                cmake_args.append(f"-DCMAKE_PREFIX_PATH={args.rocm_path}")
                cmake_args.append(f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={args.rocm_path}")
        elif args.backend == "metal":
            cmake_args.extend([
                "-DGGML_METAL=ON", "-DGGML_METAL_EMBED_LIBRARY=ON",
                "-DGGML_BLAS=ON", "-DGGML_BLAS_VENDOR=Apple",
                "-DGGML_CUDA=OFF", "-DGGML_OPENMP=OFF", "-DGGML_NATIVE=OFF",
                "-DCMAKE_OSX_ARCHITECTURES=arm64",
                "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
                "-DCMAKE_OSX_DEPLOYMENT_TARGET=" + os.environ.get("MACOSX_DEPLOYMENT_TARGET", "14.0"),
            ])
        elif args.backend == "vulkan":
            cmake_args.append("-DGGML_VULKAN=ON")
        elif args.backend == "sycl":
            cmake_args.append("-DGGML_SYCL=ON")
        cmake_args.extend(split_env_args(os.environ.get("QWENTTS_CPP_CMAKE_ARGS")))
        cmake_args.extend(args.cmake_arg)

        with (native_logging_compatibility(source),
              native_diagnostic_compatibility(source),
              metal_shader_compatibility(source, args.backend == "metal")):
            run(cmake_args)
            if args.backend == "hip":
                validate_hip_build_mode(build_dir)
            run(["cmake", "--build", str(build_dir), "--target", args.target, "-j", str(args.jobs)])
    copy_shared_libraries(build_dir, package_lib_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
