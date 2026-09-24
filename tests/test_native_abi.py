"""Compare ctypes sizes and every field offset with the actual native header."""
import ctypes
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from qwentts_cpp._binding import QT_ABI_VERSION, QtAudio, QtInitParams, QtTTSParams, QtVoiceRef


def test_native_header_layout(tmp_path):
    source = os.environ.get("QWENTTS_CPP_SOURCE")
    if not source:
        pytest.skip("QWENTTS_CPP_SOURCE not set")
    compiler = shutil.which("cc")
    assert compiler, "A C compiler is required to validate the native ABI"
    structs = {
        "qt_audio": QtAudio,
        "qt_init_params": QtInitParams,
        "qt_tts_params": QtTTSParams,
        "qt_voice_ref": QtVoiceRef,
    }
    statements = ['printf("%d\\n", QT_ABI_VERSION);']
    expected = [QT_ABI_VERSION]
    for name, cls in structs.items():
        statements.append(f'printf("%zu\\n", sizeof(struct {name}));')
        expected.append(ctypes.sizeof(cls))
        for field, _ in cls._fields_:
            statements.append(f'printf("%zu\\n", offsetof(struct {name}, {field}));')
            expected.append(getattr(cls, field).offset)
    probe = tmp_path / "abi.c"
    probe.write_text('#include <stdio.h>\n#include <stddef.h>\n#include "qwen.h"\n'
                     + 'int main(void) {\n' + '\n'.join(statements) + '\nreturn 0;\n}\n')
    executable = tmp_path / "abi"
    subprocess.run([compiler, "-std=c99", "-I", str(Path(source).resolve() / "src"),
                    str(probe), "-o", str(executable)], check=True)
    actual = subprocess.check_output([str(executable)], text=True)
    assert [int(value) for value in actual.splitlines()] == expected
