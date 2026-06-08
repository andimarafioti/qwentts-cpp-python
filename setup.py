import os

from setuptools import Distribution, setup

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:  # pragma: no cover - older setuptools fallback
    from wheel.bdist_wheel import bdist_wheel


class BinaryDistribution(Distribution):
    """Force a platform wheel even though the Python wrapper is pure Python."""

    def has_ext_modules(self):
        return True


class PlatformWheel(bdist_wheel):
    """Build a py3-none-<platform> wheel for ctypes-bundled shared libraries."""

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False
        build_tag = os.environ.get("QWENTTS_CPP_WHEEL_BUILD_TAG")
        if build_tag and not self.build_number:
            self.build_number = build_tag

    def get_tag(self):
        _python, _abi, platform = super().get_tag()
        return "py3", "none", platform


setup(distclass=BinaryDistribution, cmdclass={"bdist_wheel": PlatformWheel})
