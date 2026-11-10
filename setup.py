from __future__ import annotations

import os
import shutil
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

ROOT = Path(__file__).resolve().parent
_SUPPORTED_SOCS = frozenset(("ascend910b",))
_DEFAULT_SOC = "ascend910b"
_PRIVATE_OPAPI_NAME = "libvllm_ascend_hif8_opapi.so"


def selected_soc() -> str:
    soc = os.environ.get("HIF8_SOC", _DEFAULT_SOC)
    if soc not in _SUPPORTED_SOCS:
        supported = ", ".join(sorted(_SUPPORTED_SOCS))
        raise RuntimeError(f"unsupported HIF8_SOC={soc!r}; expected one of: {supported}")
    return soc


def opp_build_dir() -> Path:
    return Path(
        os.environ.get(
            "HIF8_OPP_BUILD_DIR",
            ROOT / "csrc" / "opp" / "build" / selected_soc(),
        )
    )


def vendor_payload_dir() -> Path:
    return (
        opp_build_dir()
        / "package"
        / "_CPack_Packages"
        / "Linux"
        / "External"
        / "custom_opp_ubuntu_aarch64.run"
        / "packages"
        / "vendors"
        / "vllm_ascend_hif8"
    )


def require_vendor_payload() -> Path:
    payload = vendor_payload_dir()
    if not payload.is_dir():
        raise RuntimeError(
            "packaged HiFloat8 OPP payload is missing:\n  "
            f"{payload}\n"
            f"Build csrc/opp for HIF8_SOC={selected_soc()} before building the wheel."
        )
    return payload


class Hif8Sdist(sdist):
    """Include the selected OPP payload in the source archive."""

    def make_release_tree(self, base_dir: str, files: list[str]) -> None:
        super().make_release_tree(base_dir, files)
        payload = require_vendor_payload()
        destination = (
            Path(base_dir)
            / "csrc"
            / "opp"
            / "build"
            / selected_soc()
            / "package"
            / "_CPack_Packages"
            / "Linux"
            / "External"
            / "custom_opp_ubuntu_aarch64.run"
            / "packages"
            / "vendors"
            / "vllm_ascend_hif8"
        )
        shutil.rmtree(destination, ignore_errors=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(payload, destination)


class Hif8BuildPy(build_py):
    """Stage the selected SoC's complete OPP vendor tree into the wheel."""

    def run(self) -> None:
        super().run()
        payload = require_vendor_payload()
        destination = (
            Path(self.build_lib)
            / "vllm_ascend_hif8"
            / "vendor"
            / selected_soc()
            / "vendors"
            / "vllm_ascend_hif8"
        )
        shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(payload, destination)


class Hif8BuildExt(build_ext):
    """Configure the native build only when build_ext actually runs."""

    def build_extensions(self) -> None:
        import torch
        import torch_npu
        from torch.utils.cpp_extension import include_paths, library_paths

        del torch_npu
        ascend_home = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.0.0"))
        soc = selected_soc()
        opp_build = opp_build_dir()
        torch_root = Path(torch.__file__).resolve().parent
        torch_npu_root = Path(__import__("torch_npu").__file__).resolve().parent
        custom_opapi = opp_build / "op_host" / _PRIVATE_OPAPI_NAME

        required = (
            ascend_home / "aarch64-linux" / "include",
            ascend_home / "lib64" / "libnnopbase.so",
            opp_build / "autogen",
            custom_opapi,
            torch_npu_root / "include",
            torch_npu_root / "lib" / "libtorch_npu.so",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(
                "native HiFloat8 prerequisites are missing:\n  "
                + "\n  ".join(missing)
                + f"\nBuild csrc/opp for HIF8_SOC={soc} before building the extension."
            )

        abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
        for extension in self.extensions:
            extension.include_dirs = [
                *include_paths(),
                str(torch_npu_root / "include"),
                str(ascend_home / "aarch64-linux" / "include"),
                str(opp_build / "autogen"),
                str(ROOT / "csrc" / "extension"),
            ]
            extension.library_dirs = [
                *library_paths(),
                str(torch_npu_root / "lib"),
                str(ascend_home / "lib64"),
                str(opp_build / "op_host"),
            ]
            extension.libraries = [
                "torch_python",
                "torch",
                "torch_cpu",
                "c10",
                "torch_npu",
                "vllm_ascend_hif8_opapi",
                "nnopbase",
                "ascendcl",
            ]
            extension.define_macros = [
                ("TORCH_EXTENSION_NAME", "_C"),
                ("_GLIBCXX_USE_CXX11_ABI", str(abi)),
            ]
            extension.extra_compile_args = [
                "-O3",
                "-std=c++17",
                "-fvisibility=hidden",
            ]
            extension.extra_link_args = [
                "-Wl,-rpath,$ORIGIN/lib",
                f"-Wl,-rpath,{torch_root / 'lib'}",
                f"-Wl,-rpath,{torch_npu_root / 'lib'}",
                f"-Wl,-rpath,{ascend_home / 'lib64'}",
            ]

        super().build_extensions()

        extension_path = Path(self.get_ext_fullpath("vllm_ascend_hif8._C"))
        private_lib = extension_path.parent / "lib" / custom_opapi.name
        private_lib.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(custom_opapi, private_lib)

    def copy_extensions_to_source(self) -> None:
        super().copy_extensions_to_source()
        custom_opapi = opp_build_dir() / "op_host" / _PRIVATE_OPAPI_NAME
        source_lib = ROOT / "src" / "vllm_ascend_hif8" / "lib" / custom_opapi.name
        source_lib.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(custom_opapi, source_lib)


setup(
    ext_modules=[
        Extension(
            "vllm_ascend_hif8._C",
            sources=[
                "csrc/extension/hif8_native.cpp",
                "csrc/extension/pybind.cpp",
            ],
            language="c++",
        )
    ],
    cmdclass={"build_ext": Hif8BuildExt, "build_py": Hif8BuildPy, "sdist": Hif8Sdist},
)
