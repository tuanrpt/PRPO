# Original Copyright (c) 2023 PRIME-RL (TTRL)
# Modifications Copyright (c) 2025 Tuan Nguyen
#
# This file is modified from TTRL: https://github.com/PRIME-RL/TTRL
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from importlib.metadata import PackageNotFoundError, version

from packaging import version as vs

from verl.utils.import_utils import is_sglang_available


def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


package_name = "vllm"
package_version = get_version(package_name)
vllm_version = None

if package_version is None:
    if not is_sglang_available():
        raise ValueError(
            f"vllm version {package_version} not supported and SGLang also not Found. Currently supported "
            f"vllm versions are 0.7.0+"
        )
elif vs.parse(package_version) >= vs.parse("0.7.0"):
    vllm_version = package_version
    from vllm import LLM
    from vllm.distributed import parallel_state
else:
    if vs.parse(package_version) in [vs.parse("0.5.4"), vs.parse("0.6.3")]:
        raise ValueError(
            f"vLLM version {package_version} support has been removed. vLLM 0.5.4 and 0.6.3 are no longer "
            f"supported. Please use vLLM 0.7.0 or later."
        )
    if not is_sglang_available():
        raise ValueError(
            f"vllm version {package_version} not supported and SGLang also not Found. Currently supported "
            f"vllm versions are 0.7.0+"
        )

__all__ = ["LLM", "parallel_state"]
