from pathlib import Path
from setuptools import setup
import sys

package_dir = Path(__file__).parent / "mlx_lm_lens"
try:
    with open("requirements.txt") as fid:
        requirements = [l.strip() for l in fid if l.strip()]
except FileNotFoundError:
    requirements = ["mlx-lm>=0.25.1", "mlx>=0.26.1"]

sys.path.append(str(package_dir))
from _version import __version__

setup(
    name="mlx-lm-lens",
    version=__version__,
    description="Find the hidden meaning of LLMs",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    author_email="goekdenizguelmez@gmail.com",
    author="Gökdeniz Gülmez",
    url="https://github.com/Goekdeniz-Guelmez/mlx-lm-lens",
    license="Apache",
    install_requires=requirements,
    packages=["mlx_lm_lens"],
    python_requires=">=3.12",
)