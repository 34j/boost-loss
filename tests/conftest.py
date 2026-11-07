import os
import platform
import subprocess


if platform.system() == "Darwin" and os.environ.get("CI"):
    subprocess.run(["brew", "install", "libomp"], check=True)
