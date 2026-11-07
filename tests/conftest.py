import os
import platform
import subprocess  # noqa: S404

if platform.system() == "Darwin" and os.environ.get("CI"):
    subprocess.run(["brew", "install", "libomp"], check=True)  # noqa: S607
