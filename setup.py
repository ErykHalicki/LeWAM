from setuptools import setup, find_namespace_packages

wam_packages = find_namespace_packages("src", exclude=["vjepa2*", "wam/scripts*"])

vjepa2_internal = find_namespace_packages("src/vjepa2", include=["app*", "evals*", "src*"])
vjepa2_external = ["vjepa2"] + [f"vjepa2.{p}" for p in vjepa2_internal]

package_dir = {
    "": "src",
    "vjepa2": "src/vjepa2",
    "app": "src/vjepa2/app",
    "evals": "src/vjepa2/evals",
    "src": "src/vjepa2/src",
}

setup(
    packages=wam_packages + vjepa2_internal + vjepa2_external,
    package_dir=package_dir,
)
