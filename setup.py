from setuptools import setup, find_namespace_packages

lewam_packages = find_namespace_packages("src", exclude=["vjepa2*", "lewam/scripts*"])

vjepa2_internal = find_namespace_packages("src/vjepa2", include=["app*", "src*"])
vjepa2_external = ["vjepa2"] + [f"vjepa2.{p}" for p in vjepa2_internal]

package_dir = {
    "": "src",
    "vjepa2": "src/vjepa2",
    "app": "src/vjepa2/app",
    "src": "src/vjepa2/src",
}

setup(
    packages=lewam_packages + vjepa2_internal + vjepa2_external,
    package_dir=package_dir,
)
