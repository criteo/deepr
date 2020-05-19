import re
import subprocess
_METADATA = dict(re.findall(r'__([a-z]+)__\s*=\s*"([^"]+)"', open("deepr/version.py").read()))
version = _METADATA['version']
process = subprocess.Popen(["git", "describe", "--tags"], stdout=subprocess.PIPE)
tagged_version = process.communicate()[0].strip().decode(encoding="utf-8")
if version == tagged_version:
    print(f"Tag and version are the same ({version}) !")
    exit(0)
else:
    print(f"Tag {tagged_version} and version {version} are not the same !")
    exit(1)
