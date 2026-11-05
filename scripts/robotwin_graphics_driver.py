#!/usr/bin/env python3
"""Resolve installed NVIDIA ICD libraries; preserve discovery evidence before failing."""
import json
from pathlib import Path
import re
import subprocess
import sys

PATTERN = re.compile(r"lib(?:GLX_nvidia|EGL_nvidia|nvidia-[A-Za-z0-9_-]+)\.so")


def discover(out, image, candidates, listing, is_file=None):
    is_file = is_file or (lambda path: Path(path).is_file())
    out = Path(out)
    (out / "ldconfig.txt").write_text(listing)
    libraries = {}
    for line in listing.splitlines():
        if "x86-64" not in line or "=>" not in line:
            continue
        soname = line.split()[0]
        source = line.rsplit("=>", 1)[1].strip()
        if PATTERN.match(soname) and is_file(source):
            libraries.setdefault(soname, source)
    records = []
    supported = []
    for path in candidates:
        record = {"file": str(path)}
        try:
            doc = json.loads(Path(path).read_text())
            if not isinstance(doc, dict) or not isinstance(doc.get("ICD"), dict):
                raise ValueError("ICD document must contain an ICD object")
            info = doc["ICD"]
            lib = info.get("library_path", "")
            if not isinstance(lib, str):
                raise ValueError("ICD library_path must be a string")
            record.update(library_path=lib, library_arch=info.get("library_arch"))
            if "nvidia" in lib.lower() and str(info.get("library_arch")) != "32":
                source = lib if Path(lib).is_absolute() and is_file(lib) else libraries.get(Path(lib).name)
                record["resolved_library"] = source
                supported.append((Path(path), doc, source))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            record["error"] = str(exc)
        records.append(record)
    evidence = {"icd_candidates": records, "installed_libraries": libraries.copy(),
                "selection": None}
    evidence_path = out / "driver_discovery.json"

    def fail(message):
        evidence["error"] = message
        evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
        raise RuntimeError(message)

    if not supported:
        fail("No NVIDIA Vulkan ICD configuration found; see driver_discovery.json and ldconfig.txt")
    egl = libraries.get("libEGL_nvidia.so.0")
    # An absolute EGL ICD is also valid evidence when absent from ldconfig.
    if not egl:
        for _, doc, source in supported:
            if source and Path(doc["ICD"]["library_path"]).name == "libEGL_nvidia.so.0":
                egl = source
                libraries["libEGL_nvidia.so.0"] = source
                break
    if not egl:
        fail("NVIDIA EGL vendor library unavailable; see driver_discovery.json and ldconfig.txt")
    resolved = [entry for entry in supported if entry[2]]
    icd, doc, source = (resolved or supported)[0]
    if source:
        key = Path(doc["ICD"]["library_path"]).name
        libraries[key] = source
        reason = "resolved_installed_icd"
    else:
        # Use the already required NVIDIA EGL vendor library as the headless ICD.
        # The later Vulkan identity probe must still prove it enumerates the allocated GPU.
        source = egl
        reason = "original_icd_library_unresolved_using_installed_nvidia_egl"
    evidence["selection"] = {"icd_file": str(icd), "host_library": source,
                             "egl_library": egl, "reason": reason}
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
    inside = "/opt/robotwin/host-driver-" + out.name
    destination = Path(image) / inside.lstrip("/")
    destination.mkdir(parents=True, exist_ok=True)
    bindings = []
    for soname, library in sorted(libraries.items()):
        (destination / soname).touch(exist_ok=True)
        bindings.append(library + "\t" + inside + "/" + soname)
    (out / "driver_binds.tsv").write_text("\n".join(bindings) + "\n")
    (out / "host_icd_path.txt").write_text(str(icd) + "\n")
    (out / "host_icd_library.txt").write_text(source + "\n")
    (out / "driver_inside_path.txt").write_text(inside + "\n")
    host = {"file_format_version": doc.get("file_format_version", "1.0.0"),
            "ICD": dict(doc["ICD"], library_path=source)}
    container = {"file_format_version": host["file_format_version"],
                 "ICD": dict(doc["ICD"], library_path=inside + "/libEGL_nvidia.so.0")}
    (out / "host_nvidia_icd.json").write_text(json.dumps(host, indent=2))
    (out / "container_nvidia_icd.json").write_text(json.dumps(container, indent=2))
    (out / "container_nvidia_egl.json").write_text(json.dumps({
        "file_format_version": "1.0.0", "ICD": {"library_path": inside + "/libEGL_nvidia.so.0"}}))
    print("NVIDIA graphics libraries: {}; selection={}".format(len(bindings), reason), flush=True)
    return evidence


def main():
    import os
    import shutil

    out, image = map(Path, sys.argv[1:])
    out.mkdir(parents=True, exist_ok=True)
    candidates = []
    for directory in ("/etc/vulkan/icd.d", "/usr/share/vulkan/icd.d", "/usr/local/share/vulkan/icd.d"):
        candidates.extend(sorted(Path(directory).glob("*.json")))
    for variable in ("VK_ICD_FILENAMES", "VK_DRIVER_FILES"):
        candidates.extend(Path(p) for p in os.environ.get(variable, "").split(":") if p)
    listing = subprocess.check_output([shutil.which("ldconfig") or "/sbin/ldconfig", "-p"],
                                     universal_newlines=True)
    discover(out, image, candidates, listing)


if __name__ == "__main__":
    main()
