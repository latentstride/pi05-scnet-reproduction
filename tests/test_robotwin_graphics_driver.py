import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from robotwin_graphics_driver import discover


class DriverDiscoveryTest(unittest.TestCase):
    def test_absolute_icd_path_uses_the_installed_soname(self):
        temporary, root, out, icd, libs, listing = self.fixture(True, True)
        with temporary:
            doc = json.loads(icd.read_text())
            doc["ICD"]["library_path"] = "/lib64/libGLX_nvidia.so.0"
            icd.write_text(json.dumps(doc))
            result = discover(out, root / "image", [icd], listing, lambda p: p in libs.values())
            self.assertEqual(result["selection"]["host_library"], libs["libGLX_nvidia.so.0"])
            self.assertEqual(result["icd_candidates"][0]["document"], doc)

    def fixture(self, glx, egl):
        temporary = TemporaryDirectory()
        root = Path(temporary.name)
        out = root / "result"
        out.mkdir()
        icd = root / "nvidia_icd.json"
        icd.write_text(json.dumps({"file_format_version": "1.0.0",
                                  "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3.0"}}))
        libs = {}
        if glx:
            libs["libGLX_nvidia.so.0"] = "/host/libGLX_nvidia.so.0"
        if egl:
            libs["libEGL_nvidia.so.0"] = "/host/libEGL_nvidia.so.0"
        listing = "\n".join("{} (libc6,x86-64) => {}".format(name, path) for name, path in libs.items())
        return temporary, root, out, icd, libs, listing

    def test_successful_prior_layout_is_preserved(self):
        temporary, root, out, icd, libs, listing = self.fixture(True, True)
        with temporary:
            result = discover(out, root / "image", [icd], listing, lambda p: p in libs.values())
            self.assertEqual(result["selection"]["host_library"], libs["libGLX_nvidia.so.0"])
            self.assertTrue((out / "driver_binds.tsv").is_file())
            container = json.loads((out / "container_nvidia_icd.json").read_text())
            self.assertTrue(container["ICD"]["library_path"].endswith("/libEGL_nvidia.so.0"))

    def test_missing_original_icd_uses_only_installed_nvidia_egl(self):
        temporary, root, out, icd, libs, listing = self.fixture(False, True)
        with temporary:
            result = discover(out, root / "image", [icd], listing, lambda p: p in libs.values())
            self.assertEqual(result["selection"]["host_library"], libs["libEGL_nvidia.so.0"])
            self.assertEqual(result["selection"]["reason"],
                             "original_icd_library_unresolved_using_installed_nvidia_egl")
            self.assertEqual(json.loads((out / "host_nvidia_icd.json").read_text())["ICD"]["library_path"],
                             libs["libEGL_nvidia.so.0"])
            self.assertIsNone(result["icd_candidates"][0]["resolved_library"])

    def test_missing_egl_fails_with_diagnostics(self):
        temporary, root, out, icd, libs, listing = self.fixture(True, False)
        with temporary:
            with self.assertRaisesRegex(RuntimeError, "EGL vendor"):
                discover(out, root / "image", [icd], listing, lambda p: p in libs.values())
            self.assertEqual((out / "ldconfig.txt").read_text(), listing)
            self.assertIn("EGL vendor", json.loads((out / "driver_discovery.json").read_text())["error"])
            self.assertFalse((out / "container_nvidia_icd.json").exists())

    def test_no_nvidia_icd_fails_with_diagnostics(self):
        temporary, root, out, icd, libs, listing = self.fixture(True, True)
        with temporary:
            with self.assertRaisesRegex(RuntimeError, "No NVIDIA"):
                discover(out, root / "image", [], listing, lambda p: p in libs.values())
            self.assertIn("error", json.loads((out / "driver_discovery.json").read_text()))

    def test_malformed_icd_keeps_diagnostic_evidence(self):
        temporary, root, out, icd, libs, listing = self.fixture(True, True)
        with temporary:
            icd.write_text('{"ICD": []}')
            with self.assertRaisesRegex(RuntimeError, "No NVIDIA"):
                discover(out, root / "image", [icd], listing, lambda p: p in libs.values())
            evidence = json.loads((out / "driver_discovery.json").read_text())
            self.assertIn("ICD object", evidence["icd_candidates"][0]["error"])


if __name__ == "__main__":
    unittest.main()
