# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests of the trussnet Python binding.

Run from the repository root (after building with -DTN_BUILD_PYTHON=ON):
    python3 -m unittest discover -s pytrussnet/tests -v
or with pytest:
    python3 -m pytest pytrussnet/tests
"""
import glob
import os
import sys
import unittest

import numpy as np

# in-tree (a CMake build drops _trussnet*.so into pytrussnet/trussnet/); else, or with
# TN_TEST_INSTALLED=1 (the wheel tests), the installed package
_here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if not os.environ.get("TN_TEST_INSTALLED") and glob.glob(
    os.path.join(_here, "trussnet", "_trussnet*")
):
    sys.path.insert(0, _here)
import trussnet  # noqa: E402


def spheres(n=48, r_out=19.0, r_in=9.0):
    """label 1 = a ball, label 2 = a concentric inner ball (voxel centres at 0..n-1)"""
    x, y, z = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    c = (n - 1) / 2.0
    r = np.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2)
    vol = (r < r_out).astype(np.uint8)
    vol[r < r_in] = 2
    return vol, r


def tet_volumes(node, elem):
    a, b, c, d = (node[elem[:, k] - 1] for k in range(4))
    return np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) / 6.0


def enclosed_volume(node, tri):
    """divergence theorem: the volume enclosed by outward-oriented triangles"""
    a, b, c = (node[tri[:, k] - 1] for k in range(3))
    return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0


class TestVersion(unittest.TestCase):
    def test_version(self):
        from trussnet import _trussnet

        self.assertEqual(trussnet.__version__, _trussnet.__version__)  # package == compiled core


class TestLabels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vol, cls.r = spheres()
        cls.out = trussnet.tetmesh(cls.vol, size=3)

    def test_outputs(self):
        out = self.out
        self.assertEqual(set(out), {"node", "elem", "face", "info"})
        node, elem, face = out["node"], out["elem"], out["face"]
        self.assertEqual(node.dtype, np.float64)
        self.assertEqual(node.shape[1], 3)
        self.assertEqual(elem.shape[1], 5)
        self.assertEqual(face.shape[1], 5)
        self.assertGreater(len(elem), 1000)

    def test_one_based(self):
        node, elem, face = self.out["node"], self.out["elem"], self.out["face"]
        self.assertEqual(elem[:, :4].min(), 1)
        self.assertEqual(elem[:, :4].max(), len(node))
        self.assertGreaterEqual(face[:, :3].min(), 1)
        self.assertLessEqual(face[:, :3].max(), len(node))
        # every node is used
        self.assertEqual(len(np.unique(elem[:, :4])), len(node))

    def test_labels_and_conformity(self):
        info = self.out["info"]
        self.assertEqual(sorted(set(self.out["elem"][:, 4].tolist())), [1, 2])
        self.assertEqual((info["bad_faces"], info["bad_edges"], info["spanning"]), (0, 0, 0))

    def test_quality(self):
        vol = np.abs(tet_volumes(self.out["node"], self.out["elem"]))
        self.assertGreater(vol.min(), 0.0)
        self.assertGreater(self.out["info"]["min_dihedral"], 5.0)
        self.assertGreater(self.out["info"]["joe_liu_median"], 0.8)

    def test_volume_matches_labels(self):
        # meshed volume of each label near its voxel count (the mesh follows the
        # smooth sub-voxel surface, not the staircase: ~7% on the small r = 9 ball)
        node, elem = self.out["node"], self.out["elem"]
        tv = np.abs(tet_volumes(node, elem))
        for lab in (1, 2):
            vox = np.count_nonzero(self.vol == lab)
            self.assertLess(abs(tv[elem[:, 4] == lab].sum() - vox) / vox, 0.10, f"label {lab}")

    def test_face_kinds_and_orientation(self):
        node, elem, face = self.out["node"], self.out["elem"], self.out["face"]
        kinds = set(map(tuple, face[:, 3:5].tolist()))
        self.assertEqual(kinds, {(1, 0), (2, 1)})  # exterior of 1; interface 2|1, once
        tv = np.abs(tet_volumes(node, elem))
        ext = face[face[:, 4] == 0]
        self.assertAlmostEqual(enclosed_volume(node, ext) / tv.sum(), 1.0, places=6)
        inner = face[(face[:, 3] == 2) & (face[:, 4] == 1)]
        self.assertAlmostEqual(
            enclosed_volume(node, inner) / tv[elem[:, 4] == 2].sum(), 1.0, places=6
        )

    def test_nodes_in_voxel_space(self):
        node = self.out["node"]
        c = (self.vol.shape[0] - 1) / 2.0
        rn = np.linalg.norm(node - c, axis=1)
        self.assertLess(rn.max(), 19.0 + 1.0)
        self.assertGreater(rn.max(), 19.0 - 1.0)


class TestOptions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vol, _ = spheres(40, 16, 7)

    def test_faces_off(self):
        out = trussnet.tetmesh(self.vol, size=3, faces=False)
        self.assertNotIn("face", out)

    def test_size_controls_density(self):
        coarse = trussnet.tetmesh(self.vol, size=4, faces=False)
        fine = trussnet.tetmesh(self.vol, size=2.5, faces=False)
        self.assertGreater(len(fine["elem"]), 1.5 * len(coarse["elem"]))

    def test_lsize_dict_and_sequence(self):
        base = trussnet.tetmesh(self.vol, size=3, faces=False)
        d = trussnet.tetmesh(self.vol, size=3, lsize={2: 1.5}, faces=False)
        s = trussnet.tetmesh(self.vol, size=3, lsize=[0, 1.5], faces=False)
        n2 = lambda o: np.count_nonzero(o["elem"][:, 4] == 2)  # noqa: E731
        # (label 2, a small r = 7 ball, is already refined by its curvature: a
        # uniform size 1.5 gives ~2.2x; the per-label size alone ~1.7x)
        self.assertGreater(n2(d), 1.4 * n2(base))
        self.assertLess(
            np.count_nonzero(d["elem"][:, 4] == 1), 2.5 * np.count_nonzero(base["elem"][:, 4] == 1)
        )
        self.assertEqual(len(d["elem"]), len(s["elem"]))

    def test_voxelsize_scales_nodes(self):
        a = trussnet.tetmesh(self.vol, size=3, faces=False)
        b = trussnet.tetmesh(self.vol, size=6, voxelsize=2.0, faces=False)  # same mesh, 2x
        self.assertEqual(len(a["elem"]), len(b["elem"]))
        np.testing.assert_allclose(b["node"], 2.0 * a["node"], rtol=1e-5, atol=1e-4)

    def test_affine(self):
        A = np.diag([2.0, 2.0, 2.0, 1.0])
        A[:3, 3] = [10.0, -5.0, 3.0]
        a = trussnet.tetmesh(self.vol, size=3, faces=False)
        b = trussnet.tetmesh(self.vol, size=6, affine=A, faces=False)  # voxel size 2 from A
        self.assertEqual(len(a["elem"]), len(b["elem"]))
        np.testing.assert_allclose(b["node"], 2.0 * a["node"] + A[:3, 3], rtol=1e-5, atol=1e-4)

    def test_names_ignore_case_and_underscores(self):
        a = trussnet.tetmesh(self.vol, size=3, sigma_thin=0.35, faces=False)
        b = trussnet.tetmesh(self.vol, SIZE=3, sigmaThin=0.35, faces=False)
        self.assertEqual(len(a["elem"]), len(b["elem"]))

    def test_deterministic(self):
        a = trussnet.tetmesh(self.vol, size=3)
        b = trussnet.tetmesh(self.vol, size=3)
        np.testing.assert_array_equal(a["elem"], b["elem"])
        np.testing.assert_array_equal(a["node"], b["node"])


class TestSizing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vol, cls.r = spheres(40, 16, 7)
        cls.base = trussnet.tetmesh(cls.vol, size=3, faces=False)

    @staticmethod
    def count(o, lab):
        return int((o["elem"][:, 4] == lab).sum())

    def test_zero_field_is_default(self):
        out = trussnet.tetmesh(self.vol, size=3, sizing=np.zeros(self.vol.shape), faces=False)
        np.testing.assert_array_equal(out["elem"], self.base["elem"])

    def test_field_refines_locally(self):
        # refine the x < 12 side only; the gradient limit (0.3) grades back to the
        # default size within ~5 mm, so the far side (x > 26) keeps its elements
        x = np.arange(self.vol.shape[0])[:, None, None] * np.ones(self.vol.shape)
        out = trussnet.tetmesh(self.vol, size=3, sizing=np.where(x < 12, 1.5, 0.0), faces=False)
        info = out["info"]
        self.assertEqual((info["bad_faces"], info["bad_edges"], info["spanning"]), (0, 0, 0))
        cx = lambda o: o["node"][o["elem"][:, :4] - 1].mean(1)[:, 0]  # noqa: E731
        near = lambda o: int((cx(o) < 10).sum())  # noqa: E731
        far = lambda o: int((cx(o) > 26).sum())  # noqa: E731
        self.assertGreater(near(out), 3 * near(self.base))
        self.assertLess(abs(far(out) - far(self.base)) / far(self.base), 0.1)

    def test_uniform_field_sets_the_size(self):
        coarse = trussnet.tetmesh(
            self.vol, size=3, sizing=np.full(self.vol.shape, 4.0), faces=False
        )
        fine = trussnet.tetmesh(self.vol, size=3, sizing=np.full(self.vol.shape, 2.0), faces=False)
        self.assertGreater(len(fine["elem"]), 4 * len(coarse["elem"]))  # ~ (4 / 2)^3

    def test_label_vector_is_lsize(self):
        ls = trussnet.tetmesh(self.vol, size=3, lsize={2: 1.5}, faces=False)
        for vec in ([0, 1.5], [0, 0, 1.5]):  # labels 1..N, or 0..N
            out = trussnet.tetmesh(self.vol, size=3, sizing=vec, faces=False)
            np.testing.assert_array_equal(out["elem"], ls["elem"])

    def test_gray_levels(self):
        gray = 20.0 - self.r
        a = trussnet.tetmesh(gray, thresholds=[5, 12], size=3, sizing=[0, 1.5], faces=False)
        b = trussnet.tetmesh(gray, thresholds=[5, 12], size=3, lsize={2: 1.5}, faces=False)
        np.testing.assert_array_equal(a["elem"], b["elem"])

    def test_tpm_channels(self):
        tpm, _ = tpm_spheres()
        a = trussnet.tetmesh(tpm, size=3, tpm_exterior=[0], sizing=[0, 0, 1.5], faces=False)
        b = trussnet.tetmesh(tpm, size=3, tpm_exterior=[0], lsize={2: 1.5}, faces=False)
        np.testing.assert_array_equal(a["elem"], b["elem"])
        f = trussnet.tetmesh(
            tpm, size=3, tpm_exterior=[0], sizing=np.full(tpm.shape[:3], 2.0), faces=False
        )
        self.assertGreater(len(f["elem"]), len(b["elem"]))  # a field on a TPM's spatial grid

    def test_bad_length(self):
        with self.assertRaises(RuntimeError):
            trussnet.tetmesh(self.vol, sizing=[1, 2, 3, 4])


class TestGray(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, r = spheres(40)
        cls.gray = 20.0 - r  # radial ramp: level t = the sphere of radius 20 - t

    def test_single_threshold(self):
        out = trussnet.tetmesh(self.gray, thresholds=[5.0], size=3)
        self.assertEqual(set(out["elem"][:, 4].tolist()), {1})
        rn = np.linalg.norm(out["node"] - 19.5, axis=1)
        self.assertAlmostEqual(rn.max(), 15.0, delta=0.3)  # nodes on the iso-surface r = 15
        self.assertEqual(out["info"]["bad_faces"], 0)

    def test_multiple_thresholds(self):
        out = trussnet.tetmesh(self.gray, thresholds=[5.0, 10.0, 14.0], size=2)
        self.assertEqual(sorted(set(out["elem"][:, 4].tolist())), [1, 2, 3])
        kinds = set(map(tuple, out["face"][:, 3:5].tolist()))
        self.assertEqual(kinds, {(1, 0), (2, 1), (3, 2)})
        info = out["info"]
        self.assertEqual((info["bad_faces"], info["spanning"]), (0, 0))

    def test_threshold_order_irrelevant(self):
        a = trussnet.tetmesh(self.gray, thresholds=[5.0, 10.0], size=3, faces=False)
        b = trussnet.tetmesh(self.gray, thresholds=[10.0, 5.0], size=3, faces=False)
        np.testing.assert_array_equal(a["elem"], b["elem"])


def tpm_spheres(n=40, r_out=16.0, r_in=7.0, width=1.5, background=True):
    """soft TPM of two nested balls: [background,] outer shell, inner ball (logistic edges)"""
    x, y, z = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    c = (n - 1) / 2.0
    r = np.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2)
    inside_out = 1.0 / (1.0 + np.exp((r - r_out) / width * 4))  # P(r < r_out)
    inside_in = 1.0 / (1.0 + np.exp((r - r_in) / width * 4))
    ch = [1 - inside_out, inside_out - inside_in, inside_in]
    if not background:
        ch = ch[1:]
    return np.stack(ch, -1).astype(np.float32), r


class TestTpm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tpm, cls.r = tpm_spheres()
        cls.out = trussnet.tetmesh(cls.tpm, size=3, tpm_exterior=[0])

    def test_labels_and_conformity(self):
        info = self.out["info"]
        self.assertEqual(sorted(set(self.out["elem"][:, 4].tolist())), [1, 2])
        self.assertEqual((info["bad_faces"], info["bad_edges"], info["spanning"]), (0, 0, 0))

    def test_interfaces_at_half_probability(self):
        node, face = self.out["node"], self.out["face"]
        c = (self.tpm.shape[0] - 1) / 2.0
        for kind, radius in (((1, 0), 16.0), ((2, 1), 7.0)):
            f = face[(face[:, 3] == kind[0]) & (face[:, 4] == kind[1])]
            rn = np.linalg.norm(node[np.unique(f[:, :3]) - 1] - c, axis=1)
            self.assertAlmostEqual(np.median(rn), radius, delta=0.5, msg=str(kind))

    def test_volumes(self):
        # default (the argmax labels' smoothed indicators): near the soft volumes; a
        # Gaussian shrinks the small r = 7 ball by ~sigma^2 / r
        node, elem = self.out["node"], self.out["elem"]
        tv = np.abs(tet_volumes(node, elem))
        for lab, tol in ((1, 0.05), (2, 0.12)):
            soft = self.tpm[..., lab].sum()
            self.assertLess(abs(tv[elem[:, 4] == lab].sum() - soft) / soft, tol, f"label {lab}")

    def test_raw_probability_interfaces(self):
        # tpm_fields with sigma = 0: the interface at p = 0.5 exactly -> the inner
        # ball within a few % of the analytic sphere (twice closer than the default)
        out = trussnet.tetmesh(
            self.tpm, size=3, tpm_exterior=[0], tpm_fields=True, sigma=0, faces=False
        )
        tv = np.abs(tet_volumes(out["node"], out["elem"]))
        ball = tv[out["elem"][:, 4] == 2].sum()
        self.assertLess(abs(ball / (4 / 3 * np.pi * 7**3) - 1), 0.05)
        info = out["info"]
        self.assertEqual((info["bad_faces"], info["bad_edges"], info["spanning"]), (0, 0, 0))

    def test_no_exterior_channel(self):
        # tissues only: the exterior is 1 - sum(tissues)
        tpm2, _ = tpm_spheres(background=False)
        out = trussnet.tetmesh(tpm2, size=3, faces=False)
        self.assertEqual(sorted(set(out["elem"][:, 4].tolist())), [1, 2])
        self.assertLess(abs(len(out["elem"]) - len(self.out["elem"])) / len(self.out["elem"]), 0.05)

    def test_exterior_default_without_names(self):
        # unnamed channels, no tpm_exterior: every channel is tissue, the box is meshed
        out = trussnet.tetmesh(self.tpm, size=4, faces=False)
        self.assertEqual(sorted(set(out["elem"][:, 4].tolist())), [1, 2, 3])

    def test_map_merges_channels(self):
        # split the shell into two channels, merged back by tpm_map
        a = self.tpm.copy()
        half = np.zeros(a.shape[:3], bool)
        half[: a.shape[0] // 2] = True
        shell = a[..., 1]
        four = np.stack(
            [a[..., 0], np.where(half, shell, 0), np.where(half, 0, shell), a[..., 2]], -1
        )
        out = trussnet.tetmesh(four, size=3, tpm_map=[0, 1, 1, 2], faces=False)
        self.assertEqual(sorted(set(out["elem"][:, 4].tolist())), [1, 2])
        self.assertLess(abs(len(out["elem"]) - len(self.out["elem"])) / len(self.out["elem"]), 0.05)

    def test_holes(self):
        a = self.tpm.copy()
        c = a.shape[0] // 2
        a[c - 13 : c - 7, c - 3 : c + 3, c - 3 : c + 3, :] = [
            1,
            0,
            0,
        ]  # a 6^3 air pocket in the shell
        filled = trussnet.tetmesh(a, size=2, tpm_exterior=[0], faces=False)
        kept = trussnet.tetmesh(a, size=2, tpm_exterior=[0], tpm_holes=True, faces=False)
        self.assertEqual(filled["info"]["tpm_filled"], 216)
        self.assertEqual(kept["info"]["tpm_filled"], 0)
        tv = lambda o: np.abs(tet_volumes(o["node"], o["elem"])).sum()  # noqa: E731
        self.assertGreater(tv(filled) - tv(kept), 100.0)  # the kept pocket is not meshed

    def test_probability_fields(self):
        out = trussnet.tetmesh(self.tpm, size=3, tpm_exterior=[0], tpm_fields=True, faces=False)
        self.assertEqual(sorted(set(out["elem"][:, 4].tolist())), [1, 2])
        self.assertEqual(out["info"]["bad_edges"], 0)

    def test_bad_map(self):
        with self.assertRaises(RuntimeError):
            trussnet.tetmesh(self.tpm, tpm_map=[0, 1])  # 2 labels for 3 channels

    def test_file(self):
        try:
            import jdata as jd
        except ImportError:
            self.skipTest("jdata not installed")
        import tempfile

        if not jdata_writes_standard_nifti(jd):
            self.skipTest("this jdata writes NIfTI data in C order, not the standard x-fastest one")

        A = np.diag([2.0, 2.0, 2.0, 1.0])
        A[:3, 3] = [-40.0, -40.0, -40.0]
        img = np.ascontiguousarray(self.tpm[..., 1:], np.float32)  # tissues only
        h = jd.nifticreate(img, headeronly=True)
        for r, k in enumerate(("srow_x", "srow_y", "srow_z")):
            h[k][:] = A[r]
        h["pixdim"][1:4] = 2.0
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "tpm.nii.gz")
            jd.savenifti(img, fn, h)
            out = trussnet.tetmesh_file(fn, size=6)
        self.assertEqual(sorted(set(out["elem"][:, 4].tolist())), [1, 2])
        ref = trussnet.tetmesh(img, size=6, affine=A, faces=False)
        np.testing.assert_array_equal(out["elem"], ref["elem"])


def jdata_writes_standard_nifti(jd):
    """True if jd.savenifti stores the data x-fastest (the NIfTI standard); jdata up to
    0.9.5 on PyPI writes the C order, so a file of it is not the array it was given"""
    import tempfile

    a = np.arange(6, dtype=np.float32).reshape(3, 2, 1)
    with tempfile.TemporaryDirectory() as d:
        fn = os.path.join(d, "probe.nii")
        jd.savenifti(a, fn)
        with open(fn, "rb") as f:
            raw = f.read()
    off = int(np.frombuffer(raw[108:112], np.float32)[0]) or 352
    return np.array_equal(np.frombuffer(raw[off : off + a.nbytes], np.float32), a.ravel(order="F"))


def disk_labels(nx=120, ny=100):
    """labels 1 (a disk), 2 / 3 (its right / upper parts): 4 junctions"""
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    r = np.hypot(x - nx / 2, y - ny / 2)
    lab = np.zeros((nx, ny), np.uint8)
    lab[r < 40] = 1
    lab[(r < 40) & (x > nx / 2)] = 2
    lab[(r < 40) & (y > ny / 2 + 10)] = 3
    return lab, r


def signed_areas(node, elem):
    a, b, c = (node[elem[:, k] - 1] for k in range(3))
    return 0.5 * ((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])


class Test2D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lab, cls.r = disk_labels()
        cls.out = trussnet.trimesh(cls.lab, size=4)

    def test_outputs(self):
        o = self.out
        self.assertEqual(o["node"].shape[1], 2)
        self.assertEqual(o["elem"].shape[1], 4)
        self.assertEqual(o["face"].shape[1], 4)
        self.assertEqual(o["elem"][:, :3].min(), 1)
        self.assertEqual(o["elem"][:, :3].max(), len(o["node"]))

    def test_conforming_and_oriented(self):
        i = self.out["info"]
        self.assertEqual((i["bad_edges"], i["spanning"]), (0, 0))
        self.assertEqual(sorted(set(self.out["elem"][:, 3].tolist())), [1, 2, 3])
        self.assertGreater(signed_areas(self.out["node"], self.out["elem"]).min(), 0)  # all ccw
        self.assertEqual(i["junctions"], 4)
        self.assertGreater(i["min_angle"], 10)
        self.assertGreater(i["q_median"], 0.95)

    def test_areas(self):
        i = self.out["info"]
        la, lp = np.array(i["label_area"]), np.array(i["label_pixels"])
        for l in (1, 2, 3):
            self.assertLess(abs(la[l] / lp[l] - 1), 0.03, f"label {l}")

    def test_edges_enclose_the_labels(self):
        # divergence theorem: each label's boundary edges (inner on the left) enclose its area
        node, elem, face = self.out["node"], self.out["elem"], self.out["face"]
        area = signed_areas(node, elem)
        for l in (1, 2, 3):
            e = [(a, b) for a, b, i, o in face.tolist() if i == l] + [
                (b, a) for a, b, i, o in face.tolist() if o == l
            ]
            e = np.array(e) - 1
            enc = 0.5 * np.sum(
                node[e[:, 0], 0] * node[e[:, 1], 1] - node[e[:, 1], 0] * node[e[:, 0], 1]
            )
            self.assertAlmostEqual(enc / area[elem[:, 3] == l].sum(), 1.0, places=6)

    def test_gray(self):
        g = 50.0 - self.r  # radial ramp: level t = the circle of radius 50 - t
        o = trussnet.trimesh(g, thresholds=[10, 25], size=3)
        i = o["info"]
        self.assertEqual((i["bad_edges"], i["spanning"]), (0, 0))
        self.assertEqual(sorted(set(o["elem"][:, 3].tolist())), [1, 2])
        c = np.array(self.lab.shape) / 2.0
        outer = np.unique(o["face"][o["face"][:, 3] == 0][:, :2]) - 1
        self.assertAlmostEqual(
            np.median(np.linalg.norm(o["node"][outer] - c, axis=1)), 40.0, delta=0.3
        )

    def test_sizing(self):
        base = trussnet.trimesh(self.lab, size=4, faces=False)
        z = trussnet.trimesh(self.lab, size=4, sizing=np.zeros(self.lab.shape), faces=False)
        np.testing.assert_array_equal(z["elem"], base["elem"])
        v = trussnet.trimesh(self.lab, size=4, sizing=[0, 2, 0], faces=False)
        ls = trussnet.trimesh(self.lab, size=4, lsize={2: 2}, faces=False)
        np.testing.assert_array_equal(v["elem"], ls["elem"])
        n2 = lambda o: int((o["elem"][:, 3] == 2).sum())  # noqa: E731
        self.assertGreater(n2(v), 2 * n2(base))
        f = trussnet.trimesh(self.lab, size=4, sizing=np.full(self.lab.shape, 2.0), faces=False)
        self.assertGreater(len(f["elem"]), 3 * len(base["elem"]))

    def test_pixelsize_and_affine(self):
        a = trussnet.trimesh(self.lab, size=4, faces=False)
        b = trussnet.trimesh(self.lab, size=8, pixelsize=2, faces=False)
        np.testing.assert_array_equal(a["elem"], b["elem"])
        np.testing.assert_allclose(b["node"], 2 * a["node"], atol=1e-6)
        A = np.array([[2.0, 0, 5], [0, 2.0, -3]])
        c = trussnet.trimesh(self.lab, size=8, affine=A, faces=False)
        np.testing.assert_array_equal(a["elem"], c["elem"])
        np.testing.assert_allclose(c["node"], 2 * a["node"] + [5, -3], atol=1e-6)

    def test_deterministic(self):
        a = trussnet.trimesh(self.lab, size=4)
        b = trussnet.trimesh(self.lab, size=4)
        np.testing.assert_array_equal(a["elem"], b["elem"])
        np.testing.assert_array_equal(a["node"], b["node"])

    def test_errors(self):
        with self.assertRaises(ValueError):
            trussnet.trimesh(np.ones((4, 4, 4)))
        with self.assertRaises(ValueError):
            trussnet.trimesh(self.lab, bogus=1)
        with self.assertRaises(RuntimeError):
            trussnet.trimesh(np.zeros((20, 20), np.uint8))


class TestErrors(unittest.TestCase):
    def test_unknown_option(self):
        vol, _ = spheres(24, 9, 4)
        with self.assertRaises(ValueError):
            trussnet.tetmesh(vol, bogus=1)

    def test_not_3d(self):
        with self.assertRaises(ValueError):
            trussnet.tetmesh(np.ones((10, 10), np.uint8))

    def test_negative_label(self):
        vol, _ = spheres(24, 9, 4)
        vol = vol.astype(np.int16)
        vol[0, 0, 0] = -1
        with self.assertRaises(ValueError):
            trussnet.tetmesh(vol)

    def test_empty_volume(self):
        with self.assertRaises(RuntimeError):
            trussnet.tetmesh(np.zeros((16, 16, 16), np.uint8))

    def test_bad_trap(self):
        vol, _ = spheres(24, 9, 4)
        with self.assertRaises(RuntimeError):
            trussnet.tetmesh(vol, trap="nope")


class TestGpu(unittest.TestCase):
    def test_gpu_matches_cpu_quality(self):
        vol, _ = spheres()
        g = trussnet.tetmesh(vol, size=3, gpu=True)
        if not g["info"]["used_gpu"]:
            self.skipTest("no OpenCL device")
        c = trussnet.tetmesh(vol, size=3, gpu=False)
        self.assertFalse(c["info"]["used_gpu"])
        for o in (g, c):
            self.assertEqual((o["info"]["bad_faces"], o["info"]["spanning"]), (0, 0))
        # the device relaxation is not bit-identical to the CPU one: sizes agree
        self.assertLess(abs(len(g["elem"]) - len(c["elem"])) / len(c["elem"]), 0.05)


if __name__ == "__main__":
    unittest.main()
