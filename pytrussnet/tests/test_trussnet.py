# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests of the trussnet Python binding.

Run from the repository root (after building with -DTN_BUILD_PYTHON=ON):
    python3 -m unittest discover -s pytrussnet/tests -v
or with pytest:
    python3 -m pytest pytrussnet/tests
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
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
