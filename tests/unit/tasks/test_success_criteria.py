# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

import types
import unittest

import numpy as np

from robosuite_env.tasks.success import (
    CupOnPlateCriterion,
    make_success_criterion,
    summarize_success_records,
)


class _Obj:
    mjOBJ_JOINT = 0
    mjOBJ_BODY = 1
    mjOBJ_GEOM = 2


class _FakeMujoco:
    mjtObj = _Obj

    def __init__(self):
        self.names = {
            _Obj.mjOBJ_JOINT: {"cube_joint0": 0, "plate_joint0": 1},
            _Obj.mjOBJ_BODY: {"world": 0, "cube_main": 1, "plate_main": 2},
            _Obj.mjOBJ_GEOM: {"cube_p_c0": 0, "plate_p_c0": 1},
        }
        self.reverse = {
            typ: {idx: name for name, idx in names.items()}
            for typ, names in self.names.items()
        }

    def mj_name2id(self, _model, typ, name):
        return self.names.get(typ, {}).get(name, -1)

    def mj_id2name(self, _model, typ, idx):
        return self.reverse.get(typ, {}).get(int(idx))


class _FakeModel:
    nbody = 3
    ngeom = 2
    body_parentid = np.array([0, 0, 0], dtype=np.int32)
    jnt_bodyid = np.array([1, 2], dtype=np.int32)
    geom_bodyid = np.array([1, 2], dtype=np.int32)
    geom_contype = np.array([1, 1], dtype=np.int32)
    geom_conaffinity = np.array([1, 1], dtype=np.int32)


class _FakeData:
    def __init__(
        self,
        cup_pos=(0.0, 0.0, 0.018912),
        plate_pos=(0.0, 0.0, 0.0),
        contacts=True,
        contact_dist=-0.001,
        contact_pos=None,
    ):
        self.xpos = np.array(
            [
                [0.0, 0.0, 0.0],
                list(cup_pos),
                list(plate_pos),
            ],
            dtype=float,
        )
        eye = np.eye(3).reshape(9)
        self.xmat = np.stack([eye, eye, eye]).astype(float)
        pos = cup_pos if contact_pos is None else contact_pos
        self.contact = [types.SimpleNamespace(geom1=0, geom2=1, dist=float(contact_dist), pos=np.asarray(pos, float))] if contacts else []
        self.ncon = len(self.contact)


def _criterion(data, **options):
    return CupOnPlateCriterion(
        mujoco=_FakeMujoco(), model=_FakeModel(), data=data, options=options or None
    )


class _SubtreeModel:
    nbody = 4
    ngeom = 2
    body_parentid = np.array([0, 0, 1, 3], dtype=np.int32)
    jnt_bodyid = np.array([1, 3], dtype=np.int32)
    geom_bodyid = np.array([2, 3], dtype=np.int32)
    geom_contype = np.array([1, 1], dtype=np.int32)
    geom_conaffinity = np.array([1, 1], dtype=np.int32)


class _SubtreeMujoco(_FakeMujoco):
    def __init__(self):
        super().__init__()
        self.names[_Obj.mjOBJ_BODY] = {
            "world": 0,
            "cube_main": 1,
            "neutral_collision_child": 2,
            "plate_main": 3,
        }
        self.names[_Obj.mjOBJ_GEOM] = {"neutral_geom": 0, "plate_geom": 1}
        self.reverse = {
            typ: {idx: name for name, idx in names.items()}
            for typ, names in self.names.items()
        }


class CupOnPlateCriterionTest(unittest.TestCase):
    def test_success_when_contacting_centered_and_on_support_plane(self):
        result = _criterion(_FakeData()).evaluate()

        self.assertTrue(result["success"])
        self.assertTrue(result["strict_success"])
        self.assertTrue(result["conditions"]["cup_plate_contact"])
        self.assertTrue(result["conditions"]["cup_plate_place_contact"])
        self.assertTrue(result["conditions"]["cup_bottom_on_plate_support"])
        self.assertTrue(result["conditions"]["cup_bottom_within_plate_vertical_span"])
        self.assertTrue(result["conditions"]["cup_bottom_inside_plate_bottom_prism"])
        self.assertTrue(result["conditions"]["cup_axis_aligned_with_plate_normal"])
        self.assertTrue(result["conditions"]["cup_resting_on_plate"])
        self.assertTrue(result["conditions"]["cup_middle_over_plate"])
        self.assertTrue(result["conditions"]["cup_above_plate"])
        self.assertEqual(result["score"]["points"], 2)
        self.assertEqual(result["score"]["reason"], "resting_on_plate_and_bottom_touch")
        self.assertGreater(result["metrics"]["inside_margin_mm"], 40.0)
        self.assertNotIn("partial_success", result)

    def test_success_when_bottom_contact_is_inside_even_if_nominal_support_site_is_offset(self):
        result = _criterion(_FakeData(cup_pos=(0.0, 0.0, 0.0032), contact_dist=-0.004)).evaluate()

        self.assertTrue(result["success"])
        self.assertFalse(result["strict_success"])
        self.assertTrue(result["conditions"]["cup_plate_contact"])
        self.assertTrue(result["conditions"]["cup_plate_place_contact"])
        self.assertFalse(result["conditions"]["cup_bottom_on_plate_support"])
        self.assertTrue(result["conditions"]["cup_bottom_within_plate_vertical_span"])
        self.assertTrue(result["conditions"]["cup_bottom_inside_plate_bottom_prism"])
        self.assertEqual(result["score"]["points"], 2)

    def test_relaxed_success_classifies_a_single_ten_mm_bottom_contact(self):
        result = _criterion(
            _FakeData(cup_pos=(0.0, 0.0, 0.0032), contact_dist=-0.010)
        ).evaluate()

        self.assertTrue(result["conditions"]["cup_plate_place_contact"])
        self.assertFalse(result["conditions"]["cup_plate_contact"])
        self.assertTrue(result["conditions"]["cup_bottom_contact"])
        self.assertFalse(result["conditions"]["cup_bottom_contact_strict"])
        self.assertEqual(result["metrics"]["bottom_contact_count"], 1)
        self.assertEqual(result["metrics"]["strict_bottom_contact_count"], 0)
        self.assertTrue(result["success"])
        self.assertFalse(result["strict_success"])
        self.assertEqual(result["score"]["points"], 2)

    def test_cup_on_the_plate_rim_beyond_the_bottom_prism_still_scores(self):
        # 60 mm off centre: inside the plate (89.371 mm) but outside the
        # bottom prism once the cup radius and slack are included.
        result = _criterion(_FakeData(cup_pos=(0.0600, 0.0, 0.018912))).evaluate()

        self.assertTrue(result["success"])
        self.assertFalse(result["conditions"]["cup_bottom_inside_plate_bottom_prism"])
        self.assertTrue(result["conditions"]["cup_middle_over_plate"])
        self.assertEqual(result["score"]["points"], 2)

    def test_fails_when_the_cup_sits_beside_the_plate(self):
        result = _criterion(_FakeData(cup_pos=(0.12, 0.0, 0.018912))).evaluate()

        self.assertFalse(result["success"])
        self.assertFalse(result["conditions"]["cup_middle_over_plate"])
        self.assertEqual(result["score"]["points"], 0)
        self.assertEqual(result["score"]["reason"], "not_resting_on_plate")

    def test_fails_when_the_cup_has_sunk_through_the_plate(self):
        # Touching and centred, but under the plate rather than on it.
        result = _criterion(_FakeData(cup_pos=(0.0, 0.0, -0.0500))).evaluate()

        self.assertFalse(result["success"])
        self.assertFalse(result["conditions"]["cup_above_plate"])

    def test_registry_accepts_case_insensitive_auto_and_off(self):
        self.assertIsNone(make_success_criterion("OFF", mujoco=_FakeMujoco(), model=_FakeModel(), data=_FakeData()))
        criterion = make_success_criterion("AUTO", mujoco=_FakeMujoco(), model=_FakeModel(), data=_FakeData())
        self.assertIsInstance(criterion, CupOnPlateCriterion)
        with self.assertRaises(ValueError):
            make_success_criterion(
                "bad_task",
                mujoco=_FakeMujoco(),
                model=_FakeModel(),
                data=_FakeData(),
            )

    def test_fails_without_contact_even_if_geometrically_aligned(self):
        result = _criterion(_FakeData(contacts=False)).evaluate()

        self.assertFalse(result["success"])
        self.assertFalse(result["conditions"]["cup_plate_contact"])
        self.assertTrue(result["conditions"]["cup_bottom_inside_plate_bottom_prism"])
        self.assertEqual(result["score"]["points"], 0)

    def test_fails_when_contact_penetration_is_too_large(self):
        result = _criterion(_FakeData(contact_dist=-0.020)).evaluate()

        self.assertFalse(result["success"])
        self.assertFalse(result["conditions"]["cup_plate_contact"])

    def test_fails_when_cup_bottom_is_outside_plate_prism(self):
        result = _criterion(_FakeData(cup_pos=(0.12, 0.0, 0.018912))).evaluate()

        self.assertFalse(result["success"])
        self.assertFalse(result["conditions"]["cup_bottom_inside_plate_bottom_prism"])
        self.assertLess(result["metrics"]["inside_margin_mm"], 0.0)

    def test_plate_footprint_boundary_uses_the_full_cup_bottom(self):
        # The bottom disc inside the plate prism is the seated test, which
        # decides strict_success rather than success.
        for offset_m, expected in ((0.05686, True), (0.05688, False)):
            with self.subTest(offset_m=offset_m):
                result = _criterion(
                    _FakeData(cup_pos=(offset_m, 0.0, 0.018912))
                ).evaluate()

                self.assertIs(
                    result["conditions"]["cup_bottom_inside_plate_bottom_prism"],
                    expected,
                )
                self.assertIs(result["strict_success"], expected)
                self.assertTrue(result["success"])

    def test_success_boundary_uses_the_plate_radius_and_the_cup_middle(self):
        # plate 89.371 mm + 3 mm slack, measured to the cup's middle.
        for offset_m, expected in ((0.0923, True), (0.0925, False)):
            with self.subTest(offset_m=offset_m):
                result = _criterion(
                    _FakeData(cup_pos=(offset_m, 0.0, 0.018912))
                ).evaluate()

                self.assertIs(result["conditions"]["cup_middle_over_plate"], expected)
                self.assertIs(result["success"], expected)

    def test_seated_tests_reject_a_cup_above_the_support_plane(self):
        # The fake contact keeps the cup touching the plate at this pose, so
        # success holds; the seated geometry is what rejects it.
        result = _criterion(_FakeData(cup_pos=(0.0, 0.0, 0.0500))).evaluate()

        self.assertFalse(result["strict_success"])
        self.assertFalse(result["conditions"]["cup_bottom_on_plate_support"])
        self.assertGreater(result["metrics"]["support_gap_mm"], 10.0)

    def test_a_cup_settled_into_the_plate_counts_under_the_default_middle_test(self):
        # The default the Duo cupplate numbers were scored with: the
        # cup's middle, which is still well above the plate 5 mm down.
        result = _criterion(_FakeData(cup_pos=(0.0, 0.0, -0.0050))).evaluate()

        self.assertTrue(result["success"])
        self.assertTrue(result["conditions"]["cup_above_plate"])
        self.assertFalse(result["strict_success"])

    def test_fails_when_the_cup_has_sunk_below_the_plate_vertical_span(self):
        # 5 mm under the plate: the cup's middle is still well above it, so the
        # "lowest" floor test has to measure the base rather than the middle.
        result = _criterion(
            _FakeData(cup_pos=(0.0, 0.0, -0.0050)), cup_above_plate_test="lowest"
        ).evaluate()

        self.assertFalse(result["success"])
        self.assertFalse(result["conditions"]["cup_above_plate"])
        self.assertFalse(result["strict_success"])
        self.assertFalse(result["conditions"]["cup_bottom_on_plate_support"])
        self.assertFalse(result["conditions"]["cup_bottom_within_plate_vertical_span"])
        self.assertAlmostEqual(result["metrics"]["cup_lowest_above_plate_origin_mm"], -5.0, places=3)

    def test_rejects_unknown_score_condition(self):
        with self.assertRaises(ValueError):
            CupOnPlateCriterion(
                mujoco=_FakeMujoco(),
                model=_FakeModel(),
                data=_FakeData(),
                options={"score_condition": "bad_score"},
            )

    def test_collects_collision_geoms_from_body_subtree(self):
        criterion = CupOnPlateCriterion(mujoco=_SubtreeMujoco(), model=_SubtreeModel(), data=_FakeData())

        self.assertEqual(criterion.cup_geoms, frozenset({0}))
        self.assertEqual(criterion.plate_geoms, frozenset({1}))

    def test_summarize_success_records(self):
        records = [
            {"frame": 10, "success": {"success": False, "score": {"enabled": True, "points": 0, "max_points": 2, "condition": "cup_on_plate_contact_points"}}},
            {"frame": 11, "success": {"success": True, "metrics": {"inside_margin_mm": 1.0}, "score": {"enabled": True, "points": 2, "max_points": 2, "condition": "cup_on_plate_contact_points"}}},
            {"frame": 12, "success": {"success": True, "metrics": {"inside_margin_mm": 2.0}, "score": {"enabled": True, "points": 1, "max_points": 2, "condition": "cup_on_plate_contact_points"}}},
            {"frame": 13, "success": {"success": False, "score": {"enabled": True, "points": 0, "max_points": 2, "condition": "cup_on_plate_contact_points"}}},
        ]

        summary = summarize_success_records(records)

        self.assertFalse(summary["new_final_success"])
        self.assertFalse(summary["success"])
        self.assertTrue(summary["success_ever"])
        self.assertEqual(summary["success_first_frame"], 11)
        self.assertEqual(summary["success_last_frame"], 12)
        self.assertEqual(summary["success_hold_frames"], 2)
        self.assertFalse(summary["success_final"]["success"])
        self.assertEqual(summary["score_total"], 3)
        self.assertEqual(summary["score_max_total"], 8)
        self.assertEqual(summary["score_final"], 0)
        self.assertEqual(summary["score_best"], 2)
        self.assertEqual(summary["score_two_point_frames"], [11])
        self.assertEqual(summary["score_one_point_frames"], [12])
        partial = summarize_success_records(
            [
                {"frame": 1, "success": {"success": False, "partial_success": True}},
                {"frame": 2, "success": {"success": False, "partial_success": False}},
            ]
        )
        self.assertFalse(partial["partial_success"])
        self.assertTrue(partial["partial_success_ever"])
        legacy = summarize_success_records(
            [{"frame": 1, "success": {"success": False}}]
        )
        self.assertNotIn("partial_success", legacy)

    def test_a_cup_off_its_support_plane_still_counts_while_touching(self):
        """Success asks for contact over the plate, not a seated bottom disc.

        The seated reading is still reported, and still gates
        ``strict_success``, so the stricter number remains readable.
        """
        result = _criterion(_FakeData(cup_pos=(0.0, 0.0, 0.0500))).evaluate()

        self.assertTrue(result["success"])
        self.assertFalse(result["strict_success"])
        self.assertFalse(result["conditions"]["cup_bottom_on_plate_support"])
        self.assertGreater(result["metrics"]["support_gap_mm"], 10.0)


class _BBoxFakeModel(_FakeModel):
    # cup geom (unused for bbox), plate geom: square slab half-extents 50mm xy,
    # z from -1mm to +17mm -> corner radial 70.71mm (AABB-corner fallback path).
    geom_dataid = np.array([-1, -1], dtype=np.int32)
    geom_aabb = np.array(
        [
            [0.0, 0.0, 0.035, 0.03, 0.03, 0.035],
            [0.0, 0.0, 0.008, 0.05, 0.05, 0.009],
        ],
        dtype=float,
    )


class _BBoxFakeData(_FakeData):
    def __init__(self, cup_pos, cup_mat=None, plate_pos=(0.0, 0.0, 0.0), **kw):
        super().__init__(cup_pos=cup_pos, plate_pos=plate_pos, **kw)
        eye = np.eye(3).reshape(9)
        cup_geom_mat = eye if cup_mat is None else np.asarray(cup_mat, dtype=float).reshape(9)
        if cup_mat is not None:
            self.xmat[1] = np.asarray(cup_mat, dtype=float).reshape(9)
        self.geom_xpos = np.array([list(cup_pos), list(plate_pos)], dtype=float)
        self.geom_xmat = np.stack([cup_geom_mat, eye]).astype(float)


class _MeshFakeModel(_FakeModel):
    """A round cup: two rings of vertices, 35.5 mm radius, 76 mm tall."""

    _RING = np.array(
        [
            [0.0355 * np.cos(a), 0.0355 * np.sin(a), z]
            for z in (0.0, 0.076)
            for a in np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
        ],
        dtype=float,
    )
    geom_dataid = np.array([0, -1], dtype=np.int32)
    mesh_vertadr = np.array([0], dtype=np.int32)
    mesh_vertnum = np.array([len(_RING)], dtype=np.int32)
    mesh_vert = _RING
    geom_aabb = np.array(
        [
            [0.0, 0.0, 0.038, 0.0355, 0.0355, 0.038],
            [0.0, 0.0, 0.008, 0.05, 0.05, 0.009],
        ],
        dtype=float,
    )


class CupOnPlateRoundCupTest(unittest.TestCase):
    @staticmethod
    def _rot_x(deg):
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)

    @staticmethod
    def _rot_z(deg):
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

    def _criterion(self, cup_pos, cup_R):
        data = _BBoxFakeData(cup_pos=cup_pos, cup_mat=cup_R.reshape(9),
                             contact_pos=(0.0, 0.0, 0.017))
        return CupOnPlateCriterion(
            mujoco=_FakeMujoco(), model=_MeshFakeModel(), data=data,
            options={"geometry_mode": "bbox"},
        )

    def test_axial_roll_does_not_move_a_toppled_cup_lowest_point(self):
        # A round cup on its side touches down 35.5 mm below its axis however
        # far it has rolled. A bounding box would answer 50.2 mm at 45 degrees
        # and call the cup sunk through the plate.
        lowest = []
        for roll_deg in (0.0, 45.0, 90.0, 23.7):
            cup_R = self._rot_x(90.0) @ self._rot_z(roll_deg)
            result = self._criterion((0.0, 0.0, 0.0525), cup_R).evaluate()
            lowest.append(result["metrics"]["cup_lowest_above_plate_origin_mm"])
            self.assertTrue(result["conditions"]["cup_above_plate"], msg=f"roll {roll_deg}")
            self.assertTrue(result["success"], msg=f"roll {roll_deg}")
        # The ring is sampled at 64 points, so a roll between two of them reads
        # a hundredth of a millimetre high; a bounding-box estimate errs by 15 mm.
        for value in lowest[1:]:
            self.assertAlmostEqual(value, lowest[0], delta=0.05)
        self.assertAlmostEqual(lowest[0], 17.0, delta=0.05)

    def test_an_upright_round_cup_touches_down_at_its_base(self):
        result = self._criterion((0.0, 0.0, 0.017), np.eye(3)).evaluate()
        self.assertAlmostEqual(
            result["metrics"]["cup_lowest_above_plate_origin_mm"], 17.0, places=3
        )
        self.assertTrue(result["success"])


def _bbox_criterion(data, **options):
    return CupOnPlateCriterion(
        mujoco=_FakeMujoco(),
        model=_BBoxFakeModel(),
        data=data,
        options={"geometry_mode": "bbox", **options},
    )


class CupOnPlateBBoxModeTest(unittest.TestCase):
    _TOPPLED = (0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0)  # cup +z axis -> world +x

    def test_full_success_on_dish_floor_where_legacy_fails(self):
        data = _BBoxFakeData(cup_pos=(0.01, 0.0, 0.003))
        result = _bbox_criterion(data).evaluate()
        self.assertTrue(result["success"])
        self.assertFalse(result["partial_success"])
        self.assertTrue(result["conditions"]["cup_center_inside_plate_disc"])
        self.assertTrue(result["conditions"]["cup_bottom_in_plate_zrange"])
        legacy = _criterion(_FakeData(cup_pos=(0.01, 0.0, 0.003))).evaluate()
        self.assertTrue(legacy["success"])
        self.assertFalse(legacy["strict_success"])

    def test_full_success_standing_on_rim_inside_disc(self):
        result = _bbox_criterion(_BBoxFakeData(cup_pos=(0.04, 0.0, 0.017))).evaluate()
        self.assertTrue(result["success"])

    def test_rejects_cup_outside_plate_disc_touching_rim(self):
        # radial 80mm > 70.71+3mm slack: standing on the table, only leaning on the plate edge
        result = _bbox_criterion(_BBoxFakeData(cup_pos=(0.08, 0.0, 0.002))).evaluate()
        self.assertFalse(result["success"])
        self.assertFalse(result["partial_success"])
        self.assertFalse(result["conditions"]["cup_center_inside_plate_disc"])

    def test_toppled_cup_on_plate_is_success(self):
        # A cup on its side on the plate counts as success. Lying on the plate's 17 mm top face: the fake cup is 60 mm across, so
        # its axis sits 30 mm above that.
        data = _BBoxFakeData(
            cup_pos=(0.01, 0.0, 0.047),
            cup_mat=self._TOPPLED,
            contact_pos=(0.045, 0.0, 0.017),
        )
        result = _bbox_criterion(data).evaluate()
        self.assertTrue(result["success"])
        self.assertFalse(result["partial_success"])
        self.assertFalse(result["strict_success"])
        self.assertTrue(result["conditions"]["cup_resting_on_plate"])
        # It meets the plate along its side, which is the one-point case: the
        # cup is there, but not seated on its base.
        self.assertFalse(result["conditions"]["cup_bottom_contact"])
        self.assertTrue(result["conditions"]["cup_side_contact"])
        self.assertEqual(result["score"]["points"], 1)
        self.assertEqual(result["score"]["reason"], "resting_on_plate_and_side_touch")

    def test_measures_the_cup_box_and_carries_it_through_a_rotation(self):
        # The cup geom is 60 x 60 x 70 mm centred 35 mm up its own axis. Upright
        # it reaches 35 mm below its middle; on its side, 30 mm.
        upright = _bbox_criterion(_BBoxFakeData(cup_pos=(0.0, 0.0, 0.003)))
        np.testing.assert_allclose(upright._cup_center_local, [0.0, 0.0, 0.035], atol=1e-9)
        np.testing.assert_allclose(upright._cup_half_extents_local, [0.03, 0.03, 0.035], atol=1e-9)
        self.assertAlmostEqual(
            upright.evaluate()["metrics"]["cup_lowest_above_plate_origin_mm"], 3.0, places=3
        )

        toppled = _bbox_criterion(
            _BBoxFakeData(cup_pos=(0.01, 0.0, 0.047), cup_mat=self._TOPPLED,
                          contact_pos=(0.045, 0.0, 0.017))
        ).evaluate()
        # Middle carried 35 mm along the cup's own +x, and the lowest point is
        # the 30 mm half-width now facing the plate.
        self.assertAlmostEqual(toppled["metrics"]["middle_center_offset_mm"], 45.0, places=3)
        self.assertAlmostEqual(toppled["metrics"]["cup_lowest_above_plate_origin_mm"], 17.0, places=3)

    def test_rejects_a_cup_sunk_below_the_measured_plate_floor(self):
        # bbox mode measures the plate floor at -1 mm; under the "lowest" test a
        # cup 20 mm under it is not resting on anything.
        result = _bbox_criterion(
            _BBoxFakeData(cup_pos=(0.0, 0.0, -0.020)), cup_above_plate_test="lowest"
        ).evaluate()
        self.assertFalse(result["success"])
        self.assertFalse(result["conditions"]["cup_above_plate"])

    def test_the_plate_floor_admits_the_penetration_tolerance_and_no_more(self):
        # Floor -1 mm, tolerance 2 mm: a base at -3 mm is still on the plate.
        for base_m, expected in ((-0.0029, True), (-0.0031, False)):
            with self.subTest(base_m=base_m):
                result = _bbox_criterion(
                    _BBoxFakeData(cup_pos=(0.0, 0.0, base_m)),
                    cup_above_plate_test="lowest",
                ).evaluate()
                self.assertIs(result["conditions"]["cup_above_plate"], expected)

    def test_rejects_a_toppled_cup_whose_middle_overhangs_the_plate(self):
        # Lying with its middle 85 mm out, past the 70.7 + 3 mm the plate reaches.
        data = _BBoxFakeData(
            cup_pos=(0.05, 0.0, 0.047),
            cup_mat=self._TOPPLED,
            contact_pos=(0.05, 0.0, 0.017),
        )
        result = _bbox_criterion(data).evaluate()
        self.assertFalse(result["success"])
        self.assertFalse(result["conditions"]["cup_middle_over_plate"])

    def test_describe_reports_measured_bbox(self):
        desc = _bbox_criterion(_BBoxFakeData(cup_pos=(0.0, 0.0, 0.003))).describe()
        self.assertEqual(desc["geometry_mode"], "bbox")
        self.assertAlmostEqual(desc["plate_bbox"]["radial_mm"], 70.711, places=2)
        self.assertAlmostEqual(desc["plate_bbox"]["z_min_mm"], -1.0, places=3)
        self.assertAlmostEqual(desc["plate_bbox"]["z_max_mm"], 17.0, places=3)

    def test_toppled_cup_on_plate_is_a_success(self):
        """A cup on its side is still on the plate.

        Success does not ask the cup to be seated on its bottom disc, so the
        upright and bottom-contact terms are reported but do not gate it. The
        stricter reading is available as ``strict_success``.
        """
        data = _BBoxFakeData(
            cup_pos=(0.01, 0.0, 0.0355),
            cup_mat=self._TOPPLED,
            contact_pos=(0.045, 0.0, 0.017),
        )
        result = _bbox_criterion(data).evaluate()
        self.assertTrue(result["success"])
        self.assertTrue(result["conditions"]["cup_middle_over_plate"])
        self.assertFalse(result["strict_success"])

if __name__ == "__main__":
    unittest.main()
