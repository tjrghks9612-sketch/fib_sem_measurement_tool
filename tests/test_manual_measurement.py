import unittest

from fib_sem_measurement_tool.core.manual_measurement import make_manual_measurement, required_manual_points
from fib_sem_measurement_tool.models.settings import MeasurementSettings


class ManualMeasurementTest(unittest.TestCase):
    def test_distance_horizontal_uses_two_points(self) -> None:
        settings = MeasurementSettings(measurement_type="distance_horizontal")

        result = make_manual_measurement([(10, 20), (42, 25)], settings)

        self.assertEqual(result.measurement_source, "manual")
        self.assertEqual(result.status, "OK")
        self.assertIsNotNone(result.horizontal_cd)
        self.assertEqual(result.horizontal_cd.selected_px, 32)
        pair = result.horizontal_cd.selected_pair
        self.assertEqual(pair.first.image_y, pair.second.image_y)

    def test_distance_vertical_constrains_display_pair(self) -> None:
        settings = MeasurementSettings(measurement_type="distance_vertical")

        result = make_manual_measurement([(10, 20), (42, 55)], settings)

        self.assertIsNotNone(result.vertical_thk)
        self.assertEqual(result.vertical_thk.selected_px, 35)
        pair = result.vertical_thk.selected_pair
        self.assertEqual(pair.first.image_x, pair.second.image_x)

    def test_distance_both_uses_horizontal_and_vertical_components(self) -> None:
        settings = MeasurementSettings(measurement_type="distance_both")

        result = make_manual_measurement([(10, 20), (42, 25), (70, 10), (74, 45)], settings)

        self.assertEqual(result.horizontal_cd.selected_px, 32)
        self.assertEqual(result.vertical_thk.selected_px, 35)

    def test_four_point_modes_require_four_points(self) -> None:
        for measurement_type in ("distance_both", "hole_cd", "crater", "taper_double"):
            with self.subTest(measurement_type=measurement_type):
                self.assertEqual(required_manual_points(measurement_type), 4)

    def test_single_taper_uses_configured_side(self) -> None:
        settings = MeasurementSettings(measurement_type="taper_single", taper_side="right")

        result = make_manual_measurement([(20, 10), (30, 60)], settings)

        self.assertIsNone(result.left_taper)
        self.assertIsNotNone(result.right_taper)
        self.assertEqual(result.right_taper.status, "OK")
        self.assertEqual(result.right_taper.fit_point_count, 2)

    def test_double_taper_requires_four_points(self) -> None:
        settings = MeasurementSettings(measurement_type="taper_double")

        self.assertEqual(required_manual_points(settings.measurement_type), 4)
        result = make_manual_measurement([(10, 10), (15, 60), (80, 10), (72, 60)], settings)

        self.assertIsNotNone(result.left_taper)
        self.assertIsNotNone(result.right_taper)
        self.assertIsNotNone(result.avg_taper_angle)

    def test_hole_cd_uses_horizontal_and_vertical_manual_axes(self) -> None:
        settings = MeasurementSettings(measurement_type="hole_cd")

        result = make_manual_measurement([(10, 40), (70, 44), (45, 20), (50, 80)], settings)

        self.assertIsNotNone(result.horizontal_cd)
        self.assertIsNotNone(result.vertical_thk)
        self.assertIsNotNone(result.hole_cd)
        self.assertEqual(result.hole_cd.horizontal_px, 60)
        self.assertEqual(result.hole_cd.vertical_px, 60)

    def test_crater_uses_horizontal_and_vertical_manual_axes(self) -> None:
        settings = MeasurementSettings(measurement_type="crater")

        result = make_manual_measurement([(10, 80), (90, 84), (50, 30), (54, 80)], settings)

        self.assertIsNotNone(result.horizontal_cd)
        self.assertIsNotNone(result.vertical_thk)
        self.assertIsNotNone(result.crater)
        self.assertEqual(result.crater.cd_px, 80)
        self.assertEqual(result.crater.thk_px, 50)
        self.assertEqual(result.crater.status, "OK")


if __name__ == "__main__":
    unittest.main()
