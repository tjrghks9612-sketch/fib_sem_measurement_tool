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

    def test_distance_both_uses_horizontal_and_vertical_components(self) -> None:
        settings = MeasurementSettings(measurement_type="distance_both")

        result = make_manual_measurement([(10, 20), (42, 55)], settings)

        self.assertEqual(result.horizontal_cd.selected_px, 32)
        self.assertEqual(result.vertical_thk.selected_px, 35)

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


if __name__ == "__main__":
    unittest.main()
