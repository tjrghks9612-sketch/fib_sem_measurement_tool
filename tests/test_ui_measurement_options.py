import unittest

from fib_sem_measurement_tool.models.settings import MeasurementSettings
from fib_sem_measurement_tool.ui.i18n import MEASUREMENT_LABELS
from fib_sem_measurement_tool.ui.option_panel import MEASUREMENT_KEYS as OPTION_MEASUREMENT_KEYS
from fib_sem_measurement_tool.ui.option_panel import option_visibility_for_measurement_type
from fib_sem_measurement_tool.ui.thumbnail_panel import MEASUREMENT_KEYS as THUMBNAIL_MEASUREMENT_KEYS


class UiMeasurementOptionsTest(unittest.TestCase):
    def test_ellipse_cd_mode_is_not_exposed_in_ui_options(self) -> None:
        self.assertNotIn("ellipse_cd", OPTION_MEASUREMENT_KEYS)
        self.assertNotIn("ellipse_cd", THUMBNAIL_MEASUREMENT_KEYS)
        self.assertIn("hole_cd", OPTION_MEASUREMENT_KEYS)
        self.assertIn("hole_cd", THUMBNAIL_MEASUREMENT_KEYS)
        self.assertIn("crater", OPTION_MEASUREMENT_KEYS)
        self.assertIn("crater", THUMBNAIL_MEASUREMENT_KEYS)
        for labels in MEASUREMENT_LABELS.values():
            self.assertNotIn("ellipse_cd", labels)

    def test_minimum_grayscale_delta_default_is_55(self) -> None:
        self.assertEqual(MeasurementSettings().minimum_grayscale_delta, 55.0)
        self.assertEqual(MeasurementSettings().crater_taper_height_percent, 15.0)

    def test_distance_modes_only_show_distance_options(self) -> None:
        for measurement_type in ("distance_horizontal", "distance_vertical", "distance_both"):
            visibility = option_visibility_for_measurement_type(measurement_type)
            self.assertTrue(visibility["representative_value"])
            self.assertTrue(visibility["normalize_signal"])
            self.assertTrue(visibility["denoise_signal"])
            self.assertTrue(visibility["boundary_angle_filter"])
            self.assertTrue(visibility["calibration"])
            self.assertFalse(visibility["taper_side"])
            self.assertFalse(visibility["taper_height"])
            self.assertFalse(visibility["crater_taper_height"])
            self.assertFalse(visibility["fit_line"])

    def test_taper_modes_only_show_taper_options(self) -> None:
        single = option_visibility_for_measurement_type("taper_single")
        double = option_visibility_for_measurement_type("taper_double")

        self.assertTrue(single["taper_side"])
        self.assertFalse(double["taper_side"])
        for visibility in (single, double):
            self.assertTrue(visibility["taper_height"])
            self.assertTrue(visibility["fit_line"])
            self.assertTrue(visibility["minimum_delta"])
            self.assertFalse(visibility["representative_value"])
            self.assertFalse(visibility["normalize_signal"])
            self.assertFalse(visibility["denoise_signal"])
            self.assertFalse(visibility["boundary_angle_filter"])
            self.assertFalse(visibility["calibration"])
            self.assertFalse(visibility["crater_taper_height"])

    def test_max_boundary_angle_only_shows_when_angle_filter_is_on(self) -> None:
        hidden = option_visibility_for_measurement_type("distance_horizontal", boundary_angle_filter_enabled=False)
        visible = option_visibility_for_measurement_type("distance_horizontal", boundary_angle_filter_enabled=True)
        taper = option_visibility_for_measurement_type("taper_single", boundary_angle_filter_enabled=True)

        self.assertFalse(hidden["max_boundary_angle"])
        self.assertTrue(visible["max_boundary_angle"])
        self.assertFalse(taper["max_boundary_angle"])

    def test_hole_cd_only_shows_hole_options(self) -> None:
        visibility = option_visibility_for_measurement_type("hole_cd")

        self.assertTrue(visibility["hole_target"])
        self.assertTrue(visibility["minimum_delta"])
        self.assertTrue(visibility["calibration"])
        self.assertFalse(visibility["representative_value"])
        self.assertFalse(visibility["edge_scan_start"])
        self.assertFalse(visibility["taper_height"])
        self.assertFalse(visibility["crater_taper_height"])

    def test_crater_only_shows_crater_relevant_options(self) -> None:
        visibility = option_visibility_for_measurement_type("crater")

        self.assertTrue(visibility["minimum_delta"])
        self.assertTrue(visibility["calibration"])
        self.assertTrue(visibility["fit_line"])
        self.assertTrue(visibility["crater_taper_height"])
        self.assertFalse(visibility["hole_target"])
        self.assertFalse(visibility["representative_value"])
        self.assertFalse(visibility["edge_scan_start"])
        self.assertFalse(visibility["taper_height"])


if __name__ == "__main__":
    unittest.main()
