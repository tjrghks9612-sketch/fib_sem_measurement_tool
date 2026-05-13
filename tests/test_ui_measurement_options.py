import unittest

from fib_sem_measurement_tool.models.settings import MeasurementSettings
from fib_sem_measurement_tool.ui.i18n import MEASUREMENT_LABELS
from fib_sem_measurement_tool.ui.option_panel import MEASUREMENT_KEYS as OPTION_MEASUREMENT_KEYS
from fib_sem_measurement_tool.ui.thumbnail_panel import MEASUREMENT_KEYS as THUMBNAIL_MEASUREMENT_KEYS


class UiMeasurementOptionsTest(unittest.TestCase):
    def test_ellipse_cd_mode_is_not_exposed_in_ui_options(self) -> None:
        self.assertNotIn("ellipse_cd", OPTION_MEASUREMENT_KEYS)
        self.assertNotIn("ellipse_cd", THUMBNAIL_MEASUREMENT_KEYS)
        for labels in MEASUREMENT_LABELS.values():
            self.assertNotIn("ellipse_cd", labels)

    def test_minimum_grayscale_delta_default_is_55(self) -> None:
        self.assertEqual(MeasurementSettings().minimum_grayscale_delta, 55.0)


if __name__ == "__main__":
    unittest.main()
