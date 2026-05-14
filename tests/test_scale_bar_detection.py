import unittest

import numpy as np

from fib_sem_measurement_tool.core.calibration import detect_scale_bar


class ScaleBarDetectionTest(unittest.TestCase):
    def test_detects_long_green_horizontal_run_in_lower_left_quadrant(self) -> None:
        image = np.zeros((120, 200, 3), dtype=np.uint8)
        image[95, 18:78] = (0, 255, 0)
        image[95, 100:190] = (0, 255, 0)

        result = detect_scale_bar(image)

        self.assertEqual(result["status"], "detected")
        self.assertEqual(result["region"], "lower_left_quadrant")
        self.assertEqual(result["pixel_length"], 60.0)
        self.assertEqual(result["line"], (18, 95, 77, 95))

    def test_ignores_short_green_noise(self) -> None:
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        image[60, 8:22] = (0, 255, 0)

        result = detect_scale_bar(image)

        self.assertEqual(result["status"], "not_found")
        self.assertEqual(result["message"], "scale_bar_not_found")

    def test_ignores_upper_or_right_quadrant_green_lines(self) -> None:
        image = np.zeros((100, 140, 3), dtype=np.uint8)
        image[20, 10:70] = (0, 255, 0)
        image[80, 80:130] = (0, 255, 0)

        result = detect_scale_bar(image)

        self.assertEqual(result["status"], "not_found")


if __name__ == "__main__":
    unittest.main()
