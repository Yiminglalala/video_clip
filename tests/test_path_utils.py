import unittest

from src.path_utils import normalize_pasted_local_path


class PathUtilsTests(unittest.TestCase):
    def test_strips_windows_copy_as_path_quotes(self):
        self.assertEqual(
            normalize_pasted_local_path('"D:\\video\\clip.mp4"'),
            "D:\\video\\clip.mp4",
        )

    def test_strips_nested_and_smart_quotes(self):
        self.assertEqual(
            normalize_pasted_local_path(' “\'D:\\video\\clip.mp4\'” '),
            "D:\\video\\clip.mp4",
        )

    def test_keeps_internal_quotes(self):
        self.assertEqual(
            normalize_pasted_local_path('"D:\\video\\a \\"quoted\\" clip.mp4"'),
            'D:\\video\\a \\"quoted\\" clip.mp4',
        )

    def test_empty_input(self):
        self.assertEqual(normalize_pasted_local_path(None), "")
        self.assertEqual(normalize_pasted_local_path("  "), "")


if __name__ == "__main__":
    unittest.main()
