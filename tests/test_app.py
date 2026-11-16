"""Exercise actual Streamlit renders and reruns without a browser dependency."""

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class AppTests(unittest.TestCase):
    def test_sample_dashboard_and_filters(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, "3")
        self.assertEqual(app.metric[1].value, "77.8%")
        self.assertEqual(app.metric[3].value, "91.7%")
        self.assertEqual(len(app.warning), 2)
        self.assertEqual(app.dataframe[0].value.iloc[0].annotator, "Amina")
        app.text_input[0].set_value("013").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe[1].value), 1)
        self.assertEqual(app.dataframe[1].value.iloc[0].status, "unknown ID")
        app.selectbox[0].select("Brian").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox[0].value, "Brian")

    def test_upload_mode_waits_for_both_files(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        app.radio[0].set_value("Upload your data").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 0)
        self.assertIn("Upload both files", app.info[0].value)


if __name__ == "__main__":
    unittest.main()
