import json
from http.server import ThreadingHTTPServer
from threading import Thread
import unittest
from urllib.request import urlopen

from ui.server import DashboardHandler


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), DashboardHandler)
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_dashboard_home_shows_substack_only(self):
        with urlopen(self.base_url) as response:
            page = response.read().decode("utf-8")

        self.assertIn("Substack article", page)
        self.assertNotIn("X thread", page)
        self.assertNotIn("Reddit", page)

    def test_state_api_returns_dashboard_settings(self):
        with urlopen(f"{self.base_url}/api/state") as response:
            state = json.loads(response.read().decode("utf-8"))

        self.assertIn("jobs", state)
        self.assertIn("library", state)
        self.assertIn("settings", state)


if __name__ == "__main__":
    unittest.main()
