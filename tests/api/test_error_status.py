"""Which status and OpenAI error type an exception leaves the gateway with."""

import unittest
import urllib.error
from http import HTTPStatus

from ai_lab.api.server import error_kind, status_for


class TimeoutStatusTests(unittest.TestCase):
    def test_an_engine_that_does_not_answer_in_time_is_a_504(self):
        # Answered as a 400 it made a client split its batch, thinking one of
        # its items had been refused.
        for error in (TimeoutError("timed out"),
                      urllib.error.URLError(TimeoutError("timed out"))):
            with self.subTest(error=error):
                self.assertEqual(status_for(error), HTTPStatus.GATEWAY_TIMEOUT)
                self.assertEqual(error_kind(error), "timeout_error")

    def test_other_errors_keep_their_meaning(self):
        self.assertEqual(error_kind(ValueError("bad")), "invalid_request_error")
        self.assertEqual(status_for(ValueError("bad")), HTTPStatus.BAD_REQUEST)
        self.assertEqual(status_for(urllib.error.URLError("refused")),
                         HTTPStatus.BAD_REQUEST)


if __name__ == "__main__":
    unittest.main()
