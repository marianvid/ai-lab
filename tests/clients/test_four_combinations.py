"""Who reaches what, for the four combinations of the two switches (the reviewer's item 21).

Callers: a client's token, the page's session (the one the page hands out with
login off, a logged-in one with login on), and nothing at all. Routes: one of
the model API and one of management. Management is the page's in every
combination; the anonymous caller reaches the model API only while
`require_token` is off; a token reaches the model API always.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from ai_lab.clients.access import Access
from ai_lab.clients.door import ANONYMOUS, WEB
from ai_lab.clients.errors import ManagementNeedsLoginError, TokenMissingError
from ai_lab.clients.sessions import COOKIE

MODEL = ("POST", "/v1/chat/completions")
MANAGEMENT = ("PATCH", "/api/security")
PASSWORD = "a long password"


class FourCombinationsTests(unittest.TestCase):
    def access(self, require_token: bool, login_required: bool) -> Access:
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        if require_token and not login_required:
            # Refused on save (m1); a machine already in it stays as it is.
            (folder / "security.json").write_text('{"require_token": true}')
            return Access.in_folder(folder, log=lambda _line: None)
        access = Access.in_folder(folder, log=lambda _line: None)
        if login_required:
            access.set_password(PASSWORD, None)
        access.update_security({"login_required": login_required, "require_token": require_token})
        return access

    def page(self, access: Access, login_required: bool) -> dict:
        if login_required:
            return {"Cookie": access.login(PASSWORD).split(";")[0]}
        return {"Cookie": f"{COOKIE}={access.sessions.start(kept=False)}"}

    def outcome(self, access: Access, route: tuple[str, str], headers: dict) -> str:
        method, path = route
        try:
            return access.door.admit(method, path, headers)
        except ManagementNeedsLoginError:
            return "for the page"
        except TokenMissingError:
            return "token missing"

    def test_every_caller_on_every_route_in_every_combination(self):
        for require_token in (False, True):
            for login_required in (False, True):
                access = self.access(require_token, login_required)
                token = {"Authorization": f"Bearer {access.register('scheduler-linux')['token']}"}
                page = self.page(access, login_required)
                anonymous_on_model = "token missing" if require_token else ANONYMOUS
                expected = {
                    ("token", MODEL): "scheduler-linux", ("token", MANAGEMENT): "for the page",
                    ("page", MODEL): WEB, ("page", MANAGEMENT): WEB,
                    ("nothing", MODEL): anonymous_on_model,
                    ("nothing", MANAGEMENT): "token missing" if require_token else "for the page",
                }
                callers = {"token": token, "page": page, "nothing": {}}
                for (caller, route), wanted in expected.items():
                    with self.subTest(require_token=require_token, login_required=login_required,
                                      caller=caller, route=route[1]):
                        self.assertEqual(self.outcome(access, route, callers[caller]), wanted)


if __name__ == "__main__":
    unittest.main()
