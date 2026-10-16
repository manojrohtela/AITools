from auth.passwords import hash_password
from auth.service import login
from web.views import login_view

USERS = {"alice": {"username": "alice", "password_hash": hash_password("wonderland")}}


def test_login_success():
    assert login("alice", "wonderland", USERS).startswith("alice:")


def test_login_wrong_password():
    assert login("alice", "nope", USERS) is None


def test_login_view_unauthorized():
    assert login_view({"username": "bob", "password": "x"}, USERS)["status"] == 401
