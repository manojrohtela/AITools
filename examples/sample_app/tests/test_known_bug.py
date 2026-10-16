from auth.tokens import parse_token
from utils.mathx import factorial


def test_factorial():
    assert factorial(5) == 120


def test_parse_token_with_colon_in_username():
    # Known bug: usernames containing ':' are truncated by parse_token.
    assert parse_token("dr:who:abcdef") == "dr:who"
