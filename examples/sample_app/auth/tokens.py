import secrets


def create_token(username: str) -> str:
    return f"{username}:{secrets.token_hex(8)}"


def parse_token(token: str) -> str:
    return token.split(":", 1)[0]
