from auth import tokens
from auth.passwords import verify_password


class AuthService:
    def __init__(self, user_repository):
        self.users = user_repository

    def authenticate(self, username: str, password: str):
        user = self.users.get(username)
        if user and verify_password(password, user["password_hash"]):
            return self._issue_token(user)
        return None

    def _issue_token(self, user):
        return tokens.create_token(user["username"])


def login(username: str, password: str, user_repository):
    service = AuthService(user_repository)
    return service.authenticate(username, password)
