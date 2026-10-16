from auth.service import login


def login_view(request, user_repository):
    token = login(request["username"], request["password"], user_repository)
    return {"status": 200 if token else 401, "token": token}


def route_table():
    # Referenced, not called: a callback registration.
    return {"/login": login}


def dynamic_dispatch(module):
    return getattr(module, "login")("admin", "secret", {})
