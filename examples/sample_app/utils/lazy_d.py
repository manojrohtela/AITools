def d_value():
    from utils import lazy_c  # deferred import: only runs when called

    return 4 if lazy_c else 0
