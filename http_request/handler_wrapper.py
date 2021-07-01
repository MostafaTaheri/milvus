import functools
import requests
import json

from .constants import Status


def handle_error(returns=tuple()):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            nonlocal returns
            try:
                return func(self, *args, **kwargs)
            except requests.exceptions.Timeout:
                status = Status(Status.UNEXPECTED_ERROR,
                                message='Request timeout')
                return returns if not returns else tuple([status]) + returns
            except json.decoder.JSONDecodeError as e:
                status = Status(Status.UNEXPECTED_ERROR, message=str(e))
                return returns if not returns else tuple([status]) + returns

        return wrapper

    return decorator
