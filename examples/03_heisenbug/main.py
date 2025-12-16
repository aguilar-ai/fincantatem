from fincantatem import finite
from typing import List
from time import sleep
import random


def flush_to_stdout(buffer: List[str]):
    if len(buffer) > 10:
        raise ValueError("Buffer is too large to flush to stdout")
    print("Events reported: ", "\n".join(buffer))


def add_event_log(event: str, _buffer: List[str] = []): # type: ignore
    _buffer.append(event)
    if len(_buffer) >= 3:
        flush_to_stdout(_buffer)
        return True
    return False


@finite
def main():
    while True:
        sleep(random.random() * 2)
        event = random.choice(["login", "logout", "view_page", "login_admin"])
        add_event_log(event)


if __name__ == "__main__":
    main()
