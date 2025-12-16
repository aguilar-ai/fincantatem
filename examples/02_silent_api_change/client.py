from requests import get
from fincantatem import finite
import random
from time import sleep


@finite
def main():
    while True:
        user_id = random.randint(1, 1000000)
        response = get(f"http://localhost:8000/users/{user_id}")
        subscription_status = response.json()["data"]["active_subscription"]["status"]
        print(f"Verified user {user_id} with subscription status: {subscription_status}")
        sleep(2)


if __name__ == "__main__":
    main()
