from fincantatem import finite

@finite
def divide(a: int, b: int):
    return a / b

def main():
    numbers = range(0, 10)
    for n_a, n_b in zip(numbers, numbers):
        divide(n_a, n_b)

if __name__ == "__main__":
    main()