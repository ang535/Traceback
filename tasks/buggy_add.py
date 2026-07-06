# This function should return the sum of two numbers
# but it has a bug — can you find and fix it?

def add(a, b):
    return a + b


if __name__ == "__main__":
    result = add(3, 5)
    print(f"3 + 5 = {result}")
    assert result == 8, f"Expected 8 but got {result}"
    print("All tests passed!")