# This function should return the product of two numbers
# but it has a bug — can you find and fix it?

def multiply(a, b):
    return a * b


if __name__ == "__main__":
    result = multiply(3, 5)
    print(f"3 * 5 = {result}")
    assert result == 15, f"Expected 15 but got {result}"
    print("All tests passed!")