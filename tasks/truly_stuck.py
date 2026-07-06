# This function must return the exact same value as SECRET_TARGET in secrets.py
# Fix the get_value() function so the assertion passes.

from secrets import SECRET_TARGET


def get_value():
    return SECRET_TARGET  # change this to match SECRET_TARGET


if __name__ == "__main__":
    result = get_value()
    print(f"Your value: {result}")
    print(f"Target value: {SECRET_TARGET}")
    assert result == SECRET_TARGET, f"Expected {SECRET_TARGET} but got {result}"
    print("All tests passed!")