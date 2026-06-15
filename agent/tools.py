import subprocess
import sys
from langchain_core.tools import tool


@tool
def read_file(filepath: str) -> str:
    """Read the contents of a Python source file.

    Args:
        filepath: Path to the Python file to read.

    Returns:
        The file contents as a string, or an error message.
    """
    try:
        with open(filepath, "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File '{filepath}' not found."
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def write_file(filepath: str, content: str) -> str:
    """Write or overwrite a Python source file with new content.

    Args:
        filepath: Path to the file to write.
        content: The full content to write to the file.

    Returns:
        A confirmation message or an error message.
    """
    try:
        with open(filepath, "w") as f:
            f.write(content)
        return f"Successfully wrote to '{filepath}'."
    except Exception as e:
        return f"Error writing file: {str(e)}"


@tool
def run_code(filepath: str) -> str:
    """Execute a Python file and return its output.

    Args:
        filepath: Path to the Python file to execute.

    Returns:
        stdout output if successful, or stderr if an error occurred.
    """
    try:
        result = subprocess.run(
            [sys.executable, filepath],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout or "Code ran successfully with no output."
        else:
            return f"Error (exit code {result.returncode}):\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out after 30 seconds."
    except Exception as e:
        return f"Error running code: {str(e)}"


TOOLS = [read_file, write_file, run_code]