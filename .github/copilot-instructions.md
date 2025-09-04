description: 'Python coding conventions and guidelines'
applyTo: '**/*.py'
---

# Python Coding Conventions

## Python Instructions

- Write clear and concise comments for each function.
- Ensure functions have descriptive names and include type hints.
- Provide docstrings following PEP 257 conventions.
- Use the `typing` module for type annotations (e.g., `List[str]`, `Dict[str, int]`).
- Break down complex functions into smaller, more manageable functions.
- Write docstrings and comments primarily in Korean (unless external libraries or public APIs require English).

## General Instructions

- Always prioritize readability and clarity.
- For algorithm-related code, include explanations of the approach used.
- Write code with good maintainability practices, including comments on why certain design decisions were made.
- Handle edge cases and write clear exception handling.
- For libraries or external dependencies, mention their usage and purpose in comments.
- Use consistent naming conventions and follow language-specific best practices.
- Ensure all referenced functions, classes, and variables are properly declared before use.
- Carefully read and comprehensively understand each function, loop, and conditional (and their combinations); when appropriate, refactor into more efficient one-liners/two-liners or leverage built-in/vectorized methods from `numpy`, `pandas`, `datetime`, `itertools`, etc., while preserving behavior and clarity.
- Write concise, efficient, and idiomatic code that is also easily understandable.

## Code Style and Formatting
- Use snake_case for variable and function names. Use CamelCase for class names. Follow PEP 8 style guidelines. Include type hints for function parameters and return types.
- Follow the **PEP 8** style guide for Python.
- Maintain proper indentation (use 4 spaces for each level of indentation).
- Ensure lines do not exceed 79 characters.
- Place function and class docstrings immediately after the `def` or `class` keyword.
- Use blank lines to separate functions, classes, and code blocks where appropriate.

## Edge Cases and Testing

- Always include test cases for critical paths of the application.
- Account for common edge cases like empty inputs, invalid data types, and large datasets.
- Include comments for edge cases and the expected behavior in those cases.
- Write unit tests for functions and document them with docstrings explaining the test cases.

## Example of Proper Documentation

```python
def calculate_area(radius: float) -> float:
    """
    원의 면적을 계산하는 함수
    
    주어진 반지름(`radius`)을 이용하여 원의 면적을 계산한다. 
    면적은 π * (반지름^2) 공식을 사용한다.
    
    :param float radius: 원의 반지름
    :return: 원의 면적 값
    :rtype: float
    """
    import math
    return math.pi * radius ** 2
