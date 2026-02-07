"""Tests for the calculator tool."""

import pytest

from src.tools.calculator import calculate


class TestCalculator:
    def test_addition(self):
        result = calculate.invoke({"expression": "2 + 3"})
        assert result == "5"

    def test_multiplication(self):
        result = calculate.invoke({"expression": "4 * 5"})
        assert result == "20"

    def test_complex_expression(self):
        result = calculate.invoke({"expression": "(2 + 3) * 4"})
        assert result == "20"

    def test_division(self):
        result = calculate.invoke({"expression": "10 / 3"})
        assert "3.333" in result

    def test_power(self):
        result = calculate.invoke({"expression": "2 ** 10"})
        assert result == "1024"

    def test_negative(self):
        result = calculate.invoke({"expression": "-5 + 3"})
        assert result == "-2"

    def test_division_by_zero(self):
        result = calculate.invoke({"expression": "1 / 0"})
        assert "Error" in result

    def test_invalid_expression(self):
        result = calculate.invoke({"expression": "import os"})
        assert "Error" in result
