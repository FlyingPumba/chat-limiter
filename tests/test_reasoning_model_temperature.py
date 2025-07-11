"""
Test temperature handling for reasoning models in OpenAI adapter.
"""

import pytest
import warnings
from unittest.mock import patch

from chat_limiter.adapters import OpenAIAdapter
from chat_limiter.types import ChatCompletionRequest, Message, MessageRole


class TestReasoningModelTemperature:
    """Test temperature handling for reasoning models."""

    def test_is_reasoning_model(self):
        """Test reasoning model detection."""
        adapter = OpenAIAdapter()
        
        # Test reasoning models
        assert adapter.is_reasoning_model("o1-preview")
        assert adapter.is_reasoning_model("o1-mini")
        assert adapter.is_reasoning_model("o3-mini")
        assert adapter.is_reasoning_model("o4-preview")
        
        # Test non-reasoning models
        assert not adapter.is_reasoning_model("gpt-4o")
        assert not adapter.is_reasoning_model("gpt-3.5-turbo")
        assert not adapter.is_reasoning_model("claude-3-5-sonnet")

    def test_reasoning_model_temperature_override(self):
        """Test that reasoning models get temperature=1 regardless of input."""
        adapter = OpenAIAdapter()
        
        # Test with very low temperature (should be overridden)
        request = ChatCompletionRequest(
            model="o3-mini",
            messages=[Message(role=MessageRole.USER, content="Test")],
            temperature=1e-19
        )
        
        with patch('builtins.print') as mock_print:
            formatted = adapter.format_request(request)
            
            # Should override to temperature=1
            assert formatted["temperature"] == 1.0
            
            # Should print warning
            mock_print.assert_called_once()
            assert "WARNING" in mock_print.call_args[0][0]
            assert "temperature=1" in mock_print.call_args[0][0]

    def test_reasoning_model_temperature_warning(self):
        """Test that warnings are issued when temperature differs from default."""
        adapter = OpenAIAdapter()
        
        request = ChatCompletionRequest(
            model="o3-mini",
            messages=[Message(role=MessageRole.USER, content="Test")],
            temperature=0.5
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            formatted = adapter.format_request(request)
            
            # Should override to temperature=1
            assert formatted["temperature"] == 1.0
            
            # Should issue warning
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "temperature=1" in str(w[0].message)

    def test_reasoning_model_temperature_no_warning_when_correct(self):
        """Test that no warning is issued when temperature=1 is provided."""
        adapter = OpenAIAdapter()
        
        request = ChatCompletionRequest(
            model="o3-mini",
            messages=[Message(role=MessageRole.USER, content="Test")],
            temperature=1.0
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            with patch('builtins.print') as mock_print:
                formatted = adapter.format_request(request)
                
                # Should keep temperature=1
                assert formatted["temperature"] == 1.0
                
                # Should not issue warning
                assert len(w) == 0
                mock_print.assert_not_called()

    def test_reasoning_model_no_temperature_provided(self):
        """Test that temperature=1 is set when no temperature is provided."""
        adapter = OpenAIAdapter()
        
        request = ChatCompletionRequest(
            model="o3-mini",
            messages=[Message(role=MessageRole.USER, content="Test")]
            # No temperature provided
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            with patch('builtins.print') as mock_print:
                formatted = adapter.format_request(request)
                
                # Should default to temperature=1
                assert formatted["temperature"] == 1.0
                
                # Should not issue warning
                assert len(w) == 0
                mock_print.assert_not_called()

    def test_non_reasoning_model_temperature_passthrough(self):
        """Test that non-reasoning models pass through temperature unchanged."""
        adapter = OpenAIAdapter()
        
        # Test with very low temperature (should be preserved)
        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[Message(role=MessageRole.USER, content="Test")],
            temperature=1e-19
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            with patch('builtins.print') as mock_print:
                formatted = adapter.format_request(request)
                
                # Should preserve original temperature
                assert formatted["temperature"] == 1e-19
                
                # Should not issue warning
                assert len(w) == 0
                mock_print.assert_not_called()

    def test_non_reasoning_model_no_temperature_provided(self):
        """Test that non-reasoning models don't get temperature when none provided."""
        adapter = OpenAIAdapter()
        
        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[Message(role=MessageRole.USER, content="Test")]
            # No temperature provided
        )
        
        formatted = adapter.format_request(request)
        
        # Should not have temperature field
        assert "temperature" not in formatted

    def test_reasoning_model_with_other_parameters(self):
        """Test that other parameters work correctly with reasoning models."""
        adapter = OpenAIAdapter()
        
        request = ChatCompletionRequest(
            model="o3-mini",
            messages=[Message(role=MessageRole.USER, content="Test")],
            temperature=0.5,  # Will be overridden
            max_tokens=100,
            top_p=0.9,
            frequency_penalty=0.1
        )
        
        with patch('builtins.print'):
            formatted = adapter.format_request(request)
            
            # Temperature should be overridden
            assert formatted["temperature"] == 1.0
            
            # Other parameters should be preserved
            assert formatted["max_completion_tokens"] == 100  # Uses max_completion_tokens for reasoning models
            assert formatted["top_p"] == 0.9
            assert formatted["frequency_penalty"] == 0.1
            
            # Should not have max_tokens (uses max_completion_tokens instead)
            assert "max_tokens" not in formatted

    def test_temperature_handling_demo(self):
        """Comprehensive test demonstrating temperature handling for reasoning models."""
        adapter = OpenAIAdapter()
        
        # Test 1: o3-mini with very low temperature (should be overridden)
        request1 = ChatCompletionRequest(
            model="o3-mini",
            messages=[Message(role=MessageRole.USER, content="What is 2+2?")],
            max_tokens=50,
            temperature=1e-19
        )
        
        with patch('builtins.print') as mock_print:
            formatted1 = adapter.format_request(request1)
            
            # Should override to temperature=1
            assert formatted1["temperature"] == 1.0
            assert "max_completion_tokens" in formatted1
            assert "max_tokens" not in formatted1
            
            # Should print warning
            mock_print.assert_called_once()
            assert "WARNING" in mock_print.call_args[0][0]
        
        # Test 2: o3-mini with no temperature (should default to 1.0)
        request2 = ChatCompletionRequest(
            model="o3-mini",
            messages=[Message(role=MessageRole.USER, content="What is 2+2?")],
            max_tokens=50
        )
        
        with patch('builtins.print') as mock_print:
            formatted2 = adapter.format_request(request2)
            
            # Should default to temperature=1
            assert formatted2["temperature"] == 1.0
            
            # Should not print warning
            mock_print.assert_not_called()
        
        # Test 3: gpt-4o with low temperature (should be preserved)
        request3 = ChatCompletionRequest(
            model="gpt-4o",
            messages=[Message(role=MessageRole.USER, content="What is 2+2?")],
            max_tokens=50,
            temperature=1e-19
        )
        
        with patch('builtins.print') as mock_print:
            formatted3 = adapter.format_request(request3)
            
            # Should preserve original temperature
            assert formatted3["temperature"] == 1e-19
            assert "max_tokens" in formatted3
            assert "max_completion_tokens" not in formatted3
            
            # Should not print warning
            mock_print.assert_not_called()
        
        # Test 4: o3-mini with temperature=0.5 (should warn and override)
        request4 = ChatCompletionRequest(
            model="o3-mini",
            messages=[Message(role=MessageRole.USER, content="What is 2+2?")],
            max_tokens=50,
            temperature=0.5
        )
        
        with patch('builtins.print') as mock_print:
            formatted4 = adapter.format_request(request4)
            
            # Should override to temperature=1
            assert formatted4["temperature"] == 1.0
            
            # Should print warning
            mock_print.assert_called_once()
            assert "WARNING" in mock_print.call_args[0][0]