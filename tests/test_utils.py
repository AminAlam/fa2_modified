"""
Tests for utility classes and functions
"""
import pytest
import time

from fa2_modified.forceatlas2 import Timer


class TestTimer:
    """Test Timer utility class"""
    
    def test_timer_initialization(self):
        """Test Timer initialization"""
        timer = Timer()
        assert timer.name == "Timer"
        assert timer.start_time == 0.0
        assert timer.total_time == 0.0
    
    def test_timer_custom_name(self):
        """Test Timer with custom name"""
        timer = Timer(name="TestTimer")
        assert timer.name == "TestTimer"
    
    def test_timer_start_stop(self):
        """Test timer start and stop"""
        timer = Timer()
        timer.start()
        time.sleep(0.01)  # Sleep for 10ms
        timer.stop()
        
        assert timer.total_time > 0.0
        assert timer.total_time < 1.0  # Should be much less than 1 second
    
    def test_timer_multiple_runs(self):
        """Test timer accumulates time over multiple runs"""
        timer = Timer()
        
        timer.start()
        time.sleep(0.01)
        timer.stop()
        first_time = timer.total_time
        
        timer.start()
        time.sleep(0.01)
        timer.stop()
        second_time = timer.total_time
        
        assert second_time > first_time
        # Allow for OS scheduling/timer resolution differences across platforms
        assert second_time >= first_time * 1.05
    
    def test_timer_display(self, capsys):
        """Test timer display output"""
        timer = Timer(name="DisplayTest")
        timer.start()
        time.sleep(0.01)
        timer.stop()
        
        timer.display()
        
        captured = capsys.readouterr()
        assert "DisplayTest" in captured.out
        assert "seconds" in captured.out
    
    def test_timer_zero_time(self):
        """Test timer with no elapsed time"""
        timer = Timer()
        assert timer.total_time == 0.0
        
        # Start and immediately stop
        timer.start()
        timer.stop()
        
        # Time should still be very small (close to zero)
        assert timer.total_time >= 0.0

