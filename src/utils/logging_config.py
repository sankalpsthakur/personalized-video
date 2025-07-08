"""
Logging configuration for video personalization pipeline
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(log_level="INFO", log_file=None, verbose=False):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        verbose: Enable verbose console output
    """
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Set up formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO if not verbose else logging.DEBUG)
    console_handler.setFormatter(simple_formatter if not verbose else detailed_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
    else:
        # Auto-generate log file name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"pipeline_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
    
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Configure specific loggers
    loggers_config = {
        'lip_sync': logging.INFO,
        'lip_sync_models': logging.INFO,
        'personalization_pipeline': logging.INFO,
        'urllib3': logging.WARNING,  # Reduce noise from HTTP requests
        'PIL': logging.WARNING,      # Reduce noise from image processing
    }
    
    for logger_name, level in loggers_config.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
    
    # Log startup info
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("Video Personalization Pipeline Started")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Verbose mode: {verbose}")
    logger.info("="*60)
    
    return log_file


def get_logger(name):
    """Get a logger instance"""
    return logging.getLogger(name)


# Performance logging utilities
class PerformanceLogger:
    """Context manager for logging performance metrics"""
    
    def __init__(self, operation_name, logger=None):
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation_name}...")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        if exc_type:
            self.logger.error(f"{self.operation_name} failed after {elapsed:.2f}s: {exc_val}")
        else:
            self.logger.info(f"✓ {self.operation_name} completed in {elapsed:.2f}s")


# Example usage
if __name__ == "__main__":
    import time
    
    # Setup logging
    setup_logging(log_level="DEBUG", verbose=True)
    
    # Test logging
    logger = get_logger(__name__)
    
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Test performance logging
    with PerformanceLogger("test operation"):
        time.sleep(1)
        
    print(f"\nLogs saved to: logs/")