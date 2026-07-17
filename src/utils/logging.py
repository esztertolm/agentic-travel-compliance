import os
import logging
from datetime import datetime
from configuration.config import AppConfig

def setup_logger():
    """
    Configures the root logger once at the application startup.
    Generates a unique log file named with the current timestamp.
    """
    os.makedirs(AppConfig.LOGS_DIR, exist_ok=True)
    
    root_logger = logging.getLogger()
    
    # Only configure if handlers are not already set
    if not root_logger.handlers:
        root_logger.setLevel(logging.INFO)
        
        # Standard format showing: Time [Level] ModuleName: Message
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        
        # Generate unique log filename using current date and time
        # Example: app_20260717_084037.log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"app_{timestamp}.log"
        log_file_path = os.path.join(AppConfig.LOGS_DIR, log_filename)
        
        # File Handler for this specific run
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Stream Handler for real-time console output (Docker/Streamlit logs)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)
        
        # Log the initialization details as the very first entry
        logging.getLogger("logging_config").info(
            "Logger is initialized. Log file created at: %s", log_file_path
        )