import logging
import os

def setup_logger():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "agentic_chatbot.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger("agentic_chatbot")
    logger.setLevel(logging.DEBUG)
    logger.info("Logger initialized.")
    return logger
