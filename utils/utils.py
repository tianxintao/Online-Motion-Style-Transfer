import os
import logging

style_labels = ["angry", "childlike", "depressed", "old", "proud", "sexy", "strutting"]


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def create_logger(output_path):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger_name = os.path.join(output_path, 'session.log')
    file_handler = logging.FileHandler(logger_name)
    console_handler = logging.StreamHandler()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    return logger

def get_style_name(style_vector):
    if style_vector.sum() == 0.0:
        style = "neutral"
    else:
        style = style_labels[(style_vector==1).nonzero(as_tuple=True)[0]]
    return style
