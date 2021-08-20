import logging

def get_logger(logger_name, log_file, f_fmt='%(message)s'):

	logger = logging.getLogger(logger_name)
	logger.setLevel(logging.DEBUG)
	stream_handler = logging.StreamHandler()
	stream_handler.setLevel(logging.DEBUG)
	stream_handler.setFormatter(logging.Formatter('%(message)s'))
	logger.addHandler(stream_handler)

	file_handler = logging.FileHandler(log_file, encoding='utf-8')
	file_handler.setLevel(logging.DEBUG)
	file_handler.setFormatter(logging.Formatter(f_fmt))
	logger.addHandler(file_handler)

	return logger