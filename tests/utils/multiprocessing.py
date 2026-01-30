import functools
import time

import psutil


def check_subprocesses_exit(max_timeout=5):
	"""If a component launches processes using pathos or multiprocessing you can use this decorator to check if it correctly kills those processes when finished"""

	def wrapper1(func):
		@functools.wraps(func)
		def wrapper2(*args, **kwargs):
			current_process = psutil.Process()

			def curr_nc():
				return len(current_process.children(recursive=True))

			num_children = curr_nc()

			func(*args, **kwargs)

			# Hacky, but killing a process from the os side might take some time.
			i = 0
			while i < max_timeout:
				i += 0.1
				time.sleep(0.1)
				if curr_nc() == num_children:
					return
			assert curr_nc() == num_children

		return wrapper2

	return wrapper1
