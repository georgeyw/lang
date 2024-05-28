CONFIG = [
    (10_000, 100),
    (20_000, 200),
    (50_000, 500),
    (100_000, 1000),
    (200_000, 2000),
    (800_001, 5000),
]

def log_steps(max_step = None, step_config=CONFIG):
  steps = []
  curr_left = 0
  for config in step_config:
    curr_right = config[0]
    step_size = config[1]
    steps += list(range(curr_left, curr_right, step_size))
    curr_left = curr_right

  if max_step is not None:
    steps = [step for step in steps if step <= max_step]
    
  return steps