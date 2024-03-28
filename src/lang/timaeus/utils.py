CONFIG = [
    (10_000, 100),
    (20_000, 200),
    (50_000, 500),
    (100_000, 1000),
    (200_000, 2000),
    (800_000, 5000),
]

def get_sparse_steps_gpt2(step_config=CONFIG):
  steps = []
  curr_left = 0
  for config in step_config:
    curr_right = config[0]
    step_size = config[1]
    steps += list(range(curr_left, curr_right, step_size))
    curr_left = curr_right
  return steps