from collections import defaultdict

# TODO: check that my own attn out calculation matches the hook
class ActivationStore:
  def __init__(self, checkpoint_step, num_samples, seed=0):
    self.checkpoint_step = checkpoint_step
    self.num_samples = num_samples
    self.seed = seed

    self.context_positions = []

    self.hook_resid_pre = defaultdict(list)
    self.attn_hook_q = defaultdict(list)
    self.attn_hook_k = defaultdict(list)
    self.attn_hook_v = defaultdict(list)
    self.attn_hook_z = defaultdict(list)
    self.attn_hook_attn_scores = defaultdict(list)
    self.attn_hook_pattern = defaultdict(list)
    self.hook_attn_out = defaultdict(list)
    self.hook_resid_post = defaultdict(list)

  def add_activations(self, cache, batch_index, context_pos):
    self.context_positions.append(context_pos)

    for layer in range(2):
      resid_pre = cache[f'blocks.{layer}.hook_resid_pre'][batch_index][context_pos].detach().cpu()
      self.hook_resid_pre[layer].append(resid_pre.numpy())

      hook_q = cache[f'blocks.{layer}.attn.hook_q'][batch_index][context_pos].detach().cpu()
      self.attn_hook_q[layer].append(hook_q.numpy())

      hook_k = cache[f'blocks.{layer}.attn.hook_k'][batch_index][context_pos].detach().cpu()
      self.attn_hook_k[layer].append(hook_k.numpy())

      hook_v = cache[f'blocks.{layer}.attn.hook_v'][batch_index][context_pos].detach().cpu()
      self.attn_hook_v[layer].append(hook_v.numpy())

      hook_z = cache[f'blocks.{layer}.attn.hook_z'][batch_index][context_pos].detach().cpu()
      self.attn_hook_z[layer].append(hook_z.numpy())

      attn_scores = cache[f'blocks.{layer}.attn.hook_attn_scores'][batch_index, :, context_pos, :].detach().cpu()
      self.attn_hook_attn_scores[layer].append(attn_scores.numpy())

      pattern = cache[f'blocks.{layer}.attn.hook_pattern'][batch_index, :, context_pos, :].detach().cpu()
      self.attn_hook_pattern[layer].append(pattern.numpy())

      attn_out = cache[f'blocks.{layer}.hook_attn_out'][batch_index][context_pos].detach().cpu()
      self.hook_attn_out[layer].append(attn_out.numpy())

      resid_post = cache[f'blocks.{layer}.hook_resid_post'][batch_index][context_pos].detach().cpu()
      self.hook_resid_post[layer].append(resid_post.numpy())

  def finalize(self):
    for layer in range(2):
      self.hook_resid_pre[layer] = np.array(self.hook_resid_pre[layer])
      self.attn_hook_q[layer] = np.array(self.attn_hook_q[layer])
      self.attn_hook_k[layer] = np.array(self.attn_hook_k[layer])
      self.attn_hook_v[layer] = np.array(self.attn_hook_v[layer])
      self.attn_hook_z[layer] = np.array(self.attn_hook_z[layer])
      self.attn_hook_attn_scores[layer] = np.array(self.attn_hook_attn_scores[layer])
      self.attn_hook_pattern[layer] = np.array(self.attn_hook_pattern[layer])
      self.hook_attn_out[layer] = np.array(self.hook_attn_out[layer])
      self.hook_resid_post[layer] = np.array(self.hook_resid_post[layer])