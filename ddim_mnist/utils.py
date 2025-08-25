import tensorflow as tf

def forward_ddpm(x, beta_start = 0.0001, beta_end = 0.02, time = None):
  if time is None:
    time = tf.random.uniform([], 0, 1)
  
  beta_t = beta_start + time * (beta_end - beta_start)
  alpha_t = 1.0 - beta_t
  
  # генерируем шум
  noise = tf.random.normal(tf.shape(x))
  
  # DDPM forward process
  noised_x = tf.sqrt(alpha_t) * x + tf.sqrt(1 - alpha_t) * noise
  
  time_tensor = tf.reshape(time, [1])
  return time_tensor, noised_x, noise