import jax
import numpy as np

dim = 256

w_key, b_key = jax.random.split(jax.random.PRNGKey(42))
weight = jax.random.normal(w_key, shape=[dim])
bias = jax.random.uniform(b_key, shape=[dim])
print(weight.shape)
print(bias.shape)
np.save('/root/torchfold3/fourier_embeddings/weight.npy', np.array(weight))
np.save('/root/torchfold3/fourier_embeddings/bias.npy', np.array(bias))