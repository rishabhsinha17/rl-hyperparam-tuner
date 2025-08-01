import jax
import jax.numpy as jnp
from jax import random, pmap
from functools import partial
import time

def rng_key():
    return random.PRNGKey(0)

def conv_block(x, w, b, s):
    x = jax.lax.conv_general_dilated(x, w, (s,s), 'SAME')
    x = x + b
    return jax.nn.relu(x)

def forward(params, x):
    w1, b1, w2, b2 = params
    x = conv_block(x, w1, b1, 1)
    x = conv_block(x, w2, b2, 2)
    return jnp.mean(x)

def main():
    devs = jax.local_devices()
    params = [random.normal(rng_key(), (3,3,3,64)), jnp.zeros(64),
              random.normal(rng_key(), (3,3,64,64)), jnp.zeros(64)]
    params = [pmap(lambda y: y)(jnp.array_split(p, len(devs))) for p in params]
    x = random.normal(rng_key(), (len(devs),128,32,32,3))
    f = pmap(partial(forward, params))
    t0 = time.time()
    y = f(x)
    t1 = time.time()
    imgs = x.shape[0] * x.shape[1]
    print(imgs / (t1 - t0))

if __name__ == '__main__':
    main()
