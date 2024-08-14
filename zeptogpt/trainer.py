from tqdm import tqdm
from flax.training import train_state

import jax
import jax.numpy as jnp

class Trainer():
    def __init__(self, state: train_state.TrainState, loss_fn, train_batch_fn, test_batch_fn, eval_iters: int, eval_interval: int, num_training_iters: int):
        self.state = state
        self.loss_fn = jax.jit(loss_fn)
        self.train_batch_fn = train_batch_fn
        self.test_batch_fn = test_batch_fn
        self.eval_iters = eval_iters
        self.eval_interval = eval_interval
        self.num_training_iters = num_training_iters
    
    def train(self, key):        
        @jax.jit
        def train_step(key, state):
            def grad_fn(params, inputs, targets):
                logits = state.apply_fn(params, inputs)
                return self.loss_fn(logits, targets)
            inputs, targets = self.train_batch_fn(key)
            grads = jax.grad(grad_fn)(state.params, inputs, targets)
            return state.apply_gradients(grads=grads)
        
        def estimate_loss(key, batch_fn):
            losses = []
            for _ in range(self.eval_iters):
                key, subkey = jax.random.split(key)
                inputs, targets = batch_fn(subkey)
                logits = jax.jit(self.state.apply_fn)(self.state.params, inputs)
                loss = self.loss_fn(logits, targets)
                losses.append(loss)
            return jnp.mean(jnp.array(losses))

        # Training loop
        for i in tqdm(range(self.num_training_iters)):
            key, subkey = jax.random.split(key)
            self.state = train_step(subkey, self.state)
            
            if i % self.eval_interval == 0 or i == self.num_training_iters - 1:
                key, subkey1, subkey2 = jax.random.split(key, num=3)
                train_loss = estimate_loss(subkey1, self.train_batch_fn)
                test_loss = estimate_loss(subkey2, self.test_batch_fn)
                print(f"Train Loss={train_loss} Test Loss={test_loss}")
        return self.state