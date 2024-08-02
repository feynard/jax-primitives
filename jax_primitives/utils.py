import jax


class RandomKey:

    def __init__(self, seed: int = 0):
        self.key = jax.random.key(seed)

    def split(self, n: int = 1):
        keys = jax.random.split(self.key, n + 1)
        self.key = keys[0]
        
        if n > 1:
            return keys[1:]
        else:
            return keys[1]

    def __floordiv__(self, n: int):
        return self.split(n)
    
    def __call__(self, n: int = 1):
        return self.split(n)
