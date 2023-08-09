from typing import Sequence, Optional, Callable
import flax.linen as nn


class MLP(nn.Module):
    features: Sequence[int]
    output_dim: Optional[int]
    activation: Callable = nn.swish

    @nn.compact
    def __call__(self, x):
        for feat in self.features:
            x = nn.Dense(features=feat)(x)
            x = self.activation(x)
        if self.output_dim is not None:
            x = nn.Dense(features=self.output_dim)(x)
        return x
