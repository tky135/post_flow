import torch
import math
import numpy as np
from torchvision import datasets, transforms
from typing import Union
from torch.distributions import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
class Distribution:
    def __init__(self):
        self.data_shape = None
        self.batch_size = 1024  # Default batch size, can be overridden by subclasses
    def sample(self, shape: Union[int, tuple, torch.Size]):
        """
        randomly sample data with shape [*shape, *self.data_shape]
        """
        if isinstance(shape, int):
            shape = (shape,)
        batch_size = math.prod(shape)
        data = self.sample_batch(batch_size)
        return data.view(*shape, *self.data_shape)
    def sample_batch(self, batch_size):
        """
        randomly sample data with shape [batch_size, *self.data_shape]
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def __iter__(self):
        return self
    def __next__(self):
        return self.sample_batch(self.batch_size)

class MGaussian1D(Distribution):
    def __init__(self):
        self.dim = 1
        self.data_shape = (self.dim,)
        self.mean = torch.tensor([1])
        # self.mean = torch.tensor([5])
        self.means = torch.ones((2, self.dim)) * self.mean
        self.means[1] = -self.means[1]
        self.var = torch.tensor([0.02])
        # self.var = torch.tensor([0.5])
        self.covs = self.var * torch.stack([torch.eye(self.dim) for _ in range(2)])
        self.probs = torch.tensor([0.5, 0.5])
        target_mix = Categorical(self.probs)
        target_comp = MultivariateNormal(self.means, self.covs)
        self.target_model = MixtureSameFamily(target_mix, target_comp)
        
        

    def sample_batch(self, batch_size):
        return self.target_model.sample([batch_size])
        
class SixGaussians2D(Distribution):
    def __init__(self):
        self.dim = 2
        self.data_shape = (self.dim,)
        D = 10.
        self.probs = torch.tensor([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
        self.means = torch.tensor([
            [D * np.sqrt(3) / 2., D / 2.], 
            [-D * np.sqrt(3) / 2., D / 2.], 
            [0.0, -D],
            [D * np.sqrt(3) / 2., - D / 2.], [-D * np.sqrt(3) / 2., - D / 2.], [0.0, D]
        ]).float()
        self.var = torch.tensor([0.5])
        self.covs = self.var * torch.stack([torch.eye(self.dim) for _ in range(6)])
        target_mix = Categorical(self.probs)
        target_comp = MultivariateNormal(self.means, self.covs)
        self.target_model = MixtureSameFamily(target_mix, target_comp)

        self.mean = self.target_model.mean
        self.std = self.target_model.stddev

    def sample_batch(self, batch_size):
        return (self.target_model.sample([batch_size]) - self.mean) / self.std
    
class EightGaussians2D(Distribution):
    def __init__(self):
        self.dim = 2
        self.data_shape = (self.dim,)
        self.probs = torch.tensor([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8])
        self.means = torch.tensor([
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        ]).float() * 5
        self.var = torch.tensor([0.1])
        self.covs = self.var * torch.stack([torch.eye(self.dim) for _ in range(8)])
        initial_mix = Categorical(self.probs)
        initial_comp = MultivariateNormal(self.means, self.covs)
        self.target_model = MixtureSameFamily(initial_mix, initial_comp)
    def sample_batch(self, batch_size):
        return self.target_model.sample([batch_size])

class Moon(Distribution):
    def __init__(self, size=100000):
        self.data_shape = (2,)
        n_samples_out = size // 2
        n_samples_in = size - n_samples_out
        outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
        outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
        inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
        inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - .5
        X = np.vstack([np.append(outer_circ_x, inner_circ_x),
                    np.append(outer_circ_y, inner_circ_y)]).T
        X += np.random.rand(size, 1) * 0.2
        samples = (torch.from_numpy(X) * 3 - 1).float()
        self.samples = samples[torch.randperm(size)]
    def sample_batch(self, batch_size):
        if batch_size > len(self.samples):
            raise ValueError(f"Requested {batch_size} samples, but only {len(self.samples)} are available.")
        random_indices = torch.randperm(len(self.samples))[:batch_size]
        return self.samples[random_indices].view(batch_size, *self.data_shape)

class Gaussian(Distribution):
    def __init__(self, data_shape=(2,)):
        self.data_shape = data_shape
    def sample_batch(self, batch_size):
        return torch.randn(batch_size, *self.data_shape)

class Cifar10(Distribution):
    def __init__(self, train=True):

        self.dataset = datasets.CIFAR10(
            root="./data",
            train=train,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
        self.data_shape = (3, 32, 32)  # CIFAR-10 images are 32x32 with 3 color channels
    def sample_batch(self, batch_size):

        if batch_size == 1:
            random_idx = torch.randint(0, len(self.dataset), (batch_size,))
            data, _ = self.dataset[random_idx]
            return data
        else:
            dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,  # Set to 0 for simplicity in this example
            )
            data, _ = next(iter(dataloader))
            return data





class FlowData(torch.utils.data.IterableDataset):

    def __init__(self, source: Distribution = None, target: Distribution = None, batch_size=1024, light_weight=False, data_shape=(2,)):

        if source is None or target is None:
            raise ValueError("Both source and target distributions must be provided.")
        self.source = source
        self.target = target
        self.batch_size = batch_size
        self.light_weight = light_weight
        self.data_shape = data_shape
        if light_weight:
            self.source.batch_size = batch_size
            self.target.batch_size = batch_size
        else:
            self.source.batch_size = 1
            self.target.batch_size = 1


    def __iter__(self):
        for source_data, target_data in zip(self.source, self.target):
            if not self.light_weight:
                source_data = source_data.view(*self.data_shape)
                target_data = target_data.view(*self.data_shape)
            else:
                source_data = source_data.view(self.batch_size, *self.data_shape)
                target_data = target_data.view(self.batch_size, *self.data_shape)
            yield source_data, target_data



if __name__ == "__main__":
    # Example usage
    cifar10_dist = Cifar10(train=True)
    sample = cifar10_dist.sample((2, 3, 32, 32))
    print(sample.shape)  # Should print torch.Size([2, 3, 32, 32])
    
    flow_data = FlowData(source=Moon(), target=Cifar10(train=False))
    for source_data, target_data in flow_data:
        print(source_data.shape, target_data.shape)
        break  # Just to demonstrate the iteration

    dataloader = torch.utils.data.DataLoader(
        flow_data,
        batch_size=4,
        num_workers=2,
    )
    for batch in dataloader:
        source_batch, target_batch = batch
        print(source_batch.shape, target_batch.shape)
        break  # Just to demonstrate the dataloader