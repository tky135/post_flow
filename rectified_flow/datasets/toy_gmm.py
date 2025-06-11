import torch
import torch.nn as nn
import torch.distributions as dist
from torch.distributions import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
import numpy as np

def mixture_sample_with_labels(self, sample_shape=torch.Size()):
    labels = self.mixture_distribution.sample(sample_shape)
    all_samples = self.component_distribution.sample(sample_shape)
    x_samples = all_samples[torch.arange(len(labels)), labels]
    return x_samples, labels

class LowDimData():
    def __init__(self, data_type, device):
        self.batchsize = 100000
        self.device = device
        self.data_init(data_type)
        self.pairs = torch.stack([self.x0, self.x1], dim=1)

    def data_init(self, data_type):
        if data_type == "1to2":
            self.dim = 1
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
            self.initial_model = MultivariateNormal(torch.zeros(self.dim), torch.eye(self.dim))
            self.x1 = self.target_model.sample([self.batchsize]).to(self.device).detach()
            self.x0 = self.initial_model.sample([self.batchsize]).to(self.device).detach()
            # print(f"means and var before norm: \n{self.means.numpy()} {self.var.numpy()}")
            # self.x1 = (self.x1-torch.mean(self.x1)) / torch.std(self.x1)
            # print(f"means and var after norm {torch.mean(self.x1).item():.3f}, var: {torch.var(self.x1).item():.3f}")
        elif data_type == "2to2":
            self.dim = 1
            self.mean = torch.tensor([1])
            self.means = torch.ones((2, self.dim)) * self.mean
            self.means[1] = -self.means[1]
            self.var = torch.tensor([0.1])
            self.covs = self.var * torch.stack([torch.eye(self.dim) for _ in range(2)])
            self.probs = torch.tensor([0.5, 0.5])
            target_mix = Categorical(self.probs)
            target_comp = MultivariateNormal(self.means, self.covs)
            self.target_model = MixtureSameFamily(target_mix, target_comp)
            initial_mix = Categorical(self.probs)
            initial_comp = MultivariateNormal(self.means, self.covs)
            self.target_model = MixtureSameFamily(target_mix, target_comp)
            self.initial_model = MixtureSameFamily(initial_mix, initial_comp)
            self.x1 = self.target_model.sample([self.batchsize]).to(self.device).detach()
            self.x0 = self.initial_model.sample([self.batchsize]).to(self.device).detach()
            # print(f"means and var before norm: \n{self.means.numpy()} {self.var.numpy()}")
            # self.x1 = (self.x1-torch.mean(self.x1)) / torch.std(self.x1)
            # print(f"means and var after norm {torch.mean(self.x1).item():.3f}, var: {torch.var(self.x1).item():.3f}")
        elif data_type == "1to5":
            self.dim = 1
            self.means = torch.ones((5, self.dim)) * 5
            for i in range(5):
                self.means[i] *= i-2
            self.var = torch.tensor([0.5])
            self.covs = self.var * torch.stack([torch.eye(self.dim) for _ in range(5)])
            self.probs = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
            target_mix = Categorical(self.probs)
            target_comp = MultivariateNormal(self.means, self.covs)
            self.target_model = MixtureSameFamily(target_mix, target_comp)
            self.initial_model = MultivariateNormal(torch.zeros(self.dim), torch.eye(self.dim))
            self.x1 = self.target_model.sample([self.batchsize]).to(self.device).detach()
            self.x0 = self.initial_model.sample([self.batchsize]).to(self.device).detach()
            print(f"means and var before norm: \n{self.means.numpy()} {self.var.numpy()}")
            self.x1 = (self.x1-torch.mean(self.x1)) / torch.std(self.x1)
            print(f"means and var after norm {torch.mean(self.x1).item():.3f}, var: {torch.var(self.x1).item():.3f}")
        elif data_type == "2D1to6":
            self.dim = 2
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
            self.initial_model = MultivariateNormal(torch.zeros(self.dim), torch.eye(self.dim))
            self.x1 = self.target_model.sample([self.batchsize]).to(self.device).detach()
            self.x0 = self.initial_model.sample([self.batchsize]).to(self.device).detach()
            print(f"means and var before norm: \n{self.means.numpy()} {self.var.numpy()}")
            self.x1 = (self.x1-torch.mean(self.x1)) / torch.std(self.x1)
            print(f"means and var after norm {torch.mean(self.x1).item():.3f}, var: {torch.var(self.x1).item():.3f}")
        elif data_type == "moon":
            self.dim = 2
            n_samples_out = self.batchsize // 2
            n_samples_in = self.batchsize - n_samples_out
            outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
            outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
            inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
            inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - .5
            X = np.vstack([np.append(outer_circ_x, inner_circ_x),
                        np.append(outer_circ_y, inner_circ_y)]).T
            X += np.random.rand(self.batchsize, 1) * 0.2
            self.x1 = (torch.from_numpy(X) * 3 - 1).float().to(self.device).detach()
            self.x1 = self.x1[torch.randperm(self.batchsize)]
            
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
            self.initial_model = MixtureSameFamily(initial_mix, initial_comp)
            self.x0 = self.initial_model.sample([self.batchsize]).to(self.device).detach()
        else:
            raise NotImplementedError
# Circular GMM Class
class CircularGMM(dist.MixtureSameFamily):
    def __init__(
        self, n_components=6, radius=10, dim=2, std=1.0, device=torch.device("cpu")
    ):
        self.device = device
        angles = torch.linspace(0, 2 * torch.pi, n_components + 1)[:-1].to(device)
        means = torch.stack(
            [radius * torch.cos(angles), radius * torch.sin(angles)], dim=1
        ).to(device)
        stds = std * torch.ones(n_components, dim).to(device)
        weights = torch.ones(n_components).to(device) / n_components

        # Initialize the MixtureSameFamily distribution
        super().__init__(
            dist.Categorical(weights), dist.Independent(dist.Normal(means, stds), 1)
        )

    def sample_with_labels(self, sample_shape=torch.Size()):
        return mixture_sample_with_labels(self, sample_shape)


# Two-point GMM Class
class TwoPointGMM(dist.MixtureSameFamily):
    def __init__(self, x=10.0, y=10.0, std=1.0, device=torch.device("cpu")):
        self.device = device
        means = torch.tensor([[x, y], [x, -y]]).to(device)
        stds = torch.ones(2, 2).to(device) * std
        weights = torch.ones(2).to(device) / 2

        # Initialize the MixtureSameFamily distribution
        super().__init__(
            dist.Categorical(weights), dist.Independent(dist.Normal(means, stds), 1)
        )

    def sample_with_labels(self, sample_shape=torch.Size()):
        return mixture_sample_with_labels(self, sample_shape)
# class CheckerboardData(Dataset):
#     def __init__(
#             self,
#             n_rc=4,
#             n_samples=1e8,
#             thickness=1.0,
#             scale=1,
#             shift=[0.0, 0.0],
#             rotation=0.0,
#             test_mode=False):
#         super().__init__()
#         self.n_rc = n_rc
#         self.n_samples = int(n_samples)
#         self.thickness = thickness
#         self.scale = scale
#         self.shift = torch.tensor(shift, dtype=torch.float32)
#         self.rotation = rotation
#         white_squares = [(i, j) for i in range(n_rc) for j in range(n_rc) if (i + j) % 2 == 0]
#         self.white_squares = torch.tensor(white_squares, dtype=torch.float32)
#         self.n_squares = len(white_squares)
#         self.samples = self.draw_samples(self.n_samples)

#     def draw_samples(self, n_samples):
#         chosen_indices = torch.randint(0, self.n_squares, size=(n_samples, ))
#         chosen_squares = self.white_squares[chosen_indices]
#         square_samples = torch.rand(n_samples, 2, dtype=torch.float32)
#         if self.thickness < 1:
#             square_samples = square_samples - 0.5
#             square_samples_r = square_samples.square().sum(dim=-1, keepdims=True)
#             square_samples_angle = torch.atan2(square_samples[:, 1], square_samples[:, 0]).unsqueeze(-1)
#             max_r = torch.minimum(
#                 0.5 / square_samples_angle.cos().abs().clamp(min=1e-6),
#                 0.5 / square_samples_angle.sin().abs().clamp(min=1e-6)).square()
#             square_samples_r_scaled = max_r - (max_r - square_samples_r) * self.thickness ** 0.5
#             square_samples *= (square_samples_r_scaled / square_samples_r).sqrt()
#             square_samples = square_samples + 0.5
#         samples = (chosen_squares + square_samples) * (2 / self.n_rc) - 1
#         if self.rotation != 0.0:
#             angle = torch.tensor(self.rotation, dtype=torch.float32) * torch.pi / 180
#             rotation_matrix = torch.tensor([[torch.cos(angle), -torch.sin(angle)],
#                                             [torch.sin(angle), torch.cos(angle)]])
#             samples = samples @ rotation_matrix
#         return samples * self.scale + self.shift

#     def __len__(self):
#         return self.n_samples

#     def __getitem__(self, idx):
#         data = dict(x=self.samples[idx])
#         return data
