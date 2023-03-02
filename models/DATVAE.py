"""TVAESynthesizer module."""

import numpy as np
import torch

from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch.nn.functional import cross_entropy, gumbel_softmax, relu
from torch.nn import Linear,Module,ReLU,Sequential,functional
from torch.optim import Adam
from packaging import version
from torch.utils.data import DataLoader, TensorDataset
from .data_sampler_edit import DataSampler
from .data_transform import DataTransformer
from .base import BaseSynthesizer, random_state


class Encoder(Module):
    """Encoder for the TVAESynthesizer.

    Args:
        data_dim (int):
            Dimensions of the data.
        compress_dims (tuple or list of ints):
            Size of each hidden layer.
        embedding_dim (int):
            Size of the output vector.
    """

    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [
                Linear(dim, item),
                ReLU()
            ]
            dim = item

        self.seq = Sequential(*seq)
        self.fc1 = Linear(dim, embedding_dim)
        self.fc2 = Linear(dim, embedding_dim)

    def forward(self, input_):
        """Encode the passed `input_`."""
        feature = self.seq(input_)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class Decoder(Module):
    """Decoder for the TVAESynthesizer.

    Args:
        embedding_dim (int):
            Size of the input vector.
        decompress_dims (tuple or list of ints):
            Size of each hidden layer.
        data_dim (int):
            Dimensions of the data.
    """

    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input_):
        """Decode the passed `input_`."""
        return self.seq(input_), self.sigma


def _loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor):
    st = 0
    loss = []
    for column_info in output_info:
        for span_info in column_info:
            if span_info.activation_fn == 'tanh':
                ed = st + span_info.dim
                std = sigmas[st]
                # eq = x[:,st] - torch.tanh(recon_x[:,st])
                eq = x[:, st] - relu(recon_x[:, st])  # *torch.sigmoid(recon_x[:, st])
                loss.append((eq ** 2 / 2 / (std ** 2)).sum())
                loss.append(torch.log(std) * x.size()[0])
                st = ed
            elif span_info.activation_fn == 'swish':
                ed = st + span_info.dim
                std = sigmas[st]
                # eq = x[:,st] - torch.tanh(recon_x[:,st])
                eq = x[:, st] - recon_x[:, st] * torch.sigmoid(recon_x[:, st])
                loss.append((eq ** 2 / 2 / (std ** 2)).sum())
                loss.append(torch.log(std) * x.size()[0])
                st = ed
            else:
                ed = st + span_info.dim
                loss.append(cross_entropy(
                    recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
                st = ed

    assert st == recon_x.size()[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
    return sum(loss) * factor / x.size()[0], KLD / x.size()[0]


class DATVAESynthesizer(BaseSynthesizer):
    """TVAESynthesizer."""

    def __init__(
            self,
            embedding_dim=200,
            compress_dims=(200, 200),
            decompress_dims=(200, 200),
            l2scale=1e-5,
            batch_size=300,
            epochs=300,
            loss_factor=2,
            con_c=0,
            cuda=True,
    ):

        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = loss_factor
        self.epochs = epochs
        self.con_c = con_c

        self.loss1 = []
        self.loss2 = []
        self.loss3 = []

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self.transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.

        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        if version.parse(torch.__version__) < version.parse('1.2.0'):
            for i in range(10):
                transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard,
                                                        eps=eps, dim=dim)
                if not torch.isnan(transformed).any():
                    return transformed
            raise ValueError('gumbel_softmax returning NaN.')

        return functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self.transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'relu':
                    ed = st + span_info.dim
                    data_t.append(torch.relu(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'swish':
                    ed = st + span_info.dim
                    data_t.append(data[:, st:ed] * torch.sigmoid(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')
        return torch.cat(data_t, dim=1)

    @random_state
    def fit(self, train_data, discrete_columns=()):
        """Fit the TVAE Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self.transformer = DataTransformer(max_clusters=self.con_c)
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)
        # dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self._device))
        # loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self._data_sampler = DataSampler(
            train_data,
            self.transformer.output_info_list,
            log_frequency=True)
        data_dim = self.transformer.output_dimensions

        encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim + self._data_sampler.dim_cond_vec(), self.decompress_dims,
                               data_dim).to(self._device)

        # def count_param(model):
        #     param_count = 0
        #     for param in model.parameters():
        #         param_count += param.view(-1).size()[0]
        #     return param_count
        # print('Norm Conv parameter count is {}'.format(count_param(encoder)+count_param(self.decoder)))
        optimizerAE = Adam(
            list(encoder.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.l2scale)
        steps_per_epoch = max(len(train_data) // self.batch_size, 1)
        for i in range(self.epochs):
            loss1, loss2, loss3 = [], [], []
            for id_ in range(steps_per_epoch):
                condvec = self._data_sampler.sample_condvec(self.batch_size)
                c1, m1, col, opt = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                m1 = torch.from_numpy(m1).to(self._device)
                perm = np.arange(self.batch_size)
                np.random.shuffle(perm)
                real = self._data_sampler.sample_data(
                    self.batch_size, col[perm], opt[perm])
                real = torch.from_numpy(real.astype('float32')).to(self._device)
                # c2 = c1[perm]
                # real = torch.cat([real, c2], dim = 1)

                optimizerAE.zero_grad()
                mu, std, logvar = encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                emb = torch.cat([emb, c1], dim=1)
                rec, sigmas = self.decoder(emb)
                loss_1, loss_2 = _loss_function(
                    rec, real, sigmas, mu, logvar,
                    self.transformer.output_info_list, self.loss_factor
                )
                cross_entropy = self._cond_loss(rec, c1, m1)
                loss = loss_1 + loss_2 + cross_entropy
                loss.backward()
                optimizerAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)
                loss1.append(loss_1.detach().cpu().item())
                loss2.append(loss_2.detach().cpu().item())
                loss3.append(cross_entropy.detach().cpu().item())
            print(f'Epoch {i + 1}, Loss 1: {np.mean(loss1): .4f},'  # noqa: T001
                  f'Loss 2: {np.mean(loss2): .4f}',
                  f'Loss 2: {np.mean(loss3): .4f}',
                  flush=True)

            self.loss1.append(np.mean(loss1))
            self.loss2.append(np.mean(loss2))
            self.loss3.append(np.mean(loss3))
            # if len(self.loss1)>10:
            #     if abs(np.mean(loss1) - np.mean(loss2)) < 1:
            #         break

    @random_state
    def sample(self, samples):
        """Sample data similar to the training data.

        Args:
            samples (int):
                Number of rows to sample.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        self.decoder.eval()

        steps = samples // self.batch_size + 1
        data = []
        for _ in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            condvec = self._data_sampler.sample_original_condvec(self.batch_size)
            c1 = condvec
            c1 = torch.from_numpy(c1).to(self._device)

            noise = torch.normal(mean=mean, std=std).to(self._device)
            noise = torch.cat([noise, c1], dim=1)
            fake, sigmas = self.decoder(noise)
            # fake = fake*torch.sigmoid(fake)
            fake = self._apply_activate(fake)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:samples]
        return self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        self.decoder.to(self._device)
