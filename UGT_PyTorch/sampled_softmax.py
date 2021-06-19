import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

from log_uniform import LogUniformSampler

"""LogUniformSampler is taken from https://github.com/rdspring1/PyTorch_GBW_LM"""

class SampledSoftmax(nn.Module):
    def __init__(self, ntokens, nsampled, nhid, device):
        super(SampledSoftmax, self).__init__()

        # Parameters
        self.ntokens = ntokens
        self.nsampled = nsampled
        self.device = device
        #
        self.sampler = LogUniformSampler(self.ntokens)
        #
        self.weight = nn.Parameter(torch.Tensor(ntokens, nhid))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs, labels):
        # sample ids according to word distribution - Unique
        sample_values = self.sampler.sample(self.nsampled, labels.data.cpu().numpy())
        return self.sampled(inputs, labels, sample_values)

    """@Dai Quoc Nguyen: Implement the sampled softmax loss function as described in the paper
    On Using Very Large Target Vocabulary for Neural Machine Translation https://www.aclweb.org/anthology/P15-1001/"""
    def sampled(self, inputs, labels, sample_values):
        assert(inputs.data.get_device() == labels.data.get_device())

        batch_size, d = inputs.size()
        sample_ids, true_freq, sample_freq = sample_values

        sample_ids = Variable(torch.LongTensor(sample_ids)).to(self.device)

        # gather true labels
        true_weights = torch.index_select(self.weight, 0, labels)

        # gather sample ids
        sample_weights = torch.index_select(self.weight, 0, sample_ids)

        # calculate logits
        true_logits = torch.exp(torch.sum(torch.mul(inputs, true_weights), dim=1))
        sample_logits = torch.exp(torch.matmul(inputs, torch.t(sample_weights)))

        logits = -torch.log(true_logits / torch.sum(sample_logits, dim=1))

        return logits
