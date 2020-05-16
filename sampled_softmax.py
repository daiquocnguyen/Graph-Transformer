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

        self.sampler = LogUniformSampler(self.ntokens)
        self.params = nn.Linear(nhid, ntokens)

        self.device = device

    def forward(self, inputs, labels):
        # sample ids according to word distribution - Unique
        sample_values = self.sampler.sample(self.nsampled, labels.data.cpu().numpy())
        return self.sampled(inputs, labels, sample_values, remove_accidental_match=True)

    """@Dai Quoc Nguyen: Implement the sampled softmax loss function as described in the paper
    On Using Very Large Target Vocabulary for Neural Machine Translation https://www.aclweb.org/anthology/P15-1001/"""
    def sampled(self, inputs, labels, sample_values, remove_accidental_match=False):
        assert(inputs.data.get_device() == labels.data.get_device())

        batch_size, d = inputs.size()
        sample_ids, true_freq, sample_freq = sample_values

        sample_ids = Variable(torch.LongTensor(sample_ids)).to(self.device)

        # gather true labels - weights and frequencies
        true_weights = torch.index_select(self.params.weight, 0, labels)

        # gather sample ids - weights and frequencies
        sample_weights = torch.index_select(self.params.weight, 0, sample_ids)

        # calculate logits
        true_logits = torch.exp(torch.sum(torch.mul(inputs, true_weights), dim=1))
        sample_logits = torch.exp(torch.matmul(inputs, torch.t(sample_weights)))

        logits = -torch.log(true_logits / torch.sum(sample_logits, dim=1))

        return logits
