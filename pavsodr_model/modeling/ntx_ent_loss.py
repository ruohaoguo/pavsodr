from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import distributed as torch_dist


class MemoryBankModule(torch.nn.Module):
    def __init__(self, size: int = 2**16):
        super(MemoryBankModule, self).__init__()

        if size < 0:
            msg = f"Illegal memory bank size {size}, must be non-negative."
            raise ValueError(msg)

        self.size = size
        self.register_buffer(
            "bank", tensor=torch.empty(0, dtype=torch.float), persistent=False
        )
        self.register_buffer(
            "bank_ptr", tensor=torch.empty(0, dtype=torch.long), persistent=False
        )

    @torch.no_grad()
    def _init_memory_bank(self, dim: int):
        self.bank = torch.randn(dim, self.size).type_as(self.bank)
        self.bank = torch.nn.functional.normalize(self.bank, dim=0)
        self.bank_ptr = torch.zeros(1).type_as(self.bank_ptr)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, batch: torch.Tensor):
        batch_size = batch.shape[0]
        ptr = int(self.bank_ptr)

        if ptr + batch_size >= self.size:
            self.bank[:, ptr:] = batch[: self.size - ptr].T.detach()
            self.bank_ptr[0] = 0
        else:
            self.bank[:, ptr : ptr + batch_size] = batch.T.detach()
            self.bank_ptr[0] = ptr + batch_size

    def forward(
        self, output: torch.Tensor, labels: torch.Tensor = None, update: bool = False
    ):
        # no memory bank, return the output
        if self.size == 0:
            return output, None

        _, dim = output.shape

        # initialize the memory bank if it is not already done
        if self.bank.nelement() == 0:
            self._init_memory_bank(dim)

        # query and update memory bank
        bank = self.bank.clone().detach()

        # only update memory bank if we later do backward pass (gradient)
        if update:
            self._dequeue_and_enqueue(output)

        return output, bank



class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        ctx.save_for_backward(input)
        output = [torch.empty_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads: torch.Tensor) -> torch.Tensor:
        (input,) = ctx.saved_tensors
        grad_out = torch.empty_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


def rank() -> int:
    """Returns the rank of the current process."""
    return dist.get_rank() if dist.is_initialized() else 0


def world_size() -> int:
    """Returns the current world size (number of distributed processes)."""
    return dist.get_world_size() if dist.is_initialized() else 1


def gather(input: torch.Tensor) -> Tuple[torch.Tensor]:
    """Gathers this tensor from all processes. Supports backprop."""
    return GatherLayer.apply(input)


def eye_rank(n: int, device: Optional[torch.device] = None) -> torch.Tensor:
    rows = torch.arange(n, device=device, dtype=torch.long)
    cols = rows + rank() * n
    diag_mask = torch.zeros((n, n * world_size()), dtype=torch.bool)
    diag_mask[(rows, cols)] = True
    return diag_mask


def rank_zero_only(fn):
    def wrapped(*args, **kwargs):
        if rank() == 0:
            return fn(*args, **kwargs)

    return wrapped


@rank_zero_only
def print_rank_zero(*args, **kwargs) -> None:
    """Equivalent to print, but only runs on the process with rank 0."""
    print(*args, **kwargs)


class NTXentLoss(MemoryBankModule):
    def __init__(
        self,
        temperature: float = 0.5,
        memory_bank_size: int = 0,
        gather_distributed: bool = False,
    ):
        super(NTXentLoss, self).__init__(size=memory_bank_size)
        self.temperature = temperature
        self.gather_distributed = gather_distributed
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.eps = 1e-8

        if abs(self.temperature) < self.eps:
            raise ValueError(
                "Illegal temperature: abs({}) < 1e-8".format(self.temperature)
            )
        if gather_distributed and not torch_dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

    def forward(self, out0: torch.Tensor, out1: torch.Tensor):
        device = out0.device
        batch_size, _ = out0.shape

        # normalize the output to length 1
        out0 = nn.functional.normalize(out0, dim=1)
        out1 = nn.functional.normalize(out1, dim=1)

        # ask memory bank for negative samples and extend it with out1 if
        # out1 requires a gradient, otherwise keep the same vectors in the
        # memory bank (this allows for keeping the memory bank constant e.g.
        # for evaluating the loss on the test set)
        # out1: shape: (batch_size, embedding_size)
        # negatives: shape: (embedding_size, memory_bank_size)
        out1, negatives = super(NTXentLoss, self).forward(
            out1, update=out0.requires_grad
        )

        # We use the cosine similarity, which is a dot product (einsum) here,
        # as all vectors are already normalized to unit length.
        # Notation in einsum: n = batch_size, c = embedding_size and k = memory_bank_size.

        if negatives is not None:
            # use negatives from memory bank
            negatives = negatives.to(device)

            # sim_pos is of shape (batch_size, 1) and sim_pos[i] denotes the similarity
            # of the i-th sample in the batch to its positive pair
            sim_pos = torch.einsum("nc,nc->n", out0, out1).unsqueeze(-1)

            # sim_neg is of shape (batch_size, memory_bank_size) and sim_neg[i,j] denotes the similarity
            # of the i-th sample to the j-th negative sample
            sim_neg = torch.einsum("nc,ck->nk", out0, negatives)

            # set the labels to the first "class", i.e. sim_pos,
            # so that it is maximized in relation to sim_neg
            logits = torch.cat([sim_pos, sim_neg], dim=1) / self.temperature
            labels = torch.zeros(logits.shape[0], device=device, dtype=torch.long)

        else:
            # user other samples from batch as negatives
            # and create diagonal mask that only selects similarities between
            # views of the same image
            if self.gather_distributed and dist.world_size() > 1:
                # gather hidden representations from other processes
                out0_large = torch.cat(dist.gather(out0), 0)
                out1_large = torch.cat(dist.gather(out1), 0)
                diag_mask = dist.eye_rank(batch_size, device=out0.device)
            else:
                # single process
                out0_large = out0
                out1_large = out1
                diag_mask = torch.eye(batch_size, device=out0.device, dtype=torch.bool)

            # calculate similiarities
            # here n = batch_size and m = batch_size * world_size
            # the resulting vectors have shape (n, m)
            logits_00 = torch.einsum("nc,mc->nm", out0, out0_large) / self.temperature
            logits_01 = torch.einsum("nc,mc->nm", out0, out1_large) / self.temperature
            logits_10 = torch.einsum("nc,mc->nm", out1, out0_large) / self.temperature
            logits_11 = torch.einsum("nc,mc->nm", out1, out1_large) / self.temperature

            # remove simliarities between same views of the same image
            logits_00 = logits_00[~diag_mask].view(batch_size, -1)
            logits_11 = logits_11[~diag_mask].view(batch_size, -1)

            # concatenate logits
            # the logits tensor in the end has shape (2*n, 2*m-1)
            logits_0100 = torch.cat([logits_01, logits_00], dim=1)
            logits_1011 = torch.cat([logits_10, logits_11], dim=1)
            logits = torch.cat([logits_0100, logits_1011], dim=0)

            # create labels
            labels = torch.arange(batch_size, device=device, dtype=torch.long)
            if self.gather_distributed:
                labels = labels + dist.rank() * batch_size
            labels = labels.repeat(2)

        loss = self.cross_entropy(logits, labels)

        return loss