import pdb 
from overrides import overrides
import torch 
import torch.nn.functional as F
from allennlp.common.registrable import Registrable

class Loss(torch.nn.Module, Registrable):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, debug_answer, label_weights):
        raise NotImplementedError

@Loss.register("ce")
class CELoss(Loss):
    def __init__(self):
        super().__init__()
        self.loss_fxn = F.cross_entropy

    @overrides 
    def forward(self, logits, labels, debug_answer, label_weights=None):
        return self.loss_fxn(logits, labels), label_weights


@Loss.register("bce")
class BCELoss(Loss):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab 
        self.loss_fxn = F.binary_cross_entropy_with_logits

    def get_label_weights(self, labels, debug_answers):
        """
        get label weights for BCE loss from per-answer counts 
        """
        weights = torch.zeros_like(labels)
        for row in range(weights.shape[0]):
            for answer, count in debug_answers[row].items():
                try:
                    ans_idx = self.vocab._token_to_index['answers'][answer]
                    weights[row, ans_idx] = count
                except KeyError:
                    # for now, just make weights all 1s 
                    print(f"cannot find answer in vocab: {answer}")
                    weights[row] = 1.0

        # normalize weights 
        weights = weights / weights.sum(dim=1, keepdim=True)
        # make zero examples also have a weight 
        # proportional to how many zeros there are
        num_zeros = weights == 0
        num_zeros = num_zeros.sum(dim=1, keepdim=True)
        weights_zero_mask = torch.ones_like(weights) / num_zeros
        weights[weights == 0] = weights_zero_mask[weights == 0]
        return weights

    @overrides 
    def forward(self, logits, labels, debug_answer, label_weights=None):
        labels = labels.squeeze(1).float()
        label_weights = self.get_label_weights(labels, debug_answer)
        return self.loss_fxn(logits, labels, weight=label_weights).mean(), label_weights


@Loss.register("wbce")
class WeightedBCELoss(Loss):
    def __init__(self): 
        super().__init__()
        self.eps = 1e-10
        self.loss_fxn = F.binary_cross_entropy_with_logits

    @overrides
    def forward(self, logits, labels, debug_answer, label_weights=None):
        labels = labels.squeeze(1)
        count_pos = torch.sum(labels)*1.0+self.eps
        count_neg = torch.sum(1.-labels)*1.0
        beta = count_neg/count_pos
        beta_back = count_pos / (count_pos + count_neg)
        bce_loss = self.loss_fxn(logits, torch.squeeze(labels, 1).float(), weight=beta).view(-1)
        vqa_loss = beta_back*bce_loss
        return vqa_loss, label_weights

@Loss.register("multilabel_ce")
class MultilabelCELoss(Loss):
    def __init__(self, vocab):
        """manually compute CE loss against multi-hot dist"""
        super().__init__() 
        self.vocab = vocab

    def get_soft_labels(self, labels, debug_answers): 
        weights = torch.zeros_like(labels)
        for row in range(weights.shape[0]):
            for answer, count in debug_answers[row].items():
                try:
                    ans_idx = self.vocab._token_to_index['answers'][answer]
                    weights[row, ans_idx] = count
                except KeyError:
                    # for now, just uniformly smooth this example 
                    print(f"cannot find answer in vocab: {answer}")
                    weights[row] = 1.0
        # normalize weights 
        weights = weights / weights.sum(dim=1, keepdim=True)
        return weights
    
    @overrides
    def forward(self, logits, labels, debug_answer, label_weights=None): 
        labels = labels.squeeze(1)
        labels = self.get_soft_labels(labels, debug_answer)
        logits = logits.log_softmax(dim=1)        
        loss = torch.mean(torch.sum(-labels * logits, dim=1))
        return loss, label_weights

@Loss.register("asym")
class AsymmetricLossMultiLabel(Loss):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super().__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y, label_weights=None):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum(), label_weights