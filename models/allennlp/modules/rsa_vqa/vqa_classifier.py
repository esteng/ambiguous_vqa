import torch

class VQAClassifier(torch.nn.Module):
    def __init__(self, hs, vs):
        super(VQAClassifier, self).__init__()
        # from: https://github.com/dandelin/ViLT
        self.vqa_classifier = torch.nn.Sequential(
                torch.nn.Linear(hs, hs * 2),
                torch.nn.LayerNorm(hs * 2),
                torch.nn.GELU(),
                torch.nn.Linear(hs * 2, vs),
            )

    def forward(self, x):
        return self.vqa_classifier(x)
