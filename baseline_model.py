import torch
import torchvision.models as models

class Baseline(torch.nn.Module):
    def __init__(self, image_height, image_width):

        super(Baseline, self).__init__()
        self.attn = torch.nn.Linear(image_width, image_width, bias = False)
        self.final_layer = torch.nn.Linear(1000, 1)

        self.image_width = image_width
        self.image_height = image_height

        self.resnet18 = models.resnet18()
        self.resnet18.train()

    def forward(self, X):

        # Attention over frames
        attn = self.attn(torch.eye(self.image_width))
        score_part1 = torch.matmul(X, attn)
        score_mat = torch.matmul(score_part1, torch.transpose(X, 2, 3))
        # Frobenius norm of each matrix
        score = torch.norm(score_mat, dim=(2,3))
        SM = torch.nn.Softmax(dim = 1)
        alpha = SM(score)

        # Make alphas coefficients same dimensions as X
        # Repeat the coefficients into the new dimensions
        # Then perform element-wise mutliplication
        # Then perform linear sum across relevant dimension

        alpha_inc1 = torch.unsqueeze(alpha, dim = 2)
        alpha_inc1_repeated = alpha_inc1.expand(-1, -1, self.image_height)
        alpha_inc2 = torch.unsqueeze(alpha_inc1_repeated, dim = 3)
        alpha_inc2_repeated = alpha_inc2.expand(-1, -1, -1, self.image_width)
        mult_X_alpha = X * alpha_inc2_repeated
        X_attn = torch.sum(mult_X_alpha, dim = 1)

        # Ensure X has 3 channels - in the future make channels not repeated
        X_attn_3 = torch.unsqueeze(X_attn, dim=1)
        X_attn_3_repeated = X_attn_3.expand(-1, 3, -1, -1)

        # Pass through resnet-18
        y_1000 = self.resnet18(X_attn_3_repeated)

        # Get single value output
        y_raw = self.final_layer(y_1000)
        y_squeeze = torch.squeeze(y_raw)
        y = torch.sigmoid(y_squeeze)

        return y
