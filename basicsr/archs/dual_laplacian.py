import torch
import torch.nn as nn
import torch.nn.functional as F


class DualGraphLaplacian(nn.Module):

    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha


    def forward(self, x, H=None, W=None):
        print("🔥 Dual Laplacian forward activated")
        print("feature:", x.shape)
        B, N, C = x.shape

        # recover spatial size
        if H is None or W is None:
            H = int(N ** 0.5)
            W = H

        if H * W != N:
            return x

        # [B,H,W,C]
        feat = x.reshape(B,H,W,C)

        # [B,N,C]
        feat_flat = feat.reshape(B,N,C)

        # Feature affinity
        feat_norm = F.normalize(
            feat_flat,
            dim=-1
        )

        # cosine similarity
        S = torch.matmul(
            feat_norm,
            feat_norm.transpose(1,2)
        )


        # Row graph

        row_ids = torch.arange(
            H,
            device=x.device
        ).repeat_interleave(W)


        M_row = (
            row_ids[:,None] ==
            row_ids[None,:]
        ).float()


        M_row = M_row.unsqueeze(0)


        # Column graph
        col_ids = torch.arange(
            W,
            device=x.device
        ).repeat(H)


        M_col = (
            col_ids[:,None] ==
            col_ids[None,:]
        ).float()


        M_col = M_col.unsqueeze(0)

        # adjacency matrices
        A_row = S * M_row
        A_col = S * M_col

        # remove self connection
        eye = torch.eye(
            N,
            device=x.device
        ).unsqueeze(0)


        A_row = A_row * (1-eye)
        A_col = A_col * (1-eye)


        # Normalizing connecitons
        A_row = F.relu(A_row)
        A_col = F.relu(A_col)


        # Normalized Laplacian
        eps = 1e-6
        D_row = A_row.sum(dim=-1) + eps
        D_col = A_col.sum(dim=-1) + eps

        D_row_inv = torch.pow(
            D_row,
            -0.5
        )

        D_col_inv = torch.pow(
            D_col,
            -0.5
        )

        A_row_norm = (
            D_row_inv.unsqueeze(-1)
            *
            A_row
            *
            D_row_inv.unsqueeze(-2)
        )


        A_col_norm = (
            D_col_inv.unsqueeze(-1)
            *
            A_col
            *
            D_col_inv.unsqueeze(-2)
        )


        I = torch.eye(
            N,
            device=x.device
        ).unsqueeze(0)



        L_row = I - A_row_norm

        L_col = I - A_col_norm



        # Laplacian filtering

        row_out = torch.bmm(
            L_row,
            feat_flat
        )


        col_out = torch.bmm(
            L_col,
            feat_flat
        )



        # dual graph response
        dual_response = row_out + col_out
        out = feat_flat + self.alpha * dual_response
        return out
        
