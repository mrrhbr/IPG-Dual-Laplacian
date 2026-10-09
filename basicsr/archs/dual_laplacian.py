import torch
import torch.nn as nn
import torch.nn.functional as F


class DualGraphLaplacian(nn.Module):

    def __init__(self, dim, alpha=0.05, debug=False):
        super().__init__()

        self.alpha = alpha
        self.debug = debug
        self._delta_printed = False


        # Attention pooling for rows
        self.row_attention = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1)
        )


        # Attention pooling for columns
        self.col_attention = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1)
        )


    def build_laplacian(self, A):

        eps = 1e-6

        N = A.size(-1)

        eye = torch.eye(
            N,
            device=A.device,
            dtype=A.dtype
        ).unsqueeze(0)


        # keep same as Exp3
        A = A * (1-eye)

        A = (A + 1) / 2


        D = A.sum(-1) + eps

        D_inv = torch.rsqrt(D)


        A_norm = (
            D_inv.unsqueeze(-1)
            *
            A
            *
            D_inv.unsqueeze(-2)
        )


        I = torch.eye(
            N,
            device=A.device,
            dtype=A.dtype
        ).unsqueeze(0)


        L = I - A_norm


        return L



    def forward(self,x,H=None,W=None):

        B,N,C = x.shape


        if H is None or W is None:

            size=int(N**0.5)

            if size*size != N:
                return x

            H=size
            W=size


        if H*W != N:
            return x



        feat = x.reshape(
            B,H,W,C
        )


        # =========================
        # Attention Row Token
        # =========================

        row_score = self.row_attention(feat)


        row_weight = F.softmax(
            row_score,
            dim=2
        )


        row_feat = torch.sum(
            row_weight * feat,
            dim=2
        )


        # B,H,C



        # =========================
        # Attention Column Token
        # =========================

        col_score = self.col_attention(feat)


        col_weight = F.softmax(
            col_score,
            dim=1
        )


        col_feat = torch.sum(
            col_weight * feat,
            dim=1
        )


        # B,W,C



        # normalize

        row_feat = F.normalize(
            row_feat,
            dim=-1
        )


        col_feat = F.normalize(
            col_feat,
            dim=-1
        )



        # =========================
        # Affinity
        # =========================

        A_row = torch.bmm(
            row_feat,
            row_feat.transpose(1,2)
        )


        A_col = torch.bmm(
            col_feat,
            col_feat.transpose(1,2)
        )



        # =========================
        # Laplacian
        # =========================

        L_row = self.build_laplacian(
            A_row
        )

        L_col = self.build_laplacian(
            A_col
        )



        # =========================
        # Propagation
        # =========================

        row_response = torch.bmm(
            L_row,
            row_feat
        )


        col_response = torch.bmm(
            L_col,
            col_feat
        )



        # back to image grid

        row_response = row_response.unsqueeze(2)


        row_response = row_response.expand(
            B,H,W,C
        )


        col_response = col_response.unsqueeze(1)


        col_response = col_response.expand(
            B,H,W,C
        )



        dual_response = (
            row_response +
            col_response
        )


        dual_response = dual_response.reshape(
            B,N,C
        )



        dual_response = (
            dual_response /
            (
                dual_response.norm(
                    dim=-1,
                    keepdim=True
                )
                +1e-6
            )
        )


        out = (
            x +
            self.alpha *
            dual_response
        )



        if self.debug and not self._delta_printed:

            diff = out-x

            print("===== Attention Dual Laplacian =====")
            print("input:",x.shape)
            print("row graph:",A_row.shape)
            print("col graph:",A_col.shape)
            print("delta:",diff.abs().mean().item())
            print("alpha:",self.alpha)

            self._delta_printed=True



        return out
