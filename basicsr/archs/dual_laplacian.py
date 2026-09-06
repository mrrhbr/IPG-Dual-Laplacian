import torch
import torch.nn as nn
import torch.nn.functional as F


class DualGraphLaplacian(nn.Module):

    def __init__(self, alpha=0.05, debug=False):
        super().__init__()

        self.alpha = alpha
        self.debug = debug
        self._printed = False
        self._delta_printed = False


    def forward(self, x, H=None, W=None):

        B, N, C = x.shape


        # infer spatial size
        if H is None or W is None:

            size = int(N ** 0.5)

            if size * size != N:
                return x

            H = size
            W = size


        if H * W != N:
            return x



        feat = x.reshape(
            B,
            H,
            W,
            C
        )


        feat_flat = feat.reshape(
            B,
            N,
            C
        )


        # =========================
        # Feature affinity
        # =========================

        feat_norm = F.normalize(
            feat_flat,
            dim=-1
        )


        S = torch.bmm(
            feat_norm,
            feat_norm.transpose(1,2)
        )


        # =========================
        # Row graph
        # =========================

        row_ids = torch.arange(
            H,
            device=x.device
        ).repeat_interleave(W)


        M_row = (
            row_ids[:,None]
            ==
            row_ids[None,:]
        ).float()



        # =========================
        # Column graph
        # =========================

        col_ids = torch.arange(
            W,
            device=x.device
        ).repeat(H)


        M_col = (
            col_ids[:,None]
            ==
            col_ids[None,:]
        ).float()



        M_row = M_row.unsqueeze(0)
        M_col = M_col.unsqueeze(0)



        # =========================
        # adjacency
        # =========================

        A_row = S * M_row
        A_col = S * M_col


        eye = torch.eye(
            N,
            device=x.device
        ).unsqueeze(0)


        A_row = A_row * (1-eye)
        A_col = A_col * (1-eye)



        A_row = F.relu(A_row)
        A_col = F.relu(A_col)



        # =========================
        # normalized Laplacian
        # =========================

        eps = 1e-6


        D_row = A_row.sum(-1) + eps
        D_col = A_col.sum(-1) + eps


        D_row_inv = torch.rsqrt(D_row)
        D_col_inv = torch.rsqrt(D_col)



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



        # =========================
        # Laplacian response
        # =========================

        row_response = torch.bmm(
            L_row,
            feat_flat
        )


        col_response = torch.bmm(
            L_col,
            feat_flat
        )



        dual_response = (
            row_response
            +
            col_response
        )


        # normalize response
        dual_response = (
            dual_response /
            (
                dual_response.norm(
                    dim=-1,
                    keepdim=True
                )
                + eps
            )
        )



        out = (
            feat_flat
            +
            self.alpha *
            dual_response
        )


        # =========================
        # Debug
        # =========================

        if self.debug and self.training:

            diff = out - feat_flat

            delta_abs = diff.abs().mean().item()

            delta_norm = diff.norm().item()

            feature_norm = feat_flat.norm().item()

            ratio = delta_norm / (feature_norm + eps)


            if not self._delta_printed:

                print("====== Dual Laplacian Debug ======")
                print("feature norm :", feature_norm)
                print("delta abs    :", delta_abs)
                print("delta norm   :", delta_norm)
                print("ratio        :", ratio)
                print("alpha        :", self.alpha)

                self._delta_printed = True



        return out
