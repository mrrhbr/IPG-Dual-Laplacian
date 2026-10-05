import torch
import torch.nn as nn
import torch.nn.functional as F


class DualGraphLaplacian(nn.Module):

    def __init__(self, alpha=0.05, debug=False):
        super().__init__()

        self.alpha = alpha
        self.debug = debug
        self._delta_printed = False


    def build_laplacian(self, A):

        eps = 1e-6

        # remove self connection
        N = A.size(-1)

        eye = torch.eye(
            N,
            device=A.device
        ).unsqueeze(0)

        A = A * (1 - eye)


        # cosine -> positive affinity

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
            device=A.device
        ).unsqueeze(0)


        L = I - A_norm


        return L



    def forward(self, x, H=None, W=None):

        B,N,C = x.shape


        if H is None or W is None:

            size = int(N ** 0.5)

            if size*size != N:
                return x

            H=size
            W=size


        if H*W != N:
            return x



        # --------------------------------
        # feature map
        # --------------------------------

        feat = x.reshape(
            B,H,W,C
        )


        # =================================
        # Row node representation
        # =================================

        row_mean = feat.mean(
            dim=2
        )

        row_max = feat.max(
            dim=2
        )[0]


        row_feat = torch.cat(
            [
                row_mean,
                row_max
            ],
            dim=-1
        )


        # B,H,2C


        # =================================
        # Column node representation
        # =================================

        col_mean = feat.mean(
            dim=1
        )


        col_max = feat.max(
            dim=1
        )[0]


        col_feat = torch.cat(
            [
                col_mean,
                col_max
            ],
            dim=-1
        )


        # B,W,2C



        # =================================
        # Normalize node features
        # =================================

        row_feat = F.normalize(
            row_feat,
            dim=-1
        )


        col_feat = F.normalize(
            col_feat,
            dim=-1
        )



        # =================================
        # Row / Column affinity
        # =================================

        A_row = torch.bmm(
            row_feat,
            row_feat.transpose(1,2)
        )


        A_col = torch.bmm(
            col_feat,
            col_feat.transpose(1,2)
        )



        # =================================
        # Laplacian
        # =================================

        L_row = self.build_laplacian(
            A_row
        )


        L_col = self.build_laplacian(
            A_col
        )



        # =================================
        # Graph propagation
        # =================================


        # row graph filtering

        row_response = torch.bmm(
            L_row,
            row_mean
        )


        # B,H,C


        row_response = row_response.unsqueeze(2)


        row_response = row_response.expand(
            B,
            H,
            W,
            C
        )



        # column graph filtering


        col_response = torch.bmm(
            L_col,
            col_mean
        )


        # B,W,C


        col_response = col_response.unsqueeze(1)


        col_response = col_response.expand(
            B,
            H,
            W,
            C
        )



        # =================================
        # Dual response
        # =================================

        dual_response = (
            row_response +
            col_response
        )


        dual_response = dual_response.reshape(
            B,
            N,
            C
        )



        # normalize update

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



        # =================================
        # Debug
        # =================================

        if self.debug and not self._delta_printed:

            diff = out-x

            print("====== Dual Row Column Laplacian ======")
            print("input:",x.shape)
            print("row graph:",A_row.shape)
            print("col graph:",A_col.shape)
            print("delta:",diff.abs().mean().item())
            print("alpha:",self.alpha)

            self._delta_printed=True



        return out
