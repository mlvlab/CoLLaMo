
import math

import torch
import torch.nn as nn
from scipy.sparse.csgraph import shortest_path

# from timm.layers import LayerNorm, LayerNorm2d
from torch_geometric.utils import to_dense_adj

from chem import mol_to_graphs

from .configuration_llamo import GraphProjectorConfig
from .graph_encoders import LayerNorm


def build_pos_embeds(
    config: GraphProjectorConfig, num_input_tokens: int, vision_hidden_size: int
):
    # pos emb
    if config.pos_emb:
        pos_emb = torch.nn.Parameter(
            torch.zeros(1, num_input_tokens, vision_hidden_size)
        )
        nn.init.trunc_normal_(pos_emb, mean=0.0, std=0.02)
    else:
        pos_emb = None

    return pos_emb


def build_eos_tokens(config: GraphProjectorConfig, output_hidden_size: int):
    # think tokens
    num_eos_tokens = config.num_eos_tokens
    if num_eos_tokens:
        eos_tokens = torch.nn.Parameter(
            torch.randn(1, num_eos_tokens, output_hidden_size)
        )
        nn.init.trunc_normal_(eos_tokens, mean=0.0, std=config.initializer_range)
    else:
        eos_tokens = None

    return eos_tokens


def build_prenorm(config: GraphProjectorConfig):
    if config.prenorm:
        prenorm = LayerNorm(config.encoder_hidden_size)
    else:
        prenorm = None
    return prenorm


def build_mlp(depth, hidden_size, output_hidden_size):
    layers = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        layers.append(nn.SiLU())
        layers.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*layers)


class Pooling(nn.Module):
    def __init__(self, hidden_size: int, num_queries: int = 32, mode: str = "attn"):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_queries = num_queries
        self.mode = mode

        self.linear = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x, mask=None):
        y = self.linear(x) / (self.linear.weight.norm() + 1e-6)
        if mask is not None:
            added_value = (~mask) * (-1000000)
            y += added_value.unsqueeze(-1)
        topk_values, topk_indices = y.topk(self.num_queries, dim=1)
        x = (
            torch.gather(x, 1, topk_indices.repeat(1, 1, x.size(-1)))
            * topk_values.tanh()
        )
        return x


class Resampler(nn.Module):
    """Base projector class"""

    def __init__(
        self,
        config: GraphProjectorConfig,
        num_query_tokens: int,
        output_hidden_size: int,
    ):
        super().__init__()
        self.config = config
        self.output_hidden_size = output_hidden_size
        self.num_query_tokens = num_query_tokens

        # think tokens
        self.eos_tokens = None
        # self.eos_tokens = build_eos_tokens(config, output_hidden_size)

        # pos emb
        # self.pos_emb = build_pos_embeds(config, num_input_tokens, config.encoder_hidden_size)
        self.pos_emb = None
        self.query_tokens = nn.Parameter(
            torch.zeros((1, num_query_tokens, self.config.encoder_hidden_size))
        )
        nn.init.trunc_normal_(self.query_tokens, mean=0.0, std=0.02)
        self.cross_attention = nn.MultiheadAttention(
            self.config.encoder_hidden_size, 4, batch_first=True
        )
        self.prenorm = build_prenorm(config)
        self.layer_norm = LayerNorm(self.config.encoder_hidden_size)

        self.build_net(self.config.encoder_hidden_size)

    def build_net(self):
        raise NotImplementedError()

    def _forward(self, x):
        raise NotImplementedError()

    def forward(
        self, x: torch.Tensor, smiles=None, h_list=None, h_pool_list=None, g_rep=None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, encoder_hidden_size) tensor from the visual backbone (CLIP visual encoder), including cls token.
        """
        if self.prenorm is not None:
            x = self.prenorm(x)

        if self.pos_emb is not None:
            x += self.pos_emb

        query_tokens = self.query_tokens.expand(x.size(0), -1, -1)
        x = self.cross_attention(query_tokens, x, x)[0]
        x = self.layer_norm(x)
        x = self._forward(x)  # (B, L, output_hidden_size)

        B = x.size(0)
        if self.eos_tokens is not None:
            x = torch.cat([query_tokens, self.eos_tokens.expand(B, -1, -1)], dim=1)
        else:
            x = query_tokens
        return x


class NodeProjector(nn.Module):
    """Base projector class"""

    def __init__(
        self,
        config: GraphProjectorConfig,
        num_query_tokens: int,
        output_hidden_size: int,
    ):
        super().__init__()
        self.config = config
        self.output_hidden_size = output_hidden_size
        self.query_type = None

        if "use_motif" in self.config.to_dict().keys() and self.config.use_motif:
            self.motif_weight = nn.Linear(
                config.num_motifs, output_hidden_size, bias=False
            )
            self.use_motif = True
            self.num_query_tokens = num_query_tokens - 8

        else:
            self.use_motif = False
            self.num_query_tokens = num_query_tokens

        if "query_type" in self.config.to_dict().keys():
            self.query_type = self.config.query_type
            if self.config.query_type == "sagpool":
                self.pool = Pooling(
                    output_hidden_size, num_queries=self.num_query_tokens
                )
            elif self.config.query_type == "lowlevel":
                pass
            else:
                raise NotImplementedError

        # think tokens
        self.eos_tokens = None
        # self.eos_tokens = build_eos_tokens(config, output_hidden_size)
        # pos emb
        # self.pos_emb = build_pos_embeds(config, num_input_tokens, config.encoder_hidden_size)
        self.pos_emb = None
        # self.emptynode_tokens = nn.Parameter(torch.zeros((1, output_hidden_size)))
        # nn.init.trunc_normal_(self.emptynode_tokens, mean=0.0, std=0.02)
        self.prenorm = build_prenorm(config)

        self.build_net(self.config.encoder_hidden_size)

    def build_net(self):
        raise NotImplementedError()

    def _forward(self, x):
        raise NotImplementedError()

    def forward(
        self, x: torch.Tensor, smiles=None, h_list=None, h_pool_list=None, g_rep=None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, encoder_hidden_size) tensor from the visual backbone (CLIP visual encoder), including cls token.
        """
        if self.prenorm is not None:
            x = self.prenorm(x)

        if self.pos_emb is not None:
            x += self.pos_emb

        if self.query_type == "lowlevel":
            x = self._forward(h_list[1][0])  # (B, L, output_hidden_size)
        else:
            x = self._forward(x)  # (B, L, output_hidden_size)
        num_nodes = x.size(1)

        if num_nodes > self.num_query_tokens:
            if self.query_type == "sagpool":
                # check the data type of graph values
                query_tokens = self.pool(x)

            else:
                query_tokens = x[:, : self.num_query_tokens, :]
        else:
            query_tokens = torch.cat(
                (
                    x,
                    torch.zeros(
                        (x.size(0), self.num_query_tokens - num_nodes, x.size(2))
                    ).to(x.device),
                ),
                dim=1,
            )

        if self.use_motif:
            motifs = []
            for s in smiles:
                try:
                    motif = mol_to_graphs(s).to(x.device).type(x.dtype)
                except Exception:
                    motif = None
                motifs.append(motif)

            for i, motif in enumerate(motifs):
                if motif is None:
                    temp_query_token = torch.zeros(8, self.output_hidden_size).to(
                        x.device
                    )
                elif motif.size(0) >= 8:
                    temp_query_token = self.motif_weight(motif[:8, :])
                else:
                    motif_token = self.motif_weight(motif)
                    temp_query_token = torch.cat(
                        (
                            motif_token,
                            torch.zeros(
                                8 - motif_token.size(0), self.output_hidden_size
                            ).to(x.device),
                        )
                    )

                if i == 0:
                    add_query_tokens = temp_query_token.unsqueeze(0)
                else:
                    add_query_tokens = torch.cat(
                        (add_query_tokens, temp_query_token.unsqueeze(0))
                    )

            query_tokens = torch.cat((query_tokens, add_query_tokens), dim=1)

        B = query_tokens.size(0)
        if self.eos_tokens is not None:
            x = torch.cat([query_tokens, self.eos_tokens.expand(B, -1, -1)], dim=1)
        else:
            x = query_tokens
        return x


class MultilevelProjector(nn.Module):
    """Base projector class"""

    def __init__(
        self,
        config: GraphProjectorConfig,
        num_query_tokens: int,
        output_hidden_size: int,
    ):
        super().__init__()
        self.config = config
        self.output_hidden_size = output_hidden_size
        self.mtl_type = None
        self.num_query_tokens = num_query_tokens

        if "no_cat" in self.config.to_dict().keys() and self.config.no_cat:
            self.no_cat = True
            self.cat_num = 0
        else:
            self.no_cat = False
            self.cat_num = 1

        if "use_motif" in self.config.to_dict().keys() and self.config.use_motif:
            self.motif_weight = nn.Linear(
                config.num_motifs, self.config.encoder_hidden_size, bias=False
            )
            self.use_motif = True
            self.num_query_tokens = num_query_tokens - 4

        else:
            self.use_motif = False
            self.num_query_tokens = num_query_tokens

        if (
            "motif_type" in self.config.to_dict().keys()
            and self.config.motif_type == "reverse"
        ):
            self.motif_reverse = True
        else:
            self.motif_reverse = False

        if (
            "motif_type" in self.config.to_dict().keys()
            and self.config.motif_type == "crossattn"
        ):
            self.motif_cross_attention = nn.MultiheadAttention(
                self.config.encoder_hidden_size, 4, batch_first=True
            )
            self.motif_query_tokens = nn.Parameter(
                torch.randn((4, self.config.encoder_hidden_size))
            )
            self.motif_type = "crossattn"
        else:
            self.motif_type = None

        if (
            "mtl_type" in self.config.to_dict().keys()
            and self.config.mtl_type == "crossattn"
        ):
            self.cross_attention = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        self.config.encoder_hidden_size, 4, batch_first=True
                    )
                ]
                * 6
            )
            self.query_tokens = nn.Parameter(
                torch.randn(
                    (
                        6,
                        (self.num_query_tokens - self.cat_num) // 6,
                        self.config.encoder_hidden_size,
                    )
                )
            )
            self.mtl_type = "crossattn"

        elif (
            "mtl_type" in self.config.to_dict().keys()
            and self.config.mtl_type == "sagpool"
        ):
            self.pool = nn.ModuleList(
                [
                    Pooling(
                        self.config.encoder_hidden_size,
                        num_queries=(self.num_query_tokens - self.cat_num) // 6,
                    )
                ]
                * 6
            )
            self.mtl_type = "sagpool"

        # think tokens
        # self.eos_tokens = build_eos_tokens(config, output_hidden_size)
        self.eos_tokens = None
        self.prenorm = build_prenorm(config)
        # self.layer_norm = nn.ModuleList(LayerNorm(output_hidden_size) for _ in range(6))
        self.layer_norm = LayerNorm(self.config.encoder_hidden_size)

        self.build_net(self.config.encoder_hidden_size)

    def build_net(self):
        raise NotImplementedError()

    def _forward(self, x):
        raise NotImplementedError()

    def forward(
        self, x: torch.Tensor, smiles=None, h_list=None, h_pool_list=None, g_rep=None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, encoder_hidden_size) tensor from the visual backbone (CLIP visual encoder), including cls token.
        """
        if self.prenorm is not None:
            x = self.prenorm(x)

        h_list = h_list[-6:]
        # h_pool_tensor = torch.stack(h_pool_list, dim=1)
        if self.mtl_type == "crossattn":
            new_query_tokens = []
            for i in range(6):
                if isinstance(h_list[i], tuple):
                    new_query_tokens.append(
                        self.cross_attention[i](
                            self.query_tokens[i].expand(x.size(0), -1, -1).to(x.device),
                            h_list[i][0],
                            h_list[i][0],
                        )[0]
                    )
                else:
                    new_query_tokens.append(
                        self.cross_attention[i](
                            self.query_tokens[i].expand(x.size(0), -1, -1).to(x.device),
                            h_list[i],
                            h_list[i],
                        )[0]
                    )
            x = torch.cat(new_query_tokens, 1)
            if not self.no_cat:
                x = torch.cat((x, h_pool_list[-1].unsqueeze(1)), 1)

        elif self.mtl_type == "sagpool":
            new_query_tokens = []
            for i in range(6):
                new_query_tokens.append(self.pool[i](h_list[i][0], mask=h_list[i][1]))
            # TODO: check the size of the query tokens
            x = torch.cat(new_query_tokens, 1)
            if not self.no_cat:
                x = torch.cat((x, h_pool_list[-1].unsqueeze(1)), 1)

        else:
            x = torch.cat([h.unsqueeze(1) for h in h_pool_list], 1)

        if self.use_motif:
            motifs = []
            for s in smiles:
                try:
                    motif = mol_to_graphs(s).to(x.device).type(x.dtype)
                except Exception:
                    motif = None
                motifs.append(motif)

            for i, motif in enumerate(motifs):
                if self.motif_type == "crossattn":
                    if motif is None:
                        temp_x = self.motif_query_tokens
                    else:
                        motif_token = self.motif_weight(motif)
                        temp_x = self.motif_cross_attention(
                            self.motif_query_tokens, motif_token, motif_token
                        )[0]

                else:
                    if motif is None:
                        temp_x = torch.zeros(6, self.config.encoder_hidden_size).to(
                            x.device
                        )
                    elif motif.size(0) >= 6:
                        temp_x = self.motif_weight(motif[:6, :])
                    else:
                        motif_token = self.motif_weight(motif)
                        temp_x = torch.cat(
                            (
                                motif_token,
                                torch.zeros(
                                    6 - motif_token.size(0),
                                    self.config.encoder_hidden_size,
                                ).to(x.device),
                            )
                        )

                if i == 0:
                    add_x = temp_x.unsqueeze(0)
                else:
                    add_x = torch.cat((add_x, temp_x.unsqueeze(0)))

            if self.motif_reverse:
                x = torch.cat((add_x, x), dim=1)
            else:
                x = torch.cat((x, add_x), dim=1)

        x = self.layer_norm(x)
        query_tokens = self._forward(x)  # (B, L, output_hidden_size)

        B = query_tokens.size(0)
        if self.eos_tokens is not None:
            x = torch.cat([query_tokens, self.eos_tokens.expand(B, -1, -1)], dim=1)
        else:
            x = query_tokens
        return x


class MotifResampler(nn.Module):
    """Base projector class"""

    def __init__(
        self,
        config: GraphProjectorConfig,
        num_query_tokens: int,
        output_hidden_size: int,
    ):
        super().__init__()
        self.config = config
        self.output_hidden_size = output_hidden_size
        self.num_query_tokens = num_query_tokens

        # think tokens
        self.eos_tokens = None
        # self.eos_tokens = build_eos_tokens(config, output_hidden_size)
        # pos emb
        # self.pos_emb = build_pos_embeds(config, num_input_tokens, config.encoder_hidden_size)
        self.pos_emb = None
        self.nomotif_query_tokens = nn.Parameter(torch.zeros((1, output_hidden_size)))
        self.motif_weight = nn.Linear(config.num_motifs, output_hidden_size, bias=False)
        nn.init.trunc_normal_(self.nomotif_query_tokens, mean=0.0, std=0.02)

        self.cross_attention = nn.MultiheadAttention(
            output_hidden_size, 8, batch_first=True
        )
        self.prenorm = build_prenorm(config)

        self.build_net(self.config.encoder_hidden_size)

    def build_net(self):
        raise NotImplementedError()

    def _forward(self, x):
        raise NotImplementedError()

    def forward(
        self, x: torch.Tensor, smiles=None, h_list=None, h_pool_list=None, g_rep=None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, encoder_hidden_size) tensor from the visual backbone (CLIP visual encoder), including cls token.
        """
        if self.prenorm is not None:
            x = self.prenorm(x)

        if self.pos_emb is not None:
            x += self.pos_emb

        x = self._forward(x)  # (B, L, output_hidden_size)
        motifs = []
        for s in smiles:
            try:
                motif = mol_to_graphs(s).to(x.device).type(x.dtype)
            except Exception:
                motif = None
            motifs.append(motif)

        for i, motif in enumerate(motifs):
            if motif is None:
                temp_query_token = self.nomotif_query_tokens.repeat(
                    (self.num_query_tokens), 1
                ).contiguous()
            elif motif.size(0) >= self.num_query_tokens:
                temp_query_token = self.motif_weight(
                    motif[: self.num_query_tokens, :]
                    / motif[: self.num_query_tokens, :].sum(-1, keepdim=True)
                ).contiguous()
            else:
                motif_token = self.motif_weight(motif / motif.sum(-1, keepdim=True))
                temp_query_token = torch.cat(
                    (
                        motif_token,
                        self.nomotif_query_tokens.repeat(
                            (self.num_query_tokens - motif_token.size(0)), 1
                        ),
                    )
                ).contiguous()
            if i == 0:
                query_tokens = temp_query_token.unsqueeze(0)
            else:
                query_tokens = torch.cat((query_tokens, temp_query_token.unsqueeze(0)))

        x = self.cross_attention(query_tokens, x, x)[0]

        B = x.size(0)
        if self.eos_tokens is not None:
            x = torch.cat([query_tokens, self.eos_tokens.expand(B, -1, -1)], dim=1)
        else:
            x = query_tokens
        return x


# class Resampler(Projector):

# class Qformer(Projector):


class MLPResampler(Resampler):
    def build_net(self, encoder_hidden_size):
        # hidden_size = self.config.hidden_size
        output_hidden_size = self.output_hidden_size
        depth = self.config.depth

        self.net = build_mlp(depth, encoder_hidden_size, output_hidden_size)

    def _forward(self, x):
        x = self.net(x)
        return x


class MLPMotifResampler(MotifResampler):
    def build_net(self, encoder_hidden_size):
        # hidden_size = self.config.hidden_size
        output_hidden_size = self.output_hidden_size
        depth = self.config.depth

        self.net = build_mlp(depth, encoder_hidden_size, output_hidden_size)

    def _forward(self, x):
        x = self.net(x)
        return x


class MLPNodeProjector(NodeProjector):
    def build_net(self, encoder_hidden_size):
        # hidden_size = self.config.hidden_size
        output_hidden_size = self.output_hidden_size
        depth = self.config.depth

        self.net = build_mlp(depth, encoder_hidden_size, output_hidden_size)

    def _forward(self, x):
        x = self.net(x)
        return x


class MLPMultilevelProjector(MultilevelProjector):
    def build_net(self, encoder_hidden_size):
        # hidden_size = self.config.hidden_size
        output_hidden_size = self.output_hidden_size
        depth = self.config.depth

        self.net = build_mlp(depth, encoder_hidden_size, output_hidden_size)

    def _forward(self, x):
        x = self.net(x)
        return x


############################################################################################################
class Resampler3d(nn.Module):
    """Base projector class"""

    def __init__(
        self,
        config: GraphProjectorConfig,
        num_query_tokens: int,
        output_hidden_size: int,
    ):
        super().__init__()
        self.config = config
        self.output_hidden_size = output_hidden_size
        self.num_query_tokens = num_query_tokens

        # think tokens
        self.eos_tokens = None
        # self.eos_tokens = build_eos_tokens(config, output_hidden_size)

        # pos emb
        # self.pos_emb = build_pos_embeds(config, num_input_tokens, config.encoder_hidden_size)
        self.pos_emb = None
        self.query_tokens = nn.Parameter(torch.zeros((1, num_query_tokens, 512)))
        nn.init.trunc_normal_(self.query_tokens, mean=0.0, std=0.02)
        self.cross_attention = nn.MultiheadAttention(512, 4, batch_first=True)
        self.prenorm = build_prenorm(config)
        self.layer_norm = LayerNorm(512)

        self.build_net(512)

    def build_net(self):
        raise NotImplementedError()

    def _forward(self, x):
        raise NotImplementedError()

    def forward(
        self, x: torch.Tensor, smiles=None, h_list=None, h_pool_list=None, g_rep=None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, encoder_hidden_size) tensor from the visual backbone (CLIP visual encoder), including cls token.
        """
        if self.prenorm is not None:
            x = self.prenorm(x)

        if self.pos_emb is not None:
            x += self.pos_emb

        query_tokens = self.query_tokens.expand(x.size(0), -1, -1)
        x = self.cross_attention(query_tokens, x, x)[0]
        x = self.layer_norm(x)
        x = self._forward(x)  # (B, L, output_hidden_size)

        B = x.size(0)
        if self.eos_tokens is not None:
            x = torch.cat([query_tokens, self.eos_tokens.expand(B, -1, -1)], dim=1)
        else:
            x = query_tokens
        return x


class NodeProjector3d(nn.Module):
    """Base projector class"""

    def __init__(
        self,
        config: GraphProjectorConfig,
        num_query_tokens: int,
        output_hidden_size: int,
    ):
        super().__init__()
        self.config = config
        self.output_hidden_size = output_hidden_size
        self.query_type = None

        if "use_motif" in self.config.to_dict().keys() and self.config.use_motif:
            self.motif_weight = nn.Linear(
                config.num_motifs, output_hidden_size, bias=False
            )
            self.use_motif = True
            self.num_query_tokens = num_query_tokens - 8

        else:
            self.use_motif = False
            self.num_query_tokens = num_query_tokens

        if "query_type" in self.config.to_dict().keys():
            self.query_type = self.config.query_type
            if self.config.query_type == "sagpool":
                self.pool = Pooling(
                    output_hidden_size, num_queries=self.num_query_tokens
                )
            elif self.config.query_type == "lowlevel":
                pass
            else:
                raise NotImplementedError

        # think tokens
        self.eos_tokens = None
        # self.eos_tokens = build_eos_tokens(config, output_hidden_size)
        # pos emb
        # self.pos_emb = build_pos_embeds(config, num_input_tokens, config.encoder_hidden_size)
        self.pos_emb = None
        # self.emptynode_tokens = nn.Parameter(torch.zeros((1, output_hidden_size)))
        # nn.init.trunc_normal_(self.emptynode_tokens, mean=0.0, std=0.02)
        self.prenorm = build_prenorm(config)

        self.build_net(512)

    def build_net(self):
        raise NotImplementedError()

    def _forward(self, x):
        raise NotImplementedError()

    def forward(
        self, x: torch.Tensor, smiles=None, h_list=None, h_pool_list=None, g_rep=None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, encoder_hidden_size) tensor from the visual backbone (CLIP visual encoder), including cls token.
        """
        if self.prenorm is not None:
            x = self.prenorm(x)

        if self.pos_emb is not None:
            x += self.pos_emb

        if self.query_type == "lowlevel":
            x = self._forward(h_list[1][0])  # (B, L, output_hidden_size)
        else:
            x = self._forward(x)  # (B, L, output_hidden_size)
        num_nodes = x.size(1)

        if num_nodes > self.num_query_tokens:
            if self.query_type == "sagpool":
                # check the data type of graph values
                query_tokens = self.pool(x)

            else:
                query_tokens = x[:, : self.num_query_tokens, :]
        else:
            query_tokens = torch.cat(
                (
                    x,
                    torch.zeros(
                        (x.size(0), self.num_query_tokens - num_nodes, x.size(2))
                    ).to(x.device),
                ),
                dim=1,
            )

        if self.use_motif:
            motifs = []
            for s in smiles:
                try:
                    motif = mol_to_graphs(s).to(x.device).type(x.dtype)
                except Exception:
                    motif = None
                motifs.append(motif)

            for i, motif in enumerate(motifs):
                if motif is None:
                    temp_query_token = torch.zeros(8, self.output_hidden_size).to(
                        x.device
                    )
                elif motif.size(0) >= 8:
                    temp_query_token = self.motif_weight(motif[:8, :])
                else:
                    motif_token = self.motif_weight(motif)
                    temp_query_token = torch.cat(
                        (
                            motif_token,
                            torch.zeros(
                                8 - motif_token.size(0), self.output_hidden_size
                            ).to(x.device),
                        )
                    )

                if i == 0:
                    add_query_tokens = temp_query_token.unsqueeze(0)
                else:
                    add_query_tokens = torch.cat(
                        (add_query_tokens, temp_query_token.unsqueeze(0))
                    )

            query_tokens = torch.cat((query_tokens, add_query_tokens), dim=1)

        B = query_tokens.size(0)
        if self.eos_tokens is not None:
            x = torch.cat([query_tokens, self.eos_tokens.expand(B, -1, -1)], dim=1)
        else:
            x = query_tokens
        return x


class MultilevelProjector3d(nn.Module):
    """Base projector class"""

    def __init__(
        self,
        config: GraphProjectorConfig,
        num_query_tokens: int,
        output_hidden_size: int,
    ):
        super().__init__()
        self.config = config
        self.output_hidden_size = output_hidden_size
        self.mtl_type = None
        self.num_query_tokens = num_query_tokens

        if "no_cat" in self.config.to_dict().keys() and self.config.no_cat:
            self.no_cat = True
            self.cat_num = 0
        else:
            self.no_cat = False
            self.cat_num = 1

        if "use_motif" in self.config.to_dict().keys() and self.config.use_motif:
            self.motif_weight = nn.Linear(config.num_motifs, 512, bias=False)
            self.use_motif = True
            self.num_query_tokens = num_query_tokens - 4

        else:
            self.use_motif = False
            self.num_query_tokens = num_query_tokens

        if (
            "motif_type" in self.config.to_dict().keys()
            and self.config.motif_type == "reverse"
        ):
            self.motif_reverse = True
        else:
            self.motif_reverse = False

        if (
            "motif_type" in self.config.to_dict().keys()
            and self.config.motif_type == "crossattn"
        ):
            self.motif_cross_attention = nn.MultiheadAttention(512, 4, batch_first=True)
            self.motif_query_tokens = nn.Parameter(torch.randn((4, 512)))
            self.motif_type = "crossattn"
        else:
            self.motif_type = None

        if (
            "mtl_type" in self.config.to_dict().keys()
            and self.config.mtl_type == "crossattn"
        ):
            self.cross_attention = nn.ModuleList(
                [nn.MultiheadAttention(512, 4, batch_first=True)] * 6
            )
            self.query_tokens = nn.Parameter(
                torch.randn((6, (self.num_query_tokens - self.cat_num) // 6, 512))
            )
            self.mtl_type = "crossattn"

        elif (
            "mtl_type" in self.config.to_dict().keys()
            and self.config.mtl_type == "sagpool"
        ):
            self.pool = nn.ModuleList(
                [Pooling(512, num_queries=(self.num_query_tokens - self.cat_num) // 6)]
                * 6
            )
            self.mtl_type = "sagpool"

        # think tokens
        # self.eos_tokens = build_eos_tokens(config, output_hidden_size)
        self.eos_tokens = None
        self.prenorm = build_prenorm(config)
        # self.layer_norm = nn.ModuleList(LayerNorm(output_hidden_size) for _ in range(6))
        self.layer_norm = LayerNorm(512)

        self.build_net(512)

    def build_net(self):
        raise NotImplementedError()

    def _forward(self, x):
        raise NotImplementedError()

    def forward(
        self, x: torch.Tensor, smiles=None, h_list=None, h_pool_list=None, g_rep=None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, encoder_hidden_size) tensor from the visual backbone (CLIP visual encoder), including cls token.
        """
        if self.prenorm is not None:
            x = self.prenorm(x)

        h_list = h_list[-6:]
        # h_pool_tensor = torch.stack(h_pool_list, dim=1)

        if self.mtl_type == "crossattn":
            new_query_tokens = []
            for i in range(6):
                if isinstance(h_list[i], tuple):
                    new_query_tokens.append(
                        self.cross_attention[i](
                            self.query_tokens[i].expand(x.size(0), -1, -1).to(x.device),
                            h_list[i][0],
                            h_list[i][0],
                        )[0]
                    )
                else:
                    new_query_tokens.append(
                        self.cross_attention[i](
                            self.query_tokens[i].expand(x.size(0), -1, -1).to(x.device),
                            h_list[i],
                            h_list[i],
                        )[0]
                    )
            x = torch.cat(new_query_tokens, 1)
            if not self.no_cat:
                x = torch.cat((x, h_pool_list[-1].unsqueeze(1)), 1)

        elif self.mtl_type == "sagpool":
            new_query_tokens = []
            for i in range(6):
                new_query_tokens.append(self.pool[i](h_list[i][0], mask=h_list[i][1]))
            # TODO: check the size of the query tokens
            x = torch.cat(new_query_tokens, 1)
            if not self.no_cat:
                x = torch.cat((x, h_pool_list[-1].unsqueeze(1)), 1)

        else:
            x = torch.cat([h.unsqueeze(1) for h in h_pool_list], 1)

        if self.use_motif:
            motifs = []
            for s in smiles:
                try:
                    motif = mol_to_graphs(s).to(x.device).type(x.dtype)
                except Exception:
                    motif = None
                motifs.append(motif)

            for i, motif in enumerate(motifs):
                if self.motif_type == "crossattn":
                    if motif is None:
                        temp_x = self.motif_query_tokens
                    else:
                        motif_token = self.motif_weight(motif)
                        temp_x = self.motif_cross_attention(
                            self.motif_query_tokens, motif_token, motif_token
                        )[0]

                else:
                    if motif is None:
                        temp_x = torch.zeros(6, self.config.encoder_hidden_size).to(
                            x.device
                        )
                    elif motif.size(0) >= 6:
                        temp_x = self.motif_weight(motif[:6, :])
                    else:
                        motif_token = self.motif_weight(motif)
                        temp_x = torch.cat(
                            (
                                motif_token,
                                torch.zeros(
                                    6 - motif_token.size(0),
                                    self.config.encoder_hidden_size,
                                ).to(x.device),
                            )
                        )

                if i == 0:
                    add_x = temp_x.unsqueeze(0)
                else:
                    add_x = torch.cat((add_x, temp_x.unsqueeze(0)))

            if self.motif_reverse:
                x = torch.cat((add_x, x), dim=1)
            else:
                x = torch.cat((x, add_x), dim=1)

        x = self.layer_norm(x)
        query_tokens = self._forward(x)  # (B, L, output_hidden_size)

        B = query_tokens.size(0)
        if self.eos_tokens is not None:
            x = torch.cat([query_tokens, self.eos_tokens.expand(B, -1, -1)], dim=1)
        else:
            x = query_tokens
        return x


class MLPResampler3d(Resampler3d):
    def build_net(self, encoder_hidden_size):
        # hidden_size = self.config.hidden_size
        output_hidden_size = self.output_hidden_size
        depth = self.config.depth

        self.net = build_mlp(depth, encoder_hidden_size, output_hidden_size)

    def _forward(self, x):
        x = self.net(x)
        return x


class MLPNodeProjector3d(NodeProjector3d):
    def build_net(self, encoder_hidden_size):
        # hidden_size = self.config.hidden_size
        output_hidden_size = self.output_hidden_size
        depth = self.config.depth

        self.net = build_mlp(depth, encoder_hidden_size, output_hidden_size)

    def _forward(self, x):
        x = self.net(x)
        return x


class MLPMultilevelProjector3d(MultilevelProjector3d):
    def build_net(self, encoder_hidden_size):
        # hidden_size = self.config.hidden_size
        output_hidden_size = self.output_hidden_size
        depth = self.config.depth

        self.net = build_mlp(depth, encoder_hidden_size, output_hidden_size)

    def _forward(self, x):
        x = self.net(x)
        return x


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        q_noise=0.0,
        qn_block_size=8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        raise NotImplementedError

    def reset_parameters(self):
        # if self.qkv_same_dim:
        # Empirically observed the convergence to be much better with
        # the scaled initialization
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        # else:
        # nn.init.xavier_uniform_(self.k_proj.weight)
        # nn.init.xavier_uniform_(self.v_proj.weight)
        # nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key=None,
        value=None,
        attn_bias=None,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        before_softmax=False,
        need_head_weights=False,
    ):
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        bsz, tgt_len, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [bsz, tgt_len, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert (src_len, bsz) == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q *= self.scaling

        q = q.contiguous().view(bsz * self.num_heads, tgt_len, self.head_dim)
        if k is not None:
            k = k.contiguous().view(bsz * self.num_heads, -1, self.head_dim)
        if v is not None:
            v = v.contiguous().view(bsz * self.num_heads, -1, self.head_dim)

        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        # attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_bias is not None:
            attn_weights += attn_bias.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(bsz, tgt_len, embed_dim)
        attn = self.out_proj(attn)

        attn_weights = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights


class BlendProjector(nn.Module):
    """Base projector class"""

    def __init__(
        self,
        config: GraphProjectorConfig,
        num_query_tokens: int,
        output_hidden_size: int,
    ):
        super().__init__()
        self.config = config
        self.output_hidden_size = output_hidden_size
        self.mtl_type = None
        self.num_query_tokens = num_query_tokens * 2

        if "sep" in self.config.to_dict().keys():
            self.sep = self.config.sep
        else:
            self.sep = False

        if "blend2" in self.config.to_dict().keys() and self.config.blend2:
            # self.d1_net = nn.ModuleList([build_mlp(2, output_hidden_size, 512)]*6)
            self.d1_net = nn.ModuleList([nn.Linear(output_hidden_size, 512)] * 6)

            self.pre_d1_layer_norm = nn.ModuleList([LayerNorm(output_hidden_size)] * 6)
            self.d1_layer_norm = nn.ModuleList([LayerNorm(512)] * 6)

            self.d1_attention = nn.ModuleList([MultiheadAttention(512, 4)] * 6)
            # self.d1_attention = nn.ModuleList([nn.MultiheadAttention(512, 4, batch_first=True)]*6)

        if "token_type" in self.config.to_dict().keys():
            self.token_type = self.config.token_type
        else:
            self.token_type = "all"

        if "use_dropout" in self.config.to_dict().keys():
            self.use_dropout = self.config.use_dropout
        else:
            self.use_dropout = True

        if "use_bias" in self.config.to_dict().keys():
            self.use_bias = self.config.use_bias
        else:
            self.use_bias = False
        # self.d2_net = nn.ModuleList([build_mlp(2, 300, 512)]*6)
        self.d2_net = nn.ModuleList([nn.Linear(300, 512)] * 6)

        self.pre_d2_layer_norm = nn.ModuleList([LayerNorm(300)] * 6)
        self.d2_layer_norm = nn.ModuleList([LayerNorm(512)] * 6)
        # self.d3_net = nn.ModuleList([build_mlp(2, 512, 512)]*6)
        self.d3_net = nn.ModuleList([nn.Linear(512, 512)] * 6)

        self.pre_d3_layer_norm = nn.ModuleList([LayerNorm(512)] * 6)
        self.d3_layer_norm = nn.ModuleList([LayerNorm(512)] * 6)
        self.ca_layer_norm = nn.ModuleList([LayerNorm(512)] * 6)

        # self.post_mlp = build_mlp(2, 512, output_hidden_size)
        # self.post_mlp = build_mlp(4, 512, output_hidden_size)
        if self.sep:
            self.post_mlp_1d = build_mlp(2, 512, output_hidden_size)
            self.post_mlp_2d = build_mlp(2, 512, output_hidden_size)
            self.post_mlp_3d = build_mlp(2, 512, output_hidden_size)

        else:
            self.post_mlp = build_mlp(6, 512, output_hidden_size)
        # self.post_mlp = nn.Linear(512, output_hidden_size)
        ###
        # self.post_mlp = nn.ModuleList([nn.Linear(512, output_hidden_size)]*6)
        ###

        self.layer_norm_post = LayerNorm(512)

        self.d2_attention = nn.ModuleList([MultiheadAttention(512, 4)] * 6)
        self.d3_attention = nn.ModuleList([MultiheadAttention(512, 4)] * 6)
        # self.d2_attention = nn.ModuleList([nn.MultiheadAttention(512, 4, batch_first=True)]*6)
        # self.d3_attention = nn.ModuleList([nn.MultiheadAttention(512, 4, batch_first=True)]*6)

        if self.use_bias:
            self.d2_attn_bias = nn.ModuleList([MoleculeAttnBias()] * 6)
            self.d3_attn_bias = nn.ModuleList([MoleculeAttnBias()] * 6)

        if (
            "mtl_type" in self.config.to_dict().keys()
            and self.config.mtl_type == "crossattn"
        ):
            if self.sep:
                self.cross_attention_1d = nn.ModuleList(
                    [nn.MultiheadAttention(512, 4, batch_first=True)] * 6
                )
                self.cross_attention_2d = nn.ModuleList(
                    [nn.MultiheadAttention(512, 4, batch_first=True)] * 6
                )
                self.cross_attention_3d = nn.ModuleList(
                    [nn.MultiheadAttention(512, 4, batch_first=True)] * 6
                )
            else:
                #####
                self.cross_attention = nn.ModuleList(
                    [nn.MultiheadAttention(512, 4, batch_first=True)] * 6
                )
                #####
            ###
            # self.query_linear = nn.ModuleList([nn.Linear(512, 512)]*6)
            # self.key_linear = nn.ModuleList([nn.Linear(512, 512)]*6)
            ###

            if "add_emb" in self.config.to_dict().keys():
                self.add_emb = self.config.add_emb
            else:
                self.add_emb = False
            if self.sep:
                self.query_tokens_1d = nn.Parameter(
                    torch.randn((6, (self.num_query_tokens) // 18, 512))
                )
                self.query_tokens_2d = nn.Parameter(
                    torch.randn((6, (self.num_query_tokens) // 18, 512))
                )
                self.query_tokens_3d = nn.Parameter(
                    torch.randn((6, (self.num_query_tokens) // 18, 512))
                )
                self.use_dropout = False
            else:
                if self.add_emb:
                    self.query_tokens = nn.Parameter(
                        torch.randn((6, (self.num_query_tokens) // 6, 512))
                    )
                    self.query1d_tokens = nn.Parameter(
                        torch.randn((6, (self.num_query_tokens) // 6, 512))
                    )
                    self.query2d_tokens = nn.Parameter(
                        torch.randn((6, (self.num_query_tokens) // 6, 512))
                    )
                    self.query3d_tokens = nn.Parameter(
                        torch.randn((6, (self.num_query_tokens) // 6, 512))
                    )

                else:
                    self.query_tokens = nn.Parameter(
                        torch.randn((6, (self.num_query_tokens) // 6, 512))
                    )

            self.mtl_type = "crossattn"

        # elif "mtl_type" in self.config.to_dict().keys() and self.config.mtl_type == "sagpool":
        # self.pool = nn.ModuleList([Pooling(512, num_queries=(self.num_query_tokens)//6)]*6)
        # self.mtl_type = "sagpool"

        # think tokens
        # self.eos_tokens = build_eos_tokens(config, output_hidden_size)
        self.eos_tokens = None
        self.prenorm = build_prenorm(config)
        # self.layer_norm = nn.ModuleList(LayerNorm(output_hidden_size) for _ in range(6))

        self.sample_idx = 0

    def build_net(self):
        raise NotImplementedError()

    def _forward(self, x):
        raise NotImplementedError()

    def forward(
        self,
        h_list,
        h3d_list,
        g3d_mask,
        new_selfies_embeds=None,
        new_selfies_masks=None,
        return_not_use_3d=None,
        edge_index=None,
        batch=None,
        dist=None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, encoder_hidden_size) tensor from the visual backbone (CLIP visual encoder), including cls token.
        """
        h_list = h_list[-6:]
        h3d_list = h3d_list[-6:]
        if self.prenorm is not None:
            h_list = [(self.prenorm(h[0]), h[1]) for h in h_list]
            h3d_list = [self.prenorm(h) for h in h3d_list]

        # num_nodes = h_list[0].size(1)
        # h_pool_tensor = torch.stack(h_pool_list, dim=1)

        if self.sep:
            new_query_tokens_1d = []
            new_query_tokens_2d = []
            new_query_tokens_3d = []
        else:
            new_query_tokens = []

        for i in range(6):
            g_node_num = h_list[i][1].size(-1)
            if new_selfies_embeds is not None:
                pre_d1_tokens = self.d1_net[i](
                    self.pre_d1_layer_norm[i](new_selfies_embeds)
                )
                d1_tokens = self.d1_layer_norm[i](pre_d1_tokens)
                d1_tokens = (
                    self.d1_attention[i](d1_tokens, key_padding_mask=new_selfies_masks)[
                        0
                    ]
                    + pre_d1_tokens
                )
                # d1_tokens = self.d1_attention[i](d1_tokens, d1_tokens, d1_tokens, key_padding_mask=new_selfies_masks)[0] + pre_d1_tokens

            pre_d2_tokens = self.d2_net[i](self.pre_d2_layer_norm[i](h_list[i][0]))
            d2_tokens = self.d2_layer_norm[i](pre_d2_tokens)

            if self.use_bias:
                d2_attn_bias = self.d2_attn_bias[i](
                    edge_index, g3d_mask, dist, batch, g_node_num, False
                )
                d2_tokens = (
                    self.d2_attention[i](
                        d2_tokens,
                        key_padding_mask=torch.logical_not(h_list[i][1]),
                        attn_bias=d2_attn_bias,
                    )[0]
                    + pre_d2_tokens
                )
            else:
                d2_tokens = (
                    self.d2_attention[i](
                        d2_tokens, key_padding_mask=torch.logical_not(h_list[i][1])
                    )[0]
                    + pre_d2_tokens
                )
                # d2_tokens = self.d2_attention[i](d2_tokens, d2_tokens, d2_tokens, key_padding_mask=torch.logical_not(h_list[i][1]))[0] + pre_d2_tokens

            pre_d3_tokens = self.d3_net[i](self.pre_d3_layer_norm[i](h3d_list[i]))
            d3_tokens = self.d3_layer_norm[i](pre_d3_tokens)
            if self.use_bias:
                d3_attn_bias = self.d3_attn_bias[i](
                    edge_index, g3d_mask, dist, batch, g_node_num, True
                )
                d3_tokens = (
                    self.d3_attention[i](
                        d3_tokens, key_padding_mask=g3d_mask, attn_bias=d3_attn_bias
                    )[0]
                    + pre_d3_tokens
                )
            else:
                d3_tokens = (
                    self.d3_attention[i](d3_tokens, key_padding_mask=g3d_mask)[0]
                    + pre_d3_tokens
                )
                # d3_tokens = self.d3_attention[i](d3_tokens, d3_tokens, d3_tokens, key_padding_mask=g3d_mask)[0] + pre_d3_tokens

            if self.sep:
                if self.token_type in ["all", "d1", "d1,d2", "d1,d3"]:
                    new_query_tokens_1d.append(
                        self.cross_attention_1d[i](
                            self.query_tokens_1d[i]
                            .expand(d1_tokens.size(0), -1, -1)
                            .to(d1_tokens.device),
                            d1_tokens,
                            d1_tokens,
                            key_padding_mask=new_selfies_masks,
                        )[0]
                    )
                if self.token_type in ["all", "d2", "d1,d2", "d2,d3"]:
                    new_query_tokens_2d.append(
                        self.cross_attention_2d[i](
                            self.query_tokens_2d[i]
                            .expand(d1_tokens.size(0), -1, -1)
                            .to(d1_tokens.device),
                            d2_tokens,
                            d2_tokens,
                            key_padding_mask=torch.logical_not(h_list[i][1]),
                        )[0]
                    )
                if self.token_type in ["all", "d3", "d1,d3", "d2,d3"]:
                    new_query_tokens_3d.append(
                        self.cross_attention_3d[i](
                            self.query_tokens_3d[i]
                            .expand(d1_tokens.size(0), -1, -1)
                            .to(d1_tokens.device),
                            d3_tokens,
                            d3_tokens,
                            key_padding_mask=g3d_mask,
                        )[0]
                    )

            else:
                if new_selfies_embeds is not None:
                    query_tokens = self.query_tokens
                    if self.training:
                        if self.token_type == "all":
                            if self.use_dropout:
                                dropout_prob = torch.ones(3) * 0.8
                                for _ in range(100):
                                    dropout_mask = torch.bernoulli(dropout_prob).to(
                                        h_list[0][0].device
                                    )
                                    dropout_mask = dropout_mask.bool()
                                    if dropout_mask[:2].sum() == 0:  # d1, d2
                                        continue
                                    if dropout_mask.sum() > 0:
                                        break

                                mask_list = list(torch.split(dropout_mask, 1))
                            else:
                                mask_list = [True, True, True]

                            if g3d_mask is None:
                                mask_list[2] = False
                            token_list = []
                            token_mask_list = []
                            if mask_list[0]:
                                token_list.append(d1_tokens)
                                token_mask_list.append(new_selfies_masks)
                                if self.add_emb:
                                    query_tokens = query_tokens + self.query1d_tokens
                            if mask_list[1]:
                                token_list.append(d2_tokens)
                                token_mask_list.append(torch.logical_not(h_list[i][1]))
                                if self.add_emb:
                                    query_tokens = query_tokens + self.query2d_tokens
                            if mask_list[2]:
                                token_list.append(d3_tokens)
                                # token_mask_list.append(g3d_mask)
                                token3d_mask_list = []
                                for idx, use_3d_flag in enumerate(return_not_use_3d):
                                    if use_3d_flag:
                                        token3d_mask_list.append(
                                            torch.ones_like(g3d_mask[idx]).bool()
                                        )
                                    else:
                                        token3d_mask_list.append(g3d_mask[idx])
                                token3d_mask_list = torch.stack(
                                    token3d_mask_list, dim=0
                                )
                                token_mask_list.append(token3d_mask_list)
                                if self.add_emb:
                                    query_tokens = query_tokens + self.query3d_tokens

                            curr_tokens = torch.cat(token_list, 1)
                            mask = torch.cat(token_mask_list, 1)
                        elif self.token_type == "d1":
                            curr_tokens = d1_tokens
                            mask = new_selfies_masks
                            if self.add_emb:
                                query_tokens = query_tokens + self.query1d_tokens
                        elif self.token_type == "d2":
                            curr_tokens = d2_tokens
                            mask = torch.logical_not(h_list[i][1])
                            if self.add_emb:
                                query_tokens = query_tokens + self.query2d_tokens
                        elif self.token_type == "d3":
                            curr_tokens = d3_tokens
                            mask = g3d_mask
                            if self.add_emb:
                                query_tokens = query_tokens + self.query3d_tokens
                        elif self.token_type == "d1,d2":
                            token_list = []
                            token_mask_list = []
                            token_list.append(d1_tokens)
                            token_mask_list.append(new_selfies_masks)
                            token_list.append(d2_tokens)
                            token_mask_list.append(torch.logical_not(h_list[i][1]))
                            curr_tokens = torch.cat(token_list, 1)
                            mask = torch.cat(token_mask_list, 1)
                            if self.add_emb:
                                query_tokens = (
                                    query_tokens
                                    + self.query1d_tokens
                                    + self.query2d_tokens
                                )
                        elif self.token_type == "d1,d3":
                            token_list = []
                            token_mask_list = []
                            token_list.append(d1_tokens)
                            token_mask_list.append(new_selfies_masks)
                            if g3d_mask is not None:
                                token_list.append(d3_tokens)
                                token3d_mask_list = []
                                for idx, use_3d_flag in enumerate(return_not_use_3d):
                                    if use_3d_flag:
                                        token3d_mask_list.append(
                                            torch.ones_like(g3d_mask[idx]).bool()
                                        )
                                    else:
                                        token3d_mask_list.append(g3d_mask[idx])
                                token3d_mask_list = torch.stack(
                                    token3d_mask_list, dim=0
                                )
                                token_mask_list.append(token3d_mask_list)
                            curr_tokens = torch.cat(token_list, 1)
                            mask = torch.cat(token_mask_list, 1)
                            if self.add_emb:
                                query_tokens = (
                                    query_tokens
                                    + self.query1d_tokens
                                    + self.query3d_tokens
                                )
                        elif self.token_type == "d2,d3":
                            token_list = []
                            token_mask_list = []
                            token_list.append(d2_tokens)
                            token_mask_list.append(torch.logical_not(h_list[i][1]))
                            if g3d_mask is not None:
                                token_list.append(d3_tokens)
                                token3d_mask_list = []
                                for idx, use_3d_flag in enumerate(return_not_use_3d):
                                    if use_3d_flag:
                                        token3d_mask_list.append(
                                            torch.ones_like(g3d_mask[idx]).bool()
                                        )
                                    else:
                                        token3d_mask_list.append(g3d_mask[idx])
                                token3d_mask_list = torch.stack(
                                    token3d_mask_list, dim=0
                                )
                                token_mask_list.append(token3d_mask_list)
                            curr_tokens = torch.cat(token_list, 1)
                            mask = torch.cat(token_mask_list, 1)
                            if self.add_emb:
                                query_tokens = (
                                    query_tokens
                                    + self.query2d_tokens
                                    + self.query3d_tokens
                                )
                        else:
                            raise NotImplementedError
                    else:
                        if self.token_type == "all":
                            mask_list = [True, True, True]
                            if g3d_mask is None:
                                mask_list[2] = False

                            token_list = []
                            token_mask_list = []
                            if mask_list[0]:
                                token_list.append(d1_tokens)
                                token_mask_list.append(new_selfies_masks)
                                if self.add_emb:
                                    query_tokens = query_tokens + self.query1d_tokens
                            if mask_list[1]:
                                token_list.append(d2_tokens)
                                token_mask_list.append(torch.logical_not(h_list[i][1]))
                                if self.add_emb:
                                    query_tokens = query_tokens + self.query2d_tokens
                            if mask_list[2]:
                                token_list.append(d3_tokens)
                                # token_mask_list.append(g3d_mask)
                                token3d_mask_list = []
                                for idx, use_3d_flag in enumerate(return_not_use_3d):
                                    if use_3d_flag:
                                        token3d_mask_list.append(
                                            torch.ones_like(g3d_mask[idx]).bool()
                                        )
                                    else:
                                        token3d_mask_list.append(g3d_mask[idx])
                                token3d_mask_list = torch.stack(
                                    token3d_mask_list, dim=0
                                )
                                token_mask_list.append(token3d_mask_list)
                                if self.add_emb:
                                    query_tokens = query_tokens + self.query3d_tokens

                            curr_tokens = torch.cat(token_list, 1)
                            mask = torch.cat(token_mask_list, 1)
                        elif self.token_type == "d1":
                            curr_tokens = d1_tokens
                            mask = new_selfies_masks
                            if self.add_emb:
                                query_tokens = query_tokens + self.query1d_tokens
                        elif self.token_type == "d2":
                            curr_tokens = d2_tokens
                            mask = torch.logical_not(h_list[i][1])
                            if self.add_emb:
                                query_tokens = query_tokens + self.query2d_tokens
                        elif self.token_type == "d3":
                            curr_tokens = d3_tokens
                            mask = g3d_mask
                            if self.add_emb:
                                query_tokens = query_tokens + self.query3d_tokens
                        elif self.token_type == "d1,d2":
                            token_list = []
                            token_mask_list = []
                            token_list.append(d1_tokens)
                            token_mask_list.append(new_selfies_masks)
                            token_list.append(d2_tokens)
                            token_mask_list.append(torch.logical_not(h_list[i][1]))
                            curr_tokens = torch.cat(token_list, 1)
                            mask = torch.cat(token_mask_list, 1)
                            if self.add_emb:
                                query_tokens = (
                                    query_tokens
                                    + self.query1d_tokens
                                    + self.query2d_tokens
                                )
                        elif self.token_type == "d1,d3":
                            token_list = []
                            token_mask_list = []
                            token_list.append(d1_tokens)
                            token_mask_list.append(new_selfies_masks)
                            if g3d_mask is not None:
                                token_list.append(d3_tokens)
                                token3d_mask_list = []
                                for idx, use_3d_flag in enumerate(return_not_use_3d):
                                    if use_3d_flag:
                                        token3d_mask_list.append(
                                            torch.ones_like(g3d_mask[idx]).bool()
                                        )
                                    else:
                                        token3d_mask_list.append(g3d_mask[idx])
                                token3d_mask_list = torch.stack(
                                    token3d_mask_list, dim=0
                                )
                                token_mask_list.append(token3d_mask_list)
                            curr_tokens = torch.cat(token_list, 1)
                            mask = torch.cat(token_mask_list, 1)
                            if self.add_emb:
                                query_tokens = (
                                    query_tokens
                                    + self.query1d_tokens
                                    + self.query3d_tokens
                                )
                        elif self.token_type == "d2,d3":
                            token_list = []
                            token_mask_list = []
                            token_list.append(d2_tokens)
                            token_mask_list.append(torch.logical_not(h_list[i][1]))
                            if g3d_mask is not None:
                                token_list.append(d3_tokens)
                                token3d_mask_list = []
                                for idx, use_3d_flag in enumerate(return_not_use_3d):
                                    if use_3d_flag:
                                        token3d_mask_list.append(
                                            torch.ones_like(g3d_mask[idx]).bool()
                                        )
                                    else:
                                        token3d_mask_list.append(g3d_mask[idx])
                                token3d_mask_list = torch.stack(
                                    token3d_mask_list, dim=0
                                )
                                token_mask_list.append(token3d_mask_list)
                            curr_tokens = torch.cat(token_list, 1)
                            mask = torch.cat(token_mask_list, 1)
                            if self.add_emb:
                                query_tokens = (
                                    query_tokens
                                    + self.query2d_tokens
                                    + self.query3d_tokens
                                )

                        else:
                            raise NotImplementedError

                else:
                    curr_tokens = torch.cat([d2_tokens, d3_tokens], 1)
                    mask = torch.cat([torch.logical_not(h_list[i][1]), g3d_mask], 1)

                #####
                curr_tokens = self.ca_layer_norm[i](curr_tokens)
                new_query_tokens.append(
                    self.cross_attention[i](
                        query_tokens[i]
                        .expand(curr_tokens.size(0), -1, -1)
                        .to(curr_tokens.device),
                        curr_tokens,
                        curr_tokens,
                        key_padding_mask=mask,
                    )[0]
                )
                # attn_weight= self.cross_attention[i](query_tokens[i].expand(curr_tokens.size(0), -1, -1).to(curr_tokens.device), curr_tokens, curr_tokens, key_padding_mask=mask, average_attn_weights=True)[1]
                # idx = 0
                # print(attn_weight[0])
                # print(mask.shape)
                # print(mask[0])
                # attn_weight = attn_weight[torch.logical_not(mask.unsqueeze(1)).repeat(1,24,1)].cpu().numpy()
                # print(attn_weight.shape)
                #####
                ###
                # new_q = nn.functional.scaled_dot_product_attention(self.query_linear[i](query_tokens[i]).expand(curr_tokens.size(0), -1, -1).to(curr_tokens.device), self.key_linear[i](curr_tokens), curr_tokens, attn_mask=mask.logical_not().unsqueeze(1).repeat(1, (self.num_query_tokens)//6, 1))
                # new_query_tokens.append(self.post_mlp[i](new_q))
                ###
        #         for idx in range(16):
        #             plt.figure(figsize=(8, 6))
        #             plt.imshow(attn_weight[idx], cmap='viridis', aspect='auto')
        #             plt.tight_layout()
        #             plt.savefig(f'qc2_attn_weight{self.sample_idx*16+idx}_{i}.png')
        # self.sample_idx += 1
        ###
        # query_tokens = torch.cat(new_query_tokens, 1)
        ###
        #####
        if not self.sep:
            x = torch.cat(new_query_tokens, 1)
            x = self.layer_norm_post(x)
            query_tokens = self.post_mlp(x)  # (B, L, output_hidden_size)
            #####
        else:
            output_query_tokens = []
            if self.token_type in ["all", "d1", "d1,d2", "d1,d3"]:
                x_1d = torch.cat(new_query_tokens_1d, 1)
                output_query_tokens.append(self.post_mlp_1d(x_1d))
            if self.token_type in ["all", "d2", "d1,d2", "d2,d3"]:
                x_2d = torch.cat(new_query_tokens_2d, 1)
                output_query_tokens.append(self.post_mlp_2d(x_2d))
            if self.token_type in ["all", "d3", "d1,d3", "d2,d3"]:
                x_3d = torch.cat(new_query_tokens_3d, 1)
                output_query_tokens.append(self.post_mlp_3d(x_3d))
            # x_1d = torch.cat(new_query_tokens_1d, 1)
            # query_tokens_1d = self.post_mlp_1d(x_1d)
            # x_2d = torch.cat(new_query_tokens_2d, 1)
            # query_tokens_2d = self.post_mlp_2d(x_2d)
            # x_3d = torch.cat(new_query_tokens_3d, 1)
            # query_tokens_3d = self.post_mlp_3d(x_3d)

            query_tokens = torch.cat(output_query_tokens, 1)

        B = query_tokens.size(0)
        if self.eos_tokens is not None:
            x = torch.cat([query_tokens, self.eos_tokens.expand(B, -1, -1)], dim=1)
        else:
            x = query_tokens

        x1, x2 = torch.split(x, x.size(1) // 2, dim=1)

        return x1.contiguous(), x2.contiguous()


class MoleculeAttnBias(nn.Module):
    def __init__(self, dim=512, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.bias_2d = nn.Embedding(513, num_heads)
        self.bias_3d = Molecule3dAttnBias(dim, num_heads)
        nn.init.zeros_(self.bias_2d.weight)

    def forward(
        self, edge_index, padding_mask, src_distance, batch, g_node_num, d3=True
    ):
        # Need to check
        num_nodes = max(src_distance.size(-1), g_node_num + 1)

        # Generate new mask for 3d
        # if padding_mask is not None:
        # temp_mask = torch.ones(padding_mask.size(0), 1, device=padding_mask.device, dtype=torch.bool)
        # padding_mask = torch.cat([temp_mask, padding_mask[:, 2:], temp_mask], dim=1)

        dense_edge_index = to_dense_adj(edge_index, batch)

        # 2d tensor
        new_tensor = (
            torch.ones(
                src_distance.size(0),
                num_nodes,
                num_nodes,
                device=src_distance.device,
                dtype=torch.long,
            )
            * 512
        )
        for i, d_i in enumerate(dense_edge_index):
            try:
                spd_i = (
                    torch.from_numpy(shortest_path(d_i.cpu().numpy()))
                    .clamp(0, 512)
                    .long()
                    .to(src_distance.device)
                )
                new_tensor[i][1 : spd_i.size(0) + 1, 1 : spd_i.size(1) + 1] = spd_i
            except Exception:
                import pdb

                pdb.set_trace()

        attn_bias_2d = self.bias_2d(new_tensor)  # (B, #Atoms, #Atoms, num_heads)
        attn_bias_2d = attn_bias_2d.permute(0, 3, 1, 2).contiguous()
        # attn_bias_2d.masked_fill_(src_tokens.eq(self.padding_idx).unsqueeze(1).unsqueeze(2), float('-inf'))

        if num_nodes == g_node_num + 1:
            pad_tensor = torch.zeros(
                src_distance.size(0),
                src_distance.size(1),
                num_nodes - src_distance.size(-1),
                device=src_distance.device,
            )
            new_src_distance = torch.cat(
                [src_distance, pad_tensor], dim=-1
            )  # (B, #Atoms, #Atoms + alpha)
            pad_tensor = torch.zeros(
                src_distance.size(0),
                num_nodes - src_distance.size(1),
                num_nodes,
                device=src_distance.device,
            )
            new_src_distance = torch.cat(
                [new_src_distance, pad_tensor], dim=1
            )  # (B, #Atoms, #Atoms + alpha)
        else:
            new_src_distance = src_distance

        attn_bias_3d = self.bias_3d(padding_mask, new_src_distance)
        if num_nodes == g_node_num + 1:
            new_attn_bias = attn_bias_2d + attn_bias_3d
            if d3:
                new_attn_bias = new_attn_bias[
                    :, :, : src_distance.size(-1), : src_distance.size(-1)
                ]
        else:
            new_attn_bias = attn_bias_2d + attn_bias_3d
        if not d3:
            new_attn_bias = new_attn_bias[:, :, 1 : g_node_num + 1, 1 : g_node_num + 1]

        return new_attn_bias


class Molecule3dAttnBias(nn.Module):
    def __init__(self, dim=512, num_heads=4, num_kernel=128):
        super(Molecule3dAttnBias, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.num_kernel = num_kernel
        self.gbf = GaussianLayer(num_kernel)
        self.gbf_proj = nn.Linear(num_kernel, num_heads)

    def forward(self, padding_mask, src_distance):
        # if padding_mask is not None:
        # new_src_distance = []
        # for i in range(padding_mask.size(0)):
        # import pdb; pdb.set_trace()
        # new_src_distance.append(src_distance[i][padding_mask[i],:][:, padding_mask[i]])
        # src_distance = torch.stack(new_src_distance, dim=0)
        dist_gbf = self.gbf(src_distance)
        dist_gbf_proj = self.gbf_proj(dist_gbf)  # (B, #Atoms, #Atoms, num_heads)
        attn_bias_3d = dist_gbf_proj.permute(0, 3, 1, 2).contiguous()

        return attn_bias_3d


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)

    def forward(self, x):
        x = x.unsqueeze(-1).repeat(1, 1, 1, self.K)  # (B, #Atoms, #Atoms, K)
        mean = self.means.weight.float().view(1, 1, 1, -1)
        std = self.stds.weight.float().view(1, 1, 1, -1).abs() + 1e-2
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()

        if hidden is None:
            hidden = input
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.functional.gelu(x)
        x = self.layer2(x)
        return x
