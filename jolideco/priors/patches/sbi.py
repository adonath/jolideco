# Loosely based on https://keras.io/examples/vision/image_classification_with_vision_transformer/

from dataclasses import dataclass

from torch import nn


@dataclass
class SBIPatchTransformerConfig:
    """Patch Transformer configuration"""

    patch_size: int = 8
    projection_dim: int = 48
    num_heads: int = 4
    transformer_layers: int = 4
    mlp_head_units: tuple[int, int] = (256, 128)
    dropout_rate: float = 0.1
    num_classes: int = 2

    @property
    def transformer_units(self):
        """Transformer units"""
        return [2 * self.projection_dim, self.projection_dim]


class Dense(nn.Module):
    """Dense layer with activation and dropout

    Attributes
    ----------
    linear : `nn.Linear`
        Linear layer
    activation : `nn.GELU`
        Activation function
    dropout : `Optional[nn.Dropout]`
        Dropout
    """

    def __init__(self, linear, activation, dropout=None):
        super().__init__()
        self.linear = linear
        self.activation = activation
        self.dropout = dropout

    @classmethod
    def from_args(cls, dim, hidden_dim, use_bias=True, dropout=None):
        """Create MLP unit from config

        Parameters
        ----------
        dim : int
            Dimension
        hidden_dim : int
            Hidden dimension
        use_bias : bool
            Use bias
        dropout : float or None
            Dropout

        Returns
        -------
        unit : `MLPUnit`
            MLP unit
        """
        if dropout is not None:
            dropout = nn.Dropout(dropout)

        return cls(
            linear=nn.Linear(dim, hidden_dim, bias=use_bias),
            activation=nn.GELU(),
            dropout=dropout,
        )

    def forward(self, x):
        """Forward pass"""
        x = self.linear(x)
        x = self.activation(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x


class MLPBlock(nn.Sequential):
    """MLP block

    Attributes
    ----------
    layers : list
        Layers
    """

    @classmethod
    def from_args(cls, dim, hidden_dims, use_bias=True, dropout=None):
        """Create MLP block from args

        Parameters
        ----------
        dim : int
            Dimension
        hidden_dims : tuple of int
            Hidden dimensions per layer
        use_bias : bool
            Use bias
        dropout : Optiona[float]
            Dropout

        Returns
        -------
        block : `MLPBlock`
            MLP block
        """
        dims = [dim] + list(hidden_dims)

        layers = []

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layer = Dense.from_args(in_dim, out_dim, use_bias, dropout)
            layers.append(layer)

        return cls(*layers)


class AttentionBlock(nn.Module):
    """Attention block

    Attributes
    ----------
    norm_1 : `nn.LayerNorm`
        Layer norm 1
    attention : `nn.MultiheadAttention`
        Multihead attention
    norm_2 : `nn.LayerNorm`
        Layer norm 2
    mlp : `nn.Sequential`
        MLP
    """

    _norm_eps = 1e-6

    def __init__(self, attention, norm_1, norm_2, mlp):
        super().__init__()
        self.norm_1 = norm_1
        self.attention = attention
        self.norm_2 = norm_2
        self.mlp = mlp

    @classmethod
    def from_args(
        cls,
        num_heads,
        projection_dim,
        mlp_hidden_dims,
        mlp_dropout=0.1,
        attention_dropout=0.1,
    ):
        """Create attention block from config

        Parameters
        ----------
        num_heads : int
            Number of heads
        projection_dim : int
            Projection dimension
        mlp_hidden_dims : list of int
            MLP units
        mlp_dropout : float
            Dropout
        attention_dropout : float
            Attention dropout

        Returns
        -------
        block : `AttentionBlock`
            Attention block
        """
        attention = nn.MultiheadAttention(
            projection_dim, num_heads, dropout=attention_dropout, batch_first=True
        )

        mlp = MLPBlock.from_args(
            projection_dim, mlp_hidden_dims, use_bias=True, dropout=mlp_dropout
        )

        return cls(
            attention=attention,
            norm_1=nn.LayerNorm(projection_dim, eps=cls._norm_eps),
            norm_2=nn.LayerNorm(projection_dim, eps=cls._norm_eps),
            mlp=mlp,
        )

    def forward(self, x):
        """Forward pass"""
        x1 = self.norm_1(x)
        attention_output, _ = self.attention(x1, x1, x1, need_weights=False)
        x2 = x1 + attention_output
        x3 = self.norm_2(x2)
        x3 = self.mlp(x3)
        return x3 + x2


class SBIPatchTransformerModel(nn.Module):
    """Patch Transformer model

    Attributes
    ----------
    patch_encoder : `nn.Linear`
        Patch encoder
    layers : `nn.Sequential`
        Layers of Attention blocks
    norm : `nn.LayerNorm`
        Layer norm
    dropout : `nn.Dropout`
        Dropout
    mlp_head : `MLPBlock`
        MLP head
    output : `nn.Linear`
        Output layer
    """

    def __init__(self, patch_encoder, layers, norm, dropout, mlp_head, output):
        super().__init__()
        self.patch_encoder = patch_encoder
        self.layers = layers
        self.norm = norm
        self.dropout = dropout
        self.mlp_head = mlp_head
        self.output = output

    @classmethod
    def from_config(cls, config):
        """Create Vision Transformer model from config

        Parameters
        ----------
        config : `SBIVisionTransformerConfig`
            Configuration

        Returns
        -------
        model : `SBIVisionTransformerModel`
            Vision Transformer model
        """

        layers = nn.Sequential()

        for _ in range(config.transformer_layers):
            block = AttentionBlock.from_args(
                num_heads=config.num_heads,
                projection_dim=config.projection_dim,
                mlp_hidden_dims=config.transformer_units,
                mlp_dropout=config.dropout_rate,
                attention_dropout=config.dropout_rate,
            )
            layers.append(block)

        patch_encoder = nn.Linear(
            in_features=config.patch_size**2, out_features=config.projection_dim
        )

        mlp_head = MLPBlock.from_args(
            config.projection_dim, config.mlp_head_units, config.dropout_rate
        )

        output = Dense.from_args(
            config.mlp_head_units[-1], config.num_classes, use_bias=True
        )

        return cls(
            patch_encoder=patch_encoder,
            layers=layers,
            norm=nn.LayerNorm(config.projection_dim, eps=1e-6),
            dropout=nn.Dropout(config.dropout_rate),
            mlp_head=mlp_head,
            output=output,
        )

    def forward(self, x):
        """Forward pass"""
        x = self.patch_encoder(x)
        x = self.layers(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = x.mean(dim=1)
        x = self.mlp_head(x)
        x = self.output(x)
        return x
