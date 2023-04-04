import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden=[64, 64],
                 hidden_activation=nn.LeakyReLU(), out_activation=nn.Identity(), bias=True):
        super(MLP, self).__init__()

        in_dims = [in_dim] + hidden
        out_dims = hidden + [out_dim]

        self.in_dim = in_dims[0]
        self.out_dim = out_dims[-1]

        self.l = nn.ModuleList()
        self.act = nn.ModuleList()

        for i, o in zip(in_dims, out_dims):
            layer = nn.Linear(i, o, bias=bias)
            # nn.init.xavier_uniform_(layer.weight)
            nn.init.orthogonal_(layer.weight)
            self.l.append(layer)

        for _ in range(len(hidden)):
            self.act.append(hidden_activation)
        self.act.append(out_activation)

    def forward(self, x):
        for l, act in zip(self.l, self.act):
            x = act(l(x))
        return x


if __name__ == "__main__":

    input_dim = 4
    output_dim = 5
    batch_size = 6
    model = MLP(in_dim=input_dim, out_dim=output_dim,
                hidden=[16, 16], out_activation=nn.Softmax(dim=-1))

    x = torch.randn(batch_size, input_dim)
    y = model(x)

    print("------- MLP test -------")
    print("input shape  : ", x.shape)
    print("output shape : ", y.shape)
