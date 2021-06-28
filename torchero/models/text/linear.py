from torch import nn

class LinearModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_size):
        super(LinearModel, self).__init__()
        self.embeddings = nn.EmbeddingBag(vocab_size,
                                          embedding_dim=embedding_dim)
        self.linear = nn.Linear(embedding_dim, output_size)
        
    def forward(self, x, offsets):
        x = self.embeddings(x, offsets)
        x = self.linear(x)
        return x