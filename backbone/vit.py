import torch
import torch.nn as nn

class PatchPositionEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, embedding_dim=768):
        super(PatchPositionEmbedding,self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_patches = (image_size // patch_size)**2
        self.projection = nn.Conv2d(3, embedding_dim, patch_size, patch_size)
        self.class_token = torch.zeros(1, 1, embedding_dim)
        self.position_token = torch.zeros(1, self.num_patches+1, embedding_dim)
    def forward(self, x):
        batch_size = x.shape[0]
        self.class_token = nn.Parameter(self.class_token.expand(batch_size, -1, -1))
        self.position_token = nn.Parameter(self.position_token.expand(batch_size, -1, -1))
        x = self.projection(x).flatten(2).transpose(1,2)
        x = torch.cat([x, self.class_token], dim=1)
        x += self.position_token
        return x
    

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=8, attention_drop_rate=0, projection_drop_rate=0):
        super(MultiHeadAttentionBlock,self).__init__()
        self.num_heads = num_heads
        self.per_head_embedding_dim = embedding_dim // num_heads
        self.calculate_qkv = nn.Linear(embedding_dim, embedding_dim*3)
        self.drop_attention = nn.Dropout(attention_drop_rate)
        self.drop_projection = nn.Dropout(projection_drop_rate)
        self.softmax = nn.Softmax(dim=-1)
        self.projection = nn.Linear(embedding_dim, embedding_dim)
    def forward(self, x):
        batch_size, num_patches_add_one, embedding_dim = x.shape
        qkv = self.calculate_qkv(x) \
            .reshape(batch_size, num_patches_add_one, 3, self.num_heads, self.per_head_embedding_dim) \
            .permute(2, 0, 3, 1, 4)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        attention = q @ k.transpose(-2,-1) / (self.num_heads)**0.5
        attention = self.drop_attention(self.softmax(attention))
        x = (attention @ v).transpose(1, 2).reshape(batch_size, num_patches_add_one, embedding_dim)
        x = self.drop_projection(self.projection(x))
        return x


class MlpBlock(nn.Module):
    def __init__(self, in_channels=768, out_channels=768, drop_rate = 0):
        super(MlpBlock,self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels*4)
        self.ac = nn.GELU()
        self.fc2 = nn.Linear(in_channels*4, out_channels)
        self.drop = nn.Dropout(drop_rate)
    def forward(self, x):
        x = self.drop(self.ac(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x
    

class EncoderBlock(nn.Module):
    def __init__(self, embedding_dim=768, path_drop_rate=0):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttentionBlock()
        self.mlp = MlpBlock()
        self.ln = nn.LayerNorm(embedding_dim)
        self.drop_path = nn.Dropout(path_drop_rate)
    def forward(self, x):
        x += self.drop_path(self.attention(self.ln(x)))
        x += self.drop_path(self.mlp(self.ln(x)))
        return x
    
class MlpHead(nn.Module):
    def __init__(self, embrdding_dim=768, num_classes=1000):
        super(MlpHead, self).__init__()
        self.head = nn.Linear(embrdding_dim, num_classes)
    def forward(self, x):
        x = self.head(x)
        return x
    

class VisionTransformer(nn.Module):
    def __init__(self, depth=12):
        super(VisionTransformer, self).__init__()
        self.position_embedding = PatchPositionEmbedding()
        self.encoder = nn.Sequential(*[EncoderBlock() for i in range(depth)])
        self.head = MlpHead()
    def forward(self, x):
        x = self.position_embedding(x)
        x = self.encoder(x)
        x = x[:,0]
        x = self.head(x)
        return x
if __name__=='__main__':
    net = VisionTransformer()
    x = torch.ones(1,3,224,224)
    x = net(x)
    print(x.shape)

