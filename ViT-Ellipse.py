#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from random import randint, random
from matplotlib import pyplot as plt

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# In[2]:


class EllipseDataset(Dataset):
    def __init__(self, size, alpha=0.2, beta=2.0):
        self.__size = size
        self.alpha = alpha
        self.beta = beta
    def __getitem__(self, index):
        data = np.zeros(self.__size)
        # random center
        x = randint(self.__size[0]//8, self.__size[0]-self.__size[0]//8)
        y = randint(self.__size[1]//8, self.__size[1]-self.__size[1]//8)
        # random degree
        ang = randint(0, 359)
        # random radius
        a = randint(self.__size[0]//8, self.__size[0]//4)
        b = randint(self.__size[1]//8, self.__size[1]//4)
        cv2.ellipse(data, (x, y), (a,b), ang, 0, 360, (255,255,255), -1)
        ydata = data.copy()
        r = random()
        if r < 0.3:
            # hori
            r = randint(x-a//3, x+a//3)
            if random() < 0.5:
                cv2.rectangle(data, (r, 0), self.__size, (0,0,0), -1)
            else:
                cv2.rectangle(data, (r, 0), (0, self.__size[1]), (0,0,0), -1)
        elif r < 0.6:
            # vert
            r = randint(x-b//3,x+b//3)
            if random() < 0.5:
                cv2.rectangle(data, (0, r), self.__size, (0,0,0), -1)
            else:
                cv2.rectangle(data, (0, r), (self.__size[0], 0), (0,0,0), -1)
        labels = torch.tensor([
            x, y, a*self.alpha,
            b*self.alpha,
            ang*self.beta/180.0*np.pi], dtype=torch.float)
        if np.sum(data) == 0:
            return self.__getitem__(0)
        return torch.tensor(data/255.0, dtype=torch.float).unsqueeze(0), labels
    def __len__(self):
        return 10000


# In[3]:


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# In[4]:


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), 
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
class Attention(nn.Module):              
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)           
        # (b, n(65), dim*3) ---> 3 * (b, n, dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        # q, k, v   (b, h, n, dim_head(64))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# In[5]:


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert  image_height % patch_height ==0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))					# nn.Parameter()定义可学习参数
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)       
        # b c (h p1) (w p2) -> b (h w) (p1 p2 c) -> b (h w) dim
        b, n, _ = x.shape          
        # b表示batchSize, n表示每个块的空间分辨率, _表示一个块内有多少个值

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  
        # self.cls_token: (1, 1, dim) -> cls_tokens: (batchSize, 1, dim)  
        x = torch.cat((cls_tokens, x), dim=1)               
        # 将cls_token拼接到patch token中去       (b, 65, dim)
        x += self.pos_embedding[:, :(n+1)]                  
        # 加位置嵌入（直接加）      (b, 65, dim)
        x = self.dropout(x)

        x = self.transformer(x)                                                 
        # (b, 65, dim)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]                   
        # (b, dim)

        x = self.to_latent(x)                                                   
        # Identity (b, dim)

        return self.mlp_head(x)


# In[6]:


batch_size_train = 64
batch_size_test = 64
random_seed = 1
torch.manual_seed(random_seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[7]:


sizes = (480, 640)
alpha=0.2
beta=5.0
dataset = EllipseDataset(sizes,alpha, beta)
train_size = len(dataset) * 80 // 100
test_size = len(dataset) - train_size
batch_size = 32

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# In[8]:


model = ViT(
    image_size = sizes,
    patch_size = (30,40),
    channels = 1,
    num_classes = 5,
    dim = 512,
    depth = 8,
    heads = 16,
    mlp_dim = 1024,
    dropout = 0.1,
    emb_dropout = 0.1
)
# for X, y in train_loader:
#     p = model(X)
#     print(p.size(), y.size())
#     break


# In[9]:


model.to(device)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
optimizer = torch.optim.Adadelta(model.parameters(), lr=0.01)
optimizer = torch.optim.Nadam(model.parameters(), lr=0.01)
loss = torch.nn.MSELoss(reduction='mean')#torch.nn.CrossEntropyLoss()
num_epoches = 1000
min_mse = 1000.
logs = {
    " v": {
        "loss": [],
        "accuracy": [],
    },
    "validation": {
        "loss": [],
        "accuracy": [],
    }
}

for epoch in range(num_epoches):
    # learn
    model.train()
    train_loss = 0
    accuracy = 0
    for X, y in train_loader:              
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()                                             # set gradient to zero
        pred = model(X)                                                   # forward pass (compute output)
        mse_loss = loss(pred, y)                                          # compute loss
        mse_loss.backward()                                               # compute gradient (backpropagation)
        optimizer.step()                                                  # update model with optimizer
        train_loss += mse_loss.detach().cuda().item() * len(X)            # accumulate loss
    train_loss = train_loss / len(train_loader.dataset)
    logs['train']['loss'].append(train_loss)
    # evaluation
    model.eval()                                                          # set model to evalutation mode
    test_loss = 0
    accuracy = 0
    for X, y in test_loader:                                              # iterate through the dataloader
        X = X.to(device)
        y = y.to(device)
        with torch.no_grad():                                             # disable gradient calculation
            pred = model(X)                                               # forward pass (compute output)
            mse_loss = loss(pred, y)                                      # compute loss
        test_loss += mse_loss.detach().cuda().item() * len(X)             # accumulate loss
    test_loss = test_loss / len(test_loader.dataset)   
    if test_loss < min_mse:
        min_mse = test_loss
    logs['validation']['loss'].append(test_loss)
    if epoch % 50 == 0:
        print(f'epoch = {epoch + 1:4d}, train loss = {train_loss:.4f}, test loss = {test_loss:.4f}')


# In[10]:


stop = 500
plt.plot(logs['train']['loss'][:stop])
plt.plot(logs['validation']['loss'][:stop])
plt.show()


# In[ ]:


np.savez('logs.npz', logs)


# In[11]:


torch.save(model, 'pupil-480640-1.pkl')


# In[105]:


model = torch.load('pupil-480640.pkl')


# In[106]:


model.to('cpu')
for X, y in train_loader:
    p = model(X)
    print(p[:2,:], y[:2,:])
    break


# In[107]:


X, y = dataset[0]
X = X.unsqueeze(0)

p = model(X)
gray = (X*255).squeeze(0).squeeze(0).numpy()
print(y,p)
gray.reshape((480,640))
y = np.array([y[0],y[1],y[2]/alpha,y[3]/alpha,y[4]/np.pi*180/beta], dtype=np.compat.long)
p = p[0].detach().numpy()
p = np.array([p[0],p[1],p[2]/alpha,p[3]/alpha,p[4]/np.pi*180/beta], dtype=np.compat.long)

# p = p[0].long().numpy()
gray = np.array(np.rint(gray), dtype=np.uint8)
colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
print(p, y)
cv2.ellipse(colored, [p[0], p[1]], [p[2], p[3]], 360-p[4], 0, 360, (255,0,0))
cv2.ellipse(colored, [y[0], y[1]], [y[2], y[3]], y[4], 0, 360, (0,0,255))
cv2.circle(colored, [p[0], p[1]], 5, (255,0,0), -1)
plt.imshow(colored)
plt.axis('off')
plt.show()


# In[108]:


from skimage.filters import threshold_sauvola
from skimage.morphology import opening, closing
from skimage import measure
from skimage.morphology import convex_hull_image
from skimage.transform import resize

def step1(gray, showing=False):
    # threshold
    ones = np.ones(gray.shape)*255
    thresh_sauvola = threshold_sauvola(gray, window_size=101, k=0.3)
    # open
    gray = opening(gray < thresh_sauvola)*ones
    # max conn
    gray_label = measure.label(gray, connectivity = 2)
    props = measure.regionprops(gray_label)
    numPix = []
    for k in range(len(props)):
        numPix += [props[k].area]
    maxnum = max(numPix)
    idx = numPix.index(maxnum)
    gray = convex_hull_image(gray_label == idx+1)*ones
    return gray


# In[114]:


video = cv2.VideoCapture('10.avi')


# In[118]:


for k in range(1000):
    ret, frame = video.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray1 = step1(gray)
    X = torch.tensor(gray1/256.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    print(X.size(), X.dtype)
    p = model(X)
    p = p[0].detach().numpy()
    p = np.array([p[0],p[1],p[2]/alpha,p[3]/alpha,p[4]/np.pi*180/beta], dtype=np.compat.long)
    gray2 = np.array(np.rint(gray1), dtype=np.uint8)
    colored = cv2.cvtColor(gray2, cv2.COLOR_GRAY2RGB)
    cv2.ellipse(colored, [p[0], p[1]], [p[2], p[3]], 360-p[4], 0, 360, (255,0,0))
    cv2.circle(colored, [p[0], p[1]], 5, (255,0,0), -1)
    plt.subplot(121)
    plt.imshow(colored)
    plt.axis('off')
    print(p)
    cv2.circle(frame, [p[0], p[1]], 5, (255,0,0), -1)
    plt.subplot(122)
    plt.imshow(frame)
    plt.axis('off')
    plt.show()

