
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
import scipy.io as scio
import numpy as np

device=torch.device('cuda' if torch.cuda.is_available() else "cpu")

def squash(x, dim=-1):
    squared_norm= (x**2).sum(dim=dim, keepdim=True)
    scale=squared_norm/(1+squared_norm)
    val=scale*x/(squared_norm.sqrt()+1e-8)
    return val

class Primary_Caps(nn.Module):
    def __init__(self,num_conv_units, in_channels, out_channels, kernel_size, stride):
        super(Primary_Caps,self).__init__()
        self.pre_flat=nn.Sequential(
            #input shape(batch,256,32,32) 
            #to further cnn to shape(batch,256,12,12)
            nn.Conv2d(in_channels, num_conv_units * out_channels, kernel_size, stride=stride),
            
        )
        self.out_channels=out_channels

    def forward(self,x):
        out=self.pre_flat(x)
        batch=out.shape[0]
        out=out.contiguous().view(batch,-1,self.out_channels) #making shape=(batch,4608,8)
        return squash(out)

class Digit_Caps(nn.Module):
    def __init__(self,in_dim, in_cap,out_cap, out_dim,num_routing):
        super(Digit_Caps,self).__init__()
        self.in_dim=in_dim
        self.in_cap=in_cap
        self.out_cap=out_cap
        self.out_dim=out_dim
        self.num_routing=num_routing
        self.device=device
        self.W= nn.Parameter(0.01*torch.rand(1,out_cap,in_cap,out_dim,in_dim),requires_grad=True)


    def forward(self, x):
        batch=x.shape[0]
        #prepare the pri_caps (batch, 4608, 8)--> (batch,1,4608,8,1)
        x=x.unsqueeze(1).unsqueeze(4)

        #apply matmul to the W prepared (1,10,4608,16,8)@ x (batch,1,4608,8,1)
        #out= (batch,10,4608,16,1)
        u_hat=torch.matmul(self.W,x)

        #to remove extra dimension (batch,10,4608,16)
        u_hat=u_hat.squeeze(-1)

        # to avoid gradient from flowing in dynamic routing
        temp_u_hat=u_hat.detach()

        #initialize zeros for b in shape(batch,10,4608,1)
        b=torch.zeros(batch,self.out_cap,self.in_cap,1).to(self.device)

        for _ in range(self.num_routing-1):
            #to softmax b along the 10 caps axis (batch,10,4608,1)
            c=b.softmax(dim=1)

            #(batch_size, 10, 4608, 1)*(batch_size, 10, 4608, 16) broadcasting--> (batch_size, 10, 4608, 16)
            #add along that 4608 axis--> (batch,10,16)
            s=(c * temp_u_hat).sum(dim=2)
            #making the 16D length becime unit vector--> (batch,10,16)
            v=squash(s)

            #get the routing agreement with dot(temp_u_hat, v)
            # (batch,10,4608,16) dot (batch,10,16)
            v=v.unsqueeze(-1) # become (batch,10,16,1)
            agree= torch.matmul(temp_u_hat,v) # become (batch,10,4608,1)

            #b<--b+agree  (batch,10,4608,1)+(batch,10,4608,1)
            b+=agree

        #last iterate done on original u_hat, so u_hat gradient only need to be updated once
        c = b.softmax(dim=1)
        s = (c * u_hat).sum(dim=2)
        v = squash(s) #

        return v #(batch,10,16)


class CapsNet(nn.Module):

    def __init__(self):
        super(CapsNet, self).__init__()

        self.layer1=nn.Sequential(
            nn.Conv2d(1,256,9,stride=1),
            nn.ReLU()
        )
        self.primary_caps=Primary_Caps(32,256,8,9,2)
        self.digit_caps=Digit_Caps(8,4608,10,16,3)

        self.decoder=nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1600),
            nn.Sigmoid()
        )

    def forward(self,x):
        out=self.layer1(x) #input (batch,1,40,40)--> (batch,256,32,32)
        out=self.primary_caps(out) #(batch 256,32,32)--> (batch,4608,8)
        out=self.digit_caps(out) #(batch,4608,8)--> (batch,10,16)

        logits=torch.norm(out,dim=-1) #become (batch,10)

        # to make a diagonal matrix and select the row according to maximum capsules activvated from logits
        pred= torch.eye(10).to(device).index_select(dim=0,index=torch.argmax(logits,dim=1)) #(batch,10)

        # Reconstruction
        batchsize=out.shape[0]
        filters=out*pred.unsqueeze(2) #to filter out the capsule to be reconstruct for each instance of the batch
        reconstruction=self.decoder(filters.contiguous().view(batchsize,-1)) # to flatten the matrix for FC

        return logits, reconstruction

class CapsuleLoss(nn.Module):
    """both margin loss and reconstruction loss"""
    def __init__(self, upper_bound=0.9, lower_bound=0.1, lmda=0.5):
        super(CapsuleLoss, self).__init__()
        self.upper = upper_bound
        self.lower = lower_bound
        self.lmda = lmda
        self.reconstruction_loss_scalar = 5e-4
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, images, labels, logits, reconstructions):
        # Shape of left / right / labels: (batch_size, num_classes)
        left = (self.upper - logits).relu() ** 2  # True negative
        right = (logits - self.lower).relu() ** 2  # False positive
        margin_loss = torch.sum(labels * left) + self.lmda * torch.sum((1 - labels) * right)

        # Reconstruction loss
        reconstruction_loss = self.mse(reconstructions.contiguous().view(images.shape), images)

        # Combine two losses
        return margin_loss + self.reconstruction_loss_scalar * reconstruction_loss

#data input
def load_expandMnist():
    transform = transforms.Compose([
        transforms.RandomCrop(40, padding=12),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    
    train_dataset = torchvision.datasets.MNIST(root='data',
                                               train=True,
                                               download=True,
                                               transform=transform)
    
    test_dataset = torchvision.datasets.MNIST(root='data',
                                              train=False,
                                              download=True,
                                              transform=transform)
    
    BATCH_SIZE = 128
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        shuffle=True)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        shuffle=True)
    
    return train_loader, test_loader

# trainer
def train (train_loader,Epoches):
    for ep in range(Epoches):
        batch_id=1
        correct, total, total_loss = 0, 0, 0.
        for images,labels in train_loader:
            images=images.to(device)
            labels = torch.eye(10).index_select(dim=0, index=labels).to(device) #for margin loss computation
            
            #forward pass
            logits, reconst=model(images) #logits:(batchsize,10) Reconst: (batchsize,1600)
            loss=criterion(images,labels,logits, reconst)
            
            #BP and Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #calculate accuracy and loss
            correct+= torch.sum(torch.argmax(logits,dim=1)==torch.argmax(labels,dim=1)).item()
            total+=len(labels)
            accuracy=correct/total
            total_loss+=loss
            if batch_id%100==0:
                print("Epoch:{},batch:{},loss:{},accuracy:{}".format(ep,batch_id,total_loss/batch_id,accuracy))
            
            batch_id+=1
        scheduler.step(ep)
        print('Total loss for epoch {}: {}'.format(ep + 1, total_loss))
        torch.save(model.state_dict(),'./affNISTcapsnet_ep_acc.pt')

def test(test_loader):
  correct,total,total_loss=0,0,0
  batchid=0
  batch_no=len(test_loader)
  with torch.no_grad():
    for images,labels in test_loader:
      images=images.to(device)
      labels = torch.eye(10).index_select(dim=0, index=labels).to(device)

      logits,reconst=model(images)
      loss=criterion(images,labels,logits, reconst)
      correct+=torch.sum(torch.argmax(logits,dim=1)==torch.argmax(labels,dim=1)).item()
      total+=len(labels)
      total_loss+=loss
      batchid+=1
      print("the progress:{}/{} /t accuracy for this batch:{}".format(batchid,batch_no,correct/total))

  print("Average test error:{},loss per batch:{} ".format(correct/total,total_loss/batch_no))

#build the affineNIST
class AffNistDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None):
        self.data=scio.loadmat(data_dir, squeeze_me=True)["affNISTdata"]
        self.img_labels = self.data.item()[5]
        self.images=np.reshape(self.data.item()[2].T,(-1,40,40))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def load_affNist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    testset= AffNistDataset("./test", transform=transform)
    batchsize=128
    test_loader=DataLoader(testset,batch_size=batchsize, shuffle=True, num_workers=4)
    return test_loader


if __name__=="__main__":
    # main running channel

    model = CapsNet().to(device)
    criterion = CapsuleLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

    train_loader, not_applicable=load_expandMnist()
    test_loader=load_affNist()

    if os.path.isfile("./affNISTcapsnet_ep_acc.pt")!=True:
      train(train_loader,30)
    else:
      model.load_state_dict(torch.load("./affNISTcapsnet_ep_acc.pt"))

    test(test_loader)


