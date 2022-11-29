

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pandas as pd

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
            #input shape(batch,256,28,28)
            #to further cnn to shape(batch,256,10,10)
            nn.Conv2d(in_channels, num_conv_units * out_channels, kernel_size, stride=stride),
            
        )
        self.out_channels=out_channels

    def forward(self,x):
        out=self.pre_flat(x)
        batch=out.shape[0]
        out=out.contiguous().view(batch,-1,self.out_channels) #making shape=(batch,3200,8)
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
        #prepare the pri_caps (batch, 3200, 8)--> (batch,1,3200,8,1)
        x=x.unsqueeze(1).unsqueeze(4)

        #apply matmul to the W prepared (1,10,3200,16,8)@ x (batch,1,3200,8,1)
        #out= (batch,10,3200,16,1)
        u_hat=torch.matmul(self.W,x)

        #to remove extra dimension (batch,10,3200,16)
        u_hat=u_hat.squeeze(-1)

        # to avoid gradient from flowing in dynamic routing
        temp_u_hat=u_hat.detach()

        #initialize zeros for b in shape(batch,10,3200,1)
        b=torch.zeros(batch,self.out_cap,self.in_cap,1).to(self.device)

        for _ in range(self.num_routing-1):
            #to softmax b along the 10 caps axis (batch,10,3200,1)
            c=b.softmax(dim=1)

            #(batch_size, 10, 3200, 1)*(batch_size, 10, 3200, 16) broadcasting--> (batch_size, 10, 3200, 16)
            #add along that 3200 axis--> (batch,10,16)
            s=(c * temp_u_hat).sum(dim=2)
            #making the 16D length becime unit vector--> (batch,10,16)
            v=squash(s)

            #get the routing agreement with dot(temp_u_hat, v)
            # (batch,10,3200,16) dot (batch,10,16)
            v=v.unsqueeze(-1) # become (batch,10,16,1)
            agree= torch.matmul(temp_u_hat,v) # become (batch,10,3200,1)

            #b<--b+agree  (batch,10,3200,1)+(batch,10,3200,1)
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
        self.digit_caps=Digit_Caps(8,3200,10,16,3)

        self.decoder=nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1296),
            nn.Sigmoid()
        )

    def forward(self,x):
        out=self.layer1(x) #input (batch,1,36,36)--> (batch,256,28,28)
        out=self.primary_caps(out) #(batch 256,28,28)--> (batch,3200,8)
        out=self.digit_caps(out) #(batch,3200,8)--> (batch,10,16)

        logits=torch.norm(out,dim=-1) #become (batch,10)

        # to make a diagonal matrix and select the row according to maximum capsules activated from logits
        _,largest_two_idx=torch.topk(logits,2,dim=1) # turple(values, idxes)--> (batchsize, 2)
        first=largest_two_idx[:,0]
        second=largest_two_idx[:,1]
        pred1= torch.eye(10).to(device).index_select(dim=0,index=first) #(batch,10,1)
        pred2= torch.eye(10).to(device).index_select(dim=0,index=second) #(batch,10,1)

        # Reconstruction
        batchsize=out.shape[0]

        #to reconstruct active capsules one at a time 
        filters1=out*pred1.unsqueeze(2) #to filter out the capsule to be reconstruct for each instance of the batch
        reconstruction1=self.decoder(filters1.contiguous().view(batchsize,-1)) # to flatten the matrix for FC shape(batch,1296)
        filters2=out*pred2.unsqueeze(2)
        reconstruction2=self.decoder(filters2.contiguous().view(batchsize,-1))
        # to combine 2 reconstruncted pixels using maximum
        reconstruction=torch.maximum(reconstruction1,reconstruction2)

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

# MultiMNIST Dataset Class Set up

class MultiMnistDataset(Dataset):
    def __init__(self, data_dir, train=True,transform=None, target_transform=None):
        if train:
              self.img_idx,self.label_idx=0,1
        else:
              self.img_idx,self.label_idx=2,3
        self.data=pd.read_pickle("./multi_mnist.pickle")
        self.img_labels = self.data[self.label_idx]
        self.images=self.data[self.img_idx]
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

# to load in MultiMnist

def load_multiMnist():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    
    train_data= MultiMnistDataset("./multi_mnist.pickle", transform=transform)
    test_data= MultiMnistDataset("./multi_mnist.pickle",train=False, transform=transform)
    batchsize=128
    train_loader=DataLoader(train_data,batch_size=batchsize, shuffle=True,num_workers=0)
    test_loader=DataLoader(test_data,batch_size=batchsize, shuffle=True,num_workers=0)
    return train_loader, test_loader

# trainer
def train (train_loader,Epoches):
    for ep in range(Epoches):
        batch_id=1
        correct, total, total_loss = 0, 0, 0.
        totalbatch=len(train_loader)
        for images,labels in train_loader:
            images=images.to(device)
            

            firstlabel=labels[:,0]
            secondlabel=labels[:,1]
            labels1 = torch.eye(10).index_select(dim=0, index=firstlabel).to(device) #for margin loss computation
            labels2 = torch.eye(10).index_select(dim=0, index=secondlabel).to(device)
            labels_in = labels1+labels2
            #labels_in.to(device)
            
            #forward pass
            logits, reconst=model(images) #logits:(batchsize,10) Reconst: (batch,1296)
            loss=criterion(images,labels_in,logits, reconst)
            
            #BP and Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #calculate accuracy and loss
            _,logits_indices= torch.topk(logits,2,dim=-1) #(val, indexes)
            logits_indices=torch.sort(logits_indices)[0] #(batch, 2)
            labels=torch.sort(labels)[0] #(batch,2)
            compare=torch.all(logits_indices==labels,dim=-1) #(batch)
            
            correct+= torch.sum(compare).item()
            total+=len(labels)
            accuracy=correct/total
            total_loss+=loss
            #if batch_id%100==0:
            print("Epoch:{},batch:{}/{},loss:{},accuracy:{}".format(ep,batch_id,totalbatch,total_loss/batch_id,accuracy))
            
            batch_id+=1
        scheduler.step(ep)
        print('Total loss for epoch {}: {}'.format(ep + 1, total_loss))
        torch.save(model.state_dict(),'./MultiNISTcapsnet_ep_acc.pt')

def test(test_loader):
    correct,total,total_loss=0,0,0
    batch_no=len(test_loader)
    batchid=1
    with torch.no_grad():
        for images,labels in test_loader:
            images=images.to(device)
            firstlabel=labels[:,0]
            secondlabel=labels[:,1]
            labels1 = torch.eye(10).index_select(dim=0, index=firstlabel).to(device) #for margin loss computation
            labels2 = torch.eye(10).index_select(dim=0, index=secondlabel).to(device)
            labels_in = labels1+labels2


            logits,reconst=model(images)
            loss=criterion(images,labels_in,logits, reconst)

            #calculate Accuracy
            _,logits_indices= torch.topk(logits,2,dim=-1) #(val, indexes)
            logits_indices=torch.sort(logits_indices)[0] #(batch, 2)
            labels=torch.sort(labels)[0] #(batch,2)
            compare=torch.all(logits_indices==labels,dim=-1) #(batch)

            correct+=torch.sum(compare).item()
            total+=len(labels)
            total_loss+=loss

            print("Progress:{}/{} \t test err per batch:{}, loss in this batch:{}".format(batchid,batch_no,1-(correct/total),loss))
            batchid += 1
    print("avg test error:{},loss per batch:{} ".format(1-(correct/total),total_loss/batch_no))

def sample_reconst():
    #to reconstruct images

    random= np.random.randint(0,128,size=1)
    sample_data=iter(test_loader).next()
    sample=sample_data[0][random]
    sample_label=sample_data[1][random]
    plt.imshow(sample.squeeze(),cmap="gray")
    plt.show()
    print("Inputs are {}".format(sample_label[0,:]))

    with torch.no_grad():
        logits,reconstruction=model(sample)
        to_reconst=reconstruction.view(sample.shape).squeeze()
        _,pred=torch.topk(logits,2,dim=-1)

        plt.imshow(to_reconst,cmap="gray")
        plt.title("The prediction are {}".format(pred.squeeze()))


if __name__ =="__main__":

    # main running channel
    model = CapsNet().to(device)
    criterion = CapsuleLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.096)

    train_loader, test_loader=load_multiMnist()
    if os.path.isfile("./MultiNISTcapsnet_ep_acc.pt")!=True:
        train(train_loader, 20)
    else:
        model.load_state_dict(torch.load("./MultiNISTcapsnet_ep_acc.pt"))

    test(test_loader)
    sample_reconst()




