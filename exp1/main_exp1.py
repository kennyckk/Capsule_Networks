import torch
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader
from Capsnet import CapsNet, CapsuleLoss

device=torch.device('cuda' if torch.cuda.is_available() else "cpu")


# data input
def load_Mnist():
    transform = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
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
def train(train_loader, Epoches):
    for ep in range(Epoches):
        batch_id = 1
        correct, total, total_loss = 0, 0, 0.
        for images, labels in train_loader:
            images = images.to(device)
            labels = torch.eye(10).index_select(dim=0, index=labels).to(device)  # for margin loss computation

            # forward pass
            logits, reconst = model(images)  # logits:(batchsize,10) Reconst: (784)
            loss = criterion(images, labels, logits, reconst)

            # BP and Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate accuracy and loss
            correct += torch.sum(torch.argmax(logits, dim=1) == torch.argmax(labels, dim=1)).item()
            total += len(labels)
            accuracy = correct / total
            total_loss += loss
            if batch_id % 100 == 0:
                print("Epoch:{},batch:{},loss:{},accuracy:{}".format(ep, batch_id, total_loss / batch_id, accuracy))

            batch_id += 1
        scheduler.step(ep)
        print('Total loss for epoch {}: {}'.format(ep + 1, total_loss))

def test(test_loader):
  correct,total,total_loss=0,0,0
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

  print("test error:{},loss per batch:{} ".format(1-(correct/total),total_loss/batch_no))

if __name__=="__main__":
    # main running channel
    model = CapsNet().to(device)
    criterion = CapsuleLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

    train_loader, test_loader = load_Mnist()

    if os.path.isfile("./capsnet_ep50_acc0.5.pt")!=True:
        # train the model for 50 epoches if parameters file not found
        train(train_loader, 50)
        torch.save(model.state_dict(), './capsnet_ep{}_acc{}.pt'.format(50, 0.5))
    else:
        model.load_state_dict(torch.load("./capsnet_ep50_acc0.5.pt"))

    test(test_loader)
