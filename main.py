import numpy as np
import torch
from torchvision import models, datasets
from torchvision import transforms as T
import cv2
import torch.nn.functional as F
from utils import calculate_outputs_and_gradients, generate_entrie_images
from integrated_gradients import random_baseline_integrated_gradients
from visualization import visualize
import matplotlib.pyplot as plt
import argparse
import os


from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models import resnet18
from torchvision import transforms as T
import matplotlib.pyplot as plt
     
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='integrated-gradients')
parser.add_argument('--cuda', action='store_true', help='if use the cuda to do the accelartion')
parser.add_argument('--model-type', type=str, default='inception', help='the type of network')
parser.add_argument('--img', type=str, default='01.jpg', help='the images name')

def fgsm(model, x, target, eps, targeted=True):
    finalout = []
    L = nn.CrossEntropyLoss()

    x.requires_grad_()
    loss = L(model(x), target.clone().detach())
    loss.backward()

    # take the largest possible step in the direction of the gradient
    finalout = torch.sign(x.grad)

    if targeted:
        finalout *= -1

    return (x + (eps * finalout)).detach()

# Ensure we remain within the L-inf ball of the original image.
# This is not needed for FGSM as it only takes a single step, and the magnitude of the step is at most epsilon.
def clip(x, x_baseline, eps):
    x = torch.minimum(x, x_baseline + eps)
    x = torch.maximum(x, x_baseline - eps)
    return x

def pgd_untargeted(model, x, labels, eps, targeted=False, k=5):
    # No targeted version was implemented, since it is not needed to make the model robust
    fgsm_in = x.clone().detach()
    fgsm_in.requires_grad_()
    for i in range(k):
      # Apply FGSM over k iterations to see if we can get a more adversarial image
      perturb = fgsm(model, fgsm_in, labels, eps, False)
      fgsm_in = clip(perturb, x, eps)
    return fgsm_in

def evaluate(model, data_loader):
    """Evaluate the model on the given dataset."""
    # Set the model to evaluation mode.
    model.eval()
    correct = 0
    # The `torch.no_grad()` context will turn off gradients for efficiency.
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            pred = output.argmax(dim=1)
            correct += (pred == labels).sum().item()
    return correct / len(data_loader.dataset)

cifar_10_train = torchvision.datasets.CIFAR10("CIFAR10", train=True, transform=T.ToTensor(), download=True)
cifar_10_test = torchvision.datasets.CIFAR10("CIFAR10", train=False, transform=T.ToTensor(), download=True)

batch_size = 64
train_loader = DataLoader(cifar_10_train, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(cifar_10_test, batch_size=batch_size, shuffle=False)
sample_img, sample_lbl = next(iter(train_loader))

def train(model, n_epoch, optimizer, scheduler, augment=False, attack_fn=None, eps=0.01):
    """Train the model on the given dataset."""
    loss_fn = nn.CrossEntropyLoss()

    loader = train_loader
    if augment:
      loader = train_loader_augmented

    for epoch in range(n_epoch):
        # Set the model to training mode.
        model.train()
        for step, (images, labels) in enumerate(loader):
            # 0. Prepare the data. Move the data to the device (CPU/GPU).
            images, labels = images.to(device), labels.to(device)
            # 1. Clear previous gradients.
            optimizer.zero_grad()

            # Train with defense against adversarial attacks
            # You can ignore this for now, it will be explained later in the notebook
            if attack_fn is not None:
                # Loss on the FGSM/PGD example
                model.eval()
                images = attack_fn(model, images, labels, eps, False)
                model.train()

            # 2. Forward pass. Calculate the output of the model.
            output = model(images)
            # 3. Calculate the loss.
            loss = loss_fn(output, labels)
            # 4. Calculate the gradients. PyTorch does this for us!
            loss.backward()
            # 5. Update the model parameters.
            optimizer.step()
            if step % 700 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss {loss.item():.4f}")
        # 6. (Optional) Update the learning rate.
        scheduler.step()
        acc = evaluate(model, valid_loader)
        print(f"Epoch {epoch}, Valid Accuracy {acc * 100:.2f}%")

        acc = evaluate(model, train_loader)
        print(f"Epoch {epoch}, Train Accuracy {acc * 100:.2f}%")

if __name__ == '__main__':
    args = parser.parse_args()
    # check if have the space to save the results
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if not os.path.exists('results/' + args.model_type):
        os.mkdir('results/' + args.model_type)
    
    # start to create models...
    if args.model_type == 'inception':
        model = models.inception_v3(pretrained=True)
    elif args.model_type == 'resnet152':
        model = models.resnet152(pretrained=True)
    elif args.model_type == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif args.model_type == 'resnet18-adv':
        model = models.resnet18(pretrained=True).to(device)
        lr = 1e-5
        gamma = 0.9
        num_epoch = 10
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
        
        train(model, 10, optimizer, scheduler, attack_fn=pgd_untargeted, eps=0.01)
    elif args.model_type == 'vgg19':
        model = models.vgg19_bn(pretrained=True)
    model.eval()
    if args.cuda:
        model.cuda()
    # read the image
    img = cv2.imread('examples/' + args.img)
    if args.model_type == 'inception':
        # the input image's size is different
        img = cv2.resize(img, (299, 299))
    img = img.astype(np.float32) 
    img = img[:, :, (2, 1, 0)]
    # calculate the gradient and the label index
    gradients, label_index = calculate_outputs_and_gradients([img], model, None, args.cuda)
    gradients = np.transpose(gradients[0], (1, 2, 0))
    img_gradient_overlay = visualize(gradients, img, clip_above_percentile=99, clip_below_percentile=0, overlay=True, mask_mode=True)
    img_gradient = visualize(gradients, img, clip_above_percentile=99, clip_below_percentile=0, overlay=False)

    # calculae the integrated gradients 
    attributions = random_baseline_integrated_gradients(img, model, label_index, calculate_outputs_and_gradients, \
                                                        steps=50, num_random_trials=10, cuda=args.cuda)
    img_integrated_gradient_overlay = visualize(attributions, img, clip_above_percentile=99, clip_below_percentile=0, \
                                                overlay=True, mask_mode=True)
    img_integrated_gradient = visualize(attributions, img, clip_above_percentile=99, clip_below_percentile=0, overlay=False)
    output_img = generate_entrie_images(img, img_gradient, img_gradient_overlay, img_integrated_gradient, \
                                        img_integrated_gradient_overlay)
    
    print('results/' + args.model_type + '/' + args.img)
    print('imshow6')
    plt.imshow(np.uint8(output_img))
    plt.axis('off')
    plt.show()
    #cv2.imshow("Image", np.uint8(output_img))
    cv2.imwrite('results/' + args.model_type + '/' + args.img, np.uint8(output_img))
