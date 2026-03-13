import os
import torch
import torch.optim as optim
from torch.nn import functional as F
import time
import yaml

from src.datasets import *
from src.util import create_experiment_folder, save_img
from src import loss
from src import network

with open('config/config.yaml', 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)
hyper_p = config["hyper_p"]
paths = config["paths"]
lambdas = config["lambdas"]
opt_p = config["opt_p"]

checkpoint_save_path, img_save_path = create_experiment_folder(paths["checkpoint_save_path"], paths["img_out_path"])

start_epoch = 0

train_dataset = TrainDataset(image_root_dir = paths["train_image_path"], mask_root_dir=paths["eval_mask_path"], mode = "train")
val_dataset = TrainDataset(image_root_dir = paths["val_image_path"], mask_root_dir=paths["eval_mask_path"], mode = "val")

train_loader = DataLoader(train_dataset, batch_size=hyper_p["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=hyper_p["batch_size"], shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net_G = network.MADIN()
net_D = network.Discriminator(in_channels = 3)

L1loss = torch.nn.L1Loss()
GANloss = loss.AdversarialLoss(type = 'nsgan')
Perloss = loss.PerceptualLoss()
Styleloss = loss.StyleLoss()

basebatch=4
leanrng_rate_p = hyper_p["batch_size"]/basebatch
optimizer_G = optim.AdamW(net_G.parameters(), 
                        lr=leanrng_rate_p*opt_p["learning_rate_G"], betas=(opt_p["beta_1"], opt_p["beta_2"]), weight_decay= opt_p["weight_decay"])
optimizer_D = optim.AdamW(net_D.parameters(),
                        lr=leanrng_rate_p*opt_p["learning_rate_D"], betas=(opt_p["beta_1"], opt_p["beta_2"]), weight_decay= opt_p["weight_decay"])
print(f"basebatch:{basebatch}, currentbatch:{hyper_p['batch_size']},\nbaselearningrate:{opt_p['learning_rate_G']}, currentlearningrate:{leanrng_rate_p * opt_p['learning_rate_G']}")


net_G = net_G.to(device)
net_D = net_D.to(device)

def train(networks, criterion, optimizer, loader, epoch, device):
    
    generator = networks[0]
    discriminator = networks[1]

    generator.train()
    discriminator.train()
    
    L1loss = criterion[0] 
    GANloss = criterion[1] 
    Perloss = criterion[2] 
    Styloss = criterion[3]
    
    optimizer_G = optimizer[0]
    optimizer_D = optimizer[1]
    
    running_L1loss = 0.0
    running_GANloss = 0.0
    running_Perloss = 0.0
    running_Styloss = 0.0
    running_Total = 0.0
    runing_Dloss = 0.0
    
    done = 0
    
    for i, (image, mask) in enumerate(loader):
        image, mask = image.to(device), mask.to(device)
        
        optimizer_D.zero_grad()
        optimizer_G.zero_grad()
        g_image = generator(image*mask, mask)
        out_image = g_image * (1 - mask) + image * mask

        
        D_real, _ = discriminator(image)
        D_fake, _ = discriminator(g_image.detach())  
        
        current_Dloss = lambdas["lambda_GAN"] * ((GANloss(D_real, True, True) + GANloss(D_fake, False, True)) / 2 ) 
        
        current_Dloss.backward()
        optimizer_D.step()
        
        D_fake_for_G, _ = discriminator(g_image)
        
        current_GAN_loss = lambdas["lambda_GAN"] * GANloss(D_fake_for_G, True, False)
        current_L1loss = lambdas["lambda_L1"] * L1loss(image, g_image)
        current_Perloss = lambdas["lambda_Per"] * Perloss(g_image, image)
        current_Styleloss = lambdas["lambda_S"] * Styloss(image * (1 - mask), g_image * (1 - mask))
        
        current_Total = current_L1loss + current_Perloss + current_GAN_loss + current_Styleloss
        
        
        current_Total.backward()
        optimizer_G.step()
        
        running_L1loss += current_L1loss.item()
        running_GANloss += current_GAN_loss.item()
        running_Perloss  += current_Perloss.item()
        running_Styloss  += current_Styleloss.item()
        running_Total  += current_Total.item()
        runing_Dloss += current_Dloss.item()
    
        tmpdone = (i+1)/len(loader)*100
        if tmpdone > done + 10 :
            done = tmpdone
            print(f"batch done: {int(done)}%")
            save_img(out_image, image*mask, image, mask,g_image, img_save_path, epoch, "train", done, hyper_p['batch_size'], None)
    return [running_Total/len(loader), running_L1loss/len(loader), running_GANloss/len(loader), running_Perloss/len(loader), running_Styloss/len(loader), runing_Dloss/len(loader)]

def validation(networks, criterion, loader, epoch, device):
    generator = networks[0]
    discriminator = networks[1]
    
    generator.eval()
    discriminator.eval()

    L1loss = criterion[0] 
    GANloss = criterion[1] 
    Perloss = criterion[2]
    Styloss = criterion[3]
    
    running_L1loss = 0.0
    running_GANloss = 0.0
    running_Perloss = 0.0
    running_Styloss = 0.0

    running_Total = 0.0
    
    done = 0
    
    with torch.no_grad():
        for i, (image, mask) in enumerate(loader):
            image, mask = image.to(device), mask.to(device)
            
            g_image = generator(image*mask, mask)
            
            out_image = g_image*(1 - mask) + image*mask
            D_fake_for_G, _ = discriminator(out_image)
            
            current_GAN_loss = GANloss(D_fake_for_G, True, False)
            current_L1loss = L1loss(image, g_image)
            current_Perloss = Perloss(g_image, image)
            current_Styleloss = Styloss(image * (1 - mask), g_image * (1 - mask))
            
            current_Total = lambdas["lambda_L1"] * current_L1loss + lambdas["lambda_Per"] * current_Perloss + lambdas["lambda_GAN"] * current_GAN_loss + lambdas["lambda_S"] * current_Styleloss
        
            running_L1loss = running_L1loss + current_L1loss.item()
            running_GANloss = running_GANloss + current_GAN_loss.item()
            running_Perloss = running_Perloss + current_Perloss.item()
            running_Styloss = running_Styloss + current_Styleloss.item()
            running_Total = running_Total + current_Total.item()
            
            tmpdone = (i+1)/len(loader)*100
            if tmpdone > done + 10 :
                done = tmpdone
                save_img(out_image, image*mask, image, mask, g_image,  img_save_path, epoch, "val", done, hyper_p['batch_size'], None)
        return [running_Total/len(loader), running_L1loss/len(loader), running_GANloss/len(loader), running_Perloss/len(loader), running_Styloss/len(loader)]


#RUN
print(f"-----Start training!! Epoch: {hyper_p['num_epochs']}, Batch size: {hyper_p['batch_size']}-----")
best_valloss = np.inf

#test_save
torch.save(net_G.state_dict(), './model_weights.pth')

for epoch in range(start_epoch, hyper_p["num_epochs"]):
    start = time.time()
    print(f"Epoch: {epoch+1}/{hyper_p['num_epochs']}")
    
    with open(f"{checkpoint_save_path}/loss_log.txt", 'a') as log_file:
        log_file.write(f"Epoch: {epoch+1}/{hyper_p['num_epochs']}\n")
    
    train_loss = train([net_G, net_D], [L1loss, GANloss, Perloss, Styleloss], [optimizer_G, optimizer_D], train_loader, epoch, device)
    print(f"Train losses: Total: {train_loss[0]:.6f}, L1: {train_loss[1]:.6f}, Gan: {train_loss[2]:.6f}, Per: {train_loss[3]:.6f}, Sty: {train_loss[4]:.6f}, Dis: {train_loss[5]:.6f}")
    
    with open(f"{checkpoint_save_path}/loss_log.txt", 'a') as log_file: 
        log_file.write(f"Train losses: Total: {train_loss[0]:.6f}, L1: {train_loss[1]:.6f}, Gan: {train_loss[2]:.6f}, Per: {train_loss[3]:.6f}, Sty: {train_loss[4]:.6f}, Dis: {train_loss[5]:.6f}\n")
    
    if (epoch + 1) % 10 == 0:
        val_loss = validation([net_G, net_D], [L1loss, GANloss, Perloss, Styleloss], val_loader, epoch, device)
        print(f"Validation losses: Total: {val_loss[0]:.6f}, L1: {val_loss[1]:.6f}, GAN: {val_loss[2]:.6f}, Per: {val_loss[3]:.6f}, Sty: {val_loss[4]:.6f}")
        
        with open(f"{checkpoint_save_path}/loss_log.txt", 'a') as log_file: 
            log_file.write(f"Validation losses: Total: {val_loss[0]:.6f}, L1: {val_loss[1]:.6f}, GAN: {val_loss[2]:.6f}, Per: {val_loss[3]:.6f}, Sty: {val_loss[4]:.6f}\n")
        
    end = time.time() - start
    
    print(f"===============================train timecost: {end:.2f}===============================")

    checkpoint = {
        'epoch': epoch + 1,
        'net_G': net_G.module.state_dict() if isinstance(net_G, torch.nn.DataParallel) else net_G.state_dict(),  
        'net_D': net_D.module.state_dict() if isinstance(net_D, torch.nn.DataParallel) else net_D.state_dict(),  
        'optimizer_G': optimizer_G.state_dict(),
        'optimizer_D': optimizer_D.state_dict(),
        'loss': train_loss[0]
    }

    if (epoch+1) % 5 == 0:
        torch.save(checkpoint, f"{checkpoint_save_path}/epoch_{epoch+1}.pth")
            