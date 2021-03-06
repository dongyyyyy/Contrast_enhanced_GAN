from include.header import *
from models.Generator.UNet import *
from utils.function.function import *
from utils.function.dataloader import *
from models.perceptual.perceptual_loss import *

def training_pix2pix_withPerceptualLoss_withoutDiscriminator(size=[256,256],blocks=[64,128,256,512,1024],batch_size=128,gpu_num=[0]):
    data_path = '/home/eslab/dataset/Nect CT Png/none_enhanced_png/'
    data_list = os.listdir(data_path)
    data_list.sort()
    data_list  = [data_path + filename + '/' for filename in data_list]
    # print(size[0],size[1])
    save_path = '/data/hdd1/kdy/git/Contrast_enhanced_GAN/NoneDiscriminator/saved_model/pix2pix/instance/ReLU/perceptualLoss_withPixelsub_version2/%d_%d_%d_%d_%d_%d_%d_%d/'%(size[0],size[1],batch_size,blocks[0],blocks[1],blocks[2],blocks[3],blocks[4])
    save_img_path = '/data/hdd1/kdy/git/Contrast_enhanced_GAN/NoneDiscriminator/saved_img//pix2pix/instance/ReLU/perceptualLoss_withPixelsub_version2/%d_%d_%d_%d_%d_%d_%d_%d/'%(size[0],size[1],batch_size,blocks[0],blocks[1],blocks[2],blocks[3],blocks[4])
    
    logging_path = '/data/hdd1/kdy/git/Contrast_enhanced_GAN/NoneDiscriminator/logging/pix2pix/instance/ReLU/perceptualLoss_withPixelsub_version2/'
    logging_filename = logging_path + 'pix2pix_instance_perceptualLoss_withPixel_sub_version2_logging_%d_%d_%d_%d_%d_%d_%d_%d.txt'%(size[0],size[1],batch_size,blocks[0],blocks[1],blocks[2],blocks[3],blocks[4])
    os.makedirs(save_img_path,exist_ok=True)
    os.makedirs(logging_path,exist_ok=True)
    os.makedirs(save_path,exist_ok=True)

    check_file = open(logging_filename, 'w')  # logging file
    n_epochs = 200
    decay_epoch = 100
    # batch_size = 8
    lr = 0.0002
    n_cpu = multiprocessing.cpu_count() // 2

    # Generator
    netG = UNet(in_channels=1,out_channels=1,blocks=blocks,kernel_size=3,norm_layer='instance', activation_func='ReLU',use_bias=False,bilinear=False,scale=2)

    VGG16 = VGG16_perceptual()

    cuda = torch.cuda.is_available()
    print(f'gpu_num ==> {gpu_num}')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{gpu_num}'
    print(f'main gpu = {gpu_num[0]}')
    device = torch.device(f"cuda:{gpu_num[0]}" if torch.cuda.is_available() else "cpu")


    if cuda:
        netG.to(device)
        
        VGG16.to(device)
    if torch.cuda.device_count() > 1:
        if len(gpu_num) > 1:
            print('Multi GPU Activation !!!', torch.cuda.device_count())
            # model = nn.DataParallel(model,device_ids=gpu_num)
            netG = nn.DataParallel(netG,device_ids =gpu_num)
            
            VGG16 = nn.DataParallel(VGG16,device_ids=gpu_num)


    criterion_GAN = torch.nn.MSELoss()
    # criterion_GAN = nn.BCELoss()
    criterion_identity = torch.nn.L1Loss()
    perceptual_ratio = 5
    lambda_ratio = 5

    patch_size = (1,size[0]//(2**4),size[1]//(2**4))

    # Optimizer for Generators
    optimizer_G = torch.optim.Adam(netG.parameters(),
                                   lr=lr, betas=(0.5, 0.999))
    

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, 0,
                                                                                       decay_epoch).step)
    


    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    # input_A = Tensor(batch_size, 1, size, size)
    # input_B = Tensor(batch_size, 1, size, size)
    # target_real = Variable(Tensor(batch_size).fill_(1.0), requires_grad=False)
    # target_fake = Variable(Tensor(batch_size).fill_(0.0), requires_grad=False)


    transforms_train = [transforms.Resize((size[0],size[1]),Image.BICUBIC),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5),(0.5))]
    transforms_val = [transforms.Resize((size[0],size[1]),Image.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5),(0.5))]

    train_split = int(len(data_list) * 0.9)
    train_list = data_list[:train_split]
    val_list = data_list[train_split:]

    train_dataset = ImageDataset(dataset_list=train_list,transforms_=transforms_train,mode='train')
    val_dataset = ImageDataset(dataset_list = val_list,transforms_=transforms_val,mode='validation')

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True, num_workers=n_cpu)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=n_cpu)

    Best_loss_G = 0
    best_epoch = 0
    for epoch in range(0,n_epochs):
        total_loss_G = 0.
        total_loss_D_N = 0.
        total_loss_D_E = 0.
        netG.train()
        VGG16.eval()
        with tqdm(train_loader, desc='Train', unit='batch') as tepoch:
            for index, batch in enumerate(tepoch):
                real_n_enhance_image = batch['n_enhanced_image'].to(device)
                real_enhance_image = batch['enhanced_image'].to(device)

                # ????????? ???????????????
                # real_target_enhance_image = real_enhance_image - real_n_enhance_image

                optimizer_G.zero_grad()
                fake_img = netG(real_n_enhance_image)
                
                fake_img = fake_img + real_n_enhance_image
                # print(d_out.shape)
                # print(target_real.shape)
                pixel_loss = criterion_identity(fake_img,real_enhance_image)
                # pixel_loss = criterion_identity(fake_img,real_enhance_image)
                with torch.no_grad():
                    _,_,_,p_fake = VGG16(fake_img)
                    _,_,_,p_real = VGG16(real_enhance_image)

                # print(p_fake,p_real)
                perceptual_loss = torch.nn.functional.l1_loss(p_fake, p_real)
                loss_G = lambda_ratio*pixel_loss+ perceptual_ratio * perceptual_loss

                loss_G.backward()
                optimizer_G.step()

                total_loss_G += loss_G.item()
                
                tepoch.set_postfix(epoch=f'{epoch}//{n_epochs}',loss_G = total_loss_G/(index+1), )
            
        output_str = f'train_dataset {epoch}//{n_epochs} loss_G = %.3f // '%(total_loss_G/(index+1))
        check_file.write(output_str)
        lr_scheduler_G.step()
        

        total_loss_G = 0.

        netG.eval()
        VGG16.eval()

        with tqdm(val_loader, desc='Val', unit='batch') as tepoch:
            for index, batch in enumerate(tepoch):
                with torch.no_grad():
                    real_n_enhance_image = batch['n_enhanced_image'].to(device)
                    real_enhance_image = batch['enhanced_image'].to(device)
                    
                    # real_target_enhance_image = real_enhance_image - real_n_enhance_image

                    fake_img = netG(real_n_enhance_image)
                
                    fake_img = fake_img + real_n_enhance_image
                    image_path = batch['path']
                    # unorm = UnNormalize(mean=(0.5),std=(0.5))

                    for i in range(len(fake_img)):
                        current_save_img_path = save_img_path+'%d_epochs/'%(epoch+1) + image_path[i].split('/')[0] +'/'
                        current_filename = image_path[i].split('/')[-1]
                        os.makedirs(current_save_img_path,exist_ok=True)
                        
                        
                        save_file = fake_img[i].cpu().numpy()
                        save_file = (save_file * 0.5) + 0.5 # Unnormalize ( X * std ) + mean
                        #Norm ==> (X - mean) / std
                        save_file = np.transpose(save_file,(1,2,0))
                        save_file = save_file * 255.
                        save_file = np.where(save_file > 255, 255 , save_file)
                        save_file = np.where(save_file < 0, 0, save_file)
                        # print('file info : ',save_file)
                        # print('save_file shape => ',save_file.shape)
                        cv2.imwrite(current_save_img_path+current_filename,save_file)
                        

                    pixel_loss = criterion_identity(fake_img,real_enhance_image)
                    # pixel_loss = criterion_identity(fake_img,real_enhance_image)
                    
                    # loss_G = gan_loss + lambda_ratio*pixel_loss
                    _,_,_,p_fake = VGG16(fake_img)
                    _,_,_,p_real = VGG16(real_enhance_image)

                    # print(p_fake,p_real)
                    perceptual_loss = torch.nn.functional.l1_loss(p_fake, p_real)
                    loss_G = lambda_ratio*pixel_loss+ perceptual_ratio * perceptual_loss

                    # Total Loss
                    total_loss_G += loss_G.item()
                    tepoch.set_postfix(epoch=f'{epoch}//{n_epochs}',loss_G=total_loss_G/(index+1))
            total_loss_G = total_loss_G / (index + 1)
            output_str = f'val_dataset {epoch}//{n_epochs} loss_G = %3.f // '%(total_loss_G)
            check_file.write(output_str)
            current_save_path = save_path + '/pix2pix_%d_%d/iteration_%d/'%(size[0],size[1],epoch+1)
            os.makedirs(current_save_path,exist_ok=True)
            if len(gpu_num) > 1:
                torch.save(netG.module.state_dict(), current_save_path+'netG_%d_%d.pth'%(size[0],size[1]))
            else:
                torch.save(netG.state_dict(), current_save_path+'netG_%d_%d.pth'%(size[0],size[1]))
                
            

    check_file.close()

