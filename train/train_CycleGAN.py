from include.header import *
from models.Generator.CycleGAN_G import *
from models.Discriminator.CycleGAN_D import *
from utils.function.function import *
from utils.function.dataloader import *

def training_CycleGAN(size=[640,480],batch_size=8,gpu_num=[0]):
    data_path = '/home/eslab/dataset/Nect CT Png/none_enhanced_png/'
    data_list = os.listdir(data_path)
    data_list.sort()
    data_list  = [data_path + filename + '/' for filename in data_list]
    # print(size[0],size[1])
    save_path = '/home/eslab/kdy/git/Contrast_enhanced_GAN/saved_model/%d_%d_%d/'%(size[0],size[1],batch_size)
    save_img_path = '/home/eslab/kdy/git/Contrast_enhanced_GAN/saved_img/%d_%d_%d/'%(size[0],size[1],batch_size)
    
    logging_path = '/home/eslab/kdy/git/Contrast_enhanced_GAN/logging/'
    logging_filename = logging_path + 'CycleGAN_logging_%d_%d_%d.txt'%(size[0],size[1],batch_size)
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
    netG_N2E = CycleGAN_G(input_channels=1,output_channels=1,features=64,norm_layer='instance',kernel_size=3,dropout_p=0.,use_bias=False,padding_type='reflect',n_residual_blocks=9)
    netG_E2N = CycleGAN_G(input_channels=1, output_channels=1, features=64, norm_layer='instance', kernel_size=3,
                          dropout_p=0., use_bias=False, padding_type='reflect', n_residual_blocks=9)

    # Discriminator
    netD_N = CycleGAN_D(input_channels=1,norm_layer='instance',use_bias=False)
    netD_E = CycleGAN_D(input_channels=1,norm_layer='instance',use_bias=False)

    cuda = torch.cuda.is_available()
    print(f'gpu_num ==> {gpu_num}')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{gpu_num}'
    print(f'main gpu = {gpu_num[0]}')
    device = torch.device(f"cuda:{gpu_num[0]}" if torch.cuda.is_available() else "cpu")


    if cuda:
        netG_N2E.to(device)
        netG_E2N.to(device)
        netD_N.to(device)
        netD_E.to(device)

    if torch.cuda.device_count() > 1:
        if len(gpu_num) > 1:
            print('Multi GPU Activation !!!', torch.cuda.device_count())
            # model = nn.DataParallel(model,device_ids=gpu_num)
            netG_N2E = nn.DataParallel(netG_N2E,device_ids =gpu_num)
            netG_E2N = nn.DataParallel(netG_E2N,device_ids =gpu_num)
            netD_N = nn.DataParallel(netD_N,device_ids =gpu_num)
            netD_E = nn.DataParallel(netD_E,device_ids =gpu_num)

    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizer for Generators
    optimizer_G = torch.optim.Adam(itertools.chain(netG_N2E.parameters(), netG_E2N.parameters()),
                                   lr=lr, betas=(0.5, 0.999))
    # Optimizer for Discriminator
    optimizer_D_N = torch.optim.Adam(netD_N.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D_E = torch.optim.Adam(netD_E.parameters(), lr=lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, 0,
                                                                                       decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_N, lr_lambda=LambdaLR(n_epochs, 0,
                                                                                           decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_E, lr_lambda=LambdaLR(n_epochs, 0,
                                                                                           decay_epoch).step)

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    # input_A = Tensor(batch_size, 1, size, size)
    # input_B = Tensor(batch_size, 1, size, size)
    # target_real = Variable(Tensor(batch_size).fill_(1.0), requires_grad=False)
    # target_fake = Variable(Tensor(batch_size).fill_(0.0), requires_grad=False)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    transforms_train = [transforms.Resize((size[0],size[1]),Image.BICUBIC),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5),(0.5))]
    transforms_val = [transforms.Resize((size[0],size[1]),Image.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5),(0.5))]

    total_len = len(data_list)
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
        with tqdm(train_loader, desc='Train', unit='batch') as tepoch:
            for index, batch in enumerate(tepoch):
                real_n_enhance_image = batch['n_enhanced_image'].to(device)
                real_enhance_image = batch['enhanced_image'].to(device)
                target_real = Variable(Tensor(len(real_n_enhance_image)).fill_(1.0), requires_grad=False)
                target_fake = Variable(Tensor(len(real_n_enhance_image)).fill_(0.0), requires_grad=False)

                optimizer_G.zero_grad()

                # Identity Loss
                same_E = netG_N2E(real_enhance_image)
                loss_identity_E = criterion_identity(same_E,real_enhance_image)*5.0

                same_NE = netG_E2N(real_n_enhance_image)
                loss_identity_NE = criterion_identity(same_NE, real_n_enhance_image) * 5.0

                # GAN Loss
                fake_E = netG_N2E(real_n_enhance_image)
                pred_fake = netD_E(fake_E)
                loss_GAN_R2F = criterion_GAN(pred_fake,target_real)

                fake_NE = netG_E2N(real_enhance_image)
                pred_fake = netD_N(fake_NE)
                loss_GAN_F2R = criterion_GAN(pred_fake, target_real)

                # Cycle Loss
                recovered_E = netG_N2E(fake_NE)
                loss_cycle_E = criterion_cycle(recovered_E,real_enhance_image)*10.0

                recovered_N = netG_E2N(fake_E)
                loss_cycle_N = criterion_cycle(recovered_N, real_n_enhance_image) * 10.0

                # Total Loss
                loss_G = loss_identity_E + loss_identity_NE + loss_GAN_F2R + loss_GAN_R2F + loss_cycle_E + loss_cycle_N
                loss_G.backward()

                optimizer_G.step()

                # Discriminator N
                optimizer_D_N.zero_grad()

                # Real Loss
                pred_N = netD_N(real_n_enhance_image) # D_X(x)
                
                loss_D_real = criterion_GAN(pred_N,target_real) # Real None enhanced Image
                # fake_NE = F(y)
                # Fake Loss
                fake_NE = fake_A_buffer.push_and_pop(fake_NE)
                pred_fake = netD_N(fake_NE.detach())
                loss_D_fake = criterion_GAN(pred_fake,target_fake)

                # Total Discriminator N Loss
                loss_D_NE = (loss_D_real + loss_D_fake)*0.5

                loss_D_NE.backward()
                optimizer_D_N.step()

                # Discriminator E
                optimizer_D_E.zero_grad()

                # Real Loss
                pred_E = netD_E(real_enhance_image)
                loss_D_real = criterion_GAN(pred_E, target_real)

                # Fake Loss
                fake_E = fake_B_buffer.push_and_pop(fake_E)
                pred_fake = netD_E(fake_E.detach())
                loss_D_fake = criterion_GAN(pred_fake, target_fake)

                # Total Discriminator N Loss
                loss_D_E = (loss_D_real + loss_D_fake) * 0.5

                loss_D_E.backward()
                optimizer_D_E.step()
                total_loss_G += loss_G.item()
                total_loss_D_E = loss_D_E.item()
                total_loss_D_N = loss_D_NE.item()
                tepoch.set_postfix(loss_G = total_loss_G/(index+1), loss_D_NE = total_loss_D_N/(index+1), loss_D_E = total_loss_D_E/(index+1))
            
        output_str = 'train_dataset loss_G = %.3f loss_D_NE = %f loss_D_E = %f // '%(total_loss_G/(index+1),total_loss_D_N/(index+1),total_loss_D_E/(index+1))
        check_file.write(output_str)
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        total_loss_G = 0.

        with tqdm(val_loader, desc='Val', unit='batch') as tepoch:
            for index, batch in enumerate(tepoch):
                with torch.no_grad():
                    real_n_enhance_image = batch['n_enhanced_image'].to(device)
                    real_enhance_image = batch['enhanced_image'].to(device)
                    target_real = Variable(Tensor(len(real_n_enhance_image)).fill_(1.0), requires_grad=False)
                    target_fake = Variable(Tensor(len(real_n_enhance_image)).fill_(0.0), requires_grad=False)

                    image_path = batch['path']
                    # print(image_path)
                    same_E = netG_N2E(real_enhance_image)
                    loss_identity_E = criterion_identity(same_E, real_enhance_image) * 5.0

                    same_NE = netG_E2N(real_n_enhance_image)
                    loss_identity_NE = criterion_identity(same_NE, real_n_enhance_image) * 5.0

                    # GAN Loss
                    fake_E = netG_N2E(real_n_enhance_image) # our's target 
                    
                    # unorm = UnNormalize(mean=(0.5),std=(0.5))

                    for i in range(len(fake_E)):
                        current_save_img_path = save_img_path+'%d_epochs/'%(epoch+1) + image_path[i].split('/')[0] +'/'
                        current_filename = image_path[i].split('/')[-1]
                        os.makedirs(current_save_img_path,exist_ok=True)
                        
                        
                        save_file = fake_E[i].cpu().numpy()
                        save_file = (save_file * 0.5) + 0.5 # Unnormalize ( X * std ) + mean
                        #Norm ==> (X - mean) / std
                        save_file = np.transpose(save_file,(1,2,0))
                        save_file = save_file * 255.
                        save_file = np.where(save_file > 255, 255 , save_file)
                        save_file = np.where(save_file < 0, 0, save_file)
                        # print('file info : ',save_file)
                        # print('save_file shape => ',save_file.shape)
                        cv2.imwrite(current_save_img_path+current_filename,save_file)
                        

                    pred_fake = netD_E(fake_E)
                    loss_GAN_R2F = criterion_GAN(pred_fake, target_real)

                    fake_NE = netG_E2N(real_enhance_image)
                    pred_fake = netD_N(fake_NE)
                    loss_GAN_F2R = criterion_GAN(pred_fake, target_real)

                    # Cycle Loss
                    recovered_E = netG_N2E(fake_NE)
                    loss_cycle_E = criterion_cycle(recovered_E, real_enhance_image) * 10.0

                    recovered_N = netG_E2N(fake_E)
                    loss_cycle_N = criterion_cycle(recovered_N, real_n_enhance_image) * 10.0

                    # Total Loss
                    loss_G = loss_identity_E + loss_identity_NE + loss_GAN_F2R + loss_GAN_R2F + loss_cycle_E + loss_cycle_N
                    total_loss_G += loss_G.item()
                    tepoch.set_postfix(loss_G=total_loss_G/(index+1))
            total_loss_G = total_loss_G / (index + 1)
            output_str = 'val_dataset loss_G = %3.f // '%(total_loss_G)
            check_file.write(output_str)
            current_save_path = save_path + '/CycleGAN_%d_%d/iteration_%d/'%(size[0],size[1],epoch+1)
            os.makedirs(current_save_path,exist_ok=True)
            if len(gpu_num) > 1:
                torch.save(netG_N2E.module.state_dict(), current_save_path+'netG_N2E_%d_%d.pth'%(size[0],size[1]))
                torch.save(netG_E2N.module.state_dict(), current_save_path+'netG_E2N_%d_%d.pth'%(size[0],size[1]))
                torch.save(netD_N.module.state_dict(), current_save_path+'netD_N_%d_%d.pth'%(size[0],size[1]))
                torch.save(netD_E.module.state_dict(), current_save_path+'netD_E_%d_%d.pth'%(size[0],size[1]))
            else:
                torch.save(netG_N2E.state_dict(), current_save_path+'netG_N2E_%d_%d.pth'%(size[0],size[1]))
                torch.save(netG_E2N.state_dict(), current_save_path+'netG_E2N_%d_%d.pth'%(size[0],size[1]))
                torch.save(netD_N.state_dict(), current_save_path+'netD_N_%d_%d.pth'%(size[0],size[1]))
                torch.save(netD_E.state_dict(), current_save_path+'netD_E_%d_%d.pth'%(size[0],size[1]))
            

    check_file.close()

