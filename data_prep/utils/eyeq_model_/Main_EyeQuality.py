import os
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import time
from progress.bar import Bar
import torchvision.transforms as transforms
from dataloader.EyeQ_loader import DatasetGenerator
from utils.trainer import train_step, validation_step, save_output
from utils.metric import compute_metric

import pandas as pd
from networks.densenet_mcf import dense121_mcs
import shutil

from tqdm import tqdm

def main():    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(0)

    data_root = '../Kaggle_DR_dataset/'

    # Setting parameters
    parser = argparse.ArgumentParser(description='EyeQ_dense121')
    parser.add_argument('--model_dir', type=str, default='./result/')
    parser.add_argument('--pre_model', type=str, default='DenseNet121_v3_v1')
    parser.add_argument('--save_model', type=str, default='DenseNet121_v3_v1')

    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--label_idx', type=list, default=['Good', 'Usable', 'Reject'])

    parser.add_argument('--n_classes', type=int, default=3)
    # Optimization options
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--loss_w', default=[0.1, 0.1, 0.1, 0.1, 0.6], type=list)

    args = parser.parse_args()

    # Images Labels
    test_images_dir = data_root + '/test'
    label_test_file = '../data/Label_EyeQ_test.csv'

    save_file_name = args.model_dir + args.save_model + '1.csv'

    best_metric = np.inf
    best_iter = 0
    # options
    cudnn.benchmark = True

    model = dense121_mcs(n_class=args.n_classes)

    if args.pre_model is not None:
        loaded_model = torch.load(os.path.join(args.model_dir, args.pre_model + '.tar'))
        model.load_state_dict(loaded_model['state_dict'])

    model.to(device)

    criterion = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    transform_list1 = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(-180, +180)),
        ])

    transformList2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])

    transform_list_val1 = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
        ])



    data_test = DatasetGenerator(data_dir=test_images_dir, list_file=label_test_file, transform1=transform_list_val1,
                                transform2=transformList2, n_class=args.n_classes, set_name='test')
    test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=args.batch_size,
                                            shuffle=False, num_workers=0, pin_memory=True)


    # Testing
    outPRED_mcs = torch.FloatTensor().cuda()
    img_name_mcs = []
    model.eval()
    iters_per_epoch = len(test_loader)
    bar = Bar('Processing {}'.format('inference'), max=len(test_loader))
    bar.check_tty = False
    for epochID, (imagesA, imagesB, imagesC, image_name) in enumerate(test_loader):
        imagesA = imagesA.cuda()
        imagesB = imagesB.cuda()
        imagesC = imagesC.cuda()

        begin_time = time.time()
        _, _, _, _, result_mcs = model(imagesA, imagesB, imagesC)
        outPRED_mcs = torch.cat((outPRED_mcs, result_mcs.data), 0)
        for i in image_name:
            img_name_mcs.append(i)

        batch_time = time.time() - begin_time
        bar.suffix = '{} / {} | Time: {batch_time:.4f}'.format(epochID + 1, len(test_loader),
                                                            batch_time=batch_time * (iters_per_epoch - epochID) / 60)
        bar.next()
    bar.finish()

    # save result into excel:
    pred= torch.argmax(outPRED_mcs,1).cpu().numpy()
    img_name_mcs= np.asarray(img_name_mcs)
    print( torch.argmax(outPRED_mcs,1) )
    print(img_name_mcs)
    print(img_name_mcs[pred==0])
    
    #copy files good images
    path=os.getcwd()+r'\result\new_dataset'

    try:
        os.mkdir(path)
    except OSError as error:
        print(error)

    goods = img_name_mcs[pred!=2]
    for img in tqdm( goods ):
        original_image_path = img
        shutil.copy2(original_image_path, path)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    
    main()