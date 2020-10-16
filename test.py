import os
from PIL import Image
import torch
from torchvision import transforms
from model3.model2_graph4_hrnet_sal import Model2
import setproctitle
import copy

def test(gpu_id, model_path, imgroot, datapath, if_save, save_root_path, group_size, img_size, img_dir_name, img_type, gt_exten):
    net = Model2().cuda()
    #net.load_state_dict(torch.load(model_path, map_location=gpu_id))
    pretrained_dict=torch.load(model_path)
    net_dict=net.state_dict()
    for key,value in pretrained_dict.items():
              net_dict[key[7:]]=value
              net_dict.update(net_dict)
              net.load_state_dict(net_dict)
    net.eval()
    ####### transformation
    img_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img_transform_gray = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.449], std=[0.226])])
    with torch.no_grad():
        for p in range(len(datapath)):
            for root, dirs, files in os.walk(os.path.join(imgroot, datapath[p], img_dir_name)):
                if files == []:
                    continue
                else:
                    rgb = torch.zeros(group_size, 3, img_size, img_size)
                    gt = torch.zeros(group_size, img_size, img_size)
                    cur_class_name = root.split('/')[-1]
                    cur_class_img_num = len(files)
                    print(cur_class_name)
                    print(cur_class_img_num)
#                    print(files)
                    cur_folderpath=os.path.join(save_root_path[p], cur_class_name)
                    if not os.path.exists(cur_folderpath):
                       os.makedirs(cur_folderpath)
                    f = open(os.path.join(imgroot, datapath[p], cur_class_name + '.txt'), 'r')
                    lines_str = f.readlines()
                    class_mask = torch.zeros(cur_class_img_num, img_size, img_size)
                    all_img_name=[]
                    for k in range(len(lines_str)):
                        line_str = lines_str[k].strip('\n').split(' ')
                        for i in range(len(line_str)):               ####### 24
                            cur_img_path = os.path.join(imgroot, datapath[p], img_dir_name, cur_class_name, line_str[i]+ img_type[p] )
                            all_img_name.append(line_str[i])
                              
                            rgb_ = Image.open(cur_img_path)
                            if rgb_.mode == 'RGB':
                                rgb_ = img_transform(rgb_)
                            else:
                                rgb_ = img_transform_gray(rgb_)
                            rgb[i,:,:,:] = rgb_
                            
                        #rgb = rgb.to(device)
                        rgb=rgb.cuda()
                        #pred_mask = net(rgb, 'error')
                        salmap, cosalmap = net(rgb)
                        salmap = salmap.cpu().sigmoid().squeeze()
                        cosalmap = cosalmap.cpu()
                        if datapath[p]=='iCoseg':
                          pred_mask=cosalmap*salmap
                        else:
                          pred_mask=cosalmap
                        pred_mask = (pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min() + 1e-8)
                        if k == len(lines_str)-1:
                            rest_img_num = cur_class_img_num - k * group_size
                            class_mask[k*group_size : k*group_size+rest_img_num, :, :] = pred_mask[0:rest_img_num, :, :]
                        else:
                            class_mask[k * group_size: (k + 1) * group_size, :, :] = pred_mask
                    ######## save results
                    if if_save:
                       len5=class_mask.size()[0]
                       for j in range(len5):
                           cur_img_name=all_img_name[j]
                           cur_img_path=os.path.join(cur_folderpath, cur_img_name+gt_exten[p])
                           #print( Image.open(os.path.join(datapath[p], img_dir_name, cur_class_name, cur_img_name+img_type[p])).size)
                           w,h = Image.open(os.path.join(imgroot, datapath[p], img_dir_name, cur_class_name, cur_img_name+img_type[p])).size
                           result=class_mask[j,:,:].numpy()
                           result=Image.fromarray(result*255)
                           result=result.resize((w,h), Image.BILINEAR)
                           result.convert('L').save(cur_img_path)
                           
        print('done')

if __name__ == '__main__':
    val_root='/home/litengpeng/cosal/dataset'
    val_datapath=['iCoseg',  'Cosal2015', 'CoSOD3k']
#    val_datapath=['/home/litengpeng/CODE/cosal/dataset/cosal']
#    val_datapath=['/home/litengpeng/CODE/cosal/dataset/coco_val/']
    gpu_id = 'cuda:0'
    model_path= '/home/litengpeng/cosal/CVPR2020-source-codes-py3/trained_models/stage3/hrnet_stage3_iter45000.pth'
    save_root_path=['./results/res-mine/hrnet_stage3_iter45000/iCoseg/',  './results/res-mine/hrnet_stage3_iter45000/Cosal2015/','./results/res-mine/hrnet_stage3_iter45000/CoSOD3k/']
    if not os.path.exists(save_root_path[0]):
       os.makedirs(save_root_path[0])
    test(gpu_id, model_path, val_root, val_datapath, True, save_root_path, 5, 224, 'image', ['.jpg', '.jpg', '.jpg'], ['.png', '.png', '.png'])
#    test(gpu_id, model_path, val_datapath, True, save_root_path, 5, 224, 'image', ['.jpg', '.jpg', '.jpg'], ['.png', '.png', '.png'])
