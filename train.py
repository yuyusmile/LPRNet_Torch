import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import *
import os, cv2, time
import numpy as np
import argparse

from load_data import *
from LPR import *
from label_code import * 

def tupe_for_ctc(t_tengths, tength):
    
    input_length = []
    target_length = []

    for ch in tength:
        
        input_length.append(t_tengths)
        target_length.append(ch)
        
    return tuple(input_length), tuple(target_length)


def sparse_tensor_to_dense(output):
    
    pred_labels = list()
    labels = list()
    
    for i in range(output.shape[0]):
        pred = output[i, :, :]
        pred_label = list()
        
        for j in range(pred.shape[1]):
            pred_label.append(np.argmax(pred[:, j], axis=0))
        no_repeat_blank_label = list()
        # pre_c = ''
        pre_c = pred_label[0]
        
        for c in pred_label:
            if (pre_c == c) or (c == len(CHARS)):
                if c == len(CHARS):
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        
        pred_labels.append(no_repeat_blank_label)
    
    return pred_labels
        
    


parser = argparse.ArgumentParser(description='lrptrain')
parser.add_argument('--image_size', default=(94, 24), help='set img size')
parser.add_argument('--train_dir_path', default='./train', type=str, help='train data path')
parser.add_argument('--vail_dir_path', default='./valid', type=str, help='vail data path')
parser.add_argument('--epoch', default=2000, type=int, help='set lpr train epcho')
parser.add_argument('--batch_size', default=128, type=int, help='set batch size numbers')
args = parser.parse_args()


if __name__ == "__main__":
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
    num_chars = len(CHARS)
    lprnet = LPR(num_chars=num_chars)
    lprnet.to(device)
    
    dataset = {"train": Load_Date(args.train_dir_path, args.image_size),
               "vail": Load_Date(args.vail_dir_path, args.image_size)}
    dataload = {"train": DataLoader(dataset=dataset['train'], num_workers=4, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn),
                "vail": DataLoader(dataset=dataset['vail'], num_workers=4, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn),}
    
    opt = torch.optim.Adam([{"params": lprnet.parameters(), 'lr': 1e-3,}])
    ctc_loss = nn.CTCLoss(blank=len(CHARS) - 1, reduction='mean')
    
    train_logging_txt = './train_logging.txt'
    val_logging_txt = './val_logging.txt'
    save_dir = './save_ckpt'
    
    if os.path.exists(save_dir):
        raise NameError(" the dir exists ")
    os.makedirs(save_dir)
    
    start_time = time.time()
    total_iters = 0
    best_acc = 0
    T_length = 15
    print("training kicked off ..")
    print("-" * 10)
    
    for epoch in range(args.epoch):
        lprnet.train()
        since = time.time()
        for img, labels, lengths in dataload['train']:
            img, labels = img.to(device), labels.to(device)
            opt.zero_grad()
            
            with torch.set_grad_enabled(True):
                logits = lprnet(img)
                log_probs = logits.permute(2, 0, 1)
                log_probs = log_probs.log_softmax(2).requires_grad_()
                input_length, target_length = tupe_for_ctc(T_length, lengths)
                loss = ctc_loss(log_probs, labels, input_lengths=input_length, target_lengths=target_length)
                loss.backward()
                opt.step()
                total_iters += 1
                
                if total_iters % 100 == 0:
                    preds = logits.cpu().detach().numpy()
                    pred_labels = sparse_tensor_to_dense(preds)
                    total = preds.shape[0]
                    start = 0
                    TP = 0
                    best_iters = total_iters
                    
                    for i, label in enumerate(labels):
                        # print('pred_labels == >', np.array(pred_labels[i]))
                        # print('label       == >', np.array(label.cpu().numpy()))                        
                        if np.array_equal(np.array(pred_labels[i]), label.cpu().numpy()):
                            print('pred_labels == >', np.array(pred_labels[i]))
                            print('label       == >', np.array(label.cpu().numpy()))
                            TP += 1
                    print('-' * 10, TP)
                            
                    time_cur = (time.time() - since) / 100
                    since = time.time()
                    
                    for p in opt.param_groups:
                        lr = p['lr']
                    print("Epoch {}/{}, Iters: {:0>6d}, loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}"
                          .format(epoch, args.epoch-1, total_iters, loss.item(), TP/total, time_cur, lr))
                    
                    with open(train_logging_txt, 'a') as f:
                        f.write("Epoch {}/{}, Iters: {:0>6d}, loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}"
                            .format(epoch, args.epoch-1, total_iters, loss.item(), TP/total, time_cur, lr)+'\n')
                        
                if total_iters % 500 == 0:
                    
                    save_dirs = os.path.join(save_dir, "{0}-Epoch-{1}-Loss-{2}.pth".format("LPR", epoch, loss))
                    torch.save(lprnet.state_dict(), save_dirs, _use_new_zipfile_serialization=False)
                
                """if total_iters % 500 == 0:
                    
                    lprnet.eval()
                    
                    ACC = eval(lprnet, dataload['vail'], dataset['vail'], device)
                                
                    if best_acc <= ACC:
                        best_acc = ACC
                        best_iters = total_iters
                    
                    print("Epoch {}/{}, Iters: {:0>6d}, validation_accuracy: {:.4f}".format(epoch, args.epoch-1, total_iters, ACC))
                    with open(val_logging_txt, 'a') as f:
                        f.write("Epoch {}/{}, Iters: {:0>6d}, validation_accuracy: {:.4f}".format(epoch, args.epoch-1, total_iters, ACC)+'\n')
                    
                    
                    lprnet.train()"""
                    
    time_elapsed = time.time() - start_time  
    print('Finally Best Accuracy: {:.4f} in iters: {}'.format(best_acc, best_iters))
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
                            
                    
        
    
    
