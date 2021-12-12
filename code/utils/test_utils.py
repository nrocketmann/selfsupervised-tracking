import torch
import numpy as np
from matplotlib import cm
import cv2

import torch.nn.functional as F
import imageio

def vis_pose(oriImg, points):

    pa = np.zeros(15)
    pa[2] = 0
    pa[12] = 8
    pa[8] = 4
    pa[4] = 0
    pa[11] = 7
    pa[7] = 3
    pa[3] = 0
    pa[0] = 1
    pa[14] = 10
    pa[10] = 6
    pa[6] = 1
    pa[13] = 9
    pa[9] = 5
    pa[5] = 1

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170,0,255],[255,0,255]]
    canvas = oriImg
    stickwidth = 4
    x = points[0, :]
    y = points[1, :]

    for n in range(len(x)):
        pair_id = int(pa[n])

        x1 = int(x[pair_id])
        y1 = int(y[pair_id])
        x2 = int(x[n])
        y2 = int(y[n])

        if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:
            cv2.line(canvas, (x1, y1), (x2, y2), colors[n], 8)

    return canvas

def hard_prop(pred):
    pred_max = pred.max(axis=0)[0]
    pred[pred <  pred_max] = 0
    pred[pred >= pred_max] = 1
    pred /= pred.sum(0)[None]
    return pred

def process_pose(pred, lbl_set, topk=3):
    # generate the coordinates:
    pred = pred[..., 1:]
    flatlbls = pred.flatten(0,1)
    topk = min(flatlbls.shape[0], topk)
    
    vals, ids = torch.topk(flatlbls, k=topk, dim=0)
    vals /= vals.sum(0)[None]
    xx, yy = ids % pred.shape[1], ids // pred.shape[1]

    current_coord = torch.stack([(xx * vals).sum(0), (yy * vals).sum(0)], dim=0)
    current_coord[:, flatlbls.sum(0) == 0] = -1

    pred_val_sharp = np.zeros((*pred.shape[:2], 3))

    for t in range(len(lbl_set) - 1):
        x = int(current_coord[0, t])
        y = int(current_coord[1, t])

        if x >=0 and y >= 0:
            pred_val_sharp[y, x, :] = lbl_set[t + 1]

    return current_coord.cpu(), pred_val_sharp


def dump_predictions(pred, lbl_set, img, dustbins, prefix, vid_idx, t):
    '''
    Save:
        1. Predicted labels for evaluation
        2. Label heatmaps for visualization
    '''
    sz = img.shape[:-1]

    # Upsample predicted soft label maps
    # pred_dist = pred.copy()
    pred_dist = cv2.resize(pred, sz[::-1])[:]
    
    # Argmax to get the hard label for index
    pred_lbl = np.argmax(pred_dist, axis=-1)
    pred_lbl = np.array(lbl_set, dtype=np.int32)[pred_lbl]      
    img_with_label = np.float32(img) * 0.5 + np.float32(pred_lbl) * 0.5

    # Visualize label distribution for object 1 (debugging/analysis)
    pred_soft = pred_dist[..., 1]
    pred_soft = cv2.resize(pred_soft, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    pred_soft = cm.jet(pred_soft)[..., :3] * 255.0
    img_with_heatmap1 =  np.float32(img) * 0.5 + np.float32(pred_soft) * 0.5

    # Save blend image for visualization
    imageio.imwrite('%s_blend.jpg' % prefix, np.uint8(img_with_label))

    relevant_dustbin = dustbins[t,-1].cpu().numpy()
    resized_dustbin = cv2.resize(relevant_dustbin, (img.shape[1],img.shape[0]), interpolation=cv2.INTER_NEAREST)
    jetted_dustbin = cm.jet(resized_dustbin)[...,:3] * 255.0
    dustbin_heatmap = np.float32(img)*.5 + np.float32(jetted_dustbin) * .5

    imageio.imwrite('%s_dustbin.jpg' % prefix, np.uint8(dustbin_heatmap))

    if prefix[-4] != '.':  # Super HACK-y
        imname2 = prefix + '_mask.png'
        # skimage.io.imsave(imname2, np.uint8(pred_val))
    else:
        imname2 = prefix.replace('jpg','png')
        # pred_val = np.uint8(pred_val)
        # skimage.io.imsave(imname2.replace('jpg','png'), pred_val)

    # Save predicted labels for evaluation
    imageio.imwrite(imname2, np.uint8(pred_lbl))

    return img_with_label, pred_lbl, img_with_heatmap1


# class Propagation:
#     def __init__(self, net, )

def context_index_bank(n_context, long_mem, N):
    '''
    Construct bank of source frames indices, for each target frame
    '''
    ll = []   # "long term" context (i.e. first frame)
    for t in long_mem:
        assert 0 <= t < N, 'context frame out of bounds'
        idx = torch.zeros(N, 1).long()
        if t > 0:
            idx += t + (n_context+1)
            idx[:n_context+t+1] = 0
        ll.append(idx)
    # "short" context    
    ss = [(torch.arange(n_context)[None].repeat(N, 1) +  torch.arange(N)[:, None])[:, :]]

    return ll + ss


def mem_efficient_batched_affinity(query, key_indices, n_context, feats, dustbin_targets, mask, temperature, topk, device, feat_shape):
    '''
    Mini-batched computation of affinity, for memory efficiency
    '''
    vid_length, HW = feats.shape[-2:]
    bsize, pbsize = 2, 100 #keys.shape[2] // 2
    Ws, Is = [], []
    dustbins = []
    mask = mask.to(device)
    # keys: 1, C, vidlen, num context frames, HW
    # query: 1, C, vidlen, HW
    # dustbin targets: 1, C, vidlen
    for b in range(0, vid_length-n_context, bsize):
        keys = feats[:, :, key_indices[b:b+bsize]].to(device)
        _q = query[:, :, b:b+bsize].to(device)
        _d = dustbin_targets[:,:,b:b+bsize].unsqueeze(-1).to(device)

        dustbin_aff = torch.einsum('ijklm,ijkn->iklmn', keys,
                                   _d).squeeze(-1)  # shape 1, vidlen, context frames, HW
        _, shape1, shape2, shape3 = dustbin_aff.shape
        dustbin_aff = dustbin_aff.view(shape1, shape2, *feat_shape)
        dustbins.append(dustbin_aff)
        keys = feats[:, :, key_indices[b:b + bsize]].to(device)
        frame_weights = []
        frame_ind = []
        for frame in range(n_context + 1):
            _k = keys[:, :, :, frame, :]
            # _k shape 1,C,bsize, HW
            # _q shape 1,C,bsize, HW

            _q_new = _q
            A = torch.einsum('ijkm, ijkn->ikmn', _k, _q_new)
            if frame > 0:
                A += mask
            # A_big[:, :, frame, :, :] = A
            f_w, f_i = torch.topk(A, topk, dim=2)
            frame_weights.append(f_w)
            frame_ind.append(f_i + frame * A.shape[-2])

        frame_weights = torch.cat(frame_weights, dim=2)
        frame_ind = torch.cat(frame_ind, dim=2)

        final_w, final_i = torch.topk(frame_weights, topk, dim=2)
        ids = torch.gather(frame_ind, dim=2, index=final_i)
        final_w /= temperature
        final_w = F.softmax(final_w, dim=2)

        # A_big = A_big.view(1, A_big.shape[1], A_big.shape[2]*A_big.shape[3], A_big.shape[4])
        # A_big /= temperature
        # weights2, ids = torch.topk(A_big, topk, dim=-2)
        # final_w = F.softmax(weights2.cpu(), dim=-2)

        Ws += [w for w in final_w[0]]
        Is += [ii for ii in ids[0]]

    dustbins = torch.cat(dustbins,dim=0)
    mean_dustbin_affs = torch.mean(dustbins[:,-1],dim=-1)
    return Ws, Is, dustbins, mean_dustbin_affs

def batched_affinity(query, keys, mask, temperature, topk, long_mem, device):
    '''
    Mini-batched computation of affinity, for memory efficiency
    (less aggressively mini-batched)
    '''
    bsize = 2
    Ws, Is = [], []
    for b in range(0, keys.shape[2], bsize):
        _k, _q = keys[:, :, b:b+bsize].to(device), query[:, :, b:b+bsize].to(device)
        w_s, i_s = [], []

        A = torch.einsum('ijklmn,ijkop->iklmnop', _k, _q) / temperature
        
        # Mask
        A[0, :, len(long_mem):] += mask.to(device)

        _, N, T, h1w1, hw = A.shape
        A = A.view(N, T*h1w1, hw)
        A /= temperature

        weights, ids = torch.topk(A, topk, dim=-2)
        weights = F.softmax(weights, dim=-2)
            
        Ws += [w for w in weights]
        Is += [ii for ii in ids]

    
    return Ws, Is

def infer_downscale(model):    
    out = model(torch.zeros(1, 10, 3, 320, 320).to(next(model.parameters()).device), just_feats=True)
    scale = out[1].shape[-2:]
    return 320 // np.array(scale)

