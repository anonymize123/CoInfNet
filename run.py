from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import torch.nn.functional as F
import numpy as np
import torch
import torch.optim as optim
import os.path as osp
from tqdm import trange


@torch.no_grad()
def evaluate(epoch, data, model, device, args, thr=None, return_best_thr=False,
             log_desc='valid'):

    x = data.x
    edge_index = data.edge_index
    offline_days = data.offline_days
    class_weight = torch.FloatTensor(data.class_weight).to(device)
    if log_desc == 'valid':
        loader = data.valid_loader
    else:
        loader = data.test_loader

    model.eval()
    loss = 0.
    invite_loss = 0.
    edge_loss = 0.
    total = 0.
    invite_total = 0.
    edge_total = 0.

    y_true, y_pred, y_score, pred_ys = [], [], [], []

    pbar = tqdm(loader)
    for batch in pbar:

        x_batch = x.to(device)
        edge_index_batch = edge_index.to(device)
        invitations = batch[0].to(device)
        offline_days_batch = offline_days.to(device)

        y_hat, edge_y_hat = model(x_batch, edge_index_batch, invitations, offline_days_batch)
        y_hat = F.log_softmax(y_hat, dim=-1)

        if not args.check_edge:
            train_loss = F.nll_loss(y_hat, batch[1].to(device), class_weight)
            loss += float(train_loss) * y_hat.size(0)
            total += y_hat.size(0)
        else:
            edge_y_hat = F.log_softmax(edge_y_hat, dim=-1)
            loss_1 = F.nll_loss(y_hat, batch[1].to(device), class_weight)
            loss_2 = calculate_edge_loss(edge_y_hat, batch[2].to(device))
            train_loss = loss_1 + args.loss_alpha * loss_2
            invite_loss += loss_1 * y_hat.size(0)
            edge_loss += loss_2 * edge_y_hat.size(0)
            invite_total += y_hat.size(0)
            edge_total += edge_y_hat.size(0)

        pred = y_hat.cpu().detach()
        y_true += batch[1].data.tolist()
        y_pred += np.argmax(pred, axis=1).tolist()
        y_score += y_hat.data[:, 1].tolist()

    model.train()

    # if thr is not None:
    #     print('using threshold {:.4f}'.format(thr))
    #     y_score = np.array(y_score)
    #     y_pred = np.zeros_like(y_score)
    #     y_pred[y_score > thr] = 1

    if not args.check_edge:
        result = result_print(y_true, y_pred, y_score, epoch, loss, total, log_desc)
        print('fuse weight is {}'.format(model.fuse_weight.data))
    else:
        result = result_print_edge(y_true, y_pred, y_score, epoch,
                                   invite_loss, edge_loss, invite_total, edge_total, log_desc)
        print('fuse weight is {}'.format(model.fuse_weight.data))
        # print('temperature of threshold_gat is {}'.format(model.graph_2.gat.fuse_weight.data))

    if return_best_thr:
        precs, recs, thrs = precision_recall_curve(y_true, y_score)
        f1s = 2 * precs * recs / (precs + recs)
        f1s = f1s[:-1]
        thrs = thrs[~np.isnan(thrs)]
        f1s = f1s[~np.isnan(f1s)]
        best_thr = thrs[np.argmax(f1s)]
        print('best threshold={:.4f}, f1={:.4f}'.format(best_thr, np.max(f1s)))
        return best_thr, float(invite_loss / invite_total)
    else:
        return result


def train(epoch, data, model, device, optimizer, args, log_desc='train'):

    x = data.x
    edge_index = data.edge_index
    offline_days = data.offline_days
    train_loader = data.train_loader
    class_weight = torch.FloatTensor(data.class_weight).to(device)

    model.train()

    loss = 0.
    invite_loss = 0.
    edge_loss = 0.
    total = 0.
    invite_total = 0.
    edge_total = 0.
    y_true, y_pred, y_score = [], [], []
    pbar = tqdm(train_loader)
    for i, batch in enumerate(pbar):  # batch : invited_ids, invite_ys, edge_ys, edge_ws
        optimizer.zero_grad()
        x_batch = x.to(device)
        edge_index_batch = edge_index.to(device)
        invitations = batch[0].to(device)
        offline_days_batch = offline_days.to(device)

        y_hat, edge_y_hat = model(x_batch, edge_index_batch, invitations, offline_days_batch)
        y_hat = F.log_softmax(y_hat, dim=-1)

        if not args.check_edge:
            train_loss = F.nll_loss(y_hat, batch[1].to(device), class_weight)
            loss += float(train_loss) * y_hat.size(0)
            total += y_hat.size(0)
        else:
            edge_y_hat = F.log_softmax(edge_y_hat, dim=-1)
            loss_1 = F.nll_loss(y_hat, batch[1].to(device), class_weight)
            loss_2 = calculate_edge_loss(edge_y_hat, batch[2].to(device))
            train_loss = loss_1 + args.loss_alpha * loss_2
            invite_loss += loss_1 * y_hat.size(0)
            edge_loss += loss_2 * edge_y_hat.size(0)
            invite_total += y_hat.size(0)
            edge_total += edge_y_hat.size(0)

        pred = y_hat.cpu().detach()
        y_true += batch[1].data.tolist()
        y_pred += np.argmax(pred, axis=1).tolist()
        y_score += y_hat.data[:, 1].tolist()
        train_loss.backward()
        optimizer.step()

    if not args.check_edge:
        auc = result_print(y_true, y_pred, y_score, epoch, loss, total, log_desc)
    else:
        auc = result_print_edge(y_true, y_pred, y_score, epoch,
                                invite_loss, edge_loss, invite_total, edge_total, log_desc)
    return train_loss


def result_print(y_true, y_pred, y_score, epoch, loss, total, log_desc='train'):
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    ap = average_precision_score(y_true, y_score, average='weighted')
    roc_auc = roc_auc_score(y_true, y_score)
    print('\n{} {}_loss: {:.4f}, ROC_AUC: {:.6f}, AP: {:.6f}, Prec: {:.6f}, Rec: {:.6f}, F1: {:.6f}'.format(
        epoch + 1, log_desc, loss / total, roc_auc, ap, prec, rec, f1))
    return [roc_auc, ap, prec, rec, f1]


def result_print_edge(y_true, y_pred, y_score, epoch,
                      invite_loss, edge_loss, invite_total, edge_total, log_desc='train'):
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    ap = average_precision_score(y_true, y_score, average='weighted')
    roc_auc = roc_auc_score(y_true, y_score)
    print('\n{} {} invite_loss: {:.4f}, edge_loss: {:.4f}, '
          'ROC_AUC: {:.6f}, AP: {:.6f}, Prec: {:.6f}, Rec: {:.6f}, F1: {:.6f}'.
          format(epoch + 1, log_desc, invite_loss / invite_total, edge_loss / edge_total,
                 roc_auc, ap, prec, rec, f1))
    return [roc_auc, ap, prec, rec, f1]


def calculate_edge_loss(edge_y_hat, edge_y_true):
    """

    :param edge_y_hat: [max_len, batch_size, edge_classes]
    :param edge_y_true: [batch_size, max_len]
    :return:
    """
    edge_y_hat = edge_y_hat.transpose(0, 1).contiguous()
    invite_idx = torch.where(edge_y_true != -1)
    edge_y_hat_split = edge_y_hat[invite_idx]  # (n , edge_classes)
    edge_y_true_split = edge_y_true[invite_idx]  # (n,)
    loss = F.nll_loss(edge_y_hat_split, edge_y_true_split)

    return loss


def run(model_path, data, model, device, args):

    best_valid_loss = None
    cnt = 0
    print('training ...')
    model.reset_parameters()
    model.fuse_weight.data.fill_(float(0.5))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=25, factor=0.25, min_lr=1e-5)
    # torch.save(model.state_dict(), osp.join(model_path, 'model.pt'))
    torch.save(model, osp.join(model_path, 'model.pt'))

    for epoch in trange(args.epochs):
        loss = train(epoch, data, model, device, optimizer, args)

        ###################################### valid ######################################
        best_thr, loss = evaluate(epoch, data, model, device, args, return_best_thr=True, log_desc='valid')
        scheduler.step(float(loss))

        ###################################### early-stop ######################################
        if args.early_stopping:
            if best_valid_loss is None:
                best_valid_loss = loss
                # torch.save(model.state_dict(), osp.join(model_path, 'model.pt'))
                torch.save(model, osp.join(model_path, 'model.pt'))
            elif loss > best_valid_loss:
                cnt += 1
                print('patience count is {}, best valid loss is {:.4f}'.format(cnt, best_valid_loss))
                if cnt > 150:
                    print("Early stopping")
                    break
            else:
                best_valid_loss = loss
                print('best valid loss is {:.4f}'.format(best_valid_loss))
                cnt = 0
                # torch.save(model.state_dict(), osp.join(model_path, 'model.pt'))
                torch.save(model, osp.join(model_path, 'model.pt'))

        ###################################### test ######################################

        if (epoch + 1) % args.check_point == 0:
            print('epoch {}, test check point!'.format(epoch + 1))
            evaluate(epoch, data, model, device, args, return_best_thr=False, log_desc='test')
            if not args.early_stopping:
                # torch.save(model.state_dict(), osp.join(model_path, 'model.pt'))
                torch.save(model, osp.join(model_path, 'model.pt'))
    print("optimization Finished!")

    print("retrieve best threshold...")
    # model.load_state_dict(torch.load(osp.join(model_path, 'model.pt')))
    model1 = torch.load(osp.join(model_path, 'model.pt'))
    best_thr, _ = evaluate(args.epochs, data, model1, device, args, return_best_thr=True, log_desc='valid')

    print('testing ...')
    result = evaluate(args.epochs, data, model1, device, args, thr=best_thr, log_desc='test')

    return result
