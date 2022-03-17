import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import models
import json
import os
import argparse
import numpy as np
import our_models
import struct_utils
from crf_dataloader import CRFDataLoader


def main():
    parser = argparse.ArgumentParser(description='Linear-Chain CRF Training')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs')
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help='select model <resnet18/resnet34/resnet50>')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        help='initial learning rate')
    parser.add_argument('--out', default=None, type=str,
                        help='output directory')
    parser.add_argument('--input_size', default=112, type=int,
                        help='image size')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='batch size')
    parser.add_argument('--num_samples', default=50, type=int,
                        help='number of sampled frames per training video')
    parser.add_argument('--lambda_p', default=1., type=float,
                        help='lambda temparature between unary/pairwise')
    parser.add_argument('--trans_weight', default=1., type=float,
                        help='weight to the transition label in motion model')
    parser.add_argument('--ce_weight', default=1., type=float,
                        help='weight to the cross entropy')

    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    if args.out:
        out_dir = args.out
    else:
        out_dir = 'pipeline_results/{}-{}/'.format(
            args.num_samples, args.trans_weight)
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir + 'params.json', 'w') as fp:
        json.dump(dict(args._get_kwargs()), fp)

    N_INPUT_FLOW = 2
    N_OUTPUT_FLOW = 2
    STAGE_TO_NUMBER = {'1-cell': 0,
                       '2-cell': 1,
                       '3-cell': 2,
                       '4-cell': 3,
                       '5-cell': 4,
                       '6-cell': 5,
                       '7-cell': 6,
                       '8-cell': 7,
                       '9+-cell': 8,
                       'morula': 9,
                       'blastocyst': 10,
                       }

    embryo_datasets = dict()
    embryo_datasets['training'] = CRFDataLoader(
        STAGE_TO_NUMBER, input_size=args.input_size,
        train=True, set_name='training', num_samples=args.num_samples)
    embryo_datasets['validation'] = CRFDataLoader(
        STAGE_TO_NUMBER, input_size=args.input_size,
        train=False, set_name='validation')

    train_loader = DataLoader(embryo_datasets['training'],
                              batch_size=args.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=args.num_workers,
                              drop_last=True)

    val_loader = DataLoader(embryo_datasets['validation'],
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=args.num_workers)

    # Initialize the model
    use_pretrained = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.backbone == "resnet18":
        backbone_model = models.resnet18(pretrained=use_pretrained)
    elif args.backbone == "resnet34":
        backbone_model = models.resnet34(pretrained=use_pretrained)
    elif args.backbone == "resnet50":
        backbone_model = models.resnet50(pretrained=use_pretrained)
    n_output_classes = len(STAGE_TO_NUMBER)
    model_im = our_models.model_building_rgb(
        n_output_classes, backbone_model, device)
    model_im = model_im.to(device)

    if args.backbone == "resnet18":
        backbone_model = models.resnet18(pretrained=use_pretrained)
    elif args.backbone == "resnet34":
        backbone_model = models.resnet34(pretrained=use_pretrained)
    elif args.backbone == "resnet50":
        backbone_model = models.resnet50(pretrained=use_pretrained)

    model_flow = our_models.model_building_flow(
        N_OUTPUT_FLOW, backbone_model, device, N_INPUT_FLOW)
    model_flow = model_flow.to(device)

    params = list(model_im.parameters()) + list(model_flow.parameters())

    optimizer = optim.Adam(params, lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               [250, 280],
                                               gamma=0.1,
                                               verbose=True)
    num_warmup = 1000
    warmup_optimizer = optim.Adam(params, lr=args.lr*.001, weight_decay=1e-5)
    warmup_scheduler = optim.lr_scheduler.StepLR(warmup_optimizer,
                                                 step_size=1,
                                                 gamma=1.0069,
                                                 verbose=False)
    train_hist = {'loss': [], 'acc': []}
    val_hist = {'loss': [], 'acc': [], 'acc_raw': []}
    best_acc = 0
    flow_weights = torch.tensor([1., args.trans_weight]).to(device)
    num_iters = 0
    for epoch in range(args.epochs):
        # TRAINNG
        print(f'\n===============\nTRAINING - Epoch {epoch+1}/{args.epochs}:')
        model_im.train()
        model_flow.train()
        running_correct, running_loss, total = 0, 0.0, 0
        for i, (inputs_im, im_label, inputs_flow, flow_label) in enumerate(
                train_loader):
            inputs_im = inputs_im.view(-1, 1, 112, 112).to(device)
            inputs_flow = inputs_flow.view(-1, 2, 112, 112).to(device)
            im_label = im_label.view(-1, 1).to(device)
            flow_label = flow_label.view(-1, 1).to(device)
            # run models
            if num_iters < num_warmup:
                warmup_optimizer.zero_grad()
            else:
                optimizer.zero_grad()
            im_pred = model_im(inputs_im)
            flow_pred = model_flow(inputs_flow)
            im_crfinput = im_pred.view(
                args.batch_size, args.num_samples, -1)
            flow_crfinput = flow_pred.view(
                args.batch_size, args.num_samples - 1, -1)
            argmax_C_by_C, argmax_labels, parts_C_by_C, parts_labels, dist = \
                struct_utils.linear_chain_distribution(
                    im_crfinput, flow_crfinput, im_label, args.num_samples,
                    img=True, flow=True, verbose=False, use_weighting=False,
                    lambda_p=args.lambda_p)
            loss = -dist.log_prob(parts_C_by_C).sum()
            loss += args.ce_weight*F.cross_entropy(im_pred,
                                                   im_label.squeeze(),
                                                   reduction='sum')
            loss += args.ce_weight*F.cross_entropy(flow_pred,
                                                   flow_label.squeeze(),
                                                   weight=flow_weights,
                                                   reduction='sum')
            loss.backward()
            if num_iters < num_warmup:
                warmup_optimizer.step()
                warmup_scheduler.step()
            else:
                optimizer.step()
            running_loss += loss.item()
            running_correct += np.sum(argmax_labels == parts_labels)
            total += parts_labels.size
            num_iters += 1
        train_loss = running_loss / len(train_loader)
        train_acc = running_correct / total
        train_hist['loss'].append(train_loss)
        train_hist['acc'].append(train_acc)

        # VALIDATION
        model_im.eval()
        model_flow.eval()
        running_correct, running_loss, total = 0, 0.0, 0
        with torch.no_grad():
            for i, (inputs_im, im_label, inputs_flow, flow_label) in enumerate(
                    val_loader):
                inputs_im = inputs_im.view(-1, 1, 112, 112)
                inputs_flow = inputs_flow.view(-1, 2, 112, 112)
                inputs_im = inputs_im.to(device)
                inputs_flow = inputs_flow.to(device)
                im_label = torch.stack(im_label, 1).view(-1, 1).to(device)
                flow_label = torch.stack(flow_label, 1).view(-1, 1).to(device)
                # run models
                im_pred = model_im(inputs_im)
                flow_pred = model_flow(inputs_flow)
                im_crfinput = im_pred.view(1, -1, n_output_classes)
                flow_crfinput = flow_pred.view(1, -1, N_OUTPUT_FLOW)
                argmax_C_by_C, argmax_labels, parts_C_by_C, parts_labels, \
                    dist = struct_utils.linear_chain_distribution(
                        im_crfinput, flow_crfinput, im_label,
                        im_label.shape[0], img=True, flow=True, verbose=False,
                        use_weighting=False, lambda_p=args.lambda_p)
                loss = -dist.log_prob(parts_C_by_C).sum()
                loss += args.ce_weight*F.cross_entropy(im_pred,
                                                       im_label.squeeze(),
                                                       reduction='sum')
                loss += args.ce_weight*F.cross_entropy(flow_pred,
                                                       flow_label.squeeze(),
                                                       weight=flow_weights,
                                                       reduction='sum')
                running_loss += loss.item()
                running_correct += np.sum(argmax_labels == parts_labels)
                total += parts_labels.size
        val_loss = running_loss / len(val_loader)
        val_acc = running_correct / total
        val_hist['loss'].append(val_loss)
        val_hist['acc'].append(val_acc)

        scheduler.step()

        print(f'Epoch {epoch+1} | train loss: \t{train_loss:.4f}, train acc: \t{train_acc:.4f}')
        print(f'Epoch {epoch+1} | val loss: \t{val_loss:.4f}, val acc: \t{val_acc:.4f}')

        state = {'epoch': epoch, 'im_state_dict': model_im.state_dict(
        ), 'flow_state_dict': model_flow.state_dict()}
        if best_acc < val_acc:
            best_acc = val_acc
            print(os.path.join(out_dir, "best-checkpoint.pth.tar"))
            torch.save(state, os.path.join(out_dir, "best-checkpoint.pth.tar"))
        print(os.path.join(out_dir, "latest-checkpoint.pth.tar"))
        torch.save(state, os.path.join(out_dir, "latest-checkpoint.pth.tar"))


if __name__ == '__main__':
    main()
