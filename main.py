import torch
import pickle
import random
import argparse
import numpy as np

from sr_gnn.dataset import Data
from sr_gnn.model import SessionGraph


def str2bool(v):
    """ Used to convert the command line arg of bool into boolean var """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def train(model, opt, opt_scheduler, loss_fn, train_data, args: dict):
    opt_scheduler.step()
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(args["batch_size"])
    for i, j in zip(slices, np.arange(len(slices))):
        alias_inputs, A, items, mask, targets = train_data.get_slice(i)
        # batch_size x max_session_length
        alias_inputs = torch.tensor(alias_inputs, dtype=torch.long, device=args["device"])
        items = torch.tensor(items, dtype=torch.long, device=args["device"])  # batch_size x max_n_node
        A = torch.tensor(A, dtype=torch.float32, device=args["device"])  # batch_size x max_n_node x (max_n_node * 2)
        mask = torch.tensor(mask, dtype=torch.long, device=args["device"])  # batch_size x max_session_length
        targets = torch.tensor(targets, dtype=torch.long, device=args["device"])

        opt.zero_grad()
        scores = model([alias_inputs, A, items, mask])  # batch_size x num_candidates
        loss = loss_fn(scores, targets - 1)
        loss.backward()
        opt.step()
        total_loss += loss.item()
        print("[{}/{}] Loss: {:.4f}".format(j, len(slices), total_loss))
    return model


def eval(model, test_data, args: dict):
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(args["batch_size"])
    for i in slices:
        alias_inputs, A, items, mask, targets = test_data.get_slice(i)
        # batch_size x max_session_length
        alias_inputs = torch.tensor(alias_inputs, dtype=torch.long, device=args["device"])
        items = torch.tensor(items, dtype=torch.long, device=args["device"])  # batch_size x max_n_node
        A = torch.tensor(A, dtype=torch.float32, device=args["device"])  # batch_size x max_n_node x (max_n_node * 2)
        mask = torch.tensor(mask, dtype=torch.long, device=args["device"])  # batch_size x max_session_length

        scores = model([alias_inputs, A, items, mask])  # batch_size x num_candidates
        sub_scores = scores.topk(args["topk"])[1].cpu().detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    return np.mean(hit), np.mean(mrr)


def main(args):
    train_data = pickle.load(open('./datasets/' + args.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('./datasets/' + args.dataset + '/test.txt', 'rb'))
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    if args.dataset == 'diginetica':
        num_candidates = 43098
    elif args.dataset == 'yoochoose1_64':
        num_candidates = 37484
    else:
        num_candidates = 310

    model = SessionGraph(num_candidates=num_candidates, args=vars(args)).to(device=args.device)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lr_decay)
    opt_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_decay_step, gamma=args.lr_decay)

    best_result = [0, 0]
    best_epoch = [0, 0]
    for epoch in range(args.num_epochs):
        print('epoch: ', epoch)
        model = train(model=model,
                      opt=opt,
                      opt_scheduler=opt_scheduler,
                      loss_fn=loss_fn,
                      train_data=train_data,
                      args=vars(args))

        if ((epoch + 1) % args.eval_freq) == 0:
            hit, mrr = eval(model, test_data, args=vars(args))
            if hit >= best_result[0]:
                best_result[0] = hit
                best_epoch[0] = epoch
            if mrr >= best_result[1]:
                best_result[1] = mrr
                best_epoch[1] = epoch
            print('Best Result of TOP@{}'.format(args.topk))
            print('Epoch: {}, Recall: {:.4f}, MMR: {:.4f}'.format(best_epoch[0], best_result[0], best_result[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--random_seed', type=int, default=2021)
    parser.add_argument('--eval_freq', type=int, default=3)
    # parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_64/sample')
    parser.add_argument('--dataset', default='yoochoose1_64', help='dataset name: diginetica/yoochoose1_64/sample')
    parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
    parser.add_argument('--dim_item', type=int, default=64, help='hidden state size')
    parser.add_argument('--num_epochs', type=int, default=10, help='the number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
    parser.add_argument('--lr_decay', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--lr_decay_step', type=int, default=3, help="lr's decay step size")
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
    parser.add_argument('--propagation_steps', type=int, default=1, help='gnn propagation steps')
    parser.add_argument('--hybrid', type=str2bool, default=False, help='only use the global preference to predict')
    parser.add_argument('--validation', type=str2bool, default=True, help='validation')
    parser.add_argument('--valid_portion', type=float, default=0.1, help='portion of validation')
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()
    print(args)

    # Set the random seed
    # [WARNING] Performance Degradation: https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    main(args=args)
