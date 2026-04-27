import os
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import json
from sampler import WarpSampler
from model import SASRec, SASRecMoE
from tqdm import tqdm
from util import *


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=201, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str, help='Device to use: cuda or cpu')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    
    # MoE模型参数
    parser.add_argument('--use_moe', type=str2bool, default=False, help='是否使用MoE价格塔')
    parser.add_argument('--cat_num', default=1696, type=int, help='品类数量')
    parser.add_argument('--num_accounts', default=8, type=int, help='账户原型数量（专家数量）')
    parser.add_argument('--top_n', default=2, type=int, help='稀疏门控TopN')
    parser.add_argument('--temperature', default=1.0, type=float, help='温度系数')
    parser.add_argument('--beta', default=0.5, type=float, help='价格塔权重')
    parser.add_argument('--lambda_reg', default=0.01, type=float, help='防坍缩正则化系数')
    parser.add_argument('--cat_emb_dim', default=32, type=int, help='品类嵌入维度')

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory
    if not os.path.isdir(args.dataset + '_' + args.train_dir):
        os.makedirs(args.dataset + '_' + args.train_dir)

    with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

    # Load dataset
    if args.use_moe:
        # 加载多特征数据
        dataset = data_partition(args.dataset, use_features=True)
        [user_train, user_valid, user_test, usernum, itemnum,
         user_train_features, user_valid_features, user_test_features] = dataset
    else:
        dataset = data_partition(args.dataset)
        [user_train, user_valid, user_test, usernum, itemnum] = dataset

    num_batch = len(user_train) // args.batch_size
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load category mapping for negative sampling (if using MoE)
    cat_item_map = {}
    all_item_list = list(range(1, itemnum + 1))
    
    if args.use_moe:
        try:
            with open("data/cat_item_map.json", "r", encoding="utf-8") as f:
                sampling_data = json.load(f)
            cat_item_map = sampling_data['cat_item_map']
            all_item_list = sampling_data['all_item_list']
            print(f'Loaded category mapping: {len(cat_item_map)} categories, {len(all_item_list)} items')
        except:
            print('Warning: Could not load category mapping, using global sampling')

    # Create sampler
    if args.use_moe:
        sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3,
                              use_features=True, user_train_features=user_train_features,
                              cat_item_map=cat_item_map, all_item_list=all_item_list)
    else:
        sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)

    # Create model
    if args.use_moe:
        model = SASRecMoE(usernum, itemnum, args.cat_num, args).to(device)
        print(f'Model: SASRecMoE (Accounts={args.num_accounts}, TopN={args.top_n}, Beta={args.beta})')
    else:
        model = SASRec(usernum, itemnum, args).to(device)
        print(f'Model: SASRec')

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    # Open log file
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')

    T = 0.0
    t0 = time.time()

    try:
        for epoch in range(1, args.num_epochs + 1):
            model.train()
            
            for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                batch_data = sampler.next_batch()
                
                if args.use_moe:
                    # 解包多特征批次
                    (u, seq, pos, neg,
                     price_seq, price_pos, price_neg,
                     cat_seq, cat_pos, cat_neg,
                     price_dev_seq, price_dev_pos, price_dev_neg,
                     time_dev_seq, time_dev_pos, time_dev_neg) = batch_data
                    
                    # 转换为numpy数组再转tensor（避免警告）
                    seq_tensor = torch.LongTensor(np.array(seq)).to(device)
                    pos_tensor = torch.LongTensor(np.array(pos)).to(device)
                    neg_tensor = torch.LongTensor(np.array(neg)).to(device)
                    
                    price_seq_t = torch.FloatTensor(np.array(price_seq)).to(device)
                    price_pos_t = torch.FloatTensor(np.array(price_pos)).to(device)
                    price_neg_t = torch.FloatTensor(np.array(price_neg)).to(device)
                    
                    cat_seq_t = torch.LongTensor(np.array(cat_seq)).to(device)
                    cat_pos_t = torch.LongTensor(np.array(cat_pos)).to(device)
                    cat_neg_t = torch.LongTensor(np.array(cat_neg)).to(device)
                    
                    price_dev_seq_t = torch.FloatTensor(np.array(price_dev_seq)).to(device)
                    price_dev_pos_t = torch.FloatTensor(np.array(price_dev_pos)).to(device)
                    price_dev_neg_t = torch.FloatTensor(np.array(price_dev_neg)).to(device)
                    
                    time_dev_seq_t = torch.FloatTensor(np.array(time_dev_seq)).to(device)
                    time_dev_pos_t = torch.FloatTensor(np.array(time_dev_pos)).to(device)
                    time_dev_neg_t = torch.FloatTensor(np.array(time_dev_neg)).to(device)
                    
                    # Forward pass
                    loss, auc = model(
                        seq_tensor, pos_tensor, neg_tensor,
                        price_seq_t, price_pos_t, price_neg_t,
                        cat_seq_t, cat_pos_t, cat_neg_t,
                        price_dev_seq_t, price_dev_pos_t, price_dev_neg_t,
                        time_dev_seq_t, time_dev_pos_t, time_dev_neg_t,
                        is_training=True
                    )
                else:
                    u, seq, pos, neg = batch_data
                    
                    seq_tensor = torch.LongTensor(np.array(seq)).to(device)
                    pos_tensor = torch.LongTensor(np.array(pos)).to(device)
                    neg_tensor = torch.LongTensor(np.array(neg)).to(device)
                    
                    loss, auc = model(seq_tensor, pos_tensor, neg_tensor, is_training=True)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % 20 == 0:
                t1 = time.time() - t0
                T += t1
                print('Evaluating', end='')
                
                t_test = evaluate(model, dataset, args, device)
                t_valid = evaluate_valid(model, dataset, args, device)
                
                print('')
                print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)' % (
                    epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

                f.write(str(t_valid) + ' ' + str(t_test) + '\n')
                f.flush()
                t0 = time.time()
                
                # Save model checkpoint
                checkpoint_path = os.path.join(args.dataset + '_' + args.train_dir, f'model_epoch_{epoch}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_ndcg': t_valid[0],
                    'valid_hr': t_valid[1],
                    'test_ndcg': t_test[0],
                    'test_hr': t_test[1],
                }, checkpoint_path)

    except KeyboardInterrupt:
        print('\nTraining interrupted by user')
    except Exception as e:
        print(f'\nError during training: {e}')
        import traceback
        traceback.print_exc()
    finally:
        sampler.close()
        f.close()

    print("Done")
