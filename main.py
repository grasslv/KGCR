import argparse
import os
import numpy as np
import torch
import random
from Dataset import TrainDataset, data_load
from KGCR import *
from torch.utils.data import DataLoader
from Train import train
from Full_rank import full_ranking
from prettytable import PrettyTable


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Seed init.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--data_path', default='Ciao', help='Dataset path')
    parser.add_argument('--save_file', default='test1231', help='Filename')

    parser.add_argument('--PATH_weight_load', default=None, help='Loading weight filename.')
    parser.add_argument('--PATH_weight_save', default=None, help='Writing weight filename.')
    parser.add_argument('--prefix', default='', help='Prefix of save_file.')
    parser.add_argument('--alpha', type=float, default=1.0, help='Learning rate.')
    parser.add_argument('--beta', type=float, default=1.0, help='Learning rate.')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin.')

    parser.add_argument('--l_r', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--reg_weight', type=float, default=1e-2, help='Weight_regularization.')
    parser.add_argument('--model_name', default='model', help='Model Name.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')  # defalut :128
    parser.add_argument('--num_epoch', type=int, default=50, help='Epoch number.')
    parser.add_argument('--num_workers', type=int, default=3, help='Workers number.')

    parser.add_argument('--dim_E', type=int, default=64, help='Embedding dimension.')
    parser.add_argument('--topK', type=int, default=20, help='Workers number.')
    parser.add_argument('--step', type=int, default=2000, help='Workers number.')

    parser.add_argument('--has_pre_trained', default='True', help='Has Pretrained Module.')
    parser.add_argument('--has_transE', default='False', help='Conduct TransE.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = init()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    data_path = args.data_path
    save_file_name = args.save_file

    alpha = args.alpha
    beta = args.beta
    margin = args.margin
    learning_rate = args.l_r
    reg_weight = args.reg_weight
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_epoch = args.num_epoch
    prefix = args.prefix
    model_name = args.model_name

    dim_E = args.dim_E
    topK = args.topK
    step = args.step
    has_pre_trained = True if args.has_pre_trained == 'True' else False
    has_transE = True if args.has_transE == 'True' else False
    writer = None
    print('Data loading ...')
    train_data, test_data, u_u_list, i_i_list, user_item_dict, user_item_dict_train, u_u_dict, num_u, num_i, num_a = data_load(data_path)
    train_dataset = TrainDataset(train_data, user_item_dict, num_i, num_u)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)

    # KG_dataset = KGDataset(kg_data, relation_list, h_r_dict, num_i+num_a, num_r)
    # KG_dataloader = DataLoader(KG_dataset, batch_size, shuffle=True, num_workers=num_workers)
    print('Data has been loaded.')
    #################################################################################################################
    print(model_name)

    model = KGCR(num_u, num_i, num_a, train_data, u_u_list, i_i_list, reg_weight, dim_E, alpha, beta, margin).cuda()
    # model = KGCR(num_u, num_i, num_a, train_data, u_u_list, i_i_list, reg_weight, dim_E, alpha, beta, margin).cuda()

    if has_pre_trained:
        pretrained_id_embed = torch.load('./datasets/' + data_path + '/id.pt').cuda()
        model.id_embedding.data = pretrained_id_embed
        pretrained_item_rep = torch.load('./datasets/' + data_path + '/item.pt').cuda()
        model.item_pre.data = pretrained_item_rep.weight
        pretrained_att_rep = torch.load('./datasets/' + data_path + '/user.pt').cuda()
        model.attribute.data = pretrained_att_rep.weight
        print('pretrained_id_embed has loaded ...')

    #################################################################################################################
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': learning_rate}])
    #################################################################################################################
    max_precision = 0.0
    max_recall = 0.0
    max_NDCG = 0.0
    num_decreases = 0
    max_val_result = list()
    max_test_result = [0, 0, 0]
    pt = PrettyTable()
    pt.field_names = ["Epoch", "Loss", "precision", "recall", "ndcg"]
    now_epoch = 0
    for epoch in range(num_epoch):
        loss = train(epoch, len(train_dataset), train_dataloader, model, optimizer, batch_size, writer)
        if torch.isnan(loss):
            print(model.result)
            with open('./datasets/' + data_path + '/result_{0}.txt'.format(save_file_name), 'a') as save_file:
                save_file.write('lr:{0} \t reg_weight:{1} is Nan\r\n'.format(learning_rate, reg_weight))
            break
        torch.cuda.empty_cache()

        # if epoch % 10 == 0: 129-160后退一个缩进
        test_result = full_ranking(epoch, model, test_data, user_item_dict_train, None, False, step, topK, model_name,
                                   'Test/', writer)
        # pt.add_row([epoch, loss, test_result[0], test_result[1], test_result[2]])

        if test_result[1] > max_recall:
            # TODO: Metric还在上升，这里记录的意义是？
            max_recall = test_result[1]
            max_test_result = test_result
            num_decreases = 0
            torch.save(model.state_dict(), f'best_model_{save_file_name}.pt')
            test_result5 = full_ranking(epoch, model, test_data, user_item_dict_train, None, False, step, 5, model_name,
                                        'Test/', writer)
            test_result10 = full_ranking(epoch, model, test_data, user_item_dict_train, None, False, step, 10,
                                         model_name, 'Test/', writer)
            test_result20 = full_ranking(epoch, model, test_data, user_item_dict_train, None, False, step, 20,
                                         model_name, 'Test/', writer)

            with open('./datasets/' + data_path + '/result_{0}.txt'.format(save_file_name), 'a') as save_file:
                save_file.write(str(args))
                save_file.write('\n epoch:' + str(epoch))
                save_file.write(
                    '\r\n-----------Test Precition@5:{0:.4f} Recall@5:{1:.4f} NDCG@5:{2:.4f}-----------'.format(
                        test_result5[0], test_result5[1], test_result5[2]))
                save_file.write(
                    '\r\n-----------Test Precition@10:{0:.4f} Recall@10:{1:.4f} NDCG@10:{2:.4f}-----------'.format(
                        test_result10[0], test_result10[1], test_result10[2]))
                save_file.write(
                    '\r\n-----------Test Precition@20:{0:.4f} Recall@20:{1:.4f} NDCG@20:{2:.4f}-----------'.format(
                        test_result20[0], test_result20[1], test_result20[2]))
                save_file.write(
                    '\r\n------Best Test Precition@20:{0:.4f} Recall@20:{1:.4f} NDCG@20:{2:.4f}-----------'.format(
                        max_test_result[0], max_test_result[1], max_test_result[2]))

        else:
            # TODO: 这是啥意思？
            # Patience 步数 设置的 5000?
            if num_decreases > 5000:
                torch.save(model.result, 'result.pt')
                torch.save(model.ua_rep, 'ua_rep.pt')
                torch.save(model.ia_rep, 'ia_rep.pt')

                torch.save(model.u_result, 'u_result.pt')
                torch.save(model.hat_u_result, 'hat_u_result.pt')
                torch.save(model.hat_i_result, 'hat_i_result.pt')
                torch.save(model.i_result, 'i_result.pt')
                test_result5 = full_ranking(epoch, model, test_data, user_item_dict_train, None, False, step, 5,
                                            model_name, 'Test/', writer)
                test_result10 = full_ranking(epoch, model, test_data, user_item_dict_train, None, False, step, 10,
                                             model_name, 'Test/', writer)
                test_result20 = full_ranking(epoch, model, test_data, user_item_dict_train, None, False, step, 20,
                                             model_name, 'Test/', writer)

                with open('./datasets/' + data_path + '/result_{0}.txt'.format(save_file_name), 'a') as save_file:
                    save_file.write(str(args))
                    save_file.write('\n epoch:' + str(epoch))
                    save_file.write(
                        '\r\n-----------Test Precition@5:{0:.4f} Recall@5:{1:.4f} NDCG@5:{2:.4f}-----------'.format(
                            test_result5[0], test_result5[1], test_result5[2]))
                    save_file.write(
                        '\r\n-----------Test Precition@10:{0:.4f} Recall@10:{1:.4f} NDCG@10:{2:.4f}-----------'.format(
                            test_result10[0], test_result10[1], test_result10[2]))
                    save_file.write(
                        '\r\n-----------Test Precition@20:{0:.4f} Recall@20:{1:.4f} NDCG@20:{2:.4f}-----------'.format(
                            test_result20[0], test_result20[1], test_result20[2]))
                    save_file.write(
                        '\r\n------Best Test Precition@20:{0:.4f} Recall@20:{1:.4f} NDCG@20:{2:.4f}-----------'.format(
                            max_test_result[0], max_test_result[1], max_test_result[2]))
                break
            else:
                num_decreases += 1
