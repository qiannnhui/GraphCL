import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
    parser.add_argument('--DS', dest='DS', help='Dataset')
    parser.add_argument('--local', dest='local', action='store_const', 
            const=True, default=False)
    parser.add_argument('--glob', dest='glob', action='store_const', 
            const=True, default=False)
    parser.add_argument('--prior', dest='prior', action='store_const', 
            const=True, default=False)

    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=5,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32,
            help='')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=128,
            help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=30,
                help='Number of epochs')
    parser.add_argument('--log_interval', dest='log_interval', type=int, default=10,
                help='Log interval')

    parser.add_argument('--aug', type=str, default='dnodes')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--or_loss', action='store_true', help='Set or_loss to True if this flag is present')
#     parser.add_argument('--path', type=str, default='/disk_194b/GCL_datasets/data') # 30
    parser.add_argument('--path', type=str, default='/disk_195a/qiannnhui/data') # 195
    parser.add_argument('--aug_ratio', dest='aug_ratio', type=int, default=1,
            help='Dropout rate of data augmentation, will be multiplied by 0.1')
    parser.add_argument('--mode', type=str, default='normal', help='normal, rm_FN(rm False Negative), cheated(pos and neg)')
    parser.add_argument('--odecay',type=float, default=1.0)  


    return parser.parse_args()

