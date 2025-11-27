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
    parser.add_argument('--neg_include_self', action='store_true', help='Include self in negative samples')
#     parser.add_argument('--path', type=str, default='/disk_194b/GCL_datasets/data') # 30
#     parser.add_argument('--path', type=str, default='/disk_195a/qiannnhui/data') # 195
#     parser.add_argument('--path', type=str, default='/disk_248a/qiannnhui/data') # 248
    parser.add_argument('--path', type=str, default='/home/qiannnhui/data/data') # 249
    parser.add_argument('--aug_ratio', dest='aug_ratio', type=int, default=1,
            help='Dropout rate of data augmentation, will be multiplied by 0.1')
    parser.add_argument('--mode', type=str, default='normal', help='normal, rm_FN(rm False Negative), cheated(pos and neg), rm_FP')
    parser.add_argument('--odecay',type=float, default=1.0)  
    parser.add_argument('--plot_theta_l2', action='store_true', help='Plot theta l2 norm')
    parser.add_argument('--plot_theta_l2_distribution', action='store_true', help='Plot single anchor FP and FN distribution')
    parser.add_argument('--plot_anchor_aug_pair_theta_per_epoch', action='store_true', help='Plot anchor-augmented pair theta per epoch')
    parser.add_argument('--similarity_measure', type=str, default='cosine', help='Similarity measure to use for the model, options: cosine, l2, cosine+l2')
    parser.add_argument('--plot_kde', action='store_true', help='Plot KDE of angles')
    parser.add_argument('--kde_reduction_method', type=str, default='tsne', help='Dimensionality reduction method for KDE, options: tsne, pca, umap')
    parser.add_argument("--shuffle_DBN", action='store_true', help="Use shuffled DBN")

    return parser.parse_args()

