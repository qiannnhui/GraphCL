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
    parser.add_argument('--get_f1_scores_by_deg_boundary', action='store_true', help='Calculate F1 scores for FN analysis')
#     parser.add_argument('--path', type=str, default='/disk_194b/GCL_datasets/data') # 30
#     parser.add_argument('--path', type=str, default='/disk_195a/qiannnhui/data') # 195
#     parser.add_argument('--path', type=str, default='/disk_248a/qiannnhui/data') # 248
    parser.add_argument('--path', type=str, default='/home/qiannnhui/data/data') # 249
    parser.add_argument('--aug_ratio', dest='aug_ratio', type=int, default=1,
            help='Dropout rate of data augmentation, will be multiplied by 0.1')
    parser.add_argument('--mode', type=str, default='normal', help='normal, rm_FN(rm False Negative), cheated(pos and neg), rm_FP')
    parser.add_argument('--odecay',type=float, default=1.0)  
    parser.add_argument('--similarity_measure', type=str, default='cosine', help='Similarity measure to use for the model, options: cosine, l2, cosine+l2')
    parser.add_argument('--kde_reduction_method', type=str, default='tsne', help='Dimensionality reduction method for KDE, options: tsne, pca, umap')
    parser.add_argument("--shuffle_DBN", action='store_true', help="Use shuffled DBN")
    parser.add_argument('--hard_sim_threshold', type=float, default=0.8, help='Hard similarity threshold for high similarity negative analysis')
    parser.add_argument("--coverage_threshold", type=float, default=0.5, help="Coverage threshold for FN identification")
    parser.add_argument('--reweight_strategy', type=str, default='1-coverage', help='Reweighting strategy for negatives: 1-coverage, thresholded, boost_TNs, none')
    parser.add_argument('--renormalization', action='store_true', help='Apply renormalization to weights')
    parser.add_argument('--RBO_anchor', action='store_true', help='Use anchor instead of aug pair to select FN by RBO')
    parser.add_argument('--denominator_anchor', action='store_true', help='Use anchor instead of aug pair in denominator for loss calculation')
    parser.add_argument('--RBO_p', type=float, default=0.9, help='Power for RBO calculation')
    parser.add_argument('--tn_weight', type=float, default=1.0, help='Weight for true negatives in the loss function')

    return parser.parse_args()

