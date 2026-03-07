import argparse


def get_argparser():
    parser = argparse.ArgumentParser(
        description='yylfix: Protein domain prediction with '
                    'axial attention + VGAE refinement')

    # Dataset options
    parser.add_argument('--compute', action='store_true',
                        help='Compute and save features for the dataset')
    parser.add_argument('--force_reprocess', action='store_true',
                        help='Force reprocessing of features')
    parser.add_argument("--dataset", type=str, default='yjy-style',
                        choices=['yjy-style', 'single-chain'],
                        help='Dataset type')
    parser.add_argument("--batch_size", type=int, default=1,
                        help='Batch size')
    parser.add_argument("--max_length", type=int, default=512,
                        help='Maximum protein length. Proteins longer than this '
                             'are dense-cropped to a contact-rich region. '
                             '-1=no limit (may OOM on long proteins)')
    parser.add_argument("--crop_size", type=int, default=512,
                        help='OOM safety-net crop size. Only used during OOM retry, '
                             'not for proactive cropping (windowed attention handles memory). '
                             '-1=skip on OOM')
    parser.add_argument("--use_checkpoint", action='store_true',
                        help='Use gradient checkpointing to reduce GPU memory '
                             '(recomputes activations during backward, ~6GB vs ~44GB)')
    parser.add_argument("--max_data_count", type=int, default=-1,
                        help='Max data samples (-1 = no limit)')
    parser.add_argument("--limit_samples", type=int, default=-1,
                        help='Limit training samples (-1 = no limit)')

    # Training options
    parser.add_argument("--device", type=str, default='cpu',
                        help='Device (cpu/cuda)')
    parser.add_argument("--epoch", type=int, default=24,
                        help='Number of epochs')
    parser.add_argument("--lr", type=float, default=3e-4,
                        help='Learning rate (Adam default: 3e-4)')
    parser.add_argument("--lr_patience", type=int, default=3,
                        help='ReduceLROnPlateau patience')
    parser.add_argument("--lr_factor", type=float, default=0.5,
                        help='ReduceLROnPlateau factor')
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help='Gradient clipping max norm')

    # Model options
    parser.add_argument("--filters", type=int, default=64,
                        help='Number of CNN filters')
    parser.add_argument("--num_layers", type=int, default=61,
                        help='Number of residual layers')
    parser.add_argument("--in_channels", type=int, default=10,
                        help='Input channels (8 features + 2 recycling)')

    # Paths
    parser.add_argument('--stride_path', type=str,
                        default='/root/autodl-tmp/yyl1/stride/stride',
                        help='Path to STRIDE executable')
    parser.add_argument("--model_saved_path", type=str, default='saved_models',
                        help='Directory for saved models')
    parser.add_argument("--load_model_path", type=str, default="",
                        help='Path to load pretrained model')

    # Dataset paths (yjy-style)
    parser.add_argument("--yjy_data_dir", type=str,
                        default='dataset/train/single/single',
                        help='Training data directory')
    parser.add_argument("--yjy_features_dir", type=str,
                        default='yjy-features',
                        help='Features cache directory')
    parser.add_argument("--yjy_label_file", type=str,
                        default='dataset/train/single/final.info',
                        help='Training label file')
    parser.add_argument("--yjy_test_data_dir", type=str,
                        default='dataset/test/single',
                        help='Test data directory')
    parser.add_argument("--yjy_test_features_dir", type=str,
                        default='yjy-test-features',
                        help='Test features cache directory')
    parser.add_argument("--yjy_test_label_file", type=str,
                        default='dataset/test/final.info',
                        help='Test label file')
    parser.add_argument("--preprocess_all_first", action='store_true',
                        help='Preprocess all samples before training')

    # Loss weights - 6 TERMS
    def parse_dict(arg_str):
        try:
            return dict(item.split("=") for item in arg_str.split(","))
        except Exception:
            raise argparse.ArgumentTypeError(
                "Dictionary must be in key1=value1,key2=value2 format")

    parser.add_argument("--loss_weight", type=parse_dict, default={
        'adj_ce_weight': 1.0,
        'lr_mse_weight': 1.0,
        'Lc_weight': 1.0,
        'vgae_weight': 1.0,           # VGAE refinement (recon + KL)
        'boundary_weight': 0.5,
        'plddt_weight': 0.3,          # pLDDT self-supervised confidence
    })

    parser.add_argument("--positional_encoding", type=str, default='linear',
                        choices=['linear', 'sinusoidal'])

    # Self-conditioned training
    parser.add_argument("--R", type=int, default=2,
                        help='Max recycling rounds')
    parser.add_argument("--K_init", type=int, default=4,
                        help='Initial domain count for greedy assignment')

    # Evaluation
    parser.add_argument("--eval_interval", type=int, default=1,
                        help='Evaluate every N epochs')
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help='IoU threshold for domain matching')

    return parser

