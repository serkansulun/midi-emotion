import os
import time
import argparse

parser = argparse.ArgumentParser(description='Generates emotion-based symbolic music')

parser.add_argument('--note', default=None, type=str,
                    help='Notes about the experiment.')
parser.add_argument("--conditioning", type=str, required=False, default="continuous_concat",
                    choices=["none", "discrete_token", "continuous_token",
                             "continuous_concat"], help='Conditioning type')
parser.add_argument("--data_folder", type=str, default="../data_files/lpd_5/lpd_5_full_transposable_debug")
parser.add_argument('--full_dataset', action="store_true",
                    help='Use LPD-full dataset')
parser.add_argument('--n_layer', type=int, default=20,
                    help='number of total layers')
parser.add_argument('--n_head', type=int, default=16,
                    help='number of heads')
parser.add_argument('--d_model', type=int, default=768,
                    help='model dimension')
parser.add_argument('--d_condition', type=int, default=192,
                    help='condition dimension (if continuous_concat is used)')
parser.add_argument('--d_inner', type=int, default=768*4,
                    help='inner dimension in FF')
parser.add_argument('--tgt_len', type=int, default=1216, #1216, #1152
                    help='number of tokens to predict')
parser.add_argument('--max_gen_input_len', type=int, default=-1,
                    help='number of tokens to predict')
parser.add_argument('--gen_len', type=int, default=2048,
                    help='Generation length')
parser.add_argument('--temp_note', type=float, default=1.2,
                    help='Temperature for generating notes')
parser.add_argument('--temp_rest', type=float, default=1.2,
                    help='Temperature for generating rests')
parser.add_argument('--n_bars', type=int, default=-1,
                    help='number of bars to use')
parser.add_argument('--no_pad', action='store_true',
                    help='dont pad sequences')
parser.add_argument('--eval_tgt_len', type=int, default=-1,
                    help='number of tokens to predict for evaluation')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='global dropout rate')
parser.add_argument("--overwrite_dropout", action="store_true",
                    help="resets dropouts")
parser.add_argument('--lr', type=float, default=2e-5,
                    help='initial learning rate (0.00025|5 for adam|sgd)')
parser.add_argument("--overwrite_lr", action="store_true", 
                    help="Overwrites learning rate if pretrained model is loaded.")
parser.add_argument('--arousal_feature', default='note_density', type=str,
                    choices=['tempo', 'note_density'],
                    help='Feature to use as arousal feature')
parser.add_argument('--scheduler', default='constant', type=str,
                    choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant', "cyclic"],
                    help='lr scheduler to use.')
parser.add_argument('--lr_min', type=float, default=5e-6,
                    help='minimum learning rate for cyclic scheduler')
parser.add_argument('--lr_max', type=float, default=5e-3,
                    help='maximum learning rate for cyclic scheduler')
parser.add_argument('--warmup_step', type=int, default=0,
                    help='upper epoch limit')
parser.add_argument('--decay_rate', type=float, default=0.5,
                    help='decay factor when ReduceLROnPlateau is used')
parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clipping')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch size')
parser.add_argument('--accumulate_step', type=int, default=1,
                    help='accumulate gradients (multiplies effective batch size')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='use CPU')
parser.add_argument('--log_step', type=int, default=1000,
                    help='report interval')
parser.add_argument('--eval_step', type=int, default=8000,
                    help='evaluation interval')
parser.add_argument('--max_eval_step', type=int, default=1000,
                    help='maximum evaluation steps')
parser.add_argument('--gen_step', type=int, default=8000,
                    help='generation interval')
parser.add_argument('--work_dir', default='../output', type=str,
                    help='experiment directory.')
parser.add_argument('--restart_dir', type=str, default=None,
                    help='restart dir')
parser.add_argument('--debug', action='store_true',
                    help='run in debug mode (do not create exp dir)')
parser.add_argument('--max_step', type=int, default=1000000000,
                    help='maximum training steps')
parser.add_argument('--overfit', action='store_true',
                    help='Works on a single sample')
parser.add_argument('--find_lr', action='store_true',
                    help='Run learning rate finder')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of cores for data loading')
parser.add_argument('--bar_start_prob', type=float, default=0.5,
                    help=('probability of training sample'
                    ' starting at a bar location'))
parser.add_argument("--n_samples", type=int, default=-1,
                    help="Limits number of training samples (for faster debugging)") 
parser.add_argument('--n_emotion_bins', type=int, default=5,
                    help='Number of emotion bins in each dimension')
parser.add_argument('--max_transpose', type=int, default=3,
                    help='Maximum transpose amount')
parser.add_argument('--no_amp', action="store_true",
                    help='Disable automatic mixed precision')
parser.add_argument('--reset_scaler', action="store_true",
                    help="Reset scaler (can help avoiding nans)")
parser.add_argument('--exhaustive_eval', action="store_true",
                    help="Use data exhaustively (for final evaluation)")
parser.add_argument('--regression', action="store_true",
                    help="Train a regression model")
parser.add_argument("--always_use_discrete_condition", action="store_true", 
                help="Discrete tokens are used for every sequence")
parser.add_argument("--regression_dir", type=str, default=None,
                    help="The path of folder with generations, to perform regression on")
parser.add_argument("--attn_type", type=str, default="causal-linear",
                    help="Attention type for fast transformers")
parser.add_argument("--fast_transformers", action="store_true",
                    help="Use the fast-transformers library")

args = parser.parse_args()

if args.regression_dir is not None:
    args.regression = True

if args.conditioning != "continuous_concat":
    args.d_condition = -1

assert not (args.exhaustive_eval and args.max_eval_step > 0)

# assert (args.attn_type == "full") or args.no_amp, "Linear attention doesn't work with AMP"

# if args.fast_transformers:
#     args.attn_type = 'causal-linear'
#     print("Using fast transformers, switching to linear attention.")

if args.full_dataset:
    assert args.conditioning in ["discrete_token", "none"] and not args.regression, "LPD-full has NaN features"

if args.regression:
    # args.n_layer = 8
    print(f"Using {args.n_layer} layers for regression")

args.batch_chunk = -1

if args.debug or args.overfit:
    args.num_workers = 0

if args.find_lr:
    args.debug = True

args.d_embed = args.d_model
    
if args.eval_tgt_len < 0:
    args.eval_tgt_len = args.tgt_len

if args.scheduler == "cyclic":
    args.lr = args.lr_min

if args.restart_dir:
    args.restart_dir = os.path.join(args.work_dir, args.restart_dir)

if args.debug:
    args.work_dir = os.path.join(args.work_dir, "DEBUG_" + time.strftime('%Y%m%d-%H%M%S'))
    args.data_folder += '_debug'
elif args.no_cuda:
    args.work_dir = os.path.join(args.work_dir, "CPU_" + time.strftime('%Y%m%d-%H%M%S'))
else:
    args.work_dir = os.path.join(args.work_dir, time.strftime('%Y%m%d-%H%M%S'))
