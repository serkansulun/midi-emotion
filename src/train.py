import time
import math
import datetime
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.build_model import build_model
from generate import generate
from data.preprocess_features import preprocess_features
from data.loader import Loader
from data.loader_exhaustive import LoaderExhaustive
from data.loader_generations import LoaderGenerations
from data.collate import filter_collate
from utils import CsvWriter, create_exp_dir, accuracy, memory, plot_performance
from config import args

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set the random seed manually for reproducibility.
if args.seed > 0:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

class Runner:
    def __init__(self):
        self.logging = create_exp_dir(args.work_dir, debug=args.debug)
        if not args.debug and args.note != None:
            # write the note in folder for ease of comparison and viewing
            open(os.path.join(args.work_dir, 'aa ' + args.note), "x")


        use_cuda = torch.cuda.is_available() and not args.no_cuda
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        if self.device == torch.device("cuda"):
            self.logging("Using GPU")
        else:
            self.logging("Using CPU")

        self.train_step = 0
        self.n_sequences_total = 0
        self.init_hours = 0
        self.epoch = 0
        self.init_time = time.time()
        self.hours_total = 0
        # best_train_loss = float("inf")

        # Load data
        n_bins = args.n_emotion_bins if args.conditioning == "discrete_token" and \
             not args.regression else None

        conditional = args.conditioning != "none" or args.regression

        # Preprocessing
        train_feats, test_feats = preprocess_features(
            "../data_files/lpd_5/features.csv", args.data_folder,
            n_bins=n_bins, conditional=conditional, 
            use_labeled_only=not args.full_dataset)

        if args.exhaustive_eval:
            # Evaluate using ENTIRE test set
            train_dataset = []
            test_dataset = LoaderExhaustive(args.data_folder, test_feats, args.tgt_len, args.conditioning,
                max_samples=args.n_samples, regression=args.regression, 
                always_use_discrete_condition=args.always_use_discrete_condition)
        else:
            train_dataset = Loader(args.data_folder, train_feats, args.tgt_len, args.conditioning,
                regression=args.regression, always_use_discrete_condition=args.always_use_discrete_condition)
            test_dataset = Loader(args.data_folder, test_feats, args.tgt_len, args.conditioning,
                regression=args.regression, always_use_discrete_condition=args.always_use_discrete_condition)

        if args.regression_dir is not None:
            # Perform emotion regression on generated samples
            train_dataset = []
            test_dataset = LoaderGenerations(args.regression_dir, args.tgt_len)

        self.null_condition = torch.FloatTensor([np.nan, np.nan]).to(self.device)

        self.maps = test_dataset.get_maps()
        self.pad_idx = test_dataset.get_pad_idx()

        self.vocab_size = test_dataset.get_vocab_len()
        args.vocab_size = self.vocab_size
        self.logging(f"Number of tokens: {self.vocab_size}")

        if args.exhaustive_eval or args.regression_dir is not None:
            self.train_loader = []
        else:
            self.train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=not args.debug,
                                                    num_workers=args.num_workers, collate_fn=filter_collate,
                                                    pin_memory=not args.no_cuda, drop_last=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, args.batch_size, shuffle=False,
                                                    num_workers=args.num_workers, collate_fn=filter_collate,
                                                    pin_memory=not args.no_cuda and args.regression_dir is None, 
                                                    drop_last=True)
        print(f"Data loader lengths\nTrain: {len(train_dataset)}")
        if not args.overfit:
            print(f"Test:{len(test_dataset)}")
        
        self.gen_dir = os.path.join(args.work_dir, "generations", "training")

        # Automatic mixed precision
        self.amp = not args.no_amp and self.device == torch.device('cuda') 

        if self.amp:
            self.logging("Using automatic mixed precision")
        else:
            self.logging("Using float32")

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.init_model()   # Build the model

        if not args.debug:
            # Save mappings
            os.makedirs(self.gen_dir, exist_ok=True)
            torch.save(self.maps, os.path.join(args.work_dir, "mappings.pt"))

        self.csv_writer = CsvWriter(os.path.join(args.work_dir, "performance.csv"),
            ["epoch", "step", "hour", "lr", "trn_loss", "val_loss", 
            #  "val_l1_v", "val_l1_a"
             ],
            in_path=self.csv_in, debug=args.debug)

        args.n_all_param = sum([p.nelement() for p in self.model.parameters()])

        self.model = self.model.to(self.device)

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx).to(self.device)
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
              
        #### scheduler
        if args.scheduler == '--':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                args.max_step, eta_min=args.eta_min)
        elif args.scheduler == 'dev_perf':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                factor=args.decay_rate, patience=args.patience, min_lr=args.lr_min)
        elif args.scheduler == 'constant':
            pass
        elif args.scheduler == 'cyclic':
            self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer,
                args.lr_min, args.lr_max, verbose=False, cycle_momentum=False)

        # Print log
        if not args.debug:
            self.logging('=' * 120)
            for k, v in args.__dict__.items():
                self.logging('    - {} : {}'.format(k, v))
            self.logging('=' * 120)
        self.logging('#params = {}'.format(args.n_all_param))

        now = datetime.datetime.now()
        now = now.strftime("%d-%m-%Y %H:%M")
        self.logging(f"Run started at {now}")
        self.once = True

    def init_model(self):
        # Initialize model
        if args.restart_dir:
            # Load existing model
            config = torch.load(os.path.join(args.restart_dir, "../model_config.pt"))
            self.model, config = build_model(None, load_config_dict=config)
            self.model = self.model.to(self.device)

            model_fp = os.path.join(args.restart_dir, 'model.pt')
            optimizer_fp = os.path.join(args.restart_dir, 'optimizer.pt')
            stats_fp = os.path.join(args.restart_dir, 'stats.pt')
            scaler_fp = os.path.join(args.restart_dir, 'scaler.pt')

            self.model.load_state_dict(
                torch.load(model_fp, map_location=lambda storage, loc: storage))
            self.logging(f"Model loaded from {model_fp}")

            self.csv_in = os.path.join(args.restart_dir, 'performance.csv')
        else:
            # Build model from scratch
            self.csv_in = None
            self.model, config = build_model(vars(args))
            self.model = self.model.to(self.device)

        # save model configuration for later load
        if not args.debug:
            torch.save(config, os.path.join(args.work_dir, "model_config.pt"))
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        # Load self.optimizer if necessary
        if args.restart_dir:
            if os.path.exists(optimizer_fp):
                try:
                    self.optimizer.load_state_dict(
                        torch.load(optimizer_fp, map_location=lambda storage, loc: storage))
                except:
                    pass
            else:
                print('Optimizer was not saved. Start from scratch.')

            try:
                stats = torch.load(stats_fp)
                self.train_step = stats["step"]
                self.init_hours = stats["hour"]
                self.epoch = stats["epoch"]
                self.n_sequences_total = stats["sample"]
            except:
                self.train_step = 0
                self.init_hours = 0
                self.epoch = 0
                self.n_sequences_total = 0
            
            if os.path.exists(scaler_fp) and not args.reset_scaler:
                try:
                    self.scaler.load_state_dict(torch.load(scaler_fp))
                except:
                    pass

            if args.overwrite_lr:
                # New learning rate
                for p in self.optimizer.param_groups:
                    p['lr'] = args.lr
    
    ###############################################################################
    # EVALUATION
    ###############################################################################

    def evaluate(self):

        # Turn on evaluation mode which disables dropout.
        self.model.eval()

        # Evaluation
        topk = (1, 5)   # find accuracy for top-1 and top-5
        n_elements_total, n_sequences_total, total_loss = 0, 0, 0.
        total_accs = {"l1_v": 0., "l1_a": 0., "l1_mean": 0., "l1_mean_normal":0
            } if args.regression else {k: 0. for k in topk}
        with torch.no_grad():
            n_batches = len(self.test_loader)
            loader = enumerate(self.test_loader)
            # if args.exhaustive_eval or args.regression:
            #     loader = tqdm(loader, total=n_batches)
            for i, (input_, condition, target) in loader:
                if args.max_eval_step > 0 and i >= args.max_eval_step:
                    break
                if input_ != []:
                    input_ = input_.to(self.device)
                    condition = condition.to(self.device)
                    if not args.regression:
                        target = target.to(self.device)
                    loss, pred = self.forward_pass(input_, condition, target)
                    if args.regression:
                        pred = torch.clamp(pred, min=-1.0, max=1.0)
                        loss = self.l1_loss(pred, condition)
                        l1_v = self.l1_loss(pred[:, 0], condition[:, 0]).item()
                        l1_a = self.l1_loss(pred[:, 1], condition[:, 1]).item()
                        accuracies = {"l1_v": l1_v, "l1_a": l1_a,
                                      "l1_mean": (l1_v + l1_a) / 2,
                                      "l1_mean_normal": (l1_v + l1_a) / 2 / 2}
                        n_elements = pred[:, 0].numel()
                    else:
                        accuracies = accuracy(pred, target, topk=topk, ignore_index=self.pad_idx)
                        n_elements = input_.numel()
                    n_sequences = input_.size(0)
                    total_loss += n_elements * loss.item()
                    for key, value in accuracies.items():
                        total_accs[key] += n_elements * value
                    n_elements_total += n_elements
                    n_sequences_total += n_sequences

            avg_loss = total_loss / n_elements_total
            avg_accs = {k: v/n_elements_total for k, v in total_accs.items()}
            if args.exhaustive_eval:
                print(f"Total number of sequences: {n_sequences_total}")

            return avg_loss, avg_accs

    def forward_pass(self, input_, condition, target):

        input_ = input_.to(self.device)
        condition = condition.to(self.device)

        with torch.cuda.amp.autocast(enabled=self.amp):
            if args.regression:
                output = self.model(input_)
                loss = self.l1_loss(output, condition)
            else:
                target = target.to(self.device)
                output = self.model(input_, condition)
                output_flat = output.reshape(-1, output.size(-1))
                target = target.reshape(-1)
                loss = self.ce_loss(output_flat, target)

        return loss, output

    def train(self):
        # Turn on training mode which enables dropout.
        self.model.train()

        train_loss = 0
        best_train_loss = float("inf")
        n_elements_total = 0
        train_interval_start = time.time()
        once = True

        while True:
            for input_, condition, target in self.train_loader:
                self.model.train()
                if input_ != []:

                    loss, _ = self.forward_pass(input_, condition, target)
                    loss_val = loss.item()
                    loss /= args.accumulate_step

                    n_elements = input_.numel()
                    if not math.isnan(loss_val):
                        train_loss += n_elements * loss_val
                        n_elements_total += n_elements
                    self.n_sequences_total += input_.size(0)

                    self.scaler.scale(loss).backward()

                    if self.train_step % args.accumulate_step == 0:
                        self.scaler.unscale_(self.optimizer)
                        if args.clip > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.model.zero_grad()

                    if args.scheduler != "constant":
                        # linear warmup stage
                        if self.train_step <= args.warmup_step:
                            curr_lr = args.lr * self.train_step / args.warmup_step
                            self.optimizer.param_groups[0]['lr'] = curr_lr
                        else:
                            self.scheduler.step()

                if (self.train_step % args.gen_step == 0) and self.train_step > 0 and not args.regression:
                    # Generate and save samples
                    with torch.no_grad():
                        self.model.eval()
                        if args.max_gen_input_len > 0:
                            max_input_len = args.max_gen_input_len
                        else:
                            max_input_len = args.tgt_len

                        primers = [["<START>"]]
                        # Use fixed set of conditions
                        if args.conditioning == "none":
                            discrete_conditions = None
                            continuous_conditions = None
                            primers = [["<START>"] for _ in range(4)]

                        elif args.conditioning == "discrete_token":
                            discrete_conditions = [
                                ["<V-2>", "<A-2>"],
                                ["<V-2>", "<A2>"],
                                ["<V2>", "<A-2>"],
                                ["<V2>", "<A2>"],
                                ]
                            continuous_conditions = None
                        elif args.conditioning in ["continuous_token", "continuous_concat"]:
                            discrete_conditions = None
                            continuous_conditions = [
                                        [-0.8, -0.8], 
                                        [-0.8, 0.8], 
                                        [0.8, -0.8],
                                        [0.8, 0.8]
                                        ]
                            
                        generate(self.model, self.maps, self.device, self.gen_dir, args.conditioning, 
                            debug=args.debug, verbose=False, amp=self.amp, discrete_conditions=discrete_conditions,
                            continuous_conditions=continuous_conditions, min_n_instruments=1,
                            gen_len=args.gen_len, max_input_len=max_input_len, 
                            step=str(self.train_step), primers=primers,
                            temperatures=[args.temp_note, args.temp_rest])
                        
                if (self.train_step % args.log_step == 0) and self.train_step > 0 and n_elements_total > 0:
                    # Print log
                    if once:
                        print(memory())
                        once = False
                    cur_loss = train_loss / n_elements_total
                    elapsed_total = time.time() - self.init_time
                    elapsed_interval = time.time() - train_interval_start
                    hours_elapsed = elapsed_total / 3600.0
                    hours_total = self.init_hours + hours_elapsed
                    lr = self.optimizer.param_groups[0]['lr']
                    log_str = '| Epoch {:3d} step {:>8d} | {:>6d} sequences  | {:>3.1f} h | lr {:.2e} ' \
                            '| ms/batch {:4.0f} | ms/sample {:4.0f} | loss {:7.4f}'.format(
                        self.epoch, self.train_step, self.n_sequences_total, hours_total, lr,
                        elapsed_interval * 1000 / args.log_step,
                        elapsed_interval * 1000 / args.log_step / args.batch_size, cur_loss)
                    self.logging(log_str)
                    self.csv_writer.update({"epoch": self.epoch, "step": self.train_step, "hour": hours_total,
                                            "lr": lr, "trn_loss": cur_loss, "val_loss": np.nan,
                                            # "val_l1_v": np.nan, "val_l1_a": np.nan
                                            })
                    
                    if not args.debug:  # Save model and learning curves
                        plot_performance(os.path.join(args.work_dir, "performance.csv"))
                        save_dir = os.path.join(args.work_dir, "latest_model")
                        self.save_model(save_dir)
                        if train_loss < best_train_loss:
                            best_train_loss = train_loss
                            save_dir = os.path.join(args.work_dir, "best_trn_model")
                            self.save_model(save_dir)
                    
                    train_loss = 0
                    n_elements_total = 0
                    self.n_good_output, self.n_nan_output = 0, 0
                    train_interval_start = time.time() 
                    
                if (self.train_step % args.eval_step == 0) and self.train_step > 0:
                    # Evaluate model
                    val_loss, val_acc = self.evaluate()
                    elapsed_total = time.time() - self.init_time
                    hours_elapsed = elapsed_total / 3600.0
                    self.hours_total = self.init_hours + hours_elapsed
                    lr = self.optimizer.param_groups[0]['lr']
                    self.logging('-' * 120)
                    log_str = '| Eval  {:3d} step {:>8d} | now: {} | {:>3.1f} h' \
                            '| valid loss {:7.4f} | ppl {:5.3f}'.format(
                        self.train_step // args.eval_step, self.train_step,
                        time.strftime("%d-%m - %H:%M"), self.hours_total, 
                        val_loss, math.exp(val_loss))
                    if args.regression:
                        log_str += " | l1_v: {:5.3f} | l1_a: {:5.3f}".format(
                            val_acc["l1_v"], val_acc["l1_a"])

                    self.csv_writer.update({"epoch": self.epoch, "step": self.train_step, "hour": self.hours_total,
                                                "lr": lr, "trn_loss": np.nan, "val_loss": val_loss})

                    self.logging(log_str)
                    self.logging('-' * 120)

                    # dev-performance based learning rate annealing
                    if args.scheduler == 'dev_perf':
                        self.scheduler.step(val_loss)

                    if not args.debug:  # Save learning curves
                        plot_performance(os.path.join(args.work_dir, "performance.csv"))
                        x=0

                if self.train_step >= args.max_step:
                    break
                self.train_step += 1
            self.epoch += 1
            if self.train_step >= args.max_step:
                break

    def save_model(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        model_fp = os.path.join(save_dir, 'model.pt')
        torch.save(self.model.state_dict(), model_fp)
        optimizer_fp = os.path.join(save_dir, 'optimizer.pt')
        torch.save(self.optimizer.state_dict(), optimizer_fp)
        scaler_fp = os.path.join(save_dir, 'scaler.pt')
        torch.save(self.scaler.state_dict(), scaler_fp)
        torch.save({"step": self.train_step, "hour": self.hours_total, "epoch": self.epoch,
                    "sample": self.n_sequences_total}, 
                    os.path.join(save_dir, 'stats.pt'))


    def run(self):

        # Loop over epochs.
        # At any point you can hit Ctrl + C to break out of training early.
        try:
            if args.exhaustive_eval or args.regression_dir is not None:
                self.logging("Exhaustive evaluation")
                if args.regression_dir is not None:
                    self.logging(f"For regression on folder {args.regression_dir}")
                loss, accuracies = self.evaluate()
                perplexity = math.exp(loss)
                elapsed_total = time.time() - self.init_time
                hours_elapsed = elapsed_total / 3600.0
                msg = f"Loss: {loss:7.4f}, ppl: {perplexity:5.2f}"
                for k, v in accuracies.items():
                    if args.regression:
                        msg += f", {k}: {v:7.4f}"
                    else:
                        msg += f", top{k:1.0f}: {v:7.4f}"
                msg += f", hours: {hours_elapsed:3.1f}"
                self.logging(msg)
            else:
                while True:
                    self.train()
                    if self.train_step >= args.max_step:
                        self.logging('-' * 120)
                        self.logging('End of training')
                        break
        except KeyboardInterrupt:
            self.logging('-' * 120)
            self.logging('Exiting from training early')

if __name__ == "__main__":
    runner = Runner()
    runner.run()