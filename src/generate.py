from argparse import ArgumentParser
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import datetime
from utils import get_n_instruments
from models.build_model import build_model
from data.data_processing_reverse import ind_tensor_to_mid, ind_tensor_to_str, ind_tensor_to_tuples
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def generate(model, conditions_tensor, maps, device, out_dir, 
                        continuous_token=False, repeat_penalty=True,
                         max_input_len=1024, amp=True, note=None, step=None,
                         gen_len=2048, temperatures=[1.5,0.7], top_k=-1, 
                         top_p=0.7, debug=False, varying_condition=None,
                         verbose=False, primers=[["<START>"]],
                         min_n_instruments=3, label_conditions=None):

    os.makedirs(out_dir, exist_ok=True)
    
    if not torch.is_tensor(conditions_tensor):
        conditions_tensor = torch.Tensor(conditions_tensor)
    model = model.to(device)
    model.eval()

    assert len(temperatures) in (1, 2)

    conditioning = torch.equal(conditions_tensor, conditions_tensor)   # doesnt have NaNs
    if varying_condition is not None:
        assert len(primers) == 1 and not conditioning
        batch_size = varying_condition[0].size(0)
    elif conditioning:
        assert len(primers) == 1 and varying_condition is None
        batch_size = conditions_tensor.size(0)
    else:
        batch_size = len(primers)

    
    
    # will be used to penalize repeats
    repeat_counts = [0 for _ in range(batch_size)]

    conditions_tensor = conditions_tensor.to(device)

    exclude_symbols = [symbol for symbol in maps["tuple2idx"].keys() if symbol[0] == "<"]

    # will have generated symbols and indices
    gen_song_tensor = torch.LongTensor([]).to(device)

    if not isinstance(primers, list):
        primers = [[primers]]
    primer_inds = [[maps["tuple2idx"][symbol] for symbol in primer] \
        for primer in primers]

    gen_inds = torch.LongTensor(primer_inds)

    if continuous_token:
        max_input_len -= conditions_tensor.size(1)

    null_conditions_tensor = torch.FloatTensor([np.nan, np.nan]).to(device)
    if varying_condition is not None:
        varying_condition[0] = varying_condition[0].to(device)
        varying_condition[1] = varying_condition[1].to(device)

    if len(primers) == 1:
        gen_inds = gen_inds.repeat(batch_size, 1)
        null_conditions_tensor = null_conditions_tensor.repeat(batch_size, 1)
    gen_inds = gen_inds.t().to(device)

    with torch.no_grad():
        i = 0
        while i < gen_len:
            i += 1
            if verbose:
                print(gen_len - i, end=" ", flush=True)

            gen_song_tensor = torch.cat((gen_song_tensor, gen_inds), 0)

            input_ = gen_song_tensor
            if len(gen_song_tensor) > max_input_len:
                input_ = input_[-max_input_len:, :]

            # INTERPOLATED CONDITIONS
            if varying_condition is not None:
                valences = varying_condition[0][:, i-1]
                arousals = varying_condition[1][:, i-1]
                conditions_tensor = torch.cat([valences[:, None], arousals[:, None]], dim=-1)

            # =================================================================
            # RUN MODEL
            # =================================================================
            with torch.cuda.amp.autocast(enabled=amp):
                input_ = input_.t()
                output = model(input_, conditions_tensor)
                output = output.permute((1, 0, 2))
                
            # =================================================================
            # PROCESS OUTPUT, GET PREDICTED TOKEN
            # =================================================================
            output = output[-1, :, :]     # Select last timestep

            # zeroing nans
            output[output != output] = 0
            # if everything becomes zero
            if torch.all(output == 0) and verbose:
                print("All predictions were NaN during generation")
                output = torch.ones(output.shape).to(device)

            # exclude certain symbols
            for symbol_exclude in exclude_symbols:
                try:
                    idx_exclude = maps["tuple2idx"][symbol_exclude]
                    output[:, idx_exclude] = -float("inf")
                except:
                    pass
            
            effective_temps = []
            for j in range(batch_size):
                gen_idx = gen_inds[0, j].item()
                gen_tuple = maps["idx2tuple"][gen_idx]
                effective_temp = temperatures[0]
                if isinstance(gen_tuple, tuple):
                    gen_event = maps["idx2event"][gen_tuple[0]]
                    if "TIMESHIFT" in gen_event:
                        effective_temp = temperatures[1]
                effective_temps.append(effective_temp)

            temp_tensor = torch.Tensor([effective_temps]).to(device)

            output = F.log_softmax(output, dim=-1)

            # Add repeat penalty to temperature
            if repeat_penalty:
                repeat_counts_array = torch.Tensor(repeat_counts).to(device)
                repeat_penalties = torch.maximum(torch.zeros_like(repeat_counts_array, device=device), 
                    torch.log((repeat_counts_array+1)/4)/5) * temp_tensor
                temp_tensor += repeat_penalties

            # Apply temperature
            output /= temp_tensor.t()
            
            # top-k
            if top_k <= 0 or top_k > output.size(-1): 
                top_k_eff = output.size(-1)
            else:
                top_k_eff = top_k
            output, top_inds = torch.topk(output, top_k_eff)

            # top-p
            if top_p > 0 and top_p < 1:
                cumulative_probs = torch.cumsum(F.softmax(output, dim=-1), dim=-1)
                remove_inds = cumulative_probs > top_p
                remove_inds[:, 0] = False   # at least keep top value
                output[remove_inds] = -float("inf")

            output = F.softmax(output, dim=-1)
        
            # Sample from probabilities
            inds_sampled = torch.multinomial(output, 1, replacement=True)
            gen_inds = top_inds.gather(1, inds_sampled).t()

            # Update repeat counts
            num_choices = torch.sum((output > 0).int(), -1)
            for j in range(batch_size):
                if num_choices[j] <= 2: repeat_counts[j] += 1
                else: repeat_counts[j] = repeat_counts[j] // 2
                
        # =================================================================
        # CONVERT TO MIDI AND SAVE
        # =================================================================
        
        for i in range(gen_song_tensor.size(-1)):
            if step is None:
                now = datetime.datetime.now()
                out_file_path = now.strftime("%Y_%m_%d_%H_%M_%S")
            else:
                out_file_path = step

            out_file_path += "_" + str(i)

            if note is not None:
                out_file_path += "_" + note

            if varying_condition is not None:
                conditions_interp =  [varying_condition[0][i, 0].item(), varying_condition[0][i, -1].item(), 
                                      varying_condition[1][i, 0].item(), varying_condition[1][i, -1].item()]
                # convert to string
                conditions_interp = [str(round(c, 2)).replace(".", "") for c in conditions_interp]
                out_file_path += f"_V{conditions_interp[0]}_{conditions_interp[1]}_A{conditions_interp[2]}_{conditions_interp[3]}"
            elif conditioning:
                condition = conditions_tensor[i, :].tolist()
                # convert to string
                condition = [str(round(c, 2)).replace(".", "") for c in condition]
                out_file_path += f"_V{condition[0]}_A{condition[1]}"
            elif label_conditions is not None:
                condition = label_conditions[i, :]
                condition = [str(round(c.item(), 2)).replace(".", "") for c in condition]
                out_file_path += f"_V{condition[0]}_A{condition[1]}"

            primer = primers[0] if len(primers) == 1 else primers[i]
            if len(primer) > 1:
                # conditioned on emotion tokens
                # write to output name
                out_file_path += "_" + primer[0][1:-1] + "_" + primer[1][1:-1]
            
            temp_str = "_".join([str(round(t, 2)).replace(".", "") for t in temperatures])
            top_p_str = str(top_p).replace(".", "")
            top_k_str = str(top_k).replace(".", "")
            out_file_path += f"_temp{temp_str}"
            if top_k > 0 and top_k < output.size(-1):
                out_file_path += f"_k{top_k_str}"
            if top_p > 0 and top_p < 1:
                out_file_path += f"_p{top_p_str}"
            if repeat_penalty:
                out_file_path += "_penalty"
            out_file_path += ".mid"
            out_path_mid = os.path.join(out_dir, out_file_path)

            symbols = ind_tensor_to_str(gen_song_tensor[:, i], maps["idx2tuple"], maps["idx2event"])
            n_instruments = get_n_instruments(symbols)
            if n_instruments >= min_n_instruments:
                mid = ind_tensor_to_mid(gen_song_tensor[:, i], maps["idx2tuple"], maps["idx2event"], verbose=verbose)
                # tuples = ind_tensor_to_tuples(gen_song_tensor[:, i], maps["idx2tuple"])
                if conditioning:
                    save_condition = conditions_tensor[i, :].cpu()
                elif label_conditions is not None:
                    save_condition = label_conditions[i, :].cpu()
                else:
                    save_condition = primer
                save_pt = {"inds": gen_song_tensor[:, i].cpu(), "condition": save_condition}

                out_path_txt = "txt_" + out_file_path.replace(".mid", ".txt")
                out_path_txt = os.path.join(out_dir, out_path_txt)
                out_path_inds = "inds_" + out_file_path.replace(".mid", ".pt")
                out_path_inds = os.path.join(out_dir, out_path_inds)

                if not debug:
                    mid.write(out_path_mid)
                    if verbose:
                        print(f"Saved to {out_path_mid}")
                    with open(out_path_txt, "w") as f:
                        f.write("\n".join(symbols))
                    torch.save(save_pt, out_path_inds)
            else:
                print(f"Only has {n_instruments}, not saving.")



if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_model_dir = os.path.abspath(os.path.join(script_dir, 'model'))
    code_utils_dir = os.path.join(code_model_dir, 'utils')
    sys.path.extend([code_model_dir, code_utils_dir])

    parser = ArgumentParser()

    parser.add_argument('--model_dir', type=str, help='Directory with model', required=True)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--num_runs', type=int, help='Number of runs', default=1)
    # parser.add_argument('--min_gen_len', type=int, help='Max generation len', default=-1)
    parser.add_argument('--gen_len', type=int, help='Max generation len', default=2048)
    parser.add_argument('--max_input_len', type=int, help='Max input len', default=1216)
    # parser.add_argument('--primer_len', type=int, help='Length of primer', default=1024)
    # parser.add_argument('--primer_track', type=str, help='Track ID of primer', default=None)
    parser.add_argument('--temp', type=float, nargs='+', help='Generation temperature', default=[1.5, 0.7])
    parser.add_argument('--topk', type=int, help='Top-k sampling', default=-1)
    parser.add_argument('--topp', type=float, help='Top-p sampling', default=0.7)
    parser.add_argument('--debug', action='store_true')
    # parser.add_argument('--custom_primer', action='store_true')
    parser.add_argument('--save_probs', action='store_true')
    parser.add_argument('--seed', action='store_true')
    parser.add_argument('--no_amp', action='store_true', help="Disable automatic mixed precision")
    parser.add_argument("--conditioning", type=str, required=True,
                    choices=["none", "discrete_token", "continuous_token",
                             "continuous_concat"], help='Conditioning type')
    parser.add_argument('--no_penalty', action='store_true')
    parser.add_argument("--quiet", action='store_true')
    parser.add_argument("--smooth_change", action='store_true')
    parser.add_argument("--abrupt_change", action='store_true')
    parser.add_argument('--batch_size', type=int, help='Batch size', default=4)
    # parser.add_argument('--condition_value', type=float, help='Conditioning value', default=0.8)
    parser.add_argument('--valence', type=float, help='Conditioning valence value', default=[0], nargs='+')
    parser.add_argument('--arousal', type=float, help='Conditioning arousal value', default=[0], nargs='+')
    parser.add_argument('--max_condition', type=float, default=[0.8])#, nargs='+')
    parser.add_argument('--constant_condition', type=float, default=[0])#, nargs='+')
    parser.add_argument("--batch_gen_dir", type=str, default="")
    parser.add_argument("--keep_unchanged", type=float, default=0)

    args = parser.parse_args()

    assert args.batch_size == len(args.valence) or args.smooth_change or args.abrupt_change

    assert not (args.abrupt_change and args.smooth_change)

    assert len(args.valence) == len(args.arousal)

    if args.seed:
        torch.manual_seed(666)
        torch.cuda.manual_seed(666)

    if args.save_probs:
        args.temp = 0
        print("Saving probabilities, temperature set to 0")

    main_output_dir = "../output"
    assert os.path.exists(os.path.join(main_output_dir, args.model_dir))
    midi_output_dir = os.path.join(main_output_dir, args.model_dir, "generations", "inference")
    if args.smooth_change:
        midi_output_dir = os.path.join(midi_output_dir, "smooth_" + args.batch_gen_dir)
    elif args.abrupt_change:
        midi_output_dir = os.path.join(midi_output_dir, "abrupt_" + args.batch_gen_dir)
    elif args.batch_gen_dir != "":
        midi_output_dir = os.path.join(midi_output_dir, args.batch_gen_dir)

    os.makedirs(midi_output_dir, exist_ok=True)

    # model_fp = os.path.join(main_output_dir, args.model_dir, 'best_model.pt')
    model_fp = os.path.join(main_output_dir, args.model_dir, 'model.pt')
    mappings_fp = os.path.join(main_output_dir, args.model_dir, 'mappings.pt')
    config_fp = os.path.join(main_output_dir, args.model_dir, 'model_config.pt')

    if os.path.exists(mappings_fp):
        maps = torch.load(mappings_fp)
    else:
        raise ValueError("Mapping file not found.")

    start_symbol = "<START>"
    
    n_emotion_bins = 5

    valence_symbols = []
    arousal_symbols = []

    emotion_bins = np.linspace(-1-1e-12, 1+1e-12, num=n_emotion_bins+1)
    if n_emotion_bins % 2 == 0:
        bin_ids = list(range(-n_emotion_bins//2, 0)) + list(range(1, n_emotion_bins//2+1))
    else:
        bin_ids = list(range(-(n_emotion_bins-1)//2, (n_emotion_bins-1)//2 + 1))
        
    for bin_id in bin_ids:
        valence_symbols.append(f"<V{bin_id}>")
        arousal_symbols.append(f"<T{bin_id}>")

    

    # print(list(maps.keys()))
    device = torch.device('cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu')
    if device == torch.device("cuda"):
        print("Using GPU")
    else:
        print("Using CPU")

    config = torch.load(config_fp)
    model, _ = build_model(None, load_config_dict=config)
    # with open(os.path.join(args.restart_dir, 'model.pt'), 'rb') as f:
    model = model.to(device)
    if os.path.exists(model_fp):
        model.load_state_dict(torch.load(model_fp, map_location=device))
    elif os.path.exists(model_fp.replace("best_", "")):
        model.load_state_dict(torch.load(model_fp.replace("best_", ""), map_location=device))
    else:
        raise ValueError("Model not found")

    null_condition = torch.FloatTensor([np.nan, np.nan]).to(device)
    note = ""
    
    varying_condition = None
    label_conditions = None

    conditions = []
    for i in range(len(args.valence)):
        conditions.append([args.valence[i], args.arousal[i]])

    if args.conditioning == "discrete_token":
       
        primers = []
        for condition in conditions:
            valence_val, arousal_val = condition
            valence_symbol = valence_symbols[np.searchsorted(
                emotion_bins, valence_val, side="right") - 1]
            arousal_symbol = arousal_symbols[np.searchsorted(
                emotion_bins, arousal_val, side="right") - 1]
            primers.append([valence_symbol, arousal_symbol, start_symbol])
        
        conditions = null_condition
    
    elif args.conditioning == "none":
        label_conditions = torch.FloatTensor(conditions)
        conditions = null_condition
        # this is just to label the file, its not effective on generation
        
        primers = [["<START>"] for _ in range(args.batch_size)]
    # elif args.conditioning == "discrete_token":
    #     conditions = null_condition
    #     primers = []
    #     for v in ["<V-2>", "<V-1>", "<V1>", "<V2>"]:
    #         for t in ["<T-2>", "<T-1>", "<T1>", "<T2>"]:
    #             primers.append([ v, t, "<START>"])

    
    elif args.conditioning in ["continuous_token", "continuous_concat"]:
        primers = [["<START>"]]
        if args.smooth_change or args.abrupt_change:
            conditions = null_condition
            # each row: [valence_start, valence_end, arousal_start, arousal_end]
            # interpolate = torch.FloatTensor([
            #                             [-0.7,  0.7 , 0,    0],
            #                             [ 0.7, -0.7 , 0,    0],
            #                             [ 0,    0 ,  -0.7,  0.7],
            #                             [ 0,    0 ,   0.7, -0.7],
            #                             ])
            max_ = args.max_condition   # 0.8
            constant = args.constant_condition    # -0.7

            # keep_unchanged = 0    # ratio of tokens with no change
            # v_start = [-max_,     max_,      constant, constant]
            # v_end =   [ max_,    -max_,      constant, constant]
            # a_start = [ constant, constant, -max_,     max_]
            # a_end = [   constant, constant,  max_,    -max_]
            v_start = [max_,     -max_,      max_, -max_]
            v_end =   [-max_,    max_,      -max_, max_]
            a_start = [ max_, -max_, -max_,     max_]
            a_end = [   -max_, max_,  max_,    -max_]

            if args.smooth_change:
                if args.keep_unchanged <= 0:
                    # smooth interpolation
                    v_varying = torch.stack([torch.linspace(v_start[j], v_end[j], args.gen_len) \
                        for j in range(len(v_start))], dim=0)
                    a_varying = torch.stack([torch.linspace(a_start[j], a_end[j], args.gen_len) \
                        for j in range(len(a_start))], dim=0)
                    
                else:
                    # constant finish and end, smooth interpolation in middle
                    v_varying = torch.stack([
                            torch.cat((
                                        torch.ones(round(args.gen_len*args.keep_unchanged)) * v_start[j],
                                        torch.linspace(v_start[j], v_end[j], round(args.gen_len * (1-2*args.keep_unchanged))),
                                        torch.ones(round(args.gen_len*args.keep_unchanged)) * v_end[j],
                                        )) \
                                for j in range(len(a_start))], dim=0)
                    a_varying = torch.stack([
                            torch.cat((
                                        torch.ones(round(args.gen_len*args.keep_unchanged)) * a_start[j],
                                        torch.linspace(a_start[j], a_end[j], round(args.gen_len * (1-2*args.keep_unchanged))),
                                        torch.ones(round(args.gen_len*args.keep_unchanged)) * a_end[j],
                                        )) \
                                for j in range(len(a_start))], dim=0)
                note = "smooth"
            
            elif args.abrupt_change:
                # abrupt change in middle
                v_varying = torch.stack([
                    torch.cat((torch.ones(args.gen_len//2)*v_start[j], torch.ones(args.gen_len//2)*v_end[j])) \
                        for j in range(len(v_start))], dim=0)
                a_varying = torch.stack([
                    torch.cat((torch.ones(args.gen_len//2)*a_start[j], torch.ones(args.gen_len//2)*a_end[j])) \
                        for j in range(len(a_start))], dim=0)
                note = "abrupt"

            varying_condition = [v_varying, a_varying]
        else:
            conditions = torch.FloatTensor(conditions)
            # conditions = torch.FloatTensor([
            #             [-args.condition_value, args.condition_value], 
            #             [-args.condition_value, args.condition_value], 
            #             [args.condition_value, args.condition_value],
            #             [args.condition_value, args.condition_value]
            #             ])

            interpolate = None
            
        

    if len(primers) > args.batch_size:
        chunks_ = list(chunks(primers, args.batch_size))
        for _ in range(args.num_runs):
            for i, primer in enumerate(chunks_):
                # print(f"{len(chunks_) - i} runs remaining")
                generate(model, conditions, maps, device, midi_output_dir, note=note,
                     repeat_penalty=not args.no_penalty, top_p=args.topp, varying_condition=varying_condition,
                    gen_len=args.gen_len, max_input_len=args.max_input_len, amp=not args.no_amp,
                    temperatures=args.temp, top_k=args.topk, debug=args.debug, verbose=not args.quiet, primers=primers)
    else:
        for i in range(args.num_runs):
            new_note = note # + "_run" + str(i)
            # print("here")
            # print(f"{ args.num_runs - i} runs remaining")
            generate(model, conditions, maps, device, midi_output_dir, note=new_note,
                        repeat_penalty=not args.no_penalty, top_p=args.topp, varying_condition=varying_condition,
                        gen_len=args.gen_len, max_input_len=args.max_input_len, amp=not args.no_amp,
                        label_conditions=label_conditions,
                        temperatures=args.temp, top_k=args.topk, debug=args.debug, verbose=not args.quiet, primers=primers)

    # # GRID SEARCH
    # song_name = "_"
    # penalties = [False, True]
    # t1s = [0.6, 1, 1.4, 1.8]
    # t2s = [0.2, 0.5, 0.8, 1.1]
    # ks = [16, 32, 48, 64]
    # ps = [0.4, 0.6, 0.8, 1]

    # n_runs = len(penalties) * len(t1s) * len(t2s) * len(ks) * len(ps)

    # i = 0
    # for penalty in penalties:
    #     for t1 in t1s:
    #         for t2 in t2s:
    #             for k in ks:
    #                 for p in ps:
    #                     print(f"{n_runs - i} runs remaining")
    #                     generate_multi(model, conditions, maps, device, midi_output_dir,
    #                             repeat_penalty=penalty, top_p=p, song_name=song_name,
    #                             gen_len=args.gen_len, max_input_len=args.max_input_len, amp=not args.no_amp,
    #                             temp=[t1,t2], top_k=k, debug=args.debug, primers=primers)
    #                     i += 1