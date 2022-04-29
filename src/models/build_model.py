import torch.nn as nn
def set_dropout(model, rate):
    for name, child in model.named_children():
        if isinstance(child, nn.Dropout):
            child.p = rate
        set_dropout(child, rate)
    return model

def build_model(args, load_config_dict=None):

    if load_config_dict is not None:
        args = load_config_dict
        
    config = {
        "vocab_size": args["vocab_size"], 
        "num_layer": args["n_layer"], 
        "num_head": args["n_head"], 
        "embedding_dim": args["d_model"], 
        "d_inner": args["d_inner"],
        "dropout": args["dropout"],
        "d_condition": args["d_condition"],
        "max_seq": 2048,
        "pad_token": 0,
    }

    if not "regression" in list(args.keys()):
        args["regression"] = False

    if args["regression"]:
        config["output_size"] = 2
        from models.music_regression \
                import MusicRegression as MusicTransformer

    elif args["conditioning"] == "continuous_token":
        from models.music_continuous_token \
                import MusicTransformerContinuousToken as MusicTransformer
        del config["d_condition"]
    else:
        from models.music_multi \
                import MusicTransformerMulti as MusicTransformer

    model = MusicTransformer(**config)
    if load_config_dict is not None and args is not None:
        if args["overwrite_dropout"]:
            model = set_dropout(model, args["dropout"])
            rate = args["dropout"]
            print(f"Dropout rate changed to {rate}")
    return model, args
