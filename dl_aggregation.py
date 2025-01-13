from options import args_parser, save_options
# load lora matrices and original weights
args = args_parser()
save_options(args)

# apply dynamic weighting algos and update global model
