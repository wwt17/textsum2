train_dataset_hparams = {
    "name": "train_dataset",
    "num_epochs": None,
    "batch_size": None, # will be set by batcher
    "shuffle": True,
    "source_dataset": {
        "files": ["./data/train.article"],
        "vocab_file": "./data/vocab",
        "bos_token": "<s>",
        "eos_token": "<\s>",
        "length_filter_mode": "discard",
        "max_seq_length": None # will be set by batcher
    },
    "target_dataset": {
        "files": ["./data/train.title"],
        "vocab_share": True,
        "processing_share": True,
        "embedding_init_share": True,
        "bos_token": "<s>",
        "eos_token": "<\s>",
        #"vocab_file": "./data/train.title.vocab.txt",
        #"eos_token": "<TARGET_EOS>",
    },
    "allow_smaller_final_batch": False,
}

val_dataset_hparams = {
    "name": "val_dataset",
    "num_epochs": 1,
    "batch_size": None, # will be assigned by batcher
    "shuffle": False,
    "source_dataset": {
        "files": ["./data/valid_zhou.article"],
        "vocab_file": "./data/vocab",
        "bos_token": "<s>",
        "eos_token": "<\s>",
        "length_filter_mode": "discard",
        "max_seq_length": None # will be set by batcher
    },
    "target_dataset": {
        "files": ["./data/valid_zhou.title"],
        "vocab_share": True,
        "processing_share": True,
        "embedding_init_share": True,
        "bos_token": "<s>",
        "eos_token": "<\s>",
        #"vocab_file": "./data/train.title.vocab.txt"
    },
    "allow_smaller_final_batch": False,
}

test_dataset_hparams = val_dataset_hparams

