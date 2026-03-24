import argparse
import os
import warnings

import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import torch
import torch.multiprocessing as mp
from pytorch_lightning import Trainer, strategies
from pytorch_lightning.loggers import CSVLogger

from llamo.configuration_llamo import GraphConfig
from llamo.conformer_encoders.unimol import SimpleUniMolModel
from pipeline.config import load_config
from pipeline.data_utils.llamo_dm import LLaMoDM
from pipeline.dictionary import Dictionary
from pipeline.trainer.stage import LLaMoStage

# torch.set_default_dtype(torch.float16)

os.environ["OPENBLAS_NUM_THREADS"] = "1"
## for pyg bug
warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
## for A5000 gpus
torch.set_float32_matmul_precision(
    "medium"
)  # can be medium (bfloat16), high (tensorfloat32), highest (float32)


class MyDDPStrategy(strategies.DDPStrategy):
    def load_model_state_dict(self, checkpoint):
        assert self.lightning_module is not None
        self.lightning_module.load_state_dict(checkpoint["state_dict"], strict=False)


def main(args):
    mp.set_sharing_strategy("file_system")
    pl.seed_everything(args.seed)

    config = load_config(args.config_file)
    config = GraphConfig.from_dict(config)

    # if args.init_checkpoint:
    # model = LLaMoStage.load_from_checkpoint(args.init_checkpoint, strict=False, map_location="cpu", args=args, config=config)
    # print(f"loaded init checkpoint from {args.init_checkpoint}")
    if args.mode == "eval" and args.stage_path:
        model = LLaMoStage(args, config)
        ckpt = torch.load(args.stage_path, map_location="cpu")
        if config.encode_mode == "3d":
            new_dict = dict()
            for k, v in ckpt["state_dict"].items():
                if ".graph_model." in k:
                    new_dict[k.replace(".graph_model.", ".graph3d_model.")] = v
                elif ".abstractor." in k:
                    new_dict[k.replace(".abstractor.", ".abstractor3d.")] = v
                else:
                    new_dict[k] = v
            ckpt["state_dict"] = new_dict
        if config.llm_tune == "lora":
            model.mllm.init_peft()
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print(f"loaded model from {args.stage_path}")

    elif args.stage_path:
        model = LLaMoStage(args, config)
        # model.mllm.init_peft()

        ckpt = torch.load(args.stage_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print(f"loaded model from {args.stage_path}")

    else:
        model = LLaMoStage(args, config)

    if config.llm_tune == "lora" and args.mode != "eval":
        model.mllm.init_peft()

    print("total params:", sum(p.numel() for p in model.parameters()))

    tokenizer = model.tokenizer
    if model.uni_dict is None:
        uni_dict = Dictionary.load("./llamo/conformer_encoders/unimol_dict.txt")
        uni_dict.add_symbol("[MASK]", is_special=True)

    else:
        uni_dict = model.uni_dict

    # data
    dm = LLaMoDM(
        args.mode,
        args.num_workers,
        args.batch_size,
        args.root_train,
        args.root_eval,
        config.text_max_len,
        tokenizer,
        uni_dict,
        args,
        config,
    )

    callbacks = []
    ## fixme save only used parameters
    # callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/", every_n_epochs=10, save_top_k=-1))
    callbacks.append(
        plc.ModelCheckpoint(
            dirpath="/hub_data2/jinyoungp/checkpoints/" + args.filename + "/",
            filename="{epoch:02d}",
            every_n_epochs=args.save_every_n_epochs,
            save_last=True,
            save_top_k=-1,
            save_on_train_epoch_end=True,
        )
    )
    if len(args.devices.split(",")) > 1:
        if args.strategy_name == "fsdp":
            strategy = strategies.DDPFullyShardedNativeStrategy()
        elif args.strategy_name == "deepspeed":
            strategy = strategies.DeepSpeedStrategy(stage=3)
        else:
            strategy = MyDDPStrategy(find_unused_parameters=True, start_method="spawn")
    else:
        strategy = "ddp_find_unused_parameters_true"
        args.devices = eval(args.devices)
    logger = CSVLogger(save_dir=f"/hub_data2/jinyoungp/checkpoints/{args.filename}/")
    # trainer = Trainer.from_argparse_args(args,
    #                                      callbacks=callbacks,
    #                                      strategy=strategy,
    #                                      logger=logger,
    #                                     #  limit_train_batches=100,
    #                                      )
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=callbacks,
        strategy=strategy,
        logger=logger,
    )
    if args.mode in {"pretrain", "ft"}:
        trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)
    elif args.mode == "eval":
        trainer.fit_loop.epoch_progress.current.completed = args.caption_eval_epoch - 1
        trainer.test(model, datamodule=dm)
    else:
        raise NotImplementedError()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="test.yaml")
    parser.add_argument("--filename", type=str, default="stage2_test")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    # MM settings
    parser.add_argument("--mode", type=str, default="pretrain")
    parser.add_argument("--strategy_name", type=str, default=None)
    parser.add_argument("--iupac_prediction", action="store_true", default=False)
    parser.add_argument("--ckpt_path", type=str, default=None)

    # parser = Trainer.add_argparse_args(parser)
    parser = LLaMoStage.add_model_specific_args(parser)
    parser = LLaMoDM.add_model_specific_args(parser)
    parser = SimpleUniMolModel.add_args(parser)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=str, default="0,1,2,3")
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    #######
    parser.add_argument("--not_use_kg", action="store_true", default=False)
    parser.add_argument("--noise_step", type=int, default=999)
    parser.add_argument("--noise_type", type=str, choices=["cd"], default="cd")
    parser.add_argument("--use_vcd", action="store_true", default=False)
    parser.add_argument("--changed", action="store_true", default=False)
    parser.add_argument("--cd_alpha", type=float, default=0.5)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--project", type=str, default="linear")
    parser.add_argument("--mlp_depth", type=int, default=2)

    args = parser.parse_args()

    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, "=", v)
    print("=========================================")
    return args


if __name__ == "__main__":
    main(get_args())
