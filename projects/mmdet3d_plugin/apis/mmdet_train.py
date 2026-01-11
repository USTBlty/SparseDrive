# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import random
import warnings

import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (
    HOOKS,
    DistSamplerSeedHook,
    EpochBasedRunner,
    Fp16OptimizerHook,
    OptimizerHook,
    build_optimizer,
    build_runner,
    get_dist_info,
)
from mmcv.utils import build_from_cfg

from mmdet.core import EvalHook

from mmdet.datasets import build_dataset, replace_ImageToTensor
from mmdet.utils import get_root_logger
import time
import os.path as osp
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.core.evaluation.eval_hooks import (
    CustomDistEvalHook,
)
from projects.mmdet3d_plugin.datasets import custom_build_dataset


def custom_train_detector(
    model,
    dataset,
    cfg,
    distributed=False,
    validate=False,
    timestamp=None,
    meta=None,
):
    # 1. 初始化日志记录器
    logger = get_root_logger(cfg.log_level)

    # 2. 准备数据加载器 (Data Loaders)
    # 确保 dataset 是列表格式（即使只有一个数据集）
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    # assert len(dataset)==1s
    
    # 处理 imgs_per_gpu 参数的兼容性问题（老版本 mmdet 的遗留问题）
    if "imgs_per_gpu" in cfg.data:
        logger.warning(
            '"imgs_per_gpu" is deprecated in MMDet V2.0. '
            'Please use "samples_per_gpu" instead'
        )
        if "samples_per_gpu" in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f"={cfg.data.imgs_per_gpu} is used in this experiments"
            )
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f"{cfg.data.imgs_per_gpu} in this experiments"
            )
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    # 确定 Runner 类型（默认 EpochBasedRunner，但 SparseDrive 常用 IterBasedRunner）
    if "runner" in cfg:
        runner_type = cfg.runner["type"]
    else:
        runner_type = "EpochBasedRunner"
        
    # 构建 DataLoader：这是数据从磁盘到内存再到 Tensor 的流水线
    # 包含了 Sampler（决定取哪个数据）和 Worker（决定起几个进程搬运数据）
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu, # batch_size
            cfg.data.workers_per_gpu, # num_workers
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            nonshuffler_sampler=dict(
                type="DistributedSampler"
            ),  # dict(type='DistributedSampler'),
            runner_type=runner_type,
        )
        for ds in dataset
    ]

    # 3. 将模型放置到 GPU 上 (Model Parallelization)
    if distributed:
        # 分布式训练 (DDP)
        find_unused_parameters = cfg.get("find_unused_parameters", False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )

    else:
        # 单卡或多卡数据并行 (DP)
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids
        )

    # 4. 构建优化器 (Optimizer)
    # 根据 config 里的 optimizer 配置（如 AdamW）创建优化器，绑定模型参数
    optimizer = build_optimizer(model, cfg.optimizer)

    # 5. 构建 Runner (执行器)
    # Runner 是 mmdet 的核心调度器，负责管理训练循环
    if "runner" not in cfg:
        cfg.runner = {
            "type": "EpochBasedRunner",
            "max_epochs": cfg.total_epochs,
        }
        warnings.warn(
            "config is now expected to have a `runner` section, "
            "please set `runner` in your config.",
            UserWarning,
        )
    else:
        if "total_epochs" in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta,
        ),
    )

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # 6. 设置混合精度训练 (FP16)
    # 如果开启，会包装优化器以支持 loss scaling
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed
        )
    elif distributed and "type" not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # 7. 注册通用 Hooks
    # 包括：学习率调整 (lr_config)、优化器步进 (optimizer_config)、
    # 模型保存 (checkpoint_config)、日志打印 (log_config) 等
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get("momentum_config", None),
    )

    # register profiler hook
    # trace_config = dict(type='tb_trace', dir_name='work_dir')
    # profiler_config = dict(on_trace_ready=trace_config)
    # runner.register_profiler_hook(profiler_config)

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # 8. 注册评估 Hook (Evaluation)
    # 如果开启验证 (validate=True)，则构建验证集 DataLoader 并注册 EvalHook
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop("samples_per_gpu", 1)
        if val_samples_per_gpu > 1:
            assert False
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline
            )
        val_dataset = custom_build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            nonshuffler_sampler=dict(type="DistributedSampler"),
        )
        eval_cfg = cfg.get("evaluation", {})
        eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"
        eval_cfg["jsonfile_prefix"] = osp.join(
            "val",
            cfg.work_dir,
            time.ctime().replace(" ", "_").replace(":", "_"),
        )
        # 这里的 CustomDistEvalHook 可能是作者修改此文件的主要原因之一
        eval_hook = CustomDistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # 9. 注册用户自定义 Hooks
    if cfg.get("custom_hooks", None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(
            custom_hooks, list
        ), f"custom_hooks expect list type, but got {type(custom_hooks)}"
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), (
                "Each item in custom_hooks expects dict type, but got "
                f"{type(hook_cfg)}"
            )
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop("priority", "NORMAL")
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    # 10. 加载 Checkpoint (如果需要恢复训练)
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
        
    # 11. 启动训练循环！
    runner.run(data_loaders, cfg.workflow)
