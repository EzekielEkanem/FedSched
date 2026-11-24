#!/usr/bin/env python3

"""
FedSched PEFT Worker
====================

A controllable CPU-based PyTorch/PEFT workload for evaluating FedSched.

Goals:
    1. Behave like a background federated-learning client.
    2. Generate sustained CPU + memory activity.
    3. Be externally controllable with:
           - cgroups v2
           - SIGSTOP / SIGCONT
           - SIGTERM / SIGINT
    4. Periodically checkpoint LoRA adapters.
    5. Emit machine-readable telemetry.
    6. Support multiple independent "FL clients".

Example:
    python peft_worker.py \
        --client-id client_01 \
        --threads 4 \
        --batch-size 4 \
        --seq-length 128 \
        --epochs 100
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)


# ---------------------------------------------------------------------------
# Global control state
# ---------------------------------------------------------------------------

shutdown_requested = False


def request_shutdown(signum, frame):
    """
    Graceful shutdown handler.

    SIGSTOP cannot be caught by a process. That is exactly what we want:
    FedSched can freeze this worker externally.

    SIGCONT resumes execution automatically.

    SIGTERM/SIGINT are handled here so the worker can save its adapter
    before exiting.
    """
    global shutdown_requested
    shutdown_requested = True

    print(
        json.dumps(
            {
                "event": "shutdown_requested",
                "pid": os.getpid(),
                "signal": signum,
                "timestamp": time.time(),
            }
        ),
        flush=True,
    )


signal.signal(signal.SIGTERM, request_shutdown)
signal.signal(signal.SIGINT, request_shutdown)


# ---------------------------------------------------------------------------
# Synthetic FL dataset
# ---------------------------------------------------------------------------

class FederatedTextDataset(Dataset):
    """
    Small local dataset representing data owned by one FL client.

    We intentionally create the data locally rather than downloading a
    massive corpus because FedSched is evaluating OS resource management,
    not language-model quality.
    """

    def __init__(
        self,
        tokenizer,
        client_id: str,
        samples: int,
        seq_length: int,
    ):
        self.examples = []

        templates = [
            "Federated learning keeps training data on local devices.",
            "Edge devices have different amounts of memory and compute.",
            "Operating systems schedule competing processes for CPU time.",
            "Memory pressure can cause applications to become unresponsive.",
            "Parameter efficient fine tuning reduces trainable parameters.",
            "LoRA adds low rank trainable matrices to transformer layers.",
            "FedSched monitors operating system health during local training.",
            "The Linux kernel exposes resource controls through cgroups.",
            "CPU pressure increases when several applications compete.",
            "A federated client may temporarily pause training under stress.",
            "Background machine learning workloads should yield to users.",
            "Adaptive scheduling can reduce federated client dropout.",
        ]

        rng = random.Random(hash(client_id) & 0xFFFFFFFF)

        for i in range(samples):
            text = (
                f"Client {client_id}. "
                f"Local record {i}. "
                f"{rng.choice(templates)} "
                f"{rng.choice(templates)} "
                f"{rng.choice(templates)}"
            )

            encoded = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=seq_length,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)

            labels = input_ids.clone()

            # Ignore padding when calculating LM loss.
            labels[attention_mask == 0] = -100

            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def emit(event: str, **kwargs):
    """
    Emit JSON-line telemetry.

    This makes it easy to:
        - redirect output to a log
        - parse results later
        - compare events with eBPF telemetry
    """

    record = {
        "event": event,
        "timestamp": time.time(),
        "pid": os.getpid(),
        **kwargs,
    }

    print(json.dumps(record), flush=True)


def save_adapter(model, output_dir: Path, client_id: str, step: int):
    checkpoint = output_dir / client_id / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(checkpoint)

    emit(
        "checkpoint",
        client_id=client_id,
        step=step,
        path=str(checkpoint),
    )


# ---------------------------------------------------------------------------
# Main training workload
# ---------------------------------------------------------------------------

def train(args):
    # ------------------------------------------------------------------
    # Linux / CPU configuration
    # ------------------------------------------------------------------

    #
    # For FedSched, CPU execution is intentional.
    #
    # If PyTorch used CUDA, cpu.max would no longer represent most of the
    # computational work we are trying to regulate.
    #

    device = torch.device("cpu")

    torch.set_num_threads(args.threads)

    # Inter-op threads should generally remain small.
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch may reject changing this after parallel work starts.
        pass

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    pid = os.getpid()

    # Write PID somewhere the shell / future daemon can find it.
    pid_file = Path(args.pid_file)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(pid))

    emit(
        "worker_start",
        client_id=args.client_id,
        model=args.model,
        threads=args.threads,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        pid_file=str(pid_file),
    )

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    emit("loading_tokenizer", client_id=args.client_id)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    emit("loading_model", client_id=args.client_id)

    model = AutoModelForCausalLM.from_pretrained(args.model)

    #
    # OPT attention layers contain:
    #
    #     q_proj
    #     k_proj
    #     v_proj
    #     out_proj
    #
    # q_proj + v_proj gives us a conventional, lightweight LoRA workload.
    #

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    model.to(device)
    model.train()

    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    dataset = FederatedTextDataset(
        tokenizer=tokenizer,
        client_id=args.client_id,
        samples=args.samples,
        seq_length=args.seq_length,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    # Only LoRA parameters should normally require gradients.
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = AdamW(
        trainable_parameters,
        lr=args.learning_rate,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    global shutdown_requested

    global_step = 0
    start_time = time.monotonic()

    emit(
        "training_start",
        client_id=args.client_id,
        dataset_samples=len(dataset),
    )

    for epoch in range(args.epochs):

        if shutdown_requested:
            break

        for batch_index, batch in enumerate(loader):

            if shutdown_requested:
                break

            batch_start = time.monotonic()

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            optimizer.zero_grad(set_to_none=True)

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            loss = outputs.loss

            loss.backward()

            optimizer.step()

            global_step += 1

            batch_time = time.monotonic() - batch_start
            elapsed = time.monotonic() - start_time

            # ----------------------------------------------------------
            # Heartbeat
            # ----------------------------------------------------------

            if global_step % args.log_every == 0:

                emit(
                    "heartbeat",
                    client_id=args.client_id,
                    epoch=epoch,
                    batch=batch_index,
                    step=global_step,
                    loss=round(loss.item(), 6),
                    batch_seconds=round(batch_time, 4),
                    elapsed_seconds=round(elapsed, 2),
                )

            # ----------------------------------------------------------
            # Checkpoint
            # ----------------------------------------------------------

            if (
                args.checkpoint_every > 0
                and global_step % args.checkpoint_every == 0
            ):
                save_adapter(
                    model,
                    output_dir,
                    args.client_id,
                    global_step,
                )

            # Optional delay useful for low-pressure experiments.
            if args.sleep_between_steps > 0:
                time.sleep(args.sleep_between_steps)

    # ------------------------------------------------------------------
    # Final checkpoint
    # ------------------------------------------------------------------

    final_dir = output_dir / args.client_id / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    emit(
        "worker_exit",
        client_id=args.client_id,
        step=global_step,
        total_seconds=round(time.monotonic() - start_time, 2),
        shutdown_requested=shutdown_requested,
        output=str(final_dir),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description="FedSched PyTorch/PEFT workload"
    )

    parser.add_argument(
        "--client-id",
        type=str,
        default="client_01",
        help="Federated client identifier",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="Hugging Face model",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="PyTorch CPU threads",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seq-length",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Large by design so workload remains alive for experiments",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
    )

    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--sleep-between-steps",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
    )

    parser.add_argument(
        "--pid-file",
        type=str,
        default="./run/fedsched-worker.pid",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    try:
        train(args)

    except Exception as exc:

        emit(
            "worker_error",
            error=repr(exc),
        )

        raise