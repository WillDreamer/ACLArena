"""Launch the MLE model API or a standalone SGLang model service."""

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
import uvicorn

from mle.core import TASK_TO_EXPERT, digest
from mle.runtime import Router, load_config
from mle.serve import create_app


def gpu_memory(gpus):
    memory = []
    for gpu in gpus.split(","):
        raw = subprocess.check_output([
            "nvidia-smi", "-i", gpu, "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits",
        ], text=True)
        memory.append(tuple(map(float, raw.strip().split(","))))
    return memory


def check_gpus(gpus, serving):
    ids = [gpu.strip() for gpu in gpus.split(",")]
    if len(ids) != serving["tp_size"] or not all(ids) or len(set(ids)) != len(ids):
        raise ValueError(f"Provide {serving['tp_size']} distinct GPU IDs separated by commas")
    if not 0 < serving["mem_fraction_static"] < 1:
        raise ValueError("mem_fraction_static must be between 0 and 1")
    for gpu, (total, free) in zip(ids, gpu_memory(",".join(ids)), strict=True):
        needed = serving["mem_fraction_static"] * total
        if free < needed + 256:
            raise RuntimeError(f"GPU {gpu} needs {needed:.0f} MiB free; available {free:.0f} MiB")
    return ",".join(ids)


class ModelService:
    """Own model processes on one GPU group, resident or switched on demand."""

    def __init__(self, services, serving, gpus, output, load_mode="on-demand"):
        self.services, self.serving, self.gpus = services, serving, gpus
        self.load_mode = load_mode
        self.exclusive = load_mode == "on-demand"
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.processes = {}
        self.logs = {}

    def check(self):
        if not self.exclusive and set(self.processes) != set(self.services):
            raise RuntimeError("Not all model services are loaded")
        for name, process in self.processes.items():
            if process.poll() is not None:
                raise RuntimeError(f"{name} exited; inspect its log")

    def ensure(self, name):
        if name in self.processes:
            if self.processes[name].poll() is not None:
                raise RuntimeError(f"{name} exited; restart the launcher after inspecting its log")
            return
        if self.exclusive:
            self.stop()
        service = self.services[name]
        parsed = urlparse(service["base_url"])
        if parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.port is None:
            raise ValueError("Managed model services must use explicit localhost ports")
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", parsed.port)) == 0:
                raise RuntimeError(f"Port {parsed.port} is occupied")
        command = [sys.executable, "-m", "sglang.launch_server", "--model-path", service["model_id"],
                   "--served-model-name", service["model_name"], "--host", "127.0.0.1", "--port", str(parsed.port)]
        serving = dict(self.serving)
        if not self.exclusive:
            memory = gpu_memory(self.gpus)
            budget = min(total for total, _ in memory) * serving["mem_fraction_static"] / len(self.services)
            free = min(available for _, available in memory)
            if free <= budget + 256:
                raise RuntimeError(f"Not enough GPU memory to load {name}; use --load-mode on-demand")
            # SGLang 0.5.10 applies this fraction to free memory before loading.
            # Recalculate it so resident models receive equal static budgets.
            serving["mem_fraction_static"] = round(budget / free, 6)
        for key, value in serving.items():
            if value is False:
                continue
            flag = "--" + key.replace("_", "-")
            command.extend([flag] if value is True else [flag, str(value)])
        log = self.logs[name] = (self.output / f"{name}.log").open("a")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": self.gpus, "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "8"}
        try:
            process = self.processes[name] = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            print(f"Loading {name}; log: {self.output / (name + '.log')}", flush=True)
            with httpx.Client(timeout=2, trust_env=False) as client:
                deadline = time.monotonic() + 1200
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f"{name} exited; inspect its log")
                    try:
                        if client.get(f"http://127.0.0.1:{parsed.port}/health").status_code == 200:
                            return
                    except httpx.HTTPError:
                        pass
                    time.sleep(1)
            raise TimeoutError(f"{name} did not become ready")
        except BaseException:
            self.stop()
            raise

    def stop(self):
        for process in reversed(list(self.processes.values())):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        self.processes.clear()
        for log in self.logs.values():
            log.close()
        self.logs.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--experts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--port", type=int, default=30610)
    parser.add_argument("--load-mode", choices=["all", "on-demand"], default="on-demand",
                        help="Keep all five models resident, or switch models on demand (default)")
    parser.add_argument("--service", choices=["mle", "gateway", "classifier", *TASK_TO_EXPERT.values()], default="mle")
    args = parser.parse_args()
    config, experts = load_config(args.config), load_config(args.experts)
    if set(experts) != set(TASK_TO_EXPERT.values()):
        raise ValueError("Exactly the four named experts must be configured")
    services = {"classifier": config["classifier"], **experts}
    selected = services if args.service in {"mle", "gateway"} else {args.service: services[args.service]}
    for name, service in selected.items():
        if not service["model_id"].strip():
            raise ValueError(f"Set model_id for {name} to its Hugging Face repository ID")
    ports = [urlparse(s["base_url"]).port for s in services.values()] + [args.port]
    if None in ports or len(set(ports)) != len(ports):
        raise ValueError("Classifier, experts, and gateway must use distinct explicit ports")
    output = Path(args.output)
    manager = None
    if args.service != "gateway":
        gpus = check_gpus(args.gpus, config["serving"])
        mode = args.load_mode if args.service == "mle" else "on-demand"
        manager = ModelService(services, config["serving"], gpus, output / "logs", mode)
        def stop(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
    try:
        if args.service in {"mle", "gateway"}:
            if manager is not None and not manager.exclusive:
                for name in services:
                    manager.ensure(name)
            state = output / "state" / digest({"config": config, "experts": experts, "prompt": Path(__file__).with_name("prompt.txt").read_text()})[:16]
            router = Router(args.config, args.experts, state, service_manager=manager)
            uvicorn.run(create_app(router), host="127.0.0.1", port=args.port)
        else:
            manager.ensure(args.service)
            while manager.processes[args.service].poll() is None:
                time.sleep(1)
            raise RuntimeError("Model service exited")
    finally:
        if manager is not None:
            manager.stop()


if __name__ == "__main__":
    main()
