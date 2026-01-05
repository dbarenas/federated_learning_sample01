import flwr as fl
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
import torch
from datetime import datetime, timezone
import argparse
import os
from typing import Dict, List, Tuple, Union, Optional

from .config import Config
from .common import (
    load_model,
    get_lora_state_dict,
    set_lora_state_dict,
    save_lora_artifact,
    ensure_artifact_dir,
    get_package_versions,
)


class FedAvgSaveStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        self.artifacts_dir = kwargs.pop("artifacts_dir", "./artifacts")
        self.num_rounds = kwargs.pop("num_rounds", Config.NUM_ROUNDS)
        super().__init__(*args, **kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregate fit results and save the final model."""

        # Call parent aggregate_fit
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None and server_round == self.num_rounds:
            print(f"Round {server_round}: Saving final global model artifact...")

            # Convert aggregated parameters to a state dictionary
            aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)

            # Load a base model to apply the aggregated weights
            model, tokenizer = load_model()

            # Get the correct order of keys
            lora_keys = get_lora_state_dict(model).keys()

            # Create the state dict, converting numpy arrays to torch tensors
            lora_state_dict = {
                k: torch.tensor(v) for k, v in zip(lora_keys, aggregated_ndarrays)
            }

            # Apply the state dict to the model
            set_lora_state_dict(model, lora_state_dict)

            # Prepare metadata
            metadata = {
                "round": server_round,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_name": Config.MODEL_NAME,
                "num_clients": len(results),
                "num_rounds": self.num_rounds,
                "metrics": aggregated_metrics,
                "package_versions": get_package_versions(),
            }

            # Ensure artifact directory exists and save
            out_dir = ensure_artifact_dir(self.artifacts_dir, server_round)
            save_lora_artifact(model, tokenizer, out_dir, metadata)

            print("Artifact saved successfully.")

        return aggregated_parameters, aggregated_metrics


def main():
    parser = argparse.ArgumentParser(description="Flower Server with Artifact Persistence")
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=os.environ.get("NUM_ROUNDS", Config.NUM_ROUNDS),
        help="Total number of federated learning rounds.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=os.environ.get("ARTIFACTS_DIR", "./artifacts"),
        help="Directory to save server-side artifacts.",
    )
    args = parser.parse_args()

    print(f"Starting Flower Server for {args.num_rounds} rounds")
    print(f"Artifacts will be saved to: {args.artifacts_dir}")
    
    # Define strategy
    strategy = FedAvgSaveStrategy(
        min_fit_clients=Config.MIN_FIT_CLIENTS,
        min_available_clients=Config.MIN_AVAILABLE_CLIENTS,
        artifacts_dir=args.artifacts_dir,
        num_rounds=args.num_rounds,
    )

    # Start Server
    fl.server.start_server(
        server_address=Config.SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
