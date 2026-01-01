import flwr as fl
from .config import Config

def main():
    print(f"Starting Flower Server on {Config.SERVER_ADDRESS}")
    
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=Config.MIN_FIT_CLIENTS,
        min_available_clients=Config.MIN_AVAILABLE_CLIENTS,
        # We can add custom metric aggregation here if needed
    )

    # Start Server
    fl.server.start_server(
        server_address=Config.SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=Config.NUM_ROUNDS),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
