import flwr as fl
import torch
from typing import Dict, List, Tuple
import numpy as np

class FederatedSyntheticLearning:
    def __init__(self, model, num_rounds=3):
        self.model = model
        self.num_rounds = num_rounds

    class SyntheticClient(fl.client.NumPyClient):
        def __init__(self, model, train_data):
            self.model = model
            self.train_data = train_data

        def get_parameters(self, config):
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.model.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            # Local training logic here
            # ...
            return self.get_parameters(config={}), len(self.train_data), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            # Evaluation logic here
            loss = 0.0
            accuracy = 0.0
            return loss, len(self.train_data), {"accuracy": accuracy}

    def weighted_average(self, metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
        """Aggregation strategy for weighted averaging."""
        total_examples = sum([num_examples for num_examples, _ in metrics])
        weighted_metrics = {}
        
        for _, metric_dict in metrics:
            for key, value in metric_dict.items():
                if key not in weighted_metrics:
                    weighted_metrics[key] = 0
                weighted_metrics[key] += value * (num_examples / total_examples)
                
        return weighted_metrics

    def start_federated_learning(self, clients_data):
        """Start federated learning process."""
        # Create FL strategy
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=len(clients_data),
            min_evaluate_clients=len(clients_data),
            min_available_clients=len(clients_data),
            evaluate_metrics_aggregation_fn=self.weighted_average,
        )

        # Start FL server
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=strategy,
        )

    def create_client(self, train_data):
        """Create a new federated learning client."""
        return self.SyntheticClient(self.model, train_data)

    def aggregate_models(self, client_models):
        """Aggregate models from different clients."""
        aggregated_dict = {}
        num_models = len(client_models)
        
        for key in client_models[0].state_dict().keys():
            aggregated_dict[key] = sum(model.state_dict()[key] for model in client_models) / num_models
            
        self.model.load_state_dict(aggregated_dict)
        return self.model 