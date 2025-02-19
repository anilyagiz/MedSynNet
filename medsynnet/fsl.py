import flwr as fl
import torch
from typing import Dict, List, Tuple
from collections import OrderedDict

class FSL:
    def __init__(self, model):
        self.model = model
        
    def get_parameters(self) -> List[np.ndarray]:
        """Get model parameters."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """Update model parameters."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def train(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        """Perform local training."""
        self.set_parameters(parameters)
        self.model.train()
        
        # Training logic implemented here
        # Simplified for this example
        num_examples = 0
        
        return self.get_parameters(), num_examples, {}
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        """Perform model evaluation."""
        self.set_parameters(parameters)
        self.model.eval()
        
        # Evaluation logic implemented here
        loss = 0.0
        num_examples = 0
        
        return loss, num_examples, {}

class FederatedClient(fl.client.NumPyClient):
    def __init__(self, fsl: FSL):
        self.fsl = fsl
    
    def get_parameters(self, config):
        return self.fsl.get_parameters()
    
    def fit(self, parameters, config):
        return self.fsl.train(parameters, config)
    
    def evaluate(self, parameters, config):
        return self.fsl.evaluate(parameters, config)

class FederatedServer(fl.server.Server):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def aggregate(self, results: List[Tuple[fl.common.Weights, int]]) -> fl.common.Weights:
        """Custom aggregation strategy."""
        # Federated Averaging implementation
        weights = [w for w, _ in results]
        examples = [n for _, n in results]
        
        total_examples = sum(examples)
        weighted_weights = [
            [layer * n / total_examples for layer in w] 
            for w, n in zip(weights, examples)
        ]
        
        return [
            sum(layer_updates) 
            for layer_updates in zip(*weighted_weights)
        ]

def start_server(model_config: Dict, num_rounds: int = 3):
    """Start federated learning server."""
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        min_fit_clients=2,
        min_evaluate_clients=2,
    )
    
    fl.server.start_server(
        server_address="[::]:8080",
        config={"num_rounds": num_rounds},
        strategy=strategy,
    )

def start_client(fsl: FSL, server_address: str = "[::]:8080"):
    """Start federated learning client."""
    client = FederatedClient(fsl)
    fl.client.start_numpy_client(server_address, client) 