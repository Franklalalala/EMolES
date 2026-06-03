import torch.nn as nn
import torch
from typing import Union, Tuple, Optional, Callable, Dict
import torch.nn.functional as F
from emoles.nn.embedding import Embedding
from emoles.data.transforms import OrbitalMapper
from emoles.nn.base import AtomicFFN, AtomicLinear, Identity
from emoles.data import AtomicDataDict
from emoles.nn.hamiltonian import E3Hamiltonian
from e3nn.o3 import Linear
from emoles.nn.rescale import E3PerSpeciesScaleShift, E3PerEdgeSpeciesScaleShift
import logging

log = logging.getLogger(__name__)

def get_neuron_config(nl):
    """Extracts the configuration of a neural network from a list of layer sizes.

    Args:
        nl: A list of integers representing the number of neurons in each layer.
            If the list has an even number of elements, the last element is assumed
            to be the output layer size.

    Returns:
        A list of dictionaries, where each dictionary describes the configuration of
        a layer in the neural network. Each dictionary has the following keys:
        - in_features: The number of input neurons for the layer.
        - hidden_features: The number of hidden neurons for the layer (if applicable).
        - out_features: The number of output neurons for the layer.
    
        e.g.
        [1, 2, 3, 4, 5, 6] -> [{'in_features': 1, 'hidden_features': 2, 'out_features': 3}, 
                               {'in_features': 3, 'hidden_features': 4, 'out_features': 5}, 
                               {'in_features': 5, 'out_features': 6}]
        [1, 2, 3, 4, 5]    -> [{'in_features': 1, 'hidden_features': 2, 'out_features': 3}, 
                               {'in_features': 3, 'hidden_features': 4, 'out_features': 5}]
    """
    
    n = len(nl)
    assert n > 1, "The neuron config should have at least 2 layers."
    if n % 2 == 0:
        d_out = nl[-1]
        nl = nl[:-1]
    config = []
    for i in range(1,len(nl)-1, 2):
        config.append({'in_features': nl[i-1], 'hidden_features': nl[i], 'out_features': nl[i+1]})

    if n % 2 == 0:
        config.append({'in_features': nl[-1], 'out_features': d_out})

    return config


class NNENV(nn.Module):
    quantities = ["hamiltonian", "energy"]
    name = "nnenv"
    def __init__(
            self,
            embedding: dict,
            prediction: dict,
            overlap: bool = False,
            basis: Dict[str, Union[str, list]]=None,
            idp: Union[OrbitalMapper, None]=None,
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
            transform: bool = True,
            has_soc: bool = False,
            scale_type: str = 'scale_w_back_grad',
            **kwargs,
    ):
        
        """The top level EMolES model class.

        Parameters
        ----------
        embedding_config : dict
            _description_
        prediction_config : dict
            _description_
        basis : Dict[str, Union[str, list], None], optional
            _description_, by default None
        idp : Union[OrbitalMapper, None], optional
            _description_, by default None
        transform : bool, optional
            _description_, decide whether to transform the irreducible matrix element to the hamiltonians
        dtype : Union[str, torch.dtype], optional
            _description_, by default torch.float32
        device : Union[str, torch.device], optional
            _description_, by default torch.device("cpu")

        Raises
        ------
        NotImplementedError
            _description_
        """
        super(NNENV, self).__init__()

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.dtype = dtype
        self.device = device
        self.model_options = {"embedding": embedding.copy(), "prediction": prediction.copy()}
        self.transform = transform

        self.method = prediction.get("method", "e3tb")
        # self.soc = prediction.get("soc", False)
        self.prediction = prediction

        prediction_copy = prediction.copy()
        scale_type = prediction_copy.get("scale_type")
        self.scale_type = scale_type

        self.has_soc = has_soc
        print(f'NNENV soc flag: {self.has_soc}')

        if basis is not None:
            self.idp = OrbitalMapper(basis, method=self.method, device=self.device, has_soc=has_soc)
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp
            
        self.basis = self.idp.basis
        self.idp.get_orbpair_maps()

        embedding.update({'has_soc': has_soc})

        n_species = len(self.basis.keys())
        # initialize the embedding layer
        self.embedding = Embedding(**embedding, dtype=dtype, device=device, idp=self.idp, n_atom=n_species)
        
        if prediction_copy.get("method") != "e3tb":
            raise NotImplementedError("The prediction model {} is not implemented.".format(prediction_copy["method"]))

        self.node_prediction_h = E3PerSpeciesScaleShift(
            field=AtomicDataDict.NODE_FEATURES_KEY,
            num_types=n_species,
            irreps_in=self.embedding.out_node_irreps,
            out_field=AtomicDataDict.NODE_FEATURES_KEY,
            shifts=0.,
            scales=1.,
            dtype=self.dtype,
            device=self.device,
            **prediction_copy,
        )

        self.edge_prediction_h = E3PerEdgeSpeciesScaleShift(
            field=AtomicDataDict.EDGE_FEATURES_KEY,
            num_types=n_species,
            irreps_in=self.embedding.out_edge_irreps,
            out_field=AtomicDataDict.EDGE_FEATURES_KEY,
            shifts=0.,
            scales=1.,
            dtype=self.dtype,
            device=self.device,
            **prediction_copy,
        )

        self.hamiltonian = E3Hamiltonian(
            edge_field=AtomicDataDict.EDGE_FEATURES_KEY,
            node_field=AtomicDataDict.NODE_FEATURES_KEY,
            idp=self.embedding.idp,
            dtype=self.dtype,
            device=self.device,
            soc=self.has_soc,
        )


    def forward(self, data: AtomicDataDict.Type):
        if data.get(AtomicDataDict.EDGE_TYPE_KEY, None) is None:
            self.idp(data)

        data = self.embedding(data)
        if self.scale_type != "no_scale":
            data = self.node_prediction_h(data)
            data = self.edge_prediction_h(data)
        
        if self.transform:
            data = self.hamiltonian(data)

        return data
    
    @classmethod
    def from_reference(
        cls, 
        checkpoint, 
        embedding: dict={},
        prediction: dict={},
        overlap: bool=None,
        basis: Dict[str, Union[str, list]]=None,
        dtype: Union[str, torch.dtype]=None,
        device: Union[str, torch.device]=None,
        transform: bool = True,
        **kwargs
        ):
        if device == 'cuda':
            if not torch.cuda.is_available():
                device = 'cpu'
                log.warning("CUDA is not available. The model will be loaded on CPU.")

        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        common_options = {
            "dtype": dtype,
            "device": device,
            "basis": basis,
            "overlap": overlap,
        }

        model_options = {
            "embedding": embedding,
            "prediction": prediction,
        }

        if len(embedding) == 0 or len(prediction) == 0:
            model_options.update(ckpt["config"]["model_options"])

        for k,v in common_options.items():
            if v is None:
                common_options[k] = ckpt["config"]["common_options"][k]
        model = cls(**model_options, **common_options, transform=transform)
        model.load_state_dict(ckpt["model_state_dict"])

        del ckpt

        return model
