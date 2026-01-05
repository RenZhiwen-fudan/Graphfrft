import torch as th
from torch_gfrft.torch_gfrft.utils import get_matvec_tensor_einsum_str, is_hermitian
from typing import Union

class GFRFT:
    def __init__(self, gft_mtx: th.Tensor) -> None:
       
        self.original_device = gft_mtx.device
        
        if is_hermitian(gft_mtx):
            self._eigvals, self._eigvecs = th.linalg.eigh(gft_mtx)
            self._inv_eigvecs = self._eigvecs.H
        else:
            self._eigvals, self._eigvecs = th.linalg.eig(gft_mtx)
            self._inv_eigvecs = th.linalg.inv(self._eigvecs)

    def gfrft_mtx(self, a: Union[float, th.Tensor]) -> th.Tensor:
        
        device = a.device if isinstance(a, th.Tensor) else self.original_device
        
       
        eigvals = self._eigvals.to(device)
        eigvecs = self._eigvecs.to(device)
        inv_eigvecs = self._inv_eigvecs.to(device)
        
        
        fractional_eigvals = eigvals**a
        
        
        return th.einsum("ij,j,jk->ik", eigvecs, fractional_eigvals, inv_eigvecs)

    def igfrft_mtx(self, a: Union[float, th.Tensor]) -> th.Tensor:
        return self.gfrft_mtx(-a)

    def gfrft(self, x: th.Tensor, a: Union[float, th.Tensor], *, dim: int = -1) -> th.Tensor:
        
        device = x.device
        
       
        if isinstance(a, th.Tensor):
            a = a.to(device)
        else:
            a = th.tensor(a, device=device)
        
        
        gfrft_mtx = self.gfrft_mtx(a)
        
        
        gfrft_mtx = gfrft_mtx.to(device)
        
        
        dtype = th.promote_types(gfrft_mtx.dtype, x.dtype)
        
        return th.einsum(
            get_matvec_tensor_einsum_str(len(x.shape), dim),
            gfrft_mtx.type(dtype),
            x.type(dtype),
        )

    def igfrft(self, x: th.Tensor, a: Union[float, th.Tensor], *, dim: int = -1) -> th.Tensor:
        return self.gfrft(x, -a, dim=dim)
    
    def to(self, device):
    
        self._eigvals = self._eigvals.to(device)
        self._eigvecs = self._eigvecs.to(device)
        self._inv_eigvecs = self._inv_eigvecs.to(device)
        self.original_device = device
        return self