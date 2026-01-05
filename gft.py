import torch as th

from torch_gfrft.torch_gfrft import ComplexSortStrategy, EigvalSortStrategy
from torch_gfrft.torch_gfrft.utils import asc_sort, get_matvec_tensor_einsum_str, is_hermitian, is_hermitian_batch, tv_sort


class GFT:
    

    def __init__(
        self,
        shift_mtx: th.Tensor,
        eigval_sort_strategy: EigvalSortStrategy = EigvalSortStrategy.NO_SORT,
        complex_sort_strategy: ComplexSortStrategy = ComplexSortStrategy.REAL,
    ) -> None:
        
       
        if shift_mtx.dim() not in [2, 3]:
            raise ValueError(f"Expected 2D or 3D tensor, got {shift_mtx.dim()}D")
        
       
        if shift_mtx.dim() == 2:
            shift_mtx = shift_mtx.unsqueeze(0)
            single_matrix = True
        else:
            single_matrix = False
        
        batch_size, n, m = shift_mtx.shape
        if n != m:
            raise ValueError("Shift matrix must be square")
        
        hermitian = is_hermitian_batch(shift_mtx)
        
        
        eigvals_list = []
        eigvecs_list = []
        
        for i in range(batch_size):
            matrix = shift_mtx[i]
            is_herm = hermitian[i] if hermitian.dim() > 0 else hermitian
            
            if is_herm:
                eigvals, eigvecs = th.linalg.eigh(matrix)
            else:
                eigvals, eigvecs = th.linalg.eig(matrix)

            if eigval_sort_strategy == EigvalSortStrategy.ASCENDING:
                eigvals, eigvecs = asc_sort(eigvals, eigvecs, complex_sort_strategy)
            elif eigval_sort_strategy == EigvalSortStrategy.TOTAL_VARIATION:
                eigvals, eigvecs = tv_sort(matrix, eigvals, eigvecs)
            
            eigvals_list.append(eigvals)
            eigvecs_list.append(eigvecs)
        
       
        self._graph_freqs = th.stack(eigvals_list)
        self._igft_mtx = th.stack(eigvecs_list)
        
        
        gft_mtx_list = []
        for i in range(batch_size):
            if hermitian[i] if hermitian.dim() > 0 else hermitian:
                gft_mtx_list.append(eigvecs_list[i].H)
            else:
                gft_mtx_list.append(th.linalg.inv(eigvecs_list[i]))
        
        self._gft_mtx = th.stack(gft_mtx_list)
        
        
        if single_matrix:
            self._graph_freqs = self._graph_freqs.squeeze(0)
            self._gft_mtx = self._gft_mtx.squeeze(0)
            self._igft_mtx = self._igft_mtx.squeeze(0)

    @property
    def graph_freqs(self) -> th.Tensor:
        """Returns the previously calculated graph frequencies."""
        return self._graph_freqs

    @property
    def gft_mtx(self) -> th.Tensor:
        """Returns the previously calculated graph Fourier transform matrix."""
        return self._gft_mtx

    @property
    def igft_mtx(self) -> th.Tensor:
        """Returns the previously calculated inverse graph Fourier transform matrix."""
        return self._igft_mtx

    def gft(self, x: th.Tensor, *, dim: int = -1) -> th.Tensor:
        """Returns the graph Fourier transform of the input with previously calculated graph Fourier transform matrix."""
        dtype = th.promote_types(x.dtype, self._gft_mtx.dtype)
        
        if self._gft_mtx.dim() == 2 and x.dim() > 2:
           
            result = []
            for i in range(x.shape[0]):
                x_i = x[i].unsqueeze(0) if x.dim() > 2 else x[i]
                transformed = th.einsum(
                    get_matvec_tensor_einsum_str(len(x_i.shape), dim),
                    self._gft_mtx.type(dtype),
                    x_i.type(dtype),
                )
                result.append(transformed.squeeze(0))
            return th.stack(result)
        else:
            return th.einsum(
                get_matvec_tensor_einsum_str(len(x.shape), dim),
                self._gft_mtx.type(dtype),
                x.type(dtype),
            )

    def igft(self, x: th.Tensor, *, dim: int = -1) -> th.Tensor:
        """Returns the inverse graph Fourier transform of the input with previously calculated graph Fourier transform matrix."""
        dtype = th.promote_types(x.dtype, self._igft_mtx.dtype)
        
        if self._igft_mtx.dim() == 2 and x.dim() > 2:
           
            result = []
            for i in range(x.shape[0]):
                x_i = x[i].unsqueeze(0) if x.dim() > 2 else x[i]
                transformed = th.einsum(
                    get_matvec_tensor_einsum_str(len(x_i.shape), dim),
                    self._igft_mtx.type(dtype),
                    x_i.type(dtype),
                )
                result.append(transformed.squeeze(0))
            return th.stack(result)
        else:
            return th.einsum(
                get_matvec_tensor_einsum_str(len(x.shape), dim),
                self._igft_mtx.type(dtype),
                x.type(dtype),
            )