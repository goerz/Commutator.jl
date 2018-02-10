module Commutator


export commutator!


"""
    commutator!(alpha, A, B, beta, C)

Calculate `C = beta * C + alpha * (A * B - B * A)` in-place.
"""
function commutator!(alpha, A, B, beta, C)
    C .*= beta
    C .+= alpha * (A * B - B * A)
end


function commutator!(
        alpha::Number, A::AbstractMatrix, B::AbstractMatrix, beta::Number,
        C::AbstractMatrix)
    C[:,:] = beta * C[:,:] + alpha * (A * B - B * A)
end


for elty in (:Float64, :Float32, :Complex128, :Complex64)
    @eval begin
        function commutator!(
                alpha::($elty), A::StridedVecOrMat{$elty},
                B::StridedVecOrMat{$elty}, beta::($elty),
                C::StridedVecOrMat{$elty})
            BLAS.gemm!('N', 'N', alpha, A, B, beta, C)
            BLAS.gemm!('N', 'N', -alpha, B, A, one(beta), C)
        end
        function commutator!(
                alpha::Number, A::StridedVecOrMat{$elty},
                B::StridedVecOrMat{$elty}, beta::Number,
                C::StridedVecOrMat{$elty})
            alpha = convert($elty, alpha)
            beta = convert($elty, beta)
            BLAS.gemm!('N', 'N', alpha, A, B, beta, C)
            BLAS.gemm!('N', 'N', -alpha, B, A, one(beta), C)
        end
    end
end


function commutator!(
        alpha::Number, A::SparseMatrixCSC, B::AbstractMatrix,
        beta::Number, C::AbstractMatrix)
    if beta != 1
        C .*= beta
    end
    for j = 1: A.n
        for k = 1:A.n, i_val = A.colptr[k] : (A.colptr[k+1]-1)
            i = A.rowval[i_val]
            A_ik = A.nzval[i_val]
            C[i,j] = C[i,j] + alpha * A_ik * B[k,j]
        end
    end
    for j = 1:A.n, i_val = A.colptr[j] : (A.colptr[j+1]-1)
        k = A.rowval[i_val]
        A_kj = A.nzval[i_val]
        for i = 1: A.n
            C[i,j] = C[i,j] - alpha * B[i, k] * A_kj
        end
    end
end

end # module
