
module Utilities

export dense_to_sparse, create_mask_matrix, khatri_rao_block

using Test

# Created using ChatGPT

function find_differences(arr1, arr2; tol=1e-8)
    if size(arr1) != size(arr2)
        error("Arrays must have the same size")
    end

    differences = []
    for i in eachindex(arr1)
        if !isapprox(arr1[i], arr2[i]; atol=tol)
            push!(differences, (index=i, arr1_value=arr1[i], arr2_value=arr2[i]))
        end
    end
    
    return differences
end


"""
    dense_to_sparse(dense_matrix, mask_matrix)

Convert a dense matrix to a sparse representation using a mask matrix.

# Arguments
- `dense_matrix` (Matrix): A dense matrix that you want to convert to a sparse representation. 
- `mask_matrix` (Matrix): A binary mask matrix of the same size as `dense_matrix`. The function keeps only the values of `dense_matrix` where `mask_matrix` has non-zero elements.

# Returns
- A vector of dictionaries where each dictionary represents a column of the `dense_matrix`. The keys in the dictionary are the row indices of the non-zero elements in the `mask_matrix`, and the values are the corresponding elements from the `dense_matrix`.

# Error
- Throws an error if the `dense_matrix` and `mask_matrix` do not have the same dimensions.
"""
function dense_to_sparse(dense_matrix, mask_matrix=ones(size(dense_matrix)))
    if size(dense_matrix) != size(mask_matrix)
        error("The dense matrix and the mask matrix must have the same dimensions")
    end
    
    m, n = size(dense_matrix)
    sparse_matrix = [
        Dict(i => dense_matrix[i, j] for i in 1:m if mask_matrix[i, j] != 0) 
        for j in 1:n
    ]
    return sparse_matrix
end

"""
    create_mask_matrix(mask_vectors, num_rows)

Create a binary mask matrix from a vector of vectors containing row indices of non-zero elements.

# Arguments
- `mask_vectors` (Vector of Vectors): A vector of vectors where each inner vector contains the row indices of non-zero elements for a specific column.
- `num_rows` (Int): The number of rows in the final mask matrix.

# Returns
- A binary mask matrix where the element at position (i, j) is 1 if row i is listed in the j-th vector in `mask_vectors`, and 0 otherwise.

# Error
- Throws an error if `num_rows` is less than the largest row index present in the `mask_vectors`.
"""
function create_mask_matrix(mask_vectors, num_rows)
    max_row_index = maximum([isempty(vector) ? 0 : maximum(vector) for vector in mask_vectors])
    
    if num_rows < max_row_index
        error("num_rows must be at least as large as the largest row index in mask_vectors")
    end

    num_columns = length(mask_vectors)
    mask_matrix = zeros(Int, num_rows, num_columns)

    for (col_idx, vector) in enumerate(mask_vectors)
        for row_idx in vector
            mask_matrix[row_idx, col_idx] = 1
        end
    end

    return mask_matrix
end


"""
    khatri_rao_block(W, Z)

Compute the block-wise Khatri-Rao product of a matrix W (of size MxM) and a tensor Z (of size NxNxM).

# Arguments
- `W` (Matrix): An MxM matrix.
- `Z` (Array): An NxNxM tensor.

# Returns
- A block matrix of size (M*N)x(M*N) where each block X_ij is computed as W[i, j] * Z[:, :, j].
"""
function khatri_rao_block(W, Z)
    # If W is a vector with a single element, reshape it to a 1x1 matrix
    if (ndims(W) == 1) && (length(W) == 1)
        W = reshape(W, (1, 1))
    end

    M, _ = size(W)
    dims = size(Z)
    
    # Handle the case when Z is a 2D array (i.e., M = 1)
    if length(dims) == 2
        N = dims[1]
        P = 1
    else
        N, _, P = dims
    end
    
    # Initialize an empty block matrix
    X = zeros(M*N, M*N)
    
    # Fill in the blocks
    for i in 1:M
        for j in 1:M
            # Compute the block X_ij
            X_ij = W[i, j] * Z[:, :, min(j, P)]
            
            # Place X_ij in the appropriate block position in the final matrix X
            X[(i-1)*N+1:i*N, (j-1)*N+1:j*N] = X_ij
        end
    end
    
    return X
end


function khatri_rao_block(W, Z)
    # If W is a vector with a single element, reshape it to a 1x1 matrix
    if (ndims(W) == 1) && (length(W) == 1)
        W = reshape(W, (1, 1))
    end

    M, _ = size(W)
    dims = size(Z)
    
    # Handle the case when Z is a 2D array (i.e., M = 1)
    if length(dims) == 2
        N = dims[1]
        P = 1
    else
        N, _, P = dims
    end
    
    # Initialize an empty block matrix
    X = zeros(M*N, M*N)
    
    # Fill in the blocks
    for i in 1:M
        for j in 1:M
            # Compute the block X_ij
            X_ij = W[i, j] * Z[:, :, min(j, P)]
            
            # Place X_ij in the appropriate block position in the final matrix X
            X[(i-1)*N+1:i*N, (j-1)*N+1:j*N] = X_ij
        end
    end
    
    return X
end

# TESTS

@testset "find_differences Function Tests" begin
    @testset "Test 1: Arrays with no differences" begin
        arr1 = [1.0, 2.0, 3.0]
        arr2 = [1.0, 2.0, 3.0]
        @test isempty(find_differences(arr1, arr2))
    end

    @testset "Test 2: Arrays with differences" begin
        arr1 = [1.0, 2.0, 3.0]
        arr2 = [1.0, 2.1, 3.0]
        differences = find_differences(arr1, arr2)
        @test length(differences) == 1
        @test differences[1] == (index=2, arr1_value=2.0, arr2_value=2.1)
    end

    @testset "Test 3: Arrays with multiple differences" begin
        arr1 = [1.0, 2.0, 3.0]
        arr2 = [1.1, 2.1, 3.1]
        differences = find_differences(arr1, arr2)
        @test length(differences) == 3
        @test differences == [(index=1, arr1_value=1.0, arr2_value=1.1), (index=2, arr1_value=2.0, arr2_value=2.1), (index=3, arr1_value=3.0, arr2_value=3.1)]
    end
    
    @testset "Test 4: Different size arrays" begin
        arr1 = [1.0, 2.0, 3.0]
        arr2 = [1.0, 2.0]
        @test_throws ErrorException find_differences(arr1, arr2)
    end
end

@testset "Dense to Sparse Tests" begin
    # Unit Test 1: Testing with a mask that includes all values
    @test begin
        dense_matrix = [1 2; 3 4]
        mask_matrix = [1 1; 1 1]
        sparse_matrix = dense_to_sparse(dense_matrix, mask_matrix)
        sparse_matrix[1] == Dict(1 => 1, 2 => 3) && sparse_matrix[2] == Dict(1 => 2, 2 => 4)
    end

    # Unit Test 2: Testing with a mask that includes some values
    @test begin
        dense_matrix = [1 2; 3 4]
        mask_matrix = [0 1; 1 0]
        sparse_matrix = dense_to_sparse(dense_matrix, mask_matrix)
        sparse_matrix[1] == Dict(2 => 3) && sparse_matrix[2] == Dict(1 => 2)
    end

    # Unit Test 3: Testing with a mask that includes no values
    @test begin
        dense_matrix = [1 2; 3 4]
        mask_matrix = [0 0; 0 0]
        sparse_matrix = dense_to_sparse(dense_matrix, mask_matrix)
        sparse_matrix[1] == Dict() && sparse_matrix[2] == Dict()
    end
end

@testset "Create Mask Matrix Tests" begin
    # Unit Test 1: Basic functionality with non-empty vectors
    @test begin
        mask_vectors = [[1, 2], [2, 3]]
        num_rows = 3
        expected_mask_matrix = [1 0; 1 1; 0 1]
        mask_matrix = create_mask_matrix(mask_vectors, num_rows)
        dense_matrix = [1 4; 2 5; 3 6]
        sparse_matrix = dense_to_sparse(dense_matrix, mask_matrix)
        all(mask_matrix .== expected_mask_matrix) && sparse_matrix[1] == Dict(1 => 1, 2 => 2) && sparse_matrix[2] == Dict(2 => 5, 3 => 6)
    end

    # Unit Test 2: Handling empty vectors
    @test begin
        mask_vectors = [[], [2, 3], []]
        num_rows = 3
        expected_mask_matrix = [0 0 0; 0 1 0; 0 1 0]
        mask_matrix = create_mask_matrix(mask_vectors, num_rows)
        dense_matrix = [1 4 7; 2 5 8; 3 6 9]
        sparse_matrix = dense_to_sparse(dense_matrix, mask_matrix)
        all(mask_matrix .== expected_mask_matrix) && sparse_matrix[1] == Dict() && sparse_matrix[2] == Dict(2 => 5, 3 => 6) && sparse_matrix[3] == Dict()
    end

    # Unit Test 3: All non-zero values in mask vectors
    @test begin
        mask_vectors = [[1, 2, 3], [1, 2, 3]]
        num_rows = 3
        expected_mask_matrix = [1 1; 1 1; 1 1]
        mask_matrix = create_mask_matrix(mask_vectors, num_rows)
        dense_matrix = [1 4; 2 5; 3 6]
        sparse_matrix = dense_to_sparse(dense_matrix, mask_matrix)
        all(mask_matrix .== expected_mask_matrix) && sparse_matrix[1] == Dict(1 => 1, 2 => 2, 3 => 3) && sparse_matrix[2] == Dict(1 => 4, 2 => 5, 3 => 6)
    end

    # Unit Test 4: All vectors are empty
    @test begin
        mask_vectors = [[], []]
        num_rows = 3
        expected_mask_matrix = [0 0; 0 0; 0 0]
        mask_matrix = create_mask_matrix(mask_vectors, num_rows)
        dense_matrix = [1 4; 2 5; 3 6]
        sparse_matrix = dense_to_sparse(dense_matrix, mask_matrix)
        all(mask_matrix .== expected_mask_matrix) && isempty(sparse_matrix[1]) && isempty(sparse_matrix[2])
    end

    # Unit Test 5: Single non-zero value in mask vectors
    @test begin
        mask_vectors = [[1], [3]]
        num_rows = 3
        expected_mask_matrix = [1 0; 0 0; 0 1]
        mask_matrix = create_mask_matrix(mask_vectors, num_rows)
        dense_matrix = [1 4; 2 5; 3 6]
        sparse_matrix = dense_to_sparse(dense_matrix, mask_matrix)
        all(mask_matrix .== expected_mask_matrix) && sparse_matrix[1] == Dict(1 => 1) && sparse_matrix[2] == Dict(3 => 6)
    end
    
end

@testset "Khatri-Rao Block Product Tests" begin

    # Test 1
    W1 = [1 3; 2 4]
    Z1 = cat([5 7; 6 8], [0.1 0.2; 0.3 0.4], dims=3)
    expected_output1 = [
        5 7 0.3 0.6; 
        6 8 0.9 1.2; 
        10 14 0.4 0.8; 
        12 16 1.2 1.6
    ]
    result = khatri_rao_block(W1, Z1)
    @test result ≈ expected_output1


    # Test 2: M = 3, N = 2
    W2 = [1 2 3; 4 5 6; 7 8 9]
    Z2 = cat([1 2; 3 4], [5 6; 7 8], [9 10; 11 12], dims=3)
    expected_output2 = [
        1 2 10 12 27 30; 
        3 4 14 16 33 36; 
        4 8 25 30 54 60; 
        12 16 35 40 66 72; 
        7 14 40 48 81 90; 
        21 28 56 64 99 108
    ]

    result2 = khatri_rao_block(W2, Z2)
    @test isempty(find_differences(result2, expected_output2))
    @test result2 ≈ expected_output2


    # Test 3: M = 2, N = 3
    W3 = [0.5 1;
          2.0 5]
    Z3 = cat([1 2 3;
              4 5 6;
              7 8 9], 
             [0.1 0.2 0.3;
              0.4 0.5 0.6;
              0.7 0.8 0.9],
             dims=3)
    expected_output3 = [
        0.5 1 1.5 0.1 0.2 0.3; 
        2 2.5 3 0.4 0.5 0.6; 
        3.5 4 4.5 0.7 0.8 0.9; 
        2 4 6 0.5 1 1.5; 
        8 10 12 2 2.5 3; 
        14 16 18 3.5 4 4.5
    ]
    @test khatri_rao_block(W3, Z3) ≈ expected_output3


    # Test 4: M = 1, N = 3
    W4 = [2.5]
    Z4 = cat([0.1 0.2 0.3;
              0.4 0.5 0.6;
              0.7 0.8 0.9], dims=3)
    expected_output4 = [
        0.25 0.5 0.75; 
        1.0 1.25 1.5; 
        1.75 2.0 2.25
    ]
    @test khatri_rao_block(W4, Z4) ≈ expected_output4


    # Test 5: M = 3, N = 1
    W5 = [1 2 3; 4 5 6; 7 8 9]
    Z5 = cat([1], [2], [3], dims=3)
    expected_output5 = [
        1 4 9; 
        4 10 18; 
        7 16 27
    ]
    result5 = khatri_rao_block(W5, Z5)
    @test isempty(find_differences(result5, expected_output5))
    @test  result5 ≈ expected_output5

end


end
