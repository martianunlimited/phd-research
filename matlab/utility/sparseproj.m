function[projectionMatrix] = sparseproj(d,density)

% sparseproj(k,d) returns a projection matrix dxd dimensions that is sparse.

k=floor(d*density);
gap=d-k;
randNormMatrix = randn(k,k); % create a random matrix.
projectionMatrix = [orth(randNormMatrix')' zeros(k,gap); zeros(gap,k) eye(gap)]; % orthogonalize the random matrix to generate the random projection matrix.



