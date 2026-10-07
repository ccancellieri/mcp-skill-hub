"""
Unit tests for Fast Random Projection (FastRP) module.

Tests coverage:
- Basic projection shape and correctness
- Distance preservation approximation
- Edge cases (zero vectors, single vector, different seeds)
- Orthogonal projection (when applicable)
"""

import pytest

np = pytest.importorskip("numpy", reason="FastRP requires the optional fastrp extra")

import skill_hub.fastrp as fastrp


class TestFastRPProjection:
    """Test the core fast_rp_projection function."""

    def test_output_shape(self):
        """Projected vectors should have correct shape."""
        vectors = np.random.randn(100, 768)  # 100 vectors, 768-dim
        result = fastrp.fast_rp_projection(vectors, n_components=128)
        assert result.shape == (100, 128)

    def test_output_dimension_match(self):
        """Projected vectors should have n_components equal to requested dim."""
        vectors = np.random.randn(50, 384)
        result = fastrp.fast_rp_projection(vectors, n_components=64)
        assert result.shape[1] == 64

    def test_zero_vector_handling(self):
        """Zero vectors should remain zero after projection."""
        vectors = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        result = fastrp.fast_rp_projection(vectors, n_components=2)
        # First row is the zero vector, should remain approximately zero
        assert np.allclose(result[0], 0.0, atol=1e-10)
        # Second row is non-zero, its projection should not be all zeros
        assert not np.allclose(result[1], 0.0)

    def test_single_vector(self):
        """Single vector should project correctly."""
        vectors = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        result = fastrp.fast_rp_projection(vectors, n_components=3)
        assert result.shape == (1, 3)

    def test_seed_reproducibility(self):
        """Same seed should produce identical projections."""
        vectors = np.random.randn(20, 256)
        result1 = fastrp.fast_rp_projection(vectors, n_components=16, seed=42)
        result2 = fastrp.fast_rp_projection(vectors, n_components=16, seed=42)
        np.testing.assert_array_equal(result1, result2)

    def test_different_seeds_different_results(self):
        """Different seeds should produce different projections."""
        vectors = np.random.randn(10, 128)
        result_a = fastrp.fast_rp_projection(vectors, n_components=64, seed=123)
        result_b = fastrp.fast_rp_projection(vectors, n_components=64, seed=456)
        # They should differ (not necessarily all different, but not guaranteed identical)
        # At minimum, they shouldn't be identical
        assert not np.allclose(result_a, result_b)

    def test_orthogonal_projection(self):
        """Orthogonal projection should be called when requested."""
        vectors = np.random.randn(30, 512)
        result = fastrp.fast_rp_projection(vectors, n_components=32, orthogonal=True)
        assert result.shape == (30, 32)

    def test_non_orthogonal_projection(self):
        """Non-orthogonal projection should work with n_components <= n_orig."""
        vectors = np.random.randn(20, 256)
        result = fastrp.fast_rp_projection(vectors, n_components=128)
        assert result.shape == (20, 128)

    def test_out_of_range_n_components(self):
        """Should raise error when n_components > original dim (non-orthogonal)."""
        vectors = np.random.randn(10, 64)
        with pytest.raises(ValueError, match="n_components.*exceeds.*original"):
            fastrp.fast_rp_projection(vectors, n_components=100)

    def test_negative_n_components(self):
        """Should raise error for negative n_components."""
        vectors = np.random.randn(5, 128)
        with pytest.raises(ValueError):
            fastrp.fast_rp_projection(vectors, n_components=-1)


class TestFastRPBatch:
    """Test the batch projection utility."""

    def test_batch_empty_input(self):
        """Empty iterator should return empty array."""
        result = fastrp.fast_rp_batch([], n_components=16)
        assert result.shape == (0, 16)

    def test_batch_with_vectors(self):
        """Batch should project vectors in chunks."""
        vectors = np.random.randn(100, 256)
        result = fastrp.fast_rp_batch(vectors, n_components=64, batch_size=50)
        assert result.shape == (100, 64)

    def test_batch_returns_correct_shape(self):
        """Batch output shape matches expected."""
        vectors = np.random.randn(200, 128)
        result = fastrp.fast_rp_batch(vectors, n_components=32, batch_size=100)
        assert result.shape == (200, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

def test_default_batch_transform_is_shared_across_chunks():
    vectors = np.random.default_rng(22).normal(size=(23, 32))
    expected = fastrp.fast_rp_projection(vectors, 8)
    np.testing.assert_allclose(fastrp.fast_rp_batch(iter(vectors), 8, batch_size=4), expected,
                               rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(fastrp.fast_rp_projection(vectors[0], 8), expected[0],
                               rtol=2e-6, atol=1e-6)


def test_projection_preserves_average_squared_distance():
    vectors = np.random.default_rng(3).normal(size=(2000, 384))
    projected = fastrp.fast_rp_projection(vectors, 128, seed=7)
    ratio = np.mean(np.sum(projected**2, axis=1)) / np.mean(np.sum(vectors**2, axis=1))
    assert 0.94 < ratio < 1.06
    assert projected.dtype == np.float32


@pytest.mark.parametrize("kwargs", [dict(n_components=0), dict(n_components=True),
                                    dict(n_components=1, batch_size=0),
                                    dict(n_components=1, batch_size=-1)])
def test_batch_invalid_sizes_fail_even_if_empty(kwargs):
    with pytest.raises(ValueError):
        fastrp.fast_rp_batch([], **kwargs)


@pytest.mark.parametrize("vectors", [np.ones((2, 2, 2)), [[np.nan, 1]], [[np.inf, 1]], []])
def test_invalid_vectors_fail(vectors):
    with pytest.raises(ValueError):
        fastrp.fast_rp_projection(vectors, 1)


def test_batch_inconsistent_dimensions_fail():
    with pytest.raises(ValueError):
        fastrp.fast_rp_batch([[1, 2], [1, 2, 3]], 1, batch_size=1)


def test_orthogonal_full_dimension_preserves_distances():
    vectors = np.random.default_rng(4).normal(size=(12, 8))
    projected = fastrp.fast_rp_projection(vectors, 8, orthogonal=True)
    np.testing.assert_allclose(np.sum((vectors[1:] - vectors[0])**2, axis=1),
                               np.sum((projected[1:] - projected[0])**2, axis=1), rtol=1e-6)
