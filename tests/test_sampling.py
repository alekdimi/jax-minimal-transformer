from multiprocessing import Value
from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp

from .. import sampling

class TestSampling(parameterized.TestCase):

    def test_greedy(self):
        logits = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = sampling.sample(logits, method="greedy")
        self.assertTrue(jnp.array_equal(result, jnp.array([[2], [2]])))

    @parameterized.parameters(
        {"k": 1, "num_samples": 4},
        {"k": 2, "num_samples": 3},
        {"k": 3, "num_samples": 2},
        {"k": 4, "num_samples": 1},
    )
    def test_top_k(self, k, num_samples):
        logits = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        result = sampling.sample(logits, method="top_k", k=k, num_samples=num_samples)
        # raise ValueError(k, num_samples, result.shape)
        #raise ValueError(result)
        self.assertEqual(result.shape, (2, num_samples))
        raise ValueError(k, num_samples, result)
        self.assertTrue(jnp.all(result > 0.0))

    def test_top_k_invalid_k(self):
        logits = jnp.array([[1.0, 2.0, 3.0]])
        with self.assertRaises(AssertionError):
            sampling._top_k(logits, k=0)
        with self.assertRaises(AssertionError):
            sampling._top_k(logits, k=4)

    # def test_nucleus(self):
    #     logits = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    #     result = sampling._nucleus(logits, p=0.9)
    #     self.assertEqual(result.shape, (2, 1))
    #     self.assertTrue(jnp.all(result < 3))

    def test_nucleus_invalid_p(self):
        logits = jnp.array([[1.0, 2.0, 3.0]])
        with self.assertRaises(AssertionError):
            sampling._nucleus(logits, p=0)
        with self.assertRaises(AssertionError):
            sampling._nucleus(logits, p=1)

    @parameterized.parameters(
        sampling.SamplingMethod.GREEDY,
        sampling.SamplingMethod.TOP_K,
        sampling.SamplingMethod.NUCLEUS,
    )
    def test_sample(self, method):
        logits = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = sampling.sample(logits, method=method)
        self.assertEqual(result.shape, (2, 1))
        self.assertTrue(jnp.all(result < 3))

    def test_sample_invalid_method(self):
        logits = jnp.array([[1.0, 2.0, 3.0]])
        with self.assertRaises(ValueError):
            sampling.sample(logits, method="invalid_method")

if __name__ == '__main__':
    absltest.main()