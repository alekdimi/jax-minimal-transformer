from absl.testing import absltest
from absl.testing import parameterized

from .. import model
import jax.numpy as jnp


class TestGPT(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(
            testcase_name="hello_world",
            prompt="Hello world",
            max_tokens=10,
        ),
        dict(
            testcase_name="long_before",
            prompt="Long before",
            max_tokens=10,
        ),
    )
    def test_generate(self, prompt, max_tokens):
        gpt = model.GPT.from_pretrained()
        without_kv_cache = gpt.generate(
            prompt, max_tokens=max_tokens, kv_cache=False, output_logits=True
        )
        with_kv_cache = gpt.generate(
            prompt, max_tokens=max_tokens, kv_cache=True, output_logits=True
        )
        self.assertEqual(without_kv_cache.text, with_kv_cache.text)
        self.assertTrue(jnp.all(without_kv_cache.tokens == with_kv_cache.tokens))
        assert without_kv_cache.logits is not None and with_kv_cache.logits is not None
        self.assertTrue(jnp.allclose(without_kv_cache.logits, with_kv_cache.logits))


if __name__ == "__main__":
    absltest.main()
