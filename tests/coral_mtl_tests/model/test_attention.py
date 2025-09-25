# tests/coral_mtl/model/test_attention.py

import pytest
import torch

from coral_mtl.model.attention import MultiTaskCrossAttentionModule, _calculate_attention


@pytest.mark.gpu
class TestMultiTaskCrossAttentionModule:
    """
    Tests for the MultiTaskCrossAttentionModule, ensuring shape correctness,
    trainability, and logical consistency.
    """

    @pytest.mark.parametrize(
        "batch_size, channels, height, width",
        [
            (2, 32, 16, 16),  # Standard case
            (1, 64, 8, 8),    # Single batch item
            (4, 16, 32, 24),  # Non-square dimensions
            (8, 8, 1, 1),     # Edge case: minimal spatial dimensions
        ],
    )
    def test_forward_pass_shapes(self, device, batch_size, channels, height, width):
        """
        Verifies that the forward pass produces output tensors of the expected shape,
        matching the input dimensions. This is a critical shape contract test.
        """
        # Setup: Create the module and input tensors on the correct device.
        module = MultiTaskCrossAttentionModule(in_channels=channels).to(device)
        f_genus = torch.randn(batch_size, channels, height, width, device=device)
        f_health = torch.randn(batch_size, channels, height, width, device=device)

        # Action: Run the forward pass.
        enriched_genus, enriched_health = module(f_genus, f_health)

        # Assertion:
        # 1. The output must be a tuple of two tensors.
        assert isinstance(enriched_genus, torch.Tensor)
        assert isinstance(enriched_health, torch.Tensor)

        # 2. Each output tensor must have the same shape as the input tensors.
        expected_shape = (batch_size, channels, height, width)
        assert enriched_genus.shape == expected_shape, "Enriched genus tensor has incorrect shape"
        assert enriched_health.shape == expected_shape, "Enriched health tensor has incorrect shape"

        # 3. Output tensors should be on the same device as the input.
        assert enriched_genus.device.type == device.type
        assert enriched_health.device.type == device.type

    def test_backward_pass_and_gradients(self, device):
        """
        Ensures that gradients can be backpropagated through the module. This
        validates that the module is trainable and can be integrated into a larger model.
        """
        # Setup:
        batch_size, channels, height, width = 2, 16, 8, 8
        module = MultiTaskCrossAttentionModule(in_channels=channels).to(device)
        f_genus = torch.randn(batch_size, channels, height, width, device=device, requires_grad=True)
        f_health = torch.randn(batch_size, channels, height, width, device=device, requires_grad=True)

        # Action:
        # 1. Run the forward pass.
        enriched_genus, enriched_health = module(f_genus, f_health)

        # 2. Compute a dummy loss and backpropagate.
        # The sum of all elements is a simple scalar loss.
        loss = enriched_genus.sum() + enriched_health.sum()
        loss.backward()

        # Assertion:
        # 1. All learnable parameters in the module must have gradients.
        for name, param in module.named_parameters():
            assert param.grad is not None, f"Parameter '{name}' has no gradient."
            assert not torch.all(param.grad == 0), f"Gradient for '{name}' is all zeros."

        # 2. The input tensors that require gradients should also have them.
        assert f_genus.grad is not None, "Input tensor f_genus has no gradient."
        assert f_health.grad is not None, "Input tensor f_health has no gradient."


@pytest.mark.gpu
def test_calculate_attention_logic(device):
    """
    Unit test for the internal `_calculate_attention` pure function to verify its
    scientific correctness with a controlled, deterministic scenario.
    """
    # Setup: Create simple, known Q, K, V tensors.
    # We create a scenario where the query is perfectly aligned with the first key.
    batch_size, seq_len_q, seq_len_kv, channels = 1, 1, 2, 2
    
    # Query vector is [1.0, 0.0]
    query = torch.tensor([[[1.0, 0.0]]], device=device, dtype=torch.float32)
    
    # Key vectors are [[1.0, 0.0], [0.0, 1.0]] (one matching, one orthogonal)
    key = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], device=device, dtype=torch.float32)
    
    # Value vectors are [[10.0, 20.0], [30.0, 40.0]]
    value = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]], device=device, dtype=torch.float32)
    
    scale = 1.0 # Use scale of 1.0 for simplicity

    # Action: Call the attention function.
    output = _calculate_attention(query, key, value, scale)

    # Assertion:
    # The dot product of query and key will be [1.0, 0.0].
    # After softmax, the attention probabilities should be heavily skewed towards the first key.
    # e.g., softmax([1, 0]) = [e^1/(e^1+e^0), e^0/(e^1+e^0)] = [0.731, 0.269]
    # The output should be a weighted average of the value vectors.
    # Expected output = 0.731 * [10, 20] + 0.269 * [30, 40] = [7.31+8.07, 14.62+10.76] = [15.38, 25.38]
    
    # Let's use a large scale to make the softmax sharper (closer to one-hot)
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) * 100.0
    attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

    # With a large scale, attention_probs should be very close to [[1.0, 0.0]]
    assert torch.allclose(attention_probs, torch.tensor([[[1.0, 0.0]]], device=device), atol=1e-6)
    
    # Re-run with a high scale to confirm the logic
    output_sharp = _calculate_attention(query, key, value, scale=100.0)

    # The output should now be almost exactly the first value vector.
    expected_output = torch.tensor([[[10.0, 20.0]]], device=device)
    assert torch.allclose(output_sharp, expected_output, atol=1e-6), \
        f"Expected output {expected_output}, but got {output_sharp}"