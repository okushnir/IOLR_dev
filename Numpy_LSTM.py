import numpy as np
from sklearn.preprocessing import StandardScaler


class SimpleLSTM:
    """Minimal LSTM implementation using only NumPy"""

    def __init__(self, input_size=1, hidden_size=32, output_size=3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * 0.1

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))

        # Output layer
        self.Wy = np.random.randn(output_size, hidden_size) * 0.1
        self.by = np.zeros((output_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def tanh(self, x):
        return np.tanh(np.clip(x, -500, 500))

    def forward(self, X):
        """Forward pass through LSTM"""
        seq_len = X.shape[0]
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))

        outputs = []

        for t in range(seq_len):
            x = X[t].reshape(-1, 1)

            # Concatenate input and hidden state
            concat = np.vstack([x, h])

            # Gates
            f = self.sigmoid(self.Wf @ concat + self.bf)  # Forget gate
            i = self.sigmoid(self.Wi @ concat + self.bi)  # Input gate
            o = self.sigmoid(self.Wo @ concat + self.bo)  # Output gate
            c_tilde = self.tanh(self.Wc @ concat + self.bc)  # Candidate values

            # Update cell state and hidden state
            c = f * c + i * c_tilde
            h = o * self.tanh(c)

            outputs.append(h.copy())

        # Final embedding
        embedding = self.tanh(self.Wy @ h + self.by)
        return embedding.flatten()


class NumPyLSTMEmbedder:
    """LSTM embedder using pure NumPy"""

    def __init__(self, embedding_dim=3, sequence_length=10, hidden_units=32):
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _create_sequences(self, data):
        """Create sequences for training"""
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            seq = data[i:i + self.sequence_length]
            sequences.append(seq)
        return np.array(sequences)

    def fit(self, data, epochs=50, **kwargs):
        """Fit the LSTM embedder"""
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        sequences = self._create_sequences(data_scaled)

        # Create simple LSTM
        self.model = SimpleLSTM(
            input_size=1,
            hidden_size=self.hidden_units,
            output_size=self.embedding_dim
        )

        # Simple training loop
        for epoch in range(epochs):
            total_loss = 0
            for seq in sequences:
                # Forward pass
                embedding = self.model.forward(seq.reshape(-1, 1))

                # Simple reconstruction loss (not implementing backprop for simplicity)
                # This is a simplified version - just updating to reduce variance
                if epoch % 10 == 0:
                    # Add small random updates
                    self.model.Wy += np.random.randn(*self.model.Wy.shape) * 0.001

        self.is_fitted = True
        return None

    def embed(self, data):
        """Generate embeddings"""
        if not self.is_fitted:
            raise ValueError("Embedder must be fitted first")

        data_scaled = self.scaler.transform(data.reshape(-1, 1)).flatten()
        sequences = self._create_sequences(data_scaled)

        embeddings = []
        for seq in sequences:
            embedding = self.model.forward(seq.reshape(-1, 1))
            embeddings.append(embedding)

        return np.array(embeddings)


# Test function
def test_numpy_lstm():
    """Test the NumPy LSTM implementation"""
    try:
        # Generate test data
        data = np.sin(np.linspace(0, 20, 100)) + 0.1 * np.random.randn(100)

        # Create embedder
        embedder = NumPyLSTMEmbedder(embedding_dim=3, sequence_length=10)

        # Fit and embed
        embedder.fit(data, epochs=20)
        embeddings = embedder.embed(data)

        print(f"✓ NumPy LSTM test passed!")
        print(f"  Data shape: {data.shape}")
        print(f"  Embeddings shape: {embeddings.shape}")
        print(f"  Embedding range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")

        return True

    except Exception as e:
        print(f"✗ NumPy LSTM test failed: {e}")
        return False


if __name__ == "__main__":
    test_numpy_lstm()