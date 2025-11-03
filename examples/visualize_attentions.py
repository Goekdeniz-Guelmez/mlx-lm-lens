# visualization_example.py
import mlx.core as mx
import numpy as np
from mlx_lm import load

def visualize_attention_analysis():
    """
    Visualize attention patterns and hidden states for polysemous word analysis
    """
    # Load model and create lens wrapper
    print("Loading model...")
    model, tokenizer = load("Goekdeniz-Guelmez/Gabliterated-Qwen3-0.6B")
    model_lens = MLX_LM_Lens_Wrapper(model)
    
    # Test sentences with polysemous word "bank"
    sentences = [
        "I sat on a bank",
        "I went in a bank"
    ]
    
    # Tokenize sentences
    print("Tokenizing sentences...")
    tokenized_sentences = []
    for sentence in sentences:
        tokens = tokenizer.encode(sentence)
        tokenized_sentences.append({
            'text': sentence,
            'tokens': tokens,
            'token_strs': [tokenizer.decode([t]) for t in tokens]
        })
    
    # Analyze each sentence
    results = []
    for i, sent_data in enumerate(tokenized_sentences):
        print(f"\nAnalyzing: '{sent_data['text']}'")
        
        # Convert to MLX array
        tokens_array = mx.array([sent_data['tokens']])
        
        # Get lens analysis
        lens_data = model_lens(
            tokens_array,
            return_dict=True,
            return_hidden_states=True
        )
        
        # Get embeddings
        embeds = model_lens.get_embeddings(tokens_array)
        
        results.append({
            'sentence': sent_data['text'],
            'tokens': sent_data['token_strs'],
            'embeddings': embeds,
            'hidden_states': lens_data['hidden_states'],
            'logits': lens_data['logits']
        })
    
    # Create visualizations
    create_visualizations(results, model_lens)
    
    # Print analysis summary
    print_analysis_summary(results)

def mlx_to_numpy(mlx_array):
    """Safely convert MLX array to NumPy array"""
    if isinstance(mlx_array, mx.array):
        # Force evaluation and convert to numpy
        return np.array(mlx_array.tolist())
    else:
        return np.array(mlx_array)

def create_visualizations(results, model_lens):
    """Create comprehensive visualizations"""
    
    # Set up the plot
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Hidden States Evolution Heatmap
    ax1 = plt.subplot(2, 3, 1)
    visualize_hidden_states_evolution(results, ax1)
    
    # 2. Token Embeddings Comparison
    ax2 = plt.subplot(2, 3, 2)
    visualize_token_embeddings(results, ax2)
    
    # 3. Layer-wise Hidden State Norms
    ax3 = plt.subplot(2, 3, 3)
    visualize_layer_norms(results, ax3)
    
    # 4. Final Layer Activations
    ax4 = plt.subplot(2, 3, 4)
    visualize_final_activations(results, ax4)
    
    # 5. Logits Comparison
    ax5 = plt.subplot(2, 3, 5)
    visualize_logits_comparison(results, ax5)
    
    # 6. Embedding Space Visualization (PCA)
    ax6 = plt.subplot(2, 3, 6)
    visualize_embedding_space(results, ax6)
    
    plt.tight_layout()
    plt.savefig('attention_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_hidden_states_evolution(results, ax):
    """Visualize how hidden states evolve through layers"""
    for i, result in enumerate(results):
        # Get hidden states for each layer (skip embedding layer)
        hidden_states = result['hidden_states'][1:]  # Skip initial embedding
        
        # Convert to numpy and compute norms
        layer_norms = []
        for hs in hidden_states:
            # Convert MLX array to numpy safely
            hs_np = mlx_to_numpy(hs[0])  # Remove batch dimension
            norms = np.linalg.norm(hs_np, axis=-1)  # Norm across hidden dimension
            layer_norms.append(norms.mean())  # Mean across sequence
        
        ax.plot(layer_norms, label=f"'{result['sentence']}'", marker='o', linewidth=2)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Average Hidden State Norm')
    ax.set_title('Hidden State Evolution Across Layers')
    ax.legend()
    ax.grid(True, alpha=0.3)

def visualize_token_embeddings(results, ax):
    """Compare token embeddings between sentences"""
    # Find "bank" token in both sentences
    bank_embeddings = []
    labels = []
    
    for result in results:
        tokens = result['tokens']
        embeddings = mlx_to_numpy(result['embeddings'][0])  # Remove batch dim
        
        # Find "bank" token
        for j, token in enumerate(tokens):
            if 'bank' in token.lower():
                bank_embeddings.append(embeddings[j])
                labels.append(f"'{result['sentence']}'")
                break
    
    if len(bank_embeddings) >= 2:
        # Compute cosine similarity
        emb1, emb2 = bank_embeddings[0], bank_embeddings[1]
        cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        # Plot embedding norms
        norms = [np.linalg.norm(emb) for emb in bank_embeddings]
        ax.bar(labels, norms, alpha=0.7)
        ax.set_ylabel('Embedding Norm')
        ax.set_title(f'Token "bank" Embeddings\nCosine Similarity: {cosine_sim:.3f}')
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.text(0.5, 0.5, 'Could not find "bank" token', ha='center', va='center')
        ax.set_title('Token Embeddings Comparison')

def visualize_layer_norms(results, ax):
    """Visualize layer-wise hidden state norms for each token"""
    colors = ['blue', 'red']
    
    for i, result in enumerate(results):
        hidden_states = result['hidden_states']
        
        # Get norms for each layer and token
        layer_norms = []
        for hs in hidden_states:
            hs_np = mlx_to_numpy(hs[0])  # Remove batch dimension
            token_norms = np.linalg.norm(hs_np, axis=-1)  # Norm for each token
            layer_norms.append(token_norms)
        
        # Plot average norm per layer
        avg_norms = [norms.mean() for norms in layer_norms]
        ax.plot(range(len(avg_norms)), avg_norms, 
                color=colors[i], label=f"'{result['sentence']}'", 
                marker='o', linewidth=2)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Average Token Norm')
    ax.set_title('Layer-wise Hidden State Norms')
    ax.legend()
    ax.grid(True, alpha=0.3)

def visualize_final_activations(results, ax):
    """Visualize final layer activations"""
    final_activations = []
    labels = []
    
    for result in results:
        # Get final hidden state (before logits)
        final_hs = result['hidden_states'][-1]
        final_hs_np = mlx_to_numpy(final_hs[0])  # Remove batch dim
        
        # Take mean across sequence length
        mean_activation = final_hs_np.mean(axis=0)
        final_activations.append(mean_activation)
        labels.append(result['sentence'])
    
    # Plot histogram of final activations
    for i, (activations, label) in enumerate(zip(final_activations, labels)):
        ax.hist(activations, bins=50, alpha=0.6, label=f"'{label}'", density=True)
    
    ax.set_xlabel('Activation Value')
    ax.set_ylabel('Density')
    ax.set_title('Final Layer Activation Distribution')
    ax.legend()

def visualize_logits_comparison(results, ax):
    """Compare top logits between sentences"""
    for i, result in enumerate(results):
        logits = mlx_to_numpy(result['logits'][0, -1])  # Last token logits
        
        # Get top 10 logits
        top_indices = np.argsort(logits)[-10:]
        top_logits = logits[top_indices]
        
        ax.barh(range(len(top_logits)), top_logits, 
                alpha=0.7, label=f"'{result['sentence']}'")
    
    ax.set_xlabel('Logit Value')
    ax.set_ylabel('Top Predictions')
    ax.set_title('Top 10 Logits Comparison')
    ax.legend()

def visualize_embedding_space(results, ax):
    """Visualize embedding space using PCA"""
    from sklearn.decomposition import PCA
    
    all_embeddings = []
    colors = []
    labels = []
    
    color_map = ['blue', 'red']
    
    for i, result in enumerate(results):
        embeddings = mlx_to_numpy(result['embeddings'][0])  # Remove batch dim
        tokens = result['tokens']
        
        for j, token in enumerate(tokens):
            all_embeddings.append(embeddings[j])
            colors.append(color_map[i])
            labels.append(f"{token} ({result['sentence'][:10]}...)")
    
    # Apply PCA
    if len(all_embeddings) > 0:
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(all_embeddings)
        
        # Plot
        for i, (emb, color, label) in enumerate(zip(embeddings_2d, colors, labels)):
            ax.scatter(emb[0], emb[1], c=color, alpha=0.7, s=100)
            ax.annotate(label.split()[0], (emb[0], emb[1]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_title('Token Embeddings in 2D Space (PCA)')
        ax.grid(True, alpha=0.3)

def print_analysis_summary(results):
    """Print summary of analysis"""
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    
    for result in results:
        print(f"\nSentence: '{result['sentence']}'")
        print(f"Tokens: {result['tokens']}")
        print(f"Number of layers: {len(result['hidden_states'])}")
        
        # Final logits info
        final_logits = mlx_to_numpy(result['logits'][0, -1])
        top_token_idx = np.argmax(final_logits)
        print(f"Top predicted token index: {top_token_idx}")
        print(f"Max logit value: {final_logits[top_token_idx]:.3f}")

if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        from mlx_lm_lens.lm_lens import MLX_LM_Lens_Wrapper
    except ImportError:
        print("Installing scikit-learn for PCA visualization...")
        import subprocess
        subprocess.run(["pip", "install", "scikit-learn", "matplotlib", "mlx_lm_lens"])
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        from mlx_lm_lens.lm_lens import MLX_LM_Lens_Wrapper
    
    visualize_attention_analysis()