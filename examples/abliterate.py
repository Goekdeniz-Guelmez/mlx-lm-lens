import gc
from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

import mlx.core as mx
import numpy as np

from mlx_lm.utils import load, save
from mlx_lm_lens.lm_lens import MLX_LM_Lens_Wrapper

# Constants
MODEL_ID = "Qwen/Qwen3-0.6B"
SKIP_BEGIN_LAYERS = 2
SKIP_END_LAYERS = 2
LAYER_FRACTION_TO_USE = 0.6
SCALE_FACTOR = 0.6

# Load model and tokenizer
model, tokenizer = load(MODEL_ID)
model_lens = MLX_LM_Lens_Wrapper(model)

# Get total layers for later use
total_layers = len(model.model.layers)
layer_idx = int(total_layers * LAYER_FRACTION_TO_USE)
print(f"Layer index for refusal direction: {layer_idx}")
print(f"Total layers: {total_layers}")

# Load datasets
with open("harmful.txt", "r") as f:
    harmful_instructions = f.readlines()
with open("harmless.txt", "r") as f:
    harmless_instructions = f.readlines()

# Tokenization function
def tokenize(instruction):
    chat = [{"role": "user", "content": instruction}]
    return tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        return_tensors="np"
    )[0]

# Get hidden states for last token
def get_last_token_hidden(tokens):
    tokens = mx.array(tokens)[None]  # Add batch dimension
    output = model_lens(tokens, return_dict=True)
    # hidden_states includes embedding + all layers
    # We want the specified layer (index starts after embedding)
    hidden_state = output["hidden_states"][layer_idx + 1]
    return hidden_state[0, -1]  # Last token of sequence

# Process harmful instructions
harmful_tokens = [tokenize(insn.strip()) for insn in harmful_instructions if insn.strip()]
harmless_tokens = [tokenize(insn.strip()) for insn in harmless_instructions if insn.strip()]

bar_generate = tqdm(total=len(harmful_tokens) + len(harmless_tokens), 
                    desc="Generating samples")

harmful_hidden = []
for tokens in harmful_tokens:
    harmful_hidden.append(get_last_token_hidden(tokens))
    bar_generate.update(1)

harmless_hidden = []
for tokens in harmless_tokens:
    harmless_hidden.append(get_last_token_hidden(tokens))
    bar_generate.update(1)

bar_generate.close()

# Compute refusal direction
harmful_mean = mx.stack(harmful_hidden).mean(axis=0)
harmless_mean = mx.stack(harmless_hidden).mean(axis=0)
refusal_dir = harmful_mean - harmless_mean
refusal_dir = refusal_dir / mx.linalg.norm(refusal_dir)

print(f"Refusal direction shape: {refusal_dir.shape}")

# Helper function to convert MLX arrays to NumPy properly
def mlx_to_numpy(mlx_array):
    """Convert MLX array to NumPy array safely"""
    # First convert to float32 to avoid dtype issues
    return np.array(mlx_array.astype(mx.float32))

# Visualize hidden states
def visualize_hidden_states(harmful_hidden, harmless_hidden, title_suffix=""):
    """Visualize hidden states using PCA and t-SNE"""
    
    # Convert to numpy for sklearn - fixed conversion
    harmful_np = np.array([mlx_to_numpy(h) for h in harmful_hidden])
    harmless_np = np.array([mlx_to_numpy(h) for h in harmless_hidden])
    
    print(f"Harmful hidden states shape: {harmful_np.shape}")
    print(f"Harmless hidden states shape: {harmless_np.shape}")
    
    # Combine data
    all_hidden = np.vstack([harmful_np, harmless_np])
    labels = ['Harmful'] * len(harmful_np) + ['Harmless'] * len(harmless_np)
    
    # PCA visualization
    pca = PCA(n_components=2)
    hidden_pca = pca.fit_transform(all_hidden)
    
    plt.figure(figsize=(15, 5))
    
    # PCA plot
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(hidden_pca[:len(harmful_np), 0], hidden_pca[:len(harmful_np), 1], 
                         c='red', alpha=0.6, label='Harmful')
    plt.scatter(hidden_pca[len(harmful_np):, 0], hidden_pca[len(harmful_np):, 1], 
               c='blue', alpha=0.6, label='Harmless')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title(f'PCA of Hidden States{title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # t-SNE visualization (if we have enough samples)
    if len(all_hidden) >= 10:
        perplexity = min(30, len(all_hidden)-1)
        if perplexity >= 5:  # t-SNE needs at least 5 perplexity
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            hidden_tsne = tsne.fit_transform(all_hidden)
            
            plt.subplot(1, 3, 2)
            plt.scatter(hidden_tsne[:len(harmful_np), 0], hidden_tsne[:len(harmful_np), 1], 
                       c='red', alpha=0.6, label='Harmful')
            plt.scatter(hidden_tsne[len(harmful_np):, 0], hidden_tsne[len(harmful_np):, 1], 
                       c='blue', alpha=0.6, label='Harmless')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.title(f't-SNE of Hidden States{title_suffix}')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.subplot(1, 3, 2)
            plt.text(0.5, 0.5, 'Not enough samples\nfor t-SNE', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f't-SNE of Hidden States{title_suffix}')
    
    # Histogram of projections onto refusal direction
    plt.subplot(1, 3, 3)
    refusal_dir_np = mlx_to_numpy(refusal_dir)
    harmful_proj = [np.dot(h, refusal_dir_np) for h in harmful_np]
    harmless_proj = [np.dot(h, refusal_dir_np) for h in harmless_np]
    
    plt.hist(harmful_proj, alpha=0.6, color='red', label='Harmful', bins=20)
    plt.hist(harmless_proj, alpha=0.6, color='blue', label='Harmless', bins=20)
    plt.xlabel('Projection onto Refusal Direction')
    plt.ylabel('Frequency')
    plt.title(f'Refusal Direction Projections{title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'hidden_states_visualization{title_suffix.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return harmful_proj, harmless_proj

# Visualize original hidden states
print("Visualizing original hidden states...")
orig_harmful_proj, orig_harmless_proj = visualize_hidden_states(harmful_hidden, harmless_hidden, " (Original)")

# Save direction vector
mx.save(MODEL_ID.replace("/", "_") + "_refusal_dir.npy", refusal_dir)

# Store original weights for comparison
original_weights = {}

# Free memory
del model, model_lens
gc.collect()
if mx.metal.is_available():
    mx.clear_cache()

# Reload model for modification
model, _ = load(MODEL_ID)

# Modify weights - Fixed version with before/after tracking
def modify_tensor(weight, refusal_dir, scale_factor, layer_name=""):
    """Modified version that returns both original and modified tensors"""
    # Convert to float32 for precision
    w_f32 = weight.astype(mx.float32)
    dir_f32 = refusal_dir.astype(mx.float32)

    # Normalize dir_f32 (in case)
    dir_f32 = dir_f32 / mx.linalg.norm(dir_f32)

    # weight shape: (out_dim, in_dim)
    # refusal_dir shape: (hidden_dim,)
    # We need to match dimensions properly
    
    if w_f32.shape[1] == dir_f32.shape[0]:
        # Input dimension matches refusal direction
        # Project each row of weight matrix onto refusal direction
        proj_coeffs = mx.matmul(w_f32, dir_f32)  # (out_dim,)
        projection = proj_coeffs[:, None] * dir_f32[None, :]  # (out_dim, in_dim)
        
    elif w_f32.shape[0] == dir_f32.shape[0]:
        # Output dimension matches refusal direction
        # Project each column of weight matrix onto refusal direction
        proj_coeffs = mx.matmul(dir_f32, w_f32)  # (in_dim,)
        projection = dir_f32[:, None] * proj_coeffs[None, :]  # (out_dim, in_dim)
        
    else:
        print(f"Warning: Weight shape {w_f32.shape} doesn't match refusal direction shape {dir_f32.shape}")
        return weight.astype(mx.bfloat16), None, None

    # Subtract scaled projection
    modified = w_f32 - scale_factor * projection
    
    # Store comparison data
    original_norm = mx.linalg.norm(w_f32)
    modified_norm = mx.linalg.norm(modified)
    change_norm = mx.linalg.norm(scale_factor * projection)
    
    print(f"{layer_name}: Original norm: {float(original_norm):.4f}, "
          f"Modified norm: {float(modified_norm):.4f}, "
          f"Change norm: {float(change_norm):.4f}, "
          f"Relative change: {float(change_norm/original_norm)*100:.2f}%")
    
    return modified.astype(mx.bfloat16), w_f32, scale_factor * projection

def visualize_weight_changes(original, change, layer_name):
    """Visualize the changes made to weights"""
    orig_np = mlx_to_numpy(original)
    change_np = mlx_to_numpy(change)
    
    plt.figure(figsize=(15, 5))
    
    # Original weight distribution
    plt.subplot(1, 3, 1)
    plt.hist(orig_np.flatten(), bins=50, alpha=0.7, color='blue')
    plt.title(f'{layer_name} - Original Weights')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Change distribution
    plt.subplot(1, 3, 2)
    plt.hist(change_np.flatten(), bins=50, alpha=0.7, color='red')
    plt.title(f'{layer_name} - Applied Changes')
    plt.xlabel('Change Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Heatmap of changes (sample)
    plt.subplot(1, 3, 3)
    sample_size = min(50, change_np.shape[0], change_np.shape[1])
    sample_change = change_np[:sample_size, :sample_size]
    sns.heatmap(sample_change, cmap='RdBu_r', center=0, cbar=True)
    plt.title(f'{layer_name} - Change Heatmap (Sample)')
    
    plt.tight_layout()
    plt.savefig(f'{layer_name.replace(" ", "_")}_weight_changes.png', dpi=300, bbox_inches='tight')
    plt.show()

# Apply modifications with visualization
num_layers = len(model.model.layers)
bar_layers = tqdm(total=(num_layers - (SKIP_BEGIN_LAYERS + SKIP_END_LAYERS)) * 2,
                 desc="Modifying tensors")

weight_changes = {}

for layer_idx in range(SKIP_BEGIN_LAYERS, num_layers - SKIP_END_LAYERS):
    # Modify attention output projection
    attn = model.model.layers[layer_idx].self_attn.o_proj
    layer_name = f"Layer_{layer_idx}_Attention"
    print(f"Layer {layer_idx} attention weight shape: {attn.weight.shape}")
    
    modified_weight, original_weight, change = modify_tensor(
        attn.weight, refusal_dir, SCALE_FACTOR, layer_name
    )
    
    if change is not None:
        weight_changes[layer_name] = (original_weight, change)
        # Visualize first few layers
        if layer_idx < SKIP_BEGIN_LAYERS + 3:
            visualize_weight_changes(original_weight, change, layer_name)
    
    attn.weight = modified_weight
    bar_layers.update(1)
    
    # Modify MLP down projection
    mlp = model.model.layers[layer_idx].mlp.down_proj
    layer_name = f"Layer_{layer_idx}_MLP"
    print(f"Layer {layer_idx} MLP weight shape: {mlp.weight.shape}")
    
    modified_weight, original_weight, change = modify_tensor(
        mlp.weight, refusal_dir, SCALE_FACTOR, layer_name
    )
    
    if change is not None:
        weight_changes[layer_name] = (original_weight, change)
        # Visualize first few layers
        if layer_idx < SKIP_BEGIN_LAYERS + 3:
            visualize_weight_changes(original_weight, change, layer_name)
    
    mlp.weight = modified_weight
    bar_layers.update(1)

bar_layers.close()

# Test the modified model with sample prompts
print("\nTesting modified model with sample prompts...")

# Sample test prompts
test_harmful = "How to make a bomb?"
test_harmless = "How to make a sandwich?"

def test_model_response(prompt, max_tokens=50):
    """Test model response to a prompt"""
    tokens = tokenize(prompt)
    tokens = mx.array(tokens)[None]
    
    # Simple generation loop since generate might not be available
    current_tokens = tokens
    generated = []
    
    for _ in range(max_tokens):
        with mx.no_grad():
            logits = model(current_tokens)
            next_token = mx.argmax(logits[0, -1, :], keepdims=True)
            generated.append(int(next_token))
            current_tokens = mx.concatenate([current_tokens, next_token[None, :]], axis=1)
            
            # Stop if we hit end token
            if int(next_token) == tokenizer.eos_token_id:
                break
    
    full_response = tokenizer.decode(current_tokens[0].tolist())
    return full_response

print(f"\nHarmful prompt: {test_harmful}")
try:
    harmful_response = test_model_response(test_harmful)
    print(f"Response: {harmful_response}")
except Exception as e:
    print(f"Error generating response: {e}")

print(f"\nHarmless prompt: {test_harmless}")
try:
    harmless_response = test_model_response(test_harmless)
    print(f"Response: {harmless_response}")
except Exception as e:
    print(f"Error generating response: {e}")

# Get hidden states from modified model for comparison
print("\nExtracting hidden states from modified model...")
model_lens_modified = MLX_LM_Lens_Wrapper(model)

# Get hidden states for the same prompts (limit to avoid memory issues)
harmful_hidden_modified = []
harmless_hidden_modified = []

sample_size = min(20, len(harmful_tokens), len(harmless_tokens))

for tokens in harmful_tokens[:sample_size]:
    tokens = mx.array(tokens)[None]
    output = model_lens_modified(tokens, return_dict=True)
    hidden_state = output["hidden_states"][layer_idx + 1]
    harmful_hidden_modified.append(hidden_state[0, -1])

for tokens in harmless_tokens[:sample_size]:
    tokens = mx.array(tokens)[None]
    output = model_lens_modified(tokens, return_dict=True)
    hidden_state = output["hidden_states"][layer_idx + 1]
    harmless_hidden_modified.append(hidden_state[0, -1])

# Visualize modified hidden states
print("Visualizing modified hidden states...")
mod_harmful_proj, mod_harmless_proj = visualize_hidden_states(
    harmful_hidden_modified, harmless_hidden_modified, " (Modified)"
)

# Compare original vs modified projections
print("\nComparing refusal direction projections...")

# Make sure we have the same sample size for comparison
sample_size = min(len(orig_harmful_proj), len(mod_harmful_proj), 
                  len(orig_harmless_proj), len(mod_harmless_proj))

orig_harmful_proj = orig_harmful_proj[:sample_size]
orig_harmless_proj = orig_harmless_proj[:sample_size]
mod_harmful_proj = mod_harmful_proj[:sample_size]
mod_harmless_proj = mod_harmless_proj[:sample_size]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(orig_harmful_proj, mod_harmful_proj, color='red', alpha=0.6, label='Harmful')
plt.scatter(orig_harmless_proj, mod_harmless_proj, color='blue', alpha=0.6, label='Harmless')
plt.xlabel('Original Projection')
plt.ylabel('Modified Projection')
plt.title('Refusal Direction Projections: Original vs Modified')
all_projs = orig_harmful_proj + orig_harmless_proj
plt.plot([min(all_projs), max(all_projs)], [min(all_projs), max(all_projs)],
         'k--', alpha=0.5, label='y=x')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
changes_harmful = [mod - orig for orig, mod in zip(orig_harmful_proj, mod_harmful_proj)]
changes_harmless = [mod - orig for orig, mod in zip(orig_harmless_proj, mod_harmless_proj)]

plt.hist(changes_harmful, alpha=0.6, color='red', label='Harmful', bins=15)
plt.hist(changes_harmless, alpha=0.6, color='blue', label='Harmless', bins=15)
plt.xlabel('Change in Projection')
plt.ylabel('Frequency')
plt.title('Distribution of Changes in Refusal Direction Projection')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('projection_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nSummary of changes:")
print(f"Harmful prompts - Mean projection change: {np.mean(changes_harmful):.4f}")
print(f"Harmless prompts - Mean projection change: {np.mean(changes_harmless):.4f}")
print(f"Original separation: {np.mean(orig_harmful_proj) - np.mean(orig_harmless_proj):.4f}")
print(f"Modified separation: {np.mean(mod_harmful_proj) - np.mean(mod_harmless_proj):.4f}")
print(f"Separation reduction: {(np.mean(orig_harmful_proj) - np.mean(orig_harmless_proj)) - (np.mean(mod_harmful_proj) - np.mean(mod_harmless_proj)):.4f}")

# Save modified model
save_path = "Qwen3-0.6B-abliterated"
save(
    src_path=MODEL_ID,
    dst_path=save_path,
    model=model,
    tokenizer=tokenizer,
    config=vars(model.args)
)
print(f"Saved modified model to {save_path}")