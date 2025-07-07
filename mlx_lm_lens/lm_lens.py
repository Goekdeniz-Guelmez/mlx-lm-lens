from typing import Optional, List, Dict, Any

import mlx.core as mx
import mlx.nn as nn


class MLX_LM_Lens_Wrapper:
    def __init__(self, model):
        self.model = model
        self.components = self._identify_model_components()
        
    def _identify_model_components(self) -> Dict[str, Any]:
        """Dynamically identify model components using common naming patterns"""
        components = {
            'embeddings': None,
            'transformer_layers': None,
            'norm': None,
            'lm_head': None,
            'tie_embeddings': False
        }
        
        # Helper to search through attributes
        def find_component(obj, candidates, recursive=True):
            for name in candidates:
                if hasattr(obj, name):
                    return getattr(obj, name)
            if recursive and hasattr(obj, 'model'):
                return find_component(obj.model, candidates, False)
            return None
        
        # 1. Check for tied embeddings (multiple ways to detect)
        components['tie_embeddings'] = (
            getattr(self.model, 'tie_word_embeddings', False) or
            getattr(self.model, 'tie_embeddings', False) or
            getattr(getattr(self.model, 'args', None), 'tie_word_embeddings', False) or
            getattr(getattr(self.model, 'config', None), 'tie_word_embeddings', False)
        )
        
        # 2. Identify embeddings
        candidates = ['embed_tokens', 'tok_emb', 'embed', 'embeddings', 'wte']
        components['embeddings'] = find_component(self.model, candidates)
        
        # 3. Identify transformer layers
        candidates = ['layers', 'h', 'blocks', 'transformer_blocks', 'encoder', 'decoder']
        components['transformer_layers'] = find_component(self.model, candidates)
        
        # 4. Identify normalization layer
        candidates = ['norm', 'ln_f', 'final_norm', 'final_layernorm']
        components['norm'] = find_component(self.model, candidates)
        
        # 5. Identify LM head (try multiple approaches)
        candidates = ['lm_head', 'head', 'output', 'lm_head_module']
        components['lm_head'] = find_component(self.model, candidates)
        
        # Special handling for model structure like in the provided example
        if hasattr(self.model, 'model') and not components['transformer_layers']:
            inner_model = self.model.model
            if hasattr(inner_model, 'layers'):
                components['transformer_layers'] = inner_model.layers
            if hasattr(inner_model, 'embed_tokens') and not components['embeddings']:
                components['embeddings'] = inner_model.embed_tokens
            if hasattr(inner_model, 'norm') and not components['norm']:
                components['norm'] = inner_model.norm
        
        # 6. Handle tied embeddings case - BEFORE validation
        if not components['lm_head'] and components['embeddings']:
            # Check if embeddings can be used as output layer
            if hasattr(components['embeddings'], 'as_linear'):
                # Model has as_linear method (like Qwen)
                components['lm_head'] = components['embeddings']
                components['tie_embeddings'] = True
                print("Using embeddings as LM head via as_linear method")
            elif hasattr(components['embeddings'], 'weight'):
                # Model has weight matrix that can be transposed
                components['lm_head'] = components['embeddings']
                components['tie_embeddings'] = True
                print("Using embeddings as LM head via weight transpose")
            else:
                # Try to detect if this is a tied embedding model by checking common patterns
                model_name = str(type(self.model)).lower()
                if any(name in model_name for name in ['llama', 'qwen', 'mistral', 'phi']):
                    components['lm_head'] = components['embeddings']
                    components['tie_embeddings'] = True
                    print(f"Assuming tied embeddings for {model_name}")
        
        # 7. If tie_embeddings is True from args/config but no lm_head found, use embeddings
        if components['tie_embeddings'] and not components['lm_head'] and components['embeddings']:
            components['lm_head'] = components['embeddings']
            print("Using embeddings as LM head based on tie_embeddings flag")
        
        # Validation
        if not components['embeddings']:
            raise RuntimeError("Could not identify embedding layer")
        if not components['transformer_layers']:
            raise RuntimeError("Could not identify transformer layers")
        if not components['lm_head']:
            # Last resort: use embeddings as lm_head
            if components['embeddings']:
                components['lm_head'] = components['embeddings']
                components['tie_embeddings'] = True
                print("Fallback: Using embeddings as LM head")
            else:
                raise RuntimeError("Could not identify LM head and no embeddings available")
        
        print(f"Identified components: "
              f"Embeddings={type(components['embeddings']).__name__}, "
              f"Layers={len(components['transformer_layers'])}, "
              f"Norm={components['norm'] is not None}, "
              f"LM Head={type(components['lm_head']).__name__}, "
              f"Tied={components['tie_embeddings']}")
        
        return components

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[List[mx.array]] = None,
        return_dict: bool = False,
        return_attention_scores: bool = False,
        return_hidden_states: bool = True,
    ):
        # Run manual forward pass to capture states
        result = self.manual_forward_with_states(inputs, mask, cache)
        
        if return_dict:
            output_dict = {"logits": result["logits"]}
            if return_hidden_states:
                output_dict["hidden_states"] = result["hidden_states"]
            if return_attention_scores:
                output_dict["attention_scores"] = result["attention_scores"]
            return output_dict
        else:
            return result["logits"]

    def manual_forward_with_states(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[List[mx.array]] = None,
    ) -> Dict[str, Any]:
        """Core forward pass that captures hidden states"""
        c = self.components
        hidden_states = []
        attention_scores = []  # Placeholder for future implementation
        
        # 1. Embed inputs
        h = c['embeddings'](inputs)
        hidden_states.append(h)
        
        # 2. Create attention mask if needed
        if mask is None:
            mask = self.create_attention_mask(h, cache)
        
        # 3. Initialize cache if not provided
        if cache is None:
            cache = [None] * len(c['transformer_layers'])
        else:
            # Ensure cache matches layer count
            cache = cache + [None] * (len(c['transformer_layers']) - len(cache))
        
        # 4. Process through transformer layers
        for i, layer in enumerate(c['transformer_layers']):
            # Store input state
            hidden_states.append(h)
            
            # Call layer
            h = layer(h, mask, cache=cache[i])
            
            # Store output state
            hidden_states.append(h)
        
        # 5. Apply final normalization if exists
        if c['norm'] is not None:
            h = c['norm'](h)
            hidden_states.append(h)
        
        # 6. Generate logits (handling various head types)
        logits = self._compute_logits(h)
        
        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "attention_scores": attention_scores
        }
    
    def _compute_logits(self, hidden_states: mx.array) -> mx.array:
        """Compute logits handling different LM head configurations"""
        c = self.components
        
        if c['tie_embeddings'] and c['lm_head'] is c['embeddings']:
            # Case 1: Tied embeddings with as_linear method
            if hasattr(c['embeddings'], 'as_linear'):
                return c['embeddings'].as_linear(hidden_states)
            
            # Case 2: Tied embeddings with weight transpose
            elif hasattr(c['embeddings'], 'weight'):
                return mx.matmul(hidden_states, c['embeddings'].weight.T)
            
            # Case 3: Try to find weight in embedding layer
            else:
                # Look for weight-like attributes
                for attr_name in ['weight', 'embedding', 'embeddings']:
                    if hasattr(c['embeddings'], attr_name):
                        weight = getattr(c['embeddings'], attr_name)
                        if hasattr(weight, 'T'):  # Check if it's a matrix
                            return mx.matmul(hidden_states, weight.T)
                
                # If all else fails, try calling the embedding layer directly
                # (some implementations might handle this internally)
                try:
                    return c['lm_head'](hidden_states)
                except Exception as e:
                    raise RuntimeError(f"Could not compute logits with tied embeddings: {e}")
        
        else:
            # Case 4: Regular linear head
            return c['lm_head'](hidden_states)

    def create_attention_mask(
        self, 
        inputs: mx.array, 
        cache: Optional[List[mx.array]] = None
    ) -> mx.array:
        """Create attention mask compatible with various MLX-LM models"""
        try:
            from mlx_lm.models.base import create_attention_mask
            return create_attention_mask(inputs, cache)
        except ImportError:
            # Fallback implementation
            seq_len = inputs.shape[1]
            mask = mx.full((seq_len, seq_len), -mx.inf)
            mask = mx.triu(mask, k=1)
            if cache and cache[0] is not None:
                mask = mask[None, None, -1:, :]
            else:
                mask = mask[None, None, :, :]
            return mask

    def get_embeds(self, inputs: mx.array) -> mx.array:
        """Get input embeddings directly"""
        return self.components['embeddings'](inputs)
    
    @property
    def layers(self) -> List[nn.Module]:
        """Access to transformer layers for direct manipulation"""
        return self.components['transformer_layers']