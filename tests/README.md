# Tests

## Model Comparison Test

### Purpose
Compare the original model against the abliterated model to see how abliteration changes response behavior.

### Prerequisites
1. Run the abliteration script first:
   ```bash
   cd examples
   python abliterate.py
   ```
   This will create a model-specific folder (e.g., `abliteration_visuals_Qwen_Qwen3-0.6B/`) containing:
   - Abliterated model in `abliterated_model/`
   - Refusal direction vector: `refusal_dir.npy`
   - All visualization images

### Running the Test

```bash
cd tests
python compare_models.py
```

### What the Test Does

The test will:
1. Load both the original and abliterated models
2. Run a series of test prompts through both models
3. Display responses side-by-side
4. Analyze differences in refusal behavior

### Expected Results

- **Original model**: May refuse potentially harmful requests
- **Abliterated model**: May be more compliant to requests
- Both models should handle benign prompts similarly

### Safety Note

⚠️ **Important**: Abliteration removes safety measures from language models. The abliterated model may generate harmful content. Use responsibly and only in controlled research environments.

