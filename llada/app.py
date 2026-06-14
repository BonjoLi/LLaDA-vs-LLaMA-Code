import os
import gc
import argparse
import inspect
import torch
import numpy as np
import gradio as gr
import torch.nn.functional as F
import threading
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForCausalLM, 
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList
)
import time
import re
import json
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
_CUDA_DEVICE_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0
if _CUDA_DEVICE_COUNT > 0:
    # Prefer explicit device strings so we can reliably place different models on different GPUs.
    device = "cuda:0"
else:
    device = "cpu"
print(f"Using device: {device} (visible cuda devices: {_CUDA_DEVICE_COUNT})")

# Global event to signal generation should stop (set by clear button)
GENERATION_STOP_EVENT = threading.Event()

# Constants
MASK_TOKEN = "[MASK]"
MASK_ID = 126336  # The token ID of [MASK] in LLaDA

DEFAULT_GEN_LENGTH = 2 ** 7
DEFAULT_STEPS = DEFAULT_GEN_LENGTH
DEFAULT_TEMPERATURE = 0.2
DEFAULT_CFG_SCALE = DEFAULT_TEMPERATURE
DEFAULT_BLOCK_LENGTH = DEFAULT_GEN_LENGTH // 2**6
DEFAULT_REMASKING = "low_confidence" # "random"
DEFAULT_VISUALIZATION_DELAY = 0.05
MASK_PLACEHOLDER = ""
DEFAULT_AUTOREG_GEN_LENGTH = 2 ** 10

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
FILE_LOCK = threading.Lock()
ACCESS_LOG_PATH = os.path.abspath(os.path.join(REPO_ROOT, "..", "access_log.json"))
MERGED_DATASET_PATH = os.path.abspath(os.path.join(REPO_ROOT, "..", "dataset", "databricks-dolly-15k_merged.jsonl"))

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def read_access_log():
    with FILE_LOCK:
        if not os.path.exists(ACCESS_LOG_PATH):
            return []
        try:
            with open(ACCESS_LOG_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading access log: {e}")
            return []

def write_access_log(log_data):
    with FILE_LOCK:
        try:
            with open(ACCESS_LOG_PATH, 'w') as f:
                json.dump(log_data, f, indent=4)
        except Exception as e:
            print(f"Error writing access log: {e}")

def get_random_prompt_and_remove():
    with FILE_LOCK:
        if not os.path.exists(MERGED_DATASET_PATH):
            print(f"Dataset not found at {MERGED_DATASET_PATH}")
            return None
        
        try:
            with open(MERGED_DATASET_PATH, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                return None
            
            idx = random.randrange(len(lines))
            selected_line = lines.pop(idx)
            
            with open(MERGED_DATASET_PATH, 'w') as f:
                f.writelines(lines)
            
            return json.loads(selected_line)
        except Exception as e:
            print(f"Error handling dataset: {e}")
            return None

def read_dataset_all():
    """Read all entries from the dataset without removing anything."""
    with FILE_LOCK:
        if not os.path.exists(MERGED_DATASET_PATH):
            return []
        try:
            with open(MERGED_DATASET_PATH, 'r') as f:
                return [json.loads(line) for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading dataset: {e}")
            return []

def _parse_cli_args(argv=None):
    """
    Parse CLI args for app.py.

    --model:
      0 -> LLaDA only
      1 -> Llama-3.1 only
      2 -> dual-model (default when flag is omitted)
    """
    default_share_env = os.environ.get("GRADIO_SHARE", "true").strip().lower()
    default_share = default_share_env in ("1", "true", "yes", "y", "on")
    default_host = os.environ.get("GRADIO_HOST", "127.0.0.1").strip() or "127.0.0.1"
    # Port defaulting:
    # - If GRADIO_PORT is set, it wins.
    # - Otherwise, we'll pick a sensible default later based on --model to avoid collisions
    #   when running two instances (e.g., model 0 and model 1) on the same machine.
    default_port_env = os.environ.get("GRADIO_PORT", "").strip()
    if default_port_env:
        try:
            default_port = int(default_port_env)
        except ValueError:
            default_port = None
    else:
        default_port = None

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--model",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Model mode: 0=LLaDA only, 1=Llama-3.1 only, 2=both w/ toggle (default).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=default_host,
        help="Host/interface for Gradio to bind (default: GRADIO_HOST or 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help="Port for Gradio to listen on (default: GRADIO_PORT, else chosen by --model).",
    )
    parser.add_argument(
        "--share",
        dest="share",
        action="store_true",
        help="Enable Gradio public share link (default: GRADIO_SHARE or true).",
    )
    parser.add_argument(
        "--no-share",
        dest="share",
        action="store_false",
        help="Disable Gradio public share link (recommended when using cloudflared).",
    )
    parser.set_defaults(share=default_share)
    return parser.parse_args(argv)


def _resolve_llama31_model_path() -> str:
    """
    Resolve local path to the `Llama-3.1-8B-Instruct` directory.

    Priority:
    1) env var `LLAMA31_MODEL_PATH`
    2) `<repo_root>/../Llama-3.1-8B-Instruct` (project layout in this repo)
    """
    env_path = os.environ.get("LLAMA31_MODEL_PATH", "").strip()
    if env_path:
        return os.path.abspath(os.path.expanduser(env_path))

    candidate = os.path.abspath(os.path.join(REPO_ROOT, "..", "Llama-3.1-8B-Instruct"))
    return candidate

LLAMA31_MODEL_PATH = _resolve_llama31_model_path()
LLADA_MODEL_NAME = "LLaDA 8B"
LLAMA3_MODEL_NAME = "Llama-3.1-8B-Instruct"
MODEL_REGISTRY = {
    LLADA_MODEL_NAME: {
        "path": "GSAI-ML/LLaDA-8B-Instruct",
        "loader": "AutoModel",
        "trust_remote_code": True,
        "mode": "diffusion",
        "torch_dtype": torch.bfloat16,
        "device_index": 0,
    },
    LLAMA3_MODEL_NAME: {
        "path": LLAMA31_MODEL_PATH,
        "loader": "AutoModelForCausalLM",
        "trust_remote_code": False,
        "mode": "autoregressive",
        "torch_dtype": torch.bfloat16,
        # If you have 2 GPUs, load this on GPU1 so both models can coexist.
        # If you only have 1 visible GPU, this will fall back to GPU0.
        "device_index": 1,
        "tokenizer_kwargs": {
            "use_fast": False,
            "fix_mistral_regex": True,
        },
    },
}
DEFAULT_MODEL_NAME = LLADA_MODEL_NAME
model_cache = {}

def parse_constraints(constraints_text):
    """Parse constraints in format: 'position:word, position:word, ...'"""
    constraints = {}
    if not constraints_text:
        return constraints
        
    parts = constraints_text.split(',')
    for part in parts:
        if ':' not in part:
            continue
        pos_str, word = part.split(':', 1)
        try:
            pos = int(pos_str.strip())
            word = word.strip()
            if word and pos >= 0:
                constraints[pos] = word
        except ValueError:
            continue
    
    return constraints

def format_chat_history(history):
    """
    Format chat history for the LLaDA model
    
    Args:
        history: List of [user_message, assistant_message] pairs
        
    Returns:
        Formatted conversation for the model
    """
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:  # Skip if None (for the latest user message)
            messages.append({"role": "assistant", "content": assistant_msg})
    
    return messages


def format_chatbot_display(history):
    """
    Convert internal history pairs into Gradio Chatbot message dicts.
    """
    messages = []
    for user_msg, assistant_msg in history:
        if user_msg is not None:
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg is not None:
            messages.append({"role": "assistant", "content": assistant_msg})
    return messages

def state_to_text(state):
    """
    Convert a list of (token, color) tuples into a textual representation
    for streaming inside the Chatbot.
    """
    text = ''.join(
        token if token != MASK_TOKEN else MASK_PLACEHOLDER
        for token, _ in state
    )
    # Collapse spaces/tabs but keep newline boundaries intact for streaming
    text = re.sub(r'[^\S\n]+', ' ', text).strip(' ')
    return text


def load_model_components(model_name):
    """
    Load (or reuse) tokenizer/model pairs for diffusion or autoregressive modes.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'")
    
    if model_name not in model_cache:
        config = MODEL_REGISTRY[model_name]
        model_path = config["path"]
        # Decide which device to place this model on.
        if torch.cuda.is_available() and _CUDA_DEVICE_COUNT > 0:
            requested_index = int(config.get("device_index", 0))
            model_device_index = requested_index if requested_index < _CUDA_DEVICE_COUNT else 0
            model_device = f"cuda:{model_device_index}"
        else:
            model_device = "cpu"

        # Prefer bf16 when supported; otherwise fall back to fp16 on CUDA.
        requested_dtype = config.get("torch_dtype", torch.float32)
        if torch.cuda.is_available() and model_device.startswith("cuda"):
            if requested_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
                requested_dtype = torch.float16
        # Fail fast with a clear message for local-path models.
        if isinstance(model_path, str) and (model_path.startswith("/") or model_path.startswith(".")):
            if not os.path.isdir(model_path):
                raise FileNotFoundError(
                    f"Local model directory not found: {model_path}\n"
                    f"For {LLAMA3_MODEL_NAME}, either:\n"
                    f"- set env var LLAMA31_MODEL_PATH to the directory containing the model files, or\n"
                    f"- place the weights at '<repo>/Llama-3.1-8B-Instruct'."
                )

            # Extra validation: make "missing weights" crash early and clearly.
            required_any = [
                "model.safetensors",
                "pytorch_model.bin",
                "model.safetensors.index.json",
                "pytorch_model.bin.index.json",
            ]
            has_any_weights = any(os.path.exists(os.path.join(model_path, f)) for f in required_any)
            if not has_any_weights:
                raise FileNotFoundError(
                    f"Local model directory exists but weights are missing in: {model_path}\n"
                    f"Expected one of: {', '.join(required_any)}"
                )

            required_config = os.path.join(model_path, "config.json")
            if not os.path.exists(required_config):
                raise FileNotFoundError(
                    f"Local model directory exists but 'config.json' is missing: {required_config}"
                )
        tokenizer_kwargs = config.get("tokenizer_kwargs", {})
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=config.get("trust_remote_code", False),
            **tokenizer_kwargs,
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if tokenizer.padding_side != "left":
            tokenizer.padding_side = "left"
        
        loader_cls = AutoModel if config["loader"] == "AutoModel" else AutoModelForCausalLM
        model_instance = loader_cls.from_pretrained(
            model_path,
            trust_remote_code=config.get("trust_remote_code", False),
            torch_dtype=requested_dtype,
            low_cpu_mem_usage=True,
        ).to(model_device)
        model_instance.eval()
        model_cache[model_name] = {
            "tokenizer": tokenizer,
            "model": model_instance,
            "mode": config["mode"],
            "device": model_device,
        }
    
    return model_cache[model_name]

def autoregressive_response_stream(
    model_instance,
    tokenizer,
    messages,
    max_new_tokens=None,
    temperature=DEFAULT_TEMPERATURE,
    device_override=None,
):
    """
    Generate autoregressive responses with true token-by-token streaming.
    Uses TextIteratorStreamer to yield tokens as they're generated.
    """
    if max_new_tokens is None:
        max_new_tokens = DEFAULT_AUTOREG_GEN_LENGTH
    model_device = (
        device_override
        if device_override is not None
        else (next(model_instance.parameters()).device if any(True for _ in model_instance.parameters()) else torch.device(device))
    )
    chat_input = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors='pt'
    ).to(model_device)
    attention_mask = (chat_input != tokenizer.pad_token_id).long().to(model_device)
    
    # Create streamer that yields text as tokens are generated
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    # Stopping criteria to allow interruption via GENERATION_STOP_EVENT
    class EventStoppingCriteria(StoppingCriteria):
        def __init__(self, stop_event):
            self.stop_event = stop_event

        def __call__(self, input_ids, scores, **kwargs):
            return self.stop_event.is_set()
    
    generation_kwargs = {
        "input_ids": chat_input,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "temperature": max(temperature, 1e-5),
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "streamer": streamer,
        "stopping_criteria": StoppingCriteriaList([EventStoppingCriteria(GENERATION_STOP_EVENT)]),
    }
    if tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
    
    # Run generation in a background thread so we can stream tokens
    def generate_thread():
        with torch.inference_mode():
            model_instance.generate(**generation_kwargs)
    
    thread = threading.Thread(target=generate_thread)
    thread.start()
    
    # Yield tokens as they arrive from the streamer
    partial_text = ""
    for new_text in streamer:
        if GENERATION_STOP_EVENT.is_set():
            break
        partial_text += new_text
        yield partial_text.strip(), False
    
    thread.join()
    yield partial_text.strip(), True


def preload_all_models():
    """Load every model in the registry so the UI toggle is seamless."""
    for model_name in MODEL_REGISTRY:
        load_model_components(model_name)


# Allow scripts to opt out of eager preloading (e.g., single-model runs).
SKIP_PRELOAD = os.environ.get("LLADA_SKIP_PRELOAD", "").lower() in ("1", "true", "yes")

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature <= 0:
        return logits
        
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def generate_response_with_visualization(model, tokenizer, device, messages, gen_length=64, steps=32, 
                                         constraints=None, temperature=0.0, cfg_scale=0.0, block_length=32,
                                         remasking='low_confidence'):
    """
    Stream visualization states for the LLaDA denoising process while generating text.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        gen_length: Length of text to generate
        steps: Number of denoising steps
        constraints: Dictionary mapping positions to words
        temperature: Sampling temperature
        cfg_scale: Classifier-free guidance scale
        block_length: Block length for semi-autoregressive generation
        remasking: Remasking strategy ('low_confidence' or 'random')
        
    Yields:
        Tuple (current_state, response_text, is_final)
        current_state: visualization data for HighlightedText
        response_text: current decoded response (final text when is_final is True)
        is_final: indicates whether generation has completed
    """
    
    # Process constraints
    if constraints is None:
        constraints = {}
        
    # Convert any string constraints to token IDs
    processed_constraints = {}
    for pos, word in constraints.items():
        tokens = tokenizer.encode(" " + word, add_special_tokens=False)
        for i, token_id in enumerate(tokens):
            processed_constraints[pos + i] = token_id
    
    # Prepare the prompt using chat template
    chat_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(chat_input)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    
    # For generation
    prompt_length = input_ids.shape[1]
    
    # Initialize the sequence with masks for the response part
    x = torch.full((1, prompt_length + gen_length), MASK_ID, dtype=torch.long).to(device)
    x[:, :prompt_length] = input_ids.clone()
    
    # Helper to decode current response span
    def decode_response():
        response_tokens = x[0, prompt_length:]
        return tokenizer.decode(
            response_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
    
    # Add initial state (all masked)
    initial_state = [(MASK_TOKEN, "#444444") for _ in range(gen_length)]
    yield initial_state, "", False
    
    # Apply constraints to the initial state
    for pos, token_id in processed_constraints.items():
        absolute_pos = prompt_length + pos
        if absolute_pos < x.shape[1]:
            x[:, absolute_pos] = token_id
    
    # Mark prompt positions to exclude them from masking during classifier-free guidance
    prompt_index = (x != MASK_ID)
    
    # Ensure block_length is valid
    if block_length > gen_length:
        block_length = gen_length
    
    # Calculate number of blocks
    num_blocks = gen_length // block_length
    if gen_length % block_length != 0:
        num_blocks += 1
    
    # Adjust steps per block
    steps_per_block = steps // num_blocks
    if steps_per_block < 1:
        steps_per_block = 1
    
    # Process each block (wrap in inference_mode to avoid gradient accumulation)
    with torch.inference_mode():
        for num_block in range(num_blocks):
            # Check for interruption
            if GENERATION_STOP_EVENT.is_set():
                return
            
            # Calculate the start and end indices for the current block
            block_start = prompt_length + num_block * block_length
            block_end = min(prompt_length + (num_block + 1) * block_length, x.shape[1])
            
            # Get mask indices for the current block
            block_mask_index = (x[:, block_start:block_end] == MASK_ID)
            
            # Skip if no masks in this block
            if not block_mask_index.any():
                continue
            
            # Calculate number of tokens to unmask at each step
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
            
            # Process each step
            for step_idx in range(steps_per_block):
                # Check for interruption
                if GENERATION_STOP_EVENT.is_set():
                    return
                
                # Get all mask positions in the current sequence
                mask_index = (x == MASK_ID)
                
                # Skip if no masks
                if not mask_index.any():
                    break
                
                # Apply classifier-free guidance if enabled
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = MASK_ID
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                    del x_, un_x, un_logits  # Free intermediate tensors
                else:
                    logits = model(x).logits
                
                # Apply Gumbel noise for sampling
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                sampled_tokens = torch.argmax(logits_with_noise, dim=-1)
                del logits_with_noise  # Free intermediate tensor
                
                # Calculate confidence scores for remasking
                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(sampled_tokens, -1)), -1)  # b, l
                    del p  # Free softmax tensor
                elif remasking == 'random':
                    x0_p = torch.rand((sampled_tokens.shape[0], sampled_tokens.shape[1]), device=sampled_tokens.device)
                else:
                    raise NotImplementedError(f"Remasking strategy '{remasking}' not implemented")
                
                del logits  # Free logits after confidence computation
                
                # Don't consider positions beyond the current block
                x0_p[:, block_end:] = -float('inf')
                
                # Apply predictions where we have masks
                x0 = torch.where(mask_index, sampled_tokens, x)
                confidence = torch.where(mask_index, x0_p, -float('inf'))
                
                # Select tokens to unmask based on confidence
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    # Only consider positions within the current block for unmasking
                    block_confidence = confidence[j, block_start:block_end]
                    if step_idx < steps_per_block - 1:  # Not the last step
                        # Take top-k confidences
                        _, select_indices = torch.topk(block_confidence, 
                                                      k=min(num_transfer_tokens[j, step_idx].item(), 
                                                           block_confidence.numel()))
                        # Adjust indices to global positions
                        select_indices = select_indices + block_start
                        transfer_index[j, select_indices] = True
                    else:  # Last step - unmask everything remaining
                        transfer_index[j, block_start:block_end] = mask_index[j, block_start:block_end]
                
                # Apply the selected tokens
                x = torch.where(transfer_index, x0, x)
                
                # Ensure constraints are maintained
                for pos, token_id in processed_constraints.items():
                    absolute_pos = prompt_length + pos
                    if absolute_pos < x.shape[1]:
                        x[:, absolute_pos] = token_id
                
                # Create visualization state only for the response part
                current_state = []
                for vis_idx in range(gen_length):
                    pos = prompt_length + vis_idx  # Absolute position in the sequence
                    
                    # Display the latest prediction (even if still masked)
                    token_id = sampled_tokens[0, pos].item()
                    token = tokenizer.decode(
                        [token_id],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )
                    token = token if token else MASK_TOKEN
                    
                    if x[0, pos] == MASK_ID:
                        # Still masked – show tentative prediction in gray
                        current_state.append((token, "#444444"))
                    elif transfer_index[0, pos]:
                        # Newly confirmed in this step
                        conf = float(x0_p[0, pos].cpu())
                        if conf < 0.3:
                            color = "#FF6666"
                        elif conf < 0.7:
                            color = "#FFAA33"
                        else:
                            color = "#66CC66"
                        current_state.append((token, color))
                    else:
                        # Previously confirmed token
                        current_state.append((token, "#6699CC"))  # Light blue
                
                partial_text = decode_response().replace(MASK_TOKEN, "").strip()
                finished = not (x == MASK_ID).any()
                
                # Clean up step tensors
                del x0, confidence, sampled_tokens, x0_p, transfer_index, mask_index
                
                if finished:
                    yield current_state, partial_text, True
                    return
                
                yield current_state, partial_text, False
    
    # Fallback: ensure final text is emitted even if loop exits without finished flag
    final_state = current_state if 'current_state' in locals() else initial_state
    final_text = decode_response().replace(MASK_TOKEN, "").strip()
    yield final_state, final_text, True

css = '''
.category-legend{display:none}
button{height:52px}
html, body{height:100%; margin:0}
.gradio-container{min-height:100vh}
#app-container{height:100vh; display:flex; flex-direction:column}
#chatbot-ui{flex:1 1 auto!important; min-height:0!important; overflow:auto!important}
#chatbot-ui > .wrap{height:100%!important}
#chatbot-ui .wrap{height:100%!important}
#chatbot-ui .message,
#chatbot-ui .message *{
    height:auto!important;
    min-height:unset!important;
    max-height:none!important;
    overflow:visible!important;
    -webkit-line-clamp:unset!important;
    line-clamp:unset!important;
}
#chatbot-ui .message .message-text,
#chatbot-ui .message .prose,
#chatbot-ui .message p,
#chatbot-ui .message span{
    display:block!important;
    white-space:pre-wrap!important;
    word-break:break-word!important;
}
#chatbot-ui .message code,
#chatbot-ui .message pre{
    white-space:pre-wrap!important;
}
/* Hide Gradio branding + API footer controls (fallback for older Gradio versions) */
footer{display:none!important}
.gradio-footer{display:none!important}

/* Research Survey Overlays */
.overlay-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background-color: rgba(0, 0, 0, 0.85);
    z-index: 10000;
    padding: 20px;
}
.overlay-content {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background-color: #2b2b2b;
    padding: 40px;
    border-radius: 12px;
    max-width: 600px;
    width: calc(100% - 40px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    text-align: center;
    color: #ffffff !important;
}
.overlay-content * {
    color: #ffffff !important;
}
.close-btn {
    position: absolute;
    top: 10px;
    right: 10px;
    background: none !important;
    border: none !important;
    color: #888 !important;
    font-size: 24px !important;
    cursor: pointer;
    min-width: unset !important;
    height: unset !important;
    padding: 5px 10px !important;
    line-height: 1 !important;
}
.close-btn:hover {
    color: #fff !important;
}
.prompt-box {
    background-color: #1a1a1a;
    border: 1px solid #444;
    padding: 15px;
    margin: 20px 0;
    border-radius: 8px;
    text-align: left;
    font-family: monospace;
    white-space: pre-wrap;
    user-select: all;
}
'''

js_copy_logic = """
function() {
    const promptField = document.querySelector('#prompt-text-field textarea');
    if (promptField) {
        promptField.select();
        document.execCommand('copy');
        // Fallback for modern browsers
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(promptField.value);
        }
    }
}
"""

def create_chatbot_demo(model_mode: int = 2):
    """
    Create the Gradio chatbot demo.

    model_mode:
      0 -> LLaDA only (no toggle)
      1 -> Llama-3.1 only (no toggle)
      2 -> both models + toggle (current behavior)
    """
    if model_mode not in (0, 1, 2):
        raise ValueError(f"Invalid model_mode: {model_mode}. Expected 0, 1, or 2.")

    fixed_model_name = None
    if model_mode == 0:
        fixed_model_name = LLADA_MODEL_NAME
    elif model_mode == 1:
        fixed_model_name = LLAMA3_MODEL_NAME

    blocks_sig = inspect.signature(gr.Blocks)
    use_blocks_css = "css" in blocks_sig.parameters
    blocks_kwargs = {}
    if use_blocks_css:
        blocks_kwargs["css"] = css
    if "fill_height" in blocks_sig.parameters:
        blocks_kwargs["fill_height"] = True
    blocks_kwargs["theme"] = gr.themes.Soft(mode="light")
    with gr.Blocks(**blocks_kwargs) as demo:
        if not use_blocks_css:
            gr.HTML(f"<style>{css}</style>")

        # SURVEY STATE
        prolific_id_state = gr.State("")
        interaction_count_state = gr.State(0)
        session_prompts_state = gr.State([])
        current_prompt_entry_state = gr.State(None)
        next_prompt_entry_state = gr.State(None)
        start_time_state = gr.State(0.0)
        model_type_state = gr.State(0 if model_mode == 0 else (1 if model_mode == 1 else 0))

        # ID ENTRY OVERLAY
        with gr.Column(visible=True, elem_classes="overlay-container") as id_entry_col:
            with gr.Column(elem_classes="overlay-content"):
                close_id_btn = gr.Button("✕", elem_classes="close-btn")
                gr.Markdown("# Welcome to the chatbot interface!")
                gr.Markdown("Please enter your Prolific ID to begin:")
                gr.Markdown("(If you closed the Qualtrics survey, please click the back button in your browser, and open this link again in a separate tab.)")
                id_input = gr.Textbox(label="Prolific ID", placeholder="Enter ID here...")
                start_btn = gr.Button("Start Study", variant="primary")

        # PROMPT DISPLAY OVERLAY
        with gr.Column(visible=False, elem_classes="overlay-container") as prompt_display_col:
            with gr.Column(elem_classes="overlay-content"):
                close_prompt_btn = gr.Button("✕", elem_classes="close-btn")
                gr.Markdown("## Please copy the instruction below and paste it into the chat interface.")
                gr.Markdown("")
                prompt_text_box = gr.Textbox(
                    show_label=False,
                    interactive=False,
                    show_copy_button=True,
                    container=False,
                    elem_classes="prompt-box",
                    elem_id="prompt-text-field"
                )
                continue_btn = gr.Button("Copy Prompt", variant="primary", elem_id="continue-btn")

        # COMPLETION OVERLAY
        with gr.Column(visible=False, elem_classes="overlay-container") as completion_col:
            with gr.Column(elem_classes="overlay-content"):
                close_completion_btn = gr.Button("✕", elem_classes="close-btn")
                gr.Markdown("# Study Completed")
                gr.Markdown("Thank you for your participation! You have completed all 5 interactions for this model.")
                gr.Markdown("Please close this tab.")

        model_selector = None
        with gr.Column(elem_id="app-container", visible=False) as main_app_col:
            # STATE MANAGEMENT
            chat_history = gr.State([])

            # UI COMPONENTS
            chatbot_ui = gr.Chatbot(
                show_label=False,
                render_markdown=True,
                elem_id="chatbot-ui",
                type="messages",
            )

            # Message input
            with gr.Row():
                user_input = gr.Textbox(
                    label="Your Message",
                    placeholder="Type your message here...",
                    show_label=False,
                )
                send_btn = gr.Button("Send")
            
            next_task_btn = gr.Button("Move to Next Task", variant="primary", visible=False)
            show_prompt_again_btn = gr.Button("Show Prompt Again", variant="secondary")

            # Model toggle + clear
            with gr.Row():
                if fixed_model_name is None:
                    model_selector = gr.Radio(
                        choices=[LLADA_MODEL_NAME, LLAMA3_MODEL_NAME],
                        value=DEFAULT_MODEL_NAME,
                        label="",
                        show_label=False,
                    )
                clear_btn = gr.Button("Clear Conversation")
        
        # HELPER FUNCTIONS FOR SURVEY
        def start_study(pid, model_mode_val, model_selector_val=None):
            if not pid.strip():
                return {id_entry_col: gr.update(visible=True)}
            
            pid = pid.strip()
            # Determine current model type (0 for LLaDA, 1 for Llama)
            # If model_mode_val is 2, use model_selector_val
            current_model_name = model_selector_val if model_mode_val == 2 else (LLADA_MODEL_NAME if model_mode_val == 0 else LLAMA3_MODEL_NAME)
            current_model_type = 0 if "LLaDA" in current_model_name else 1
            
            log = read_access_log()
            
            # Check for existing entries for this ID
            existing_entries = [e for e in log if e.get("prolific_id") == pid]
            same_model_entry = next((e for e in existing_entries if e.get("model_type") == current_model_type), None)
            other_model_entry = next((e for e in existing_entries if e.get("model_type") != current_model_type), None)
            
            session_prompts = []
            is_second_session = False
            
            if other_model_entry:
                # Second session: use prompts from first session
                is_second_session = True
                first_session_interactions = other_model_entry.get("interactions", [])
                # Extract the prompt entries that were saved
                session_prompts = [i.get("prompt_entry") for i in first_session_interactions if i.get("prompt_entry")]
                random.shuffle(session_prompts)
            
            if not same_model_entry:
                # Create new log entry for this ID + model
                new_entry = {
                    "prolific_id": pid,
                    "model_type": current_model_type,
                    "timestamp": time.time(),
                    "interactions": []
                }
                log.append(new_entry)
                write_access_log(log)
                same_model_entry = new_entry
            
            interactions = same_model_entry.get("interactions", [])
            interaction_count = len(interactions)
            
            if interaction_count >= 5:
                return {
                    id_entry_col: gr.update(visible=False),
                    completion_col: gr.update(visible=True),
                }

            # Prepare first prompt for this session (might be interaction 0 or more if resumed)
            current_prompt_entry = None
            if is_second_session:
                if session_prompts and interaction_count < len(session_prompts):
                    current_prompt_entry = session_prompts[interaction_count]
            else:
                current_prompt_entry = get_random_prompt_and_remove()
            
            if not current_prompt_entry:
                return {
                    id_entry_col: gr.update(visible=True),
                    main_app_col: gr.update(visible=False),
                }

            return {
                id_entry_col: gr.update(visible=False),
                prompt_display_col: gr.update(visible=True),
                prompt_text_box: current_prompt_entry.get("instruction", "No instruction found."),
                prolific_id_state: pid,
                model_type_state: current_model_type,
                session_prompts_state: session_prompts,
                current_prompt_entry_state: current_prompt_entry,
                interaction_count_state: interaction_count,
                main_app_col: gr.update(visible=False),
                completion_col: gr.update(visible=False),
                # Disable model selector if it exists
                **({model_selector: gr.update(interactive=False)} if model_mode_val == 2 else {})
            }

        def continue_to_chat():
            return {
                prompt_display_col: gr.update(visible=False),
                main_app_col: gr.update(visible=True),
            }

        def hide_overlays():
            return {
                id_entry_col: gr.update(visible=False),
                prompt_display_col: gr.update(visible=False),
                completion_col: gr.update(visible=False),
                main_app_col: gr.update(visible=True),
            }

        def show_prompt_overlay():
            return {
                prompt_display_col: gr.update(visible=True),
                main_app_col: gr.update(visible=False),
            }

        # EVENT HANDLERS FOR SURVEY
        close_id_btn.click(fn=hide_overlays, outputs=[id_entry_col, prompt_display_col, completion_col, main_app_col])
        close_prompt_btn.click(fn=hide_overlays, outputs=[id_entry_col, prompt_display_col, completion_col, main_app_col])
        close_completion_btn.click(fn=hide_overlays, outputs=[id_entry_col, prompt_display_col, completion_col, main_app_col])
        
        show_prompt_again_btn.click(fn=show_prompt_overlay, outputs=[prompt_display_col, main_app_col])

        start_btn_outputs = [id_entry_col, prompt_display_col, prompt_text_box, prolific_id_state, model_type_state, session_prompts_state, current_prompt_entry_state, interaction_count_state, main_app_col, completion_col]
        if model_selector is not None:
            start_btn_outputs.append(model_selector)
            
        start_btn.click(
            fn=start_study,
            inputs=[id_input, gr.State(model_mode), (model_selector if model_selector is not None else id_input)],
            outputs=start_btn_outputs
        )
        
        continue_btn.click(
            fn=continue_to_chat,
            outputs=[prompt_display_col, main_app_col],
            js=js_copy_logic
        )

        # HELPER FUNCTIONS
        def add_message(history, message, response):
            """Add a message pair to the history and return the updated history"""
            history = history.copy()
            history.append([message, response])
            return history
            
        def user_message_submitted(message, history):
            """Process a submitted user message"""
            # Skip empty messages
            if not message.strip():
                # Return current state unchanged
                history_for_display = format_chatbot_display(history)
                return history, history_for_display, "", time.time()
                
            # Add user message to history
            history = add_message(history, message, None)
            
            # Format for display - temporarily show user message with empty response
            history_for_display = format_chatbot_display(history)
            
            # Clear the input
            message_out = ""
            
            # Return immediately to update UI with user message
            return history, history_for_display, message_out, time.time()
            
        def bot_response(history, model_choice=None):
            """Generate bot response for the latest message"""
            # Clear any previous stop signal before starting new generation
            GENERATION_STOP_EVENT.clear()
            
            if not history:
                return history, []
                
            # Get the last user message
            last_user_message = history[-1][0]
            
            try:
                chosen_model = fixed_model_name if fixed_model_name is not None else model_choice
                if chosen_model is None:
                    raise ValueError("Model not selected.")

                components = load_model_components(chosen_model)
                tokenizer_instance = components["tokenizer"]
                model_instance = components["model"]
                mode = components["mode"]
                model_device = components.get("device", device)
                
                # Format all messages except the last one (which has no response yet)
                messages = format_chat_history(history[:-1])
                
                # Add the last user message
                messages.append({"role": "user", "content": last_user_message})
                
                if mode == "diffusion":
                    vis_stream = generate_response_with_visualization(
                        model_instance, tokenizer_instance, model_device, 
                        messages, 
                        gen_length=DEFAULT_GEN_LENGTH, 
                        steps=DEFAULT_STEPS,
                        constraints=None,
                        temperature=DEFAULT_TEMPERATURE,
                        cfg_scale=DEFAULT_CFG_SCALE,
                        block_length=DEFAULT_BLOCK_LENGTH,
                        remasking=DEFAULT_REMASKING
                    )
                    
                    for state, partial_text, is_final in vis_stream:
                        if GENERATION_STOP_EVENT.is_set():
                            return
                            
                        display_text = partial_text if is_final else state_to_text(state)
                        history[-1][1] = display_text
                        
                        history_for_display = format_chatbot_display(history)
                        yield history, history_for_display
                        
                        if not is_final and DEFAULT_VISUALIZATION_DELAY > 0:
                            time.sleep(DEFAULT_VISUALIZATION_DELAY)
                else:
                    text_stream = autoregressive_response_stream(
                        model_instance,
                        tokenizer_instance,
                        messages,
                        temperature=DEFAULT_TEMPERATURE,
                        device_override=model_device,
                    )
                    
                    for partial_text, is_final in text_stream:
                        if GENERATION_STOP_EVENT.is_set():
                            return
                            
                        history[-1][1] = partial_text
                        history_for_display = format_chatbot_display(history)
                        yield history, history_for_display
                        
                        if not is_final and DEFAULT_VISUALIZATION_DELAY > 0:
                            time.sleep(DEFAULT_VISUALIZATION_DELAY)
                    
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                print(error_msg)
                
                history[-1][1] = error_msg
                history_for_display = format_chatbot_display(history)
                yield history, history_for_display
        
        def clear_conversation():
            """Clear the conversation history and signal any running generation to stop"""
            GENERATION_STOP_EVENT.set()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return [], [], ""

        def finalize_interaction(history, pid, mtype, count, session_prompts, current_prompt_entry, start_time):
            if not pid or current_prompt_entry is None:
                return count, session_prompts, None, gr.update(), gr.update(), gr.update(), gr.update()

            duration = time.time() - start_time
            last_user_msg, last_bot_msg = history[-1]
            
            orig_instruction = current_prompt_entry.get("instruction", "")
            lev_dist = levenshtein_distance(orig_instruction, last_user_msg)
            
            interaction = {
                "prompt_entry": current_prompt_entry,
                "user_input": last_user_msg if lev_dist != 0 else None,
                "lev_dist": lev_dist,
                "model_output": last_bot_msg,
                "duration_seconds": duration,
                "timestamp": time.time()
            }
            if lev_dist == 0:
                interaction.pop("user_input", None)

            # Update log
            log = read_access_log()
            for entry in log:
                if entry.get("prolific_id") == pid and entry.get("model_type") == mtype:
                    if "interactions" not in entry:
                        entry["interactions"] = []
                    entry["interactions"].append(interaction)
                    break
            write_access_log(log)
            
            new_count = count + 1
            
            if new_count >= 5:
                # Completion! Show completion overlay
                return new_count, session_prompts, None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
            
            # Prepare next prompt entry now, but don't show the overlay yet
            next_prompt_entry = None
            if session_prompts and len(session_prompts) > 0:
                if new_count < len(session_prompts):
                    next_prompt_entry = session_prompts[new_count]
            else:
                next_prompt_entry = get_random_prompt_and_remove()
            
            if not next_prompt_entry:
                return new_count, session_prompts, None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

            # Show "Next Task" button in the current chat view
            return new_count, session_prompts, next_prompt_entry, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

        def go_to_next_prompt(next_prompt_entry):
            """Triggered by the 'Next Task' button in the chat interface"""
            if next_prompt_entry is None:
                return gr.update(visible=False), gr.update(visible=False), "", None, gr.update(visible=False), gr.update(visible=True)
            
            return (
                gr.update(visible=False), 
                gr.update(visible=True), 
                next_prompt_entry.get("instruction", "No instruction found."),
                next_prompt_entry,
                gr.update(visible=False),
                gr.update(visible=False)
            )

        # EVENT HANDLERS
        
        # User message submission flow (3-step process for survey)
        # Step 1: Add user message to history and update UI
        msg_submit = user_input.submit(
            fn=user_message_submitted,
            inputs=[user_input, chat_history],
            outputs=[chat_history, chatbot_ui, user_input, start_time_state]
        )
        
        # Also connect the send button
        send_click = send_btn.click(
            fn=user_message_submitted,
            inputs=[user_input, chat_history],
            outputs=[chat_history, chatbot_ui, user_input, start_time_state]
        )
        
        # Step 2: Generate bot response
        if fixed_model_name is None:
            bot_res_submit = msg_submit.then(
                fn=bot_response,
                inputs=[chat_history, model_selector],
                outputs=[chat_history, chatbot_ui],
            )
        else:
            bot_res_submit = msg_submit.then(
                fn=bot_response,
                inputs=[chat_history],
                outputs=[chat_history, chatbot_ui],
            )
        
        if fixed_model_name is None:
            bot_res_click = send_click.then(
                fn=bot_response,
                inputs=[chat_history, model_selector],
                outputs=[chat_history, chatbot_ui],
            )
        else:
            bot_res_click = send_click.then(
                fn=bot_response,
                inputs=[chat_history],
                outputs=[chat_history, chatbot_ui],
            )

        # Step 3: Finalize interaction and show next task button
        bot_res_submit.then(
            fn=finalize_interaction,
            inputs=[chat_history, prolific_id_state, model_type_state, interaction_count_state, session_prompts_state, current_prompt_entry_state, start_time_state],
            outputs=[interaction_count_state, session_prompts_state, next_prompt_entry_state, main_app_col, prompt_display_col, completion_col, next_task_btn]
        )
        
        bot_res_click.then(
            fn=finalize_interaction,
            inputs=[chat_history, prolific_id_state, model_type_state, interaction_count_state, session_prompts_state, current_prompt_entry_state, start_time_state],
            outputs=[interaction_count_state, session_prompts_state, next_prompt_entry_state, main_app_col, prompt_display_col, completion_col, next_task_btn]
        )

        # Handler for Next Task button
        next_task_btn.click(
            fn=go_to_next_prompt,
            inputs=[next_prompt_entry_state],
            outputs=[main_app_col, prompt_display_col, prompt_text_box, current_prompt_entry_state, next_task_btn, completion_col]
        ).then(
            fn=clear_conversation,
            outputs=[chat_history, chatbot_ui, user_input]
        )

        # Clear button handler

        # Clear button handler - cancels any running generation by setting the stop event
        clear_btn.click(
            fn=clear_conversation,
            inputs=[],
            outputs=[chat_history, chatbot_ui, user_input],
            queue=False
        )
        
    return demo

# Launch the demo
if __name__ == "__main__":
    args = _parse_cli_args()
    # Choose a default port if none was provided via --port or GRADIO_PORT.
    # This makes it easy to run two instances on one machine:
    # - --model 0 defaults to 7860
    # - --model 1 defaults to 7861
    # - --model 2 defaults to 7860 (single instance)
    if args.port is None:
        args.port = 7861 if args.model == 1 else 7860

    if args.model == 2:
        # Dual-model behavior (toggle available). Preserve existing default of eager preload,
        # unless a caller opted out via LLADA_SKIP_PRELOAD.
        if not SKIP_PRELOAD:
            preload_all_models()
        demo = create_chatbot_demo(model_mode=2)
    else:
        # Single-model behavior (no toggle). Only load the selected model.
        selected_model = LLADA_MODEL_NAME if args.model == 0 else LLAMA3_MODEL_NAME
        load_model_components(selected_model)
        demo = create_chatbot_demo(model_mode=args.model)
    queued_demo = demo.queue()
    # Hide "Use via API" and Gradio footer branding when the installed Gradio version supports it.
    launch_kwargs = {"share": bool(args.share)}
    try:
        launch_sig = inspect.signature(queued_demo.launch)
        if "server_name" in launch_sig.parameters:
            launch_kwargs["server_name"] = args.host
        if "server_port" in launch_sig.parameters:
            launch_kwargs["server_port"] = int(args.port)
        if "show_api" in launch_sig.parameters:
            launch_kwargs["show_api"] = False
        if "show_footer" in launch_sig.parameters:
            launch_kwargs["show_footer"] = False
    except Exception:
        # If signature inspection fails, CSS above still hides the footer in most versions.
        pass
    queued_demo.launch(**launch_kwargs)
