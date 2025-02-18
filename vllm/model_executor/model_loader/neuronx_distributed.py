import copy
import importlib
import hashlib
import os
import shutil
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import multiprocessing
from transformers import AutoModelForCausalLM, PretrainedConfig, AutoTokenizer

from vllm.config import ModelConfig, ParallelConfig, SchedulerConfig, SpeculativeConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import (CompletionSequenceGroupOutput, Logprob,
                           SequenceOutput)

from neuronx_distributed_inference.models.mllama.utils import create_vision_mask
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.models.config import FusedSpecNeuronConfig, OnDeviceSamplingConfig

logger = init_logger(__name__)

TORCH_DTYPE_TO_NEURON_AMP = {
    "auto": "float32",
    "half": "float16",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float": "float32",
    "float32": "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}

# Models supported by Neuronx distributed for inference.
_NEURON_SUPPORTED_MODELS: Dict[str, Tuple[str, str]] = {
    "LlamaForCausalLM": ("neuronx_distributed_inference.models.llama.modeling_llama",
                         "NeuronLlamaForCausalLM"),
    "DbrxForCausalLM": ("neuronx_distributed_inference.models.dbrx.modeling_dbrx",
                         "NeuronDbrxForCausalLM"),
    "MixtralForCausalLM": ("neuronx_distributed_inference.models.mixtral.modeling_mixtral",
                         "NeuronMixtralForCausalLM"),
    "MllamaForConditionalGeneration": ("neuronx_distributed_inference.models.mllama.modeling_mllama",
                         "NeuronMllamaForCausalLM"),
    "MistralForCausalLM": ("neuronx_distributed_inference.models.llama.modeling_llama",
                         "NeuronLlamaForCausalLM"),
    "PixtralForConditionalGeneration": ("neuronx_distributed_inference.models.pixtral.modeling_pixtral",
                         "NeuronPixtralForConditionalGeneration"),
}

class NeuronPixtralConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        on_device_sampling_disabled: bool = False) -> None:
        super().__init__()
        self.config = config
        self.logits_processor = LogitsProcessor(config.get_text_config().vocab_size,
                                                logits_as_input=True)

        self.on_device_sampling_disabled = on_device_sampling_disabled
        if self.on_device_sampling_disabled:
            # Use default sampler
            self.sampler = Sampler()

        # Lazy initialized
        self.model: nn.Module
        
        @cached_property
        def sampler(self):
            if hasattr(self.text_model, "sampler"):
                return self.language_model.sampler
    
            return get_sampler()
    
        def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
            image_input, image_tokens = self._parse_and_validate_image_input(
                **kwargs)
            if image_input is None:
                return None
    
            vision_embeddings = self._process_image_input(image_input)
    
            # NOTE: We patch the outputs of the vision encoder with embeddings
            # from `[IMG_BREAK]` and `[IMG_END]` tokens.
            image_embeds = self.language_model.get_input_embeddings(image_tokens)
            image_token_mask = image_tokens == self.vision_args.image_token_id
            image_embeds[image_token_mask] = vision_embeddings
    
            # NOTE: Image embeddings are split into separate tensors for each image
            # by the indices of `[IMG_END]` token.
            image_end_mask = image_tokens == self.vision_args.image_end_token_id
            split_indices = torch.where(image_end_mask)[0] + 1
            if len(split_indices) <= 1:
                # Do not split, return as tensor of shape [1, fs, hs]
                return image_embeds.unsqueeze(0)
    
            # If the last split index is the last index in image_tokens, we
            # ignore it to avoid empty split tensor
            if split_indices[-1] == len(image_tokens):
                split_indices = split_indices[:-1]
    
            image_embeds = image_embeds.tensor_split(split_indices.cpu())
            return image_embeds
    
        def get_input_embeddings(
            self,
            input_ids: torch.Tensor,
            multimodal_embeddings: Optional[NestedTensors] = None,
        ) -> torch.Tensor:
            inputs_embeds = self.language_model.get_input_embeddings(input_ids)
            if multimodal_embeddings is not None:
                inputs_embeds = merge_multimodal_embeddings(
                    input_ids, inputs_embeds, multimodal_embeddings, [
                        self.vision_args.image_token_id,
                        self.vision_args.image_break_token_id,
                        self.vision_args.image_end_token_id,
                    ])
            return inputs_embeds

        # attn_metadata: AttentionMetadata, is a vllm parameter, maybe just drop
        def forward(self,
                    input_ids: torch.Tensor,
                    attention_mask = None, 
                    positions: torch.Tensor = None,
                    seq_ids = None, 
                    sampling_params = None, # maybe send from Neuron config
                    intermediate_tensors: Optional[IntermediateTensors] = None, # maybe maps to prev_hidden
                    adapter_ids = None,
                    accepted_indices = None,
                    current_length = None,
                    inputs_embeds: Optional[torch.Tensor] = None,
                    kv_caches: List[torch.Tensor] = None,
                    **kwargs: object,
        ) -> Union[torch.Tensor, IntermediateTensors]:
            """Run forward pass for pixtral.
            """
            if intermediate_tensors is not None:
                inputs_embeds = None
    
            # NOTE: In v1, inputs_embeds is always generated at model runner, this
            # condition is for v0 compatibility.
            elif inputs_embeds is None:
                vision_embeddings = self.get_multimodal_embeddings(**kwargs)
                inputs_embeds = self.get_input_embeddings(input_ids,
                                                          vision_embeddings)
                input_ids = None

            hidden_states = self.text_model.forward(input_ids,
                                                    None, 
                                                    positions,
                                                    None, 
                                                    None, # maybe send sampling params from Neuron config
                                                    intermediate_tensors,
                                                    None,
                                                    None,
                                                    None,
                                                    inputs_embeds,
                                                    kv_caches)

        # from the original
        # hidden_states = self.language_model.model(input_ids,
        #                                           positions,
        #                                           kv_caches,
        #                                           attn_metadata,
        #                                           intermediate_tensors,
        #                                           inputs_embeds=inputs_embeds)
    
            return hidden_states
    
        def _parse_and_validate_image_input(
            self,
            images: Optional[Union[List[List[torch.Tensor]], List[torch.Tensor],
                                   torch.Tensor]] = None,
            image_tokens: Optional[torch.Tensor] = None,
        ) -> Tuple[Optional[List[torch.Tensor]], Optional[torch.Tensor]]:
            if images is None:
                return None, None
    
            if isinstance(images, torch.Tensor):
                # if passed as batch take all images
                N, B, C, W, H = images.shape
                images = images.reshape(N * B, C, W, H)
                images = [images[i] for i in range(images.size(0))]
            elif isinstance(images, list):
                # if passed as list flatten lists of tensors
                flatten_images = []
                for imgs_per_req in images:
                    imgs_per_req = [
                        imgs_per_req[i] for i in range(imgs_per_req.size(0))
                    ] if isinstance(imgs_per_req, torch.Tensor) else imgs_per_req
    
                    flatten_images.extend(imgs_per_req)
    
                images = flatten_images
    
            if isinstance(image_tokens, torch.Tensor):
                # image_tokens are batched
                image_tokens = image_tokens.flatten()
            elif isinstance(image_tokens, list):
                # image_tokens are of different lengths thus passed as a list
                image_tokens = torch.cat(image_tokens)
    
            assert image_tokens.dim() == 1
    
            return images, image_tokens
    
        def _process_image_input(self,
                                 image_input: List[torch.Tensor]) -> torch.Tensor:
            return self.vision_language_adapter(self.vision_encoder(image_input))
    
        def compute_logits(
            self,
            hidden_states: torch.Tensor,
            sampling_metadata: SamplingMetadata,
        ) -> Optional[torch.Tensor]:
            return self.language_model.compute_logits(hidden_states,
                                                      sampling_metadata)
    
        def sample(
            self,
            logits: torch.Tensor,
            sampling_metadata: SamplingMetadata,
        ) -> Optional[SamplerOutput]:
            return self.language_model.sample(logits, sampling_metadata)
    
        def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
    
            def is_vision_encoder_weights(weight: Tuple[str, torch.Tensor]):
                return weight[0].startswith("vision_encoder")
    
            def is_vision_lang_adapter_weights(weight: Tuple[str, torch.Tensor]):
                return weight[0].startswith("vision_language_adapter")
    
            # Get references to parameters for direct loading
            vision_encoder_dict = dict(self.vision_encoder.named_parameters())
            vision_lang_adapter_dict = dict(
                self.vision_language_adapter.named_parameters())
    
            def llm_weights_generator():
                # Single pass over weights
                for name, w in weights:
                    if is_vision_encoder_weights((name, w)):
                        # Load vision encoder weights directly
                        trimmed_name = '.'.join(name.split(".")[1:])
                        param = vision_encoder_dict[trimmed_name]
                        with torch.no_grad():
                            default_weight_loader(param, w)
                    elif is_vision_lang_adapter_weights((name, w)):
                        # Load vision-language adapter weights directly
                        trimmed_name = '.'.join(name.split(".")[1:])
                        param = vision_lang_adapter_dict[trimmed_name]
                        with torch.no_grad():
                            default_weight_loader(param, w)
                    else:
                        # LLM weights: yield them to be loaded
                        # by language_model.load_weights
                        yield (name, w)
    
            # Now we call the language model load with the generator
            self.language_model.load_weights(llm_weights_generator())

class NeuronCasualLM(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.logits_processor = LogitsProcessor(config.vocab_size,
                                                logits_as_input=True)
        self.sampler = Sampler()

        # Lazy initialized
        self.model: nn.Module

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_block_ids: torch.Tensor,
        sampling_params: torch.Tensor,
    ) -> torch.Tensor:
        output = self.model(input_ids,
                            attention_mask=None,
                            position_ids=positions,
                            seq_ids=input_block_ids,
                            sampling_params=sampling_params)
        # on-device sampling
        if self.config.neuron_config.on_device_sampling_config:
            return output.hidden_states
        else:
            return output.logits[:, -1, :]

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(None, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        # on-device sampling
        if self.config.neuron_config.on_device_sampling_config:
            batch_size = logits.shape
            seq_ids = [seq_id for sg in sampling_metadata.seq_groups for seq_id in sg.seq_ids]
            assert len(seq_ids) == list(batch_size)[0], "batch size mismatch"
            # Organize input tensors by step instead of by sequence.
            accepted_token_ids_by_step = logits.flatten()
            accepted_token_ids_by_step = accepted_token_ids_by_step.tolist()

            step_output_token_ids = []
            for i, seq_id in enumerate(seq_ids):
                token_id = accepted_token_ids_by_step[i]
                step_output_token_ids.append(CompletionSequenceGroupOutput(samples=[SequenceOutput(parent_seq_id=seq_id, output_token=token_id, logprobs={token_id: Logprob(token_id)})], prompt_logprobs=None))
            return SamplerOutput(outputs=step_output_token_ids)
        else:
            return self.sampler(logits, sampling_metadata)

    def load_weights(self, model_name_or_path: str, **kwargs):
        arch = _get_model_architecture(self.config)
        neuronx_module_path, neuronx_model_cls_name = (
            _NEURON_SUPPORTED_MODELS[arch])
        neuronx_module = importlib.import_module(neuronx_module_path)
        neuronx_model_cls = getattr(neuronx_module, neuronx_model_cls_name)
        neuron_config = neuronx_model_cls.get_neuron_config_cls()(**kwargs['neuron_config'])
        self.config.neuron_config = neuron_config
        config = neuronx_model_cls.get_config_cls()(
            neuron_config, load_config=load_pretrained_config(model_name_or_path)
        )
        if os.getenv("NEURON_COMPILED_ARTIFACTS") is not None:
            compiled_model_path = os.getenv("NEURON_COMPILED_ARTIFACTS")
        elif os.path.exists(model_name_or_path):
            compiled_model_path = os.path.join(model_name_or_path,
                f"neuron-compiled-artifacts/{hashlib.md5(config.to_json_string().encode('utf-8')).hexdigest()}/")
            shutil.rmtree(compiled_model_path, ignore_errors=True)
        else:
            compiled_model_path = os.path.join("local-models", model_name_or_path,
                f"neuron-compiled-artifacts/{hashlib.md5(config.to_json_string().encode('utf-8')).hexdigest()}/")
            shutil.rmtree(compiled_model_path, ignore_errors=True)
        try:
            self.model = neuronx_model_cls(compiled_model_path)
            override_neuron_config = kwargs["override_neuron_config"]
            for k, v in override_neuron_config.items():
                setattr(self.model.config.neuron_config, k, v)
            self.model.load(compiled_model_path)
            return
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Exception: {e}")
            logger.warning(f"Failed to load the model from {compiled_model_path}, Recompiling...")
        if not os.path.exists(model_name_or_path):
            hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
            saved_path = os.path.join("local-models", model_name_or_path)
            hf_model.save_pretrained(saved_path)
            model_name_or_path = saved_path
        self.model = neuronx_model_cls(model_name_or_path, config)
        self.model.compile(compiled_model_path)
        self.model.load(compiled_model_path)

class NeuronMllamaForCausalLM(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        on_device_sampling_disabled: bool = False) -> None:
        super().__init__()
        self.config = config
        self.logits_processor = LogitsProcessor(config.get_text_config().vocab_size,
                                                logits_as_input=True)

        self.on_device_sampling_disabled = on_device_sampling_disabled
        if self.on_device_sampling_disabled:
            # Use default sampler
            self.sampler = Sampler()

        # Lazy initialized
        self.model: nn.Module

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            seq_ids: torch.Tensor,
            pixel_values: torch.Tensor,
            aspect_ratios: torch.Tensor,
            num_chunks: torch.Tensor,
            has_image: torch.Tensor,
            sampling_params
    ) -> torch.Tensor:
        self.vision_mask = create_vision_mask(input_ids, self.vision_token_id)
        output = self.model(input_ids.to(torch.int32),
                            attention_mask=None,
                            position_ids=positions.to(torch.int32),
                            seq_ids= seq_ids.flatten().to(torch.int32),
                            pixel_values=pixel_values.to(self.config.vision_config.torch_dtype),
                            aspect_ratios=aspect_ratios.to(torch.int32),
                            vision_mask=self.vision_mask.to(torch.int32),
                            sampling_params=sampling_params,
                            num_chunks=num_chunks.to(torch.int32),
                            has_image=has_image.to(torch.int32),
                            )
        if self.config.neuron_config.on_device_sampling_config:
            return output.hidden_states
        return output.logits[:, -1, :]

    def compute_logits(self, hidden_states: torch.Tensor,
                    sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(None, hidden_states, sampling_metadata)
        return logits

    def sample(self, hidden_states, sampling_metadata):
        if not self.on_device_sampling_disabled:
            with torch.profiler.record_function("sample"):
                hidden_states = hidden_states.flatten()
                res = []
                sample_idx = 0
                for seq_group in sampling_metadata.seq_groups:
                    seq_ids = seq_group.seq_ids
                    samples = []
                    for seq_id in seq_ids:
                        token_id = hidden_states[sample_idx].item()
                        samples.append(SequenceOutput(parent_seq_id=seq_id, output_token=token_id,
                                                    logprobs={token_id: Logprob(token_id)}))
                        sample_idx += 1
                    res.append(CompletionSequenceGroupOutput(samples=samples, prompt_logprobs=None))
                next_tokens = SamplerOutput(outputs=res)
        else:
            next_tokens = self.sampler(None, hidden_states, sampling_metadata)
        return next_tokens

    def load_weights(self, model_name_or_path: str, **kwargs):
        arch = _get_model_architecture(self.config)
        neuronx_module_path, neuronx_model_cls_name = (
            _NEURON_SUPPORTED_MODELS[arch])
        neuronx_module = importlib.import_module(neuronx_module_path)
        neuronx_model_cls = getattr(neuronx_module, neuronx_model_cls_name)
        neuron_config = neuronx_model_cls.get_neuron_config_cls()(**kwargs['neuron_config'])
        self.config.neuron_config = neuron_config
        print(f"neuron_config buckets: {self.config.neuron_config.buckets}")
        config = neuronx_model_cls.get_config_cls()(
            neuron_config, load_config=load_pretrained_config(model_name_or_path)
        )
        if os.getenv("NEURON_COMPILED_ARTIFACTS") is not None:
            compiled_model_path = os.getenv("NEURON_COMPILED_ARTIFACTS")
        elif os.path.exists(model_name_or_path):
            compiled_model_path = os.path.join(model_name_or_path,
                f"neuron-compiled-artifacts/{hashlib.md5(config.to_json_string().encode('utf-8')).hexdigest()}/")
        else:
            compiled_model_path = os.path.join("local-models", model_name_or_path,
                f"neuron-compiled-artifacts/{hashlib.md5(config.to_json_string().encode('utf-8')).hexdigest()}/")
        try:
            self.model = neuronx_model_cls(compiled_model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.vision_token_id = tokenizer("<|image|>", add_special_tokens=False).input_ids
            self.model.load(compiled_model_path)
            return
        except (FileNotFoundError, ValueError):
            logger.warning(f"Failed to load the model from {compiled_model_path}, Recompiling...")
        if not os.path.exists(model_name_or_path):
            hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
            saved_path = os.path.join("local-models", model_name_or_path)
            hf_model.save_pretrained(saved_path)
            model_name_or_path = saved_path
        self.model = neuronx_model_cls(model_name_or_path, config)

        logger.info(f"\nCompiling and saving model to {model_name_or_path}...")
        p = multiprocessing.Process(target=compile_model, args=(self, compiled_model_path))
        p.start()
        p.join()

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        tokenizer.save_pretrained(compiled_model_path)
        logger.info(f"successfully compiled and saved the model in {compiled_model_path}")

        # Read "<|image|>" token_id from the tokenizer
        self.vision_token_id = tokenizer("<|image|>", add_special_tokens=False).input_ids
        logger.info("\nLoading model from compiled checkpoint...")
        self.model.load(compiled_model_path)


def compile_model(neuron_model, traced_model_path):
    neuron_model.model.compile(traced_model_path)


class NeuronSpeculationCasualLM(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.logits_processor = LogitsProcessor(config.vocab_size,
                                                logits_as_input=True)
        # Lazy initialized
        self.model: nn.Module

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_block_ids: torch.Tensor,
        sampling_params: torch.Tensor,
    ) -> torch.Tensor:
        output = self.model(input_ids,
                            attention_mask=None,
                            position_ids=positions,
                            seq_ids=input_block_ids,
                            sampling_params=sampling_params)
        if output.fused_outputs[1].shape[-1] == 1:
            # CTX encoding
            return output.fused_outputs[1].view(1, -1)
        draft_new_tokens = output.fused_outputs[0].view(1, -1)
        target_tokens = output.fused_outputs[1].view(1, -1)
        if self.config.neuron_config.enable_eagle_speculation:
            candidate_new_tokens = draft_new_tokens[:, 1:]
        else:
            candidate_new_tokens = draft_new_tokens[:,:-1]
        selected_tokens = target_tokens[:,:-1]
        n_matches = ((~(candidate_new_tokens == selected_tokens)).cumsum(dim=-1) < 1).sum()
        accepted_tokens = target_tokens[:,:n_matches+1]
        return accepted_tokens

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[List[SamplerOutput]]:
        batch_size, num_steps = logits.shape
        seq_ids = [seq_id for sg in sampling_metadata.seq_groups for seq_id in sg.seq_ids]
        # Organize input tensors by step instead of by sequence.
        accepted_token_ids_by_step = logits.transpose(0, 1)
        accepted_token_ids_by_step = accepted_token_ids_by_step.tolist()

        sampler_output_list = []
        for step_index in range(num_steps):
            if all(token_id == -1 for token_id in accepted_token_ids_by_step[step_index]):
                break
            step_output_token_ids = []
            for sequence_index in range(batch_size):
                token_id = accepted_token_ids_by_step[step_index][sequence_index]
                step_output_token_ids.append(CompletionSequenceGroupOutput(samples=[SequenceOutput(parent_seq_id=seq_ids[sequence_index], output_token=token_id, logprobs={token_id: Logprob(token_id)})], prompt_logprobs=None))
            sampler_output_list.append(
                SamplerOutput(outputs=step_output_token_ids))
        return sampler_output_list

    def load_weights(self, model_name_or_path: str, draft_model_name_or_path: str, **kwargs):
        arch = _get_model_architecture(self.config)
        neuronx_module_path, neuronx_model_cls_name = (
            _NEURON_SUPPORTED_MODELS[arch])
        neuronx_module = importlib.import_module(neuronx_module_path)
        neuronx_model_cls = getattr(neuronx_module, neuronx_model_cls_name)
        neuron_config = neuronx_model_cls.get_neuron_config_cls()(**kwargs['neuron_config'])
        config = neuronx_model_cls.get_config_cls()(
            neuron_config, load_config=load_pretrained_config(model_name_or_path)
        )

        draft_neuron_config = copy.deepcopy(config.neuron_config)
        if not config.neuron_config.enable_eagle_speculation:
            draft_neuron_config.speculation_length = 0
        draft_neuron_config.trace_tokengen_model = True
        draft_neuron_config.enable_fused_speculation = False
        if config.neuron_config.enable_eagle_speculation:
            draft_neuron_config.is_eagle_draft = True
            draft_neuron_config.sequence_parallel_enabled = False
        draft_config = neuronx_model_cls.get_config_cls()(
            draft_neuron_config, load_config=load_pretrained_config(draft_model_name_or_path)
        )
        fused_spec_config = FusedSpecNeuronConfig(neuronx_model_cls._model_cls, draft_config=draft_config, draft_model_path=draft_model_name_or_path)
        config.fused_spec_config = fused_spec_config
        self.config.neuron_config = neuron_config
        
        if os.getenv("NEURON_COMPILED_ARTIFACTS") is not None:
            compiled_model_path = os.getenv("NEURON_COMPILED_ARTIFACTS")
        elif os.path.exists(model_name_or_path):
            compiled_model_path = os.path.join(model_name_or_path,
                f"neuron-compiled-artifacts/{hashlib.md5(config.to_json_string().encode('utf-8')).hexdigest()}/")
            shutil.rmtree(compiled_model_path, ignore_errors=True)
        else:
            compiled_model_path = os.path.join("local-models", model_name_or_path,
                f"neuron-compiled-artifacts/{hashlib.md5(config.to_json_string().encode('utf-8')).hexdigest()}/")
            shutil.rmtree(compiled_model_path, ignore_errors=True)
        try:
            self.model = neuronx_model_cls(compiled_model_path)
            override_neuron_config = kwargs["override_neuron_config"]
            for k, v in override_neuron_config.items():
                setattr(self.model.config.neuron_config, k, v)
            self.model.load(compiled_model_path)
            return
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Exception: {e}")
            logger.warning(f"Failed to load the model from {compiled_model_path}, Recompiling...")
        if draft_model_name_or_path == model_name_or_path:
            draft_checkpoint_download = False
        if not os.path.exists(model_name_or_path):
            hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
            saved_path = os.path.join("local-models", model_name_or_path)
            hf_model.save_pretrained(saved_path)
            model_name_or_path = saved_path
        if not os.path.exists(draft_model_name_or_path):
            if draft_checkpoint_download:
                hf_model = AutoModelForCausalLM.from_pretrained(draft_model_name_or_path)
                saved_path = os.path.join("local-models", draft_model_name_or_path)
                hf_model.save_pretrained(saved_path)
                draft_model_name_or_path = saved_path
            else:
                draft_model_name_or_path = model_name_or_path
            config.fused_spec_config.draft_model_path = draft_model_name_or_path
        self.model = neuronx_model_cls(model_name_or_path, config)
        self.model.compile(compiled_model_path)
        self.model.load(compiled_model_path)

def _get_model_architecture(config: PretrainedConfig) -> str:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _NEURON_SUPPORTED_MODELS:
            return arch
    raise ValueError(
        f"Model architectures {architectures} are not supported on Neuron "
        f"for now. Supported architectures: "
        f"{list(_NEURON_SUPPORTED_MODELS.keys())}")

def _get_default_neuron_config(model_config: ModelConfig,
                               parallel_config: ParallelConfig,
                               scheduler_config: SchedulerConfig):

    logger.info(f"Initializing OnDeviceSampling config with global_topk=64")
    on_device_sampling_config = OnDeviceSamplingConfig(global_topk=64,
                                                    dynamic=True,
                                                    deterministic=False)
    batch_size = scheduler_config.max_num_seqs

    neuron_config = dict(
        tp_degree=parallel_config.tensor_parallel_size,
        ctx_batch_size=1,
        batch_size=batch_size,
        max_context_length=scheduler_config.max_model_len,
        seq_len=scheduler_config.max_model_len,
        enable_bucketing=True,
        is_continuous_batching=(batch_size>1),
        quantized=False,
        torch_dtype=TORCH_DTYPE_TO_NEURON_AMP[model_config.dtype],
        padding_side="right",
        on_device_sampling_config=on_device_sampling_config,
        sequence_parallel_enabled=True,
    )
    return neuron_config


def _get_default_neuron_speculation_config(model_config: ModelConfig,
                                           parallel_config: ParallelConfig,
                                           scheduler_config: SchedulerConfig,
                                           speculation_config: SpeculativeConfig):
    neuron_config = dict(
        tp_degree=parallel_config.tensor_parallel_size,
        batch_size=scheduler_config.max_num_seqs,
        max_context_length=scheduler_config.max_model_len,
        seq_len=scheduler_config.max_model_len,
        speculation_length=speculation_config.num_speculative_tokens,
        trace_tokengen_model=False,
        enable_fused_speculation=True,
        enable_bucketing=True,
        quantized=False,
        torch_dtype=TORCH_DTYPE_TO_NEURON_AMP[model_config.dtype],
        on_device_sampling_config= dict(top_k=1, do_sample=False,)
    )
    return neuron_config


def _get_neuron_config_after_override(default_neuron_config,
                                      overridden_neuron_config):
    overridden_neuron_config = overridden_neuron_config or {}
    default_neuron_config.update(overridden_neuron_config)
    return default_neuron_config

def get_neuron_model(model_config: ModelConfig,
                     parallel_config: ParallelConfig,
                     scheduler_config: SchedulerConfig) -> nn.Module:
    model_arch = _get_model_architecture(model_config.hf_config)
    if model_arch == "MllamaForConditionalGeneration":
        model = NeuronMllamaForCausalLM(model_config.hf_config)
    else:
        model = NeuronCasualLM(model_config.hf_config)
    default_neuron_config_args = _get_default_neuron_config(
        model_config, parallel_config, scheduler_config)
    neuron_config = _get_neuron_config_after_override(default_neuron_config_args,
        model_config.override_neuron_config)
    model.load_weights(model_config.model,
                       neuron_config=neuron_config,
                       override_neuron_config=model_config.override_neuron_config)
    return model.eval()

def get_neuron_speculation_model(model_config: ModelConfig,
                                 parallel_config: ParallelConfig,
                                 scheduler_config: SchedulerConfig,
                                 speculation_config: SpeculativeConfig):
    model = NeuronSpeculationCasualLM(model_config.hf_config)
    default_neuron_config_args = _get_default_neuron_speculation_config(
        model_config, parallel_config, scheduler_config, speculation_config)
    neuron_config = _get_neuron_config_after_override(default_neuron_config_args,
        model_config.override_neuron_config)
    model.load_weights(model_config.model,
                       speculation_config.draft_model_config.model,
                       neuron_config=neuron_config, 
                       override_neuron_config=model_config.override_neuron_config)
    return model.eval()
