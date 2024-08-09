from typing import Optional, Union, Dict, Tuple, List
from pathlib import Path
import json

import torch
from torch import nn
import safetensors

from huggingface_hub import PyTorchModelHubMixin, snapshot_download
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers import PretrainedConfig, PreTrainedModel, LlamaConfig, Qwen2Config, GemmaConfig, MistralConfig
from peft import LoraConfig, get_peft_model

from .models import MistralBiModel, LlamaBiModel, GemmaBiModel, Qwen2BiModel

DECODER_MODEL_MAPPING = {
    "MistralConfig": MistralBiModel,
    "LlamaConfig": LlamaBiModel,
    "GemmaConfig": GemmaBiModel,
    "Qwen2Config": Qwen2BiModel
}

MODEL_CONFIG_MAPPING = {
    'llama': LlamaConfig,
    'qwen2': Qwen2Config,
    'gemma': GemmaConfig,
    'mistral': MistralConfig
}

class AutoLLMEncoder(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config, adapter_config: Optional[LoraConfig] = None):
        super().__init__()
        self.config = config
        self.adapter_config = adapter_config

        config_class_name = config.__class__.__name__
        model_cls = self._get_model_class(config_class_name)

        self.model = model_cls(config)

        if adapter_config:
            self.model = get_peft_model(self.model, adapter_config)

    def _get_model_class(self, config_class_name):
        if config_class_name in DECODER_MODEL_MAPPING:
            return DECODER_MODEL_MAPPING[config_class_name]
        else:
            raise ValueError(
                f"{config_class_name} is not supported yet with bidirectional models."
            )

    @classmethod
    def _get_config_class(cls, model_type):
        if model_type in MODEL_CONFIG_MAPPING:
            return MODEL_CONFIG_MAPPING[model_type]
        else:
            raise ValueError(
                f"{model_type} is not supported yet with bidirectional models."
            )
    

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    @classmethod
    def load_model(cls, model, model_weights_file, map_location='cpu', strict=False):
        state_dict = safetensors.torch.load_file(model_weights_file, device=map_location)
        model_state_dict = model.state_dict()
        
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k in model_state_dict:
                filtered_state_dict[k] = v
            elif f'model.{k}' in model_state_dict:
                filtered_state_dict[f'model.{k}'] = v
        
        model.load_state_dict(filtered_state_dict, strict=strict)

    @classmethod
    def _from_pretrained(cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        adapter_model: Optional[str] = None,
        strict: bool = False,
        **model_kwargs,
    ):
        model_dir = Path(model_id) 
        if not model_dir.exists():
            model_dir = snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )

        config_file = Path(model_dir) / "config.json"
        model_weights_file = Path(model_dir) / "model.safetensors"

        if adapter_model is not None:
            adapter_model_dir = Path(adapter_model)
            if not adapter_model_dir.exists():
                adapter_model_dir = snapshot_download(
                    repo_id=adapter_model,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            adapter_config_file = Path(adapter_model_dir) / "adapter_config.json"
        else:
            adapter_config_file = Path(model_dir) / "adapter_config.json"

        config_ = json.load(open(config_file))
        model_type = config_.pop("model_type")
        config_class = cls._get_config_class(model_type)
        config = config_class(**config_, **model_kwargs)

        if adapter_config_file.exists():
            adapter_config_ = json.load(open(adapter_config_file))
            adapter_config = LoraConfig(**adapter_config_)
        else:
            adapter_config = None
        
        model = cls(config, adapter_config)

        cls.load_model(model, model_weights_file, map_location, strict)

        return model


    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        *,
        config: Optional[PretrainedConfig] = None,
        repo_id: Optional[str] = None,
        push_to_hub: bool = False,
        **push_to_hub_kwargs,
    ) -> Optional[str]:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # save model weights/files
        safetensors.torch.save_model(
            self,
            save_directory / "model.safetensors",
        )

        # save config (if provided)
        if config is None:
            config = self.config
        if config is not None:
            config.to_json_file(save_directory / "config.json")

        if self.adapter_config is None:
            adapter_config = self.adapter_config
        if adapter_config is not None:
            adapter_config.to_json_file(save_directory / "adapter_config.json")

        # push to the Hub if required
        if push_to_hub:
            kwargs = push_to_hub_kwargs.copy()  # soft-copy to avoid mutating input
            if config is not None:  # kwarg for `push_to_hub`
                kwargs["config"] = config
            if adapter_config is not None:
                kwargs["adapter_config"] = config
            if repo_id is None:
                repo_id = save_directory.name  # Defaults to `save_directory` name
            return self.push_to_hub(repo_id=repo_id, **kwargs)
        return None

class AutoLLMEncoderForSequenceClassification(AutoLLMEncoder):
    def __init__(self, config, adapter_config):
        super().__init__(config, adapter_config)
        self.num_labels = config.num_labels
        self.head = nn.Linear(config.hidden_size, self.num_labels, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        hidden_states = transformer_outputs[0]
        logits = self.head(hidden_states)

        pooled_logits = logits[:, 0, :]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

class AutoLLMEncoderForQuestionAnswering(AutoLLMEncoder):
    def __init__(self, config, adapter_config):
        super().__init__(config, adapter_config)
        self.qa_outputs = nn.Linear(config.hidden_size, self.num_labels, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1).to(start_logits.device)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1).to(end_logits.device)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class AutoLLMEncoderForTokenClassification(AutoLLMEncoder):
    def __init__(self, config, adapter_config):
        super().__init__(config, adapter_config)
        self.num_labels = config.num_labels
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.head = nn.Linear(config.hidden_size, self.num_labels, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )