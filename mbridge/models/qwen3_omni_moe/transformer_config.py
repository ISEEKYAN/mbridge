from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig


def get_audio_model_config(hf_config):
    audio_config = Qwen3OmniMoeAudioEncoderConfig()
    audio_config.num_mel_bins = hf_config.num_mel_bins
    audio_config.d_model = hf_config.d_model
    audio_config.encoder_layers = hf_config.encoder_layers
    audio_config.encoder_attention_heads = hf_config.encoder_attention_heads
    audio_config.encoder_ffn_dim = hf_config.encoder_ffn_dim
    audio_config.dropout = hf_config.dropout
    audio_config.attention_dropout = hf_config.attention_dropout
    audio_config.activation_function = hf_config.activation_function
    audio_config.activation_dropout = hf_config.activation_dropout
    audio_config.num_hidden_layers = hf_config.num_hidden_layers
    audio_config.initializer_range = hf_config.initializer_range
    audio_config.scale_embedding = hf_config.scale_embedding
    audio_config.max_source_positions = hf_config.max_source_positions
    audio_config.n_window = hf_config.n_window
    audio_config.output_dim = hf_config.output_dim
    audio_config.n_window_infer = hf_config.n_window_infer
    audio_config.conv_chunksize = hf_config.conv_chunksize
    audio_config.downsample_hidden_size = hf_config.downsample_hidden_size

    return audio_config
