from ...configuration_utils import PretrainedConfig

class DeepSeekV2Config(PretrainedConfig):
    """
    Configuration class for DeepSeekV2 model.

    This class extends the PretrainedConfig class and provides configuration options
    for the DeepSeekV2 model. It includes parameters for the model architecture,
    such as the vocabulary size, hidden size, number of layers, and attention heads.
    It also includes options for the routing mechanism, such as the number of shared
    and routed experts, the scaling factor for routed experts, and the dimensions
    of the key, query, and value heads.

    Args:
        vocab_size (`int`, *optional*, defaults to 102400):
            Size of the vocabulary.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the intermediate representations.
        moe_intermediate_size (`int`, *optional*, defaults to 1407):
            Dimension of the MoE intermediate representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads.
        n_shared_experts (`int`, *optional*, defaults to None):
            Number of shared experts.
        n_routed_experts (`int`, *optional*, defaults to None):
            Number of routed experts.
        ep_size (`int`, *optional*, defaults to 1)::
            Expert parallelism size.
        routed_scaling_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for routed experts.
        kv_lora_rank (`int`, *optional*, defaults to 512):
            Rank of the key-value LoRA.
        q_lora_rank (`int`, *optional*, defaults to 1536):
            Rank of the query LoRA.
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            Dimension of the query-key RoPE head.
        v_head_dim (`int`, *optional*, defaults to 128):
            Dimension of the value head.
        qk_nope_head_dim (`int`, *optional*, defaults to 128):
            Dimension of the query-key NoPE head.
        topk_method (`str`, *optional*, defaults to `gready`):
            Method for top-k selection.
        n_group (`int`, *optional*, defaults to None):
            Number of groups for routing.
        topk_group (`int`, *optional*, defaults to None):
            Number of top-k groups.
        num_experts_per_tok (`int`, *optional*, defaults to None):
            Number of experts per token.
        moe_layer_freq (`int`, *optional*, defaults to 1):
            Frequency of MoE layers.
        first_k_dense_replace (`int`, *optional*, defaults to 0):
            Number of dense layers to replace.
        norm_topk_prob (`bool`, *optional*, defaults to False):
            Whether to normalize the top-k probabilities.
        scoring_func (`str`, *optional*, defaults to 'softmax'):
            Scoring function for routing.
        aux_loss_alpha (`float`, *optional*, defaults to 0.001):
            Weight of the auxiliary loss.
        seq_aux (`bool`, *optional*, defaults to True):
            Whether to use sequence-level auxiliary loss.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            Number of key-value heads.
        hidden_act  (`str` or `function`, *optional*, defaults to `"silu"`):
            Activation function for the hidden layer.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            Maximum length of the input sequence.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Range for the initializer.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            Epsilon for the RMS normalization.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        pad_token_id (`int`, *optional*, defaults to `None`):
            Token id for the padding token.
        bos_token_id (`int`, *optional*, defaults to `100000`):
            Token id for the beginning of sequence token.
        eos_token_id (`int`, *optional*, defaults to `100001`):
            Token id for the end of sequence token.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Number of tokens per training example.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie word embeddings.
        rope_theta  (`float`, *optional*, defaults to 10000.0):
            Theta parameter for the RoPE.
        rope_scaling (`Dict`, *optional*):
            Scaling parameters for the RoPE.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use attention bias.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout rate for the attention layer.

    Example:

    ```python
    >>> from transformers import DeepSeekV2Config, DeepSeekV2Model

    >>> # Initializing a DeepSeekV2 google/deepseek-v2-base style configuration
    >>> configuration = DeepSeekV2Config()

    >>> # Initializing a model (with random weights) from the google/deepseek-v2-base style configuration
    >>> model = DeepSeekV2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """
    model_type = "deepseek_v2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=102400,
        hidden_size=4096,
        intermediate_size=11008,
        moe_intermediate_size=1407,
        num_hidden_layers=30,
        num_attention_heads=32,
        num_key_value_heads=32,
        n_shared_experts=None,
        n_routed_experts=None,
        ep_size=1,
        routed_scaling_factor=1.0,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=128,
        topk_method='gready',
        n_group=None,
        topk_group=None,
        num_experts_per_tok=None,
        moe_layer_freq=1,
        first_k_dense_replace=0,
        norm_topk_prob=False,
        scoring_func="softmax",
        aux_loss_alpha=0.001,
        seq_aux=True,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=100000,
        eos_token_id=100001,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,        
        **kwargs    
        ):


        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.ep_size = ep_size
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.tie_word_embeddings = tie_word_embeddings
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

