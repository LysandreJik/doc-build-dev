import{S as Xn,i as Sn,s as En,e as s,k as m,w as y,t as i,M as qn,c as a,d as t,m as f,a as r,x as T,h as d,b as c,G as o,g,y as w,q as k,o as P,B as $,v as Cn,L as Wt}from"../../chunks/vendor-hf-doc-builder.js";import{T as xn}from"../../chunks/Tip-hf-doc-builder.js";import{D as to}from"../../chunks/Docstring-hf-doc-builder.js";import{C as Rt}from"../../chunks/CodeBlock-hf-doc-builder.js";import{I as no}from"../../chunks/IconCopyLink-hf-doc-builder.js";import{E as Dt}from"../../chunks/ExampleCodeBlock-hf-doc-builder.js";function Mn(S){let l,b,_,u,v;return u=new Rt({props:{code:`from transformers import PegasusXModel, PegasusXConfig

# Initializing a PEGASUS zphang/pegasus-x-large style configuration
configuration = PegasusXConfig()

# Initializing a model from the zphang/pegasus-x-large style configuration
model = PegasusXModel(configuration)

# Accessing the model configuration
configuration = model.config`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> PegasusXModel, PegasusXConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a PEGASUS zphang/pegasus-x-large style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = PegasusXConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the zphang/pegasus-x-large style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = PegasusXModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`}}),{c(){l=s("p"),b=i("Example:"),_=m(),y(u.$$.fragment)},l(n){l=a(n,"P",{});var h=r(l);b=d(h,"Example:"),h.forEach(t),_=f(n),T(u.$$.fragment,n)},m(n,h){g(n,l,h),o(l,b),g(n,_,h),w(u,n,h),v=!0},p:Wt,i(n){v||(k(u.$$.fragment,n),v=!0)},o(n){P(u.$$.fragment,n),v=!1},d(n){n&&t(l),n&&t(_),$(u,n)}}}function Fn(S){let l,b,_,u,v;return{c(){l=s("p"),b=i("Although the recipe for forward pass needs to be defined within this function, one should call the "),_=s("code"),u=i("Module"),v=i(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(n){l=a(n,"P",{});var h=r(l);b=d(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),_=a(h,"CODE",{});var C=r(_);u=d(C,"Module"),C.forEach(t),v=d(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(t)},m(n,h){g(n,l,h),o(l,b),o(l,_),o(_,u),o(l,v)},d(n){n&&t(l)}}}function An(S){let l,b,_,u,v;return u=new Rt({props:{code:`from transformers import PegasusTokenizer, PegasusModel

tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large")
model = PegasusModel.from_pretrained("zphang/pegasus-x-large")

inputs = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt")
decoder_inputs = tokenizer("Studies show that", return_tensors="pt")
outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_inputs.input_ids)

last_hidden_states = outputs.last_hidden_state
list(last_hidden_states.shape)`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> PegasusTokenizer, PegasusModel

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = PegasusTokenizer.from_pretrained(<span class="hljs-string">&quot;google/pegasus-large&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = PegasusModel.from_pretrained(<span class="hljs-string">&quot;zphang/pegasus-x-large&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Studies have been shown that owning a dog is good for you&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>decoder_inputs = tokenizer(<span class="hljs-string">&quot;Studies show that&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_inputs.input_ids)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(last_hidden_states.shape)
[<span class="hljs-number">1</span>, <span class="hljs-number">4</span>, <span class="hljs-number">1024</span>]`}}),{c(){l=s("p"),b=i("Example:"),_=m(),y(u.$$.fragment)},l(n){l=a(n,"P",{});var h=r(l);b=d(h,"Example:"),h.forEach(t),_=f(n),T(u.$$.fragment,n)},m(n,h){g(n,l,h),o(l,b),g(n,_,h),w(u,n,h),v=!0},p:Wt,i(n){v||(k(u.$$.fragment,n),v=!0)},o(n){P(u.$$.fragment,n),v=!1},d(n){n&&t(l),n&&t(_),$(u,n)}}}function Gn(S){let l,b,_,u,v;return{c(){l=s("p"),b=i("Although the recipe for forward pass needs to be defined within this function, one should call the "),_=s("code"),u=i("Module"),v=i(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(n){l=a(n,"P",{});var h=r(l);b=d(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),_=a(h,"CODE",{});var C=r(_);u=d(C,"Module"),C.forEach(t),v=d(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(t)},m(n,h){g(n,l,h),o(l,b),o(l,_),o(_,u),o(l,v)},d(n){n&&t(l)}}}function jn(S){let l,b,_,u,v;return u=new Rt({props:{code:`from transformers import PegasusTokenizer, PegasusXForConditionalGeneration

model = PegasusXForConditionalGeneration.from_pretrained("zphang/pegasus-x-base")
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large")

ARTICLE_TO_SUMMARIZE = (
    "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
)
inputs = tokenizer(ARTICLE_TO_SUMMARIZE, max_length=1024, return_tensors="pt")

# Generate Summary
summary_ids = model.generate(inputs["input_ids"])
tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> PegasusTokenizer, PegasusXForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>model = PegasusXForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;zphang/pegasus-x-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = PegasusTokenizer.from_pretrained(<span class="hljs-string">&quot;google/pegasus-large&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>ARTICLE_TO_SUMMARIZE = (
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;PG&amp;E stated it scheduled the blackouts in response to forecasts for high winds &quot;</span>
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were &quot;</span>
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow.&quot;</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(ARTICLE_TO_SUMMARIZE, max_length=<span class="hljs-number">1024</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate Summary</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>summary_ids = model.generate(inputs[<span class="hljs-string">&quot;input_ids&quot;</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(summary_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;California&#x27;s largest electricity provider has turned off power to hundreds of thousands of customers.&quot;</span>`}}),{c(){l=s("p"),b=i("Summarization example:"),_=m(),y(u.$$.fragment)},l(n){l=a(n,"P",{});var h=r(l);b=d(h,"Summarization example:"),h.forEach(t),_=f(n),T(u.$$.fragment,n)},m(n,h){g(n,l,h),o(l,b),g(n,_,h),w(u,n,h),v=!0},p:Wt,i(n){v||(k(u.$$.fragment,n),v=!0)},o(n){P(u.$$.fragment,n),v=!1},d(n){n&&t(l),n&&t(_),$(u,n)}}}function In(S){let l,b,_,u,v,n,h,C,Xo,so,F,L,Ne,ne,So,Le,Eo,ao,D,qo,se,Co,Mo,ro,ke,Fo,io,Pe,Ao,lo,$e,De,Go,co,ze,jo,po,xe,We,Io,uo,M,Oo,Xe,Uo,No,ae,Lo,Do,ho,A,W,Re,re,Wo,Ve,Ro,mo,z,ie,Vo,G,Ho,Se,Bo,Zo,de,Jo,Ko,Yo,j,Qo,Ee,et,ot,qe,tt,nt,st,R,fo,I,V,He,le,at,Be,rt,go,x,ce,it,pe,dt,Ce,lt,ct,pt,ue,ut,he,ht,mt,ft,E,me,gt,O,_t,Me,vt,bt,Ze,yt,Tt,wt,H,kt,B,_o,U,Z,Je,fe,Pt,Ke,$t,vo,X,ge,zt,_e,xt,Fe,Xt,St,Et,ve,qt,be,Ct,Mt,Ft,q,ye,At,N,Gt,Ae,jt,It,Ye,Ot,Ut,Nt,J,Lt,K,bo;return n=new no({}),ne=new no({}),re=new no({}),ie=new to({props:{name:"class transformers.PegasusXConfig",anchor:"transformers.PegasusXConfig",parameters:[{name:"vocab_size",val:" = 96103"},{name:"max_position_embeddings",val:" = 16384"},{name:"encoder_layers",val:" = 16"},{name:"encoder_ffn_dim",val:" = 4096"},{name:"encoder_attention_heads",val:" = 16"},{name:"decoder_layers",val:" = 16"},{name:"decoder_ffn_dim",val:" = 4096"},{name:"decoder_attention_heads",val:" = 16"},{name:"encoder_layerdrop",val:" = 0.0"},{name:"decoder_layerdrop",val:" = 0.0"},{name:"use_cache",val:" = True"},{name:"is_encoder_decoder",val:" = True"},{name:"activation_function",val:" = 'gelu'"},{name:"d_model",val:" = 1024"},{name:"dropout",val:" = 0.1"},{name:"attention_dropout",val:" = 0.0"},{name:"activation_dropout",val:" = 0.0"},{name:"init_std",val:" = 0.02"},{name:"decoder_start_token_id",val:" = 0"},{name:"classifier_dropout",val:" = 0.0"},{name:"scale_embedding",val:" = True"},{name:"pad_token_id",val:" = 0"},{name:"eos_token_id",val:" = 1"},{name:"forced_eos_token_id",val:" = 1"},{name:"num_global_tokens",val:" = 32"},{name:"block_size",val:" = 512"},{name:"stagger_local_blocks",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PegasusXConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 96103) &#x2014;
Vocabulary size of the PEGASUS-X model. Defines the number of different tokens that can be represented by
the <code>inputs_ids</code> passed when calling <a href="/docs/transformers/pr_18551/en/model_doc/pegasus_x#transformers.PegasusXModel">PegasusXModel</a>.`,name:"vocab_size"},{anchor:"transformers.PegasusXConfig.d_model",description:`<strong>d_model</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
Dimension of the layers and the pooler layer.`,name:"d_model"},{anchor:"transformers.PegasusXConfig.encoder_layers",description:`<strong>encoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of encoder layers.`,name:"encoder_layers"},{anchor:"transformers.PegasusXConfig.decoder_layers",description:`<strong>decoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of decoder layers.`,name:"decoder_layers"},{anchor:"transformers.PegasusXConfig.encoder_attention_heads",description:`<strong>encoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"encoder_attention_heads"},{anchor:"transformers.PegasusXConfig.decoder_attention_heads",description:`<strong>decoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"decoder_attention_heads"},{anchor:"transformers.PegasusXConfig.decoder_ffn_dim",description:`<strong>decoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimension of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"decoder_ffn_dim"},{anchor:"transformers.PegasusXConfig.encoder_ffn_dim",description:`<strong>encoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimension of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"encoder_ffn_dim"},{anchor:"transformers.PegasusXConfig.activation_function",description:`<strong>activation_function</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"activation_function"},{anchor:"transformers.PegasusXConfig.dropout",description:`<strong>dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"dropout"},{anchor:"transformers.PegasusXConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.PegasusXConfig.activation_dropout",description:`<strong>activation_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for activations inside the fully connected layer.`,name:"activation_dropout"},{anchor:"transformers.PegasusXConfig.classifier_dropout",description:`<strong>classifier_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for classifier.`,name:"classifier_dropout"},{anchor:"transformers.PegasusXConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 16384) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.PegasusXConfig.init_std",description:`<strong>init_std</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
encoder_layerdrop &#x2014; (<code>float</code>, <em>optional</em>, defaults to 0.0):
The LayerDrop probability for the encoder. See the [LayerDrop paper](see <a href="https://arxiv.org/abs/1909.11556" rel="nofollow">https://arxiv.org/abs/1909.11556</a>)
for more details.
decoder_layerdrop &#x2014; (<code>float</code>, <em>optional</em>, defaults to 0.0):
The LayerDrop probability for the decoder. See the [LayerDrop paper](see <a href="https://arxiv.org/abs/1909.11556" rel="nofollow">https://arxiv.org/abs/1909.11556</a>)
for more details.`,name:"init_std"},{anchor:"transformers.PegasusXConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models)`,name:"use_cache"},{anchor:"transformers.PegasusXConfig.forced_eos_token_id",description:`<strong>forced_eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
The id of the token to force as the last generated token when <code>max_length</code> is reached. Usually set to
<code>eos_token_id</code>.`,name:"forced_eos_token_id"},{anchor:"transformers.PegasusXConfig.num_global_tokens",description:`<strong>num_global_tokens</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
Number of global tokens to use for the encoder`,name:"num_global_tokens"},{anchor:"transformers.PegasusXConfig.block_size=512",description:`<strong>block_size=512</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
Block size for encoder local attention. Sequence length should be an exact multiple of block size.
block_size must be a multiple of 2 if stagger_local_block is True`,name:"block_size=512"},{anchor:"transformers.PegasusXConfig.stagger_local_block",description:`<strong>stagger_local_block</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to stagger every other local attention by half a block`,name:"stagger_local_block"}],source:"https://github.com/huggingface/transformers/blob/vr_18551/src/transformers/models/pegasus_x/configuration_pegasus_x.py#L30"}}),R=new Dt({props:{anchor:"transformers.PegasusXConfig.example",$$slots:{default:[Mn]},$$scope:{ctx:S}}}),le=new no({}),ce=new to({props:{name:"class transformers.PegasusXModel",anchor:"transformers.PegasusXModel",parameters:[{name:"config",val:": PegasusXConfig"}],parametersDescription:[{anchor:"transformers.PegasusXModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18551/en/model_doc/pegasus_x#transformers.PegasusXConfig">PegasusXConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/pr_18551/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18551/src/transformers/models/pegasus_x/modeling_pegasus_x.py#L1344"}}),me=new to({props:{name:"forward",anchor:"transformers.PegasusXModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[typing.Tuple[torch.FloatTensor]] = None"},{name:"past_key_values",val:": typing.Optional[typing.Tuple[torch.FloatTensor]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.PegasusXModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
it.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18551/en/model_doc/pegasus#transformers.PegasusTokenizer">PegasusTokenizer</a>. See <a href="/docs/transformers/pr_18551/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18551/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.PegasusXModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.PegasusXModel.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18551/en/model_doc/pegasus#transformers.PegasusTokenizer">PegasusTokenizer</a>. See <a href="/docs/transformers/pr_18551/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18551/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>PEGASUS-X uses the <code>pad_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_ids</code> have to be input (see
<code>past_key_values</code>).`,name:"decoder_input_ids"},{anchor:"transformers.PegasusXModel.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.PegasusXModel.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple(tuple(torch.FloatTensor)</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.PegasusXModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple(tuple(torch.FloatTensor))</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) &#x2014;
Tuple of <code>tuple(torch.FloatTensor)</code> of length <code>config.n_layers</code>, with each tuple having 2 tensors of shape
<code>(batch_size, num_heads, sequence_length, embed_size_per_head)</code>) and 2 additional tensors of shape
<code>(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)</code>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
<p>If <code>past_key_values</code> are used, the user can optionally input only the last <code>decoder_input_ids</code> (those that
don&#x2019;t have their past key value states given to this model) of shape <code>(batch_size, 1)</code> instead of all
<code>decoder_input_ids</code> of shape <code>(batch_size, sequence_length)</code>. inputs_embeds (<code>torch.FloatTensor</code> of shape
<code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>): Optionally, instead of passing <code>input_ids</code> you
can choose to directly pass an embedded representation. This is useful if you want more control over how to
convert <code>input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.`,name:"past_key_values"},{anchor:"transformers.PegasusXModel.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.PegasusXModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.PegasusXModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.PegasusXModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.PegasusXModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18551/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18551/src/transformers/models/pegasus_x/modeling_pegasus_x.py#L1394",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18551/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput"
>transformers.modeling_outputs.Seq2SeqModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18551/en/model_doc/pegasus_x#transformers.PegasusXConfig"
>PegasusXConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) \u2014 Sequence of hidden-states at the output of the last layer of the decoder of the model.</p>
<p>If <code>past_key_values</code> is used only the last hidden-state of the sequences of shape <code>(batch_size, 1, hidden_size)</code> is output.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>tuple(tuple(torch.FloatTensor))</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) \u2014 Tuple of <code>tuple(torch.FloatTensor)</code> of length <code>config.n_layers</code>, with each tuple having 2 tensors of shape
<code>(batch_size, num_heads, sequence_length, embed_size_per_head)</code>) and 2 additional tensors of shape
<code>(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)</code>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>decoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>decoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder\u2019s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
<li>
<p><strong>encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) \u2014 Sequence of hidden-states at the output of the last layer of the encoder of the model.</p>
</li>
<li>
<p><strong>encoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>encoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18551/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput"
>transformers.modeling_outputs.Seq2SeqModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),H=new xn({props:{$$slots:{default:[Fn]},$$scope:{ctx:S}}}),B=new Dt({props:{anchor:"transformers.PegasusXModel.forward.example",$$slots:{default:[An]},$$scope:{ctx:S}}}),fe=new no({}),ge=new to({props:{name:"class transformers.PegasusXForConditionalGeneration",anchor:"transformers.PegasusXForConditionalGeneration",parameters:[{name:"config",val:": PegasusXConfig"}],parametersDescription:[{anchor:"transformers.PegasusXForConditionalGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18551/en/model_doc/pegasus_x#transformers.PegasusXConfig">PegasusXConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/pr_18551/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18551/src/transformers/models/pegasus_x/modeling_pegasus_x.py#L1485"}}),ye=new to({props:{name:"forward",anchor:"transformers.PegasusXForConditionalGeneration.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[typing.Tuple[torch.FloatTensor]] = None"},{name:"past_key_values",val:": typing.Optional[typing.Tuple[torch.FloatTensor]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.PegasusXForConditionalGeneration.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
it.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18551/en/model_doc/pegasus#transformers.PegasusTokenizer">PegasusTokenizer</a>. See <a href="/docs/transformers/pr_18551/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18551/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.PegasusXForConditionalGeneration.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.PegasusXForConditionalGeneration.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18551/en/model_doc/pegasus#transformers.PegasusTokenizer">PegasusTokenizer</a>. See <a href="/docs/transformers/pr_18551/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18551/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>PEGASUS-X uses the <code>pad_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_ids</code> have to be input (see
<code>past_key_values</code>).`,name:"decoder_input_ids"},{anchor:"transformers.PegasusXForConditionalGeneration.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.PegasusXForConditionalGeneration.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple(tuple(torch.FloatTensor)</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.PegasusXForConditionalGeneration.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple(tuple(torch.FloatTensor))</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) &#x2014;
Tuple of <code>tuple(torch.FloatTensor)</code> of length <code>config.n_layers</code>, with each tuple having 2 tensors of shape
<code>(batch_size, num_heads, sequence_length, embed_size_per_head)</code>) and 2 additional tensors of shape
<code>(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)</code>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
<p>If <code>past_key_values</code> are used, the user can optionally input only the last <code>decoder_input_ids</code> (those that
don&#x2019;t have their past key value states given to this model) of shape <code>(batch_size, 1)</code> instead of all
<code>decoder_input_ids</code> of shape <code>(batch_size, sequence_length)</code>. inputs_embeds (<code>torch.FloatTensor</code> of shape
<code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>): Optionally, instead of passing <code>input_ids</code> you
can choose to directly pass an embedded representation. This is useful if you want more control over how to
convert <code>input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.`,name:"past_key_values"},{anchor:"transformers.PegasusXForConditionalGeneration.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.PegasusXForConditionalGeneration.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.PegasusXForConditionalGeneration.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.PegasusXForConditionalGeneration.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.PegasusXForConditionalGeneration.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18551/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.PegasusXForConditionalGeneration.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18551/src/transformers/models/pegasus_x/modeling_pegasus_x.py#L1541",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18551/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput"
>transformers.modeling_outputs.Seq2SeqLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18551/en/model_doc/pegasus_x#transformers.PegasusXConfig"
>PegasusXConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) \u2014 Language modeling loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) \u2014 Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>tuple(tuple(torch.FloatTensor))</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) \u2014 Tuple of <code>tuple(torch.FloatTensor)</code> of length <code>config.n_layers</code>, with each tuple having 2 tensors of shape
<code>(batch_size, num_heads, sequence_length, embed_size_per_head)</code>) and 2 additional tensors of shape
<code>(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)</code>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>decoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>decoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder\u2019s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
<li>
<p><strong>encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) \u2014 Sequence of hidden-states at the output of the last layer of the encoder of the model.</p>
</li>
<li>
<p><strong>encoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>encoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18551/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput"
>transformers.modeling_outputs.Seq2SeqLMOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),J=new xn({props:{$$slots:{default:[Gn]},$$scope:{ctx:S}}}),K=new Dt({props:{anchor:"transformers.PegasusXForConditionalGeneration.forward.example",$$slots:{default:[jn]},$$scope:{ctx:S}}}),{c(){l=s("meta"),b=m(),_=s("h1"),u=s("a"),v=s("span"),y(n.$$.fragment),h=m(),C=s("span"),Xo=i("PEGASUS-X"),so=m(),F=s("h2"),L=s("a"),Ne=s("span"),y(ne.$$.fragment),So=m(),Le=s("span"),Eo=i("Overview"),ao=m(),D=s("p"),qo=i("The PEGASUS-X model was proposed in "),se=s("a"),Co=i("Investigating Efficiently Extending Transformers for Long Input Summarization"),Mo=i("  by Jason Phang, Yao Zhao and Peter J. Liu."),ro=m(),ke=s("p"),Fo=i("PEGASUS-X (PEGASUS eXtended) extends the PEGASUS models for long input summarization through additional long input pretraining and using staggered block-local attention with global tokens in the encoder."),io=m(),Pe=s("p"),Ao=i("The abstract from the paper is the following:"),lo=m(),$e=s("p"),De=s("em"),Go=i("While large pretrained Transformer models have proven highly capable at tackling natural language tasks, handling long sequence inputs continues to be a significant challenge. One such task is long input summarization, where inputs are longer than the maximum input context of most pretrained models. Through an extensive set of experiments, we investigate what model architectural changes and pretraining paradigms can most efficiently adapt a pretrained Transformer for long input summarization. We find that a staggered, block-local Transformer with global encoder tokens strikes a good balance of performance and efficiency, and that an additional pretraining phase on long sequences meaningfully improves downstream summarization performance. Based on our findings, we introduce PEGASUS-X, an extension of the PEGASUS model with additional long input pretraining to handle inputs of up to 16K tokens. PEGASUS-X achieves strong performance on long input summarization tasks comparable with much larger models while adding few additional parameters and not requiring model parallelism to train."),co=m(),ze=s("p"),jo=i("Tips:"),po=m(),xe=s("ul"),We=s("li"),Io=i("PEGASUS-X uses the same tokenizer as PEGASUS."),uo=m(),M=s("p"),Oo=i("This model was contributed by "),Xe=s("a"),Uo=i("zphang"),No=i(". The original code can be found "),ae=s("a"),Lo=i("here"),Do=i("."),ho=m(),A=s("h2"),W=s("a"),Re=s("span"),y(re.$$.fragment),Wo=m(),Ve=s("span"),Ro=i("PegasusXConfig"),mo=m(),z=s("div"),y(ie.$$.fragment),Vo=m(),G=s("p"),Ho=i("This is the configuration class to store the configuration of a "),Se=s("a"),Bo=i("PegasusXModel"),Zo=i(`. It is used to instantiate a
PEGASUS-X model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the PEGASUS-X
`),de=s("a"),Jo=i("zphang/pegasus-x-large"),Ko=i(" architecture."),Yo=m(),j=s("p"),Qo=i("Configuration objects inherit from "),Ee=s("a"),et=i("PretrainedConfig"),ot=i(` and can be used to control the model outputs. Read the
documentation from `),qe=s("a"),tt=i("PretrainedConfig"),nt=i(" for more information."),st=m(),y(R.$$.fragment),fo=m(),I=s("h2"),V=s("a"),He=s("span"),y(le.$$.fragment),at=m(),Be=s("span"),rt=i("PegasusXModel"),go=m(),x=s("div"),y(ce.$$.fragment),it=m(),pe=s("p"),dt=i(`The bare PEGASUS-X Model outputting raw hidden-states without any specific head on top.
This model inherits from `),Ce=s("a"),lt=i("PreTrainedModel"),ct=i(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),pt=m(),ue=s("p"),ut=i("This model is also a PyTorch "),he=s("a"),ht=i("torch.nn.Module"),mt=i(` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),ft=m(),E=s("div"),y(me.$$.fragment),gt=m(),O=s("p"),_t=i("The "),Me=s("a"),vt=i("PegasusXModel"),bt=i(" forward method, overrides the "),Ze=s("code"),yt=i("__call__"),Tt=i(" special method."),wt=m(),y(H.$$.fragment),kt=m(),y(B.$$.fragment),_o=m(),U=s("h2"),Z=s("a"),Je=s("span"),y(fe.$$.fragment),Pt=m(),Ke=s("span"),$t=i("PegasusXForConditionalGeneration"),vo=m(),X=s("div"),y(ge.$$.fragment),zt=m(),_e=s("p"),xt=i(`The PEGASUS-X for conditional generation (e.g. summarization).
This model inherits from `),Fe=s("a"),Xt=i("PreTrainedModel"),St=i(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Et=m(),ve=s("p"),qt=i("This model is also a PyTorch "),be=s("a"),Ct=i("torch.nn.Module"),Mt=i(` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Ft=m(),q=s("div"),y(ye.$$.fragment),At=m(),N=s("p"),Gt=i("The "),Ae=s("a"),jt=i("PegasusXForConditionalGeneration"),It=i(" forward method, overrides the "),Ye=s("code"),Ot=i("__call__"),Ut=i(" special method."),Nt=m(),y(J.$$.fragment),Lt=m(),y(K.$$.fragment),this.h()},l(e){const p=qn('[data-svelte="svelte-1phssyn"]',document.head);l=a(p,"META",{name:!0,content:!0}),p.forEach(t),b=f(e),_=a(e,"H1",{class:!0});var Te=r(_);u=a(Te,"A",{id:!0,class:!0,href:!0});var Qe=r(u);v=a(Qe,"SPAN",{});var eo=r(v);T(n.$$.fragment,eo),eo.forEach(t),Qe.forEach(t),h=f(Te),C=a(Te,"SPAN",{});var oo=r(C);Xo=d(oo,"PEGASUS-X"),oo.forEach(t),Te.forEach(t),so=f(e),F=a(e,"H2",{class:!0});var we=r(F);L=a(we,"A",{id:!0,class:!0,href:!0});var Vt=r(L);Ne=a(Vt,"SPAN",{});var Ht=r(Ne);T(ne.$$.fragment,Ht),Ht.forEach(t),Vt.forEach(t),So=f(we),Le=a(we,"SPAN",{});var Bt=r(Le);Eo=d(Bt,"Overview"),Bt.forEach(t),we.forEach(t),ao=f(e),D=a(e,"P",{});var yo=r(D);qo=d(yo,"The PEGASUS-X model was proposed in "),se=a(yo,"A",{href:!0,rel:!0});var Zt=r(se);Co=d(Zt,"Investigating Efficiently Extending Transformers for Long Input Summarization"),Zt.forEach(t),Mo=d(yo,"  by Jason Phang, Yao Zhao and Peter J. Liu."),yo.forEach(t),ro=f(e),ke=a(e,"P",{});var Jt=r(ke);Fo=d(Jt,"PEGASUS-X (PEGASUS eXtended) extends the PEGASUS models for long input summarization through additional long input pretraining and using staggered block-local attention with global tokens in the encoder."),Jt.forEach(t),io=f(e),Pe=a(e,"P",{});var Kt=r(Pe);Ao=d(Kt,"The abstract from the paper is the following:"),Kt.forEach(t),lo=f(e),$e=a(e,"P",{});var Yt=r($e);De=a(Yt,"EM",{});var Qt=r(De);Go=d(Qt,"While large pretrained Transformer models have proven highly capable at tackling natural language tasks, handling long sequence inputs continues to be a significant challenge. One such task is long input summarization, where inputs are longer than the maximum input context of most pretrained models. Through an extensive set of experiments, we investigate what model architectural changes and pretraining paradigms can most efficiently adapt a pretrained Transformer for long input summarization. We find that a staggered, block-local Transformer with global encoder tokens strikes a good balance of performance and efficiency, and that an additional pretraining phase on long sequences meaningfully improves downstream summarization performance. Based on our findings, we introduce PEGASUS-X, an extension of the PEGASUS model with additional long input pretraining to handle inputs of up to 16K tokens. PEGASUS-X achieves strong performance on long input summarization tasks comparable with much larger models while adding few additional parameters and not requiring model parallelism to train."),Qt.forEach(t),Yt.forEach(t),co=f(e),ze=a(e,"P",{});var en=r(ze);jo=d(en,"Tips:"),en.forEach(t),po=f(e),xe=a(e,"UL",{});var on=r(xe);We=a(on,"LI",{});var tn=r(We);Io=d(tn,"PEGASUS-X uses the same tokenizer as PEGASUS."),tn.forEach(t),on.forEach(t),uo=f(e),M=a(e,"P",{});var Ge=r(M);Oo=d(Ge,"This model was contributed by "),Xe=a(Ge,"A",{href:!0});var nn=r(Xe);Uo=d(nn,"zphang"),nn.forEach(t),No=d(Ge,". The original code can be found "),ae=a(Ge,"A",{href:!0,rel:!0});var sn=r(ae);Lo=d(sn,"here"),sn.forEach(t),Do=d(Ge,"."),Ge.forEach(t),ho=f(e),A=a(e,"H2",{class:!0});var To=r(A);W=a(To,"A",{id:!0,class:!0,href:!0});var an=r(W);Re=a(an,"SPAN",{});var rn=r(Re);T(re.$$.fragment,rn),rn.forEach(t),an.forEach(t),Wo=f(To),Ve=a(To,"SPAN",{});var dn=r(Ve);Ro=d(dn,"PegasusXConfig"),dn.forEach(t),To.forEach(t),mo=f(e),z=a(e,"DIV",{class:!0});var Y=r(z);T(ie.$$.fragment,Y),Vo=f(Y),G=a(Y,"P",{});var je=r(G);Ho=d(je,"This is the configuration class to store the configuration of a "),Se=a(je,"A",{href:!0});var ln=r(Se);Bo=d(ln,"PegasusXModel"),ln.forEach(t),Zo=d(je,`. It is used to instantiate a
PEGASUS-X model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the PEGASUS-X
`),de=a(je,"A",{href:!0,rel:!0});var cn=r(de);Jo=d(cn,"zphang/pegasus-x-large"),cn.forEach(t),Ko=d(je," architecture."),je.forEach(t),Yo=f(Y),j=a(Y,"P",{});var Ie=r(j);Qo=d(Ie,"Configuration objects inherit from "),Ee=a(Ie,"A",{href:!0});var pn=r(Ee);et=d(pn,"PretrainedConfig"),pn.forEach(t),ot=d(Ie,` and can be used to control the model outputs. Read the
documentation from `),qe=a(Ie,"A",{href:!0});var un=r(qe);tt=d(un,"PretrainedConfig"),un.forEach(t),nt=d(Ie," for more information."),Ie.forEach(t),st=f(Y),T(R.$$.fragment,Y),Y.forEach(t),fo=f(e),I=a(e,"H2",{class:!0});var wo=r(I);V=a(wo,"A",{id:!0,class:!0,href:!0});var hn=r(V);He=a(hn,"SPAN",{});var mn=r(He);T(le.$$.fragment,mn),mn.forEach(t),hn.forEach(t),at=f(wo),Be=a(wo,"SPAN",{});var fn=r(Be);rt=d(fn,"PegasusXModel"),fn.forEach(t),wo.forEach(t),go=f(e),x=a(e,"DIV",{class:!0});var Q=r(x);T(ce.$$.fragment,Q),it=f(Q),pe=a(Q,"P",{});var ko=r(pe);dt=d(ko,`The bare PEGASUS-X Model outputting raw hidden-states without any specific head on top.
This model inherits from `),Ce=a(ko,"A",{href:!0});var gn=r(Ce);lt=d(gn,"PreTrainedModel"),gn.forEach(t),ct=d(ko,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),ko.forEach(t),pt=f(Q),ue=a(Q,"P",{});var Po=r(ue);ut=d(Po,"This model is also a PyTorch "),he=a(Po,"A",{href:!0,rel:!0});var _n=r(he);ht=d(_n,"torch.nn.Module"),_n.forEach(t),mt=d(Po,` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Po.forEach(t),ft=f(Q),E=a(Q,"DIV",{class:!0});var ee=r(E);T(me.$$.fragment,ee),gt=f(ee),O=a(ee,"P",{});var Oe=r(O);_t=d(Oe,"The "),Me=a(Oe,"A",{href:!0});var vn=r(Me);vt=d(vn,"PegasusXModel"),vn.forEach(t),bt=d(Oe," forward method, overrides the "),Ze=a(Oe,"CODE",{});var bn=r(Ze);yt=d(bn,"__call__"),bn.forEach(t),Tt=d(Oe," special method."),Oe.forEach(t),wt=f(ee),T(H.$$.fragment,ee),kt=f(ee),T(B.$$.fragment,ee),ee.forEach(t),Q.forEach(t),_o=f(e),U=a(e,"H2",{class:!0});var $o=r(U);Z=a($o,"A",{id:!0,class:!0,href:!0});var yn=r(Z);Je=a(yn,"SPAN",{});var Tn=r(Je);T(fe.$$.fragment,Tn),Tn.forEach(t),yn.forEach(t),Pt=f($o),Ke=a($o,"SPAN",{});var wn=r(Ke);$t=d(wn,"PegasusXForConditionalGeneration"),wn.forEach(t),$o.forEach(t),vo=f(e),X=a(e,"DIV",{class:!0});var oe=r(X);T(ge.$$.fragment,oe),zt=f(oe),_e=a(oe,"P",{});var zo=r(_e);xt=d(zo,`The PEGASUS-X for conditional generation (e.g. summarization).
This model inherits from `),Fe=a(zo,"A",{href:!0});var kn=r(Fe);Xt=d(kn,"PreTrainedModel"),kn.forEach(t),St=d(zo,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),zo.forEach(t),Et=f(oe),ve=a(oe,"P",{});var xo=r(ve);qt=d(xo,"This model is also a PyTorch "),be=a(xo,"A",{href:!0,rel:!0});var Pn=r(be);Ct=d(Pn,"torch.nn.Module"),Pn.forEach(t),Mt=d(xo,` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),xo.forEach(t),Ft=f(oe),q=a(oe,"DIV",{class:!0});var te=r(q);T(ye.$$.fragment,te),At=f(te),N=a(te,"P",{});var Ue=r(N);Gt=d(Ue,"The "),Ae=a(Ue,"A",{href:!0});var $n=r(Ae);jt=d($n,"PegasusXForConditionalGeneration"),$n.forEach(t),It=d(Ue," forward method, overrides the "),Ye=a(Ue,"CODE",{});var zn=r(Ye);Ot=d(zn,"__call__"),zn.forEach(t),Ut=d(Ue," special method."),Ue.forEach(t),Nt=f(te),T(J.$$.fragment,te),Lt=f(te),T(K.$$.fragment,te),te.forEach(t),oe.forEach(t),this.h()},h(){c(l,"name","hf:doc:metadata"),c(l,"content",JSON.stringify(On)),c(u,"id","pegasusx"),c(u,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(u,"href","#pegasusx"),c(_,"class","relative group"),c(L,"id","overview"),c(L,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(L,"href","#overview"),c(F,"class","relative group"),c(se,"href","https://arxiv.org/abs/2208.04347"),c(se,"rel","nofollow"),c(Xe,"href","<https://huggingface.co/zphang"),c(ae,"href","https://github.com/google-research/pegasus"),c(ae,"rel","nofollow"),c(W,"id","transformers.PegasusXConfig"),c(W,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(W,"href","#transformers.PegasusXConfig"),c(A,"class","relative group"),c(Se,"href","/docs/transformers/pr_18551/en/model_doc/pegasus_x#transformers.PegasusXModel"),c(de,"href","https://huggingface.co/zphang/pegasus-x-large"),c(de,"rel","nofollow"),c(Ee,"href","/docs/transformers/pr_18551/en/main_classes/configuration#transformers.PretrainedConfig"),c(qe,"href","/docs/transformers/pr_18551/en/main_classes/configuration#transformers.PretrainedConfig"),c(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(V,"id","transformers.PegasusXModel"),c(V,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(V,"href","#transformers.PegasusXModel"),c(I,"class","relative group"),c(Ce,"href","/docs/transformers/pr_18551/en/main_classes/model#transformers.PreTrainedModel"),c(he,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),c(he,"rel","nofollow"),c(Me,"href","/docs/transformers/pr_18551/en/model_doc/pegasus_x#transformers.PegasusXModel"),c(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(Z,"id","transformers.PegasusXForConditionalGeneration"),c(Z,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(Z,"href","#transformers.PegasusXForConditionalGeneration"),c(U,"class","relative group"),c(Fe,"href","/docs/transformers/pr_18551/en/main_classes/model#transformers.PreTrainedModel"),c(be,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),c(be,"rel","nofollow"),c(Ae,"href","/docs/transformers/pr_18551/en/model_doc/pegasus_x#transformers.PegasusXForConditionalGeneration"),c(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,p){o(document.head,l),g(e,b,p),g(e,_,p),o(_,u),o(u,v),w(n,v,null),o(_,h),o(_,C),o(C,Xo),g(e,so,p),g(e,F,p),o(F,L),o(L,Ne),w(ne,Ne,null),o(F,So),o(F,Le),o(Le,Eo),g(e,ao,p),g(e,D,p),o(D,qo),o(D,se),o(se,Co),o(D,Mo),g(e,ro,p),g(e,ke,p),o(ke,Fo),g(e,io,p),g(e,Pe,p),o(Pe,Ao),g(e,lo,p),g(e,$e,p),o($e,De),o(De,Go),g(e,co,p),g(e,ze,p),o(ze,jo),g(e,po,p),g(e,xe,p),o(xe,We),o(We,Io),g(e,uo,p),g(e,M,p),o(M,Oo),o(M,Xe),o(Xe,Uo),o(M,No),o(M,ae),o(ae,Lo),o(M,Do),g(e,ho,p),g(e,A,p),o(A,W),o(W,Re),w(re,Re,null),o(A,Wo),o(A,Ve),o(Ve,Ro),g(e,mo,p),g(e,z,p),w(ie,z,null),o(z,Vo),o(z,G),o(G,Ho),o(G,Se),o(Se,Bo),o(G,Zo),o(G,de),o(de,Jo),o(G,Ko),o(z,Yo),o(z,j),o(j,Qo),o(j,Ee),o(Ee,et),o(j,ot),o(j,qe),o(qe,tt),o(j,nt),o(z,st),w(R,z,null),g(e,fo,p),g(e,I,p),o(I,V),o(V,He),w(le,He,null),o(I,at),o(I,Be),o(Be,rt),g(e,go,p),g(e,x,p),w(ce,x,null),o(x,it),o(x,pe),o(pe,dt),o(pe,Ce),o(Ce,lt),o(pe,ct),o(x,pt),o(x,ue),o(ue,ut),o(ue,he),o(he,ht),o(ue,mt),o(x,ft),o(x,E),w(me,E,null),o(E,gt),o(E,O),o(O,_t),o(O,Me),o(Me,vt),o(O,bt),o(O,Ze),o(Ze,yt),o(O,Tt),o(E,wt),w(H,E,null),o(E,kt),w(B,E,null),g(e,_o,p),g(e,U,p),o(U,Z),o(Z,Je),w(fe,Je,null),o(U,Pt),o(U,Ke),o(Ke,$t),g(e,vo,p),g(e,X,p),w(ge,X,null),o(X,zt),o(X,_e),o(_e,xt),o(_e,Fe),o(Fe,Xt),o(_e,St),o(X,Et),o(X,ve),o(ve,qt),o(ve,be),o(be,Ct),o(ve,Mt),o(X,Ft),o(X,q),w(ye,q,null),o(q,At),o(q,N),o(N,Gt),o(N,Ae),o(Ae,jt),o(N,It),o(N,Ye),o(Ye,Ot),o(N,Ut),o(q,Nt),w(J,q,null),o(q,Lt),w(K,q,null),bo=!0},p(e,[p]){const Te={};p&2&&(Te.$$scope={dirty:p,ctx:e}),R.$set(Te);const Qe={};p&2&&(Qe.$$scope={dirty:p,ctx:e}),H.$set(Qe);const eo={};p&2&&(eo.$$scope={dirty:p,ctx:e}),B.$set(eo);const oo={};p&2&&(oo.$$scope={dirty:p,ctx:e}),J.$set(oo);const we={};p&2&&(we.$$scope={dirty:p,ctx:e}),K.$set(we)},i(e){bo||(k(n.$$.fragment,e),k(ne.$$.fragment,e),k(re.$$.fragment,e),k(ie.$$.fragment,e),k(R.$$.fragment,e),k(le.$$.fragment,e),k(ce.$$.fragment,e),k(me.$$.fragment,e),k(H.$$.fragment,e),k(B.$$.fragment,e),k(fe.$$.fragment,e),k(ge.$$.fragment,e),k(ye.$$.fragment,e),k(J.$$.fragment,e),k(K.$$.fragment,e),bo=!0)},o(e){P(n.$$.fragment,e),P(ne.$$.fragment,e),P(re.$$.fragment,e),P(ie.$$.fragment,e),P(R.$$.fragment,e),P(le.$$.fragment,e),P(ce.$$.fragment,e),P(me.$$.fragment,e),P(H.$$.fragment,e),P(B.$$.fragment,e),P(fe.$$.fragment,e),P(ge.$$.fragment,e),P(ye.$$.fragment,e),P(J.$$.fragment,e),P(K.$$.fragment,e),bo=!1},d(e){t(l),e&&t(b),e&&t(_),$(n),e&&t(so),e&&t(F),$(ne),e&&t(ao),e&&t(D),e&&t(ro),e&&t(ke),e&&t(io),e&&t(Pe),e&&t(lo),e&&t($e),e&&t(co),e&&t(ze),e&&t(po),e&&t(xe),e&&t(uo),e&&t(M),e&&t(ho),e&&t(A),$(re),e&&t(mo),e&&t(z),$(ie),$(R),e&&t(fo),e&&t(I),$(le),e&&t(go),e&&t(x),$(ce),$(me),$(H),$(B),e&&t(_o),e&&t(U),$(fe),e&&t(vo),e&&t(X),$(ge),$(ye),$(J),$(K)}}}const On={local:"pegasusx",sections:[{local:"overview",title:"Overview"},{local:"transformers.PegasusXConfig",title:"PegasusXConfig"},{local:"transformers.PegasusXModel",title:"PegasusXModel"},{local:"transformers.PegasusXForConditionalGeneration",title:"PegasusXForConditionalGeneration"}],title:"PEGASUS-X"};function Un(S){return Cn(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Hn extends Xn{constructor(l){super();Sn(this,l,Un,In,En,{})}}export{Hn as default,On as metadata};
