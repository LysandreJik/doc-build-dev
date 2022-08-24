import{S as oi,i as ai,s as si,e as s,k as c,w as C,t as o,M as ni,c as n,d as r,m as h,a as i,x as w,h as a,b as l,N as ii,G as e,g as p,y as k,q as $,o as O,B as E,v as li,L as ri}from"../../chunks/vendor-hf-doc-builder.js";import{T as ei}from"../../chunks/Tip-hf-doc-builder.js";import{D as te}from"../../chunks/Docstring-hf-doc-builder.js";import{C as As}from"../../chunks/CodeBlock-hf-doc-builder.js";import{I as xt}from"../../chunks/IconCopyLink-hf-doc-builder.js";import{E as ti}from"../../chunks/ExampleCodeBlock-hf-doc-builder.js";function di(S){let f,R,u,g,T;return g=new As({props:{code:`from transformers import TrOCRForCausalLM, TrOCRConfig

# Initializing a TrOCR-base style configuration
configuration = TrOCRConfig()

# Initializing a model from the TrOCR-base style configuration
model = TrOCRForCausalLM(configuration)

# Accessing the model configuration
configuration = model.config`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> TrOCRForCausalLM, TrOCRConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a TrOCR-base style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = TrOCRConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the TrOCR-base style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = TrOCRForCausalLM(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`}}),{c(){f=s("p"),R=o("Example:"),u=c(),C(g.$$.fragment)},l(m){f=n(m,"P",{});var _=i(f);R=a(_,"Example:"),_.forEach(r),u=h(m),w(g.$$.fragment,m)},m(m,_){p(m,f,_),e(f,R),p(m,u,_),k(g,m,_),T=!0},p:ri,i(m){T||($(g.$$.fragment,m),T=!0)},o(m){O(g.$$.fragment,m),T=!1},d(m){m&&r(f),m&&r(u),E(g,m)}}}function ci(S){let f,R,u,g,T,m,_,D;return{c(){f=s("p"),R=o(`This class method is simply calling the feature extractor
`),u=s("a"),g=o("from_pretrained()"),T=o(` and the tokenizer
`),m=s("code"),_=o("from_pretrained"),D=o(` methods. Please refer to the docstrings of the
methods above for more information.`),this.h()},l(j){f=n(j,"P",{});var b=i(f);R=a(b,`This class method is simply calling the feature extractor
`),u=n(b,"A",{href:!0});var z=i(u);g=a(z,"from_pretrained()"),z.forEach(r),T=a(b,` and the tokenizer
`),m=n(b,"CODE",{});var L=i(m);_=a(L,"from_pretrained"),L.forEach(r),D=a(b,` methods. Please refer to the docstrings of the
methods above for more information.`),b.forEach(r),this.h()},h(){l(u,"href","/docs/transformers/pr_18429/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained")},m(j,b){p(j,f,b),e(f,R),e(f,u),e(u,g),e(f,T),e(f,m),e(m,_),e(f,D)},d(j){j&&r(f)}}}function hi(S){let f,R,u,g,T,m,_,D;return{c(){f=s("p"),R=o("This class method is simply calling "),u=s("a"),g=o("save_pretrained()"),T=o(` and
`),m=s("code"),_=o("save_pretrained"),D=o(`. Please refer to the docstrings of the methods
above for more information.`),this.h()},l(j){f=n(j,"P",{});var b=i(f);R=a(b,"This class method is simply calling "),u=n(b,"A",{href:!0});var z=i(u);g=a(z,"save_pretrained()"),z.forEach(r),T=a(b,` and
`),m=n(b,"CODE",{});var L=i(m);_=a(L,"save_pretrained"),L.forEach(r),D=a(b,`. Please refer to the docstrings of the methods
above for more information.`),b.forEach(r),this.h()},h(){l(u,"href","/docs/transformers/pr_18429/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained")},m(j,b){p(j,f,b),e(f,R),e(f,u),e(u,g),e(f,T),e(f,m),e(m,_),e(f,D)},d(j){j&&r(f)}}}function mi(S){let f,R,u,g,T;return g=new As({props:{code:`from transformers import (
    TrOCRConfig,
    TrOCRProcessor,
    TrOCRForCausalLM,
    ViTConfig,
    ViTModel,
    VisionEncoderDecoderModel,
)
import requests
from PIL import Image

# TrOCR is a decoder model and should be used within a VisionEncoderDecoderModel
# init vision2text model with random weights
encoder = ViTModel(ViTConfig())
decoder = TrOCRForCausalLM(TrOCRConfig())
model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)

# If you want to start from the pretrained model, load the checkpoint with \`VisionEncoderDecoderModel\`
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# load image from the IAM dataset
url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
pixel_values = processor(image, return_tensors="pt").pixel_values
text = "industry, ' Mr. Brown commented icily. ' Let us have a"

# training
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

labels = processor.tokenizer(text, return_tensors="pt").input_ids
outputs = model(pixel_values, labels=labels)
loss = outputs.loss
round(loss.item(), 2)

# inference
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
generated_text`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> (
<span class="hljs-meta">... </span>    TrOCRConfig,
<span class="hljs-meta">... </span>    TrOCRProcessor,
<span class="hljs-meta">... </span>    TrOCRForCausalLM,
<span class="hljs-meta">... </span>    ViTConfig,
<span class="hljs-meta">... </span>    ViTModel,
<span class="hljs-meta">... </span>    VisionEncoderDecoderModel,
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># TrOCR is a decoder model and should be used within a VisionEncoderDecoderModel</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># init vision2text model with random weights</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>encoder = ViTModel(ViTConfig())
<span class="hljs-meta">&gt;&gt;&gt; </span>decoder = TrOCRForCausalLM(TrOCRConfig())
<span class="hljs-meta">&gt;&gt;&gt; </span>model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># If you want to start from the pretrained model, load the checkpoint with \`VisionEncoderDecoderModel\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>processor = TrOCRProcessor.from_pretrained(<span class="hljs-string">&quot;microsoft/trocr-base-handwritten&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = VisionEncoderDecoderModel.from_pretrained(<span class="hljs-string">&quot;microsoft/trocr-base-handwritten&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># load image from the IAM dataset</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw).convert(<span class="hljs-string">&quot;RGB&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>pixel_values = processor(image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).pixel_values
<span class="hljs-meta">&gt;&gt;&gt; </span>text = <span class="hljs-string">&quot;industry, &#x27; Mr. Brown commented icily. &#x27; Let us have a&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># training</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.pad_token_id = processor.tokenizer.pad_token_id
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.vocab_size = model.config.decoder.vocab_size

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = processor.tokenizer(text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).input_ids
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(pixel_values, labels=labels)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
<span class="hljs-number">5.30</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># inference</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generated_ids = model.generate(pixel_values)
<span class="hljs-meta">&gt;&gt;&gt; </span>generated_text = processor.batch_decode(generated_ids, skip_special_tokens=<span class="hljs-literal">True</span>)[<span class="hljs-number">0</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>generated_text
<span class="hljs-string">&#x27;industry, &quot; Mr. Brown commented icily. &quot; Let us have a&#x27;</span>`}}),{c(){f=s("p"),R=o("Example:"),u=c(),C(g.$$.fragment)},l(m){f=n(m,"P",{});var _=i(f);R=a(_,"Example:"),_.forEach(r),u=h(m),w(g.$$.fragment,m)},m(m,_){p(m,f,_),e(f,R),p(m,u,_),k(g,m,_),T=!0},p:ri,i(m){T||($(g.$$.fragment,m),T=!0)},o(m){O(g.$$.fragment,m),T=!1},d(m){m&&r(f),m&&r(u),E(g,m)}}}function pi(S){let f,R,u,g,T,m,_,D,j,b,z,L,Pt,ke,Kr,Mt,Qr,mr,W,eo,$e,to,ro,Oe,oo,ao,pr,ot,so,fr,at,jt,no,ur,re,Is,gr,oe,io,st,lo,co,_r,ae,ho,zt,mo,po,vr,B,fo,Ee,uo,go,ye,_o,vo,Tr,nt,To,br,H,Re,bo,xe,Co,wo,ko,I,$o,Pe,Oo,Eo,Me,yo,Ro,je,xo,Po,Mo,ze,jo,it,zo,Fo,Cr,Z,se,Ft,Fe,Lo,Lt,Do,wr,G,qo,Dt,Ao,Io,lt,Vo,No,kr,v,So,qt,Wo,Bo,At,Ho,Go,It,Xo,Uo,Vt,Zo,Jo,dt,Yo,Ko,Nt,Qo,ea,St,ta,ra,Wt,oa,aa,Bt,sa,na,$r,ct,Ht,ia,Or,Le,Er,ne,la,De,da,ca,yr,J,ie,Gt,qe,ha,Xt,ma,Rr,q,Ae,pa,Y,fa,ht,ua,ga,Ie,_a,va,Ta,K,ba,mt,Ca,wa,pt,ka,$a,Oa,le,xr,Q,de,Ut,Ve,Ea,Zt,ya,Pr,y,Ne,Ra,Jt,xa,Pa,P,ft,Ma,ja,Yt,za,Fa,Kt,La,Da,Qt,qa,Aa,er,Ia,Va,Se,tr,Na,Sa,Wa,ut,Ba,Ha,Ga,ce,We,Xa,V,Ua,rr,Za,Ja,or,Ya,Ka,ar,Qa,es,ts,X,Be,rs,sr,os,as,he,ss,U,He,ns,Ge,is,gt,ls,ds,cs,me,hs,pe,Xe,ms,Ue,ps,_t,fs,us,gs,fe,Ze,_s,Je,vs,vt,Ts,bs,Mr,ee,ue,nr,Ye,Cs,ir,ws,jr,A,Ke,ks,N,$s,Tt,Os,Es,lr,ys,Rs,bt,xs,Ps,Ms,Qe,js,et,zs,Fs,Ls,ge,tt,Ds,_e,zr;return m=new xt({}),ke=new xt({}),Fe=new xt({}),Le=new As({props:{code:`from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import requests 
from PIL import Image

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten") 
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# load image from the IAM dataset 
url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg" 
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

pixel_values = processor(image, return_tensors="pt").pixel_values 
generated_ids = model.generate(pixel_values)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0] `,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> TrOCRProcessor, VisionEncoderDecoderModel
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests 
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = TrOCRProcessor.from_pretrained(<span class="hljs-string">&quot;microsoft/trocr-base-handwritten&quot;</span>) 
<span class="hljs-meta">&gt;&gt;&gt; </span>model = VisionEncoderDecoderModel.from_pretrained(<span class="hljs-string">&quot;microsoft/trocr-base-handwritten&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># load image from the IAM dataset </span>
<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg&quot;</span> 
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw).convert(<span class="hljs-string">&quot;RGB&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>pixel_values = processor(image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).pixel_values 
<span class="hljs-meta">&gt;&gt;&gt; </span>generated_ids = model.generate(pixel_values)

<span class="hljs-meta">&gt;&gt;&gt; </span>generated_text = processor.batch_decode(generated_ids, skip_special_tokens=<span class="hljs-literal">True</span>)[<span class="hljs-number">0</span>] `}}),qe=new xt({}),Ae=new te({props:{name:"class transformers.TrOCRConfig",anchor:"transformers.TrOCRConfig",parameters:[{name:"vocab_size",val:" = 50265"},{name:"d_model",val:" = 1024"},{name:"decoder_layers",val:" = 12"},{name:"decoder_attention_heads",val:" = 16"},{name:"decoder_ffn_dim",val:" = 4096"},{name:"activation_function",val:" = 'gelu'"},{name:"max_position_embeddings",val:" = 512"},{name:"dropout",val:" = 0.1"},{name:"attention_dropout",val:" = 0.0"},{name:"activation_dropout",val:" = 0.0"},{name:"decoder_start_token_id",val:" = 2"},{name:"classifier_dropout",val:" = 0.0"},{name:"init_std",val:" = 0.02"},{name:"decoder_layerdrop",val:" = 0.0"},{name:"use_cache",val:" = False"},{name:"scale_embedding",val:" = False"},{name:"use_learned_position_embeddings",val:" = True"},{name:"layernorm_embedding",val:" = True"},{name:"pad_token_id",val:" = 1"},{name:"bos_token_id",val:" = 0"},{name:"eos_token_id",val:" = 2"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TrOCRConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 50265) &#x2014;
Vocabulary size of the TrOCR model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/pr_18429/en/model_doc/trocr#transformers.TrOCRForCausalLM">TrOCRForCausalLM</a>.`,name:"vocab_size"},{anchor:"transformers.TrOCRConfig.d_model",description:`<strong>d_model</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
Dimensionality of the layers and the pooler layer.`,name:"d_model"},{anchor:"transformers.TrOCRConfig.decoder_layers",description:`<strong>decoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of decoder layers.`,name:"decoder_layers"},{anchor:"transformers.TrOCRConfig.decoder_attention_heads",description:`<strong>decoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"decoder_attention_heads"},{anchor:"transformers.TrOCRConfig.decoder_ffn_dim",description:`<strong>decoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"decoder_ffn_dim"},{anchor:"transformers.TrOCRConfig.activation_function",description:`<strong>activation_function</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the pooler. If string, <code>&quot;gelu&quot;</code>, <code>&quot;relu&quot;</code>,
<code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"activation_function"},{anchor:"transformers.TrOCRConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.TrOCRConfig.dropout",description:`<strong>dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, and pooler.`,name:"dropout"},{anchor:"transformers.TrOCRConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.TrOCRConfig.activation_dropout",description:`<strong>activation_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for activations inside the fully connected layer.`,name:"activation_dropout"},{anchor:"transformers.TrOCRConfig.classifier_dropout",description:`<strong>classifier_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for classifier.`,name:"classifier_dropout"},{anchor:"transformers.TrOCRConfig.init_std",description:`<strong>init_std</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
decoder_layerdrop &#x2014; (<code>float</code>, <em>optional</em>, defaults to 0.0):
The LayerDrop probability for the decoder. See the [LayerDrop paper](see <a href="https://arxiv.org/abs/1909.11556" rel="nofollow">https://arxiv.org/abs/1909.11556</a>)
for more details.`,name:"init_std"},{anchor:"transformers.TrOCRConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models).`,name:"use_cache"},{anchor:"transformers.TrOCRConfig.scale_embedding",description:`<strong>scale_embedding</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to scale the word embeddings by sqrt(d_model).`,name:"scale_embedding"},{anchor:"transformers.TrOCRConfig.use_learned_position_embeddings",description:`<strong>use_learned_position_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to use learned position embeddings. If not, sinusoidal position embeddings will be used.`,name:"use_learned_position_embeddings"},{anchor:"transformers.TrOCRConfig.layernorm_embedding",description:`<strong>layernorm_embedding</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to use a layernorm after the word + position embeddings.`,name:"layernorm_embedding"}],source:"https://github.com/huggingface/transformers/blob/vr_18429/src/transformers/models/trocr/configuration_trocr.py#L31"}}),le=new ti({props:{anchor:"transformers.TrOCRConfig.example",$$slots:{default:[di]},$$scope:{ctx:S}}}),Ve=new xt({}),Ne=new te({props:{name:"class transformers.TrOCRProcessor",anchor:"transformers.TrOCRProcessor",parameters:[{name:"feature_extractor",val:""},{name:"tokenizer",val:""}],parametersDescription:[{anchor:"transformers.TrOCRProcessor.feature_extractor",description:`<strong>feature_extractor</strong> ([<code>ViTFeatureExtractor</code>/<code>DeiTFeatureExtractor</code>]) &#x2014;
An instance of [<code>ViTFeatureExtractor</code>/<code>DeiTFeatureExtractor</code>]. The feature extractor is a required input.`,name:"feature_extractor"},{anchor:"transformers.TrOCRProcessor.tokenizer",description:`<strong>tokenizer</strong> ([<code>RobertaTokenizer</code>/<code>XLMRobertaTokenizer</code>]) &#x2014;
An instance of [<code>RobertaTokenizer</code>/<code>XLMRobertaTokenizer</code>]. The tokenizer is a required input.`,name:"tokenizer"}],source:"https://github.com/huggingface/transformers/blob/vr_18429/src/transformers/models/trocr/processing_trocr.py#L24"}}),We=new te({props:{name:"__call__",anchor:"transformers.TrOCRProcessor.__call__",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18429/src/transformers/models/trocr/processing_trocr.py#L46"}}),Be=new te({props:{name:"from_pretrained",anchor:"transformers.TrOCRProcessor.from_pretrained",parameters:[{name:"pretrained_model_name_or_path",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TrOCRProcessor.from_pretrained.pretrained_model_name_or_path",description:`<strong>pretrained_model_name_or_path</strong> (<code>str</code> or <code>os.PathLike</code>) &#x2014;
This can be either:</p>
<ul>
<li>a string, the <em>model id</em> of a pretrained feature_extractor hosted inside a model repo on
huggingface.co. Valid model ids can be located at the root-level, like <code>bert-base-uncased</code>, or
namespaced under a user or organization name, like <code>dbmdz/bert-base-german-cased</code>.</li>
<li>a path to a <em>directory</em> containing a feature extractor file saved using the
<a href="/docs/transformers/pr_18429/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained">save_pretrained()</a> method, e.g., <code>./my_model_directory/</code>.</li>
<li>a path or url to a saved feature extractor JSON <em>file</em>, e.g.,
<code>./my_model_directory/preprocessor_config.json</code>.
**kwargs &#x2014;
Additional keyword arguments passed along to both
<a href="/docs/transformers/pr_18429/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained">from_pretrained()</a> and
<code>from_pretrained</code>.</li>
</ul>`,name:"pretrained_model_name_or_path"}],source:"https://github.com/huggingface/transformers/blob/vr_18429/src/transformers/processing_utils.py#L152"}}),he=new ei({props:{$$slots:{default:[ci]},$$scope:{ctx:S}}}),He=new te({props:{name:"save_pretrained",anchor:"transformers.TrOCRProcessor.save_pretrained",parameters:[{name:"save_directory",val:""},{name:"push_to_hub",val:": bool = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TrOCRProcessor.save_pretrained.save_directory",description:`<strong>save_directory</strong> (<code>str</code> or <code>os.PathLike</code>) &#x2014;
Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
be created if it does not exist).`,name:"save_directory"},{anchor:"transformers.TrOCRProcessor.save_pretrained.push_to_hub",description:`<strong>push_to_hub</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
repository you want to push to with <code>repo_id</code> (will default to the name of <code>save_directory</code> in your
namespace).
kwargs &#x2014;
Additional key word arguments passed along to the <a href="/docs/transformers/pr_18429/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.push_to_hub">push_to_hub()</a> method.`,name:"push_to_hub"}],source:"https://github.com/huggingface/transformers/blob/vr_18429/src/transformers/processing_utils.py#L94"}}),me=new ei({props:{$$slots:{default:[hi]},$$scope:{ctx:S}}}),Xe=new te({props:{name:"batch_decode",anchor:"transformers.TrOCRProcessor.batch_decode",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18429/src/transformers/models/trocr/processing_trocr.py#L79"}}),Ze=new te({props:{name:"decode",anchor:"transformers.TrOCRProcessor.decode",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18429/src/transformers/models/trocr/processing_trocr.py#L86"}}),Ye=new xt({}),Ke=new te({props:{name:"class transformers.TrOCRForCausalLM",anchor:"transformers.TrOCRForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.TrOCRForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18429/en/model_doc/trocr#transformers.TrOCRConfig">TrOCRConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/pr_18429/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18429/src/transformers/models/trocr/modeling_trocr.py#L782"}}),tt=new te({props:{name:"forward",anchor:"transformers.TrOCRForCausalLM.forward",parameters:[{name:"input_ids",val:" = None"},{name:"attention_mask",val:" = None"},{name:"encoder_hidden_states",val:" = None"},{name:"encoder_attention_mask",val:" = None"},{name:"head_mask",val:" = None"},{name:"cross_attn_head_mask",val:" = None"},{name:"past_key_values",val:" = None"},{name:"inputs_embeds",val:" = None"},{name:"labels",val:" = None"},{name:"use_cache",val:" = None"},{name:"output_attentions",val:" = None"},{name:"output_hidden_states",val:" = None"},{name:"return_dict",val:" = None"}],parametersDescription:[{anchor:"transformers.TrOCRForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
provide it.</p>
<p>Indices can be obtained using <code>TrOCRTokenizer</code>. See <a href="/docs/transformers/pr_18429/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18429/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.TrOCRForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.TrOCRForCausalLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong>  (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.TrOCRForCausalLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used
in the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:`,name:"encoder_attention_mask"},{anchor:"transformers.TrOCRForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.TrOCRForCausalLM.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.TrOCRForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple(tuple(torch.FloatTensor))</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) &#x2014;
Tuple of <code>tuple(torch.FloatTensor)</code> of length <code>config.n_layers</code>, with each tuple having 2 tensors of
shape <code>(batch_size, num_heads, sequence_length, embed_size_per_head)</code>) and 2 additional tensors of
shape <code>(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)</code>. The two additional
tensors are only required when the model is used as a decoder in a Sequence to Sequence model.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
cross-attention blocks) that can be used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
<p>If <code>past_key_values</code> are used, the user can optionally input only the last <code>decoder_input_ids</code> (those
that don&#x2019;t have their past key value states given to this model) of shape <code>(batch_size, 1)</code> instead of
all <code>decoder_input_ids</code> of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.TrOCRForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.TrOCRForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding
(see <code>past_key_values</code>).</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"use_cache"},{anchor:"transformers.TrOCRForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under
returned tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.TrOCRForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors
for more detail.`,name:"output_hidden_states"},{anchor:"transformers.TrOCRForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18429/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18429/src/transformers/models/trocr/modeling_trocr.py#L813",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18429/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18429/en/model_doc/trocr#transformers.TrOCRConfig"
>TrOCRConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) \u2014 Language modeling loss (for next-token prediction).</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) \u2014 Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Cross attentions weights after the attention softmax, used to compute the weighted average in the
cross-attention heads.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>tuple(tuple(torch.FloatTensor))</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) \u2014 Tuple of <code>torch.FloatTensor</code> tuples of length <code>config.n_layers</code>, with each tuple containing the cached key,
value states of the self-attention and the cross-attention layers if model is used in encoder-decoder
setting. Only relevant if <code>config.is_decoder = True</code>.</p>
<p>Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
<code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18429/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),_e=new ti({props:{anchor:"transformers.TrOCRForCausalLM.forward.example",$$slots:{default:[mi]},$$scope:{ctx:S}}}),{c(){f=s("meta"),R=c(),u=s("h1"),g=s("a"),T=s("span"),C(m.$$.fragment),_=c(),D=s("span"),j=o("TrOCR"),b=c(),z=s("h2"),L=s("a"),Pt=s("span"),C(ke.$$.fragment),Kr=c(),Mt=s("span"),Qr=o("Overview"),mr=c(),W=s("p"),eo=o("The TrOCR model was proposed in "),$e=s("a"),to=o(`TrOCR: Transformer-based Optical Character Recognition with Pre-trained
Models`),ro=o(` by Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang,
Zhoujun Li, Furu Wei. TrOCR consists of an image Transformer encoder and an autoregressive text Transformer decoder to
perform `),Oe=s("a"),oo=o("optical character recognition (OCR)"),ao=o("."),pr=c(),ot=s("p"),so=o("The abstract from the paper is the following:"),fr=c(),at=s("p"),jt=s("em"),no=o(`Text recognition is a long-standing research problem for document digitalization. Existing approaches for text recognition
are usually built based on CNN for image understanding and RNN for char-level text generation. In addition, another language
model is usually needed to improve the overall accuracy as a post-processing step. In this paper, we propose an end-to-end
text recognition approach with pre-trained image Transformer and text Transformer models, namely TrOCR, which leverages the
Transformer architecture for both image understanding and wordpiece-level text generation. The TrOCR model is simple but
effective, and can be pre-trained with large-scale synthetic data and fine-tuned with human-labeled datasets. Experiments
show that the TrOCR model outperforms the current state-of-the-art models on both printed and handwritten text recognition
tasks.`),ur=c(),re=s("img"),gr=c(),oe=s("small"),io=o("TrOCR architecture. Taken from the "),st=s("a"),lo=o("original paper"),co=o("."),_r=c(),ae=s("p"),ho=o("Please refer to the "),zt=s("code"),mo=o("VisionEncoderDecoder"),po=o(" class on how to use this model."),vr=c(),B=s("p"),fo=o("This model was contributed by "),Ee=s("a"),uo=o("nielsr"),go=o(`. The original code can be found
`),ye=s("a"),_o=o("here"),vo=o("."),Tr=c(),nt=s("p"),To=o("Tips:"),br=c(),H=s("ul"),Re=s("li"),bo=o("The quickest way to get started with TrOCR is by checking the "),xe=s("a"),Co=o(`tutorial
notebooks`),wo=o(`, which show how to use the model
at inference time as well as fine-tuning on custom data.`),ko=c(),I=s("li"),$o=o(`TrOCR is pre-trained in 2 stages before being fine-tuned on downstream datasets. It achieves state-of-the-art results
on both printed (e.g. the `),Pe=s("a"),Oo=o("SROIE dataset"),Eo=o(" and handwritten (e.g. the "),Me=s("a"),yo=o(`IAM
Handwriting dataset`),Ro=o(` text recognition tasks. For more
information, see the `),je=s("a"),xo=o("official models"),Po=o("."),Mo=c(),ze=s("li"),jo=o("TrOCR is always used within the "),it=s("a"),zo=o("VisionEncoderDecoder"),Fo=o(" framework."),Cr=c(),Z=s("h2"),se=s("a"),Ft=s("span"),C(Fe.$$.fragment),Lo=c(),Lt=s("span"),Do=o("Inference"),wr=c(),G=s("p"),qo=o("TrOCR\u2019s "),Dt=s("code"),Ao=o("VisionEncoderDecoder"),Io=o(` model accepts images as input and makes use of
`),lt=s("a"),Vo=o("generate()"),No=o(" to autoregressively generate text given the input image."),kr=c(),v=s("p"),So=o("The ["),qt=s("code"),Wo=o("ViTFeatureExtractor"),Bo=o("/"),At=s("code"),Ho=o("DeiTFeatureExtractor"),Go=o(`] class is responsible for preprocessing the input image and
[`),It=s("code"),Xo=o("RobertaTokenizer"),Uo=o("/"),Vt=s("code"),Zo=o("XLMRobertaTokenizer"),Jo=o(`] decodes the generated target tokens to the target string. The
`),dt=s("a"),Yo=o("TrOCRProcessor"),Ko=o(" wraps ["),Nt=s("code"),Qo=o("ViTFeatureExtractor"),ea=o("/"),St=s("code"),ta=o("DeiTFeatureExtractor"),ra=o("] and ["),Wt=s("code"),oa=o("RobertaTokenizer"),aa=o("/"),Bt=s("code"),sa=o("XLMRobertaTokenizer"),na=o(`]
into a single instance to both extract the input features and decode the predicted token ids.`),$r=c(),ct=s("ul"),Ht=s("li"),ia=o("Step-by-step Optical Character Recognition (OCR)"),Or=c(),C(Le.$$.fragment),Er=c(),ne=s("p"),la=o("See the "),De=s("a"),da=o("model hub"),ca=o(" to look for TrOCR checkpoints."),yr=c(),J=s("h2"),ie=s("a"),Gt=s("span"),C(qe.$$.fragment),ha=c(),Xt=s("span"),ma=o("TrOCRConfig"),Rr=c(),q=s("div"),C(Ae.$$.fragment),pa=c(),Y=s("p"),fa=o("This is the configuration class to store the configuration of a "),ht=s("a"),ua=o("TrOCRForCausalLM"),ga=o(`. It is used to instantiate an
TrOCR model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the TrOCR
`),Ie=s("a"),_a=o("microsoft/trocr-base-handwritten"),va=o(" architecture."),Ta=c(),K=s("p"),ba=o("Configuration objects inherit from "),mt=s("a"),Ca=o("PretrainedConfig"),wa=o(` and can be used to control the model outputs. Read the
documentation from `),pt=s("a"),ka=o("PretrainedConfig"),$a=o(" for more information."),Oa=c(),C(le.$$.fragment),xr=c(),Q=s("h2"),de=s("a"),Ut=s("span"),C(Ve.$$.fragment),Ea=c(),Zt=s("span"),ya=o("TrOCRProcessor"),Pr=c(),y=s("div"),C(Ne.$$.fragment),Ra=c(),Jt=s("p"),xa=o("Constructs a TrOCR processor which wraps a vision feature extractor and a TrOCR tokenizer into a single processor."),Pa=c(),P=s("p"),ft=s("a"),Ma=o("TrOCRProcessor"),ja=o(" offers all the functionalities of ["),Yt=s("code"),za=o("ViTFeatureExtractor"),Fa=o("/"),Kt=s("code"),La=o("DeiTFeatureExtractor"),Da=o(`] and
[`),Qt=s("code"),qa=o("RobertaTokenizer"),Aa=o("/"),er=s("code"),Ia=o("XLMRobertaTokenizer"),Va=o("]. See the "),Se=s("a"),tr=s("strong"),Na=o("call"),Sa=o("()"),Wa=o(" and "),ut=s("a"),Ba=o("decode()"),Ha=o(` for
more information.`),Ga=c(),ce=s("div"),C(We.$$.fragment),Xa=c(),V=s("p"),Ua=o(`When used in normal mode, this method forwards all its arguments to AutoFeatureExtractor\u2019s
`),rr=s("code"),Za=o("__call__()"),Ja=o(` and returns its output. If used in the context
`),or=s("code"),Ya=o("as_target_processor()"),Ka=o(` this method forwards all its arguments to TrOCRTokenizer\u2019s
`),ar=s("code"),Qa=o("__call__"),es=o(". Please refer to the doctsring of the above two methods for more information."),ts=c(),X=s("div"),C(Be.$$.fragment),rs=c(),sr=s("p"),os=o("Instantiate a processor associated with a pretrained model."),as=c(),C(he.$$.fragment),ss=c(),U=s("div"),C(He.$$.fragment),ns=c(),Ge=s("p"),is=o(`Saves the attributes of this processor (feature extractor, tokenizer\u2026) in the specified directory so that it
can be reloaded using the `),gt=s("a"),ls=o("from_pretrained()"),ds=o(" method."),cs=c(),C(me.$$.fragment),hs=c(),pe=s("div"),C(Xe.$$.fragment),ms=c(),Ue=s("p"),ps=o("This method forwards all its arguments to TrOCRTokenizer\u2019s "),_t=s("a"),fs=o("batch_decode()"),us=o(`. Please refer
to the docstring of this method for more information.`),gs=c(),fe=s("div"),C(Ze.$$.fragment),_s=c(),Je=s("p"),vs=o("This method forwards all its arguments to TrOCRTokenizer\u2019s "),vt=s("a"),Ts=o("decode()"),bs=o(`. Please refer to the
docstring of this method for more information.`),Mr=c(),ee=s("h2"),ue=s("a"),nr=s("span"),C(Ye.$$.fragment),Cs=c(),ir=s("span"),ws=o("TrOCRForCausalLM"),jr=c(),A=s("div"),C(Ke.$$.fragment),ks=c(),N=s("p"),$s=o("The TrOCR Decoder with a language modeling head. Can be used as the decoder part of "),Tt=s("a"),Os=o("EncoderDecoderModel"),Es=o(" and "),lr=s("code"),ys=o("VisionEncoderDecoder"),Rs=o(`.
This model inherits from `),bt=s("a"),xs=o("PreTrainedModel"),Ps=o(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Ms=c(),Qe=s("p"),js=o("This model is also a PyTorch "),et=s("a"),zs=o("torch.nn.Module"),Fs=o(` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Ls=c(),ge=s("div"),C(tt.$$.fragment),Ds=c(),C(_e.$$.fragment),this.h()},l(t){const d=ni('[data-svelte="svelte-1phssyn"]',document.head);f=n(d,"META",{name:!0,content:!0}),d.forEach(r),R=h(t),u=n(t,"H1",{class:!0});var rt=i(u);g=n(rt,"A",{id:!0,class:!0,href:!0});var dr=i(g);T=n(dr,"SPAN",{});var cr=i(T);w(m.$$.fragment,cr),cr.forEach(r),dr.forEach(r),_=h(rt),D=n(rt,"SPAN",{});var hr=i(D);j=a(hr,"TrOCR"),hr.forEach(r),rt.forEach(r),b=h(t),z=n(t,"H2",{class:!0});var Fr=i(z);L=n(Fr,"A",{id:!0,class:!0,href:!0});var Vs=i(L);Pt=n(Vs,"SPAN",{});var Ns=i(Pt);w(ke.$$.fragment,Ns),Ns.forEach(r),Vs.forEach(r),Kr=h(Fr),Mt=n(Fr,"SPAN",{});var Ss=i(Mt);Qr=a(Ss,"Overview"),Ss.forEach(r),Fr.forEach(r),mr=h(t),W=n(t,"P",{});var Ct=i(W);eo=a(Ct,"The TrOCR model was proposed in "),$e=n(Ct,"A",{href:!0,rel:!0});var Ws=i($e);to=a(Ws,`TrOCR: Transformer-based Optical Character Recognition with Pre-trained
Models`),Ws.forEach(r),ro=a(Ct,` by Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang,
Zhoujun Li, Furu Wei. TrOCR consists of an image Transformer encoder and an autoregressive text Transformer decoder to
perform `),Oe=n(Ct,"A",{href:!0,rel:!0});var Bs=i(Oe);oo=a(Bs,"optical character recognition (OCR)"),Bs.forEach(r),ao=a(Ct,"."),Ct.forEach(r),pr=h(t),ot=n(t,"P",{});var Hs=i(ot);so=a(Hs,"The abstract from the paper is the following:"),Hs.forEach(r),fr=h(t),at=n(t,"P",{});var Gs=i(at);jt=n(Gs,"EM",{});var Xs=i(jt);no=a(Xs,`Text recognition is a long-standing research problem for document digitalization. Existing approaches for text recognition
are usually built based on CNN for image understanding and RNN for char-level text generation. In addition, another language
model is usually needed to improve the overall accuracy as a post-processing step. In this paper, we propose an end-to-end
text recognition approach with pre-trained image Transformer and text Transformer models, namely TrOCR, which leverages the
Transformer architecture for both image understanding and wordpiece-level text generation. The TrOCR model is simple but
effective, and can be pre-trained with large-scale synthetic data and fine-tuned with human-labeled datasets. Experiments
show that the TrOCR model outperforms the current state-of-the-art models on both printed and handwritten text recognition
tasks.`),Xs.forEach(r),Gs.forEach(r),ur=h(t),re=n(t,"IMG",{src:!0,alt:!0,width:!0}),gr=h(t),oe=n(t,"SMALL",{});var Lr=i(oe);io=a(Lr,"TrOCR architecture. Taken from the "),st=n(Lr,"A",{href:!0});var Us=i(st);lo=a(Us,"original paper"),Us.forEach(r),co=a(Lr,"."),Lr.forEach(r),_r=h(t),ae=n(t,"P",{});var Dr=i(ae);ho=a(Dr,"Please refer to the "),zt=n(Dr,"CODE",{});var Zs=i(zt);mo=a(Zs,"VisionEncoderDecoder"),Zs.forEach(r),po=a(Dr," class on how to use this model."),Dr.forEach(r),vr=h(t),B=n(t,"P",{});var wt=i(B);fo=a(wt,"This model was contributed by "),Ee=n(wt,"A",{href:!0,rel:!0});var Js=i(Ee);uo=a(Js,"nielsr"),Js.forEach(r),go=a(wt,`. The original code can be found
`),ye=n(wt,"A",{href:!0,rel:!0});var Ys=i(ye);_o=a(Ys,"here"),Ys.forEach(r),vo=a(wt,"."),wt.forEach(r),Tr=h(t),nt=n(t,"P",{});var Ks=i(nt);To=a(Ks,"Tips:"),Ks.forEach(r),br=h(t),H=n(t,"UL",{});var kt=i(H);Re=n(kt,"LI",{});var qr=i(Re);bo=a(qr,"The quickest way to get started with TrOCR is by checking the "),xe=n(qr,"A",{href:!0,rel:!0});var Qs=i(xe);Co=a(Qs,`tutorial
notebooks`),Qs.forEach(r),wo=a(qr,`, which show how to use the model
at inference time as well as fine-tuning on custom data.`),qr.forEach(r),ko=h(kt),I=n(kt,"LI",{});var ve=i(I);$o=a(ve,`TrOCR is pre-trained in 2 stages before being fine-tuned on downstream datasets. It achieves state-of-the-art results
on both printed (e.g. the `),Pe=n(ve,"A",{href:!0,rel:!0});var en=i(Pe);Oo=a(en,"SROIE dataset"),en.forEach(r),Eo=a(ve," and handwritten (e.g. the "),Me=n(ve,"A",{href:!0,rel:!0});var tn=i(Me);yo=a(tn,`IAM
Handwriting dataset`),tn.forEach(r),Ro=a(ve,` text recognition tasks. For more
information, see the `),je=n(ve,"A",{href:!0,rel:!0});var rn=i(je);xo=a(rn,"official models"),rn.forEach(r),Po=a(ve,"."),ve.forEach(r),Mo=h(kt),ze=n(kt,"LI",{});var Ar=i(ze);jo=a(Ar,"TrOCR is always used within the "),it=n(Ar,"A",{href:!0});var on=i(it);zo=a(on,"VisionEncoderDecoder"),on.forEach(r),Fo=a(Ar," framework."),Ar.forEach(r),kt.forEach(r),Cr=h(t),Z=n(t,"H2",{class:!0});var Ir=i(Z);se=n(Ir,"A",{id:!0,class:!0,href:!0});var an=i(se);Ft=n(an,"SPAN",{});var sn=i(Ft);w(Fe.$$.fragment,sn),sn.forEach(r),an.forEach(r),Lo=h(Ir),Lt=n(Ir,"SPAN",{});var nn=i(Lt);Do=a(nn,"Inference"),nn.forEach(r),Ir.forEach(r),wr=h(t),G=n(t,"P",{});var $t=i(G);qo=a($t,"TrOCR\u2019s "),Dt=n($t,"CODE",{});var ln=i(Dt);Ao=a(ln,"VisionEncoderDecoder"),ln.forEach(r),Io=a($t,` model accepts images as input and makes use of
`),lt=n($t,"A",{href:!0});var dn=i(lt);Vo=a(dn,"generate()"),dn.forEach(r),No=a($t," to autoregressively generate text given the input image."),$t.forEach(r),kr=h(t),v=n(t,"P",{});var x=i(v);So=a(x,"The ["),qt=n(x,"CODE",{});var cn=i(qt);Wo=a(cn,"ViTFeatureExtractor"),cn.forEach(r),Bo=a(x,"/"),At=n(x,"CODE",{});var hn=i(At);Ho=a(hn,"DeiTFeatureExtractor"),hn.forEach(r),Go=a(x,`] class is responsible for preprocessing the input image and
[`),It=n(x,"CODE",{});var mn=i(It);Xo=a(mn,"RobertaTokenizer"),mn.forEach(r),Uo=a(x,"/"),Vt=n(x,"CODE",{});var pn=i(Vt);Zo=a(pn,"XLMRobertaTokenizer"),pn.forEach(r),Jo=a(x,`] decodes the generated target tokens to the target string. The
`),dt=n(x,"A",{href:!0});var fn=i(dt);Yo=a(fn,"TrOCRProcessor"),fn.forEach(r),Ko=a(x," wraps ["),Nt=n(x,"CODE",{});var un=i(Nt);Qo=a(un,"ViTFeatureExtractor"),un.forEach(r),ea=a(x,"/"),St=n(x,"CODE",{});var gn=i(St);ta=a(gn,"DeiTFeatureExtractor"),gn.forEach(r),ra=a(x,"] and ["),Wt=n(x,"CODE",{});var _n=i(Wt);oa=a(_n,"RobertaTokenizer"),_n.forEach(r),aa=a(x,"/"),Bt=n(x,"CODE",{});var vn=i(Bt);sa=a(vn,"XLMRobertaTokenizer"),vn.forEach(r),na=a(x,`]
into a single instance to both extract the input features and decode the predicted token ids.`),x.forEach(r),$r=h(t),ct=n(t,"UL",{});var Tn=i(ct);Ht=n(Tn,"LI",{});var bn=i(Ht);ia=a(bn,"Step-by-step Optical Character Recognition (OCR)"),bn.forEach(r),Tn.forEach(r),Or=h(t),w(Le.$$.fragment,t),Er=h(t),ne=n(t,"P",{});var Vr=i(ne);la=a(Vr,"See the "),De=n(Vr,"A",{href:!0,rel:!0});var Cn=i(De);da=a(Cn,"model hub"),Cn.forEach(r),ca=a(Vr," to look for TrOCR checkpoints."),Vr.forEach(r),yr=h(t),J=n(t,"H2",{class:!0});var Nr=i(J);ie=n(Nr,"A",{id:!0,class:!0,href:!0});var wn=i(ie);Gt=n(wn,"SPAN",{});var kn=i(Gt);w(qe.$$.fragment,kn),kn.forEach(r),wn.forEach(r),ha=h(Nr),Xt=n(Nr,"SPAN",{});var $n=i(Xt);ma=a($n,"TrOCRConfig"),$n.forEach(r),Nr.forEach(r),Rr=h(t),q=n(t,"DIV",{class:!0});var Te=i(q);w(Ae.$$.fragment,Te),pa=h(Te),Y=n(Te,"P",{});var Ot=i(Y);fa=a(Ot,"This is the configuration class to store the configuration of a "),ht=n(Ot,"A",{href:!0});var On=i(ht);ua=a(On,"TrOCRForCausalLM"),On.forEach(r),ga=a(Ot,`. It is used to instantiate an
TrOCR model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the TrOCR
`),Ie=n(Ot,"A",{href:!0,rel:!0});var En=i(Ie);_a=a(En,"microsoft/trocr-base-handwritten"),En.forEach(r),va=a(Ot," architecture."),Ot.forEach(r),Ta=h(Te),K=n(Te,"P",{});var Et=i(K);ba=a(Et,"Configuration objects inherit from "),mt=n(Et,"A",{href:!0});var yn=i(mt);Ca=a(yn,"PretrainedConfig"),yn.forEach(r),wa=a(Et,` and can be used to control the model outputs. Read the
documentation from `),pt=n(Et,"A",{href:!0});var Rn=i(pt);ka=a(Rn,"PretrainedConfig"),Rn.forEach(r),$a=a(Et," for more information."),Et.forEach(r),Oa=h(Te),w(le.$$.fragment,Te),Te.forEach(r),xr=h(t),Q=n(t,"H2",{class:!0});var Sr=i(Q);de=n(Sr,"A",{id:!0,class:!0,href:!0});var xn=i(de);Ut=n(xn,"SPAN",{});var Pn=i(Ut);w(Ve.$$.fragment,Pn),Pn.forEach(r),xn.forEach(r),Ea=h(Sr),Zt=n(Sr,"SPAN",{});var Mn=i(Zt);ya=a(Mn,"TrOCRProcessor"),Mn.forEach(r),Sr.forEach(r),Pr=h(t),y=n(t,"DIV",{class:!0});var M=i(y);w(Ne.$$.fragment,M),Ra=h(M),Jt=n(M,"P",{});var jn=i(Jt);xa=a(jn,"Constructs a TrOCR processor which wraps a vision feature extractor and a TrOCR tokenizer into a single processor."),jn.forEach(r),Pa=h(M),P=n(M,"P",{});var F=i(P);ft=n(F,"A",{href:!0});var zn=i(ft);Ma=a(zn,"TrOCRProcessor"),zn.forEach(r),ja=a(F," offers all the functionalities of ["),Yt=n(F,"CODE",{});var Fn=i(Yt);za=a(Fn,"ViTFeatureExtractor"),Fn.forEach(r),Fa=a(F,"/"),Kt=n(F,"CODE",{});var Ln=i(Kt);La=a(Ln,"DeiTFeatureExtractor"),Ln.forEach(r),Da=a(F,`] and
[`),Qt=n(F,"CODE",{});var Dn=i(Qt);qa=a(Dn,"RobertaTokenizer"),Dn.forEach(r),Aa=a(F,"/"),er=n(F,"CODE",{});var qn=i(er);Ia=a(qn,"XLMRobertaTokenizer"),qn.forEach(r),Va=a(F,"]. See the "),Se=n(F,"A",{href:!0});var qs=i(Se);tr=n(qs,"STRONG",{});var An=i(tr);Na=a(An,"call"),An.forEach(r),Sa=a(qs,"()"),qs.forEach(r),Wa=a(F," and "),ut=n(F,"A",{href:!0});var In=i(ut);Ba=a(In,"decode()"),In.forEach(r),Ha=a(F,` for
more information.`),F.forEach(r),Ga=h(M),ce=n(M,"DIV",{class:!0});var Wr=i(ce);w(We.$$.fragment,Wr),Xa=h(Wr),V=n(Wr,"P",{});var be=i(V);Ua=a(be,`When used in normal mode, this method forwards all its arguments to AutoFeatureExtractor\u2019s
`),rr=n(be,"CODE",{});var Vn=i(rr);Za=a(Vn,"__call__()"),Vn.forEach(r),Ja=a(be,` and returns its output. If used in the context
`),or=n(be,"CODE",{});var Nn=i(or);Ya=a(Nn,"as_target_processor()"),Nn.forEach(r),Ka=a(be,` this method forwards all its arguments to TrOCRTokenizer\u2019s
`),ar=n(be,"CODE",{});var Sn=i(ar);Qa=a(Sn,"__call__"),Sn.forEach(r),es=a(be,". Please refer to the doctsring of the above two methods for more information."),be.forEach(r),Wr.forEach(r),ts=h(M),X=n(M,"DIV",{class:!0});var yt=i(X);w(Be.$$.fragment,yt),rs=h(yt),sr=n(yt,"P",{});var Wn=i(sr);os=a(Wn,"Instantiate a processor associated with a pretrained model."),Wn.forEach(r),as=h(yt),w(he.$$.fragment,yt),yt.forEach(r),ss=h(M),U=n(M,"DIV",{class:!0});var Rt=i(U);w(He.$$.fragment,Rt),ns=h(Rt),Ge=n(Rt,"P",{});var Br=i(Ge);is=a(Br,`Saves the attributes of this processor (feature extractor, tokenizer\u2026) in the specified directory so that it
can be reloaded using the `),gt=n(Br,"A",{href:!0});var Bn=i(gt);ls=a(Bn,"from_pretrained()"),Bn.forEach(r),ds=a(Br," method."),Br.forEach(r),cs=h(Rt),w(me.$$.fragment,Rt),Rt.forEach(r),hs=h(M),pe=n(M,"DIV",{class:!0});var Hr=i(pe);w(Xe.$$.fragment,Hr),ms=h(Hr),Ue=n(Hr,"P",{});var Gr=i(Ue);ps=a(Gr,"This method forwards all its arguments to TrOCRTokenizer\u2019s "),_t=n(Gr,"A",{href:!0});var Hn=i(_t);fs=a(Hn,"batch_decode()"),Hn.forEach(r),us=a(Gr,`. Please refer
to the docstring of this method for more information.`),Gr.forEach(r),Hr.forEach(r),gs=h(M),fe=n(M,"DIV",{class:!0});var Xr=i(fe);w(Ze.$$.fragment,Xr),_s=h(Xr),Je=n(Xr,"P",{});var Ur=i(Je);vs=a(Ur,"This method forwards all its arguments to TrOCRTokenizer\u2019s "),vt=n(Ur,"A",{href:!0});var Gn=i(vt);Ts=a(Gn,"decode()"),Gn.forEach(r),bs=a(Ur,`. Please refer to the
docstring of this method for more information.`),Ur.forEach(r),Xr.forEach(r),M.forEach(r),Mr=h(t),ee=n(t,"H2",{class:!0});var Zr=i(ee);ue=n(Zr,"A",{id:!0,class:!0,href:!0});var Xn=i(ue);nr=n(Xn,"SPAN",{});var Un=i(nr);w(Ye.$$.fragment,Un),Un.forEach(r),Xn.forEach(r),Cs=h(Zr),ir=n(Zr,"SPAN",{});var Zn=i(ir);ws=a(Zn,"TrOCRForCausalLM"),Zn.forEach(r),Zr.forEach(r),jr=h(t),A=n(t,"DIV",{class:!0});var Ce=i(A);w(Ke.$$.fragment,Ce),ks=h(Ce),N=n(Ce,"P",{});var we=i(N);$s=a(we,"The TrOCR Decoder with a language modeling head. Can be used as the decoder part of "),Tt=n(we,"A",{href:!0});var Jn=i(Tt);Os=a(Jn,"EncoderDecoderModel"),Jn.forEach(r),Es=a(we," and "),lr=n(we,"CODE",{});var Yn=i(lr);ys=a(Yn,"VisionEncoderDecoder"),Yn.forEach(r),Rs=a(we,`.
This model inherits from `),bt=n(we,"A",{href:!0});var Kn=i(bt);xs=a(Kn,"PreTrainedModel"),Kn.forEach(r),Ps=a(we,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),we.forEach(r),Ms=h(Ce),Qe=n(Ce,"P",{});var Jr=i(Qe);js=a(Jr,"This model is also a PyTorch "),et=n(Jr,"A",{href:!0,rel:!0});var Qn=i(et);zs=a(Qn,"torch.nn.Module"),Qn.forEach(r),Fs=a(Jr,` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Jr.forEach(r),Ls=h(Ce),ge=n(Ce,"DIV",{class:!0});var Yr=i(ge);w(tt.$$.fragment,Yr),Ds=h(Yr),w(_e.$$.fragment,Yr),Yr.forEach(r),Ce.forEach(r),this.h()},h(){l(f,"name","hf:doc:metadata"),l(f,"content",JSON.stringify(fi)),l(g,"id","trocr"),l(g,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),l(g,"href","#trocr"),l(u,"class","relative group"),l(L,"id","overview"),l(L,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),l(L,"href","#overview"),l(z,"class","relative group"),l($e,"href","https://arxiv.org/abs/2109.10282"),l($e,"rel","nofollow"),l(Oe,"href","https://en.wikipedia.org/wiki/Optical_character_recognition"),l(Oe,"rel","nofollow"),ii(re.src,Is="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/trocr_architecture.jpg")||l(re,"src",Is),l(re,"alt","drawing"),l(re,"width","600"),l(st,"href","https://arxiv.org/abs/2109.10282"),l(Ee,"href","https://huggingface.co/nielsr"),l(Ee,"rel","nofollow"),l(ye,"href","https://github.com/microsoft/unilm/tree/6f60612e7cc86a2a1ae85c47231507a587ab4e01/trocr"),l(ye,"rel","nofollow"),l(xe,"href","https://github.com/NielsRogge/Transformers-Tutorials/tree/master/TrOCR"),l(xe,"rel","nofollow"),l(Pe,"href","https://paperswithcode.com/dataset/sroie"),l(Pe,"rel","nofollow"),l(Me,"href","https://fki.tic.heia-fr.ch/databases/iam-handwriting-database%3E"),l(Me,"rel","nofollow"),l(je,"href","https://huggingface.co/models?other=trocr%3E"),l(je,"rel","nofollow"),l(it,"href","vision-encoder-decoder"),l(se,"id","inference"),l(se,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),l(se,"href","#inference"),l(Z,"class","relative group"),l(lt,"href","/docs/transformers/pr_18429/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate"),l(dt,"href","/docs/transformers/pr_18429/en/model_doc/trocr#transformers.TrOCRProcessor"),l(De,"href","https://huggingface.co/models?filter=trocr"),l(De,"rel","nofollow"),l(ie,"id","transformers.TrOCRConfig"),l(ie,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),l(ie,"href","#transformers.TrOCRConfig"),l(J,"class","relative group"),l(ht,"href","/docs/transformers/pr_18429/en/model_doc/trocr#transformers.TrOCRForCausalLM"),l(Ie,"href","https://huggingface.co/microsoft/trocr-base-handwritten"),l(Ie,"rel","nofollow"),l(mt,"href","/docs/transformers/pr_18429/en/main_classes/configuration#transformers.PretrainedConfig"),l(pt,"href","/docs/transformers/pr_18429/en/main_classes/configuration#transformers.PretrainedConfig"),l(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),l(de,"id","transformers.TrOCRProcessor"),l(de,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),l(de,"href","#transformers.TrOCRProcessor"),l(Q,"class","relative group"),l(ft,"href","/docs/transformers/pr_18429/en/model_doc/trocr#transformers.TrOCRProcessor"),l(Se,"href","/docs/transformers/pr_18429/en/model_doc/trocr#transformers.TrOCRProcessor.__call__"),l(ut,"href","/docs/transformers/pr_18429/en/model_doc/trocr#transformers.TrOCRProcessor.decode"),l(ce,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),l(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),l(gt,"href","/docs/transformers/pr_18429/en/main_classes/processors#transformers.ProcessorMixin.from_pretrained"),l(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),l(_t,"href","/docs/transformers/pr_18429/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.batch_decode"),l(pe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),l(vt,"href","/docs/transformers/pr_18429/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.decode"),l(fe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),l(y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),l(ue,"id","transformers.TrOCRForCausalLM"),l(ue,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),l(ue,"href","#transformers.TrOCRForCausalLM"),l(ee,"class","relative group"),l(Tt,"href","/docs/transformers/pr_18429/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel"),l(bt,"href","/docs/transformers/pr_18429/en/main_classes/model#transformers.PreTrainedModel"),l(et,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),l(et,"rel","nofollow"),l(ge,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),l(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(t,d){e(document.head,f),p(t,R,d),p(t,u,d),e(u,g),e(g,T),k(m,T,null),e(u,_),e(u,D),e(D,j),p(t,b,d),p(t,z,d),e(z,L),e(L,Pt),k(ke,Pt,null),e(z,Kr),e(z,Mt),e(Mt,Qr),p(t,mr,d),p(t,W,d),e(W,eo),e(W,$e),e($e,to),e(W,ro),e(W,Oe),e(Oe,oo),e(W,ao),p(t,pr,d),p(t,ot,d),e(ot,so),p(t,fr,d),p(t,at,d),e(at,jt),e(jt,no),p(t,ur,d),p(t,re,d),p(t,gr,d),p(t,oe,d),e(oe,io),e(oe,st),e(st,lo),e(oe,co),p(t,_r,d),p(t,ae,d),e(ae,ho),e(ae,zt),e(zt,mo),e(ae,po),p(t,vr,d),p(t,B,d),e(B,fo),e(B,Ee),e(Ee,uo),e(B,go),e(B,ye),e(ye,_o),e(B,vo),p(t,Tr,d),p(t,nt,d),e(nt,To),p(t,br,d),p(t,H,d),e(H,Re),e(Re,bo),e(Re,xe),e(xe,Co),e(Re,wo),e(H,ko),e(H,I),e(I,$o),e(I,Pe),e(Pe,Oo),e(I,Eo),e(I,Me),e(Me,yo),e(I,Ro),e(I,je),e(je,xo),e(I,Po),e(H,Mo),e(H,ze),e(ze,jo),e(ze,it),e(it,zo),e(ze,Fo),p(t,Cr,d),p(t,Z,d),e(Z,se),e(se,Ft),k(Fe,Ft,null),e(Z,Lo),e(Z,Lt),e(Lt,Do),p(t,wr,d),p(t,G,d),e(G,qo),e(G,Dt),e(Dt,Ao),e(G,Io),e(G,lt),e(lt,Vo),e(G,No),p(t,kr,d),p(t,v,d),e(v,So),e(v,qt),e(qt,Wo),e(v,Bo),e(v,At),e(At,Ho),e(v,Go),e(v,It),e(It,Xo),e(v,Uo),e(v,Vt),e(Vt,Zo),e(v,Jo),e(v,dt),e(dt,Yo),e(v,Ko),e(v,Nt),e(Nt,Qo),e(v,ea),e(v,St),e(St,ta),e(v,ra),e(v,Wt),e(Wt,oa),e(v,aa),e(v,Bt),e(Bt,sa),e(v,na),p(t,$r,d),p(t,ct,d),e(ct,Ht),e(Ht,ia),p(t,Or,d),k(Le,t,d),p(t,Er,d),p(t,ne,d),e(ne,la),e(ne,De),e(De,da),e(ne,ca),p(t,yr,d),p(t,J,d),e(J,ie),e(ie,Gt),k(qe,Gt,null),e(J,ha),e(J,Xt),e(Xt,ma),p(t,Rr,d),p(t,q,d),k(Ae,q,null),e(q,pa),e(q,Y),e(Y,fa),e(Y,ht),e(ht,ua),e(Y,ga),e(Y,Ie),e(Ie,_a),e(Y,va),e(q,Ta),e(q,K),e(K,ba),e(K,mt),e(mt,Ca),e(K,wa),e(K,pt),e(pt,ka),e(K,$a),e(q,Oa),k(le,q,null),p(t,xr,d),p(t,Q,d),e(Q,de),e(de,Ut),k(Ve,Ut,null),e(Q,Ea),e(Q,Zt),e(Zt,ya),p(t,Pr,d),p(t,y,d),k(Ne,y,null),e(y,Ra),e(y,Jt),e(Jt,xa),e(y,Pa),e(y,P),e(P,ft),e(ft,Ma),e(P,ja),e(P,Yt),e(Yt,za),e(P,Fa),e(P,Kt),e(Kt,La),e(P,Da),e(P,Qt),e(Qt,qa),e(P,Aa),e(P,er),e(er,Ia),e(P,Va),e(P,Se),e(Se,tr),e(tr,Na),e(Se,Sa),e(P,Wa),e(P,ut),e(ut,Ba),e(P,Ha),e(y,Ga),e(y,ce),k(We,ce,null),e(ce,Xa),e(ce,V),e(V,Ua),e(V,rr),e(rr,Za),e(V,Ja),e(V,or),e(or,Ya),e(V,Ka),e(V,ar),e(ar,Qa),e(V,es),e(y,ts),e(y,X),k(Be,X,null),e(X,rs),e(X,sr),e(sr,os),e(X,as),k(he,X,null),e(y,ss),e(y,U),k(He,U,null),e(U,ns),e(U,Ge),e(Ge,is),e(Ge,gt),e(gt,ls),e(Ge,ds),e(U,cs),k(me,U,null),e(y,hs),e(y,pe),k(Xe,pe,null),e(pe,ms),e(pe,Ue),e(Ue,ps),e(Ue,_t),e(_t,fs),e(Ue,us),e(y,gs),e(y,fe),k(Ze,fe,null),e(fe,_s),e(fe,Je),e(Je,vs),e(Je,vt),e(vt,Ts),e(Je,bs),p(t,Mr,d),p(t,ee,d),e(ee,ue),e(ue,nr),k(Ye,nr,null),e(ee,Cs),e(ee,ir),e(ir,ws),p(t,jr,d),p(t,A,d),k(Ke,A,null),e(A,ks),e(A,N),e(N,$s),e(N,Tt),e(Tt,Os),e(N,Es),e(N,lr),e(lr,ys),e(N,Rs),e(N,bt),e(bt,xs),e(N,Ps),e(A,Ms),e(A,Qe),e(Qe,js),e(Qe,et),e(et,zs),e(Qe,Fs),e(A,Ls),e(A,ge),k(tt,ge,null),e(ge,Ds),k(_e,ge,null),zr=!0},p(t,[d]){const rt={};d&2&&(rt.$$scope={dirty:d,ctx:t}),le.$set(rt);const dr={};d&2&&(dr.$$scope={dirty:d,ctx:t}),he.$set(dr);const cr={};d&2&&(cr.$$scope={dirty:d,ctx:t}),me.$set(cr);const hr={};d&2&&(hr.$$scope={dirty:d,ctx:t}),_e.$set(hr)},i(t){zr||($(m.$$.fragment,t),$(ke.$$.fragment,t),$(Fe.$$.fragment,t),$(Le.$$.fragment,t),$(qe.$$.fragment,t),$(Ae.$$.fragment,t),$(le.$$.fragment,t),$(Ve.$$.fragment,t),$(Ne.$$.fragment,t),$(We.$$.fragment,t),$(Be.$$.fragment,t),$(he.$$.fragment,t),$(He.$$.fragment,t),$(me.$$.fragment,t),$(Xe.$$.fragment,t),$(Ze.$$.fragment,t),$(Ye.$$.fragment,t),$(Ke.$$.fragment,t),$(tt.$$.fragment,t),$(_e.$$.fragment,t),zr=!0)},o(t){O(m.$$.fragment,t),O(ke.$$.fragment,t),O(Fe.$$.fragment,t),O(Le.$$.fragment,t),O(qe.$$.fragment,t),O(Ae.$$.fragment,t),O(le.$$.fragment,t),O(Ve.$$.fragment,t),O(Ne.$$.fragment,t),O(We.$$.fragment,t),O(Be.$$.fragment,t),O(he.$$.fragment,t),O(He.$$.fragment,t),O(me.$$.fragment,t),O(Xe.$$.fragment,t),O(Ze.$$.fragment,t),O(Ye.$$.fragment,t),O(Ke.$$.fragment,t),O(tt.$$.fragment,t),O(_e.$$.fragment,t),zr=!1},d(t){r(f),t&&r(R),t&&r(u),E(m),t&&r(b),t&&r(z),E(ke),t&&r(mr),t&&r(W),t&&r(pr),t&&r(ot),t&&r(fr),t&&r(at),t&&r(ur),t&&r(re),t&&r(gr),t&&r(oe),t&&r(_r),t&&r(ae),t&&r(vr),t&&r(B),t&&r(Tr),t&&r(nt),t&&r(br),t&&r(H),t&&r(Cr),t&&r(Z),E(Fe),t&&r(wr),t&&r(G),t&&r(kr),t&&r(v),t&&r($r),t&&r(ct),t&&r(Or),E(Le,t),t&&r(Er),t&&r(ne),t&&r(yr),t&&r(J),E(qe),t&&r(Rr),t&&r(q),E(Ae),E(le),t&&r(xr),t&&r(Q),E(Ve),t&&r(Pr),t&&r(y),E(Ne),E(We),E(Be),E(he),E(He),E(me),E(Xe),E(Ze),t&&r(Mr),t&&r(ee),E(Ye),t&&r(jr),t&&r(A),E(Ke),E(tt),E(_e)}}}const fi={local:"trocr",sections:[{local:"overview",title:"Overview"},{local:"inference",title:"Inference"},{local:"transformers.TrOCRConfig",title:"TrOCRConfig"},{local:"transformers.TrOCRProcessor",title:"TrOCRProcessor"},{local:"transformers.TrOCRForCausalLM",title:"TrOCRForCausalLM"}],title:"TrOCR"};function ui(S){return li(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class wi extends oi{constructor(f){super();ai(this,f,ui,pi,si,{})}}export{wi as default,fi as metadata};
