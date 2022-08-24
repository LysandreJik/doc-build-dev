import{S as Op,i as Gp,s as Up,e as r,k as l,w as b,t as n,M as Wp,c as a,d as t,m as p,a as i,x as k,h as s,b as c,G as e,g as f,y,q as w,o as S,B as $,v as Vp,L as Ys}from"../../chunks/vendor-hf-doc-builder.js";import{T as eo}from"../../chunks/Tip-hf-doc-builder.js";import{D as L}from"../../chunks/Docstring-hf-doc-builder.js";import{C as dn}from"../../chunks/CodeBlock-hf-doc-builder.js";import{I as We}from"../../chunks/IconCopyLink-hf-doc-builder.js";import{E as Js}from"../../chunks/ExampleCodeBlock-hf-doc-builder.js";function Rp(C){let h,x,_,m,v;return m=new dn({props:{code:`from transformers import Speech2TextModel, Speech2TextConfig

# Initializing a Speech2Text s2t_transformer_s style configuration
configuration = Speech2TextConfig()

# Initializing a model from the s2t_transformer_s style configuration
model = Speech2TextModel(configuration)

# Accessing the model configuration
configuration = model.config`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Speech2TextModel, Speech2TextConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Speech2Text s2t_transformer_s style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = Speech2TextConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the s2t_transformer_s style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Speech2TextModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`}}),{c(){h=r("p"),x=n("Example:"),_=l(),b(m.$$.fragment)},l(d){h=a(d,"P",{});var g=i(h);x=s(g,"Example:"),g.forEach(t),_=p(d),k(m.$$.fragment,d)},m(d,g){f(d,h,g),e(h,x),f(d,_,g),y(m,d,g),v=!0},p:Ys,i(d){v||(w(m.$$.fragment,d),v=!0)},o(d){S(m.$$.fragment,d),v=!1},d(d){d&&t(h),d&&t(_),$(m,d)}}}function Hp(C){let h,x,_,m,v,d,g,E;return{c(){h=r("p"),x=n(`This class method is simply calling the feature extractor
`),_=r("a"),m=n("from_pretrained()"),v=n(` and the tokenizer
`),d=r("code"),g=n("from_pretrained"),E=n(` methods. Please refer to the docstrings of the
methods above for more information.`),this.h()},l(A){h=a(A,"P",{});var F=i(h);x=s(F,`This class method is simply calling the feature extractor
`),_=a(F,"A",{href:!0});var q=i(_);m=s(q,"from_pretrained()"),q.forEach(t),v=s(F,` and the tokenizer
`),d=a(F,"CODE",{});var D=i(d);g=s(D,"from_pretrained"),D.forEach(t),E=s(F,` methods. Please refer to the docstrings of the
methods above for more information.`),F.forEach(t),this.h()},h(){c(_,"href","/docs/transformers/pr_18590/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained")},m(A,F){f(A,h,F),e(h,x),e(h,_),e(_,m),e(h,v),e(h,d),e(d,g),e(h,E)},d(A){A&&t(h)}}}function Bp(C){let h,x,_,m,v,d,g,E;return{c(){h=r("p"),x=n("This class method is simply calling "),_=r("a"),m=n("save_pretrained()"),v=n(` and
`),d=r("code"),g=n("save_pretrained"),E=n(`. Please refer to the docstrings of the methods
above for more information.`),this.h()},l(A){h=a(A,"P",{});var F=i(h);x=s(F,"This class method is simply calling "),_=a(F,"A",{href:!0});var q=i(_);m=s(q,"save_pretrained()"),q.forEach(t),v=s(F,` and
`),d=a(F,"CODE",{});var D=i(d);g=s(D,"save_pretrained"),D.forEach(t),E=s(F,`. Please refer to the docstrings of the methods
above for more information.`),F.forEach(t),this.h()},h(){c(_,"href","/docs/transformers/pr_18590/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained")},m(A,F){f(A,h,F),e(h,x),e(h,_),e(_,m),e(h,v),e(h,d),e(d,g),e(h,E)},d(A){A&&t(h)}}}function Kp(C){let h,x,_,m,v;return{c(){h=r("p"),x=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),_=r("code"),m=n("Module"),v=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(d){h=a(d,"P",{});var g=i(h);x=s(g,"Although the recipe for forward pass needs to be defined within this function, one should call the "),_=a(g,"CODE",{});var E=i(_);m=s(E,"Module"),E.forEach(t),v=s(g,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),g.forEach(t)},m(d,g){f(d,h,g),e(h,x),e(h,_),e(_,m),e(h,v)},d(d){d&&t(h)}}}function Jp(C){let h,x,_,m,v;return m=new dn({props:{code:`import torch
from transformers import Speech2TextModel, Speech2TextFeatureExtractor
from datasets import load_dataset

model = Speech2TextModel.from_pretrained("facebook/s2t-small-librispeech-asr")
feature_extractor = Speech2TextFeatureExtractor.from_pretrained("facebook/s2t-small-librispeech-asr")
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
inputs = feature_extractor(
    ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt"
)
input_features = inputs.input_features
decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
list(last_hidden_state.shape)`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Speech2TextModel, Speech2TextFeatureExtractor
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset

<span class="hljs-meta">&gt;&gt;&gt; </span>model = Speech2TextModel.from_pretrained(<span class="hljs-string">&quot;facebook/s2t-small-librispeech-asr&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>feature_extractor = Speech2TextFeatureExtractor.from_pretrained(<span class="hljs-string">&quot;facebook/s2t-small-librispeech-asr&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>ds = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/librispeech_asr_dummy&quot;</span>, <span class="hljs-string">&quot;clean&quot;</span>, split=<span class="hljs-string">&quot;validation&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = feature_extractor(
<span class="hljs-meta">... </span>    ds[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;array&quot;</span>], sampling_rate=ds[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;sampling_rate&quot;</span>], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>input_features = inputs.input_features
<span class="hljs-meta">&gt;&gt;&gt; </span>decoder_input_ids = torch.tensor([[<span class="hljs-number">1</span>, <span class="hljs-number">1</span>]]) * model.config.decoder_start_token_id
<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(last_hidden_state.shape)
[<span class="hljs-number">1</span>, <span class="hljs-number">2</span>, <span class="hljs-number">256</span>]`}}),{c(){h=r("p"),x=n("Example:"),_=l(),b(m.$$.fragment)},l(d){h=a(d,"P",{});var g=i(h);x=s(g,"Example:"),g.forEach(t),_=p(d),k(m.$$.fragment,d)},m(d,g){f(d,h,g),e(h,x),f(d,_,g),y(m,d,g),v=!0},p:Ys,i(d){v||(w(m.$$.fragment,d),v=!0)},o(d){S(m.$$.fragment,d),v=!1},d(d){d&&t(h),d&&t(_),$(m,d)}}}function Yp(C){let h,x,_,m,v;return{c(){h=r("p"),x=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),_=r("code"),m=n("Module"),v=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(d){h=a(d,"P",{});var g=i(h);x=s(g,"Although the recipe for forward pass needs to be defined within this function, one should call the "),_=a(g,"CODE",{});var E=i(_);m=s(E,"Module"),E.forEach(t),v=s(g,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),g.forEach(t)},m(d,g){f(d,h,g),e(h,x),e(h,_),e(_,m),e(h,v)},d(d){d&&t(h)}}}function Xp(C){let h,x,_,m,v;return m=new dn({props:{code:`import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from datasets import load_dataset

model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")


ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

inputs = processor(
    ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt"
)
input_features = inputs.input_features

generated_ids = model.generate(inputs=input_features)

transcription = processor.batch_decode(generated_ids)[0]
transcription`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Speech2TextProcessor, Speech2TextForConditionalGeneration
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset

<span class="hljs-meta">&gt;&gt;&gt; </span>model = Speech2TextForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;facebook/s2t-small-librispeech-asr&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor = Speech2TextProcessor.from_pretrained(<span class="hljs-string">&quot;facebook/s2t-small-librispeech-asr&quot;</span>)


<span class="hljs-meta">&gt;&gt;&gt; </span>ds = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/librispeech_asr_dummy&quot;</span>, <span class="hljs-string">&quot;clean&quot;</span>, split=<span class="hljs-string">&quot;validation&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(
<span class="hljs-meta">... </span>    ds[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;array&quot;</span>], sampling_rate=ds[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;sampling_rate&quot;</span>], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>input_features = inputs.input_features

<span class="hljs-meta">&gt;&gt;&gt; </span>generated_ids = model.generate(inputs=input_features)

<span class="hljs-meta">&gt;&gt;&gt; </span>transcription = processor.batch_decode(generated_ids)[<span class="hljs-number">0</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>transcription
<span class="hljs-string">&#x27;mister quilter is the apostle of the middle classes and we are glad to welcome his gospel&#x27;</span>`}}),{c(){h=r("p"),x=n("Example:"),_=l(),b(m.$$.fragment)},l(d){h=a(d,"P",{});var g=i(h);x=s(g,"Example:"),g.forEach(t),_=p(d),k(m.$$.fragment,d)},m(d,g){f(d,h,g),e(h,x),f(d,_,g),y(m,d,g),v=!0},p:Ys,i(d){v||(w(m.$$.fragment,d),v=!0)},o(d){S(m.$$.fragment,d),v=!1},d(d){d&&t(h),d&&t(_),$(m,d)}}}function Qp(C){let h,x,_,m,v,d,g,E,A,F,q,D,R,ee,Ce,H,Pe,ye,N,B,te,Te,j,M,je,oe,ne,Me,se,re,Ne,O,we,I,Ae,ae,ie,De,de,K,Ie,Q,J;return{c(){h=r("p"),x=n("TF 2.0 models accepts two formats as inputs:"),_=l(),m=r("ul"),v=r("li"),d=n("having all inputs as keyword arguments (like PyTorch models), or"),g=l(),E=r("li"),A=n("having all inputs as a list, tuple or dict in the first positional arguments."),F=l(),q=r("p"),D=n("This second option is useful when using "),R=r("code"),ee=n("tf.keras.Model.fit"),Ce=n(` method which currently requires having all the
tensors in the first argument of the model call function: `),H=r("code"),Pe=n("model(inputs)"),ye=n("."),N=l(),B=r("p"),te=n(`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),Te=l(),j=r("ul"),M=r("li"),je=n("a single Tensor with "),oe=r("code"),ne=n("input_ids"),Me=n(" only and nothing else: "),se=r("code"),re=n("model(input_ids)"),Ne=l(),O=r("li"),we=n(`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),I=r("code"),Ae=n("model([input_ids, attention_mask])"),ae=n(" or "),ie=r("code"),De=n("model([input_ids, attention_mask, token_type_ids])"),de=l(),K=r("li"),Ie=n(`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),Q=r("code"),J=n('model({"input_ids": input_ids, "token_type_ids": token_type_ids})')},l(T){h=a(T,"P",{});var z=i(h);x=s(z,"TF 2.0 models accepts two formats as inputs:"),z.forEach(t),_=p(T),m=a(T,"UL",{});var Y=i(m);v=a(Y,"LI",{});var Ze=i(v);d=s(Ze,"having all inputs as keyword arguments (like PyTorch models), or"),Ze.forEach(t),g=p(Y),E=a(Y,"LI",{});var Le=i(E);A=s(Le,"having all inputs as a list, tuple or dict in the first positional arguments."),Le.forEach(t),Y.forEach(t),F=p(T),q=a(T,"P",{});var X=i(q);D=s(X,"This second option is useful when using "),R=a(X,"CODE",{});var Ve=i(R);ee=s(Ve,"tf.keras.Model.fit"),Ve.forEach(t),Ce=s(X,` method which currently requires having all the
tensors in the first argument of the model call function: `),H=a(X,"CODE",{});var ue=i(H);Pe=s(ue,"model(inputs)"),ue.forEach(t),ye=s(X,"."),X.forEach(t),N=p(T),B=a(T,"P",{});var et=i(B);te=s(et,`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),et.forEach(t),Te=p(T),j=a(T,"UL",{});var U=i(j);M=a(U,"LI",{});var ce=i(M);je=s(ce,"a single Tensor with "),oe=a(ce,"CODE",{});var tt=i(oe);ne=s(tt,"input_ids"),tt.forEach(t),Me=s(ce," only and nothing else: "),se=a(ce,"CODE",{});var Re=i(se);re=s(Re,"model(input_ids)"),Re.forEach(t),ce.forEach(t),Ne=p(U),O=a(U,"LI",{});var P=i(O);we=s(P,`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),I=a(P,"CODE",{});var ot=i(I);Ae=s(ot,"model([input_ids, attention_mask])"),ot.forEach(t),ae=s(P," or "),ie=a(P,"CODE",{});var Se=i(ie);De=s(Se,"model([input_ids, attention_mask, token_type_ids])"),Se.forEach(t),P.forEach(t),de=p(U),K=a(U,"LI",{});var Oe=i(K);Ie=s(Oe,`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),Q=a(Oe,"CODE",{});var nt=i(Q);J=s(nt,'model({"input_ids": input_ids, "token_type_ids": token_type_ids})'),nt.forEach(t),Oe.forEach(t),U.forEach(t)},m(T,z){f(T,h,z),e(h,x),f(T,_,z),f(T,m,z),e(m,v),e(v,d),e(m,g),e(m,E),e(E,A),f(T,F,z),f(T,q,z),e(q,D),e(q,R),e(R,ee),e(q,Ce),e(q,H),e(H,Pe),e(q,ye),f(T,N,z),f(T,B,z),e(B,te),f(T,Te,z),f(T,j,z),e(j,M),e(M,je),e(M,oe),e(oe,ne),e(M,Me),e(M,se),e(se,re),e(j,Ne),e(j,O),e(O,we),e(O,I),e(I,Ae),e(O,ae),e(O,ie),e(ie,De),e(j,de),e(j,K),e(K,Ie),e(K,Q),e(Q,J)},d(T){T&&t(h),T&&t(_),T&&t(m),T&&t(F),T&&t(q),T&&t(N),T&&t(B),T&&t(Te),T&&t(j)}}}function Zp(C){let h,x,_,m,v;return{c(){h=r("p"),x=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),_=r("code"),m=n("Module"),v=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(d){h=a(d,"P",{});var g=i(h);x=s(g,"Although the recipe for forward pass needs to be defined within this function, one should call the "),_=a(g,"CODE",{});var E=i(_);m=s(E,"Module"),E.forEach(t),v=s(g,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),g.forEach(t)},m(d,g){f(d,h,g),e(h,x),e(h,_),e(_,m),e(h,v)},d(d){d&&t(h)}}}function eh(C){let h,x,_,m,v;return m=new dn({props:{code:`from transformers import Speech2TextTokenizer, TFSpeech2TextModel
import tensorflow as tf

tokenizer = Speech2TextTokenizer.from_pretrained("facebook/s2t-small-librispeech-asr")
model = TFSpeech2TextModel.from_pretrained("facebook/s2t-small-librispeech-asr")

inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
outputs = model(inputs)

last_hidden_states = outputs.last_hidden_state`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Speech2TextTokenizer, TFSpeech2TextModel
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> tensorflow <span class="hljs-keyword">as</span> tf

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = Speech2TextTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/s2t-small-librispeech-asr&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFSpeech2TextModel.from_pretrained(<span class="hljs-string">&quot;facebook/s2t-small-librispeech-asr&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;tf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`}}),{c(){h=r("p"),x=n("Example:"),_=l(),b(m.$$.fragment)},l(d){h=a(d,"P",{});var g=i(h);x=s(g,"Example:"),g.forEach(t),_=p(d),k(m.$$.fragment,d)},m(d,g){f(d,h,g),e(h,x),f(d,_,g),y(m,d,g),v=!0},p:Ys,i(d){v||(w(m.$$.fragment,d),v=!0)},o(d){S(m.$$.fragment,d),v=!1},d(d){d&&t(h),d&&t(_),$(m,d)}}}function th(C){let h,x,_,m,v,d,g,E,A,F,q,D,R,ee,Ce,H,Pe,ye,N,B,te,Te,j,M,je,oe,ne,Me,se,re,Ne,O,we,I,Ae,ae,ie,De,de,K,Ie,Q,J;return{c(){h=r("p"),x=n("TF 2.0 models accepts two formats as inputs:"),_=l(),m=r("ul"),v=r("li"),d=n("having all inputs as keyword arguments (like PyTorch models), or"),g=l(),E=r("li"),A=n("having all inputs as a list, tuple or dict in the first positional arguments."),F=l(),q=r("p"),D=n("This second option is useful when using "),R=r("code"),ee=n("tf.keras.Model.fit"),Ce=n(` method which currently requires having all the
tensors in the first argument of the model call function: `),H=r("code"),Pe=n("model(inputs)"),ye=n("."),N=l(),B=r("p"),te=n(`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),Te=l(),j=r("ul"),M=r("li"),je=n("a single Tensor with "),oe=r("code"),ne=n("input_ids"),Me=n(" only and nothing else: "),se=r("code"),re=n("model(input_ids)"),Ne=l(),O=r("li"),we=n(`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),I=r("code"),Ae=n("model([input_ids, attention_mask])"),ae=n(" or "),ie=r("code"),De=n("model([input_ids, attention_mask, token_type_ids])"),de=l(),K=r("li"),Ie=n(`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),Q=r("code"),J=n('model({"input_ids": input_ids, "token_type_ids": token_type_ids})')},l(T){h=a(T,"P",{});var z=i(h);x=s(z,"TF 2.0 models accepts two formats as inputs:"),z.forEach(t),_=p(T),m=a(T,"UL",{});var Y=i(m);v=a(Y,"LI",{});var Ze=i(v);d=s(Ze,"having all inputs as keyword arguments (like PyTorch models), or"),Ze.forEach(t),g=p(Y),E=a(Y,"LI",{});var Le=i(E);A=s(Le,"having all inputs as a list, tuple or dict in the first positional arguments."),Le.forEach(t),Y.forEach(t),F=p(T),q=a(T,"P",{});var X=i(q);D=s(X,"This second option is useful when using "),R=a(X,"CODE",{});var Ve=i(R);ee=s(Ve,"tf.keras.Model.fit"),Ve.forEach(t),Ce=s(X,` method which currently requires having all the
tensors in the first argument of the model call function: `),H=a(X,"CODE",{});var ue=i(H);Pe=s(ue,"model(inputs)"),ue.forEach(t),ye=s(X,"."),X.forEach(t),N=p(T),B=a(T,"P",{});var et=i(B);te=s(et,`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),et.forEach(t),Te=p(T),j=a(T,"UL",{});var U=i(j);M=a(U,"LI",{});var ce=i(M);je=s(ce,"a single Tensor with "),oe=a(ce,"CODE",{});var tt=i(oe);ne=s(tt,"input_ids"),tt.forEach(t),Me=s(ce," only and nothing else: "),se=a(ce,"CODE",{});var Re=i(se);re=s(Re,"model(input_ids)"),Re.forEach(t),ce.forEach(t),Ne=p(U),O=a(U,"LI",{});var P=i(O);we=s(P,`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),I=a(P,"CODE",{});var ot=i(I);Ae=s(ot,"model([input_ids, attention_mask])"),ot.forEach(t),ae=s(P," or "),ie=a(P,"CODE",{});var Se=i(ie);De=s(Se,"model([input_ids, attention_mask, token_type_ids])"),Se.forEach(t),P.forEach(t),de=p(U),K=a(U,"LI",{});var Oe=i(K);Ie=s(Oe,`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),Q=a(Oe,"CODE",{});var nt=i(Q);J=s(nt,'model({"input_ids": input_ids, "token_type_ids": token_type_ids})'),nt.forEach(t),Oe.forEach(t),U.forEach(t)},m(T,z){f(T,h,z),e(h,x),f(T,_,z),f(T,m,z),e(m,v),e(v,d),e(m,g),e(m,E),e(E,A),f(T,F,z),f(T,q,z),e(q,D),e(q,R),e(R,ee),e(q,Ce),e(q,H),e(H,Pe),e(q,ye),f(T,N,z),f(T,B,z),e(B,te),f(T,Te,z),f(T,j,z),e(j,M),e(M,je),e(M,oe),e(oe,ne),e(M,Me),e(M,se),e(se,re),e(j,Ne),e(j,O),e(O,we),e(O,I),e(I,Ae),e(O,ae),e(O,ie),e(ie,De),e(j,de),e(j,K),e(K,Ie),e(K,Q),e(Q,J)},d(T){T&&t(h),T&&t(_),T&&t(m),T&&t(F),T&&t(q),T&&t(N),T&&t(B),T&&t(Te),T&&t(j)}}}function oh(C){let h,x,_,m,v;return{c(){h=r("p"),x=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),_=r("code"),m=n("Module"),v=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(d){h=a(d,"P",{});var g=i(h);x=s(g,"Although the recipe for forward pass needs to be defined within this function, one should call the "),_=a(g,"CODE",{});var E=i(_);m=s(E,"Module"),E.forEach(t),v=s(g,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),g.forEach(t)},m(d,g){f(d,h,g),e(h,x),e(h,_),e(_,m),e(h,v)},d(d){d&&t(h)}}}function nh(C){let h,x,_,m,v;return m=new dn({props:{code:`import tensorflow as tf
from transformers import Speech2TextProcessor, TFSpeech2TextForConditionalGeneration
from datasets import load_dataset
import soundfile as sf

model = TFSpeech2TextForConditionalGeneration.from_pretrained(
    "facebook/s2t-small-librispeech-asr", from_pt=True
)
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")


def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch


ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.map(map_to_array)
ds.set_format(type="tf")

input_features = processor(
    ds["speech"][0], sampling_rate=16000, return_tensors="tf"
).input_features  # Batch size 1
generated_ids = model.generate(input_features)

transcription = processor.batch_decode(generated_ids)`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> tensorflow <span class="hljs-keyword">as</span> tf
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Speech2TextProcessor, TFSpeech2TextForConditionalGeneration
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> soundfile <span class="hljs-keyword">as</span> sf

<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFSpeech2TextForConditionalGeneration.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;facebook/s2t-small-librispeech-asr&quot;</span>, from_pt=<span class="hljs-literal">True</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor = Speech2TextProcessor.from_pretrained(<span class="hljs-string">&quot;facebook/s2t-small-librispeech-asr&quot;</span>)


<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">def</span> <span class="hljs-title function_">map_to_array</span>(<span class="hljs-params">batch</span>):
<span class="hljs-meta">... </span>    speech, _ = sf.read(batch[<span class="hljs-string">&quot;file&quot;</span>])
<span class="hljs-meta">... </span>    batch[<span class="hljs-string">&quot;speech&quot;</span>] = speech
<span class="hljs-meta">... </span>    <span class="hljs-keyword">return</span> batch


<span class="hljs-meta">&gt;&gt;&gt; </span>ds = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/librispeech_asr_dummy&quot;</span>, <span class="hljs-string">&quot;clean&quot;</span>, split=<span class="hljs-string">&quot;validation&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>ds = ds.<span class="hljs-built_in">map</span>(map_to_array)
<span class="hljs-meta">&gt;&gt;&gt; </span>ds.set_format(<span class="hljs-built_in">type</span>=<span class="hljs-string">&quot;tf&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>input_features = processor(
<span class="hljs-meta">... </span>    ds[<span class="hljs-string">&quot;speech&quot;</span>][<span class="hljs-number">0</span>], sampling_rate=<span class="hljs-number">16000</span>, return_tensors=<span class="hljs-string">&quot;tf&quot;</span>
<span class="hljs-meta">... </span>).input_features  <span class="hljs-comment"># Batch size 1</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generated_ids = model.generate(input_features)

<span class="hljs-meta">&gt;&gt;&gt; </span>transcription = processor.batch_decode(generated_ids)`}}),{c(){h=r("p"),x=n("Example:"),_=l(),b(m.$$.fragment)},l(d){h=a(d,"P",{});var g=i(h);x=s(g,"Example:"),g.forEach(t),_=p(d),k(m.$$.fragment,d)},m(d,g){f(d,h,g),e(h,x),f(d,_,g),y(m,d,g),v=!0},p:Ys,i(d){v||(w(m.$$.fragment,d),v=!0)},o(d){S(m.$$.fragment,d),v=!1},d(d){d&&t(h),d&&t(_),$(m,d)}}}function sh(C){let h,x,_,m,v,d,g,E,A,F,q,D,R,ee,Ce,H,Pe,ye,N,B,te,Te,j,M,je,oe,ne,Me,se,re,Ne,O,we,I,Ae,ae,ie,De,de,K,Ie,Q,J,T,z,Y,Ze,Le,X,Ve,ue,et,U,ce,tt,Re,P,ot,Se,Oe,nt,cn,Qr,Zr,ln,ea,ta,pn,oa,na,Xs,W,sa,Bn,ra,aa,Kn,ia,da,Jn,ca,la,Yn,pa,ha,Xn,ma,fa,to,ua,_a,Qn,ga,Qs,hn,Zn,Ta,Zs,oo,er,mn,no,es,va,xa,le,ba,ts,ka,ya,os,wa,Sa,ns,$a,Ea,ss,qa,Fa,rs,za,Ca,tr,so,or,Tt,Pa,ro,ja,Ma,nr,st,vt,as,ao,Na,is,Aa,sr,ve,io,Da,rt,Ia,fn,La,Oa,co,Ga,Ua,Wa,at,Va,un,Ra,Ha,_n,Ba,Ka,Ja,xt,rr,it,bt,ds,lo,Ya,cs,Xa,ar,V,po,Qa,ls,Za,ei,ho,ti,gn,oi,ni,si,kt,mo,ri,ps,ai,ii,yt,fo,di,uo,ci,hs,li,pi,hi,He,_o,mi,Tn,fi,vn,ui,_i,ms,gi,Ti,xn,go,ir,dt,wt,fs,To,vi,us,xi,dr,pe,vo,bi,_s,ki,yi,xo,wi,bn,Si,$i,Ei,gs,qi,Fi,St,bo,zi,Ts,Ci,cr,ct,$t,vs,ko,Pi,xs,ji,lr,G,yo,Mi,bs,Ni,Ai,_e,kn,Di,Ii,yn,Li,Oi,wn,Gi,Ui,wo,ks,Wi,Vi,Ri,Sn,Hi,Bi,Ki,Et,So,Ji,Ge,Yi,$o,ys,Xi,Qi,Zi,ws,ed,td,Eo,Ss,od,nd,sd,rd,Be,qo,ad,$s,id,dd,qt,cd,Ke,Fo,ld,zo,pd,$n,hd,md,fd,Ft,ud,zt,Co,_d,Po,gd,En,Td,vd,xd,Ct,jo,bd,Mo,kd,qn,yd,wd,pr,lt,Pt,Es,No,Sd,qs,$d,hr,xe,Ao,Ed,Do,qd,Fn,Fd,zd,Cd,Io,Pd,Lo,jd,Md,Nd,$e,Oo,Ad,pt,Dd,zn,Id,Ld,Fs,Od,Gd,Ud,jt,Wd,Mt,mr,ht,Nt,zs,Go,Vd,Cs,Rd,fr,be,Uo,Hd,Wo,Bd,Cn,Kd,Jd,Yd,Vo,Xd,Ro,Qd,Zd,ec,Ee,Ho,tc,mt,oc,Pn,nc,sc,Ps,rc,ac,ic,At,dc,Dt,ur,ft,It,js,Bo,cc,Ms,lc,_r,he,Ko,pc,Jo,hc,jn,mc,fc,uc,Yo,_c,Xo,gc,Tc,vc,Lt,xc,qe,Qo,bc,ut,kc,Mn,yc,wc,Ns,Sc,$c,Ec,Ot,qc,Gt,gr,_t,Ut,As,Zo,Fc,Ds,zc,Tr,me,en,Cc,tn,Pc,Nn,jc,Mc,Nc,on,Ac,nn,Dc,Ic,Lc,Wt,Oc,Fe,sn,Gc,gt,Uc,An,Wc,Vc,Is,Rc,Hc,Bc,Vt,Kc,Rt,vr;return d=new We({}),ee=new We({}),Y=new We({}),oo=new dn({props:{code:`import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from datasets import load_dataset

model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")


ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")

inputs = processor(ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt")
generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])

transcription = processor.batch_decode(generated_ids)
transcription`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Speech2TextProcessor, Speech2TextForConditionalGeneration
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset

<span class="hljs-meta">&gt;&gt;&gt; </span>model = Speech2TextForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;facebook/s2t-small-librispeech-asr&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor = Speech2TextProcessor.from_pretrained(<span class="hljs-string">&quot;facebook/s2t-small-librispeech-asr&quot;</span>)


<span class="hljs-meta">&gt;&gt;&gt; </span>ds = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/librispeech_asr_demo&quot;</span>, <span class="hljs-string">&quot;clean&quot;</span>, split=<span class="hljs-string">&quot;validation&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(ds[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;array&quot;</span>], sampling_rate=ds[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;sampling_rate&quot;</span>], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>generated_ids = model.generate(inputs[<span class="hljs-string">&quot;input_features&quot;</span>], attention_mask=inputs[<span class="hljs-string">&quot;attention_mask&quot;</span>])

<span class="hljs-meta">&gt;&gt;&gt; </span>transcription = processor.batch_decode(generated_ids)
<span class="hljs-meta">&gt;&gt;&gt; </span>transcription
[<span class="hljs-string">&#x27;mister quilter is the apostle of the middle classes and we are glad to welcome his gospel&#x27;</span>]`}}),so=new dn({props:{code:`import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from datasets import load_dataset

model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")

ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")

inputs = processor(ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt")
generated_ids = model.generate(
    inputs["input_features"],
    attention_mask=inputs["attention_mask"],
    forced_bos_token_id=processor.tokenizer.lang_code_to_id["fr"],
)

translation = processor.batch_decode(generated_ids)
translation`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Speech2TextProcessor, Speech2TextForConditionalGeneration
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset

<span class="hljs-meta">&gt;&gt;&gt; </span>model = Speech2TextForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;facebook/s2t-medium-mustc-multilingual-st&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor = Speech2TextProcessor.from_pretrained(<span class="hljs-string">&quot;facebook/s2t-medium-mustc-multilingual-st&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>ds = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/librispeech_asr_demo&quot;</span>, <span class="hljs-string">&quot;clean&quot;</span>, split=<span class="hljs-string">&quot;validation&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(ds[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;array&quot;</span>], sampling_rate=ds[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;sampling_rate&quot;</span>], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>generated_ids = model.generate(
<span class="hljs-meta">... </span>    inputs[<span class="hljs-string">&quot;input_features&quot;</span>],
<span class="hljs-meta">... </span>    attention_mask=inputs[<span class="hljs-string">&quot;attention_mask&quot;</span>],
<span class="hljs-meta">... </span>    forced_bos_token_id=processor.tokenizer.lang_code_to_id[<span class="hljs-string">&quot;fr&quot;</span>],
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>translation = processor.batch_decode(generated_ids)
<span class="hljs-meta">&gt;&gt;&gt; </span>translation
[<span class="hljs-string">&quot;&lt;lang:fr&gt; (Vid\xE9o) Si M. Kilder est l&#x27;apossible des classes moyennes, et nous sommes heureux d&#x27;\xEAtre accueillis dans son \xE9vangile.&quot;</span>]`}}),ao=new We({}),io=new L({props:{name:"class transformers.Speech2TextConfig",anchor:"transformers.Speech2TextConfig",parameters:[{name:"vocab_size",val:" = 10000"},{name:"encoder_layers",val:" = 12"},{name:"encoder_ffn_dim",val:" = 2048"},{name:"encoder_attention_heads",val:" = 4"},{name:"decoder_layers",val:" = 6"},{name:"decoder_ffn_dim",val:" = 2048"},{name:"decoder_attention_heads",val:" = 4"},{name:"encoder_layerdrop",val:" = 0.0"},{name:"decoder_layerdrop",val:" = 0.0"},{name:"use_cache",val:" = True"},{name:"is_encoder_decoder",val:" = True"},{name:"activation_function",val:" = 'relu'"},{name:"d_model",val:" = 256"},{name:"dropout",val:" = 0.1"},{name:"attention_dropout",val:" = 0.0"},{name:"activation_dropout",val:" = 0.0"},{name:"init_std",val:" = 0.02"},{name:"decoder_start_token_id",val:" = 2"},{name:"classifier_dropout",val:" = 0.0"},{name:"scale_embedding",val:" = True"},{name:"pad_token_id",val:" = 1"},{name:"bos_token_id",val:" = 0"},{name:"eos_token_id",val:" = 2"},{name:"max_source_positions",val:" = 6000"},{name:"max_target_positions",val:" = 1024"},{name:"num_conv_layers",val:" = 2"},{name:"conv_kernel_sizes",val:" = (5, 5)"},{name:"conv_channels",val:" = 1024"},{name:"input_feat_per_channel",val:" = 80"},{name:"input_channels",val:" = 1"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Speech2TextConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 50265) &#x2014;
Vocabulary size of the Speech2Text model. Defines the number of different tokens that can be represented by
the <code>inputs_ids</code> passed when calling <a href="/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextModel">Speech2TextModel</a>`,name:"vocab_size"},{anchor:"transformers.Speech2TextConfig.d_model",description:`<strong>d_model</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
Dimensionality of the layers and the pooler layer.`,name:"d_model"},{anchor:"transformers.Speech2TextConfig.encoder_layers",description:`<strong>encoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of encoder layers.`,name:"encoder_layers"},{anchor:"transformers.Speech2TextConfig.decoder_layers",description:`<strong>decoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of decoder layers.`,name:"decoder_layers"},{anchor:"transformers.Speech2TextConfig.encoder_attention_heads",description:`<strong>encoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"encoder_attention_heads"},{anchor:"transformers.Speech2TextConfig.decoder_attention_heads",description:`<strong>decoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"decoder_attention_heads"},{anchor:"transformers.Speech2TextConfig.decoder_ffn_dim",description:`<strong>decoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"decoder_ffn_dim"},{anchor:"transformers.Speech2TextConfig.encoder_ffn_dim",description:`<strong>encoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"encoder_ffn_dim"},{anchor:"transformers.Speech2TextConfig.activation_function",description:`<strong>activation_function</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"activation_function"},{anchor:"transformers.Speech2TextConfig.dropout",description:`<strong>dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"dropout"},{anchor:"transformers.Speech2TextConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.Speech2TextConfig.activation_dropout",description:`<strong>activation_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for activations inside the fully connected layer.`,name:"activation_dropout"},{anchor:"transformers.Speech2TextConfig.classifier_dropout",description:`<strong>classifier_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for classifier.`,name:"classifier_dropout"},{anchor:"transformers.Speech2TextConfig.init_std",description:`<strong>init_std</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
encoder_layerdrop &#x2014; (<code>float</code>, <em>optional</em>, defaults to 0.0):
The LayerDrop probability for the encoder. See the [LayerDrop paper](see <a href="https://arxiv.org/abs/1909.11556" rel="nofollow">https://arxiv.org/abs/1909.11556</a>)
for more details.
decoder_layerdrop &#x2014; (<code>float</code>, <em>optional</em>, defaults to 0.0):
The LayerDrop probability for the decoder. See the [LayerDrop paper](see <a href="https://arxiv.org/abs/1909.11556" rel="nofollow">https://arxiv.org/abs/1909.11556</a>)
for more details.`,name:"init_std"},{anchor:"transformers.Speech2TextConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models).`,name:"use_cache"},{anchor:"transformers.Speech2TextConfig.max_source_positions",description:`<strong>max_source_positions</strong> (<code>int</code>, <em>optional</em>, defaults to 6000) &#x2014;
The maximum sequence length of log-mel filter-bank features that this model might ever be used with.`,name:"max_source_positions"},{anchor:"transformers.Speech2TextConfig.max_target_positions",description:`<strong>max_target_positions</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_target_positions"},{anchor:"transformers.Speech2TextConfig.num_conv_layers",description:`<strong>num_conv_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Number of 1D convolutional layers in the conv module.`,name:"num_conv_layers"},{anchor:"transformers.Speech2TextConfig.conv_kernel_sizes",description:`<strong>conv_kernel_sizes</strong> (<code>Tuple[int]</code>, <em>optional</em>, defaults to <code>(5, 5)</code>) &#x2014;
A tuple of integers defining the kernel size of each 1D convolutional layer in the conv module. The length
of <code>conv_kernel_sizes</code> has to match <code>num_conv_layers</code>.`,name:"conv_kernel_sizes"},{anchor:"transformers.Speech2TextConfig.conv_channels",description:`<strong>conv_channels</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
An integer defining the number of output channels of each convolution layers except the final one in the
conv module.`,name:"conv_channels"},{anchor:"transformers.Speech2TextConfig.input_feat_per_channel",description:`<strong>input_feat_per_channel</strong> (<code>int</code>, <em>optional</em>, defaults to 80) &#x2014;
An integer specifying the size of feature vector. This is also the dimensions of log-mel filter-bank
features.`,name:"input_feat_per_channel"},{anchor:"transformers.Speech2TextConfig.input_channels",description:`<strong>input_channels</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
An integer specifying number of input channels of the input feature vector.`,name:"input_channels"}],source:"https://github.com/huggingface/transformers/blob/vr_18590/src/transformers/models/speech_to_text/configuration_speech_to_text.py#L31"}}),xt=new Js({props:{anchor:"transformers.Speech2TextConfig.example",$$slots:{default:[Rp]},$$scope:{ctx:C}}}),lo=new We({}),po=new L({props:{name:"class transformers.Speech2TextTokenizer",anchor:"transformers.Speech2TextTokenizer",parameters:[{name:"vocab_file",val:""},{name:"spm_file",val:""},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"pad_token",val:" = '<pad>'"},{name:"unk_token",val:" = '<unk>'"},{name:"do_upper_case",val:" = False"},{name:"do_lower_case",val:" = False"},{name:"tgt_lang",val:" = None"},{name:"lang_codes",val:" = None"},{name:"sp_model_kwargs",val:": typing.Union[typing.Dict[str, typing.Any], NoneType] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Speech2TextTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
File containing the vocabulary.`,name:"vocab_file"},{anchor:"transformers.Speech2TextTokenizer.spm_file",description:`<strong>spm_file</strong> (<code>str</code>) &#x2014;
Path to the <a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a> model file`,name:"spm_file"},{anchor:"transformers.Speech2TextTokenizer.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sentence token.`,name:"bos_token"},{anchor:"transformers.Speech2TextTokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sentence token.`,name:"eos_token"},{anchor:"transformers.Speech2TextTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.Speech2TextTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.Speech2TextTokenizer.do_upper_case",description:`<strong>do_upper_case</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to uppercase the output when decoding.`,name:"do_upper_case"},{anchor:"transformers.Speech2TextTokenizer.do_lower_case",description:`<strong>do_lower_case</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to lowercase the input when tokenizing.`,name:"do_lower_case"},{anchor:"transformers.Speech2TextTokenizer.tgt_lang",description:`<strong>tgt_lang</strong> (<code>str</code>, <em>optional</em>) &#x2014;
A string representing the target language.`,name:"tgt_lang"},{anchor:"transformers.Speech2TextTokenizer.sp_model_kwargs",description:`<strong>sp_model_kwargs</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
Will be passed to the <code>SentencePieceProcessor.__init__()</code> method. The <a href="https://github.com/google/sentencepiece/tree/master/python" rel="nofollow">Python wrapper for
SentencePiece</a> can be used, among other things,
to set:</p>
<ul>
<li>
<p><code>enable_sampling</code>: Enable subword regularization.</p>
</li>
<li>
<p><code>nbest_size</code>: Sampling parameters for unigram. Invalid for BPE-Dropout.</p>
<ul>
<li><code>nbest_size = {0,1}</code>: No sampling is performed.</li>
<li><code>nbest_size &gt; 1</code>: samples from the nbest_size results.</li>
<li><code>nbest_size &lt; 0</code>: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
using forward-filtering-and-backward-sampling algorithm.</li>
</ul>
</li>
<li>
<p><code>alpha</code>: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
BPE-dropout.</p>
</li>
</ul>
<p>**kwargs &#x2014;
Additional keyword arguments passed along to <a href="/docs/transformers/pr_18590/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a>`,name:"sp_model_kwargs"}],source:"https://github.com/huggingface/transformers/blob/vr_18590/src/transformers/models/speech_to_text/tokenization_speech_to_text.py#L59"}}),mo=new L({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.Speech2TextTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:""},{name:"token_ids_1",val:" = None"}],source:"https://github.com/huggingface/transformers/blob/vr_18590/src/transformers/models/speech_to_text/tokenization_speech_to_text.py#L199"}}),fo=new L({props:{name:"get_special_tokens_mask",anchor:"transformers.Speech2TextTokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": typing.List[int]"},{name:"token_ids_1",val:": typing.Optional[typing.List[int]] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.Speech2TextTokenizer.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.Speech2TextTokenizer.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"},{anchor:"transformers.Speech2TextTokenizer.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/vr_18590/src/transformers/models/speech_to_text/tokenization_speech_to_text.py#L206",returnDescription:`
<p>A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.</p>
`,returnType:`
<p><code>List[int]</code></p>
`}}),_o=new L({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.Speech2TextTokenizer.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": typing.List[int]"},{name:"token_ids_1",val:": typing.Optional[typing.List[int]] = None"}],parametersDescription:[{anchor:"transformers.Speech2TextTokenizer.create_token_type_ids_from_sequences.token_ids_0",description:"<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014; The first tokenized sequence.",name:"token_ids_0"},{anchor:"transformers.Speech2TextTokenizer.create_token_type_ids_from_sequences.token_ids_1",description:"<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014; The second tokenized sequence.",name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/vr_18590/src/transformers/tokenization_utils_base.py#L2962",returnDescription:`
<p>The token type ids.</p>
`,returnType:`
<p><code>List[int]</code></p>
`}}),go=new L({props:{name:"save_vocabulary",anchor:"transformers.Speech2TextTokenizer.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/vr_18590/src/transformers/models/speech_to_text/tokenization_speech_to_text.py#L255"}}),To=new We({}),vo=new L({props:{name:"class transformers.Speech2TextFeatureExtractor",anchor:"transformers.Speech2TextFeatureExtractor",parameters:[{name:"feature_size",val:" = 80"},{name:"sampling_rate",val:" = 16000"},{name:"num_mel_bins",val:" = 80"},{name:"padding_value",val:" = 0.0"},{name:"do_ceptral_normalize",val:" = True"},{name:"normalize_means",val:" = True"},{name:"normalize_vars",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Speech2TextFeatureExtractor.feature_size",description:`<strong>feature_size</strong> (<code>int</code>, defaults to 80) &#x2014;
The feature dimension of the extracted features.`,name:"feature_size"},{anchor:"transformers.Speech2TextFeatureExtractor.sampling_rate",description:`<strong>sampling_rate</strong> (<code>int</code>, defaults to 16000) &#x2014;
The sampling rate at which the audio files should be digitalized expressed in Hertz per second (Hz).`,name:"sampling_rate"},{anchor:"transformers.Speech2TextFeatureExtractor.num_mel_bins",description:`<strong>num_mel_bins</strong> (<code>int</code>, defaults to 80) &#x2014;
Number of Mel-frequency bins.`,name:"num_mel_bins"},{anchor:"transformers.Speech2TextFeatureExtractor.padding_value",description:`<strong>padding_value</strong> (<code>float</code>, defaults to 0.0) &#x2014;
The value that is used to fill the padding vectors.`,name:"padding_value"},{anchor:"transformers.Speech2TextFeatureExtractor.do_ceptral_normalize",description:`<strong>do_ceptral_normalize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to apply utterance-level cepstral mean and variance normalization to extracted features.`,name:"do_ceptral_normalize"},{anchor:"transformers.Speech2TextFeatureExtractor.normalize_means",description:`<strong>normalize_means</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to zero-mean normalize the extracted features.`,name:"normalize_means"},{anchor:"transformers.Speech2TextFeatureExtractor.normalize_vars",description:`<strong>normalize_vars</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to unit-variance normalize the extracted features.`,name:"normalize_vars"}],source:"https://github.com/huggingface/transformers/blob/vr_18590/src/transformers/models/speech_to_text/feature_extraction_speech_to_text.py#L33"}}),bo=new L({props:{name:"__call__",anchor:"transformers.Speech2TextFeatureExtractor.__call__",parameters:[{name:"raw_speech",val:": typing.Union[numpy.ndarray, typing.List[float], typing.List[numpy.ndarray], typing.List[typing.List[float]]]"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"truncation",val:": bool = False"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"sampling_rate",val:": typing.Optional[int] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Speech2TextFeatureExtractor.__call__.raw_speech",description:`<strong>raw_speech</strong> (<code>np.ndarray</code>, <code>List[float]</code>, <code>List[np.ndarray]</code>, <code>List[List[float]]</code>) &#x2014;
The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
values, a list of numpy arrays or a list of list of float values.`,name:"raw_speech"},{anchor:"transformers.Speech2TextFeatureExtractor.__call__.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/pr_18590/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Select a strategy to pad the returned sequences (according to the model&#x2019;s padding side and padding
index) among:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence if provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.Speech2TextFeatureExtractor.__call__.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Maximum length of the returned list and optionally padding length (see above).`,name:"max_length"},{anchor:"transformers.Speech2TextFeatureExtractor.__call__.truncation",description:`<strong>truncation</strong> (<code>bool</code>) &#x2014;
Activates truncation to cut input sequences longer than <em>max_length</em> to <em>max_length</em>.`,name:"truncation"},{anchor:"transformers.Speech2TextFeatureExtractor.__call__.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value.</p>
<p>This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability</p>
<blockquote>
<p>= 7.5 (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.</p>
</blockquote>`,name:"pad_to_multiple_of"},{anchor:"transformers.Speech2TextFeatureExtractor.__call__.return_attention_mask",description:`<strong>return_attention_mask</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return the attention mask. If left to the default, will return the attention mask according
to the specific feature_extractor&#x2019;s default.</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a></p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>For Speech2TextTransoformer models, <code>attention_mask</code> should alwys be passed for batched inference, to
avoid subtle bugs.</p>

					</div>`,name:"return_attention_mask"},{anchor:"transformers.Speech2TextFeatureExtractor.__call__.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/pr_18590/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.Speech2TextFeatureExtractor.__call__.sampling_rate",description:`<strong>sampling_rate</strong> (<code>int</code>, <em>optional</em>) &#x2014;
The sampling rate at which the <code>raw_speech</code> input was sampled. It is strongly recommended to pass
<code>sampling_rate</code> at the forward call to prevent silent errors.`,name:"sampling_rate"},{anchor:"transformers.Speech2TextFeatureExtractor.__call__.padding_value",description:`<strong>padding_value</strong> (<code>float</code>, defaults to 0.0) &#x2014;
The value that is used to fill the padding values / vectors.`,name:"padding_value"}],source:"https://github.com/huggingface/transformers/blob/vr_18590/src/transformers/models/speech_to_text/feature_extraction_speech_to_text.py#L126"}}),ko=new We({}),yo=new L({props:{name:"class transformers.Speech2TextProcessor",anchor:"transformers.Speech2TextProcessor",parameters:[{name:"feature_extractor",val:""},{name:"tokenizer",val:""}],parametersDescription:[{anchor:"transformers.Speech2TextProcessor.feature_extractor",description:`<strong>feature_extractor</strong> (<code>Speech2TextFeatureExtractor</code>) &#x2014;
An instance of <a href="/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor">Speech2TextFeatureExtractor</a>. The feature extractor is a required input.`,name:"feature_extractor"},{anchor:"transformers.Speech2TextProcessor.tokenizer",description:`<strong>tokenizer</strong> (<code>Speech2TextTokenizer</code>) &#x2014;
An instance of <a href="/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextTokenizer">Speech2TextTokenizer</a>. The tokenizer is a required input.`,name:"tokenizer"}],source:"https://github.com/huggingface/transformers/blob/vr_18590/src/transformers/models/speech_to_text/processing_speech_to_text.py#L24"}}),So=new L({props:{name:"__call__",anchor:"transformers.Speech2TextProcessor.__call__",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18590/src/transformers/models/speech_to_text/processing_speech_to_text.py#L47"}}),qo=new L({props:{name:"from_pretrained",anchor:"transformers.Speech2TextProcessor.from_pretrained",parameters:[{name:"pretrained_model_name_or_path",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Speech2TextProcessor.from_pretrained.pretrained_model_name_or_path",description:`<strong>pretrained_model_name_or_path</strong> (<code>str</code> or <code>os.PathLike</code>) &#x2014;
This can be either:</p>
<ul>
<li>a string, the <em>model id</em> of a pretrained feature_extractor hosted inside a model repo on
huggingface.co. Valid model ids can be located at the root-level, like <code>bert-base-uncased</code>, or
namespaced under a user or organization name, like <code>dbmdz/bert-base-german-cased</code>.</li>
<li>a path to a <em>directory</em> containing a feature extractor file saved using the
<a href="/docs/transformers/pr_18590/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained">save_pretrained()</a> method, e.g., <code>./my_model_directory/</code>.</li>
<li>a path or url to a saved feature extractor JSON <em>file</em>, e.g.,
<code>./my_model_directory/preprocessor_config.json</code>.
**kwargs &#x2014;
Additional keyword arguments passed along to both
<a href="/docs/transformers/pr_18590/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained">from_pretrained()</a> and
<code>from_pretrained</code>.</li>
</ul>`,name:"pretrained_model_name_or_path"}],source:"https://github.com/huggingface/transformers/blob/vr_18590/src/transformers/processing_utils.py#L152"}}),qt=new eo({props:{$$slots:{default:[Hp]},$$scope:{ctx:C}}}),Fo=new L({props:{name:"save_pretrained",anchor:"transformers.Speech2TextProcessor.save_pretrained",parameters:[{name:"save_directory",val:""},{name:"push_to_hub",val:": bool = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Speech2TextProcessor.save_pretrained.save_directory",description:`<strong>save_directory</strong> (<code>str</code> or <code>os.PathLike</code>) &#x2014;
Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
be created if it does not exist).`,name:"save_directory"},{anchor:"transformers.Speech2TextProcessor.save_pretrained.push_to_hub",description:`<strong>push_to_hub</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
repository you want to push to with <code>repo_id</code> (will default to the name of <code>save_directory</code> in your
namespace).
kwargs &#x2014;
Additional key word arguments passed along to the <a href="/docs/transformers/pr_18590/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub">push_to_hub()</a> method.`,name:"push_to_hub"}],source:"https://github.com/huggingface/transformers/blob/vr_18590/src/transformers/processing_utils.py#L94"}}),Ft=new eo({props:{$$slots:{default:[Bp]},$$scope:{ctx:C}}}),Co=new L({props:{name:"batch_decode",anchor:"transformers.Speech2TextProcessor.batch_decode",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18590/src/transformers/models/speech_to_text/processing_speech_to_text.py#L85"}}),jo=new L({props:{name:"decode",anchor:"transformers.Speech2TextProcessor.decode",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18590/src/transformers/models/speech_to_text/processing_speech_to_text.py#L92"}}),No=new We({}),Ao=new L({props:{name:"class transformers.Speech2TextModel",anchor:"transformers.Speech2TextModel",parameters:[{name:"config",val:": Speech2TextConfig"}],parametersDescription:[{anchor:"transformers.Speech2TextModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextConfig">Speech2TextConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/pr_18590/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18590/src/transformers/models/speech_to_text/modeling_speech_to_text.py#L1121"}}),Oo=new L({props:{name:"forward",anchor:"transformers.Speech2TextModel.forward",parameters:[{name:"input_features",val:" = None"},{name:"attention_mask",val:" = None"},{name:"decoder_input_ids",val:" = None"},{name:"decoder_attention_mask",val:" = None"},{name:"head_mask",val:" = None"},{name:"decoder_head_mask",val:" = None"},{name:"cross_attn_head_mask",val:" = None"},{name:"encoder_outputs",val:" = None"},{name:"past_key_values",val:" = None"},{name:"decoder_inputs_embeds",val:" = None"},{name:"use_cache",val:" = None"},{name:"output_attentions",val:" = None"},{name:"output_hidden_states",val:" = None"},{name:"return_dict",val:" = None"}],parametersDescription:[{anchor:"transformers.Speech2TextModel.forward.input_features",description:`<strong>input_features</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, feature_size)</code>) &#x2014;
Float values of fbank features extracted from the raw speech waveform. Raw speech waveform can be obtained
by loading a <code>.flac</code> or <code>.wav</code> audio file into an array of type <code>List[float]</code> or a <code>numpy.ndarray</code>, <em>e.g.</em>
via the soundfile library (<code>pip install soundfile</code>). To prepare the array into <code>input_features</code>, the
<a href="/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor">Speech2TextFeatureExtractor</a> should be used for extracting the fbank features, padding and conversion
into a tensor of type <code>torch.FloatTensor</code>. See <a href="/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor.__call__"><strong>call</strong>()</a>`,name:"input_features"},{anchor:"transformers.Speech2TextModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing convolution and attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Speech2TextModel.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <code>SpeechToTextTokenizer</code>. See <a href="/docs/transformers/pr_18590/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18590/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>SpeechToText uses the <code>eos_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_ids</code> have to be input (see
<code>past_key_values</code>).`,name:"decoder_input_ids"},{anchor:"transformers.Speech2TextModel.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.</p>
<p>If you want to change padding behavior, you should read
<code>modeling_speech_to_text._prepare_decoder_attention_mask</code> and modify to your needs. See diagram 1 in <a href="https://arxiv.org/abs/1910.13461" rel="nofollow">the
paper</a> for more information on the default strategy.`,name:"decoder_attention_mask"},{anchor:"transformers.Speech2TextModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(encoder_layers, encoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.Speech2TextModel.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.Speech2TextModel.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.Speech2TextModel.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple(tuple(torch.FloatTensor)</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.Speech2TextModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple(tuple(torch.FloatTensor))</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) &#x2014;
Tuple of <code>tuple(torch.FloatTensor)</code> of length <code>config.n_layers</code>, with each tuple having 2 tensors of shape
<code>(batch_size, num_heads, sequence_length, embed_size_per_head)</code>) and 2 additional tensors of shape
<code>(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)</code>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
<p>If <code>past_key_values</code> are used, the user can optionally input only the last <code>decoder_input_ids</code> (those that
don&#x2019;t have their past key value states given to this model) of shape <code>(batch_size, 1)</code> instead of all
<code>decoder_input_ids</code> of shape <code>(batch_size, sequence_length)</code>. decoder_inputs_embeds (<code>torch.FloatTensor</code> of
shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>): Optionally, instead of passing
<code>decoder_input_ids</code> you can choose to directly pass an embedded representation. If <code>past_key_values</code> is
used, optionally only the last <code>decoder_inputs_embeds</code> have to be input (see <code>past_key_values</code>). This is
useful if you want more control over how to convert <code>decoder_input_ids</code> indices into associated vectors
than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"past_key_values"},{anchor:"transformers.Speech2TextModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Speech2TextModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Speech2TextModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Speech2TextModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18590/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18590/src/transformers/models/speech_to_text/modeling_speech_to_text.py#L1143",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18590/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput"
>transformers.modeling_outputs.Seq2SeqLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextConfig"
>Speech2TextConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18590/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput"
>transformers.modeling_outputs.Seq2SeqLMOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),jt=new eo({props:{$$slots:{default:[Kp]},$$scope:{ctx:C}}}),Mt=new Js({props:{anchor:"transformers.Speech2TextModel.forward.example",$$slots:{default:[Jp]},$$scope:{ctx:C}}}),Go=new We({}),Uo=new L({props:{name:"class transformers.Speech2TextForConditionalGeneration",anchor:"transformers.Speech2TextForConditionalGeneration",parameters:[{name:"config",val:": Speech2TextConfig"}],parametersDescription:[{anchor:"transformers.Speech2TextForConditionalGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextConfig">Speech2TextConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/pr_18590/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18590/src/transformers/models/speech_to_text/modeling_speech_to_text.py#L1252"}}),Ho=new L({props:{name:"forward",anchor:"transformers.Speech2TextForConditionalGeneration.forward",parameters:[{name:"input_features",val:" = None"},{name:"attention_mask",val:" = None"},{name:"decoder_input_ids",val:" = None"},{name:"decoder_attention_mask",val:" = None"},{name:"head_mask",val:" = None"},{name:"decoder_head_mask",val:" = None"},{name:"cross_attn_head_mask",val:" = None"},{name:"encoder_outputs",val:" = None"},{name:"past_key_values",val:" = None"},{name:"decoder_inputs_embeds",val:" = None"},{name:"labels",val:" = None"},{name:"use_cache",val:" = None"},{name:"output_attentions",val:" = None"},{name:"output_hidden_states",val:" = None"},{name:"return_dict",val:" = None"}],parametersDescription:[{anchor:"transformers.Speech2TextForConditionalGeneration.forward.input_features",description:`<strong>input_features</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, feature_size)</code>) &#x2014;
Float values of fbank features extracted from the raw speech waveform. Raw speech waveform can be obtained
by loading a <code>.flac</code> or <code>.wav</code> audio file into an array of type <code>List[float]</code> or a <code>numpy.ndarray</code>, <em>e.g.</em>
via the soundfile library (<code>pip install soundfile</code>). To prepare the array into <code>input_features</code>, the
<a href="/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor">Speech2TextFeatureExtractor</a> should be used for extracting the fbank features, padding and conversion
into a tensor of type <code>torch.FloatTensor</code>. See <a href="/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor.__call__"><strong>call</strong>()</a>`,name:"input_features"},{anchor:"transformers.Speech2TextForConditionalGeneration.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing convolution and attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Speech2TextForConditionalGeneration.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <code>SpeechToTextTokenizer</code>. See <a href="/docs/transformers/pr_18590/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18590/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>SpeechToText uses the <code>eos_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_ids</code> have to be input (see
<code>past_key_values</code>).`,name:"decoder_input_ids"},{anchor:"transformers.Speech2TextForConditionalGeneration.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.</p>
<p>If you want to change padding behavior, you should read
<code>modeling_speech_to_text._prepare_decoder_attention_mask</code> and modify to your needs. See diagram 1 in <a href="https://arxiv.org/abs/1910.13461" rel="nofollow">the
paper</a> for more information on the default strategy.`,name:"decoder_attention_mask"},{anchor:"transformers.Speech2TextForConditionalGeneration.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(encoder_layers, encoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.Speech2TextForConditionalGeneration.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.Speech2TextForConditionalGeneration.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.Speech2TextForConditionalGeneration.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple(tuple(torch.FloatTensor)</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.Speech2TextForConditionalGeneration.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple(tuple(torch.FloatTensor))</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) &#x2014;
Tuple of <code>tuple(torch.FloatTensor)</code> of length <code>config.n_layers</code>, with each tuple having 2 tensors of shape
<code>(batch_size, num_heads, sequence_length, embed_size_per_head)</code>) and 2 additional tensors of shape
<code>(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)</code>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
<p>If <code>past_key_values</code> are used, the user can optionally input only the last <code>decoder_input_ids</code> (those that
don&#x2019;t have their past key value states given to this model) of shape <code>(batch_size, 1)</code> instead of all
<code>decoder_input_ids</code> of shape <code>(batch_size, sequence_length)</code>. decoder_inputs_embeds (<code>torch.FloatTensor</code> of
shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>): Optionally, instead of passing
<code>decoder_input_ids</code> you can choose to directly pass an embedded representation. If <code>past_key_values</code> is
used, optionally only the last <code>decoder_inputs_embeds</code> have to be input (see <code>past_key_values</code>). This is
useful if you want more control over how to convert <code>decoder_input_ids</code> indices into associated vectors
than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"past_key_values"},{anchor:"transformers.Speech2TextForConditionalGeneration.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Speech2TextForConditionalGeneration.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Speech2TextForConditionalGeneration.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Speech2TextForConditionalGeneration.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18590/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.Speech2TextForConditionalGeneration.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code>
or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored (masked), the loss is
only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18590/src/transformers/models/speech_to_text/modeling_speech_to_text.py#L1289",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18590/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput"
>transformers.modeling_outputs.Seq2SeqLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextConfig"
>Speech2TextConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18590/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput"
>transformers.modeling_outputs.Seq2SeqLMOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),At=new eo({props:{$$slots:{default:[Yp]},$$scope:{ctx:C}}}),Dt=new Js({props:{anchor:"transformers.Speech2TextForConditionalGeneration.forward.example",$$slots:{default:[Xp]},$$scope:{ctx:C}}}),Bo=new We({}),Ko=new L({props:{name:"class transformers.TFSpeech2TextModel",anchor:"transformers.TFSpeech2TextModel",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFSpeech2TextModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextConfig">Speech2TextConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/pr_18590/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18590/src/transformers/models/speech_to_text/modeling_tf_speech_to_text.py#L1235"}}),Lt=new eo({props:{$$slots:{default:[Qp]},$$scope:{ctx:C}}}),Qo=new L({props:{name:"call",anchor:"transformers.TFSpeech2TextModel.call",parameters:[{name:"input_features",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"decoder_input_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"decoder_attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"head_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"decoder_head_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"cross_attn_head_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"encoder_outputs",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"past_key_values",val:": typing.Union[typing.Tuple[typing.Tuple[typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor]]], NoneType] = None"},{name:"decoder_inputs_embeds",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"training",val:": bool = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFSpeech2TextModel.call.input_features",description:`<strong>input_features</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, feature_size)</code>) &#x2014;
Float values of fbank features extracted from the raw speech waveform. Raw speech waveform can be obtained
by loading a <code>.flac</code> or <code>.wav</code> audio file into an array of type <code>List[float]</code> or a <code>numpy.ndarray</code>, <em>e.g.</em>
via the soundfile library (<code>pip install soundfile</code>). To prepare the array into <code>input_features</code>, the
<a href="/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor">Speech2TextFeatureExtractor</a> should be used for extracting the fbank features, padding and conversion
into a tensor of floats. See <a href="/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor.__call__"><strong>call</strong>()</a>`,name:"input_features"},{anchor:"transformers.TFSpeech2TextModel.call.attention_mask",description:`<strong>attention_mask</strong> (<code>tf.Tensor</code> of shape <code>({0})</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.TFSpeech2TextModel.call.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextTokenizer">Speech2TextTokenizer</a>. See <a href="/docs/transformers/pr_18590/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18590/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>Bart uses the <code>eos_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If <code>past_key_values</code>
is used, optionally only the last <code>decoder_input_ids</code> have to be input (see <code>past_key_values</code>).</p>
<p>For translation and summarization training, <code>decoder_input_ids</code> should be provided. If no
<code>decoder_input_ids</code> is provided, the model will create this tensor by shifting the <code>input_ids</code> to the right
for denoising pre-training following the paper.`,name:"decoder_input_ids"},{anchor:"transformers.TFSpeech2TextModel.call.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
will be made by default and ignore pad tokens. It is not recommended to set this for most use cases.`,name:"decoder_attention_mask"},{anchor:"transformers.TFSpeech2TextModel.call.head_mask",description:`<strong>head_mask</strong> (<code>tf.Tensor</code> of shape <code>(encoder_layers, encoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.TFSpeech2TextModel.call.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>tf.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.TFSpeech2TextModel.call.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>tf.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.TFSpeech2TextModel.call.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tf.FloatTensor</code>, <em>optional</em>) &#x2014;
hidden states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
of shape <code>(batch_size, sequence_length, hidden_size)</code> is a sequence of`,name:"encoder_outputs"},{anchor:"transformers.TFSpeech2TextModel.call.past_key_values",description:`<strong>past_key_values</strong> (<code>Tuple[Tuple[tf.Tensor]]</code> of length <code>config.n_layers</code>) &#x2014;
contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
If <code>past_key_values</code> are used, the user can optionally input only the last <code>decoder_input_ids</code> (those that
don&#x2019;t have their past key value states given to this model) of shape <code>(batch_size, 1)</code> instead of all
<code>decoder_input_ids</code> of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.TFSpeech2TextModel.call.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
config will be used instead.`,name:"output_attentions"},{anchor:"transformers.TFSpeech2TextModel.call.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
used instead.`,name:"output_hidden_states"},{anchor:"transformers.TFSpeech2TextModel.call.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18590/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple. This argument can be used in
eager mode, in graph mode the value will always be set to True.`,name:"return_dict"},{anchor:"transformers.TFSpeech2TextModel.call.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"}],source:"https://github.com/huggingface/transformers/blob/vr_18590/src/transformers/models/speech_to_text/modeling_tf_speech_to_text.py#L1247",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18590/en/main_classes/output#transformers.modeling_tf_outputs.TFSeq2SeqModelOutput"
>transformers.modeling_tf_outputs.TFSeq2SeqModelOutput</a> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<a
  href="/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextConfig"
>Speech2TextConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) \u2014 Sequence of hidden-states at the output of the last layer of the decoder of the model.</p>
<p>If <code>past_key_values</code> is used only the last hidden-state of the sequences of shape <code>(batch_size, 1, hidden_size)</code> is output.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>List[tf.Tensor]</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) \u2014 List of <code>tf.Tensor</code> of length <code>config.n_layers</code>, with each tensor of shape <code>(2, batch_size, num_heads, sequence_length, embed_size_per_head)</code>).</p>
<p>Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>decoder_hidden_states</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>decoder_attentions</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder\u2019s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
<li>
<p><strong>encoder_last_hidden_state</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) \u2014 Sequence of hidden-states at the output of the last layer of the encoder of the model.</p>
</li>
<li>
<p><strong>encoder_hidden_states</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>encoder_attentions</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18590/en/main_classes/output#transformers.modeling_tf_outputs.TFSeq2SeqModelOutput"
>transformers.modeling_tf_outputs.TFSeq2SeqModelOutput</a> or <code>tuple(tf.Tensor)</code></p>
`}}),Ot=new eo({props:{$$slots:{default:[Zp]},$$scope:{ctx:C}}}),Gt=new Js({props:{anchor:"transformers.TFSpeech2TextModel.call.example",$$slots:{default:[eh]},$$scope:{ctx:C}}}),Zo=new We({}),en=new L({props:{name:"class transformers.TFSpeech2TextForConditionalGeneration",anchor:"transformers.TFSpeech2TextForConditionalGeneration",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFSpeech2TextForConditionalGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextConfig">Speech2TextConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/pr_18590/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18590/src/transformers/models/speech_to_text/modeling_tf_speech_to_text.py#L1318"}}),Wt=new eo({props:{$$slots:{default:[th]},$$scope:{ctx:C}}}),sn=new L({props:{name:"call",anchor:"transformers.TFSpeech2TextForConditionalGeneration.call",parameters:[{name:"input_features",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"decoder_input_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"decoder_attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"head_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"decoder_head_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"cross_attn_head_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"encoder_outputs",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"past_key_values",val:": typing.Union[typing.Tuple[typing.Tuple[typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor]]], NoneType] = None"},{name:"decoder_inputs_embeds",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"labels",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"training",val:": typing.Optional[bool] = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFSpeech2TextForConditionalGeneration.call.input_features",description:`<strong>input_features</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, feature_size)</code>) &#x2014;
Float values of fbank features extracted from the raw speech waveform. Raw speech waveform can be obtained
by loading a <code>.flac</code> or <code>.wav</code> audio file into an array of type <code>List[float]</code> or a <code>numpy.ndarray</code>, <em>e.g.</em>
via the soundfile library (<code>pip install soundfile</code>). To prepare the array into <code>input_features</code>, the
<a href="/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor">Speech2TextFeatureExtractor</a> should be used for extracting the fbank features, padding and conversion
into a tensor of floats. See <a href="/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor.__call__"><strong>call</strong>()</a>`,name:"input_features"},{anchor:"transformers.TFSpeech2TextForConditionalGeneration.call.attention_mask",description:`<strong>attention_mask</strong> (<code>tf.Tensor</code> of shape <code>({0})</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.TFSpeech2TextForConditionalGeneration.call.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextTokenizer">Speech2TextTokenizer</a>. See <a href="/docs/transformers/pr_18590/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18590/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>Bart uses the <code>eos_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If <code>past_key_values</code>
is used, optionally only the last <code>decoder_input_ids</code> have to be input (see <code>past_key_values</code>).</p>
<p>For translation and summarization training, <code>decoder_input_ids</code> should be provided. If no
<code>decoder_input_ids</code> is provided, the model will create this tensor by shifting the <code>input_ids</code> to the right
for denoising pre-training following the paper.`,name:"decoder_input_ids"},{anchor:"transformers.TFSpeech2TextForConditionalGeneration.call.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
will be made by default and ignore pad tokens. It is not recommended to set this for most use cases.`,name:"decoder_attention_mask"},{anchor:"transformers.TFSpeech2TextForConditionalGeneration.call.head_mask",description:`<strong>head_mask</strong> (<code>tf.Tensor</code> of shape <code>(encoder_layers, encoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.TFSpeech2TextForConditionalGeneration.call.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>tf.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.TFSpeech2TextForConditionalGeneration.call.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>tf.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.TFSpeech2TextForConditionalGeneration.call.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tf.FloatTensor</code>, <em>optional</em>) &#x2014;
hidden states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
of shape <code>(batch_size, sequence_length, hidden_size)</code> is a sequence of`,name:"encoder_outputs"},{anchor:"transformers.TFSpeech2TextForConditionalGeneration.call.past_key_values",description:`<strong>past_key_values</strong> (<code>Tuple[Tuple[tf.Tensor]]</code> of length <code>config.n_layers</code>) &#x2014;
contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
If <code>past_key_values</code> are used, the user can optionally input only the last <code>decoder_input_ids</code> (those that
don&#x2019;t have their past key value states given to this model) of shape <code>(batch_size, 1)</code> instead of all
<code>decoder_input_ids</code> of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.TFSpeech2TextForConditionalGeneration.call.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
config will be used instead.`,name:"output_attentions"},{anchor:"transformers.TFSpeech2TextForConditionalGeneration.call.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
used instead.`,name:"output_hidden_states"},{anchor:"transformers.TFSpeech2TextForConditionalGeneration.call.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18590/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple. This argument can be used in
eager mode, in graph mode the value will always be set to True.`,name:"return_dict"},{anchor:"transformers.TFSpeech2TextForConditionalGeneration.call.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"},{anchor:"transformers.TFSpeech2TextForConditionalGeneration.call.labels",description:`<strong>labels</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18590/src/transformers/models/speech_to_text/modeling_tf_speech_to_text.py#L1342",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18590/en/main_classes/output#transformers.modeling_tf_outputs.TFSeq2SeqLMOutput"
>transformers.modeling_tf_outputs.TFSeq2SeqLMOutput</a> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<a
  href="/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextConfig"
>Speech2TextConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>tf.Tensor</code> of shape <code>(n,)</code>, <em>optional</em>, where n is the number of non-masked labels, returned when <code>labels</code> is provided) \u2014 Language modeling loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) \u2014 Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>List[tf.Tensor]</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) \u2014 List of <code>tf.Tensor</code> of length <code>config.n_layers</code>, with each tensor of shape <code>(2, batch_size, num_heads, sequence_length, embed_size_per_head)</code>).</p>
<p>Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>decoder_hidden_states</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>decoder_attentions</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder\u2019s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
<li>
<p><strong>encoder_last_hidden_state</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) \u2014 Sequence of hidden-states at the output of the last layer of the encoder of the model.</p>
</li>
<li>
<p><strong>encoder_hidden_states</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>encoder_attentions</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18590/en/main_classes/output#transformers.modeling_tf_outputs.TFSeq2SeqLMOutput"
>transformers.modeling_tf_outputs.TFSeq2SeqLMOutput</a> or <code>tuple(tf.Tensor)</code></p>
`}}),Vt=new eo({props:{$$slots:{default:[oh]},$$scope:{ctx:C}}}),Rt=new Js({props:{anchor:"transformers.TFSpeech2TextForConditionalGeneration.call.example",$$slots:{default:[nh]},$$scope:{ctx:C}}}),{c(){h=r("meta"),x=l(),_=r("h1"),m=r("a"),v=r("span"),b(d.$$.fragment),g=l(),E=r("span"),A=n("Speech2Text"),F=l(),q=r("h2"),D=r("a"),R=r("span"),b(ee.$$.fragment),Ce=l(),H=r("span"),Pe=n("Overview"),ye=l(),N=r("p"),B=n("The Speech2Text model was proposed in "),te=r("a"),Te=n("fairseq S2T: Fast Speech-to-Text Modeling with fairseq"),j=n(` by Changhan Wang, Yun Tang, Xutai Ma, Anne Wu, Dmytro Okhonko, Juan Pino. It\u2019s a
transformer-based seq2seq (encoder-decoder) model designed for end-to-end Automatic Speech Recognition (ASR) and Speech
Translation (ST). It uses a convolutional downsampler to reduce the length of speech inputs by 3/4th before they are
fed into the encoder. The model is trained with standard autoregressive cross-entropy loss and generates the
transcripts/translations autoregressively. Speech2Text has been fine-tuned on several datasets for ASR and ST:
`),M=r("a"),je=n("LibriSpeech"),oe=n(", "),ne=r("a"),Me=n("CoVoST 2"),se=n(", "),re=r("a"),Ne=n("MuST-C"),O=n("."),we=l(),I=r("p"),Ae=n("This model was contributed by "),ae=r("a"),ie=n("valhalla"),De=n(". The original code can be found "),de=r("a"),K=n("here"),Ie=n("."),Q=l(),J=r("h2"),T=r("a"),z=r("span"),b(Y.$$.fragment),Ze=l(),Le=r("span"),X=n("Inference"),Ve=l(),ue=r("p"),et=n(`Speech2Text is a speech model that accepts a float tensor of log-mel filter-bank features extracted from the speech
signal. It\u2019s a transformer-based seq2seq model, so the transcripts/translations are generated autoregressively. The
`),U=r("code"),ce=n("generate()"),tt=n(" method can be used for inference."),Re=l(),P=r("p"),ot=n("The "),Se=r("a"),Oe=n("Speech2TextFeatureExtractor"),nt=n(` class is responsible for extracting the log-mel filter-bank
features. The `),cn=r("a"),Qr=n("Speech2TextProcessor"),Zr=n(" wraps "),ln=r("a"),ea=n("Speech2TextFeatureExtractor"),ta=n(` and
`),pn=r("a"),oa=n("Speech2TextTokenizer"),na=n(` into a single instance to both extract the input features and decode the
predicted token ids.`),Xs=l(),W=r("p"),sa=n("The feature extractor depends on "),Bn=r("code"),ra=n("torchaudio"),aa=n(" and the tokenizer depends on "),Kn=r("code"),ia=n("sentencepiece"),da=n(` so be sure to
install those packages before running the examples. You could either install those as extra speech dependencies with
`),Jn=r("code"),ca=n('pip install transformers"[speech, sentencepiece]"'),la=n(" or install the packages separately with "),Yn=r("code"),pa=n("pip install torchaudio sentencepiece"),ha=n(". Also "),Xn=r("code"),ma=n("torchaudio"),fa=n(" requires the development version of the "),to=r("a"),ua=n("libsndfile"),_a=n(` package which can be installed via a system package manager. On Ubuntu it can
be installed as follows: `),Qn=r("code"),ga=n("apt install libsndfile1-dev"),Qs=l(),hn=r("ul"),Zn=r("li"),Ta=n("ASR and Speech Translation"),Zs=l(),b(oo.$$.fragment),er=l(),mn=r("ul"),no=r("li"),es=r("p"),va=n("Multilingual speech translation"),xa=l(),le=r("p"),ba=n("For multilingual speech translation models, "),ts=r("code"),ka=n("eos_token_id"),ya=n(" is used as the "),os=r("code"),wa=n("decoder_start_token_id"),Sa=n(` and
the target language id is forced as the first generated token. To force the target language id as the first
generated token, pass the `),ns=r("code"),$a=n("forced_bos_token_id"),Ea=n(" parameter to the "),ss=r("code"),qa=n("generate()"),Fa=n(` method. The following
example shows how to transate English speech to French text using the `),rs=r("em"),za=n("facebook/s2t-medium-mustc-multilingual-st"),Ca=n(`
checkpoint.`),tr=l(),b(so.$$.fragment),or=l(),Tt=r("p"),Pa=n("See the "),ro=r("a"),ja=n("model hub"),Ma=n(" to look for Speech2Text checkpoints."),nr=l(),st=r("h2"),vt=r("a"),as=r("span"),b(ao.$$.fragment),Na=l(),is=r("span"),Aa=n("Speech2TextConfig"),sr=l(),ve=r("div"),b(io.$$.fragment),Da=l(),rt=r("p"),Ia=n("This is the configuration class to store the configuration of a "),fn=r("a"),La=n("Speech2TextModel"),Oa=n(`. It is used to instantiate an
Speech2Text model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the Speech2Text
`),co=r("a"),Ga=n("facebook/s2t-small-librispeech-asr"),Ua=n(" architecture."),Wa=l(),at=r("p"),Va=n("Configuration objects inherit from "),un=r("a"),Ra=n("PretrainedConfig"),Ha=n(` and can be used to control the model outputs. Read the
documentation from `),_n=r("a"),Ba=n("PretrainedConfig"),Ka=n(" for more information."),Ja=l(),b(xt.$$.fragment),rr=l(),it=r("h2"),bt=r("a"),ds=r("span"),b(lo.$$.fragment),Ya=l(),cs=r("span"),Xa=n("Speech2TextTokenizer"),ar=l(),V=r("div"),b(po.$$.fragment),Qa=l(),ls=r("p"),Za=n("Construct an Speech2Text tokenizer."),ei=l(),ho=r("p"),ti=n("This tokenizer inherits from "),gn=r("a"),oi=n("PreTrainedTokenizer"),ni=n(` which contains some of the main methods. Users should refer to
the superclass for more information regarding such methods.`),si=l(),kt=r("div"),b(mo.$$.fragment),ri=l(),ps=r("p"),ai=n("Build model inputs from a sequence by appending eos_token_id."),ii=l(),yt=r("div"),b(fo.$$.fragment),di=l(),uo=r("p"),ci=n(`Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer `),hs=r("code"),li=n("prepare_for_model"),pi=n(" method."),hi=l(),He=r("div"),b(_o.$$.fragment),mi=l(),Tn=r("p"),fi=n("Create the token type IDs corresponding to the sequences passed. "),vn=r("a"),ui=n(`What are token type
IDs?`),_i=l(),ms=r("p"),gi=n("Should be overridden in a subclass if the model has a special way of building those."),Ti=l(),xn=r("div"),b(go.$$.fragment),ir=l(),dt=r("h2"),wt=r("a"),fs=r("span"),b(To.$$.fragment),vi=l(),us=r("span"),xi=n("Speech2TextFeatureExtractor"),dr=l(),pe=r("div"),b(vo.$$.fragment),bi=l(),_s=r("p"),ki=n("Constructs a Speech2Text feature extractor."),yi=l(),xo=r("p"),wi=n("This feature extractor inherits from "),bn=r("a"),Si=n("Speech2TextFeatureExtractor"),$i=n(` which contains most of the main methods. Users
should refer to this superclass for more information regarding those methods.`),Ei=l(),gs=r("p"),qi=n(`This class extracts mel-filter bank features from raw speech using TorchAudio and applies utterance-level cepstral
mean and variance normalization to the extracted features.`),Fi=l(),St=r("div"),b(bo.$$.fragment),zi=l(),Ts=r("p"),Ci=n("Main method to featurize and prepare for the model one or several sequence(s). sequences."),cr=l(),ct=r("h2"),$t=r("a"),vs=r("span"),b(ko.$$.fragment),Pi=l(),xs=r("span"),ji=n("Speech2TextProcessor"),lr=l(),G=r("div"),b(yo.$$.fragment),Mi=l(),bs=r("p"),Ni=n(`Constructs a Speech2Text processor which wraps a Speech2Text feature extractor and a Speech2Text tokenizer into a
single processor.`),Ai=l(),_e=r("p"),kn=r("a"),Di=n("Speech2TextProcessor"),Ii=n(" offers all the functionalities of "),yn=r("a"),Li=n("Speech2TextFeatureExtractor"),Oi=n(` and
`),wn=r("a"),Gi=n("Speech2TextTokenizer"),Ui=n(". See the "),wo=r("a"),ks=r("strong"),Wi=n("call"),Vi=n("()"),Ri=n(" and "),Sn=r("a"),Hi=n("decode()"),Bi=n(` for more
information.`),Ki=l(),Et=r("div"),b(So.$$.fragment),Ji=l(),Ge=r("p"),Yi=n(`When used in normal mode, this method forwards all its arguments to Speech2TextFeatureExtractor\u2019s
`),$o=r("a"),ys=r("strong"),Xi=n("call"),Qi=n("()"),Zi=n(` and returns its output. If used in the context
`),ws=r("code"),ed=n("as_target_processor()"),td=n(` this method forwards all its arguments to Speech2TextTokenizer\u2019s
`),Eo=r("a"),Ss=r("strong"),od=n("call"),nd=n("()"),sd=n(`. Please refer to the doctsring of the above two methods for more
information.`),rd=l(),Be=r("div"),b(qo.$$.fragment),ad=l(),$s=r("p"),id=n("Instantiate a processor associated with a pretrained model."),dd=l(),b(qt.$$.fragment),cd=l(),Ke=r("div"),b(Fo.$$.fragment),ld=l(),zo=r("p"),pd=n(`Saves the attributes of this processor (feature extractor, tokenizer\u2026) in the specified directory so that it
can be reloaded using the `),$n=r("a"),hd=n("from_pretrained()"),md=n(" method."),fd=l(),b(Ft.$$.fragment),ud=l(),zt=r("div"),b(Co.$$.fragment),_d=l(),Po=r("p"),gd=n("This method forwards all its arguments to Speech2TextTokenizer\u2019s "),En=r("a"),Td=n("batch_decode()"),vd=n(`. Please
refer to the docstring of this method for more information.`),xd=l(),Ct=r("div"),b(jo.$$.fragment),bd=l(),Mo=r("p"),kd=n("This method forwards all its arguments to Speech2TextTokenizer\u2019s "),qn=r("a"),yd=n("decode()"),wd=n(`. Please refer
to the docstring of this method for more information.`),pr=l(),lt=r("h2"),Pt=r("a"),Es=r("span"),b(No.$$.fragment),Sd=l(),qs=r("span"),$d=n("Speech2TextModel"),hr=l(),xe=r("div"),b(Ao.$$.fragment),Ed=l(),Do=r("p"),qd=n(`The bare Speech2Text Model outputting raw hidden-states without any specific head on top.
This model inherits from `),Fn=r("a"),Fd=n("PreTrainedModel"),zd=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Cd=l(),Io=r("p"),Pd=n("This model is also a PyTorch "),Lo=r("a"),jd=n("torch.nn.Module"),Md=n(` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Nd=l(),$e=r("div"),b(Oo.$$.fragment),Ad=l(),pt=r("p"),Dd=n("The "),zn=r("a"),Id=n("Speech2TextModel"),Ld=n(" forward method, overrides the "),Fs=r("code"),Od=n("__call__"),Gd=n(" special method."),Ud=l(),b(jt.$$.fragment),Wd=l(),b(Mt.$$.fragment),mr=l(),ht=r("h2"),Nt=r("a"),zs=r("span"),b(Go.$$.fragment),Vd=l(),Cs=r("span"),Rd=n("Speech2TextForConditionalGeneration"),fr=l(),be=r("div"),b(Uo.$$.fragment),Hd=l(),Wo=r("p"),Bd=n(`The Speech2Text Model with a language modeling head. Can be used for summarization.
This model inherits from `),Cn=r("a"),Kd=n("PreTrainedModel"),Jd=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Yd=l(),Vo=r("p"),Xd=n("This model is also a PyTorch "),Ro=r("a"),Qd=n("torch.nn.Module"),Zd=n(` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),ec=l(),Ee=r("div"),b(Ho.$$.fragment),tc=l(),mt=r("p"),oc=n("The "),Pn=r("a"),nc=n("Speech2TextForConditionalGeneration"),sc=n(" forward method, overrides the "),Ps=r("code"),rc=n("__call__"),ac=n(" special method."),ic=l(),b(At.$$.fragment),dc=l(),b(Dt.$$.fragment),ur=l(),ft=r("h2"),It=r("a"),js=r("span"),b(Bo.$$.fragment),cc=l(),Ms=r("span"),lc=n("TFSpeech2TextModel"),_r=l(),he=r("div"),b(Ko.$$.fragment),pc=l(),Jo=r("p"),hc=n(`The bare Speech2Text Model outputting raw hidden-states without any specific head on top.
This model inherits from `),jn=r("a"),mc=n("TFPreTrainedModel"),fc=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),uc=l(),Yo=r("p"),_c=n("This model is also a "),Xo=r("a"),gc=n("tf.keras.Model"),Tc=n(` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),vc=l(),b(Lt.$$.fragment),xc=l(),qe=r("div"),b(Qo.$$.fragment),bc=l(),ut=r("p"),kc=n("The "),Mn=r("a"),yc=n("TFSpeech2TextModel"),wc=n(" forward method, overrides the "),Ns=r("code"),Sc=n("__call__"),$c=n(" special method."),Ec=l(),b(Ot.$$.fragment),qc=l(),b(Gt.$$.fragment),gr=l(),_t=r("h2"),Ut=r("a"),As=r("span"),b(Zo.$$.fragment),Fc=l(),Ds=r("span"),zc=n("TFSpeech2TextForConditionalGeneration"),Tr=l(),me=r("div"),b(en.$$.fragment),Cc=l(),tn=r("p"),Pc=n(`The Speech2Text Model with a language modeling head. Can be used for summarization.
This model inherits from `),Nn=r("a"),jc=n("TFPreTrainedModel"),Mc=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Nc=l(),on=r("p"),Ac=n("This model is also a "),nn=r("a"),Dc=n("tf.keras.Model"),Ic=n(` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),Lc=l(),b(Wt.$$.fragment),Oc=l(),Fe=r("div"),b(sn.$$.fragment),Gc=l(),gt=r("p"),Uc=n("The "),An=r("a"),Wc=n("TFSpeech2TextForConditionalGeneration"),Vc=n(" forward method, overrides the "),Is=r("code"),Rc=n("__call__"),Hc=n(" special method."),Bc=l(),b(Vt.$$.fragment),Kc=l(),b(Rt.$$.fragment),this.h()},l(o){const u=Wp('[data-svelte="svelte-1phssyn"]',document.head);h=a(u,"META",{name:!0,content:!0}),u.forEach(t),x=p(o),_=a(o,"H1",{class:!0});var rn=i(_);m=a(rn,"A",{id:!0,class:!0,href:!0});var Ls=i(m);v=a(Ls,"SPAN",{});var Os=i(v);k(d.$$.fragment,Os),Os.forEach(t),Ls.forEach(t),g=p(rn),E=a(rn,"SPAN",{});var Gs=i(E);A=s(Gs,"Speech2Text"),Gs.forEach(t),rn.forEach(t),F=p(o),q=a(o,"H2",{class:!0});var an=i(q);D=a(an,"A",{id:!0,class:!0,href:!0});var Us=i(D);R=a(Us,"SPAN",{});var Ws=i(R);k(ee.$$.fragment,Ws),Ws.forEach(t),Us.forEach(t),Ce=p(an),H=a(an,"SPAN",{});var Vs=i(H);Pe=s(Vs,"Overview"),Vs.forEach(t),an.forEach(t),ye=p(o),N=a(o,"P",{});var ke=i(N);B=s(ke,"The Speech2Text model was proposed in "),te=a(ke,"A",{href:!0,rel:!0});var Rs=i(te);Te=s(Rs,"fairseq S2T: Fast Speech-to-Text Modeling with fairseq"),Rs.forEach(t),j=s(ke,` by Changhan Wang, Yun Tang, Xutai Ma, Anne Wu, Dmytro Okhonko, Juan Pino. It\u2019s a
transformer-based seq2seq (encoder-decoder) model designed for end-to-end Automatic Speech Recognition (ASR) and Speech
Translation (ST). It uses a convolutional downsampler to reduce the length of speech inputs by 3/4th before they are
fed into the encoder. The model is trained with standard autoregressive cross-entropy loss and generates the
transcripts/translations autoregressively. Speech2Text has been fine-tuned on several datasets for ASR and ST:
`),M=a(ke,"A",{href:!0,rel:!0});var Hs=i(M);je=s(Hs,"LibriSpeech"),Hs.forEach(t),oe=s(ke,", "),ne=a(ke,"A",{href:!0,rel:!0});var Bs=i(ne);Me=s(Bs,"CoVoST 2"),Bs.forEach(t),se=s(ke,", "),re=a(ke,"A",{href:!0,rel:!0});var Ks=i(re);Ne=s(Ks,"MuST-C"),Ks.forEach(t),O=s(ke,"."),ke.forEach(t),we=p(o),I=a(o,"P",{});var Dn=i(I);Ae=s(Dn,"This model was contributed by "),ae=a(Dn,"A",{href:!0,rel:!0});var Zc=i(ae);ie=s(Zc,"valhalla"),Zc.forEach(t),De=s(Dn,". The original code can be found "),de=a(Dn,"A",{href:!0,rel:!0});var el=i(de);K=s(el,"here"),el.forEach(t),Ie=s(Dn,"."),Dn.forEach(t),Q=p(o),J=a(o,"H2",{class:!0});var xr=i(J);T=a(xr,"A",{id:!0,class:!0,href:!0});var tl=i(T);z=a(tl,"SPAN",{});var ol=i(z);k(Y.$$.fragment,ol),ol.forEach(t),tl.forEach(t),Ze=p(xr),Le=a(xr,"SPAN",{});var nl=i(Le);X=s(nl,"Inference"),nl.forEach(t),xr.forEach(t),Ve=p(o),ue=a(o,"P",{});var br=i(ue);et=s(br,`Speech2Text is a speech model that accepts a float tensor of log-mel filter-bank features extracted from the speech
signal. It\u2019s a transformer-based seq2seq model, so the transcripts/translations are generated autoregressively. The
`),U=a(br,"CODE",{});var sl=i(U);ce=s(sl,"generate()"),sl.forEach(t),tt=s(br," method can be used for inference."),br.forEach(t),Re=p(o),P=a(o,"P",{});var Je=i(P);ot=s(Je,"The "),Se=a(Je,"A",{href:!0});var rl=i(Se);Oe=s(rl,"Speech2TextFeatureExtractor"),rl.forEach(t),nt=s(Je,` class is responsible for extracting the log-mel filter-bank
features. The `),cn=a(Je,"A",{href:!0});var al=i(cn);Qr=s(al,"Speech2TextProcessor"),al.forEach(t),Zr=s(Je," wraps "),ln=a(Je,"A",{href:!0});var il=i(ln);ea=s(il,"Speech2TextFeatureExtractor"),il.forEach(t),ta=s(Je,` and
`),pn=a(Je,"A",{href:!0});var dl=i(pn);oa=s(dl,"Speech2TextTokenizer"),dl.forEach(t),na=s(Je,` into a single instance to both extract the input features and decode the
predicted token ids.`),Je.forEach(t),Xs=p(o),W=a(o,"P",{});var fe=i(W);sa=s(fe,"The feature extractor depends on "),Bn=a(fe,"CODE",{});var cl=i(Bn);ra=s(cl,"torchaudio"),cl.forEach(t),aa=s(fe," and the tokenizer depends on "),Kn=a(fe,"CODE",{});var ll=i(Kn);ia=s(ll,"sentencepiece"),ll.forEach(t),da=s(fe,` so be sure to
install those packages before running the examples. You could either install those as extra speech dependencies with
`),Jn=a(fe,"CODE",{});var pl=i(Jn);ca=s(pl,'pip install transformers"[speech, sentencepiece]"'),pl.forEach(t),la=s(fe," or install the packages separately with "),Yn=a(fe,"CODE",{});var hl=i(Yn);pa=s(hl,"pip install torchaudio sentencepiece"),hl.forEach(t),ha=s(fe,". Also "),Xn=a(fe,"CODE",{});var ml=i(Xn);ma=s(ml,"torchaudio"),ml.forEach(t),fa=s(fe," requires the development version of the "),to=a(fe,"A",{href:!0,rel:!0});var fl=i(to);ua=s(fl,"libsndfile"),fl.forEach(t),_a=s(fe,` package which can be installed via a system package manager. On Ubuntu it can
be installed as follows: `),Qn=a(fe,"CODE",{});var ul=i(Qn);ga=s(ul,"apt install libsndfile1-dev"),ul.forEach(t),fe.forEach(t),Qs=p(o),hn=a(o,"UL",{});var _l=i(hn);Zn=a(_l,"LI",{});var gl=i(Zn);Ta=s(gl,"ASR and Speech Translation"),gl.forEach(t),_l.forEach(t),Zs=p(o),k(oo.$$.fragment,o),er=p(o),mn=a(o,"UL",{});var Tl=i(mn);no=a(Tl,"LI",{});var kr=i(no);es=a(kr,"P",{});var vl=i(es);va=s(vl,"Multilingual speech translation"),vl.forEach(t),xa=p(kr),le=a(kr,"P",{});var ze=i(le);ba=s(ze,"For multilingual speech translation models, "),ts=a(ze,"CODE",{});var xl=i(ts);ka=s(xl,"eos_token_id"),xl.forEach(t),ya=s(ze," is used as the "),os=a(ze,"CODE",{});var bl=i(os);wa=s(bl,"decoder_start_token_id"),bl.forEach(t),Sa=s(ze,` and
the target language id is forced as the first generated token. To force the target language id as the first
generated token, pass the `),ns=a(ze,"CODE",{});var kl=i(ns);$a=s(kl,"forced_bos_token_id"),kl.forEach(t),Ea=s(ze," parameter to the "),ss=a(ze,"CODE",{});var yl=i(ss);qa=s(yl,"generate()"),yl.forEach(t),Fa=s(ze,` method. The following
example shows how to transate English speech to French text using the `),rs=a(ze,"EM",{});var wl=i(rs);za=s(wl,"facebook/s2t-medium-mustc-multilingual-st"),wl.forEach(t),Ca=s(ze,`
checkpoint.`),ze.forEach(t),kr.forEach(t),Tl.forEach(t),tr=p(o),k(so.$$.fragment,o),or=p(o),Tt=a(o,"P",{});var yr=i(Tt);Pa=s(yr,"See the "),ro=a(yr,"A",{href:!0,rel:!0});var Sl=i(ro);ja=s(Sl,"model hub"),Sl.forEach(t),Ma=s(yr," to look for Speech2Text checkpoints."),yr.forEach(t),nr=p(o),st=a(o,"H2",{class:!0});var wr=i(st);vt=a(wr,"A",{id:!0,class:!0,href:!0});var $l=i(vt);as=a($l,"SPAN",{});var El=i(as);k(ao.$$.fragment,El),El.forEach(t),$l.forEach(t),Na=p(wr),is=a(wr,"SPAN",{});var ql=i(is);Aa=s(ql,"Speech2TextConfig"),ql.forEach(t),wr.forEach(t),sr=p(o),ve=a(o,"DIV",{class:!0});var Ht=i(ve);k(io.$$.fragment,Ht),Da=p(Ht),rt=a(Ht,"P",{});var In=i(rt);Ia=s(In,"This is the configuration class to store the configuration of a "),fn=a(In,"A",{href:!0});var Fl=i(fn);La=s(Fl,"Speech2TextModel"),Fl.forEach(t),Oa=s(In,`. It is used to instantiate an
Speech2Text model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the Speech2Text
`),co=a(In,"A",{href:!0,rel:!0});var zl=i(co);Ga=s(zl,"facebook/s2t-small-librispeech-asr"),zl.forEach(t),Ua=s(In," architecture."),In.forEach(t),Wa=p(Ht),at=a(Ht,"P",{});var Ln=i(at);Va=s(Ln,"Configuration objects inherit from "),un=a(Ln,"A",{href:!0});var Cl=i(un);Ra=s(Cl,"PretrainedConfig"),Cl.forEach(t),Ha=s(Ln,` and can be used to control the model outputs. Read the
documentation from `),_n=a(Ln,"A",{href:!0});var Pl=i(_n);Ba=s(Pl,"PretrainedConfig"),Pl.forEach(t),Ka=s(Ln," for more information."),Ln.forEach(t),Ja=p(Ht),k(xt.$$.fragment,Ht),Ht.forEach(t),rr=p(o),it=a(o,"H2",{class:!0});var Sr=i(it);bt=a(Sr,"A",{id:!0,class:!0,href:!0});var jl=i(bt);ds=a(jl,"SPAN",{});var Ml=i(ds);k(lo.$$.fragment,Ml),Ml.forEach(t),jl.forEach(t),Ya=p(Sr),cs=a(Sr,"SPAN",{});var Nl=i(cs);Xa=s(Nl,"Speech2TextTokenizer"),Nl.forEach(t),Sr.forEach(t),ar=p(o),V=a(o,"DIV",{class:!0});var ge=i(V);k(po.$$.fragment,ge),Qa=p(ge),ls=a(ge,"P",{});var Al=i(ls);Za=s(Al,"Construct an Speech2Text tokenizer."),Al.forEach(t),ei=p(ge),ho=a(ge,"P",{});var $r=i(ho);ti=s($r,"This tokenizer inherits from "),gn=a($r,"A",{href:!0});var Dl=i(gn);oi=s(Dl,"PreTrainedTokenizer"),Dl.forEach(t),ni=s($r,` which contains some of the main methods. Users should refer to
the superclass for more information regarding such methods.`),$r.forEach(t),si=p(ge),kt=a(ge,"DIV",{class:!0});var Er=i(kt);k(mo.$$.fragment,Er),ri=p(Er),ps=a(Er,"P",{});var Il=i(ps);ai=s(Il,"Build model inputs from a sequence by appending eos_token_id."),Il.forEach(t),Er.forEach(t),ii=p(ge),yt=a(ge,"DIV",{class:!0});var qr=i(yt);k(fo.$$.fragment,qr),di=p(qr),uo=a(qr,"P",{});var Fr=i(uo);ci=s(Fr,`Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer `),hs=a(Fr,"CODE",{});var Ll=i(hs);li=s(Ll,"prepare_for_model"),Ll.forEach(t),pi=s(Fr," method."),Fr.forEach(t),qr.forEach(t),hi=p(ge),He=a(ge,"DIV",{class:!0});var On=i(He);k(_o.$$.fragment,On),mi=p(On),Tn=a(On,"P",{});var Jc=i(Tn);fi=s(Jc,"Create the token type IDs corresponding to the sequences passed. "),vn=a(Jc,"A",{href:!0});var Ol=i(vn);ui=s(Ol,`What are token type
IDs?`),Ol.forEach(t),Jc.forEach(t),_i=p(On),ms=a(On,"P",{});var Gl=i(ms);gi=s(Gl,"Should be overridden in a subclass if the model has a special way of building those."),Gl.forEach(t),On.forEach(t),Ti=p(ge),xn=a(ge,"DIV",{class:!0});var Ul=i(xn);k(go.$$.fragment,Ul),Ul.forEach(t),ge.forEach(t),ir=p(o),dt=a(o,"H2",{class:!0});var zr=i(dt);wt=a(zr,"A",{id:!0,class:!0,href:!0});var Wl=i(wt);fs=a(Wl,"SPAN",{});var Vl=i(fs);k(To.$$.fragment,Vl),Vl.forEach(t),Wl.forEach(t),vi=p(zr),us=a(zr,"SPAN",{});var Rl=i(us);xi=s(Rl,"Speech2TextFeatureExtractor"),Rl.forEach(t),zr.forEach(t),dr=p(o),pe=a(o,"DIV",{class:!0});var Ye=i(pe);k(vo.$$.fragment,Ye),bi=p(Ye),_s=a(Ye,"P",{});var Hl=i(_s);ki=s(Hl,"Constructs a Speech2Text feature extractor."),Hl.forEach(t),yi=p(Ye),xo=a(Ye,"P",{});var Cr=i(xo);wi=s(Cr,"This feature extractor inherits from "),bn=a(Cr,"A",{href:!0});var Bl=i(bn);Si=s(Bl,"Speech2TextFeatureExtractor"),Bl.forEach(t),$i=s(Cr,` which contains most of the main methods. Users
should refer to this superclass for more information regarding those methods.`),Cr.forEach(t),Ei=p(Ye),gs=a(Ye,"P",{});var Kl=i(gs);qi=s(Kl,`This class extracts mel-filter bank features from raw speech using TorchAudio and applies utterance-level cepstral
mean and variance normalization to the extracted features.`),Kl.forEach(t),Fi=p(Ye),St=a(Ye,"DIV",{class:!0});var Pr=i(St);k(bo.$$.fragment,Pr),zi=p(Pr),Ts=a(Pr,"P",{});var Jl=i(Ts);Ci=s(Jl,"Main method to featurize and prepare for the model one or several sequence(s). sequences."),Jl.forEach(t),Pr.forEach(t),Ye.forEach(t),cr=p(o),ct=a(o,"H2",{class:!0});var jr=i(ct);$t=a(jr,"A",{id:!0,class:!0,href:!0});var Yl=i($t);vs=a(Yl,"SPAN",{});var Xl=i(vs);k(ko.$$.fragment,Xl),Xl.forEach(t),Yl.forEach(t),Pi=p(jr),xs=a(jr,"SPAN",{});var Ql=i(xs);ji=s(Ql,"Speech2TextProcessor"),Ql.forEach(t),jr.forEach(t),lr=p(o),G=a(o,"DIV",{class:!0});var Z=i(G);k(yo.$$.fragment,Z),Mi=p(Z),bs=a(Z,"P",{});var Zl=i(bs);Ni=s(Zl,`Constructs a Speech2Text processor which wraps a Speech2Text feature extractor and a Speech2Text tokenizer into a
single processor.`),Zl.forEach(t),Ai=p(Z),_e=a(Z,"P",{});var Ue=i(_e);kn=a(Ue,"A",{href:!0});var ep=i(kn);Di=s(ep,"Speech2TextProcessor"),ep.forEach(t),Ii=s(Ue," offers all the functionalities of "),yn=a(Ue,"A",{href:!0});var tp=i(yn);Li=s(tp,"Speech2TextFeatureExtractor"),tp.forEach(t),Oi=s(Ue,` and
`),wn=a(Ue,"A",{href:!0});var op=i(wn);Gi=s(op,"Speech2TextTokenizer"),op.forEach(t),Ui=s(Ue,". See the "),wo=a(Ue,"A",{href:!0});var Yc=i(wo);ks=a(Yc,"STRONG",{});var np=i(ks);Wi=s(np,"call"),np.forEach(t),Vi=s(Yc,"()"),Yc.forEach(t),Ri=s(Ue," and "),Sn=a(Ue,"A",{href:!0});var sp=i(Sn);Hi=s(sp,"decode()"),sp.forEach(t),Bi=s(Ue,` for more
information.`),Ue.forEach(t),Ki=p(Z),Et=a(Z,"DIV",{class:!0});var Mr=i(Et);k(So.$$.fragment,Mr),Ji=p(Mr),Ge=a(Mr,"P",{});var Bt=i(Ge);Yi=s(Bt,`When used in normal mode, this method forwards all its arguments to Speech2TextFeatureExtractor\u2019s
`),$o=a(Bt,"A",{href:!0});var Xc=i($o);ys=a(Xc,"STRONG",{});var rp=i(ys);Xi=s(rp,"call"),rp.forEach(t),Qi=s(Xc,"()"),Xc.forEach(t),Zi=s(Bt,` and returns its output. If used in the context
`),ws=a(Bt,"CODE",{});var ap=i(ws);ed=s(ap,"as_target_processor()"),ap.forEach(t),td=s(Bt,` this method forwards all its arguments to Speech2TextTokenizer\u2019s
`),Eo=a(Bt,"A",{href:!0});var Qc=i(Eo);Ss=a(Qc,"STRONG",{});var ip=i(Ss);od=s(ip,"call"),ip.forEach(t),nd=s(Qc,"()"),Qc.forEach(t),sd=s(Bt,`. Please refer to the doctsring of the above two methods for more
information.`),Bt.forEach(t),Mr.forEach(t),rd=p(Z),Be=a(Z,"DIV",{class:!0});var Gn=i(Be);k(qo.$$.fragment,Gn),ad=p(Gn),$s=a(Gn,"P",{});var dp=i($s);id=s(dp,"Instantiate a processor associated with a pretrained model."),dp.forEach(t),dd=p(Gn),k(qt.$$.fragment,Gn),Gn.forEach(t),cd=p(Z),Ke=a(Z,"DIV",{class:!0});var Un=i(Ke);k(Fo.$$.fragment,Un),ld=p(Un),zo=a(Un,"P",{});var Nr=i(zo);pd=s(Nr,`Saves the attributes of this processor (feature extractor, tokenizer\u2026) in the specified directory so that it
can be reloaded using the `),$n=a(Nr,"A",{href:!0});var cp=i($n);hd=s(cp,"from_pretrained()"),cp.forEach(t),md=s(Nr," method."),Nr.forEach(t),fd=p(Un),k(Ft.$$.fragment,Un),Un.forEach(t),ud=p(Z),zt=a(Z,"DIV",{class:!0});var Ar=i(zt);k(Co.$$.fragment,Ar),_d=p(Ar),Po=a(Ar,"P",{});var Dr=i(Po);gd=s(Dr,"This method forwards all its arguments to Speech2TextTokenizer\u2019s "),En=a(Dr,"A",{href:!0});var lp=i(En);Td=s(lp,"batch_decode()"),lp.forEach(t),vd=s(Dr,`. Please
refer to the docstring of this method for more information.`),Dr.forEach(t),Ar.forEach(t),xd=p(Z),Ct=a(Z,"DIV",{class:!0});var Ir=i(Ct);k(jo.$$.fragment,Ir),bd=p(Ir),Mo=a(Ir,"P",{});var Lr=i(Mo);kd=s(Lr,"This method forwards all its arguments to Speech2TextTokenizer\u2019s "),qn=a(Lr,"A",{href:!0});var pp=i(qn);yd=s(pp,"decode()"),pp.forEach(t),wd=s(Lr,`. Please refer
to the docstring of this method for more information.`),Lr.forEach(t),Ir.forEach(t),Z.forEach(t),pr=p(o),lt=a(o,"H2",{class:!0});var Or=i(lt);Pt=a(Or,"A",{id:!0,class:!0,href:!0});var hp=i(Pt);Es=a(hp,"SPAN",{});var mp=i(Es);k(No.$$.fragment,mp),mp.forEach(t),hp.forEach(t),Sd=p(Or),qs=a(Or,"SPAN",{});var fp=i(qs);$d=s(fp,"Speech2TextModel"),fp.forEach(t),Or.forEach(t),hr=p(o),xe=a(o,"DIV",{class:!0});var Kt=i(xe);k(Ao.$$.fragment,Kt),Ed=p(Kt),Do=a(Kt,"P",{});var Gr=i(Do);qd=s(Gr,`The bare Speech2Text Model outputting raw hidden-states without any specific head on top.
This model inherits from `),Fn=a(Gr,"A",{href:!0});var up=i(Fn);Fd=s(up,"PreTrainedModel"),up.forEach(t),zd=s(Gr,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Gr.forEach(t),Cd=p(Kt),Io=a(Kt,"P",{});var Ur=i(Io);Pd=s(Ur,"This model is also a PyTorch "),Lo=a(Ur,"A",{href:!0,rel:!0});var _p=i(Lo);jd=s(_p,"torch.nn.Module"),_p.forEach(t),Md=s(Ur,` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Ur.forEach(t),Nd=p(Kt),$e=a(Kt,"DIV",{class:!0});var Jt=i($e);k(Oo.$$.fragment,Jt),Ad=p(Jt),pt=a(Jt,"P",{});var Wn=i(pt);Dd=s(Wn,"The "),zn=a(Wn,"A",{href:!0});var gp=i(zn);Id=s(gp,"Speech2TextModel"),gp.forEach(t),Ld=s(Wn," forward method, overrides the "),Fs=a(Wn,"CODE",{});var Tp=i(Fs);Od=s(Tp,"__call__"),Tp.forEach(t),Gd=s(Wn," special method."),Wn.forEach(t),Ud=p(Jt),k(jt.$$.fragment,Jt),Wd=p(Jt),k(Mt.$$.fragment,Jt),Jt.forEach(t),Kt.forEach(t),mr=p(o),ht=a(o,"H2",{class:!0});var Wr=i(ht);Nt=a(Wr,"A",{id:!0,class:!0,href:!0});var vp=i(Nt);zs=a(vp,"SPAN",{});var xp=i(zs);k(Go.$$.fragment,xp),xp.forEach(t),vp.forEach(t),Vd=p(Wr),Cs=a(Wr,"SPAN",{});var bp=i(Cs);Rd=s(bp,"Speech2TextForConditionalGeneration"),bp.forEach(t),Wr.forEach(t),fr=p(o),be=a(o,"DIV",{class:!0});var Yt=i(be);k(Uo.$$.fragment,Yt),Hd=p(Yt),Wo=a(Yt,"P",{});var Vr=i(Wo);Bd=s(Vr,`The Speech2Text Model with a language modeling head. Can be used for summarization.
This model inherits from `),Cn=a(Vr,"A",{href:!0});var kp=i(Cn);Kd=s(kp,"PreTrainedModel"),kp.forEach(t),Jd=s(Vr,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Vr.forEach(t),Yd=p(Yt),Vo=a(Yt,"P",{});var Rr=i(Vo);Xd=s(Rr,"This model is also a PyTorch "),Ro=a(Rr,"A",{href:!0,rel:!0});var yp=i(Ro);Qd=s(yp,"torch.nn.Module"),yp.forEach(t),Zd=s(Rr,` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Rr.forEach(t),ec=p(Yt),Ee=a(Yt,"DIV",{class:!0});var Xt=i(Ee);k(Ho.$$.fragment,Xt),tc=p(Xt),mt=a(Xt,"P",{});var Vn=i(mt);oc=s(Vn,"The "),Pn=a(Vn,"A",{href:!0});var wp=i(Pn);nc=s(wp,"Speech2TextForConditionalGeneration"),wp.forEach(t),sc=s(Vn," forward method, overrides the "),Ps=a(Vn,"CODE",{});var Sp=i(Ps);rc=s(Sp,"__call__"),Sp.forEach(t),ac=s(Vn," special method."),Vn.forEach(t),ic=p(Xt),k(At.$$.fragment,Xt),dc=p(Xt),k(Dt.$$.fragment,Xt),Xt.forEach(t),Yt.forEach(t),ur=p(o),ft=a(o,"H2",{class:!0});var Hr=i(ft);It=a(Hr,"A",{id:!0,class:!0,href:!0});var $p=i(It);js=a($p,"SPAN",{});var Ep=i(js);k(Bo.$$.fragment,Ep),Ep.forEach(t),$p.forEach(t),cc=p(Hr),Ms=a(Hr,"SPAN",{});var qp=i(Ms);lc=s(qp,"TFSpeech2TextModel"),qp.forEach(t),Hr.forEach(t),_r=p(o),he=a(o,"DIV",{class:!0});var Xe=i(he);k(Ko.$$.fragment,Xe),pc=p(Xe),Jo=a(Xe,"P",{});var Br=i(Jo);hc=s(Br,`The bare Speech2Text Model outputting raw hidden-states without any specific head on top.
This model inherits from `),jn=a(Br,"A",{href:!0});var Fp=i(jn);mc=s(Fp,"TFPreTrainedModel"),Fp.forEach(t),fc=s(Br,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Br.forEach(t),uc=p(Xe),Yo=a(Xe,"P",{});var Kr=i(Yo);_c=s(Kr,"This model is also a "),Xo=a(Kr,"A",{href:!0,rel:!0});var zp=i(Xo);gc=s(zp,"tf.keras.Model"),zp.forEach(t),Tc=s(Kr,` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),Kr.forEach(t),vc=p(Xe),k(Lt.$$.fragment,Xe),xc=p(Xe),qe=a(Xe,"DIV",{class:!0});var Qt=i(qe);k(Qo.$$.fragment,Qt),bc=p(Qt),ut=a(Qt,"P",{});var Rn=i(ut);kc=s(Rn,"The "),Mn=a(Rn,"A",{href:!0});var Cp=i(Mn);yc=s(Cp,"TFSpeech2TextModel"),Cp.forEach(t),wc=s(Rn," forward method, overrides the "),Ns=a(Rn,"CODE",{});var Pp=i(Ns);Sc=s(Pp,"__call__"),Pp.forEach(t),$c=s(Rn," special method."),Rn.forEach(t),Ec=p(Qt),k(Ot.$$.fragment,Qt),qc=p(Qt),k(Gt.$$.fragment,Qt),Qt.forEach(t),Xe.forEach(t),gr=p(o),_t=a(o,"H2",{class:!0});var Jr=i(_t);Ut=a(Jr,"A",{id:!0,class:!0,href:!0});var jp=i(Ut);As=a(jp,"SPAN",{});var Mp=i(As);k(Zo.$$.fragment,Mp),Mp.forEach(t),jp.forEach(t),Fc=p(Jr),Ds=a(Jr,"SPAN",{});var Np=i(Ds);zc=s(Np,"TFSpeech2TextForConditionalGeneration"),Np.forEach(t),Jr.forEach(t),Tr=p(o),me=a(o,"DIV",{class:!0});var Qe=i(me);k(en.$$.fragment,Qe),Cc=p(Qe),tn=a(Qe,"P",{});var Yr=i(tn);Pc=s(Yr,`The Speech2Text Model with a language modeling head. Can be used for summarization.
This model inherits from `),Nn=a(Yr,"A",{href:!0});var Ap=i(Nn);jc=s(Ap,"TFPreTrainedModel"),Ap.forEach(t),Mc=s(Yr,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Yr.forEach(t),Nc=p(Qe),on=a(Qe,"P",{});var Xr=i(on);Ac=s(Xr,"This model is also a "),nn=a(Xr,"A",{href:!0,rel:!0});var Dp=i(nn);Dc=s(Dp,"tf.keras.Model"),Dp.forEach(t),Ic=s(Xr,` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),Xr.forEach(t),Lc=p(Qe),k(Wt.$$.fragment,Qe),Oc=p(Qe),Fe=a(Qe,"DIV",{class:!0});var Zt=i(Fe);k(sn.$$.fragment,Zt),Gc=p(Zt),gt=a(Zt,"P",{});var Hn=i(gt);Uc=s(Hn,"The "),An=a(Hn,"A",{href:!0});var Ip=i(An);Wc=s(Ip,"TFSpeech2TextForConditionalGeneration"),Ip.forEach(t),Vc=s(Hn," forward method, overrides the "),Is=a(Hn,"CODE",{});var Lp=i(Is);Rc=s(Lp,"__call__"),Lp.forEach(t),Hc=s(Hn," special method."),Hn.forEach(t),Bc=p(Zt),k(Vt.$$.fragment,Zt),Kc=p(Zt),k(Rt.$$.fragment,Zt),Zt.forEach(t),Qe.forEach(t),this.h()},h(){c(h,"name","hf:doc:metadata"),c(h,"content",JSON.stringify(rh)),c(m,"id","speech2text"),c(m,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(m,"href","#speech2text"),c(_,"class","relative group"),c(D,"id","overview"),c(D,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(D,"href","#overview"),c(q,"class","relative group"),c(te,"href","https://arxiv.org/abs/2010.05171"),c(te,"rel","nofollow"),c(M,"href","http://www.openslr.org/12"),c(M,"rel","nofollow"),c(ne,"href","https://github.com/facebookresearch/covost"),c(ne,"rel","nofollow"),c(re,"href","https://ict.fbk.eu/must-c/"),c(re,"rel","nofollow"),c(ae,"href","https://huggingface.co/valhalla"),c(ae,"rel","nofollow"),c(de,"href","https://github.com/pytorch/fairseq/tree/master/examples/speech_to_text"),c(de,"rel","nofollow"),c(T,"id","inference"),c(T,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(T,"href","#inference"),c(J,"class","relative group"),c(Se,"href","/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor"),c(cn,"href","/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextProcessor"),c(ln,"href","/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor"),c(pn,"href","/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextTokenizer"),c(to,"href","http://www.mega-nerd.com/libsndfile/"),c(to,"rel","nofollow"),c(ro,"href","https://huggingface.co/models?filter=speech_to_text"),c(ro,"rel","nofollow"),c(vt,"id","transformers.Speech2TextConfig"),c(vt,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(vt,"href","#transformers.Speech2TextConfig"),c(st,"class","relative group"),c(fn,"href","/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextModel"),c(co,"href","https://huggingface.co/facebook/s2t-small-librispeech-asr"),c(co,"rel","nofollow"),c(un,"href","/docs/transformers/pr_18590/en/main_classes/configuration#transformers.PretrainedConfig"),c(_n,"href","/docs/transformers/pr_18590/en/main_classes/configuration#transformers.PretrainedConfig"),c(ve,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(bt,"id","transformers.Speech2TextTokenizer"),c(bt,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(bt,"href","#transformers.Speech2TextTokenizer"),c(it,"class","relative group"),c(gn,"href","/docs/transformers/pr_18590/en/main_classes/tokenizer#transformers.PreTrainedTokenizer"),c(kt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(yt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(vn,"href","../glossary#token-type-ids"),c(He,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(xn,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(wt,"id","transformers.Speech2TextFeatureExtractor"),c(wt,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(wt,"href","#transformers.Speech2TextFeatureExtractor"),c(dt,"class","relative group"),c(bn,"href","/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor"),c(St,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(pe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c($t,"id","transformers.Speech2TextProcessor"),c($t,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c($t,"href","#transformers.Speech2TextProcessor"),c(ct,"class","relative group"),c(kn,"href","/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextProcessor"),c(yn,"href","/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor"),c(wn,"href","/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextTokenizer"),c(wo,"href","/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextProcessor.__call__"),c(Sn,"href","/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextProcessor.decode"),c($o,"href","/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor.__call__"),c(Eo,"href","/docs/transformers/pr_18590/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__"),c(Et,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(Be,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c($n,"href","/docs/transformers/pr_18590/en/model_doc/speech_to_text_2#transformers.Speech2Text2Processor.from_pretrained"),c(Ke,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(En,"href","/docs/transformers/pr_18590/en/model_doc/speech_to_text_2#transformers.Speech2Text2Tokenizer.batch_decode"),c(zt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(qn,"href","/docs/transformers/pr_18590/en/model_doc/speech_to_text_2#transformers.Speech2Text2Tokenizer.decode"),c(Ct,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(Pt,"id","transformers.Speech2TextModel"),c(Pt,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(Pt,"href","#transformers.Speech2TextModel"),c(lt,"class","relative group"),c(Fn,"href","/docs/transformers/pr_18590/en/main_classes/model#transformers.PreTrainedModel"),c(Lo,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),c(Lo,"rel","nofollow"),c(zn,"href","/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextModel"),c($e,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(xe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(Nt,"id","transformers.Speech2TextForConditionalGeneration"),c(Nt,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(Nt,"href","#transformers.Speech2TextForConditionalGeneration"),c(ht,"class","relative group"),c(Cn,"href","/docs/transformers/pr_18590/en/main_classes/model#transformers.PreTrainedModel"),c(Ro,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),c(Ro,"rel","nofollow"),c(Pn,"href","/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.Speech2TextForConditionalGeneration"),c(Ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(be,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(It,"id","transformers.TFSpeech2TextModel"),c(It,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(It,"href","#transformers.TFSpeech2TextModel"),c(ft,"class","relative group"),c(jn,"href","/docs/transformers/pr_18590/en/main_classes/model#transformers.TFPreTrainedModel"),c(Xo,"href","https://www.tensorflow.org/api_docs/python/tf/keras/Model"),c(Xo,"rel","nofollow"),c(Mn,"href","/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.TFSpeech2TextModel"),c(qe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(he,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(Ut,"id","transformers.TFSpeech2TextForConditionalGeneration"),c(Ut,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(Ut,"href","#transformers.TFSpeech2TextForConditionalGeneration"),c(_t,"class","relative group"),c(Nn,"href","/docs/transformers/pr_18590/en/main_classes/model#transformers.TFPreTrainedModel"),c(nn,"href","https://www.tensorflow.org/api_docs/python/tf/keras/Model"),c(nn,"rel","nofollow"),c(An,"href","/docs/transformers/pr_18590/en/model_doc/speech_to_text#transformers.TFSpeech2TextForConditionalGeneration"),c(Fe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(me,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(o,u){e(document.head,h),f(o,x,u),f(o,_,u),e(_,m),e(m,v),y(d,v,null),e(_,g),e(_,E),e(E,A),f(o,F,u),f(o,q,u),e(q,D),e(D,R),y(ee,R,null),e(q,Ce),e(q,H),e(H,Pe),f(o,ye,u),f(o,N,u),e(N,B),e(N,te),e(te,Te),e(N,j),e(N,M),e(M,je),e(N,oe),e(N,ne),e(ne,Me),e(N,se),e(N,re),e(re,Ne),e(N,O),f(o,we,u),f(o,I,u),e(I,Ae),e(I,ae),e(ae,ie),e(I,De),e(I,de),e(de,K),e(I,Ie),f(o,Q,u),f(o,J,u),e(J,T),e(T,z),y(Y,z,null),e(J,Ze),e(J,Le),e(Le,X),f(o,Ve,u),f(o,ue,u),e(ue,et),e(ue,U),e(U,ce),e(ue,tt),f(o,Re,u),f(o,P,u),e(P,ot),e(P,Se),e(Se,Oe),e(P,nt),e(P,cn),e(cn,Qr),e(P,Zr),e(P,ln),e(ln,ea),e(P,ta),e(P,pn),e(pn,oa),e(P,na),f(o,Xs,u),f(o,W,u),e(W,sa),e(W,Bn),e(Bn,ra),e(W,aa),e(W,Kn),e(Kn,ia),e(W,da),e(W,Jn),e(Jn,ca),e(W,la),e(W,Yn),e(Yn,pa),e(W,ha),e(W,Xn),e(Xn,ma),e(W,fa),e(W,to),e(to,ua),e(W,_a),e(W,Qn),e(Qn,ga),f(o,Qs,u),f(o,hn,u),e(hn,Zn),e(Zn,Ta),f(o,Zs,u),y(oo,o,u),f(o,er,u),f(o,mn,u),e(mn,no),e(no,es),e(es,va),e(no,xa),e(no,le),e(le,ba),e(le,ts),e(ts,ka),e(le,ya),e(le,os),e(os,wa),e(le,Sa),e(le,ns),e(ns,$a),e(le,Ea),e(le,ss),e(ss,qa),e(le,Fa),e(le,rs),e(rs,za),e(le,Ca),f(o,tr,u),y(so,o,u),f(o,or,u),f(o,Tt,u),e(Tt,Pa),e(Tt,ro),e(ro,ja),e(Tt,Ma),f(o,nr,u),f(o,st,u),e(st,vt),e(vt,as),y(ao,as,null),e(st,Na),e(st,is),e(is,Aa),f(o,sr,u),f(o,ve,u),y(io,ve,null),e(ve,Da),e(ve,rt),e(rt,Ia),e(rt,fn),e(fn,La),e(rt,Oa),e(rt,co),e(co,Ga),e(rt,Ua),e(ve,Wa),e(ve,at),e(at,Va),e(at,un),e(un,Ra),e(at,Ha),e(at,_n),e(_n,Ba),e(at,Ka),e(ve,Ja),y(xt,ve,null),f(o,rr,u),f(o,it,u),e(it,bt),e(bt,ds),y(lo,ds,null),e(it,Ya),e(it,cs),e(cs,Xa),f(o,ar,u),f(o,V,u),y(po,V,null),e(V,Qa),e(V,ls),e(ls,Za),e(V,ei),e(V,ho),e(ho,ti),e(ho,gn),e(gn,oi),e(ho,ni),e(V,si),e(V,kt),y(mo,kt,null),e(kt,ri),e(kt,ps),e(ps,ai),e(V,ii),e(V,yt),y(fo,yt,null),e(yt,di),e(yt,uo),e(uo,ci),e(uo,hs),e(hs,li),e(uo,pi),e(V,hi),e(V,He),y(_o,He,null),e(He,mi),e(He,Tn),e(Tn,fi),e(Tn,vn),e(vn,ui),e(He,_i),e(He,ms),e(ms,gi),e(V,Ti),e(V,xn),y(go,xn,null),f(o,ir,u),f(o,dt,u),e(dt,wt),e(wt,fs),y(To,fs,null),e(dt,vi),e(dt,us),e(us,xi),f(o,dr,u),f(o,pe,u),y(vo,pe,null),e(pe,bi),e(pe,_s),e(_s,ki),e(pe,yi),e(pe,xo),e(xo,wi),e(xo,bn),e(bn,Si),e(xo,$i),e(pe,Ei),e(pe,gs),e(gs,qi),e(pe,Fi),e(pe,St),y(bo,St,null),e(St,zi),e(St,Ts),e(Ts,Ci),f(o,cr,u),f(o,ct,u),e(ct,$t),e($t,vs),y(ko,vs,null),e(ct,Pi),e(ct,xs),e(xs,ji),f(o,lr,u),f(o,G,u),y(yo,G,null),e(G,Mi),e(G,bs),e(bs,Ni),e(G,Ai),e(G,_e),e(_e,kn),e(kn,Di),e(_e,Ii),e(_e,yn),e(yn,Li),e(_e,Oi),e(_e,wn),e(wn,Gi),e(_e,Ui),e(_e,wo),e(wo,ks),e(ks,Wi),e(wo,Vi),e(_e,Ri),e(_e,Sn),e(Sn,Hi),e(_e,Bi),e(G,Ki),e(G,Et),y(So,Et,null),e(Et,Ji),e(Et,Ge),e(Ge,Yi),e(Ge,$o),e($o,ys),e(ys,Xi),e($o,Qi),e(Ge,Zi),e(Ge,ws),e(ws,ed),e(Ge,td),e(Ge,Eo),e(Eo,Ss),e(Ss,od),e(Eo,nd),e(Ge,sd),e(G,rd),e(G,Be),y(qo,Be,null),e(Be,ad),e(Be,$s),e($s,id),e(Be,dd),y(qt,Be,null),e(G,cd),e(G,Ke),y(Fo,Ke,null),e(Ke,ld),e(Ke,zo),e(zo,pd),e(zo,$n),e($n,hd),e(zo,md),e(Ke,fd),y(Ft,Ke,null),e(G,ud),e(G,zt),y(Co,zt,null),e(zt,_d),e(zt,Po),e(Po,gd),e(Po,En),e(En,Td),e(Po,vd),e(G,xd),e(G,Ct),y(jo,Ct,null),e(Ct,bd),e(Ct,Mo),e(Mo,kd),e(Mo,qn),e(qn,yd),e(Mo,wd),f(o,pr,u),f(o,lt,u),e(lt,Pt),e(Pt,Es),y(No,Es,null),e(lt,Sd),e(lt,qs),e(qs,$d),f(o,hr,u),f(o,xe,u),y(Ao,xe,null),e(xe,Ed),e(xe,Do),e(Do,qd),e(Do,Fn),e(Fn,Fd),e(Do,zd),e(xe,Cd),e(xe,Io),e(Io,Pd),e(Io,Lo),e(Lo,jd),e(Io,Md),e(xe,Nd),e(xe,$e),y(Oo,$e,null),e($e,Ad),e($e,pt),e(pt,Dd),e(pt,zn),e(zn,Id),e(pt,Ld),e(pt,Fs),e(Fs,Od),e(pt,Gd),e($e,Ud),y(jt,$e,null),e($e,Wd),y(Mt,$e,null),f(o,mr,u),f(o,ht,u),e(ht,Nt),e(Nt,zs),y(Go,zs,null),e(ht,Vd),e(ht,Cs),e(Cs,Rd),f(o,fr,u),f(o,be,u),y(Uo,be,null),e(be,Hd),e(be,Wo),e(Wo,Bd),e(Wo,Cn),e(Cn,Kd),e(Wo,Jd),e(be,Yd),e(be,Vo),e(Vo,Xd),e(Vo,Ro),e(Ro,Qd),e(Vo,Zd),e(be,ec),e(be,Ee),y(Ho,Ee,null),e(Ee,tc),e(Ee,mt),e(mt,oc),e(mt,Pn),e(Pn,nc),e(mt,sc),e(mt,Ps),e(Ps,rc),e(mt,ac),e(Ee,ic),y(At,Ee,null),e(Ee,dc),y(Dt,Ee,null),f(o,ur,u),f(o,ft,u),e(ft,It),e(It,js),y(Bo,js,null),e(ft,cc),e(ft,Ms),e(Ms,lc),f(o,_r,u),f(o,he,u),y(Ko,he,null),e(he,pc),e(he,Jo),e(Jo,hc),e(Jo,jn),e(jn,mc),e(Jo,fc),e(he,uc),e(he,Yo),e(Yo,_c),e(Yo,Xo),e(Xo,gc),e(Yo,Tc),e(he,vc),y(Lt,he,null),e(he,xc),e(he,qe),y(Qo,qe,null),e(qe,bc),e(qe,ut),e(ut,kc),e(ut,Mn),e(Mn,yc),e(ut,wc),e(ut,Ns),e(Ns,Sc),e(ut,$c),e(qe,Ec),y(Ot,qe,null),e(qe,qc),y(Gt,qe,null),f(o,gr,u),f(o,_t,u),e(_t,Ut),e(Ut,As),y(Zo,As,null),e(_t,Fc),e(_t,Ds),e(Ds,zc),f(o,Tr,u),f(o,me,u),y(en,me,null),e(me,Cc),e(me,tn),e(tn,Pc),e(tn,Nn),e(Nn,jc),e(tn,Mc),e(me,Nc),e(me,on),e(on,Ac),e(on,nn),e(nn,Dc),e(on,Ic),e(me,Lc),y(Wt,me,null),e(me,Oc),e(me,Fe),y(sn,Fe,null),e(Fe,Gc),e(Fe,gt),e(gt,Uc),e(gt,An),e(An,Wc),e(gt,Vc),e(gt,Is),e(Is,Rc),e(gt,Hc),e(Fe,Bc),y(Vt,Fe,null),e(Fe,Kc),y(Rt,Fe,null),vr=!0},p(o,[u]){const rn={};u&2&&(rn.$$scope={dirty:u,ctx:o}),xt.$set(rn);const Ls={};u&2&&(Ls.$$scope={dirty:u,ctx:o}),qt.$set(Ls);const Os={};u&2&&(Os.$$scope={dirty:u,ctx:o}),Ft.$set(Os);const Gs={};u&2&&(Gs.$$scope={dirty:u,ctx:o}),jt.$set(Gs);const an={};u&2&&(an.$$scope={dirty:u,ctx:o}),Mt.$set(an);const Us={};u&2&&(Us.$$scope={dirty:u,ctx:o}),At.$set(Us);const Ws={};u&2&&(Ws.$$scope={dirty:u,ctx:o}),Dt.$set(Ws);const Vs={};u&2&&(Vs.$$scope={dirty:u,ctx:o}),Lt.$set(Vs);const ke={};u&2&&(ke.$$scope={dirty:u,ctx:o}),Ot.$set(ke);const Rs={};u&2&&(Rs.$$scope={dirty:u,ctx:o}),Gt.$set(Rs);const Hs={};u&2&&(Hs.$$scope={dirty:u,ctx:o}),Wt.$set(Hs);const Bs={};u&2&&(Bs.$$scope={dirty:u,ctx:o}),Vt.$set(Bs);const Ks={};u&2&&(Ks.$$scope={dirty:u,ctx:o}),Rt.$set(Ks)},i(o){vr||(w(d.$$.fragment,o),w(ee.$$.fragment,o),w(Y.$$.fragment,o),w(oo.$$.fragment,o),w(so.$$.fragment,o),w(ao.$$.fragment,o),w(io.$$.fragment,o),w(xt.$$.fragment,o),w(lo.$$.fragment,o),w(po.$$.fragment,o),w(mo.$$.fragment,o),w(fo.$$.fragment,o),w(_o.$$.fragment,o),w(go.$$.fragment,o),w(To.$$.fragment,o),w(vo.$$.fragment,o),w(bo.$$.fragment,o),w(ko.$$.fragment,o),w(yo.$$.fragment,o),w(So.$$.fragment,o),w(qo.$$.fragment,o),w(qt.$$.fragment,o),w(Fo.$$.fragment,o),w(Ft.$$.fragment,o),w(Co.$$.fragment,o),w(jo.$$.fragment,o),w(No.$$.fragment,o),w(Ao.$$.fragment,o),w(Oo.$$.fragment,o),w(jt.$$.fragment,o),w(Mt.$$.fragment,o),w(Go.$$.fragment,o),w(Uo.$$.fragment,o),w(Ho.$$.fragment,o),w(At.$$.fragment,o),w(Dt.$$.fragment,o),w(Bo.$$.fragment,o),w(Ko.$$.fragment,o),w(Lt.$$.fragment,o),w(Qo.$$.fragment,o),w(Ot.$$.fragment,o),w(Gt.$$.fragment,o),w(Zo.$$.fragment,o),w(en.$$.fragment,o),w(Wt.$$.fragment,o),w(sn.$$.fragment,o),w(Vt.$$.fragment,o),w(Rt.$$.fragment,o),vr=!0)},o(o){S(d.$$.fragment,o),S(ee.$$.fragment,o),S(Y.$$.fragment,o),S(oo.$$.fragment,o),S(so.$$.fragment,o),S(ao.$$.fragment,o),S(io.$$.fragment,o),S(xt.$$.fragment,o),S(lo.$$.fragment,o),S(po.$$.fragment,o),S(mo.$$.fragment,o),S(fo.$$.fragment,o),S(_o.$$.fragment,o),S(go.$$.fragment,o),S(To.$$.fragment,o),S(vo.$$.fragment,o),S(bo.$$.fragment,o),S(ko.$$.fragment,o),S(yo.$$.fragment,o),S(So.$$.fragment,o),S(qo.$$.fragment,o),S(qt.$$.fragment,o),S(Fo.$$.fragment,o),S(Ft.$$.fragment,o),S(Co.$$.fragment,o),S(jo.$$.fragment,o),S(No.$$.fragment,o),S(Ao.$$.fragment,o),S(Oo.$$.fragment,o),S(jt.$$.fragment,o),S(Mt.$$.fragment,o),S(Go.$$.fragment,o),S(Uo.$$.fragment,o),S(Ho.$$.fragment,o),S(At.$$.fragment,o),S(Dt.$$.fragment,o),S(Bo.$$.fragment,o),S(Ko.$$.fragment,o),S(Lt.$$.fragment,o),S(Qo.$$.fragment,o),S(Ot.$$.fragment,o),S(Gt.$$.fragment,o),S(Zo.$$.fragment,o),S(en.$$.fragment,o),S(Wt.$$.fragment,o),S(sn.$$.fragment,o),S(Vt.$$.fragment,o),S(Rt.$$.fragment,o),vr=!1},d(o){t(h),o&&t(x),o&&t(_),$(d),o&&t(F),o&&t(q),$(ee),o&&t(ye),o&&t(N),o&&t(we),o&&t(I),o&&t(Q),o&&t(J),$(Y),o&&t(Ve),o&&t(ue),o&&t(Re),o&&t(P),o&&t(Xs),o&&t(W),o&&t(Qs),o&&t(hn),o&&t(Zs),$(oo,o),o&&t(er),o&&t(mn),o&&t(tr),$(so,o),o&&t(or),o&&t(Tt),o&&t(nr),o&&t(st),$(ao),o&&t(sr),o&&t(ve),$(io),$(xt),o&&t(rr),o&&t(it),$(lo),o&&t(ar),o&&t(V),$(po),$(mo),$(fo),$(_o),$(go),o&&t(ir),o&&t(dt),$(To),o&&t(dr),o&&t(pe),$(vo),$(bo),o&&t(cr),o&&t(ct),$(ko),o&&t(lr),o&&t(G),$(yo),$(So),$(qo),$(qt),$(Fo),$(Ft),$(Co),$(jo),o&&t(pr),o&&t(lt),$(No),o&&t(hr),o&&t(xe),$(Ao),$(Oo),$(jt),$(Mt),o&&t(mr),o&&t(ht),$(Go),o&&t(fr),o&&t(be),$(Uo),$(Ho),$(At),$(Dt),o&&t(ur),o&&t(ft),$(Bo),o&&t(_r),o&&t(he),$(Ko),$(Lt),$(Qo),$(Ot),$(Gt),o&&t(gr),o&&t(_t),$(Zo),o&&t(Tr),o&&t(me),$(en),$(Wt),$(sn),$(Vt),$(Rt)}}}const rh={local:"speech2text",sections:[{local:"overview",title:"Overview"},{local:"inference",title:"Inference"},{local:"transformers.Speech2TextConfig",title:"Speech2TextConfig"},{local:"transformers.Speech2TextTokenizer",title:"Speech2TextTokenizer"},{local:"transformers.Speech2TextFeatureExtractor",title:"Speech2TextFeatureExtractor"},{local:"transformers.Speech2TextProcessor",title:"Speech2TextProcessor"},{local:"transformers.Speech2TextModel",title:"Speech2TextModel"},{local:"transformers.Speech2TextForConditionalGeneration",title:"Speech2TextForConditionalGeneration"},{local:"transformers.TFSpeech2TextModel",title:"TFSpeech2TextModel"},{local:"transformers.TFSpeech2TextForConditionalGeneration",title:"TFSpeech2TextForConditionalGeneration"}],title:"Speech2Text"};function ah(C){return Vp(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class mh extends Op{constructor(h){super();Gp(this,h,ah,sh,Up,{})}}export{mh as default,rh as metadata};
