import{S as W6,i as H6,s as R6,e as r,k as c,w as v,t as n,M as V6,c as a,d as t,m as p,a as i,x as y,h as s,b as u,G as e,g as b,y as w,q as $,o as x,B as F,v as Q6,L as te}from"../../chunks/vendor-hf-doc-builder.js";import{T as we}from"../../chunks/Tip-hf-doc-builder.js";import{D as C}from"../../chunks/Docstring-hf-doc-builder.js";import{C as oe}from"../../chunks/CodeBlock-hf-doc-builder.js";import{I as $e}from"../../chunks/IconCopyLink-hf-doc-builder.js";import{E as ee}from"../../chunks/ExampleCodeBlock-hf-doc-builder.js";function K6(B){let d,_,m,h,g;return h=new oe({props:{code:`from transformers import BertModel, BertConfig

# Initializing a BERT bert-base-uncased style configuration
configuration = BertConfig()

# Initializing a model from the bert-base-uncased style configuration
model = BertModel(configuration)

# Accessing the model configuration
configuration = model.config`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertModel, BertConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a BERT bert-base-uncased style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = BertConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the bert-base-uncased style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BertModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`}}),{c(){d=r("p"),_=n("Examples:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Examples:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function J6(B){let d,_,m,h,g;return h=new oe({props:{code:`0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
| first sequence    | second sequence |`,highlighted:`0<span class="hljs-number"> 0 </span>0<span class="hljs-number"> 0 </span>0<span class="hljs-number"> 0 </span>0<span class="hljs-number"> 0 </span>0<span class="hljs-number"> 0 </span>0<span class="hljs-number"> 1 </span>1<span class="hljs-number"> 1 </span>1<span class="hljs-number"> 1 </span>1<span class="hljs-number"> 1 </span>1 1
| first sequence    | second sequence |`}}),{c(){d=r("p"),_=n("pair mask has the following format:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"pair mask has the following format:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function G6(B){let d,_,m,h,g;return h=new oe({props:{code:`0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
| first sequence    | second sequence |`,highlighted:`0<span class="hljs-number"> 0 </span>0<span class="hljs-number"> 0 </span>0<span class="hljs-number"> 0 </span>0<span class="hljs-number"> 0 </span>0<span class="hljs-number"> 0 </span>0<span class="hljs-number"> 1 </span>1<span class="hljs-number"> 1 </span>1<span class="hljs-number"> 1 </span>1<span class="hljs-number"> 1 </span>1 1
| first sequence    | second sequence |`}}),{c(){d=r("p"),_=n("pair mask has the following format:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"pair mask has the following format:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function X6(B){let d,_,m,h,g;return h=new oe({props:{code:`from transformers import TFBertTokenizer

tf_tokenizer = TFBertTokenizer.from_pretrained("bert-base-uncased")`,highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> TFBertTokenizer

tf_tokenizer = TFBertTokenizer.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)`}}),{c(){d=r("p"),_=n("Examples:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Examples:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function Y6(B){let d,_,m,h,g;return h=new oe({props:{code:`from transformers import AutoTokenizer, TFBertTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tf_tokenizer = TFBertTokenizer.from_tokenizer(tokenizer)`,highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, TFBertTokenizer

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)
tf_tokenizer = TFBertTokenizer.from_tokenizer(tokenizer)`}}),{c(){d=r("p"),_=n("Examples:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Examples:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function Z6(B){let d,_,m,h,g;return{c(){d=r("p"),_=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r("code"),h=n("Module"),g=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=a(f,"CODE",{});var E=i(m);h=s(E,"Module"),E.forEach(t),g=s(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,_),e(d,m),e(m,h),e(d,g)},d(l){l&&t(d)}}}function eN(B){let d,_,m,h,g;return h=new oe({props:{code:`from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertTokenizer, BertModel
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BertModel.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`}}),{c(){d=r("p"),_=n("Example:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Example:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function tN(B){let d,_,m,h,g;return{c(){d=r("p"),_=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r("code"),h=n("Module"),g=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=a(f,"CODE",{});var E=i(m);h=s(E,"Module"),E.forEach(t),g=s(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,_),e(d,m),e(m,h),e(d,g)},d(l){l&&t(d)}}}function oN(B){let d,_,m,h,g;return h=new oe({props:{code:`from transformers import BertTokenizer, BertForPreTraining
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForPreTraining.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

prediction_logits = outputs.prediction_logits
seq_relationship_logits = outputs.seq_relationship_logits`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertTokenizer, BertForPreTraining
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BertForPreTraining.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>prediction_logits = outputs.prediction_logits
<span class="hljs-meta">&gt;&gt;&gt; </span>seq_relationship_logits = outputs.seq_relationship_logits`}}),{c(){d=r("p"),_=n("Example:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Example:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function nN(B){let d,_,m,h,g;return{c(){d=r("p"),_=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r("code"),h=n("Module"),g=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=a(f,"CODE",{});var E=i(m);h=s(E,"Module"),E.forEach(t),g=s(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,_),e(d,m),e(m,h),e(d,g)},d(l){l&&t(d)}}}function sN(B){let d,_,m,h,g;return h=new oe({props:{code:`import torch
from transformers import BertTokenizer, BertLMHeadModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertLMHeadModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
logits = outputs.logits`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertTokenizer, BertLMHeadModel

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BertLMHeadModel.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs, labels=inputs[<span class="hljs-string">&quot;input_ids&quot;</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`}}),{c(){d=r("p"),_=n("Example:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Example:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function rN(B){let d,_,m,h,g;return{c(){d=r("p"),_=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r("code"),h=n("Module"),g=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=a(f,"CODE",{});var E=i(m);h=s(E,"Module"),E.forEach(t),g=s(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,_),e(d,m),e(m,h),e(d,g)},d(l){l&&t(d)}}}function aN(B){let d,_,m,h,g;return h=new oe({props:{code:`from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# retrieve index of [MASK]
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
tokenizer.decode(predicted_token_id)`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertTokenizer, BertForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BertForMaskedLM.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;The capital of France is [MASK].&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># retrieve index of [MASK]</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[<span class="hljs-number">0</span>].nonzero(as_tuple=<span class="hljs-literal">True</span>)[<span class="hljs-number">0</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_token_id = logits[<span class="hljs-number">0</span>, mask_token_index].argmax(axis=-<span class="hljs-number">1</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.decode(predicted_token_id)
<span class="hljs-string">&#x27;paris&#x27;</span>`}}),{c(){d=r("p"),_=n("Example:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Example:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function iN(B){let d,_;return d=new oe({props:{code:`labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
# mask labels of non-[MASK] tokens
labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

outputs = model(**inputs, labels=labels)
round(outputs.loss.item(), 2)`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span>labels = tokenizer(<span class="hljs-string">&quot;The capital of France is Paris.&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># mask labels of non-[MASK] tokens</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -<span class="hljs-number">100</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs, labels=labels)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(outputs.loss.item(), <span class="hljs-number">2</span>)
<span class="hljs-number">0.88</span>`}}),{c(){v(d.$$.fragment)},l(m){y(d.$$.fragment,m)},m(m,h){w(d,m,h),_=!0},p:te,i(m){_||($(d.$$.fragment,m),_=!0)},o(m){x(d.$$.fragment,m),_=!1},d(m){F(d,m)}}}function lN(B){let d,_,m,h,g;return{c(){d=r("p"),_=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r("code"),h=n("Module"),g=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=a(f,"CODE",{});var E=i(m);h=s(E,"Module"),E.forEach(t),g=s(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,_),e(d,m),e(m,h),e(d,g)},d(l){l&&t(d)}}}function dN(B){let d,_,m,h,g;return h=new oe({props:{code:`from transformers import BertTokenizer, BertForNextSentencePrediction
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")

prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
next_sentence = "The sky is blue due to the shorter wavelength of blue light."
encoding = tokenizer(prompt, next_sentence, return_tensors="pt")

outputs = model(**encoding, labels=torch.LongTensor([1]))
logits = outputs.logits
assert logits[0, 0] < logits[0, 1]  # next sentence was random`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertTokenizer, BertForNextSentencePrediction
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BertForNextSentencePrediction.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>next_sentence = <span class="hljs-string">&quot;The sky is blue due to the shorter wavelength of blue light.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer(prompt, next_sentence, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**encoding, labels=torch.LongTensor([<span class="hljs-number">1</span>]))
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">assert</span> logits[<span class="hljs-number">0</span>, <span class="hljs-number">0</span>] &lt; logits[<span class="hljs-number">0</span>, <span class="hljs-number">1</span>]  <span class="hljs-comment"># next sentence was random</span>`}}),{c(){d=r("p"),_=n("Example:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Example:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function cN(B){let d,_,m,h,g;return{c(){d=r("p"),_=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r("code"),h=n("Module"),g=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=a(f,"CODE",{});var E=i(m);h=s(E,"Module"),E.forEach(t),g=s(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,_),e(d,m),e(m,h),e(d,g)},d(l){l&&t(d)}}}function pN(B){let d,_,m,h,g;return h=new oe({props:{code:`import torch
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("textattack/bert-base-uncased-yelp-polarity")
model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertTokenizer, BertForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;textattack/bert-base-uncased-yelp-polarity&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;textattack/bert-base-uncased-yelp-polarity&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
<span class="hljs-string">&#x27;LABEL_1&#x27;</span>`}}),{c(){d=r("p"),_=n("Example of single-label classification:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Example of single-label classification:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function hN(B){let d,_;return d=new oe({props:{code:'# To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`\nnum_labels = len(model.config.id2label)\nmodel = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity", num_labels=num_labels)\n\nlabels = torch.tensor(1)\nloss = model(**inputs, labels=labels).loss\nround(loss.item(), 2)',highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;textattack/bert-base-uncased-yelp-polarity&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">1</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
<span class="hljs-number">0.01</span>`}}),{c(){v(d.$$.fragment)},l(m){y(d.$$.fragment,m)},m(m,h){w(d,m,h),_=!0},p:te,i(m){_||($(d.$$.fragment,m),_=!0)},o(m){x(d.$$.fragment,m),_=!1},d(m){F(d,m)}}}function mN(B){let d,_,m,h,g;return h=new oe({props:{code:`import torch
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("textattack/bert-base-uncased-yelp-polarity")
model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity", problem_type="multi_label_classification")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertTokenizer, BertForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;textattack/bert-base-uncased-yelp-polarity&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;textattack/bert-base-uncased-yelp-polarity&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
<span class="hljs-string">&#x27;LABEL_1&#x27;</span>`}}),{c(){d=r("p"),_=n("Example of multi-label classification:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Example of multi-label classification:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function fN(B){let d,_;return d=new oe({props:{code:`# To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`
num_labels = len(model.config.id2label)
model = BertForSequenceClassification.from_pretrained(
    "textattack/bert-base-uncased-yelp-polarity", num_labels=num_labels, problem_type="multi_label_classification"
)

labels = torch.nn.functional.one_hot(torch.tensor([predicted_class_id]), num_classes=num_labels).to(
    torch.float
)
loss = model(**inputs, labels=labels).loss
loss.backward()`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BertForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;textattack/bert-base-uncased-yelp-polarity&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.nn.functional.one_hot(torch.tensor([predicted_class_id]), num_classes=num_labels).to(
<span class="hljs-meta">... </span>    torch.<span class="hljs-built_in">float</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span>loss.backward()`}}),{c(){v(d.$$.fragment)},l(m){y(d.$$.fragment,m)},m(m,h){w(d,m,h),_=!0},p:te,i(m){_||($(d.$$.fragment,m),_=!0)},o(m){x(d.$$.fragment,m),_=!1},d(m){F(d,m)}}}function uN(B){let d,_,m,h,g;return{c(){d=r("p"),_=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r("code"),h=n("Module"),g=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=a(f,"CODE",{});var E=i(m);h=s(E,"Module"),E.forEach(t),g=s(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,_),e(d,m),e(m,h),e(d,g)},d(l){l&&t(d)}}}function gN(B){let d,_,m,h,g;return h=new oe({props:{code:`from transformers import BertTokenizer, BertForMultipleChoice
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMultipleChoice.from_pretrained("bert-base-uncased")

prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
choice0 = "It is eaten with a fork and a knife."
choice1 = "It is eaten while held in the hand."
labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="pt", padding=True)
outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)  # batch size is 1

# the linear classifier still needs to be trained
loss = outputs.loss
logits = outputs.logits`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertTokenizer, BertForMultipleChoice
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BertForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">0</span>).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># choice0 is correct (according to Wikipedia ;)), batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**{k: v.unsqueeze(<span class="hljs-number">0</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()}, labels=labels)  <span class="hljs-comment"># batch size is 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the linear classifier still needs to be trained</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`}}),{c(){d=r("p"),_=n("Example:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Example:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function _N(B){let d,_,m,h,g;return{c(){d=r("p"),_=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r("code"),h=n("Module"),g=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=a(f,"CODE",{});var E=i(m);h=s(E,"Module"),E.forEach(t),g=s(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,_),e(d,m),e(m,h),e(d,g)},d(l){l&&t(d)}}}function bN(B){let d,_,m,h,g;return h=new oe({props:{code:`from transformers import BertTokenizer, BertForTokenClassification
import torch

tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = BertForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

inputs = tokenizer(
    "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="pt"
)

with torch.no_grad():
    logits = model(**inputs).logits

predicted_token_class_ids = logits.argmax(-1)

# Note that tokens are classified rather then input words which means that
# there might be more predicted token classes than words.
# Multiple token classes might account for the same word
predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
predicted_tokens_classes`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertTokenizer, BertForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;dbmdz/bert-large-cased-finetuned-conll03-english&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BertForTokenClassification.from_pretrained(<span class="hljs-string">&quot;dbmdz/bert-large-cased-finetuned-conll03-english&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;HuggingFace is a company based in Paris and New York&quot;</span>, add_special_tokens=<span class="hljs-literal">False</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_token_class_ids = logits.argmax(-<span class="hljs-number">1</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Note that tokens are classified rather then input words which means that</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># there might be more predicted token classes than words.</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Multiple token classes might account for the same word</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_tokens_classes = [model.config.id2label[t.item()] <span class="hljs-keyword">for</span> t <span class="hljs-keyword">in</span> predicted_token_class_ids[<span class="hljs-number">0</span>]]
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_tokens_classes
[<span class="hljs-string">&#x27;O&#x27;</span>, <span class="hljs-string">&#x27;I-ORG&#x27;</span>, <span class="hljs-string">&#x27;I-ORG&#x27;</span>, <span class="hljs-string">&#x27;I-ORG&#x27;</span>, <span class="hljs-string">&#x27;O&#x27;</span>, <span class="hljs-string">&#x27;O&#x27;</span>, <span class="hljs-string">&#x27;O&#x27;</span>, <span class="hljs-string">&#x27;O&#x27;</span>, <span class="hljs-string">&#x27;O&#x27;</span>, <span class="hljs-string">&#x27;I-LOC&#x27;</span>, <span class="hljs-string">&#x27;O&#x27;</span>, <span class="hljs-string">&#x27;I-LOC&#x27;</span>, <span class="hljs-string">&#x27;I-LOC&#x27;</span>] `}}),{c(){d=r("p"),_=n("Example:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Example:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function kN(B){let d,_;return d=new oe({props:{code:`labels = predicted_token_class_ids
loss = model(**inputs, labels=labels).loss
round(loss.item(), 2)`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span>labels = predicted_token_class_ids
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
<span class="hljs-number">0.01</span>`}}),{c(){v(d.$$.fragment)},l(m){y(d.$$.fragment,m)},m(m,h){w(d,m,h),_=!0},p:te,i(m){_||($(d.$$.fragment,m),_=!0)},o(m){x(d.$$.fragment,m),_=!1},d(m){F(d,m)}}}function TN(B){let d,_,m,h,g;return{c(){d=r("p"),_=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r("code"),h=n("Module"),g=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=a(f,"CODE",{});var E=i(m);h=s(E,"Module"),E.forEach(t),g=s(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,_),e(d,m),e(m,h),e(d,g)},d(l){l&&t(d)}}}function vN(B){let d,_,m,h,g;return h=new oe({props:{code:`from transformers import BertTokenizer, BertForQuestionAnswering
import torch

tokenizer = BertTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
model = BertForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")

question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

inputs = tokenizer(question, text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
tokenizer.decode(predict_answer_tokens)`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertTokenizer, BertForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;deepset/bert-base-cased-squad2&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BertForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;deepset/bert-base-cased-squad2&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>question, text = <span class="hljs-string">&quot;Who was Jim Henson?&quot;</span>, <span class="hljs-string">&quot;Jim Henson was a nice puppet&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(question, text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>answer_start_index = outputs.start_logits.argmax()
<span class="hljs-meta">&gt;&gt;&gt; </span>answer_end_index = outputs.end_logits.argmax()

<span class="hljs-meta">&gt;&gt;&gt; </span>predict_answer_tokens = inputs.input_ids[<span class="hljs-number">0</span>, answer_start_index : answer_end_index + <span class="hljs-number">1</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.decode(predict_answer_tokens)
<span class="hljs-string">&#x27;a nice puppet&#x27;</span>`}}),{c(){d=r("p"),_=n("Example:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Example:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function yN(B){let d,_;return d=new oe({props:{code:`# target is "nice puppet"
target_start_index = torch.tensor([14])
target_end_index = torch.tensor([15])

outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
loss = outputs.loss
round(loss.item(), 2)`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># target is &quot;nice puppet&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>target_start_index = torch.tensor([<span class="hljs-number">14</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>target_end_index = torch.tensor([<span class="hljs-number">15</span>])

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
<span class="hljs-number">7.41</span>`}}),{c(){v(d.$$.fragment)},l(m){y(d.$$.fragment,m)},m(m,h){w(d,m,h),_=!0},p:te,i(m){_||($(d.$$.fragment,m),_=!0)},o(m){x(d.$$.fragment,m),_=!1},d(m){F(d,m)}}}function wN(B){let d,_,m,h,g,l,f,E,be,X,z,ne,L,re,ke,D,Te,me,J,O,ae,Y,P,j,ie,H,fe,le,S,ve,ue,q,ce,R,ge,de,V,_e,se,N,ye,Q,pe;return{c(){d=r("p"),_=n("TF 2.0 models accepts two formats as inputs:"),m=c(),h=r("ul"),g=r("li"),l=n("having all inputs as keyword arguments (like PyTorch models), or"),f=c(),E=r("li"),be=n("having all inputs as a list, tuple or dict in the first positional arguments."),X=c(),z=r("p"),ne=n("This second option is useful when using "),L=r("code"),re=n("tf.keras.Model.fit"),ke=n(` method which currently requires having all the
tensors in the first argument of the model call function: `),D=r("code"),Te=n("model(inputs)"),me=n("."),J=c(),O=r("p"),ae=n(`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),Y=c(),P=r("ul"),j=r("li"),ie=n("a single Tensor with "),H=r("code"),fe=n("input_ids"),le=n(" only and nothing else: "),S=r("code"),ve=n("model(inputs_ids)"),ue=c(),q=r("li"),ce=n(`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),R=r("code"),ge=n("model([input_ids, attention_mask])"),de=n(" or "),V=r("code"),_e=n("model([input_ids, attention_mask, token_type_ids])"),se=c(),N=r("li"),ye=n(`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),Q=r("code"),pe=n('model({"input_ids": input_ids, "token_type_ids": token_type_ids})')},l(T){d=a(T,"P",{});var M=i(d);_=s(M,"TF 2.0 models accepts two formats as inputs:"),M.forEach(t),m=p(T),h=a(T,"UL",{});var K=i(h);g=a(K,"LI",{});var ze=i(g);l=s(ze,"having all inputs as keyword arguments (like PyTorch models), or"),ze.forEach(t),f=p(K),E=a(K,"LI",{});var Be=i(E);be=s(Be,"having all inputs as a list, tuple or dict in the first positional arguments."),Be.forEach(t),K.forEach(t),X=p(T),z=a(T,"P",{});var I=i(z);ne=s(I,"This second option is useful when using "),L=a(I,"CODE",{});var Pe=i(L);re=s(Pe,"tf.keras.Model.fit"),Pe.forEach(t),ke=s(I,` method which currently requires having all the
tensors in the first argument of the model call function: `),D=a(I,"CODE",{});var Ee=i(D);Te=s(Ee,"model(inputs)"),Ee.forEach(t),me=s(I,"."),I.forEach(t),J=p(T),O=a(T,"P",{});var qe=i(O);ae=s(qe,`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),qe.forEach(t),Y=p(T),P=a(T,"UL",{});var A=i(P);j=a(A,"LI",{});var W=i(j);ie=s(W,"a single Tensor with "),H=a(W,"CODE",{});var xe=i(H);fe=s(xe,"input_ids"),xe.forEach(t),le=s(W," only and nothing else: "),S=a(W,"CODE",{});var Fe=i(S);ve=s(Fe,"model(inputs_ids)"),Fe.forEach(t),W.forEach(t),ue=p(A),q=a(A,"LI",{});var U=i(q);ce=s(U,`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),R=a(U,"CODE",{});var Me=i(R);ge=s(Me,"model([input_ids, attention_mask])"),Me.forEach(t),de=s(U," or "),V=a(U,"CODE",{});var je=i(V);_e=s(je,"model([input_ids, attention_mask, token_type_ids])"),je.forEach(t),U.forEach(t),se=p(A),N=a(A,"LI",{});var he=i(N);ye=s(he,`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),Q=a(he,"CODE",{});var Ce=i(Q);pe=s(Ce,'model({"input_ids": input_ids, "token_type_ids": token_type_ids})'),Ce.forEach(t),he.forEach(t),A.forEach(t)},m(T,M){b(T,d,M),e(d,_),b(T,m,M),b(T,h,M),e(h,g),e(g,l),e(h,f),e(h,E),e(E,be),b(T,X,M),b(T,z,M),e(z,ne),e(z,L),e(L,re),e(z,ke),e(z,D),e(D,Te),e(z,me),b(T,J,M),b(T,O,M),e(O,ae),b(T,Y,M),b(T,P,M),e(P,j),e(j,ie),e(j,H),e(H,fe),e(j,le),e(j,S),e(S,ve),e(P,ue),e(P,q),e(q,ce),e(q,R),e(R,ge),e(q,de),e(q,V),e(V,_e),e(P,se),e(P,N),e(N,ye),e(N,Q),e(Q,pe)},d(T){T&&t(d),T&&t(m),T&&t(h),T&&t(X),T&&t(z),T&&t(J),T&&t(O),T&&t(Y),T&&t(P)}}}function $N(B){let d,_,m,h,g;return{c(){d=r("p"),_=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r("code"),h=n("Module"),g=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=a(f,"CODE",{});var E=i(m);h=s(E,"Module"),E.forEach(t),g=s(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,_),e(d,m),e(m,h),e(d,g)},d(l){l&&t(d)}}}function xN(B){let d,_,m,h,g;return h=new oe({props:{code:`from transformers import BertTokenizer, TFBertModel
import tensorflow as tf

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TFBertModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
outputs = model(inputs)

last_hidden_states = outputs.last_hidden_state`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertTokenizer, TFBertModel
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> tensorflow <span class="hljs-keyword">as</span> tf

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFBertModel.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;tf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`}}),{c(){d=r("p"),_=n("Example:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Example:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function FN(B){let d,_,m,h,g,l,f,E,be,X,z,ne,L,re,ke,D,Te,me,J,O,ae,Y,P,j,ie,H,fe,le,S,ve,ue,q,ce,R,ge,de,V,_e,se,N,ye,Q,pe;return{c(){d=r("p"),_=n("TF 2.0 models accepts two formats as inputs:"),m=c(),h=r("ul"),g=r("li"),l=n("having all inputs as keyword arguments (like PyTorch models), or"),f=c(),E=r("li"),be=n("having all inputs as a list, tuple or dict in the first positional arguments."),X=c(),z=r("p"),ne=n("This second option is useful when using "),L=r("code"),re=n("tf.keras.Model.fit"),ke=n(` method which currently requires having all the
tensors in the first argument of the model call function: `),D=r("code"),Te=n("model(inputs)"),me=n("."),J=c(),O=r("p"),ae=n(`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),Y=c(),P=r("ul"),j=r("li"),ie=n("a single Tensor with "),H=r("code"),fe=n("input_ids"),le=n(" only and nothing else: "),S=r("code"),ve=n("model(inputs_ids)"),ue=c(),q=r("li"),ce=n(`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),R=r("code"),ge=n("model([input_ids, attention_mask])"),de=n(" or "),V=r("code"),_e=n("model([input_ids, attention_mask, token_type_ids])"),se=c(),N=r("li"),ye=n(`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),Q=r("code"),pe=n('model({"input_ids": input_ids, "token_type_ids": token_type_ids})')},l(T){d=a(T,"P",{});var M=i(d);_=s(M,"TF 2.0 models accepts two formats as inputs:"),M.forEach(t),m=p(T),h=a(T,"UL",{});var K=i(h);g=a(K,"LI",{});var ze=i(g);l=s(ze,"having all inputs as keyword arguments (like PyTorch models), or"),ze.forEach(t),f=p(K),E=a(K,"LI",{});var Be=i(E);be=s(Be,"having all inputs as a list, tuple or dict in the first positional arguments."),Be.forEach(t),K.forEach(t),X=p(T),z=a(T,"P",{});var I=i(z);ne=s(I,"This second option is useful when using "),L=a(I,"CODE",{});var Pe=i(L);re=s(Pe,"tf.keras.Model.fit"),Pe.forEach(t),ke=s(I,` method which currently requires having all the
tensors in the first argument of the model call function: `),D=a(I,"CODE",{});var Ee=i(D);Te=s(Ee,"model(inputs)"),Ee.forEach(t),me=s(I,"."),I.forEach(t),J=p(T),O=a(T,"P",{});var qe=i(O);ae=s(qe,`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),qe.forEach(t),Y=p(T),P=a(T,"UL",{});var A=i(P);j=a(A,"LI",{});var W=i(j);ie=s(W,"a single Tensor with "),H=a(W,"CODE",{});var xe=i(H);fe=s(xe,"input_ids"),xe.forEach(t),le=s(W," only and nothing else: "),S=a(W,"CODE",{});var Fe=i(S);ve=s(Fe,"model(inputs_ids)"),Fe.forEach(t),W.forEach(t),ue=p(A),q=a(A,"LI",{});var U=i(q);ce=s(U,`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),R=a(U,"CODE",{});var Me=i(R);ge=s(Me,"model([input_ids, attention_mask])"),Me.forEach(t),de=s(U," or "),V=a(U,"CODE",{});var je=i(V);_e=s(je,"model([input_ids, attention_mask, token_type_ids])"),je.forEach(t),U.forEach(t),se=p(A),N=a(A,"LI",{});var he=i(N);ye=s(he,`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),Q=a(he,"CODE",{});var Ce=i(Q);pe=s(Ce,'model({"input_ids": input_ids, "token_type_ids": token_type_ids})'),Ce.forEach(t),he.forEach(t),A.forEach(t)},m(T,M){b(T,d,M),e(d,_),b(T,m,M),b(T,h,M),e(h,g),e(g,l),e(h,f),e(h,E),e(E,be),b(T,X,M),b(T,z,M),e(z,ne),e(z,L),e(L,re),e(z,ke),e(z,D),e(D,Te),e(z,me),b(T,J,M),b(T,O,M),e(O,ae),b(T,Y,M),b(T,P,M),e(P,j),e(j,ie),e(j,H),e(H,fe),e(j,le),e(j,S),e(S,ve),e(P,ue),e(P,q),e(q,ce),e(q,R),e(R,ge),e(q,de),e(q,V),e(V,_e),e(P,se),e(P,N),e(N,ye),e(N,Q),e(Q,pe)},d(T){T&&t(d),T&&t(m),T&&t(h),T&&t(X),T&&t(z),T&&t(J),T&&t(O),T&&t(Y),T&&t(P)}}}function BN(B){let d,_,m,h,g;return{c(){d=r("p"),_=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r("code"),h=n("Module"),g=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=a(f,"CODE",{});var E=i(m);h=s(E,"Module"),E.forEach(t),g=s(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,_),e(d,m),e(m,h),e(d,g)},d(l){l&&t(d)}}}function EN(B){let d,_,m,h,g;return h=new oe({props:{code:`import tensorflow as tf
from transformers import BertTokenizer, TFBertForPreTraining

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TFBertForPreTraining.from_pretrained("bert-base-uncased")
input_ids = tokenizer("Hello, my dog is cute", add_special_tokens=True, return_tensors="tf")
# Batch size 1

outputs = model(input_ids)
prediction_logits, seq_relationship_logits = outputs[:2]`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> tensorflow <span class="hljs-keyword">as</span> tf
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertTokenizer, TFBertForPreTraining

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFBertForPreTraining.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, add_special_tokens=<span class="hljs-literal">True</span>, return_tensors=<span class="hljs-string">&quot;tf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids)
<span class="hljs-meta">&gt;&gt;&gt; </span>prediction_logits, seq_relationship_logits = outputs[:<span class="hljs-number">2</span>]`}}),{c(){d=r("p"),_=n("Examples:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Examples:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function MN(B){let d,_,m,h,g;return h=new oe({props:{code:`from transformers import BertTokenizer, TFBertLMHeadModel
import tensorflow as tf

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TFBertLMHeadModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
outputs = model(inputs)
logits = outputs.logits`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertTokenizer, TFBertLMHeadModel
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> tensorflow <span class="hljs-keyword">as</span> tf

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFBertLMHeadModel.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;tf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`}}),{c(){d=r("p"),_=n("Example:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Example:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function zN(B){let d,_,m,h,g,l,f,E,be,X,z,ne,L,re,ke,D,Te,me,J,O,ae,Y,P,j,ie,H,fe,le,S,ve,ue,q,ce,R,ge,de,V,_e,se,N,ye,Q,pe;return{c(){d=r("p"),_=n("TF 2.0 models accepts two formats as inputs:"),m=c(),h=r("ul"),g=r("li"),l=n("having all inputs as keyword arguments (like PyTorch models), or"),f=c(),E=r("li"),be=n("having all inputs as a list, tuple or dict in the first positional arguments."),X=c(),z=r("p"),ne=n("This second option is useful when using "),L=r("code"),re=n("tf.keras.Model.fit"),ke=n(` method which currently requires having all the
tensors in the first argument of the model call function: `),D=r("code"),Te=n("model(inputs)"),me=n("."),J=c(),O=r("p"),ae=n(`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),Y=c(),P=r("ul"),j=r("li"),ie=n("a single Tensor with "),H=r("code"),fe=n("input_ids"),le=n(" only and nothing else: "),S=r("code"),ve=n("model(inputs_ids)"),ue=c(),q=r("li"),ce=n(`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),R=r("code"),ge=n("model([input_ids, attention_mask])"),de=n(" or "),V=r("code"),_e=n("model([input_ids, attention_mask, token_type_ids])"),se=c(),N=r("li"),ye=n(`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),Q=r("code"),pe=n('model({"input_ids": input_ids, "token_type_ids": token_type_ids})')},l(T){d=a(T,"P",{});var M=i(d);_=s(M,"TF 2.0 models accepts two formats as inputs:"),M.forEach(t),m=p(T),h=a(T,"UL",{});var K=i(h);g=a(K,"LI",{});var ze=i(g);l=s(ze,"having all inputs as keyword arguments (like PyTorch models), or"),ze.forEach(t),f=p(K),E=a(K,"LI",{});var Be=i(E);be=s(Be,"having all inputs as a list, tuple or dict in the first positional arguments."),Be.forEach(t),K.forEach(t),X=p(T),z=a(T,"P",{});var I=i(z);ne=s(I,"This second option is useful when using "),L=a(I,"CODE",{});var Pe=i(L);re=s(Pe,"tf.keras.Model.fit"),Pe.forEach(t),ke=s(I,` method which currently requires having all the
tensors in the first argument of the model call function: `),D=a(I,"CODE",{});var Ee=i(D);Te=s(Ee,"model(inputs)"),Ee.forEach(t),me=s(I,"."),I.forEach(t),J=p(T),O=a(T,"P",{});var qe=i(O);ae=s(qe,`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),qe.forEach(t),Y=p(T),P=a(T,"UL",{});var A=i(P);j=a(A,"LI",{});var W=i(j);ie=s(W,"a single Tensor with "),H=a(W,"CODE",{});var xe=i(H);fe=s(xe,"input_ids"),xe.forEach(t),le=s(W," only and nothing else: "),S=a(W,"CODE",{});var Fe=i(S);ve=s(Fe,"model(inputs_ids)"),Fe.forEach(t),W.forEach(t),ue=p(A),q=a(A,"LI",{});var U=i(q);ce=s(U,`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),R=a(U,"CODE",{});var Me=i(R);ge=s(Me,"model([input_ids, attention_mask])"),Me.forEach(t),de=s(U," or "),V=a(U,"CODE",{});var je=i(V);_e=s(je,"model([input_ids, attention_mask, token_type_ids])"),je.forEach(t),U.forEach(t),se=p(A),N=a(A,"LI",{});var he=i(N);ye=s(he,`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),Q=a(he,"CODE",{});var Ce=i(Q);pe=s(Ce,'model({"input_ids": input_ids, "token_type_ids": token_type_ids})'),Ce.forEach(t),he.forEach(t),A.forEach(t)},m(T,M){b(T,d,M),e(d,_),b(T,m,M),b(T,h,M),e(h,g),e(g,l),e(h,f),e(h,E),e(E,be),b(T,X,M),b(T,z,M),e(z,ne),e(z,L),e(L,re),e(z,ke),e(z,D),e(D,Te),e(z,me),b(T,J,M),b(T,O,M),e(O,ae),b(T,Y,M),b(T,P,M),e(P,j),e(j,ie),e(j,H),e(H,fe),e(j,le),e(j,S),e(S,ve),e(P,ue),e(P,q),e(q,ce),e(q,R),e(R,ge),e(q,de),e(q,V),e(V,_e),e(P,se),e(P,N),e(N,ye),e(N,Q),e(Q,pe)},d(T){T&&t(d),T&&t(m),T&&t(h),T&&t(X),T&&t(z),T&&t(J),T&&t(O),T&&t(Y),T&&t(P)}}}function PN(B){let d,_,m,h,g;return{c(){d=r("p"),_=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r("code"),h=n("Module"),g=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=a(f,"CODE",{});var E=i(m);h=s(E,"Module"),E.forEach(t),g=s(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,_),e(d,m),e(m,h),e(d,g)},d(l){l&&t(d)}}}function qN(B){let d,_,m,h,g;return h=new oe({props:{code:`from transformers import BertTokenizer, TFBertForMaskedLM
import tensorflow as tf

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TFBertForMaskedLM.from_pretrained("bert-base-uncased")

inputs = tokenizer("The capital of France is [MASK].", return_tensors="tf")
logits = model(**inputs).logits

# retrieve index of [MASK]
mask_token_index = tf.where((inputs.input_ids == tokenizer.mask_token_id)[0])
selected_logits = tf.gather_nd(logits[0], indices=mask_token_index)

predicted_token_id = tf.math.argmax(selected_logits, axis=-1)
tokenizer.decode(predicted_token_id)`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertTokenizer, TFBertForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> tensorflow <span class="hljs-keyword">as</span> tf

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFBertForMaskedLM.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;The capital of France is [MASK].&quot;</span>, return_tensors=<span class="hljs-string">&quot;tf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># retrieve index of [MASK]</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>mask_token_index = tf.where((inputs.input_ids == tokenizer.mask_token_id)[<span class="hljs-number">0</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>selected_logits = tf.gather_nd(logits[<span class="hljs-number">0</span>], indices=mask_token_index)

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_token_id = tf.math.argmax(selected_logits, axis=-<span class="hljs-number">1</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.decode(predicted_token_id)
<span class="hljs-string">&#x27;paris&#x27;</span>`}}),{c(){d=r("p"),_=n("Example:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Example:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function jN(B){let d,_;return d=new oe({props:{code:`labels = tokenizer("The capital of France is Paris.", return_tensors="tf")["input_ids"]
# mask labels of non-[MASK] tokens
labels = tf.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

outputs = model(**inputs, labels=labels)
round(float(outputs.loss), 2)`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span>labels = tokenizer(<span class="hljs-string">&quot;The capital of France is Paris.&quot;</span>, return_tensors=<span class="hljs-string">&quot;tf&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># mask labels of non-[MASK] tokens</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = tf.where(inputs.input_ids == tokenizer.mask_token_id, labels, -<span class="hljs-number">100</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs, labels=labels)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(<span class="hljs-built_in">float</span>(outputs.loss), <span class="hljs-number">2</span>)
<span class="hljs-number">0.88</span>`}}),{c(){v(d.$$.fragment)},l(m){y(d.$$.fragment,m)},m(m,h){w(d,m,h),_=!0},p:te,i(m){_||($(d.$$.fragment,m),_=!0)},o(m){x(d.$$.fragment,m),_=!1},d(m){F(d,m)}}}function CN(B){let d,_,m,h,g,l,f,E,be,X,z,ne,L,re,ke,D,Te,me,J,O,ae,Y,P,j,ie,H,fe,le,S,ve,ue,q,ce,R,ge,de,V,_e,se,N,ye,Q,pe;return{c(){d=r("p"),_=n("TF 2.0 models accepts two formats as inputs:"),m=c(),h=r("ul"),g=r("li"),l=n("having all inputs as keyword arguments (like PyTorch models), or"),f=c(),E=r("li"),be=n("having all inputs as a list, tuple or dict in the first positional arguments."),X=c(),z=r("p"),ne=n("This second option is useful when using "),L=r("code"),re=n("tf.keras.Model.fit"),ke=n(` method which currently requires having all the
tensors in the first argument of the model call function: `),D=r("code"),Te=n("model(inputs)"),me=n("."),J=c(),O=r("p"),ae=n(`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),Y=c(),P=r("ul"),j=r("li"),ie=n("a single Tensor with "),H=r("code"),fe=n("input_ids"),le=n(" only and nothing else: "),S=r("code"),ve=n("model(inputs_ids)"),ue=c(),q=r("li"),ce=n(`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),R=r("code"),ge=n("model([input_ids, attention_mask])"),de=n(" or "),V=r("code"),_e=n("model([input_ids, attention_mask, token_type_ids])"),se=c(),N=r("li"),ye=n(`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),Q=r("code"),pe=n('model({"input_ids": input_ids, "token_type_ids": token_type_ids})')},l(T){d=a(T,"P",{});var M=i(d);_=s(M,"TF 2.0 models accepts two formats as inputs:"),M.forEach(t),m=p(T),h=a(T,"UL",{});var K=i(h);g=a(K,"LI",{});var ze=i(g);l=s(ze,"having all inputs as keyword arguments (like PyTorch models), or"),ze.forEach(t),f=p(K),E=a(K,"LI",{});var Be=i(E);be=s(Be,"having all inputs as a list, tuple or dict in the first positional arguments."),Be.forEach(t),K.forEach(t),X=p(T),z=a(T,"P",{});var I=i(z);ne=s(I,"This second option is useful when using "),L=a(I,"CODE",{});var Pe=i(L);re=s(Pe,"tf.keras.Model.fit"),Pe.forEach(t),ke=s(I,` method which currently requires having all the
tensors in the first argument of the model call function: `),D=a(I,"CODE",{});var Ee=i(D);Te=s(Ee,"model(inputs)"),Ee.forEach(t),me=s(I,"."),I.forEach(t),J=p(T),O=a(T,"P",{});var qe=i(O);ae=s(qe,`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),qe.forEach(t),Y=p(T),P=a(T,"UL",{});var A=i(P);j=a(A,"LI",{});var W=i(j);ie=s(W,"a single Tensor with "),H=a(W,"CODE",{});var xe=i(H);fe=s(xe,"input_ids"),xe.forEach(t),le=s(W," only and nothing else: "),S=a(W,"CODE",{});var Fe=i(S);ve=s(Fe,"model(inputs_ids)"),Fe.forEach(t),W.forEach(t),ue=p(A),q=a(A,"LI",{});var U=i(q);ce=s(U,`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),R=a(U,"CODE",{});var Me=i(R);ge=s(Me,"model([input_ids, attention_mask])"),Me.forEach(t),de=s(U," or "),V=a(U,"CODE",{});var je=i(V);_e=s(je,"model([input_ids, attention_mask, token_type_ids])"),je.forEach(t),U.forEach(t),se=p(A),N=a(A,"LI",{});var he=i(N);ye=s(he,`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),Q=a(he,"CODE",{});var Ce=i(Q);pe=s(Ce,'model({"input_ids": input_ids, "token_type_ids": token_type_ids})'),Ce.forEach(t),he.forEach(t),A.forEach(t)},m(T,M){b(T,d,M),e(d,_),b(T,m,M),b(T,h,M),e(h,g),e(g,l),e(h,f),e(h,E),e(E,be),b(T,X,M),b(T,z,M),e(z,ne),e(z,L),e(L,re),e(z,ke),e(z,D),e(D,Te),e(z,me),b(T,J,M),b(T,O,M),e(O,ae),b(T,Y,M),b(T,P,M),e(P,j),e(j,ie),e(j,H),e(H,fe),e(j,le),e(j,S),e(S,ve),e(P,ue),e(P,q),e(q,ce),e(q,R),e(R,ge),e(q,de),e(q,V),e(V,_e),e(P,se),e(P,N),e(N,ye),e(N,Q),e(Q,pe)},d(T){T&&t(d),T&&t(m),T&&t(h),T&&t(X),T&&t(z),T&&t(J),T&&t(O),T&&t(Y),T&&t(P)}}}function NN(B){let d,_,m,h,g;return{c(){d=r("p"),_=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r("code"),h=n("Module"),g=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=a(f,"CODE",{});var E=i(m);h=s(E,"Module"),E.forEach(t),g=s(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,_),e(d,m),e(m,h),e(d,g)},d(l){l&&t(d)}}}function ON(B){let d,_,m,h,g;return h=new oe({props:{code:`import tensorflow as tf
from transformers import BertTokenizer, TFBertForNextSentencePrediction

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TFBertForNextSentencePrediction.from_pretrained("bert-base-uncased")

prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
next_sentence = "The sky is blue due to the shorter wavelength of blue light."
encoding = tokenizer(prompt, next_sentence, return_tensors="tf")

logits = model(encoding["input_ids"], token_type_ids=encoding["token_type_ids"])[0]
assert logits[0][0] < logits[0][1]  # the next sentence was random`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> tensorflow <span class="hljs-keyword">as</span> tf
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertTokenizer, TFBertForNextSentencePrediction

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFBertForNextSentencePrediction.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>next_sentence = <span class="hljs-string">&quot;The sky is blue due to the shorter wavelength of blue light.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer(prompt, next_sentence, return_tensors=<span class="hljs-string">&quot;tf&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>logits = model(encoding[<span class="hljs-string">&quot;input_ids&quot;</span>], token_type_ids=encoding[<span class="hljs-string">&quot;token_type_ids&quot;</span>])[<span class="hljs-number">0</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">assert</span> logits[<span class="hljs-number">0</span>][<span class="hljs-number">0</span>] &lt; logits[<span class="hljs-number">0</span>][<span class="hljs-number">1</span>]  <span class="hljs-comment"># the next sentence was random</span>`}}),{c(){d=r("p"),_=n("Examples:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Examples:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function IN(B){let d,_,m,h,g,l,f,E,be,X,z,ne,L,re,ke,D,Te,me,J,O,ae,Y,P,j,ie,H,fe,le,S,ve,ue,q,ce,R,ge,de,V,_e,se,N,ye,Q,pe;return{c(){d=r("p"),_=n("TF 2.0 models accepts two formats as inputs:"),m=c(),h=r("ul"),g=r("li"),l=n("having all inputs as keyword arguments (like PyTorch models), or"),f=c(),E=r("li"),be=n("having all inputs as a list, tuple or dict in the first positional arguments."),X=c(),z=r("p"),ne=n("This second option is useful when using "),L=r("code"),re=n("tf.keras.Model.fit"),ke=n(` method which currently requires having all the
tensors in the first argument of the model call function: `),D=r("code"),Te=n("model(inputs)"),me=n("."),J=c(),O=r("p"),ae=n(`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),Y=c(),P=r("ul"),j=r("li"),ie=n("a single Tensor with "),H=r("code"),fe=n("input_ids"),le=n(" only and nothing else: "),S=r("code"),ve=n("model(inputs_ids)"),ue=c(),q=r("li"),ce=n(`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),R=r("code"),ge=n("model([input_ids, attention_mask])"),de=n(" or "),V=r("code"),_e=n("model([input_ids, attention_mask, token_type_ids])"),se=c(),N=r("li"),ye=n(`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),Q=r("code"),pe=n('model({"input_ids": input_ids, "token_type_ids": token_type_ids})')},l(T){d=a(T,"P",{});var M=i(d);_=s(M,"TF 2.0 models accepts two formats as inputs:"),M.forEach(t),m=p(T),h=a(T,"UL",{});var K=i(h);g=a(K,"LI",{});var ze=i(g);l=s(ze,"having all inputs as keyword arguments (like PyTorch models), or"),ze.forEach(t),f=p(K),E=a(K,"LI",{});var Be=i(E);be=s(Be,"having all inputs as a list, tuple or dict in the first positional arguments."),Be.forEach(t),K.forEach(t),X=p(T),z=a(T,"P",{});var I=i(z);ne=s(I,"This second option is useful when using "),L=a(I,"CODE",{});var Pe=i(L);re=s(Pe,"tf.keras.Model.fit"),Pe.forEach(t),ke=s(I,` method which currently requires having all the
tensors in the first argument of the model call function: `),D=a(I,"CODE",{});var Ee=i(D);Te=s(Ee,"model(inputs)"),Ee.forEach(t),me=s(I,"."),I.forEach(t),J=p(T),O=a(T,"P",{});var qe=i(O);ae=s(qe,`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),qe.forEach(t),Y=p(T),P=a(T,"UL",{});var A=i(P);j=a(A,"LI",{});var W=i(j);ie=s(W,"a single Tensor with "),H=a(W,"CODE",{});var xe=i(H);fe=s(xe,"input_ids"),xe.forEach(t),le=s(W," only and nothing else: "),S=a(W,"CODE",{});var Fe=i(S);ve=s(Fe,"model(inputs_ids)"),Fe.forEach(t),W.forEach(t),ue=p(A),q=a(A,"LI",{});var U=i(q);ce=s(U,`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),R=a(U,"CODE",{});var Me=i(R);ge=s(Me,"model([input_ids, attention_mask])"),Me.forEach(t),de=s(U," or "),V=a(U,"CODE",{});var je=i(V);_e=s(je,"model([input_ids, attention_mask, token_type_ids])"),je.forEach(t),U.forEach(t),se=p(A),N=a(A,"LI",{});var he=i(N);ye=s(he,`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),Q=a(he,"CODE",{});var Ce=i(Q);pe=s(Ce,'model({"input_ids": input_ids, "token_type_ids": token_type_ids})'),Ce.forEach(t),he.forEach(t),A.forEach(t)},m(T,M){b(T,d,M),e(d,_),b(T,m,M),b(T,h,M),e(h,g),e(g,l),e(h,f),e(h,E),e(E,be),b(T,X,M),b(T,z,M),e(z,ne),e(z,L),e(L,re),e(z,ke),e(z,D),e(D,Te),e(z,me),b(T,J,M),b(T,O,M),e(O,ae),b(T,Y,M),b(T,P,M),e(P,j),e(j,ie),e(j,H),e(H,fe),e(j,le),e(j,S),e(S,ve),e(P,ue),e(P,q),e(q,ce),e(q,R),e(R,ge),e(q,de),e(q,V),e(V,_e),e(P,se),e(P,N),e(N,ye),e(N,Q),e(Q,pe)},d(T){T&&t(d),T&&t(m),T&&t(h),T&&t(X),T&&t(z),T&&t(J),T&&t(O),T&&t(Y),T&&t(P)}}}function AN(B){let d,_,m,h,g;return{c(){d=r("p"),_=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r("code"),h=n("Module"),g=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=a(f,"CODE",{});var E=i(m);h=s(E,"Module"),E.forEach(t),g=s(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,_),e(d,m),e(m,h),e(d,g)},d(l){l&&t(d)}}}function LN(B){let d,_,m,h,g;return h=new oe({props:{code:`from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

tokenizer = BertTokenizer.from_pretrained("ydshieh/bert-base-uncased-yelp-polarity")
model = TFBertForSequenceClassification.from_pretrained("ydshieh/bert-base-uncased-yelp-polarity")

inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")

logits = model(**inputs).logits

predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
model.config.id2label[predicted_class_id]`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertTokenizer, TFBertForSequenceClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> tensorflow <span class="hljs-keyword">as</span> tf

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;ydshieh/bert-base-uncased-yelp-polarity&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFBertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;ydshieh/bert-base-uncased-yelp-polarity&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;tf&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = <span class="hljs-built_in">int</span>(tf.math.argmax(logits, axis=-<span class="hljs-number">1</span>)[<span class="hljs-number">0</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
<span class="hljs-string">&#x27;LABEL_1&#x27;</span>`}}),{c(){d=r("p"),_=n("Example:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Example:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function DN(B){let d,_;return d=new oe({props:{code:'# To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`\nnum_labels = len(model.config.id2label)\nmodel = TFBertForSequenceClassification.from_pretrained("ydshieh/bert-base-uncased-yelp-polarity", num_labels=num_labels)\n\nlabels = tf.constant(1)\nloss = model(**inputs, labels=labels).loss\nround(float(loss), 2)',highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFBertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;ydshieh/bert-base-uncased-yelp-polarity&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = tf.constant(<span class="hljs-number">1</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(<span class="hljs-built_in">float</span>(loss), <span class="hljs-number">2</span>)
<span class="hljs-number">0.01</span>`}}),{c(){v(d.$$.fragment)},l(m){y(d.$$.fragment,m)},m(m,h){w(d,m,h),_=!0},p:te,i(m){_||($(d.$$.fragment,m),_=!0)},o(m){x(d.$$.fragment,m),_=!1},d(m){F(d,m)}}}function SN(B){let d,_,m,h,g,l,f,E,be,X,z,ne,L,re,ke,D,Te,me,J,O,ae,Y,P,j,ie,H,fe,le,S,ve,ue,q,ce,R,ge,de,V,_e,se,N,ye,Q,pe;return{c(){d=r("p"),_=n("TF 2.0 models accepts two formats as inputs:"),m=c(),h=r("ul"),g=r("li"),l=n("having all inputs as keyword arguments (like PyTorch models), or"),f=c(),E=r("li"),be=n("having all inputs as a list, tuple or dict in the first positional arguments."),X=c(),z=r("p"),ne=n("This second option is useful when using "),L=r("code"),re=n("tf.keras.Model.fit"),ke=n(` method which currently requires having all the
tensors in the first argument of the model call function: `),D=r("code"),Te=n("model(inputs)"),me=n("."),J=c(),O=r("p"),ae=n(`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),Y=c(),P=r("ul"),j=r("li"),ie=n("a single Tensor with "),H=r("code"),fe=n("input_ids"),le=n(" only and nothing else: "),S=r("code"),ve=n("model(inputs_ids)"),ue=c(),q=r("li"),ce=n(`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),R=r("code"),ge=n("model([input_ids, attention_mask])"),de=n(" or "),V=r("code"),_e=n("model([input_ids, attention_mask, token_type_ids])"),se=c(),N=r("li"),ye=n(`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),Q=r("code"),pe=n('model({"input_ids": input_ids, "token_type_ids": token_type_ids})')},l(T){d=a(T,"P",{});var M=i(d);_=s(M,"TF 2.0 models accepts two formats as inputs:"),M.forEach(t),m=p(T),h=a(T,"UL",{});var K=i(h);g=a(K,"LI",{});var ze=i(g);l=s(ze,"having all inputs as keyword arguments (like PyTorch models), or"),ze.forEach(t),f=p(K),E=a(K,"LI",{});var Be=i(E);be=s(Be,"having all inputs as a list, tuple or dict in the first positional arguments."),Be.forEach(t),K.forEach(t),X=p(T),z=a(T,"P",{});var I=i(z);ne=s(I,"This second option is useful when using "),L=a(I,"CODE",{});var Pe=i(L);re=s(Pe,"tf.keras.Model.fit"),Pe.forEach(t),ke=s(I,` method which currently requires having all the
tensors in the first argument of the model call function: `),D=a(I,"CODE",{});var Ee=i(D);Te=s(Ee,"model(inputs)"),Ee.forEach(t),me=s(I,"."),I.forEach(t),J=p(T),O=a(T,"P",{});var qe=i(O);ae=s(qe,`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),qe.forEach(t),Y=p(T),P=a(T,"UL",{});var A=i(P);j=a(A,"LI",{});var W=i(j);ie=s(W,"a single Tensor with "),H=a(W,"CODE",{});var xe=i(H);fe=s(xe,"input_ids"),xe.forEach(t),le=s(W," only and nothing else: "),S=a(W,"CODE",{});var Fe=i(S);ve=s(Fe,"model(inputs_ids)"),Fe.forEach(t),W.forEach(t),ue=p(A),q=a(A,"LI",{});var U=i(q);ce=s(U,`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),R=a(U,"CODE",{});var Me=i(R);ge=s(Me,"model([input_ids, attention_mask])"),Me.forEach(t),de=s(U," or "),V=a(U,"CODE",{});var je=i(V);_e=s(je,"model([input_ids, attention_mask, token_type_ids])"),je.forEach(t),U.forEach(t),se=p(A),N=a(A,"LI",{});var he=i(N);ye=s(he,`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),Q=a(he,"CODE",{});var Ce=i(Q);pe=s(Ce,'model({"input_ids": input_ids, "token_type_ids": token_type_ids})'),Ce.forEach(t),he.forEach(t),A.forEach(t)},m(T,M){b(T,d,M),e(d,_),b(T,m,M),b(T,h,M),e(h,g),e(g,l),e(h,f),e(h,E),e(E,be),b(T,X,M),b(T,z,M),e(z,ne),e(z,L),e(L,re),e(z,ke),e(z,D),e(D,Te),e(z,me),b(T,J,M),b(T,O,M),e(O,ae),b(T,Y,M),b(T,P,M),e(P,j),e(j,ie),e(j,H),e(H,fe),e(j,le),e(j,S),e(S,ve),e(P,ue),e(P,q),e(q,ce),e(q,R),e(R,ge),e(q,de),e(q,V),e(V,_e),e(P,se),e(P,N),e(N,ye),e(N,Q),e(Q,pe)},d(T){T&&t(d),T&&t(m),T&&t(h),T&&t(X),T&&t(z),T&&t(J),T&&t(O),T&&t(Y),T&&t(P)}}}function UN(B){let d,_,m,h,g;return{c(){d=r("p"),_=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r("code"),h=n("Module"),g=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=a(f,"CODE",{});var E=i(m);h=s(E,"Module"),E.forEach(t),g=s(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,_),e(d,m),e(m,h),e(d,g)},d(l){l&&t(d)}}}function WN(B){let d,_,m,h,g;return h=new oe({props:{code:`from transformers import BertTokenizer, TFBertForMultipleChoice
import tensorflow as tf

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TFBertForMultipleChoice.from_pretrained("bert-base-uncased")

prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
choice0 = "It is eaten with a fork and a knife."
choice1 = "It is eaten while held in the hand."

encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="tf", padding=True)
inputs = {k: tf.expand_dims(v, 0) for k, v in encoding.items()}
outputs = model(inputs)  # batch size is 1

# the linear classifier still needs to be trained
logits = outputs.logits`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertTokenizer, TFBertForMultipleChoice
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> tensorflow <span class="hljs-keyword">as</span> tf

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFBertForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors=<span class="hljs-string">&quot;tf&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = {k: tf.expand_dims(v, <span class="hljs-number">0</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()}
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(inputs)  <span class="hljs-comment"># batch size is 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the linear classifier still needs to be trained</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`}}),{c(){d=r("p"),_=n("Example:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Example:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function HN(B){let d,_,m,h,g,l,f,E,be,X,z,ne,L,re,ke,D,Te,me,J,O,ae,Y,P,j,ie,H,fe,le,S,ve,ue,q,ce,R,ge,de,V,_e,se,N,ye,Q,pe;return{c(){d=r("p"),_=n("TF 2.0 models accepts two formats as inputs:"),m=c(),h=r("ul"),g=r("li"),l=n("having all inputs as keyword arguments (like PyTorch models), or"),f=c(),E=r("li"),be=n("having all inputs as a list, tuple or dict in the first positional arguments."),X=c(),z=r("p"),ne=n("This second option is useful when using "),L=r("code"),re=n("tf.keras.Model.fit"),ke=n(` method which currently requires having all the
tensors in the first argument of the model call function: `),D=r("code"),Te=n("model(inputs)"),me=n("."),J=c(),O=r("p"),ae=n(`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),Y=c(),P=r("ul"),j=r("li"),ie=n("a single Tensor with "),H=r("code"),fe=n("input_ids"),le=n(" only and nothing else: "),S=r("code"),ve=n("model(inputs_ids)"),ue=c(),q=r("li"),ce=n(`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),R=r("code"),ge=n("model([input_ids, attention_mask])"),de=n(" or "),V=r("code"),_e=n("model([input_ids, attention_mask, token_type_ids])"),se=c(),N=r("li"),ye=n(`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),Q=r("code"),pe=n('model({"input_ids": input_ids, "token_type_ids": token_type_ids})')},l(T){d=a(T,"P",{});var M=i(d);_=s(M,"TF 2.0 models accepts two formats as inputs:"),M.forEach(t),m=p(T),h=a(T,"UL",{});var K=i(h);g=a(K,"LI",{});var ze=i(g);l=s(ze,"having all inputs as keyword arguments (like PyTorch models), or"),ze.forEach(t),f=p(K),E=a(K,"LI",{});var Be=i(E);be=s(Be,"having all inputs as a list, tuple or dict in the first positional arguments."),Be.forEach(t),K.forEach(t),X=p(T),z=a(T,"P",{});var I=i(z);ne=s(I,"This second option is useful when using "),L=a(I,"CODE",{});var Pe=i(L);re=s(Pe,"tf.keras.Model.fit"),Pe.forEach(t),ke=s(I,` method which currently requires having all the
tensors in the first argument of the model call function: `),D=a(I,"CODE",{});var Ee=i(D);Te=s(Ee,"model(inputs)"),Ee.forEach(t),me=s(I,"."),I.forEach(t),J=p(T),O=a(T,"P",{});var qe=i(O);ae=s(qe,`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),qe.forEach(t),Y=p(T),P=a(T,"UL",{});var A=i(P);j=a(A,"LI",{});var W=i(j);ie=s(W,"a single Tensor with "),H=a(W,"CODE",{});var xe=i(H);fe=s(xe,"input_ids"),xe.forEach(t),le=s(W," only and nothing else: "),S=a(W,"CODE",{});var Fe=i(S);ve=s(Fe,"model(inputs_ids)"),Fe.forEach(t),W.forEach(t),ue=p(A),q=a(A,"LI",{});var U=i(q);ce=s(U,`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),R=a(U,"CODE",{});var Me=i(R);ge=s(Me,"model([input_ids, attention_mask])"),Me.forEach(t),de=s(U," or "),V=a(U,"CODE",{});var je=i(V);_e=s(je,"model([input_ids, attention_mask, token_type_ids])"),je.forEach(t),U.forEach(t),se=p(A),N=a(A,"LI",{});var he=i(N);ye=s(he,`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),Q=a(he,"CODE",{});var Ce=i(Q);pe=s(Ce,'model({"input_ids": input_ids, "token_type_ids": token_type_ids})'),Ce.forEach(t),he.forEach(t),A.forEach(t)},m(T,M){b(T,d,M),e(d,_),b(T,m,M),b(T,h,M),e(h,g),e(g,l),e(h,f),e(h,E),e(E,be),b(T,X,M),b(T,z,M),e(z,ne),e(z,L),e(L,re),e(z,ke),e(z,D),e(D,Te),e(z,me),b(T,J,M),b(T,O,M),e(O,ae),b(T,Y,M),b(T,P,M),e(P,j),e(j,ie),e(j,H),e(H,fe),e(j,le),e(j,S),e(S,ve),e(P,ue),e(P,q),e(q,ce),e(q,R),e(R,ge),e(q,de),e(q,V),e(V,_e),e(P,se),e(P,N),e(N,ye),e(N,Q),e(Q,pe)},d(T){T&&t(d),T&&t(m),T&&t(h),T&&t(X),T&&t(z),T&&t(J),T&&t(O),T&&t(Y),T&&t(P)}}}function RN(B){let d,_,m,h,g;return{c(){d=r("p"),_=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r("code"),h=n("Module"),g=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=a(f,"CODE",{});var E=i(m);h=s(E,"Module"),E.forEach(t),g=s(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,_),e(d,m),e(m,h),e(d,g)},d(l){l&&t(d)}}}function VN(B){let d,_,m,h,g;return h=new oe({props:{code:`from transformers import BertTokenizer, TFBertForTokenClassification
import tensorflow as tf

tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = TFBertForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

inputs = tokenizer(
    "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="tf"
)

logits = model(**inputs).logits
predicted_token_class_ids = tf.math.argmax(logits, axis=-1)

# Note that tokens are classified rather then input words which means that
# there might be more predicted token classes than words.
# Multiple token classes might account for the same word
predicted_tokens_classes = [model.config.id2label[t] for t in predicted_token_class_ids[0].numpy().tolist()]
predicted_tokens_classes`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertTokenizer, TFBertForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> tensorflow <span class="hljs-keyword">as</span> tf

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;dbmdz/bert-large-cased-finetuned-conll03-english&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFBertForTokenClassification.from_pretrained(<span class="hljs-string">&quot;dbmdz/bert-large-cased-finetuned-conll03-english&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;HuggingFace is a company based in Paris and New York&quot;</span>, add_special_tokens=<span class="hljs-literal">False</span>, return_tensors=<span class="hljs-string">&quot;tf&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>logits = model(**inputs).logits
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_token_class_ids = tf.math.argmax(logits, axis=-<span class="hljs-number">1</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Note that tokens are classified rather then input words which means that</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># there might be more predicted token classes than words.</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Multiple token classes might account for the same word</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_tokens_classes = [model.config.id2label[t] <span class="hljs-keyword">for</span> t <span class="hljs-keyword">in</span> predicted_token_class_ids[<span class="hljs-number">0</span>].numpy().tolist()]
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_tokens_classes
[<span class="hljs-string">&#x27;O&#x27;</span>, <span class="hljs-string">&#x27;I-ORG&#x27;</span>, <span class="hljs-string">&#x27;I-ORG&#x27;</span>, <span class="hljs-string">&#x27;I-ORG&#x27;</span>, <span class="hljs-string">&#x27;O&#x27;</span>, <span class="hljs-string">&#x27;O&#x27;</span>, <span class="hljs-string">&#x27;O&#x27;</span>, <span class="hljs-string">&#x27;O&#x27;</span>, <span class="hljs-string">&#x27;O&#x27;</span>, <span class="hljs-string">&#x27;I-LOC&#x27;</span>, <span class="hljs-string">&#x27;O&#x27;</span>, <span class="hljs-string">&#x27;I-LOC&#x27;</span>, <span class="hljs-string">&#x27;I-LOC&#x27;</span>] `}}),{c(){d=r("p"),_=n("Example:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Example:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function QN(B){let d,_;return d=new oe({props:{code:`labels = predicted_token_class_ids
loss = tf.math.reduce_mean(model(**inputs, labels=labels).loss)
round(float(loss), 2)`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span>labels = predicted_token_class_ids
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = tf.math.reduce_mean(model(**inputs, labels=labels).loss)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(<span class="hljs-built_in">float</span>(loss), <span class="hljs-number">2</span>)
<span class="hljs-number">0.01</span>`}}),{c(){v(d.$$.fragment)},l(m){y(d.$$.fragment,m)},m(m,h){w(d,m,h),_=!0},p:te,i(m){_||($(d.$$.fragment,m),_=!0)},o(m){x(d.$$.fragment,m),_=!1},d(m){F(d,m)}}}function KN(B){let d,_,m,h,g,l,f,E,be,X,z,ne,L,re,ke,D,Te,me,J,O,ae,Y,P,j,ie,H,fe,le,S,ve,ue,q,ce,R,ge,de,V,_e,se,N,ye,Q,pe;return{c(){d=r("p"),_=n("TF 2.0 models accepts two formats as inputs:"),m=c(),h=r("ul"),g=r("li"),l=n("having all inputs as keyword arguments (like PyTorch models), or"),f=c(),E=r("li"),be=n("having all inputs as a list, tuple or dict in the first positional arguments."),X=c(),z=r("p"),ne=n("This second option is useful when using "),L=r("code"),re=n("tf.keras.Model.fit"),ke=n(` method which currently requires having all the
tensors in the first argument of the model call function: `),D=r("code"),Te=n("model(inputs)"),me=n("."),J=c(),O=r("p"),ae=n(`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),Y=c(),P=r("ul"),j=r("li"),ie=n("a single Tensor with "),H=r("code"),fe=n("input_ids"),le=n(" only and nothing else: "),S=r("code"),ve=n("model(inputs_ids)"),ue=c(),q=r("li"),ce=n(`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),R=r("code"),ge=n("model([input_ids, attention_mask])"),de=n(" or "),V=r("code"),_e=n("model([input_ids, attention_mask, token_type_ids])"),se=c(),N=r("li"),ye=n(`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),Q=r("code"),pe=n('model({"input_ids": input_ids, "token_type_ids": token_type_ids})')},l(T){d=a(T,"P",{});var M=i(d);_=s(M,"TF 2.0 models accepts two formats as inputs:"),M.forEach(t),m=p(T),h=a(T,"UL",{});var K=i(h);g=a(K,"LI",{});var ze=i(g);l=s(ze,"having all inputs as keyword arguments (like PyTorch models), or"),ze.forEach(t),f=p(K),E=a(K,"LI",{});var Be=i(E);be=s(Be,"having all inputs as a list, tuple or dict in the first positional arguments."),Be.forEach(t),K.forEach(t),X=p(T),z=a(T,"P",{});var I=i(z);ne=s(I,"This second option is useful when using "),L=a(I,"CODE",{});var Pe=i(L);re=s(Pe,"tf.keras.Model.fit"),Pe.forEach(t),ke=s(I,` method which currently requires having all the
tensors in the first argument of the model call function: `),D=a(I,"CODE",{});var Ee=i(D);Te=s(Ee,"model(inputs)"),Ee.forEach(t),me=s(I,"."),I.forEach(t),J=p(T),O=a(T,"P",{});var qe=i(O);ae=s(qe,`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),qe.forEach(t),Y=p(T),P=a(T,"UL",{});var A=i(P);j=a(A,"LI",{});var W=i(j);ie=s(W,"a single Tensor with "),H=a(W,"CODE",{});var xe=i(H);fe=s(xe,"input_ids"),xe.forEach(t),le=s(W," only and nothing else: "),S=a(W,"CODE",{});var Fe=i(S);ve=s(Fe,"model(inputs_ids)"),Fe.forEach(t),W.forEach(t),ue=p(A),q=a(A,"LI",{});var U=i(q);ce=s(U,`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),R=a(U,"CODE",{});var Me=i(R);ge=s(Me,"model([input_ids, attention_mask])"),Me.forEach(t),de=s(U," or "),V=a(U,"CODE",{});var je=i(V);_e=s(je,"model([input_ids, attention_mask, token_type_ids])"),je.forEach(t),U.forEach(t),se=p(A),N=a(A,"LI",{});var he=i(N);ye=s(he,`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),Q=a(he,"CODE",{});var Ce=i(Q);pe=s(Ce,'model({"input_ids": input_ids, "token_type_ids": token_type_ids})'),Ce.forEach(t),he.forEach(t),A.forEach(t)},m(T,M){b(T,d,M),e(d,_),b(T,m,M),b(T,h,M),e(h,g),e(g,l),e(h,f),e(h,E),e(E,be),b(T,X,M),b(T,z,M),e(z,ne),e(z,L),e(L,re),e(z,ke),e(z,D),e(D,Te),e(z,me),b(T,J,M),b(T,O,M),e(O,ae),b(T,Y,M),b(T,P,M),e(P,j),e(j,ie),e(j,H),e(H,fe),e(j,le),e(j,S),e(S,ve),e(P,ue),e(P,q),e(q,ce),e(q,R),e(R,ge),e(q,de),e(q,V),e(V,_e),e(P,se),e(P,N),e(N,ye),e(N,Q),e(Q,pe)},d(T){T&&t(d),T&&t(m),T&&t(h),T&&t(X),T&&t(z),T&&t(J),T&&t(O),T&&t(Y),T&&t(P)}}}function JN(B){let d,_,m,h,g;return{c(){d=r("p"),_=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r("code"),h=n("Module"),g=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=a(f,"CODE",{});var E=i(m);h=s(E,"Module"),E.forEach(t),g=s(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,_),e(d,m),e(m,h),e(d,g)},d(l){l&&t(d)}}}function GN(B){let d,_,m,h,g;return h=new oe({props:{code:`from transformers import BertTokenizer, TFBertForQuestionAnswering
import tensorflow as tf

tokenizer = BertTokenizer.from_pretrained("ydshieh/bert-base-cased-squad2")
model = TFBertForQuestionAnswering.from_pretrained("ydshieh/bert-base-cased-squad2")

question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

inputs = tokenizer(question, text, return_tensors="tf")
outputs = model(**inputs)

answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
tokenizer.decode(predict_answer_tokens)`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertTokenizer, TFBertForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> tensorflow <span class="hljs-keyword">as</span> tf

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;ydshieh/bert-base-cased-squad2&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFBertForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;ydshieh/bert-base-cased-squad2&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>question, text = <span class="hljs-string">&quot;Who was Jim Henson?&quot;</span>, <span class="hljs-string">&quot;Jim Henson was a nice puppet&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(question, text, return_tensors=<span class="hljs-string">&quot;tf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>answer_start_index = <span class="hljs-built_in">int</span>(tf.math.argmax(outputs.start_logits, axis=-<span class="hljs-number">1</span>)[<span class="hljs-number">0</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>answer_end_index = <span class="hljs-built_in">int</span>(tf.math.argmax(outputs.end_logits, axis=-<span class="hljs-number">1</span>)[<span class="hljs-number">0</span>])

<span class="hljs-meta">&gt;&gt;&gt; </span>predict_answer_tokens = inputs.input_ids[<span class="hljs-number">0</span>, answer_start_index : answer_end_index + <span class="hljs-number">1</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.decode(predict_answer_tokens)
<span class="hljs-string">&#x27;a nice puppet&#x27;</span>`}}),{c(){d=r("p"),_=n("Example:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Example:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function XN(B){let d,_;return d=new oe({props:{code:`# target is "nice puppet"
target_start_index = tf.constant([14])
target_end_index = tf.constant([15])

outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
loss = tf.math.reduce_mean(outputs.loss)
round(float(loss), 2)`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># target is &quot;nice puppet&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>target_start_index = tf.constant([<span class="hljs-number">14</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>target_end_index = tf.constant([<span class="hljs-number">15</span>])

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = tf.math.reduce_mean(outputs.loss)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(<span class="hljs-built_in">float</span>(loss), <span class="hljs-number">2</span>)
<span class="hljs-number">7.41</span>`}}),{c(){v(d.$$.fragment)},l(m){y(d.$$.fragment,m)},m(m,h){w(d,m,h),_=!0},p:te,i(m){_||($(d.$$.fragment,m),_=!0)},o(m){x(d.$$.fragment,m),_=!1},d(m){F(d,m)}}}function YN(B){let d,_,m,h,g;return{c(){d=r("p"),_=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r("code"),h=n("Module"),g=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=a(f,"CODE",{});var E=i(m);h=s(E,"Module"),E.forEach(t),g=s(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,_),e(d,m),e(m,h),e(d,g)},d(l){l&&t(d)}}}function ZN(B){let d,_,m,h,g;return h=new oe({props:{code:`from transformers import BertTokenizer, FlaxBertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = FlaxBertModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertTokenizer, FlaxBertModel

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FlaxBertModel.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;jax&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`}}),{c(){d=r("p"),_=n("Example:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Example:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function eO(B){let d,_,m,h,g;return{c(){d=r("p"),_=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r("code"),h=n("Module"),g=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=a(f,"CODE",{});var E=i(m);h=s(E,"Module"),E.forEach(t),g=s(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,_),e(d,m),e(m,h),e(d,g)},d(l){l&&t(d)}}}function tO(B){let d,_,m,h,g;return h=new oe({props:{code:`from transformers import BertTokenizer, FlaxBertForPreTraining

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = FlaxBertForPreTraining.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="np")
outputs = model(**inputs)

prediction_logits = outputs.prediction_logits
seq_relationship_logits = outputs.seq_relationship_logits`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertTokenizer, FlaxBertForPreTraining

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FlaxBertForPreTraining.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;np&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>prediction_logits = outputs.prediction_logits
<span class="hljs-meta">&gt;&gt;&gt; </span>seq_relationship_logits = outputs.seq_relationship_logits`}}),{c(){d=r("p"),_=n("Example:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Example:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function oO(B){let d,_,m,h,g;return{c(){d=r("p"),_=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r("code"),h=n("Module"),g=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=a(f,"CODE",{});var E=i(m);h=s(E,"Module"),E.forEach(t),g=s(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,_),e(d,m),e(m,h),e(d,g)},d(l){l&&t(d)}}}function nO(B){let d,_,m,h,g;return h=new oe({props:{code:`from transformers import BertTokenizer, FlaxBertForCausalLM

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = FlaxBertForCausalLM.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="np")
outputs = model(**inputs)

# retrieve logts for next token
next_token_logits = outputs.logits[:, -1]`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertTokenizer, FlaxBertForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FlaxBertForCausalLM.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;np&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># retrieve logts for next token</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>next_token_logits = outputs.logits[:, -<span class="hljs-number">1</span>]`}}),{c(){d=r("p"),_=n("Example:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Example:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function sO(B){let d,_,m,h,g;return{c(){d=r("p"),_=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r("code"),h=n("Module"),g=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=a(f,"CODE",{});var E=i(m);h=s(E,"Module"),E.forEach(t),g=s(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,_),e(d,m),e(m,h),e(d,g)},d(l){l&&t(d)}}}function rO(B){let d,_,m,h,g;return h=new oe({props:{code:`from transformers import BertTokenizer, FlaxBertForMaskedLM

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = FlaxBertForMaskedLM.from_pretrained("bert-base-uncased")

inputs = tokenizer("The capital of France is [MASK].", return_tensors="jax")

outputs = model(**inputs)
logits = outputs.logits`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertTokenizer, FlaxBertForMaskedLM

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FlaxBertForMaskedLM.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;The capital of France is [MASK].&quot;</span>, return_tensors=<span class="hljs-string">&quot;jax&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`}}),{c(){d=r("p"),_=n("Example:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Example:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function aO(B){let d,_,m,h,g;return{c(){d=r("p"),_=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r("code"),h=n("Module"),g=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=a(f,"CODE",{});var E=i(m);h=s(E,"Module"),E.forEach(t),g=s(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,_),e(d,m),e(m,h),e(d,g)},d(l){l&&t(d)}}}function iO(B){let d,_,m,h,g;return h=new oe({props:{code:`from transformers import BertTokenizer, FlaxBertForNextSentencePrediction

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = FlaxBertForNextSentencePrediction.from_pretrained("bert-base-uncased")

prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
next_sentence = "The sky is blue due to the shorter wavelength of blue light."
encoding = tokenizer(prompt, next_sentence, return_tensors="jax")

outputs = model(**encoding)
logits = outputs.logits
assert logits[0, 0] < logits[0, 1]  # next sentence was random`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertTokenizer, FlaxBertForNextSentencePrediction

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FlaxBertForNextSentencePrediction.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>next_sentence = <span class="hljs-string">&quot;The sky is blue due to the shorter wavelength of blue light.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer(prompt, next_sentence, return_tensors=<span class="hljs-string">&quot;jax&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**encoding)
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">assert</span> logits[<span class="hljs-number">0</span>, <span class="hljs-number">0</span>] &lt; logits[<span class="hljs-number">0</span>, <span class="hljs-number">1</span>]  <span class="hljs-comment"># next sentence was random</span>`}}),{c(){d=r("p"),_=n("Example:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Example:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function lO(B){let d,_,m,h,g;return{c(){d=r("p"),_=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r("code"),h=n("Module"),g=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=a(f,"CODE",{});var E=i(m);h=s(E,"Module"),E.forEach(t),g=s(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,_),e(d,m),e(m,h),e(d,g)},d(l){l&&t(d)}}}function dO(B){let d,_,m,h,g;return h=new oe({props:{code:`from transformers import BertTokenizer, FlaxBertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = FlaxBertForSequenceClassification.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")

outputs = model(**inputs)
logits = outputs.logits`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertTokenizer, FlaxBertForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FlaxBertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;jax&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`}}),{c(){d=r("p"),_=n("Example:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Example:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function cO(B){let d,_,m,h,g;return{c(){d=r("p"),_=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r("code"),h=n("Module"),g=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=a(f,"CODE",{});var E=i(m);h=s(E,"Module"),E.forEach(t),g=s(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,_),e(d,m),e(m,h),e(d,g)},d(l){l&&t(d)}}}function pO(B){let d,_,m,h,g;return h=new oe({props:{code:`from transformers import BertTokenizer, FlaxBertForMultipleChoice

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = FlaxBertForMultipleChoice.from_pretrained("bert-base-uncased")

prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
choice0 = "It is eaten with a fork and a knife."
choice1 = "It is eaten while held in the hand."

encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="jax", padding=True)
outputs = model(**{k: v[None, :] for k, v in encoding.items()})

logits = outputs.logits`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertTokenizer, FlaxBertForMultipleChoice

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FlaxBertForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors=<span class="hljs-string">&quot;jax&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**{k: v[<span class="hljs-literal">None</span>, :] <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()})

<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`}}),{c(){d=r("p"),_=n("Example:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Example:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function hO(B){let d,_,m,h,g;return{c(){d=r("p"),_=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r("code"),h=n("Module"),g=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=a(f,"CODE",{});var E=i(m);h=s(E,"Module"),E.forEach(t),g=s(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,_),e(d,m),e(m,h),e(d,g)},d(l){l&&t(d)}}}function mO(B){let d,_,m,h,g;return h=new oe({props:{code:`from transformers import BertTokenizer, FlaxBertForTokenClassification

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = FlaxBertForTokenClassification.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")

outputs = model(**inputs)
logits = outputs.logits`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertTokenizer, FlaxBertForTokenClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FlaxBertForTokenClassification.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;jax&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`}}),{c(){d=r("p"),_=n("Example:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Example:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function fO(B){let d,_,m,h,g;return{c(){d=r("p"),_=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r("code"),h=n("Module"),g=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=a(f,"CODE",{});var E=i(m);h=s(E,"Module"),E.forEach(t),g=s(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,_),e(d,m),e(m,h),e(d,g)},d(l){l&&t(d)}}}function uO(B){let d,_,m,h,g;return h=new oe({props:{code:`from transformers import BertTokenizer, FlaxBertForQuestionAnswering

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = FlaxBertForQuestionAnswering.from_pretrained("bert-base-uncased")

question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
inputs = tokenizer(question, text, return_tensors="jax")

outputs = model(**inputs)
start_scores = outputs.start_logits
end_scores = outputs.end_logits`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertTokenizer, FlaxBertForQuestionAnswering

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FlaxBertForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>question, text = <span class="hljs-string">&quot;Who was Jim Henson?&quot;</span>, <span class="hljs-string">&quot;Jim Henson was a nice puppet&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(question, text, return_tensors=<span class="hljs-string">&quot;jax&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>start_scores = outputs.start_logits
<span class="hljs-meta">&gt;&gt;&gt; </span>end_scores = outputs.end_logits`}}),{c(){d=r("p"),_=n("Example:"),m=c(),v(h.$$.fragment)},l(l){d=a(l,"P",{});var f=i(d);_=s(f,"Example:"),f.forEach(t),m=p(l),y(h.$$.fragment,l)},m(l,f){b(l,d,f),e(d,_),b(l,m,f),w(h,l,f),g=!0},p:te,i(l){g||($(h.$$.fragment,l),g=!0)},o(l){x(h.$$.fragment,l),g=!1},d(l){l&&t(d),l&&t(m),F(h,l)}}}function gO(B){let d,_,m,h,g,l,f,E,be,X,z,ne,L,re,ke,D,Te,me,J,O,ae,Y,P,j,ie,H,fe,le,S,ve,ue,q,ce,R,ge,de,V,_e,se,N,ye,Q,pe,T,M,K,ze,Be,I,Pe,Ee,qe,A,W,xe,Fe,U,Me,je,he,Ce,Eb,Nt,Ra,jv,ho,Cv,Ep,Nv,Ov,Mp,Iv,Av,Va,Lv,Dv,Sv,Ko,Uv,zp,Wv,Hv,Pp,Rv,Vv,Qv,_s,Mb,Jo,bs,km,Qa,Kv,Tm,Jv,zb,Ne,Ka,Gv,vm,Xv,Yv,Ja,Zv,qp,ey,ty,oy,Bo,Ga,ny,ym,sy,ry,Xa,jp,ay,wm,iy,ly,Cp,dy,$m,cy,py,ks,Ya,hy,Za,my,xm,fy,uy,gy,It,ei,_y,Fm,by,ky,Ts,Ty,Go,vy,Bm,yy,wy,Em,$y,xy,Fy,Np,ti,Pb,Xo,vs,Mm,oi,By,zm,Ey,qb,rt,ni,My,si,zy,Pm,Py,qy,jy,ri,Cy,Op,Ny,Oy,Iy,Eo,ai,Ay,qm,Ly,Dy,ii,Ip,Sy,jm,Uy,Wy,Ap,Hy,Cm,Ry,Vy,At,li,Qy,Nm,Ky,Jy,ys,Gy,Yo,Xy,Om,Yy,Zy,Im,e1,t1,jb,Zo,ws,Am,di,o1,Lm,n1,Cb,at,ci,s1,en,r1,Dm,a1,i1,Sm,l1,d1,c1,pi,p1,Um,h1,m1,f1,Mo,hi,u1,mi,g1,Wm,_1,b1,k1,$s,T1,zo,fi,v1,tn,y1,Hm,w1,$1,Rm,x1,F1,B1,xs,Nb,on,Fs,Vm,ui,E1,Qm,M1,Ob,nn,gi,z1,_i,P1,Lp,q1,j1,Ib,sn,bi,C1,ki,N1,Dp,O1,I1,Ab,mo,Ti,A1,vi,L1,Sp,D1,S1,U1,Bs,yi,W1,Km,H1,Lb,rn,Es,Jm,wi,R1,Gm,V1,Db,Oe,$i,Q1,Xm,K1,J1,xi,G1,Up,X1,Y1,Z1,Fi,ew,Bi,tw,ow,nw,Ei,sw,Mi,rw,aw,iw,Ke,lw,Ym,dw,cw,Zm,pw,hw,ef,mw,fw,tf,uw,gw,of,_w,bw,nf,kw,Tw,vw,Lt,zi,yw,an,ww,Wp,$w,xw,sf,Fw,Bw,Ew,Ms,Mw,zs,Sb,ln,Ps,rf,Pi,zw,af,Pw,Ub,it,qi,qw,dn,jw,lf,Cw,Nw,df,Ow,Iw,Aw,ji,Lw,Hp,Dw,Sw,Uw,Ci,Ww,Ni,Hw,Rw,Vw,Dt,Oi,Qw,cn,Kw,Rp,Jw,Gw,cf,Xw,Yw,Zw,qs,e2,js,Wb,pn,Cs,pf,Ii,t2,hf,o2,Hb,lt,Ai,n2,Li,s2,mf,r2,a2,i2,Di,l2,Vp,d2,c2,p2,Si,h2,Ui,m2,f2,u2,St,Wi,g2,hn,_2,Qp,b2,k2,ff,T2,v2,y2,Ns,w2,Os,Rb,mn,Is,uf,Hi,$2,gf,x2,Vb,dt,Ri,F2,Vi,B2,_f,E2,M2,z2,Qi,P2,Kp,q2,j2,C2,Ki,N2,Ji,O2,I2,A2,ut,Gi,L2,fn,D2,Jp,S2,U2,bf,W2,H2,R2,As,V2,Ls,Q2,Ds,Qb,un,Ss,kf,Xi,K2,Tf,J2,Kb,ct,Yi,G2,Zi,X2,vf,Y2,Z2,e$,el,t$,Gp,o$,n$,s$,tl,r$,ol,a$,i$,l$,Ut,nl,d$,gn,c$,Xp,p$,h$,yf,m$,f$,u$,Us,g$,Ws,Jb,_n,Hs,wf,sl,_$,$f,b$,Gb,pt,rl,k$,xf,T$,v$,al,y$,Yp,w$,$$,x$,il,F$,ll,B$,E$,M$,Qe,dl,z$,bn,P$,Zp,q$,j$,Ff,C$,N$,O$,Rs,I$,Vs,A$,Qs,L$,Ks,D$,Js,Xb,kn,Gs,Bf,cl,S$,Ef,U$,Yb,ht,pl,W$,Mf,H$,R$,hl,V$,eh,Q$,K$,J$,ml,G$,fl,X$,Y$,Z$,Wt,ul,ex,Tn,tx,th,ox,nx,zf,sx,rx,ax,Xs,ix,Ys,Zb,vn,Zs,Pf,gl,lx,qf,dx,ek,mt,_l,cx,jf,px,hx,bl,mx,oh,fx,ux,gx,kl,_x,Tl,bx,kx,Tx,gt,vl,vx,yn,yx,nh,wx,$x,Cf,xx,Fx,Bx,er,Ex,tr,Mx,or,tk,wn,nr,Nf,yl,zx,Of,Px,ok,ft,wl,qx,$n,jx,If,Cx,Nx,Af,Ox,Ix,Ax,$l,Lx,sh,Dx,Sx,Ux,xl,Wx,Fl,Hx,Rx,Vx,_t,Bl,Qx,xn,Kx,rh,Jx,Gx,Lf,Xx,Yx,Zx,sr,eF,rr,tF,ar,nk,Fn,ir,Df,El,oF,Sf,nF,sk,Je,Ml,sF,Uf,rF,aF,zl,iF,ah,lF,dF,cF,Pl,pF,ql,hF,mF,fF,lr,uF,Ht,jl,gF,Bn,_F,ih,bF,kF,Wf,TF,vF,yF,dr,wF,cr,rk,En,pr,Hf,Cl,$F,Rf,xF,ak,Ge,Nl,FF,Mn,BF,Vf,EF,MF,Qf,zF,PF,qF,Ol,jF,lh,CF,NF,OF,Il,IF,Al,AF,LF,DF,hr,SF,Rt,Ll,UF,zn,WF,dh,HF,RF,Kf,VF,QF,KF,mr,JF,fr,ik,Pn,ur,Jf,Dl,GF,Gf,XF,lk,qn,Sl,YF,bt,Ul,ZF,Ie,e0,Xf,t0,o0,Yf,n0,s0,Zf,r0,a0,eu,i0,l0,tu,d0,c0,ou,p0,h0,nu,m0,f0,u0,Wl,Hl,g0,su,_0,b0,k0,Rl,T0,ru,v0,y0,w0,G,$0,au,x0,F0,iu,B0,E0,lu,M0,z0,du,P0,q0,cu,j0,C0,pu,N0,O0,hu,I0,A0,mu,L0,D0,fu,S0,U0,uu,W0,H0,gu,R0,V0,_u,Q0,K0,bu,J0,G0,ku,X0,Y0,Tu,Z0,eB,vu,tB,oB,yu,nB,sB,wu,rB,aB,$u,iB,lB,xu,dB,cB,pB,gr,dk,jn,_r,Fu,Vl,hB,Bu,mB,ck,Xe,Ql,fB,Kl,uB,Eu,gB,_B,bB,Jl,kB,ch,TB,vB,yB,Gl,wB,Xl,$B,xB,FB,br,BB,kt,Yl,EB,Cn,MB,ph,zB,PB,Mu,qB,jB,CB,kr,NB,Tr,OB,vr,pk,Nn,yr,zu,Zl,IB,Pu,AB,hk,Ye,ed,LB,td,DB,qu,SB,UB,WB,od,HB,hh,RB,VB,QB,nd,KB,sd,JB,GB,XB,wr,YB,Vt,rd,ZB,On,eE,mh,tE,oE,ju,nE,sE,rE,$r,aE,xr,mk,In,Fr,Cu,ad,iE,Nu,lE,fk,Ze,id,dE,Ou,cE,pE,ld,hE,fh,mE,fE,uE,dd,gE,cd,_E,bE,kE,Br,TE,Tt,pd,vE,An,yE,uh,wE,$E,Iu,xE,FE,BE,Er,EE,Mr,ME,zr,uk,Ln,Pr,Au,hd,zE,Lu,PE,gk,et,md,qE,Du,jE,CE,fd,NE,gh,OE,IE,AE,ud,LE,gd,DE,SE,UE,qr,WE,Qt,_d,HE,Dn,RE,_h,VE,QE,Su,KE,JE,GE,jr,XE,Cr,_k,Sn,Nr,Uu,bd,YE,Wu,ZE,bk,tt,kd,eM,Hu,tM,oM,Td,nM,bh,sM,rM,aM,vd,iM,yd,lM,dM,cM,Or,pM,vt,wd,hM,Un,mM,kh,fM,uM,Ru,gM,_M,bM,Ir,kM,Ar,TM,Lr,kk,Wn,Dr,Vu,$d,vM,Qu,yM,Tk,ot,xd,wM,Hn,$M,Ku,xM,FM,Ju,BM,EM,MM,Fd,zM,Th,PM,qM,jM,Bd,CM,Ed,NM,OM,IM,Sr,AM,yt,Md,LM,Rn,DM,vh,SM,UM,Gu,WM,HM,RM,Ur,VM,Wr,QM,Hr,vk,Vn,Rr,Xu,zd,KM,Yu,JM,yk,Ae,Pd,GM,Zu,XM,YM,qd,ZM,yh,ez,tz,oz,jd,nz,Cd,sz,rz,az,eg,iz,lz,fo,tg,Nd,dz,cz,og,Od,pz,hz,ng,Id,mz,fz,sg,Ad,uz,gz,Kt,Ld,_z,Qn,bz,rg,kz,Tz,ag,vz,yz,wz,Vr,$z,Qr,wk,Kn,Kr,ig,Dd,xz,lg,Fz,$k,Le,Sd,Bz,Jn,Ez,dg,Mz,zz,cg,Pz,qz,jz,Ud,Cz,wh,Nz,Oz,Iz,Wd,Az,Hd,Lz,Dz,Sz,pg,Uz,Wz,uo,hg,Rd,Hz,Rz,mg,Vd,Vz,Qz,fg,Qd,Kz,Jz,ug,Kd,Gz,Xz,Jt,Jd,Yz,Gn,Zz,gg,e4,t4,_g,o4,n4,s4,Jr,r4,Gr,xk,Xn,Xr,bg,Gd,a4,kg,i4,Fk,De,Xd,l4,Tg,d4,c4,Yd,p4,$h,h4,m4,f4,Zd,u4,ec,g4,_4,b4,vg,k4,T4,go,yg,tc,v4,y4,wg,oc,w4,$4,$g,nc,x4,F4,xg,sc,B4,E4,Gt,rc,M4,Yn,z4,Fg,P4,q4,Bg,j4,C4,N4,Yr,O4,Zr,Bk,Zn,ea,Eg,ac,I4,Mg,A4,Ek,Se,ic,L4,lc,D4,zg,S4,U4,W4,dc,H4,xh,R4,V4,Q4,cc,K4,pc,J4,G4,X4,Pg,Y4,Z4,_o,qg,hc,eP,tP,jg,mc,oP,nP,Cg,fc,sP,rP,Ng,uc,aP,iP,Xt,gc,lP,es,dP,Og,cP,pP,Ig,hP,mP,fP,ta,uP,oa,Mk,ts,na,Ag,_c,gP,Lg,_P,zk,Ue,bc,bP,kc,kP,Dg,TP,vP,yP,Tc,wP,Fh,$P,xP,FP,vc,BP,yc,EP,MP,zP,Sg,PP,qP,bo,Ug,wc,jP,CP,Wg,$c,NP,OP,Hg,xc,IP,AP,Rg,Fc,LP,DP,Yt,Bc,SP,os,UP,Vg,WP,HP,Qg,RP,VP,QP,sa,KP,ra,Pk,ns,aa,Kg,Ec,JP,Jg,GP,qk,We,Mc,XP,Gg,YP,ZP,zc,e8,Bh,t8,o8,n8,Pc,s8,qc,r8,a8,i8,Xg,l8,d8,ko,Yg,jc,c8,p8,Zg,Cc,h8,m8,e_,Nc,f8,u8,t_,Oc,g8,_8,Zt,Ic,b8,ss,k8,o_,T8,v8,n_,y8,w8,$8,ia,x8,la,jk,rs,da,s_,Ac,F8,r_,B8,Ck,He,Lc,E8,a_,M8,z8,Dc,P8,Eh,q8,j8,C8,Sc,N8,Uc,O8,I8,A8,i_,L8,D8,To,l_,Wc,S8,U8,d_,Hc,W8,H8,c_,Rc,R8,V8,p_,Vc,Q8,K8,eo,Qc,J8,as,G8,h_,X8,Y8,m_,Z8,eq,tq,ca,oq,pa,Nk,is,ha,f_,Kc,nq,u_,sq,Ok,Re,Jc,rq,g_,aq,iq,Gc,lq,Mh,dq,cq,pq,Xc,hq,Yc,mq,fq,uq,__,gq,_q,vo,b_,Zc,bq,kq,k_,ep,Tq,vq,T_,tp,yq,wq,v_,op,$q,xq,to,np,Fq,ls,Bq,y_,Eq,Mq,w_,zq,Pq,qq,ma,jq,fa,Ik,ds,ua,$_,sp,Cq,x_,Nq,Ak,Ve,rp,Oq,cs,Iq,F_,Aq,Lq,B_,Dq,Sq,Uq,ap,Wq,zh,Hq,Rq,Vq,ip,Qq,lp,Kq,Jq,Gq,E_,Xq,Yq,yo,M_,dp,Zq,e7,z_,cp,t7,o7,P_,pp,n7,s7,q_,hp,r7,a7,oo,mp,i7,ps,l7,j_,d7,c7,C_,p7,h7,m7,ga,f7,_a,Lk;return l=new $e({}),re=new $e({}),Me=new $e({}),Ra=new C({props:{name:"class transformers.BertConfig",anchor:"transformers.BertConfig",parameters:[{name:"vocab_size",val:" = 30522"},{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 512"},{name:"type_vocab_size",val:" = 2"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"pad_token_id",val:" = 0"},{name:"position_embedding_type",val:" = 'absolute'"},{name:"use_cache",val:" = True"},{name:"classifier_dropout",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BertConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 30522) &#x2014;
Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertModel">BertModel</a> or <a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.TFBertModel">TFBertModel</a>.`,name:"vocab_size"},{anchor:"transformers.BertConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.BertConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.BertConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.BertConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.BertConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>Callable</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.BertConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.BertConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.BertConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.BertConfig.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed when calling <a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertModel">BertModel</a> or <a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.TFBertModel">TFBertModel</a>.`,name:"type_vocab_size"},{anchor:"transformers.BertConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.BertConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.BertConfig.position_embedding_type",description:`<strong>position_embedding_type</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;absolute&quot;</code>) &#x2014;
Type of position embedding. Choose one of <code>&quot;absolute&quot;</code>, <code>&quot;relative_key&quot;</code>, <code>&quot;relative_key_query&quot;</code>. For
positional embeddings use <code>&quot;absolute&quot;</code>. For more information on <code>&quot;relative_key&quot;</code>, please refer to
<a href="https://arxiv.org/abs/1803.02155" rel="nofollow">Self-Attention with Relative Position Representations (Shaw et al.)</a>.
For more information on <code>&quot;relative_key_query&quot;</code>, please refer to <em>Method 4</em> in <a href="https://arxiv.org/abs/2009.13658" rel="nofollow">Improve Transformer Models
with Better Relative Position Embeddings (Huang et al.)</a>.`,name:"position_embedding_type"},{anchor:"transformers.BertConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.BertConfig.classifier_dropout",description:`<strong>classifier_dropout</strong> (<code>float</code>, <em>optional</em>) &#x2014;
The dropout ratio for the classification head.`,name:"classifier_dropout"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/configuration_bert.py#L72"}}),_s=new ee({props:{anchor:"transformers.BertConfig.example",$$slots:{default:[K6]},$$scope:{ctx:B}}}),Qa=new $e({}),Ka=new C({props:{name:"class transformers.BertTokenizer",anchor:"transformers.BertTokenizer",parameters:[{name:"vocab_file",val:""},{name:"do_lower_case",val:" = True"},{name:"do_basic_tokenize",val:" = True"},{name:"never_split",val:" = None"},{name:"unk_token",val:" = '[UNK]'"},{name:"sep_token",val:" = '[SEP]'"},{name:"pad_token",val:" = '[PAD]'"},{name:"cls_token",val:" = '[CLS]'"},{name:"mask_token",val:" = '[MASK]'"},{name:"tokenize_chinese_chars",val:" = True"},{name:"strip_accents",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BertTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
File containing the vocabulary.`,name:"vocab_file"},{anchor:"transformers.BertTokenizer.do_lower_case",description:`<strong>do_lower_case</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to lowercase the input when tokenizing.`,name:"do_lower_case"},{anchor:"transformers.BertTokenizer.do_basic_tokenize",description:`<strong>do_basic_tokenize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to do basic tokenization before WordPiece.`,name:"do_basic_tokenize"},{anchor:"transformers.BertTokenizer.never_split",description:`<strong>never_split</strong> (<code>Iterable</code>, <em>optional</em>) &#x2014;
Collection of tokens which will never be split during tokenization. Only has an effect when
<code>do_basic_tokenize=True</code>`,name:"never_split"},{anchor:"transformers.BertTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[UNK]&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.BertTokenizer.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[SEP]&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.BertTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[PAD]&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.BertTokenizer.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[CLS]&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.BertTokenizer.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[MASK]&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.BertTokenizer.tokenize_chinese_chars",description:`<strong>tokenize_chinese_chars</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to tokenize Chinese characters.</p>
<p>This should likely be deactivated for Japanese (see this
<a href="https://github.com/huggingface/transformers/issues/328" rel="nofollow">issue</a>).`,name:"tokenize_chinese_chars"},{anchor:"transformers.BertTokenizer.strip_accents",description:`<strong>strip_accents</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to strip all accents. If this option is not specified, then it will be determined by the
value for <code>lowercase</code> (as in the original BERT).`,name:"strip_accents"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/tokenization_bert.py#L137"}}),Ga=new C({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.BertTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": typing.List[int]"},{name:"token_ids_1",val:": typing.Optional[typing.List[int]] = None"}],parametersDescription:[{anchor:"transformers.BertTokenizer.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.BertTokenizer.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/tokenization_bert.py#L268",returnDescription:`
<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`
<p><code>List[int]</code></p>
`}}),Ya=new C({props:{name:"get_special_tokens_mask",anchor:"transformers.BertTokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": typing.List[int]"},{name:"token_ids_1",val:": typing.Optional[typing.List[int]] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.BertTokenizer.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.BertTokenizer.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"},{anchor:"transformers.BertTokenizer.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/tokenization_bert.py#L293",returnDescription:`
<p>A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.</p>
`,returnType:`
<p><code>List[int]</code></p>
`}}),ei=new C({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.BertTokenizer.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": typing.List[int]"},{name:"token_ids_1",val:": typing.Optional[typing.List[int]] = None"}],parametersDescription:[{anchor:"transformers.BertTokenizer.create_token_type_ids_from_sequences.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.BertTokenizer.create_token_type_ids_from_sequences.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/tokenization_bert.py#L321",returnDescription:`
<p>List of <a href="../glossary#token-type-ids">token type IDs</a> according to the given sequence(s).</p>
`,returnType:`
<p><code>List[int]</code></p>
`}}),Ts=new ee({props:{anchor:"transformers.BertTokenizer.create_token_type_ids_from_sequences.example",$$slots:{default:[J6]},$$scope:{ctx:B}}}),ti=new C({props:{name:"save_vocabulary",anchor:"transformers.BertTokenizer.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/tokenization_bert.py#L350"}}),oi=new $e({}),ni=new C({props:{name:"class transformers.BertTokenizerFast",anchor:"transformers.BertTokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"do_lower_case",val:" = True"},{name:"unk_token",val:" = '[UNK]'"},{name:"sep_token",val:" = '[SEP]'"},{name:"pad_token",val:" = '[PAD]'"},{name:"cls_token",val:" = '[CLS]'"},{name:"mask_token",val:" = '[MASK]'"},{name:"tokenize_chinese_chars",val:" = True"},{name:"strip_accents",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BertTokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
File containing the vocabulary.`,name:"vocab_file"},{anchor:"transformers.BertTokenizerFast.do_lower_case",description:`<strong>do_lower_case</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to lowercase the input when tokenizing.`,name:"do_lower_case"},{anchor:"transformers.BertTokenizerFast.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[UNK]&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.BertTokenizerFast.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[SEP]&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.BertTokenizerFast.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[PAD]&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.BertTokenizerFast.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[CLS]&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.BertTokenizerFast.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[MASK]&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.BertTokenizerFast.clean_text",description:`<strong>clean_text</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to clean the text before tokenization by removing any control characters and replacing all
whitespaces by the classic one.`,name:"clean_text"},{anchor:"transformers.BertTokenizerFast.tokenize_chinese_chars",description:`<strong>tokenize_chinese_chars</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see <a href="https://github.com/huggingface/transformers/issues/328" rel="nofollow">this
issue</a>).`,name:"tokenize_chinese_chars"},{anchor:"transformers.BertTokenizerFast.strip_accents",description:`<strong>strip_accents</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to strip all accents. If this option is not specified, then it will be determined by the
value for <code>lowercase</code> (as in the original BERT).`,name:"strip_accents"},{anchor:"transformers.BertTokenizerFast.wordpieces_prefix",description:`<strong>wordpieces_prefix</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;##&quot;</code>) &#x2014;
The prefix for subwords.`,name:"wordpieces_prefix"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/tokenization_bert_fast.py#L161"}}),ai=new C({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.BertTokenizerFast.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:""},{name:"token_ids_1",val:" = None"}],parametersDescription:[{anchor:"transformers.BertTokenizerFast.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.BertTokenizerFast.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/tokenization_bert_fast.py#L249",returnDescription:`
<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`
<p><code>List[int]</code></p>
`}}),li=new C({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.BertTokenizerFast.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": typing.List[int]"},{name:"token_ids_1",val:": typing.Optional[typing.List[int]] = None"}],parametersDescription:[{anchor:"transformers.BertTokenizerFast.create_token_type_ids_from_sequences.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.BertTokenizerFast.create_token_type_ids_from_sequences.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/tokenization_bert_fast.py#L273",returnDescription:`
<p>List of <a href="../glossary#token-type-ids">token type IDs</a> according to the given sequence(s).</p>
`,returnType:`
<p><code>List[int]</code></p>
`}}),ys=new ee({props:{anchor:"transformers.BertTokenizerFast.create_token_type_ids_from_sequences.example",$$slots:{default:[G6]},$$scope:{ctx:B}}}),di=new $e({}),ci=new C({props:{name:"class transformers.TFBertTokenizer",anchor:"transformers.TFBertTokenizer",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFBertTokenizer.vocab_list",description:`<strong>vocab_list</strong> (<code>list</code>) &#x2014;
List containing the vocabulary.`,name:"vocab_list"},{anchor:"transformers.TFBertTokenizer.do_lower_case",description:`<strong>do_lower_case</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to lowercase the input when tokenizing.`,name:"do_lower_case"},{anchor:"transformers.TFBertTokenizer.cls_token_id",description:`<strong>cls_token_id</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[CLS]&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token_id"},{anchor:"transformers.TFBertTokenizer.sep_token_id",description:`<strong>sep_token_id</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[SEP]&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token_id"},{anchor:"transformers.TFBertTokenizer.pad_token_id",description:`<strong>pad_token_id</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[PAD]&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token_id"},{anchor:"transformers.TFBertTokenizer.padding",description:`<strong>padding</strong> (<code>str</code>, defaults to <code>&quot;longest&quot;</code>) &#x2014;
The type of padding to use. Can be either <code>&quot;longest&quot;</code>, to pad only up to the longest sample in the batch,
or \`&#x201C;max_length&#x201D;, to pad all inputs to the maximum length supported by the tokenizer.`,name:"padding"},{anchor:"transformers.TFBertTokenizer.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to truncate the sequence to the maximum length.`,name:"truncation"},{anchor:"transformers.TFBertTokenizer.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>, defaults to <code>512</code>) &#x2014;
The maximum length of the sequence, used for padding (if <code>padding</code> is &#x201C;max_length&#x201D;) and/or truncation (if
<code>truncation</code> is <code>True</code>).`,name:"max_length"},{anchor:"transformers.TFBertTokenizer.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>, defaults to <code>None</code>) &#x2014;
If set, the sequence will be padded to a multiple of this value.`,name:"pad_to_multiple_of"},{anchor:"transformers.TFBertTokenizer.return_token_type_ids",description:`<strong>return_token_type_ids</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to return token_type_ids.`,name:"return_token_type_ids"},{anchor:"transformers.TFBertTokenizer.return_attention_mask",description:`<strong>return_attention_mask</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to return the attention_mask.`,name:"return_attention_mask"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/tokenization_bert_tf.py#L11"}}),hi=new C({props:{name:"from_pretrained",anchor:"transformers.TFBertTokenizer.from_pretrained",parameters:[{name:"pretrained_model_name_or_path",val:": typing.Union[str, os.PathLike]"},{name:"*init_inputs",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFBertTokenizer.from_pretrained.pretrained_model_name_or_path",description:`<strong>pretrained_model_name_or_path</strong> (<code>str</code> or <code>os.PathLike</code>) &#x2014;
The name or path to the pre-trained tokenizer.`,name:"pretrained_model_name_or_path"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/tokenization_bert_tf.py#L113"}}),$s=new ee({props:{anchor:"transformers.TFBertTokenizer.from_pretrained.example",$$slots:{default:[X6]},$$scope:{ctx:B}}}),fi=new C({props:{name:"from_tokenizer",anchor:"transformers.TFBertTokenizer.from_tokenizer",parameters:[{name:"tokenizer",val:": PreTrainedTokenizerBase"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFBertTokenizer.from_tokenizer.tokenizer",description:`<strong>tokenizer</strong> (<code>PreTrainedTokenizerBase</code>) &#x2014;
The tokenizer to use to initialize the <code>TFBertTokenizer</code>.`,name:"tokenizer"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/tokenization_bert_tf.py#L83"}}),xs=new ee({props:{anchor:"transformers.TFBertTokenizer.from_tokenizer.example",$$slots:{default:[Y6]},$$scope:{ctx:B}}}),ui=new $e({}),gi=new C({props:{name:"class transformers.models.bert.modeling_bert.BertForPreTrainingOutput",anchor:"transformers.models.bert.modeling_bert.BertForPreTrainingOutput",parameters:[{name:"loss",val:": typing.Optional[torch.FloatTensor] = None"},{name:"prediction_logits",val:": FloatTensor = None"},{name:"seq_relationship_logits",val:": FloatTensor = None"},{name:"hidden_states",val:": typing.Optional[typing.Tuple[torch.FloatTensor]] = None"},{name:"attentions",val:": typing.Optional[typing.Tuple[torch.FloatTensor]] = None"}],parametersDescription:[{anchor:"transformers.models.bert.modeling_bert.BertForPreTrainingOutput.loss",description:`<strong>loss</strong> (<em>optional</em>, returned when <code>labels</code> is provided, <code>torch.FloatTensor</code> of shape <code>(1,)</code>) &#x2014;
Total loss as the sum of the masked language modeling loss and the next sequence prediction
(classification) loss.`,name:"loss"},{anchor:"transformers.models.bert.modeling_bert.BertForPreTrainingOutput.prediction_logits",description:`<strong>prediction_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) &#x2014;
Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).`,name:"prediction_logits"},{anchor:"transformers.models.bert.modeling_bert.BertForPreTrainingOutput.seq_relationship_logits",description:`<strong>seq_relationship_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, 2)</code>) &#x2014;
Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
before SoftMax).`,name:"seq_relationship_logits"},{anchor:"transformers.models.bert.modeling_bert.BertForPreTrainingOutput.hidden_states",description:`<strong>hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the initial embedding outputs.`,name:"hidden_states"},{anchor:"transformers.models.bert.modeling_bert.BertForPreTrainingOutput.attentions",description:`<strong>attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.`,name:"attentions"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_bert.py#L771"}}),bi=new C({props:{name:"class transformers.models.bert.modeling_tf_bert.TFBertForPreTrainingOutput",anchor:"transformers.models.bert.modeling_tf_bert.TFBertForPreTrainingOutput",parameters:[{name:"loss",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"prediction_logits",val:": Tensor = None"},{name:"seq_relationship_logits",val:": Tensor = None"},{name:"hidden_states",val:": typing.Union[typing.Tuple[tensorflow.python.framework.ops.Tensor], tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"attentions",val:": typing.Union[typing.Tuple[tensorflow.python.framework.ops.Tensor], tensorflow.python.framework.ops.Tensor, NoneType] = None"}],parametersDescription:[{anchor:"transformers.models.bert.modeling_tf_bert.TFBertForPreTrainingOutput.prediction_logits",description:`<strong>prediction_logits</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) &#x2014;
Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).`,name:"prediction_logits"},{anchor:"transformers.models.bert.modeling_tf_bert.TFBertForPreTrainingOutput.seq_relationship_logits",description:`<strong>seq_relationship_logits</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, 2)</code>) &#x2014;
Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
before SoftMax).`,name:"seq_relationship_logits"},{anchor:"transformers.models.bert.modeling_tf_bert.TFBertForPreTrainingOutput.hidden_states",description:`<strong>hidden_states</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>tf.Tensor</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the initial embedding outputs.`,name:"hidden_states"},{anchor:"transformers.models.bert.modeling_tf_bert.TFBertForPreTrainingOutput.attentions",description:`<strong>attentions</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>tf.Tensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.`,name:"attentions"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_tf_bert.py#L925"}}),Ti=new C({props:{name:"class transformers.models.bert.modeling_flax_bert.FlaxBertForPreTrainingOutput",anchor:"transformers.models.bert.modeling_flax_bert.FlaxBertForPreTrainingOutput",parameters:[{name:"prediction_logits",val:": ndarray = None"},{name:"seq_relationship_logits",val:": ndarray = None"},{name:"hidden_states",val:": typing.Optional[typing.Tuple[jax._src.numpy.ndarray.ndarray]] = None"},{name:"attentions",val:": typing.Optional[typing.Tuple[jax._src.numpy.ndarray.ndarray]] = None"}],parametersDescription:[{anchor:"transformers.models.bert.modeling_flax_bert.FlaxBertForPreTrainingOutput.prediction_logits",description:`<strong>prediction_logits</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) &#x2014;
Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).`,name:"prediction_logits"},{anchor:"transformers.models.bert.modeling_flax_bert.FlaxBertForPreTrainingOutput.seq_relationship_logits",description:`<strong>seq_relationship_logits</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, 2)</code>) &#x2014;
Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
before SoftMax).`,name:"seq_relationship_logits"},{anchor:"transformers.models.bert.modeling_flax_bert.FlaxBertForPreTrainingOutput.hidden_states",description:`<strong>hidden_states</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>jnp.ndarray</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the initial embedding outputs.`,name:"hidden_states"},{anchor:"transformers.models.bert.modeling_flax_bert.FlaxBertForPreTrainingOutput.attentions",description:`<strong>attentions</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>jnp.ndarray</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.`,name:"attentions"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_flax_bert.py#L64"}}),yi=new C({props:{name:"replace",anchor:"transformers.models.bert.modeling_flax_bert.FlaxBertForPreTrainingOutput.replace",parameters:[{name:"**updates",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/flax/struct.py#L108"}}),wi=new $e({}),$i=new C({props:{name:"class transformers.BertModel",anchor:"transformers.BertModel",parameters:[{name:"config",val:""},{name:"add_pooling_layer",val:" = True"}],parametersDescription:[{anchor:"transformers.BertModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig">BertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_bert.py#L874"}}),zi=new C({props:{name:"forward",anchor:"transformers.BertModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[typing.List[torch.FloatTensor]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.BertModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18729/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18729/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BertModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BertModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.BertModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BertModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BertModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BertModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BertModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BertModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18729/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BertModel.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong>  (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.BertModel.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.BertModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple(tuple(torch.FloatTensor))</code> of length <code>config.n_layers</code> with each tuple having 4 tensors of shape <code>(batch_size, num_heads, sequence_length - 1, embed_size_per_head)</code>) &#x2014;
Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.</p>
<p>If <code>past_key_values</code> are used, the user can optionally input only the last <code>decoder_input_ids</code> (those that
don&#x2019;t have their past key value states given to this model) of shape <code>(batch_size, 1)</code> instead of all
<code>decoder_input_ids</code> of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.BertModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_bert.py#L913",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) \u2014 Sequence of hidden-states at the output of the last layer of the model.</p>
</li>
<li>
<p><strong>pooler_output</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, hidden_size)</code>) \u2014 Last layer hidden-state of the first token of the sequence (classification token) after further processing
through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
the classification token after processing through a linear layer and a tanh activation function. The linear
layer weights are trained from the next sentence prediction (classification) objective during pretraining.</p>
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
<p><strong>cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> and <code>config.add_cross_attention=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder\u2019s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>tuple(tuple(torch.FloatTensor))</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) \u2014 Tuple of <code>tuple(torch.FloatTensor)</code> of length <code>config.n_layers</code>, with each tuple having 2 tensors of shape
<code>(batch_size, num_heads, sequence_length, embed_size_per_head)</code>) and optionally if
<code>config.is_encoder_decoder=True</code> 2 additional tensors of shape <code>(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)</code>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
<code>config.is_encoder_decoder=True</code> in the cross-attention blocks) that can be used (see <code>past_key_values</code>
input) to speed up sequential decoding.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Ms=new we({props:{$$slots:{default:[Z6]},$$scope:{ctx:B}}}),zs=new ee({props:{anchor:"transformers.BertModel.forward.example",$$slots:{default:[eN]},$$scope:{ctx:B}}}),Pi=new $e({}),qi=new C({props:{name:"class transformers.BertForPreTraining",anchor:"transformers.BertForPreTraining",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.BertForPreTraining.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig">BertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_bert.py#L1057"}}),Oi=new C({props:{name:"forward",anchor:"transformers.BertForPreTraining.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"next_sentence_label",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.BertForPreTraining.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18729/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18729/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BertForPreTraining.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BertForPreTraining.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.BertForPreTraining.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BertForPreTraining.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BertForPreTraining.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BertForPreTraining.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BertForPreTraining.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BertForPreTraining.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18729/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.</p>
<p>labels (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>):
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked),
the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>
next_sentence_label (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>):
Labels for computing the next sequence prediction (classification) loss. Input should be a sequence
pair (see <code>input_ids</code> docstring) Indices should be in <code>[0, 1]</code>:</p>
<ul>
<li>0 indicates sequence B is a continuation of sequence A,</li>
<li>1 indicates sequence B is a random sequence.
kwargs (<code>Dict[str, any]</code>, optional, defaults to <em>{}</em>):
Used to hide legacy arguments that have been deprecated.</li>
</ul>`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_bert.py#L1073",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.models.bert.modeling_bert.BertForPreTrainingOutput"
>transformers.models.bert.modeling_bert.BertForPreTrainingOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<em>optional</em>, returned when <code>labels</code> is provided, <code>torch.FloatTensor</code> of shape <code>(1,)</code>) \u2014 Total loss as the sum of the masked language modeling loss and the next sequence prediction
(classification) loss.</p>
</li>
<li>
<p><strong>prediction_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) \u2014 Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>seq_relationship_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, 2)</code>) \u2014 Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.models.bert.modeling_bert.BertForPreTrainingOutput"
>transformers.models.bert.modeling_bert.BertForPreTrainingOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),qs=new we({props:{$$slots:{default:[tN]},$$scope:{ctx:B}}}),js=new ee({props:{anchor:"transformers.BertForPreTraining.forward.example",$$slots:{default:[oN]},$$scope:{ctx:B}}}),Ii=new $e({}),Ai=new C({props:{name:"class transformers.BertLMHeadModel",anchor:"transformers.BertLMHeadModel",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.BertLMHeadModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig">BertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_bert.py#L1161"}}),Wi=new C({props:{name:"forward",anchor:"transformers.BertLMHeadModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[typing.List[torch.Tensor]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.BertLMHeadModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18729/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18729/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BertLMHeadModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BertLMHeadModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.BertLMHeadModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BertLMHeadModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BertLMHeadModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BertLMHeadModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BertLMHeadModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BertLMHeadModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18729/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BertLMHeadModel.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong>  (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.BertLMHeadModel.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.BertLMHeadModel.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
<code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are
ignored (masked), the loss is only computed for the tokens with labels n <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.BertLMHeadModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple(tuple(torch.FloatTensor))</code> of length <code>config.n_layers</code> with each tuple having 4 tensors of shape <code>(batch_size, num_heads, sequence_length - 1, embed_size_per_head)</code>) &#x2014;
Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.</p>
<p>If <code>past_key_values</code> are used, the user can optionally input only the last <code>decoder_input_ids</code> (those that
don&#x2019;t have their past key value states given to this model) of shape <code>(batch_size, 1)</code> instead of all
<code>decoder_input_ids</code> of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.BertLMHeadModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_bert.py#L1184",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Ns=new we({props:{$$slots:{default:[nN]},$$scope:{ctx:B}}}),Os=new ee({props:{anchor:"transformers.BertLMHeadModel.forward.example",$$slots:{default:[sN]},$$scope:{ctx:B}}}),Hi=new $e({}),Ri=new C({props:{name:"class transformers.BertForMaskedLM",anchor:"transformers.BertForMaskedLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.BertForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig">BertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_bert.py#L1296"}}),Gi=new C({props:{name:"forward",anchor:"transformers.BertForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.BertForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18729/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18729/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BertForMaskedLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BertForMaskedLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.BertForMaskedLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BertForMaskedLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BertForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BertForMaskedLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BertForMaskedLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BertForMaskedLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18729/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BertForMaskedLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_bert.py#L1322",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) \u2014 Masked language modeling (MLM) loss.</p>
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
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),As=new we({props:{$$slots:{default:[rN]},$$scope:{ctx:B}}}),Ls=new ee({props:{anchor:"transformers.BertForMaskedLM.forward.example",$$slots:{default:[aN]},$$scope:{ctx:B}}}),Ds=new ee({props:{anchor:"transformers.BertForMaskedLM.forward.example-2",$$slots:{default:[iN]},$$scope:{ctx:B}}}),Xi=new $e({}),Yi=new C({props:{name:"class transformers.BertForNextSentencePrediction",anchor:"transformers.BertForNextSentencePrediction",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.BertForNextSentencePrediction.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig">BertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_bert.py#L1409"}}),nl=new C({props:{name:"forward",anchor:"transformers.BertForNextSentencePrediction.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BertForNextSentencePrediction.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18729/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18729/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BertForNextSentencePrediction.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BertForNextSentencePrediction.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.BertForNextSentencePrediction.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BertForNextSentencePrediction.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BertForNextSentencePrediction.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BertForNextSentencePrediction.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BertForNextSentencePrediction.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BertForNextSentencePrediction.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18729/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BertForNextSentencePrediction.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
(see <code>input_ids</code> docstring). Indices should be in <code>[0, 1]</code>:</p>
<ul>
<li>0 indicates sequence B is a continuation of sequence A,</li>
<li>1 indicates sequence B is a random sequence.</li>
</ul>`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_bert.py#L1419",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_outputs.NextSentencePredictorOutput"
>transformers.modeling_outputs.NextSentencePredictorOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>next_sentence_label</code> is provided) \u2014 Next sequence prediction (classification) loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, 2)</code>) \u2014 Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
before SoftMax).</p>
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
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_outputs.NextSentencePredictorOutput"
>transformers.modeling_outputs.NextSentencePredictorOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Us=new we({props:{$$slots:{default:[lN]},$$scope:{ctx:B}}}),Ws=new ee({props:{anchor:"transformers.BertForNextSentencePrediction.forward.example",$$slots:{default:[dN]},$$scope:{ctx:B}}}),sl=new $e({}),rl=new C({props:{name:"class transformers.BertForSequenceClassification",anchor:"transformers.BertForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.BertForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig">BertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_bert.py#L1514"}}),dl=new C({props:{name:"forward",anchor:"transformers.BertForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.BertForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18729/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18729/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BertForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BertForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.BertForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BertForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BertForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BertForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BertForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BertForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18729/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BertForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_bert.py#L1530",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) \u2014 Classification (or regression if config.num_labels==1) loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.num_labels)</code>) \u2014 Classification (or regression if config.num_labels==1) scores (before SoftMax).</p>
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
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Rs=new we({props:{$$slots:{default:[cN]},$$scope:{ctx:B}}}),Vs=new ee({props:{anchor:"transformers.BertForSequenceClassification.forward.example",$$slots:{default:[pN]},$$scope:{ctx:B}}}),Qs=new ee({props:{anchor:"transformers.BertForSequenceClassification.forward.example-2",$$slots:{default:[hN]},$$scope:{ctx:B}}}),Ks=new ee({props:{anchor:"transformers.BertForSequenceClassification.forward.example-3",$$slots:{default:[mN]},$$scope:{ctx:B}}}),Js=new ee({props:{anchor:"transformers.BertForSequenceClassification.forward.example-4",$$slots:{default:[fN]},$$scope:{ctx:B}}}),cl=new $e({}),pl=new C({props:{name:"class transformers.BertForMultipleChoice",anchor:"transformers.BertForMultipleChoice",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.BertForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig">BertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_bert.py#L1618"}}),ul=new C({props:{name:"forward",anchor:"transformers.BertForMultipleChoice.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.BertForMultipleChoice.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18729/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18729/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BertForMultipleChoice.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BertForMultipleChoice.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.BertForMultipleChoice.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BertForMultipleChoice.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BertForMultipleChoice.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BertForMultipleChoice.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BertForMultipleChoice.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BertForMultipleChoice.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18729/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BertForMultipleChoice.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices-1]</code> where <code>num_choices</code> is the size of the second dimension of the input tensors. (See
<code>input_ids</code> above)`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_bert.py#L1632",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <em>(1,)</em>, <em>optional</em>, returned when <code>labels</code> is provided) \u2014 Classification loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices)</code>) \u2014 <em>num_choices</em> is the second dimension of the input tensors. (see <em>input_ids</em> above).</p>
<p>Classification scores (before SoftMax).</p>
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
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Xs=new we({props:{$$slots:{default:[uN]},$$scope:{ctx:B}}}),Ys=new ee({props:{anchor:"transformers.BertForMultipleChoice.forward.example",$$slots:{default:[gN]},$$scope:{ctx:B}}}),gl=new $e({}),_l=new C({props:{name:"class transformers.BertForTokenClassification",anchor:"transformers.BertForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.BertForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig">BertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_bert.py#L1713"}}),vl=new C({props:{name:"forward",anchor:"transformers.BertForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.BertForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18729/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18729/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BertForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BertForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.BertForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BertForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BertForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BertForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BertForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BertForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18729/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BertForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_bert.py#L1731",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided)  \u2014 Classification loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.num_labels)</code>) \u2014 Classification scores (before SoftMax).</p>
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
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),er=new we({props:{$$slots:{default:[_N]},$$scope:{ctx:B}}}),tr=new ee({props:{anchor:"transformers.BertForTokenClassification.forward.example",$$slots:{default:[bN]},$$scope:{ctx:B}}}),or=new ee({props:{anchor:"transformers.BertForTokenClassification.forward.example-2",$$slots:{default:[kN]},$$scope:{ctx:B}}}),yl=new $e({}),wl=new C({props:{name:"class transformers.BertForQuestionAnswering",anchor:"transformers.BertForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.BertForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig">BertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_bert.py#L1800"}}),Bl=new C({props:{name:"forward",anchor:"transformers.BertForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"start_positions",val:": typing.Optional[torch.Tensor] = None"},{name:"end_positions",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.BertForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18729/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18729/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BertForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BertForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.BertForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BertForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BertForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BertForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BertForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BertForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18729/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BertForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.BertForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_bert.py#L1814",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) \u2014 Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.</p>
</li>
<li>
<p><strong>start_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>) \u2014 Span-start scores (before SoftMax).</p>
</li>
<li>
<p><strong>end_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>) \u2014 Span-end scores (before SoftMax).</p>
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
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),sr=new we({props:{$$slots:{default:[TN]},$$scope:{ctx:B}}}),rr=new ee({props:{anchor:"transformers.BertForQuestionAnswering.forward.example",$$slots:{default:[vN]},$$scope:{ctx:B}}}),ar=new ee({props:{anchor:"transformers.BertForQuestionAnswering.forward.example-2",$$slots:{default:[yN]},$$scope:{ctx:B}}}),El=new $e({}),Ml=new C({props:{name:"class transformers.TFBertModel",anchor:"transformers.TFBertModel",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFBertModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig">BertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_tf_bert.py#L1052"}}),lr=new we({props:{$$slots:{default:[wN]},$$scope:{ctx:B}}}),jl=new C({props:{name:"call",anchor:"transformers.TFBertModel.call",parameters:[{name:"input_ids",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"token_type_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"position_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"head_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"inputs_embeds",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"encoder_hidden_states",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"encoder_attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"past_key_values",val:": typing.Union[typing.Tuple[typing.Tuple[typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor]]], NoneType] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"training",val:": typing.Optional[bool] = False"}],parametersDescription:[{anchor:"transformers.TFBertModel.call.input_ids",description:`<strong>input_ids</strong> (<code>np.ndarray</code>, <code>tf.Tensor</code>, <code>List[tf.Tensor]</code> \`<code>Dict[str, tf.Tensor]</code> or <code>Dict[str, np.ndarray]</code> and each example must have the shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18729/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> and
<a href="/docs/transformers/pr_18729/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.TFBertModel.call.attention_mask",description:`<strong>attention_mask</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.TFBertModel.call.token_type_ids",description:`<strong>token_type_ids</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.TFBertModel.call.position_ids",description:`<strong>position_ids</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.TFBertModel.call.head_mask",description:`<strong>head_mask</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.TFBertModel.call.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.TFBertModel.call.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
config will be used instead.`,name:"output_attentions"},{anchor:"transformers.TFBertModel.call.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
used instead.`,name:"output_hidden_states"},{anchor:"transformers.TFBertModel.call.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18729/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple. This argument can be used in
eager mode, in graph mode the value will always be set to True.`,name:"return_dict"},{anchor:"transformers.TFBertModel.call.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to \`False&#x201C;) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"},{anchor:"transformers.TFBertModel.call.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong>  (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.TFBertModel.call.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.TFBertModel.call.past_key_values",description:`<strong>past_key_values</strong> (<code>Tuple[Tuple[tf.Tensor]]</code> of length <code>config.n_layers</code>) &#x2014;
contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
If <code>past_key_values</code> are used, the user can optionally input only the last <code>decoder_input_ids</code> (those that
don&#x2019;t have their past key value states given to this model) of shape <code>(batch_size, 1)</code> instead of all
<code>decoder_input_ids</code> of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.TFBertModel.call.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>). Set to <code>False</code> during training, <code>True</code> during generation`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_tf_bert.py#L1058",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutputWithPoolingAndCrossAttentions"
>transformers.modeling_tf_outputs.TFBaseModelOutputWithPoolingAndCrossAttentions</a> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) \u2014 Sequence of hidden-states at the output of the last layer of the model.</p>
</li>
<li>
<p><strong>pooler_output</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, hidden_size)</code>) \u2014 Last layer hidden-state of the first token of the sequence (classification token) further processed by a
Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence
prediction (classification) objective during pretraining.</p>
<p>This output is usually <em>not</em> a good summary of the semantic content of the input, you\u2019re often better with
averaging or pooling the sequence of hidden-states for the whole input sequence.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>List[tf.Tensor]</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) \u2014 List of <code>tf.Tensor</code> of length <code>config.n_layers</code>, with each tensor of shape <code>(2, batch_size, num_heads, sequence_length, embed_size_per_head)</code>).</p>
<p>Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
<code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder\u2019s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutputWithPoolingAndCrossAttentions"
>transformers.modeling_tf_outputs.TFBaseModelOutputWithPoolingAndCrossAttentions</a> or <code>tuple(tf.Tensor)</code></p>
`}}),dr=new we({props:{$$slots:{default:[$N]},$$scope:{ctx:B}}}),cr=new ee({props:{anchor:"transformers.TFBertModel.call.example",$$slots:{default:[xN]},$$scope:{ctx:B}}}),Cl=new $e({}),Nl=new C({props:{name:"class transformers.TFBertForPreTraining",anchor:"transformers.TFBertForPreTraining",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFBertForPreTraining.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig">BertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_tf_bert.py#L1149"}}),hr=new we({props:{$$slots:{default:[FN]},$$scope:{ctx:B}}}),Ll=new C({props:{name:"call",anchor:"transformers.TFBertForPreTraining.call",parameters:[{name:"input_ids",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"token_type_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"position_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"head_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"inputs_embeds",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"labels",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"next_sentence_label",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"training",val:": typing.Optional[bool] = False"}],parametersDescription:[{anchor:"transformers.TFBertForPreTraining.call.input_ids",description:`<strong>input_ids</strong> (<code>np.ndarray</code>, <code>tf.Tensor</code>, <code>List[tf.Tensor]</code> \`<code>Dict[str, tf.Tensor]</code> or <code>Dict[str, np.ndarray]</code> and each example must have the shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18729/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> and
<a href="/docs/transformers/pr_18729/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.TFBertForPreTraining.call.attention_mask",description:`<strong>attention_mask</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.TFBertForPreTraining.call.token_type_ids",description:`<strong>token_type_ids</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.TFBertForPreTraining.call.position_ids",description:`<strong>position_ids</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.TFBertForPreTraining.call.head_mask",description:`<strong>head_mask</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.TFBertForPreTraining.call.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.TFBertForPreTraining.call.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
config will be used instead.`,name:"output_attentions"},{anchor:"transformers.TFBertForPreTraining.call.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
used instead.`,name:"output_hidden_states"},{anchor:"transformers.TFBertForPreTraining.call.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18729/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple. This argument can be used in
eager mode, in graph mode the value will always be set to True.`,name:"return_dict"},{anchor:"transformers.TFBertForPreTraining.call.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to \`False&#x201C;) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"},{anchor:"transformers.TFBertForPreTraining.call.labels",description:`<strong>labels</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.TFBertForPreTraining.call.next_sentence_label",description:`<strong>next_sentence_label</strong> (<code>tf.Tensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
(see <code>input_ids</code> docstring) Indices should be in <code>[0, 1]</code>:</p>
<ul>
<li>0 indicates sequence B is a continuation of sequence A,</li>
<li>1 indicates sequence B is a random sequence.</li>
</ul>`,name:"next_sentence_label"},{anchor:"transformers.TFBertForPreTraining.call.kwargs",description:`<strong>kwargs</strong> (<code>Dict[str, any]</code>, optional, defaults to <em>{}</em>) &#x2014;
Used to hide legacy arguments that have been deprecated.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_tf_bert.py#L1171",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.models.bert.modeling_tf_bert.TFBertForPreTrainingOutput"
>transformers.models.bert.modeling_tf_bert.TFBertForPreTrainingOutput</a> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>prediction_logits</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) \u2014 Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>seq_relationship_logits</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, 2)</code>) \u2014 Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.models.bert.modeling_tf_bert.TFBertForPreTrainingOutput"
>transformers.models.bert.modeling_tf_bert.TFBertForPreTrainingOutput</a> or <code>tuple(tf.Tensor)</code></p>
`}}),mr=new we({props:{$$slots:{default:[BN]},$$scope:{ctx:B}}}),fr=new ee({props:{anchor:"transformers.TFBertForPreTraining.call.example",$$slots:{default:[EN]},$$scope:{ctx:B}}}),Dl=new $e({}),Sl=new C({props:{name:"class transformers.TFBertLMHeadModel",anchor:"transformers.TFBertLMHeadModel",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_tf_bert.py#L1358"}}),Ul=new C({props:{name:"call",anchor:"transformers.TFBertLMHeadModel.call",parameters:[{name:"input_ids",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"token_type_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"position_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"head_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"inputs_embeds",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"encoder_hidden_states",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"encoder_attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"past_key_values",val:": typing.Union[typing.Tuple[typing.Tuple[typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor]]], NoneType] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"labels",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"training",val:": typing.Optional[bool] = False"},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_tf_bert.py#L1395",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_tf_outputs.TFCausalLMOutputWithCrossAttentions"
>transformers.modeling_tf_outputs.TFCausalLMOutputWithCrossAttentions</a> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>tf.Tensor</code> of shape <code>(n,)</code>, <em>optional</em>, where n is the number of non-masked labels, returned when <code>labels</code> is provided) \u2014 Language modeling loss (for next-token prediction).</p>
</li>
<li>
<p><strong>logits</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) \u2014 Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder\u2019s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>List[tf.Tensor]</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) \u2014 List of <code>tf.Tensor</code> of length <code>config.n_layers</code>, with each tensor of shape <code>(2, batch_size, num_heads, sequence_length, embed_size_per_head)</code>).</p>
<p>Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
<code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_tf_outputs.TFCausalLMOutputWithCrossAttentions"
>transformers.modeling_tf_outputs.TFCausalLMOutputWithCrossAttentions</a> or <code>tuple(tf.Tensor)</code></p>
`}}),gr=new ee({props:{anchor:"transformers.TFBertLMHeadModel.call.example",$$slots:{default:[MN]},$$scope:{ctx:B}}}),Vl=new $e({}),Ql=new C({props:{name:"class transformers.TFBertForMaskedLM",anchor:"transformers.TFBertForMaskedLM",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFBertForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig">BertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_tf_bert.py#L1266"}}),br=new we({props:{$$slots:{default:[zN]},$$scope:{ctx:B}}}),Yl=new C({props:{name:"call",anchor:"transformers.TFBertForMaskedLM.call",parameters:[{name:"input_ids",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"token_type_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"position_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"head_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"inputs_embeds",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"labels",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"training",val:": typing.Optional[bool] = False"}],parametersDescription:[{anchor:"transformers.TFBertForMaskedLM.call.input_ids",description:`<strong>input_ids</strong> (<code>np.ndarray</code>, <code>tf.Tensor</code>, <code>List[tf.Tensor]</code> \`<code>Dict[str, tf.Tensor]</code> or <code>Dict[str, np.ndarray]</code> and each example must have the shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18729/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> and
<a href="/docs/transformers/pr_18729/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.TFBertForMaskedLM.call.attention_mask",description:`<strong>attention_mask</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.TFBertForMaskedLM.call.token_type_ids",description:`<strong>token_type_ids</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.TFBertForMaskedLM.call.position_ids",description:`<strong>position_ids</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.TFBertForMaskedLM.call.head_mask",description:`<strong>head_mask</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.TFBertForMaskedLM.call.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.TFBertForMaskedLM.call.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
config will be used instead.`,name:"output_attentions"},{anchor:"transformers.TFBertForMaskedLM.call.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
used instead.`,name:"output_hidden_states"},{anchor:"transformers.TFBertForMaskedLM.call.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18729/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple. This argument can be used in
eager mode, in graph mode the value will always be set to True.`,name:"return_dict"},{anchor:"transformers.TFBertForMaskedLM.call.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to \`False&#x201C;) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"},{anchor:"transformers.TFBertForMaskedLM.call.labels",description:`<strong>labels</strong> (<code>tf.Tensor</code> or <code>np.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_tf_bert.py#L1294",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_tf_outputs.TFMaskedLMOutput"
>transformers.modeling_tf_outputs.TFMaskedLMOutput</a> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>tf.Tensor</code> of shape <code>(n,)</code>, <em>optional</em>, where n is the number of non-masked labels, returned when <code>labels</code> is provided) \u2014 Masked language modeling (MLM) loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) \u2014 Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_tf_outputs.TFMaskedLMOutput"
>transformers.modeling_tf_outputs.TFMaskedLMOutput</a> or <code>tuple(tf.Tensor)</code></p>
`}}),kr=new we({props:{$$slots:{default:[PN]},$$scope:{ctx:B}}}),Tr=new ee({props:{anchor:"transformers.TFBertForMaskedLM.call.example",$$slots:{default:[qN]},$$scope:{ctx:B}}}),vr=new ee({props:{anchor:"transformers.TFBertForMaskedLM.call.example-2",$$slots:{default:[jN]},$$scope:{ctx:B}}}),Zl=new $e({}),ed=new C({props:{name:"class transformers.TFBertForNextSentencePrediction",anchor:"transformers.TFBertForNextSentencePrediction",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFBertForNextSentencePrediction.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig">BertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_tf_bert.py#L1508"}}),wr=new we({props:{$$slots:{default:[CN]},$$scope:{ctx:B}}}),rd=new C({props:{name:"call",anchor:"transformers.TFBertForNextSentencePrediction.call",parameters:[{name:"input_ids",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"token_type_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"position_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"head_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"inputs_embeds",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"next_sentence_label",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"training",val:": typing.Optional[bool] = False"}],parametersDescription:[{anchor:"transformers.TFBertForNextSentencePrediction.call.input_ids",description:`<strong>input_ids</strong> (<code>np.ndarray</code>, <code>tf.Tensor</code>, <code>List[tf.Tensor]</code> \`<code>Dict[str, tf.Tensor]</code> or <code>Dict[str, np.ndarray]</code> and each example must have the shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18729/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> and
<a href="/docs/transformers/pr_18729/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.TFBertForNextSentencePrediction.call.attention_mask",description:`<strong>attention_mask</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.TFBertForNextSentencePrediction.call.token_type_ids",description:`<strong>token_type_ids</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.TFBertForNextSentencePrediction.call.position_ids",description:`<strong>position_ids</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.TFBertForNextSentencePrediction.call.head_mask",description:`<strong>head_mask</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.TFBertForNextSentencePrediction.call.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.TFBertForNextSentencePrediction.call.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
config will be used instead.`,name:"output_attentions"},{anchor:"transformers.TFBertForNextSentencePrediction.call.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
used instead.`,name:"output_hidden_states"},{anchor:"transformers.TFBertForNextSentencePrediction.call.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18729/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple. This argument can be used in
eager mode, in graph mode the value will always be set to True.`,name:"return_dict"},{anchor:"transformers.TFBertForNextSentencePrediction.call.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to \`False&#x201C;) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_tf_bert.py#L1518",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_tf_outputs.TFNextSentencePredictorOutput"
>transformers.modeling_tf_outputs.TFNextSentencePredictorOutput</a> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>tf.Tensor</code> of shape <code>(n,)</code>, <em>optional</em>, where n is the number of non-masked labels, returned when <code>next_sentence_label</code> is provided) \u2014 Next sentence prediction loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, 2)</code>) \u2014 Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_tf_outputs.TFNextSentencePredictorOutput"
>transformers.modeling_tf_outputs.TFNextSentencePredictorOutput</a> or <code>tuple(tf.Tensor)</code></p>
`}}),$r=new we({props:{$$slots:{default:[NN]},$$scope:{ctx:B}}}),xr=new ee({props:{anchor:"transformers.TFBertForNextSentencePrediction.call.example",$$slots:{default:[ON]},$$scope:{ctx:B}}}),ad=new $e({}),id=new C({props:{name:"class transformers.TFBertForSequenceClassification",anchor:"transformers.TFBertForSequenceClassification",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFBertForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig">BertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_tf_bert.py#L1599"}}),Br=new we({props:{$$slots:{default:[IN]},$$scope:{ctx:B}}}),pd=new C({props:{name:"call",anchor:"transformers.TFBertForSequenceClassification.call",parameters:[{name:"input_ids",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"token_type_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"position_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"head_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"inputs_embeds",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"labels",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"training",val:": typing.Optional[bool] = False"}],parametersDescription:[{anchor:"transformers.TFBertForSequenceClassification.call.input_ids",description:`<strong>input_ids</strong> (<code>np.ndarray</code>, <code>tf.Tensor</code>, <code>List[tf.Tensor]</code> \`<code>Dict[str, tf.Tensor]</code> or <code>Dict[str, np.ndarray]</code> and each example must have the shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18729/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> and
<a href="/docs/transformers/pr_18729/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.TFBertForSequenceClassification.call.attention_mask",description:`<strong>attention_mask</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.TFBertForSequenceClassification.call.token_type_ids",description:`<strong>token_type_ids</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.TFBertForSequenceClassification.call.position_ids",description:`<strong>position_ids</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.TFBertForSequenceClassification.call.head_mask",description:`<strong>head_mask</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.TFBertForSequenceClassification.call.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.TFBertForSequenceClassification.call.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
config will be used instead.`,name:"output_attentions"},{anchor:"transformers.TFBertForSequenceClassification.call.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
used instead.`,name:"output_hidden_states"},{anchor:"transformers.TFBertForSequenceClassification.call.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18729/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple. This argument can be used in
eager mode, in graph mode the value will always be set to True.`,name:"return_dict"},{anchor:"transformers.TFBertForSequenceClassification.call.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to \`False&#x201C;) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"},{anchor:"transformers.TFBertForSequenceClassification.call.labels",description:`<strong>labels</strong> (<code>tf.Tensor</code> or <code>np.ndarray</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_tf_bert.py#L1620",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_tf_outputs.TFSequenceClassifierOutput"
>transformers.modeling_tf_outputs.TFSequenceClassifierOutput</a> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, )</code>, <em>optional</em>, returned when <code>labels</code> is provided) \u2014 Classification (or regression if config.num_labels==1) loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, config.num_labels)</code>) \u2014 Classification (or regression if config.num_labels==1) scores (before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_tf_outputs.TFSequenceClassifierOutput"
>transformers.modeling_tf_outputs.TFSequenceClassifierOutput</a> or <code>tuple(tf.Tensor)</code></p>
`}}),Er=new we({props:{$$slots:{default:[AN]},$$scope:{ctx:B}}}),Mr=new ee({props:{anchor:"transformers.TFBertForSequenceClassification.call.example",$$slots:{default:[LN]},$$scope:{ctx:B}}}),zr=new ee({props:{anchor:"transformers.TFBertForSequenceClassification.call.example-2",$$slots:{default:[DN]},$$scope:{ctx:B}}}),hd=new $e({}),md=new C({props:{name:"class transformers.TFBertForMultipleChoice",anchor:"transformers.TFBertForMultipleChoice",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFBertForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig">BertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_tf_bert.py#L1692"}}),qr=new we({props:{$$slots:{default:[SN]},$$scope:{ctx:B}}}),_d=new C({props:{name:"call",anchor:"transformers.TFBertForMultipleChoice.call",parameters:[{name:"input_ids",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"token_type_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"position_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"head_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"inputs_embeds",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"labels",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"training",val:": typing.Optional[bool] = False"}],parametersDescription:[{anchor:"transformers.TFBertForMultipleChoice.call.input_ids",description:`<strong>input_ids</strong> (<code>np.ndarray</code>, <code>tf.Tensor</code>, <code>List[tf.Tensor]</code> \`<code>Dict[str, tf.Tensor]</code> or <code>Dict[str, np.ndarray]</code> and each example must have the shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18729/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> and
<a href="/docs/transformers/pr_18729/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.TFBertForMultipleChoice.call.attention_mask",description:`<strong>attention_mask</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.TFBertForMultipleChoice.call.token_type_ids",description:`<strong>token_type_ids</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.TFBertForMultipleChoice.call.position_ids",description:`<strong>position_ids</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.TFBertForMultipleChoice.call.head_mask",description:`<strong>head_mask</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.TFBertForMultipleChoice.call.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, num_choices, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.TFBertForMultipleChoice.call.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
config will be used instead.`,name:"output_attentions"},{anchor:"transformers.TFBertForMultipleChoice.call.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
used instead.`,name:"output_hidden_states"},{anchor:"transformers.TFBertForMultipleChoice.call.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18729/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple. This argument can be used in
eager mode, in graph mode the value will always be set to True.`,name:"return_dict"},{anchor:"transformers.TFBertForMultipleChoice.call.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to \`False&#x201C;) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"},{anchor:"transformers.TFBertForMultipleChoice.call.labels",description:`<strong>labels</strong> (<code>tf.Tensor</code> or <code>np.ndarray</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices]</code>
where <code>num_choices</code> is the size of the second dimension of the input tensors. (See <code>input_ids</code> above)`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_tf_bert.py#L1716",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_tf_outputs.TFMultipleChoiceModelOutput"
>transformers.modeling_tf_outputs.TFMultipleChoiceModelOutput</a> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>tf.Tensor</code> of shape <em>(batch_size, )</em>, <em>optional</em>, returned when <code>labels</code> is provided) \u2014 Classification loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, num_choices)</code>) \u2014 <em>num_choices</em> is the second dimension of the input tensors. (see <em>input_ids</em> above).</p>
<p>Classification scores (before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_tf_outputs.TFMultipleChoiceModelOutput"
>transformers.modeling_tf_outputs.TFMultipleChoiceModelOutput</a> or <code>tuple(tf.Tensor)</code></p>
`}}),jr=new we({props:{$$slots:{default:[UN]},$$scope:{ctx:B}}}),Cr=new ee({props:{anchor:"transformers.TFBertForMultipleChoice.call.example",$$slots:{default:[WN]},$$scope:{ctx:B}}}),bd=new $e({}),kd=new C({props:{name:"class transformers.TFBertForTokenClassification",anchor:"transformers.TFBertForTokenClassification",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFBertForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig">BertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_tf_bert.py#L1822"}}),Or=new we({props:{$$slots:{default:[HN]},$$scope:{ctx:B}}}),wd=new C({props:{name:"call",anchor:"transformers.TFBertForTokenClassification.call",parameters:[{name:"input_ids",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"token_type_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"position_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"head_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"inputs_embeds",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"labels",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"training",val:": typing.Optional[bool] = False"}],parametersDescription:[{anchor:"transformers.TFBertForTokenClassification.call.input_ids",description:`<strong>input_ids</strong> (<code>np.ndarray</code>, <code>tf.Tensor</code>, <code>List[tf.Tensor]</code> \`<code>Dict[str, tf.Tensor]</code> or <code>Dict[str, np.ndarray]</code> and each example must have the shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18729/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> and
<a href="/docs/transformers/pr_18729/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.TFBertForTokenClassification.call.attention_mask",description:`<strong>attention_mask</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.TFBertForTokenClassification.call.token_type_ids",description:`<strong>token_type_ids</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.TFBertForTokenClassification.call.position_ids",description:`<strong>position_ids</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.TFBertForTokenClassification.call.head_mask",description:`<strong>head_mask</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.TFBertForTokenClassification.call.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.TFBertForTokenClassification.call.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
config will be used instead.`,name:"output_attentions"},{anchor:"transformers.TFBertForTokenClassification.call.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
used instead.`,name:"output_hidden_states"},{anchor:"transformers.TFBertForTokenClassification.call.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18729/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple. This argument can be used in
eager mode, in graph mode the value will always be set to True.`,name:"return_dict"},{anchor:"transformers.TFBertForTokenClassification.call.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to \`False&#x201C;) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"},{anchor:"transformers.TFBertForTokenClassification.call.labels",description:`<strong>labels</strong> (<code>tf.Tensor</code> or <code>np.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_tf_bert.py#L1849",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_tf_outputs.TFTokenClassifierOutput"
>transformers.modeling_tf_outputs.TFTokenClassifierOutput</a> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>tf.Tensor</code> of shape <code>(n,)</code>, <em>optional</em>, where n is the number of unmasked labels, returned when <code>labels</code> is provided)  \u2014 Classification loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, config.num_labels)</code>) \u2014 Classification scores (before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_tf_outputs.TFTokenClassifierOutput"
>transformers.modeling_tf_outputs.TFTokenClassifierOutput</a> or <code>tuple(tf.Tensor)</code></p>
`}}),Ir=new we({props:{$$slots:{default:[RN]},$$scope:{ctx:B}}}),Ar=new ee({props:{anchor:"transformers.TFBertForTokenClassification.call.example",$$slots:{default:[VN]},$$scope:{ctx:B}}}),Lr=new ee({props:{anchor:"transformers.TFBertForTokenClassification.call.example-2",$$slots:{default:[QN]},$$scope:{ctx:B}}}),$d=new $e({}),xd=new C({props:{name:"class transformers.TFBertForQuestionAnswering",anchor:"transformers.TFBertForQuestionAnswering",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFBertForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig">BertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_tf_bert.py#L1919"}}),Sr=new we({props:{$$slots:{default:[KN]},$$scope:{ctx:B}}}),Md=new C({props:{name:"call",anchor:"transformers.TFBertForQuestionAnswering.call",parameters:[{name:"input_ids",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"token_type_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"position_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"head_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"inputs_embeds",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"start_positions",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"end_positions",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"training",val:": typing.Optional[bool] = False"}],parametersDescription:[{anchor:"transformers.TFBertForQuestionAnswering.call.input_ids",description:`<strong>input_ids</strong> (<code>np.ndarray</code>, <code>tf.Tensor</code>, <code>List[tf.Tensor]</code> \`<code>Dict[str, tf.Tensor]</code> or <code>Dict[str, np.ndarray]</code> and each example must have the shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18729/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> and
<a href="/docs/transformers/pr_18729/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.TFBertForQuestionAnswering.call.attention_mask",description:`<strong>attention_mask</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.TFBertForQuestionAnswering.call.token_type_ids",description:`<strong>token_type_ids</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.TFBertForQuestionAnswering.call.position_ids",description:`<strong>position_ids</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.TFBertForQuestionAnswering.call.head_mask",description:`<strong>head_mask</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.TFBertForQuestionAnswering.call.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.TFBertForQuestionAnswering.call.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
config will be used instead.`,name:"output_attentions"},{anchor:"transformers.TFBertForQuestionAnswering.call.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
used instead.`,name:"output_hidden_states"},{anchor:"transformers.TFBertForQuestionAnswering.call.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18729/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple. This argument can be used in
eager mode, in graph mode the value will always be set to True.`,name:"return_dict"},{anchor:"transformers.TFBertForQuestionAnswering.call.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to \`False&#x201C;) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"},{anchor:"transformers.TFBertForQuestionAnswering.call.start_positions",description:`<strong>start_positions</strong> (<code>tf.Tensor</code> or <code>np.ndarray</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.TFBertForQuestionAnswering.call.end_positions",description:`<strong>end_positions</strong> (<code>tf.Tensor</code> or <code>np.ndarray</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_tf_bert.py#L1941",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_tf_outputs.TFQuestionAnsweringModelOutput"
>transformers.modeling_tf_outputs.TFQuestionAnsweringModelOutput</a> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, )</code>, <em>optional</em>, returned when <code>start_positions</code> and <code>end_positions</code> are provided) \u2014 Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.</p>
</li>
<li>
<p><strong>start_logits</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>) \u2014 Span-start scores (before SoftMax).</p>
</li>
<li>
<p><strong>end_logits</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>) \u2014 Span-end scores (before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_tf_outputs.TFQuestionAnsweringModelOutput"
>transformers.modeling_tf_outputs.TFQuestionAnsweringModelOutput</a> or <code>tuple(tf.Tensor)</code></p>
`}}),Ur=new we({props:{$$slots:{default:[JN]},$$scope:{ctx:B}}}),Wr=new ee({props:{anchor:"transformers.TFBertForQuestionAnswering.call.example",$$slots:{default:[GN]},$$scope:{ctx:B}}}),Hr=new ee({props:{anchor:"transformers.TFBertForQuestionAnswering.call.example-2",$$slots:{default:[XN]},$$scope:{ctx:B}}}),zd=new $e({}),Pd=new C({props:{name:"class transformers.FlaxBertModel",anchor:"transformers.FlaxBertModel",parameters:[{name:"config",val:": BertConfig"},{name:"input_shape",val:": typing.Tuple = (1, 1)"},{name:"seed",val:": int = 0"},{name:"dtype",val:": dtype = <class 'jax.numpy.float32'>"},{name:"_do_init",val:": bool = True"},{name:"gradient_checkpointing",val:": bool = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.FlaxBertModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig">BertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.FlaxBertModel.dtype",description:`<strong>dtype</strong> (<code>jax.numpy.dtype</code>, <em>optional</em>, defaults to <code>jax.numpy.float32</code>) &#x2014;
The data type of the computation. Can be one of <code>jax.numpy.float32</code>, <code>jax.numpy.float16</code> (on GPUs) and
<code>jax.numpy.bfloat16</code> (on TPUs).</p>
<p>This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
specified all the computation will be performed with the given <code>dtype</code>.</p>
<p><strong>Note that this only specifies the dtype of the computation and does not influence the dtype of model
parameters.</strong></p>
<p>If you wish to change the dtype of the model parameters, see <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16">to_fp16()</a> and
<a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16">to_bf16()</a>.`,name:"dtype"},{anchor:"transformers.FlaxBertModel.dtype",description:`<strong>dtype</strong> (<code>jax.numpy.dtype</code>, <em>optional</em>, defaults to <code>jax.numpy.float32</code>) &#x2014;
The data type of the computation. Can be one of <code>jax.numpy.float32</code>, <code>jax.numpy.float16</code> (on GPUs) and
<code>jax.numpy.bfloat16</code> (on TPUs).</p>
<p>This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
specified all the computation will be performed with the given <code>dtype</code>.</p>
<p><strong>Note that this only specifies the dtype of the computation and does not influence the dtype of model
parameters.</strong></p>
<p>If you wish to change the dtype of the model parameters, see <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16">to_fp16()</a> and
<a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16">to_bf16()</a>.`,name:"dtype"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_flax_bert.py#L1028"}}),Ld=new C({props:{name:"__call__",anchor:"transformers.FlaxBertModel.__call__",parameters:[{name:"input_ids",val:""},{name:"attention_mask",val:" = None"},{name:"token_type_ids",val:" = None"},{name:"position_ids",val:" = None"},{name:"head_mask",val:" = None"},{name:"encoder_hidden_states",val:" = None"},{name:"encoder_attention_mask",val:" = None"},{name:"params",val:": dict = None"},{name:"dropout_rng",val:": PRNGKey = None"},{name:"train",val:": bool = False"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"past_key_values",val:": dict = None"}],parametersDescription:[{anchor:"transformers.FlaxBertModel.__call__.input_ids",description:`<strong>input_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18729/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18729/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FlaxBertModel.__call__.attention_mask",description:`<strong>attention_mask</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.FlaxBertModel.__call__.token_type_ids",description:`<strong>token_type_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.FlaxBertModel.__call__.position_ids",description:`<strong>position_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.`,name:"position_ids"},{anchor:"transformers.FlaxBertModel.__call__.head_mask",description:`<strong>head_mask</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <code>optional) -- Mask to nullify selected heads of the attention modules. Mask values selected in </code>[0, 1]\`:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.FlaxBertModel.__call__.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18729/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_flax_bert.py#L855",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutputWithPooling"
>transformers.modeling_flax_outputs.FlaxBaseModelOutputWithPooling</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) \u2014 Sequence of hidden-states at the output of the last layer of the model.</p>
</li>
<li>
<p><strong>pooler_output</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, hidden_size)</code>) \u2014 Last layer hidden-state of the first token of the sequence (classification token) further processed by a
Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence
prediction (classification) objective during pretraining.</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutputWithPooling"
>transformers.modeling_flax_outputs.FlaxBaseModelOutputWithPooling</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Vr=new we({props:{$$slots:{default:[YN]},$$scope:{ctx:B}}}),Qr=new ee({props:{anchor:"transformers.FlaxBertModel.__call__.example",$$slots:{default:[ZN]},$$scope:{ctx:B}}}),Dd=new $e({}),Sd=new C({props:{name:"class transformers.FlaxBertForPreTraining",anchor:"transformers.FlaxBertForPreTraining",parameters:[{name:"config",val:": BertConfig"},{name:"input_shape",val:": typing.Tuple = (1, 1)"},{name:"seed",val:": int = 0"},{name:"dtype",val:": dtype = <class 'jax.numpy.float32'>"},{name:"_do_init",val:": bool = True"},{name:"gradient_checkpointing",val:": bool = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.FlaxBertForPreTraining.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig">BertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.FlaxBertForPreTraining.dtype",description:`<strong>dtype</strong> (<code>jax.numpy.dtype</code>, <em>optional</em>, defaults to <code>jax.numpy.float32</code>) &#x2014;
The data type of the computation. Can be one of <code>jax.numpy.float32</code>, <code>jax.numpy.float16</code> (on GPUs) and
<code>jax.numpy.bfloat16</code> (on TPUs).</p>
<p>This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
specified all the computation will be performed with the given <code>dtype</code>.</p>
<p><strong>Note that this only specifies the dtype of the computation and does not influence the dtype of model
parameters.</strong></p>
<p>If you wish to change the dtype of the model parameters, see <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16">to_fp16()</a> and
<a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16">to_bf16()</a>.`,name:"dtype"},{anchor:"transformers.FlaxBertForPreTraining.dtype",description:`<strong>dtype</strong> (<code>jax.numpy.dtype</code>, <em>optional</em>, defaults to <code>jax.numpy.float32</code>) &#x2014;
The data type of the computation. Can be one of <code>jax.numpy.float32</code>, <code>jax.numpy.float16</code> (on GPUs) and
<code>jax.numpy.bfloat16</code> (on TPUs).</p>
<p>This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
specified all the computation will be performed with the given <code>dtype</code>.</p>
<p><strong>Note that this only specifies the dtype of the computation and does not influence the dtype of model
parameters.</strong></p>
<p>If you wish to change the dtype of the model parameters, see <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16">to_fp16()</a> and
<a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16">to_bf16()</a>.`,name:"dtype"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_flax_bert.py#L1106"}}),Jd=new C({props:{name:"__call__",anchor:"transformers.FlaxBertForPreTraining.__call__",parameters:[{name:"input_ids",val:""},{name:"attention_mask",val:" = None"},{name:"token_type_ids",val:" = None"},{name:"position_ids",val:" = None"},{name:"head_mask",val:" = None"},{name:"encoder_hidden_states",val:" = None"},{name:"encoder_attention_mask",val:" = None"},{name:"params",val:": dict = None"},{name:"dropout_rng",val:": PRNGKey = None"},{name:"train",val:": bool = False"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"past_key_values",val:": dict = None"}],parametersDescription:[{anchor:"transformers.FlaxBertForPreTraining.__call__.input_ids",description:`<strong>input_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18729/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18729/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FlaxBertForPreTraining.__call__.attention_mask",description:`<strong>attention_mask</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.FlaxBertForPreTraining.__call__.token_type_ids",description:`<strong>token_type_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.FlaxBertForPreTraining.__call__.position_ids",description:`<strong>position_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.`,name:"position_ids"},{anchor:"transformers.FlaxBertForPreTraining.__call__.head_mask",description:`<strong>head_mask</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <code>optional) -- Mask to nullify selected heads of the attention modules. Mask values selected in </code>[0, 1]\`:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.FlaxBertForPreTraining.__call__.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18729/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_flax_bert.py#L855",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.models.bert.modeling_flax_bert.FlaxBertForPreTrainingOutput"
>transformers.models.bert.modeling_flax_bert.FlaxBertForPreTrainingOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>prediction_logits</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) \u2014 Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>seq_relationship_logits</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, 2)</code>) \u2014 Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.models.bert.modeling_flax_bert.FlaxBertForPreTrainingOutput"
>transformers.models.bert.modeling_flax_bert.FlaxBertForPreTrainingOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Jr=new we({props:{$$slots:{default:[eO]},$$scope:{ctx:B}}}),Gr=new ee({props:{anchor:"transformers.FlaxBertForPreTraining.__call__.example",$$slots:{default:[tO]},$$scope:{ctx:B}}}),Gd=new $e({}),Xd=new C({props:{name:"class transformers.FlaxBertForCausalLM",anchor:"transformers.FlaxBertForCausalLM",parameters:[{name:"config",val:": BertConfig"},{name:"input_shape",val:": typing.Tuple = (1, 1)"},{name:"seed",val:": int = 0"},{name:"dtype",val:": dtype = <class 'jax.numpy.float32'>"},{name:"_do_init",val:": bool = True"},{name:"gradient_checkpointing",val:": bool = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.FlaxBertForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig">BertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.FlaxBertForCausalLM.dtype",description:`<strong>dtype</strong> (<code>jax.numpy.dtype</code>, <em>optional</em>, defaults to <code>jax.numpy.float32</code>) &#x2014;
The data type of the computation. Can be one of <code>jax.numpy.float32</code>, <code>jax.numpy.float16</code> (on GPUs) and
<code>jax.numpy.bfloat16</code> (on TPUs).</p>
<p>This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
specified all the computation will be performed with the given <code>dtype</code>.</p>
<p><strong>Note that this only specifies the dtype of the computation and does not influence the dtype of model
parameters.</strong></p>
<p>If you wish to change the dtype of the model parameters, see <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16">to_fp16()</a> and
<a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16">to_bf16()</a>.`,name:"dtype"},{anchor:"transformers.FlaxBertForCausalLM.dtype",description:`<strong>dtype</strong> (<code>jax.numpy.dtype</code>, <em>optional</em>, defaults to <code>jax.numpy.float32</code>) &#x2014;
The data type of the computation. Can be one of <code>jax.numpy.float32</code>, <code>jax.numpy.float16</code> (on GPUs) and
<code>jax.numpy.bfloat16</code> (on TPUs).</p>
<p>This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
specified all the computation will be performed with the given <code>dtype</code>.</p>
<p><strong>Note that this only specifies the dtype of the computation and does not influence the dtype of model
parameters.</strong></p>
<p>If you wish to change the dtype of the model parameters, see <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16">to_fp16()</a> and
<a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16">to_bf16()</a>.`,name:"dtype"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_flax_bert.py#L1683"}}),rc=new C({props:{name:"__call__",anchor:"transformers.FlaxBertForCausalLM.__call__",parameters:[{name:"input_ids",val:""},{name:"attention_mask",val:" = None"},{name:"token_type_ids",val:" = None"},{name:"position_ids",val:" = None"},{name:"head_mask",val:" = None"},{name:"encoder_hidden_states",val:" = None"},{name:"encoder_attention_mask",val:" = None"},{name:"params",val:": dict = None"},{name:"dropout_rng",val:": PRNGKey = None"},{name:"train",val:": bool = False"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"past_key_values",val:": dict = None"}],parametersDescription:[{anchor:"transformers.FlaxBertForCausalLM.__call__.input_ids",description:`<strong>input_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18729/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18729/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FlaxBertForCausalLM.__call__.attention_mask",description:`<strong>attention_mask</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.FlaxBertForCausalLM.__call__.token_type_ids",description:`<strong>token_type_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.FlaxBertForCausalLM.__call__.position_ids",description:`<strong>position_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.`,name:"position_ids"},{anchor:"transformers.FlaxBertForCausalLM.__call__.head_mask",description:`<strong>head_mask</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <code>optional) -- Mask to nullify selected heads of the attention modules. Mask values selected in </code>[0, 1]\`:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.FlaxBertForCausalLM.__call__.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18729/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_flax_bert.py#L855",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_flax_outputs.FlaxCausalLMOutputWithCrossAttentions"
>transformers.modeling_flax_outputs.FlaxCausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>logits</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) \u2014 Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Cross attentions weights after the attention softmax, used to compute the weighted average in the
cross-attention heads.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>tuple(tuple(jnp.ndarray))</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> tuples of length <code>config.n_layers</code>, with each tuple containing the cached key, value
states of the self-attention and the cross-attention layers if model is used in encoder-decoder setting.
Only relevant if <code>config.is_decoder = True</code>.</p>
<p>Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
<code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_flax_outputs.FlaxCausalLMOutputWithCrossAttentions"
>transformers.modeling_flax_outputs.FlaxCausalLMOutputWithCrossAttentions</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Yr=new we({props:{$$slots:{default:[oO]},$$scope:{ctx:B}}}),Zr=new ee({props:{anchor:"transformers.FlaxBertForCausalLM.__call__.example",$$slots:{default:[nO]},$$scope:{ctx:B}}}),ac=new $e({}),ic=new C({props:{name:"class transformers.FlaxBertForMaskedLM",anchor:"transformers.FlaxBertForMaskedLM",parameters:[{name:"config",val:": BertConfig"},{name:"input_shape",val:": typing.Tuple = (1, 1)"},{name:"seed",val:": int = 0"},{name:"dtype",val:": dtype = <class 'jax.numpy.float32'>"},{name:"_do_init",val:": bool = True"},{name:"gradient_checkpointing",val:": bool = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.FlaxBertForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig">BertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.FlaxBertForMaskedLM.dtype",description:`<strong>dtype</strong> (<code>jax.numpy.dtype</code>, <em>optional</em>, defaults to <code>jax.numpy.float32</code>) &#x2014;
The data type of the computation. Can be one of <code>jax.numpy.float32</code>, <code>jax.numpy.float16</code> (on GPUs) and
<code>jax.numpy.bfloat16</code> (on TPUs).</p>
<p>This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
specified all the computation will be performed with the given <code>dtype</code>.</p>
<p><strong>Note that this only specifies the dtype of the computation and does not influence the dtype of model
parameters.</strong></p>
<p>If you wish to change the dtype of the model parameters, see <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16">to_fp16()</a> and
<a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16">to_bf16()</a>.`,name:"dtype"},{anchor:"transformers.FlaxBertForMaskedLM.dtype",description:`<strong>dtype</strong> (<code>jax.numpy.dtype</code>, <em>optional</em>, defaults to <code>jax.numpy.float32</code>) &#x2014;
The data type of the computation. Can be one of <code>jax.numpy.float32</code>, <code>jax.numpy.float16</code> (on GPUs) and
<code>jax.numpy.bfloat16</code> (on TPUs).</p>
<p>This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
specified all the computation will be performed with the given <code>dtype</code>.</p>
<p><strong>Note that this only specifies the dtype of the computation and does not influence the dtype of model
parameters.</strong></p>
<p>If you wish to change the dtype of the model parameters, see <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16">to_fp16()</a> and
<a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16">to_bf16()</a>.`,name:"dtype"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_flax_bert.py#L1197"}}),gc=new C({props:{name:"__call__",anchor:"transformers.FlaxBertForMaskedLM.__call__",parameters:[{name:"input_ids",val:""},{name:"attention_mask",val:" = None"},{name:"token_type_ids",val:" = None"},{name:"position_ids",val:" = None"},{name:"head_mask",val:" = None"},{name:"encoder_hidden_states",val:" = None"},{name:"encoder_attention_mask",val:" = None"},{name:"params",val:": dict = None"},{name:"dropout_rng",val:": PRNGKey = None"},{name:"train",val:": bool = False"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"past_key_values",val:": dict = None"}],parametersDescription:[{anchor:"transformers.FlaxBertForMaskedLM.__call__.input_ids",description:`<strong>input_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18729/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18729/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FlaxBertForMaskedLM.__call__.attention_mask",description:`<strong>attention_mask</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.FlaxBertForMaskedLM.__call__.token_type_ids",description:`<strong>token_type_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.FlaxBertForMaskedLM.__call__.position_ids",description:`<strong>position_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.`,name:"position_ids"},{anchor:"transformers.FlaxBertForMaskedLM.__call__.head_mask",description:`<strong>head_mask</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <code>optional) -- Mask to nullify selected heads of the attention modules. Mask values selected in </code>[0, 1]\`:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.FlaxBertForMaskedLM.__call__.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18729/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_flax_bert.py#L855",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_flax_outputs.FlaxMaskedLMOutput"
>transformers.modeling_flax_outputs.FlaxMaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>logits</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) \u2014 Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_flax_outputs.FlaxMaskedLMOutput"
>transformers.modeling_flax_outputs.FlaxMaskedLMOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ta=new we({props:{$$slots:{default:[sO]},$$scope:{ctx:B}}}),oa=new ee({props:{anchor:"transformers.FlaxBertForMaskedLM.__call__.example",$$slots:{default:[rO]},$$scope:{ctx:B}}}),_c=new $e({}),bc=new C({props:{name:"class transformers.FlaxBertForNextSentencePrediction",anchor:"transformers.FlaxBertForNextSentencePrediction",parameters:[{name:"config",val:": BertConfig"},{name:"input_shape",val:": typing.Tuple = (1, 1)"},{name:"seed",val:": int = 0"},{name:"dtype",val:": dtype = <class 'jax.numpy.float32'>"},{name:"_do_init",val:": bool = True"},{name:"gradient_checkpointing",val:": bool = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.FlaxBertForNextSentencePrediction.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig">BertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.FlaxBertForNextSentencePrediction.dtype",description:`<strong>dtype</strong> (<code>jax.numpy.dtype</code>, <em>optional</em>, defaults to <code>jax.numpy.float32</code>) &#x2014;
The data type of the computation. Can be one of <code>jax.numpy.float32</code>, <code>jax.numpy.float16</code> (on GPUs) and
<code>jax.numpy.bfloat16</code> (on TPUs).</p>
<p>This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
specified all the computation will be performed with the given <code>dtype</code>.</p>
<p><strong>Note that this only specifies the dtype of the computation and does not influence the dtype of model
parameters.</strong></p>
<p>If you wish to change the dtype of the model parameters, see <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16">to_fp16()</a> and
<a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16">to_bf16()</a>.`,name:"dtype"},{anchor:"transformers.FlaxBertForNextSentencePrediction.dtype",description:`<strong>dtype</strong> (<code>jax.numpy.dtype</code>, <em>optional</em>, defaults to <code>jax.numpy.float32</code>) &#x2014;
The data type of the computation. Can be one of <code>jax.numpy.float32</code>, <code>jax.numpy.float16</code> (on GPUs) and
<code>jax.numpy.bfloat16</code> (on TPUs).</p>
<p>This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
specified all the computation will be performed with the given <code>dtype</code>.</p>
<p><strong>Note that this only specifies the dtype of the computation and does not influence the dtype of model
parameters.</strong></p>
<p>If you wish to change the dtype of the model parameters, see <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16">to_fp16()</a> and
<a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16">to_bf16()</a>.`,name:"dtype"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_flax_bert.py#L1263"}}),Bc=new C({props:{name:"__call__",anchor:"transformers.FlaxBertForNextSentencePrediction.__call__",parameters:[{name:"input_ids",val:""},{name:"attention_mask",val:" = None"},{name:"token_type_ids",val:" = None"},{name:"position_ids",val:" = None"},{name:"head_mask",val:" = None"},{name:"encoder_hidden_states",val:" = None"},{name:"encoder_attention_mask",val:" = None"},{name:"params",val:": dict = None"},{name:"dropout_rng",val:": PRNGKey = None"},{name:"train",val:": bool = False"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"past_key_values",val:": dict = None"}],parametersDescription:[{anchor:"transformers.FlaxBertForNextSentencePrediction.__call__.input_ids",description:`<strong>input_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18729/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18729/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FlaxBertForNextSentencePrediction.__call__.attention_mask",description:`<strong>attention_mask</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.FlaxBertForNextSentencePrediction.__call__.token_type_ids",description:`<strong>token_type_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.FlaxBertForNextSentencePrediction.__call__.position_ids",description:`<strong>position_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.`,name:"position_ids"},{anchor:"transformers.FlaxBertForNextSentencePrediction.__call__.head_mask",description:`<strong>head_mask</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <code>optional) -- Mask to nullify selected heads of the attention modules. Mask values selected in </code>[0, 1]\`:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.FlaxBertForNextSentencePrediction.__call__.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18729/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_flax_bert.py#L855",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_flax_outputs.FlaxNextSentencePredictorOutput"
>transformers.modeling_flax_outputs.FlaxNextSentencePredictorOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>logits</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, 2)</code>) \u2014 Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_flax_outputs.FlaxNextSentencePredictorOutput"
>transformers.modeling_flax_outputs.FlaxNextSentencePredictorOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),sa=new we({props:{$$slots:{default:[aO]},$$scope:{ctx:B}}}),ra=new ee({props:{anchor:"transformers.FlaxBertForNextSentencePrediction.__call__.example",$$slots:{default:[iO]},$$scope:{ctx:B}}}),Ec=new $e({}),Mc=new C({props:{name:"class transformers.FlaxBertForSequenceClassification",anchor:"transformers.FlaxBertForSequenceClassification",parameters:[{name:"config",val:": BertConfig"},{name:"input_shape",val:": typing.Tuple = (1, 1)"},{name:"seed",val:": int = 0"},{name:"dtype",val:": dtype = <class 'jax.numpy.float32'>"},{name:"_do_init",val:": bool = True"},{name:"gradient_checkpointing",val:": bool = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.FlaxBertForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig">BertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.FlaxBertForSequenceClassification.dtype",description:`<strong>dtype</strong> (<code>jax.numpy.dtype</code>, <em>optional</em>, defaults to <code>jax.numpy.float32</code>) &#x2014;
The data type of the computation. Can be one of <code>jax.numpy.float32</code>, <code>jax.numpy.float16</code> (on GPUs) and
<code>jax.numpy.bfloat16</code> (on TPUs).</p>
<p>This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
specified all the computation will be performed with the given <code>dtype</code>.</p>
<p><strong>Note that this only specifies the dtype of the computation and does not influence the dtype of model
parameters.</strong></p>
<p>If you wish to change the dtype of the model parameters, see <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16">to_fp16()</a> and
<a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16">to_bf16()</a>.`,name:"dtype"},{anchor:"transformers.FlaxBertForSequenceClassification.dtype",description:`<strong>dtype</strong> (<code>jax.numpy.dtype</code>, <em>optional</em>, defaults to <code>jax.numpy.float32</code>) &#x2014;
The data type of the computation. Can be one of <code>jax.numpy.float32</code>, <code>jax.numpy.float16</code> (on GPUs) and
<code>jax.numpy.bfloat16</code> (on TPUs).</p>
<p>This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
specified all the computation will be performed with the given <code>dtype</code>.</p>
<p><strong>Note that this only specifies the dtype of the computation and does not influence the dtype of model
parameters.</strong></p>
<p>If you wish to change the dtype of the model parameters, see <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16">to_fp16()</a> and
<a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16">to_bf16()</a>.`,name:"dtype"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_flax_bert.py#L1366"}}),Ic=new C({props:{name:"__call__",anchor:"transformers.FlaxBertForSequenceClassification.__call__",parameters:[{name:"input_ids",val:""},{name:"attention_mask",val:" = None"},{name:"token_type_ids",val:" = None"},{name:"position_ids",val:" = None"},{name:"head_mask",val:" = None"},{name:"encoder_hidden_states",val:" = None"},{name:"encoder_attention_mask",val:" = None"},{name:"params",val:": dict = None"},{name:"dropout_rng",val:": PRNGKey = None"},{name:"train",val:": bool = False"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"past_key_values",val:": dict = None"}],parametersDescription:[{anchor:"transformers.FlaxBertForSequenceClassification.__call__.input_ids",description:`<strong>input_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18729/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18729/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FlaxBertForSequenceClassification.__call__.attention_mask",description:`<strong>attention_mask</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.FlaxBertForSequenceClassification.__call__.token_type_ids",description:`<strong>token_type_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.FlaxBertForSequenceClassification.__call__.position_ids",description:`<strong>position_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.`,name:"position_ids"},{anchor:"transformers.FlaxBertForSequenceClassification.__call__.head_mask",description:`<strong>head_mask</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <code>optional) -- Mask to nullify selected heads of the attention modules. Mask values selected in </code>[0, 1]\`:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.FlaxBertForSequenceClassification.__call__.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18729/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_flax_bert.py#L855",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSequenceClassifierOutput"
>transformers.modeling_flax_outputs.FlaxSequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>logits</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, config.num_labels)</code>) \u2014 Classification (or regression if config.num_labels==1) scores (before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSequenceClassifierOutput"
>transformers.modeling_flax_outputs.FlaxSequenceClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ia=new we({props:{$$slots:{default:[lO]},$$scope:{ctx:B}}}),la=new ee({props:{anchor:"transformers.FlaxBertForSequenceClassification.__call__.example",$$slots:{default:[dO]},$$scope:{ctx:B}}}),Ac=new $e({}),Lc=new C({props:{name:"class transformers.FlaxBertForMultipleChoice",anchor:"transformers.FlaxBertForMultipleChoice",parameters:[{name:"config",val:": BertConfig"},{name:"input_shape",val:": typing.Tuple = (1, 1)"},{name:"seed",val:": int = 0"},{name:"dtype",val:": dtype = <class 'jax.numpy.float32'>"},{name:"_do_init",val:": bool = True"},{name:"gradient_checkpointing",val:": bool = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.FlaxBertForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig">BertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.FlaxBertForMultipleChoice.dtype",description:`<strong>dtype</strong> (<code>jax.numpy.dtype</code>, <em>optional</em>, defaults to <code>jax.numpy.float32</code>) &#x2014;
The data type of the computation. Can be one of <code>jax.numpy.float32</code>, <code>jax.numpy.float16</code> (on GPUs) and
<code>jax.numpy.bfloat16</code> (on TPUs).</p>
<p>This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
specified all the computation will be performed with the given <code>dtype</code>.</p>
<p><strong>Note that this only specifies the dtype of the computation and does not influence the dtype of model
parameters.</strong></p>
<p>If you wish to change the dtype of the model parameters, see <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16">to_fp16()</a> and
<a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16">to_bf16()</a>.`,name:"dtype"},{anchor:"transformers.FlaxBertForMultipleChoice.dtype",description:`<strong>dtype</strong> (<code>jax.numpy.dtype</code>, <em>optional</em>, defaults to <code>jax.numpy.float32</code>) &#x2014;
The data type of the computation. Can be one of <code>jax.numpy.float32</code>, <code>jax.numpy.float16</code> (on GPUs) and
<code>jax.numpy.bfloat16</code> (on TPUs).</p>
<p>This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
specified all the computation will be performed with the given <code>dtype</code>.</p>
<p><strong>Note that this only specifies the dtype of the computation and does not influence the dtype of model
parameters.</strong></p>
<p>If you wish to change the dtype of the model parameters, see <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16">to_fp16()</a> and
<a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16">to_bf16()</a>.`,name:"dtype"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_flax_bert.py#L1447"}}),Qc=new C({props:{name:"__call__",anchor:"transformers.FlaxBertForMultipleChoice.__call__",parameters:[{name:"input_ids",val:""},{name:"attention_mask",val:" = None"},{name:"token_type_ids",val:" = None"},{name:"position_ids",val:" = None"},{name:"head_mask",val:" = None"},{name:"encoder_hidden_states",val:" = None"},{name:"encoder_attention_mask",val:" = None"},{name:"params",val:": dict = None"},{name:"dropout_rng",val:": PRNGKey = None"},{name:"train",val:": bool = False"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"past_key_values",val:": dict = None"}],parametersDescription:[{anchor:"transformers.FlaxBertForMultipleChoice.__call__.input_ids",description:`<strong>input_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18729/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18729/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FlaxBertForMultipleChoice.__call__.attention_mask",description:`<strong>attention_mask</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.FlaxBertForMultipleChoice.__call__.token_type_ids",description:`<strong>token_type_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.FlaxBertForMultipleChoice.__call__.position_ids",description:`<strong>position_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.`,name:"position_ids"},{anchor:"transformers.FlaxBertForMultipleChoice.__call__.head_mask",description:`<strong>head_mask</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <code>optional) -- Mask to nullify selected heads of the attention modules. Mask values selected in </code>[0, 1]\`:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.FlaxBertForMultipleChoice.__call__.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18729/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_flax_bert.py#L855",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_flax_outputs.FlaxMultipleChoiceModelOutput"
>transformers.modeling_flax_outputs.FlaxMultipleChoiceModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>logits</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, num_choices)</code>) \u2014 <em>num_choices</em> is the second dimension of the input tensors. (see <em>input_ids</em> above).</p>
<p>Classification scores (before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_flax_outputs.FlaxMultipleChoiceModelOutput"
>transformers.modeling_flax_outputs.FlaxMultipleChoiceModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ca=new we({props:{$$slots:{default:[cO]},$$scope:{ctx:B}}}),pa=new ee({props:{anchor:"transformers.FlaxBertForMultipleChoice.__call__.example",$$slots:{default:[pO]},$$scope:{ctx:B}}}),Kc=new $e({}),Jc=new C({props:{name:"class transformers.FlaxBertForTokenClassification",anchor:"transformers.FlaxBertForTokenClassification",parameters:[{name:"config",val:": BertConfig"},{name:"input_shape",val:": typing.Tuple = (1, 1)"},{name:"seed",val:": int = 0"},{name:"dtype",val:": dtype = <class 'jax.numpy.float32'>"},{name:"_do_init",val:": bool = True"},{name:"gradient_checkpointing",val:": bool = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.FlaxBertForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig">BertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.FlaxBertForTokenClassification.dtype",description:`<strong>dtype</strong> (<code>jax.numpy.dtype</code>, <em>optional</em>, defaults to <code>jax.numpy.float32</code>) &#x2014;
The data type of the computation. Can be one of <code>jax.numpy.float32</code>, <code>jax.numpy.float16</code> (on GPUs) and
<code>jax.numpy.bfloat16</code> (on TPUs).</p>
<p>This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
specified all the computation will be performed with the given <code>dtype</code>.</p>
<p><strong>Note that this only specifies the dtype of the computation and does not influence the dtype of model
parameters.</strong></p>
<p>If you wish to change the dtype of the model parameters, see <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16">to_fp16()</a> and
<a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16">to_bf16()</a>.`,name:"dtype"},{anchor:"transformers.FlaxBertForTokenClassification.dtype",description:`<strong>dtype</strong> (<code>jax.numpy.dtype</code>, <em>optional</em>, defaults to <code>jax.numpy.float32</code>) &#x2014;
The data type of the computation. Can be one of <code>jax.numpy.float32</code>, <code>jax.numpy.float16</code> (on GPUs) and
<code>jax.numpy.bfloat16</code> (on TPUs).</p>
<p>This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
specified all the computation will be performed with the given <code>dtype</code>.</p>
<p><strong>Note that this only specifies the dtype of the computation and does not influence the dtype of model
parameters.</strong></p>
<p>If you wish to change the dtype of the model parameters, see <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16">to_fp16()</a> and
<a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16">to_bf16()</a>.`,name:"dtype"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_flax_bert.py#L1525"}}),np=new C({props:{name:"__call__",anchor:"transformers.FlaxBertForTokenClassification.__call__",parameters:[{name:"input_ids",val:""},{name:"attention_mask",val:" = None"},{name:"token_type_ids",val:" = None"},{name:"position_ids",val:" = None"},{name:"head_mask",val:" = None"},{name:"encoder_hidden_states",val:" = None"},{name:"encoder_attention_mask",val:" = None"},{name:"params",val:": dict = None"},{name:"dropout_rng",val:": PRNGKey = None"},{name:"train",val:": bool = False"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"past_key_values",val:": dict = None"}],parametersDescription:[{anchor:"transformers.FlaxBertForTokenClassification.__call__.input_ids",description:`<strong>input_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18729/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18729/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FlaxBertForTokenClassification.__call__.attention_mask",description:`<strong>attention_mask</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.FlaxBertForTokenClassification.__call__.token_type_ids",description:`<strong>token_type_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.FlaxBertForTokenClassification.__call__.position_ids",description:`<strong>position_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.`,name:"position_ids"},{anchor:"transformers.FlaxBertForTokenClassification.__call__.head_mask",description:`<strong>head_mask</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <code>optional) -- Mask to nullify selected heads of the attention modules. Mask values selected in </code>[0, 1]\`:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.FlaxBertForTokenClassification.__call__.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18729/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_flax_bert.py#L855",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_flax_outputs.FlaxTokenClassifierOutput"
>transformers.modeling_flax_outputs.FlaxTokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>logits</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length, config.num_labels)</code>) \u2014 Classification scores (before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_flax_outputs.FlaxTokenClassifierOutput"
>transformers.modeling_flax_outputs.FlaxTokenClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ma=new we({props:{$$slots:{default:[hO]},$$scope:{ctx:B}}}),fa=new ee({props:{anchor:"transformers.FlaxBertForTokenClassification.__call__.example",$$slots:{default:[mO]},$$scope:{ctx:B}}}),sp=new $e({}),rp=new C({props:{name:"class transformers.FlaxBertForQuestionAnswering",anchor:"transformers.FlaxBertForQuestionAnswering",parameters:[{name:"config",val:": BertConfig"},{name:"input_shape",val:": typing.Tuple = (1, 1)"},{name:"seed",val:": int = 0"},{name:"dtype",val:": dtype = <class 'jax.numpy.float32'>"},{name:"_do_init",val:": bool = True"},{name:"gradient_checkpointing",val:": bool = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.FlaxBertForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig">BertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.FlaxBertForQuestionAnswering.dtype",description:`<strong>dtype</strong> (<code>jax.numpy.dtype</code>, <em>optional</em>, defaults to <code>jax.numpy.float32</code>) &#x2014;
The data type of the computation. Can be one of <code>jax.numpy.float32</code>, <code>jax.numpy.float16</code> (on GPUs) and
<code>jax.numpy.bfloat16</code> (on TPUs).</p>
<p>This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
specified all the computation will be performed with the given <code>dtype</code>.</p>
<p><strong>Note that this only specifies the dtype of the computation and does not influence the dtype of model
parameters.</strong></p>
<p>If you wish to change the dtype of the model parameters, see <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16">to_fp16()</a> and
<a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16">to_bf16()</a>.`,name:"dtype"},{anchor:"transformers.FlaxBertForQuestionAnswering.dtype",description:`<strong>dtype</strong> (<code>jax.numpy.dtype</code>, <em>optional</em>, defaults to <code>jax.numpy.float32</code>) &#x2014;
The data type of the computation. Can be one of <code>jax.numpy.float32</code>, <code>jax.numpy.float16</code> (on GPUs) and
<code>jax.numpy.bfloat16</code> (on TPUs).</p>
<p>This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
specified all the computation will be performed with the given <code>dtype</code>.</p>
<p><strong>Note that this only specifies the dtype of the computation and does not influence the dtype of model
parameters.</strong></p>
<p>If you wish to change the dtype of the model parameters, see <a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16">to_fp16()</a> and
<a href="/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16">to_bf16()</a>.`,name:"dtype"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_flax_bert.py#L1598"}}),mp=new C({props:{name:"__call__",anchor:"transformers.FlaxBertForQuestionAnswering.__call__",parameters:[{name:"input_ids",val:""},{name:"attention_mask",val:" = None"},{name:"token_type_ids",val:" = None"},{name:"position_ids",val:" = None"},{name:"head_mask",val:" = None"},{name:"encoder_hidden_states",val:" = None"},{name:"encoder_attention_mask",val:" = None"},{name:"params",val:": dict = None"},{name:"dropout_rng",val:": PRNGKey = None"},{name:"train",val:": bool = False"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"past_key_values",val:": dict = None"}],parametersDescription:[{anchor:"transformers.FlaxBertForQuestionAnswering.__call__.input_ids",description:`<strong>input_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18729/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18729/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FlaxBertForQuestionAnswering.__call__.attention_mask",description:`<strong>attention_mask</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.FlaxBertForQuestionAnswering.__call__.token_type_ids",description:`<strong>token_type_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.FlaxBertForQuestionAnswering.__call__.position_ids",description:`<strong>position_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.`,name:"position_ids"},{anchor:"transformers.FlaxBertForQuestionAnswering.__call__.head_mask",description:`<strong>head_mask</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <code>optional) -- Mask to nullify selected heads of the attention modules. Mask values selected in </code>[0, 1]\`:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.FlaxBertForQuestionAnswering.__call__.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18729/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18729/src/transformers/models/bert/modeling_flax_bert.py#L855",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_flax_outputs.FlaxQuestionAnsweringModelOutput"
>transformers.modeling_flax_outputs.FlaxQuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>start_logits</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length)</code>) \u2014 Span-start scores (before SoftMax).</p>
</li>
<li>
<p><strong>end_logits</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length)</code>) \u2014 Span-end scores (before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18729/en/main_classes/output#transformers.modeling_flax_outputs.FlaxQuestionAnsweringModelOutput"
>transformers.modeling_flax_outputs.FlaxQuestionAnsweringModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ga=new we({props:{$$slots:{default:[fO]},$$scope:{ctx:B}}}),_a=new ee({props:{anchor:"transformers.FlaxBertForQuestionAnswering.__call__.example",$$slots:{default:[uO]},$$scope:{ctx:B}}}),{c(){d=r("meta"),_=c(),m=r("h1"),h=r("a"),g=r("span"),v(l.$$.fragment),f=c(),E=r("span"),be=n("BERT"),X=c(),z=r("h2"),ne=r("a"),L=r("span"),v(re.$$.fragment),ke=c(),D=r("span"),Te=n("Overview"),me=c(),J=r("p"),O=n("The BERT model was proposed in "),ae=r("a"),Y=n("BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"),P=n(` by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. It\u2019s a
bidirectional transformer pretrained using a combination of masked language modeling objective and next sentence
prediction on a large corpus comprising the Toronto Book Corpus and Wikipedia.`),j=c(),ie=r("p"),H=n("The abstract from the paper is the following:"),fe=c(),le=r("p"),S=r("em"),ve=n(`We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations
from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional
representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result,
the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models
for a wide range of tasks, such as question answering and language inference, without substantial task-specific
architecture modifications.`),ue=c(),q=r("p"),ce=r("em"),R=n(`BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural
language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI
accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute
improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).`),ge=c(),de=r("p"),V=n("Tips:"),_e=c(),se=r("ul"),N=r("li"),ye=n(`BERT is a model with absolute position embeddings so it\u2019s usually advised to pad the inputs on the right rather than
the left.`),Q=c(),pe=r("li"),T=n(`BERT was trained with the masked language modeling (MLM) and next sentence prediction (NSP) objectives. It is
efficient at predicting masked tokens and at NLU in general, but is not optimal for text generation.`),M=c(),K=r("p"),ze=n("This model was contributed by "),Be=r("a"),I=n("thomwolf"),Pe=n(". The original code can be found "),Ee=r("a"),qe=n("here"),A=n("."),W=c(),xe=r("h2"),Fe=r("a"),U=r("span"),v(Me.$$.fragment),je=c(),he=r("span"),Ce=n("BertConfig"),Eb=c(),Nt=r("div"),v(Ra.$$.fragment),jv=c(),ho=r("p"),Cv=n("This is the configuration class to store the configuration of a "),Ep=r("a"),Nv=n("BertModel"),Ov=n(" or a "),Mp=r("a"),Iv=n("TFBertModel"),Av=n(`. It is used to
instantiate a BERT model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the BERT
`),Va=r("a"),Lv=n("bert-base-uncased"),Dv=n(" architecture."),Sv=c(),Ko=r("p"),Uv=n("Configuration objects inherit from "),zp=r("a"),Wv=n("PretrainedConfig"),Hv=n(` and can be used to control the model outputs. Read the
documentation from `),Pp=r("a"),Rv=n("PretrainedConfig"),Vv=n(" for more information."),Qv=c(),v(_s.$$.fragment),Mb=c(),Jo=r("h2"),bs=r("a"),km=r("span"),v(Qa.$$.fragment),Kv=c(),Tm=r("span"),Jv=n("BertTokenizer"),zb=c(),Ne=r("div"),v(Ka.$$.fragment),Gv=c(),vm=r("p"),Xv=n("Construct a BERT tokenizer. Based on WordPiece."),Yv=c(),Ja=r("p"),Zv=n("This tokenizer inherits from "),qp=r("a"),ey=n("PreTrainedTokenizer"),ty=n(` which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`),oy=c(),Bo=r("div"),v(Ga.$$.fragment),ny=c(),ym=r("p"),sy=n(`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A BERT sequence has the following format:`),ry=c(),Xa=r("ul"),jp=r("li"),ay=n("single sequence: "),wm=r("code"),iy=n("[CLS] X [SEP]"),ly=c(),Cp=r("li"),dy=n("pair of sequences: "),$m=r("code"),cy=n("[CLS] A [SEP] B [SEP]"),py=c(),ks=r("div"),v(Ya.$$.fragment),hy=c(),Za=r("p"),my=n(`Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer `),xm=r("code"),fy=n("prepare_for_model"),uy=n(" method."),gy=c(),It=r("div"),v(ei.$$.fragment),_y=c(),Fm=r("p"),by=n("Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence"),ky=c(),v(Ts.$$.fragment),Ty=c(),Go=r("p"),vy=n("If "),Bm=r("code"),yy=n("token_ids_1"),wy=n(" is "),Em=r("code"),$y=n("None"),xy=n(", this method only returns the first portion of the mask (0s)."),Fy=c(),Np=r("div"),v(ti.$$.fragment),Pb=c(),Xo=r("h2"),vs=r("a"),Mm=r("span"),v(oi.$$.fragment),By=c(),zm=r("span"),Ey=n("BertTokenizerFast"),qb=c(),rt=r("div"),v(ni.$$.fragment),My=c(),si=r("p"),zy=n("Construct a \u201Cfast\u201D BERT tokenizer (backed by HuggingFace\u2019s "),Pm=r("em"),Py=n("tokenizers"),qy=n(" library). Based on WordPiece."),jy=c(),ri=r("p"),Cy=n("This tokenizer inherits from "),Op=r("a"),Ny=n("PreTrainedTokenizerFast"),Oy=n(` which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`),Iy=c(),Eo=r("div"),v(ai.$$.fragment),Ay=c(),qm=r("p"),Ly=n(`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A BERT sequence has the following format:`),Dy=c(),ii=r("ul"),Ip=r("li"),Sy=n("single sequence: "),jm=r("code"),Uy=n("[CLS] X [SEP]"),Wy=c(),Ap=r("li"),Hy=n("pair of sequences: "),Cm=r("code"),Ry=n("[CLS] A [SEP] B [SEP]"),Vy=c(),At=r("div"),v(li.$$.fragment),Qy=c(),Nm=r("p"),Ky=n("Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence"),Jy=c(),v(ys.$$.fragment),Gy=c(),Yo=r("p"),Xy=n("If "),Om=r("code"),Yy=n("token_ids_1"),Zy=n(" is "),Im=r("code"),e1=n("None"),t1=n(", this method only returns the first portion of the mask (0s)."),jb=c(),Zo=r("h2"),ws=r("a"),Am=r("span"),v(di.$$.fragment),o1=c(),Lm=r("span"),n1=n("TFBertTokenizer"),Cb=c(),at=r("div"),v(ci.$$.fragment),s1=c(),en=r("p"),r1=n(`This is an in-graph tokenizer for BERT. It should be initialized similarly to other tokenizers, using the
`),Dm=r("code"),a1=n("from_pretrained()"),i1=n(" method. It can also be initialized with the "),Sm=r("code"),l1=n("from_tokenizer()"),d1=n(` method, which imports settings
from an existing standard tokenizer object.`),c1=c(),pi=r("p"),p1=n(`In-graph tokenizers, unlike other Hugging Face tokenizers, are actually Keras layers and are designed to be run
when the model is called, rather than during preprocessing. As a result, they have somewhat more limited options
than standard tokenizer classes. They are most useful when you want to create an end-to-end model that goes
straight from `),Um=r("code"),h1=n("tf.string"),m1=n(" inputs to outputs."),f1=c(),Mo=r("div"),v(hi.$$.fragment),u1=c(),mi=r("p"),g1=n("Instantiate a "),Wm=r("code"),_1=n("TFBertTokenizer"),b1=n(" from a pre-trained tokenizer."),k1=c(),v($s.$$.fragment),T1=c(),zo=r("div"),v(fi.$$.fragment),v1=c(),tn=r("p"),y1=n("Initialize a "),Hm=r("code"),w1=n("TFBertTokenizer"),$1=n(" from an existing "),Rm=r("code"),x1=n("Tokenizer"),F1=n("."),B1=c(),v(xs.$$.fragment),Nb=c(),on=r("h2"),Fs=r("a"),Vm=r("span"),v(ui.$$.fragment),E1=c(),Qm=r("span"),M1=n("Bert specific outputs"),Ob=c(),nn=r("div"),v(gi.$$.fragment),z1=c(),_i=r("p"),P1=n("Output type of "),Lp=r("a"),q1=n("BertForPreTraining"),j1=n("."),Ib=c(),sn=r("div"),v(bi.$$.fragment),C1=c(),ki=r("p"),N1=n("Output type of "),Dp=r("a"),O1=n("TFBertForPreTraining"),I1=n("."),Ab=c(),mo=r("div"),v(Ti.$$.fragment),A1=c(),vi=r("p"),L1=n("Output type of "),Sp=r("a"),D1=n("BertForPreTraining"),S1=n("."),U1=c(),Bs=r("div"),v(yi.$$.fragment),W1=c(),Km=r("p"),H1=n("\u201CReturns a new object replacing the specified fields with new values."),Lb=c(),rn=r("h2"),Es=r("a"),Jm=r("span"),v(wi.$$.fragment),R1=c(),Gm=r("span"),V1=n("BertModel"),Db=c(),Oe=r("div"),v($i.$$.fragment),Q1=c(),Xm=r("p"),K1=n("The bare Bert Model transformer outputting raw hidden-states without any specific head on top."),J1=c(),xi=r("p"),G1=n("This model inherits from "),Up=r("a"),X1=n("PreTrainedModel"),Y1=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Z1=c(),Fi=r("p"),ew=n("This model is also a PyTorch "),Bi=r("a"),tw=n("torch.nn.Module"),ow=n(` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),nw=c(),Ei=r("p"),sw=n(`The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
cross-attention is added between the self-attention layers, following the architecture described in `),Mi=r("a"),rw=n(`Attention is
all you need`),aw=n(` by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.`),iw=c(),Ke=r("p"),lw=n("To behave as an decoder the model needs to be initialized with the "),Ym=r("code"),dw=n("is_decoder"),cw=n(` argument of the configuration set
to `),Zm=r("code"),pw=n("True"),hw=n(". To be used in a Seq2Seq model, the model needs to initialized with both "),ef=r("code"),mw=n("is_decoder"),fw=n(` argument and
`),tf=r("code"),uw=n("add_cross_attention"),gw=n(" set to "),of=r("code"),_w=n("True"),bw=n("; an "),nf=r("code"),kw=n("encoder_hidden_states"),Tw=n(" is then expected as an input to the forward pass."),vw=c(),Lt=r("div"),v(zi.$$.fragment),yw=c(),an=r("p"),ww=n("The "),Wp=r("a"),$w=n("BertModel"),xw=n(" forward method, overrides the "),sf=r("code"),Fw=n("__call__"),Bw=n(" special method."),Ew=c(),v(Ms.$$.fragment),Mw=c(),v(zs.$$.fragment),Sb=c(),ln=r("h2"),Ps=r("a"),rf=r("span"),v(Pi.$$.fragment),zw=c(),af=r("span"),Pw=n("BertForPreTraining"),Ub=c(),it=r("div"),v(qi.$$.fragment),qw=c(),dn=r("p"),jw=n("Bert Model with two heads on top as done during the pretraining: a "),lf=r("code"),Cw=n("masked language modeling"),Nw=n(" head and a "),df=r("code"),Ow=n("next sentence prediction (classification)"),Iw=n(" head."),Aw=c(),ji=r("p"),Lw=n("This model inherits from "),Hp=r("a"),Dw=n("PreTrainedModel"),Sw=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Uw=c(),Ci=r("p"),Ww=n("This model is also a PyTorch "),Ni=r("a"),Hw=n("torch.nn.Module"),Rw=n(` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Vw=c(),Dt=r("div"),v(Oi.$$.fragment),Qw=c(),cn=r("p"),Kw=n("The "),Rp=r("a"),Jw=n("BertForPreTraining"),Gw=n(" forward method, overrides the "),cf=r("code"),Xw=n("__call__"),Yw=n(" special method."),Zw=c(),v(qs.$$.fragment),e2=c(),v(js.$$.fragment),Wb=c(),pn=r("h2"),Cs=r("a"),pf=r("span"),v(Ii.$$.fragment),t2=c(),hf=r("span"),o2=n("BertLMHeadModel"),Hb=c(),lt=r("div"),v(Ai.$$.fragment),n2=c(),Li=r("p"),s2=n("Bert Model with a "),mf=r("code"),r2=n("language modeling"),a2=n(" head on top for CLM fine-tuning."),i2=c(),Di=r("p"),l2=n("This model inherits from "),Vp=r("a"),d2=n("PreTrainedModel"),c2=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),p2=c(),Si=r("p"),h2=n("This model is also a PyTorch "),Ui=r("a"),m2=n("torch.nn.Module"),f2=n(` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),u2=c(),St=r("div"),v(Wi.$$.fragment),g2=c(),hn=r("p"),_2=n("The "),Qp=r("a"),b2=n("BertLMHeadModel"),k2=n(" forward method, overrides the "),ff=r("code"),T2=n("__call__"),v2=n(" special method."),y2=c(),v(Ns.$$.fragment),w2=c(),v(Os.$$.fragment),Rb=c(),mn=r("h2"),Is=r("a"),uf=r("span"),v(Hi.$$.fragment),$2=c(),gf=r("span"),x2=n("BertForMaskedLM"),Vb=c(),dt=r("div"),v(Ri.$$.fragment),F2=c(),Vi=r("p"),B2=n("Bert Model with a "),_f=r("code"),E2=n("language modeling"),M2=n(" head on top."),z2=c(),Qi=r("p"),P2=n("This model inherits from "),Kp=r("a"),q2=n("PreTrainedModel"),j2=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),C2=c(),Ki=r("p"),N2=n("This model is also a PyTorch "),Ji=r("a"),O2=n("torch.nn.Module"),I2=n(` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),A2=c(),ut=r("div"),v(Gi.$$.fragment),L2=c(),fn=r("p"),D2=n("The "),Jp=r("a"),S2=n("BertForMaskedLM"),U2=n(" forward method, overrides the "),bf=r("code"),W2=n("__call__"),H2=n(" special method."),R2=c(),v(As.$$.fragment),V2=c(),v(Ls.$$.fragment),Q2=c(),v(Ds.$$.fragment),Qb=c(),un=r("h2"),Ss=r("a"),kf=r("span"),v(Xi.$$.fragment),K2=c(),Tf=r("span"),J2=n("BertForNextSentencePrediction"),Kb=c(),ct=r("div"),v(Yi.$$.fragment),G2=c(),Zi=r("p"),X2=n("Bert Model with a "),vf=r("code"),Y2=n("next sentence prediction (classification)"),Z2=n(" head on top."),e$=c(),el=r("p"),t$=n("This model inherits from "),Gp=r("a"),o$=n("PreTrainedModel"),n$=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),s$=c(),tl=r("p"),r$=n("This model is also a PyTorch "),ol=r("a"),a$=n("torch.nn.Module"),i$=n(` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),l$=c(),Ut=r("div"),v(nl.$$.fragment),d$=c(),gn=r("p"),c$=n("The "),Xp=r("a"),p$=n("BertForNextSentencePrediction"),h$=n(" forward method, overrides the "),yf=r("code"),m$=n("__call__"),f$=n(" special method."),u$=c(),v(Us.$$.fragment),g$=c(),v(Ws.$$.fragment),Jb=c(),_n=r("h2"),Hs=r("a"),wf=r("span"),v(sl.$$.fragment),_$=c(),$f=r("span"),b$=n("BertForSequenceClassification"),Gb=c(),pt=r("div"),v(rl.$$.fragment),k$=c(),xf=r("p"),T$=n(`Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
output) e.g. for GLUE tasks.`),v$=c(),al=r("p"),y$=n("This model inherits from "),Yp=r("a"),w$=n("PreTrainedModel"),$$=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),x$=c(),il=r("p"),F$=n("This model is also a PyTorch "),ll=r("a"),B$=n("torch.nn.Module"),E$=n(` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),M$=c(),Qe=r("div"),v(dl.$$.fragment),z$=c(),bn=r("p"),P$=n("The "),Zp=r("a"),q$=n("BertForSequenceClassification"),j$=n(" forward method, overrides the "),Ff=r("code"),C$=n("__call__"),N$=n(" special method."),O$=c(),v(Rs.$$.fragment),I$=c(),v(Vs.$$.fragment),A$=c(),v(Qs.$$.fragment),L$=c(),v(Ks.$$.fragment),D$=c(),v(Js.$$.fragment),Xb=c(),kn=r("h2"),Gs=r("a"),Bf=r("span"),v(cl.$$.fragment),S$=c(),Ef=r("span"),U$=n("BertForMultipleChoice"),Yb=c(),ht=r("div"),v(pl.$$.fragment),W$=c(),Mf=r("p"),H$=n(`Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`),R$=c(),hl=r("p"),V$=n("This model inherits from "),eh=r("a"),Q$=n("PreTrainedModel"),K$=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),J$=c(),ml=r("p"),G$=n("This model is also a PyTorch "),fl=r("a"),X$=n("torch.nn.Module"),Y$=n(` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Z$=c(),Wt=r("div"),v(ul.$$.fragment),ex=c(),Tn=r("p"),tx=n("The "),th=r("a"),ox=n("BertForMultipleChoice"),nx=n(" forward method, overrides the "),zf=r("code"),sx=n("__call__"),rx=n(" special method."),ax=c(),v(Xs.$$.fragment),ix=c(),v(Ys.$$.fragment),Zb=c(),vn=r("h2"),Zs=r("a"),Pf=r("span"),v(gl.$$.fragment),lx=c(),qf=r("span"),dx=n("BertForTokenClassification"),ek=c(),mt=r("div"),v(_l.$$.fragment),cx=c(),jf=r("p"),px=n(`Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
Named-Entity-Recognition (NER) tasks.`),hx=c(),bl=r("p"),mx=n("This model inherits from "),oh=r("a"),fx=n("PreTrainedModel"),ux=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),gx=c(),kl=r("p"),_x=n("This model is also a PyTorch "),Tl=r("a"),bx=n("torch.nn.Module"),kx=n(` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Tx=c(),gt=r("div"),v(vl.$$.fragment),vx=c(),yn=r("p"),yx=n("The "),nh=r("a"),wx=n("BertForTokenClassification"),$x=n(" forward method, overrides the "),Cf=r("code"),xx=n("__call__"),Fx=n(" special method."),Bx=c(),v(er.$$.fragment),Ex=c(),v(tr.$$.fragment),Mx=c(),v(or.$$.fragment),tk=c(),wn=r("h2"),nr=r("a"),Nf=r("span"),v(yl.$$.fragment),zx=c(),Of=r("span"),Px=n("BertForQuestionAnswering"),ok=c(),ft=r("div"),v(wl.$$.fragment),qx=c(),$n=r("p"),jx=n(`Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
layers on top of the hidden-states output to compute `),If=r("code"),Cx=n("span start logits"),Nx=n(" and "),Af=r("code"),Ox=n("span end logits"),Ix=n(")."),Ax=c(),$l=r("p"),Lx=n("This model inherits from "),sh=r("a"),Dx=n("PreTrainedModel"),Sx=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Ux=c(),xl=r("p"),Wx=n("This model is also a PyTorch "),Fl=r("a"),Hx=n("torch.nn.Module"),Rx=n(` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Vx=c(),_t=r("div"),v(Bl.$$.fragment),Qx=c(),xn=r("p"),Kx=n("The "),rh=r("a"),Jx=n("BertForQuestionAnswering"),Gx=n(" forward method, overrides the "),Lf=r("code"),Xx=n("__call__"),Yx=n(" special method."),Zx=c(),v(sr.$$.fragment),eF=c(),v(rr.$$.fragment),tF=c(),v(ar.$$.fragment),nk=c(),Fn=r("h2"),ir=r("a"),Df=r("span"),v(El.$$.fragment),oF=c(),Sf=r("span"),nF=n("TFBertModel"),sk=c(),Je=r("div"),v(Ml.$$.fragment),sF=c(),Uf=r("p"),rF=n("The bare Bert Model transformer outputting raw hidden-states without any specific head on top."),aF=c(),zl=r("p"),iF=n("This model inherits from "),ah=r("a"),lF=n("TFPreTrainedModel"),dF=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),cF=c(),Pl=r("p"),pF=n("This model is also a "),ql=r("a"),hF=n("tf.keras.Model"),mF=n(` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),fF=c(),v(lr.$$.fragment),uF=c(),Ht=r("div"),v(jl.$$.fragment),gF=c(),Bn=r("p"),_F=n("The "),ih=r("a"),bF=n("TFBertModel"),kF=n(" forward method, overrides the "),Wf=r("code"),TF=n("__call__"),vF=n(" special method."),yF=c(),v(dr.$$.fragment),wF=c(),v(cr.$$.fragment),rk=c(),En=r("h2"),pr=r("a"),Hf=r("span"),v(Cl.$$.fragment),$F=c(),Rf=r("span"),xF=n("TFBertForPreTraining"),ak=c(),Ge=r("div"),v(Nl.$$.fragment),FF=c(),Mn=r("p"),BF=n(`Bert Model with two heads on top as done during the pretraining:
a `),Vf=r("code"),EF=n("masked language modeling"),MF=n(" head and a "),Qf=r("code"),zF=n("next sentence prediction (classification)"),PF=n(" head."),qF=c(),Ol=r("p"),jF=n("This model inherits from "),lh=r("a"),CF=n("TFPreTrainedModel"),NF=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),OF=c(),Il=r("p"),IF=n("This model is also a "),Al=r("a"),AF=n("tf.keras.Model"),LF=n(` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),DF=c(),v(hr.$$.fragment),SF=c(),Rt=r("div"),v(Ll.$$.fragment),UF=c(),zn=r("p"),WF=n("The "),dh=r("a"),HF=n("TFBertForPreTraining"),RF=n(" forward method, overrides the "),Kf=r("code"),VF=n("__call__"),QF=n(" special method."),KF=c(),v(mr.$$.fragment),JF=c(),v(fr.$$.fragment),ik=c(),Pn=r("h2"),ur=r("a"),Jf=r("span"),v(Dl.$$.fragment),GF=c(),Gf=r("span"),XF=n("TFBertModelLMHeadModel"),lk=c(),qn=r("div"),v(Sl.$$.fragment),YF=c(),bt=r("div"),v(Ul.$$.fragment),ZF=c(),Ie=r("p"),e0=n("encoder_hidden_states  ("),Xf=r("code"),t0=n("tf.Tensor"),o0=n(" of shape "),Yf=r("code"),n0=n("(batch_size, sequence_length, hidden_size)"),s0=n(", "),Zf=r("em"),r0=n("optional"),a0=n(`):
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
the model is configured as a decoder.
encoder_attention_mask (`),eu=r("code"),i0=n("tf.Tensor"),l0=n(" of shape "),tu=r("code"),d0=n("(batch_size, sequence_length)"),c0=n(", "),ou=r("em"),p0=n("optional"),h0=n(`):
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in `),nu=r("code"),m0=n("[0, 1]"),f0=n(":"),u0=c(),Wl=r("ul"),Hl=r("li"),g0=n("1 for tokens that are "),su=r("strong"),_0=n("not masked"),b0=n(","),k0=c(),Rl=r("li"),T0=n("0 for tokens that are "),ru=r("strong"),v0=n("masked"),y0=n("."),w0=c(),G=r("p"),$0=n("past_key_values ("),au=r("code"),x0=n("Tuple[Tuple[tf.Tensor]]"),F0=n(" of length "),iu=r("code"),B0=n("config.n_layers"),E0=n(`)
contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
If `),lu=r("code"),M0=n("past_key_values"),z0=n(" are used, the user can optionally input only the last "),du=r("code"),P0=n("decoder_input_ids"),q0=n(` (those that
don\u2019t have their past key value states given to this model) of shape `),cu=r("code"),j0=n("(batch_size, 1)"),C0=n(` instead of all
`),pu=r("code"),N0=n("decoder_input_ids"),O0=n(" of shape "),hu=r("code"),I0=n("(batch_size, sequence_length)"),A0=n(`.
use_cache (`),mu=r("code"),L0=n("bool"),D0=n(", "),fu=r("em"),S0=n("optional"),U0=n(", defaults to "),uu=r("code"),W0=n("True"),H0=n(`):
If set to `),gu=r("code"),R0=n("True"),V0=n(", "),_u=r("code"),Q0=n("past_key_values"),K0=n(` key value states are returned and can be used to speed up decoding (see
`),bu=r("code"),J0=n("past_key_values"),G0=n("). Set to "),ku=r("code"),X0=n("False"),Y0=n(" during training, "),Tu=r("code"),Z0=n("True"),eB=n(` during generation
labels (`),vu=r("code"),tB=n("tf.Tensor"),oB=n(" or "),yu=r("code"),nB=n("np.ndarray"),sB=n(" of shape "),wu=r("code"),rB=n("(batch_size, sequence_length)"),aB=n(", "),$u=r("em"),iB=n("optional"),lB=n(`):
Labels for computing the cross entropy classification loss. Indices should be in `),xu=r("code"),dB=n("[0, ..., config.vocab_size - 1]"),cB=n("."),pB=c(),v(gr.$$.fragment),dk=c(),jn=r("h2"),_r=r("a"),Fu=r("span"),v(Vl.$$.fragment),hB=c(),Bu=r("span"),mB=n("TFBertForMaskedLM"),ck=c(),Xe=r("div"),v(Ql.$$.fragment),fB=c(),Kl=r("p"),uB=n("Bert Model with a "),Eu=r("code"),gB=n("language modeling"),_B=n(" head on top."),bB=c(),Jl=r("p"),kB=n("This model inherits from "),ch=r("a"),TB=n("TFPreTrainedModel"),vB=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),yB=c(),Gl=r("p"),wB=n("This model is also a "),Xl=r("a"),$B=n("tf.keras.Model"),xB=n(` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),FB=c(),v(br.$$.fragment),BB=c(),kt=r("div"),v(Yl.$$.fragment),EB=c(),Cn=r("p"),MB=n("The "),ph=r("a"),zB=n("TFBertForMaskedLM"),PB=n(" forward method, overrides the "),Mu=r("code"),qB=n("__call__"),jB=n(" special method."),CB=c(),v(kr.$$.fragment),NB=c(),v(Tr.$$.fragment),OB=c(),v(vr.$$.fragment),pk=c(),Nn=r("h2"),yr=r("a"),zu=r("span"),v(Zl.$$.fragment),IB=c(),Pu=r("span"),AB=n("TFBertForNextSentencePrediction"),hk=c(),Ye=r("div"),v(ed.$$.fragment),LB=c(),td=r("p"),DB=n("Bert Model with a "),qu=r("code"),SB=n("next sentence prediction (classification)"),UB=n(" head on top."),WB=c(),od=r("p"),HB=n("This model inherits from "),hh=r("a"),RB=n("TFPreTrainedModel"),VB=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),QB=c(),nd=r("p"),KB=n("This model is also a "),sd=r("a"),JB=n("tf.keras.Model"),GB=n(` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),XB=c(),v(wr.$$.fragment),YB=c(),Vt=r("div"),v(rd.$$.fragment),ZB=c(),On=r("p"),eE=n("The "),mh=r("a"),tE=n("TFBertForNextSentencePrediction"),oE=n(" forward method, overrides the "),ju=r("code"),nE=n("__call__"),sE=n(" special method."),rE=c(),v($r.$$.fragment),aE=c(),v(xr.$$.fragment),mk=c(),In=r("h2"),Fr=r("a"),Cu=r("span"),v(ad.$$.fragment),iE=c(),Nu=r("span"),lE=n("TFBertForSequenceClassification"),fk=c(),Ze=r("div"),v(id.$$.fragment),dE=c(),Ou=r("p"),cE=n(`Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
output) e.g. for GLUE tasks.`),pE=c(),ld=r("p"),hE=n("This model inherits from "),fh=r("a"),mE=n("TFPreTrainedModel"),fE=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),uE=c(),dd=r("p"),gE=n("This model is also a "),cd=r("a"),_E=n("tf.keras.Model"),bE=n(` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),kE=c(),v(Br.$$.fragment),TE=c(),Tt=r("div"),v(pd.$$.fragment),vE=c(),An=r("p"),yE=n("The "),uh=r("a"),wE=n("TFBertForSequenceClassification"),$E=n(" forward method, overrides the "),Iu=r("code"),xE=n("__call__"),FE=n(" special method."),BE=c(),v(Er.$$.fragment),EE=c(),v(Mr.$$.fragment),ME=c(),v(zr.$$.fragment),uk=c(),Ln=r("h2"),Pr=r("a"),Au=r("span"),v(hd.$$.fragment),zE=c(),Lu=r("span"),PE=n("TFBertForMultipleChoice"),gk=c(),et=r("div"),v(md.$$.fragment),qE=c(),Du=r("p"),jE=n(`Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`),CE=c(),fd=r("p"),NE=n("This model inherits from "),gh=r("a"),OE=n("TFPreTrainedModel"),IE=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),AE=c(),ud=r("p"),LE=n("This model is also a "),gd=r("a"),DE=n("tf.keras.Model"),SE=n(` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),UE=c(),v(qr.$$.fragment),WE=c(),Qt=r("div"),v(_d.$$.fragment),HE=c(),Dn=r("p"),RE=n("The "),_h=r("a"),VE=n("TFBertForMultipleChoice"),QE=n(" forward method, overrides the "),Su=r("code"),KE=n("__call__"),JE=n(" special method."),GE=c(),v(jr.$$.fragment),XE=c(),v(Cr.$$.fragment),_k=c(),Sn=r("h2"),Nr=r("a"),Uu=r("span"),v(bd.$$.fragment),YE=c(),Wu=r("span"),ZE=n("TFBertForTokenClassification"),bk=c(),tt=r("div"),v(kd.$$.fragment),eM=c(),Hu=r("p"),tM=n(`Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
Named-Entity-Recognition (NER) tasks.`),oM=c(),Td=r("p"),nM=n("This model inherits from "),bh=r("a"),sM=n("TFPreTrainedModel"),rM=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),aM=c(),vd=r("p"),iM=n("This model is also a "),yd=r("a"),lM=n("tf.keras.Model"),dM=n(` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),cM=c(),v(Or.$$.fragment),pM=c(),vt=r("div"),v(wd.$$.fragment),hM=c(),Un=r("p"),mM=n("The "),kh=r("a"),fM=n("TFBertForTokenClassification"),uM=n(" forward method, overrides the "),Ru=r("code"),gM=n("__call__"),_M=n(" special method."),bM=c(),v(Ir.$$.fragment),kM=c(),v(Ar.$$.fragment),TM=c(),v(Lr.$$.fragment),kk=c(),Wn=r("h2"),Dr=r("a"),Vu=r("span"),v($d.$$.fragment),vM=c(),Qu=r("span"),yM=n("TFBertForQuestionAnswering"),Tk=c(),ot=r("div"),v(xd.$$.fragment),wM=c(),Hn=r("p"),$M=n(`Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
layer on top of the hidden-states output to compute `),Ku=r("code"),xM=n("span start logits"),FM=n(" and "),Ju=r("code"),BM=n("span end logits"),EM=n(")."),MM=c(),Fd=r("p"),zM=n("This model inherits from "),Th=r("a"),PM=n("TFPreTrainedModel"),qM=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),jM=c(),Bd=r("p"),CM=n("This model is also a "),Ed=r("a"),NM=n("tf.keras.Model"),OM=n(` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),IM=c(),v(Sr.$$.fragment),AM=c(),yt=r("div"),v(Md.$$.fragment),LM=c(),Rn=r("p"),DM=n("The "),vh=r("a"),SM=n("TFBertForQuestionAnswering"),UM=n(" forward method, overrides the "),Gu=r("code"),WM=n("__call__"),HM=n(" special method."),RM=c(),v(Ur.$$.fragment),VM=c(),v(Wr.$$.fragment),QM=c(),v(Hr.$$.fragment),vk=c(),Vn=r("h2"),Rr=r("a"),Xu=r("span"),v(zd.$$.fragment),KM=c(),Yu=r("span"),JM=n("FlaxBertModel"),yk=c(),Ae=r("div"),v(Pd.$$.fragment),GM=c(),Zu=r("p"),XM=n("The bare Bert Model transformer outputting raw hidden-states without any specific head on top."),YM=c(),qd=r("p"),ZM=n("This model inherits from "),yh=r("a"),ez=n("FlaxPreTrainedModel"),tz=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading, saving and converting weights from PyTorch models)`),oz=c(),jd=r("p"),nz=n("This model is also a Flax Linen "),Cd=r("a"),sz=n("flax.linen.Module"),rz=n(`
subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
general usage and behavior.`),az=c(),eg=r("p"),iz=n("Finally, this model supports inherent JAX features such as:"),lz=c(),fo=r("ul"),tg=r("li"),Nd=r("a"),dz=n("Just-In-Time (JIT) compilation"),cz=c(),og=r("li"),Od=r("a"),pz=n("Automatic Differentiation"),hz=c(),ng=r("li"),Id=r("a"),mz=n("Vectorization"),fz=c(),sg=r("li"),Ad=r("a"),uz=n("Parallelization"),gz=c(),Kt=r("div"),v(Ld.$$.fragment),_z=c(),Qn=r("p"),bz=n("The "),rg=r("code"),kz=n("FlaxBertPreTrainedModel"),Tz=n(" forward method, overrides the "),ag=r("code"),vz=n("__call__"),yz=n(" special method."),wz=c(),v(Vr.$$.fragment),$z=c(),v(Qr.$$.fragment),wk=c(),Kn=r("h2"),Kr=r("a"),ig=r("span"),v(Dd.$$.fragment),xz=c(),lg=r("span"),Fz=n("FlaxBertForPreTraining"),$k=c(),Le=r("div"),v(Sd.$$.fragment),Bz=c(),Jn=r("p"),Ez=n("Bert Model with two heads on top as done during the pretraining: a "),dg=r("code"),Mz=n("masked language modeling"),zz=n(" head and a "),cg=r("code"),Pz=n("next sentence prediction (classification)"),qz=n(" head."),jz=c(),Ud=r("p"),Cz=n("This model inherits from "),wh=r("a"),Nz=n("FlaxPreTrainedModel"),Oz=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading, saving and converting weights from PyTorch models)`),Iz=c(),Wd=r("p"),Az=n("This model is also a Flax Linen "),Hd=r("a"),Lz=n("flax.linen.Module"),Dz=n(`
subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
general usage and behavior.`),Sz=c(),pg=r("p"),Uz=n("Finally, this model supports inherent JAX features such as:"),Wz=c(),uo=r("ul"),hg=r("li"),Rd=r("a"),Hz=n("Just-In-Time (JIT) compilation"),Rz=c(),mg=r("li"),Vd=r("a"),Vz=n("Automatic Differentiation"),Qz=c(),fg=r("li"),Qd=r("a"),Kz=n("Vectorization"),Jz=c(),ug=r("li"),Kd=r("a"),Gz=n("Parallelization"),Xz=c(),Jt=r("div"),v(Jd.$$.fragment),Yz=c(),Gn=r("p"),Zz=n("The "),gg=r("code"),e4=n("FlaxBertPreTrainedModel"),t4=n(" forward method, overrides the "),_g=r("code"),o4=n("__call__"),n4=n(" special method."),s4=c(),v(Jr.$$.fragment),r4=c(),v(Gr.$$.fragment),xk=c(),Xn=r("h2"),Xr=r("a"),bg=r("span"),v(Gd.$$.fragment),a4=c(),kg=r("span"),i4=n("FlaxBertForCausalLM"),Fk=c(),De=r("div"),v(Xd.$$.fragment),l4=c(),Tg=r("p"),d4=n(`Bert Model with a language modeling head on top (a linear layer on top of the hidden-states output) e.g for
autoregressive tasks.`),c4=c(),Yd=r("p"),p4=n("This model inherits from "),$h=r("a"),h4=n("FlaxPreTrainedModel"),m4=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading, saving and converting weights from PyTorch models)`),f4=c(),Zd=r("p"),u4=n("This model is also a Flax Linen "),ec=r("a"),g4=n("flax.linen.Module"),_4=n(`
subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
general usage and behavior.`),b4=c(),vg=r("p"),k4=n("Finally, this model supports inherent JAX features such as:"),T4=c(),go=r("ul"),yg=r("li"),tc=r("a"),v4=n("Just-In-Time (JIT) compilation"),y4=c(),wg=r("li"),oc=r("a"),w4=n("Automatic Differentiation"),$4=c(),$g=r("li"),nc=r("a"),x4=n("Vectorization"),F4=c(),xg=r("li"),sc=r("a"),B4=n("Parallelization"),E4=c(),Gt=r("div"),v(rc.$$.fragment),M4=c(),Yn=r("p"),z4=n("The "),Fg=r("code"),P4=n("FlaxBertPreTrainedModel"),q4=n(" forward method, overrides the "),Bg=r("code"),j4=n("__call__"),C4=n(" special method."),N4=c(),v(Yr.$$.fragment),O4=c(),v(Zr.$$.fragment),Bk=c(),Zn=r("h2"),ea=r("a"),Eg=r("span"),v(ac.$$.fragment),I4=c(),Mg=r("span"),A4=n("FlaxBertForMaskedLM"),Ek=c(),Se=r("div"),v(ic.$$.fragment),L4=c(),lc=r("p"),D4=n("Bert Model with a "),zg=r("code"),S4=n("language modeling"),U4=n(" head on top."),W4=c(),dc=r("p"),H4=n("This model inherits from "),xh=r("a"),R4=n("FlaxPreTrainedModel"),V4=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading, saving and converting weights from PyTorch models)`),Q4=c(),cc=r("p"),K4=n("This model is also a Flax Linen "),pc=r("a"),J4=n("flax.linen.Module"),G4=n(`
subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
general usage and behavior.`),X4=c(),Pg=r("p"),Y4=n("Finally, this model supports inherent JAX features such as:"),Z4=c(),_o=r("ul"),qg=r("li"),hc=r("a"),eP=n("Just-In-Time (JIT) compilation"),tP=c(),jg=r("li"),mc=r("a"),oP=n("Automatic Differentiation"),nP=c(),Cg=r("li"),fc=r("a"),sP=n("Vectorization"),rP=c(),Ng=r("li"),uc=r("a"),aP=n("Parallelization"),iP=c(),Xt=r("div"),v(gc.$$.fragment),lP=c(),es=r("p"),dP=n("The "),Og=r("code"),cP=n("FlaxBertPreTrainedModel"),pP=n(" forward method, overrides the "),Ig=r("code"),hP=n("__call__"),mP=n(" special method."),fP=c(),v(ta.$$.fragment),uP=c(),v(oa.$$.fragment),Mk=c(),ts=r("h2"),na=r("a"),Ag=r("span"),v(_c.$$.fragment),gP=c(),Lg=r("span"),_P=n("FlaxBertForNextSentencePrediction"),zk=c(),Ue=r("div"),v(bc.$$.fragment),bP=c(),kc=r("p"),kP=n("Bert Model with a "),Dg=r("code"),TP=n("next sentence prediction (classification)"),vP=n(" head on top."),yP=c(),Tc=r("p"),wP=n("This model inherits from "),Fh=r("a"),$P=n("FlaxPreTrainedModel"),xP=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading, saving and converting weights from PyTorch models)`),FP=c(),vc=r("p"),BP=n("This model is also a Flax Linen "),yc=r("a"),EP=n("flax.linen.Module"),MP=n(`
subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
general usage and behavior.`),zP=c(),Sg=r("p"),PP=n("Finally, this model supports inherent JAX features such as:"),qP=c(),bo=r("ul"),Ug=r("li"),wc=r("a"),jP=n("Just-In-Time (JIT) compilation"),CP=c(),Wg=r("li"),$c=r("a"),NP=n("Automatic Differentiation"),OP=c(),Hg=r("li"),xc=r("a"),IP=n("Vectorization"),AP=c(),Rg=r("li"),Fc=r("a"),LP=n("Parallelization"),DP=c(),Yt=r("div"),v(Bc.$$.fragment),SP=c(),os=r("p"),UP=n("The "),Vg=r("code"),WP=n("FlaxBertPreTrainedModel"),HP=n(" forward method, overrides the "),Qg=r("code"),RP=n("__call__"),VP=n(" special method."),QP=c(),v(sa.$$.fragment),KP=c(),v(ra.$$.fragment),Pk=c(),ns=r("h2"),aa=r("a"),Kg=r("span"),v(Ec.$$.fragment),JP=c(),Jg=r("span"),GP=n("FlaxBertForSequenceClassification"),qk=c(),We=r("div"),v(Mc.$$.fragment),XP=c(),Gg=r("p"),YP=n(`Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
output) e.g. for GLUE tasks.`),ZP=c(),zc=r("p"),e8=n("This model inherits from "),Bh=r("a"),t8=n("FlaxPreTrainedModel"),o8=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading, saving and converting weights from PyTorch models)`),n8=c(),Pc=r("p"),s8=n("This model is also a Flax Linen "),qc=r("a"),r8=n("flax.linen.Module"),a8=n(`
subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
general usage and behavior.`),i8=c(),Xg=r("p"),l8=n("Finally, this model supports inherent JAX features such as:"),d8=c(),ko=r("ul"),Yg=r("li"),jc=r("a"),c8=n("Just-In-Time (JIT) compilation"),p8=c(),Zg=r("li"),Cc=r("a"),h8=n("Automatic Differentiation"),m8=c(),e_=r("li"),Nc=r("a"),f8=n("Vectorization"),u8=c(),t_=r("li"),Oc=r("a"),g8=n("Parallelization"),_8=c(),Zt=r("div"),v(Ic.$$.fragment),b8=c(),ss=r("p"),k8=n("The "),o_=r("code"),T8=n("FlaxBertPreTrainedModel"),v8=n(" forward method, overrides the "),n_=r("code"),y8=n("__call__"),w8=n(" special method."),$8=c(),v(ia.$$.fragment),x8=c(),v(la.$$.fragment),jk=c(),rs=r("h2"),da=r("a"),s_=r("span"),v(Ac.$$.fragment),F8=c(),r_=r("span"),B8=n("FlaxBertForMultipleChoice"),Ck=c(),He=r("div"),v(Lc.$$.fragment),E8=c(),a_=r("p"),M8=n(`Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`),z8=c(),Dc=r("p"),P8=n("This model inherits from "),Eh=r("a"),q8=n("FlaxPreTrainedModel"),j8=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading, saving and converting weights from PyTorch models)`),C8=c(),Sc=r("p"),N8=n("This model is also a Flax Linen "),Uc=r("a"),O8=n("flax.linen.Module"),I8=n(`
subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
general usage and behavior.`),A8=c(),i_=r("p"),L8=n("Finally, this model supports inherent JAX features such as:"),D8=c(),To=r("ul"),l_=r("li"),Wc=r("a"),S8=n("Just-In-Time (JIT) compilation"),U8=c(),d_=r("li"),Hc=r("a"),W8=n("Automatic Differentiation"),H8=c(),c_=r("li"),Rc=r("a"),R8=n("Vectorization"),V8=c(),p_=r("li"),Vc=r("a"),Q8=n("Parallelization"),K8=c(),eo=r("div"),v(Qc.$$.fragment),J8=c(),as=r("p"),G8=n("The "),h_=r("code"),X8=n("FlaxBertPreTrainedModel"),Y8=n(" forward method, overrides the "),m_=r("code"),Z8=n("__call__"),eq=n(" special method."),tq=c(),v(ca.$$.fragment),oq=c(),v(pa.$$.fragment),Nk=c(),is=r("h2"),ha=r("a"),f_=r("span"),v(Kc.$$.fragment),nq=c(),u_=r("span"),sq=n("FlaxBertForTokenClassification"),Ok=c(),Re=r("div"),v(Jc.$$.fragment),rq=c(),g_=r("p"),aq=n(`Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
Named-Entity-Recognition (NER) tasks.`),iq=c(),Gc=r("p"),lq=n("This model inherits from "),Mh=r("a"),dq=n("FlaxPreTrainedModel"),cq=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading, saving and converting weights from PyTorch models)`),pq=c(),Xc=r("p"),hq=n("This model is also a Flax Linen "),Yc=r("a"),mq=n("flax.linen.Module"),fq=n(`
subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
general usage and behavior.`),uq=c(),__=r("p"),gq=n("Finally, this model supports inherent JAX features such as:"),_q=c(),vo=r("ul"),b_=r("li"),Zc=r("a"),bq=n("Just-In-Time (JIT) compilation"),kq=c(),k_=r("li"),ep=r("a"),Tq=n("Automatic Differentiation"),vq=c(),T_=r("li"),tp=r("a"),yq=n("Vectorization"),wq=c(),v_=r("li"),op=r("a"),$q=n("Parallelization"),xq=c(),to=r("div"),v(np.$$.fragment),Fq=c(),ls=r("p"),Bq=n("The "),y_=r("code"),Eq=n("FlaxBertPreTrainedModel"),Mq=n(" forward method, overrides the "),w_=r("code"),zq=n("__call__"),Pq=n(" special method."),qq=c(),v(ma.$$.fragment),jq=c(),v(fa.$$.fragment),Ik=c(),ds=r("h2"),ua=r("a"),$_=r("span"),v(sp.$$.fragment),Cq=c(),x_=r("span"),Nq=n("FlaxBertForQuestionAnswering"),Ak=c(),Ve=r("div"),v(rp.$$.fragment),Oq=c(),cs=r("p"),Iq=n(`Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
layers on top of the hidden-states output to compute `),F_=r("code"),Aq=n("span start logits"),Lq=n(" and "),B_=r("code"),Dq=n("span end logits"),Sq=n(")."),Uq=c(),ap=r("p"),Wq=n("This model inherits from "),zh=r("a"),Hq=n("FlaxPreTrainedModel"),Rq=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading, saving and converting weights from PyTorch models)`),Vq=c(),ip=r("p"),Qq=n("This model is also a Flax Linen "),lp=r("a"),Kq=n("flax.linen.Module"),Jq=n(`
subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
general usage and behavior.`),Gq=c(),E_=r("p"),Xq=n("Finally, this model supports inherent JAX features such as:"),Yq=c(),yo=r("ul"),M_=r("li"),dp=r("a"),Zq=n("Just-In-Time (JIT) compilation"),e7=c(),z_=r("li"),cp=r("a"),t7=n("Automatic Differentiation"),o7=c(),P_=r("li"),pp=r("a"),n7=n("Vectorization"),s7=c(),q_=r("li"),hp=r("a"),r7=n("Parallelization"),a7=c(),oo=r("div"),v(mp.$$.fragment),i7=c(),ps=r("p"),l7=n("The "),j_=r("code"),d7=n("FlaxBertPreTrainedModel"),c7=n(" forward method, overrides the "),C_=r("code"),p7=n("__call__"),h7=n(" special method."),m7=c(),v(ga.$$.fragment),f7=c(),v(_a.$$.fragment),this.h()},l(o){const k=V6('[data-svelte="svelte-1phssyn"]',document.head);d=a(k,"META",{name:!0,content:!0}),k.forEach(t),_=p(o),m=a(o,"H1",{class:!0});var fp=i(m);h=a(fp,"A",{id:!0,class:!0,href:!0});var N_=i(h);g=a(N_,"SPAN",{});var O_=i(g);y(l.$$.fragment,O_),O_.forEach(t),N_.forEach(t),f=p(fp),E=a(fp,"SPAN",{});var I_=i(E);be=s(I_,"BERT"),I_.forEach(t),fp.forEach(t),X=p(o),z=a(o,"H2",{class:!0});var up=i(z);ne=a(up,"A",{id:!0,class:!0,href:!0});var A_=i(ne);L=a(A_,"SPAN",{});var L_=i(L);y(re.$$.fragment,L_),L_.forEach(t),A_.forEach(t),ke=p(up),D=a(up,"SPAN",{});var D_=i(D);Te=s(D_,"Overview"),D_.forEach(t),up.forEach(t),me=p(o),J=a(o,"P",{});var gp=i(J);O=s(gp,"The BERT model was proposed in "),ae=a(gp,"A",{href:!0,rel:!0});var S_=i(ae);Y=s(S_,"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"),S_.forEach(t),P=s(gp,` by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. It\u2019s a
bidirectional transformer pretrained using a combination of masked language modeling objective and next sentence
prediction on a large corpus comprising the Toronto Book Corpus and Wikipedia.`),gp.forEach(t),j=p(o),ie=a(o,"P",{});var U_=i(ie);H=s(U_,"The abstract from the paper is the following:"),U_.forEach(t),fe=p(o),le=a(o,"P",{});var W_=i(le);S=a(W_,"EM",{});var H_=i(S);ve=s(H_,`We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations
from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional
representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result,
the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models
for a wide range of tasks, such as question answering and language inference, without substantial task-specific
architecture modifications.`),H_.forEach(t),W_.forEach(t),ue=p(o),q=a(o,"P",{});var R_=i(q);ce=a(R_,"EM",{});var V_=i(ce);R=s(V_,`BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural
language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI
accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute
improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).`),V_.forEach(t),R_.forEach(t),ge=p(o),de=a(o,"P",{});var Q_=i(de);V=s(Q_,"Tips:"),Q_.forEach(t),_e=p(o),se=a(o,"UL",{});var _p=i(se);N=a(_p,"LI",{});var K_=i(N);ye=s(K_,`BERT is a model with absolute position embeddings so it\u2019s usually advised to pad the inputs on the right rather than
the left.`),K_.forEach(t),Q=p(_p),pe=a(_p,"LI",{});var J_=i(pe);T=s(J_,`BERT was trained with the masked language modeling (MLM) and next sentence prediction (NSP) objectives. It is
efficient at predicting masked tokens and at NLU in general, but is not optimal for text generation.`),J_.forEach(t),_p.forEach(t),M=p(o),K=a(o,"P",{});var hs=i(K);ze=s(hs,"This model was contributed by "),Be=a(hs,"A",{href:!0,rel:!0});var G_=i(Be);I=s(G_,"thomwolf"),G_.forEach(t),Pe=s(hs,". The original code can be found "),Ee=a(hs,"A",{href:!0,rel:!0});var X_=i(Ee);qe=s(X_,"here"),X_.forEach(t),A=s(hs,"."),hs.forEach(t),W=p(o),xe=a(o,"H2",{class:!0});var bp=i(xe);Fe=a(bp,"A",{id:!0,class:!0,href:!0});var Y_=i(Fe);U=a(Y_,"SPAN",{});var Z_=i(U);y(Me.$$.fragment,Z_),Z_.forEach(t),Y_.forEach(t),je=p(bp),he=a(bp,"SPAN",{});var eb=i(he);Ce=s(eb,"BertConfig"),eb.forEach(t),bp.forEach(t),Eb=p(o),Nt=a(o,"DIV",{class:!0});var wo=i(Nt);y(Ra.$$.fragment,wo),jv=p(wo),ho=a(wo,"P",{});var $o=i(ho);Cv=s($o,"This is the configuration class to store the configuration of a "),Ep=a($o,"A",{href:!0});var tb=i(Ep);Nv=s(tb,"BertModel"),tb.forEach(t),Ov=s($o," or a "),Mp=a($o,"A",{href:!0});var ob=i(Mp);Iv=s(ob,"TFBertModel"),ob.forEach(t),Av=s($o,`. It is used to
instantiate a BERT model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the BERT
`),Va=a($o,"A",{href:!0,rel:!0});var nb=i(Va);Lv=s(nb,"bert-base-uncased"),nb.forEach(t),Dv=s($o," architecture."),$o.forEach(t),Sv=p(wo),Ko=a(wo,"P",{});var ms=i(Ko);Uv=s(ms,"Configuration objects inherit from "),zp=a(ms,"A",{href:!0});var sb=i(zp);Wv=s(sb,"PretrainedConfig"),sb.forEach(t),Hv=s(ms,` and can be used to control the model outputs. Read the
documentation from `),Pp=a(ms,"A",{href:!0});var rb=i(Pp);Rv=s(rb,"PretrainedConfig"),rb.forEach(t),Vv=s(ms," for more information."),ms.forEach(t),Qv=p(wo),y(_s.$$.fragment,wo),wo.forEach(t),Mb=p(o),Jo=a(o,"H2",{class:!0});var kp=i(Jo);bs=a(kp,"A",{id:!0,class:!0,href:!0});var ab=i(bs);km=a(ab,"SPAN",{});var ib=i(km);y(Qa.$$.fragment,ib),ib.forEach(t),ab.forEach(t),Kv=p(kp),Tm=a(kp,"SPAN",{});var lb=i(Tm);Jv=s(lb,"BertTokenizer"),lb.forEach(t),kp.forEach(t),zb=p(o),Ne=a(o,"DIV",{class:!0});var nt=i(Ne);y(Ka.$$.fragment,nt),Gv=p(nt),vm=a(nt,"P",{});var db=i(vm);Xv=s(db,"Construct a BERT tokenizer. Based on WordPiece."),db.forEach(t),Yv=p(nt),Ja=a(nt,"P",{});var Tp=i(Ja);Zv=s(Tp,"This tokenizer inherits from "),qp=a(Tp,"A",{href:!0});var cb=i(qp);ey=s(cb,"PreTrainedTokenizer"),cb.forEach(t),ty=s(Tp,` which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`),Tp.forEach(t),oy=p(nt),Bo=a(nt,"DIV",{class:!0});var fs=i(Bo);y(Ga.$$.fragment,fs),ny=p(fs),ym=a(fs,"P",{});var pb=i(ym);sy=s(pb,`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A BERT sequence has the following format:`),pb.forEach(t),ry=p(fs),Xa=a(fs,"UL",{});var vp=i(Xa);jp=a(vp,"LI",{});var Ph=i(jp);ay=s(Ph,"single sequence: "),wm=a(Ph,"CODE",{});var hb=i(wm);iy=s(hb,"[CLS] X [SEP]"),hb.forEach(t),Ph.forEach(t),ly=p(vp),Cp=a(vp,"LI",{});var qh=i(Cp);dy=s(qh,"pair of sequences: "),$m=a(qh,"CODE",{});var mb=i($m);cy=s(mb,"[CLS] A [SEP] B [SEP]"),mb.forEach(t),qh.forEach(t),vp.forEach(t),fs.forEach(t),py=p(nt),ks=a(nt,"DIV",{class:!0});var yp=i(ks);y(Ya.$$.fragment,yp),hy=p(yp),Za=a(yp,"P",{});var wp=i(Za);my=s(wp,`Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer `),xm=a(wp,"CODE",{});var fb=i(xm);fy=s(fb,"prepare_for_model"),fb.forEach(t),uy=s(wp," method."),wp.forEach(t),yp.forEach(t),gy=p(nt),It=a(nt,"DIV",{class:!0});var xo=i(It);y(ei.$$.fragment,xo),_y=p(xo),Fm=a(xo,"P",{});var ub=i(Fm);by=s(ub,"Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence"),ub.forEach(t),ky=p(xo),y(Ts.$$.fragment,xo),Ty=p(xo),Go=a(xo,"P",{});var us=i(Go);vy=s(us,"If "),Bm=a(us,"CODE",{});var gb=i(Bm);yy=s(gb,"token_ids_1"),gb.forEach(t),wy=s(us," is "),Em=a(us,"CODE",{});var _b=i(Em);$y=s(_b,"None"),_b.forEach(t),xy=s(us,", this method only returns the first portion of the mask (0s)."),us.forEach(t),xo.forEach(t),Fy=p(nt),Np=a(nt,"DIV",{class:!0});var bb=i(Np);y(ti.$$.fragment,bb),bb.forEach(t),nt.forEach(t),Pb=p(o),Xo=a(o,"H2",{class:!0});var $p=i(Xo);vs=a($p,"A",{id:!0,class:!0,href:!0});var kb=i(vs);Mm=a(kb,"SPAN",{});var Tb=i(Mm);y(oi.$$.fragment,Tb),Tb.forEach(t),kb.forEach(t),By=p($p),zm=a($p,"SPAN",{});var vb=i(zm);Ey=s(vb,"BertTokenizerFast"),vb.forEach(t),$p.forEach(t),qb=p(o),rt=a(o,"DIV",{class:!0});var Ot=i(rt);y(ni.$$.fragment,Ot),My=p(Ot),si=a(Ot,"P",{});var xp=i(si);zy=s(xp,"Construct a \u201Cfast\u201D BERT tokenizer (backed by HuggingFace\u2019s "),Pm=a(xp,"EM",{});var yb=i(Pm);Py=s(yb,"tokenizers"),yb.forEach(t),qy=s(xp," library). Based on WordPiece."),xp.forEach(t),jy=p(Ot),ri=a(Ot,"P",{});var Fp=i(ri);Cy=s(Fp,"This tokenizer inherits from "),Op=a(Fp,"A",{href:!0});var wb=i(Op);Ny=s(wb,"PreTrainedTokenizerFast"),wb.forEach(t),Oy=s(Fp,` which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`),Fp.forEach(t),Iy=p(Ot),Eo=a(Ot,"DIV",{class:!0});var gs=i(Eo);y(ai.$$.fragment,gs),Ay=p(gs),qm=a(gs,"P",{});var $b=i(qm);Ly=s($b,`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A BERT sequence has the following format:`),$b.forEach(t),Dy=p(gs),ii=a(gs,"UL",{});var Bp=i(ii);Ip=a(Bp,"LI",{});var jh=i(Ip);Sy=s(jh,"single sequence: "),jm=a(jh,"CODE",{});var xb=i(jm);Uy=s(xb,"[CLS] X [SEP]"),xb.forEach(t),jh.forEach(t),Wy=p(Bp),Ap=a(Bp,"LI",{});var Ch=i(Ap);Hy=s(Ch,"pair of sequences: "),Cm=a(Ch,"CODE",{});var Fb=i(Cm);Ry=s(Fb,"[CLS] A [SEP] B [SEP]"),Fb.forEach(t),Ch.forEach(t),Bp.forEach(t),gs.forEach(t),Vy=p(Ot),At=a(Ot,"DIV",{class:!0});var Fo=i(At);y(li.$$.fragment,Fo),Qy=p(Fo),Nm=a(Fo,"P",{});var Bb=i(Nm);Ky=s(Bb,"Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence"),Bb.forEach(t),Jy=p(Fo),y(ys.$$.fragment,Fo),Gy=p(Fo),Yo=a(Fo,"P",{});var Nh=i(Yo);Xy=s(Nh,"If "),Om=a(Nh,"CODE",{});var u7=i(Om);Yy=s(u7,"token_ids_1"),u7.forEach(t),Zy=s(Nh," is "),Im=a(Nh,"CODE",{});var g7=i(Im);e1=s(g7,"None"),g7.forEach(t),t1=s(Nh,", this method only returns the first portion of the mask (0s)."),Nh.forEach(t),Fo.forEach(t),Ot.forEach(t),jb=p(o),Zo=a(o,"H2",{class:!0});var Dk=i(Zo);ws=a(Dk,"A",{id:!0,class:!0,href:!0});var _7=i(ws);Am=a(_7,"SPAN",{});var b7=i(Am);y(di.$$.fragment,b7),b7.forEach(t),_7.forEach(t),o1=p(Dk),Lm=a(Dk,"SPAN",{});var k7=i(Lm);n1=s(k7,"TFBertTokenizer"),k7.forEach(t),Dk.forEach(t),Cb=p(o),at=a(o,"DIV",{class:!0});var Po=i(at);y(ci.$$.fragment,Po),s1=p(Po),en=a(Po,"P",{});var Oh=i(en);r1=s(Oh,`This is an in-graph tokenizer for BERT. It should be initialized similarly to other tokenizers, using the
`),Dm=a(Oh,"CODE",{});var T7=i(Dm);a1=s(T7,"from_pretrained()"),T7.forEach(t),i1=s(Oh," method. It can also be initialized with the "),Sm=a(Oh,"CODE",{});var v7=i(Sm);l1=s(v7,"from_tokenizer()"),v7.forEach(t),d1=s(Oh,` method, which imports settings
from an existing standard tokenizer object.`),Oh.forEach(t),c1=p(Po),pi=a(Po,"P",{});var Sk=i(pi);p1=s(Sk,`In-graph tokenizers, unlike other Hugging Face tokenizers, are actually Keras layers and are designed to be run
when the model is called, rather than during preprocessing. As a result, they have somewhat more limited options
than standard tokenizer classes. They are most useful when you want to create an end-to-end model that goes
straight from `),Um=a(Sk,"CODE",{});var y7=i(Um);h1=s(y7,"tf.string"),y7.forEach(t),m1=s(Sk," inputs to outputs."),Sk.forEach(t),f1=p(Po),Mo=a(Po,"DIV",{class:!0});var Ih=i(Mo);y(hi.$$.fragment,Ih),u1=p(Ih),mi=a(Ih,"P",{});var Uk=i(mi);g1=s(Uk,"Instantiate a "),Wm=a(Uk,"CODE",{});var w7=i(Wm);_1=s(w7,"TFBertTokenizer"),w7.forEach(t),b1=s(Uk," from a pre-trained tokenizer."),Uk.forEach(t),k1=p(Ih),y($s.$$.fragment,Ih),Ih.forEach(t),T1=p(Po),zo=a(Po,"DIV",{class:!0});var Ah=i(zo);y(fi.$$.fragment,Ah),v1=p(Ah),tn=a(Ah,"P",{});var Lh=i(tn);y1=s(Lh,"Initialize a "),Hm=a(Lh,"CODE",{});var $7=i(Hm);w1=s($7,"TFBertTokenizer"),$7.forEach(t),$1=s(Lh," from an existing "),Rm=a(Lh,"CODE",{});var x7=i(Rm);x1=s(x7,"Tokenizer"),x7.forEach(t),F1=s(Lh,"."),Lh.forEach(t),B1=p(Ah),y(xs.$$.fragment,Ah),Ah.forEach(t),Po.forEach(t),Nb=p(o),on=a(o,"H2",{class:!0});var Wk=i(on);Fs=a(Wk,"A",{id:!0,class:!0,href:!0});var F7=i(Fs);Vm=a(F7,"SPAN",{});var B7=i(Vm);y(ui.$$.fragment,B7),B7.forEach(t),F7.forEach(t),E1=p(Wk),Qm=a(Wk,"SPAN",{});var E7=i(Qm);M1=s(E7,"Bert specific outputs"),E7.forEach(t),Wk.forEach(t),Ob=p(o),nn=a(o,"DIV",{class:!0});var Hk=i(nn);y(gi.$$.fragment,Hk),z1=p(Hk),_i=a(Hk,"P",{});var Rk=i(_i);P1=s(Rk,"Output type of "),Lp=a(Rk,"A",{href:!0});var M7=i(Lp);q1=s(M7,"BertForPreTraining"),M7.forEach(t),j1=s(Rk,"."),Rk.forEach(t),Hk.forEach(t),Ib=p(o),sn=a(o,"DIV",{class:!0});var Vk=i(sn);y(bi.$$.fragment,Vk),C1=p(Vk),ki=a(Vk,"P",{});var Qk=i(ki);N1=s(Qk,"Output type of "),Dp=a(Qk,"A",{href:!0});var z7=i(Dp);O1=s(z7,"TFBertForPreTraining"),z7.forEach(t),I1=s(Qk,"."),Qk.forEach(t),Vk.forEach(t),Ab=p(o),mo=a(o,"DIV",{class:!0});var Dh=i(mo);y(Ti.$$.fragment,Dh),A1=p(Dh),vi=a(Dh,"P",{});var Kk=i(vi);L1=s(Kk,"Output type of "),Sp=a(Kk,"A",{href:!0});var P7=i(Sp);D1=s(P7,"BertForPreTraining"),P7.forEach(t),S1=s(Kk,"."),Kk.forEach(t),U1=p(Dh),Bs=a(Dh,"DIV",{class:!0});var Jk=i(Bs);y(yi.$$.fragment,Jk),W1=p(Jk),Km=a(Jk,"P",{});var q7=i(Km);H1=s(q7,"\u201CReturns a new object replacing the specified fields with new values."),q7.forEach(t),Jk.forEach(t),Dh.forEach(t),Lb=p(o),rn=a(o,"H2",{class:!0});var Gk=i(rn);Es=a(Gk,"A",{id:!0,class:!0,href:!0});var j7=i(Es);Jm=a(j7,"SPAN",{});var C7=i(Jm);y(wi.$$.fragment,C7),C7.forEach(t),j7.forEach(t),R1=p(Gk),Gm=a(Gk,"SPAN",{});var N7=i(Gm);V1=s(N7,"BertModel"),N7.forEach(t),Gk.forEach(t),Db=p(o),Oe=a(o,"DIV",{class:!0});var wt=i(Oe);y($i.$$.fragment,wt),Q1=p(wt),Xm=a(wt,"P",{});var O7=i(Xm);K1=s(O7,"The bare Bert Model transformer outputting raw hidden-states without any specific head on top."),O7.forEach(t),J1=p(wt),xi=a(wt,"P",{});var Xk=i(xi);G1=s(Xk,"This model inherits from "),Up=a(Xk,"A",{href:!0});var I7=i(Up);X1=s(I7,"PreTrainedModel"),I7.forEach(t),Y1=s(Xk,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Xk.forEach(t),Z1=p(wt),Fi=a(wt,"P",{});var Yk=i(Fi);ew=s(Yk,"This model is also a PyTorch "),Bi=a(Yk,"A",{href:!0,rel:!0});var A7=i(Bi);tw=s(A7,"torch.nn.Module"),A7.forEach(t),ow=s(Yk,` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Yk.forEach(t),nw=p(wt),Ei=a(wt,"P",{});var Zk=i(Ei);sw=s(Zk,`The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
cross-attention is added between the self-attention layers, following the architecture described in `),Mi=a(Zk,"A",{href:!0,rel:!0});var L7=i(Mi);rw=s(L7,`Attention is
all you need`),L7.forEach(t),aw=s(Zk,` by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.`),Zk.forEach(t),iw=p(wt),Ke=a(wt,"P",{});var $t=i(Ke);lw=s($t,"To behave as an decoder the model needs to be initialized with the "),Ym=a($t,"CODE",{});var D7=i(Ym);dw=s(D7,"is_decoder"),D7.forEach(t),cw=s($t,` argument of the configuration set
to `),Zm=a($t,"CODE",{});var S7=i(Zm);pw=s(S7,"True"),S7.forEach(t),hw=s($t,". To be used in a Seq2Seq model, the model needs to initialized with both "),ef=a($t,"CODE",{});var U7=i(ef);mw=s(U7,"is_decoder"),U7.forEach(t),fw=s($t,` argument and
`),tf=a($t,"CODE",{});var W7=i(tf);uw=s(W7,"add_cross_attention"),W7.forEach(t),gw=s($t," set to "),of=a($t,"CODE",{});var H7=i(of);_w=s(H7,"True"),H7.forEach(t),bw=s($t,"; an "),nf=a($t,"CODE",{});var R7=i(nf);kw=s(R7,"encoder_hidden_states"),R7.forEach(t),Tw=s($t," is then expected as an input to the forward pass."),$t.forEach(t),vw=p(wt),Lt=a(wt,"DIV",{class:!0});var ba=i(Lt);y(zi.$$.fragment,ba),yw=p(ba),an=a(ba,"P",{});var Sh=i(an);ww=s(Sh,"The "),Wp=a(Sh,"A",{href:!0});var V7=i(Wp);$w=s(V7,"BertModel"),V7.forEach(t),xw=s(Sh," forward method, overrides the "),sf=a(Sh,"CODE",{});var Q7=i(sf);Fw=s(Q7,"__call__"),Q7.forEach(t),Bw=s(Sh," special method."),Sh.forEach(t),Ew=p(ba),y(Ms.$$.fragment,ba),Mw=p(ba),y(zs.$$.fragment,ba),ba.forEach(t),wt.forEach(t),Sb=p(o),ln=a(o,"H2",{class:!0});var eT=i(ln);Ps=a(eT,"A",{id:!0,class:!0,href:!0});var K7=i(Ps);rf=a(K7,"SPAN",{});var J7=i(rf);y(Pi.$$.fragment,J7),J7.forEach(t),K7.forEach(t),zw=p(eT),af=a(eT,"SPAN",{});var G7=i(af);Pw=s(G7,"BertForPreTraining"),G7.forEach(t),eT.forEach(t),Ub=p(o),it=a(o,"DIV",{class:!0});var qo=i(it);y(qi.$$.fragment,qo),qw=p(qo),dn=a(qo,"P",{});var Uh=i(dn);jw=s(Uh,"Bert Model with two heads on top as done during the pretraining: a "),lf=a(Uh,"CODE",{});var X7=i(lf);Cw=s(X7,"masked language modeling"),X7.forEach(t),Nw=s(Uh," head and a "),df=a(Uh,"CODE",{});var Y7=i(df);Ow=s(Y7,"next sentence prediction (classification)"),Y7.forEach(t),Iw=s(Uh," head."),Uh.forEach(t),Aw=p(qo),ji=a(qo,"P",{});var tT=i(ji);Lw=s(tT,"This model inherits from "),Hp=a(tT,"A",{href:!0});var Z7=i(Hp);Dw=s(Z7,"PreTrainedModel"),Z7.forEach(t),Sw=s(tT,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),tT.forEach(t),Uw=p(qo),Ci=a(qo,"P",{});var oT=i(Ci);Ww=s(oT,"This model is also a PyTorch "),Ni=a(oT,"A",{href:!0,rel:!0});var ej=i(Ni);Hw=s(ej,"torch.nn.Module"),ej.forEach(t),Rw=s(oT,` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),oT.forEach(t),Vw=p(qo),Dt=a(qo,"DIV",{class:!0});var ka=i(Dt);y(Oi.$$.fragment,ka),Qw=p(ka),cn=a(ka,"P",{});var Wh=i(cn);Kw=s(Wh,"The "),Rp=a(Wh,"A",{href:!0});var tj=i(Rp);Jw=s(tj,"BertForPreTraining"),tj.forEach(t),Gw=s(Wh," forward method, overrides the "),cf=a(Wh,"CODE",{});var oj=i(cf);Xw=s(oj,"__call__"),oj.forEach(t),Yw=s(Wh," special method."),Wh.forEach(t),Zw=p(ka),y(qs.$$.fragment,ka),e2=p(ka),y(js.$$.fragment,ka),ka.forEach(t),qo.forEach(t),Wb=p(o),pn=a(o,"H2",{class:!0});var nT=i(pn);Cs=a(nT,"A",{id:!0,class:!0,href:!0});var nj=i(Cs);pf=a(nj,"SPAN",{});var sj=i(pf);y(Ii.$$.fragment,sj),sj.forEach(t),nj.forEach(t),t2=p(nT),hf=a(nT,"SPAN",{});var rj=i(hf);o2=s(rj,"BertLMHeadModel"),rj.forEach(t),nT.forEach(t),Hb=p(o),lt=a(o,"DIV",{class:!0});var jo=i(lt);y(Ai.$$.fragment,jo),n2=p(jo),Li=a(jo,"P",{});var sT=i(Li);s2=s(sT,"Bert Model with a "),mf=a(sT,"CODE",{});var aj=i(mf);r2=s(aj,"language modeling"),aj.forEach(t),a2=s(sT," head on top for CLM fine-tuning."),sT.forEach(t),i2=p(jo),Di=a(jo,"P",{});var rT=i(Di);l2=s(rT,"This model inherits from "),Vp=a(rT,"A",{href:!0});var ij=i(Vp);d2=s(ij,"PreTrainedModel"),ij.forEach(t),c2=s(rT,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),rT.forEach(t),p2=p(jo),Si=a(jo,"P",{});var aT=i(Si);h2=s(aT,"This model is also a PyTorch "),Ui=a(aT,"A",{href:!0,rel:!0});var lj=i(Ui);m2=s(lj,"torch.nn.Module"),lj.forEach(t),f2=s(aT,` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),aT.forEach(t),u2=p(jo),St=a(jo,"DIV",{class:!0});var Ta=i(St);y(Wi.$$.fragment,Ta),g2=p(Ta),hn=a(Ta,"P",{});var Hh=i(hn);_2=s(Hh,"The "),Qp=a(Hh,"A",{href:!0});var dj=i(Qp);b2=s(dj,"BertLMHeadModel"),dj.forEach(t),k2=s(Hh," forward method, overrides the "),ff=a(Hh,"CODE",{});var cj=i(ff);T2=s(cj,"__call__"),cj.forEach(t),v2=s(Hh," special method."),Hh.forEach(t),y2=p(Ta),y(Ns.$$.fragment,Ta),w2=p(Ta),y(Os.$$.fragment,Ta),Ta.forEach(t),jo.forEach(t),Rb=p(o),mn=a(o,"H2",{class:!0});var iT=i(mn);Is=a(iT,"A",{id:!0,class:!0,href:!0});var pj=i(Is);uf=a(pj,"SPAN",{});var hj=i(uf);y(Hi.$$.fragment,hj),hj.forEach(t),pj.forEach(t),$2=p(iT),gf=a(iT,"SPAN",{});var mj=i(gf);x2=s(mj,"BertForMaskedLM"),mj.forEach(t),iT.forEach(t),Vb=p(o),dt=a(o,"DIV",{class:!0});var Co=i(dt);y(Ri.$$.fragment,Co),F2=p(Co),Vi=a(Co,"P",{});var lT=i(Vi);B2=s(lT,"Bert Model with a "),_f=a(lT,"CODE",{});var fj=i(_f);E2=s(fj,"language modeling"),fj.forEach(t),M2=s(lT," head on top."),lT.forEach(t),z2=p(Co),Qi=a(Co,"P",{});var dT=i(Qi);P2=s(dT,"This model inherits from "),Kp=a(dT,"A",{href:!0});var uj=i(Kp);q2=s(uj,"PreTrainedModel"),uj.forEach(t),j2=s(dT,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),dT.forEach(t),C2=p(Co),Ki=a(Co,"P",{});var cT=i(Ki);N2=s(cT,"This model is also a PyTorch "),Ji=a(cT,"A",{href:!0,rel:!0});var gj=i(Ji);O2=s(gj,"torch.nn.Module"),gj.forEach(t),I2=s(cT,` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),cT.forEach(t),A2=p(Co),ut=a(Co,"DIV",{class:!0});var No=i(ut);y(Gi.$$.fragment,No),L2=p(No),fn=a(No,"P",{});var Rh=i(fn);D2=s(Rh,"The "),Jp=a(Rh,"A",{href:!0});var _j=i(Jp);S2=s(_j,"BertForMaskedLM"),_j.forEach(t),U2=s(Rh," forward method, overrides the "),bf=a(Rh,"CODE",{});var bj=i(bf);W2=s(bj,"__call__"),bj.forEach(t),H2=s(Rh," special method."),Rh.forEach(t),R2=p(No),y(As.$$.fragment,No),V2=p(No),y(Ls.$$.fragment,No),Q2=p(No),y(Ds.$$.fragment,No),No.forEach(t),Co.forEach(t),Qb=p(o),un=a(o,"H2",{class:!0});var pT=i(un);Ss=a(pT,"A",{id:!0,class:!0,href:!0});var kj=i(Ss);kf=a(kj,"SPAN",{});var Tj=i(kf);y(Xi.$$.fragment,Tj),Tj.forEach(t),kj.forEach(t),K2=p(pT),Tf=a(pT,"SPAN",{});var vj=i(Tf);J2=s(vj,"BertForNextSentencePrediction"),vj.forEach(t),pT.forEach(t),Kb=p(o),ct=a(o,"DIV",{class:!0});var Oo=i(ct);y(Yi.$$.fragment,Oo),G2=p(Oo),Zi=a(Oo,"P",{});var hT=i(Zi);X2=s(hT,"Bert Model with a "),vf=a(hT,"CODE",{});var yj=i(vf);Y2=s(yj,"next sentence prediction (classification)"),yj.forEach(t),Z2=s(hT," head on top."),hT.forEach(t),e$=p(Oo),el=a(Oo,"P",{});var mT=i(el);t$=s(mT,"This model inherits from "),Gp=a(mT,"A",{href:!0});var wj=i(Gp);o$=s(wj,"PreTrainedModel"),wj.forEach(t),n$=s(mT,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),mT.forEach(t),s$=p(Oo),tl=a(Oo,"P",{});var fT=i(tl);r$=s(fT,"This model is also a PyTorch "),ol=a(fT,"A",{href:!0,rel:!0});var $j=i(ol);a$=s($j,"torch.nn.Module"),$j.forEach(t),i$=s(fT,` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),fT.forEach(t),l$=p(Oo),Ut=a(Oo,"DIV",{class:!0});var va=i(Ut);y(nl.$$.fragment,va),d$=p(va),gn=a(va,"P",{});var Vh=i(gn);c$=s(Vh,"The "),Xp=a(Vh,"A",{href:!0});var xj=i(Xp);p$=s(xj,"BertForNextSentencePrediction"),xj.forEach(t),h$=s(Vh," forward method, overrides the "),yf=a(Vh,"CODE",{});var Fj=i(yf);m$=s(Fj,"__call__"),Fj.forEach(t),f$=s(Vh," special method."),Vh.forEach(t),u$=p(va),y(Us.$$.fragment,va),g$=p(va),y(Ws.$$.fragment,va),va.forEach(t),Oo.forEach(t),Jb=p(o),_n=a(o,"H2",{class:!0});var uT=i(_n);Hs=a(uT,"A",{id:!0,class:!0,href:!0});var Bj=i(Hs);wf=a(Bj,"SPAN",{});var Ej=i(wf);y(sl.$$.fragment,Ej),Ej.forEach(t),Bj.forEach(t),_$=p(uT),$f=a(uT,"SPAN",{});var Mj=i($f);b$=s(Mj,"BertForSequenceClassification"),Mj.forEach(t),uT.forEach(t),Gb=p(o),pt=a(o,"DIV",{class:!0});var Io=i(pt);y(rl.$$.fragment,Io),k$=p(Io),xf=a(Io,"P",{});var zj=i(xf);T$=s(zj,`Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
output) e.g. for GLUE tasks.`),zj.forEach(t),v$=p(Io),al=a(Io,"P",{});var gT=i(al);y$=s(gT,"This model inherits from "),Yp=a(gT,"A",{href:!0});var Pj=i(Yp);w$=s(Pj,"PreTrainedModel"),Pj.forEach(t),$$=s(gT,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),gT.forEach(t),x$=p(Io),il=a(Io,"P",{});var _T=i(il);F$=s(_T,"This model is also a PyTorch "),ll=a(_T,"A",{href:!0,rel:!0});var qj=i(ll);B$=s(qj,"torch.nn.Module"),qj.forEach(t),E$=s(_T,` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),_T.forEach(t),M$=p(Io),Qe=a(Io,"DIV",{class:!0});var xt=i(Qe);y(dl.$$.fragment,xt),z$=p(xt),bn=a(xt,"P",{});var Qh=i(bn);P$=s(Qh,"The "),Zp=a(Qh,"A",{href:!0});var jj=i(Zp);q$=s(jj,"BertForSequenceClassification"),jj.forEach(t),j$=s(Qh," forward method, overrides the "),Ff=a(Qh,"CODE",{});var Cj=i(Ff);C$=s(Cj,"__call__"),Cj.forEach(t),N$=s(Qh," special method."),Qh.forEach(t),O$=p(xt),y(Rs.$$.fragment,xt),I$=p(xt),y(Vs.$$.fragment,xt),A$=p(xt),y(Qs.$$.fragment,xt),L$=p(xt),y(Ks.$$.fragment,xt),D$=p(xt),y(Js.$$.fragment,xt),xt.forEach(t),Io.forEach(t),Xb=p(o),kn=a(o,"H2",{class:!0});var bT=i(kn);Gs=a(bT,"A",{id:!0,class:!0,href:!0});var Nj=i(Gs);Bf=a(Nj,"SPAN",{});var Oj=i(Bf);y(cl.$$.fragment,Oj),Oj.forEach(t),Nj.forEach(t),S$=p(bT),Ef=a(bT,"SPAN",{});var Ij=i(Ef);U$=s(Ij,"BertForMultipleChoice"),Ij.forEach(t),bT.forEach(t),Yb=p(o),ht=a(o,"DIV",{class:!0});var Ao=i(ht);y(pl.$$.fragment,Ao),W$=p(Ao),Mf=a(Ao,"P",{});var Aj=i(Mf);H$=s(Aj,`Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`),Aj.forEach(t),R$=p(Ao),hl=a(Ao,"P",{});var kT=i(hl);V$=s(kT,"This model inherits from "),eh=a(kT,"A",{href:!0});var Lj=i(eh);Q$=s(Lj,"PreTrainedModel"),Lj.forEach(t),K$=s(kT,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),kT.forEach(t),J$=p(Ao),ml=a(Ao,"P",{});var TT=i(ml);G$=s(TT,"This model is also a PyTorch "),fl=a(TT,"A",{href:!0,rel:!0});var Dj=i(fl);X$=s(Dj,"torch.nn.Module"),Dj.forEach(t),Y$=s(TT,` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),TT.forEach(t),Z$=p(Ao),Wt=a(Ao,"DIV",{class:!0});var ya=i(Wt);y(ul.$$.fragment,ya),ex=p(ya),Tn=a(ya,"P",{});var Kh=i(Tn);tx=s(Kh,"The "),th=a(Kh,"A",{href:!0});var Sj=i(th);ox=s(Sj,"BertForMultipleChoice"),Sj.forEach(t),nx=s(Kh," forward method, overrides the "),zf=a(Kh,"CODE",{});var Uj=i(zf);sx=s(Uj,"__call__"),Uj.forEach(t),rx=s(Kh," special method."),Kh.forEach(t),ax=p(ya),y(Xs.$$.fragment,ya),ix=p(ya),y(Ys.$$.fragment,ya),ya.forEach(t),Ao.forEach(t),Zb=p(o),vn=a(o,"H2",{class:!0});var vT=i(vn);Zs=a(vT,"A",{id:!0,class:!0,href:!0});var Wj=i(Zs);Pf=a(Wj,"SPAN",{});var Hj=i(Pf);y(gl.$$.fragment,Hj),Hj.forEach(t),Wj.forEach(t),lx=p(vT),qf=a(vT,"SPAN",{});var Rj=i(qf);dx=s(Rj,"BertForTokenClassification"),Rj.forEach(t),vT.forEach(t),ek=p(o),mt=a(o,"DIV",{class:!0});var Lo=i(mt);y(_l.$$.fragment,Lo),cx=p(Lo),jf=a(Lo,"P",{});var Vj=i(jf);px=s(Vj,`Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
Named-Entity-Recognition (NER) tasks.`),Vj.forEach(t),hx=p(Lo),bl=a(Lo,"P",{});var yT=i(bl);mx=s(yT,"This model inherits from "),oh=a(yT,"A",{href:!0});var Qj=i(oh);fx=s(Qj,"PreTrainedModel"),Qj.forEach(t),ux=s(yT,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),yT.forEach(t),gx=p(Lo),kl=a(Lo,"P",{});var wT=i(kl);_x=s(wT,"This model is also a PyTorch "),Tl=a(wT,"A",{href:!0,rel:!0});var Kj=i(Tl);bx=s(Kj,"torch.nn.Module"),Kj.forEach(t),kx=s(wT,` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),wT.forEach(t),Tx=p(Lo),gt=a(Lo,"DIV",{class:!0});var Do=i(gt);y(vl.$$.fragment,Do),vx=p(Do),yn=a(Do,"P",{});var Jh=i(yn);yx=s(Jh,"The "),nh=a(Jh,"A",{href:!0});var Jj=i(nh);wx=s(Jj,"BertForTokenClassification"),Jj.forEach(t),$x=s(Jh," forward method, overrides the "),Cf=a(Jh,"CODE",{});var Gj=i(Cf);xx=s(Gj,"__call__"),Gj.forEach(t),Fx=s(Jh," special method."),Jh.forEach(t),Bx=p(Do),y(er.$$.fragment,Do),Ex=p(Do),y(tr.$$.fragment,Do),Mx=p(Do),y(or.$$.fragment,Do),Do.forEach(t),Lo.forEach(t),tk=p(o),wn=a(o,"H2",{class:!0});var $T=i(wn);nr=a($T,"A",{id:!0,class:!0,href:!0});var Xj=i(nr);Nf=a(Xj,"SPAN",{});var Yj=i(Nf);y(yl.$$.fragment,Yj),Yj.forEach(t),Xj.forEach(t),zx=p($T),Of=a($T,"SPAN",{});var Zj=i(Of);Px=s(Zj,"BertForQuestionAnswering"),Zj.forEach(t),$T.forEach(t),ok=p(o),ft=a(o,"DIV",{class:!0});var So=i(ft);y(wl.$$.fragment,So),qx=p(So),$n=a(So,"P",{});var Gh=i($n);jx=s(Gh,`Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
layers on top of the hidden-states output to compute `),If=a(Gh,"CODE",{});var e9=i(If);Cx=s(e9,"span start logits"),e9.forEach(t),Nx=s(Gh," and "),Af=a(Gh,"CODE",{});var t9=i(Af);Ox=s(t9,"span end logits"),t9.forEach(t),Ix=s(Gh,")."),Gh.forEach(t),Ax=p(So),$l=a(So,"P",{});var xT=i($l);Lx=s(xT,"This model inherits from "),sh=a(xT,"A",{href:!0});var o9=i(sh);Dx=s(o9,"PreTrainedModel"),o9.forEach(t),Sx=s(xT,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),xT.forEach(t),Ux=p(So),xl=a(So,"P",{});var FT=i(xl);Wx=s(FT,"This model is also a PyTorch "),Fl=a(FT,"A",{href:!0,rel:!0});var n9=i(Fl);Hx=s(n9,"torch.nn.Module"),n9.forEach(t),Rx=s(FT,` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),FT.forEach(t),Vx=p(So),_t=a(So,"DIV",{class:!0});var Uo=i(_t);y(Bl.$$.fragment,Uo),Qx=p(Uo),xn=a(Uo,"P",{});var Xh=i(xn);Kx=s(Xh,"The "),rh=a(Xh,"A",{href:!0});var s9=i(rh);Jx=s(s9,"BertForQuestionAnswering"),s9.forEach(t),Gx=s(Xh," forward method, overrides the "),Lf=a(Xh,"CODE",{});var r9=i(Lf);Xx=s(r9,"__call__"),r9.forEach(t),Yx=s(Xh," special method."),Xh.forEach(t),Zx=p(Uo),y(sr.$$.fragment,Uo),eF=p(Uo),y(rr.$$.fragment,Uo),tF=p(Uo),y(ar.$$.fragment,Uo),Uo.forEach(t),So.forEach(t),nk=p(o),Fn=a(o,"H2",{class:!0});var BT=i(Fn);ir=a(BT,"A",{id:!0,class:!0,href:!0});var a9=i(ir);Df=a(a9,"SPAN",{});var i9=i(Df);y(El.$$.fragment,i9),i9.forEach(t),a9.forEach(t),oF=p(BT),Sf=a(BT,"SPAN",{});var l9=i(Sf);nF=s(l9,"TFBertModel"),l9.forEach(t),BT.forEach(t),sk=p(o),Je=a(o,"DIV",{class:!0});var no=i(Je);y(Ml.$$.fragment,no),sF=p(no),Uf=a(no,"P",{});var d9=i(Uf);rF=s(d9,"The bare Bert Model transformer outputting raw hidden-states without any specific head on top."),d9.forEach(t),aF=p(no),zl=a(no,"P",{});var ET=i(zl);iF=s(ET,"This model inherits from "),ah=a(ET,"A",{href:!0});var c9=i(ah);lF=s(c9,"TFPreTrainedModel"),c9.forEach(t),dF=s(ET,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),ET.forEach(t),cF=p(no),Pl=a(no,"P",{});var MT=i(Pl);pF=s(MT,"This model is also a "),ql=a(MT,"A",{href:!0,rel:!0});var p9=i(ql);hF=s(p9,"tf.keras.Model"),p9.forEach(t),mF=s(MT,` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),MT.forEach(t),fF=p(no),y(lr.$$.fragment,no),uF=p(no),Ht=a(no,"DIV",{class:!0});var wa=i(Ht);y(jl.$$.fragment,wa),gF=p(wa),Bn=a(wa,"P",{});var Yh=i(Bn);_F=s(Yh,"The "),ih=a(Yh,"A",{href:!0});var h9=i(ih);bF=s(h9,"TFBertModel"),h9.forEach(t),kF=s(Yh," forward method, overrides the "),Wf=a(Yh,"CODE",{});var m9=i(Wf);TF=s(m9,"__call__"),m9.forEach(t),vF=s(Yh," special method."),Yh.forEach(t),yF=p(wa),y(dr.$$.fragment,wa),wF=p(wa),y(cr.$$.fragment,wa),wa.forEach(t),no.forEach(t),rk=p(o),En=a(o,"H2",{class:!0});var zT=i(En);pr=a(zT,"A",{id:!0,class:!0,href:!0});var f9=i(pr);Hf=a(f9,"SPAN",{});var u9=i(Hf);y(Cl.$$.fragment,u9),u9.forEach(t),f9.forEach(t),$F=p(zT),Rf=a(zT,"SPAN",{});var g9=i(Rf);xF=s(g9,"TFBertForPreTraining"),g9.forEach(t),zT.forEach(t),ak=p(o),Ge=a(o,"DIV",{class:!0});var so=i(Ge);y(Nl.$$.fragment,so),FF=p(so),Mn=a(so,"P",{});var Zh=i(Mn);BF=s(Zh,`Bert Model with two heads on top as done during the pretraining:
a `),Vf=a(Zh,"CODE",{});var _9=i(Vf);EF=s(_9,"masked language modeling"),_9.forEach(t),MF=s(Zh," head and a "),Qf=a(Zh,"CODE",{});var b9=i(Qf);zF=s(b9,"next sentence prediction (classification)"),b9.forEach(t),PF=s(Zh," head."),Zh.forEach(t),qF=p(so),Ol=a(so,"P",{});var PT=i(Ol);jF=s(PT,"This model inherits from "),lh=a(PT,"A",{href:!0});var k9=i(lh);CF=s(k9,"TFPreTrainedModel"),k9.forEach(t),NF=s(PT,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),PT.forEach(t),OF=p(so),Il=a(so,"P",{});var qT=i(Il);IF=s(qT,"This model is also a "),Al=a(qT,"A",{href:!0,rel:!0});var T9=i(Al);AF=s(T9,"tf.keras.Model"),T9.forEach(t),LF=s(qT,` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),qT.forEach(t),DF=p(so),y(hr.$$.fragment,so),SF=p(so),Rt=a(so,"DIV",{class:!0});var $a=i(Rt);y(Ll.$$.fragment,$a),UF=p($a),zn=a($a,"P",{});var em=i(zn);WF=s(em,"The "),dh=a(em,"A",{href:!0});var v9=i(dh);HF=s(v9,"TFBertForPreTraining"),v9.forEach(t),RF=s(em," forward method, overrides the "),Kf=a(em,"CODE",{});var y9=i(Kf);VF=s(y9,"__call__"),y9.forEach(t),QF=s(em," special method."),em.forEach(t),KF=p($a),y(mr.$$.fragment,$a),JF=p($a),y(fr.$$.fragment,$a),$a.forEach(t),so.forEach(t),ik=p(o),Pn=a(o,"H2",{class:!0});var jT=i(Pn);ur=a(jT,"A",{id:!0,class:!0,href:!0});var w9=i(ur);Jf=a(w9,"SPAN",{});var $9=i(Jf);y(Dl.$$.fragment,$9),$9.forEach(t),w9.forEach(t),GF=p(jT),Gf=a(jT,"SPAN",{});var x9=i(Gf);XF=s(x9,"TFBertModelLMHeadModel"),x9.forEach(t),jT.forEach(t),lk=p(o),qn=a(o,"DIV",{class:!0});var CT=i(qn);y(Sl.$$.fragment,CT),YF=p(CT),bt=a(CT,"DIV",{class:!0});var Wo=i(bt);y(Ul.$$.fragment,Wo),ZF=p(Wo),Ie=a(Wo,"P",{});var st=i(Ie);e0=s(st,"encoder_hidden_states  ("),Xf=a(st,"CODE",{});var F9=i(Xf);t0=s(F9,"tf.Tensor"),F9.forEach(t),o0=s(st," of shape "),Yf=a(st,"CODE",{});var B9=i(Yf);n0=s(B9,"(batch_size, sequence_length, hidden_size)"),B9.forEach(t),s0=s(st,", "),Zf=a(st,"EM",{});var E9=i(Zf);r0=s(E9,"optional"),E9.forEach(t),a0=s(st,`):
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
the model is configured as a decoder.
encoder_attention_mask (`),eu=a(st,"CODE",{});var M9=i(eu);i0=s(M9,"tf.Tensor"),M9.forEach(t),l0=s(st," of shape "),tu=a(st,"CODE",{});var z9=i(tu);d0=s(z9,"(batch_size, sequence_length)"),z9.forEach(t),c0=s(st,", "),ou=a(st,"EM",{});var P9=i(ou);p0=s(P9,"optional"),P9.forEach(t),h0=s(st,`):
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in `),nu=a(st,"CODE",{});var q9=i(nu);m0=s(q9,"[0, 1]"),q9.forEach(t),f0=s(st,":"),st.forEach(t),u0=p(Wo),Wl=a(Wo,"UL",{});var NT=i(Wl);Hl=a(NT,"LI",{});var OT=i(Hl);g0=s(OT,"1 for tokens that are "),su=a(OT,"STRONG",{});var j9=i(su);_0=s(j9,"not masked"),j9.forEach(t),b0=s(OT,","),OT.forEach(t),k0=p(NT),Rl=a(NT,"LI",{});var IT=i(Rl);T0=s(IT,"0 for tokens that are "),ru=a(IT,"STRONG",{});var C9=i(ru);v0=s(C9,"masked"),C9.forEach(t),y0=s(IT,"."),IT.forEach(t),NT.forEach(t),w0=p(Wo),G=a(Wo,"P",{});var Z=i(G);$0=s(Z,"past_key_values ("),au=a(Z,"CODE",{});var N9=i(au);x0=s(N9,"Tuple[Tuple[tf.Tensor]]"),N9.forEach(t),F0=s(Z," of length "),iu=a(Z,"CODE",{});var O9=i(iu);B0=s(O9,"config.n_layers"),O9.forEach(t),E0=s(Z,`)
contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
If `),lu=a(Z,"CODE",{});var I9=i(lu);M0=s(I9,"past_key_values"),I9.forEach(t),z0=s(Z," are used, the user can optionally input only the last "),du=a(Z,"CODE",{});var A9=i(du);P0=s(A9,"decoder_input_ids"),A9.forEach(t),q0=s(Z,` (those that
don\u2019t have their past key value states given to this model) of shape `),cu=a(Z,"CODE",{});var L9=i(cu);j0=s(L9,"(batch_size, 1)"),L9.forEach(t),C0=s(Z,` instead of all
`),pu=a(Z,"CODE",{});var D9=i(pu);N0=s(D9,"decoder_input_ids"),D9.forEach(t),O0=s(Z," of shape "),hu=a(Z,"CODE",{});var S9=i(hu);I0=s(S9,"(batch_size, sequence_length)"),S9.forEach(t),A0=s(Z,`.
use_cache (`),mu=a(Z,"CODE",{});var U9=i(mu);L0=s(U9,"bool"),U9.forEach(t),D0=s(Z,", "),fu=a(Z,"EM",{});var W9=i(fu);S0=s(W9,"optional"),W9.forEach(t),U0=s(Z,", defaults to "),uu=a(Z,"CODE",{});var H9=i(uu);W0=s(H9,"True"),H9.forEach(t),H0=s(Z,`):
If set to `),gu=a(Z,"CODE",{});var R9=i(gu);R0=s(R9,"True"),R9.forEach(t),V0=s(Z,", "),_u=a(Z,"CODE",{});var V9=i(_u);Q0=s(V9,"past_key_values"),V9.forEach(t),K0=s(Z,` key value states are returned and can be used to speed up decoding (see
`),bu=a(Z,"CODE",{});var Q9=i(bu);J0=s(Q9,"past_key_values"),Q9.forEach(t),G0=s(Z,"). Set to "),ku=a(Z,"CODE",{});var K9=i(ku);X0=s(K9,"False"),K9.forEach(t),Y0=s(Z," during training, "),Tu=a(Z,"CODE",{});var J9=i(Tu);Z0=s(J9,"True"),J9.forEach(t),eB=s(Z,` during generation
labels (`),vu=a(Z,"CODE",{});var G9=i(vu);tB=s(G9,"tf.Tensor"),G9.forEach(t),oB=s(Z," or "),yu=a(Z,"CODE",{});var X9=i(yu);nB=s(X9,"np.ndarray"),X9.forEach(t),sB=s(Z," of shape "),wu=a(Z,"CODE",{});var Y9=i(wu);rB=s(Y9,"(batch_size, sequence_length)"),Y9.forEach(t),aB=s(Z,", "),$u=a(Z,"EM",{});var Z9=i($u);iB=s(Z9,"optional"),Z9.forEach(t),lB=s(Z,`):
Labels for computing the cross entropy classification loss. Indices should be in `),xu=a(Z,"CODE",{});var eC=i(xu);dB=s(eC,"[0, ..., config.vocab_size - 1]"),eC.forEach(t),cB=s(Z,"."),Z.forEach(t),pB=p(Wo),y(gr.$$.fragment,Wo),Wo.forEach(t),CT.forEach(t),dk=p(o),jn=a(o,"H2",{class:!0});var AT=i(jn);_r=a(AT,"A",{id:!0,class:!0,href:!0});var tC=i(_r);Fu=a(tC,"SPAN",{});var oC=i(Fu);y(Vl.$$.fragment,oC),oC.forEach(t),tC.forEach(t),hB=p(AT),Bu=a(AT,"SPAN",{});var nC=i(Bu);mB=s(nC,"TFBertForMaskedLM"),nC.forEach(t),AT.forEach(t),ck=p(o),Xe=a(o,"DIV",{class:!0});var ro=i(Xe);y(Ql.$$.fragment,ro),fB=p(ro),Kl=a(ro,"P",{});var LT=i(Kl);uB=s(LT,"Bert Model with a "),Eu=a(LT,"CODE",{});var sC=i(Eu);gB=s(sC,"language modeling"),sC.forEach(t),_B=s(LT," head on top."),LT.forEach(t),bB=p(ro),Jl=a(ro,"P",{});var DT=i(Jl);kB=s(DT,"This model inherits from "),ch=a(DT,"A",{href:!0});var rC=i(ch);TB=s(rC,"TFPreTrainedModel"),rC.forEach(t),vB=s(DT,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),DT.forEach(t),yB=p(ro),Gl=a(ro,"P",{});var ST=i(Gl);wB=s(ST,"This model is also a "),Xl=a(ST,"A",{href:!0,rel:!0});var aC=i(Xl);$B=s(aC,"tf.keras.Model"),aC.forEach(t),xB=s(ST,` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),ST.forEach(t),FB=p(ro),y(br.$$.fragment,ro),BB=p(ro),kt=a(ro,"DIV",{class:!0});var Ho=i(kt);y(Yl.$$.fragment,Ho),EB=p(Ho),Cn=a(Ho,"P",{});var tm=i(Cn);MB=s(tm,"The "),ph=a(tm,"A",{href:!0});var iC=i(ph);zB=s(iC,"TFBertForMaskedLM"),iC.forEach(t),PB=s(tm," forward method, overrides the "),Mu=a(tm,"CODE",{});var lC=i(Mu);qB=s(lC,"__call__"),lC.forEach(t),jB=s(tm," special method."),tm.forEach(t),CB=p(Ho),y(kr.$$.fragment,Ho),NB=p(Ho),y(Tr.$$.fragment,Ho),OB=p(Ho),y(vr.$$.fragment,Ho),Ho.forEach(t),ro.forEach(t),pk=p(o),Nn=a(o,"H2",{class:!0});var UT=i(Nn);yr=a(UT,"A",{id:!0,class:!0,href:!0});var dC=i(yr);zu=a(dC,"SPAN",{});var cC=i(zu);y(Zl.$$.fragment,cC),cC.forEach(t),dC.forEach(t),IB=p(UT),Pu=a(UT,"SPAN",{});var pC=i(Pu);AB=s(pC,"TFBertForNextSentencePrediction"),pC.forEach(t),UT.forEach(t),hk=p(o),Ye=a(o,"DIV",{class:!0});var ao=i(Ye);y(ed.$$.fragment,ao),LB=p(ao),td=a(ao,"P",{});var WT=i(td);DB=s(WT,"Bert Model with a "),qu=a(WT,"CODE",{});var hC=i(qu);SB=s(hC,"next sentence prediction (classification)"),hC.forEach(t),UB=s(WT," head on top."),WT.forEach(t),WB=p(ao),od=a(ao,"P",{});var HT=i(od);HB=s(HT,"This model inherits from "),hh=a(HT,"A",{href:!0});var mC=i(hh);RB=s(mC,"TFPreTrainedModel"),mC.forEach(t),VB=s(HT,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),HT.forEach(t),QB=p(ao),nd=a(ao,"P",{});var RT=i(nd);KB=s(RT,"This model is also a "),sd=a(RT,"A",{href:!0,rel:!0});var fC=i(sd);JB=s(fC,"tf.keras.Model"),fC.forEach(t),GB=s(RT,` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),RT.forEach(t),XB=p(ao),y(wr.$$.fragment,ao),YB=p(ao),Vt=a(ao,"DIV",{class:!0});var xa=i(Vt);y(rd.$$.fragment,xa),ZB=p(xa),On=a(xa,"P",{});var om=i(On);eE=s(om,"The "),mh=a(om,"A",{href:!0});var uC=i(mh);tE=s(uC,"TFBertForNextSentencePrediction"),uC.forEach(t),oE=s(om," forward method, overrides the "),ju=a(om,"CODE",{});var gC=i(ju);nE=s(gC,"__call__"),gC.forEach(t),sE=s(om," special method."),om.forEach(t),rE=p(xa),y($r.$$.fragment,xa),aE=p(xa),y(xr.$$.fragment,xa),xa.forEach(t),ao.forEach(t),mk=p(o),In=a(o,"H2",{class:!0});var VT=i(In);Fr=a(VT,"A",{id:!0,class:!0,href:!0});var _C=i(Fr);Cu=a(_C,"SPAN",{});var bC=i(Cu);y(ad.$$.fragment,bC),bC.forEach(t),_C.forEach(t),iE=p(VT),Nu=a(VT,"SPAN",{});var kC=i(Nu);lE=s(kC,"TFBertForSequenceClassification"),kC.forEach(t),VT.forEach(t),fk=p(o),Ze=a(o,"DIV",{class:!0});var io=i(Ze);y(id.$$.fragment,io),dE=p(io),Ou=a(io,"P",{});var TC=i(Ou);cE=s(TC,`Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
output) e.g. for GLUE tasks.`),TC.forEach(t),pE=p(io),ld=a(io,"P",{});var QT=i(ld);hE=s(QT,"This model inherits from "),fh=a(QT,"A",{href:!0});var vC=i(fh);mE=s(vC,"TFPreTrainedModel"),vC.forEach(t),fE=s(QT,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),QT.forEach(t),uE=p(io),dd=a(io,"P",{});var KT=i(dd);gE=s(KT,"This model is also a "),cd=a(KT,"A",{href:!0,rel:!0});var yC=i(cd);_E=s(yC,"tf.keras.Model"),yC.forEach(t),bE=s(KT,` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),KT.forEach(t),kE=p(io),y(Br.$$.fragment,io),TE=p(io),Tt=a(io,"DIV",{class:!0});var Ro=i(Tt);y(pd.$$.fragment,Ro),vE=p(Ro),An=a(Ro,"P",{});var nm=i(An);yE=s(nm,"The "),uh=a(nm,"A",{href:!0});var wC=i(uh);wE=s(wC,"TFBertForSequenceClassification"),wC.forEach(t),$E=s(nm," forward method, overrides the "),Iu=a(nm,"CODE",{});var $C=i(Iu);xE=s($C,"__call__"),$C.forEach(t),FE=s(nm," special method."),nm.forEach(t),BE=p(Ro),y(Er.$$.fragment,Ro),EE=p(Ro),y(Mr.$$.fragment,Ro),ME=p(Ro),y(zr.$$.fragment,Ro),Ro.forEach(t),io.forEach(t),uk=p(o),Ln=a(o,"H2",{class:!0});var JT=i(Ln);Pr=a(JT,"A",{id:!0,class:!0,href:!0});var xC=i(Pr);Au=a(xC,"SPAN",{});var FC=i(Au);y(hd.$$.fragment,FC),FC.forEach(t),xC.forEach(t),zE=p(JT),Lu=a(JT,"SPAN",{});var BC=i(Lu);PE=s(BC,"TFBertForMultipleChoice"),BC.forEach(t),JT.forEach(t),gk=p(o),et=a(o,"DIV",{class:!0});var lo=i(et);y(md.$$.fragment,lo),qE=p(lo),Du=a(lo,"P",{});var EC=i(Du);jE=s(EC,`Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`),EC.forEach(t),CE=p(lo),fd=a(lo,"P",{});var GT=i(fd);NE=s(GT,"This model inherits from "),gh=a(GT,"A",{href:!0});var MC=i(gh);OE=s(MC,"TFPreTrainedModel"),MC.forEach(t),IE=s(GT,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),GT.forEach(t),AE=p(lo),ud=a(lo,"P",{});var XT=i(ud);LE=s(XT,"This model is also a "),gd=a(XT,"A",{href:!0,rel:!0});var zC=i(gd);DE=s(zC,"tf.keras.Model"),zC.forEach(t),SE=s(XT,` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),XT.forEach(t),UE=p(lo),y(qr.$$.fragment,lo),WE=p(lo),Qt=a(lo,"DIV",{class:!0});var Fa=i(Qt);y(_d.$$.fragment,Fa),HE=p(Fa),Dn=a(Fa,"P",{});var sm=i(Dn);RE=s(sm,"The "),_h=a(sm,"A",{href:!0});var PC=i(_h);VE=s(PC,"TFBertForMultipleChoice"),PC.forEach(t),QE=s(sm," forward method, overrides the "),Su=a(sm,"CODE",{});var qC=i(Su);KE=s(qC,"__call__"),qC.forEach(t),JE=s(sm," special method."),sm.forEach(t),GE=p(Fa),y(jr.$$.fragment,Fa),XE=p(Fa),y(Cr.$$.fragment,Fa),Fa.forEach(t),lo.forEach(t),_k=p(o),Sn=a(o,"H2",{class:!0});var YT=i(Sn);Nr=a(YT,"A",{id:!0,class:!0,href:!0});var jC=i(Nr);Uu=a(jC,"SPAN",{});var CC=i(Uu);y(bd.$$.fragment,CC),CC.forEach(t),jC.forEach(t),YE=p(YT),Wu=a(YT,"SPAN",{});var NC=i(Wu);ZE=s(NC,"TFBertForTokenClassification"),NC.forEach(t),YT.forEach(t),bk=p(o),tt=a(o,"DIV",{class:!0});var co=i(tt);y(kd.$$.fragment,co),eM=p(co),Hu=a(co,"P",{});var OC=i(Hu);tM=s(OC,`Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
Named-Entity-Recognition (NER) tasks.`),OC.forEach(t),oM=p(co),Td=a(co,"P",{});var ZT=i(Td);nM=s(ZT,"This model inherits from "),bh=a(ZT,"A",{href:!0});var IC=i(bh);sM=s(IC,"TFPreTrainedModel"),IC.forEach(t),rM=s(ZT,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),ZT.forEach(t),aM=p(co),vd=a(co,"P",{});var ev=i(vd);iM=s(ev,"This model is also a "),yd=a(ev,"A",{href:!0,rel:!0});var AC=i(yd);lM=s(AC,"tf.keras.Model"),AC.forEach(t),dM=s(ev,` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),ev.forEach(t),cM=p(co),y(Or.$$.fragment,co),pM=p(co),vt=a(co,"DIV",{class:!0});var Vo=i(vt);y(wd.$$.fragment,Vo),hM=p(Vo),Un=a(Vo,"P",{});var rm=i(Un);mM=s(rm,"The "),kh=a(rm,"A",{href:!0});var LC=i(kh);fM=s(LC,"TFBertForTokenClassification"),LC.forEach(t),uM=s(rm," forward method, overrides the "),Ru=a(rm,"CODE",{});var DC=i(Ru);gM=s(DC,"__call__"),DC.forEach(t),_M=s(rm," special method."),rm.forEach(t),bM=p(Vo),y(Ir.$$.fragment,Vo),kM=p(Vo),y(Ar.$$.fragment,Vo),TM=p(Vo),y(Lr.$$.fragment,Vo),Vo.forEach(t),co.forEach(t),kk=p(o),Wn=a(o,"H2",{class:!0});var tv=i(Wn);Dr=a(tv,"A",{id:!0,class:!0,href:!0});var SC=i(Dr);Vu=a(SC,"SPAN",{});var UC=i(Vu);y($d.$$.fragment,UC),UC.forEach(t),SC.forEach(t),vM=p(tv),Qu=a(tv,"SPAN",{});var WC=i(Qu);yM=s(WC,"TFBertForQuestionAnswering"),WC.forEach(t),tv.forEach(t),Tk=p(o),ot=a(o,"DIV",{class:!0});var po=i(ot);y(xd.$$.fragment,po),wM=p(po),Hn=a(po,"P",{});var am=i(Hn);$M=s(am,`Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
layer on top of the hidden-states output to compute `),Ku=a(am,"CODE",{});var HC=i(Ku);xM=s(HC,"span start logits"),HC.forEach(t),FM=s(am," and "),Ju=a(am,"CODE",{});var RC=i(Ju);BM=s(RC,"span end logits"),RC.forEach(t),EM=s(am,")."),am.forEach(t),MM=p(po),Fd=a(po,"P",{});var ov=i(Fd);zM=s(ov,"This model inherits from "),Th=a(ov,"A",{href:!0});var VC=i(Th);PM=s(VC,"TFPreTrainedModel"),VC.forEach(t),qM=s(ov,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),ov.forEach(t),jM=p(po),Bd=a(po,"P",{});var nv=i(Bd);CM=s(nv,"This model is also a "),Ed=a(nv,"A",{href:!0,rel:!0});var QC=i(Ed);NM=s(QC,"tf.keras.Model"),QC.forEach(t),OM=s(nv,` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),nv.forEach(t),IM=p(po),y(Sr.$$.fragment,po),AM=p(po),yt=a(po,"DIV",{class:!0});var Qo=i(yt);y(Md.$$.fragment,Qo),LM=p(Qo),Rn=a(Qo,"P",{});var im=i(Rn);DM=s(im,"The "),vh=a(im,"A",{href:!0});var KC=i(vh);SM=s(KC,"TFBertForQuestionAnswering"),KC.forEach(t),UM=s(im," forward method, overrides the "),Gu=a(im,"CODE",{});var JC=i(Gu);WM=s(JC,"__call__"),JC.forEach(t),HM=s(im," special method."),im.forEach(t),RM=p(Qo),y(Ur.$$.fragment,Qo),VM=p(Qo),y(Wr.$$.fragment,Qo),QM=p(Qo),y(Hr.$$.fragment,Qo),Qo.forEach(t),po.forEach(t),vk=p(o),Vn=a(o,"H2",{class:!0});var sv=i(Vn);Rr=a(sv,"A",{id:!0,class:!0,href:!0});var GC=i(Rr);Xu=a(GC,"SPAN",{});var XC=i(Xu);y(zd.$$.fragment,XC),XC.forEach(t),GC.forEach(t),KM=p(sv),Yu=a(sv,"SPAN",{});var YC=i(Yu);JM=s(YC,"FlaxBertModel"),YC.forEach(t),sv.forEach(t),yk=p(o),Ae=a(o,"DIV",{class:!0});var Ft=i(Ae);y(Pd.$$.fragment,Ft),GM=p(Ft),Zu=a(Ft,"P",{});var ZC=i(Zu);XM=s(ZC,"The bare Bert Model transformer outputting raw hidden-states without any specific head on top."),ZC.forEach(t),YM=p(Ft),qd=a(Ft,"P",{});var rv=i(qd);ZM=s(rv,"This model inherits from "),yh=a(rv,"A",{href:!0});var e3=i(yh);ez=s(e3,"FlaxPreTrainedModel"),e3.forEach(t),tz=s(rv,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading, saving and converting weights from PyTorch models)`),rv.forEach(t),oz=p(Ft),jd=a(Ft,"P",{});var av=i(jd);nz=s(av,"This model is also a Flax Linen "),Cd=a(av,"A",{href:!0,rel:!0});var t3=i(Cd);sz=s(t3,"flax.linen.Module"),t3.forEach(t),rz=s(av,`
subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
general usage and behavior.`),av.forEach(t),az=p(Ft),eg=a(Ft,"P",{});var o3=i(eg);iz=s(o3,"Finally, this model supports inherent JAX features such as:"),o3.forEach(t),lz=p(Ft),fo=a(Ft,"UL",{});var Ba=i(fo);tg=a(Ba,"LI",{});var n3=i(tg);Nd=a(n3,"A",{href:!0,rel:!0});var s3=i(Nd);dz=s(s3,"Just-In-Time (JIT) compilation"),s3.forEach(t),n3.forEach(t),cz=p(Ba),og=a(Ba,"LI",{});var r3=i(og);Od=a(r3,"A",{href:!0,rel:!0});var a3=i(Od);pz=s(a3,"Automatic Differentiation"),a3.forEach(t),r3.forEach(t),hz=p(Ba),ng=a(Ba,"LI",{});var i3=i(ng);Id=a(i3,"A",{href:!0,rel:!0});var l3=i(Id);mz=s(l3,"Vectorization"),l3.forEach(t),i3.forEach(t),fz=p(Ba),sg=a(Ba,"LI",{});var d3=i(sg);Ad=a(d3,"A",{href:!0,rel:!0});var c3=i(Ad);uz=s(c3,"Parallelization"),c3.forEach(t),d3.forEach(t),Ba.forEach(t),gz=p(Ft),Kt=a(Ft,"DIV",{class:!0});var Ea=i(Kt);y(Ld.$$.fragment,Ea),_z=p(Ea),Qn=a(Ea,"P",{});var lm=i(Qn);bz=s(lm,"The "),rg=a(lm,"CODE",{});var p3=i(rg);kz=s(p3,"FlaxBertPreTrainedModel"),p3.forEach(t),Tz=s(lm," forward method, overrides the "),ag=a(lm,"CODE",{});var h3=i(ag);vz=s(h3,"__call__"),h3.forEach(t),yz=s(lm," special method."),lm.forEach(t),wz=p(Ea),y(Vr.$$.fragment,Ea),$z=p(Ea),y(Qr.$$.fragment,Ea),Ea.forEach(t),Ft.forEach(t),wk=p(o),Kn=a(o,"H2",{class:!0});var iv=i(Kn);Kr=a(iv,"A",{id:!0,class:!0,href:!0});var m3=i(Kr);ig=a(m3,"SPAN",{});var f3=i(ig);y(Dd.$$.fragment,f3),f3.forEach(t),m3.forEach(t),xz=p(iv),lg=a(iv,"SPAN",{});var u3=i(lg);Fz=s(u3,"FlaxBertForPreTraining"),u3.forEach(t),iv.forEach(t),$k=p(o),Le=a(o,"DIV",{class:!0});var Bt=i(Le);y(Sd.$$.fragment,Bt),Bz=p(Bt),Jn=a(Bt,"P",{});var dm=i(Jn);Ez=s(dm,"Bert Model with two heads on top as done during the pretraining: a "),dg=a(dm,"CODE",{});var g3=i(dg);Mz=s(g3,"masked language modeling"),g3.forEach(t),zz=s(dm," head and a "),cg=a(dm,"CODE",{});var _3=i(cg);Pz=s(_3,"next sentence prediction (classification)"),_3.forEach(t),qz=s(dm," head."),dm.forEach(t),jz=p(Bt),Ud=a(Bt,"P",{});var lv=i(Ud);Cz=s(lv,"This model inherits from "),wh=a(lv,"A",{href:!0});var b3=i(wh);Nz=s(b3,"FlaxPreTrainedModel"),b3.forEach(t),Oz=s(lv,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading, saving and converting weights from PyTorch models)`),lv.forEach(t),Iz=p(Bt),Wd=a(Bt,"P",{});var dv=i(Wd);Az=s(dv,"This model is also a Flax Linen "),Hd=a(dv,"A",{href:!0,rel:!0});var k3=i(Hd);Lz=s(k3,"flax.linen.Module"),k3.forEach(t),Dz=s(dv,`
subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
general usage and behavior.`),dv.forEach(t),Sz=p(Bt),pg=a(Bt,"P",{});var T3=i(pg);Uz=s(T3,"Finally, this model supports inherent JAX features such as:"),T3.forEach(t),Wz=p(Bt),uo=a(Bt,"UL",{});var Ma=i(uo);hg=a(Ma,"LI",{});var v3=i(hg);Rd=a(v3,"A",{href:!0,rel:!0});var y3=i(Rd);Hz=s(y3,"Just-In-Time (JIT) compilation"),y3.forEach(t),v3.forEach(t),Rz=p(Ma),mg=a(Ma,"LI",{});var w3=i(mg);Vd=a(w3,"A",{href:!0,rel:!0});var $3=i(Vd);Vz=s($3,"Automatic Differentiation"),$3.forEach(t),w3.forEach(t),Qz=p(Ma),fg=a(Ma,"LI",{});var x3=i(fg);Qd=a(x3,"A",{href:!0,rel:!0});var F3=i(Qd);Kz=s(F3,"Vectorization"),F3.forEach(t),x3.forEach(t),Jz=p(Ma),ug=a(Ma,"LI",{});var B3=i(ug);Kd=a(B3,"A",{href:!0,rel:!0});var E3=i(Kd);Gz=s(E3,"Parallelization"),E3.forEach(t),B3.forEach(t),Ma.forEach(t),Xz=p(Bt),Jt=a(Bt,"DIV",{class:!0});var za=i(Jt);y(Jd.$$.fragment,za),Yz=p(za),Gn=a(za,"P",{});var cm=i(Gn);Zz=s(cm,"The "),gg=a(cm,"CODE",{});var M3=i(gg);e4=s(M3,"FlaxBertPreTrainedModel"),M3.forEach(t),t4=s(cm," forward method, overrides the "),_g=a(cm,"CODE",{});var z3=i(_g);o4=s(z3,"__call__"),z3.forEach(t),n4=s(cm," special method."),cm.forEach(t),s4=p(za),y(Jr.$$.fragment,za),r4=p(za),y(Gr.$$.fragment,za),za.forEach(t),Bt.forEach(t),xk=p(o),Xn=a(o,"H2",{class:!0});var cv=i(Xn);Xr=a(cv,"A",{id:!0,class:!0,href:!0});var P3=i(Xr);bg=a(P3,"SPAN",{});var q3=i(bg);y(Gd.$$.fragment,q3),q3.forEach(t),P3.forEach(t),a4=p(cv),kg=a(cv,"SPAN",{});var j3=i(kg);i4=s(j3,"FlaxBertForCausalLM"),j3.forEach(t),cv.forEach(t),Fk=p(o),De=a(o,"DIV",{class:!0});var Et=i(De);y(Xd.$$.fragment,Et),l4=p(Et),Tg=a(Et,"P",{});var C3=i(Tg);d4=s(C3,`Bert Model with a language modeling head on top (a linear layer on top of the hidden-states output) e.g for
autoregressive tasks.`),C3.forEach(t),c4=p(Et),Yd=a(Et,"P",{});var pv=i(Yd);p4=s(pv,"This model inherits from "),$h=a(pv,"A",{href:!0});var N3=i($h);h4=s(N3,"FlaxPreTrainedModel"),N3.forEach(t),m4=s(pv,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading, saving and converting weights from PyTorch models)`),pv.forEach(t),f4=p(Et),Zd=a(Et,"P",{});var hv=i(Zd);u4=s(hv,"This model is also a Flax Linen "),ec=a(hv,"A",{href:!0,rel:!0});var O3=i(ec);g4=s(O3,"flax.linen.Module"),O3.forEach(t),_4=s(hv,`
subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
general usage and behavior.`),hv.forEach(t),b4=p(Et),vg=a(Et,"P",{});var I3=i(vg);k4=s(I3,"Finally, this model supports inherent JAX features such as:"),I3.forEach(t),T4=p(Et),go=a(Et,"UL",{});var Pa=i(go);yg=a(Pa,"LI",{});var A3=i(yg);tc=a(A3,"A",{href:!0,rel:!0});var L3=i(tc);v4=s(L3,"Just-In-Time (JIT) compilation"),L3.forEach(t),A3.forEach(t),y4=p(Pa),wg=a(Pa,"LI",{});var D3=i(wg);oc=a(D3,"A",{href:!0,rel:!0});var S3=i(oc);w4=s(S3,"Automatic Differentiation"),S3.forEach(t),D3.forEach(t),$4=p(Pa),$g=a(Pa,"LI",{});var U3=i($g);nc=a(U3,"A",{href:!0,rel:!0});var W3=i(nc);x4=s(W3,"Vectorization"),W3.forEach(t),U3.forEach(t),F4=p(Pa),xg=a(Pa,"LI",{});var H3=i(xg);sc=a(H3,"A",{href:!0,rel:!0});var R3=i(sc);B4=s(R3,"Parallelization"),R3.forEach(t),H3.forEach(t),Pa.forEach(t),E4=p(Et),Gt=a(Et,"DIV",{class:!0});var qa=i(Gt);y(rc.$$.fragment,qa),M4=p(qa),Yn=a(qa,"P",{});var pm=i(Yn);z4=s(pm,"The "),Fg=a(pm,"CODE",{});var V3=i(Fg);P4=s(V3,"FlaxBertPreTrainedModel"),V3.forEach(t),q4=s(pm," forward method, overrides the "),Bg=a(pm,"CODE",{});var Q3=i(Bg);j4=s(Q3,"__call__"),Q3.forEach(t),C4=s(pm," special method."),pm.forEach(t),N4=p(qa),y(Yr.$$.fragment,qa),O4=p(qa),y(Zr.$$.fragment,qa),qa.forEach(t),Et.forEach(t),Bk=p(o),Zn=a(o,"H2",{class:!0});var mv=i(Zn);ea=a(mv,"A",{id:!0,class:!0,href:!0});var K3=i(ea);Eg=a(K3,"SPAN",{});var J3=i(Eg);y(ac.$$.fragment,J3),J3.forEach(t),K3.forEach(t),I4=p(mv),Mg=a(mv,"SPAN",{});var G3=i(Mg);A4=s(G3,"FlaxBertForMaskedLM"),G3.forEach(t),mv.forEach(t),Ek=p(o),Se=a(o,"DIV",{class:!0});var Mt=i(Se);y(ic.$$.fragment,Mt),L4=p(Mt),lc=a(Mt,"P",{});var fv=i(lc);D4=s(fv,"Bert Model with a "),zg=a(fv,"CODE",{});var X3=i(zg);S4=s(X3,"language modeling"),X3.forEach(t),U4=s(fv," head on top."),fv.forEach(t),W4=p(Mt),dc=a(Mt,"P",{});var uv=i(dc);H4=s(uv,"This model inherits from "),xh=a(uv,"A",{href:!0});var Y3=i(xh);R4=s(Y3,"FlaxPreTrainedModel"),Y3.forEach(t),V4=s(uv,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading, saving and converting weights from PyTorch models)`),uv.forEach(t),Q4=p(Mt),cc=a(Mt,"P",{});var gv=i(cc);K4=s(gv,"This model is also a Flax Linen "),pc=a(gv,"A",{href:!0,rel:!0});var Z3=i(pc);J4=s(Z3,"flax.linen.Module"),Z3.forEach(t),G4=s(gv,`
subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
general usage and behavior.`),gv.forEach(t),X4=p(Mt),Pg=a(Mt,"P",{});var e5=i(Pg);Y4=s(e5,"Finally, this model supports inherent JAX features such as:"),e5.forEach(t),Z4=p(Mt),_o=a(Mt,"UL",{});var ja=i(_o);qg=a(ja,"LI",{});var t5=i(qg);hc=a(t5,"A",{href:!0,rel:!0});var o5=i(hc);eP=s(o5,"Just-In-Time (JIT) compilation"),o5.forEach(t),t5.forEach(t),tP=p(ja),jg=a(ja,"LI",{});var n5=i(jg);mc=a(n5,"A",{href:!0,rel:!0});var s5=i(mc);oP=s(s5,"Automatic Differentiation"),s5.forEach(t),n5.forEach(t),nP=p(ja),Cg=a(ja,"LI",{});var r5=i(Cg);fc=a(r5,"A",{href:!0,rel:!0});var a5=i(fc);sP=s(a5,"Vectorization"),a5.forEach(t),r5.forEach(t),rP=p(ja),Ng=a(ja,"LI",{});var i5=i(Ng);uc=a(i5,"A",{href:!0,rel:!0});var l5=i(uc);aP=s(l5,"Parallelization"),l5.forEach(t),i5.forEach(t),ja.forEach(t),iP=p(Mt),Xt=a(Mt,"DIV",{class:!0});var Ca=i(Xt);y(gc.$$.fragment,Ca),lP=p(Ca),es=a(Ca,"P",{});var hm=i(es);dP=s(hm,"The "),Og=a(hm,"CODE",{});var d5=i(Og);cP=s(d5,"FlaxBertPreTrainedModel"),d5.forEach(t),pP=s(hm," forward method, overrides the "),Ig=a(hm,"CODE",{});var c5=i(Ig);hP=s(c5,"__call__"),c5.forEach(t),mP=s(hm," special method."),hm.forEach(t),fP=p(Ca),y(ta.$$.fragment,Ca),uP=p(Ca),y(oa.$$.fragment,Ca),Ca.forEach(t),Mt.forEach(t),Mk=p(o),ts=a(o,"H2",{class:!0});var _v=i(ts);na=a(_v,"A",{id:!0,class:!0,href:!0});var p5=i(na);Ag=a(p5,"SPAN",{});var h5=i(Ag);y(_c.$$.fragment,h5),h5.forEach(t),p5.forEach(t),gP=p(_v),Lg=a(_v,"SPAN",{});var m5=i(Lg);_P=s(m5,"FlaxBertForNextSentencePrediction"),m5.forEach(t),_v.forEach(t),zk=p(o),Ue=a(o,"DIV",{class:!0});var zt=i(Ue);y(bc.$$.fragment,zt),bP=p(zt),kc=a(zt,"P",{});var bv=i(kc);kP=s(bv,"Bert Model with a "),Dg=a(bv,"CODE",{});var f5=i(Dg);TP=s(f5,"next sentence prediction (classification)"),f5.forEach(t),vP=s(bv," head on top."),bv.forEach(t),yP=p(zt),Tc=a(zt,"P",{});var kv=i(Tc);wP=s(kv,"This model inherits from "),Fh=a(kv,"A",{href:!0});var u5=i(Fh);$P=s(u5,"FlaxPreTrainedModel"),u5.forEach(t),xP=s(kv,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading, saving and converting weights from PyTorch models)`),kv.forEach(t),FP=p(zt),vc=a(zt,"P",{});var Tv=i(vc);BP=s(Tv,"This model is also a Flax Linen "),yc=a(Tv,"A",{href:!0,rel:!0});var g5=i(yc);EP=s(g5,"flax.linen.Module"),g5.forEach(t),MP=s(Tv,`
subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
general usage and behavior.`),Tv.forEach(t),zP=p(zt),Sg=a(zt,"P",{});var _5=i(Sg);PP=s(_5,"Finally, this model supports inherent JAX features such as:"),_5.forEach(t),qP=p(zt),bo=a(zt,"UL",{});var Na=i(bo);Ug=a(Na,"LI",{});var b5=i(Ug);wc=a(b5,"A",{href:!0,rel:!0});var k5=i(wc);jP=s(k5,"Just-In-Time (JIT) compilation"),k5.forEach(t),b5.forEach(t),CP=p(Na),Wg=a(Na,"LI",{});var T5=i(Wg);$c=a(T5,"A",{href:!0,rel:!0});var v5=i($c);NP=s(v5,"Automatic Differentiation"),v5.forEach(t),T5.forEach(t),OP=p(Na),Hg=a(Na,"LI",{});var y5=i(Hg);xc=a(y5,"A",{href:!0,rel:!0});var w5=i(xc);IP=s(w5,"Vectorization"),w5.forEach(t),y5.forEach(t),AP=p(Na),Rg=a(Na,"LI",{});var $5=i(Rg);Fc=a($5,"A",{href:!0,rel:!0});var x5=i(Fc);LP=s(x5,"Parallelization"),x5.forEach(t),$5.forEach(t),Na.forEach(t),DP=p(zt),Yt=a(zt,"DIV",{class:!0});var Oa=i(Yt);y(Bc.$$.fragment,Oa),SP=p(Oa),os=a(Oa,"P",{});var mm=i(os);UP=s(mm,"The "),Vg=a(mm,"CODE",{});var F5=i(Vg);WP=s(F5,"FlaxBertPreTrainedModel"),F5.forEach(t),HP=s(mm," forward method, overrides the "),Qg=a(mm,"CODE",{});var B5=i(Qg);RP=s(B5,"__call__"),B5.forEach(t),VP=s(mm," special method."),mm.forEach(t),QP=p(Oa),y(sa.$$.fragment,Oa),KP=p(Oa),y(ra.$$.fragment,Oa),Oa.forEach(t),zt.forEach(t),Pk=p(o),ns=a(o,"H2",{class:!0});var vv=i(ns);aa=a(vv,"A",{id:!0,class:!0,href:!0});var E5=i(aa);Kg=a(E5,"SPAN",{});var M5=i(Kg);y(Ec.$$.fragment,M5),M5.forEach(t),E5.forEach(t),JP=p(vv),Jg=a(vv,"SPAN",{});var z5=i(Jg);GP=s(z5,"FlaxBertForSequenceClassification"),z5.forEach(t),vv.forEach(t),qk=p(o),We=a(o,"DIV",{class:!0});var Pt=i(We);y(Mc.$$.fragment,Pt),XP=p(Pt),Gg=a(Pt,"P",{});var P5=i(Gg);YP=s(P5,`Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
output) e.g. for GLUE tasks.`),P5.forEach(t),ZP=p(Pt),zc=a(Pt,"P",{});var yv=i(zc);e8=s(yv,"This model inherits from "),Bh=a(yv,"A",{href:!0});var q5=i(Bh);t8=s(q5,"FlaxPreTrainedModel"),q5.forEach(t),o8=s(yv,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading, saving and converting weights from PyTorch models)`),yv.forEach(t),n8=p(Pt),Pc=a(Pt,"P",{});var wv=i(Pc);s8=s(wv,"This model is also a Flax Linen "),qc=a(wv,"A",{href:!0,rel:!0});var j5=i(qc);r8=s(j5,"flax.linen.Module"),j5.forEach(t),a8=s(wv,`
subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
general usage and behavior.`),wv.forEach(t),i8=p(Pt),Xg=a(Pt,"P",{});var C5=i(Xg);l8=s(C5,"Finally, this model supports inherent JAX features such as:"),C5.forEach(t),d8=p(Pt),ko=a(Pt,"UL",{});var Ia=i(ko);Yg=a(Ia,"LI",{});var N5=i(Yg);jc=a(N5,"A",{href:!0,rel:!0});var O5=i(jc);c8=s(O5,"Just-In-Time (JIT) compilation"),O5.forEach(t),N5.forEach(t),p8=p(Ia),Zg=a(Ia,"LI",{});var I5=i(Zg);Cc=a(I5,"A",{href:!0,rel:!0});var A5=i(Cc);h8=s(A5,"Automatic Differentiation"),A5.forEach(t),I5.forEach(t),m8=p(Ia),e_=a(Ia,"LI",{});var L5=i(e_);Nc=a(L5,"A",{href:!0,rel:!0});var D5=i(Nc);f8=s(D5,"Vectorization"),D5.forEach(t),L5.forEach(t),u8=p(Ia),t_=a(Ia,"LI",{});var S5=i(t_);Oc=a(S5,"A",{href:!0,rel:!0});var U5=i(Oc);g8=s(U5,"Parallelization"),U5.forEach(t),S5.forEach(t),Ia.forEach(t),_8=p(Pt),Zt=a(Pt,"DIV",{class:!0});var Aa=i(Zt);y(Ic.$$.fragment,Aa),b8=p(Aa),ss=a(Aa,"P",{});var fm=i(ss);k8=s(fm,"The "),o_=a(fm,"CODE",{});var W5=i(o_);T8=s(W5,"FlaxBertPreTrainedModel"),W5.forEach(t),v8=s(fm," forward method, overrides the "),n_=a(fm,"CODE",{});var H5=i(n_);y8=s(H5,"__call__"),H5.forEach(t),w8=s(fm," special method."),fm.forEach(t),$8=p(Aa),y(ia.$$.fragment,Aa),x8=p(Aa),y(la.$$.fragment,Aa),Aa.forEach(t),Pt.forEach(t),jk=p(o),rs=a(o,"H2",{class:!0});var $v=i(rs);da=a($v,"A",{id:!0,class:!0,href:!0});var R5=i(da);s_=a(R5,"SPAN",{});var V5=i(s_);y(Ac.$$.fragment,V5),V5.forEach(t),R5.forEach(t),F8=p($v),r_=a($v,"SPAN",{});var Q5=i(r_);B8=s(Q5,"FlaxBertForMultipleChoice"),Q5.forEach(t),$v.forEach(t),Ck=p(o),He=a(o,"DIV",{class:!0});var qt=i(He);y(Lc.$$.fragment,qt),E8=p(qt),a_=a(qt,"P",{});var K5=i(a_);M8=s(K5,`Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`),K5.forEach(t),z8=p(qt),Dc=a(qt,"P",{});var xv=i(Dc);P8=s(xv,"This model inherits from "),Eh=a(xv,"A",{href:!0});var J5=i(Eh);q8=s(J5,"FlaxPreTrainedModel"),J5.forEach(t),j8=s(xv,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading, saving and converting weights from PyTorch models)`),xv.forEach(t),C8=p(qt),Sc=a(qt,"P",{});var Fv=i(Sc);N8=s(Fv,"This model is also a Flax Linen "),Uc=a(Fv,"A",{href:!0,rel:!0});var G5=i(Uc);O8=s(G5,"flax.linen.Module"),G5.forEach(t),I8=s(Fv,`
subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
general usage and behavior.`),Fv.forEach(t),A8=p(qt),i_=a(qt,"P",{});var X5=i(i_);L8=s(X5,"Finally, this model supports inherent JAX features such as:"),X5.forEach(t),D8=p(qt),To=a(qt,"UL",{});var La=i(To);l_=a(La,"LI",{});var Y5=i(l_);Wc=a(Y5,"A",{href:!0,rel:!0});var Z5=i(Wc);S8=s(Z5,"Just-In-Time (JIT) compilation"),Z5.forEach(t),Y5.forEach(t),U8=p(La),d_=a(La,"LI",{});var e6=i(d_);Hc=a(e6,"A",{href:!0,rel:!0});var t6=i(Hc);W8=s(t6,"Automatic Differentiation"),t6.forEach(t),e6.forEach(t),H8=p(La),c_=a(La,"LI",{});var o6=i(c_);Rc=a(o6,"A",{href:!0,rel:!0});var n6=i(Rc);R8=s(n6,"Vectorization"),n6.forEach(t),o6.forEach(t),V8=p(La),p_=a(La,"LI",{});var s6=i(p_);Vc=a(s6,"A",{href:!0,rel:!0});var r6=i(Vc);Q8=s(r6,"Parallelization"),r6.forEach(t),s6.forEach(t),La.forEach(t),K8=p(qt),eo=a(qt,"DIV",{class:!0});var Da=i(eo);y(Qc.$$.fragment,Da),J8=p(Da),as=a(Da,"P",{});var um=i(as);G8=s(um,"The "),h_=a(um,"CODE",{});var a6=i(h_);X8=s(a6,"FlaxBertPreTrainedModel"),a6.forEach(t),Y8=s(um," forward method, overrides the "),m_=a(um,"CODE",{});var i6=i(m_);Z8=s(i6,"__call__"),i6.forEach(t),eq=s(um," special method."),um.forEach(t),tq=p(Da),y(ca.$$.fragment,Da),oq=p(Da),y(pa.$$.fragment,Da),Da.forEach(t),qt.forEach(t),Nk=p(o),is=a(o,"H2",{class:!0});var Bv=i(is);ha=a(Bv,"A",{id:!0,class:!0,href:!0});var l6=i(ha);f_=a(l6,"SPAN",{});var d6=i(f_);y(Kc.$$.fragment,d6),d6.forEach(t),l6.forEach(t),nq=p(Bv),u_=a(Bv,"SPAN",{});var c6=i(u_);sq=s(c6,"FlaxBertForTokenClassification"),c6.forEach(t),Bv.forEach(t),Ok=p(o),Re=a(o,"DIV",{class:!0});var jt=i(Re);y(Jc.$$.fragment,jt),rq=p(jt),g_=a(jt,"P",{});var p6=i(g_);aq=s(p6,`Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
Named-Entity-Recognition (NER) tasks.`),p6.forEach(t),iq=p(jt),Gc=a(jt,"P",{});var Ev=i(Gc);lq=s(Ev,"This model inherits from "),Mh=a(Ev,"A",{href:!0});var h6=i(Mh);dq=s(h6,"FlaxPreTrainedModel"),h6.forEach(t),cq=s(Ev,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading, saving and converting weights from PyTorch models)`),Ev.forEach(t),pq=p(jt),Xc=a(jt,"P",{});var Mv=i(Xc);hq=s(Mv,"This model is also a Flax Linen "),Yc=a(Mv,"A",{href:!0,rel:!0});var m6=i(Yc);mq=s(m6,"flax.linen.Module"),m6.forEach(t),fq=s(Mv,`
subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
general usage and behavior.`),Mv.forEach(t),uq=p(jt),__=a(jt,"P",{});var f6=i(__);gq=s(f6,"Finally, this model supports inherent JAX features such as:"),f6.forEach(t),_q=p(jt),vo=a(jt,"UL",{});var Sa=i(vo);b_=a(Sa,"LI",{});var u6=i(b_);Zc=a(u6,"A",{href:!0,rel:!0});var g6=i(Zc);bq=s(g6,"Just-In-Time (JIT) compilation"),g6.forEach(t),u6.forEach(t),kq=p(Sa),k_=a(Sa,"LI",{});var _6=i(k_);ep=a(_6,"A",{href:!0,rel:!0});var b6=i(ep);Tq=s(b6,"Automatic Differentiation"),b6.forEach(t),_6.forEach(t),vq=p(Sa),T_=a(Sa,"LI",{});var k6=i(T_);tp=a(k6,"A",{href:!0,rel:!0});var T6=i(tp);yq=s(T6,"Vectorization"),T6.forEach(t),k6.forEach(t),wq=p(Sa),v_=a(Sa,"LI",{});var v6=i(v_);op=a(v6,"A",{href:!0,rel:!0});var y6=i(op);$q=s(y6,"Parallelization"),y6.forEach(t),v6.forEach(t),Sa.forEach(t),xq=p(jt),to=a(jt,"DIV",{class:!0});var Ua=i(to);y(np.$$.fragment,Ua),Fq=p(Ua),ls=a(Ua,"P",{});var gm=i(ls);Bq=s(gm,"The "),y_=a(gm,"CODE",{});var w6=i(y_);Eq=s(w6,"FlaxBertPreTrainedModel"),w6.forEach(t),Mq=s(gm," forward method, overrides the "),w_=a(gm,"CODE",{});var $6=i(w_);zq=s($6,"__call__"),$6.forEach(t),Pq=s(gm," special method."),gm.forEach(t),qq=p(Ua),y(ma.$$.fragment,Ua),jq=p(Ua),y(fa.$$.fragment,Ua),Ua.forEach(t),jt.forEach(t),Ik=p(o),ds=a(o,"H2",{class:!0});var zv=i(ds);ua=a(zv,"A",{id:!0,class:!0,href:!0});var x6=i(ua);$_=a(x6,"SPAN",{});var F6=i($_);y(sp.$$.fragment,F6),F6.forEach(t),x6.forEach(t),Cq=p(zv),x_=a(zv,"SPAN",{});var B6=i(x_);Nq=s(B6,"FlaxBertForQuestionAnswering"),B6.forEach(t),zv.forEach(t),Ak=p(o),Ve=a(o,"DIV",{class:!0});var Ct=i(Ve);y(rp.$$.fragment,Ct),Oq=p(Ct),cs=a(Ct,"P",{});var _m=i(cs);Iq=s(_m,`Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
layers on top of the hidden-states output to compute `),F_=a(_m,"CODE",{});var E6=i(F_);Aq=s(E6,"span start logits"),E6.forEach(t),Lq=s(_m," and "),B_=a(_m,"CODE",{});var M6=i(B_);Dq=s(M6,"span end logits"),M6.forEach(t),Sq=s(_m,")."),_m.forEach(t),Uq=p(Ct),ap=a(Ct,"P",{});var Pv=i(ap);Wq=s(Pv,"This model inherits from "),zh=a(Pv,"A",{href:!0});var z6=i(zh);Hq=s(z6,"FlaxPreTrainedModel"),z6.forEach(t),Rq=s(Pv,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading, saving and converting weights from PyTorch models)`),Pv.forEach(t),Vq=p(Ct),ip=a(Ct,"P",{});var qv=i(ip);Qq=s(qv,"This model is also a Flax Linen "),lp=a(qv,"A",{href:!0,rel:!0});var P6=i(lp);Kq=s(P6,"flax.linen.Module"),P6.forEach(t),Jq=s(qv,`
subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
general usage and behavior.`),qv.forEach(t),Gq=p(Ct),E_=a(Ct,"P",{});var q6=i(E_);Xq=s(q6,"Finally, this model supports inherent JAX features such as:"),q6.forEach(t),Yq=p(Ct),yo=a(Ct,"UL",{});var Wa=i(yo);M_=a(Wa,"LI",{});var j6=i(M_);dp=a(j6,"A",{href:!0,rel:!0});var C6=i(dp);Zq=s(C6,"Just-In-Time (JIT) compilation"),C6.forEach(t),j6.forEach(t),e7=p(Wa),z_=a(Wa,"LI",{});var N6=i(z_);cp=a(N6,"A",{href:!0,rel:!0});var O6=i(cp);t7=s(O6,"Automatic Differentiation"),O6.forEach(t),N6.forEach(t),o7=p(Wa),P_=a(Wa,"LI",{});var I6=i(P_);pp=a(I6,"A",{href:!0,rel:!0});var A6=i(pp);n7=s(A6,"Vectorization"),A6.forEach(t),I6.forEach(t),s7=p(Wa),q_=a(Wa,"LI",{});var L6=i(q_);hp=a(L6,"A",{href:!0,rel:!0});var D6=i(hp);r7=s(D6,"Parallelization"),D6.forEach(t),L6.forEach(t),Wa.forEach(t),a7=p(Ct),oo=a(Ct,"DIV",{class:!0});var Ha=i(oo);y(mp.$$.fragment,Ha),i7=p(Ha),ps=a(Ha,"P",{});var bm=i(ps);l7=s(bm,"The "),j_=a(bm,"CODE",{});var S6=i(j_);d7=s(S6,"FlaxBertPreTrainedModel"),S6.forEach(t),c7=s(bm," forward method, overrides the "),C_=a(bm,"CODE",{});var U6=i(C_);p7=s(U6,"__call__"),U6.forEach(t),h7=s(bm," special method."),bm.forEach(t),m7=p(Ha),y(ga.$$.fragment,Ha),f7=p(Ha),y(_a.$$.fragment,Ha),Ha.forEach(t),Ct.forEach(t),this.h()},h(){u(d,"name","hf:doc:metadata"),u(d,"content",JSON.stringify(_O)),u(h,"id","bert"),u(h,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(h,"href","#bert"),u(m,"class","relative group"),u(ne,"id","overview"),u(ne,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(ne,"href","#overview"),u(z,"class","relative group"),u(ae,"href","https://arxiv.org/abs/1810.04805"),u(ae,"rel","nofollow"),u(Be,"href","https://huggingface.co/thomwolf"),u(Be,"rel","nofollow"),u(Ee,"href","https://github.com/google-research/bert"),u(Ee,"rel","nofollow"),u(Fe,"id","transformers.BertConfig"),u(Fe,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(Fe,"href","#transformers.BertConfig"),u(xe,"class","relative group"),u(Ep,"href","/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertModel"),u(Mp,"href","/docs/transformers/pr_18729/en/model_doc/bert#transformers.TFBertModel"),u(Va,"href","https://huggingface.co/bert-base-uncased"),u(Va,"rel","nofollow"),u(zp,"href","/docs/transformers/pr_18729/en/main_classes/configuration#transformers.PretrainedConfig"),u(Pp,"href","/docs/transformers/pr_18729/en/main_classes/configuration#transformers.PretrainedConfig"),u(Nt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(bs,"id","transformers.BertTokenizer"),u(bs,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(bs,"href","#transformers.BertTokenizer"),u(Jo,"class","relative group"),u(qp,"href","/docs/transformers/pr_18729/en/main_classes/tokenizer#transformers.PreTrainedTokenizer"),u(Bo,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(ks,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(It,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Np,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Ne,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(vs,"id","transformers.BertTokenizerFast"),u(vs,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(vs,"href","#transformers.BertTokenizerFast"),u(Xo,"class","relative group"),u(Op,"href","/docs/transformers/pr_18729/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast"),u(Eo,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(At,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(rt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(ws,"id","transformers.TFBertTokenizer"),u(ws,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(ws,"href","#transformers.TFBertTokenizer"),u(Zo,"class","relative group"),u(Mo,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(zo,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(at,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Fs,"id","transformers.models.bert.modeling_bert.BertForPreTrainingOutput"),u(Fs,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(Fs,"href","#transformers.models.bert.modeling_bert.BertForPreTrainingOutput"),u(on,"class","relative group"),u(Lp,"href","/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertForPreTraining"),u(nn,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Dp,"href","/docs/transformers/pr_18729/en/model_doc/bert#transformers.TFBertForPreTraining"),u(sn,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Sp,"href","/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertForPreTraining"),u(Bs,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(mo,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Es,"id","transformers.BertModel"),u(Es,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(Es,"href","#transformers.BertModel"),u(rn,"class","relative group"),u(Up,"href","/docs/transformers/pr_18729/en/main_classes/model#transformers.PreTrainedModel"),u(Bi,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),u(Bi,"rel","nofollow"),u(Mi,"href","https://arxiv.org/abs/1706.03762"),u(Mi,"rel","nofollow"),u(Wp,"href","/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertModel"),u(Lt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Oe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Ps,"id","transformers.BertForPreTraining"),u(Ps,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(Ps,"href","#transformers.BertForPreTraining"),u(ln,"class","relative group"),u(Hp,"href","/docs/transformers/pr_18729/en/main_classes/model#transformers.PreTrainedModel"),u(Ni,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),u(Ni,"rel","nofollow"),u(Rp,"href","/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertForPreTraining"),u(Dt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(it,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Cs,"id","transformers.BertLMHeadModel"),u(Cs,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(Cs,"href","#transformers.BertLMHeadModel"),u(pn,"class","relative group"),u(Vp,"href","/docs/transformers/pr_18729/en/main_classes/model#transformers.PreTrainedModel"),u(Ui,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),u(Ui,"rel","nofollow"),u(Qp,"href","/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertLMHeadModel"),u(St,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(lt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Is,"id","transformers.BertForMaskedLM"),u(Is,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(Is,"href","#transformers.BertForMaskedLM"),u(mn,"class","relative group"),u(Kp,"href","/docs/transformers/pr_18729/en/main_classes/model#transformers.PreTrainedModel"),u(Ji,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),u(Ji,"rel","nofollow"),u(Jp,"href","/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertForMaskedLM"),u(ut,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(dt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Ss,"id","transformers.BertForNextSentencePrediction"),u(Ss,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(Ss,"href","#transformers.BertForNextSentencePrediction"),u(un,"class","relative group"),u(Gp,"href","/docs/transformers/pr_18729/en/main_classes/model#transformers.PreTrainedModel"),u(ol,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),u(ol,"rel","nofollow"),u(Xp,"href","/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertForNextSentencePrediction"),u(Ut,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(ct,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Hs,"id","transformers.BertForSequenceClassification"),u(Hs,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(Hs,"href","#transformers.BertForSequenceClassification"),u(_n,"class","relative group"),u(Yp,"href","/docs/transformers/pr_18729/en/main_classes/model#transformers.PreTrainedModel"),u(ll,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),u(ll,"rel","nofollow"),u(Zp,"href","/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertForSequenceClassification"),u(Qe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(pt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Gs,"id","transformers.BertForMultipleChoice"),u(Gs,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(Gs,"href","#transformers.BertForMultipleChoice"),u(kn,"class","relative group"),u(eh,"href","/docs/transformers/pr_18729/en/main_classes/model#transformers.PreTrainedModel"),u(fl,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),u(fl,"rel","nofollow"),u(th,"href","/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertForMultipleChoice"),u(Wt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(ht,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Zs,"id","transformers.BertForTokenClassification"),u(Zs,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(Zs,"href","#transformers.BertForTokenClassification"),u(vn,"class","relative group"),u(oh,"href","/docs/transformers/pr_18729/en/main_classes/model#transformers.PreTrainedModel"),u(Tl,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),u(Tl,"rel","nofollow"),u(nh,"href","/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertForTokenClassification"),u(gt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(mt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(nr,"id","transformers.BertForQuestionAnswering"),u(nr,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(nr,"href","#transformers.BertForQuestionAnswering"),u(wn,"class","relative group"),u(sh,"href","/docs/transformers/pr_18729/en/main_classes/model#transformers.PreTrainedModel"),u(Fl,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),u(Fl,"rel","nofollow"),u(rh,"href","/docs/transformers/pr_18729/en/model_doc/bert#transformers.BertForQuestionAnswering"),u(_t,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(ft,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(ir,"id","transformers.TFBertModel"),u(ir,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(ir,"href","#transformers.TFBertModel"),u(Fn,"class","relative group"),u(ah,"href","/docs/transformers/pr_18729/en/main_classes/model#transformers.TFPreTrainedModel"),u(ql,"href","https://www.tensorflow.org/api_docs/python/tf/keras/Model"),u(ql,"rel","nofollow"),u(ih,"href","/docs/transformers/pr_18729/en/model_doc/bert#transformers.TFBertModel"),u(Ht,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Je,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(pr,"id","transformers.TFBertForPreTraining"),u(pr,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(pr,"href","#transformers.TFBertForPreTraining"),u(En,"class","relative group"),u(lh,"href","/docs/transformers/pr_18729/en/main_classes/model#transformers.TFPreTrainedModel"),u(Al,"href","https://www.tensorflow.org/api_docs/python/tf/keras/Model"),u(Al,"rel","nofollow"),u(dh,"href","/docs/transformers/pr_18729/en/model_doc/bert#transformers.TFBertForPreTraining"),u(Rt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Ge,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(ur,"id","transformers.TFBertLMHeadModel"),u(ur,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(ur,"href","#transformers.TFBertLMHeadModel"),u(Pn,"class","relative group"),u(bt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(qn,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(_r,"id","transformers.TFBertForMaskedLM"),u(_r,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(_r,"href","#transformers.TFBertForMaskedLM"),u(jn,"class","relative group"),u(ch,"href","/docs/transformers/pr_18729/en/main_classes/model#transformers.TFPreTrainedModel"),u(Xl,"href","https://www.tensorflow.org/api_docs/python/tf/keras/Model"),u(Xl,"rel","nofollow"),u(ph,"href","/docs/transformers/pr_18729/en/model_doc/bert#transformers.TFBertForMaskedLM"),u(kt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Xe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(yr,"id","transformers.TFBertForNextSentencePrediction"),u(yr,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(yr,"href","#transformers.TFBertForNextSentencePrediction"),u(Nn,"class","relative group"),u(hh,"href","/docs/transformers/pr_18729/en/main_classes/model#transformers.TFPreTrainedModel"),u(sd,"href","https://www.tensorflow.org/api_docs/python/tf/keras/Model"),u(sd,"rel","nofollow"),u(mh,"href","/docs/transformers/pr_18729/en/model_doc/bert#transformers.TFBertForNextSentencePrediction"),u(Vt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Ye,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Fr,"id","transformers.TFBertForSequenceClassification"),u(Fr,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(Fr,"href","#transformers.TFBertForSequenceClassification"),u(In,"class","relative group"),u(fh,"href","/docs/transformers/pr_18729/en/main_classes/model#transformers.TFPreTrainedModel"),u(cd,"href","https://www.tensorflow.org/api_docs/python/tf/keras/Model"),u(cd,"rel","nofollow"),u(uh,"href","/docs/transformers/pr_18729/en/model_doc/bert#transformers.TFBertForSequenceClassification"),u(Tt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Ze,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Pr,"id","transformers.TFBertForMultipleChoice"),u(Pr,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(Pr,"href","#transformers.TFBertForMultipleChoice"),u(Ln,"class","relative group"),u(gh,"href","/docs/transformers/pr_18729/en/main_classes/model#transformers.TFPreTrainedModel"),u(gd,"href","https://www.tensorflow.org/api_docs/python/tf/keras/Model"),u(gd,"rel","nofollow"),u(_h,"href","/docs/transformers/pr_18729/en/model_doc/bert#transformers.TFBertForMultipleChoice"),u(Qt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(et,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Nr,"id","transformers.TFBertForTokenClassification"),u(Nr,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(Nr,"href","#transformers.TFBertForTokenClassification"),u(Sn,"class","relative group"),u(bh,"href","/docs/transformers/pr_18729/en/main_classes/model#transformers.TFPreTrainedModel"),u(yd,"href","https://www.tensorflow.org/api_docs/python/tf/keras/Model"),u(yd,"rel","nofollow"),u(kh,"href","/docs/transformers/pr_18729/en/model_doc/bert#transformers.TFBertForTokenClassification"),u(vt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(tt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Dr,"id","transformers.TFBertForQuestionAnswering"),u(Dr,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(Dr,"href","#transformers.TFBertForQuestionAnswering"),u(Wn,"class","relative group"),u(Th,"href","/docs/transformers/pr_18729/en/main_classes/model#transformers.TFPreTrainedModel"),u(Ed,"href","https://www.tensorflow.org/api_docs/python/tf/keras/Model"),u(Ed,"rel","nofollow"),u(vh,"href","/docs/transformers/pr_18729/en/model_doc/bert#transformers.TFBertForQuestionAnswering"),u(yt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(ot,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Rr,"id","transformers.FlaxBertModel"),u(Rr,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(Rr,"href","#transformers.FlaxBertModel"),u(Vn,"class","relative group"),u(yh,"href","/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel"),u(Cd,"href","https://flax.readthedocs.io/en/latest/flax.linen.html#module"),u(Cd,"rel","nofollow"),u(Nd,"href","https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit"),u(Nd,"rel","nofollow"),u(Od,"href","https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation"),u(Od,"rel","nofollow"),u(Id,"href","https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap"),u(Id,"rel","nofollow"),u(Ad,"href","https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap"),u(Ad,"rel","nofollow"),u(Kt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Ae,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Kr,"id","transformers.FlaxBertForPreTraining"),u(Kr,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(Kr,"href","#transformers.FlaxBertForPreTraining"),u(Kn,"class","relative group"),u(wh,"href","/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel"),u(Hd,"href","https://flax.readthedocs.io/en/latest/flax.linen.html#module"),u(Hd,"rel","nofollow"),u(Rd,"href","https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit"),u(Rd,"rel","nofollow"),u(Vd,"href","https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation"),u(Vd,"rel","nofollow"),u(Qd,"href","https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap"),u(Qd,"rel","nofollow"),u(Kd,"href","https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap"),u(Kd,"rel","nofollow"),u(Jt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Le,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Xr,"id","transformers.FlaxBertForCausalLM"),u(Xr,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(Xr,"href","#transformers.FlaxBertForCausalLM"),u(Xn,"class","relative group"),u($h,"href","/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel"),u(ec,"href","https://flax.readthedocs.io/en/latest/flax.linen.html#module"),u(ec,"rel","nofollow"),u(tc,"href","https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit"),u(tc,"rel","nofollow"),u(oc,"href","https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation"),u(oc,"rel","nofollow"),u(nc,"href","https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap"),u(nc,"rel","nofollow"),u(sc,"href","https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap"),u(sc,"rel","nofollow"),u(Gt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(De,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(ea,"id","transformers.FlaxBertForMaskedLM"),u(ea,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(ea,"href","#transformers.FlaxBertForMaskedLM"),u(Zn,"class","relative group"),u(xh,"href","/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel"),u(pc,"href","https://flax.readthedocs.io/en/latest/flax.linen.html#module"),u(pc,"rel","nofollow"),u(hc,"href","https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit"),u(hc,"rel","nofollow"),u(mc,"href","https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation"),u(mc,"rel","nofollow"),u(fc,"href","https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap"),u(fc,"rel","nofollow"),u(uc,"href","https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap"),u(uc,"rel","nofollow"),u(Xt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Se,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(na,"id","transformers.FlaxBertForNextSentencePrediction"),u(na,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(na,"href","#transformers.FlaxBertForNextSentencePrediction"),u(ts,"class","relative group"),u(Fh,"href","/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel"),u(yc,"href","https://flax.readthedocs.io/en/latest/flax.linen.html#module"),u(yc,"rel","nofollow"),u(wc,"href","https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit"),u(wc,"rel","nofollow"),u($c,"href","https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation"),u($c,"rel","nofollow"),u(xc,"href","https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap"),u(xc,"rel","nofollow"),u(Fc,"href","https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap"),u(Fc,"rel","nofollow"),u(Yt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Ue,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(aa,"id","transformers.FlaxBertForSequenceClassification"),u(aa,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(aa,"href","#transformers.FlaxBertForSequenceClassification"),u(ns,"class","relative group"),u(Bh,"href","/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel"),u(qc,"href","https://flax.readthedocs.io/en/latest/flax.linen.html#module"),u(qc,"rel","nofollow"),u(jc,"href","https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit"),u(jc,"rel","nofollow"),u(Cc,"href","https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation"),u(Cc,"rel","nofollow"),u(Nc,"href","https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap"),u(Nc,"rel","nofollow"),u(Oc,"href","https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap"),u(Oc,"rel","nofollow"),u(Zt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(We,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(da,"id","transformers.FlaxBertForMultipleChoice"),u(da,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(da,"href","#transformers.FlaxBertForMultipleChoice"),u(rs,"class","relative group"),u(Eh,"href","/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel"),u(Uc,"href","https://flax.readthedocs.io/en/latest/flax.linen.html#module"),u(Uc,"rel","nofollow"),u(Wc,"href","https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit"),u(Wc,"rel","nofollow"),u(Hc,"href","https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation"),u(Hc,"rel","nofollow"),u(Rc,"href","https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap"),u(Rc,"rel","nofollow"),u(Vc,"href","https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap"),u(Vc,"rel","nofollow"),u(eo,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(He,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(ha,"id","transformers.FlaxBertForTokenClassification"),u(ha,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(ha,"href","#transformers.FlaxBertForTokenClassification"),u(is,"class","relative group"),u(Mh,"href","/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel"),u(Yc,"href","https://flax.readthedocs.io/en/latest/flax.linen.html#module"),u(Yc,"rel","nofollow"),u(Zc,"href","https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit"),u(Zc,"rel","nofollow"),u(ep,"href","https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation"),u(ep,"rel","nofollow"),u(tp,"href","https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap"),u(tp,"rel","nofollow"),u(op,"href","https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap"),u(op,"rel","nofollow"),u(to,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Re,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(ua,"id","transformers.FlaxBertForQuestionAnswering"),u(ua,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(ua,"href","#transformers.FlaxBertForQuestionAnswering"),u(ds,"class","relative group"),u(zh,"href","/docs/transformers/pr_18729/en/main_classes/model#transformers.FlaxPreTrainedModel"),u(lp,"href","https://flax.readthedocs.io/en/latest/flax.linen.html#module"),u(lp,"rel","nofollow"),u(dp,"href","https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit"),u(dp,"rel","nofollow"),u(cp,"href","https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation"),u(cp,"rel","nofollow"),u(pp,"href","https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap"),u(pp,"rel","nofollow"),u(hp,"href","https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap"),u(hp,"rel","nofollow"),u(oo,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Ve,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(o,k){e(document.head,d),b(o,_,k),b(o,m,k),e(m,h),e(h,g),w(l,g,null),e(m,f),e(m,E),e(E,be),b(o,X,k),b(o,z,k),e(z,ne),e(ne,L),w(re,L,null),e(z,ke),e(z,D),e(D,Te),b(o,me,k),b(o,J,k),e(J,O),e(J,ae),e(ae,Y),e(J,P),b(o,j,k),b(o,ie,k),e(ie,H),b(o,fe,k),b(o,le,k),e(le,S),e(S,ve),b(o,ue,k),b(o,q,k),e(q,ce),e(ce,R),b(o,ge,k),b(o,de,k),e(de,V),b(o,_e,k),b(o,se,k),e(se,N),e(N,ye),e(se,Q),e(se,pe),e(pe,T),b(o,M,k),b(o,K,k),e(K,ze),e(K,Be),e(Be,I),e(K,Pe),e(K,Ee),e(Ee,qe),e(K,A),b(o,W,k),b(o,xe,k),e(xe,Fe),e(Fe,U),w(Me,U,null),e(xe,je),e(xe,he),e(he,Ce),b(o,Eb,k),b(o,Nt,k),w(Ra,Nt,null),e(Nt,jv),e(Nt,ho),e(ho,Cv),e(ho,Ep),e(Ep,Nv),e(ho,Ov),e(ho,Mp),e(Mp,Iv),e(ho,Av),e(ho,Va),e(Va,Lv),e(ho,Dv),e(Nt,Sv),e(Nt,Ko),e(Ko,Uv),e(Ko,zp),e(zp,Wv),e(Ko,Hv),e(Ko,Pp),e(Pp,Rv),e(Ko,Vv),e(Nt,Qv),w(_s,Nt,null),b(o,Mb,k),b(o,Jo,k),e(Jo,bs),e(bs,km),w(Qa,km,null),e(Jo,Kv),e(Jo,Tm),e(Tm,Jv),b(o,zb,k),b(o,Ne,k),w(Ka,Ne,null),e(Ne,Gv),e(Ne,vm),e(vm,Xv),e(Ne,Yv),e(Ne,Ja),e(Ja,Zv),e(Ja,qp),e(qp,ey),e(Ja,ty),e(Ne,oy),e(Ne,Bo),w(Ga,Bo,null),e(Bo,ny),e(Bo,ym),e(ym,sy),e(Bo,ry),e(Bo,Xa),e(Xa,jp),e(jp,ay),e(jp,wm),e(wm,iy),e(Xa,ly),e(Xa,Cp),e(Cp,dy),e(Cp,$m),e($m,cy),e(Ne,py),e(Ne,ks),w(Ya,ks,null),e(ks,hy),e(ks,Za),e(Za,my),e(Za,xm),e(xm,fy),e(Za,uy),e(Ne,gy),e(Ne,It),w(ei,It,null),e(It,_y),e(It,Fm),e(Fm,by),e(It,ky),w(Ts,It,null),e(It,Ty),e(It,Go),e(Go,vy),e(Go,Bm),e(Bm,yy),e(Go,wy),e(Go,Em),e(Em,$y),e(Go,xy),e(Ne,Fy),e(Ne,Np),w(ti,Np,null),b(o,Pb,k),b(o,Xo,k),e(Xo,vs),e(vs,Mm),w(oi,Mm,null),e(Xo,By),e(Xo,zm),e(zm,Ey),b(o,qb,k),b(o,rt,k),w(ni,rt,null),e(rt,My),e(rt,si),e(si,zy),e(si,Pm),e(Pm,Py),e(si,qy),e(rt,jy),e(rt,ri),e(ri,Cy),e(ri,Op),e(Op,Ny),e(ri,Oy),e(rt,Iy),e(rt,Eo),w(ai,Eo,null),e(Eo,Ay),e(Eo,qm),e(qm,Ly),e(Eo,Dy),e(Eo,ii),e(ii,Ip),e(Ip,Sy),e(Ip,jm),e(jm,Uy),e(ii,Wy),e(ii,Ap),e(Ap,Hy),e(Ap,Cm),e(Cm,Ry),e(rt,Vy),e(rt,At),w(li,At,null),e(At,Qy),e(At,Nm),e(Nm,Ky),e(At,Jy),w(ys,At,null),e(At,Gy),e(At,Yo),e(Yo,Xy),e(Yo,Om),e(Om,Yy),e(Yo,Zy),e(Yo,Im),e(Im,e1),e(Yo,t1),b(o,jb,k),b(o,Zo,k),e(Zo,ws),e(ws,Am),w(di,Am,null),e(Zo,o1),e(Zo,Lm),e(Lm,n1),b(o,Cb,k),b(o,at,k),w(ci,at,null),e(at,s1),e(at,en),e(en,r1),e(en,Dm),e(Dm,a1),e(en,i1),e(en,Sm),e(Sm,l1),e(en,d1),e(at,c1),e(at,pi),e(pi,p1),e(pi,Um),e(Um,h1),e(pi,m1),e(at,f1),e(at,Mo),w(hi,Mo,null),e(Mo,u1),e(Mo,mi),e(mi,g1),e(mi,Wm),e(Wm,_1),e(mi,b1),e(Mo,k1),w($s,Mo,null),e(at,T1),e(at,zo),w(fi,zo,null),e(zo,v1),e(zo,tn),e(tn,y1),e(tn,Hm),e(Hm,w1),e(tn,$1),e(tn,Rm),e(Rm,x1),e(tn,F1),e(zo,B1),w(xs,zo,null),b(o,Nb,k),b(o,on,k),e(on,Fs),e(Fs,Vm),w(ui,Vm,null),e(on,E1),e(on,Qm),e(Qm,M1),b(o,Ob,k),b(o,nn,k),w(gi,nn,null),e(nn,z1),e(nn,_i),e(_i,P1),e(_i,Lp),e(Lp,q1),e(_i,j1),b(o,Ib,k),b(o,sn,k),w(bi,sn,null),e(sn,C1),e(sn,ki),e(ki,N1),e(ki,Dp),e(Dp,O1),e(ki,I1),b(o,Ab,k),b(o,mo,k),w(Ti,mo,null),e(mo,A1),e(mo,vi),e(vi,L1),e(vi,Sp),e(Sp,D1),e(vi,S1),e(mo,U1),e(mo,Bs),w(yi,Bs,null),e(Bs,W1),e(Bs,Km),e(Km,H1),b(o,Lb,k),b(o,rn,k),e(rn,Es),e(Es,Jm),w(wi,Jm,null),e(rn,R1),e(rn,Gm),e(Gm,V1),b(o,Db,k),b(o,Oe,k),w($i,Oe,null),e(Oe,Q1),e(Oe,Xm),e(Xm,K1),e(Oe,J1),e(Oe,xi),e(xi,G1),e(xi,Up),e(Up,X1),e(xi,Y1),e(Oe,Z1),e(Oe,Fi),e(Fi,ew),e(Fi,Bi),e(Bi,tw),e(Fi,ow),e(Oe,nw),e(Oe,Ei),e(Ei,sw),e(Ei,Mi),e(Mi,rw),e(Ei,aw),e(Oe,iw),e(Oe,Ke),e(Ke,lw),e(Ke,Ym),e(Ym,dw),e(Ke,cw),e(Ke,Zm),e(Zm,pw),e(Ke,hw),e(Ke,ef),e(ef,mw),e(Ke,fw),e(Ke,tf),e(tf,uw),e(Ke,gw),e(Ke,of),e(of,_w),e(Ke,bw),e(Ke,nf),e(nf,kw),e(Ke,Tw),e(Oe,vw),e(Oe,Lt),w(zi,Lt,null),e(Lt,yw),e(Lt,an),e(an,ww),e(an,Wp),e(Wp,$w),e(an,xw),e(an,sf),e(sf,Fw),e(an,Bw),e(Lt,Ew),w(Ms,Lt,null),e(Lt,Mw),w(zs,Lt,null),b(o,Sb,k),b(o,ln,k),e(ln,Ps),e(Ps,rf),w(Pi,rf,null),e(ln,zw),e(ln,af),e(af,Pw),b(o,Ub,k),b(o,it,k),w(qi,it,null),e(it,qw),e(it,dn),e(dn,jw),e(dn,lf),e(lf,Cw),e(dn,Nw),e(dn,df),e(df,Ow),e(dn,Iw),e(it,Aw),e(it,ji),e(ji,Lw),e(ji,Hp),e(Hp,Dw),e(ji,Sw),e(it,Uw),e(it,Ci),e(Ci,Ww),e(Ci,Ni),e(Ni,Hw),e(Ci,Rw),e(it,Vw),e(it,Dt),w(Oi,Dt,null),e(Dt,Qw),e(Dt,cn),e(cn,Kw),e(cn,Rp),e(Rp,Jw),e(cn,Gw),e(cn,cf),e(cf,Xw),e(cn,Yw),e(Dt,Zw),w(qs,Dt,null),e(Dt,e2),w(js,Dt,null),b(o,Wb,k),b(o,pn,k),e(pn,Cs),e(Cs,pf),w(Ii,pf,null),e(pn,t2),e(pn,hf),e(hf,o2),b(o,Hb,k),b(o,lt,k),w(Ai,lt,null),e(lt,n2),e(lt,Li),e(Li,s2),e(Li,mf),e(mf,r2),e(Li,a2),e(lt,i2),e(lt,Di),e(Di,l2),e(Di,Vp),e(Vp,d2),e(Di,c2),e(lt,p2),e(lt,Si),e(Si,h2),e(Si,Ui),e(Ui,m2),e(Si,f2),e(lt,u2),e(lt,St),w(Wi,St,null),e(St,g2),e(St,hn),e(hn,_2),e(hn,Qp),e(Qp,b2),e(hn,k2),e(hn,ff),e(ff,T2),e(hn,v2),e(St,y2),w(Ns,St,null),e(St,w2),w(Os,St,null),b(o,Rb,k),b(o,mn,k),e(mn,Is),e(Is,uf),w(Hi,uf,null),e(mn,$2),e(mn,gf),e(gf,x2),b(o,Vb,k),b(o,dt,k),w(Ri,dt,null),e(dt,F2),e(dt,Vi),e(Vi,B2),e(Vi,_f),e(_f,E2),e(Vi,M2),e(dt,z2),e(dt,Qi),e(Qi,P2),e(Qi,Kp),e(Kp,q2),e(Qi,j2),e(dt,C2),e(dt,Ki),e(Ki,N2),e(Ki,Ji),e(Ji,O2),e(Ki,I2),e(dt,A2),e(dt,ut),w(Gi,ut,null),e(ut,L2),e(ut,fn),e(fn,D2),e(fn,Jp),e(Jp,S2),e(fn,U2),e(fn,bf),e(bf,W2),e(fn,H2),e(ut,R2),w(As,ut,null),e(ut,V2),w(Ls,ut,null),e(ut,Q2),w(Ds,ut,null),b(o,Qb,k),b(o,un,k),e(un,Ss),e(Ss,kf),w(Xi,kf,null),e(un,K2),e(un,Tf),e(Tf,J2),b(o,Kb,k),b(o,ct,k),w(Yi,ct,null),e(ct,G2),e(ct,Zi),e(Zi,X2),e(Zi,vf),e(vf,Y2),e(Zi,Z2),e(ct,e$),e(ct,el),e(el,t$),e(el,Gp),e(Gp,o$),e(el,n$),e(ct,s$),e(ct,tl),e(tl,r$),e(tl,ol),e(ol,a$),e(tl,i$),e(ct,l$),e(ct,Ut),w(nl,Ut,null),e(Ut,d$),e(Ut,gn),e(gn,c$),e(gn,Xp),e(Xp,p$),e(gn,h$),e(gn,yf),e(yf,m$),e(gn,f$),e(Ut,u$),w(Us,Ut,null),e(Ut,g$),w(Ws,Ut,null),b(o,Jb,k),b(o,_n,k),e(_n,Hs),e(Hs,wf),w(sl,wf,null),e(_n,_$),e(_n,$f),e($f,b$),b(o,Gb,k),b(o,pt,k),w(rl,pt,null),e(pt,k$),e(pt,xf),e(xf,T$),e(pt,v$),e(pt,al),e(al,y$),e(al,Yp),e(Yp,w$),e(al,$$),e(pt,x$),e(pt,il),e(il,F$),e(il,ll),e(ll,B$),e(il,E$),e(pt,M$),e(pt,Qe),w(dl,Qe,null),e(Qe,z$),e(Qe,bn),e(bn,P$),e(bn,Zp),e(Zp,q$),e(bn,j$),e(bn,Ff),e(Ff,C$),e(bn,N$),e(Qe,O$),w(Rs,Qe,null),e(Qe,I$),w(Vs,Qe,null),e(Qe,A$),w(Qs,Qe,null),e(Qe,L$),w(Ks,Qe,null),e(Qe,D$),w(Js,Qe,null),b(o,Xb,k),b(o,kn,k),e(kn,Gs),e(Gs,Bf),w(cl,Bf,null),e(kn,S$),e(kn,Ef),e(Ef,U$),b(o,Yb,k),b(o,ht,k),w(pl,ht,null),e(ht,W$),e(ht,Mf),e(Mf,H$),e(ht,R$),e(ht,hl),e(hl,V$),e(hl,eh),e(eh,Q$),e(hl,K$),e(ht,J$),e(ht,ml),e(ml,G$),e(ml,fl),e(fl,X$),e(ml,Y$),e(ht,Z$),e(ht,Wt),w(ul,Wt,null),e(Wt,ex),e(Wt,Tn),e(Tn,tx),e(Tn,th),e(th,ox),e(Tn,nx),e(Tn,zf),e(zf,sx),e(Tn,rx),e(Wt,ax),w(Xs,Wt,null),e(Wt,ix),w(Ys,Wt,null),b(o,Zb,k),b(o,vn,k),e(vn,Zs),e(Zs,Pf),w(gl,Pf,null),e(vn,lx),e(vn,qf),e(qf,dx),b(o,ek,k),b(o,mt,k),w(_l,mt,null),e(mt,cx),e(mt,jf),e(jf,px),e(mt,hx),e(mt,bl),e(bl,mx),e(bl,oh),e(oh,fx),e(bl,ux),e(mt,gx),e(mt,kl),e(kl,_x),e(kl,Tl),e(Tl,bx),e(kl,kx),e(mt,Tx),e(mt,gt),w(vl,gt,null),e(gt,vx),e(gt,yn),e(yn,yx),e(yn,nh),e(nh,wx),e(yn,$x),e(yn,Cf),e(Cf,xx),e(yn,Fx),e(gt,Bx),w(er,gt,null),e(gt,Ex),w(tr,gt,null),e(gt,Mx),w(or,gt,null),b(o,tk,k),b(o,wn,k),e(wn,nr),e(nr,Nf),w(yl,Nf,null),e(wn,zx),e(wn,Of),e(Of,Px),b(o,ok,k),b(o,ft,k),w(wl,ft,null),e(ft,qx),e(ft,$n),e($n,jx),e($n,If),e(If,Cx),e($n,Nx),e($n,Af),e(Af,Ox),e($n,Ix),e(ft,Ax),e(ft,$l),e($l,Lx),e($l,sh),e(sh,Dx),e($l,Sx),e(ft,Ux),e(ft,xl),e(xl,Wx),e(xl,Fl),e(Fl,Hx),e(xl,Rx),e(ft,Vx),e(ft,_t),w(Bl,_t,null),e(_t,Qx),e(_t,xn),e(xn,Kx),e(xn,rh),e(rh,Jx),e(xn,Gx),e(xn,Lf),e(Lf,Xx),e(xn,Yx),e(_t,Zx),w(sr,_t,null),e(_t,eF),w(rr,_t,null),e(_t,tF),w(ar,_t,null),b(o,nk,k),b(o,Fn,k),e(Fn,ir),e(ir,Df),w(El,Df,null),e(Fn,oF),e(Fn,Sf),e(Sf,nF),b(o,sk,k),b(o,Je,k),w(Ml,Je,null),e(Je,sF),e(Je,Uf),e(Uf,rF),e(Je,aF),e(Je,zl),e(zl,iF),e(zl,ah),e(ah,lF),e(zl,dF),e(Je,cF),e(Je,Pl),e(Pl,pF),e(Pl,ql),e(ql,hF),e(Pl,mF),e(Je,fF),w(lr,Je,null),e(Je,uF),e(Je,Ht),w(jl,Ht,null),e(Ht,gF),e(Ht,Bn),e(Bn,_F),e(Bn,ih),e(ih,bF),e(Bn,kF),e(Bn,Wf),e(Wf,TF),e(Bn,vF),e(Ht,yF),w(dr,Ht,null),e(Ht,wF),w(cr,Ht,null),b(o,rk,k),b(o,En,k),e(En,pr),e(pr,Hf),w(Cl,Hf,null),e(En,$F),e(En,Rf),e(Rf,xF),b(o,ak,k),b(o,Ge,k),w(Nl,Ge,null),e(Ge,FF),e(Ge,Mn),e(Mn,BF),e(Mn,Vf),e(Vf,EF),e(Mn,MF),e(Mn,Qf),e(Qf,zF),e(Mn,PF),e(Ge,qF),e(Ge,Ol),e(Ol,jF),e(Ol,lh),e(lh,CF),e(Ol,NF),e(Ge,OF),e(Ge,Il),e(Il,IF),e(Il,Al),e(Al,AF),e(Il,LF),e(Ge,DF),w(hr,Ge,null),e(Ge,SF),e(Ge,Rt),w(Ll,Rt,null),e(Rt,UF),e(Rt,zn),e(zn,WF),e(zn,dh),e(dh,HF),e(zn,RF),e(zn,Kf),e(Kf,VF),e(zn,QF),e(Rt,KF),w(mr,Rt,null),e(Rt,JF),w(fr,Rt,null),b(o,ik,k),b(o,Pn,k),e(Pn,ur),e(ur,Jf),w(Dl,Jf,null),e(Pn,GF),e(Pn,Gf),e(Gf,XF),b(o,lk,k),b(o,qn,k),w(Sl,qn,null),e(qn,YF),e(qn,bt),w(Ul,bt,null),e(bt,ZF),e(bt,Ie),e(Ie,e0),e(Ie,Xf),e(Xf,t0),e(Ie,o0),e(Ie,Yf),e(Yf,n0),e(Ie,s0),e(Ie,Zf),e(Zf,r0),e(Ie,a0),e(Ie,eu),e(eu,i0),e(Ie,l0),e(Ie,tu),e(tu,d0),e(Ie,c0),e(Ie,ou),e(ou,p0),e(Ie,h0),e(Ie,nu),e(nu,m0),e(Ie,f0),e(bt,u0),e(bt,Wl),e(Wl,Hl),e(Hl,g0),e(Hl,su),e(su,_0),e(Hl,b0),e(Wl,k0),e(Wl,Rl),e(Rl,T0),e(Rl,ru),e(ru,v0),e(Rl,y0),e(bt,w0),e(bt,G),e(G,$0),e(G,au),e(au,x0),e(G,F0),e(G,iu),e(iu,B0),e(G,E0),e(G,lu),e(lu,M0),e(G,z0),e(G,du),e(du,P0),e(G,q0),e(G,cu),e(cu,j0),e(G,C0),e(G,pu),e(pu,N0),e(G,O0),e(G,hu),e(hu,I0),e(G,A0),e(G,mu),e(mu,L0),e(G,D0),e(G,fu),e(fu,S0),e(G,U0),e(G,uu),e(uu,W0),e(G,H0),e(G,gu),e(gu,R0),e(G,V0),e(G,_u),e(_u,Q0),e(G,K0),e(G,bu),e(bu,J0),e(G,G0),e(G,ku),e(ku,X0),e(G,Y0),e(G,Tu),e(Tu,Z0),e(G,eB),e(G,vu),e(vu,tB),e(G,oB),e(G,yu),e(yu,nB),e(G,sB),e(G,wu),e(wu,rB),e(G,aB),e(G,$u),e($u,iB),e(G,lB),e(G,xu),e(xu,dB),e(G,cB),e(bt,pB),w(gr,bt,null),b(o,dk,k),b(o,jn,k),e(jn,_r),e(_r,Fu),w(Vl,Fu,null),e(jn,hB),e(jn,Bu),e(Bu,mB),b(o,ck,k),b(o,Xe,k),w(Ql,Xe,null),e(Xe,fB),e(Xe,Kl),e(Kl,uB),e(Kl,Eu),e(Eu,gB),e(Kl,_B),e(Xe,bB),e(Xe,Jl),e(Jl,kB),e(Jl,ch),e(ch,TB),e(Jl,vB),e(Xe,yB),e(Xe,Gl),e(Gl,wB),e(Gl,Xl),e(Xl,$B),e(Gl,xB),e(Xe,FB),w(br,Xe,null),e(Xe,BB),e(Xe,kt),w(Yl,kt,null),e(kt,EB),e(kt,Cn),e(Cn,MB),e(Cn,ph),e(ph,zB),e(Cn,PB),e(Cn,Mu),e(Mu,qB),e(Cn,jB),e(kt,CB),w(kr,kt,null),e(kt,NB),w(Tr,kt,null),e(kt,OB),w(vr,kt,null),b(o,pk,k),b(o,Nn,k),e(Nn,yr),e(yr,zu),w(Zl,zu,null),e(Nn,IB),e(Nn,Pu),e(Pu,AB),b(o,hk,k),b(o,Ye,k),w(ed,Ye,null),e(Ye,LB),e(Ye,td),e(td,DB),e(td,qu),e(qu,SB),e(td,UB),e(Ye,WB),e(Ye,od),e(od,HB),e(od,hh),e(hh,RB),e(od,VB),e(Ye,QB),e(Ye,nd),e(nd,KB),e(nd,sd),e(sd,JB),e(nd,GB),e(Ye,XB),w(wr,Ye,null),e(Ye,YB),e(Ye,Vt),w(rd,Vt,null),e(Vt,ZB),e(Vt,On),e(On,eE),e(On,mh),e(mh,tE),e(On,oE),e(On,ju),e(ju,nE),e(On,sE),e(Vt,rE),w($r,Vt,null),e(Vt,aE),w(xr,Vt,null),b(o,mk,k),b(o,In,k),e(In,Fr),e(Fr,Cu),w(ad,Cu,null),e(In,iE),e(In,Nu),e(Nu,lE),b(o,fk,k),b(o,Ze,k),w(id,Ze,null),e(Ze,dE),e(Ze,Ou),e(Ou,cE),e(Ze,pE),e(Ze,ld),e(ld,hE),e(ld,fh),e(fh,mE),e(ld,fE),e(Ze,uE),e(Ze,dd),e(dd,gE),e(dd,cd),e(cd,_E),e(dd,bE),e(Ze,kE),w(Br,Ze,null),e(Ze,TE),e(Ze,Tt),w(pd,Tt,null),e(Tt,vE),e(Tt,An),e(An,yE),e(An,uh),e(uh,wE),e(An,$E),e(An,Iu),e(Iu,xE),e(An,FE),e(Tt,BE),w(Er,Tt,null),e(Tt,EE),w(Mr,Tt,null),e(Tt,ME),w(zr,Tt,null),b(o,uk,k),b(o,Ln,k),e(Ln,Pr),e(Pr,Au),w(hd,Au,null),e(Ln,zE),e(Ln,Lu),e(Lu,PE),b(o,gk,k),b(o,et,k),w(md,et,null),e(et,qE),e(et,Du),e(Du,jE),e(et,CE),e(et,fd),e(fd,NE),e(fd,gh),e(gh,OE),e(fd,IE),e(et,AE),e(et,ud),e(ud,LE),e(ud,gd),e(gd,DE),e(ud,SE),e(et,UE),w(qr,et,null),e(et,WE),e(et,Qt),w(_d,Qt,null),e(Qt,HE),e(Qt,Dn),e(Dn,RE),e(Dn,_h),e(_h,VE),e(Dn,QE),e(Dn,Su),e(Su,KE),e(Dn,JE),e(Qt,GE),w(jr,Qt,null),e(Qt,XE),w(Cr,Qt,null),b(o,_k,k),b(o,Sn,k),e(Sn,Nr),e(Nr,Uu),w(bd,Uu,null),e(Sn,YE),e(Sn,Wu),e(Wu,ZE),b(o,bk,k),b(o,tt,k),w(kd,tt,null),e(tt,eM),e(tt,Hu),e(Hu,tM),e(tt,oM),e(tt,Td),e(Td,nM),e(Td,bh),e(bh,sM),e(Td,rM),e(tt,aM),e(tt,vd),e(vd,iM),e(vd,yd),e(yd,lM),e(vd,dM),e(tt,cM),w(Or,tt,null),e(tt,pM),e(tt,vt),w(wd,vt,null),e(vt,hM),e(vt,Un),e(Un,mM),e(Un,kh),e(kh,fM),e(Un,uM),e(Un,Ru),e(Ru,gM),e(Un,_M),e(vt,bM),w(Ir,vt,null),e(vt,kM),w(Ar,vt,null),e(vt,TM),w(Lr,vt,null),b(o,kk,k),b(o,Wn,k),e(Wn,Dr),e(Dr,Vu),w($d,Vu,null),e(Wn,vM),e(Wn,Qu),e(Qu,yM),b(o,Tk,k),b(o,ot,k),w(xd,ot,null),e(ot,wM),e(ot,Hn),e(Hn,$M),e(Hn,Ku),e(Ku,xM),e(Hn,FM),e(Hn,Ju),e(Ju,BM),e(Hn,EM),e(ot,MM),e(ot,Fd),e(Fd,zM),e(Fd,Th),e(Th,PM),e(Fd,qM),e(ot,jM),e(ot,Bd),e(Bd,CM),e(Bd,Ed),e(Ed,NM),e(Bd,OM),e(ot,IM),w(Sr,ot,null),e(ot,AM),e(ot,yt),w(Md,yt,null),e(yt,LM),e(yt,Rn),e(Rn,DM),e(Rn,vh),e(vh,SM),e(Rn,UM),e(Rn,Gu),e(Gu,WM),e(Rn,HM),e(yt,RM),w(Ur,yt,null),e(yt,VM),w(Wr,yt,null),e(yt,QM),w(Hr,yt,null),b(o,vk,k),b(o,Vn,k),e(Vn,Rr),e(Rr,Xu),w(zd,Xu,null),e(Vn,KM),e(Vn,Yu),e(Yu,JM),b(o,yk,k),b(o,Ae,k),w(Pd,Ae,null),e(Ae,GM),e(Ae,Zu),e(Zu,XM),e(Ae,YM),e(Ae,qd),e(qd,ZM),e(qd,yh),e(yh,ez),e(qd,tz),e(Ae,oz),e(Ae,jd),e(jd,nz),e(jd,Cd),e(Cd,sz),e(jd,rz),e(Ae,az),e(Ae,eg),e(eg,iz),e(Ae,lz),e(Ae,fo),e(fo,tg),e(tg,Nd),e(Nd,dz),e(fo,cz),e(fo,og),e(og,Od),e(Od,pz),e(fo,hz),e(fo,ng),e(ng,Id),e(Id,mz),e(fo,fz),e(fo,sg),e(sg,Ad),e(Ad,uz),e(Ae,gz),e(Ae,Kt),w(Ld,Kt,null),e(Kt,_z),e(Kt,Qn),e(Qn,bz),e(Qn,rg),e(rg,kz),e(Qn,Tz),e(Qn,ag),e(ag,vz),e(Qn,yz),e(Kt,wz),w(Vr,Kt,null),e(Kt,$z),w(Qr,Kt,null),b(o,wk,k),b(o,Kn,k),e(Kn,Kr),e(Kr,ig),w(Dd,ig,null),e(Kn,xz),e(Kn,lg),e(lg,Fz),b(o,$k,k),b(o,Le,k),w(Sd,Le,null),e(Le,Bz),e(Le,Jn),e(Jn,Ez),e(Jn,dg),e(dg,Mz),e(Jn,zz),e(Jn,cg),e(cg,Pz),e(Jn,qz),e(Le,jz),e(Le,Ud),e(Ud,Cz),e(Ud,wh),e(wh,Nz),e(Ud,Oz),e(Le,Iz),e(Le,Wd),e(Wd,Az),e(Wd,Hd),e(Hd,Lz),e(Wd,Dz),e(Le,Sz),e(Le,pg),e(pg,Uz),e(Le,Wz),e(Le,uo),e(uo,hg),e(hg,Rd),e(Rd,Hz),e(uo,Rz),e(uo,mg),e(mg,Vd),e(Vd,Vz),e(uo,Qz),e(uo,fg),e(fg,Qd),e(Qd,Kz),e(uo,Jz),e(uo,ug),e(ug,Kd),e(Kd,Gz),e(Le,Xz),e(Le,Jt),w(Jd,Jt,null),e(Jt,Yz),e(Jt,Gn),e(Gn,Zz),e(Gn,gg),e(gg,e4),e(Gn,t4),e(Gn,_g),e(_g,o4),e(Gn,n4),e(Jt,s4),w(Jr,Jt,null),e(Jt,r4),w(Gr,Jt,null),b(o,xk,k),b(o,Xn,k),e(Xn,Xr),e(Xr,bg),w(Gd,bg,null),e(Xn,a4),e(Xn,kg),e(kg,i4),b(o,Fk,k),b(o,De,k),w(Xd,De,null),e(De,l4),e(De,Tg),e(Tg,d4),e(De,c4),e(De,Yd),e(Yd,p4),e(Yd,$h),e($h,h4),e(Yd,m4),e(De,f4),e(De,Zd),e(Zd,u4),e(Zd,ec),e(ec,g4),e(Zd,_4),e(De,b4),e(De,vg),e(vg,k4),e(De,T4),e(De,go),e(go,yg),e(yg,tc),e(tc,v4),e(go,y4),e(go,wg),e(wg,oc),e(oc,w4),e(go,$4),e(go,$g),e($g,nc),e(nc,x4),e(go,F4),e(go,xg),e(xg,sc),e(sc,B4),e(De,E4),e(De,Gt),w(rc,Gt,null),e(Gt,M4),e(Gt,Yn),e(Yn,z4),e(Yn,Fg),e(Fg,P4),e(Yn,q4),e(Yn,Bg),e(Bg,j4),e(Yn,C4),e(Gt,N4),w(Yr,Gt,null),e(Gt,O4),w(Zr,Gt,null),b(o,Bk,k),b(o,Zn,k),e(Zn,ea),e(ea,Eg),w(ac,Eg,null),e(Zn,I4),e(Zn,Mg),e(Mg,A4),b(o,Ek,k),b(o,Se,k),w(ic,Se,null),e(Se,L4),e(Se,lc),e(lc,D4),e(lc,zg),e(zg,S4),e(lc,U4),e(Se,W4),e(Se,dc),e(dc,H4),e(dc,xh),e(xh,R4),e(dc,V4),e(Se,Q4),e(Se,cc),e(cc,K4),e(cc,pc),e(pc,J4),e(cc,G4),e(Se,X4),e(Se,Pg),e(Pg,Y4),e(Se,Z4),e(Se,_o),e(_o,qg),e(qg,hc),e(hc,eP),e(_o,tP),e(_o,jg),e(jg,mc),e(mc,oP),e(_o,nP),e(_o,Cg),e(Cg,fc),e(fc,sP),e(_o,rP),e(_o,Ng),e(Ng,uc),e(uc,aP),e(Se,iP),e(Se,Xt),w(gc,Xt,null),e(Xt,lP),e(Xt,es),e(es,dP),e(es,Og),e(Og,cP),e(es,pP),e(es,Ig),e(Ig,hP),e(es,mP),e(Xt,fP),w(ta,Xt,null),e(Xt,uP),w(oa,Xt,null),b(o,Mk,k),b(o,ts,k),e(ts,na),e(na,Ag),w(_c,Ag,null),e(ts,gP),e(ts,Lg),e(Lg,_P),b(o,zk,k),b(o,Ue,k),w(bc,Ue,null),e(Ue,bP),e(Ue,kc),e(kc,kP),e(kc,Dg),e(Dg,TP),e(kc,vP),e(Ue,yP),e(Ue,Tc),e(Tc,wP),e(Tc,Fh),e(Fh,$P),e(Tc,xP),e(Ue,FP),e(Ue,vc),e(vc,BP),e(vc,yc),e(yc,EP),e(vc,MP),e(Ue,zP),e(Ue,Sg),e(Sg,PP),e(Ue,qP),e(Ue,bo),e(bo,Ug),e(Ug,wc),e(wc,jP),e(bo,CP),e(bo,Wg),e(Wg,$c),e($c,NP),e(bo,OP),e(bo,Hg),e(Hg,xc),e(xc,IP),e(bo,AP),e(bo,Rg),e(Rg,Fc),e(Fc,LP),e(Ue,DP),e(Ue,Yt),w(Bc,Yt,null),e(Yt,SP),e(Yt,os),e(os,UP),e(os,Vg),e(Vg,WP),e(os,HP),e(os,Qg),e(Qg,RP),e(os,VP),e(Yt,QP),w(sa,Yt,null),e(Yt,KP),w(ra,Yt,null),b(o,Pk,k),b(o,ns,k),e(ns,aa),e(aa,Kg),w(Ec,Kg,null),e(ns,JP),e(ns,Jg),e(Jg,GP),b(o,qk,k),b(o,We,k),w(Mc,We,null),e(We,XP),e(We,Gg),e(Gg,YP),e(We,ZP),e(We,zc),e(zc,e8),e(zc,Bh),e(Bh,t8),e(zc,o8),e(We,n8),e(We,Pc),e(Pc,s8),e(Pc,qc),e(qc,r8),e(Pc,a8),e(We,i8),e(We,Xg),e(Xg,l8),e(We,d8),e(We,ko),e(ko,Yg),e(Yg,jc),e(jc,c8),e(ko,p8),e(ko,Zg),e(Zg,Cc),e(Cc,h8),e(ko,m8),e(ko,e_),e(e_,Nc),e(Nc,f8),e(ko,u8),e(ko,t_),e(t_,Oc),e(Oc,g8),e(We,_8),e(We,Zt),w(Ic,Zt,null),e(Zt,b8),e(Zt,ss),e(ss,k8),e(ss,o_),e(o_,T8),e(ss,v8),e(ss,n_),e(n_,y8),e(ss,w8),e(Zt,$8),w(ia,Zt,null),e(Zt,x8),w(la,Zt,null),b(o,jk,k),b(o,rs,k),e(rs,da),e(da,s_),w(Ac,s_,null),e(rs,F8),e(rs,r_),e(r_,B8),b(o,Ck,k),b(o,He,k),w(Lc,He,null),e(He,E8),e(He,a_),e(a_,M8),e(He,z8),e(He,Dc),e(Dc,P8),e(Dc,Eh),e(Eh,q8),e(Dc,j8),e(He,C8),e(He,Sc),e(Sc,N8),e(Sc,Uc),e(Uc,O8),e(Sc,I8),e(He,A8),e(He,i_),e(i_,L8),e(He,D8),e(He,To),e(To,l_),e(l_,Wc),e(Wc,S8),e(To,U8),e(To,d_),e(d_,Hc),e(Hc,W8),e(To,H8),e(To,c_),e(c_,Rc),e(Rc,R8),e(To,V8),e(To,p_),e(p_,Vc),e(Vc,Q8),e(He,K8),e(He,eo),w(Qc,eo,null),e(eo,J8),e(eo,as),e(as,G8),e(as,h_),e(h_,X8),e(as,Y8),e(as,m_),e(m_,Z8),e(as,eq),e(eo,tq),w(ca,eo,null),e(eo,oq),w(pa,eo,null),b(o,Nk,k),b(o,is,k),e(is,ha),e(ha,f_),w(Kc,f_,null),e(is,nq),e(is,u_),e(u_,sq),b(o,Ok,k),b(o,Re,k),w(Jc,Re,null),e(Re,rq),e(Re,g_),e(g_,aq),e(Re,iq),e(Re,Gc),e(Gc,lq),e(Gc,Mh),e(Mh,dq),e(Gc,cq),e(Re,pq),e(Re,Xc),e(Xc,hq),e(Xc,Yc),e(Yc,mq),e(Xc,fq),e(Re,uq),e(Re,__),e(__,gq),e(Re,_q),e(Re,vo),e(vo,b_),e(b_,Zc),e(Zc,bq),e(vo,kq),e(vo,k_),e(k_,ep),e(ep,Tq),e(vo,vq),e(vo,T_),e(T_,tp),e(tp,yq),e(vo,wq),e(vo,v_),e(v_,op),e(op,$q),e(Re,xq),e(Re,to),w(np,to,null),e(to,Fq),e(to,ls),e(ls,Bq),e(ls,y_),e(y_,Eq),e(ls,Mq),e(ls,w_),e(w_,zq),e(ls,Pq),e(to,qq),w(ma,to,null),e(to,jq),w(fa,to,null),b(o,Ik,k),b(o,ds,k),e(ds,ua),e(ua,$_),w(sp,$_,null),e(ds,Cq),e(ds,x_),e(x_,Nq),b(o,Ak,k),b(o,Ve,k),w(rp,Ve,null),e(Ve,Oq),e(Ve,cs),e(cs,Iq),e(cs,F_),e(F_,Aq),e(cs,Lq),e(cs,B_),e(B_,Dq),e(cs,Sq),e(Ve,Uq),e(Ve,ap),e(ap,Wq),e(ap,zh),e(zh,Hq),e(ap,Rq),e(Ve,Vq),e(Ve,ip),e(ip,Qq),e(ip,lp),e(lp,Kq),e(ip,Jq),e(Ve,Gq),e(Ve,E_),e(E_,Xq),e(Ve,Yq),e(Ve,yo),e(yo,M_),e(M_,dp),e(dp,Zq),e(yo,e7),e(yo,z_),e(z_,cp),e(cp,t7),e(yo,o7),e(yo,P_),e(P_,pp),e(pp,n7),e(yo,s7),e(yo,q_),e(q_,hp),e(hp,r7),e(Ve,a7),e(Ve,oo),w(mp,oo,null),e(oo,i7),e(oo,ps),e(ps,l7),e(ps,j_),e(j_,d7),e(ps,c7),e(ps,C_),e(C_,p7),e(ps,h7),e(oo,m7),w(ga,oo,null),e(oo,f7),w(_a,oo,null),Lk=!0},p(o,[k]){const fp={};k&2&&(fp.$$scope={dirty:k,ctx:o}),_s.$set(fp);const N_={};k&2&&(N_.$$scope={dirty:k,ctx:o}),Ts.$set(N_);const O_={};k&2&&(O_.$$scope={dirty:k,ctx:o}),ys.$set(O_);const I_={};k&2&&(I_.$$scope={dirty:k,ctx:o}),$s.$set(I_);const up={};k&2&&(up.$$scope={dirty:k,ctx:o}),xs.$set(up);const A_={};k&2&&(A_.$$scope={dirty:k,ctx:o}),Ms.$set(A_);const L_={};k&2&&(L_.$$scope={dirty:k,ctx:o}),zs.$set(L_);const D_={};k&2&&(D_.$$scope={dirty:k,ctx:o}),qs.$set(D_);const gp={};k&2&&(gp.$$scope={dirty:k,ctx:o}),js.$set(gp);const S_={};k&2&&(S_.$$scope={dirty:k,ctx:o}),Ns.$set(S_);const U_={};k&2&&(U_.$$scope={dirty:k,ctx:o}),Os.$set(U_);const W_={};k&2&&(W_.$$scope={dirty:k,ctx:o}),As.$set(W_);const H_={};k&2&&(H_.$$scope={dirty:k,ctx:o}),Ls.$set(H_);const R_={};k&2&&(R_.$$scope={dirty:k,ctx:o}),Ds.$set(R_);const V_={};k&2&&(V_.$$scope={dirty:k,ctx:o}),Us.$set(V_);const Q_={};k&2&&(Q_.$$scope={dirty:k,ctx:o}),Ws.$set(Q_);const _p={};k&2&&(_p.$$scope={dirty:k,ctx:o}),Rs.$set(_p);const K_={};k&2&&(K_.$$scope={dirty:k,ctx:o}),Vs.$set(K_);const J_={};k&2&&(J_.$$scope={dirty:k,ctx:o}),Qs.$set(J_);const hs={};k&2&&(hs.$$scope={dirty:k,ctx:o}),Ks.$set(hs);const G_={};k&2&&(G_.$$scope={dirty:k,ctx:o}),Js.$set(G_);const X_={};k&2&&(X_.$$scope={dirty:k,ctx:o}),Xs.$set(X_);const bp={};k&2&&(bp.$$scope={dirty:k,ctx:o}),Ys.$set(bp);const Y_={};k&2&&(Y_.$$scope={dirty:k,ctx:o}),er.$set(Y_);const Z_={};k&2&&(Z_.$$scope={dirty:k,ctx:o}),tr.$set(Z_);const eb={};k&2&&(eb.$$scope={dirty:k,ctx:o}),or.$set(eb);const wo={};k&2&&(wo.$$scope={dirty:k,ctx:o}),sr.$set(wo);const $o={};k&2&&($o.$$scope={dirty:k,ctx:o}),rr.$set($o);const tb={};k&2&&(tb.$$scope={dirty:k,ctx:o}),ar.$set(tb);const ob={};k&2&&(ob.$$scope={dirty:k,ctx:o}),lr.$set(ob);const nb={};k&2&&(nb.$$scope={dirty:k,ctx:o}),dr.$set(nb);const ms={};k&2&&(ms.$$scope={dirty:k,ctx:o}),cr.$set(ms);const sb={};k&2&&(sb.$$scope={dirty:k,ctx:o}),hr.$set(sb);const rb={};k&2&&(rb.$$scope={dirty:k,ctx:o}),mr.$set(rb);const kp={};k&2&&(kp.$$scope={dirty:k,ctx:o}),fr.$set(kp);const ab={};k&2&&(ab.$$scope={dirty:k,ctx:o}),gr.$set(ab);const ib={};k&2&&(ib.$$scope={dirty:k,ctx:o}),br.$set(ib);const lb={};k&2&&(lb.$$scope={dirty:k,ctx:o}),kr.$set(lb);const nt={};k&2&&(nt.$$scope={dirty:k,ctx:o}),Tr.$set(nt);const db={};k&2&&(db.$$scope={dirty:k,ctx:o}),vr.$set(db);const Tp={};k&2&&(Tp.$$scope={dirty:k,ctx:o}),wr.$set(Tp);const cb={};k&2&&(cb.$$scope={dirty:k,ctx:o}),$r.$set(cb);const fs={};k&2&&(fs.$$scope={dirty:k,ctx:o}),xr.$set(fs);const pb={};k&2&&(pb.$$scope={dirty:k,ctx:o}),Br.$set(pb);const vp={};k&2&&(vp.$$scope={dirty:k,ctx:o}),Er.$set(vp);const Ph={};k&2&&(Ph.$$scope={dirty:k,ctx:o}),Mr.$set(Ph);const hb={};k&2&&(hb.$$scope={dirty:k,ctx:o}),zr.$set(hb);const qh={};k&2&&(qh.$$scope={dirty:k,ctx:o}),qr.$set(qh);const mb={};k&2&&(mb.$$scope={dirty:k,ctx:o}),jr.$set(mb);const yp={};k&2&&(yp.$$scope={dirty:k,ctx:o}),Cr.$set(yp);const wp={};k&2&&(wp.$$scope={dirty:k,ctx:o}),Or.$set(wp);const fb={};k&2&&(fb.$$scope={dirty:k,ctx:o}),Ir.$set(fb);const xo={};k&2&&(xo.$$scope={dirty:k,ctx:o}),Ar.$set(xo);const ub={};k&2&&(ub.$$scope={dirty:k,ctx:o}),Lr.$set(ub);const us={};k&2&&(us.$$scope={dirty:k,ctx:o}),Sr.$set(us);const gb={};k&2&&(gb.$$scope={dirty:k,ctx:o}),Ur.$set(gb);const _b={};k&2&&(_b.$$scope={dirty:k,ctx:o}),Wr.$set(_b);const bb={};k&2&&(bb.$$scope={dirty:k,ctx:o}),Hr.$set(bb);const $p={};k&2&&($p.$$scope={dirty:k,ctx:o}),Vr.$set($p);const kb={};k&2&&(kb.$$scope={dirty:k,ctx:o}),Qr.$set(kb);const Tb={};k&2&&(Tb.$$scope={dirty:k,ctx:o}),Jr.$set(Tb);const vb={};k&2&&(vb.$$scope={dirty:k,ctx:o}),Gr.$set(vb);const Ot={};k&2&&(Ot.$$scope={dirty:k,ctx:o}),Yr.$set(Ot);const xp={};k&2&&(xp.$$scope={dirty:k,ctx:o}),Zr.$set(xp);const yb={};k&2&&(yb.$$scope={dirty:k,ctx:o}),ta.$set(yb);const Fp={};k&2&&(Fp.$$scope={dirty:k,ctx:o}),oa.$set(Fp);const wb={};k&2&&(wb.$$scope={dirty:k,ctx:o}),sa.$set(wb);const gs={};k&2&&(gs.$$scope={dirty:k,ctx:o}),ra.$set(gs);const $b={};k&2&&($b.$$scope={dirty:k,ctx:o}),ia.$set($b);const Bp={};k&2&&(Bp.$$scope={dirty:k,ctx:o}),la.$set(Bp);const jh={};k&2&&(jh.$$scope={dirty:k,ctx:o}),ca.$set(jh);const xb={};k&2&&(xb.$$scope={dirty:k,ctx:o}),pa.$set(xb);const Ch={};k&2&&(Ch.$$scope={dirty:k,ctx:o}),ma.$set(Ch);const Fb={};k&2&&(Fb.$$scope={dirty:k,ctx:o}),fa.$set(Fb);const Fo={};k&2&&(Fo.$$scope={dirty:k,ctx:o}),ga.$set(Fo);const Bb={};k&2&&(Bb.$$scope={dirty:k,ctx:o}),_a.$set(Bb)},i(o){Lk||($(l.$$.fragment,o),$(re.$$.fragment,o),$(Me.$$.fragment,o),$(Ra.$$.fragment,o),$(_s.$$.fragment,o),$(Qa.$$.fragment,o),$(Ka.$$.fragment,o),$(Ga.$$.fragment,o),$(Ya.$$.fragment,o),$(ei.$$.fragment,o),$(Ts.$$.fragment,o),$(ti.$$.fragment,o),$(oi.$$.fragment,o),$(ni.$$.fragment,o),$(ai.$$.fragment,o),$(li.$$.fragment,o),$(ys.$$.fragment,o),$(di.$$.fragment,o),$(ci.$$.fragment,o),$(hi.$$.fragment,o),$($s.$$.fragment,o),$(fi.$$.fragment,o),$(xs.$$.fragment,o),$(ui.$$.fragment,o),$(gi.$$.fragment,o),$(bi.$$.fragment,o),$(Ti.$$.fragment,o),$(yi.$$.fragment,o),$(wi.$$.fragment,o),$($i.$$.fragment,o),$(zi.$$.fragment,o),$(Ms.$$.fragment,o),$(zs.$$.fragment,o),$(Pi.$$.fragment,o),$(qi.$$.fragment,o),$(Oi.$$.fragment,o),$(qs.$$.fragment,o),$(js.$$.fragment,o),$(Ii.$$.fragment,o),$(Ai.$$.fragment,o),$(Wi.$$.fragment,o),$(Ns.$$.fragment,o),$(Os.$$.fragment,o),$(Hi.$$.fragment,o),$(Ri.$$.fragment,o),$(Gi.$$.fragment,o),$(As.$$.fragment,o),$(Ls.$$.fragment,o),$(Ds.$$.fragment,o),$(Xi.$$.fragment,o),$(Yi.$$.fragment,o),$(nl.$$.fragment,o),$(Us.$$.fragment,o),$(Ws.$$.fragment,o),$(sl.$$.fragment,o),$(rl.$$.fragment,o),$(dl.$$.fragment,o),$(Rs.$$.fragment,o),$(Vs.$$.fragment,o),$(Qs.$$.fragment,o),$(Ks.$$.fragment,o),$(Js.$$.fragment,o),$(cl.$$.fragment,o),$(pl.$$.fragment,o),$(ul.$$.fragment,o),$(Xs.$$.fragment,o),$(Ys.$$.fragment,o),$(gl.$$.fragment,o),$(_l.$$.fragment,o),$(vl.$$.fragment,o),$(er.$$.fragment,o),$(tr.$$.fragment,o),$(or.$$.fragment,o),$(yl.$$.fragment,o),$(wl.$$.fragment,o),$(Bl.$$.fragment,o),$(sr.$$.fragment,o),$(rr.$$.fragment,o),$(ar.$$.fragment,o),$(El.$$.fragment,o),$(Ml.$$.fragment,o),$(lr.$$.fragment,o),$(jl.$$.fragment,o),$(dr.$$.fragment,o),$(cr.$$.fragment,o),$(Cl.$$.fragment,o),$(Nl.$$.fragment,o),$(hr.$$.fragment,o),$(Ll.$$.fragment,o),$(mr.$$.fragment,o),$(fr.$$.fragment,o),$(Dl.$$.fragment,o),$(Sl.$$.fragment,o),$(Ul.$$.fragment,o),$(gr.$$.fragment,o),$(Vl.$$.fragment,o),$(Ql.$$.fragment,o),$(br.$$.fragment,o),$(Yl.$$.fragment,o),$(kr.$$.fragment,o),$(Tr.$$.fragment,o),$(vr.$$.fragment,o),$(Zl.$$.fragment,o),$(ed.$$.fragment,o),$(wr.$$.fragment,o),$(rd.$$.fragment,o),$($r.$$.fragment,o),$(xr.$$.fragment,o),$(ad.$$.fragment,o),$(id.$$.fragment,o),$(Br.$$.fragment,o),$(pd.$$.fragment,o),$(Er.$$.fragment,o),$(Mr.$$.fragment,o),$(zr.$$.fragment,o),$(hd.$$.fragment,o),$(md.$$.fragment,o),$(qr.$$.fragment,o),$(_d.$$.fragment,o),$(jr.$$.fragment,o),$(Cr.$$.fragment,o),$(bd.$$.fragment,o),$(kd.$$.fragment,o),$(Or.$$.fragment,o),$(wd.$$.fragment,o),$(Ir.$$.fragment,o),$(Ar.$$.fragment,o),$(Lr.$$.fragment,o),$($d.$$.fragment,o),$(xd.$$.fragment,o),$(Sr.$$.fragment,o),$(Md.$$.fragment,o),$(Ur.$$.fragment,o),$(Wr.$$.fragment,o),$(Hr.$$.fragment,o),$(zd.$$.fragment,o),$(Pd.$$.fragment,o),$(Ld.$$.fragment,o),$(Vr.$$.fragment,o),$(Qr.$$.fragment,o),$(Dd.$$.fragment,o),$(Sd.$$.fragment,o),$(Jd.$$.fragment,o),$(Jr.$$.fragment,o),$(Gr.$$.fragment,o),$(Gd.$$.fragment,o),$(Xd.$$.fragment,o),$(rc.$$.fragment,o),$(Yr.$$.fragment,o),$(Zr.$$.fragment,o),$(ac.$$.fragment,o),$(ic.$$.fragment,o),$(gc.$$.fragment,o),$(ta.$$.fragment,o),$(oa.$$.fragment,o),$(_c.$$.fragment,o),$(bc.$$.fragment,o),$(Bc.$$.fragment,o),$(sa.$$.fragment,o),$(ra.$$.fragment,o),$(Ec.$$.fragment,o),$(Mc.$$.fragment,o),$(Ic.$$.fragment,o),$(ia.$$.fragment,o),$(la.$$.fragment,o),$(Ac.$$.fragment,o),$(Lc.$$.fragment,o),$(Qc.$$.fragment,o),$(ca.$$.fragment,o),$(pa.$$.fragment,o),$(Kc.$$.fragment,o),$(Jc.$$.fragment,o),$(np.$$.fragment,o),$(ma.$$.fragment,o),$(fa.$$.fragment,o),$(sp.$$.fragment,o),$(rp.$$.fragment,o),$(mp.$$.fragment,o),$(ga.$$.fragment,o),$(_a.$$.fragment,o),Lk=!0)},o(o){x(l.$$.fragment,o),x(re.$$.fragment,o),x(Me.$$.fragment,o),x(Ra.$$.fragment,o),x(_s.$$.fragment,o),x(Qa.$$.fragment,o),x(Ka.$$.fragment,o),x(Ga.$$.fragment,o),x(Ya.$$.fragment,o),x(ei.$$.fragment,o),x(Ts.$$.fragment,o),x(ti.$$.fragment,o),x(oi.$$.fragment,o),x(ni.$$.fragment,o),x(ai.$$.fragment,o),x(li.$$.fragment,o),x(ys.$$.fragment,o),x(di.$$.fragment,o),x(ci.$$.fragment,o),x(hi.$$.fragment,o),x($s.$$.fragment,o),x(fi.$$.fragment,o),x(xs.$$.fragment,o),x(ui.$$.fragment,o),x(gi.$$.fragment,o),x(bi.$$.fragment,o),x(Ti.$$.fragment,o),x(yi.$$.fragment,o),x(wi.$$.fragment,o),x($i.$$.fragment,o),x(zi.$$.fragment,o),x(Ms.$$.fragment,o),x(zs.$$.fragment,o),x(Pi.$$.fragment,o),x(qi.$$.fragment,o),x(Oi.$$.fragment,o),x(qs.$$.fragment,o),x(js.$$.fragment,o),x(Ii.$$.fragment,o),x(Ai.$$.fragment,o),x(Wi.$$.fragment,o),x(Ns.$$.fragment,o),x(Os.$$.fragment,o),x(Hi.$$.fragment,o),x(Ri.$$.fragment,o),x(Gi.$$.fragment,o),x(As.$$.fragment,o),x(Ls.$$.fragment,o),x(Ds.$$.fragment,o),x(Xi.$$.fragment,o),x(Yi.$$.fragment,o),x(nl.$$.fragment,o),x(Us.$$.fragment,o),x(Ws.$$.fragment,o),x(sl.$$.fragment,o),x(rl.$$.fragment,o),x(dl.$$.fragment,o),x(Rs.$$.fragment,o),x(Vs.$$.fragment,o),x(Qs.$$.fragment,o),x(Ks.$$.fragment,o),x(Js.$$.fragment,o),x(cl.$$.fragment,o),x(pl.$$.fragment,o),x(ul.$$.fragment,o),x(Xs.$$.fragment,o),x(Ys.$$.fragment,o),x(gl.$$.fragment,o),x(_l.$$.fragment,o),x(vl.$$.fragment,o),x(er.$$.fragment,o),x(tr.$$.fragment,o),x(or.$$.fragment,o),x(yl.$$.fragment,o),x(wl.$$.fragment,o),x(Bl.$$.fragment,o),x(sr.$$.fragment,o),x(rr.$$.fragment,o),x(ar.$$.fragment,o),x(El.$$.fragment,o),x(Ml.$$.fragment,o),x(lr.$$.fragment,o),x(jl.$$.fragment,o),x(dr.$$.fragment,o),x(cr.$$.fragment,o),x(Cl.$$.fragment,o),x(Nl.$$.fragment,o),x(hr.$$.fragment,o),x(Ll.$$.fragment,o),x(mr.$$.fragment,o),x(fr.$$.fragment,o),x(Dl.$$.fragment,o),x(Sl.$$.fragment,o),x(Ul.$$.fragment,o),x(gr.$$.fragment,o),x(Vl.$$.fragment,o),x(Ql.$$.fragment,o),x(br.$$.fragment,o),x(Yl.$$.fragment,o),x(kr.$$.fragment,o),x(Tr.$$.fragment,o),x(vr.$$.fragment,o),x(Zl.$$.fragment,o),x(ed.$$.fragment,o),x(wr.$$.fragment,o),x(rd.$$.fragment,o),x($r.$$.fragment,o),x(xr.$$.fragment,o),x(ad.$$.fragment,o),x(id.$$.fragment,o),x(Br.$$.fragment,o),x(pd.$$.fragment,o),x(Er.$$.fragment,o),x(Mr.$$.fragment,o),x(zr.$$.fragment,o),x(hd.$$.fragment,o),x(md.$$.fragment,o),x(qr.$$.fragment,o),x(_d.$$.fragment,o),x(jr.$$.fragment,o),x(Cr.$$.fragment,o),x(bd.$$.fragment,o),x(kd.$$.fragment,o),x(Or.$$.fragment,o),x(wd.$$.fragment,o),x(Ir.$$.fragment,o),x(Ar.$$.fragment,o),x(Lr.$$.fragment,o),x($d.$$.fragment,o),x(xd.$$.fragment,o),x(Sr.$$.fragment,o),x(Md.$$.fragment,o),x(Ur.$$.fragment,o),x(Wr.$$.fragment,o),x(Hr.$$.fragment,o),x(zd.$$.fragment,o),x(Pd.$$.fragment,o),x(Ld.$$.fragment,o),x(Vr.$$.fragment,o),x(Qr.$$.fragment,o),x(Dd.$$.fragment,o),x(Sd.$$.fragment,o),x(Jd.$$.fragment,o),x(Jr.$$.fragment,o),x(Gr.$$.fragment,o),x(Gd.$$.fragment,o),x(Xd.$$.fragment,o),x(rc.$$.fragment,o),x(Yr.$$.fragment,o),x(Zr.$$.fragment,o),x(ac.$$.fragment,o),x(ic.$$.fragment,o),x(gc.$$.fragment,o),x(ta.$$.fragment,o),x(oa.$$.fragment,o),x(_c.$$.fragment,o),x(bc.$$.fragment,o),x(Bc.$$.fragment,o),x(sa.$$.fragment,o),x(ra.$$.fragment,o),x(Ec.$$.fragment,o),x(Mc.$$.fragment,o),x(Ic.$$.fragment,o),x(ia.$$.fragment,o),x(la.$$.fragment,o),x(Ac.$$.fragment,o),x(Lc.$$.fragment,o),x(Qc.$$.fragment,o),x(ca.$$.fragment,o),x(pa.$$.fragment,o),x(Kc.$$.fragment,o),x(Jc.$$.fragment,o),x(np.$$.fragment,o),x(ma.$$.fragment,o),x(fa.$$.fragment,o),x(sp.$$.fragment,o),x(rp.$$.fragment,o),x(mp.$$.fragment,o),x(ga.$$.fragment,o),x(_a.$$.fragment,o),Lk=!1},d(o){t(d),o&&t(_),o&&t(m),F(l),o&&t(X),o&&t(z),F(re),o&&t(me),o&&t(J),o&&t(j),o&&t(ie),o&&t(fe),o&&t(le),o&&t(ue),o&&t(q),o&&t(ge),o&&t(de),o&&t(_e),o&&t(se),o&&t(M),o&&t(K),o&&t(W),o&&t(xe),F(Me),o&&t(Eb),o&&t(Nt),F(Ra),F(_s),o&&t(Mb),o&&t(Jo),F(Qa),o&&t(zb),o&&t(Ne),F(Ka),F(Ga),F(Ya),F(ei),F(Ts),F(ti),o&&t(Pb),o&&t(Xo),F(oi),o&&t(qb),o&&t(rt),F(ni),F(ai),F(li),F(ys),o&&t(jb),o&&t(Zo),F(di),o&&t(Cb),o&&t(at),F(ci),F(hi),F($s),F(fi),F(xs),o&&t(Nb),o&&t(on),F(ui),o&&t(Ob),o&&t(nn),F(gi),o&&t(Ib),o&&t(sn),F(bi),o&&t(Ab),o&&t(mo),F(Ti),F(yi),o&&t(Lb),o&&t(rn),F(wi),o&&t(Db),o&&t(Oe),F($i),F(zi),F(Ms),F(zs),o&&t(Sb),o&&t(ln),F(Pi),o&&t(Ub),o&&t(it),F(qi),F(Oi),F(qs),F(js),o&&t(Wb),o&&t(pn),F(Ii),o&&t(Hb),o&&t(lt),F(Ai),F(Wi),F(Ns),F(Os),o&&t(Rb),o&&t(mn),F(Hi),o&&t(Vb),o&&t(dt),F(Ri),F(Gi),F(As),F(Ls),F(Ds),o&&t(Qb),o&&t(un),F(Xi),o&&t(Kb),o&&t(ct),F(Yi),F(nl),F(Us),F(Ws),o&&t(Jb),o&&t(_n),F(sl),o&&t(Gb),o&&t(pt),F(rl),F(dl),F(Rs),F(Vs),F(Qs),F(Ks),F(Js),o&&t(Xb),o&&t(kn),F(cl),o&&t(Yb),o&&t(ht),F(pl),F(ul),F(Xs),F(Ys),o&&t(Zb),o&&t(vn),F(gl),o&&t(ek),o&&t(mt),F(_l),F(vl),F(er),F(tr),F(or),o&&t(tk),o&&t(wn),F(yl),o&&t(ok),o&&t(ft),F(wl),F(Bl),F(sr),F(rr),F(ar),o&&t(nk),o&&t(Fn),F(El),o&&t(sk),o&&t(Je),F(Ml),F(lr),F(jl),F(dr),F(cr),o&&t(rk),o&&t(En),F(Cl),o&&t(ak),o&&t(Ge),F(Nl),F(hr),F(Ll),F(mr),F(fr),o&&t(ik),o&&t(Pn),F(Dl),o&&t(lk),o&&t(qn),F(Sl),F(Ul),F(gr),o&&t(dk),o&&t(jn),F(Vl),o&&t(ck),o&&t(Xe),F(Ql),F(br),F(Yl),F(kr),F(Tr),F(vr),o&&t(pk),o&&t(Nn),F(Zl),o&&t(hk),o&&t(Ye),F(ed),F(wr),F(rd),F($r),F(xr),o&&t(mk),o&&t(In),F(ad),o&&t(fk),o&&t(Ze),F(id),F(Br),F(pd),F(Er),F(Mr),F(zr),o&&t(uk),o&&t(Ln),F(hd),o&&t(gk),o&&t(et),F(md),F(qr),F(_d),F(jr),F(Cr),o&&t(_k),o&&t(Sn),F(bd),o&&t(bk),o&&t(tt),F(kd),F(Or),F(wd),F(Ir),F(Ar),F(Lr),o&&t(kk),o&&t(Wn),F($d),o&&t(Tk),o&&t(ot),F(xd),F(Sr),F(Md),F(Ur),F(Wr),F(Hr),o&&t(vk),o&&t(Vn),F(zd),o&&t(yk),o&&t(Ae),F(Pd),F(Ld),F(Vr),F(Qr),o&&t(wk),o&&t(Kn),F(Dd),o&&t($k),o&&t(Le),F(Sd),F(Jd),F(Jr),F(Gr),o&&t(xk),o&&t(Xn),F(Gd),o&&t(Fk),o&&t(De),F(Xd),F(rc),F(Yr),F(Zr),o&&t(Bk),o&&t(Zn),F(ac),o&&t(Ek),o&&t(Se),F(ic),F(gc),F(ta),F(oa),o&&t(Mk),o&&t(ts),F(_c),o&&t(zk),o&&t(Ue),F(bc),F(Bc),F(sa),F(ra),o&&t(Pk),o&&t(ns),F(Ec),o&&t(qk),o&&t(We),F(Mc),F(Ic),F(ia),F(la),o&&t(jk),o&&t(rs),F(Ac),o&&t(Ck),o&&t(He),F(Lc),F(Qc),F(ca),F(pa),o&&t(Nk),o&&t(is),F(Kc),o&&t(Ok),o&&t(Re),F(Jc),F(np),F(ma),F(fa),o&&t(Ik),o&&t(ds),F(sp),o&&t(Ak),o&&t(Ve),F(rp),F(mp),F(ga),F(_a)}}}const _O={local:"bert",sections:[{local:"overview",title:"Overview"},{local:"transformers.BertConfig",title:"BertConfig"},{local:"transformers.BertTokenizer",title:"BertTokenizer"},{local:"transformers.BertTokenizerFast",title:"BertTokenizerFast"},{local:"transformers.TFBertTokenizer",title:"TFBertTokenizer"},{local:"transformers.models.bert.modeling_bert.BertForPreTrainingOutput",title:"Bert specific outputs"},{local:"transformers.BertModel",title:"BertModel"},{local:"transformers.BertForPreTraining",title:"BertForPreTraining"},{local:"transformers.BertLMHeadModel",title:"BertLMHeadModel"},{local:"transformers.BertForMaskedLM",title:"BertForMaskedLM"},{local:"transformers.BertForNextSentencePrediction",title:"BertForNextSentencePrediction"},{local:"transformers.BertForSequenceClassification",title:"BertForSequenceClassification"},{local:"transformers.BertForMultipleChoice",title:"BertForMultipleChoice"},{local:"transformers.BertForTokenClassification",title:"BertForTokenClassification"},{local:"transformers.BertForQuestionAnswering",title:"BertForQuestionAnswering"},{local:"transformers.TFBertModel",title:"TFBertModel"},{local:"transformers.TFBertForPreTraining",title:"TFBertForPreTraining"},{local:"transformers.TFBertLMHeadModel",title:"TFBertModelLMHeadModel"},{local:"transformers.TFBertForMaskedLM",title:"TFBertForMaskedLM"},{local:"transformers.TFBertForNextSentencePrediction",title:"TFBertForNextSentencePrediction"},{local:"transformers.TFBertForSequenceClassification",title:"TFBertForSequenceClassification"},{local:"transformers.TFBertForMultipleChoice",title:"TFBertForMultipleChoice"},{local:"transformers.TFBertForTokenClassification",title:"TFBertForTokenClassification"},{local:"transformers.TFBertForQuestionAnswering",title:"TFBertForQuestionAnswering"},{local:"transformers.FlaxBertModel",title:"FlaxBertModel"},{local:"transformers.FlaxBertForPreTraining",title:"FlaxBertForPreTraining"},{local:"transformers.FlaxBertForCausalLM",title:"FlaxBertForCausalLM"},{local:"transformers.FlaxBertForMaskedLM",title:"FlaxBertForMaskedLM"},{local:"transformers.FlaxBertForNextSentencePrediction",title:"FlaxBertForNextSentencePrediction"},{local:"transformers.FlaxBertForSequenceClassification",title:"FlaxBertForSequenceClassification"},{local:"transformers.FlaxBertForMultipleChoice",title:"FlaxBertForMultipleChoice"},{local:"transformers.FlaxBertForTokenClassification",title:"FlaxBertForTokenClassification"},{local:"transformers.FlaxBertForQuestionAnswering",title:"FlaxBertForQuestionAnswering"}],title:"BERT"};function bO(B){return Q6(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class xO extends W6{constructor(d){super();H6(this,d,bO,gO,R6,{})}}export{xO as default,_O as metadata};
