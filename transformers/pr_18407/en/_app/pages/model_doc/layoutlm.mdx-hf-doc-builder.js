import{S as fg,i as gg,s as _g,e as a,k as c,w as L,t as n,M as yg,c as r,d as t,m as p,a as i,x as w,h as s,b as u,G as e,g,y as T,q as v,o as M,B as $,v as bg,L as ut}from"../../chunks/vendor-hf-doc-builder.js";import{T as Ue}from"../../chunks/Tip-hf-doc-builder.js";import{D as _e}from"../../chunks/Docstring-hf-doc-builder.js";import{C as Ke}from"../../chunks/CodeBlock-hf-doc-builder.js";import{I as De}from"../../chunks/IconCopyLink-hf-doc-builder.js";import{E as pt}from"../../chunks/ExampleCodeBlock-hf-doc-builder.js";function kg(z){let d,k,_,h,b;return h=new Ke({props:{code:`from transformers import LayoutLMModel, LayoutLMConfig

# Initializing a LayoutLM configuration
configuration = LayoutLMConfig()

# Initializing a model from the configuration
model = LayoutLMModel(configuration)

# Accessing the model configuration
configuration = model.config`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> LayoutLMModel, LayoutLMConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a LayoutLM configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = LayoutLMConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LayoutLMModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`}}),{c(){d=a("p"),k=n("Examples:"),_=c(),L(h.$$.fragment)},l(l){d=r(l,"P",{});var m=i(d);k=s(m,"Examples:"),m.forEach(t),_=p(l),w(h.$$.fragment,l)},m(l,m){g(l,d,m),e(d,k),g(l,_,m),T(h,l,m),b=!0},p:ut,i(l){b||(v(h.$$.fragment,l),b=!0)},o(l){M(h.$$.fragment,l),b=!1},d(l){l&&t(d),l&&t(_),$(h,l)}}}function Lg(z){let d,k,_,h,b;return{c(){d=a("p"),k=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),_=a("code"),h=n("Module"),b=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=r(l,"P",{});var m=i(d);k=s(m,"Although the recipe for forward pass needs to be defined within this function, one should call the "),_=r(m,"CODE",{});var j=i(_);h=s(j,"Module"),j.forEach(t),b=s(m,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),m.forEach(t)},m(l,m){g(l,d,m),e(d,k),e(d,_),e(_,h),e(d,b)},d(l){l&&t(d)}}}function wg(z){let d,k,_,h,b;return h=new Ke({props:{code:`from transformers import AutoTokenizer, LayoutLMModel
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
model = LayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased")

words = ["Hello", "world"]
normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

token_boxes = []
for word, box in zip(words, normalized_word_boxes):
    word_tokens = tokenizer.tokenize(word)
    token_boxes.extend([box] * len(word_tokens))
# add bounding boxes of cls + sep tokens
token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

encoding = tokenizer(" ".join(words), return_tensors="pt")
input_ids = encoding["input_ids"]
attention_mask = encoding["attention_mask"]
token_type_ids = encoding["token_type_ids"]
bbox = torch.tensor([token_boxes])

outputs = model(
    input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids
)

last_hidden_states = outputs.last_hidden_state`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, LayoutLMModel
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlm-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LayoutLMModel.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlm-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>words = [<span class="hljs-string">&quot;Hello&quot;</span>, <span class="hljs-string">&quot;world&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>normalized_word_boxes = [<span class="hljs-number">637</span>, <span class="hljs-number">773</span>, <span class="hljs-number">693</span>, <span class="hljs-number">782</span>], [<span class="hljs-number">698</span>, <span class="hljs-number">773</span>, <span class="hljs-number">733</span>, <span class="hljs-number">782</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>token_boxes = []
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> word, box <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(words, normalized_word_boxes):
<span class="hljs-meta">... </span>    word_tokens = tokenizer.tokenize(word)
<span class="hljs-meta">... </span>    token_boxes.extend([box] * <span class="hljs-built_in">len</span>(word_tokens))
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># add bounding boxes of cls + sep tokens</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>token_boxes = [[<span class="hljs-number">0</span>, <span class="hljs-number">0</span>, <span class="hljs-number">0</span>, <span class="hljs-number">0</span>]] + token_boxes + [[<span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>]]

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer(<span class="hljs-string">&quot; &quot;</span>.join(words), return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = encoding[<span class="hljs-string">&quot;input_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>attention_mask = encoding[<span class="hljs-string">&quot;attention_mask&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>token_type_ids = encoding[<span class="hljs-string">&quot;token_type_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>bbox = torch.tensor([token_boxes])

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(
<span class="hljs-meta">... </span>    input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`}}),{c(){d=a("p"),k=n("Examples:"),_=c(),L(h.$$.fragment)},l(l){d=r(l,"P",{});var m=i(d);k=s(m,"Examples:"),m.forEach(t),_=p(l),w(h.$$.fragment,l)},m(l,m){g(l,d,m),e(d,k),g(l,_,m),T(h,l,m),b=!0},p:ut,i(l){b||(v(h.$$.fragment,l),b=!0)},o(l){M(h.$$.fragment,l),b=!1},d(l){l&&t(d),l&&t(_),$(h,l)}}}function Tg(z){let d,k,_,h,b;return{c(){d=a("p"),k=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),_=a("code"),h=n("Module"),b=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=r(l,"P",{});var m=i(d);k=s(m,"Although the recipe for forward pass needs to be defined within this function, one should call the "),_=r(m,"CODE",{});var j=i(_);h=s(j,"Module"),j.forEach(t),b=s(m,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),m.forEach(t)},m(l,m){g(l,d,m),e(d,k),e(d,_),e(_,h),e(d,b)},d(l){l&&t(d)}}}function vg(z){let d,k,_,h,b;return h=new Ke({props:{code:`from transformers import LayoutLMTokenizer, LayoutLMForMaskedLM
import torch

tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
model = LayoutLMForMaskedLM.from_pretrained("microsoft/layoutlm-base-uncased")

words = ["Hello", "[MASK]"]
normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

token_boxes = []
for word, box in zip(words, normalized_word_boxes):
    word_tokens = tokenizer.tokenize(word)
    token_boxes.extend([box] * len(word_tokens))
# add bounding boxes of cls + sep tokens
token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

encoding = tokenizer(" ".join(words), return_tensors="pt")
input_ids = encoding["input_ids"]
attention_mask = encoding["attention_mask"]
token_type_ids = encoding["token_type_ids"]
bbox = torch.tensor([token_boxes])

labels = tokenizer("Hello world", return_tensors="pt")["input_ids"]

outputs = model(
    input_ids=input_ids,
    bbox=bbox,
    attention_mask=attention_mask,
    token_type_ids=token_type_ids,
    labels=labels,
)

loss = outputs.loss`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> LayoutLMTokenizer, LayoutLMForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = LayoutLMTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlm-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LayoutLMForMaskedLM.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlm-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>words = [<span class="hljs-string">&quot;Hello&quot;</span>, <span class="hljs-string">&quot;[MASK]&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>normalized_word_boxes = [<span class="hljs-number">637</span>, <span class="hljs-number">773</span>, <span class="hljs-number">693</span>, <span class="hljs-number">782</span>], [<span class="hljs-number">698</span>, <span class="hljs-number">773</span>, <span class="hljs-number">733</span>, <span class="hljs-number">782</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>token_boxes = []
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> word, box <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(words, normalized_word_boxes):
<span class="hljs-meta">... </span>    word_tokens = tokenizer.tokenize(word)
<span class="hljs-meta">... </span>    token_boxes.extend([box] * <span class="hljs-built_in">len</span>(word_tokens))
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># add bounding boxes of cls + sep tokens</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>token_boxes = [[<span class="hljs-number">0</span>, <span class="hljs-number">0</span>, <span class="hljs-number">0</span>, <span class="hljs-number">0</span>]] + token_boxes + [[<span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>]]

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer(<span class="hljs-string">&quot; &quot;</span>.join(words), return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = encoding[<span class="hljs-string">&quot;input_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>attention_mask = encoding[<span class="hljs-string">&quot;attention_mask&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>token_type_ids = encoding[<span class="hljs-string">&quot;token_type_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>bbox = torch.tensor([token_boxes])

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = tokenizer(<span class="hljs-string">&quot;Hello world&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(
<span class="hljs-meta">... </span>    input_ids=input_ids,
<span class="hljs-meta">... </span>    bbox=bbox,
<span class="hljs-meta">... </span>    attention_mask=attention_mask,
<span class="hljs-meta">... </span>    token_type_ids=token_type_ids,
<span class="hljs-meta">... </span>    labels=labels,
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss`}}),{c(){d=a("p"),k=n("Examples:"),_=c(),L(h.$$.fragment)},l(l){d=r(l,"P",{});var m=i(d);k=s(m,"Examples:"),m.forEach(t),_=p(l),w(h.$$.fragment,l)},m(l,m){g(l,d,m),e(d,k),g(l,_,m),T(h,l,m),b=!0},p:ut,i(l){b||(v(h.$$.fragment,l),b=!0)},o(l){M(h.$$.fragment,l),b=!1},d(l){l&&t(d),l&&t(_),$(h,l)}}}function Mg(z){let d,k,_,h,b;return{c(){d=a("p"),k=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),_=a("code"),h=n("Module"),b=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=r(l,"P",{});var m=i(d);k=s(m,"Although the recipe for forward pass needs to be defined within this function, one should call the "),_=r(m,"CODE",{});var j=i(_);h=s(j,"Module"),j.forEach(t),b=s(m,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),m.forEach(t)},m(l,m){g(l,d,m),e(d,k),e(d,_),e(_,h),e(d,b)},d(l){l&&t(d)}}}function $g(z){let d,k,_,h,b;return h=new Ke({props:{code:`from transformers import LayoutLMTokenizer, LayoutLMForSequenceClassification
import torch

tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
model = LayoutLMForSequenceClassification.from_pretrained("microsoft/layoutlm-base-uncased")

words = ["Hello", "world"]
normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

token_boxes = []
for word, box in zip(words, normalized_word_boxes):
    word_tokens = tokenizer.tokenize(word)
    token_boxes.extend([box] * len(word_tokens))
# add bounding boxes of cls + sep tokens
token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

encoding = tokenizer(" ".join(words), return_tensors="pt")
input_ids = encoding["input_ids"]
attention_mask = encoding["attention_mask"]
token_type_ids = encoding["token_type_ids"]
bbox = torch.tensor([token_boxes])
sequence_label = torch.tensor([1])

outputs = model(
    input_ids=input_ids,
    bbox=bbox,
    attention_mask=attention_mask,
    token_type_ids=token_type_ids,
    labels=sequence_label,
)

loss = outputs.loss
logits = outputs.logits`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> LayoutLMTokenizer, LayoutLMForSequenceClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = LayoutLMTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlm-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LayoutLMForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlm-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>words = [<span class="hljs-string">&quot;Hello&quot;</span>, <span class="hljs-string">&quot;world&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>normalized_word_boxes = [<span class="hljs-number">637</span>, <span class="hljs-number">773</span>, <span class="hljs-number">693</span>, <span class="hljs-number">782</span>], [<span class="hljs-number">698</span>, <span class="hljs-number">773</span>, <span class="hljs-number">733</span>, <span class="hljs-number">782</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>token_boxes = []
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> word, box <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(words, normalized_word_boxes):
<span class="hljs-meta">... </span>    word_tokens = tokenizer.tokenize(word)
<span class="hljs-meta">... </span>    token_boxes.extend([box] * <span class="hljs-built_in">len</span>(word_tokens))
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># add bounding boxes of cls + sep tokens</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>token_boxes = [[<span class="hljs-number">0</span>, <span class="hljs-number">0</span>, <span class="hljs-number">0</span>, <span class="hljs-number">0</span>]] + token_boxes + [[<span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>]]

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer(<span class="hljs-string">&quot; &quot;</span>.join(words), return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = encoding[<span class="hljs-string">&quot;input_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>attention_mask = encoding[<span class="hljs-string">&quot;attention_mask&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>token_type_ids = encoding[<span class="hljs-string">&quot;token_type_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>bbox = torch.tensor([token_boxes])
<span class="hljs-meta">&gt;&gt;&gt; </span>sequence_label = torch.tensor([<span class="hljs-number">1</span>])

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(
<span class="hljs-meta">... </span>    input_ids=input_ids,
<span class="hljs-meta">... </span>    bbox=bbox,
<span class="hljs-meta">... </span>    attention_mask=attention_mask,
<span class="hljs-meta">... </span>    token_type_ids=token_type_ids,
<span class="hljs-meta">... </span>    labels=sequence_label,
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`}}),{c(){d=a("p"),k=n("Examples:"),_=c(),L(h.$$.fragment)},l(l){d=r(l,"P",{});var m=i(d);k=s(m,"Examples:"),m.forEach(t),_=p(l),w(h.$$.fragment,l)},m(l,m){g(l,d,m),e(d,k),g(l,_,m),T(h,l,m),b=!0},p:ut,i(l){b||(v(h.$$.fragment,l),b=!0)},o(l){M(h.$$.fragment,l),b=!1},d(l){l&&t(d),l&&t(_),$(h,l)}}}function xg(z){let d,k,_,h,b;return{c(){d=a("p"),k=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),_=a("code"),h=n("Module"),b=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=r(l,"P",{});var m=i(d);k=s(m,"Although the recipe for forward pass needs to be defined within this function, one should call the "),_=r(m,"CODE",{});var j=i(_);h=s(j,"Module"),j.forEach(t),b=s(m,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),m.forEach(t)},m(l,m){g(l,d,m),e(d,k),e(d,_),e(_,h),e(d,b)},d(l){l&&t(d)}}}function jg(z){let d,k,_,h,b;return h=new Ke({props:{code:`from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification
import torch

tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased")

words = ["Hello", "world"]
normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

token_boxes = []
for word, box in zip(words, normalized_word_boxes):
    word_tokens = tokenizer.tokenize(word)
    token_boxes.extend([box] * len(word_tokens))
# add bounding boxes of cls + sep tokens
token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

encoding = tokenizer(" ".join(words), return_tensors="pt")
input_ids = encoding["input_ids"]
attention_mask = encoding["attention_mask"]
token_type_ids = encoding["token_type_ids"]
bbox = torch.tensor([token_boxes])
token_labels = torch.tensor([1, 1, 0, 0]).unsqueeze(0)  # batch size of 1

outputs = model(
    input_ids=input_ids,
    bbox=bbox,
    attention_mask=attention_mask,
    token_type_ids=token_type_ids,
    labels=token_labels,
)

loss = outputs.loss
logits = outputs.logits`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> LayoutLMTokenizer, LayoutLMForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = LayoutLMTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlm-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LayoutLMForTokenClassification.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlm-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>words = [<span class="hljs-string">&quot;Hello&quot;</span>, <span class="hljs-string">&quot;world&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>normalized_word_boxes = [<span class="hljs-number">637</span>, <span class="hljs-number">773</span>, <span class="hljs-number">693</span>, <span class="hljs-number">782</span>], [<span class="hljs-number">698</span>, <span class="hljs-number">773</span>, <span class="hljs-number">733</span>, <span class="hljs-number">782</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>token_boxes = []
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> word, box <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(words, normalized_word_boxes):
<span class="hljs-meta">... </span>    word_tokens = tokenizer.tokenize(word)
<span class="hljs-meta">... </span>    token_boxes.extend([box] * <span class="hljs-built_in">len</span>(word_tokens))
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># add bounding boxes of cls + sep tokens</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>token_boxes = [[<span class="hljs-number">0</span>, <span class="hljs-number">0</span>, <span class="hljs-number">0</span>, <span class="hljs-number">0</span>]] + token_boxes + [[<span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>]]

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer(<span class="hljs-string">&quot; &quot;</span>.join(words), return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = encoding[<span class="hljs-string">&quot;input_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>attention_mask = encoding[<span class="hljs-string">&quot;attention_mask&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>token_type_ids = encoding[<span class="hljs-string">&quot;token_type_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>bbox = torch.tensor([token_boxes])
<span class="hljs-meta">&gt;&gt;&gt; </span>token_labels = torch.tensor([<span class="hljs-number">1</span>, <span class="hljs-number">1</span>, <span class="hljs-number">0</span>, <span class="hljs-number">0</span>]).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># batch size of 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(
<span class="hljs-meta">... </span>    input_ids=input_ids,
<span class="hljs-meta">... </span>    bbox=bbox,
<span class="hljs-meta">... </span>    attention_mask=attention_mask,
<span class="hljs-meta">... </span>    token_type_ids=token_type_ids,
<span class="hljs-meta">... </span>    labels=token_labels,
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`}}),{c(){d=a("p"),k=n("Examples:"),_=c(),L(h.$$.fragment)},l(l){d=r(l,"P",{});var m=i(d);k=s(m,"Examples:"),m.forEach(t),_=p(l),w(h.$$.fragment,l)},m(l,m){g(l,d,m),e(d,k),g(l,_,m),T(h,l,m),b=!0},p:ut,i(l){b||(v(h.$$.fragment,l),b=!0)},o(l){M(h.$$.fragment,l),b=!1},d(l){l&&t(d),l&&t(_),$(h,l)}}}function Fg(z){let d,k;return d=new Ke({props:{code:`from transformers import AutoTokenizer, LayoutLMForQuestionAnswering
from datasets import load_dataset
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased", add_prefix_space=True)
model = LayoutLMForQuestionAnswering.from_pretrained("microsoft/layoutlm-base-uncased")

dataset = load_dataset("nielsr/funsd", split="train")
example = dataset[0]
question = "what's his name?"
words = example["words"]
boxes = example["bboxes"]

encoding = tokenizer(
    question.split(), words, is_split_into_words=True, return_token_type_ids=True, return_tensors="pt"
)
bbox = []
for i, s, w in zip(encoding.input_ids[0], encoding.sequence_ids(0), encoding.word_ids(0)):
    if s == 1:
        bbox.append(boxes[w])
    elif i == tokenizer.sep_token_id:
        bbox.append([1000] * 4)
    else:
        bbox.append([0] * 4)
encoding["bbox"] = torch.tensor([bbox])

outputs = model(**encoding)
loss = outputs.loss
start_scores = outputs.start_logits
end_scores = outputs.end_logits`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, LayoutLMForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlm-base-uncased&quot;</span>, add_prefix_space=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LayoutLMForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlm-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = load_dataset(<span class="hljs-string">&quot;nielsr/funsd&quot;</span>, split=<span class="hljs-string">&quot;train&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>example = dataset[<span class="hljs-number">0</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>question = <span class="hljs-string">&quot;what&#x27;s his name?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>words = example[<span class="hljs-string">&quot;words&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>boxes = example[<span class="hljs-string">&quot;bboxes&quot;</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer(
<span class="hljs-meta">... </span>    question.split(), words, is_split_into_words=<span class="hljs-literal">True</span>, return_token_type_ids=<span class="hljs-literal">True</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>bbox = []
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> i, s, w <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(encoding.input_ids[<span class="hljs-number">0</span>], encoding.sequence_ids(<span class="hljs-number">0</span>), encoding.word_ids(<span class="hljs-number">0</span>)):
<span class="hljs-meta">... </span>    <span class="hljs-keyword">if</span> s == <span class="hljs-number">1</span>:
<span class="hljs-meta">... </span>        bbox.append(boxes[w])
<span class="hljs-meta">... </span>    <span class="hljs-keyword">elif</span> i == tokenizer.sep_token_id:
<span class="hljs-meta">... </span>        bbox.append([<span class="hljs-number">1000</span>] * <span class="hljs-number">4</span>)
<span class="hljs-meta">... </span>    <span class="hljs-keyword">else</span>:
<span class="hljs-meta">... </span>        bbox.append([<span class="hljs-number">0</span>] * <span class="hljs-number">4</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>encoding[<span class="hljs-string">&quot;bbox&quot;</span>] = torch.tensor([bbox])

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**encoding)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>start_scores = outputs.start_logits
<span class="hljs-meta">&gt;&gt;&gt; </span>end_scores = outputs.end_logits`}}),{c(){L(d.$$.fragment)},l(_){w(d.$$.fragment,_)},m(_,h){T(d,_,h),k=!0},p:ut,i(_){k||(v(d.$$.fragment,_),k=!0)},o(_){M(d.$$.fragment,_),k=!1},d(_){$(d,_)}}}function Eg(z){let d,k,_,h,b,l,m,j,re,X,E,ae,I,Y,ne,S,ie,se,Z,D,G,J,F,q,le,H,V,ee,Q,te,de,C,ce,U,pe,oe,B,ue,he,N,me,W,fe;return{c(){d=a("p"),k=n("TF 2.0 models accepts two formats as inputs:"),_=c(),h=a("ul"),b=a("li"),l=n("having all inputs as keyword arguments (like PyTorch models), or"),m=c(),j=a("li"),re=n("having all inputs as a list, tuple or dict in the first positional arguments."),X=c(),E=a("p"),ae=n("This second option is useful when using "),I=a("code"),Y=n("tf.keras.Model.fit"),ne=n(` method which currently requires having all the
tensors in the first argument of the model call function: `),S=a("code"),ie=n("model(inputs)"),se=n("."),Z=c(),D=a("p"),G=n(`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),J=c(),F=a("ul"),q=a("li"),le=n("a single Tensor with "),H=a("code"),V=n("input_ids"),ee=n(" only and nothing else: "),Q=a("code"),te=n("model(inputs_ids)"),de=c(),C=a("li"),ce=n(`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),U=a("code"),pe=n("model([input_ids, attention_mask])"),oe=n(" or "),B=a("code"),ue=n("model([input_ids, attention_mask, token_type_ids])"),he=c(),N=a("li"),me=n(`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),W=a("code"),fe=n('model({"input_ids": input_ids, "token_type_ids": token_type_ids})')},l(f){d=r(f,"P",{});var x=i(d);k=s(x,"TF 2.0 models accepts two formats as inputs:"),x.forEach(t),_=p(f),h=r(f,"UL",{});var K=i(h);b=r(K,"LI",{});var Me=i(b);l=s(Me,"having all inputs as keyword arguments (like PyTorch models), or"),Me.forEach(t),m=p(K),j=r(K,"LI",{});var we=i(j);re=s(we,"having all inputs as a list, tuple or dict in the first positional arguments."),we.forEach(t),K.forEach(t),X=p(f),E=r(f,"P",{});var A=i(E);ae=s(A,"This second option is useful when using "),I=r(A,"CODE",{});var ke=i(I);Y=s(ke,"tf.keras.Model.fit"),ke.forEach(t),ne=s(A,` method which currently requires having all the
tensors in the first argument of the model call function: `),S=r(A,"CODE",{});var $e=i(S);ie=s($e,"model(inputs)"),$e.forEach(t),se=s(A,"."),A.forEach(t),Z=p(f),D=r(f,"P",{});var Te=i(D);G=s(Te,`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),Te.forEach(t),J=p(f),F=r(f,"UL",{});var O=i(F);q=r(O,"LI",{});var R=i(q);le=s(R,"a single Tensor with "),H=r(R,"CODE",{});var ve=i(H);V=s(ve,"input_ids"),ve.forEach(t),ee=s(R," only and nothing else: "),Q=r(R,"CODE",{});var ye=i(Q);te=s(ye,"model(inputs_ids)"),ye.forEach(t),R.forEach(t),de=p(O),C=r(O,"LI",{});var P=i(C);ce=s(P,`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),U=r(P,"CODE",{});var xe=i(U);pe=s(xe,"model([input_ids, attention_mask])"),xe.forEach(t),oe=s(P," or "),B=r(P,"CODE",{});var Le=i(B);ue=s(Le,"model([input_ids, attention_mask, token_type_ids])"),Le.forEach(t),P.forEach(t),he=p(O),N=r(O,"LI",{});var ge=i(N);me=s(ge,`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),W=r(ge,"CODE",{});var je=i(W);fe=s(je,'model({"input_ids": input_ids, "token_type_ids": token_type_ids})'),je.forEach(t),ge.forEach(t),O.forEach(t)},m(f,x){g(f,d,x),e(d,k),g(f,_,x),g(f,h,x),e(h,b),e(b,l),e(h,m),e(h,j),e(j,re),g(f,X,x),g(f,E,x),e(E,ae),e(E,I),e(I,Y),e(E,ne),e(E,S),e(S,ie),e(E,se),g(f,Z,x),g(f,D,x),e(D,G),g(f,J,x),g(f,F,x),e(F,q),e(q,le),e(q,H),e(H,V),e(q,ee),e(q,Q),e(Q,te),e(F,de),e(F,C),e(C,ce),e(C,U),e(U,pe),e(C,oe),e(C,B),e(B,ue),e(F,he),e(F,N),e(N,me),e(N,W),e(W,fe)},d(f){f&&t(d),f&&t(_),f&&t(h),f&&t(X),f&&t(E),f&&t(Z),f&&t(D),f&&t(J),f&&t(F)}}}function zg(z){let d,k,_,h,b;return{c(){d=a("p"),k=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),_=a("code"),h=n("Module"),b=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=r(l,"P",{});var m=i(d);k=s(m,"Although the recipe for forward pass needs to be defined within this function, one should call the "),_=r(m,"CODE",{});var j=i(_);h=s(j,"Module"),j.forEach(t),b=s(m,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),m.forEach(t)},m(l,m){g(l,d,m),e(d,k),e(d,_),e(_,h),e(d,b)},d(l){l&&t(d)}}}function qg(z){let d,k,_,h,b;return h=new Ke({props:{code:`from transformers import LayoutLMTokenizer, TFLayoutLMModel
import tensorflow as tf

tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
model = TFLayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased")

words = ["Hello", "world"]
normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

token_boxes = []
for word, box in zip(words, normalized_word_boxes):
    word_tokens = tokenizer.tokenize(word)
    token_boxes.extend([box] * len(word_tokens))
# add bounding boxes of cls + sep tokens
token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

encoding = tokenizer(" ".join(words), return_tensors="tf")
input_ids = encoding["input_ids"]
attention_mask = encoding["attention_mask"]
token_type_ids = encoding["token_type_ids"]
bbox = tf.convert_to_tensor([token_boxes])

outputs = model(
    input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids
)

last_hidden_states = outputs.last_hidden_state`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> LayoutLMTokenizer, TFLayoutLMModel
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> tensorflow <span class="hljs-keyword">as</span> tf

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = LayoutLMTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlm-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFLayoutLMModel.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlm-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>words = [<span class="hljs-string">&quot;Hello&quot;</span>, <span class="hljs-string">&quot;world&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>normalized_word_boxes = [<span class="hljs-number">637</span>, <span class="hljs-number">773</span>, <span class="hljs-number">693</span>, <span class="hljs-number">782</span>], [<span class="hljs-number">698</span>, <span class="hljs-number">773</span>, <span class="hljs-number">733</span>, <span class="hljs-number">782</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>token_boxes = []
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> word, box <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(words, normalized_word_boxes):
<span class="hljs-meta">... </span>    word_tokens = tokenizer.tokenize(word)
<span class="hljs-meta">... </span>    token_boxes.extend([box] * <span class="hljs-built_in">len</span>(word_tokens))
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># add bounding boxes of cls + sep tokens</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>token_boxes = [[<span class="hljs-number">0</span>, <span class="hljs-number">0</span>, <span class="hljs-number">0</span>, <span class="hljs-number">0</span>]] + token_boxes + [[<span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>]]

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer(<span class="hljs-string">&quot; &quot;</span>.join(words), return_tensors=<span class="hljs-string">&quot;tf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = encoding[<span class="hljs-string">&quot;input_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>attention_mask = encoding[<span class="hljs-string">&quot;attention_mask&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>token_type_ids = encoding[<span class="hljs-string">&quot;token_type_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>bbox = tf.convert_to_tensor([token_boxes])

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(
<span class="hljs-meta">... </span>    input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`}}),{c(){d=a("p"),k=n("Examples:"),_=c(),L(h.$$.fragment)},l(l){d=r(l,"P",{});var m=i(d);k=s(m,"Examples:"),m.forEach(t),_=p(l),w(h.$$.fragment,l)},m(l,m){g(l,d,m),e(d,k),g(l,_,m),T(h,l,m),b=!0},p:ut,i(l){b||(v(h.$$.fragment,l),b=!0)},o(l){M(h.$$.fragment,l),b=!1},d(l){l&&t(d),l&&t(_),$(h,l)}}}function Cg(z){let d,k,_,h,b,l,m,j,re,X,E,ae,I,Y,ne,S,ie,se,Z,D,G,J,F,q,le,H,V,ee,Q,te,de,C,ce,U,pe,oe,B,ue,he,N,me,W,fe;return{c(){d=a("p"),k=n("TF 2.0 models accepts two formats as inputs:"),_=c(),h=a("ul"),b=a("li"),l=n("having all inputs as keyword arguments (like PyTorch models), or"),m=c(),j=a("li"),re=n("having all inputs as a list, tuple or dict in the first positional arguments."),X=c(),E=a("p"),ae=n("This second option is useful when using "),I=a("code"),Y=n("tf.keras.Model.fit"),ne=n(` method which currently requires having all the
tensors in the first argument of the model call function: `),S=a("code"),ie=n("model(inputs)"),se=n("."),Z=c(),D=a("p"),G=n(`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),J=c(),F=a("ul"),q=a("li"),le=n("a single Tensor with "),H=a("code"),V=n("input_ids"),ee=n(" only and nothing else: "),Q=a("code"),te=n("model(inputs_ids)"),de=c(),C=a("li"),ce=n(`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),U=a("code"),pe=n("model([input_ids, attention_mask])"),oe=n(" or "),B=a("code"),ue=n("model([input_ids, attention_mask, token_type_ids])"),he=c(),N=a("li"),me=n(`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),W=a("code"),fe=n('model({"input_ids": input_ids, "token_type_ids": token_type_ids})')},l(f){d=r(f,"P",{});var x=i(d);k=s(x,"TF 2.0 models accepts two formats as inputs:"),x.forEach(t),_=p(f),h=r(f,"UL",{});var K=i(h);b=r(K,"LI",{});var Me=i(b);l=s(Me,"having all inputs as keyword arguments (like PyTorch models), or"),Me.forEach(t),m=p(K),j=r(K,"LI",{});var we=i(j);re=s(we,"having all inputs as a list, tuple or dict in the first positional arguments."),we.forEach(t),K.forEach(t),X=p(f),E=r(f,"P",{});var A=i(E);ae=s(A,"This second option is useful when using "),I=r(A,"CODE",{});var ke=i(I);Y=s(ke,"tf.keras.Model.fit"),ke.forEach(t),ne=s(A,` method which currently requires having all the
tensors in the first argument of the model call function: `),S=r(A,"CODE",{});var $e=i(S);ie=s($e,"model(inputs)"),$e.forEach(t),se=s(A,"."),A.forEach(t),Z=p(f),D=r(f,"P",{});var Te=i(D);G=s(Te,`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),Te.forEach(t),J=p(f),F=r(f,"UL",{});var O=i(F);q=r(O,"LI",{});var R=i(q);le=s(R,"a single Tensor with "),H=r(R,"CODE",{});var ve=i(H);V=s(ve,"input_ids"),ve.forEach(t),ee=s(R," only and nothing else: "),Q=r(R,"CODE",{});var ye=i(Q);te=s(ye,"model(inputs_ids)"),ye.forEach(t),R.forEach(t),de=p(O),C=r(O,"LI",{});var P=i(C);ce=s(P,`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),U=r(P,"CODE",{});var xe=i(U);pe=s(xe,"model([input_ids, attention_mask])"),xe.forEach(t),oe=s(P," or "),B=r(P,"CODE",{});var Le=i(B);ue=s(Le,"model([input_ids, attention_mask, token_type_ids])"),Le.forEach(t),P.forEach(t),he=p(O),N=r(O,"LI",{});var ge=i(N);me=s(ge,`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),W=r(ge,"CODE",{});var je=i(W);fe=s(je,'model({"input_ids": input_ids, "token_type_ids": token_type_ids})'),je.forEach(t),ge.forEach(t),O.forEach(t)},m(f,x){g(f,d,x),e(d,k),g(f,_,x),g(f,h,x),e(h,b),e(b,l),e(h,m),e(h,j),e(j,re),g(f,X,x),g(f,E,x),e(E,ae),e(E,I),e(I,Y),e(E,ne),e(E,S),e(S,ie),e(E,se),g(f,Z,x),g(f,D,x),e(D,G),g(f,J,x),g(f,F,x),e(F,q),e(q,le),e(q,H),e(H,V),e(q,ee),e(q,Q),e(Q,te),e(F,de),e(F,C),e(C,ce),e(C,U),e(U,pe),e(C,oe),e(C,B),e(B,ue),e(F,he),e(F,N),e(N,me),e(N,W),e(W,fe)},d(f){f&&t(d),f&&t(_),f&&t(h),f&&t(X),f&&t(E),f&&t(Z),f&&t(D),f&&t(J),f&&t(F)}}}function Pg(z){let d,k,_,h,b;return{c(){d=a("p"),k=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),_=a("code"),h=n("Module"),b=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=r(l,"P",{});var m=i(d);k=s(m,"Although the recipe for forward pass needs to be defined within this function, one should call the "),_=r(m,"CODE",{});var j=i(_);h=s(j,"Module"),j.forEach(t),b=s(m,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),m.forEach(t)},m(l,m){g(l,d,m),e(d,k),e(d,_),e(_,h),e(d,b)},d(l){l&&t(d)}}}function Ng(z){let d,k,_,h,b;return h=new Ke({props:{code:`from transformers import LayoutLMTokenizer, TFLayoutLMForMaskedLM
import tensorflow as tf

tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
model = TFLayoutLMForMaskedLM.from_pretrained("microsoft/layoutlm-base-uncased")

words = ["Hello", "[MASK]"]
normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

token_boxes = []
for word, box in zip(words, normalized_word_boxes):
    word_tokens = tokenizer.tokenize(word)
    token_boxes.extend([box] * len(word_tokens))
# add bounding boxes of cls + sep tokens
token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

encoding = tokenizer(" ".join(words), return_tensors="tf")
input_ids = encoding["input_ids"]
attention_mask = encoding["attention_mask"]
token_type_ids = encoding["token_type_ids"]
bbox = tf.convert_to_tensor([token_boxes])

labels = tokenizer("Hello world", return_tensors="tf")["input_ids"]

outputs = model(
    input_ids=input_ids,
    bbox=bbox,
    attention_mask=attention_mask,
    token_type_ids=token_type_ids,
    labels=labels,
)

loss = outputs.loss`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> LayoutLMTokenizer, TFLayoutLMForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> tensorflow <span class="hljs-keyword">as</span> tf

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = LayoutLMTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlm-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFLayoutLMForMaskedLM.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlm-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>words = [<span class="hljs-string">&quot;Hello&quot;</span>, <span class="hljs-string">&quot;[MASK]&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>normalized_word_boxes = [<span class="hljs-number">637</span>, <span class="hljs-number">773</span>, <span class="hljs-number">693</span>, <span class="hljs-number">782</span>], [<span class="hljs-number">698</span>, <span class="hljs-number">773</span>, <span class="hljs-number">733</span>, <span class="hljs-number">782</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>token_boxes = []
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> word, box <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(words, normalized_word_boxes):
<span class="hljs-meta">... </span>    word_tokens = tokenizer.tokenize(word)
<span class="hljs-meta">... </span>    token_boxes.extend([box] * <span class="hljs-built_in">len</span>(word_tokens))
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># add bounding boxes of cls + sep tokens</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>token_boxes = [[<span class="hljs-number">0</span>, <span class="hljs-number">0</span>, <span class="hljs-number">0</span>, <span class="hljs-number">0</span>]] + token_boxes + [[<span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>]]

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer(<span class="hljs-string">&quot; &quot;</span>.join(words), return_tensors=<span class="hljs-string">&quot;tf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = encoding[<span class="hljs-string">&quot;input_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>attention_mask = encoding[<span class="hljs-string">&quot;attention_mask&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>token_type_ids = encoding[<span class="hljs-string">&quot;token_type_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>bbox = tf.convert_to_tensor([token_boxes])

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = tokenizer(<span class="hljs-string">&quot;Hello world&quot;</span>, return_tensors=<span class="hljs-string">&quot;tf&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(
<span class="hljs-meta">... </span>    input_ids=input_ids,
<span class="hljs-meta">... </span>    bbox=bbox,
<span class="hljs-meta">... </span>    attention_mask=attention_mask,
<span class="hljs-meta">... </span>    token_type_ids=token_type_ids,
<span class="hljs-meta">... </span>    labels=labels,
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss`}}),{c(){d=a("p"),k=n("Examples:"),_=c(),L(h.$$.fragment)},l(l){d=r(l,"P",{});var m=i(d);k=s(m,"Examples:"),m.forEach(t),_=p(l),w(h.$$.fragment,l)},m(l,m){g(l,d,m),e(d,k),g(l,_,m),T(h,l,m),b=!0},p:ut,i(l){b||(v(h.$$.fragment,l),b=!0)},o(l){M(h.$$.fragment,l),b=!1},d(l){l&&t(d),l&&t(_),$(h,l)}}}function Ag(z){let d,k,_,h,b,l,m,j,re,X,E,ae,I,Y,ne,S,ie,se,Z,D,G,J,F,q,le,H,V,ee,Q,te,de,C,ce,U,pe,oe,B,ue,he,N,me,W,fe;return{c(){d=a("p"),k=n("TF 2.0 models accepts two formats as inputs:"),_=c(),h=a("ul"),b=a("li"),l=n("having all inputs as keyword arguments (like PyTorch models), or"),m=c(),j=a("li"),re=n("having all inputs as a list, tuple or dict in the first positional arguments."),X=c(),E=a("p"),ae=n("This second option is useful when using "),I=a("code"),Y=n("tf.keras.Model.fit"),ne=n(` method which currently requires having all the
tensors in the first argument of the model call function: `),S=a("code"),ie=n("model(inputs)"),se=n("."),Z=c(),D=a("p"),G=n(`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),J=c(),F=a("ul"),q=a("li"),le=n("a single Tensor with "),H=a("code"),V=n("input_ids"),ee=n(" only and nothing else: "),Q=a("code"),te=n("model(inputs_ids)"),de=c(),C=a("li"),ce=n(`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),U=a("code"),pe=n("model([input_ids, attention_mask])"),oe=n(" or "),B=a("code"),ue=n("model([input_ids, attention_mask, token_type_ids])"),he=c(),N=a("li"),me=n(`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),W=a("code"),fe=n('model({"input_ids": input_ids, "token_type_ids": token_type_ids})')},l(f){d=r(f,"P",{});var x=i(d);k=s(x,"TF 2.0 models accepts two formats as inputs:"),x.forEach(t),_=p(f),h=r(f,"UL",{});var K=i(h);b=r(K,"LI",{});var Me=i(b);l=s(Me,"having all inputs as keyword arguments (like PyTorch models), or"),Me.forEach(t),m=p(K),j=r(K,"LI",{});var we=i(j);re=s(we,"having all inputs as a list, tuple or dict in the first positional arguments."),we.forEach(t),K.forEach(t),X=p(f),E=r(f,"P",{});var A=i(E);ae=s(A,"This second option is useful when using "),I=r(A,"CODE",{});var ke=i(I);Y=s(ke,"tf.keras.Model.fit"),ke.forEach(t),ne=s(A,` method which currently requires having all the
tensors in the first argument of the model call function: `),S=r(A,"CODE",{});var $e=i(S);ie=s($e,"model(inputs)"),$e.forEach(t),se=s(A,"."),A.forEach(t),Z=p(f),D=r(f,"P",{});var Te=i(D);G=s(Te,`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),Te.forEach(t),J=p(f),F=r(f,"UL",{});var O=i(F);q=r(O,"LI",{});var R=i(q);le=s(R,"a single Tensor with "),H=r(R,"CODE",{});var ve=i(H);V=s(ve,"input_ids"),ve.forEach(t),ee=s(R," only and nothing else: "),Q=r(R,"CODE",{});var ye=i(Q);te=s(ye,"model(inputs_ids)"),ye.forEach(t),R.forEach(t),de=p(O),C=r(O,"LI",{});var P=i(C);ce=s(P,`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),U=r(P,"CODE",{});var xe=i(U);pe=s(xe,"model([input_ids, attention_mask])"),xe.forEach(t),oe=s(P," or "),B=r(P,"CODE",{});var Le=i(B);ue=s(Le,"model([input_ids, attention_mask, token_type_ids])"),Le.forEach(t),P.forEach(t),he=p(O),N=r(O,"LI",{});var ge=i(N);me=s(ge,`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),W=r(ge,"CODE",{});var je=i(W);fe=s(je,'model({"input_ids": input_ids, "token_type_ids": token_type_ids})'),je.forEach(t),ge.forEach(t),O.forEach(t)},m(f,x){g(f,d,x),e(d,k),g(f,_,x),g(f,h,x),e(h,b),e(b,l),e(h,m),e(h,j),e(j,re),g(f,X,x),g(f,E,x),e(E,ae),e(E,I),e(I,Y),e(E,ne),e(E,S),e(S,ie),e(E,se),g(f,Z,x),g(f,D,x),e(D,G),g(f,J,x),g(f,F,x),e(F,q),e(q,le),e(q,H),e(H,V),e(q,ee),e(q,Q),e(Q,te),e(F,de),e(F,C),e(C,ce),e(C,U),e(U,pe),e(C,oe),e(C,B),e(B,ue),e(F,he),e(F,N),e(N,me),e(N,W),e(W,fe)},d(f){f&&t(d),f&&t(_),f&&t(h),f&&t(X),f&&t(E),f&&t(Z),f&&t(D),f&&t(J),f&&t(F)}}}function Og(z){let d,k,_,h,b;return{c(){d=a("p"),k=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),_=a("code"),h=n("Module"),b=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=r(l,"P",{});var m=i(d);k=s(m,"Although the recipe for forward pass needs to be defined within this function, one should call the "),_=r(m,"CODE",{});var j=i(_);h=s(j,"Module"),j.forEach(t),b=s(m,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),m.forEach(t)},m(l,m){g(l,d,m),e(d,k),e(d,_),e(_,h),e(d,b)},d(l){l&&t(d)}}}function Ig(z){let d,k,_,h,b;return h=new Ke({props:{code:`from transformers import LayoutLMTokenizer, TFLayoutLMForSequenceClassification
import tensorflow as tf

tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
model = TFLayoutLMForSequenceClassification.from_pretrained("microsoft/layoutlm-base-uncased")

words = ["Hello", "world"]
normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

token_boxes = []
for word, box in zip(words, normalized_word_boxes):
    word_tokens = tokenizer.tokenize(word)
    token_boxes.extend([box] * len(word_tokens))
# add bounding boxes of cls + sep tokens
token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

encoding = tokenizer(" ".join(words), return_tensors="tf")
input_ids = encoding["input_ids"]
attention_mask = encoding["attention_mask"]
token_type_ids = encoding["token_type_ids"]
bbox = tf.convert_to_tensor([token_boxes])
sequence_label = tf.convert_to_tensor([1])

outputs = model(
    input_ids=input_ids,
    bbox=bbox,
    attention_mask=attention_mask,
    token_type_ids=token_type_ids,
    labels=sequence_label,
)

loss = outputs.loss
logits = outputs.logits`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> LayoutLMTokenizer, TFLayoutLMForSequenceClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> tensorflow <span class="hljs-keyword">as</span> tf

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = LayoutLMTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlm-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFLayoutLMForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlm-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>words = [<span class="hljs-string">&quot;Hello&quot;</span>, <span class="hljs-string">&quot;world&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>normalized_word_boxes = [<span class="hljs-number">637</span>, <span class="hljs-number">773</span>, <span class="hljs-number">693</span>, <span class="hljs-number">782</span>], [<span class="hljs-number">698</span>, <span class="hljs-number">773</span>, <span class="hljs-number">733</span>, <span class="hljs-number">782</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>token_boxes = []
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> word, box <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(words, normalized_word_boxes):
<span class="hljs-meta">... </span>    word_tokens = tokenizer.tokenize(word)
<span class="hljs-meta">... </span>    token_boxes.extend([box] * <span class="hljs-built_in">len</span>(word_tokens))
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># add bounding boxes of cls + sep tokens</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>token_boxes = [[<span class="hljs-number">0</span>, <span class="hljs-number">0</span>, <span class="hljs-number">0</span>, <span class="hljs-number">0</span>]] + token_boxes + [[<span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>]]

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer(<span class="hljs-string">&quot; &quot;</span>.join(words), return_tensors=<span class="hljs-string">&quot;tf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = encoding[<span class="hljs-string">&quot;input_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>attention_mask = encoding[<span class="hljs-string">&quot;attention_mask&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>token_type_ids = encoding[<span class="hljs-string">&quot;token_type_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>bbox = tf.convert_to_tensor([token_boxes])
<span class="hljs-meta">&gt;&gt;&gt; </span>sequence_label = tf.convert_to_tensor([<span class="hljs-number">1</span>])

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(
<span class="hljs-meta">... </span>    input_ids=input_ids,
<span class="hljs-meta">... </span>    bbox=bbox,
<span class="hljs-meta">... </span>    attention_mask=attention_mask,
<span class="hljs-meta">... </span>    token_type_ids=token_type_ids,
<span class="hljs-meta">... </span>    labels=sequence_label,
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`}}),{c(){d=a("p"),k=n("Examples:"),_=c(),L(h.$$.fragment)},l(l){d=r(l,"P",{});var m=i(d);k=s(m,"Examples:"),m.forEach(t),_=p(l),w(h.$$.fragment,l)},m(l,m){g(l,d,m),e(d,k),g(l,_,m),T(h,l,m),b=!0},p:ut,i(l){b||(v(h.$$.fragment,l),b=!0)},o(l){M(h.$$.fragment,l),b=!1},d(l){l&&t(d),l&&t(_),$(h,l)}}}function Dg(z){let d,k,_,h,b,l,m,j,re,X,E,ae,I,Y,ne,S,ie,se,Z,D,G,J,F,q,le,H,V,ee,Q,te,de,C,ce,U,pe,oe,B,ue,he,N,me,W,fe;return{c(){d=a("p"),k=n("TF 2.0 models accepts two formats as inputs:"),_=c(),h=a("ul"),b=a("li"),l=n("having all inputs as keyword arguments (like PyTorch models), or"),m=c(),j=a("li"),re=n("having all inputs as a list, tuple or dict in the first positional arguments."),X=c(),E=a("p"),ae=n("This second option is useful when using "),I=a("code"),Y=n("tf.keras.Model.fit"),ne=n(` method which currently requires having all the
tensors in the first argument of the model call function: `),S=a("code"),ie=n("model(inputs)"),se=n("."),Z=c(),D=a("p"),G=n(`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),J=c(),F=a("ul"),q=a("li"),le=n("a single Tensor with "),H=a("code"),V=n("input_ids"),ee=n(" only and nothing else: "),Q=a("code"),te=n("model(inputs_ids)"),de=c(),C=a("li"),ce=n(`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),U=a("code"),pe=n("model([input_ids, attention_mask])"),oe=n(" or "),B=a("code"),ue=n("model([input_ids, attention_mask, token_type_ids])"),he=c(),N=a("li"),me=n(`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),W=a("code"),fe=n('model({"input_ids": input_ids, "token_type_ids": token_type_ids})')},l(f){d=r(f,"P",{});var x=i(d);k=s(x,"TF 2.0 models accepts two formats as inputs:"),x.forEach(t),_=p(f),h=r(f,"UL",{});var K=i(h);b=r(K,"LI",{});var Me=i(b);l=s(Me,"having all inputs as keyword arguments (like PyTorch models), or"),Me.forEach(t),m=p(K),j=r(K,"LI",{});var we=i(j);re=s(we,"having all inputs as a list, tuple or dict in the first positional arguments."),we.forEach(t),K.forEach(t),X=p(f),E=r(f,"P",{});var A=i(E);ae=s(A,"This second option is useful when using "),I=r(A,"CODE",{});var ke=i(I);Y=s(ke,"tf.keras.Model.fit"),ke.forEach(t),ne=s(A,` method which currently requires having all the
tensors in the first argument of the model call function: `),S=r(A,"CODE",{});var $e=i(S);ie=s($e,"model(inputs)"),$e.forEach(t),se=s(A,"."),A.forEach(t),Z=p(f),D=r(f,"P",{});var Te=i(D);G=s(Te,`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),Te.forEach(t),J=p(f),F=r(f,"UL",{});var O=i(F);q=r(O,"LI",{});var R=i(q);le=s(R,"a single Tensor with "),H=r(R,"CODE",{});var ve=i(H);V=s(ve,"input_ids"),ve.forEach(t),ee=s(R," only and nothing else: "),Q=r(R,"CODE",{});var ye=i(Q);te=s(ye,"model(inputs_ids)"),ye.forEach(t),R.forEach(t),de=p(O),C=r(O,"LI",{});var P=i(C);ce=s(P,`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),U=r(P,"CODE",{});var xe=i(U);pe=s(xe,"model([input_ids, attention_mask])"),xe.forEach(t),oe=s(P," or "),B=r(P,"CODE",{});var Le=i(B);ue=s(Le,"model([input_ids, attention_mask, token_type_ids])"),Le.forEach(t),P.forEach(t),he=p(O),N=r(O,"LI",{});var ge=i(N);me=s(ge,`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),W=r(ge,"CODE",{});var je=i(W);fe=s(je,'model({"input_ids": input_ids, "token_type_ids": token_type_ids})'),je.forEach(t),ge.forEach(t),O.forEach(t)},m(f,x){g(f,d,x),e(d,k),g(f,_,x),g(f,h,x),e(h,b),e(b,l),e(h,m),e(h,j),e(j,re),g(f,X,x),g(f,E,x),e(E,ae),e(E,I),e(I,Y),e(E,ne),e(E,S),e(S,ie),e(E,se),g(f,Z,x),g(f,D,x),e(D,G),g(f,J,x),g(f,F,x),e(F,q),e(q,le),e(q,H),e(H,V),e(q,ee),e(q,Q),e(Q,te),e(F,de),e(F,C),e(C,ce),e(C,U),e(U,pe),e(C,oe),e(C,B),e(B,ue),e(F,he),e(F,N),e(N,me),e(N,W),e(W,fe)},d(f){f&&t(d),f&&t(_),f&&t(h),f&&t(X),f&&t(E),f&&t(Z),f&&t(D),f&&t(J),f&&t(F)}}}function Sg(z){let d,k,_,h,b;return{c(){d=a("p"),k=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),_=a("code"),h=n("Module"),b=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=r(l,"P",{});var m=i(d);k=s(m,"Although the recipe for forward pass needs to be defined within this function, one should call the "),_=r(m,"CODE",{});var j=i(_);h=s(j,"Module"),j.forEach(t),b=s(m,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),m.forEach(t)},m(l,m){g(l,d,m),e(d,k),e(d,_),e(_,h),e(d,b)},d(l){l&&t(d)}}}function Ug(z){let d,k,_,h,b;return h=new Ke({props:{code:`import tensorflow as tf
from transformers import LayoutLMTokenizer, TFLayoutLMForTokenClassification

tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
model = TFLayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased")

words = ["Hello", "world"]
normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

token_boxes = []
for word, box in zip(words, normalized_word_boxes):
    word_tokens = tokenizer.tokenize(word)
    token_boxes.extend([box] * len(word_tokens))
# add bounding boxes of cls + sep tokens
token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

encoding = tokenizer(" ".join(words), return_tensors="tf")
input_ids = encoding["input_ids"]
attention_mask = encoding["attention_mask"]
token_type_ids = encoding["token_type_ids"]
bbox = tf.convert_to_tensor([token_boxes])
token_labels = tf.convert_to_tensor([1, 1, 0, 0])

outputs = model(
    input_ids=input_ids,
    bbox=bbox,
    attention_mask=attention_mask,
    token_type_ids=token_type_ids,
    labels=token_labels,
)

loss = outputs.loss
logits = outputs.logits`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> tensorflow <span class="hljs-keyword">as</span> tf
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> LayoutLMTokenizer, TFLayoutLMForTokenClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = LayoutLMTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlm-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFLayoutLMForTokenClassification.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlm-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>words = [<span class="hljs-string">&quot;Hello&quot;</span>, <span class="hljs-string">&quot;world&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>normalized_word_boxes = [<span class="hljs-number">637</span>, <span class="hljs-number">773</span>, <span class="hljs-number">693</span>, <span class="hljs-number">782</span>], [<span class="hljs-number">698</span>, <span class="hljs-number">773</span>, <span class="hljs-number">733</span>, <span class="hljs-number">782</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>token_boxes = []
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> word, box <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(words, normalized_word_boxes):
<span class="hljs-meta">... </span>    word_tokens = tokenizer.tokenize(word)
<span class="hljs-meta">... </span>    token_boxes.extend([box] * <span class="hljs-built_in">len</span>(word_tokens))
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># add bounding boxes of cls + sep tokens</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>token_boxes = [[<span class="hljs-number">0</span>, <span class="hljs-number">0</span>, <span class="hljs-number">0</span>, <span class="hljs-number">0</span>]] + token_boxes + [[<span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>]]

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer(<span class="hljs-string">&quot; &quot;</span>.join(words), return_tensors=<span class="hljs-string">&quot;tf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = encoding[<span class="hljs-string">&quot;input_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>attention_mask = encoding[<span class="hljs-string">&quot;attention_mask&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>token_type_ids = encoding[<span class="hljs-string">&quot;token_type_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>bbox = tf.convert_to_tensor([token_boxes])
<span class="hljs-meta">&gt;&gt;&gt; </span>token_labels = tf.convert_to_tensor([<span class="hljs-number">1</span>, <span class="hljs-number">1</span>, <span class="hljs-number">0</span>, <span class="hljs-number">0</span>])

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(
<span class="hljs-meta">... </span>    input_ids=input_ids,
<span class="hljs-meta">... </span>    bbox=bbox,
<span class="hljs-meta">... </span>    attention_mask=attention_mask,
<span class="hljs-meta">... </span>    token_type_ids=token_type_ids,
<span class="hljs-meta">... </span>    labels=token_labels,
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`}}),{c(){d=a("p"),k=n("Examples:"),_=c(),L(h.$$.fragment)},l(l){d=r(l,"P",{});var m=i(d);k=s(m,"Examples:"),m.forEach(t),_=p(l),w(h.$$.fragment,l)},m(l,m){g(l,d,m),e(d,k),g(l,_,m),T(h,l,m),b=!0},p:ut,i(l){b||(v(h.$$.fragment,l),b=!0)},o(l){M(h.$$.fragment,l),b=!1},d(l){l&&t(d),l&&t(_),$(h,l)}}}function Wg(z){let d,k,_,h,b,l,m,j,re,X,E,ae,I,Y,ne,S,ie,se,Z,D,G,J,F,q,le,H,V,ee,Q,te,de,C,ce,U,pe,oe,B,ue,he,N,me,W,fe;return{c(){d=a("p"),k=n("TF 2.0 models accepts two formats as inputs:"),_=c(),h=a("ul"),b=a("li"),l=n("having all inputs as keyword arguments (like PyTorch models), or"),m=c(),j=a("li"),re=n("having all inputs as a list, tuple or dict in the first positional arguments."),X=c(),E=a("p"),ae=n("This second option is useful when using "),I=a("code"),Y=n("tf.keras.Model.fit"),ne=n(` method which currently requires having all the
tensors in the first argument of the model call function: `),S=a("code"),ie=n("model(inputs)"),se=n("."),Z=c(),D=a("p"),G=n(`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),J=c(),F=a("ul"),q=a("li"),le=n("a single Tensor with "),H=a("code"),V=n("input_ids"),ee=n(" only and nothing else: "),Q=a("code"),te=n("model(inputs_ids)"),de=c(),C=a("li"),ce=n(`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),U=a("code"),pe=n("model([input_ids, attention_mask])"),oe=n(" or "),B=a("code"),ue=n("model([input_ids, attention_mask, token_type_ids])"),he=c(),N=a("li"),me=n(`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),W=a("code"),fe=n('model({"input_ids": input_ids, "token_type_ids": token_type_ids})')},l(f){d=r(f,"P",{});var x=i(d);k=s(x,"TF 2.0 models accepts two formats as inputs:"),x.forEach(t),_=p(f),h=r(f,"UL",{});var K=i(h);b=r(K,"LI",{});var Me=i(b);l=s(Me,"having all inputs as keyword arguments (like PyTorch models), or"),Me.forEach(t),m=p(K),j=r(K,"LI",{});var we=i(j);re=s(we,"having all inputs as a list, tuple or dict in the first positional arguments."),we.forEach(t),K.forEach(t),X=p(f),E=r(f,"P",{});var A=i(E);ae=s(A,"This second option is useful when using "),I=r(A,"CODE",{});var ke=i(I);Y=s(ke,"tf.keras.Model.fit"),ke.forEach(t),ne=s(A,` method which currently requires having all the
tensors in the first argument of the model call function: `),S=r(A,"CODE",{});var $e=i(S);ie=s($e,"model(inputs)"),$e.forEach(t),se=s(A,"."),A.forEach(t),Z=p(f),D=r(f,"P",{});var Te=i(D);G=s(Te,`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),Te.forEach(t),J=p(f),F=r(f,"UL",{});var O=i(F);q=r(O,"LI",{});var R=i(q);le=s(R,"a single Tensor with "),H=r(R,"CODE",{});var ve=i(H);V=s(ve,"input_ids"),ve.forEach(t),ee=s(R," only and nothing else: "),Q=r(R,"CODE",{});var ye=i(Q);te=s(ye,"model(inputs_ids)"),ye.forEach(t),R.forEach(t),de=p(O),C=r(O,"LI",{});var P=i(C);ce=s(P,`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),U=r(P,"CODE",{});var xe=i(U);pe=s(xe,"model([input_ids, attention_mask])"),xe.forEach(t),oe=s(P," or "),B=r(P,"CODE",{});var Le=i(B);ue=s(Le,"model([input_ids, attention_mask, token_type_ids])"),Le.forEach(t),P.forEach(t),he=p(O),N=r(O,"LI",{});var ge=i(N);me=s(ge,`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),W=r(ge,"CODE",{});var je=i(W);fe=s(je,'model({"input_ids": input_ids, "token_type_ids": token_type_ids})'),je.forEach(t),ge.forEach(t),O.forEach(t)},m(f,x){g(f,d,x),e(d,k),g(f,_,x),g(f,h,x),e(h,b),e(b,l),e(h,m),e(h,j),e(j,re),g(f,X,x),g(f,E,x),e(E,ae),e(E,I),e(I,Y),e(E,ne),e(E,S),e(S,ie),e(E,se),g(f,Z,x),g(f,D,x),e(D,G),g(f,J,x),g(f,F,x),e(F,q),e(q,le),e(q,H),e(H,V),e(q,ee),e(q,Q),e(Q,te),e(F,de),e(F,C),e(C,ce),e(C,U),e(U,pe),e(C,oe),e(C,B),e(B,ue),e(F,he),e(F,N),e(N,me),e(N,W),e(W,fe)},d(f){f&&t(d),f&&t(_),f&&t(h),f&&t(X),f&&t(E),f&&t(Z),f&&t(D),f&&t(J),f&&t(F)}}}function Hg(z){let d,k,_,h,b;return{c(){d=a("p"),k=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),_=a("code"),h=n("Module"),b=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=r(l,"P",{});var m=i(d);k=s(m,"Although the recipe for forward pass needs to be defined within this function, one should call the "),_=r(m,"CODE",{});var j=i(_);h=s(j,"Module"),j.forEach(t),b=s(m,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),m.forEach(t)},m(l,m){g(l,d,m),e(d,k),e(d,_),e(_,h),e(d,b)},d(l){l&&t(d)}}}function Qg(z){let d,k,_,h,b;return h=new Ke({props:{code:`import tensorflow as tf
from transformers import AutoTokenizer, TFLayoutLMForQuestionAnswering
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
model = TFLayoutLMForQuestionAnswering.from_pretrained("microsoft/layoutlm-base-uncased")

dataset = load_dataset("nielsr/funsd", split="train")
example = dataset[0]
question = "what's his name?"
words = example["words"]
boxes = example["bboxes"]

encoding = tokenizer(
    question.split(), words, is_split_into_words=True, return_token_type_ids=True, return_tensors="tf"
)
bbox = []
for i, s, w in zip(encoding.input_ids[0], encoding.sequence_ids(0), encoding.word_ids(0)):
    if s == 1:
        bbox.append(boxes[w])
    elif i == tokenizer.sep_token_id:
        bbox.append([1000] * 4)
    else:
        bbox.append([0] * 4)
encoding["bbox"] = tf.convert_to_tensor([bbox])

outputs = model(**encoding)
loss = outputs.loss
start_scores = outputs.start_logits
end_scores = outputs.end_logits`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> tensorflow <span class="hljs-keyword">as</span> tf
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, TFLayoutLMForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlm-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFLayoutLMForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlm-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = load_dataset(<span class="hljs-string">&quot;nielsr/funsd&quot;</span>, split=<span class="hljs-string">&quot;train&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>example = dataset[<span class="hljs-number">0</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>question = <span class="hljs-string">&quot;what&#x27;s his name?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>words = example[<span class="hljs-string">&quot;words&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>boxes = example[<span class="hljs-string">&quot;bboxes&quot;</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer(
<span class="hljs-meta">... </span>    question.split(), words, is_split_into_words=<span class="hljs-literal">True</span>, return_token_type_ids=<span class="hljs-literal">True</span>, return_tensors=<span class="hljs-string">&quot;tf&quot;</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>bbox = []
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> i, s, w <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(encoding.input_ids[<span class="hljs-number">0</span>], encoding.sequence_ids(<span class="hljs-number">0</span>), encoding.word_ids(<span class="hljs-number">0</span>)):
<span class="hljs-meta">... </span>    <span class="hljs-keyword">if</span> s == <span class="hljs-number">1</span>:
<span class="hljs-meta">... </span>        bbox.append(boxes[w])
<span class="hljs-meta">... </span>    <span class="hljs-keyword">elif</span> i == tokenizer.sep_token_id:
<span class="hljs-meta">... </span>        bbox.append([<span class="hljs-number">1000</span>] * <span class="hljs-number">4</span>)
<span class="hljs-meta">... </span>    <span class="hljs-keyword">else</span>:
<span class="hljs-meta">... </span>        bbox.append([<span class="hljs-number">0</span>] * <span class="hljs-number">4</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>encoding[<span class="hljs-string">&quot;bbox&quot;</span>] = tf.convert_to_tensor([bbox])

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**encoding)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>start_scores = outputs.start_logits
<span class="hljs-meta">&gt;&gt;&gt; </span>end_scores = outputs.end_logits`}}),{c(){d=a("p"),k=n("Examples:"),_=c(),L(h.$$.fragment)},l(l){d=r(l,"P",{});var m=i(d);k=s(m,"Examples:"),m.forEach(t),_=p(l),w(h.$$.fragment,l)},m(l,m){g(l,d,m),e(d,k),g(l,_,m),T(h,l,m),b=!0},p:ut,i(l){b||(v(h.$$.fragment,l),b=!0)},o(l){M(h.$$.fragment,l),b=!1},d(l){l&&t(d),l&&t(_),$(h,l)}}}function Bg(z){let d,k,_,h,b,l,m,j,re,X,E,ae,I,Y,ne,S,ie,se,Z,D,G,J,F,q,le,H,V,ee,Q,te,de,C,ce,U,pe,oe,B,ue,he,N,me,W,fe,f,x,K,Me,we,A,ke,$e,Te,O,R,ve,ye,P,xe,Le,ge,je,Os,ll,dl,va,cl,pl,Yo,ul,hl,Zo,ml,fl,Vr,Go,Xr,ht,gl,Ma,_l,yl,$a,bl,kl,Yr,Jo,Zr,Is,lt,Ll,Ds,wl,Tl,en,vl,Ml,tn,$l,xl,Gr,mt,jl,on,Fl,El,nn,zl,ql,Jr,bt,Rt,xa,sn,Cl,ja,Pl,ei,We,an,Nl,kt,Al,Ss,Ol,Il,rn,Dl,Sl,Ul,Lt,Wl,Us,Hl,Ql,Ws,Bl,Rl,Kl,Kt,ti,wt,Vt,Fa,ln,Vl,Ea,Xl,oi,He,dn,Yl,za,Zl,Gl,Xt,Hs,Jl,ed,Qs,td,od,nd,cn,sd,Bs,ad,rd,ni,Tt,Yt,qa,pn,id,Ca,ld,si,Qe,un,dd,Pa,cd,pd,Zt,Rs,ud,hd,Ks,md,fd,gd,hn,_d,Vs,yd,bd,ai,vt,Gt,Na,mn,kd,Aa,Ld,ri,Be,fn,wd,gn,Td,_n,vd,Md,$d,yn,xd,bn,jd,Fd,Ed,Ve,kn,zd,Mt,qd,Xs,Cd,Pd,Oa,Nd,Ad,Od,Jt,Id,eo,ii,$t,to,Ia,Ln,Dd,Da,Sd,li,Re,wn,Ud,xt,Wd,Sa,Hd,Qd,Tn,Bd,Rd,Kd,vn,Vd,Mn,Xd,Yd,Zd,Xe,$n,Gd,jt,Jd,Ys,ec,tc,Ua,oc,nc,sc,oo,ac,no,di,Ft,so,Wa,xn,rc,Ha,ic,ci,Ne,jn,lc,Fn,dc,En,cc,pc,uc,zn,hc,qn,mc,fc,gc,Cn,_c,Pn,yc,bc,kc,Ye,Nn,Lc,Et,wc,Zs,Tc,vc,Qa,Mc,$c,xc,ao,jc,ro,pi,zt,io,Ba,An,Fc,Ra,Ec,ui,Ae,On,zc,qt,qc,In,Cc,Pc,Dn,Nc,Ac,Oc,Sn,Ic,Un,Dc,Sc,Uc,Wn,Wc,Hn,Hc,Qc,Bc,Ze,Qn,Rc,Ct,Kc,Gs,Vc,Xc,Ka,Yc,Zc,Gc,lo,Jc,co,hi,Pt,po,Va,Bn,ep,Xa,tp,mi,Oe,Rn,op,dt,np,Kn,sp,ap,Ya,rp,ip,Za,lp,dp,cp,Vn,pp,Xn,up,hp,mp,Yn,fp,Zn,gp,_p,yp,Se,Gn,bp,be,kp,Ga,Lp,wp,Ja,Tp,vp,er,Mp,$p,tr,xp,jp,or,Fp,Ep,nr,zp,qp,sr,Cp,Pp,ar,Np,Ap,Op,rr,Ip,Dp,ir,Sp,Up,uo,fi,Nt,ho,lr,Jn,Wp,dr,Hp,gi,Ee,es,Qp,cr,Bp,Rp,ts,Kp,Js,Vp,Xp,Yp,os,Zp,ns,Gp,Jp,eu,mo,tu,Ge,ss,ou,At,nu,ea,su,au,pr,ru,iu,lu,fo,du,go,_i,Ot,_o,ur,as,cu,hr,pu,yi,ze,rs,uu,is,hu,mr,mu,fu,gu,ls,_u,ta,yu,bu,ku,ds,Lu,cs,wu,Tu,vu,yo,Mu,Je,ps,$u,It,xu,oa,ju,Fu,fr,Eu,zu,qu,bo,Cu,ko,bi,Dt,Lo,gr,us,Pu,_r,Nu,ki,qe,hs,Au,yr,Ou,Iu,ms,Du,na,Su,Uu,Wu,fs,Hu,gs,Qu,Bu,Ru,wo,Ku,et,_s,Vu,St,Xu,sa,Yu,Zu,br,Gu,Ju,eh,To,th,vo,Li,Ut,Mo,kr,ys,oh,Lr,nh,wi,Ce,bs,sh,wr,ah,rh,ks,ih,aa,lh,dh,ch,Ls,ph,ws,uh,hh,mh,$o,fh,tt,Ts,gh,Wt,_h,ra,yh,bh,Tr,kh,Lh,wh,xo,Th,jo,Ti,Ht,Fo,vr,vs,vh,Mr,Mh,vi,Pe,Ms,$h,ct,xh,$s,jh,Fh,$r,Eh,zh,xr,qh,Ch,Ph,xs,Nh,ia,Ah,Oh,Ih,js,Dh,Fs,Sh,Uh,Wh,Eo,Hh,ot,Es,Qh,Qt,Bh,la,Rh,Kh,jr,Vh,Xh,Yh,zo,Zh,qo,Mi;return l=new De({}),S=new De({}),Go=new Ke({props:{code:`def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]`,highlighted:`<span class="hljs-keyword">def</span> <span class="hljs-title function_">normalize_bbox</span>(<span class="hljs-params">bbox, width, height</span>):
    <span class="hljs-keyword">return</span> [
        <span class="hljs-built_in">int</span>(<span class="hljs-number">1000</span> * (bbox[<span class="hljs-number">0</span>] / width)),
        <span class="hljs-built_in">int</span>(<span class="hljs-number">1000</span> * (bbox[<span class="hljs-number">1</span>] / height)),
        <span class="hljs-built_in">int</span>(<span class="hljs-number">1000</span> * (bbox[<span class="hljs-number">2</span>] / width)),
        <span class="hljs-built_in">int</span>(<span class="hljs-number">1000</span> * (bbox[<span class="hljs-number">3</span>] / height)),
    ]`}}),Jo=new Ke({props:{code:`from PIL import Image

image = Image.open("name_of_your_document - can be a png file, pdf, etc.")

width, height = image.size`,highlighted:`<span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image

image = Image.<span class="hljs-built_in">open</span>(<span class="hljs-string">&quot;name_of_your_document - can be a png file, pdf, etc.&quot;</span>)

width, height = image.size`}}),sn=new De({}),an=new _e({props:{name:"class transformers.LayoutLMConfig",anchor:"transformers.LayoutLMConfig",parameters:[{name:"vocab_size",val:" = 30522"},{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 512"},{name:"type_vocab_size",val:" = 2"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"pad_token_id",val:" = 0"},{name:"max_2d_position_embeddings",val:" = 1024"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.LayoutLMConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 30522) &#x2014;
Vocabulary size of the LayoutLM model. Defines the different tokens that can be represented by the
<em>inputs_ids</em> passed to the forward method of <a href="/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMModel">LayoutLMModel</a>.`,name:"vocab_size"},{anchor:"transformers.LayoutLMConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.LayoutLMConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.LayoutLMConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.LayoutLMConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (i.e., feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.LayoutLMConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.LayoutLMConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.LayoutLMConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.LayoutLMConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.LayoutLMConfig.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed into <a href="/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMModel">LayoutLMModel</a>.`,name:"type_vocab_size"},{anchor:"transformers.LayoutLMConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.LayoutLMConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.LayoutLMConfig.max_2d_position_embeddings",description:`<strong>max_2d_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
The maximum value that the 2D position embedding might ever used. Typically set this to something large
just in case (e.g., 1024).`,name:"max_2d_position_embeddings"}],source:"https://github.com/huggingface/transformers/blob/vr_18407/src/transformers/models/layoutlm/configuration_layoutlm.py#L39"}}),Kt=new pt({props:{anchor:"transformers.LayoutLMConfig.example",$$slots:{default:[kg]},$$scope:{ctx:z}}}),ln=new De({}),dn=new _e({props:{name:"class transformers.LayoutLMTokenizer",anchor:"transformers.LayoutLMTokenizer",parameters:[{name:"vocab_file",val:""},{name:"do_lower_case",val:" = True"},{name:"do_basic_tokenize",val:" = True"},{name:"never_split",val:" = None"},{name:"unk_token",val:" = '[UNK]'"},{name:"sep_token",val:" = '[SEP]'"},{name:"pad_token",val:" = '[PAD]'"},{name:"cls_token",val:" = '[CLS]'"},{name:"mask_token",val:" = '[MASK]'"},{name:"tokenize_chinese_chars",val:" = True"},{name:"strip_accents",val:" = None"},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18407/src/transformers/models/layoutlm/tokenization_layoutlm.py#L50"}}),pn=new De({}),un=new _e({props:{name:"class transformers.LayoutLMTokenizerFast",anchor:"transformers.LayoutLMTokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"do_lower_case",val:" = True"},{name:"unk_token",val:" = '[UNK]'"},{name:"sep_token",val:" = '[SEP]'"},{name:"pad_token",val:" = '[PAD]'"},{name:"cls_token",val:" = '[CLS]'"},{name:"mask_token",val:" = '[MASK]'"},{name:"tokenize_chinese_chars",val:" = True"},{name:"strip_accents",val:" = None"},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18407/src/transformers/models/layoutlm/tokenization_layoutlm_fast.py#L59"}}),mn=new De({}),fn=new _e({props:{name:"class transformers.LayoutLMModel",anchor:"transformers.LayoutLMModel",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.LayoutLMModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMConfig">LayoutLMConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18407/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18407/src/transformers/models/layoutlm/modeling_layoutlm.py#L704"}}),kn=new _e({props:{name:"forward",anchor:"transformers.LayoutLMModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"bbox",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.LayoutLMModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMTokenizer">LayoutLMTokenizer</a>. See <a href="/docs/transformers/pr_18407/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18407/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LayoutLMModel.forward.bbox",description:`<strong>bbox</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length, 4)</code>, <em>optional</em>) &#x2014;
Bounding boxes of each input sequence tokens. Selected in the range <code>[0, config.max_2d_position_embeddings-1]</code>. Each bounding box should be a normalized version in (x0, y0, x1, y1)
format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
y1) represents the position of the lower right corner. See <a href="#Overview">Overview</a> for normalization.`,name:"bbox"},{anchor:"transformers.LayoutLMModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>: <code>1</code> for
tokens that are NOT MASKED, <code>0</code> for MASKED tokens.</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LayoutLMModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>: <code>0</code> corresponds to a <em>sentence A</em> token, <code>1</code> corresponds to a <em>sentence B</em> token</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.LayoutLMModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LayoutLMModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>: <code>1</code>
indicates the head is <strong>not masked</strong>, <code>0</code> indicates the head is <strong>masked</strong>.`,name:"head_mask"},{anchor:"transformers.LayoutLMModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <em>input_ids</em> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LayoutLMModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, the attentions tensors of all attention layers are returned. See <code>attentions</code> under
returned tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.LayoutLMModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, the hidden states of all layers are returned. See <code>hidden_states</code> under returned tensors
for more detail.`,name:"output_hidden_states"},{anchor:"transformers.LayoutLMModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, the model will return a <a href="/docs/transformers/pr_18407/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18407/src/transformers/models/layoutlm/modeling_layoutlm.py#L730",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18407/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMConfig"
>LayoutLMConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18407/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Jt=new Ue({props:{$$slots:{default:[Lg]},$$scope:{ctx:z}}}),eo=new pt({props:{anchor:"transformers.LayoutLMModel.forward.example",$$slots:{default:[wg]},$$scope:{ctx:z}}}),Ln=new De({}),wn=new _e({props:{name:"class transformers.LayoutLMForMaskedLM",anchor:"transformers.LayoutLMForMaskedLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.LayoutLMForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMConfig">LayoutLMConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18407/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18407/src/transformers/models/layoutlm/modeling_layoutlm.py#L852"}}),$n=new _e({props:{name:"forward",anchor:"transformers.LayoutLMForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"bbox",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"encoder_hidden_states",val:" = None"},{name:"encoder_attention_mask",val:" = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.LayoutLMForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMTokenizer">LayoutLMTokenizer</a>. See <a href="/docs/transformers/pr_18407/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18407/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LayoutLMForMaskedLM.forward.bbox",description:`<strong>bbox</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length, 4)</code>, <em>optional</em>) &#x2014;
Bounding boxes of each input sequence tokens. Selected in the range <code>[0, config.max_2d_position_embeddings-1]</code>. Each bounding box should be a normalized version in (x0, y0, x1, y1)
format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
y1) represents the position of the lower right corner. See <a href="#Overview">Overview</a> for normalization.`,name:"bbox"},{anchor:"transformers.LayoutLMForMaskedLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>: <code>1</code> for
tokens that are NOT MASKED, <code>0</code> for MASKED tokens.</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LayoutLMForMaskedLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>: <code>0</code> corresponds to a <em>sentence A</em> token, <code>1</code> corresponds to a <em>sentence B</em> token</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.LayoutLMForMaskedLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LayoutLMForMaskedLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>: <code>1</code>
indicates the head is <strong>not masked</strong>, <code>0</code> indicates the head is <strong>masked</strong>.`,name:"head_mask"},{anchor:"transformers.LayoutLMForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <em>input_ids</em> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LayoutLMForMaskedLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, the attentions tensors of all attention layers are returned. See <code>attentions</code> under
returned tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.LayoutLMForMaskedLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, the hidden states of all layers are returned. See <code>hidden_states</code> under returned tensors
for more detail.`,name:"output_hidden_states"},{anchor:"transformers.LayoutLMForMaskedLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, the model will return a <a href="/docs/transformers/pr_18407/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.LayoutLMForMaskedLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18407/src/transformers/models/layoutlm/modeling_layoutlm.py#L871",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18407/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMConfig"
>LayoutLMConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18407/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),oo=new Ue({props:{$$slots:{default:[Tg]},$$scope:{ctx:z}}}),no=new pt({props:{anchor:"transformers.LayoutLMForMaskedLM.forward.example",$$slots:{default:[vg]},$$scope:{ctx:z}}}),xn=new De({}),jn=new _e({props:{name:"class transformers.LayoutLMForSequenceClassification",anchor:"transformers.LayoutLMForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.LayoutLMForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMConfig">LayoutLMConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18407/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18407/src/transformers/models/layoutlm/modeling_layoutlm.py#L981"}}),Nn=new _e({props:{name:"forward",anchor:"transformers.LayoutLMForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"bbox",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.LayoutLMForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMTokenizer">LayoutLMTokenizer</a>. See <a href="/docs/transformers/pr_18407/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18407/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LayoutLMForSequenceClassification.forward.bbox",description:`<strong>bbox</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length, 4)</code>, <em>optional</em>) &#x2014;
Bounding boxes of each input sequence tokens. Selected in the range <code>[0, config.max_2d_position_embeddings-1]</code>. Each bounding box should be a normalized version in (x0, y0, x1, y1)
format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
y1) represents the position of the lower right corner. See <a href="#Overview">Overview</a> for normalization.`,name:"bbox"},{anchor:"transformers.LayoutLMForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>: <code>1</code> for
tokens that are NOT MASKED, <code>0</code> for MASKED tokens.</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LayoutLMForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>: <code>0</code> corresponds to a <em>sentence A</em> token, <code>1</code> corresponds to a <em>sentence B</em> token</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.LayoutLMForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LayoutLMForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>: <code>1</code>
indicates the head is <strong>not masked</strong>, <code>0</code> indicates the head is <strong>masked</strong>.`,name:"head_mask"},{anchor:"transformers.LayoutLMForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <em>input_ids</em> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LayoutLMForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, the attentions tensors of all attention layers are returned. See <code>attentions</code> under
returned tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.LayoutLMForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, the hidden states of all layers are returned. See <code>hidden_states</code> under returned tensors
for more detail.`,name:"output_hidden_states"},{anchor:"transformers.LayoutLMForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, the model will return a <a href="/docs/transformers/pr_18407/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.LayoutLMForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18407/src/transformers/models/layoutlm/modeling_layoutlm.py#L995",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18407/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMConfig"
>LayoutLMConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18407/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ao=new Ue({props:{$$slots:{default:[Mg]},$$scope:{ctx:z}}}),ro=new pt({props:{anchor:"transformers.LayoutLMForSequenceClassification.forward.example",$$slots:{default:[$g]},$$scope:{ctx:z}}}),An=new De({}),On=new _e({props:{name:"class transformers.LayoutLMForTokenClassification",anchor:"transformers.LayoutLMForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.LayoutLMForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMConfig">LayoutLMConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18407/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18407/src/transformers/models/layoutlm/modeling_layoutlm.py#L1118"}}),Qn=new _e({props:{name:"forward",anchor:"transformers.LayoutLMForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"bbox",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.LayoutLMForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMTokenizer">LayoutLMTokenizer</a>. See <a href="/docs/transformers/pr_18407/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18407/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LayoutLMForTokenClassification.forward.bbox",description:`<strong>bbox</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length, 4)</code>, <em>optional</em>) &#x2014;
Bounding boxes of each input sequence tokens. Selected in the range <code>[0, config.max_2d_position_embeddings-1]</code>. Each bounding box should be a normalized version in (x0, y0, x1, y1)
format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
y1) represents the position of the lower right corner. See <a href="#Overview">Overview</a> for normalization.`,name:"bbox"},{anchor:"transformers.LayoutLMForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>: <code>1</code> for
tokens that are NOT MASKED, <code>0</code> for MASKED tokens.</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LayoutLMForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>: <code>0</code> corresponds to a <em>sentence A</em> token, <code>1</code> corresponds to a <em>sentence B</em> token</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.LayoutLMForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LayoutLMForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>: <code>1</code>
indicates the head is <strong>not masked</strong>, <code>0</code> indicates the head is <strong>masked</strong>.`,name:"head_mask"},{anchor:"transformers.LayoutLMForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <em>input_ids</em> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LayoutLMForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, the attentions tensors of all attention layers are returned. See <code>attentions</code> under
returned tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.LayoutLMForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, the hidden states of all layers are returned. See <code>hidden_states</code> under returned tensors
for more detail.`,name:"output_hidden_states"},{anchor:"transformers.LayoutLMForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, the model will return a <a href="/docs/transformers/pr_18407/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.LayoutLMForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18407/src/transformers/models/layoutlm/modeling_layoutlm.py#L1132",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18407/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMConfig"
>LayoutLMConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18407/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),lo=new Ue({props:{$$slots:{default:[xg]},$$scope:{ctx:z}}}),co=new pt({props:{anchor:"transformers.LayoutLMForTokenClassification.forward.example",$$slots:{default:[jg]},$$scope:{ctx:z}}}),Bn=new De({}),Rn=new _e({props:{name:"class transformers.LayoutLMForQuestionAnswering",anchor:"transformers.LayoutLMForQuestionAnswering",parameters:[{name:"config",val:""},{name:"has_visual_segment_embedding",val:" = True"}],parametersDescription:[{anchor:"transformers.LayoutLMForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMConfig">LayoutLMConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18407/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18407/src/transformers/models/layoutlm/modeling_layoutlm.py#L1236"}}),Gn=new _e({props:{name:"forward",anchor:"transformers.LayoutLMForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"bbox",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],source:"https://github.com/huggingface/transformers/blob/vr_18407/src/transformers/models/layoutlm/modeling_layoutlm.py#L1250",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18407/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMConfig"
>LayoutLMConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18407/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),uo=new pt({props:{anchor:"transformers.LayoutLMForQuestionAnswering.forward.example",$$slots:{default:[Fg]},$$scope:{ctx:z}}}),Jn=new De({}),es=new _e({props:{name:"class transformers.TFLayoutLMModel",anchor:"transformers.TFLayoutLMModel",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFLayoutLMModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMConfig">LayoutLMConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18407/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18407/src/transformers/models/layoutlm/modeling_tf_layoutlm.py#L906"}}),mo=new Ue({props:{$$slots:{default:[Eg]},$$scope:{ctx:z}}}),ss=new _e({props:{name:"call",anchor:"transformers.TFLayoutLMModel.call",parameters:[{name:"input_ids",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"bbox",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"token_type_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"position_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"head_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"inputs_embeds",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"encoder_hidden_states",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"encoder_attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"training",val:": typing.Optional[bool] = False"}],parametersDescription:[{anchor:"transformers.TFLayoutLMModel.call.input_ids",description:`<strong>input_ids</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMTokenizer">LayoutLMTokenizer</a>. See <a href="/docs/transformers/pr_18407/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> and
<a href="/docs/transformers/pr_18407/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.TFLayoutLMModel.call.bbox",description:`<strong>bbox</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, 4)</code>, <em>optional</em>) &#x2014;
Bounding Boxes of each input sequence tokens. Selected in the range <code>[0, config.max_2d_position_embeddings- 1]</code>.`,name:"bbox"},{anchor:"transformers.TFLayoutLMModel.call.attention_mask",description:`<strong>attention_mask</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.TFLayoutLMModel.call.token_type_ids",description:`<strong>token_type_ids</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.TFLayoutLMModel.call.position_ids",description:`<strong>position_ids</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.TFLayoutLMModel.call.head_mask",description:`<strong>head_mask</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.TFLayoutLMModel.call.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.TFLayoutLMModel.call.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.TFLayoutLMModel.call.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.TFLayoutLMModel.call.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18407/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.TFLayoutLMModel.call.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"}],source:"https://github.com/huggingface/transformers/blob/vr_18407/src/transformers/models/layoutlm/modeling_tf_layoutlm.py#L912",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18407/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutputWithPoolingAndCrossAttentions"
>transformers.modeling_tf_outputs.TFBaseModelOutputWithPoolingAndCrossAttentions</a> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<a
  href="/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMConfig"
>LayoutLMConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18407/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutputWithPoolingAndCrossAttentions"
>transformers.modeling_tf_outputs.TFBaseModelOutputWithPoolingAndCrossAttentions</a> or <code>tuple(tf.Tensor)</code></p>
`}}),fo=new Ue({props:{$$slots:{default:[zg]},$$scope:{ctx:z}}}),go=new pt({props:{anchor:"transformers.TFLayoutLMModel.call.example",$$slots:{default:[qg]},$$scope:{ctx:z}}}),as=new De({}),rs=new _e({props:{name:"class transformers.TFLayoutLMForMaskedLM",anchor:"transformers.TFLayoutLMForMaskedLM",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFLayoutLMForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMConfig">LayoutLMConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18407/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18407/src/transformers/models/layoutlm/modeling_tf_layoutlm.py#L1006"}}),yo=new Ue({props:{$$slots:{default:[Cg]},$$scope:{ctx:z}}}),ps=new _e({props:{name:"call",anchor:"transformers.TFLayoutLMForMaskedLM.call",parameters:[{name:"input_ids",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"bbox",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"token_type_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"position_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"head_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"inputs_embeds",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"labels",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"training",val:": typing.Optional[bool] = False"}],parametersDescription:[{anchor:"transformers.TFLayoutLMForMaskedLM.call.input_ids",description:`<strong>input_ids</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMTokenizer">LayoutLMTokenizer</a>. See <a href="/docs/transformers/pr_18407/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> and
<a href="/docs/transformers/pr_18407/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.TFLayoutLMForMaskedLM.call.bbox",description:`<strong>bbox</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, 4)</code>, <em>optional</em>) &#x2014;
Bounding Boxes of each input sequence tokens. Selected in the range <code>[0, config.max_2d_position_embeddings- 1]</code>.`,name:"bbox"},{anchor:"transformers.TFLayoutLMForMaskedLM.call.attention_mask",description:`<strong>attention_mask</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.TFLayoutLMForMaskedLM.call.token_type_ids",description:`<strong>token_type_ids</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.TFLayoutLMForMaskedLM.call.position_ids",description:`<strong>position_ids</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.TFLayoutLMForMaskedLM.call.head_mask",description:`<strong>head_mask</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.TFLayoutLMForMaskedLM.call.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.TFLayoutLMForMaskedLM.call.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.TFLayoutLMForMaskedLM.call.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.TFLayoutLMForMaskedLM.call.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18407/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.TFLayoutLMForMaskedLM.call.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"},{anchor:"transformers.TFLayoutLMForMaskedLM.call.labels",description:`<strong>labels</strong> (<code>tf.Tensor</code> or <code>np.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18407/src/transformers/models/layoutlm/modeling_tf_layoutlm.py#L1034",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18407/en/main_classes/output#transformers.modeling_tf_outputs.TFMaskedLMOutput"
>transformers.modeling_tf_outputs.TFMaskedLMOutput</a> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<a
  href="/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMConfig"
>LayoutLMConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18407/en/main_classes/output#transformers.modeling_tf_outputs.TFMaskedLMOutput"
>transformers.modeling_tf_outputs.TFMaskedLMOutput</a> or <code>tuple(tf.Tensor)</code></p>
`}}),bo=new Ue({props:{$$slots:{default:[Pg]},$$scope:{ctx:z}}}),ko=new pt({props:{anchor:"transformers.TFLayoutLMForMaskedLM.call.example",$$slots:{default:[Ng]},$$scope:{ctx:z}}}),us=new De({}),hs=new _e({props:{name:"class transformers.TFLayoutLMForSequenceClassification",anchor:"transformers.TFLayoutLMForSequenceClassification",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFLayoutLMForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMConfig">LayoutLMConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18407/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18407/src/transformers/models/layoutlm/modeling_tf_layoutlm.py#L1139"}}),wo=new Ue({props:{$$slots:{default:[Ag]},$$scope:{ctx:z}}}),_s=new _e({props:{name:"call",anchor:"transformers.TFLayoutLMForSequenceClassification.call",parameters:[{name:"input_ids",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"bbox",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"token_type_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"position_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"head_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"inputs_embeds",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"labels",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"training",val:": typing.Optional[bool] = False"}],parametersDescription:[{anchor:"transformers.TFLayoutLMForSequenceClassification.call.input_ids",description:`<strong>input_ids</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMTokenizer">LayoutLMTokenizer</a>. See <a href="/docs/transformers/pr_18407/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> and
<a href="/docs/transformers/pr_18407/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.TFLayoutLMForSequenceClassification.call.bbox",description:`<strong>bbox</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, 4)</code>, <em>optional</em>) &#x2014;
Bounding Boxes of each input sequence tokens. Selected in the range <code>[0, config.max_2d_position_embeddings- 1]</code>.`,name:"bbox"},{anchor:"transformers.TFLayoutLMForSequenceClassification.call.attention_mask",description:`<strong>attention_mask</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.TFLayoutLMForSequenceClassification.call.token_type_ids",description:`<strong>token_type_ids</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.TFLayoutLMForSequenceClassification.call.position_ids",description:`<strong>position_ids</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.TFLayoutLMForSequenceClassification.call.head_mask",description:`<strong>head_mask</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.TFLayoutLMForSequenceClassification.call.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.TFLayoutLMForSequenceClassification.call.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.TFLayoutLMForSequenceClassification.call.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.TFLayoutLMForSequenceClassification.call.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18407/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.TFLayoutLMForSequenceClassification.call.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"},{anchor:"transformers.TFLayoutLMForSequenceClassification.call.labels",description:`<strong>labels</strong> (<code>tf.Tensor</code> or <code>np.ndarray</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18407/src/transformers/models/layoutlm/modeling_tf_layoutlm.py#L1157",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18407/en/main_classes/output#transformers.modeling_tf_outputs.TFSequenceClassifierOutput"
>transformers.modeling_tf_outputs.TFSequenceClassifierOutput</a> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<a
  href="/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMConfig"
>LayoutLMConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18407/en/main_classes/output#transformers.modeling_tf_outputs.TFSequenceClassifierOutput"
>transformers.modeling_tf_outputs.TFSequenceClassifierOutput</a> or <code>tuple(tf.Tensor)</code></p>
`}}),To=new Ue({props:{$$slots:{default:[Og]},$$scope:{ctx:z}}}),vo=new pt({props:{anchor:"transformers.TFLayoutLMForSequenceClassification.call.example",$$slots:{default:[Ig]},$$scope:{ctx:z}}}),ys=new De({}),bs=new _e({props:{name:"class transformers.TFLayoutLMForTokenClassification",anchor:"transformers.TFLayoutLMForTokenClassification",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFLayoutLMForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMConfig">LayoutLMConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18407/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18407/src/transformers/models/layoutlm/modeling_tf_layoutlm.py#L1263"}}),$o=new Ue({props:{$$slots:{default:[Dg]},$$scope:{ctx:z}}}),Ts=new _e({props:{name:"call",anchor:"transformers.TFLayoutLMForTokenClassification.call",parameters:[{name:"input_ids",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"bbox",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"token_type_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"position_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"head_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"inputs_embeds",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"labels",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"training",val:": typing.Optional[bool] = False"}],parametersDescription:[{anchor:"transformers.TFLayoutLMForTokenClassification.call.input_ids",description:`<strong>input_ids</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMTokenizer">LayoutLMTokenizer</a>. See <a href="/docs/transformers/pr_18407/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> and
<a href="/docs/transformers/pr_18407/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.TFLayoutLMForTokenClassification.call.bbox",description:`<strong>bbox</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, 4)</code>, <em>optional</em>) &#x2014;
Bounding Boxes of each input sequence tokens. Selected in the range <code>[0, config.max_2d_position_embeddings- 1]</code>.`,name:"bbox"},{anchor:"transformers.TFLayoutLMForTokenClassification.call.attention_mask",description:`<strong>attention_mask</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.TFLayoutLMForTokenClassification.call.token_type_ids",description:`<strong>token_type_ids</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.TFLayoutLMForTokenClassification.call.position_ids",description:`<strong>position_ids</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.TFLayoutLMForTokenClassification.call.head_mask",description:`<strong>head_mask</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.TFLayoutLMForTokenClassification.call.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.TFLayoutLMForTokenClassification.call.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.TFLayoutLMForTokenClassification.call.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.TFLayoutLMForTokenClassification.call.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18407/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.TFLayoutLMForTokenClassification.call.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"},{anchor:"transformers.TFLayoutLMForTokenClassification.call.labels",description:`<strong>labels</strong> (<code>tf.Tensor</code> or <code>np.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18407/src/transformers/models/layoutlm/modeling_tf_layoutlm.py#L1287",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18407/en/main_classes/output#transformers.modeling_tf_outputs.TFTokenClassifierOutput"
>transformers.modeling_tf_outputs.TFTokenClassifierOutput</a> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<a
  href="/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMConfig"
>LayoutLMConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18407/en/main_classes/output#transformers.modeling_tf_outputs.TFTokenClassifierOutput"
>transformers.modeling_tf_outputs.TFTokenClassifierOutput</a> or <code>tuple(tf.Tensor)</code></p>
`}}),xo=new Ue({props:{$$slots:{default:[Sg]},$$scope:{ctx:z}}}),jo=new pt({props:{anchor:"transformers.TFLayoutLMForTokenClassification.call.example",$$slots:{default:[Ug]},$$scope:{ctx:z}}}),vs=new De({}),Ms=new _e({props:{name:"class transformers.TFLayoutLMForQuestionAnswering",anchor:"transformers.TFLayoutLMForQuestionAnswering",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFLayoutLMForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMConfig">LayoutLMConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18407/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18407/src/transformers/models/layoutlm/modeling_tf_layoutlm.py#L1392"}}),Eo=new Ue({props:{$$slots:{default:[Wg]},$$scope:{ctx:z}}}),Es=new _e({props:{name:"call",anchor:"transformers.TFLayoutLMForQuestionAnswering.call",parameters:[{name:"input_ids",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"bbox",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"token_type_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"position_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"head_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"inputs_embeds",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"start_positions",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"end_positions",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"training",val:": typing.Optional[bool] = False"}],parametersDescription:[{anchor:"transformers.TFLayoutLMForQuestionAnswering.call.input_ids",description:`<strong>input_ids</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMTokenizer">LayoutLMTokenizer</a>. See <a href="/docs/transformers/pr_18407/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> and
<a href="/docs/transformers/pr_18407/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.TFLayoutLMForQuestionAnswering.call.bbox",description:`<strong>bbox</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, 4)</code>, <em>optional</em>) &#x2014;
Bounding Boxes of each input sequence tokens. Selected in the range <code>[0, config.max_2d_position_embeddings- 1]</code>.`,name:"bbox"},{anchor:"transformers.TFLayoutLMForQuestionAnswering.call.attention_mask",description:`<strong>attention_mask</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.TFLayoutLMForQuestionAnswering.call.token_type_ids",description:`<strong>token_type_ids</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.TFLayoutLMForQuestionAnswering.call.position_ids",description:`<strong>position_ids</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.TFLayoutLMForQuestionAnswering.call.head_mask",description:`<strong>head_mask</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.TFLayoutLMForQuestionAnswering.call.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.TFLayoutLMForQuestionAnswering.call.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.TFLayoutLMForQuestionAnswering.call.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.TFLayoutLMForQuestionAnswering.call.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18407/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.TFLayoutLMForQuestionAnswering.call.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"},{anchor:"transformers.TFLayoutLMForQuestionAnswering.call.start_positions",description:`<strong>start_positions</strong> (<code>tf.Tensor</code> or <code>np.ndarray</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.TFLayoutLMForQuestionAnswering.call.end_positions",description:`<strong>end_positions</strong> (<code>tf.Tensor</code> or <code>np.ndarray</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"}],source:"https://github.com/huggingface/transformers/blob/vr_18407/src/transformers/models/layoutlm/modeling_tf_layoutlm.py#L1413",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18407/en/main_classes/output#transformers.modeling_tf_outputs.TFQuestionAnsweringModelOutput"
>transformers.modeling_tf_outputs.TFQuestionAnsweringModelOutput</a> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<a
  href="/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMConfig"
>LayoutLMConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18407/en/main_classes/output#transformers.modeling_tf_outputs.TFQuestionAnsweringModelOutput"
>transformers.modeling_tf_outputs.TFQuestionAnsweringModelOutput</a> or <code>tuple(tf.Tensor)</code></p>
`}}),zo=new Ue({props:{$$slots:{default:[Hg]},$$scope:{ctx:z}}}),qo=new pt({props:{anchor:"transformers.TFLayoutLMForQuestionAnswering.call.example",$$slots:{default:[Qg]},$$scope:{ctx:z}}}),{c(){d=a("meta"),k=c(),_=a("h1"),h=a("a"),b=a("span"),L(l.$$.fragment),m=c(),j=a("span"),re=n("LayoutLM"),X=c(),E=a("a"),ae=c(),I=a("h2"),Y=a("a"),ne=a("span"),L(S.$$.fragment),ie=c(),se=a("span"),Z=n("Overview"),D=c(),G=a("p"),J=n("The LayoutLM model was proposed in the paper "),F=a("a"),q=n(`LayoutLM: Pre-training of Text and Layout for Document Image
Understanding`),le=n(` by Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and
Ming Zhou. It\u2019s a simple but effective pretraining method of text and layout for document image understanding and
information extraction tasks, such as form understanding and receipt understanding. It obtains state-of-the-art results
on several downstream tasks:`),H=c(),V=a("ul"),ee=a("li"),Q=n("form understanding: the "),te=a("a"),de=n("FUNSD"),C=n(` dataset (a collection of 199 annotated
forms comprising more than 30,000 words).`),ce=c(),U=a("li"),pe=n("receipt understanding: the "),oe=a("a"),B=n("SROIE"),ue=n(` dataset (a collection of 626 receipts for
training and 347 receipts for testing).`),he=c(),N=a("li"),me=n("document image classification: the "),W=a("a"),fe=n("RVL-CDIP"),f=n(` dataset (a collection of
400,000 images belonging to one of 16 classes).`),x=c(),K=a("p"),Me=n("The abstract from the paper is the following:"),we=c(),A=a("p"),ke=a("em"),$e=n(`Pre-training techniques have been verified successfully in a variety of NLP tasks in recent years. Despite the
widespread use of pretraining models for NLP applications, they almost exclusively focus on text-level manipulation,
while neglecting layout and style information that is vital for document image understanding. In this paper, we propose
the LayoutLM to jointly model interactions between text and layout information across scanned document images, which is
beneficial for a great number of real-world document image understanding tasks such as information extraction from
scanned documents. Furthermore, we also leverage image features to incorporate words\u2019 visual information into LayoutLM.
To the best of our knowledge, this is the first time that text and layout are jointly learned in a single framework for
document-level pretraining. It achieves new state-of-the-art results in several downstream tasks, including form
understanding (from 70.72 to 79.27), receipt understanding (from 94.02 to 95.24) and document image classification
(from 93.07 to 94.42).`),Te=c(),O=a("p"),R=n("Tips:"),ve=c(),ye=a("ul"),P=a("li"),xe=n("In addition to "),Le=a("em"),ge=n("input_ids"),je=n(", "),Os=a("a"),ll=n("forward()"),dl=n(" also expects the input "),va=a("code"),cl=n("bbox"),pl=n(`, which are
the bounding boxes (i.e. 2D-positions) of the input tokens. These can be obtained using an external OCR engine such
as Google\u2019s `),Yo=a("a"),ul=n("Tesseract"),hl=n(" (there\u2019s a "),Zo=a("a"),ml=n("Python wrapper"),fl=n(` available). Each bounding box should be in (x0, y0, x1, y1) format, where
(x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1, y1) represents the
position of the lower right corner. Note that one first needs to normalize the bounding boxes to be on a 0-1000
scale. To normalize, you can use the following function:`),Vr=c(),L(Go.$$.fragment),Xr=c(),ht=a("p"),gl=n("Here, "),Ma=a("code"),_l=n("width"),yl=n(" and "),$a=a("code"),bl=n("height"),kl=n(` correspond to the width and height of the original document in which the token
occurs. Those can be obtained using the Python Image Library (PIL) library for example, as follows:`),Yr=c(),L(Jo.$$.fragment),Zr=c(),Is=a("ul"),lt=a("li"),Ll=n("For a demo which shows how to fine-tune "),Ds=a("a"),wl=n("LayoutLMForTokenClassification"),Tl=n(" on the "),en=a("a"),vl=n("FUNSD dataset"),Ml=n(" (a collection of annotated forms), see "),tn=a("a"),$l=n("this notebook"),xl=n(`.
It includes an inference part, which shows how to use Google\u2019s Tesseract on a new document.`),Gr=c(),mt=a("p"),jl=n("This model was contributed by "),on=a("a"),Fl=n("liminghao1630"),El=n(`. The original code can be found
`),nn=a("a"),zl=n("here"),ql=n("."),Jr=c(),bt=a("h2"),Rt=a("a"),xa=a("span"),L(sn.$$.fragment),Cl=c(),ja=a("span"),Pl=n("LayoutLMConfig"),ei=c(),We=a("div"),L(an.$$.fragment),Nl=c(),kt=a("p"),Al=n("This is the configuration class to store the configuration of a "),Ss=a("a"),Ol=n("LayoutLMModel"),Il=n(`. It is used to instantiate a
LayoutLM model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the LayoutLM
`),rn=a("a"),Dl=n("microsoft/layoutlm-base-uncased"),Sl=n(" architecture."),Ul=c(),Lt=a("p"),Wl=n("Configuration objects inherit from "),Us=a("a"),Hl=n("BertConfig"),Ql=n(` and can be used to control the model outputs. Read the
documentation from `),Ws=a("a"),Bl=n("BertConfig"),Rl=n(" for more information."),Kl=c(),L(Kt.$$.fragment),ti=c(),wt=a("h2"),Vt=a("a"),Fa=a("span"),L(ln.$$.fragment),Vl=c(),Ea=a("span"),Xl=n("LayoutLMTokenizer"),oi=c(),He=a("div"),L(dn.$$.fragment),Yl=c(),za=a("p"),Zl=n("Constructs a LayoutLM tokenizer."),Gl=c(),Xt=a("p"),Hs=a("a"),Jl=n("LayoutLMTokenizer"),ed=n(" is identical to "),Qs=a("a"),td=n("BertTokenizer"),od=n(` and runs end-to-end tokenization: punctuation splitting +
wordpiece.`),nd=c(),cn=a("p"),sd=n("Refer to superclass "),Bs=a("a"),ad=n("BertTokenizer"),rd=n(" for usage examples and documentation concerning parameters."),ni=c(),Tt=a("h2"),Yt=a("a"),qa=a("span"),L(pn.$$.fragment),id=c(),Ca=a("span"),ld=n("LayoutLMTokenizerFast"),si=c(),Qe=a("div"),L(un.$$.fragment),dd=c(),Pa=a("p"),cd=n("Constructs a \u201CFast\u201D LayoutLMTokenizer."),pd=c(),Zt=a("p"),Rs=a("a"),ud=n("LayoutLMTokenizerFast"),hd=n(" is identical to "),Ks=a("a"),md=n("BertTokenizerFast"),fd=n(` and runs end-to-end tokenization: punctuation
splitting + wordpiece.`),gd=c(),hn=a("p"),_d=n("Refer to superclass "),Vs=a("a"),yd=n("BertTokenizerFast"),bd=n(" for usage examples and documentation concerning parameters."),ai=c(),vt=a("h2"),Gt=a("a"),Na=a("span"),L(mn.$$.fragment),kd=c(),Aa=a("span"),Ld=n("LayoutLMModel"),ri=c(),Be=a("div"),L(fn.$$.fragment),wd=c(),gn=a("p"),Td=n(`The bare LayoutLM Model transformer outputting raw hidden-states without any specific head on top.
The LayoutLM model was proposed in `),_n=a("a"),vd=n(`LayoutLM: Pre-training of Text and Layout for Document Image
Understanding`),Md=n(` by Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei and
Ming Zhou.`),$d=c(),yn=a("p"),xd=n("This model is a PyTorch "),bn=a("a"),jd=n("torch.nn.Module"),Fd=n(` sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),Ed=c(),Ve=a("div"),L(kn.$$.fragment),zd=c(),Mt=a("p"),qd=n("The "),Xs=a("a"),Cd=n("LayoutLMModel"),Pd=n(" forward method, overrides the "),Oa=a("code"),Nd=n("__call__"),Ad=n(" special method."),Od=c(),L(Jt.$$.fragment),Id=c(),L(eo.$$.fragment),ii=c(),$t=a("h2"),to=a("a"),Ia=a("span"),L(Ln.$$.fragment),Dd=c(),Da=a("span"),Sd=n("LayoutLMForMaskedLM"),li=c(),Re=a("div"),L(wn.$$.fragment),Ud=c(),xt=a("p"),Wd=n("LayoutLM Model with a "),Sa=a("code"),Hd=n("language modeling"),Qd=n(` head on top.
The LayoutLM model was proposed in `),Tn=a("a"),Bd=n(`LayoutLM: Pre-training of Text and Layout for Document Image
Understanding`),Rd=n(` by Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei and
Ming Zhou.`),Kd=c(),vn=a("p"),Vd=n("This model is a PyTorch "),Mn=a("a"),Xd=n("torch.nn.Module"),Yd=n(` sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),Zd=c(),Xe=a("div"),L($n.$$.fragment),Gd=c(),jt=a("p"),Jd=n("The "),Ys=a("a"),ec=n("LayoutLMForMaskedLM"),tc=n(" forward method, overrides the "),Ua=a("code"),oc=n("__call__"),nc=n(" special method."),sc=c(),L(oo.$$.fragment),ac=c(),L(no.$$.fragment),di=c(),Ft=a("h2"),so=a("a"),Wa=a("span"),L(xn.$$.fragment),rc=c(),Ha=a("span"),ic=n("LayoutLMForSequenceClassification"),ci=c(),Ne=a("div"),L(jn.$$.fragment),lc=c(),Fn=a("p"),dc=n(`LayoutLM Model with a sequence classification head on top (a linear layer on top of the pooled output) e.g. for
document image classification tasks such as the `),En=a("a"),cc=n("RVL-CDIP"),pc=n(" dataset."),uc=c(),zn=a("p"),hc=n("The LayoutLM model was proposed in "),qn=a("a"),mc=n(`LayoutLM: Pre-training of Text and Layout for Document Image
Understanding`),fc=n(` by Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei and
Ming Zhou.`),gc=c(),Cn=a("p"),_c=n("This model is a PyTorch "),Pn=a("a"),yc=n("torch.nn.Module"),bc=n(` sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),kc=c(),Ye=a("div"),L(Nn.$$.fragment),Lc=c(),Et=a("p"),wc=n("The "),Zs=a("a"),Tc=n("LayoutLMForSequenceClassification"),vc=n(" forward method, overrides the "),Qa=a("code"),Mc=n("__call__"),$c=n(" special method."),xc=c(),L(ao.$$.fragment),jc=c(),L(ro.$$.fragment),pi=c(),zt=a("h2"),io=a("a"),Ba=a("span"),L(An.$$.fragment),Fc=c(),Ra=a("span"),Ec=n("LayoutLMForTokenClassification"),ui=c(),Ae=a("div"),L(On.$$.fragment),zc=c(),qt=a("p"),qc=n(`LayoutLM Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
sequence labeling (information extraction) tasks such as the `),In=a("a"),Cc=n("FUNSD"),Pc=n(`
dataset and the `),Dn=a("a"),Nc=n("SROIE"),Ac=n(" dataset."),Oc=c(),Sn=a("p"),Ic=n("The LayoutLM model was proposed in "),Un=a("a"),Dc=n(`LayoutLM: Pre-training of Text and Layout for Document Image
Understanding`),Sc=n(` by Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei and
Ming Zhou.`),Uc=c(),Wn=a("p"),Wc=n("This model is a PyTorch "),Hn=a("a"),Hc=n("torch.nn.Module"),Qc=n(` sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),Bc=c(),Ze=a("div"),L(Qn.$$.fragment),Rc=c(),Ct=a("p"),Kc=n("The "),Gs=a("a"),Vc=n("LayoutLMForTokenClassification"),Xc=n(" forward method, overrides the "),Ka=a("code"),Yc=n("__call__"),Zc=n(" special method."),Gc=c(),L(lo.$$.fragment),Jc=c(),L(co.$$.fragment),hi=c(),Pt=a("h2"),po=a("a"),Va=a("span"),L(Bn.$$.fragment),ep=c(),Xa=a("span"),tp=n("LayoutLMForQuestionAnswering"),mi=c(),Oe=a("div"),L(Rn.$$.fragment),op=c(),dt=a("p"),np=n(`LayoutLM Model with a span classification head on top for extractive question-answering tasks such as
`),Kn=a("a"),sp=n("DocVQA"),ap=n(" (a linear layer on top of the final hidden-states output to compute "),Ya=a("code"),rp=n("span start logits"),ip=n(" and "),Za=a("code"),lp=n("span end logits"),dp=n(")."),cp=c(),Vn=a("p"),pp=n("The LayoutLM model was proposed in "),Xn=a("a"),up=n(`LayoutLM: Pre-training of Text and Layout for Document Image
Understanding`),hp=n(` by Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei and
Ming Zhou.`),mp=c(),Yn=a("p"),fp=n("This model is a PyTorch "),Zn=a("a"),gp=n("torch.nn.Module"),_p=n(` sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),yp=c(),Se=a("div"),L(Gn.$$.fragment),bp=c(),be=a("p"),kp=n("start_positions ("),Ga=a("code"),Lp=n("torch.LongTensor"),wp=n(" of shape "),Ja=a("code"),Tp=n("(batch_size,)"),vp=n(", "),er=a("em"),Mp=n("optional"),$p=n(`):
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (`),tr=a("code"),xp=n("sequence_length"),jp=n(`). Position outside of the sequence
are not taken into account for computing the loss.
end_positions (`),or=a("code"),Fp=n("torch.LongTensor"),Ep=n(" of shape "),nr=a("code"),zp=n("(batch_size,)"),qp=n(", "),sr=a("em"),Cp=n("optional"),Pp=n(`):
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (`),ar=a("code"),Np=n("sequence_length"),Ap=n(`). Position outside of the sequence
are not taken into account for computing the loss.`),Op=c(),rr=a("p"),Ip=n("Example:"),Dp=c(),ir=a("p"),Sp=n(`In the example below, we prepare a question + context pair for the LayoutLM model. It will give us a prediction
of what it thinks the answer is (the span of the answer within the texts parsed from the image).`),Up=c(),L(uo.$$.fragment),fi=c(),Nt=a("h2"),ho=a("a"),lr=a("span"),L(Jn.$$.fragment),Wp=c(),dr=a("span"),Hp=n("TFLayoutLMModel"),gi=c(),Ee=a("div"),L(es.$$.fragment),Qp=c(),cr=a("p"),Bp=n("The bare LayoutLM Model transformer outputting raw hidden-states without any specific head on top."),Rp=c(),ts=a("p"),Kp=n("This model inherits from "),Js=a("a"),Vp=n("TFPreTrainedModel"),Xp=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Yp=c(),os=a("p"),Zp=n("This model is also a "),ns=a("a"),Gp=n("tf.keras.Model"),Jp=n(` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),eu=c(),L(mo.$$.fragment),tu=c(),Ge=a("div"),L(ss.$$.fragment),ou=c(),At=a("p"),nu=n("The "),ea=a("a"),su=n("TFLayoutLMModel"),au=n(" forward method, overrides the "),pr=a("code"),ru=n("__call__"),iu=n(" special method."),lu=c(),L(fo.$$.fragment),du=c(),L(go.$$.fragment),_i=c(),Ot=a("h2"),_o=a("a"),ur=a("span"),L(as.$$.fragment),cu=c(),hr=a("span"),pu=n("TFLayoutLMForMaskedLM"),yi=c(),ze=a("div"),L(rs.$$.fragment),uu=c(),is=a("p"),hu=n("LayoutLM Model with a "),mr=a("code"),mu=n("language modeling"),fu=n(" head on top."),gu=c(),ls=a("p"),_u=n("This model inherits from "),ta=a("a"),yu=n("TFPreTrainedModel"),bu=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),ku=c(),ds=a("p"),Lu=n("This model is also a "),cs=a("a"),wu=n("tf.keras.Model"),Tu=n(` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),vu=c(),L(yo.$$.fragment),Mu=c(),Je=a("div"),L(ps.$$.fragment),$u=c(),It=a("p"),xu=n("The "),oa=a("a"),ju=n("TFLayoutLMForMaskedLM"),Fu=n(" forward method, overrides the "),fr=a("code"),Eu=n("__call__"),zu=n(" special method."),qu=c(),L(bo.$$.fragment),Cu=c(),L(ko.$$.fragment),bi=c(),Dt=a("h2"),Lo=a("a"),gr=a("span"),L(us.$$.fragment),Pu=c(),_r=a("span"),Nu=n("TFLayoutLMForSequenceClassification"),ki=c(),qe=a("div"),L(hs.$$.fragment),Au=c(),yr=a("p"),Ou=n(`LayoutLM Model transformer with a sequence classification/regression head on top (a linear layer on top of the
pooled output) e.g. for GLUE tasks.`),Iu=c(),ms=a("p"),Du=n("This model inherits from "),na=a("a"),Su=n("TFPreTrainedModel"),Uu=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Wu=c(),fs=a("p"),Hu=n("This model is also a "),gs=a("a"),Qu=n("tf.keras.Model"),Bu=n(` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),Ru=c(),L(wo.$$.fragment),Ku=c(),et=a("div"),L(_s.$$.fragment),Vu=c(),St=a("p"),Xu=n("The "),sa=a("a"),Yu=n("TFLayoutLMForSequenceClassification"),Zu=n(" forward method, overrides the "),br=a("code"),Gu=n("__call__"),Ju=n(" special method."),eh=c(),L(To.$$.fragment),th=c(),L(vo.$$.fragment),Li=c(),Ut=a("h2"),Mo=a("a"),kr=a("span"),L(ys.$$.fragment),oh=c(),Lr=a("span"),nh=n("TFLayoutLMForTokenClassification"),wi=c(),Ce=a("div"),L(bs.$$.fragment),sh=c(),wr=a("p"),ah=n(`LayoutLM Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
Named-Entity-Recognition (NER) tasks.`),rh=c(),ks=a("p"),ih=n("This model inherits from "),aa=a("a"),lh=n("TFPreTrainedModel"),dh=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),ch=c(),Ls=a("p"),ph=n("This model is also a "),ws=a("a"),uh=n("tf.keras.Model"),hh=n(` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),mh=c(),L($o.$$.fragment),fh=c(),tt=a("div"),L(Ts.$$.fragment),gh=c(),Wt=a("p"),_h=n("The "),ra=a("a"),yh=n("TFLayoutLMForTokenClassification"),bh=n(" forward method, overrides the "),Tr=a("code"),kh=n("__call__"),Lh=n(" special method."),wh=c(),L(xo.$$.fragment),Th=c(),L(jo.$$.fragment),Ti=c(),Ht=a("h2"),Fo=a("a"),vr=a("span"),L(vs.$$.fragment),vh=c(),Mr=a("span"),Mh=n("TFLayoutLMForQuestionAnswering"),vi=c(),Pe=a("div"),L(Ms.$$.fragment),$h=c(),ct=a("p"),xh=n(`LayoutLM Model with a span classification head on top for extractive question-answering tasks such as
`),$s=a("a"),jh=n("DocVQA"),Fh=n(" (a linear layer on top of the final hidden-states output to compute "),$r=a("code"),Eh=n("span start logits"),zh=n(" and "),xr=a("code"),qh=n("span end logits"),Ch=n(")."),Ph=c(),xs=a("p"),Nh=n("This model inherits from "),ia=a("a"),Ah=n("TFPreTrainedModel"),Oh=n(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Ih=c(),js=a("p"),Dh=n("This model is also a "),Fs=a("a"),Sh=n("tf.keras.Model"),Uh=n(` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),Wh=c(),L(Eo.$$.fragment),Hh=c(),ot=a("div"),L(Es.$$.fragment),Qh=c(),Qt=a("p"),Bh=n("The "),la=a("a"),Rh=n("TFLayoutLMForQuestionAnswering"),Kh=n(" forward method, overrides the "),jr=a("code"),Vh=n("__call__"),Xh=n(" special method."),Yh=c(),L(zo.$$.fragment),Zh=c(),L(qo.$$.fragment),this.h()},l(o){const y=yg('[data-svelte="svelte-1phssyn"]',document.head);d=r(y,"META",{name:!0,content:!0}),y.forEach(t),k=p(o),_=r(o,"H1",{class:!0});var zs=i(_);h=r(zs,"A",{id:!0,class:!0,href:!0});var Fr=i(h);b=r(Fr,"SPAN",{});var Er=i(b);w(l.$$.fragment,Er),Er.forEach(t),Fr.forEach(t),m=p(zs),j=r(zs,"SPAN",{});var zr=i(j);re=s(zr,"LayoutLM"),zr.forEach(t),zs.forEach(t),X=p(o),E=r(o,"A",{id:!0}),i(E).forEach(t),ae=p(o),I=r(o,"H2",{class:!0});var qs=i(I);Y=r(qs,"A",{id:!0,class:!0,href:!0});var qr=i(Y);ne=r(qr,"SPAN",{});var Cr=i(ne);w(S.$$.fragment,Cr),Cr.forEach(t),qr.forEach(t),ie=p(qs),se=r(qs,"SPAN",{});var Pr=i(se);Z=s(Pr,"Overview"),Pr.forEach(t),qs.forEach(t),D=p(o),G=r(o,"P",{});var Cs=i(G);J=s(Cs,"The LayoutLM model was proposed in the paper "),F=r(Cs,"A",{href:!0,rel:!0});var Nr=i(F);q=s(Nr,`LayoutLM: Pre-training of Text and Layout for Document Image
Understanding`),Nr.forEach(t),le=s(Cs,` by Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and
Ming Zhou. It\u2019s a simple but effective pretraining method of text and layout for document image understanding and
information extraction tasks, such as form understanding and receipt understanding. It obtains state-of-the-art results
on several downstream tasks:`),Cs.forEach(t),H=p(o),V=r(o,"UL",{});var Bt=i(V);ee=r(Bt,"LI",{});var Ps=i(ee);Q=s(Ps,"form understanding: the "),te=r(Ps,"A",{href:!0,rel:!0});var Ar=i(te);de=s(Ar,"FUNSD"),Ar.forEach(t),C=s(Ps,` dataset (a collection of 199 annotated
forms comprising more than 30,000 words).`),Ps.forEach(t),ce=p(Bt),U=r(Bt,"LI",{});var Ns=i(U);pe=s(Ns,"receipt understanding: the "),oe=r(Ns,"A",{href:!0,rel:!0});var Or=i(oe);B=s(Or,"SROIE"),Or.forEach(t),ue=s(Ns,` dataset (a collection of 626 receipts for
training and 347 receipts for testing).`),Ns.forEach(t),he=p(Bt),N=r(Bt,"LI",{});var As=i(N);me=s(As,"document image classification: the "),W=r(As,"A",{href:!0,rel:!0});var Ir=i(W);fe=s(Ir,"RVL-CDIP"),Ir.forEach(t),f=s(As,` dataset (a collection of
400,000 images belonging to one of 16 classes).`),As.forEach(t),Bt.forEach(t),x=p(o),K=r(o,"P",{});var Dr=i(K);Me=s(Dr,"The abstract from the paper is the following:"),Dr.forEach(t),we=p(o),A=r(o,"P",{});var Sr=i(A);ke=r(Sr,"EM",{});var Ur=i(ke);$e=s(Ur,`Pre-training techniques have been verified successfully in a variety of NLP tasks in recent years. Despite the
widespread use of pretraining models for NLP applications, they almost exclusively focus on text-level manipulation,
while neglecting layout and style information that is vital for document image understanding. In this paper, we propose
the LayoutLM to jointly model interactions between text and layout information across scanned document images, which is
beneficial for a great number of real-world document image understanding tasks such as information extraction from
scanned documents. Furthermore, we also leverage image features to incorporate words\u2019 visual information into LayoutLM.
To the best of our knowledge, this is the first time that text and layout are jointly learned in a single framework for
document-level pretraining. It achieves new state-of-the-art results in several downstream tasks, including form
understanding (from 70.72 to 79.27), receipt understanding (from 94.02 to 95.24) and document image classification
(from 93.07 to 94.42).`),Ur.forEach(t),Sr.forEach(t),Te=p(o),O=r(o,"P",{});var Wr=i(O);R=s(Wr,"Tips:"),Wr.forEach(t),ve=p(o),ye=r(o,"UL",{});var Hr=i(ye);P=r(Hr,"LI",{});var Ie=i(P);xe=s(Ie,"In addition to "),Le=r(Ie,"EM",{});var Qr=i(Le);ge=s(Qr,"input_ids"),Qr.forEach(t),je=s(Ie,", "),Os=r(Ie,"A",{href:!0});var Br=i(Os);ll=s(Br,"forward()"),Br.forEach(t),dl=s(Ie," also expects the input "),va=r(Ie,"CODE",{});var Gh=i(va);cl=s(Gh,"bbox"),Gh.forEach(t),pl=s(Ie,`, which are
the bounding boxes (i.e. 2D-positions) of the input tokens. These can be obtained using an external OCR engine such
as Google\u2019s `),Yo=r(Ie,"A",{href:!0,rel:!0});var Jh=i(Yo);ul=s(Jh,"Tesseract"),Jh.forEach(t),hl=s(Ie," (there\u2019s a "),Zo=r(Ie,"A",{href:!0,rel:!0});var em=i(Zo);ml=s(em,"Python wrapper"),em.forEach(t),fl=s(Ie,` available). Each bounding box should be in (x0, y0, x1, y1) format, where
(x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1, y1) represents the
position of the lower right corner. Note that one first needs to normalize the bounding boxes to be on a 0-1000
scale. To normalize, you can use the following function:`),Ie.forEach(t),Hr.forEach(t),Vr=p(o),w(Go.$$.fragment,o),Xr=p(o),ht=r(o,"P",{});var da=i(ht);gl=s(da,"Here, "),Ma=r(da,"CODE",{});var tm=i(Ma);_l=s(tm,"width"),tm.forEach(t),yl=s(da," and "),$a=r(da,"CODE",{});var om=i($a);bl=s(om,"height"),om.forEach(t),kl=s(da,` correspond to the width and height of the original document in which the token
occurs. Those can be obtained using the Python Image Library (PIL) library for example, as follows:`),da.forEach(t),Yr=p(o),w(Jo.$$.fragment,o),Zr=p(o),Is=r(o,"UL",{});var nm=i(Is);lt=r(nm,"LI",{});var Co=i(lt);Ll=s(Co,"For a demo which shows how to fine-tune "),Ds=r(Co,"A",{href:!0});var sm=i(Ds);wl=s(sm,"LayoutLMForTokenClassification"),sm.forEach(t),Tl=s(Co," on the "),en=r(Co,"A",{href:!0,rel:!0});var am=i(en);vl=s(am,"FUNSD dataset"),am.forEach(t),Ml=s(Co," (a collection of annotated forms), see "),tn=r(Co,"A",{href:!0,rel:!0});var rm=i(tn);$l=s(rm,"this notebook"),rm.forEach(t),xl=s(Co,`.
It includes an inference part, which shows how to use Google\u2019s Tesseract on a new document.`),Co.forEach(t),nm.forEach(t),Gr=p(o),mt=r(o,"P",{});var ca=i(mt);jl=s(ca,"This model was contributed by "),on=r(ca,"A",{href:!0,rel:!0});var im=i(on);Fl=s(im,"liminghao1630"),im.forEach(t),El=s(ca,`. The original code can be found
`),nn=r(ca,"A",{href:!0,rel:!0});var lm=i(nn);zl=s(lm,"here"),lm.forEach(t),ql=s(ca,"."),ca.forEach(t),Jr=p(o),bt=r(o,"H2",{class:!0});var $i=i(bt);Rt=r($i,"A",{id:!0,class:!0,href:!0});var dm=i(Rt);xa=r(dm,"SPAN",{});var cm=i(xa);w(sn.$$.fragment,cm),cm.forEach(t),dm.forEach(t),Cl=p($i),ja=r($i,"SPAN",{});var pm=i(ja);Pl=s(pm,"LayoutLMConfig"),pm.forEach(t),$i.forEach(t),ei=p(o),We=r(o,"DIV",{class:!0});var Po=i(We);w(an.$$.fragment,Po),Nl=p(Po),kt=r(Po,"P",{});var pa=i(kt);Al=s(pa,"This is the configuration class to store the configuration of a "),Ss=r(pa,"A",{href:!0});var um=i(Ss);Ol=s(um,"LayoutLMModel"),um.forEach(t),Il=s(pa,`. It is used to instantiate a
LayoutLM model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the LayoutLM
`),rn=r(pa,"A",{href:!0,rel:!0});var hm=i(rn);Dl=s(hm,"microsoft/layoutlm-base-uncased"),hm.forEach(t),Sl=s(pa," architecture."),pa.forEach(t),Ul=p(Po),Lt=r(Po,"P",{});var ua=i(Lt);Wl=s(ua,"Configuration objects inherit from "),Us=r(ua,"A",{href:!0});var mm=i(Us);Hl=s(mm,"BertConfig"),mm.forEach(t),Ql=s(ua,` and can be used to control the model outputs. Read the
documentation from `),Ws=r(ua,"A",{href:!0});var fm=i(Ws);Bl=s(fm,"BertConfig"),fm.forEach(t),Rl=s(ua," for more information."),ua.forEach(t),Kl=p(Po),w(Kt.$$.fragment,Po),Po.forEach(t),ti=p(o),wt=r(o,"H2",{class:!0});var xi=i(wt);Vt=r(xi,"A",{id:!0,class:!0,href:!0});var gm=i(Vt);Fa=r(gm,"SPAN",{});var _m=i(Fa);w(ln.$$.fragment,_m),_m.forEach(t),gm.forEach(t),Vl=p(xi),Ea=r(xi,"SPAN",{});var ym=i(Ea);Xl=s(ym,"LayoutLMTokenizer"),ym.forEach(t),xi.forEach(t),oi=p(o),He=r(o,"DIV",{class:!0});var No=i(He);w(dn.$$.fragment,No),Yl=p(No),za=r(No,"P",{});var bm=i(za);Zl=s(bm,"Constructs a LayoutLM tokenizer."),bm.forEach(t),Gl=p(No),Xt=r(No,"P",{});var Rr=i(Xt);Hs=r(Rr,"A",{href:!0});var km=i(Hs);Jl=s(km,"LayoutLMTokenizer"),km.forEach(t),ed=s(Rr," is identical to "),Qs=r(Rr,"A",{href:!0});var Lm=i(Qs);td=s(Lm,"BertTokenizer"),Lm.forEach(t),od=s(Rr,` and runs end-to-end tokenization: punctuation splitting +
wordpiece.`),Rr.forEach(t),nd=p(No),cn=r(No,"P",{});var ji=i(cn);sd=s(ji,"Refer to superclass "),Bs=r(ji,"A",{href:!0});var wm=i(Bs);ad=s(wm,"BertTokenizer"),wm.forEach(t),rd=s(ji," for usage examples and documentation concerning parameters."),ji.forEach(t),No.forEach(t),ni=p(o),Tt=r(o,"H2",{class:!0});var Fi=i(Tt);Yt=r(Fi,"A",{id:!0,class:!0,href:!0});var Tm=i(Yt);qa=r(Tm,"SPAN",{});var vm=i(qa);w(pn.$$.fragment,vm),vm.forEach(t),Tm.forEach(t),id=p(Fi),Ca=r(Fi,"SPAN",{});var Mm=i(Ca);ld=s(Mm,"LayoutLMTokenizerFast"),Mm.forEach(t),Fi.forEach(t),si=p(o),Qe=r(o,"DIV",{class:!0});var Ao=i(Qe);w(un.$$.fragment,Ao),dd=p(Ao),Pa=r(Ao,"P",{});var $m=i(Pa);cd=s($m,"Constructs a \u201CFast\u201D LayoutLMTokenizer."),$m.forEach(t),pd=p(Ao),Zt=r(Ao,"P",{});var Kr=i(Zt);Rs=r(Kr,"A",{href:!0});var xm=i(Rs);ud=s(xm,"LayoutLMTokenizerFast"),xm.forEach(t),hd=s(Kr," is identical to "),Ks=r(Kr,"A",{href:!0});var jm=i(Ks);md=s(jm,"BertTokenizerFast"),jm.forEach(t),fd=s(Kr,` and runs end-to-end tokenization: punctuation
splitting + wordpiece.`),Kr.forEach(t),gd=p(Ao),hn=r(Ao,"P",{});var Ei=i(hn);_d=s(Ei,"Refer to superclass "),Vs=r(Ei,"A",{href:!0});var Fm=i(Vs);yd=s(Fm,"BertTokenizerFast"),Fm.forEach(t),bd=s(Ei," for usage examples and documentation concerning parameters."),Ei.forEach(t),Ao.forEach(t),ai=p(o),vt=r(o,"H2",{class:!0});var zi=i(vt);Gt=r(zi,"A",{id:!0,class:!0,href:!0});var Em=i(Gt);Na=r(Em,"SPAN",{});var zm=i(Na);w(mn.$$.fragment,zm),zm.forEach(t),Em.forEach(t),kd=p(zi),Aa=r(zi,"SPAN",{});var qm=i(Aa);Ld=s(qm,"LayoutLMModel"),qm.forEach(t),zi.forEach(t),ri=p(o),Be=r(o,"DIV",{class:!0});var Oo=i(Be);w(fn.$$.fragment,Oo),wd=p(Oo),gn=r(Oo,"P",{});var qi=i(gn);Td=s(qi,`The bare LayoutLM Model transformer outputting raw hidden-states without any specific head on top.
The LayoutLM model was proposed in `),_n=r(qi,"A",{href:!0,rel:!0});var Cm=i(_n);vd=s(Cm,`LayoutLM: Pre-training of Text and Layout for Document Image
Understanding`),Cm.forEach(t),Md=s(qi,` by Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei and
Ming Zhou.`),qi.forEach(t),$d=p(Oo),yn=r(Oo,"P",{});var Ci=i(yn);xd=s(Ci,"This model is a PyTorch "),bn=r(Ci,"A",{href:!0,rel:!0});var Pm=i(bn);jd=s(Pm,"torch.nn.Module"),Pm.forEach(t),Fd=s(Ci,` sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),Ci.forEach(t),Ed=p(Oo),Ve=r(Oo,"DIV",{class:!0});var Io=i(Ve);w(kn.$$.fragment,Io),zd=p(Io),Mt=r(Io,"P",{});var ha=i(Mt);qd=s(ha,"The "),Xs=r(ha,"A",{href:!0});var Nm=i(Xs);Cd=s(Nm,"LayoutLMModel"),Nm.forEach(t),Pd=s(ha," forward method, overrides the "),Oa=r(ha,"CODE",{});var Am=i(Oa);Nd=s(Am,"__call__"),Am.forEach(t),Ad=s(ha," special method."),ha.forEach(t),Od=p(Io),w(Jt.$$.fragment,Io),Id=p(Io),w(eo.$$.fragment,Io),Io.forEach(t),Oo.forEach(t),ii=p(o),$t=r(o,"H2",{class:!0});var Pi=i($t);to=r(Pi,"A",{id:!0,class:!0,href:!0});var Om=i(to);Ia=r(Om,"SPAN",{});var Im=i(Ia);w(Ln.$$.fragment,Im),Im.forEach(t),Om.forEach(t),Dd=p(Pi),Da=r(Pi,"SPAN",{});var Dm=i(Da);Sd=s(Dm,"LayoutLMForMaskedLM"),Dm.forEach(t),Pi.forEach(t),li=p(o),Re=r(o,"DIV",{class:!0});var Do=i(Re);w(wn.$$.fragment,Do),Ud=p(Do),xt=r(Do,"P",{});var ma=i(xt);Wd=s(ma,"LayoutLM Model with a "),Sa=r(ma,"CODE",{});var Sm=i(Sa);Hd=s(Sm,"language modeling"),Sm.forEach(t),Qd=s(ma,` head on top.
The LayoutLM model was proposed in `),Tn=r(ma,"A",{href:!0,rel:!0});var Um=i(Tn);Bd=s(Um,`LayoutLM: Pre-training of Text and Layout for Document Image
Understanding`),Um.forEach(t),Rd=s(ma,` by Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei and
Ming Zhou.`),ma.forEach(t),Kd=p(Do),vn=r(Do,"P",{});var Ni=i(vn);Vd=s(Ni,"This model is a PyTorch "),Mn=r(Ni,"A",{href:!0,rel:!0});var Wm=i(Mn);Xd=s(Wm,"torch.nn.Module"),Wm.forEach(t),Yd=s(Ni,` sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),Ni.forEach(t),Zd=p(Do),Xe=r(Do,"DIV",{class:!0});var So=i(Xe);w($n.$$.fragment,So),Gd=p(So),jt=r(So,"P",{});var fa=i(jt);Jd=s(fa,"The "),Ys=r(fa,"A",{href:!0});var Hm=i(Ys);ec=s(Hm,"LayoutLMForMaskedLM"),Hm.forEach(t),tc=s(fa," forward method, overrides the "),Ua=r(fa,"CODE",{});var Qm=i(Ua);oc=s(Qm,"__call__"),Qm.forEach(t),nc=s(fa," special method."),fa.forEach(t),sc=p(So),w(oo.$$.fragment,So),ac=p(So),w(no.$$.fragment,So),So.forEach(t),Do.forEach(t),di=p(o),Ft=r(o,"H2",{class:!0});var Ai=i(Ft);so=r(Ai,"A",{id:!0,class:!0,href:!0});var Bm=i(so);Wa=r(Bm,"SPAN",{});var Rm=i(Wa);w(xn.$$.fragment,Rm),Rm.forEach(t),Bm.forEach(t),rc=p(Ai),Ha=r(Ai,"SPAN",{});var Km=i(Ha);ic=s(Km,"LayoutLMForSequenceClassification"),Km.forEach(t),Ai.forEach(t),ci=p(o),Ne=r(o,"DIV",{class:!0});var ft=i(Ne);w(jn.$$.fragment,ft),lc=p(ft),Fn=r(ft,"P",{});var Oi=i(Fn);dc=s(Oi,`LayoutLM Model with a sequence classification head on top (a linear layer on top of the pooled output) e.g. for
document image classification tasks such as the `),En=r(Oi,"A",{href:!0,rel:!0});var Vm=i(En);cc=s(Vm,"RVL-CDIP"),Vm.forEach(t),pc=s(Oi," dataset."),Oi.forEach(t),uc=p(ft),zn=r(ft,"P",{});var Ii=i(zn);hc=s(Ii,"The LayoutLM model was proposed in "),qn=r(Ii,"A",{href:!0,rel:!0});var Xm=i(qn);mc=s(Xm,`LayoutLM: Pre-training of Text and Layout for Document Image
Understanding`),Xm.forEach(t),fc=s(Ii,` by Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei and
Ming Zhou.`),Ii.forEach(t),gc=p(ft),Cn=r(ft,"P",{});var Di=i(Cn);_c=s(Di,"This model is a PyTorch "),Pn=r(Di,"A",{href:!0,rel:!0});var Ym=i(Pn);yc=s(Ym,"torch.nn.Module"),Ym.forEach(t),bc=s(Di,` sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),Di.forEach(t),kc=p(ft),Ye=r(ft,"DIV",{class:!0});var Uo=i(Ye);w(Nn.$$.fragment,Uo),Lc=p(Uo),Et=r(Uo,"P",{});var ga=i(Et);wc=s(ga,"The "),Zs=r(ga,"A",{href:!0});var Zm=i(Zs);Tc=s(Zm,"LayoutLMForSequenceClassification"),Zm.forEach(t),vc=s(ga," forward method, overrides the "),Qa=r(ga,"CODE",{});var Gm=i(Qa);Mc=s(Gm,"__call__"),Gm.forEach(t),$c=s(ga," special method."),ga.forEach(t),xc=p(Uo),w(ao.$$.fragment,Uo),jc=p(Uo),w(ro.$$.fragment,Uo),Uo.forEach(t),ft.forEach(t),pi=p(o),zt=r(o,"H2",{class:!0});var Si=i(zt);io=r(Si,"A",{id:!0,class:!0,href:!0});var Jm=i(io);Ba=r(Jm,"SPAN",{});var ef=i(Ba);w(An.$$.fragment,ef),ef.forEach(t),Jm.forEach(t),Fc=p(Si),Ra=r(Si,"SPAN",{});var tf=i(Ra);Ec=s(tf,"LayoutLMForTokenClassification"),tf.forEach(t),Si.forEach(t),ui=p(o),Ae=r(o,"DIV",{class:!0});var gt=i(Ae);w(On.$$.fragment,gt),zc=p(gt),qt=r(gt,"P",{});var _a=i(qt);qc=s(_a,`LayoutLM Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
sequence labeling (information extraction) tasks such as the `),In=r(_a,"A",{href:!0,rel:!0});var of=i(In);Cc=s(of,"FUNSD"),of.forEach(t),Pc=s(_a,`
dataset and the `),Dn=r(_a,"A",{href:!0,rel:!0});var nf=i(Dn);Nc=s(nf,"SROIE"),nf.forEach(t),Ac=s(_a," dataset."),_a.forEach(t),Oc=p(gt),Sn=r(gt,"P",{});var Ui=i(Sn);Ic=s(Ui,"The LayoutLM model was proposed in "),Un=r(Ui,"A",{href:!0,rel:!0});var sf=i(Un);Dc=s(sf,`LayoutLM: Pre-training of Text and Layout for Document Image
Understanding`),sf.forEach(t),Sc=s(Ui,` by Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei and
Ming Zhou.`),Ui.forEach(t),Uc=p(gt),Wn=r(gt,"P",{});var Wi=i(Wn);Wc=s(Wi,"This model is a PyTorch "),Hn=r(Wi,"A",{href:!0,rel:!0});var af=i(Hn);Hc=s(af,"torch.nn.Module"),af.forEach(t),Qc=s(Wi,` sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),Wi.forEach(t),Bc=p(gt),Ze=r(gt,"DIV",{class:!0});var Wo=i(Ze);w(Qn.$$.fragment,Wo),Rc=p(Wo),Ct=r(Wo,"P",{});var ya=i(Ct);Kc=s(ya,"The "),Gs=r(ya,"A",{href:!0});var rf=i(Gs);Vc=s(rf,"LayoutLMForTokenClassification"),rf.forEach(t),Xc=s(ya," forward method, overrides the "),Ka=r(ya,"CODE",{});var lf=i(Ka);Yc=s(lf,"__call__"),lf.forEach(t),Zc=s(ya," special method."),ya.forEach(t),Gc=p(Wo),w(lo.$$.fragment,Wo),Jc=p(Wo),w(co.$$.fragment,Wo),Wo.forEach(t),gt.forEach(t),hi=p(o),Pt=r(o,"H2",{class:!0});var Hi=i(Pt);po=r(Hi,"A",{id:!0,class:!0,href:!0});var df=i(po);Va=r(df,"SPAN",{});var cf=i(Va);w(Bn.$$.fragment,cf),cf.forEach(t),df.forEach(t),ep=p(Hi),Xa=r(Hi,"SPAN",{});var pf=i(Xa);tp=s(pf,"LayoutLMForQuestionAnswering"),pf.forEach(t),Hi.forEach(t),mi=p(o),Oe=r(o,"DIV",{class:!0});var _t=i(Oe);w(Rn.$$.fragment,_t),op=p(_t),dt=r(_t,"P",{});var Ho=i(dt);np=s(Ho,`LayoutLM Model with a span classification head on top for extractive question-answering tasks such as
`),Kn=r(Ho,"A",{href:!0,rel:!0});var uf=i(Kn);sp=s(uf,"DocVQA"),uf.forEach(t),ap=s(Ho," (a linear layer on top of the final hidden-states output to compute "),Ya=r(Ho,"CODE",{});var hf=i(Ya);rp=s(hf,"span start logits"),hf.forEach(t),ip=s(Ho," and "),Za=r(Ho,"CODE",{});var mf=i(Za);lp=s(mf,"span end logits"),mf.forEach(t),dp=s(Ho,")."),Ho.forEach(t),cp=p(_t),Vn=r(_t,"P",{});var Qi=i(Vn);pp=s(Qi,"The LayoutLM model was proposed in "),Xn=r(Qi,"A",{href:!0,rel:!0});var ff=i(Xn);up=s(ff,`LayoutLM: Pre-training of Text and Layout for Document Image
Understanding`),ff.forEach(t),hp=s(Qi,` by Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei and
Ming Zhou.`),Qi.forEach(t),mp=p(_t),Yn=r(_t,"P",{});var Bi=i(Yn);fp=s(Bi,"This model is a PyTorch "),Zn=r(Bi,"A",{href:!0,rel:!0});var gf=i(Zn);gp=s(gf,"torch.nn.Module"),gf.forEach(t),_p=s(Bi,` sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),Bi.forEach(t),yp=p(_t),Se=r(_t,"DIV",{class:!0});var yt=i(Se);w(Gn.$$.fragment,yt),bp=p(yt),be=r(yt,"P",{});var Fe=i(be);kp=s(Fe,"start_positions ("),Ga=r(Fe,"CODE",{});var _f=i(Ga);Lp=s(_f,"torch.LongTensor"),_f.forEach(t),wp=s(Fe," of shape "),Ja=r(Fe,"CODE",{});var yf=i(Ja);Tp=s(yf,"(batch_size,)"),yf.forEach(t),vp=s(Fe,", "),er=r(Fe,"EM",{});var bf=i(er);Mp=s(bf,"optional"),bf.forEach(t),$p=s(Fe,`):
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (`),tr=r(Fe,"CODE",{});var kf=i(tr);xp=s(kf,"sequence_length"),kf.forEach(t),jp=s(Fe,`). Position outside of the sequence
are not taken into account for computing the loss.
end_positions (`),or=r(Fe,"CODE",{});var Lf=i(or);Fp=s(Lf,"torch.LongTensor"),Lf.forEach(t),Ep=s(Fe," of shape "),nr=r(Fe,"CODE",{});var wf=i(nr);zp=s(wf,"(batch_size,)"),wf.forEach(t),qp=s(Fe,", "),sr=r(Fe,"EM",{});var Tf=i(sr);Cp=s(Tf,"optional"),Tf.forEach(t),Pp=s(Fe,`):
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (`),ar=r(Fe,"CODE",{});var vf=i(ar);Np=s(vf,"sequence_length"),vf.forEach(t),Ap=s(Fe,`). Position outside of the sequence
are not taken into account for computing the loss.`),Fe.forEach(t),Op=p(yt),rr=r(yt,"P",{});var Mf=i(rr);Ip=s(Mf,"Example:"),Mf.forEach(t),Dp=p(yt),ir=r(yt,"P",{});var $f=i(ir);Sp=s($f,`In the example below, we prepare a question + context pair for the LayoutLM model. It will give us a prediction
of what it thinks the answer is (the span of the answer within the texts parsed from the image).`),$f.forEach(t),Up=p(yt),w(uo.$$.fragment,yt),yt.forEach(t),_t.forEach(t),fi=p(o),Nt=r(o,"H2",{class:!0});var Ri=i(Nt);ho=r(Ri,"A",{id:!0,class:!0,href:!0});var xf=i(ho);lr=r(xf,"SPAN",{});var jf=i(lr);w(Jn.$$.fragment,jf),jf.forEach(t),xf.forEach(t),Wp=p(Ri),dr=r(Ri,"SPAN",{});var Ff=i(dr);Hp=s(Ff,"TFLayoutLMModel"),Ff.forEach(t),Ri.forEach(t),gi=p(o),Ee=r(o,"DIV",{class:!0});var nt=i(Ee);w(es.$$.fragment,nt),Qp=p(nt),cr=r(nt,"P",{});var Ef=i(cr);Bp=s(Ef,"The bare LayoutLM Model transformer outputting raw hidden-states without any specific head on top."),Ef.forEach(t),Rp=p(nt),ts=r(nt,"P",{});var Ki=i(ts);Kp=s(Ki,"This model inherits from "),Js=r(Ki,"A",{href:!0});var zf=i(Js);Vp=s(zf,"TFPreTrainedModel"),zf.forEach(t),Xp=s(Ki,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Ki.forEach(t),Yp=p(nt),os=r(nt,"P",{});var Vi=i(os);Zp=s(Vi,"This model is also a "),ns=r(Vi,"A",{href:!0,rel:!0});var qf=i(ns);Gp=s(qf,"tf.keras.Model"),qf.forEach(t),Jp=s(Vi,` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),Vi.forEach(t),eu=p(nt),w(mo.$$.fragment,nt),tu=p(nt),Ge=r(nt,"DIV",{class:!0});var Qo=i(Ge);w(ss.$$.fragment,Qo),ou=p(Qo),At=r(Qo,"P",{});var ba=i(At);nu=s(ba,"The "),ea=r(ba,"A",{href:!0});var Cf=i(ea);su=s(Cf,"TFLayoutLMModel"),Cf.forEach(t),au=s(ba," forward method, overrides the "),pr=r(ba,"CODE",{});var Pf=i(pr);ru=s(Pf,"__call__"),Pf.forEach(t),iu=s(ba," special method."),ba.forEach(t),lu=p(Qo),w(fo.$$.fragment,Qo),du=p(Qo),w(go.$$.fragment,Qo),Qo.forEach(t),nt.forEach(t),_i=p(o),Ot=r(o,"H2",{class:!0});var Xi=i(Ot);_o=r(Xi,"A",{id:!0,class:!0,href:!0});var Nf=i(_o);ur=r(Nf,"SPAN",{});var Af=i(ur);w(as.$$.fragment,Af),Af.forEach(t),Nf.forEach(t),cu=p(Xi),hr=r(Xi,"SPAN",{});var Of=i(hr);pu=s(Of,"TFLayoutLMForMaskedLM"),Of.forEach(t),Xi.forEach(t),yi=p(o),ze=r(o,"DIV",{class:!0});var st=i(ze);w(rs.$$.fragment,st),uu=p(st),is=r(st,"P",{});var Yi=i(is);hu=s(Yi,"LayoutLM Model with a "),mr=r(Yi,"CODE",{});var If=i(mr);mu=s(If,"language modeling"),If.forEach(t),fu=s(Yi," head on top."),Yi.forEach(t),gu=p(st),ls=r(st,"P",{});var Zi=i(ls);_u=s(Zi,"This model inherits from "),ta=r(Zi,"A",{href:!0});var Df=i(ta);yu=s(Df,"TFPreTrainedModel"),Df.forEach(t),bu=s(Zi,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Zi.forEach(t),ku=p(st),ds=r(st,"P",{});var Gi=i(ds);Lu=s(Gi,"This model is also a "),cs=r(Gi,"A",{href:!0,rel:!0});var Sf=i(cs);wu=s(Sf,"tf.keras.Model"),Sf.forEach(t),Tu=s(Gi,` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),Gi.forEach(t),vu=p(st),w(yo.$$.fragment,st),Mu=p(st),Je=r(st,"DIV",{class:!0});var Bo=i(Je);w(ps.$$.fragment,Bo),$u=p(Bo),It=r(Bo,"P",{});var ka=i(It);xu=s(ka,"The "),oa=r(ka,"A",{href:!0});var Uf=i(oa);ju=s(Uf,"TFLayoutLMForMaskedLM"),Uf.forEach(t),Fu=s(ka," forward method, overrides the "),fr=r(ka,"CODE",{});var Wf=i(fr);Eu=s(Wf,"__call__"),Wf.forEach(t),zu=s(ka," special method."),ka.forEach(t),qu=p(Bo),w(bo.$$.fragment,Bo),Cu=p(Bo),w(ko.$$.fragment,Bo),Bo.forEach(t),st.forEach(t),bi=p(o),Dt=r(o,"H2",{class:!0});var Ji=i(Dt);Lo=r(Ji,"A",{id:!0,class:!0,href:!0});var Hf=i(Lo);gr=r(Hf,"SPAN",{});var Qf=i(gr);w(us.$$.fragment,Qf),Qf.forEach(t),Hf.forEach(t),Pu=p(Ji),_r=r(Ji,"SPAN",{});var Bf=i(_r);Nu=s(Bf,"TFLayoutLMForSequenceClassification"),Bf.forEach(t),Ji.forEach(t),ki=p(o),qe=r(o,"DIV",{class:!0});var at=i(qe);w(hs.$$.fragment,at),Au=p(at),yr=r(at,"P",{});var Rf=i(yr);Ou=s(Rf,`LayoutLM Model transformer with a sequence classification/regression head on top (a linear layer on top of the
pooled output) e.g. for GLUE tasks.`),Rf.forEach(t),Iu=p(at),ms=r(at,"P",{});var el=i(ms);Du=s(el,"This model inherits from "),na=r(el,"A",{href:!0});var Kf=i(na);Su=s(Kf,"TFPreTrainedModel"),Kf.forEach(t),Uu=s(el,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),el.forEach(t),Wu=p(at),fs=r(at,"P",{});var tl=i(fs);Hu=s(tl,"This model is also a "),gs=r(tl,"A",{href:!0,rel:!0});var Vf=i(gs);Qu=s(Vf,"tf.keras.Model"),Vf.forEach(t),Bu=s(tl,` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),tl.forEach(t),Ru=p(at),w(wo.$$.fragment,at),Ku=p(at),et=r(at,"DIV",{class:!0});var Ro=i(et);w(_s.$$.fragment,Ro),Vu=p(Ro),St=r(Ro,"P",{});var La=i(St);Xu=s(La,"The "),sa=r(La,"A",{href:!0});var Xf=i(sa);Yu=s(Xf,"TFLayoutLMForSequenceClassification"),Xf.forEach(t),Zu=s(La," forward method, overrides the "),br=r(La,"CODE",{});var Yf=i(br);Gu=s(Yf,"__call__"),Yf.forEach(t),Ju=s(La," special method."),La.forEach(t),eh=p(Ro),w(To.$$.fragment,Ro),th=p(Ro),w(vo.$$.fragment,Ro),Ro.forEach(t),at.forEach(t),Li=p(o),Ut=r(o,"H2",{class:!0});var ol=i(Ut);Mo=r(ol,"A",{id:!0,class:!0,href:!0});var Zf=i(Mo);kr=r(Zf,"SPAN",{});var Gf=i(kr);w(ys.$$.fragment,Gf),Gf.forEach(t),Zf.forEach(t),oh=p(ol),Lr=r(ol,"SPAN",{});var Jf=i(Lr);nh=s(Jf,"TFLayoutLMForTokenClassification"),Jf.forEach(t),ol.forEach(t),wi=p(o),Ce=r(o,"DIV",{class:!0});var rt=i(Ce);w(bs.$$.fragment,rt),sh=p(rt),wr=r(rt,"P",{});var eg=i(wr);ah=s(eg,`LayoutLM Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
Named-Entity-Recognition (NER) tasks.`),eg.forEach(t),rh=p(rt),ks=r(rt,"P",{});var nl=i(ks);ih=s(nl,"This model inherits from "),aa=r(nl,"A",{href:!0});var tg=i(aa);lh=s(tg,"TFPreTrainedModel"),tg.forEach(t),dh=s(nl,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),nl.forEach(t),ch=p(rt),Ls=r(rt,"P",{});var sl=i(Ls);ph=s(sl,"This model is also a "),ws=r(sl,"A",{href:!0,rel:!0});var og=i(ws);uh=s(og,"tf.keras.Model"),og.forEach(t),hh=s(sl,` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),sl.forEach(t),mh=p(rt),w($o.$$.fragment,rt),fh=p(rt),tt=r(rt,"DIV",{class:!0});var Ko=i(tt);w(Ts.$$.fragment,Ko),gh=p(Ko),Wt=r(Ko,"P",{});var wa=i(Wt);_h=s(wa,"The "),ra=r(wa,"A",{href:!0});var ng=i(ra);yh=s(ng,"TFLayoutLMForTokenClassification"),ng.forEach(t),bh=s(wa," forward method, overrides the "),Tr=r(wa,"CODE",{});var sg=i(Tr);kh=s(sg,"__call__"),sg.forEach(t),Lh=s(wa," special method."),wa.forEach(t),wh=p(Ko),w(xo.$$.fragment,Ko),Th=p(Ko),w(jo.$$.fragment,Ko),Ko.forEach(t),rt.forEach(t),Ti=p(o),Ht=r(o,"H2",{class:!0});var al=i(Ht);Fo=r(al,"A",{id:!0,class:!0,href:!0});var ag=i(Fo);vr=r(ag,"SPAN",{});var rg=i(vr);w(vs.$$.fragment,rg),rg.forEach(t),ag.forEach(t),vh=p(al),Mr=r(al,"SPAN",{});var ig=i(Mr);Mh=s(ig,"TFLayoutLMForQuestionAnswering"),ig.forEach(t),al.forEach(t),vi=p(o),Pe=r(o,"DIV",{class:!0});var it=i(Pe);w(Ms.$$.fragment,it),$h=p(it),ct=r(it,"P",{});var Vo=i(ct);xh=s(Vo,`LayoutLM Model with a span classification head on top for extractive question-answering tasks such as
`),$s=r(Vo,"A",{href:!0,rel:!0});var lg=i($s);jh=s(lg,"DocVQA"),lg.forEach(t),Fh=s(Vo," (a linear layer on top of the final hidden-states output to compute "),$r=r(Vo,"CODE",{});var dg=i($r);Eh=s(dg,"span start logits"),dg.forEach(t),zh=s(Vo," and "),xr=r(Vo,"CODE",{});var cg=i(xr);qh=s(cg,"span end logits"),cg.forEach(t),Ch=s(Vo,")."),Vo.forEach(t),Ph=p(it),xs=r(it,"P",{});var rl=i(xs);Nh=s(rl,"This model inherits from "),ia=r(rl,"A",{href:!0});var pg=i(ia);Ah=s(pg,"TFPreTrainedModel"),pg.forEach(t),Oh=s(rl,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),rl.forEach(t),Ih=p(it),js=r(it,"P",{});var il=i(js);Dh=s(il,"This model is also a "),Fs=r(il,"A",{href:!0,rel:!0});var ug=i(Fs);Sh=s(ug,"tf.keras.Model"),ug.forEach(t),Uh=s(il,` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),il.forEach(t),Wh=p(it),w(Eo.$$.fragment,it),Hh=p(it),ot=r(it,"DIV",{class:!0});var Xo=i(ot);w(Es.$$.fragment,Xo),Qh=p(Xo),Qt=r(Xo,"P",{});var Ta=i(Qt);Bh=s(Ta,"The "),la=r(Ta,"A",{href:!0});var hg=i(la);Rh=s(hg,"TFLayoutLMForQuestionAnswering"),hg.forEach(t),Kh=s(Ta," forward method, overrides the "),jr=r(Ta,"CODE",{});var mg=i(jr);Vh=s(mg,"__call__"),mg.forEach(t),Xh=s(Ta," special method."),Ta.forEach(t),Yh=p(Xo),w(zo.$$.fragment,Xo),Zh=p(Xo),w(qo.$$.fragment,Xo),Xo.forEach(t),it.forEach(t),this.h()},h(){u(d,"name","hf:doc:metadata"),u(d,"content",JSON.stringify(Rg)),u(h,"id","layoutlm"),u(h,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(h,"href","#layoutlm"),u(_,"class","relative group"),u(E,"id","Overview"),u(Y,"id","overview"),u(Y,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(Y,"href","#overview"),u(I,"class","relative group"),u(F,"href","https://arxiv.org/abs/1912.13318"),u(F,"rel","nofollow"),u(te,"href","https://guillaumejaume.github.io/FUNSD/"),u(te,"rel","nofollow"),u(oe,"href","https://rrc.cvc.uab.es/?ch=13"),u(oe,"rel","nofollow"),u(W,"href","https://www.cs.cmu.edu/~aharley/rvl-cdip/"),u(W,"rel","nofollow"),u(Os,"href","/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMModel.forward"),u(Yo,"href","https://github.com/tesseract-ocr/tesseract"),u(Yo,"rel","nofollow"),u(Zo,"href","https://pypi.org/project/pytesseract/"),u(Zo,"rel","nofollow"),u(Ds,"href","/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMForTokenClassification"),u(en,"href","https://guillaumejaume.github.io/FUNSD/"),u(en,"rel","nofollow"),u(tn,"href","https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LayoutLM/Fine_tuning_LayoutLMForTokenClassification_on_FUNSD.ipynb"),u(tn,"rel","nofollow"),u(on,"href","https://huggingface.co/liminghao1630"),u(on,"rel","nofollow"),u(nn,"href","https://github.com/microsoft/unilm/tree/master/layoutlm"),u(nn,"rel","nofollow"),u(Rt,"id","transformers.LayoutLMConfig"),u(Rt,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(Rt,"href","#transformers.LayoutLMConfig"),u(bt,"class","relative group"),u(Ss,"href","/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMModel"),u(rn,"href","https://huggingface.co/microsoft/layoutlm-base-uncased"),u(rn,"rel","nofollow"),u(Us,"href","/docs/transformers/pr_18407/en/model_doc/bert#transformers.BertConfig"),u(Ws,"href","/docs/transformers/pr_18407/en/model_doc/bert#transformers.BertConfig"),u(We,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Vt,"id","transformers.LayoutLMTokenizer"),u(Vt,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(Vt,"href","#transformers.LayoutLMTokenizer"),u(wt,"class","relative group"),u(Hs,"href","/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMTokenizer"),u(Qs,"href","/docs/transformers/pr_18407/en/model_doc/bert#transformers.BertTokenizer"),u(Bs,"href","/docs/transformers/pr_18407/en/model_doc/bert#transformers.BertTokenizer"),u(He,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Yt,"id","transformers.LayoutLMTokenizerFast"),u(Yt,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(Yt,"href","#transformers.LayoutLMTokenizerFast"),u(Tt,"class","relative group"),u(Rs,"href","/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMTokenizerFast"),u(Ks,"href","/docs/transformers/pr_18407/en/model_doc/bert#transformers.BertTokenizerFast"),u(Vs,"href","/docs/transformers/pr_18407/en/model_doc/bert#transformers.BertTokenizerFast"),u(Qe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Gt,"id","transformers.LayoutLMModel"),u(Gt,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(Gt,"href","#transformers.LayoutLMModel"),u(vt,"class","relative group"),u(_n,"href","https://arxiv.org/abs/1912.13318"),u(_n,"rel","nofollow"),u(bn,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),u(bn,"rel","nofollow"),u(Xs,"href","/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMModel"),u(Ve,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Be,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(to,"id","transformers.LayoutLMForMaskedLM"),u(to,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(to,"href","#transformers.LayoutLMForMaskedLM"),u($t,"class","relative group"),u(Tn,"href","https://arxiv.org/abs/1912.13318"),u(Tn,"rel","nofollow"),u(Mn,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),u(Mn,"rel","nofollow"),u(Ys,"href","/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMForMaskedLM"),u(Xe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Re,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(so,"id","transformers.LayoutLMForSequenceClassification"),u(so,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(so,"href","#transformers.LayoutLMForSequenceClassification"),u(Ft,"class","relative group"),u(En,"href","https://www.cs.cmu.edu/~aharley/rvl-cdip/"),u(En,"rel","nofollow"),u(qn,"href","https://arxiv.org/abs/1912.13318"),u(qn,"rel","nofollow"),u(Pn,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),u(Pn,"rel","nofollow"),u(Zs,"href","/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMForSequenceClassification"),u(Ye,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Ne,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(io,"id","transformers.LayoutLMForTokenClassification"),u(io,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(io,"href","#transformers.LayoutLMForTokenClassification"),u(zt,"class","relative group"),u(In,"href","https://guillaumejaume.github.io/FUNSD/"),u(In,"rel","nofollow"),u(Dn,"href","https://rrc.cvc.uab.es/?ch=13"),u(Dn,"rel","nofollow"),u(Un,"href","https://arxiv.org/abs/1912.13318"),u(Un,"rel","nofollow"),u(Hn,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),u(Hn,"rel","nofollow"),u(Gs,"href","/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.LayoutLMForTokenClassification"),u(Ze,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Ae,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(po,"id","transformers.LayoutLMForQuestionAnswering"),u(po,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(po,"href","#transformers.LayoutLMForQuestionAnswering"),u(Pt,"class","relative group"),u(Kn,"href","https://rrc.cvc.uab.es/?ch=17"),u(Kn,"rel","nofollow"),u(Xn,"href","https://arxiv.org/abs/1912.13318"),u(Xn,"rel","nofollow"),u(Zn,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),u(Zn,"rel","nofollow"),u(Se,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Oe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(ho,"id","transformers.TFLayoutLMModel"),u(ho,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(ho,"href","#transformers.TFLayoutLMModel"),u(Nt,"class","relative group"),u(Js,"href","/docs/transformers/pr_18407/en/main_classes/model#transformers.TFPreTrainedModel"),u(ns,"href","https://www.tensorflow.org/api_docs/python/tf/keras/Model"),u(ns,"rel","nofollow"),u(ea,"href","/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.TFLayoutLMModel"),u(Ge,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(_o,"id","transformers.TFLayoutLMForMaskedLM"),u(_o,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(_o,"href","#transformers.TFLayoutLMForMaskedLM"),u(Ot,"class","relative group"),u(ta,"href","/docs/transformers/pr_18407/en/main_classes/model#transformers.TFPreTrainedModel"),u(cs,"href","https://www.tensorflow.org/api_docs/python/tf/keras/Model"),u(cs,"rel","nofollow"),u(oa,"href","/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.TFLayoutLMForMaskedLM"),u(Je,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(ze,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Lo,"id","transformers.TFLayoutLMForSequenceClassification"),u(Lo,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(Lo,"href","#transformers.TFLayoutLMForSequenceClassification"),u(Dt,"class","relative group"),u(na,"href","/docs/transformers/pr_18407/en/main_classes/model#transformers.TFPreTrainedModel"),u(gs,"href","https://www.tensorflow.org/api_docs/python/tf/keras/Model"),u(gs,"rel","nofollow"),u(sa,"href","/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.TFLayoutLMForSequenceClassification"),u(et,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(qe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Mo,"id","transformers.TFLayoutLMForTokenClassification"),u(Mo,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(Mo,"href","#transformers.TFLayoutLMForTokenClassification"),u(Ut,"class","relative group"),u(aa,"href","/docs/transformers/pr_18407/en/main_classes/model#transformers.TFPreTrainedModel"),u(ws,"href","https://www.tensorflow.org/api_docs/python/tf/keras/Model"),u(ws,"rel","nofollow"),u(ra,"href","/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.TFLayoutLMForTokenClassification"),u(tt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Ce,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Fo,"id","transformers.TFLayoutLMForQuestionAnswering"),u(Fo,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(Fo,"href","#transformers.TFLayoutLMForQuestionAnswering"),u(Ht,"class","relative group"),u($s,"href","https://rrc.cvc.uab.es/?ch=17"),u($s,"rel","nofollow"),u(ia,"href","/docs/transformers/pr_18407/en/main_classes/model#transformers.TFPreTrainedModel"),u(Fs,"href","https://www.tensorflow.org/api_docs/python/tf/keras/Model"),u(Fs,"rel","nofollow"),u(la,"href","/docs/transformers/pr_18407/en/model_doc/layoutlm#transformers.TFLayoutLMForQuestionAnswering"),u(ot,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),u(Pe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(o,y){e(document.head,d),g(o,k,y),g(o,_,y),e(_,h),e(h,b),T(l,b,null),e(_,m),e(_,j),e(j,re),g(o,X,y),g(o,E,y),g(o,ae,y),g(o,I,y),e(I,Y),e(Y,ne),T(S,ne,null),e(I,ie),e(I,se),e(se,Z),g(o,D,y),g(o,G,y),e(G,J),e(G,F),e(F,q),e(G,le),g(o,H,y),g(o,V,y),e(V,ee),e(ee,Q),e(ee,te),e(te,de),e(ee,C),e(V,ce),e(V,U),e(U,pe),e(U,oe),e(oe,B),e(U,ue),e(V,he),e(V,N),e(N,me),e(N,W),e(W,fe),e(N,f),g(o,x,y),g(o,K,y),e(K,Me),g(o,we,y),g(o,A,y),e(A,ke),e(ke,$e),g(o,Te,y),g(o,O,y),e(O,R),g(o,ve,y),g(o,ye,y),e(ye,P),e(P,xe),e(P,Le),e(Le,ge),e(P,je),e(P,Os),e(Os,ll),e(P,dl),e(P,va),e(va,cl),e(P,pl),e(P,Yo),e(Yo,ul),e(P,hl),e(P,Zo),e(Zo,ml),e(P,fl),g(o,Vr,y),T(Go,o,y),g(o,Xr,y),g(o,ht,y),e(ht,gl),e(ht,Ma),e(Ma,_l),e(ht,yl),e(ht,$a),e($a,bl),e(ht,kl),g(o,Yr,y),T(Jo,o,y),g(o,Zr,y),g(o,Is,y),e(Is,lt),e(lt,Ll),e(lt,Ds),e(Ds,wl),e(lt,Tl),e(lt,en),e(en,vl),e(lt,Ml),e(lt,tn),e(tn,$l),e(lt,xl),g(o,Gr,y),g(o,mt,y),e(mt,jl),e(mt,on),e(on,Fl),e(mt,El),e(mt,nn),e(nn,zl),e(mt,ql),g(o,Jr,y),g(o,bt,y),e(bt,Rt),e(Rt,xa),T(sn,xa,null),e(bt,Cl),e(bt,ja),e(ja,Pl),g(o,ei,y),g(o,We,y),T(an,We,null),e(We,Nl),e(We,kt),e(kt,Al),e(kt,Ss),e(Ss,Ol),e(kt,Il),e(kt,rn),e(rn,Dl),e(kt,Sl),e(We,Ul),e(We,Lt),e(Lt,Wl),e(Lt,Us),e(Us,Hl),e(Lt,Ql),e(Lt,Ws),e(Ws,Bl),e(Lt,Rl),e(We,Kl),T(Kt,We,null),g(o,ti,y),g(o,wt,y),e(wt,Vt),e(Vt,Fa),T(ln,Fa,null),e(wt,Vl),e(wt,Ea),e(Ea,Xl),g(o,oi,y),g(o,He,y),T(dn,He,null),e(He,Yl),e(He,za),e(za,Zl),e(He,Gl),e(He,Xt),e(Xt,Hs),e(Hs,Jl),e(Xt,ed),e(Xt,Qs),e(Qs,td),e(Xt,od),e(He,nd),e(He,cn),e(cn,sd),e(cn,Bs),e(Bs,ad),e(cn,rd),g(o,ni,y),g(o,Tt,y),e(Tt,Yt),e(Yt,qa),T(pn,qa,null),e(Tt,id),e(Tt,Ca),e(Ca,ld),g(o,si,y),g(o,Qe,y),T(un,Qe,null),e(Qe,dd),e(Qe,Pa),e(Pa,cd),e(Qe,pd),e(Qe,Zt),e(Zt,Rs),e(Rs,ud),e(Zt,hd),e(Zt,Ks),e(Ks,md),e(Zt,fd),e(Qe,gd),e(Qe,hn),e(hn,_d),e(hn,Vs),e(Vs,yd),e(hn,bd),g(o,ai,y),g(o,vt,y),e(vt,Gt),e(Gt,Na),T(mn,Na,null),e(vt,kd),e(vt,Aa),e(Aa,Ld),g(o,ri,y),g(o,Be,y),T(fn,Be,null),e(Be,wd),e(Be,gn),e(gn,Td),e(gn,_n),e(_n,vd),e(gn,Md),e(Be,$d),e(Be,yn),e(yn,xd),e(yn,bn),e(bn,jd),e(yn,Fd),e(Be,Ed),e(Be,Ve),T(kn,Ve,null),e(Ve,zd),e(Ve,Mt),e(Mt,qd),e(Mt,Xs),e(Xs,Cd),e(Mt,Pd),e(Mt,Oa),e(Oa,Nd),e(Mt,Ad),e(Ve,Od),T(Jt,Ve,null),e(Ve,Id),T(eo,Ve,null),g(o,ii,y),g(o,$t,y),e($t,to),e(to,Ia),T(Ln,Ia,null),e($t,Dd),e($t,Da),e(Da,Sd),g(o,li,y),g(o,Re,y),T(wn,Re,null),e(Re,Ud),e(Re,xt),e(xt,Wd),e(xt,Sa),e(Sa,Hd),e(xt,Qd),e(xt,Tn),e(Tn,Bd),e(xt,Rd),e(Re,Kd),e(Re,vn),e(vn,Vd),e(vn,Mn),e(Mn,Xd),e(vn,Yd),e(Re,Zd),e(Re,Xe),T($n,Xe,null),e(Xe,Gd),e(Xe,jt),e(jt,Jd),e(jt,Ys),e(Ys,ec),e(jt,tc),e(jt,Ua),e(Ua,oc),e(jt,nc),e(Xe,sc),T(oo,Xe,null),e(Xe,ac),T(no,Xe,null),g(o,di,y),g(o,Ft,y),e(Ft,so),e(so,Wa),T(xn,Wa,null),e(Ft,rc),e(Ft,Ha),e(Ha,ic),g(o,ci,y),g(o,Ne,y),T(jn,Ne,null),e(Ne,lc),e(Ne,Fn),e(Fn,dc),e(Fn,En),e(En,cc),e(Fn,pc),e(Ne,uc),e(Ne,zn),e(zn,hc),e(zn,qn),e(qn,mc),e(zn,fc),e(Ne,gc),e(Ne,Cn),e(Cn,_c),e(Cn,Pn),e(Pn,yc),e(Cn,bc),e(Ne,kc),e(Ne,Ye),T(Nn,Ye,null),e(Ye,Lc),e(Ye,Et),e(Et,wc),e(Et,Zs),e(Zs,Tc),e(Et,vc),e(Et,Qa),e(Qa,Mc),e(Et,$c),e(Ye,xc),T(ao,Ye,null),e(Ye,jc),T(ro,Ye,null),g(o,pi,y),g(o,zt,y),e(zt,io),e(io,Ba),T(An,Ba,null),e(zt,Fc),e(zt,Ra),e(Ra,Ec),g(o,ui,y),g(o,Ae,y),T(On,Ae,null),e(Ae,zc),e(Ae,qt),e(qt,qc),e(qt,In),e(In,Cc),e(qt,Pc),e(qt,Dn),e(Dn,Nc),e(qt,Ac),e(Ae,Oc),e(Ae,Sn),e(Sn,Ic),e(Sn,Un),e(Un,Dc),e(Sn,Sc),e(Ae,Uc),e(Ae,Wn),e(Wn,Wc),e(Wn,Hn),e(Hn,Hc),e(Wn,Qc),e(Ae,Bc),e(Ae,Ze),T(Qn,Ze,null),e(Ze,Rc),e(Ze,Ct),e(Ct,Kc),e(Ct,Gs),e(Gs,Vc),e(Ct,Xc),e(Ct,Ka),e(Ka,Yc),e(Ct,Zc),e(Ze,Gc),T(lo,Ze,null),e(Ze,Jc),T(co,Ze,null),g(o,hi,y),g(o,Pt,y),e(Pt,po),e(po,Va),T(Bn,Va,null),e(Pt,ep),e(Pt,Xa),e(Xa,tp),g(o,mi,y),g(o,Oe,y),T(Rn,Oe,null),e(Oe,op),e(Oe,dt),e(dt,np),e(dt,Kn),e(Kn,sp),e(dt,ap),e(dt,Ya),e(Ya,rp),e(dt,ip),e(dt,Za),e(Za,lp),e(dt,dp),e(Oe,cp),e(Oe,Vn),e(Vn,pp),e(Vn,Xn),e(Xn,up),e(Vn,hp),e(Oe,mp),e(Oe,Yn),e(Yn,fp),e(Yn,Zn),e(Zn,gp),e(Yn,_p),e(Oe,yp),e(Oe,Se),T(Gn,Se,null),e(Se,bp),e(Se,be),e(be,kp),e(be,Ga),e(Ga,Lp),e(be,wp),e(be,Ja),e(Ja,Tp),e(be,vp),e(be,er),e(er,Mp),e(be,$p),e(be,tr),e(tr,xp),e(be,jp),e(be,or),e(or,Fp),e(be,Ep),e(be,nr),e(nr,zp),e(be,qp),e(be,sr),e(sr,Cp),e(be,Pp),e(be,ar),e(ar,Np),e(be,Ap),e(Se,Op),e(Se,rr),e(rr,Ip),e(Se,Dp),e(Se,ir),e(ir,Sp),e(Se,Up),T(uo,Se,null),g(o,fi,y),g(o,Nt,y),e(Nt,ho),e(ho,lr),T(Jn,lr,null),e(Nt,Wp),e(Nt,dr),e(dr,Hp),g(o,gi,y),g(o,Ee,y),T(es,Ee,null),e(Ee,Qp),e(Ee,cr),e(cr,Bp),e(Ee,Rp),e(Ee,ts),e(ts,Kp),e(ts,Js),e(Js,Vp),e(ts,Xp),e(Ee,Yp),e(Ee,os),e(os,Zp),e(os,ns),e(ns,Gp),e(os,Jp),e(Ee,eu),T(mo,Ee,null),e(Ee,tu),e(Ee,Ge),T(ss,Ge,null),e(Ge,ou),e(Ge,At),e(At,nu),e(At,ea),e(ea,su),e(At,au),e(At,pr),e(pr,ru),e(At,iu),e(Ge,lu),T(fo,Ge,null),e(Ge,du),T(go,Ge,null),g(o,_i,y),g(o,Ot,y),e(Ot,_o),e(_o,ur),T(as,ur,null),e(Ot,cu),e(Ot,hr),e(hr,pu),g(o,yi,y),g(o,ze,y),T(rs,ze,null),e(ze,uu),e(ze,is),e(is,hu),e(is,mr),e(mr,mu),e(is,fu),e(ze,gu),e(ze,ls),e(ls,_u),e(ls,ta),e(ta,yu),e(ls,bu),e(ze,ku),e(ze,ds),e(ds,Lu),e(ds,cs),e(cs,wu),e(ds,Tu),e(ze,vu),T(yo,ze,null),e(ze,Mu),e(ze,Je),T(ps,Je,null),e(Je,$u),e(Je,It),e(It,xu),e(It,oa),e(oa,ju),e(It,Fu),e(It,fr),e(fr,Eu),e(It,zu),e(Je,qu),T(bo,Je,null),e(Je,Cu),T(ko,Je,null),g(o,bi,y),g(o,Dt,y),e(Dt,Lo),e(Lo,gr),T(us,gr,null),e(Dt,Pu),e(Dt,_r),e(_r,Nu),g(o,ki,y),g(o,qe,y),T(hs,qe,null),e(qe,Au),e(qe,yr),e(yr,Ou),e(qe,Iu),e(qe,ms),e(ms,Du),e(ms,na),e(na,Su),e(ms,Uu),e(qe,Wu),e(qe,fs),e(fs,Hu),e(fs,gs),e(gs,Qu),e(fs,Bu),e(qe,Ru),T(wo,qe,null),e(qe,Ku),e(qe,et),T(_s,et,null),e(et,Vu),e(et,St),e(St,Xu),e(St,sa),e(sa,Yu),e(St,Zu),e(St,br),e(br,Gu),e(St,Ju),e(et,eh),T(To,et,null),e(et,th),T(vo,et,null),g(o,Li,y),g(o,Ut,y),e(Ut,Mo),e(Mo,kr),T(ys,kr,null),e(Ut,oh),e(Ut,Lr),e(Lr,nh),g(o,wi,y),g(o,Ce,y),T(bs,Ce,null),e(Ce,sh),e(Ce,wr),e(wr,ah),e(Ce,rh),e(Ce,ks),e(ks,ih),e(ks,aa),e(aa,lh),e(ks,dh),e(Ce,ch),e(Ce,Ls),e(Ls,ph),e(Ls,ws),e(ws,uh),e(Ls,hh),e(Ce,mh),T($o,Ce,null),e(Ce,fh),e(Ce,tt),T(Ts,tt,null),e(tt,gh),e(tt,Wt),e(Wt,_h),e(Wt,ra),e(ra,yh),e(Wt,bh),e(Wt,Tr),e(Tr,kh),e(Wt,Lh),e(tt,wh),T(xo,tt,null),e(tt,Th),T(jo,tt,null),g(o,Ti,y),g(o,Ht,y),e(Ht,Fo),e(Fo,vr),T(vs,vr,null),e(Ht,vh),e(Ht,Mr),e(Mr,Mh),g(o,vi,y),g(o,Pe,y),T(Ms,Pe,null),e(Pe,$h),e(Pe,ct),e(ct,xh),e(ct,$s),e($s,jh),e(ct,Fh),e(ct,$r),e($r,Eh),e(ct,zh),e(ct,xr),e(xr,qh),e(ct,Ch),e(Pe,Ph),e(Pe,xs),e(xs,Nh),e(xs,ia),e(ia,Ah),e(xs,Oh),e(Pe,Ih),e(Pe,js),e(js,Dh),e(js,Fs),e(Fs,Sh),e(js,Uh),e(Pe,Wh),T(Eo,Pe,null),e(Pe,Hh),e(Pe,ot),T(Es,ot,null),e(ot,Qh),e(ot,Qt),e(Qt,Bh),e(Qt,la),e(la,Rh),e(Qt,Kh),e(Qt,jr),e(jr,Vh),e(Qt,Xh),e(ot,Yh),T(zo,ot,null),e(ot,Zh),T(qo,ot,null),Mi=!0},p(o,[y]){const zs={};y&2&&(zs.$$scope={dirty:y,ctx:o}),Kt.$set(zs);const Fr={};y&2&&(Fr.$$scope={dirty:y,ctx:o}),Jt.$set(Fr);const Er={};y&2&&(Er.$$scope={dirty:y,ctx:o}),eo.$set(Er);const zr={};y&2&&(zr.$$scope={dirty:y,ctx:o}),oo.$set(zr);const qs={};y&2&&(qs.$$scope={dirty:y,ctx:o}),no.$set(qs);const qr={};y&2&&(qr.$$scope={dirty:y,ctx:o}),ao.$set(qr);const Cr={};y&2&&(Cr.$$scope={dirty:y,ctx:o}),ro.$set(Cr);const Pr={};y&2&&(Pr.$$scope={dirty:y,ctx:o}),lo.$set(Pr);const Cs={};y&2&&(Cs.$$scope={dirty:y,ctx:o}),co.$set(Cs);const Nr={};y&2&&(Nr.$$scope={dirty:y,ctx:o}),uo.$set(Nr);const Bt={};y&2&&(Bt.$$scope={dirty:y,ctx:o}),mo.$set(Bt);const Ps={};y&2&&(Ps.$$scope={dirty:y,ctx:o}),fo.$set(Ps);const Ar={};y&2&&(Ar.$$scope={dirty:y,ctx:o}),go.$set(Ar);const Ns={};y&2&&(Ns.$$scope={dirty:y,ctx:o}),yo.$set(Ns);const Or={};y&2&&(Or.$$scope={dirty:y,ctx:o}),bo.$set(Or);const As={};y&2&&(As.$$scope={dirty:y,ctx:o}),ko.$set(As);const Ir={};y&2&&(Ir.$$scope={dirty:y,ctx:o}),wo.$set(Ir);const Dr={};y&2&&(Dr.$$scope={dirty:y,ctx:o}),To.$set(Dr);const Sr={};y&2&&(Sr.$$scope={dirty:y,ctx:o}),vo.$set(Sr);const Ur={};y&2&&(Ur.$$scope={dirty:y,ctx:o}),$o.$set(Ur);const Wr={};y&2&&(Wr.$$scope={dirty:y,ctx:o}),xo.$set(Wr);const Hr={};y&2&&(Hr.$$scope={dirty:y,ctx:o}),jo.$set(Hr);const Ie={};y&2&&(Ie.$$scope={dirty:y,ctx:o}),Eo.$set(Ie);const Qr={};y&2&&(Qr.$$scope={dirty:y,ctx:o}),zo.$set(Qr);const Br={};y&2&&(Br.$$scope={dirty:y,ctx:o}),qo.$set(Br)},i(o){Mi||(v(l.$$.fragment,o),v(S.$$.fragment,o),v(Go.$$.fragment,o),v(Jo.$$.fragment,o),v(sn.$$.fragment,o),v(an.$$.fragment,o),v(Kt.$$.fragment,o),v(ln.$$.fragment,o),v(dn.$$.fragment,o),v(pn.$$.fragment,o),v(un.$$.fragment,o),v(mn.$$.fragment,o),v(fn.$$.fragment,o),v(kn.$$.fragment,o),v(Jt.$$.fragment,o),v(eo.$$.fragment,o),v(Ln.$$.fragment,o),v(wn.$$.fragment,o),v($n.$$.fragment,o),v(oo.$$.fragment,o),v(no.$$.fragment,o),v(xn.$$.fragment,o),v(jn.$$.fragment,o),v(Nn.$$.fragment,o),v(ao.$$.fragment,o),v(ro.$$.fragment,o),v(An.$$.fragment,o),v(On.$$.fragment,o),v(Qn.$$.fragment,o),v(lo.$$.fragment,o),v(co.$$.fragment,o),v(Bn.$$.fragment,o),v(Rn.$$.fragment,o),v(Gn.$$.fragment,o),v(uo.$$.fragment,o),v(Jn.$$.fragment,o),v(es.$$.fragment,o),v(mo.$$.fragment,o),v(ss.$$.fragment,o),v(fo.$$.fragment,o),v(go.$$.fragment,o),v(as.$$.fragment,o),v(rs.$$.fragment,o),v(yo.$$.fragment,o),v(ps.$$.fragment,o),v(bo.$$.fragment,o),v(ko.$$.fragment,o),v(us.$$.fragment,o),v(hs.$$.fragment,o),v(wo.$$.fragment,o),v(_s.$$.fragment,o),v(To.$$.fragment,o),v(vo.$$.fragment,o),v(ys.$$.fragment,o),v(bs.$$.fragment,o),v($o.$$.fragment,o),v(Ts.$$.fragment,o),v(xo.$$.fragment,o),v(jo.$$.fragment,o),v(vs.$$.fragment,o),v(Ms.$$.fragment,o),v(Eo.$$.fragment,o),v(Es.$$.fragment,o),v(zo.$$.fragment,o),v(qo.$$.fragment,o),Mi=!0)},o(o){M(l.$$.fragment,o),M(S.$$.fragment,o),M(Go.$$.fragment,o),M(Jo.$$.fragment,o),M(sn.$$.fragment,o),M(an.$$.fragment,o),M(Kt.$$.fragment,o),M(ln.$$.fragment,o),M(dn.$$.fragment,o),M(pn.$$.fragment,o),M(un.$$.fragment,o),M(mn.$$.fragment,o),M(fn.$$.fragment,o),M(kn.$$.fragment,o),M(Jt.$$.fragment,o),M(eo.$$.fragment,o),M(Ln.$$.fragment,o),M(wn.$$.fragment,o),M($n.$$.fragment,o),M(oo.$$.fragment,o),M(no.$$.fragment,o),M(xn.$$.fragment,o),M(jn.$$.fragment,o),M(Nn.$$.fragment,o),M(ao.$$.fragment,o),M(ro.$$.fragment,o),M(An.$$.fragment,o),M(On.$$.fragment,o),M(Qn.$$.fragment,o),M(lo.$$.fragment,o),M(co.$$.fragment,o),M(Bn.$$.fragment,o),M(Rn.$$.fragment,o),M(Gn.$$.fragment,o),M(uo.$$.fragment,o),M(Jn.$$.fragment,o),M(es.$$.fragment,o),M(mo.$$.fragment,o),M(ss.$$.fragment,o),M(fo.$$.fragment,o),M(go.$$.fragment,o),M(as.$$.fragment,o),M(rs.$$.fragment,o),M(yo.$$.fragment,o),M(ps.$$.fragment,o),M(bo.$$.fragment,o),M(ko.$$.fragment,o),M(us.$$.fragment,o),M(hs.$$.fragment,o),M(wo.$$.fragment,o),M(_s.$$.fragment,o),M(To.$$.fragment,o),M(vo.$$.fragment,o),M(ys.$$.fragment,o),M(bs.$$.fragment,o),M($o.$$.fragment,o),M(Ts.$$.fragment,o),M(xo.$$.fragment,o),M(jo.$$.fragment,o),M(vs.$$.fragment,o),M(Ms.$$.fragment,o),M(Eo.$$.fragment,o),M(Es.$$.fragment,o),M(zo.$$.fragment,o),M(qo.$$.fragment,o),Mi=!1},d(o){t(d),o&&t(k),o&&t(_),$(l),o&&t(X),o&&t(E),o&&t(ae),o&&t(I),$(S),o&&t(D),o&&t(G),o&&t(H),o&&t(V),o&&t(x),o&&t(K),o&&t(we),o&&t(A),o&&t(Te),o&&t(O),o&&t(ve),o&&t(ye),o&&t(Vr),$(Go,o),o&&t(Xr),o&&t(ht),o&&t(Yr),$(Jo,o),o&&t(Zr),o&&t(Is),o&&t(Gr),o&&t(mt),o&&t(Jr),o&&t(bt),$(sn),o&&t(ei),o&&t(We),$(an),$(Kt),o&&t(ti),o&&t(wt),$(ln),o&&t(oi),o&&t(He),$(dn),o&&t(ni),o&&t(Tt),$(pn),o&&t(si),o&&t(Qe),$(un),o&&t(ai),o&&t(vt),$(mn),o&&t(ri),o&&t(Be),$(fn),$(kn),$(Jt),$(eo),o&&t(ii),o&&t($t),$(Ln),o&&t(li),o&&t(Re),$(wn),$($n),$(oo),$(no),o&&t(di),o&&t(Ft),$(xn),o&&t(ci),o&&t(Ne),$(jn),$(Nn),$(ao),$(ro),o&&t(pi),o&&t(zt),$(An),o&&t(ui),o&&t(Ae),$(On),$(Qn),$(lo),$(co),o&&t(hi),o&&t(Pt),$(Bn),o&&t(mi),o&&t(Oe),$(Rn),$(Gn),$(uo),o&&t(fi),o&&t(Nt),$(Jn),o&&t(gi),o&&t(Ee),$(es),$(mo),$(ss),$(fo),$(go),o&&t(_i),o&&t(Ot),$(as),o&&t(yi),o&&t(ze),$(rs),$(yo),$(ps),$(bo),$(ko),o&&t(bi),o&&t(Dt),$(us),o&&t(ki),o&&t(qe),$(hs),$(wo),$(_s),$(To),$(vo),o&&t(Li),o&&t(Ut),$(ys),o&&t(wi),o&&t(Ce),$(bs),$($o),$(Ts),$(xo),$(jo),o&&t(Ti),o&&t(Ht),$(vs),o&&t(vi),o&&t(Pe),$(Ms),$(Eo),$(Es),$(zo),$(qo)}}}const Rg={local:"layoutlm",sections:[{local:"overview",title:"Overview"},{local:"transformers.LayoutLMConfig",title:"LayoutLMConfig"},{local:"transformers.LayoutLMTokenizer",title:"LayoutLMTokenizer"},{local:"transformers.LayoutLMTokenizerFast",title:"LayoutLMTokenizerFast"},{local:"transformers.LayoutLMModel",title:"LayoutLMModel"},{local:"transformers.LayoutLMForMaskedLM",title:"LayoutLMForMaskedLM"},{local:"transformers.LayoutLMForSequenceClassification",title:"LayoutLMForSequenceClassification"},{local:"transformers.LayoutLMForTokenClassification",title:"LayoutLMForTokenClassification"},{local:"transformers.LayoutLMForQuestionAnswering",title:"LayoutLMForQuestionAnswering"},{local:"transformers.TFLayoutLMModel",title:"TFLayoutLMModel"},{local:"transformers.TFLayoutLMForMaskedLM",title:"TFLayoutLMForMaskedLM"},{local:"transformers.TFLayoutLMForSequenceClassification",title:"TFLayoutLMForSequenceClassification"},{local:"transformers.TFLayoutLMForTokenClassification",title:"TFLayoutLMForTokenClassification"},{local:"transformers.TFLayoutLMForQuestionAnswering",title:"TFLayoutLMForQuestionAnswering"}],title:"LayoutLM"};function Kg(z){return bg(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class e_ extends fg{constructor(d){super();gg(this,d,Kg,Bg,_g,{})}}export{e_ as default,Rg as metadata};
