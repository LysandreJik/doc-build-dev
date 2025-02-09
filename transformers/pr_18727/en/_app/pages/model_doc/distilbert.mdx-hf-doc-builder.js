import{S as Zy,i as eF,s as tF,e as o,k as p,w as T,t as a,M as sF,c as n,d as t,m as h,a as r,x as $,h as i,b as m,G as e,g as b,y as w,q as y,o as F,B as D,v as oF,L as ae}from"../../chunks/vendor-hf-doc-builder.js";import{T as be}from"../../chunks/Tip-hf-doc-builder.js";import{D as H}from"../../chunks/Docstring-hf-doc-builder.js";import{C as ie}from"../../chunks/CodeBlock-hf-doc-builder.js";import{I as Te}from"../../chunks/IconCopyLink-hf-doc-builder.js";import{E as re}from"../../chunks/ExampleCodeBlock-hf-doc-builder.js";function nF(x){let d,g,c,u,_;return u=new ie({props:{code:`from transformers import DistilBertModel, DistilBertConfig

# Initializing a DistilBERT configuration
configuration = DistilBertConfig()

# Initializing a model from the configuration
model = DistilBertModel(configuration)

# Accessing the model configuration
configuration = model.config`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> DistilBertModel, DistilBertConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a DistilBERT configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = DistilBertConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = DistilBertModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`}}),{c(){d=o("p"),g=a("Examples:"),c=p(),T(u.$$.fragment)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Examples:"),f.forEach(t),c=h(l),$(u.$$.fragment,l)},m(l,f){b(l,d,f),e(d,g),b(l,c,f),w(u,l,f),_=!0},p:ae,i(l){_||(y(u.$$.fragment,l),_=!0)},o(l){F(u.$$.fragment,l),_=!1},d(l){l&&t(d),l&&t(c),D(u,l)}}}function rF(x){let d,g,c,u,_;return{c(){d=o("p"),g=a("Although the recipe for forward pass needs to be defined within this function, one should call the "),c=o("code"),u=a("Module"),_=a(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),c=n(f,"CODE",{});var B=r(c);u=i(B,"Module"),B.forEach(t),_=i(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,g),e(d,c),e(c,u),e(d,_)},d(l){l&&t(d)}}}function aF(x){let d,g,c,u,_;return u=new ie({props:{code:`from transformers import DistilBertTokenizer, DistilBertModel
import torch

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> DistilBertTokenizer, DistilBertModel
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = DistilBertTokenizer.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = DistilBertModel.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`}}),{c(){d=o("p"),g=a("Example:"),c=p(),T(u.$$.fragment)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Example:"),f.forEach(t),c=h(l),$(u.$$.fragment,l)},m(l,f){b(l,d,f),e(d,g),b(l,c,f),w(u,l,f),_=!0},p:ae,i(l){_||(y(u.$$.fragment,l),_=!0)},o(l){F(u.$$.fragment,l),_=!1},d(l){l&&t(d),l&&t(c),D(u,l)}}}function iF(x){let d,g,c,u,_;return{c(){d=o("p"),g=a("Although the recipe for forward pass needs to be defined within this function, one should call the "),c=o("code"),u=a("Module"),_=a(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),c=n(f,"CODE",{});var B=r(c);u=i(B,"Module"),B.forEach(t),_=i(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,g),e(d,c),e(c,u),e(d,_)},d(l){l&&t(d)}}}function lF(x){let d,g,c,u,_;return u=new ie({props:{code:`from transformers import DistilBertTokenizer, DistilBertForMaskedLM
import torch

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")

inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# retrieve index of [MASK]
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
tokenizer.decode(predicted_token_id)
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> DistilBertTokenizer, DistilBertForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = DistilBertTokenizer.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = DistilBertForMaskedLM.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;The capital of France is [MASK].&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># retrieve index of [MASK]</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[<span class="hljs-number">0</span>].nonzero(as_tuple=<span class="hljs-literal">True</span>)[<span class="hljs-number">0</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_token_id = logits[<span class="hljs-number">0</span>, mask_token_index].argmax(axis=-<span class="hljs-number">1</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.decode(predicted_token_id)
`}}),{c(){d=o("p"),g=a("Example:"),c=p(),T(u.$$.fragment)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Example:"),f.forEach(t),c=h(l),$(u.$$.fragment,l)},m(l,f){b(l,d,f),e(d,g),b(l,c,f),w(u,l,f),_=!0},p:ae,i(l){_||(y(u.$$.fragment,l),_=!0)},o(l){F(u.$$.fragment,l),_=!1},d(l){l&&t(d),l&&t(c),D(u,l)}}}function dF(x){let d,g;return d=new ie({props:{code:`labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
# mask labels of non-[MASK] tokens
labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

outputs = model(**inputs, labels=labels)
round(outputs.loss.item(), 2)
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span>labels = tokenizer(<span class="hljs-string">&quot;The capital of France is Paris.&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># mask labels of non-[MASK] tokens</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -<span class="hljs-number">100</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs, labels=labels)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(outputs.loss.item(), <span class="hljs-number">2</span>)
`}}),{c(){T(d.$$.fragment)},l(c){$(d.$$.fragment,c)},m(c,u){w(d,c,u),g=!0},p:ae,i(c){g||(y(d.$$.fragment,c),g=!0)},o(c){F(d.$$.fragment,c),g=!1},d(c){D(d,c)}}}function cF(x){let d,g,c,u,_;return{c(){d=o("p"),g=a("Although the recipe for forward pass needs to be defined within this function, one should call the "),c=o("code"),u=a("Module"),_=a(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),c=n(f,"CODE",{});var B=r(c);u=i(B,"Module"),B.forEach(t),_=i(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,g),e(d,c),e(c,u),e(d,_)},d(l){l&&t(d)}}}function pF(x){let d,g,c,u,_;return u=new ie({props:{code:`import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> DistilBertTokenizer, DistilBertForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = DistilBertTokenizer.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = DistilBertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
`}}),{c(){d=o("p"),g=a("Example of single-label classification:"),c=p(),T(u.$$.fragment)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Example of single-label classification:"),f.forEach(t),c=h(l),$(u.$$.fragment,l)},m(l,f){b(l,d,f),e(d,g),b(l,c,f),w(u,l,f),_=!0},p:ae,i(l){_||(y(u.$$.fragment,l),_=!0)},o(l){F(u.$$.fragment,l),_=!1},d(l){l&&t(d),l&&t(c),D(u,l)}}}function hF(x){let d,g;return d=new ie({props:{code:`# To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`
num_labels = len(model.config.id2label)
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

labels = torch.tensor(1)
loss = model(**inputs, labels=labels).loss
round(loss.item(), 2)
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = DistilBertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">1</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
`}}),{c(){T(d.$$.fragment)},l(c){$(d.$$.fragment,c)},m(c,u){w(d,c,u),g=!0},p:ae,i(c){g||(y(d.$$.fragment,c),g=!0)},o(c){F(d.$$.fragment,c),g=!1},d(c){D(d,c)}}}function uF(x){let d,g,c,u,_;return u=new ie({props:{code:`import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", problem_type="multi_label_classification")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> DistilBertTokenizer, DistilBertForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = DistilBertTokenizer.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = DistilBertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
`}}),{c(){d=o("p"),g=a("Example of multi-label classification:"),c=p(),T(u.$$.fragment)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Example of multi-label classification:"),f.forEach(t),c=h(l),$(u.$$.fragment,l)},m(l,f){b(l,d,f),e(d,g),b(l,c,f),w(u,l,f),_=!0},p:ae,i(l){_||(y(u.$$.fragment,l),_=!0)},o(l){F(u.$$.fragment,l),_=!1},d(l){l&&t(d),l&&t(c),D(u,l)}}}function fF(x){let d,g;return d=new ie({props:{code:`# To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`
num_labels = len(model.config.id2label)
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=num_labels, problem_type="multi_label_classification"
)

labels = torch.nn.functional.one_hot(torch.tensor([predicted_class_id]), num_classes=num_labels).to(
    torch.float
)
loss = model(**inputs, labels=labels).loss
loss.backward()`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = DistilBertForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.nn.functional.one_hot(torch.tensor([predicted_class_id]), num_classes=num_labels).to(
<span class="hljs-meta">... </span>    torch.<span class="hljs-built_in">float</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span>loss.backward()`}}),{c(){T(d.$$.fragment)},l(c){$(d.$$.fragment,c)},m(c,u){w(d,c,u),g=!0},p:ae,i(c){g||(y(d.$$.fragment,c),g=!0)},o(c){F(d.$$.fragment,c),g=!1},d(c){D(d,c)}}}function mF(x){let d,g,c,u,_;return{c(){d=o("p"),g=a("Although the recipe for forward pass needs to be defined within this function, one should call the "),c=o("code"),u=a("Module"),_=a(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),c=n(f,"CODE",{});var B=r(c);u=i(B,"Module"),B.forEach(t),_=i(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,g),e(d,c),e(c,u),e(d,_)},d(l){l&&t(d)}}}function gF(x){let d,g,c,u,_;return u=new ie({props:{code:`from transformers import DistilBertTokenizer, DistilBertForMultipleChoice
import torch

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")
model = DistilBertForMultipleChoice.from_pretrained("distilbert-base-cased")

prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
choice0 = "It is eaten with a fork and a knife."
choice1 = "It is eaten while held in the hand."
labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

encoding = tokenizer([[prompt, choice0], [prompt, choice1]], return_tensors="pt", padding=True)
outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)  # batch size is 1

# the linear classifier still needs to be trained
loss = outputs.loss
logits = outputs.logits`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> DistilBertTokenizer, DistilBertForMultipleChoice
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = DistilBertTokenizer.from_pretrained(<span class="hljs-string">&quot;distilbert-base-cased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = DistilBertForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;distilbert-base-cased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">0</span>).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># choice0 is correct (according to Wikipedia ;)), batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([[prompt, choice0], [prompt, choice1]], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**{k: v.unsqueeze(<span class="hljs-number">0</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()}, labels=labels)  <span class="hljs-comment"># batch size is 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the linear classifier still needs to be trained</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`}}),{c(){d=o("p"),g=a("Examples:"),c=p(),T(u.$$.fragment)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Examples:"),f.forEach(t),c=h(l),$(u.$$.fragment,l)},m(l,f){b(l,d,f),e(d,g),b(l,c,f),w(u,l,f),_=!0},p:ae,i(l){_||(y(u.$$.fragment,l),_=!0)},o(l){F(u.$$.fragment,l),_=!1},d(l){l&&t(d),l&&t(c),D(u,l)}}}function _F(x){let d,g,c,u,_;return{c(){d=o("p"),g=a("Although the recipe for forward pass needs to be defined within this function, one should call the "),c=o("code"),u=a("Module"),_=a(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),c=n(f,"CODE",{});var B=r(c);u=i(B,"Module"),B.forEach(t),_=i(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,g),e(d,c),e(c,u),e(d,_)},d(l){l&&t(d)}}}function bF(x){let d,g,c,u,_;return u=new ie({props:{code:`from transformers import DistilBertTokenizer, DistilBertForTokenClassification
import torch

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForTokenClassification.from_pretrained("distilbert-base-uncased")

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
predicted_tokens_classes
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> DistilBertTokenizer, DistilBertForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = DistilBertTokenizer.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = DistilBertForTokenClassification.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)

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
`}}),{c(){d=o("p"),g=a("Example:"),c=p(),T(u.$$.fragment)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Example:"),f.forEach(t),c=h(l),$(u.$$.fragment,l)},m(l,f){b(l,d,f),e(d,g),b(l,c,f),w(u,l,f),_=!0},p:ae,i(l){_||(y(u.$$.fragment,l),_=!0)},o(l){F(u.$$.fragment,l),_=!1},d(l){l&&t(d),l&&t(c),D(u,l)}}}function vF(x){let d,g;return d=new ie({props:{code:`labels = predicted_token_class_ids
loss = model(**inputs, labels=labels).loss
round(loss.item(), 2)
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span>labels = predicted_token_class_ids
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
`}}),{c(){T(d.$$.fragment)},l(c){$(d.$$.fragment,c)},m(c,u){w(d,c,u),g=!0},p:ae,i(c){g||(y(d.$$.fragment,c),g=!0)},o(c){F(d.$$.fragment,c),g=!1},d(c){D(d,c)}}}function kF(x){let d,g,c,u,_;return{c(){d=o("p"),g=a("Although the recipe for forward pass needs to be defined within this function, one should call the "),c=o("code"),u=a("Module"),_=a(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),c=n(f,"CODE",{});var B=r(c);u=i(B,"Module"),B.forEach(t),_=i(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,g),e(d,c),e(c,u),e(d,_)},d(l){l&&t(d)}}}function TF(x){let d,g,c,u,_;return u=new ie({props:{code:`from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

inputs = tokenizer(question, text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
tokenizer.decode(predict_answer_tokens)
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> DistilBertTokenizer, DistilBertForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = DistilBertTokenizer.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = DistilBertForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>question, text = <span class="hljs-string">&quot;Who was Jim Henson?&quot;</span>, <span class="hljs-string">&quot;Jim Henson was a nice puppet&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(question, text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>answer_start_index = outputs.start_logits.argmax()
<span class="hljs-meta">&gt;&gt;&gt; </span>answer_end_index = outputs.end_logits.argmax()

<span class="hljs-meta">&gt;&gt;&gt; </span>predict_answer_tokens = inputs.input_ids[<span class="hljs-number">0</span>, answer_start_index : answer_end_index + <span class="hljs-number">1</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.decode(predict_answer_tokens)
`}}),{c(){d=o("p"),g=a("Example:"),c=p(),T(u.$$.fragment)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Example:"),f.forEach(t),c=h(l),$(u.$$.fragment,l)},m(l,f){b(l,d,f),e(d,g),b(l,c,f),w(u,l,f),_=!0},p:ae,i(l){_||(y(u.$$.fragment,l),_=!0)},o(l){F(u.$$.fragment,l),_=!1},d(l){l&&t(d),l&&t(c),D(u,l)}}}function $F(x){let d,g;return d=new ie({props:{code:`# target is "nice puppet"
target_start_index = torch.tensor([14])
target_end_index = torch.tensor([15])

outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
loss = outputs.loss
round(loss.item(), 2)
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># target is &quot;nice puppet&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>target_start_index = torch.tensor([<span class="hljs-number">14</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>target_end_index = torch.tensor([<span class="hljs-number">15</span>])

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
`}}),{c(){T(d.$$.fragment)},l(c){$(d.$$.fragment,c)},m(c,u){w(d,c,u),g=!0},p:ae,i(c){g||(y(d.$$.fragment,c),g=!0)},o(c){F(d.$$.fragment,c),g=!1},d(c){D(d,c)}}}function wF(x){let d,g,c,u,_,l,f,B,he,V,E,G,S,X,ue,I,fe,le,N,P,Y,K,z,j,me,W,se,ge,Q,de,ee,A,ce,L,oe,_e,q,te,R,pe;return{c(){d=o("p"),g=a("TF 2.0 models accepts two formats as inputs:"),c=p(),u=o("ul"),_=o("li"),l=a("having all inputs as keyword arguments (like PyTorch models), or"),f=p(),B=o("li"),he=a("having all inputs as a list, tuple or dict in the first positional arguments."),V=p(),E=o("p"),G=a("This second option is useful when using "),S=o("code"),X=a("tf.keras.Model.fit"),ue=a(` method which currently requires having all the
tensors in the first argument of the model call function: `),I=o("code"),fe=a("model(inputs)"),le=a("."),N=p(),P=o("p"),Y=a(`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),K=p(),z=o("ul"),j=o("li"),me=a("a single Tensor with "),W=o("code"),se=a("input_ids"),ge=a(" only and nothing else: "),Q=o("code"),de=a("model(inputs_ids)"),ee=p(),A=o("li"),ce=a(`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),L=o("code"),oe=a("model([input_ids, attention_mask])"),_e=p(),q=o("li"),te=a(`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),R=o("code"),pe=a('model({"input_ids": input_ids})')},l(k){d=n(k,"P",{});var M=r(d);g=i(M,"TF 2.0 models accepts two formats as inputs:"),M.forEach(t),c=h(k),u=n(k,"UL",{});var J=r(u);_=n(J,"LI",{});var ve=r(_);l=i(ve,"having all inputs as keyword arguments (like PyTorch models), or"),ve.forEach(t),f=h(J),B=n(J,"LI",{});var $e=r(B);he=i($e,"having all inputs as a list, tuple or dict in the first positional arguments."),$e.forEach(t),J.forEach(t),V=h(k),E=n(k,"P",{});var O=r(E);G=i(O,"This second option is useful when using "),S=n(O,"CODE",{});var ke=r(S);X=i(ke,"tf.keras.Model.fit"),ke.forEach(t),ue=i(O,` method which currently requires having all the
tensors in the first argument of the model call function: `),I=n(O,"CODE",{});var we=r(I);fe=i(we,"model(inputs)"),we.forEach(t),le=i(O,"."),O.forEach(t),N=h(k),P=n(k,"P",{});var ye=r(P);Y=i(ye,`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),ye.forEach(t),K=h(k),z=n(k,"UL",{});var C=r(z);j=n(C,"LI",{});var U=r(j);me=i(U,"a single Tensor with "),W=n(U,"CODE",{});var Fe=r(W);se=i(Fe,"input_ids"),Fe.forEach(t),ge=i(U," only and nothing else: "),Q=n(U,"CODE",{});var De=r(Q);de=i(De,"model(inputs_ids)"),De.forEach(t),U.forEach(t),ee=h(C),A=n(C,"LI",{});var Z=r(A);ce=i(Z,`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),L=n(Z,"CODE",{});var xe=r(L);oe=i(xe,"model([input_ids, attention_mask])"),xe.forEach(t),Z.forEach(t),_e=h(C),q=n(C,"LI",{});var ne=r(q);te=i(ne,`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),R=n(ne,"CODE",{});var Be=r(R);pe=i(Be,'model({"input_ids": input_ids})'),Be.forEach(t),ne.forEach(t),C.forEach(t)},m(k,M){b(k,d,M),e(d,g),b(k,c,M),b(k,u,M),e(u,_),e(_,l),e(u,f),e(u,B),e(B,he),b(k,V,M),b(k,E,M),e(E,G),e(E,S),e(S,X),e(E,ue),e(E,I),e(I,fe),e(E,le),b(k,N,M),b(k,P,M),e(P,Y),b(k,K,M),b(k,z,M),e(z,j),e(j,me),e(j,W),e(W,se),e(j,ge),e(j,Q),e(Q,de),e(z,ee),e(z,A),e(A,ce),e(A,L),e(L,oe),e(z,_e),e(z,q),e(q,te),e(q,R),e(R,pe)},d(k){k&&t(d),k&&t(c),k&&t(u),k&&t(V),k&&t(E),k&&t(N),k&&t(P),k&&t(K),k&&t(z)}}}function yF(x){let d,g,c,u,_;return{c(){d=o("p"),g=a("Although the recipe for forward pass needs to be defined within this function, one should call the "),c=o("code"),u=a("Module"),_=a(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),c=n(f,"CODE",{});var B=r(c);u=i(B,"Module"),B.forEach(t),_=i(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,g),e(d,c),e(c,u),e(d,_)},d(l){l&&t(d)}}}function FF(x){let d,g,c,u,_;return u=new ie({props:{code:`from transformers import DistilBertTokenizer, TFDistilBertModel
import tensorflow as tf

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
outputs = model(inputs)

last_hidden_states = outputs.last_hidden_state`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> DistilBertTokenizer, TFDistilBertModel
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> tensorflow <span class="hljs-keyword">as</span> tf

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = DistilBertTokenizer.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFDistilBertModel.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;tf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`}}),{c(){d=o("p"),g=a("Example:"),c=p(),T(u.$$.fragment)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Example:"),f.forEach(t),c=h(l),$(u.$$.fragment,l)},m(l,f){b(l,d,f),e(d,g),b(l,c,f),w(u,l,f),_=!0},p:ae,i(l){_||(y(u.$$.fragment,l),_=!0)},o(l){F(u.$$.fragment,l),_=!1},d(l){l&&t(d),l&&t(c),D(u,l)}}}function DF(x){let d,g,c,u,_,l,f,B,he,V,E,G,S,X,ue,I,fe,le,N,P,Y,K,z,j,me,W,se,ge,Q,de,ee,A,ce,L,oe,_e,q,te,R,pe;return{c(){d=o("p"),g=a("TF 2.0 models accepts two formats as inputs:"),c=p(),u=o("ul"),_=o("li"),l=a("having all inputs as keyword arguments (like PyTorch models), or"),f=p(),B=o("li"),he=a("having all inputs as a list, tuple or dict in the first positional arguments."),V=p(),E=o("p"),G=a("This second option is useful when using "),S=o("code"),X=a("tf.keras.Model.fit"),ue=a(` method which currently requires having all the
tensors in the first argument of the model call function: `),I=o("code"),fe=a("model(inputs)"),le=a("."),N=p(),P=o("p"),Y=a(`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),K=p(),z=o("ul"),j=o("li"),me=a("a single Tensor with "),W=o("code"),se=a("input_ids"),ge=a(" only and nothing else: "),Q=o("code"),de=a("model(inputs_ids)"),ee=p(),A=o("li"),ce=a(`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),L=o("code"),oe=a("model([input_ids, attention_mask])"),_e=p(),q=o("li"),te=a(`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),R=o("code"),pe=a('model({"input_ids": input_ids})')},l(k){d=n(k,"P",{});var M=r(d);g=i(M,"TF 2.0 models accepts two formats as inputs:"),M.forEach(t),c=h(k),u=n(k,"UL",{});var J=r(u);_=n(J,"LI",{});var ve=r(_);l=i(ve,"having all inputs as keyword arguments (like PyTorch models), or"),ve.forEach(t),f=h(J),B=n(J,"LI",{});var $e=r(B);he=i($e,"having all inputs as a list, tuple or dict in the first positional arguments."),$e.forEach(t),J.forEach(t),V=h(k),E=n(k,"P",{});var O=r(E);G=i(O,"This second option is useful when using "),S=n(O,"CODE",{});var ke=r(S);X=i(ke,"tf.keras.Model.fit"),ke.forEach(t),ue=i(O,` method which currently requires having all the
tensors in the first argument of the model call function: `),I=n(O,"CODE",{});var we=r(I);fe=i(we,"model(inputs)"),we.forEach(t),le=i(O,"."),O.forEach(t),N=h(k),P=n(k,"P",{});var ye=r(P);Y=i(ye,`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),ye.forEach(t),K=h(k),z=n(k,"UL",{});var C=r(z);j=n(C,"LI",{});var U=r(j);me=i(U,"a single Tensor with "),W=n(U,"CODE",{});var Fe=r(W);se=i(Fe,"input_ids"),Fe.forEach(t),ge=i(U," only and nothing else: "),Q=n(U,"CODE",{});var De=r(Q);de=i(De,"model(inputs_ids)"),De.forEach(t),U.forEach(t),ee=h(C),A=n(C,"LI",{});var Z=r(A);ce=i(Z,`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),L=n(Z,"CODE",{});var xe=r(L);oe=i(xe,"model([input_ids, attention_mask])"),xe.forEach(t),Z.forEach(t),_e=h(C),q=n(C,"LI",{});var ne=r(q);te=i(ne,`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),R=n(ne,"CODE",{});var Be=r(R);pe=i(Be,'model({"input_ids": input_ids})'),Be.forEach(t),ne.forEach(t),C.forEach(t)},m(k,M){b(k,d,M),e(d,g),b(k,c,M),b(k,u,M),e(u,_),e(_,l),e(u,f),e(u,B),e(B,he),b(k,V,M),b(k,E,M),e(E,G),e(E,S),e(S,X),e(E,ue),e(E,I),e(I,fe),e(E,le),b(k,N,M),b(k,P,M),e(P,Y),b(k,K,M),b(k,z,M),e(z,j),e(j,me),e(j,W),e(W,se),e(j,ge),e(j,Q),e(Q,de),e(z,ee),e(z,A),e(A,ce),e(A,L),e(L,oe),e(z,_e),e(z,q),e(q,te),e(q,R),e(R,pe)},d(k){k&&t(d),k&&t(c),k&&t(u),k&&t(V),k&&t(E),k&&t(N),k&&t(P),k&&t(K),k&&t(z)}}}function xF(x){let d,g,c,u,_;return{c(){d=o("p"),g=a("Although the recipe for forward pass needs to be defined within this function, one should call the "),c=o("code"),u=a("Module"),_=a(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),c=n(f,"CODE",{});var B=r(c);u=i(B,"Module"),B.forEach(t),_=i(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,g),e(d,c),e(c,u),e(d,_)},d(l){l&&t(d)}}}function BF(x){let d,g,c,u,_;return u=new ie({props:{code:`from transformers import DistilBertTokenizer, TFDistilBertForMaskedLM
import tensorflow as tf

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = TFDistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")

inputs = tokenizer("The capital of France is [MASK].", return_tensors="tf")
logits = model(**inputs).logits

# retrieve index of [MASK]
mask_token_index = tf.where((inputs.input_ids == tokenizer.mask_token_id)[0])
selected_logits = tf.gather_nd(logits[0], indices=mask_token_index)

predicted_token_id = tf.math.argmax(selected_logits, axis=-1)
tokenizer.decode(predicted_token_id)
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> DistilBertTokenizer, TFDistilBertForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> tensorflow <span class="hljs-keyword">as</span> tf

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = DistilBertTokenizer.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFDistilBertForMaskedLM.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;The capital of France is [MASK].&quot;</span>, return_tensors=<span class="hljs-string">&quot;tf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># retrieve index of [MASK]</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>mask_token_index = tf.where((inputs.input_ids == tokenizer.mask_token_id)[<span class="hljs-number">0</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>selected_logits = tf.gather_nd(logits[<span class="hljs-number">0</span>], indices=mask_token_index)

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_token_id = tf.math.argmax(selected_logits, axis=-<span class="hljs-number">1</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.decode(predicted_token_id)
`}}),{c(){d=o("p"),g=a("Example:"),c=p(),T(u.$$.fragment)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Example:"),f.forEach(t),c=h(l),$(u.$$.fragment,l)},m(l,f){b(l,d,f),e(d,g),b(l,c,f),w(u,l,f),_=!0},p:ae,i(l){_||(y(u.$$.fragment,l),_=!0)},o(l){F(u.$$.fragment,l),_=!1},d(l){l&&t(d),l&&t(c),D(u,l)}}}function MF(x){let d,g;return d=new ie({props:{code:`labels = tokenizer("The capital of France is Paris.", return_tensors="tf")["input_ids"]
# mask labels of non-[MASK] tokens
labels = tf.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

outputs = model(**inputs, labels=labels)
round(float(outputs.loss), 2)
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span>labels = tokenizer(<span class="hljs-string">&quot;The capital of France is Paris.&quot;</span>, return_tensors=<span class="hljs-string">&quot;tf&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># mask labels of non-[MASK] tokens</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = tf.where(inputs.input_ids == tokenizer.mask_token_id, labels, -<span class="hljs-number">100</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs, labels=labels)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(<span class="hljs-built_in">float</span>(outputs.loss), <span class="hljs-number">2</span>)
`}}),{c(){T(d.$$.fragment)},l(c){$(d.$$.fragment,c)},m(c,u){w(d,c,u),g=!0},p:ae,i(c){g||(y(d.$$.fragment,c),g=!0)},o(c){F(d.$$.fragment,c),g=!1},d(c){D(d,c)}}}function EF(x){let d,g,c,u,_,l,f,B,he,V,E,G,S,X,ue,I,fe,le,N,P,Y,K,z,j,me,W,se,ge,Q,de,ee,A,ce,L,oe,_e,q,te,R,pe;return{c(){d=o("p"),g=a("TF 2.0 models accepts two formats as inputs:"),c=p(),u=o("ul"),_=o("li"),l=a("having all inputs as keyword arguments (like PyTorch models), or"),f=p(),B=o("li"),he=a("having all inputs as a list, tuple or dict in the first positional arguments."),V=p(),E=o("p"),G=a("This second option is useful when using "),S=o("code"),X=a("tf.keras.Model.fit"),ue=a(` method which currently requires having all the
tensors in the first argument of the model call function: `),I=o("code"),fe=a("model(inputs)"),le=a("."),N=p(),P=o("p"),Y=a(`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),K=p(),z=o("ul"),j=o("li"),me=a("a single Tensor with "),W=o("code"),se=a("input_ids"),ge=a(" only and nothing else: "),Q=o("code"),de=a("model(inputs_ids)"),ee=p(),A=o("li"),ce=a(`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),L=o("code"),oe=a("model([input_ids, attention_mask])"),_e=p(),q=o("li"),te=a(`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),R=o("code"),pe=a('model({"input_ids": input_ids})')},l(k){d=n(k,"P",{});var M=r(d);g=i(M,"TF 2.0 models accepts two formats as inputs:"),M.forEach(t),c=h(k),u=n(k,"UL",{});var J=r(u);_=n(J,"LI",{});var ve=r(_);l=i(ve,"having all inputs as keyword arguments (like PyTorch models), or"),ve.forEach(t),f=h(J),B=n(J,"LI",{});var $e=r(B);he=i($e,"having all inputs as a list, tuple or dict in the first positional arguments."),$e.forEach(t),J.forEach(t),V=h(k),E=n(k,"P",{});var O=r(E);G=i(O,"This second option is useful when using "),S=n(O,"CODE",{});var ke=r(S);X=i(ke,"tf.keras.Model.fit"),ke.forEach(t),ue=i(O,` method which currently requires having all the
tensors in the first argument of the model call function: `),I=n(O,"CODE",{});var we=r(I);fe=i(we,"model(inputs)"),we.forEach(t),le=i(O,"."),O.forEach(t),N=h(k),P=n(k,"P",{});var ye=r(P);Y=i(ye,`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),ye.forEach(t),K=h(k),z=n(k,"UL",{});var C=r(z);j=n(C,"LI",{});var U=r(j);me=i(U,"a single Tensor with "),W=n(U,"CODE",{});var Fe=r(W);se=i(Fe,"input_ids"),Fe.forEach(t),ge=i(U," only and nothing else: "),Q=n(U,"CODE",{});var De=r(Q);de=i(De,"model(inputs_ids)"),De.forEach(t),U.forEach(t),ee=h(C),A=n(C,"LI",{});var Z=r(A);ce=i(Z,`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),L=n(Z,"CODE",{});var xe=r(L);oe=i(xe,"model([input_ids, attention_mask])"),xe.forEach(t),Z.forEach(t),_e=h(C),q=n(C,"LI",{});var ne=r(q);te=i(ne,`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),R=n(ne,"CODE",{});var Be=r(R);pe=i(Be,'model({"input_ids": input_ids})'),Be.forEach(t),ne.forEach(t),C.forEach(t)},m(k,M){b(k,d,M),e(d,g),b(k,c,M),b(k,u,M),e(u,_),e(_,l),e(u,f),e(u,B),e(B,he),b(k,V,M),b(k,E,M),e(E,G),e(E,S),e(S,X),e(E,ue),e(E,I),e(I,fe),e(E,le),b(k,N,M),b(k,P,M),e(P,Y),b(k,K,M),b(k,z,M),e(z,j),e(j,me),e(j,W),e(W,se),e(j,ge),e(j,Q),e(Q,de),e(z,ee),e(z,A),e(A,ce),e(A,L),e(L,oe),e(z,_e),e(z,q),e(q,te),e(q,R),e(R,pe)},d(k){k&&t(d),k&&t(c),k&&t(u),k&&t(V),k&&t(E),k&&t(N),k&&t(P),k&&t(K),k&&t(z)}}}function zF(x){let d,g,c,u,_;return{c(){d=o("p"),g=a("Although the recipe for forward pass needs to be defined within this function, one should call the "),c=o("code"),u=a("Module"),_=a(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),c=n(f,"CODE",{});var B=r(c);u=i(B,"Module"),B.forEach(t),_=i(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,g),e(d,c),e(c,u),e(d,_)},d(l){l&&t(d)}}}function jF(x){let d,g,c,u,_;return u=new ie({props:{code:`from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")

logits = model(**inputs).logits

predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
model.config.id2label[predicted_class_id]
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> DistilBertTokenizer, TFDistilBertForSequenceClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> tensorflow <span class="hljs-keyword">as</span> tf

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = DistilBertTokenizer.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFDistilBertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;tf&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = <span class="hljs-built_in">int</span>(tf.math.argmax(logits, axis=-<span class="hljs-number">1</span>)[<span class="hljs-number">0</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
`}}),{c(){d=o("p"),g=a("Example:"),c=p(),T(u.$$.fragment)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Example:"),f.forEach(t),c=h(l),$(u.$$.fragment,l)},m(l,f){b(l,d,f),e(d,g),b(l,c,f),w(u,l,f),_=!0},p:ae,i(l){_||(y(u.$$.fragment,l),_=!0)},o(l){F(u.$$.fragment,l),_=!1},d(l){l&&t(d),l&&t(c),D(u,l)}}}function CF(x){let d,g;return d=new ie({props:{code:`# To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`
num_labels = len(model.config.id2label)
model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

labels = tf.constant(1)
loss = model(**inputs, labels=labels).loss
round(float(loss), 2)
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFDistilBertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = tf.constant(<span class="hljs-number">1</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(<span class="hljs-built_in">float</span>(loss), <span class="hljs-number">2</span>)
`}}),{c(){T(d.$$.fragment)},l(c){$(d.$$.fragment,c)},m(c,u){w(d,c,u),g=!0},p:ae,i(c){g||(y(d.$$.fragment,c),g=!0)},o(c){F(d.$$.fragment,c),g=!1},d(c){D(d,c)}}}function qF(x){let d,g,c,u,_,l,f,B,he,V,E,G,S,X,ue,I,fe,le,N,P,Y,K,z,j,me,W,se,ge,Q,de,ee,A,ce,L,oe,_e,q,te,R,pe;return{c(){d=o("p"),g=a("TF 2.0 models accepts two formats as inputs:"),c=p(),u=o("ul"),_=o("li"),l=a("having all inputs as keyword arguments (like PyTorch models), or"),f=p(),B=o("li"),he=a("having all inputs as a list, tuple or dict in the first positional arguments."),V=p(),E=o("p"),G=a("This second option is useful when using "),S=o("code"),X=a("tf.keras.Model.fit"),ue=a(` method which currently requires having all the
tensors in the first argument of the model call function: `),I=o("code"),fe=a("model(inputs)"),le=a("."),N=p(),P=o("p"),Y=a(`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),K=p(),z=o("ul"),j=o("li"),me=a("a single Tensor with "),W=o("code"),se=a("input_ids"),ge=a(" only and nothing else: "),Q=o("code"),de=a("model(inputs_ids)"),ee=p(),A=o("li"),ce=a(`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),L=o("code"),oe=a("model([input_ids, attention_mask])"),_e=p(),q=o("li"),te=a(`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),R=o("code"),pe=a('model({"input_ids": input_ids})')},l(k){d=n(k,"P",{});var M=r(d);g=i(M,"TF 2.0 models accepts two formats as inputs:"),M.forEach(t),c=h(k),u=n(k,"UL",{});var J=r(u);_=n(J,"LI",{});var ve=r(_);l=i(ve,"having all inputs as keyword arguments (like PyTorch models), or"),ve.forEach(t),f=h(J),B=n(J,"LI",{});var $e=r(B);he=i($e,"having all inputs as a list, tuple or dict in the first positional arguments."),$e.forEach(t),J.forEach(t),V=h(k),E=n(k,"P",{});var O=r(E);G=i(O,"This second option is useful when using "),S=n(O,"CODE",{});var ke=r(S);X=i(ke,"tf.keras.Model.fit"),ke.forEach(t),ue=i(O,` method which currently requires having all the
tensors in the first argument of the model call function: `),I=n(O,"CODE",{});var we=r(I);fe=i(we,"model(inputs)"),we.forEach(t),le=i(O,"."),O.forEach(t),N=h(k),P=n(k,"P",{});var ye=r(P);Y=i(ye,`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),ye.forEach(t),K=h(k),z=n(k,"UL",{});var C=r(z);j=n(C,"LI",{});var U=r(j);me=i(U,"a single Tensor with "),W=n(U,"CODE",{});var Fe=r(W);se=i(Fe,"input_ids"),Fe.forEach(t),ge=i(U," only and nothing else: "),Q=n(U,"CODE",{});var De=r(Q);de=i(De,"model(inputs_ids)"),De.forEach(t),U.forEach(t),ee=h(C),A=n(C,"LI",{});var Z=r(A);ce=i(Z,`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),L=n(Z,"CODE",{});var xe=r(L);oe=i(xe,"model([input_ids, attention_mask])"),xe.forEach(t),Z.forEach(t),_e=h(C),q=n(C,"LI",{});var ne=r(q);te=i(ne,`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),R=n(ne,"CODE",{});var Be=r(R);pe=i(Be,'model({"input_ids": input_ids})'),Be.forEach(t),ne.forEach(t),C.forEach(t)},m(k,M){b(k,d,M),e(d,g),b(k,c,M),b(k,u,M),e(u,_),e(_,l),e(u,f),e(u,B),e(B,he),b(k,V,M),b(k,E,M),e(E,G),e(E,S),e(S,X),e(E,ue),e(E,I),e(I,fe),e(E,le),b(k,N,M),b(k,P,M),e(P,Y),b(k,K,M),b(k,z,M),e(z,j),e(j,me),e(j,W),e(W,se),e(j,ge),e(j,Q),e(Q,de),e(z,ee),e(z,A),e(A,ce),e(A,L),e(L,oe),e(z,_e),e(z,q),e(q,te),e(q,R),e(R,pe)},d(k){k&&t(d),k&&t(c),k&&t(u),k&&t(V),k&&t(E),k&&t(N),k&&t(P),k&&t(K),k&&t(z)}}}function PF(x){let d,g,c,u,_;return{c(){d=o("p"),g=a("Although the recipe for forward pass needs to be defined within this function, one should call the "),c=o("code"),u=a("Module"),_=a(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),c=n(f,"CODE",{});var B=r(c);u=i(B,"Module"),B.forEach(t),_=i(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,g),e(d,c),e(c,u),e(d,_)},d(l){l&&t(d)}}}function AF(x){let d,g,c,u,_;return u=new ie({props:{code:`from transformers import DistilBertTokenizer, TFDistilBertForMultipleChoice
import tensorflow as tf

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = TFDistilBertForMultipleChoice.from_pretrained("distilbert-base-uncased")

prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
choice0 = "It is eaten with a fork and a knife."
choice1 = "It is eaten while held in the hand."

encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="tf", padding=True)
inputs = {k: tf.expand_dims(v, 0) for k, v in encoding.items()}
outputs = model(inputs)  # batch size is 1

# the linear classifier still needs to be trained
logits = outputs.logits`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> DistilBertTokenizer, TFDistilBertForMultipleChoice
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> tensorflow <span class="hljs-keyword">as</span> tf

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = DistilBertTokenizer.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFDistilBertForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors=<span class="hljs-string">&quot;tf&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = {k: tf.expand_dims(v, <span class="hljs-number">0</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()}
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(inputs)  <span class="hljs-comment"># batch size is 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the linear classifier still needs to be trained</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`}}),{c(){d=o("p"),g=a("Example:"),c=p(),T(u.$$.fragment)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Example:"),f.forEach(t),c=h(l),$(u.$$.fragment,l)},m(l,f){b(l,d,f),e(d,g),b(l,c,f),w(u,l,f),_=!0},p:ae,i(l){_||(y(u.$$.fragment,l),_=!0)},o(l){F(u.$$.fragment,l),_=!1},d(l){l&&t(d),l&&t(c),D(u,l)}}}function OF(x){let d,g,c,u,_,l,f,B,he,V,E,G,S,X,ue,I,fe,le,N,P,Y,K,z,j,me,W,se,ge,Q,de,ee,A,ce,L,oe,_e,q,te,R,pe;return{c(){d=o("p"),g=a("TF 2.0 models accepts two formats as inputs:"),c=p(),u=o("ul"),_=o("li"),l=a("having all inputs as keyword arguments (like PyTorch models), or"),f=p(),B=o("li"),he=a("having all inputs as a list, tuple or dict in the first positional arguments."),V=p(),E=o("p"),G=a("This second option is useful when using "),S=o("code"),X=a("tf.keras.Model.fit"),ue=a(` method which currently requires having all the
tensors in the first argument of the model call function: `),I=o("code"),fe=a("model(inputs)"),le=a("."),N=p(),P=o("p"),Y=a(`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),K=p(),z=o("ul"),j=o("li"),me=a("a single Tensor with "),W=o("code"),se=a("input_ids"),ge=a(" only and nothing else: "),Q=o("code"),de=a("model(inputs_ids)"),ee=p(),A=o("li"),ce=a(`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),L=o("code"),oe=a("model([input_ids, attention_mask])"),_e=p(),q=o("li"),te=a(`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),R=o("code"),pe=a('model({"input_ids": input_ids})')},l(k){d=n(k,"P",{});var M=r(d);g=i(M,"TF 2.0 models accepts two formats as inputs:"),M.forEach(t),c=h(k),u=n(k,"UL",{});var J=r(u);_=n(J,"LI",{});var ve=r(_);l=i(ve,"having all inputs as keyword arguments (like PyTorch models), or"),ve.forEach(t),f=h(J),B=n(J,"LI",{});var $e=r(B);he=i($e,"having all inputs as a list, tuple or dict in the first positional arguments."),$e.forEach(t),J.forEach(t),V=h(k),E=n(k,"P",{});var O=r(E);G=i(O,"This second option is useful when using "),S=n(O,"CODE",{});var ke=r(S);X=i(ke,"tf.keras.Model.fit"),ke.forEach(t),ue=i(O,` method which currently requires having all the
tensors in the first argument of the model call function: `),I=n(O,"CODE",{});var we=r(I);fe=i(we,"model(inputs)"),we.forEach(t),le=i(O,"."),O.forEach(t),N=h(k),P=n(k,"P",{});var ye=r(P);Y=i(ye,`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),ye.forEach(t),K=h(k),z=n(k,"UL",{});var C=r(z);j=n(C,"LI",{});var U=r(j);me=i(U,"a single Tensor with "),W=n(U,"CODE",{});var Fe=r(W);se=i(Fe,"input_ids"),Fe.forEach(t),ge=i(U," only and nothing else: "),Q=n(U,"CODE",{});var De=r(Q);de=i(De,"model(inputs_ids)"),De.forEach(t),U.forEach(t),ee=h(C),A=n(C,"LI",{});var Z=r(A);ce=i(Z,`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),L=n(Z,"CODE",{});var xe=r(L);oe=i(xe,"model([input_ids, attention_mask])"),xe.forEach(t),Z.forEach(t),_e=h(C),q=n(C,"LI",{});var ne=r(q);te=i(ne,`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),R=n(ne,"CODE",{});var Be=r(R);pe=i(Be,'model({"input_ids": input_ids})'),Be.forEach(t),ne.forEach(t),C.forEach(t)},m(k,M){b(k,d,M),e(d,g),b(k,c,M),b(k,u,M),e(u,_),e(_,l),e(u,f),e(u,B),e(B,he),b(k,V,M),b(k,E,M),e(E,G),e(E,S),e(S,X),e(E,ue),e(E,I),e(I,fe),e(E,le),b(k,N,M),b(k,P,M),e(P,Y),b(k,K,M),b(k,z,M),e(z,j),e(j,me),e(j,W),e(W,se),e(j,ge),e(j,Q),e(Q,de),e(z,ee),e(z,A),e(A,ce),e(A,L),e(L,oe),e(z,_e),e(z,q),e(q,te),e(q,R),e(R,pe)},d(k){k&&t(d),k&&t(c),k&&t(u),k&&t(V),k&&t(E),k&&t(N),k&&t(P),k&&t(K),k&&t(z)}}}function NF(x){let d,g,c,u,_;return{c(){d=o("p"),g=a("Although the recipe for forward pass needs to be defined within this function, one should call the "),c=o("code"),u=a("Module"),_=a(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),c=n(f,"CODE",{});var B=r(c);u=i(B,"Module"),B.forEach(t),_=i(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,g),e(d,c),e(c,u),e(d,_)},d(l){l&&t(d)}}}function LF(x){let d,g,c,u,_;return u=new ie({props:{code:`from transformers import DistilBertTokenizer, TFDistilBertForTokenClassification
import tensorflow as tf

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = TFDistilBertForTokenClassification.from_pretrained("distilbert-base-uncased")

inputs = tokenizer(
    "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="tf"
)

logits = model(**inputs).logits
predicted_token_class_ids = tf.math.argmax(logits, axis=-1)

# Note that tokens are classified rather then input words which means that
# there might be more predicted token classes than words.
# Multiple token classes might account for the same word
predicted_tokens_classes = [model.config.id2label[t] for t in predicted_token_class_ids[0].numpy().tolist()]
predicted_tokens_classes
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> DistilBertTokenizer, TFDistilBertForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> tensorflow <span class="hljs-keyword">as</span> tf

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = DistilBertTokenizer.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFDistilBertForTokenClassification.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)

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
`}}),{c(){d=o("p"),g=a("Example:"),c=p(),T(u.$$.fragment)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Example:"),f.forEach(t),c=h(l),$(u.$$.fragment,l)},m(l,f){b(l,d,f),e(d,g),b(l,c,f),w(u,l,f),_=!0},p:ae,i(l){_||(y(u.$$.fragment,l),_=!0)},o(l){F(u.$$.fragment,l),_=!1},d(l){l&&t(d),l&&t(c),D(u,l)}}}function SF(x){let d,g;return d=new ie({props:{code:`labels = predicted_token_class_ids
loss = tf.math.reduce_mean(model(**inputs, labels=labels).loss)
round(float(loss), 2)
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span>labels = predicted_token_class_ids
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = tf.math.reduce_mean(model(**inputs, labels=labels).loss)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(<span class="hljs-built_in">float</span>(loss), <span class="hljs-number">2</span>)
`}}),{c(){T(d.$$.fragment)},l(c){$(d.$$.fragment,c)},m(c,u){w(d,c,u),g=!0},p:ae,i(c){g||(y(d.$$.fragment,c),g=!0)},o(c){F(d.$$.fragment,c),g=!1},d(c){D(d,c)}}}function IF(x){let d,g,c,u,_,l,f,B,he,V,E,G,S,X,ue,I,fe,le,N,P,Y,K,z,j,me,W,se,ge,Q,de,ee,A,ce,L,oe,_e,q,te,R,pe;return{c(){d=o("p"),g=a("TF 2.0 models accepts two formats as inputs:"),c=p(),u=o("ul"),_=o("li"),l=a("having all inputs as keyword arguments (like PyTorch models), or"),f=p(),B=o("li"),he=a("having all inputs as a list, tuple or dict in the first positional arguments."),V=p(),E=o("p"),G=a("This second option is useful when using "),S=o("code"),X=a("tf.keras.Model.fit"),ue=a(` method which currently requires having all the
tensors in the first argument of the model call function: `),I=o("code"),fe=a("model(inputs)"),le=a("."),N=p(),P=o("p"),Y=a(`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),K=p(),z=o("ul"),j=o("li"),me=a("a single Tensor with "),W=o("code"),se=a("input_ids"),ge=a(" only and nothing else: "),Q=o("code"),de=a("model(inputs_ids)"),ee=p(),A=o("li"),ce=a(`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),L=o("code"),oe=a("model([input_ids, attention_mask])"),_e=p(),q=o("li"),te=a(`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),R=o("code"),pe=a('model({"input_ids": input_ids})')},l(k){d=n(k,"P",{});var M=r(d);g=i(M,"TF 2.0 models accepts two formats as inputs:"),M.forEach(t),c=h(k),u=n(k,"UL",{});var J=r(u);_=n(J,"LI",{});var ve=r(_);l=i(ve,"having all inputs as keyword arguments (like PyTorch models), or"),ve.forEach(t),f=h(J),B=n(J,"LI",{});var $e=r(B);he=i($e,"having all inputs as a list, tuple or dict in the first positional arguments."),$e.forEach(t),J.forEach(t),V=h(k),E=n(k,"P",{});var O=r(E);G=i(O,"This second option is useful when using "),S=n(O,"CODE",{});var ke=r(S);X=i(ke,"tf.keras.Model.fit"),ke.forEach(t),ue=i(O,` method which currently requires having all the
tensors in the first argument of the model call function: `),I=n(O,"CODE",{});var we=r(I);fe=i(we,"model(inputs)"),we.forEach(t),le=i(O,"."),O.forEach(t),N=h(k),P=n(k,"P",{});var ye=r(P);Y=i(ye,`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),ye.forEach(t),K=h(k),z=n(k,"UL",{});var C=r(z);j=n(C,"LI",{});var U=r(j);me=i(U,"a single Tensor with "),W=n(U,"CODE",{});var Fe=r(W);se=i(Fe,"input_ids"),Fe.forEach(t),ge=i(U," only and nothing else: "),Q=n(U,"CODE",{});var De=r(Q);de=i(De,"model(inputs_ids)"),De.forEach(t),U.forEach(t),ee=h(C),A=n(C,"LI",{});var Z=r(A);ce=i(Z,`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),L=n(Z,"CODE",{});var xe=r(L);oe=i(xe,"model([input_ids, attention_mask])"),xe.forEach(t),Z.forEach(t),_e=h(C),q=n(C,"LI",{});var ne=r(q);te=i(ne,`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),R=n(ne,"CODE",{});var Be=r(R);pe=i(Be,'model({"input_ids": input_ids})'),Be.forEach(t),ne.forEach(t),C.forEach(t)},m(k,M){b(k,d,M),e(d,g),b(k,c,M),b(k,u,M),e(u,_),e(_,l),e(u,f),e(u,B),e(B,he),b(k,V,M),b(k,E,M),e(E,G),e(E,S),e(S,X),e(E,ue),e(E,I),e(I,fe),e(E,le),b(k,N,M),b(k,P,M),e(P,Y),b(k,K,M),b(k,z,M),e(z,j),e(j,me),e(j,W),e(W,se),e(j,ge),e(j,Q),e(Q,de),e(z,ee),e(z,A),e(A,ce),e(A,L),e(L,oe),e(z,_e),e(z,q),e(q,te),e(q,R),e(R,pe)},d(k){k&&t(d),k&&t(c),k&&t(u),k&&t(V),k&&t(E),k&&t(N),k&&t(P),k&&t(K),k&&t(z)}}}function WF(x){let d,g,c,u,_;return{c(){d=o("p"),g=a("Although the recipe for forward pass needs to be defined within this function, one should call the "),c=o("code"),u=a("Module"),_=a(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),c=n(f,"CODE",{});var B=r(c);u=i(B,"Module"),B.forEach(t),_=i(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,g),e(d,c),e(c,u),e(d,_)},d(l){l&&t(d)}}}function QF(x){let d,g,c,u,_;return u=new ie({props:{code:`from transformers import DistilBertTokenizer, TFDistilBertForQuestionAnswering
import tensorflow as tf

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = TFDistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

inputs = tokenizer(question, text, return_tensors="tf")
outputs = model(**inputs)

answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
tokenizer.decode(predict_answer_tokens)
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> DistilBertTokenizer, TFDistilBertForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> tensorflow <span class="hljs-keyword">as</span> tf

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = DistilBertTokenizer.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFDistilBertForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>question, text = <span class="hljs-string">&quot;Who was Jim Henson?&quot;</span>, <span class="hljs-string">&quot;Jim Henson was a nice puppet&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(question, text, return_tensors=<span class="hljs-string">&quot;tf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>answer_start_index = <span class="hljs-built_in">int</span>(tf.math.argmax(outputs.start_logits, axis=-<span class="hljs-number">1</span>)[<span class="hljs-number">0</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>answer_end_index = <span class="hljs-built_in">int</span>(tf.math.argmax(outputs.end_logits, axis=-<span class="hljs-number">1</span>)[<span class="hljs-number">0</span>])

<span class="hljs-meta">&gt;&gt;&gt; </span>predict_answer_tokens = inputs.input_ids[<span class="hljs-number">0</span>, answer_start_index : answer_end_index + <span class="hljs-number">1</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.decode(predict_answer_tokens)
`}}),{c(){d=o("p"),g=a("Example:"),c=p(),T(u.$$.fragment)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Example:"),f.forEach(t),c=h(l),$(u.$$.fragment,l)},m(l,f){b(l,d,f),e(d,g),b(l,c,f),w(u,l,f),_=!0},p:ae,i(l){_||(y(u.$$.fragment,l),_=!0)},o(l){F(u.$$.fragment,l),_=!1},d(l){l&&t(d),l&&t(c),D(u,l)}}}function RF(x){let d,g;return d=new ie({props:{code:`# target is "nice puppet"
target_start_index = tf.constant([14])
target_end_index = tf.constant([15])

outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
loss = tf.math.reduce_mean(outputs.loss)
round(float(loss), 2)
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># target is &quot;nice puppet&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>target_start_index = tf.constant([<span class="hljs-number">14</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>target_end_index = tf.constant([<span class="hljs-number">15</span>])

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = tf.math.reduce_mean(outputs.loss)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(<span class="hljs-built_in">float</span>(loss), <span class="hljs-number">2</span>)
`}}),{c(){T(d.$$.fragment)},l(c){$(d.$$.fragment,c)},m(c,u){w(d,c,u),g=!0},p:ae,i(c){g||(y(d.$$.fragment,c),g=!0)},o(c){F(d.$$.fragment,c),g=!1},d(c){D(d,c)}}}function UF(x){let d,g,c,u,_;return{c(){d=o("p"),g=a("Although the recipe for forward pass needs to be defined within this function, one should call the "),c=o("code"),u=a("Module"),_=a(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),c=n(f,"CODE",{});var B=r(c);u=i(B,"Module"),B.forEach(t),_=i(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,g),e(d,c),e(c,u),e(d,_)},d(l){l&&t(d)}}}function HF(x){let d,g,c,u,_;return u=new ie({props:{code:`from transformers import DistilBertTokenizer, FlaxDistilBertModel

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = FlaxDistilBertModel.from_pretrained("distilbert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> DistilBertTokenizer, FlaxDistilBertModel

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = DistilBertTokenizer.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FlaxDistilBertModel.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;jax&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`}}),{c(){d=o("p"),g=a("Example:"),c=p(),T(u.$$.fragment)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Example:"),f.forEach(t),c=h(l),$(u.$$.fragment,l)},m(l,f){b(l,d,f),e(d,g),b(l,c,f),w(u,l,f),_=!0},p:ae,i(l){_||(y(u.$$.fragment,l),_=!0)},o(l){F(u.$$.fragment,l),_=!1},d(l){l&&t(d),l&&t(c),D(u,l)}}}function VF(x){let d,g,c,u,_;return{c(){d=o("p"),g=a("Although the recipe for forward pass needs to be defined within this function, one should call the "),c=o("code"),u=a("Module"),_=a(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),c=n(f,"CODE",{});var B=r(c);u=i(B,"Module"),B.forEach(t),_=i(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,g),e(d,c),e(c,u),e(d,_)},d(l){l&&t(d)}}}function KF(x){let d,g,c,u,_;return u=new ie({props:{code:`from transformers import DistilBertTokenizer, FlaxDistilBertForMaskedLM

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = FlaxDistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")

inputs = tokenizer("The capital of France is [MASK].", return_tensors="jax")

outputs = model(**inputs)
logits = outputs.logits`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> DistilBertTokenizer, FlaxDistilBertForMaskedLM

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = DistilBertTokenizer.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FlaxDistilBertForMaskedLM.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;The capital of France is [MASK].&quot;</span>, return_tensors=<span class="hljs-string">&quot;jax&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`}}),{c(){d=o("p"),g=a("Example:"),c=p(),T(u.$$.fragment)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Example:"),f.forEach(t),c=h(l),$(u.$$.fragment,l)},m(l,f){b(l,d,f),e(d,g),b(l,c,f),w(u,l,f),_=!0},p:ae,i(l){_||(y(u.$$.fragment,l),_=!0)},o(l){F(u.$$.fragment,l),_=!1},d(l){l&&t(d),l&&t(c),D(u,l)}}}function JF(x){let d,g,c,u,_;return{c(){d=o("p"),g=a("Although the recipe for forward pass needs to be defined within this function, one should call the "),c=o("code"),u=a("Module"),_=a(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),c=n(f,"CODE",{});var B=r(c);u=i(B,"Module"),B.forEach(t),_=i(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,g),e(d,c),e(c,u),e(d,_)},d(l){l&&t(d)}}}function GF(x){let d,g,c,u,_;return u=new ie({props:{code:`from transformers import DistilBertTokenizer, FlaxDistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = FlaxDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")

outputs = model(**inputs)
logits = outputs.logits`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> DistilBertTokenizer, FlaxDistilBertForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = DistilBertTokenizer.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FlaxDistilBertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;jax&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`}}),{c(){d=o("p"),g=a("Example:"),c=p(),T(u.$$.fragment)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Example:"),f.forEach(t),c=h(l),$(u.$$.fragment,l)},m(l,f){b(l,d,f),e(d,g),b(l,c,f),w(u,l,f),_=!0},p:ae,i(l){_||(y(u.$$.fragment,l),_=!0)},o(l){F(u.$$.fragment,l),_=!1},d(l){l&&t(d),l&&t(c),D(u,l)}}}function XF(x){let d,g,c,u,_;return{c(){d=o("p"),g=a("Although the recipe for forward pass needs to be defined within this function, one should call the "),c=o("code"),u=a("Module"),_=a(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),c=n(f,"CODE",{});var B=r(c);u=i(B,"Module"),B.forEach(t),_=i(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,g),e(d,c),e(c,u),e(d,_)},d(l){l&&t(d)}}}function YF(x){let d,g,c,u,_;return u=new ie({props:{code:`from transformers import DistilBertTokenizer, FlaxDistilBertForMultipleChoice

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = FlaxDistilBertForMultipleChoice.from_pretrained("distilbert-base-uncased")

prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
choice0 = "It is eaten with a fork and a knife."
choice1 = "It is eaten while held in the hand."

encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="jax", padding=True)
outputs = model(**{k: v[None, :] for k, v in encoding.items()})

logits = outputs.logits`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> DistilBertTokenizer, FlaxDistilBertForMultipleChoice

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = DistilBertTokenizer.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FlaxDistilBertForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors=<span class="hljs-string">&quot;jax&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**{k: v[<span class="hljs-literal">None</span>, :] <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()})

<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`}}),{c(){d=o("p"),g=a("Example:"),c=p(),T(u.$$.fragment)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Example:"),f.forEach(t),c=h(l),$(u.$$.fragment,l)},m(l,f){b(l,d,f),e(d,g),b(l,c,f),w(u,l,f),_=!0},p:ae,i(l){_||(y(u.$$.fragment,l),_=!0)},o(l){F(u.$$.fragment,l),_=!1},d(l){l&&t(d),l&&t(c),D(u,l)}}}function ZF(x){let d,g,c,u,_;return{c(){d=o("p"),g=a("Although the recipe for forward pass needs to be defined within this function, one should call the "),c=o("code"),u=a("Module"),_=a(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),c=n(f,"CODE",{});var B=r(c);u=i(B,"Module"),B.forEach(t),_=i(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,g),e(d,c),e(c,u),e(d,_)},d(l){l&&t(d)}}}function eD(x){let d,g,c,u,_;return u=new ie({props:{code:`from transformers import DistilBertTokenizer, FlaxDistilBertForTokenClassification

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = FlaxDistilBertForTokenClassification.from_pretrained("distilbert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")

outputs = model(**inputs)
logits = outputs.logits`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> DistilBertTokenizer, FlaxDistilBertForTokenClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = DistilBertTokenizer.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FlaxDistilBertForTokenClassification.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;jax&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`}}),{c(){d=o("p"),g=a("Example:"),c=p(),T(u.$$.fragment)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Example:"),f.forEach(t),c=h(l),$(u.$$.fragment,l)},m(l,f){b(l,d,f),e(d,g),b(l,c,f),w(u,l,f),_=!0},p:ae,i(l){_||(y(u.$$.fragment,l),_=!0)},o(l){F(u.$$.fragment,l),_=!1},d(l){l&&t(d),l&&t(c),D(u,l)}}}function tD(x){let d,g,c,u,_;return{c(){d=o("p"),g=a("Although the recipe for forward pass needs to be defined within this function, one should call the "),c=o("code"),u=a("Module"),_=a(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Although the recipe for forward pass needs to be defined within this function, one should call the "),c=n(f,"CODE",{});var B=r(c);u=i(B,"Module"),B.forEach(t),_=i(f,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),f.forEach(t)},m(l,f){b(l,d,f),e(d,g),e(d,c),e(c,u),e(d,_)},d(l){l&&t(d)}}}function sD(x){let d,g,c,u,_;return u=new ie({props:{code:`from transformers import DistilBertTokenizer, FlaxDistilBertForQuestionAnswering

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = FlaxDistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
inputs = tokenizer(question, text, return_tensors="jax")

outputs = model(**inputs)
start_scores = outputs.start_logits
end_scores = outputs.end_logits`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> DistilBertTokenizer, FlaxDistilBertForQuestionAnswering

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = DistilBertTokenizer.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FlaxDistilBertForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>question, text = <span class="hljs-string">&quot;Who was Jim Henson?&quot;</span>, <span class="hljs-string">&quot;Jim Henson was a nice puppet&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(question, text, return_tensors=<span class="hljs-string">&quot;jax&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>start_scores = outputs.start_logits
<span class="hljs-meta">&gt;&gt;&gt; </span>end_scores = outputs.end_logits`}}),{c(){d=o("p"),g=a("Example:"),c=p(),T(u.$$.fragment)},l(l){d=n(l,"P",{});var f=r(d);g=i(f,"Example:"),f.forEach(t),c=h(l),$(u.$$.fragment,l)},m(l,f){b(l,d,f),e(d,g),b(l,c,f),w(u,l,f),_=!0},p:ae,i(l){_||(y(u.$$.fragment,l),_=!0)},o(l){F(u.$$.fragment,l),_=!1},d(l){l&&t(d),l&&t(c),D(u,l)}}}function oD(x){let d,g,c,u,_,l,f,B,he,V,E,G,S,X,ue,I,fe,le,N,P,Y,K,z,j,me,W,se,ge,Q,de,ee,A,ce,L,oe,_e,q,te,R,pe,k,M,J,ve,$e,O,ke,we,ye,C,U,Fe,De,Z,xe,ne,Be,wu,Bp,pt,yu,jn,Fu,Du,Cn,xu,Bu,qn,Mu,Eu,Mp,es,Rs,Cl,Pn,zu,ql,ju,Ep,lt,An,Cu,Bt,qu,Di,Pu,Au,xi,Ou,Nu,On,Lu,Su,Iu,ts,Wu,Bi,Qu,Ru,Mi,Uu,Hu,Vu,Us,zp,ss,Hs,Pl,Nn,Ku,Al,Ju,jp,dt,Ln,Gu,Ol,Xu,Yu,Vs,Ei,Zu,ef,zi,tf,sf,of,Sn,nf,ji,rf,af,Cp,os,Ks,Nl,In,lf,Ll,df,qp,ct,Wn,cf,Qn,pf,Sl,hf,uf,ff,Js,Ci,mf,gf,qi,_f,bf,vf,Rn,kf,Pi,Tf,$f,Pp,ns,Gs,Il,Un,wf,Wl,yf,Ap,We,Hn,Ff,Ql,Df,xf,Vn,Bf,Ai,Mf,Ef,zf,Kn,jf,Jn,Cf,qf,Pf,ht,Gn,Af,rs,Of,Oi,Nf,Lf,Rl,Sf,If,Wf,Xs,Qf,Ys,Op,as,Zs,Ul,Xn,Rf,Hl,Uf,Np,Qe,Yn,Hf,Zn,Vf,Vl,Kf,Jf,Gf,er,Xf,Ni,Yf,Zf,em,tr,tm,sr,sm,om,nm,Ke,or,rm,is,am,Li,im,lm,Kl,dm,cm,pm,eo,hm,to,um,so,Lp,ls,oo,Jl,nr,fm,Gl,mm,Sp,Re,rr,gm,Xl,_m,bm,ar,vm,Si,km,Tm,$m,ir,wm,lr,ym,Fm,Dm,Pe,dr,xm,ds,Bm,Ii,Mm,Em,Yl,zm,jm,Cm,no,qm,ro,Pm,ao,Am,io,Om,lo,Ip,cs,co,Zl,cr,Nm,ed,Lm,Wp,Ue,pr,Sm,td,Im,Wm,hr,Qm,Wi,Rm,Um,Hm,ur,Vm,fr,Km,Jm,Gm,ut,mr,Xm,ps,Ym,Qi,Zm,eg,sd,tg,sg,og,po,ng,ho,Qp,hs,uo,od,gr,rg,nd,ag,Rp,He,_r,ig,rd,lg,dg,br,cg,Ri,pg,hg,ug,vr,fg,kr,mg,gg,_g,Je,Tr,bg,us,vg,Ui,kg,Tg,ad,$g,wg,yg,fo,Fg,mo,Dg,go,Up,fs,_o,id,$r,xg,ld,Bg,Hp,Ve,wr,Mg,ms,Eg,dd,zg,jg,cd,Cg,qg,Pg,yr,Ag,Hi,Og,Ng,Lg,Fr,Sg,Dr,Ig,Wg,Qg,Ge,xr,Rg,gs,Ug,Vi,Hg,Vg,pd,Kg,Jg,Gg,bo,Xg,vo,Yg,ko,Vp,_s,To,hd,Br,Zg,ud,e_,Kp,Ae,Mr,t_,fd,s_,o_,Er,n_,Ki,r_,a_,i_,zr,l_,jr,d_,c_,p_,$o,h_,ft,Cr,u_,bs,f_,Ji,m_,g_,md,__,b_,v_,wo,k_,yo,Jp,vs,Fo,gd,qr,T_,_d,$_,Gp,Oe,Pr,w_,Ar,y_,bd,F_,D_,x_,Or,B_,Gi,M_,E_,z_,Nr,j_,Lr,C_,q_,P_,Do,A_,Xe,Sr,O_,ks,N_,Xi,L_,S_,vd,I_,W_,Q_,xo,R_,Bo,U_,Mo,Xp,Ts,Eo,kd,Ir,H_,Td,V_,Yp,Ne,Wr,K_,$d,J_,G_,Qr,X_,Yi,Y_,Z_,eb,Rr,tb,Ur,sb,ob,nb,zo,rb,Ye,Hr,ab,$s,ib,Zi,lb,db,wd,cb,pb,hb,jo,ub,Co,fb,qo,Zp,ws,Po,yd,Vr,mb,Fd,gb,eh,Le,Kr,_b,Dd,bb,vb,Jr,kb,el,Tb,$b,wb,Gr,yb,Xr,Fb,Db,xb,Ao,Bb,mt,Yr,Mb,ys,Eb,tl,zb,jb,xd,Cb,qb,Pb,Oo,Ab,No,th,Fs,Lo,Bd,Zr,Ob,Md,Nb,sh,Se,ea,Lb,Ed,Sb,Ib,ta,Wb,sl,Qb,Rb,Ub,sa,Hb,oa,Vb,Kb,Jb,So,Gb,Ze,na,Xb,Ds,Yb,ol,Zb,ev,zd,tv,sv,ov,Io,nv,Wo,rv,Qo,oh,xs,Ro,jd,ra,av,Cd,iv,nh,Ie,aa,lv,Bs,dv,qd,cv,pv,Pd,hv,uv,fv,ia,mv,nl,gv,_v,bv,la,vv,da,kv,Tv,$v,Uo,wv,et,ca,yv,Ms,Fv,rl,Dv,xv,Ad,Bv,Mv,Ev,Ho,zv,Vo,jv,Ko,rh,Es,Jo,Od,pa,Cv,Nd,qv,ah,Me,ha,Pv,Ld,Av,Ov,ua,Nv,al,Lv,Sv,Iv,fa,Wv,ma,Qv,Rv,Uv,Sd,Hv,Vv,Mt,Id,ga,Kv,Jv,Wd,_a,Gv,Xv,Qd,ba,Yv,Zv,Rd,va,ek,tk,gt,ka,sk,zs,ok,Ud,nk,rk,Hd,ak,ik,lk,Go,dk,Xo,ih,js,Yo,Vd,Ta,ck,Kd,pk,lh,Ee,$a,hk,wa,uk,Jd,fk,mk,gk,ya,_k,il,bk,vk,kk,Fa,Tk,Da,$k,wk,yk,Gd,Fk,Dk,Et,Xd,xa,xk,Bk,Yd,Ba,Mk,Ek,Zd,Ma,zk,jk,ec,Ea,Ck,qk,_t,za,Pk,Cs,Ak,tc,Ok,Nk,sc,Lk,Sk,Ik,Zo,Wk,en,dh,qs,tn,oc,ja,Qk,nc,Rk,ch,ze,Ca,Uk,rc,Hk,Vk,qa,Kk,ll,Jk,Gk,Xk,Pa,Yk,Aa,Zk,eT,tT,ac,sT,oT,zt,ic,Oa,nT,rT,lc,Na,aT,iT,dc,La,lT,dT,cc,Sa,cT,pT,bt,Ia,hT,Ps,uT,pc,fT,mT,hc,gT,_T,bT,sn,vT,on,ph,As,nn,uc,Wa,kT,fc,TT,hh,je,Qa,$T,mc,wT,yT,Ra,FT,dl,DT,xT,BT,Ua,MT,Ha,ET,zT,jT,gc,CT,qT,jt,_c,Va,PT,AT,bc,Ka,OT,NT,vc,Ja,LT,ST,kc,Ga,IT,WT,vt,Xa,QT,Os,RT,Tc,UT,HT,$c,VT,KT,JT,rn,GT,an,uh,Ns,ln,wc,Ya,XT,yc,YT,fh,Ce,Za,ZT,Fc,e$,t$,ei,s$,cl,o$,n$,r$,ti,a$,si,i$,l$,d$,Dc,c$,p$,Ct,xc,oi,h$,u$,Bc,ni,f$,m$,Mc,ri,g$,_$,Ec,ai,b$,v$,kt,ii,k$,Ls,T$,zc,$$,w$,jc,y$,F$,D$,dn,x$,cn,mh,Ss,pn,Cc,li,B$,qc,M$,gh,qe,di,E$,Is,z$,Pc,j$,C$,Ac,q$,P$,A$,ci,O$,pl,N$,L$,S$,pi,I$,hi,W$,Q$,R$,Oc,U$,H$,qt,Nc,ui,V$,K$,Lc,fi,J$,G$,Sc,mi,X$,Y$,Ic,gi,Z$,ew,Tt,_i,tw,Ws,sw,Wc,ow,nw,Qc,rw,aw,iw,hn,lw,un,_h;return l=new Te({}),X=new Te({}),Pn=new Te({}),An=new H({props:{name:"class transformers.DistilBertConfig",anchor:"transformers.DistilBertConfig",parameters:[{name:"vocab_size",val:" = 30522"},{name:"max_position_embeddings",val:" = 512"},{name:"sinusoidal_pos_embds",val:" = False"},{name:"n_layers",val:" = 6"},{name:"n_heads",val:" = 12"},{name:"dim",val:" = 768"},{name:"hidden_dim",val:" = 3072"},{name:"dropout",val:" = 0.1"},{name:"attention_dropout",val:" = 0.1"},{name:"activation",val:" = 'gelu'"},{name:"initializer_range",val:" = 0.02"},{name:"qa_dropout",val:" = 0.1"},{name:"seq_classif_dropout",val:" = 0.2"},{name:"pad_token_id",val:" = 0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.DistilBertConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 30522) &#x2014;
Vocabulary size of the DistilBERT model. Defines the number of different tokens that can be represented by
the <code>inputs_ids</code> passed when calling <a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertModel">DistilBertModel</a> or <a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.TFDistilBertModel">TFDistilBertModel</a>.`,name:"vocab_size"},{anchor:"transformers.DistilBertConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.DistilBertConfig.sinusoidal_pos_embds",description:`<strong>sinusoidal_pos_embds</strong> (<code>boolean</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use sinusoidal positional embeddings.`,name:"sinusoidal_pos_embds"},{anchor:"transformers.DistilBertConfig.n_layers",description:`<strong>n_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 6) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"n_layers"},{anchor:"transformers.DistilBertConfig.n_heads",description:`<strong>n_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"n_heads"},{anchor:"transformers.DistilBertConfig.dim",description:`<strong>dim</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"dim"},{anchor:"transformers.DistilBertConfig.hidden_dim",description:`<strong>hidden_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
The size of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in the Transformer encoder.`,name:"hidden_dim"},{anchor:"transformers.DistilBertConfig.dropout",description:`<strong>dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"dropout"},{anchor:"transformers.DistilBertConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.DistilBertConfig.activation",description:`<strong>activation</strong> (<code>str</code> or <code>Callable</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"activation"},{anchor:"transformers.DistilBertConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.DistilBertConfig.qa_dropout",description:`<strong>qa_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probabilities used in the question answering model <a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertForQuestionAnswering">DistilBertForQuestionAnswering</a>.`,name:"qa_dropout"},{anchor:"transformers.DistilBertConfig.seq_classif_dropout",description:`<strong>seq_classif_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.2) &#x2014;
The dropout probabilities used in the sequence classification and the multiple choice model
<a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertForSequenceClassification">DistilBertForSequenceClassification</a>.`,name:"seq_classif_dropout"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/configuration_distilbert.py#L45"}}),Us=new re({props:{anchor:"transformers.DistilBertConfig.example",$$slots:{default:[nF]},$$scope:{ctx:x}}}),Nn=new Te({}),Ln=new H({props:{name:"class transformers.DistilBertTokenizer",anchor:"transformers.DistilBertTokenizer",parameters:[{name:"vocab_file",val:""},{name:"do_lower_case",val:" = True"},{name:"do_basic_tokenize",val:" = True"},{name:"never_split",val:" = None"},{name:"unk_token",val:" = '[UNK]'"},{name:"sep_token",val:" = '[SEP]'"},{name:"pad_token",val:" = '[PAD]'"},{name:"cls_token",val:" = '[CLS]'"},{name:"mask_token",val:" = '[MASK]'"},{name:"tokenize_chinese_chars",val:" = True"},{name:"strip_accents",val:" = None"},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/tokenization_distilbert.py#L62"}}),In=new Te({}),Wn=new H({props:{name:"class transformers.DistilBertTokenizerFast",anchor:"transformers.DistilBertTokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"do_lower_case",val:" = True"},{name:"unk_token",val:" = '[UNK]'"},{name:"sep_token",val:" = '[SEP]'"},{name:"pad_token",val:" = '[PAD]'"},{name:"cls_token",val:" = '[CLS]'"},{name:"mask_token",val:" = '[MASK]'"},{name:"tokenize_chinese_chars",val:" = True"},{name:"strip_accents",val:" = None"},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/tokenization_distilbert_fast.py#L79"}}),Un=new Te({}),Hn=new H({props:{name:"class transformers.DistilBertModel",anchor:"transformers.DistilBertModel",parameters:[{name:"config",val:": PretrainedConfig"}],parametersDescription:[{anchor:"transformers.DistilBertModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig">DistilBertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18727/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_distilbert.py#L459"}}),Gn=new H({props:{name:"forward",anchor:"transformers.DistilBertModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.DistilBertModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertTokenizer">DistilBertTokenizer</a>. See <a href="/docs/transformers/pr_18727/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18727/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.DistilBertModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.DistilBertModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.DistilBertModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.DistilBertModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.DistilBertModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.DistilBertModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18727/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_distilbert.py#L531",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput"
>transformers.modeling_outputs.BaseModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig"
>DistilBertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) \u2014 Sequence of hidden-states at the output of the last layer of the model.</p>
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
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput"
>transformers.modeling_outputs.BaseModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Xs=new be({props:{$$slots:{default:[rF]},$$scope:{ctx:x}}}),Ys=new re({props:{anchor:"transformers.DistilBertModel.forward.example",$$slots:{default:[aF]},$$scope:{ctx:x}}}),Xn=new Te({}),Yn=new H({props:{name:"class transformers.DistilBertForMaskedLM",anchor:"transformers.DistilBertForMaskedLM",parameters:[{name:"config",val:": PretrainedConfig"}],parametersDescription:[{anchor:"transformers.DistilBertForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig">DistilBertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18727/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_distilbert.py#L587"}}),or=new H({props:{name:"forward",anchor:"transformers.DistilBertForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.DistilBertForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertTokenizer">DistilBertTokenizer</a>. See <a href="/docs/transformers/pr_18727/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18727/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.DistilBertForMaskedLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.DistilBertForMaskedLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.DistilBertForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.DistilBertForMaskedLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.DistilBertForMaskedLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.DistilBertForMaskedLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18727/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.DistilBertForMaskedLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_distilbert.py#L629",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig"
>DistilBertConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),eo=new be({props:{$$slots:{default:[iF]},$$scope:{ctx:x}}}),to=new re({props:{anchor:"transformers.DistilBertForMaskedLM.forward.example",$$slots:{default:[lF]},$$scope:{ctx:x}}}),so=new re({props:{anchor:"transformers.DistilBertForMaskedLM.forward.example-2",$$slots:{default:[dF]},$$scope:{ctx:x}}}),nr=new Te({}),rr=new H({props:{name:"class transformers.DistilBertForSequenceClassification",anchor:"transformers.DistilBertForSequenceClassification",parameters:[{name:"config",val:": PretrainedConfig"}],parametersDescription:[{anchor:"transformers.DistilBertForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig">DistilBertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18727/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_distilbert.py#L693"}}),dr=new H({props:{name:"forward",anchor:"transformers.DistilBertForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.DistilBertForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertTokenizer">DistilBertTokenizer</a>. See <a href="/docs/transformers/pr_18727/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18727/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.DistilBertForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.DistilBertForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.DistilBertForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.DistilBertForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.DistilBertForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.DistilBertForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18727/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.DistilBertForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_distilbert.py#L727",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig"
>DistilBertConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),no=new be({props:{$$slots:{default:[cF]},$$scope:{ctx:x}}}),ro=new re({props:{anchor:"transformers.DistilBertForSequenceClassification.forward.example",$$slots:{default:[pF]},$$scope:{ctx:x}}}),ao=new re({props:{anchor:"transformers.DistilBertForSequenceClassification.forward.example-2",$$slots:{default:[hF]},$$scope:{ctx:x}}}),io=new re({props:{anchor:"transformers.DistilBertForSequenceClassification.forward.example-3",$$slots:{default:[uF]},$$scope:{ctx:x}}}),lo=new re({props:{anchor:"transformers.DistilBertForSequenceClassification.forward.example-4",$$slots:{default:[fF]},$$scope:{ctx:x}}}),cr=new Te({}),pr=new H({props:{name:"class transformers.DistilBertForMultipleChoice",anchor:"transformers.DistilBertForMultipleChoice",parameters:[{name:"config",val:": PretrainedConfig"}],parametersDescription:[{anchor:"transformers.DistilBertForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig">DistilBertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18727/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_distilbert.py#L1023"}}),mr=new H({props:{name:"forward",anchor:"transformers.DistilBertForMultipleChoice.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.DistilBertForMultipleChoice.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertTokenizer">DistilBertTokenizer</a>. See <a href="/docs/transformers/pr_18727/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18727/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.DistilBertForMultipleChoice.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.DistilBertForMultipleChoice.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.DistilBertForMultipleChoice.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.DistilBertForMultipleChoice.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.DistilBertForMultipleChoice.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.DistilBertForMultipleChoice.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18727/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.DistilBertForMultipleChoice.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices-1]</code> where <code>num_choices</code> is the size of the second dimension of the input tensors. (See
<code>input_ids</code> above)`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_distilbert.py#L1055",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig"
>DistilBertConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),po=new be({props:{$$slots:{default:[mF]},$$scope:{ctx:x}}}),ho=new re({props:{anchor:"transformers.DistilBertForMultipleChoice.forward.example",$$slots:{default:[gF]},$$scope:{ctx:x}}}),gr=new Te({}),_r=new H({props:{name:"class transformers.DistilBertForTokenClassification",anchor:"transformers.DistilBertForTokenClassification",parameters:[{name:"config",val:": PretrainedConfig"}],parametersDescription:[{anchor:"transformers.DistilBertForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig">DistilBertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18727/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_distilbert.py#L928"}}),Tr=new H({props:{name:"forward",anchor:"transformers.DistilBertForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.DistilBertForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>({0})</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertTokenizer">DistilBertTokenizer</a>. See <a href="/docs/transformers/pr_18727/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18727/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.DistilBertForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>({0})</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.DistilBertForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.DistilBertForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>({0}, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.DistilBertForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.DistilBertForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.DistilBertForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18727/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.DistilBertForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_distilbert.py#L960",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig"
>DistilBertConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),fo=new be({props:{$$slots:{default:[_F]},$$scope:{ctx:x}}}),mo=new re({props:{anchor:"transformers.DistilBertForTokenClassification.forward.example",$$slots:{default:[bF]},$$scope:{ctx:x}}}),go=new re({props:{anchor:"transformers.DistilBertForTokenClassification.forward.example-2",$$slots:{default:[vF]},$$scope:{ctx:x}}}),$r=new Te({}),wr=new H({props:{name:"class transformers.DistilBertForQuestionAnswering",anchor:"transformers.DistilBertForQuestionAnswering",parameters:[{name:"config",val:": PretrainedConfig"}],parametersDescription:[{anchor:"transformers.DistilBertForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig">DistilBertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18727/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_distilbert.py#L811"}}),xr=new H({props:{name:"forward",anchor:"transformers.DistilBertForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"start_positions",val:": typing.Optional[torch.Tensor] = None"},{name:"end_positions",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.DistilBertForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertTokenizer">DistilBertTokenizer</a>. See <a href="/docs/transformers/pr_18727/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18727/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.DistilBertForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.DistilBertForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.DistilBertForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.DistilBertForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.DistilBertForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.DistilBertForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18727/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.DistilBertForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.DistilBertForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_distilbert.py#L843",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig"
>DistilBertConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),bo=new be({props:{$$slots:{default:[kF]},$$scope:{ctx:x}}}),vo=new re({props:{anchor:"transformers.DistilBertForQuestionAnswering.forward.example",$$slots:{default:[TF]},$$scope:{ctx:x}}}),ko=new re({props:{anchor:"transformers.DistilBertForQuestionAnswering.forward.example-2",$$slots:{default:[$F]},$$scope:{ctx:x}}}),Br=new Te({}),Mr=new H({props:{name:"class transformers.TFDistilBertModel",anchor:"transformers.TFDistilBertModel",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFDistilBertModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig">DistilBertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18727/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_tf_distilbert.py#L522"}}),$o=new be({props:{$$slots:{default:[wF]},$$scope:{ctx:x}}}),Cr=new H({props:{name:"call",anchor:"transformers.TFDistilBertModel.call",parameters:[{name:"input_ids",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"head_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"inputs_embeds",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"training",val:": typing.Optional[bool] = False"}],parametersDescription:[{anchor:"transformers.TFDistilBertModel.call.input_ids",description:`<strong>input_ids</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertTokenizer">DistilBertTokenizer</a>. See <a href="/docs/transformers/pr_18727/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> and
<a href="/docs/transformers/pr_18727/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.TFDistilBertModel.call.attention_mask",description:`<strong>attention_mask</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.TFDistilBertModel.call.head_mask",description:`<strong>head_mask</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.TFDistilBertModel.call.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.TFDistilBertModel.call.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
config will be used instead.`,name:"output_attentions"},{anchor:"transformers.TFDistilBertModel.call.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
used instead.`,name:"output_hidden_states"},{anchor:"transformers.TFDistilBertModel.call.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18727/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple. This argument can be used in
eager mode, in graph mode the value will always be set to True.`,name:"return_dict"},{anchor:"transformers.TFDistilBertModel.call.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_tf_distilbert.py#L527",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutput"
>transformers.modeling_tf_outputs.TFBaseModelOutput</a> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<a
  href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig"
>DistilBertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) \u2014 Sequence of hidden-states at the output of the last layer of the model.</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(tf.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for the output of the embeddings + one for the output of each layer) of shape
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
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutput"
>transformers.modeling_tf_outputs.TFBaseModelOutput</a> or <code>tuple(tf.Tensor)</code></p>
`}}),wo=new be({props:{$$slots:{default:[yF]},$$scope:{ctx:x}}}),yo=new re({props:{anchor:"transformers.TFDistilBertModel.call.example",$$slots:{default:[FF]},$$scope:{ctx:x}}}),qr=new Te({}),Pr=new H({props:{name:"class transformers.TFDistilBertForMaskedLM",anchor:"transformers.TFDistilBertForMaskedLM",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFDistilBertForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig">DistilBertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18727/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_tf_distilbert.py#L609"}}),Do=new be({props:{$$slots:{default:[DF]},$$scope:{ctx:x}}}),Sr=new H({props:{name:"call",anchor:"transformers.TFDistilBertForMaskedLM.call",parameters:[{name:"input_ids",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"head_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"inputs_embeds",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"labels",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"training",val:": typing.Optional[bool] = False"}],parametersDescription:[{anchor:"transformers.TFDistilBertForMaskedLM.call.input_ids",description:`<strong>input_ids</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertTokenizer">DistilBertTokenizer</a>. See <a href="/docs/transformers/pr_18727/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> and
<a href="/docs/transformers/pr_18727/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.TFDistilBertForMaskedLM.call.attention_mask",description:`<strong>attention_mask</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.TFDistilBertForMaskedLM.call.head_mask",description:`<strong>head_mask</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.TFDistilBertForMaskedLM.call.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.TFDistilBertForMaskedLM.call.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
config will be used instead.`,name:"output_attentions"},{anchor:"transformers.TFDistilBertForMaskedLM.call.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
used instead.`,name:"output_hidden_states"},{anchor:"transformers.TFDistilBertForMaskedLM.call.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18727/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple. This argument can be used in
eager mode, in graph mode the value will always be set to True.`,name:"return_dict"},{anchor:"transformers.TFDistilBertForMaskedLM.call.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"},{anchor:"transformers.TFDistilBertForMaskedLM.call.labels",description:`<strong>labels</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_tf_distilbert.py#L629",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_tf_outputs.TFMaskedLMOutput"
>transformers.modeling_tf_outputs.TFMaskedLMOutput</a> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<a
  href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig"
>DistilBertConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_tf_outputs.TFMaskedLMOutput"
>transformers.modeling_tf_outputs.TFMaskedLMOutput</a> or <code>tuple(tf.Tensor)</code></p>
`}}),xo=new be({props:{$$slots:{default:[xF]},$$scope:{ctx:x}}}),Bo=new re({props:{anchor:"transformers.TFDistilBertForMaskedLM.call.example",$$slots:{default:[BF]},$$scope:{ctx:x}}}),Mo=new re({props:{anchor:"transformers.TFDistilBertForMaskedLM.call.example-2",$$slots:{default:[MF]},$$scope:{ctx:x}}}),Ir=new Te({}),Wr=new H({props:{name:"class transformers.TFDistilBertForSequenceClassification",anchor:"transformers.TFDistilBertForSequenceClassification",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFDistilBertForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig">DistilBertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18727/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_tf_distilbert.py#L699"}}),zo=new be({props:{$$slots:{default:[EF]},$$scope:{ctx:x}}}),Hr=new H({props:{name:"call",anchor:"transformers.TFDistilBertForSequenceClassification.call",parameters:[{name:"input_ids",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"head_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"inputs_embeds",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"labels",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"training",val:": typing.Optional[bool] = False"}],parametersDescription:[{anchor:"transformers.TFDistilBertForSequenceClassification.call.input_ids",description:`<strong>input_ids</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertTokenizer">DistilBertTokenizer</a>. See <a href="/docs/transformers/pr_18727/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> and
<a href="/docs/transformers/pr_18727/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.TFDistilBertForSequenceClassification.call.attention_mask",description:`<strong>attention_mask</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.TFDistilBertForSequenceClassification.call.head_mask",description:`<strong>head_mask</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.TFDistilBertForSequenceClassification.call.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.TFDistilBertForSequenceClassification.call.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
config will be used instead.`,name:"output_attentions"},{anchor:"transformers.TFDistilBertForSequenceClassification.call.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
used instead.`,name:"output_hidden_states"},{anchor:"transformers.TFDistilBertForSequenceClassification.call.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18727/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple. This argument can be used in
eager mode, in graph mode the value will always be set to True.`,name:"return_dict"},{anchor:"transformers.TFDistilBertForSequenceClassification.call.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"},{anchor:"transformers.TFDistilBertForSequenceClassification.call.labels",description:`<strong>labels</strong> (<code>tf.Tensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_tf_distilbert.py#L716",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_tf_outputs.TFSequenceClassifierOutput"
>transformers.modeling_tf_outputs.TFSequenceClassifierOutput</a> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<a
  href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig"
>DistilBertConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_tf_outputs.TFSequenceClassifierOutput"
>transformers.modeling_tf_outputs.TFSequenceClassifierOutput</a> or <code>tuple(tf.Tensor)</code></p>
`}}),jo=new be({props:{$$slots:{default:[zF]},$$scope:{ctx:x}}}),Co=new re({props:{anchor:"transformers.TFDistilBertForSequenceClassification.call.example",$$slots:{default:[jF]},$$scope:{ctx:x}}}),qo=new re({props:{anchor:"transformers.TFDistilBertForSequenceClassification.call.example-2",$$slots:{default:[CF]},$$scope:{ctx:x}}}),Vr=new Te({}),Kr=new H({props:{name:"class transformers.TFDistilBertForMultipleChoice",anchor:"transformers.TFDistilBertForMultipleChoice",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFDistilBertForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig">DistilBertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18727/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_tf_distilbert.py#L862"}}),Ao=new be({props:{$$slots:{default:[qF]},$$scope:{ctx:x}}}),Yr=new H({props:{name:"call",anchor:"transformers.TFDistilBertForMultipleChoice.call",parameters:[{name:"input_ids",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"head_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"inputs_embeds",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"labels",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"training",val:": typing.Optional[bool] = False"}],parametersDescription:[{anchor:"transformers.TFDistilBertForMultipleChoice.call.input_ids",description:`<strong>input_ids</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertTokenizer">DistilBertTokenizer</a>. See <a href="/docs/transformers/pr_18727/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> and
<a href="/docs/transformers/pr_18727/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.TFDistilBertForMultipleChoice.call.attention_mask",description:`<strong>attention_mask</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.TFDistilBertForMultipleChoice.call.head_mask",description:`<strong>head_mask</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.TFDistilBertForMultipleChoice.call.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, num_choices, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.TFDistilBertForMultipleChoice.call.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
config will be used instead.`,name:"output_attentions"},{anchor:"transformers.TFDistilBertForMultipleChoice.call.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
used instead.`,name:"output_hidden_states"},{anchor:"transformers.TFDistilBertForMultipleChoice.call.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18727/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple. This argument can be used in
eager mode, in graph mode the value will always be set to True.`,name:"return_dict"},{anchor:"transformers.TFDistilBertForMultipleChoice.call.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"},{anchor:"transformers.TFDistilBertForMultipleChoice.call.labels",description:`<strong>labels</strong> (<code>tf.Tensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices]</code>
where <code>num_choices</code> is the size of the second dimension of the input tensors. (See <code>input_ids</code> above)`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_tf_distilbert.py#L888",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_tf_outputs.TFMultipleChoiceModelOutput"
>transformers.modeling_tf_outputs.TFMultipleChoiceModelOutput</a> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<a
  href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig"
>DistilBertConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_tf_outputs.TFMultipleChoiceModelOutput"
>transformers.modeling_tf_outputs.TFMultipleChoiceModelOutput</a> or <code>tuple(tf.Tensor)</code></p>
`}}),Oo=new be({props:{$$slots:{default:[PF]},$$scope:{ctx:x}}}),No=new re({props:{anchor:"transformers.TFDistilBertForMultipleChoice.call.example",$$slots:{default:[AF]},$$scope:{ctx:x}}}),Zr=new Te({}),ea=new H({props:{name:"class transformers.TFDistilBertForTokenClassification",anchor:"transformers.TFDistilBertForTokenClassification",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFDistilBertForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig">DistilBertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18727/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_tf_distilbert.py#L786"}}),So=new be({props:{$$slots:{default:[OF]},$$scope:{ctx:x}}}),na=new H({props:{name:"call",anchor:"transformers.TFDistilBertForTokenClassification.call",parameters:[{name:"input_ids",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"head_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"inputs_embeds",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"labels",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"training",val:": typing.Optional[bool] = False"}],parametersDescription:[{anchor:"transformers.TFDistilBertForTokenClassification.call.input_ids",description:`<strong>input_ids</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertTokenizer">DistilBertTokenizer</a>. See <a href="/docs/transformers/pr_18727/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> and
<a href="/docs/transformers/pr_18727/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.TFDistilBertForTokenClassification.call.attention_mask",description:`<strong>attention_mask</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.TFDistilBertForTokenClassification.call.head_mask",description:`<strong>head_mask</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.TFDistilBertForTokenClassification.call.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.TFDistilBertForTokenClassification.call.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
config will be used instead.`,name:"output_attentions"},{anchor:"transformers.TFDistilBertForTokenClassification.call.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
used instead.`,name:"output_hidden_states"},{anchor:"transformers.TFDistilBertForTokenClassification.call.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18727/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple. This argument can be used in
eager mode, in graph mode the value will always be set to True.`,name:"return_dict"},{anchor:"transformers.TFDistilBertForTokenClassification.call.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"},{anchor:"transformers.TFDistilBertForTokenClassification.call.labels",description:`<strong>labels</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_tf_distilbert.py#L797",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_tf_outputs.TFTokenClassifierOutput"
>transformers.modeling_tf_outputs.TFTokenClassifierOutput</a> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<a
  href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig"
>DistilBertConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_tf_outputs.TFTokenClassifierOutput"
>transformers.modeling_tf_outputs.TFTokenClassifierOutput</a> or <code>tuple(tf.Tensor)</code></p>
`}}),Io=new be({props:{$$slots:{default:[NF]},$$scope:{ctx:x}}}),Wo=new re({props:{anchor:"transformers.TFDistilBertForTokenClassification.call.example",$$slots:{default:[LF]},$$scope:{ctx:x}}}),Qo=new re({props:{anchor:"transformers.TFDistilBertForTokenClassification.call.example-2",$$slots:{default:[SF]},$$scope:{ctx:x}}}),ra=new Te({}),aa=new H({props:{name:"class transformers.TFDistilBertForQuestionAnswering",anchor:"transformers.TFDistilBertForQuestionAnswering",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFDistilBertForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig">DistilBertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18727/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_tf_distilbert.py#L987"}}),Uo=new be({props:{$$slots:{default:[IF]},$$scope:{ctx:x}}}),ca=new H({props:{name:"call",anchor:"transformers.TFDistilBertForQuestionAnswering.call",parameters:[{name:"input_ids",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"head_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"inputs_embeds",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"start_positions",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"end_positions",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"training",val:": typing.Optional[bool] = False"}],parametersDescription:[{anchor:"transformers.TFDistilBertForQuestionAnswering.call.input_ids",description:`<strong>input_ids</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertTokenizer">DistilBertTokenizer</a>. See <a href="/docs/transformers/pr_18727/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> and
<a href="/docs/transformers/pr_18727/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.TFDistilBertForQuestionAnswering.call.attention_mask",description:`<strong>attention_mask</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.TFDistilBertForQuestionAnswering.call.head_mask",description:`<strong>head_mask</strong> (<code>Numpy array</code> or <code>tf.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.TFDistilBertForQuestionAnswering.call.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.TFDistilBertForQuestionAnswering.call.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
config will be used instead.`,name:"output_attentions"},{anchor:"transformers.TFDistilBertForQuestionAnswering.call.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
used instead.`,name:"output_hidden_states"},{anchor:"transformers.TFDistilBertForQuestionAnswering.call.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18727/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple. This argument can be used in
eager mode, in graph mode the value will always be set to True.`,name:"return_dict"},{anchor:"transformers.TFDistilBertForQuestionAnswering.call.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"},{anchor:"transformers.TFDistilBertForQuestionAnswering.call.start_positions",description:`<strong>start_positions</strong> (<code>tf.Tensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.TFDistilBertForQuestionAnswering.call.end_positions",description:`<strong>end_positions</strong> (<code>tf.Tensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_tf_distilbert.py#L998",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_tf_outputs.TFQuestionAnsweringModelOutput"
>transformers.modeling_tf_outputs.TFQuestionAnsweringModelOutput</a> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<a
  href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig"
>DistilBertConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_tf_outputs.TFQuestionAnsweringModelOutput"
>transformers.modeling_tf_outputs.TFQuestionAnsweringModelOutput</a> or <code>tuple(tf.Tensor)</code></p>
`}}),Ho=new be({props:{$$slots:{default:[WF]},$$scope:{ctx:x}}}),Vo=new re({props:{anchor:"transformers.TFDistilBertForQuestionAnswering.call.example",$$slots:{default:[QF]},$$scope:{ctx:x}}}),Ko=new re({props:{anchor:"transformers.TFDistilBertForQuestionAnswering.call.example-2",$$slots:{default:[RF]},$$scope:{ctx:x}}}),pa=new Te({}),ha=new H({props:{name:"class transformers.FlaxDistilBertModel",anchor:"transformers.FlaxDistilBertModel",parameters:[{name:"config",val:": DistilBertConfig"},{name:"input_shape",val:": typing.Tuple = (1, 1)"},{name:"seed",val:": int = 0"},{name:"dtype",val:": dtype = <class 'jax.numpy.float32'>"},{name:"_do_init",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.FlaxDistilBertModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig">DistilBertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18727/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_flax_distilbert.py#L535"}}),ka=new H({props:{name:"__call__",anchor:"transformers.FlaxDistilBertModel.__call__",parameters:[{name:"input_ids",val:""},{name:"attention_mask",val:" = None"},{name:"head_mask",val:" = None"},{name:"params",val:": dict = None"},{name:"dropout_rng",val:": PRNGKey = None"},{name:"train",val:": bool = False"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.FlaxDistilBertModel.__call__.input_ids",description:`<strong>input_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18727/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18727/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18727/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FlaxDistilBertModel.__call__.attention_mask",description:`<strong>attention_mask</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.FlaxDistilBertModel.__call__.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.FlaxDistilBertModel.__call__.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.FlaxDistilBertModel.__call__.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18727/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_flax_distilbert.py#L458"}}),Go=new be({props:{$$slots:{default:[UF]},$$scope:{ctx:x}}}),Xo=new re({props:{anchor:"transformers.FlaxDistilBertModel.__call__.example",$$slots:{default:[HF]},$$scope:{ctx:x}}}),Ta=new Te({}),$a=new H({props:{name:"class transformers.FlaxDistilBertForMaskedLM",anchor:"transformers.FlaxDistilBertForMaskedLM",parameters:[{name:"config",val:": DistilBertConfig"},{name:"input_shape",val:": typing.Tuple = (1, 1)"},{name:"seed",val:": int = 0"},{name:"dtype",val:": dtype = <class 'jax.numpy.float32'>"},{name:"_do_init",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.FlaxDistilBertForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig">DistilBertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18727/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_flax_distilbert.py#L608"}}),za=new H({props:{name:"__call__",anchor:"transformers.FlaxDistilBertForMaskedLM.__call__",parameters:[{name:"input_ids",val:""},{name:"attention_mask",val:" = None"},{name:"head_mask",val:" = None"},{name:"params",val:": dict = None"},{name:"dropout_rng",val:": PRNGKey = None"},{name:"train",val:": bool = False"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.FlaxDistilBertForMaskedLM.__call__.input_ids",description:`<strong>input_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18727/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18727/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18727/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FlaxDistilBertForMaskedLM.__call__.attention_mask",description:`<strong>attention_mask</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.FlaxDistilBertForMaskedLM.__call__.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.FlaxDistilBertForMaskedLM.__call__.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.FlaxDistilBertForMaskedLM.__call__.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18727/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_flax_distilbert.py#L458",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_flax_outputs.FlaxMaskedLMOutput"
>transformers.modeling_flax_outputs.FlaxMaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig"
>DistilBertConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_flax_outputs.FlaxMaskedLMOutput"
>transformers.modeling_flax_outputs.FlaxMaskedLMOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Zo=new be({props:{$$slots:{default:[VF]},$$scope:{ctx:x}}}),en=new re({props:{anchor:"transformers.FlaxDistilBertForMaskedLM.__call__.example",$$slots:{default:[KF]},$$scope:{ctx:x}}}),ja=new Te({}),Ca=new H({props:{name:"class transformers.FlaxDistilBertForSequenceClassification",anchor:"transformers.FlaxDistilBertForSequenceClassification",parameters:[{name:"config",val:": DistilBertConfig"},{name:"input_shape",val:": typing.Tuple = (1, 1)"},{name:"seed",val:": int = 0"},{name:"dtype",val:": dtype = <class 'jax.numpy.float32'>"},{name:"_do_init",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.FlaxDistilBertForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig">DistilBertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18727/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_flax_distilbert.py#L677"}}),Ia=new H({props:{name:"__call__",anchor:"transformers.FlaxDistilBertForSequenceClassification.__call__",parameters:[{name:"input_ids",val:""},{name:"attention_mask",val:" = None"},{name:"head_mask",val:" = None"},{name:"params",val:": dict = None"},{name:"dropout_rng",val:": PRNGKey = None"},{name:"train",val:": bool = False"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.FlaxDistilBertForSequenceClassification.__call__.input_ids",description:`<strong>input_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18727/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18727/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18727/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FlaxDistilBertForSequenceClassification.__call__.attention_mask",description:`<strong>attention_mask</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.FlaxDistilBertForSequenceClassification.__call__.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.FlaxDistilBertForSequenceClassification.__call__.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.FlaxDistilBertForSequenceClassification.__call__.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18727/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_flax_distilbert.py#L458",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSequenceClassifierOutput"
>transformers.modeling_flax_outputs.FlaxSequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig"
>DistilBertConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSequenceClassifierOutput"
>transformers.modeling_flax_outputs.FlaxSequenceClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),sn=new be({props:{$$slots:{default:[JF]},$$scope:{ctx:x}}}),on=new re({props:{anchor:"transformers.FlaxDistilBertForSequenceClassification.__call__.example",$$slots:{default:[GF]},$$scope:{ctx:x}}}),Wa=new Te({}),Qa=new H({props:{name:"class transformers.FlaxDistilBertForMultipleChoice",anchor:"transformers.FlaxDistilBertForMultipleChoice",parameters:[{name:"config",val:": DistilBertConfig"},{name:"input_shape",val:": typing.Tuple = (1, 1)"},{name:"seed",val:": int = 0"},{name:"dtype",val:": dtype = <class 'jax.numpy.float32'>"},{name:"_do_init",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.FlaxDistilBertForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig">DistilBertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18727/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_flax_distilbert.py#L757"}}),Xa=new H({props:{name:"__call__",anchor:"transformers.FlaxDistilBertForMultipleChoice.__call__",parameters:[{name:"input_ids",val:""},{name:"attention_mask",val:" = None"},{name:"head_mask",val:" = None"},{name:"params",val:": dict = None"},{name:"dropout_rng",val:": PRNGKey = None"},{name:"train",val:": bool = False"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.FlaxDistilBertForMultipleChoice.__call__.input_ids",description:`<strong>input_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18727/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18727/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18727/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FlaxDistilBertForMultipleChoice.__call__.attention_mask",description:`<strong>attention_mask</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.FlaxDistilBertForMultipleChoice.__call__.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.FlaxDistilBertForMultipleChoice.__call__.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.FlaxDistilBertForMultipleChoice.__call__.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18727/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_flax_distilbert.py#L458",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_flax_outputs.FlaxMultipleChoiceModelOutput"
>transformers.modeling_flax_outputs.FlaxMultipleChoiceModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig"
>DistilBertConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_flax_outputs.FlaxMultipleChoiceModelOutput"
>transformers.modeling_flax_outputs.FlaxMultipleChoiceModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),rn=new be({props:{$$slots:{default:[XF]},$$scope:{ctx:x}}}),an=new re({props:{anchor:"transformers.FlaxDistilBertForMultipleChoice.__call__.example",$$slots:{default:[YF]},$$scope:{ctx:x}}}),Ya=new Te({}),Za=new H({props:{name:"class transformers.FlaxDistilBertForTokenClassification",anchor:"transformers.FlaxDistilBertForTokenClassification",parameters:[{name:"config",val:": DistilBertConfig"},{name:"input_shape",val:": typing.Tuple = (1, 1)"},{name:"seed",val:": int = 0"},{name:"dtype",val:": dtype = <class 'jax.numpy.float32'>"},{name:"_do_init",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.FlaxDistilBertForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig">DistilBertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18727/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_flax_distilbert.py#L823"}}),ii=new H({props:{name:"__call__",anchor:"transformers.FlaxDistilBertForTokenClassification.__call__",parameters:[{name:"input_ids",val:""},{name:"attention_mask",val:" = None"},{name:"head_mask",val:" = None"},{name:"params",val:": dict = None"},{name:"dropout_rng",val:": PRNGKey = None"},{name:"train",val:": bool = False"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.FlaxDistilBertForTokenClassification.__call__.input_ids",description:`<strong>input_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18727/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18727/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18727/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FlaxDistilBertForTokenClassification.__call__.attention_mask",description:`<strong>attention_mask</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.FlaxDistilBertForTokenClassification.__call__.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.FlaxDistilBertForTokenClassification.__call__.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.FlaxDistilBertForTokenClassification.__call__.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18727/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_flax_distilbert.py#L458",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_flax_outputs.FlaxTokenClassifierOutput"
>transformers.modeling_flax_outputs.FlaxTokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig"
>DistilBertConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_flax_outputs.FlaxTokenClassifierOutput"
>transformers.modeling_flax_outputs.FlaxTokenClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),dn=new be({props:{$$slots:{default:[ZF]},$$scope:{ctx:x}}}),cn=new re({props:{anchor:"transformers.FlaxDistilBertForTokenClassification.__call__.example",$$slots:{default:[eD]},$$scope:{ctx:x}}}),li=new Te({}),di=new H({props:{name:"class transformers.FlaxDistilBertForQuestionAnswering",anchor:"transformers.FlaxDistilBertForQuestionAnswering",parameters:[{name:"config",val:": DistilBertConfig"},{name:"input_shape",val:": typing.Tuple = (1, 1)"},{name:"seed",val:": int = 0"},{name:"dtype",val:": dtype = <class 'jax.numpy.float32'>"},{name:"_do_init",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.FlaxDistilBertForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig">DistilBertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18727/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_flax_distilbert.py#L893"}}),_i=new H({props:{name:"__call__",anchor:"transformers.FlaxDistilBertForQuestionAnswering.__call__",parameters:[{name:"input_ids",val:""},{name:"attention_mask",val:" = None"},{name:"head_mask",val:" = None"},{name:"params",val:": dict = None"},{name:"dropout_rng",val:": PRNGKey = None"},{name:"train",val:": bool = False"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.FlaxDistilBertForQuestionAnswering.__call__.input_ids",description:`<strong>input_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18727/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18727/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18727/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FlaxDistilBertForQuestionAnswering.__call__.attention_mask",description:`<strong>attention_mask</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.FlaxDistilBertForQuestionAnswering.__call__.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.FlaxDistilBertForQuestionAnswering.__call__.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.FlaxDistilBertForQuestionAnswering.__call__.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18727/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18727/src/transformers/models/distilbert/modeling_flax_distilbert.py#L458",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_flax_outputs.FlaxQuestionAnsweringModelOutput"
>transformers.modeling_flax_outputs.FlaxQuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertConfig"
>DistilBertConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18727/en/main_classes/output#transformers.modeling_flax_outputs.FlaxQuestionAnsweringModelOutput"
>transformers.modeling_flax_outputs.FlaxQuestionAnsweringModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),hn=new be({props:{$$slots:{default:[tD]},$$scope:{ctx:x}}}),un=new re({props:{anchor:"transformers.FlaxDistilBertForQuestionAnswering.__call__.example",$$slots:{default:[sD]},$$scope:{ctx:x}}}),{c(){d=o("meta"),g=p(),c=o("h1"),u=o("a"),_=o("span"),T(l.$$.fragment),f=p(),B=o("span"),he=a("DistilBERT"),V=p(),E=o("h2"),G=o("a"),S=o("span"),T(X.$$.fragment),ue=p(),I=o("span"),fe=a("Overview"),le=p(),N=o("p"),P=a("The DistilBERT model was proposed in the blog post "),Y=o("a"),K=a(`Smaller, faster, cheaper, lighter: Introducing DistilBERT, a
distilled version of BERT`),z=a(", and the paper "),j=o("a"),me=a(`DistilBERT, a
distilled version of BERT: smaller, faster, cheaper and lighter`),W=a(`. DistilBERT is a
small, fast, cheap and light Transformer model trained by distilling BERT base. It has 40% less parameters than
`),se=o("em"),ge=a("bert-base-uncased"),Q=a(`, runs 60% faster while preserving over 95% of BERT\u2019s performances as measured on the GLUE language
understanding benchmark.`),de=p(),ee=o("p"),A=a("The abstract from the paper is the following:"),ce=p(),L=o("p"),oe=o("em"),_e=a(`As Transfer Learning from large-scale pre-trained models becomes more prevalent in Natural Language Processing (NLP),
operating these large models in on-the-edge and/or under constrained computational training or inference budgets
remains challenging. In this work, we propose a method to pre-train a smaller general-purpose language representation
model, called DistilBERT, which can then be fine-tuned with good performances on a wide range of tasks like its larger
counterparts. While most prior work investigated the use of distillation for building task-specific models, we leverage
knowledge distillation during the pretraining phase and show that it is possible to reduce the size of a BERT model by
40%, while retaining 97% of its language understanding capabilities and being 60% faster. To leverage the inductive
biases learned by larger models during pretraining, we introduce a triple loss combining language modeling,
distillation and cosine-distance losses. Our smaller, faster and lighter model is cheaper to pre-train and we
demonstrate its capabilities for on-device computations in a proof-of-concept experiment and a comparative on-device
study.`),q=p(),te=o("p"),R=a("Tips:"),pe=p(),k=o("ul"),M=o("li"),J=a("DistilBERT doesn\u2019t have "),ve=o("code"),$e=a("token_type_ids"),O=a(`, you don\u2019t need to indicate which token belongs to which segment. Just
separate your segments with the separation token `),ke=o("code"),we=a("tokenizer.sep_token"),ye=a(" (or "),C=o("code"),U=a("[SEP]"),Fe=a(")."),De=p(),Z=o("li"),xe=a("DistilBERT doesn\u2019t have options to select the input positions ("),ne=o("code"),Be=a("position_ids"),wu=a(` input). This could be added if
necessary though, just let us know if you need this option.`),Bp=p(),pt=o("p"),yu=a("This model was contributed by "),jn=o("a"),Fu=a("victorsanh"),Du=a(`. This model jax version was
contributed by `),Cn=o("a"),xu=a("kamalkraj"),Bu=a(". The original code can be found "),qn=o("a"),Mu=a("here"),Eu=a("."),Mp=p(),es=o("h2"),Rs=o("a"),Cl=o("span"),T(Pn.$$.fragment),zu=p(),ql=o("span"),ju=a("DistilBertConfig"),Ep=p(),lt=o("div"),T(An.$$.fragment),Cu=p(),Bt=o("p"),qu=a("This is the configuration class to store the configuration of a "),Di=o("a"),Pu=a("DistilBertModel"),Au=a(" or a "),xi=o("a"),Ou=a("TFDistilBertModel"),Nu=a(`. It
is used to instantiate a DistilBERT model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the DistilBERT
`),On=o("a"),Lu=a("distilbert-base-uncased"),Su=a(" architecture."),Iu=p(),ts=o("p"),Wu=a("Configuration objects inherit from "),Bi=o("a"),Qu=a("PretrainedConfig"),Ru=a(` and can be used to control the model outputs. Read the
documentation from `),Mi=o("a"),Uu=a("PretrainedConfig"),Hu=a(" for more information."),Vu=p(),T(Us.$$.fragment),zp=p(),ss=o("h2"),Hs=o("a"),Pl=o("span"),T(Nn.$$.fragment),Ku=p(),Al=o("span"),Ju=a("DistilBertTokenizer"),jp=p(),dt=o("div"),T(Ln.$$.fragment),Gu=p(),Ol=o("p"),Xu=a("Construct a DistilBERT tokenizer."),Yu=p(),Vs=o("p"),Ei=o("a"),Zu=a("DistilBertTokenizer"),ef=a(" is identical to "),zi=o("a"),tf=a("BertTokenizer"),sf=a(` and runs end-to-end tokenization: punctuation splitting
and wordpiece.`),of=p(),Sn=o("p"),nf=a("Refer to superclass "),ji=o("a"),rf=a("BertTokenizer"),af=a(" for usage examples and documentation concerning parameters."),Cp=p(),os=o("h2"),Ks=o("a"),Nl=o("span"),T(In.$$.fragment),lf=p(),Ll=o("span"),df=a("DistilBertTokenizerFast"),qp=p(),ct=o("div"),T(Wn.$$.fragment),cf=p(),Qn=o("p"),pf=a("Construct a \u201Cfast\u201D DistilBERT tokenizer (backed by HuggingFace\u2019s "),Sl=o("em"),hf=a("tokenizers"),uf=a(" library)."),ff=p(),Js=o("p"),Ci=o("a"),mf=a("DistilBertTokenizerFast"),gf=a(" is identical to "),qi=o("a"),_f=a("BertTokenizerFast"),bf=a(` and runs end-to-end tokenization: punctuation
splitting and wordpiece.`),vf=p(),Rn=o("p"),kf=a("Refer to superclass "),Pi=o("a"),Tf=a("BertTokenizerFast"),$f=a(" for usage examples and documentation concerning parameters."),Pp=p(),ns=o("h2"),Gs=o("a"),Il=o("span"),T(Un.$$.fragment),wf=p(),Wl=o("span"),yf=a("DistilBertModel"),Ap=p(),We=o("div"),T(Hn.$$.fragment),Ff=p(),Ql=o("p"),Df=a("The bare DistilBERT encoder/transformer outputting raw hidden-states without any specific head on top."),xf=p(),Vn=o("p"),Bf=a("This model inherits from "),Ai=o("a"),Mf=a("PreTrainedModel"),Ef=a(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),zf=p(),Kn=o("p"),jf=a("This model is also a PyTorch "),Jn=o("a"),Cf=a("torch.nn.Module"),qf=a(` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Pf=p(),ht=o("div"),T(Gn.$$.fragment),Af=p(),rs=o("p"),Of=a("The "),Oi=o("a"),Nf=a("DistilBertModel"),Lf=a(" forward method, overrides the "),Rl=o("code"),Sf=a("__call__"),If=a(" special method."),Wf=p(),T(Xs.$$.fragment),Qf=p(),T(Ys.$$.fragment),Op=p(),as=o("h2"),Zs=o("a"),Ul=o("span"),T(Xn.$$.fragment),Rf=p(),Hl=o("span"),Uf=a("DistilBertForMaskedLM"),Np=p(),Qe=o("div"),T(Yn.$$.fragment),Hf=p(),Zn=o("p"),Vf=a("DistilBert Model with a "),Vl=o("code"),Kf=a("masked language modeling"),Jf=a(" head on top."),Gf=p(),er=o("p"),Xf=a("This model inherits from "),Ni=o("a"),Yf=a("PreTrainedModel"),Zf=a(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),em=p(),tr=o("p"),tm=a("This model is also a PyTorch "),sr=o("a"),sm=a("torch.nn.Module"),om=a(` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),nm=p(),Ke=o("div"),T(or.$$.fragment),rm=p(),is=o("p"),am=a("The "),Li=o("a"),im=a("DistilBertForMaskedLM"),lm=a(" forward method, overrides the "),Kl=o("code"),dm=a("__call__"),cm=a(" special method."),pm=p(),T(eo.$$.fragment),hm=p(),T(to.$$.fragment),um=p(),T(so.$$.fragment),Lp=p(),ls=o("h2"),oo=o("a"),Jl=o("span"),T(nr.$$.fragment),fm=p(),Gl=o("span"),mm=a("DistilBertForSequenceClassification"),Sp=p(),Re=o("div"),T(rr.$$.fragment),gm=p(),Xl=o("p"),_m=a(`DistilBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the
pooled output) e.g. for GLUE tasks.`),bm=p(),ar=o("p"),vm=a("This model inherits from "),Si=o("a"),km=a("PreTrainedModel"),Tm=a(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),$m=p(),ir=o("p"),wm=a("This model is also a PyTorch "),lr=o("a"),ym=a("torch.nn.Module"),Fm=a(` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Dm=p(),Pe=o("div"),T(dr.$$.fragment),xm=p(),ds=o("p"),Bm=a("The "),Ii=o("a"),Mm=a("DistilBertForSequenceClassification"),Em=a(" forward method, overrides the "),Yl=o("code"),zm=a("__call__"),jm=a(" special method."),Cm=p(),T(no.$$.fragment),qm=p(),T(ro.$$.fragment),Pm=p(),T(ao.$$.fragment),Am=p(),T(io.$$.fragment),Om=p(),T(lo.$$.fragment),Ip=p(),cs=o("h2"),co=o("a"),Zl=o("span"),T(cr.$$.fragment),Nm=p(),ed=o("span"),Lm=a("DistilBertForMultipleChoice"),Wp=p(),Ue=o("div"),T(pr.$$.fragment),Sm=p(),td=o("p"),Im=a(`DistilBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
a softmax) e.g. for RocStories/SWAG tasks.`),Wm=p(),hr=o("p"),Qm=a("This model inherits from "),Wi=o("a"),Rm=a("PreTrainedModel"),Um=a(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Hm=p(),ur=o("p"),Vm=a("This model is also a PyTorch "),fr=o("a"),Km=a("torch.nn.Module"),Jm=a(` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Gm=p(),ut=o("div"),T(mr.$$.fragment),Xm=p(),ps=o("p"),Ym=a("The "),Qi=o("a"),Zm=a("DistilBertForMultipleChoice"),eg=a(" forward method, overrides the "),sd=o("code"),tg=a("__call__"),sg=a(" special method."),og=p(),T(po.$$.fragment),ng=p(),T(ho.$$.fragment),Qp=p(),hs=o("h2"),uo=o("a"),od=o("span"),T(gr.$$.fragment),rg=p(),nd=o("span"),ag=a("DistilBertForTokenClassification"),Rp=p(),He=o("div"),T(_r.$$.fragment),ig=p(),rd=o("p"),lg=a(`DistilBert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
for Named-Entity-Recognition (NER) tasks.`),dg=p(),br=o("p"),cg=a("This model inherits from "),Ri=o("a"),pg=a("PreTrainedModel"),hg=a(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),ug=p(),vr=o("p"),fg=a("This model is also a PyTorch "),kr=o("a"),mg=a("torch.nn.Module"),gg=a(` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),_g=p(),Je=o("div"),T(Tr.$$.fragment),bg=p(),us=o("p"),vg=a("The "),Ui=o("a"),kg=a("DistilBertForTokenClassification"),Tg=a(" forward method, overrides the "),ad=o("code"),$g=a("__call__"),wg=a(" special method."),yg=p(),T(fo.$$.fragment),Fg=p(),T(mo.$$.fragment),Dg=p(),T(go.$$.fragment),Up=p(),fs=o("h2"),_o=o("a"),id=o("span"),T($r.$$.fragment),xg=p(),ld=o("span"),Bg=a("DistilBertForQuestionAnswering"),Hp=p(),Ve=o("div"),T(wr.$$.fragment),Mg=p(),ms=o("p"),Eg=a(`DistilBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
linear layers on top of the hidden-states output to compute `),dd=o("code"),zg=a("span start logits"),jg=a(" and "),cd=o("code"),Cg=a("span end logits"),qg=a(")."),Pg=p(),yr=o("p"),Ag=a("This model inherits from "),Hi=o("a"),Og=a("PreTrainedModel"),Ng=a(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Lg=p(),Fr=o("p"),Sg=a("This model is also a PyTorch "),Dr=o("a"),Ig=a("torch.nn.Module"),Wg=a(` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Qg=p(),Ge=o("div"),T(xr.$$.fragment),Rg=p(),gs=o("p"),Ug=a("The "),Vi=o("a"),Hg=a("DistilBertForQuestionAnswering"),Vg=a(" forward method, overrides the "),pd=o("code"),Kg=a("__call__"),Jg=a(" special method."),Gg=p(),T(bo.$$.fragment),Xg=p(),T(vo.$$.fragment),Yg=p(),T(ko.$$.fragment),Vp=p(),_s=o("h2"),To=o("a"),hd=o("span"),T(Br.$$.fragment),Zg=p(),ud=o("span"),e_=a("TFDistilBertModel"),Kp=p(),Ae=o("div"),T(Mr.$$.fragment),t_=p(),fd=o("p"),s_=a("The bare DistilBERT encoder/transformer outputting raw hidden-states without any specific head on top."),o_=p(),Er=o("p"),n_=a("This model inherits from "),Ki=o("a"),r_=a("TFPreTrainedModel"),a_=a(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),i_=p(),zr=o("p"),l_=a("This model is also a "),jr=o("a"),d_=a("tf.keras.Model"),c_=a(` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),p_=p(),T($o.$$.fragment),h_=p(),ft=o("div"),T(Cr.$$.fragment),u_=p(),bs=o("p"),f_=a("The "),Ji=o("a"),m_=a("TFDistilBertModel"),g_=a(" forward method, overrides the "),md=o("code"),__=a("__call__"),b_=a(" special method."),v_=p(),T(wo.$$.fragment),k_=p(),T(yo.$$.fragment),Jp=p(),vs=o("h2"),Fo=o("a"),gd=o("span"),T(qr.$$.fragment),T_=p(),_d=o("span"),$_=a("TFDistilBertForMaskedLM"),Gp=p(),Oe=o("div"),T(Pr.$$.fragment),w_=p(),Ar=o("p"),y_=a("DistilBert Model with a "),bd=o("code"),F_=a("masked language modeling"),D_=a(" head on top."),x_=p(),Or=o("p"),B_=a("This model inherits from "),Gi=o("a"),M_=a("TFPreTrainedModel"),E_=a(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),z_=p(),Nr=o("p"),j_=a("This model is also a "),Lr=o("a"),C_=a("tf.keras.Model"),q_=a(` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),P_=p(),T(Do.$$.fragment),A_=p(),Xe=o("div"),T(Sr.$$.fragment),O_=p(),ks=o("p"),N_=a("The "),Xi=o("a"),L_=a("TFDistilBertForMaskedLM"),S_=a(" forward method, overrides the "),vd=o("code"),I_=a("__call__"),W_=a(" special method."),Q_=p(),T(xo.$$.fragment),R_=p(),T(Bo.$$.fragment),U_=p(),T(Mo.$$.fragment),Xp=p(),Ts=o("h2"),Eo=o("a"),kd=o("span"),T(Ir.$$.fragment),H_=p(),Td=o("span"),V_=a("TFDistilBertForSequenceClassification"),Yp=p(),Ne=o("div"),T(Wr.$$.fragment),K_=p(),$d=o("p"),J_=a(`DistilBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the
pooled output) e.g. for GLUE tasks.`),G_=p(),Qr=o("p"),X_=a("This model inherits from "),Yi=o("a"),Y_=a("TFPreTrainedModel"),Z_=a(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),eb=p(),Rr=o("p"),tb=a("This model is also a "),Ur=o("a"),sb=a("tf.keras.Model"),ob=a(` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),nb=p(),T(zo.$$.fragment),rb=p(),Ye=o("div"),T(Hr.$$.fragment),ab=p(),$s=o("p"),ib=a("The "),Zi=o("a"),lb=a("TFDistilBertForSequenceClassification"),db=a(" forward method, overrides the "),wd=o("code"),cb=a("__call__"),pb=a(" special method."),hb=p(),T(jo.$$.fragment),ub=p(),T(Co.$$.fragment),fb=p(),T(qo.$$.fragment),Zp=p(),ws=o("h2"),Po=o("a"),yd=o("span"),T(Vr.$$.fragment),mb=p(),Fd=o("span"),gb=a("TFDistilBertForMultipleChoice"),eh=p(),Le=o("div"),T(Kr.$$.fragment),_b=p(),Dd=o("p"),bb=a(`DistilBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
a softmax) e.g. for RocStories/SWAG tasks.`),vb=p(),Jr=o("p"),kb=a("This model inherits from "),el=o("a"),Tb=a("TFPreTrainedModel"),$b=a(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),wb=p(),Gr=o("p"),yb=a("This model is also a "),Xr=o("a"),Fb=a("tf.keras.Model"),Db=a(` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),xb=p(),T(Ao.$$.fragment),Bb=p(),mt=o("div"),T(Yr.$$.fragment),Mb=p(),ys=o("p"),Eb=a("The "),tl=o("a"),zb=a("TFDistilBertForMultipleChoice"),jb=a(" forward method, overrides the "),xd=o("code"),Cb=a("__call__"),qb=a(" special method."),Pb=p(),T(Oo.$$.fragment),Ab=p(),T(No.$$.fragment),th=p(),Fs=o("h2"),Lo=o("a"),Bd=o("span"),T(Zr.$$.fragment),Ob=p(),Md=o("span"),Nb=a("TFDistilBertForTokenClassification"),sh=p(),Se=o("div"),T(ea.$$.fragment),Lb=p(),Ed=o("p"),Sb=a(`DistilBert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
for Named-Entity-Recognition (NER) tasks.`),Ib=p(),ta=o("p"),Wb=a("This model inherits from "),sl=o("a"),Qb=a("TFPreTrainedModel"),Rb=a(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Ub=p(),sa=o("p"),Hb=a("This model is also a "),oa=o("a"),Vb=a("tf.keras.Model"),Kb=a(` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),Jb=p(),T(So.$$.fragment),Gb=p(),Ze=o("div"),T(na.$$.fragment),Xb=p(),Ds=o("p"),Yb=a("The "),ol=o("a"),Zb=a("TFDistilBertForTokenClassification"),ev=a(" forward method, overrides the "),zd=o("code"),tv=a("__call__"),sv=a(" special method."),ov=p(),T(Io.$$.fragment),nv=p(),T(Wo.$$.fragment),rv=p(),T(Qo.$$.fragment),oh=p(),xs=o("h2"),Ro=o("a"),jd=o("span"),T(ra.$$.fragment),av=p(),Cd=o("span"),iv=a("TFDistilBertForQuestionAnswering"),nh=p(),Ie=o("div"),T(aa.$$.fragment),lv=p(),Bs=o("p"),dv=a(`DistilBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
linear layer on top of the hidden-states output to compute `),qd=o("code"),cv=a("span start logits"),pv=a(" and "),Pd=o("code"),hv=a("span end logits"),uv=a(")."),fv=p(),ia=o("p"),mv=a("This model inherits from "),nl=o("a"),gv=a("TFPreTrainedModel"),_v=a(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),bv=p(),la=o("p"),vv=a("This model is also a "),da=o("a"),kv=a("tf.keras.Model"),Tv=a(` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),$v=p(),T(Uo.$$.fragment),wv=p(),et=o("div"),T(ca.$$.fragment),yv=p(),Ms=o("p"),Fv=a("The "),rl=o("a"),Dv=a("TFDistilBertForQuestionAnswering"),xv=a(" forward method, overrides the "),Ad=o("code"),Bv=a("__call__"),Mv=a(" special method."),Ev=p(),T(Ho.$$.fragment),zv=p(),T(Vo.$$.fragment),jv=p(),T(Ko.$$.fragment),rh=p(),Es=o("h2"),Jo=o("a"),Od=o("span"),T(pa.$$.fragment),Cv=p(),Nd=o("span"),qv=a("FlaxDistilBertModel"),ah=p(),Me=o("div"),T(ha.$$.fragment),Pv=p(),Ld=o("p"),Av=a("The bare DistilBert Model transformer outputting raw hidden-states without any specific head on top."),Ov=p(),ua=o("p"),Nv=a("This model inherits from "),al=o("a"),Lv=a("FlaxPreTrainedModel"),Sv=a(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading, saving and converting weights from PyTorch models)`),Iv=p(),fa=o("p"),Wv=a("This model is also a Flax Linen "),ma=o("a"),Qv=a("flax.linen.Module"),Rv=a(`
subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
general usage and behavior.`),Uv=p(),Sd=o("p"),Hv=a("Finally, this model supports inherent JAX features such as:"),Vv=p(),Mt=o("ul"),Id=o("li"),ga=o("a"),Kv=a("Just-In-Time (JIT) compilation"),Jv=p(),Wd=o("li"),_a=o("a"),Gv=a("Automatic Differentiation"),Xv=p(),Qd=o("li"),ba=o("a"),Yv=a("Vectorization"),Zv=p(),Rd=o("li"),va=o("a"),ek=a("Parallelization"),tk=p(),gt=o("div"),T(ka.$$.fragment),sk=p(),zs=o("p"),ok=a("The "),Ud=o("code"),nk=a("FlaxDistilBertPreTrainedModel"),rk=a(" forward method, overrides the "),Hd=o("code"),ak=a("__call__"),ik=a(" special method."),lk=p(),T(Go.$$.fragment),dk=p(),T(Xo.$$.fragment),ih=p(),js=o("h2"),Yo=o("a"),Vd=o("span"),T(Ta.$$.fragment),ck=p(),Kd=o("span"),pk=a("FlaxDistilBertForMaskedLM"),lh=p(),Ee=o("div"),T($a.$$.fragment),hk=p(),wa=o("p"),uk=a("DistilBert Model with a "),Jd=o("code"),fk=a("language modeling"),mk=a(" head on top."),gk=p(),ya=o("p"),_k=a("This model inherits from "),il=o("a"),bk=a("FlaxPreTrainedModel"),vk=a(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading, saving and converting weights from PyTorch models)`),kk=p(),Fa=o("p"),Tk=a("This model is also a Flax Linen "),Da=o("a"),$k=a("flax.linen.Module"),wk=a(`
subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
general usage and behavior.`),yk=p(),Gd=o("p"),Fk=a("Finally, this model supports inherent JAX features such as:"),Dk=p(),Et=o("ul"),Xd=o("li"),xa=o("a"),xk=a("Just-In-Time (JIT) compilation"),Bk=p(),Yd=o("li"),Ba=o("a"),Mk=a("Automatic Differentiation"),Ek=p(),Zd=o("li"),Ma=o("a"),zk=a("Vectorization"),jk=p(),ec=o("li"),Ea=o("a"),Ck=a("Parallelization"),qk=p(),_t=o("div"),T(za.$$.fragment),Pk=p(),Cs=o("p"),Ak=a("The "),tc=o("code"),Ok=a("FlaxDistilBertPreTrainedModel"),Nk=a(" forward method, overrides the "),sc=o("code"),Lk=a("__call__"),Sk=a(" special method."),Ik=p(),T(Zo.$$.fragment),Wk=p(),T(en.$$.fragment),dh=p(),qs=o("h2"),tn=o("a"),oc=o("span"),T(ja.$$.fragment),Qk=p(),nc=o("span"),Rk=a("FlaxDistilBertForSequenceClassification"),ch=p(),ze=o("div"),T(Ca.$$.fragment),Uk=p(),rc=o("p"),Hk=a(`DistilBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the
pooled output) e.g. for GLUE tasks.`),Vk=p(),qa=o("p"),Kk=a("This model inherits from "),ll=o("a"),Jk=a("FlaxPreTrainedModel"),Gk=a(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading, saving and converting weights from PyTorch models)`),Xk=p(),Pa=o("p"),Yk=a("This model is also a Flax Linen "),Aa=o("a"),Zk=a("flax.linen.Module"),eT=a(`
subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
general usage and behavior.`),tT=p(),ac=o("p"),sT=a("Finally, this model supports inherent JAX features such as:"),oT=p(),zt=o("ul"),ic=o("li"),Oa=o("a"),nT=a("Just-In-Time (JIT) compilation"),rT=p(),lc=o("li"),Na=o("a"),aT=a("Automatic Differentiation"),iT=p(),dc=o("li"),La=o("a"),lT=a("Vectorization"),dT=p(),cc=o("li"),Sa=o("a"),cT=a("Parallelization"),pT=p(),bt=o("div"),T(Ia.$$.fragment),hT=p(),Ps=o("p"),uT=a("The "),pc=o("code"),fT=a("FlaxDistilBertPreTrainedModel"),mT=a(" forward method, overrides the "),hc=o("code"),gT=a("__call__"),_T=a(" special method."),bT=p(),T(sn.$$.fragment),vT=p(),T(on.$$.fragment),ph=p(),As=o("h2"),nn=o("a"),uc=o("span"),T(Wa.$$.fragment),kT=p(),fc=o("span"),TT=a("FlaxDistilBertForMultipleChoice"),hh=p(),je=o("div"),T(Qa.$$.fragment),$T=p(),mc=o("p"),wT=a(`DistilBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
a softmax) e.g. for RocStories/SWAG tasks.`),yT=p(),Ra=o("p"),FT=a("This model inherits from "),dl=o("a"),DT=a("FlaxPreTrainedModel"),xT=a(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading, saving and converting weights from PyTorch models)`),BT=p(),Ua=o("p"),MT=a("This model is also a Flax Linen "),Ha=o("a"),ET=a("flax.linen.Module"),zT=a(`
subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
general usage and behavior.`),jT=p(),gc=o("p"),CT=a("Finally, this model supports inherent JAX features such as:"),qT=p(),jt=o("ul"),_c=o("li"),Va=o("a"),PT=a("Just-In-Time (JIT) compilation"),AT=p(),bc=o("li"),Ka=o("a"),OT=a("Automatic Differentiation"),NT=p(),vc=o("li"),Ja=o("a"),LT=a("Vectorization"),ST=p(),kc=o("li"),Ga=o("a"),IT=a("Parallelization"),WT=p(),vt=o("div"),T(Xa.$$.fragment),QT=p(),Os=o("p"),RT=a("The "),Tc=o("code"),UT=a("FlaxDistilBertPreTrainedModel"),HT=a(" forward method, overrides the "),$c=o("code"),VT=a("__call__"),KT=a(" special method."),JT=p(),T(rn.$$.fragment),GT=p(),T(an.$$.fragment),uh=p(),Ns=o("h2"),ln=o("a"),wc=o("span"),T(Ya.$$.fragment),XT=p(),yc=o("span"),YT=a("FlaxDistilBertForTokenClassification"),fh=p(),Ce=o("div"),T(Za.$$.fragment),ZT=p(),Fc=o("p"),e$=a(`DistilBert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
for Named-Entity-Recognition (NER) tasks.`),t$=p(),ei=o("p"),s$=a("This model inherits from "),cl=o("a"),o$=a("FlaxPreTrainedModel"),n$=a(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading, saving and converting weights from PyTorch models)`),r$=p(),ti=o("p"),a$=a("This model is also a Flax Linen "),si=o("a"),i$=a("flax.linen.Module"),l$=a(`
subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
general usage and behavior.`),d$=p(),Dc=o("p"),c$=a("Finally, this model supports inherent JAX features such as:"),p$=p(),Ct=o("ul"),xc=o("li"),oi=o("a"),h$=a("Just-In-Time (JIT) compilation"),u$=p(),Bc=o("li"),ni=o("a"),f$=a("Automatic Differentiation"),m$=p(),Mc=o("li"),ri=o("a"),g$=a("Vectorization"),_$=p(),Ec=o("li"),ai=o("a"),b$=a("Parallelization"),v$=p(),kt=o("div"),T(ii.$$.fragment),k$=p(),Ls=o("p"),T$=a("The "),zc=o("code"),$$=a("FlaxDistilBertPreTrainedModel"),w$=a(" forward method, overrides the "),jc=o("code"),y$=a("__call__"),F$=a(" special method."),D$=p(),T(dn.$$.fragment),x$=p(),T(cn.$$.fragment),mh=p(),Ss=o("h2"),pn=o("a"),Cc=o("span"),T(li.$$.fragment),B$=p(),qc=o("span"),M$=a("FlaxDistilBertForQuestionAnswering"),gh=p(),qe=o("div"),T(di.$$.fragment),E$=p(),Is=o("p"),z$=a(`DistilBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
linear layers on top of the hidden-states output to compute `),Pc=o("code"),j$=a("span start logits"),C$=a(" and "),Ac=o("code"),q$=a("span end logits"),P$=a(")."),A$=p(),ci=o("p"),O$=a("This model inherits from "),pl=o("a"),N$=a("FlaxPreTrainedModel"),L$=a(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading, saving and converting weights from PyTorch models)`),S$=p(),pi=o("p"),I$=a("This model is also a Flax Linen "),hi=o("a"),W$=a("flax.linen.Module"),Q$=a(`
subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
general usage and behavior.`),R$=p(),Oc=o("p"),U$=a("Finally, this model supports inherent JAX features such as:"),H$=p(),qt=o("ul"),Nc=o("li"),ui=o("a"),V$=a("Just-In-Time (JIT) compilation"),K$=p(),Lc=o("li"),fi=o("a"),J$=a("Automatic Differentiation"),G$=p(),Sc=o("li"),mi=o("a"),X$=a("Vectorization"),Y$=p(),Ic=o("li"),gi=o("a"),Z$=a("Parallelization"),ew=p(),Tt=o("div"),T(_i.$$.fragment),tw=p(),Ws=o("p"),sw=a("The "),Wc=o("code"),ow=a("FlaxDistilBertPreTrainedModel"),nw=a(" forward method, overrides the "),Qc=o("code"),rw=a("__call__"),aw=a(" special method."),iw=p(),T(hn.$$.fragment),lw=p(),T(un.$$.fragment),this.h()},l(s){const v=sF('[data-svelte="svelte-1phssyn"]',document.head);d=n(v,"META",{name:!0,content:!0}),v.forEach(t),g=h(s),c=n(s,"H1",{class:!0});var bi=r(c);u=n(bi,"A",{id:!0,class:!0,href:!0});var Rc=r(u);_=n(Rc,"SPAN",{});var Uc=r(_);$(l.$$.fragment,Uc),Uc.forEach(t),Rc.forEach(t),f=h(bi),B=n(bi,"SPAN",{});var Hc=r(B);he=i(Hc,"DistilBERT"),Hc.forEach(t),bi.forEach(t),V=h(s),E=n(s,"H2",{class:!0});var vi=r(E);G=n(vi,"A",{id:!0,class:!0,href:!0});var Vc=r(G);S=n(Vc,"SPAN",{});var Kc=r(S);$(X.$$.fragment,Kc),Kc.forEach(t),Vc.forEach(t),ue=h(vi),I=n(vi,"SPAN",{});var Jc=r(I);fe=i(Jc,"Overview"),Jc.forEach(t),vi.forEach(t),le=h(s),N=n(s,"P",{});var Pt=r(N);P=i(Pt,"The DistilBERT model was proposed in the blog post "),Y=n(Pt,"A",{href:!0,rel:!0});var Gc=r(Y);K=i(Gc,`Smaller, faster, cheaper, lighter: Introducing DistilBERT, a
distilled version of BERT`),Gc.forEach(t),z=i(Pt,", and the paper "),j=n(Pt,"A",{href:!0,rel:!0});var Xc=r(j);me=i(Xc,`DistilBERT, a
distilled version of BERT: smaller, faster, cheaper and lighter`),Xc.forEach(t),W=i(Pt,`. DistilBERT is a
small, fast, cheap and light Transformer model trained by distilling BERT base. It has 40% less parameters than
`),se=n(Pt,"EM",{});var Yc=r(se);ge=i(Yc,"bert-base-uncased"),Yc.forEach(t),Q=i(Pt,`, runs 60% faster while preserving over 95% of BERT\u2019s performances as measured on the GLUE language
understanding benchmark.`),Pt.forEach(t),de=h(s),ee=n(s,"P",{});var Zc=r(ee);A=i(Zc,"The abstract from the paper is the following:"),Zc.forEach(t),ce=h(s),L=n(s,"P",{});var ep=r(L);oe=n(ep,"EM",{});var tp=r(oe);_e=i(tp,`As Transfer Learning from large-scale pre-trained models becomes more prevalent in Natural Language Processing (NLP),
operating these large models in on-the-edge and/or under constrained computational training or inference budgets
remains challenging. In this work, we propose a method to pre-train a smaller general-purpose language representation
model, called DistilBERT, which can then be fine-tuned with good performances on a wide range of tasks like its larger
counterparts. While most prior work investigated the use of distillation for building task-specific models, we leverage
knowledge distillation during the pretraining phase and show that it is possible to reduce the size of a BERT model by
40%, while retaining 97% of its language understanding capabilities and being 60% faster. To leverage the inductive
biases learned by larger models during pretraining, we introduce a triple loss combining language modeling,
distillation and cosine-distance losses. Our smaller, faster and lighter model is cheaper to pre-train and we
demonstrate its capabilities for on-device computations in a proof-of-concept experiment and a comparative on-device
study.`),tp.forEach(t),ep.forEach(t),q=h(s),te=n(s,"P",{});var sp=r(te);R=i(sp,"Tips:"),sp.forEach(t),pe=h(s),k=n(s,"UL",{});var ki=r(k);M=n(ki,"LI",{});var At=r(M);J=i(At,"DistilBERT doesn\u2019t have "),ve=n(At,"CODE",{});var op=r(ve);$e=i(op,"token_type_ids"),op.forEach(t),O=i(At,`, you don\u2019t need to indicate which token belongs to which segment. Just
separate your segments with the separation token `),ke=n(At,"CODE",{});var np=r(ke);we=i(np,"tokenizer.sep_token"),np.forEach(t),ye=i(At," (or "),C=n(At,"CODE",{});var rp=r(C);U=i(rp,"[SEP]"),rp.forEach(t),Fe=i(At,")."),At.forEach(t),De=h(ki),Z=n(ki,"LI",{});var Ti=r(Z);xe=i(Ti,"DistilBERT doesn\u2019t have options to select the input positions ("),ne=n(Ti,"CODE",{});var ap=r(ne);Be=i(ap,"position_ids"),ap.forEach(t),wu=i(Ti,` input). This could be added if
necessary though, just let us know if you need this option.`),Ti.forEach(t),ki.forEach(t),Bp=h(s),pt=n(s,"P",{});var Ot=r(pt);yu=i(Ot,"This model was contributed by "),jn=n(Ot,"A",{href:!0,rel:!0});var ip=r(jn);Fu=i(ip,"victorsanh"),ip.forEach(t),Du=i(Ot,`. This model jax version was
contributed by `),Cn=n(Ot,"A",{href:!0,rel:!0});var lp=r(Cn);xu=i(lp,"kamalkraj"),lp.forEach(t),Bu=i(Ot,". The original code can be found "),qn=n(Ot,"A",{href:!0,rel:!0});var dp=r(qn);Mu=i(dp,"here"),dp.forEach(t),Eu=i(Ot,"."),Ot.forEach(t),Mp=h(s),es=n(s,"H2",{class:!0});var $i=r(es);Rs=n($i,"A",{id:!0,class:!0,href:!0});var cp=r(Rs);Cl=n(cp,"SPAN",{});var pp=r(Cl);$(Pn.$$.fragment,pp),pp.forEach(t),cp.forEach(t),zu=h($i),ql=n($i,"SPAN",{});var hp=r(ql);ju=i(hp,"DistilBertConfig"),hp.forEach(t),$i.forEach(t),Ep=h(s),lt=n(s,"DIV",{class:!0});var Nt=r(lt);$(An.$$.fragment,Nt),Cu=h(Nt),Bt=n(Nt,"P",{});var Lt=r(Bt);qu=i(Lt,"This is the configuration class to store the configuration of a "),Di=n(Lt,"A",{href:!0});var up=r(Di);Pu=i(up,"DistilBertModel"),up.forEach(t),Au=i(Lt," or a "),xi=n(Lt,"A",{href:!0});var fp=r(xi);Ou=i(fp,"TFDistilBertModel"),fp.forEach(t),Nu=i(Lt,`. It
is used to instantiate a DistilBERT model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the DistilBERT
`),On=n(Lt,"A",{href:!0,rel:!0});var mp=r(On);Lu=i(mp,"distilbert-base-uncased"),mp.forEach(t),Su=i(Lt," architecture."),Lt.forEach(t),Iu=h(Nt),ts=n(Nt,"P",{});var Qs=r(ts);Wu=i(Qs,"Configuration objects inherit from "),Bi=n(Qs,"A",{href:!0});var gp=r(Bi);Qu=i(gp,"PretrainedConfig"),gp.forEach(t),Ru=i(Qs,` and can be used to control the model outputs. Read the
documentation from `),Mi=n(Qs,"A",{href:!0});var _p=r(Mi);Uu=i(_p,"PretrainedConfig"),_p.forEach(t),Hu=i(Qs," for more information."),Qs.forEach(t),Vu=h(Nt),$(Us.$$.fragment,Nt),Nt.forEach(t),zp=h(s),ss=n(s,"H2",{class:!0});var wi=r(ss);Hs=n(wi,"A",{id:!0,class:!0,href:!0});var bp=r(Hs);Pl=n(bp,"SPAN",{});var vp=r(Pl);$(Nn.$$.fragment,vp),vp.forEach(t),bp.forEach(t),Ku=h(wi),Al=n(wi,"SPAN",{});var kp=r(Al);Ju=i(kp,"DistilBertTokenizer"),kp.forEach(t),wi.forEach(t),jp=h(s),dt=n(s,"DIV",{class:!0});var St=r(dt);$(Ln.$$.fragment,St),Gu=h(St),Ol=n(St,"P",{});var Tp=r(Ol);Xu=i(Tp,"Construct a DistilBERT tokenizer."),Tp.forEach(t),Yu=h(St),Vs=n(St,"P",{});var fn=r(Vs);Ei=n(fn,"A",{href:!0});var $p=r(Ei);Zu=i($p,"DistilBertTokenizer"),$p.forEach(t),ef=i(fn," is identical to "),zi=n(fn,"A",{href:!0});var wp=r(zi);tf=i(wp,"BertTokenizer"),wp.forEach(t),sf=i(fn,` and runs end-to-end tokenization: punctuation splitting
and wordpiece.`),fn.forEach(t),of=h(St),Sn=n(St,"P",{});var yi=r(Sn);nf=i(yi,"Refer to superclass "),ji=n(yi,"A",{href:!0});var yp=r(ji);rf=i(yp,"BertTokenizer"),yp.forEach(t),af=i(yi," for usage examples and documentation concerning parameters."),yi.forEach(t),St.forEach(t),Cp=h(s),os=n(s,"H2",{class:!0});var Fi=r(os);Ks=n(Fi,"A",{id:!0,class:!0,href:!0});var Fp=r(Ks);Nl=n(Fp,"SPAN",{});var Dp=r(Nl);$(In.$$.fragment,Dp),Dp.forEach(t),Fp.forEach(t),lf=h(Fi),Ll=n(Fi,"SPAN",{});var dw=r(Ll);df=i(dw,"DistilBertTokenizerFast"),dw.forEach(t),Fi.forEach(t),qp=h(s),ct=n(s,"DIV",{class:!0});var mn=r(ct);$(Wn.$$.fragment,mn),cf=h(mn),Qn=n(mn,"P",{});var bh=r(Qn);pf=i(bh,"Construct a \u201Cfast\u201D DistilBERT tokenizer (backed by HuggingFace\u2019s "),Sl=n(bh,"EM",{});var cw=r(Sl);hf=i(cw,"tokenizers"),cw.forEach(t),uf=i(bh," library)."),bh.forEach(t),ff=h(mn),Js=n(mn,"P",{});var xp=r(Js);Ci=n(xp,"A",{href:!0});var pw=r(Ci);mf=i(pw,"DistilBertTokenizerFast"),pw.forEach(t),gf=i(xp," is identical to "),qi=n(xp,"A",{href:!0});var hw=r(qi);_f=i(hw,"BertTokenizerFast"),hw.forEach(t),bf=i(xp,` and runs end-to-end tokenization: punctuation
splitting and wordpiece.`),xp.forEach(t),vf=h(mn),Rn=n(mn,"P",{});var vh=r(Rn);kf=i(vh,"Refer to superclass "),Pi=n(vh,"A",{href:!0});var uw=r(Pi);Tf=i(uw,"BertTokenizerFast"),uw.forEach(t),$f=i(vh," for usage examples and documentation concerning parameters."),vh.forEach(t),mn.forEach(t),Pp=h(s),ns=n(s,"H2",{class:!0});var kh=r(ns);Gs=n(kh,"A",{id:!0,class:!0,href:!0});var fw=r(Gs);Il=n(fw,"SPAN",{});var mw=r(Il);$(Un.$$.fragment,mw),mw.forEach(t),fw.forEach(t),wf=h(kh),Wl=n(kh,"SPAN",{});var gw=r(Wl);yf=i(gw,"DistilBertModel"),gw.forEach(t),kh.forEach(t),Ap=h(s),We=n(s,"DIV",{class:!0});var It=r(We);$(Hn.$$.fragment,It),Ff=h(It),Ql=n(It,"P",{});var _w=r(Ql);Df=i(_w,"The bare DistilBERT encoder/transformer outputting raw hidden-states without any specific head on top."),_w.forEach(t),xf=h(It),Vn=n(It,"P",{});var Th=r(Vn);Bf=i(Th,"This model inherits from "),Ai=n(Th,"A",{href:!0});var bw=r(Ai);Mf=i(bw,"PreTrainedModel"),bw.forEach(t),Ef=i(Th,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Th.forEach(t),zf=h(It),Kn=n(It,"P",{});var $h=r(Kn);jf=i($h,"This model is also a PyTorch "),Jn=n($h,"A",{href:!0,rel:!0});var vw=r(Jn);Cf=i(vw,"torch.nn.Module"),vw.forEach(t),qf=i($h,` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),$h.forEach(t),Pf=h(It),ht=n(It,"DIV",{class:!0});var gn=r(ht);$(Gn.$$.fragment,gn),Af=h(gn),rs=n(gn,"P",{});var hl=r(rs);Of=i(hl,"The "),Oi=n(hl,"A",{href:!0});var kw=r(Oi);Nf=i(kw,"DistilBertModel"),kw.forEach(t),Lf=i(hl," forward method, overrides the "),Rl=n(hl,"CODE",{});var Tw=r(Rl);Sf=i(Tw,"__call__"),Tw.forEach(t),If=i(hl," special method."),hl.forEach(t),Wf=h(gn),$(Xs.$$.fragment,gn),Qf=h(gn),$(Ys.$$.fragment,gn),gn.forEach(t),It.forEach(t),Op=h(s),as=n(s,"H2",{class:!0});var wh=r(as);Zs=n(wh,"A",{id:!0,class:!0,href:!0});var $w=r(Zs);Ul=n($w,"SPAN",{});var ww=r(Ul);$(Xn.$$.fragment,ww),ww.forEach(t),$w.forEach(t),Rf=h(wh),Hl=n(wh,"SPAN",{});var yw=r(Hl);Uf=i(yw,"DistilBertForMaskedLM"),yw.forEach(t),wh.forEach(t),Np=h(s),Qe=n(s,"DIV",{class:!0});var Wt=r(Qe);$(Yn.$$.fragment,Wt),Hf=h(Wt),Zn=n(Wt,"P",{});var yh=r(Zn);Vf=i(yh,"DistilBert Model with a "),Vl=n(yh,"CODE",{});var Fw=r(Vl);Kf=i(Fw,"masked language modeling"),Fw.forEach(t),Jf=i(yh," head on top."),yh.forEach(t),Gf=h(Wt),er=n(Wt,"P",{});var Fh=r(er);Xf=i(Fh,"This model inherits from "),Ni=n(Fh,"A",{href:!0});var Dw=r(Ni);Yf=i(Dw,"PreTrainedModel"),Dw.forEach(t),Zf=i(Fh,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Fh.forEach(t),em=h(Wt),tr=n(Wt,"P",{});var Dh=r(tr);tm=i(Dh,"This model is also a PyTorch "),sr=n(Dh,"A",{href:!0,rel:!0});var xw=r(sr);sm=i(xw,"torch.nn.Module"),xw.forEach(t),om=i(Dh,` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Dh.forEach(t),nm=h(Wt),Ke=n(Wt,"DIV",{class:!0});var Qt=r(Ke);$(or.$$.fragment,Qt),rm=h(Qt),is=n(Qt,"P",{});var ul=r(is);am=i(ul,"The "),Li=n(ul,"A",{href:!0});var Bw=r(Li);im=i(Bw,"DistilBertForMaskedLM"),Bw.forEach(t),lm=i(ul," forward method, overrides the "),Kl=n(ul,"CODE",{});var Mw=r(Kl);dm=i(Mw,"__call__"),Mw.forEach(t),cm=i(ul," special method."),ul.forEach(t),pm=h(Qt),$(eo.$$.fragment,Qt),hm=h(Qt),$(to.$$.fragment,Qt),um=h(Qt),$(so.$$.fragment,Qt),Qt.forEach(t),Wt.forEach(t),Lp=h(s),ls=n(s,"H2",{class:!0});var xh=r(ls);oo=n(xh,"A",{id:!0,class:!0,href:!0});var Ew=r(oo);Jl=n(Ew,"SPAN",{});var zw=r(Jl);$(nr.$$.fragment,zw),zw.forEach(t),Ew.forEach(t),fm=h(xh),Gl=n(xh,"SPAN",{});var jw=r(Gl);mm=i(jw,"DistilBertForSequenceClassification"),jw.forEach(t),xh.forEach(t),Sp=h(s),Re=n(s,"DIV",{class:!0});var Rt=r(Re);$(rr.$$.fragment,Rt),gm=h(Rt),Xl=n(Rt,"P",{});var Cw=r(Xl);_m=i(Cw,`DistilBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the
pooled output) e.g. for GLUE tasks.`),Cw.forEach(t),bm=h(Rt),ar=n(Rt,"P",{});var Bh=r(ar);vm=i(Bh,"This model inherits from "),Si=n(Bh,"A",{href:!0});var qw=r(Si);km=i(qw,"PreTrainedModel"),qw.forEach(t),Tm=i(Bh,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Bh.forEach(t),$m=h(Rt),ir=n(Rt,"P",{});var Mh=r(ir);wm=i(Mh,"This model is also a PyTorch "),lr=n(Mh,"A",{href:!0,rel:!0});var Pw=r(lr);ym=i(Pw,"torch.nn.Module"),Pw.forEach(t),Fm=i(Mh,` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Mh.forEach(t),Dm=h(Rt),Pe=n(Rt,"DIV",{class:!0});var tt=r(Pe);$(dr.$$.fragment,tt),xm=h(tt),ds=n(tt,"P",{});var fl=r(ds);Bm=i(fl,"The "),Ii=n(fl,"A",{href:!0});var Aw=r(Ii);Mm=i(Aw,"DistilBertForSequenceClassification"),Aw.forEach(t),Em=i(fl," forward method, overrides the "),Yl=n(fl,"CODE",{});var Ow=r(Yl);zm=i(Ow,"__call__"),Ow.forEach(t),jm=i(fl," special method."),fl.forEach(t),Cm=h(tt),$(no.$$.fragment,tt),qm=h(tt),$(ro.$$.fragment,tt),Pm=h(tt),$(ao.$$.fragment,tt),Am=h(tt),$(io.$$.fragment,tt),Om=h(tt),$(lo.$$.fragment,tt),tt.forEach(t),Rt.forEach(t),Ip=h(s),cs=n(s,"H2",{class:!0});var Eh=r(cs);co=n(Eh,"A",{id:!0,class:!0,href:!0});var Nw=r(co);Zl=n(Nw,"SPAN",{});var Lw=r(Zl);$(cr.$$.fragment,Lw),Lw.forEach(t),Nw.forEach(t),Nm=h(Eh),ed=n(Eh,"SPAN",{});var Sw=r(ed);Lm=i(Sw,"DistilBertForMultipleChoice"),Sw.forEach(t),Eh.forEach(t),Wp=h(s),Ue=n(s,"DIV",{class:!0});var Ut=r(Ue);$(pr.$$.fragment,Ut),Sm=h(Ut),td=n(Ut,"P",{});var Iw=r(td);Im=i(Iw,`DistilBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
a softmax) e.g. for RocStories/SWAG tasks.`),Iw.forEach(t),Wm=h(Ut),hr=n(Ut,"P",{});var zh=r(hr);Qm=i(zh,"This model inherits from "),Wi=n(zh,"A",{href:!0});var Ww=r(Wi);Rm=i(Ww,"PreTrainedModel"),Ww.forEach(t),Um=i(zh,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),zh.forEach(t),Hm=h(Ut),ur=n(Ut,"P",{});var jh=r(ur);Vm=i(jh,"This model is also a PyTorch "),fr=n(jh,"A",{href:!0,rel:!0});var Qw=r(fr);Km=i(Qw,"torch.nn.Module"),Qw.forEach(t),Jm=i(jh,` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),jh.forEach(t),Gm=h(Ut),ut=n(Ut,"DIV",{class:!0});var _n=r(ut);$(mr.$$.fragment,_n),Xm=h(_n),ps=n(_n,"P",{});var ml=r(ps);Ym=i(ml,"The "),Qi=n(ml,"A",{href:!0});var Rw=r(Qi);Zm=i(Rw,"DistilBertForMultipleChoice"),Rw.forEach(t),eg=i(ml," forward method, overrides the "),sd=n(ml,"CODE",{});var Uw=r(sd);tg=i(Uw,"__call__"),Uw.forEach(t),sg=i(ml," special method."),ml.forEach(t),og=h(_n),$(po.$$.fragment,_n),ng=h(_n),$(ho.$$.fragment,_n),_n.forEach(t),Ut.forEach(t),Qp=h(s),hs=n(s,"H2",{class:!0});var Ch=r(hs);uo=n(Ch,"A",{id:!0,class:!0,href:!0});var Hw=r(uo);od=n(Hw,"SPAN",{});var Vw=r(od);$(gr.$$.fragment,Vw),Vw.forEach(t),Hw.forEach(t),rg=h(Ch),nd=n(Ch,"SPAN",{});var Kw=r(nd);ag=i(Kw,"DistilBertForTokenClassification"),Kw.forEach(t),Ch.forEach(t),Rp=h(s),He=n(s,"DIV",{class:!0});var Ht=r(He);$(_r.$$.fragment,Ht),ig=h(Ht),rd=n(Ht,"P",{});var Jw=r(rd);lg=i(Jw,`DistilBert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
for Named-Entity-Recognition (NER) tasks.`),Jw.forEach(t),dg=h(Ht),br=n(Ht,"P",{});var qh=r(br);cg=i(qh,"This model inherits from "),Ri=n(qh,"A",{href:!0});var Gw=r(Ri);pg=i(Gw,"PreTrainedModel"),Gw.forEach(t),hg=i(qh,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),qh.forEach(t),ug=h(Ht),vr=n(Ht,"P",{});var Ph=r(vr);fg=i(Ph,"This model is also a PyTorch "),kr=n(Ph,"A",{href:!0,rel:!0});var Xw=r(kr);mg=i(Xw,"torch.nn.Module"),Xw.forEach(t),gg=i(Ph,` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Ph.forEach(t),_g=h(Ht),Je=n(Ht,"DIV",{class:!0});var Vt=r(Je);$(Tr.$$.fragment,Vt),bg=h(Vt),us=n(Vt,"P",{});var gl=r(us);vg=i(gl,"The "),Ui=n(gl,"A",{href:!0});var Yw=r(Ui);kg=i(Yw,"DistilBertForTokenClassification"),Yw.forEach(t),Tg=i(gl," forward method, overrides the "),ad=n(gl,"CODE",{});var Zw=r(ad);$g=i(Zw,"__call__"),Zw.forEach(t),wg=i(gl," special method."),gl.forEach(t),yg=h(Vt),$(fo.$$.fragment,Vt),Fg=h(Vt),$(mo.$$.fragment,Vt),Dg=h(Vt),$(go.$$.fragment,Vt),Vt.forEach(t),Ht.forEach(t),Up=h(s),fs=n(s,"H2",{class:!0});var Ah=r(fs);_o=n(Ah,"A",{id:!0,class:!0,href:!0});var e1=r(_o);id=n(e1,"SPAN",{});var t1=r(id);$($r.$$.fragment,t1),t1.forEach(t),e1.forEach(t),xg=h(Ah),ld=n(Ah,"SPAN",{});var s1=r(ld);Bg=i(s1,"DistilBertForQuestionAnswering"),s1.forEach(t),Ah.forEach(t),Hp=h(s),Ve=n(s,"DIV",{class:!0});var Kt=r(Ve);$(wr.$$.fragment,Kt),Mg=h(Kt),ms=n(Kt,"P",{});var _l=r(ms);Eg=i(_l,`DistilBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
linear layers on top of the hidden-states output to compute `),dd=n(_l,"CODE",{});var o1=r(dd);zg=i(o1,"span start logits"),o1.forEach(t),jg=i(_l," and "),cd=n(_l,"CODE",{});var n1=r(cd);Cg=i(n1,"span end logits"),n1.forEach(t),qg=i(_l,")."),_l.forEach(t),Pg=h(Kt),yr=n(Kt,"P",{});var Oh=r(yr);Ag=i(Oh,"This model inherits from "),Hi=n(Oh,"A",{href:!0});var r1=r(Hi);Og=i(r1,"PreTrainedModel"),r1.forEach(t),Ng=i(Oh,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Oh.forEach(t),Lg=h(Kt),Fr=n(Kt,"P",{});var Nh=r(Fr);Sg=i(Nh,"This model is also a PyTorch "),Dr=n(Nh,"A",{href:!0,rel:!0});var a1=r(Dr);Ig=i(a1,"torch.nn.Module"),a1.forEach(t),Wg=i(Nh,` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Nh.forEach(t),Qg=h(Kt),Ge=n(Kt,"DIV",{class:!0});var Jt=r(Ge);$(xr.$$.fragment,Jt),Rg=h(Jt),gs=n(Jt,"P",{});var bl=r(gs);Ug=i(bl,"The "),Vi=n(bl,"A",{href:!0});var i1=r(Vi);Hg=i(i1,"DistilBertForQuestionAnswering"),i1.forEach(t),Vg=i(bl," forward method, overrides the "),pd=n(bl,"CODE",{});var l1=r(pd);Kg=i(l1,"__call__"),l1.forEach(t),Jg=i(bl," special method."),bl.forEach(t),Gg=h(Jt),$(bo.$$.fragment,Jt),Xg=h(Jt),$(vo.$$.fragment,Jt),Yg=h(Jt),$(ko.$$.fragment,Jt),Jt.forEach(t),Kt.forEach(t),Vp=h(s),_s=n(s,"H2",{class:!0});var Lh=r(_s);To=n(Lh,"A",{id:!0,class:!0,href:!0});var d1=r(To);hd=n(d1,"SPAN",{});var c1=r(hd);$(Br.$$.fragment,c1),c1.forEach(t),d1.forEach(t),Zg=h(Lh),ud=n(Lh,"SPAN",{});var p1=r(ud);e_=i(p1,"TFDistilBertModel"),p1.forEach(t),Lh.forEach(t),Kp=h(s),Ae=n(s,"DIV",{class:!0});var $t=r(Ae);$(Mr.$$.fragment,$t),t_=h($t),fd=n($t,"P",{});var h1=r(fd);s_=i(h1,"The bare DistilBERT encoder/transformer outputting raw hidden-states without any specific head on top."),h1.forEach(t),o_=h($t),Er=n($t,"P",{});var Sh=r(Er);n_=i(Sh,"This model inherits from "),Ki=n(Sh,"A",{href:!0});var u1=r(Ki);r_=i(u1,"TFPreTrainedModel"),u1.forEach(t),a_=i(Sh,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Sh.forEach(t),i_=h($t),zr=n($t,"P",{});var Ih=r(zr);l_=i(Ih,"This model is also a "),jr=n(Ih,"A",{href:!0,rel:!0});var f1=r(jr);d_=i(f1,"tf.keras.Model"),f1.forEach(t),c_=i(Ih,` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),Ih.forEach(t),p_=h($t),$($o.$$.fragment,$t),h_=h($t),ft=n($t,"DIV",{class:!0});var bn=r(ft);$(Cr.$$.fragment,bn),u_=h(bn),bs=n(bn,"P",{});var vl=r(bs);f_=i(vl,"The "),Ji=n(vl,"A",{href:!0});var m1=r(Ji);m_=i(m1,"TFDistilBertModel"),m1.forEach(t),g_=i(vl," forward method, overrides the "),md=n(vl,"CODE",{});var g1=r(md);__=i(g1,"__call__"),g1.forEach(t),b_=i(vl," special method."),vl.forEach(t),v_=h(bn),$(wo.$$.fragment,bn),k_=h(bn),$(yo.$$.fragment,bn),bn.forEach(t),$t.forEach(t),Jp=h(s),vs=n(s,"H2",{class:!0});var Wh=r(vs);Fo=n(Wh,"A",{id:!0,class:!0,href:!0});var _1=r(Fo);gd=n(_1,"SPAN",{});var b1=r(gd);$(qr.$$.fragment,b1),b1.forEach(t),_1.forEach(t),T_=h(Wh),_d=n(Wh,"SPAN",{});var v1=r(_d);$_=i(v1,"TFDistilBertForMaskedLM"),v1.forEach(t),Wh.forEach(t),Gp=h(s),Oe=n(s,"DIV",{class:!0});var wt=r(Oe);$(Pr.$$.fragment,wt),w_=h(wt),Ar=n(wt,"P",{});var Qh=r(Ar);y_=i(Qh,"DistilBert Model with a "),bd=n(Qh,"CODE",{});var k1=r(bd);F_=i(k1,"masked language modeling"),k1.forEach(t),D_=i(Qh," head on top."),Qh.forEach(t),x_=h(wt),Or=n(wt,"P",{});var Rh=r(Or);B_=i(Rh,"This model inherits from "),Gi=n(Rh,"A",{href:!0});var T1=r(Gi);M_=i(T1,"TFPreTrainedModel"),T1.forEach(t),E_=i(Rh,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Rh.forEach(t),z_=h(wt),Nr=n(wt,"P",{});var Uh=r(Nr);j_=i(Uh,"This model is also a "),Lr=n(Uh,"A",{href:!0,rel:!0});var $1=r(Lr);C_=i($1,"tf.keras.Model"),$1.forEach(t),q_=i(Uh,` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),Uh.forEach(t),P_=h(wt),$(Do.$$.fragment,wt),A_=h(wt),Xe=n(wt,"DIV",{class:!0});var Gt=r(Xe);$(Sr.$$.fragment,Gt),O_=h(Gt),ks=n(Gt,"P",{});var kl=r(ks);N_=i(kl,"The "),Xi=n(kl,"A",{href:!0});var w1=r(Xi);L_=i(w1,"TFDistilBertForMaskedLM"),w1.forEach(t),S_=i(kl," forward method, overrides the "),vd=n(kl,"CODE",{});var y1=r(vd);I_=i(y1,"__call__"),y1.forEach(t),W_=i(kl," special method."),kl.forEach(t),Q_=h(Gt),$(xo.$$.fragment,Gt),R_=h(Gt),$(Bo.$$.fragment,Gt),U_=h(Gt),$(Mo.$$.fragment,Gt),Gt.forEach(t),wt.forEach(t),Xp=h(s),Ts=n(s,"H2",{class:!0});var Hh=r(Ts);Eo=n(Hh,"A",{id:!0,class:!0,href:!0});var F1=r(Eo);kd=n(F1,"SPAN",{});var D1=r(kd);$(Ir.$$.fragment,D1),D1.forEach(t),F1.forEach(t),H_=h(Hh),Td=n(Hh,"SPAN",{});var x1=r(Td);V_=i(x1,"TFDistilBertForSequenceClassification"),x1.forEach(t),Hh.forEach(t),Yp=h(s),Ne=n(s,"DIV",{class:!0});var yt=r(Ne);$(Wr.$$.fragment,yt),K_=h(yt),$d=n(yt,"P",{});var B1=r($d);J_=i(B1,`DistilBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the
pooled output) e.g. for GLUE tasks.`),B1.forEach(t),G_=h(yt),Qr=n(yt,"P",{});var Vh=r(Qr);X_=i(Vh,"This model inherits from "),Yi=n(Vh,"A",{href:!0});var M1=r(Yi);Y_=i(M1,"TFPreTrainedModel"),M1.forEach(t),Z_=i(Vh,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Vh.forEach(t),eb=h(yt),Rr=n(yt,"P",{});var Kh=r(Rr);tb=i(Kh,"This model is also a "),Ur=n(Kh,"A",{href:!0,rel:!0});var E1=r(Ur);sb=i(E1,"tf.keras.Model"),E1.forEach(t),ob=i(Kh,` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),Kh.forEach(t),nb=h(yt),$(zo.$$.fragment,yt),rb=h(yt),Ye=n(yt,"DIV",{class:!0});var Xt=r(Ye);$(Hr.$$.fragment,Xt),ab=h(Xt),$s=n(Xt,"P",{});var Tl=r($s);ib=i(Tl,"The "),Zi=n(Tl,"A",{href:!0});var z1=r(Zi);lb=i(z1,"TFDistilBertForSequenceClassification"),z1.forEach(t),db=i(Tl," forward method, overrides the "),wd=n(Tl,"CODE",{});var j1=r(wd);cb=i(j1,"__call__"),j1.forEach(t),pb=i(Tl," special method."),Tl.forEach(t),hb=h(Xt),$(jo.$$.fragment,Xt),ub=h(Xt),$(Co.$$.fragment,Xt),fb=h(Xt),$(qo.$$.fragment,Xt),Xt.forEach(t),yt.forEach(t),Zp=h(s),ws=n(s,"H2",{class:!0});var Jh=r(ws);Po=n(Jh,"A",{id:!0,class:!0,href:!0});var C1=r(Po);yd=n(C1,"SPAN",{});var q1=r(yd);$(Vr.$$.fragment,q1),q1.forEach(t),C1.forEach(t),mb=h(Jh),Fd=n(Jh,"SPAN",{});var P1=r(Fd);gb=i(P1,"TFDistilBertForMultipleChoice"),P1.forEach(t),Jh.forEach(t),eh=h(s),Le=n(s,"DIV",{class:!0});var Ft=r(Le);$(Kr.$$.fragment,Ft),_b=h(Ft),Dd=n(Ft,"P",{});var A1=r(Dd);bb=i(A1,`DistilBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
a softmax) e.g. for RocStories/SWAG tasks.`),A1.forEach(t),vb=h(Ft),Jr=n(Ft,"P",{});var Gh=r(Jr);kb=i(Gh,"This model inherits from "),el=n(Gh,"A",{href:!0});var O1=r(el);Tb=i(O1,"TFPreTrainedModel"),O1.forEach(t),$b=i(Gh,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Gh.forEach(t),wb=h(Ft),Gr=n(Ft,"P",{});var Xh=r(Gr);yb=i(Xh,"This model is also a "),Xr=n(Xh,"A",{href:!0,rel:!0});var N1=r(Xr);Fb=i(N1,"tf.keras.Model"),N1.forEach(t),Db=i(Xh,` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),Xh.forEach(t),xb=h(Ft),$(Ao.$$.fragment,Ft),Bb=h(Ft),mt=n(Ft,"DIV",{class:!0});var vn=r(mt);$(Yr.$$.fragment,vn),Mb=h(vn),ys=n(vn,"P",{});var $l=r(ys);Eb=i($l,"The "),tl=n($l,"A",{href:!0});var L1=r(tl);zb=i(L1,"TFDistilBertForMultipleChoice"),L1.forEach(t),jb=i($l," forward method, overrides the "),xd=n($l,"CODE",{});var S1=r(xd);Cb=i(S1,"__call__"),S1.forEach(t),qb=i($l," special method."),$l.forEach(t),Pb=h(vn),$(Oo.$$.fragment,vn),Ab=h(vn),$(No.$$.fragment,vn),vn.forEach(t),Ft.forEach(t),th=h(s),Fs=n(s,"H2",{class:!0});var Yh=r(Fs);Lo=n(Yh,"A",{id:!0,class:!0,href:!0});var I1=r(Lo);Bd=n(I1,"SPAN",{});var W1=r(Bd);$(Zr.$$.fragment,W1),W1.forEach(t),I1.forEach(t),Ob=h(Yh),Md=n(Yh,"SPAN",{});var Q1=r(Md);Nb=i(Q1,"TFDistilBertForTokenClassification"),Q1.forEach(t),Yh.forEach(t),sh=h(s),Se=n(s,"DIV",{class:!0});var Dt=r(Se);$(ea.$$.fragment,Dt),Lb=h(Dt),Ed=n(Dt,"P",{});var R1=r(Ed);Sb=i(R1,`DistilBert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
for Named-Entity-Recognition (NER) tasks.`),R1.forEach(t),Ib=h(Dt),ta=n(Dt,"P",{});var Zh=r(ta);Wb=i(Zh,"This model inherits from "),sl=n(Zh,"A",{href:!0});var U1=r(sl);Qb=i(U1,"TFPreTrainedModel"),U1.forEach(t),Rb=i(Zh,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Zh.forEach(t),Ub=h(Dt),sa=n(Dt,"P",{});var eu=r(sa);Hb=i(eu,"This model is also a "),oa=n(eu,"A",{href:!0,rel:!0});var H1=r(oa);Vb=i(H1,"tf.keras.Model"),H1.forEach(t),Kb=i(eu,` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),eu.forEach(t),Jb=h(Dt),$(So.$$.fragment,Dt),Gb=h(Dt),Ze=n(Dt,"DIV",{class:!0});var Yt=r(Ze);$(na.$$.fragment,Yt),Xb=h(Yt),Ds=n(Yt,"P",{});var wl=r(Ds);Yb=i(wl,"The "),ol=n(wl,"A",{href:!0});var V1=r(ol);Zb=i(V1,"TFDistilBertForTokenClassification"),V1.forEach(t),ev=i(wl," forward method, overrides the "),zd=n(wl,"CODE",{});var K1=r(zd);tv=i(K1,"__call__"),K1.forEach(t),sv=i(wl," special method."),wl.forEach(t),ov=h(Yt),$(Io.$$.fragment,Yt),nv=h(Yt),$(Wo.$$.fragment,Yt),rv=h(Yt),$(Qo.$$.fragment,Yt),Yt.forEach(t),Dt.forEach(t),oh=h(s),xs=n(s,"H2",{class:!0});var tu=r(xs);Ro=n(tu,"A",{id:!0,class:!0,href:!0});var J1=r(Ro);jd=n(J1,"SPAN",{});var G1=r(jd);$(ra.$$.fragment,G1),G1.forEach(t),J1.forEach(t),av=h(tu),Cd=n(tu,"SPAN",{});var X1=r(Cd);iv=i(X1,"TFDistilBertForQuestionAnswering"),X1.forEach(t),tu.forEach(t),nh=h(s),Ie=n(s,"DIV",{class:!0});var xt=r(Ie);$(aa.$$.fragment,xt),lv=h(xt),Bs=n(xt,"P",{});var yl=r(Bs);dv=i(yl,`DistilBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
linear layer on top of the hidden-states output to compute `),qd=n(yl,"CODE",{});var Y1=r(qd);cv=i(Y1,"span start logits"),Y1.forEach(t),pv=i(yl," and "),Pd=n(yl,"CODE",{});var Z1=r(Pd);hv=i(Z1,"span end logits"),Z1.forEach(t),uv=i(yl,")."),yl.forEach(t),fv=h(xt),ia=n(xt,"P",{});var su=r(ia);mv=i(su,"This model inherits from "),nl=n(su,"A",{href:!0});var e2=r(nl);gv=i(e2,"TFPreTrainedModel"),e2.forEach(t),_v=i(su,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),su.forEach(t),bv=h(xt),la=n(xt,"P",{});var ou=r(la);vv=i(ou,"This model is also a "),da=n(ou,"A",{href:!0,rel:!0});var t2=r(da);kv=i(t2,"tf.keras.Model"),t2.forEach(t),Tv=i(ou,` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),ou.forEach(t),$v=h(xt),$(Uo.$$.fragment,xt),wv=h(xt),et=n(xt,"DIV",{class:!0});var Zt=r(et);$(ca.$$.fragment,Zt),yv=h(Zt),Ms=n(Zt,"P",{});var Fl=r(Ms);Fv=i(Fl,"The "),rl=n(Fl,"A",{href:!0});var s2=r(rl);Dv=i(s2,"TFDistilBertForQuestionAnswering"),s2.forEach(t),xv=i(Fl," forward method, overrides the "),Ad=n(Fl,"CODE",{});var o2=r(Ad);Bv=i(o2,"__call__"),o2.forEach(t),Mv=i(Fl," special method."),Fl.forEach(t),Ev=h(Zt),$(Ho.$$.fragment,Zt),zv=h(Zt),$(Vo.$$.fragment,Zt),jv=h(Zt),$(Ko.$$.fragment,Zt),Zt.forEach(t),xt.forEach(t),rh=h(s),Es=n(s,"H2",{class:!0});var nu=r(Es);Jo=n(nu,"A",{id:!0,class:!0,href:!0});var n2=r(Jo);Od=n(n2,"SPAN",{});var r2=r(Od);$(pa.$$.fragment,r2),r2.forEach(t),n2.forEach(t),Cv=h(nu),Nd=n(nu,"SPAN",{});var a2=r(Nd);qv=i(a2,"FlaxDistilBertModel"),a2.forEach(t),nu.forEach(t),ah=h(s),Me=n(s,"DIV",{class:!0});var st=r(Me);$(ha.$$.fragment,st),Pv=h(st),Ld=n(st,"P",{});var i2=r(Ld);Av=i(i2,"The bare DistilBert Model transformer outputting raw hidden-states without any specific head on top."),i2.forEach(t),Ov=h(st),ua=n(st,"P",{});var ru=r(ua);Nv=i(ru,"This model inherits from "),al=n(ru,"A",{href:!0});var l2=r(al);Lv=i(l2,"FlaxPreTrainedModel"),l2.forEach(t),Sv=i(ru,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading, saving and converting weights from PyTorch models)`),ru.forEach(t),Iv=h(st),fa=n(st,"P",{});var au=r(fa);Wv=i(au,"This model is also a Flax Linen "),ma=n(au,"A",{href:!0,rel:!0});var d2=r(ma);Qv=i(d2,"flax.linen.Module"),d2.forEach(t),Rv=i(au,`
subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
general usage and behavior.`),au.forEach(t),Uv=h(st),Sd=n(st,"P",{});var c2=r(Sd);Hv=i(c2,"Finally, this model supports inherent JAX features such as:"),c2.forEach(t),Vv=h(st),Mt=n(st,"UL",{});var kn=r(Mt);Id=n(kn,"LI",{});var p2=r(Id);ga=n(p2,"A",{href:!0,rel:!0});var h2=r(ga);Kv=i(h2,"Just-In-Time (JIT) compilation"),h2.forEach(t),p2.forEach(t),Jv=h(kn),Wd=n(kn,"LI",{});var u2=r(Wd);_a=n(u2,"A",{href:!0,rel:!0});var f2=r(_a);Gv=i(f2,"Automatic Differentiation"),f2.forEach(t),u2.forEach(t),Xv=h(kn),Qd=n(kn,"LI",{});var m2=r(Qd);ba=n(m2,"A",{href:!0,rel:!0});var g2=r(ba);Yv=i(g2,"Vectorization"),g2.forEach(t),m2.forEach(t),Zv=h(kn),Rd=n(kn,"LI",{});var _2=r(Rd);va=n(_2,"A",{href:!0,rel:!0});var b2=r(va);ek=i(b2,"Parallelization"),b2.forEach(t),_2.forEach(t),kn.forEach(t),tk=h(st),gt=n(st,"DIV",{class:!0});var Tn=r(gt);$(ka.$$.fragment,Tn),sk=h(Tn),zs=n(Tn,"P",{});var Dl=r(zs);ok=i(Dl,"The "),Ud=n(Dl,"CODE",{});var v2=r(Ud);nk=i(v2,"FlaxDistilBertPreTrainedModel"),v2.forEach(t),rk=i(Dl," forward method, overrides the "),Hd=n(Dl,"CODE",{});var k2=r(Hd);ak=i(k2,"__call__"),k2.forEach(t),ik=i(Dl," special method."),Dl.forEach(t),lk=h(Tn),$(Go.$$.fragment,Tn),dk=h(Tn),$(Xo.$$.fragment,Tn),Tn.forEach(t),st.forEach(t),ih=h(s),js=n(s,"H2",{class:!0});var iu=r(js);Yo=n(iu,"A",{id:!0,class:!0,href:!0});var T2=r(Yo);Vd=n(T2,"SPAN",{});var $2=r(Vd);$(Ta.$$.fragment,$2),$2.forEach(t),T2.forEach(t),ck=h(iu),Kd=n(iu,"SPAN",{});var w2=r(Kd);pk=i(w2,"FlaxDistilBertForMaskedLM"),w2.forEach(t),iu.forEach(t),lh=h(s),Ee=n(s,"DIV",{class:!0});var ot=r(Ee);$($a.$$.fragment,ot),hk=h(ot),wa=n(ot,"P",{});var lu=r(wa);uk=i(lu,"DistilBert Model with a "),Jd=n(lu,"CODE",{});var y2=r(Jd);fk=i(y2,"language modeling"),y2.forEach(t),mk=i(lu," head on top."),lu.forEach(t),gk=h(ot),ya=n(ot,"P",{});var du=r(ya);_k=i(du,"This model inherits from "),il=n(du,"A",{href:!0});var F2=r(il);bk=i(F2,"FlaxPreTrainedModel"),F2.forEach(t),vk=i(du,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading, saving and converting weights from PyTorch models)`),du.forEach(t),kk=h(ot),Fa=n(ot,"P",{});var cu=r(Fa);Tk=i(cu,"This model is also a Flax Linen "),Da=n(cu,"A",{href:!0,rel:!0});var D2=r(Da);$k=i(D2,"flax.linen.Module"),D2.forEach(t),wk=i(cu,`
subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
general usage and behavior.`),cu.forEach(t),yk=h(ot),Gd=n(ot,"P",{});var x2=r(Gd);Fk=i(x2,"Finally, this model supports inherent JAX features such as:"),x2.forEach(t),Dk=h(ot),Et=n(ot,"UL",{});var $n=r(Et);Xd=n($n,"LI",{});var B2=r(Xd);xa=n(B2,"A",{href:!0,rel:!0});var M2=r(xa);xk=i(M2,"Just-In-Time (JIT) compilation"),M2.forEach(t),B2.forEach(t),Bk=h($n),Yd=n($n,"LI",{});var E2=r(Yd);Ba=n(E2,"A",{href:!0,rel:!0});var z2=r(Ba);Mk=i(z2,"Automatic Differentiation"),z2.forEach(t),E2.forEach(t),Ek=h($n),Zd=n($n,"LI",{});var j2=r(Zd);Ma=n(j2,"A",{href:!0,rel:!0});var C2=r(Ma);zk=i(C2,"Vectorization"),C2.forEach(t),j2.forEach(t),jk=h($n),ec=n($n,"LI",{});var q2=r(ec);Ea=n(q2,"A",{href:!0,rel:!0});var P2=r(Ea);Ck=i(P2,"Parallelization"),P2.forEach(t),q2.forEach(t),$n.forEach(t),qk=h(ot),_t=n(ot,"DIV",{class:!0});var wn=r(_t);$(za.$$.fragment,wn),Pk=h(wn),Cs=n(wn,"P",{});var xl=r(Cs);Ak=i(xl,"The "),tc=n(xl,"CODE",{});var A2=r(tc);Ok=i(A2,"FlaxDistilBertPreTrainedModel"),A2.forEach(t),Nk=i(xl," forward method, overrides the "),sc=n(xl,"CODE",{});var O2=r(sc);Lk=i(O2,"__call__"),O2.forEach(t),Sk=i(xl," special method."),xl.forEach(t),Ik=h(wn),$(Zo.$$.fragment,wn),Wk=h(wn),$(en.$$.fragment,wn),wn.forEach(t),ot.forEach(t),dh=h(s),qs=n(s,"H2",{class:!0});var pu=r(qs);tn=n(pu,"A",{id:!0,class:!0,href:!0});var N2=r(tn);oc=n(N2,"SPAN",{});var L2=r(oc);$(ja.$$.fragment,L2),L2.forEach(t),N2.forEach(t),Qk=h(pu),nc=n(pu,"SPAN",{});var S2=r(nc);Rk=i(S2,"FlaxDistilBertForSequenceClassification"),S2.forEach(t),pu.forEach(t),ch=h(s),ze=n(s,"DIV",{class:!0});var nt=r(ze);$(Ca.$$.fragment,nt),Uk=h(nt),rc=n(nt,"P",{});var I2=r(rc);Hk=i(I2,`DistilBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the
pooled output) e.g. for GLUE tasks.`),I2.forEach(t),Vk=h(nt),qa=n(nt,"P",{});var hu=r(qa);Kk=i(hu,"This model inherits from "),ll=n(hu,"A",{href:!0});var W2=r(ll);Jk=i(W2,"FlaxPreTrainedModel"),W2.forEach(t),Gk=i(hu,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading, saving and converting weights from PyTorch models)`),hu.forEach(t),Xk=h(nt),Pa=n(nt,"P",{});var uu=r(Pa);Yk=i(uu,"This model is also a Flax Linen "),Aa=n(uu,"A",{href:!0,rel:!0});var Q2=r(Aa);Zk=i(Q2,"flax.linen.Module"),Q2.forEach(t),eT=i(uu,`
subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
general usage and behavior.`),uu.forEach(t),tT=h(nt),ac=n(nt,"P",{});var R2=r(ac);sT=i(R2,"Finally, this model supports inherent JAX features such as:"),R2.forEach(t),oT=h(nt),zt=n(nt,"UL",{});var yn=r(zt);ic=n(yn,"LI",{});var U2=r(ic);Oa=n(U2,"A",{href:!0,rel:!0});var H2=r(Oa);nT=i(H2,"Just-In-Time (JIT) compilation"),H2.forEach(t),U2.forEach(t),rT=h(yn),lc=n(yn,"LI",{});var V2=r(lc);Na=n(V2,"A",{href:!0,rel:!0});var K2=r(Na);aT=i(K2,"Automatic Differentiation"),K2.forEach(t),V2.forEach(t),iT=h(yn),dc=n(yn,"LI",{});var J2=r(dc);La=n(J2,"A",{href:!0,rel:!0});var G2=r(La);lT=i(G2,"Vectorization"),G2.forEach(t),J2.forEach(t),dT=h(yn),cc=n(yn,"LI",{});var X2=r(cc);Sa=n(X2,"A",{href:!0,rel:!0});var Y2=r(Sa);cT=i(Y2,"Parallelization"),Y2.forEach(t),X2.forEach(t),yn.forEach(t),pT=h(nt),bt=n(nt,"DIV",{class:!0});var Fn=r(bt);$(Ia.$$.fragment,Fn),hT=h(Fn),Ps=n(Fn,"P",{});var Bl=r(Ps);uT=i(Bl,"The "),pc=n(Bl,"CODE",{});var Z2=r(pc);fT=i(Z2,"FlaxDistilBertPreTrainedModel"),Z2.forEach(t),mT=i(Bl," forward method, overrides the "),hc=n(Bl,"CODE",{});var ey=r(hc);gT=i(ey,"__call__"),ey.forEach(t),_T=i(Bl," special method."),Bl.forEach(t),bT=h(Fn),$(sn.$$.fragment,Fn),vT=h(Fn),$(on.$$.fragment,Fn),Fn.forEach(t),nt.forEach(t),ph=h(s),As=n(s,"H2",{class:!0});var fu=r(As);nn=n(fu,"A",{id:!0,class:!0,href:!0});var ty=r(nn);uc=n(ty,"SPAN",{});var sy=r(uc);$(Wa.$$.fragment,sy),sy.forEach(t),ty.forEach(t),kT=h(fu),fc=n(fu,"SPAN",{});var oy=r(fc);TT=i(oy,"FlaxDistilBertForMultipleChoice"),oy.forEach(t),fu.forEach(t),hh=h(s),je=n(s,"DIV",{class:!0});var rt=r(je);$(Qa.$$.fragment,rt),$T=h(rt),mc=n(rt,"P",{});var ny=r(mc);wT=i(ny,`DistilBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
a softmax) e.g. for RocStories/SWAG tasks.`),ny.forEach(t),yT=h(rt),Ra=n(rt,"P",{});var mu=r(Ra);FT=i(mu,"This model inherits from "),dl=n(mu,"A",{href:!0});var ry=r(dl);DT=i(ry,"FlaxPreTrainedModel"),ry.forEach(t),xT=i(mu,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading, saving and converting weights from PyTorch models)`),mu.forEach(t),BT=h(rt),Ua=n(rt,"P",{});var gu=r(Ua);MT=i(gu,"This model is also a Flax Linen "),Ha=n(gu,"A",{href:!0,rel:!0});var ay=r(Ha);ET=i(ay,"flax.linen.Module"),ay.forEach(t),zT=i(gu,`
subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
general usage and behavior.`),gu.forEach(t),jT=h(rt),gc=n(rt,"P",{});var iy=r(gc);CT=i(iy,"Finally, this model supports inherent JAX features such as:"),iy.forEach(t),qT=h(rt),jt=n(rt,"UL",{});var Dn=r(jt);_c=n(Dn,"LI",{});var ly=r(_c);Va=n(ly,"A",{href:!0,rel:!0});var dy=r(Va);PT=i(dy,"Just-In-Time (JIT) compilation"),dy.forEach(t),ly.forEach(t),AT=h(Dn),bc=n(Dn,"LI",{});var cy=r(bc);Ka=n(cy,"A",{href:!0,rel:!0});var py=r(Ka);OT=i(py,"Automatic Differentiation"),py.forEach(t),cy.forEach(t),NT=h(Dn),vc=n(Dn,"LI",{});var hy=r(vc);Ja=n(hy,"A",{href:!0,rel:!0});var uy=r(Ja);LT=i(uy,"Vectorization"),uy.forEach(t),hy.forEach(t),ST=h(Dn),kc=n(Dn,"LI",{});var fy=r(kc);Ga=n(fy,"A",{href:!0,rel:!0});var my=r(Ga);IT=i(my,"Parallelization"),my.forEach(t),fy.forEach(t),Dn.forEach(t),WT=h(rt),vt=n(rt,"DIV",{class:!0});var xn=r(vt);$(Xa.$$.fragment,xn),QT=h(xn),Os=n(xn,"P",{});var Ml=r(Os);RT=i(Ml,"The "),Tc=n(Ml,"CODE",{});var gy=r(Tc);UT=i(gy,"FlaxDistilBertPreTrainedModel"),gy.forEach(t),HT=i(Ml," forward method, overrides the "),$c=n(Ml,"CODE",{});var _y=r($c);VT=i(_y,"__call__"),_y.forEach(t),KT=i(Ml," special method."),Ml.forEach(t),JT=h(xn),$(rn.$$.fragment,xn),GT=h(xn),$(an.$$.fragment,xn),xn.forEach(t),rt.forEach(t),uh=h(s),Ns=n(s,"H2",{class:!0});var _u=r(Ns);ln=n(_u,"A",{id:!0,class:!0,href:!0});var by=r(ln);wc=n(by,"SPAN",{});var vy=r(wc);$(Ya.$$.fragment,vy),vy.forEach(t),by.forEach(t),XT=h(_u),yc=n(_u,"SPAN",{});var ky=r(yc);YT=i(ky,"FlaxDistilBertForTokenClassification"),ky.forEach(t),_u.forEach(t),fh=h(s),Ce=n(s,"DIV",{class:!0});var at=r(Ce);$(Za.$$.fragment,at),ZT=h(at),Fc=n(at,"P",{});var Ty=r(Fc);e$=i(Ty,`DistilBert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
for Named-Entity-Recognition (NER) tasks.`),Ty.forEach(t),t$=h(at),ei=n(at,"P",{});var bu=r(ei);s$=i(bu,"This model inherits from "),cl=n(bu,"A",{href:!0});var $y=r(cl);o$=i($y,"FlaxPreTrainedModel"),$y.forEach(t),n$=i(bu,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading, saving and converting weights from PyTorch models)`),bu.forEach(t),r$=h(at),ti=n(at,"P",{});var vu=r(ti);a$=i(vu,"This model is also a Flax Linen "),si=n(vu,"A",{href:!0,rel:!0});var wy=r(si);i$=i(wy,"flax.linen.Module"),wy.forEach(t),l$=i(vu,`
subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
general usage and behavior.`),vu.forEach(t),d$=h(at),Dc=n(at,"P",{});var yy=r(Dc);c$=i(yy,"Finally, this model supports inherent JAX features such as:"),yy.forEach(t),p$=h(at),Ct=n(at,"UL",{});var Bn=r(Ct);xc=n(Bn,"LI",{});var Fy=r(xc);oi=n(Fy,"A",{href:!0,rel:!0});var Dy=r(oi);h$=i(Dy,"Just-In-Time (JIT) compilation"),Dy.forEach(t),Fy.forEach(t),u$=h(Bn),Bc=n(Bn,"LI",{});var xy=r(Bc);ni=n(xy,"A",{href:!0,rel:!0});var By=r(ni);f$=i(By,"Automatic Differentiation"),By.forEach(t),xy.forEach(t),m$=h(Bn),Mc=n(Bn,"LI",{});var My=r(Mc);ri=n(My,"A",{href:!0,rel:!0});var Ey=r(ri);g$=i(Ey,"Vectorization"),Ey.forEach(t),My.forEach(t),_$=h(Bn),Ec=n(Bn,"LI",{});var zy=r(Ec);ai=n(zy,"A",{href:!0,rel:!0});var jy=r(ai);b$=i(jy,"Parallelization"),jy.forEach(t),zy.forEach(t),Bn.forEach(t),v$=h(at),kt=n(at,"DIV",{class:!0});var Mn=r(kt);$(ii.$$.fragment,Mn),k$=h(Mn),Ls=n(Mn,"P",{});var El=r(Ls);T$=i(El,"The "),zc=n(El,"CODE",{});var Cy=r(zc);$$=i(Cy,"FlaxDistilBertPreTrainedModel"),Cy.forEach(t),w$=i(El," forward method, overrides the "),jc=n(El,"CODE",{});var qy=r(jc);y$=i(qy,"__call__"),qy.forEach(t),F$=i(El," special method."),El.forEach(t),D$=h(Mn),$(dn.$$.fragment,Mn),x$=h(Mn),$(cn.$$.fragment,Mn),Mn.forEach(t),at.forEach(t),mh=h(s),Ss=n(s,"H2",{class:!0});var ku=r(Ss);pn=n(ku,"A",{id:!0,class:!0,href:!0});var Py=r(pn);Cc=n(Py,"SPAN",{});var Ay=r(Cc);$(li.$$.fragment,Ay),Ay.forEach(t),Py.forEach(t),B$=h(ku),qc=n(ku,"SPAN",{});var Oy=r(qc);M$=i(Oy,"FlaxDistilBertForQuestionAnswering"),Oy.forEach(t),ku.forEach(t),gh=h(s),qe=n(s,"DIV",{class:!0});var it=r(qe);$(di.$$.fragment,it),E$=h(it),Is=n(it,"P",{});var zl=r(Is);z$=i(zl,`DistilBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
linear layers on top of the hidden-states output to compute `),Pc=n(zl,"CODE",{});var Ny=r(Pc);j$=i(Ny,"span start logits"),Ny.forEach(t),C$=i(zl," and "),Ac=n(zl,"CODE",{});var Ly=r(Ac);q$=i(Ly,"span end logits"),Ly.forEach(t),P$=i(zl,")."),zl.forEach(t),A$=h(it),ci=n(it,"P",{});var Tu=r(ci);O$=i(Tu,"This model inherits from "),pl=n(Tu,"A",{href:!0});var Sy=r(pl);N$=i(Sy,"FlaxPreTrainedModel"),Sy.forEach(t),L$=i(Tu,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading, saving and converting weights from PyTorch models)`),Tu.forEach(t),S$=h(it),pi=n(it,"P",{});var $u=r(pi);I$=i($u,"This model is also a Flax Linen "),hi=n($u,"A",{href:!0,rel:!0});var Iy=r(hi);W$=i(Iy,"flax.linen.Module"),Iy.forEach(t),Q$=i($u,`
subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
general usage and behavior.`),$u.forEach(t),R$=h(it),Oc=n(it,"P",{});var Wy=r(Oc);U$=i(Wy,"Finally, this model supports inherent JAX features such as:"),Wy.forEach(t),H$=h(it),qt=n(it,"UL",{});var En=r(qt);Nc=n(En,"LI",{});var Qy=r(Nc);ui=n(Qy,"A",{href:!0,rel:!0});var Ry=r(ui);V$=i(Ry,"Just-In-Time (JIT) compilation"),Ry.forEach(t),Qy.forEach(t),K$=h(En),Lc=n(En,"LI",{});var Uy=r(Lc);fi=n(Uy,"A",{href:!0,rel:!0});var Hy=r(fi);J$=i(Hy,"Automatic Differentiation"),Hy.forEach(t),Uy.forEach(t),G$=h(En),Sc=n(En,"LI",{});var Vy=r(Sc);mi=n(Vy,"A",{href:!0,rel:!0});var Ky=r(mi);X$=i(Ky,"Vectorization"),Ky.forEach(t),Vy.forEach(t),Y$=h(En),Ic=n(En,"LI",{});var Jy=r(Ic);gi=n(Jy,"A",{href:!0,rel:!0});var Gy=r(gi);Z$=i(Gy,"Parallelization"),Gy.forEach(t),Jy.forEach(t),En.forEach(t),ew=h(it),Tt=n(it,"DIV",{class:!0});var zn=r(Tt);$(_i.$$.fragment,zn),tw=h(zn),Ws=n(zn,"P",{});var jl=r(Ws);sw=i(jl,"The "),Wc=n(jl,"CODE",{});var Xy=r(Wc);ow=i(Xy,"FlaxDistilBertPreTrainedModel"),Xy.forEach(t),nw=i(jl," forward method, overrides the "),Qc=n(jl,"CODE",{});var Yy=r(Qc);rw=i(Yy,"__call__"),Yy.forEach(t),aw=i(jl," special method."),jl.forEach(t),iw=h(zn),$(hn.$$.fragment,zn),lw=h(zn),$(un.$$.fragment,zn),zn.forEach(t),it.forEach(t),this.h()},h(){m(d,"name","hf:doc:metadata"),m(d,"content",JSON.stringify(nD)),m(u,"id","distilbert"),m(u,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(u,"href","#distilbert"),m(c,"class","relative group"),m(G,"id","overview"),m(G,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(G,"href","#overview"),m(E,"class","relative group"),m(Y,"href","https://medium.com/huggingface/distilbert-8cf3380435b5"),m(Y,"rel","nofollow"),m(j,"href","https://arxiv.org/abs/1910.01108"),m(j,"rel","nofollow"),m(jn,"href","https://huggingface.co/victorsanh"),m(jn,"rel","nofollow"),m(Cn,"href","https://huggingface.co/kamalkraj"),m(Cn,"rel","nofollow"),m(qn,"href","https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation"),m(qn,"rel","nofollow"),m(Rs,"id","transformers.DistilBertConfig"),m(Rs,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(Rs,"href","#transformers.DistilBertConfig"),m(es,"class","relative group"),m(Di,"href","/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertModel"),m(xi,"href","/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.TFDistilBertModel"),m(On,"href","https://huggingface.co/distilbert-base-uncased"),m(On,"rel","nofollow"),m(Bi,"href","/docs/transformers/pr_18727/en/main_classes/configuration#transformers.PretrainedConfig"),m(Mi,"href","/docs/transformers/pr_18727/en/main_classes/configuration#transformers.PretrainedConfig"),m(lt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Hs,"id","transformers.DistilBertTokenizer"),m(Hs,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(Hs,"href","#transformers.DistilBertTokenizer"),m(ss,"class","relative group"),m(Ei,"href","/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertTokenizer"),m(zi,"href","/docs/transformers/pr_18727/en/model_doc/bert#transformers.BertTokenizer"),m(ji,"href","/docs/transformers/pr_18727/en/model_doc/bert#transformers.BertTokenizer"),m(dt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Ks,"id","transformers.DistilBertTokenizerFast"),m(Ks,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(Ks,"href","#transformers.DistilBertTokenizerFast"),m(os,"class","relative group"),m(Ci,"href","/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertTokenizerFast"),m(qi,"href","/docs/transformers/pr_18727/en/model_doc/bert#transformers.BertTokenizerFast"),m(Pi,"href","/docs/transformers/pr_18727/en/model_doc/bert#transformers.BertTokenizerFast"),m(ct,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Gs,"id","transformers.DistilBertModel"),m(Gs,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(Gs,"href","#transformers.DistilBertModel"),m(ns,"class","relative group"),m(Ai,"href","/docs/transformers/pr_18727/en/main_classes/model#transformers.PreTrainedModel"),m(Jn,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),m(Jn,"rel","nofollow"),m(Oi,"href","/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertModel"),m(ht,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(We,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Zs,"id","transformers.DistilBertForMaskedLM"),m(Zs,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(Zs,"href","#transformers.DistilBertForMaskedLM"),m(as,"class","relative group"),m(Ni,"href","/docs/transformers/pr_18727/en/main_classes/model#transformers.PreTrainedModel"),m(sr,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),m(sr,"rel","nofollow"),m(Li,"href","/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertForMaskedLM"),m(Ke,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Qe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(oo,"id","transformers.DistilBertForSequenceClassification"),m(oo,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(oo,"href","#transformers.DistilBertForSequenceClassification"),m(ls,"class","relative group"),m(Si,"href","/docs/transformers/pr_18727/en/main_classes/model#transformers.PreTrainedModel"),m(lr,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),m(lr,"rel","nofollow"),m(Ii,"href","/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertForSequenceClassification"),m(Pe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Re,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(co,"id","transformers.DistilBertForMultipleChoice"),m(co,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(co,"href","#transformers.DistilBertForMultipleChoice"),m(cs,"class","relative group"),m(Wi,"href","/docs/transformers/pr_18727/en/main_classes/model#transformers.PreTrainedModel"),m(fr,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),m(fr,"rel","nofollow"),m(Qi,"href","/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertForMultipleChoice"),m(ut,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Ue,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(uo,"id","transformers.DistilBertForTokenClassification"),m(uo,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(uo,"href","#transformers.DistilBertForTokenClassification"),m(hs,"class","relative group"),m(Ri,"href","/docs/transformers/pr_18727/en/main_classes/model#transformers.PreTrainedModel"),m(kr,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),m(kr,"rel","nofollow"),m(Ui,"href","/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertForTokenClassification"),m(Je,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(He,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(_o,"id","transformers.DistilBertForQuestionAnswering"),m(_o,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(_o,"href","#transformers.DistilBertForQuestionAnswering"),m(fs,"class","relative group"),m(Hi,"href","/docs/transformers/pr_18727/en/main_classes/model#transformers.PreTrainedModel"),m(Dr,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),m(Dr,"rel","nofollow"),m(Vi,"href","/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.DistilBertForQuestionAnswering"),m(Ge,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Ve,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(To,"id","transformers.TFDistilBertModel"),m(To,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(To,"href","#transformers.TFDistilBertModel"),m(_s,"class","relative group"),m(Ki,"href","/docs/transformers/pr_18727/en/main_classes/model#transformers.TFPreTrainedModel"),m(jr,"href","https://www.tensorflow.org/api_docs/python/tf/keras/Model"),m(jr,"rel","nofollow"),m(Ji,"href","/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.TFDistilBertModel"),m(ft,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Ae,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Fo,"id","transformers.TFDistilBertForMaskedLM"),m(Fo,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(Fo,"href","#transformers.TFDistilBertForMaskedLM"),m(vs,"class","relative group"),m(Gi,"href","/docs/transformers/pr_18727/en/main_classes/model#transformers.TFPreTrainedModel"),m(Lr,"href","https://www.tensorflow.org/api_docs/python/tf/keras/Model"),m(Lr,"rel","nofollow"),m(Xi,"href","/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.TFDistilBertForMaskedLM"),m(Xe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Oe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Eo,"id","transformers.TFDistilBertForSequenceClassification"),m(Eo,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(Eo,"href","#transformers.TFDistilBertForSequenceClassification"),m(Ts,"class","relative group"),m(Yi,"href","/docs/transformers/pr_18727/en/main_classes/model#transformers.TFPreTrainedModel"),m(Ur,"href","https://www.tensorflow.org/api_docs/python/tf/keras/Model"),m(Ur,"rel","nofollow"),m(Zi,"href","/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.TFDistilBertForSequenceClassification"),m(Ye,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Ne,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Po,"id","transformers.TFDistilBertForMultipleChoice"),m(Po,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(Po,"href","#transformers.TFDistilBertForMultipleChoice"),m(ws,"class","relative group"),m(el,"href","/docs/transformers/pr_18727/en/main_classes/model#transformers.TFPreTrainedModel"),m(Xr,"href","https://www.tensorflow.org/api_docs/python/tf/keras/Model"),m(Xr,"rel","nofollow"),m(tl,"href","/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.TFDistilBertForMultipleChoice"),m(mt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Le,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Lo,"id","transformers.TFDistilBertForTokenClassification"),m(Lo,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(Lo,"href","#transformers.TFDistilBertForTokenClassification"),m(Fs,"class","relative group"),m(sl,"href","/docs/transformers/pr_18727/en/main_classes/model#transformers.TFPreTrainedModel"),m(oa,"href","https://www.tensorflow.org/api_docs/python/tf/keras/Model"),m(oa,"rel","nofollow"),m(ol,"href","/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.TFDistilBertForTokenClassification"),m(Ze,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Se,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Ro,"id","transformers.TFDistilBertForQuestionAnswering"),m(Ro,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(Ro,"href","#transformers.TFDistilBertForQuestionAnswering"),m(xs,"class","relative group"),m(nl,"href","/docs/transformers/pr_18727/en/main_classes/model#transformers.TFPreTrainedModel"),m(da,"href","https://www.tensorflow.org/api_docs/python/tf/keras/Model"),m(da,"rel","nofollow"),m(rl,"href","/docs/transformers/pr_18727/en/model_doc/distilbert#transformers.TFDistilBertForQuestionAnswering"),m(et,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Ie,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Jo,"id","transformers.FlaxDistilBertModel"),m(Jo,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(Jo,"href","#transformers.FlaxDistilBertModel"),m(Es,"class","relative group"),m(al,"href","/docs/transformers/pr_18727/en/main_classes/model#transformers.FlaxPreTrainedModel"),m(ma,"href","https://flax.readthedocs.io/en/latest/flax.linen.html#module"),m(ma,"rel","nofollow"),m(ga,"href","https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit"),m(ga,"rel","nofollow"),m(_a,"href","https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation"),m(_a,"rel","nofollow"),m(ba,"href","https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap"),m(ba,"rel","nofollow"),m(va,"href","https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap"),m(va,"rel","nofollow"),m(gt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Me,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Yo,"id","transformers.FlaxDistilBertForMaskedLM"),m(Yo,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(Yo,"href","#transformers.FlaxDistilBertForMaskedLM"),m(js,"class","relative group"),m(il,"href","/docs/transformers/pr_18727/en/main_classes/model#transformers.FlaxPreTrainedModel"),m(Da,"href","https://flax.readthedocs.io/en/latest/flax.linen.html#module"),m(Da,"rel","nofollow"),m(xa,"href","https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit"),m(xa,"rel","nofollow"),m(Ba,"href","https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation"),m(Ba,"rel","nofollow"),m(Ma,"href","https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap"),m(Ma,"rel","nofollow"),m(Ea,"href","https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap"),m(Ea,"rel","nofollow"),m(_t,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(tn,"id","transformers.FlaxDistilBertForSequenceClassification"),m(tn,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(tn,"href","#transformers.FlaxDistilBertForSequenceClassification"),m(qs,"class","relative group"),m(ll,"href","/docs/transformers/pr_18727/en/main_classes/model#transformers.FlaxPreTrainedModel"),m(Aa,"href","https://flax.readthedocs.io/en/latest/flax.linen.html#module"),m(Aa,"rel","nofollow"),m(Oa,"href","https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit"),m(Oa,"rel","nofollow"),m(Na,"href","https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation"),m(Na,"rel","nofollow"),m(La,"href","https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap"),m(La,"rel","nofollow"),m(Sa,"href","https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap"),m(Sa,"rel","nofollow"),m(bt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(ze,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(nn,"id","transformers.FlaxDistilBertForMultipleChoice"),m(nn,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(nn,"href","#transformers.FlaxDistilBertForMultipleChoice"),m(As,"class","relative group"),m(dl,"href","/docs/transformers/pr_18727/en/main_classes/model#transformers.FlaxPreTrainedModel"),m(Ha,"href","https://flax.readthedocs.io/en/latest/flax.linen.html#module"),m(Ha,"rel","nofollow"),m(Va,"href","https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit"),m(Va,"rel","nofollow"),m(Ka,"href","https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation"),m(Ka,"rel","nofollow"),m(Ja,"href","https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap"),m(Ja,"rel","nofollow"),m(Ga,"href","https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap"),m(Ga,"rel","nofollow"),m(vt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(je,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(ln,"id","transformers.FlaxDistilBertForTokenClassification"),m(ln,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(ln,"href","#transformers.FlaxDistilBertForTokenClassification"),m(Ns,"class","relative group"),m(cl,"href","/docs/transformers/pr_18727/en/main_classes/model#transformers.FlaxPreTrainedModel"),m(si,"href","https://flax.readthedocs.io/en/latest/flax.linen.html#module"),m(si,"rel","nofollow"),m(oi,"href","https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit"),m(oi,"rel","nofollow"),m(ni,"href","https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation"),m(ni,"rel","nofollow"),m(ri,"href","https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap"),m(ri,"rel","nofollow"),m(ai,"href","https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap"),m(ai,"rel","nofollow"),m(kt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Ce,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(pn,"id","transformers.FlaxDistilBertForQuestionAnswering"),m(pn,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(pn,"href","#transformers.FlaxDistilBertForQuestionAnswering"),m(Ss,"class","relative group"),m(pl,"href","/docs/transformers/pr_18727/en/main_classes/model#transformers.FlaxPreTrainedModel"),m(hi,"href","https://flax.readthedocs.io/en/latest/flax.linen.html#module"),m(hi,"rel","nofollow"),m(ui,"href","https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit"),m(ui,"rel","nofollow"),m(fi,"href","https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation"),m(fi,"rel","nofollow"),m(mi,"href","https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap"),m(mi,"rel","nofollow"),m(gi,"href","https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap"),m(gi,"rel","nofollow"),m(Tt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(qe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(s,v){e(document.head,d),b(s,g,v),b(s,c,v),e(c,u),e(u,_),w(l,_,null),e(c,f),e(c,B),e(B,he),b(s,V,v),b(s,E,v),e(E,G),e(G,S),w(X,S,null),e(E,ue),e(E,I),e(I,fe),b(s,le,v),b(s,N,v),e(N,P),e(N,Y),e(Y,K),e(N,z),e(N,j),e(j,me),e(N,W),e(N,se),e(se,ge),e(N,Q),b(s,de,v),b(s,ee,v),e(ee,A),b(s,ce,v),b(s,L,v),e(L,oe),e(oe,_e),b(s,q,v),b(s,te,v),e(te,R),b(s,pe,v),b(s,k,v),e(k,M),e(M,J),e(M,ve),e(ve,$e),e(M,O),e(M,ke),e(ke,we),e(M,ye),e(M,C),e(C,U),e(M,Fe),e(k,De),e(k,Z),e(Z,xe),e(Z,ne),e(ne,Be),e(Z,wu),b(s,Bp,v),b(s,pt,v),e(pt,yu),e(pt,jn),e(jn,Fu),e(pt,Du),e(pt,Cn),e(Cn,xu),e(pt,Bu),e(pt,qn),e(qn,Mu),e(pt,Eu),b(s,Mp,v),b(s,es,v),e(es,Rs),e(Rs,Cl),w(Pn,Cl,null),e(es,zu),e(es,ql),e(ql,ju),b(s,Ep,v),b(s,lt,v),w(An,lt,null),e(lt,Cu),e(lt,Bt),e(Bt,qu),e(Bt,Di),e(Di,Pu),e(Bt,Au),e(Bt,xi),e(xi,Ou),e(Bt,Nu),e(Bt,On),e(On,Lu),e(Bt,Su),e(lt,Iu),e(lt,ts),e(ts,Wu),e(ts,Bi),e(Bi,Qu),e(ts,Ru),e(ts,Mi),e(Mi,Uu),e(ts,Hu),e(lt,Vu),w(Us,lt,null),b(s,zp,v),b(s,ss,v),e(ss,Hs),e(Hs,Pl),w(Nn,Pl,null),e(ss,Ku),e(ss,Al),e(Al,Ju),b(s,jp,v),b(s,dt,v),w(Ln,dt,null),e(dt,Gu),e(dt,Ol),e(Ol,Xu),e(dt,Yu),e(dt,Vs),e(Vs,Ei),e(Ei,Zu),e(Vs,ef),e(Vs,zi),e(zi,tf),e(Vs,sf),e(dt,of),e(dt,Sn),e(Sn,nf),e(Sn,ji),e(ji,rf),e(Sn,af),b(s,Cp,v),b(s,os,v),e(os,Ks),e(Ks,Nl),w(In,Nl,null),e(os,lf),e(os,Ll),e(Ll,df),b(s,qp,v),b(s,ct,v),w(Wn,ct,null),e(ct,cf),e(ct,Qn),e(Qn,pf),e(Qn,Sl),e(Sl,hf),e(Qn,uf),e(ct,ff),e(ct,Js),e(Js,Ci),e(Ci,mf),e(Js,gf),e(Js,qi),e(qi,_f),e(Js,bf),e(ct,vf),e(ct,Rn),e(Rn,kf),e(Rn,Pi),e(Pi,Tf),e(Rn,$f),b(s,Pp,v),b(s,ns,v),e(ns,Gs),e(Gs,Il),w(Un,Il,null),e(ns,wf),e(ns,Wl),e(Wl,yf),b(s,Ap,v),b(s,We,v),w(Hn,We,null),e(We,Ff),e(We,Ql),e(Ql,Df),e(We,xf),e(We,Vn),e(Vn,Bf),e(Vn,Ai),e(Ai,Mf),e(Vn,Ef),e(We,zf),e(We,Kn),e(Kn,jf),e(Kn,Jn),e(Jn,Cf),e(Kn,qf),e(We,Pf),e(We,ht),w(Gn,ht,null),e(ht,Af),e(ht,rs),e(rs,Of),e(rs,Oi),e(Oi,Nf),e(rs,Lf),e(rs,Rl),e(Rl,Sf),e(rs,If),e(ht,Wf),w(Xs,ht,null),e(ht,Qf),w(Ys,ht,null),b(s,Op,v),b(s,as,v),e(as,Zs),e(Zs,Ul),w(Xn,Ul,null),e(as,Rf),e(as,Hl),e(Hl,Uf),b(s,Np,v),b(s,Qe,v),w(Yn,Qe,null),e(Qe,Hf),e(Qe,Zn),e(Zn,Vf),e(Zn,Vl),e(Vl,Kf),e(Zn,Jf),e(Qe,Gf),e(Qe,er),e(er,Xf),e(er,Ni),e(Ni,Yf),e(er,Zf),e(Qe,em),e(Qe,tr),e(tr,tm),e(tr,sr),e(sr,sm),e(tr,om),e(Qe,nm),e(Qe,Ke),w(or,Ke,null),e(Ke,rm),e(Ke,is),e(is,am),e(is,Li),e(Li,im),e(is,lm),e(is,Kl),e(Kl,dm),e(is,cm),e(Ke,pm),w(eo,Ke,null),e(Ke,hm),w(to,Ke,null),e(Ke,um),w(so,Ke,null),b(s,Lp,v),b(s,ls,v),e(ls,oo),e(oo,Jl),w(nr,Jl,null),e(ls,fm),e(ls,Gl),e(Gl,mm),b(s,Sp,v),b(s,Re,v),w(rr,Re,null),e(Re,gm),e(Re,Xl),e(Xl,_m),e(Re,bm),e(Re,ar),e(ar,vm),e(ar,Si),e(Si,km),e(ar,Tm),e(Re,$m),e(Re,ir),e(ir,wm),e(ir,lr),e(lr,ym),e(ir,Fm),e(Re,Dm),e(Re,Pe),w(dr,Pe,null),e(Pe,xm),e(Pe,ds),e(ds,Bm),e(ds,Ii),e(Ii,Mm),e(ds,Em),e(ds,Yl),e(Yl,zm),e(ds,jm),e(Pe,Cm),w(no,Pe,null),e(Pe,qm),w(ro,Pe,null),e(Pe,Pm),w(ao,Pe,null),e(Pe,Am),w(io,Pe,null),e(Pe,Om),w(lo,Pe,null),b(s,Ip,v),b(s,cs,v),e(cs,co),e(co,Zl),w(cr,Zl,null),e(cs,Nm),e(cs,ed),e(ed,Lm),b(s,Wp,v),b(s,Ue,v),w(pr,Ue,null),e(Ue,Sm),e(Ue,td),e(td,Im),e(Ue,Wm),e(Ue,hr),e(hr,Qm),e(hr,Wi),e(Wi,Rm),e(hr,Um),e(Ue,Hm),e(Ue,ur),e(ur,Vm),e(ur,fr),e(fr,Km),e(ur,Jm),e(Ue,Gm),e(Ue,ut),w(mr,ut,null),e(ut,Xm),e(ut,ps),e(ps,Ym),e(ps,Qi),e(Qi,Zm),e(ps,eg),e(ps,sd),e(sd,tg),e(ps,sg),e(ut,og),w(po,ut,null),e(ut,ng),w(ho,ut,null),b(s,Qp,v),b(s,hs,v),e(hs,uo),e(uo,od),w(gr,od,null),e(hs,rg),e(hs,nd),e(nd,ag),b(s,Rp,v),b(s,He,v),w(_r,He,null),e(He,ig),e(He,rd),e(rd,lg),e(He,dg),e(He,br),e(br,cg),e(br,Ri),e(Ri,pg),e(br,hg),e(He,ug),e(He,vr),e(vr,fg),e(vr,kr),e(kr,mg),e(vr,gg),e(He,_g),e(He,Je),w(Tr,Je,null),e(Je,bg),e(Je,us),e(us,vg),e(us,Ui),e(Ui,kg),e(us,Tg),e(us,ad),e(ad,$g),e(us,wg),e(Je,yg),w(fo,Je,null),e(Je,Fg),w(mo,Je,null),e(Je,Dg),w(go,Je,null),b(s,Up,v),b(s,fs,v),e(fs,_o),e(_o,id),w($r,id,null),e(fs,xg),e(fs,ld),e(ld,Bg),b(s,Hp,v),b(s,Ve,v),w(wr,Ve,null),e(Ve,Mg),e(Ve,ms),e(ms,Eg),e(ms,dd),e(dd,zg),e(ms,jg),e(ms,cd),e(cd,Cg),e(ms,qg),e(Ve,Pg),e(Ve,yr),e(yr,Ag),e(yr,Hi),e(Hi,Og),e(yr,Ng),e(Ve,Lg),e(Ve,Fr),e(Fr,Sg),e(Fr,Dr),e(Dr,Ig),e(Fr,Wg),e(Ve,Qg),e(Ve,Ge),w(xr,Ge,null),e(Ge,Rg),e(Ge,gs),e(gs,Ug),e(gs,Vi),e(Vi,Hg),e(gs,Vg),e(gs,pd),e(pd,Kg),e(gs,Jg),e(Ge,Gg),w(bo,Ge,null),e(Ge,Xg),w(vo,Ge,null),e(Ge,Yg),w(ko,Ge,null),b(s,Vp,v),b(s,_s,v),e(_s,To),e(To,hd),w(Br,hd,null),e(_s,Zg),e(_s,ud),e(ud,e_),b(s,Kp,v),b(s,Ae,v),w(Mr,Ae,null),e(Ae,t_),e(Ae,fd),e(fd,s_),e(Ae,o_),e(Ae,Er),e(Er,n_),e(Er,Ki),e(Ki,r_),e(Er,a_),e(Ae,i_),e(Ae,zr),e(zr,l_),e(zr,jr),e(jr,d_),e(zr,c_),e(Ae,p_),w($o,Ae,null),e(Ae,h_),e(Ae,ft),w(Cr,ft,null),e(ft,u_),e(ft,bs),e(bs,f_),e(bs,Ji),e(Ji,m_),e(bs,g_),e(bs,md),e(md,__),e(bs,b_),e(ft,v_),w(wo,ft,null),e(ft,k_),w(yo,ft,null),b(s,Jp,v),b(s,vs,v),e(vs,Fo),e(Fo,gd),w(qr,gd,null),e(vs,T_),e(vs,_d),e(_d,$_),b(s,Gp,v),b(s,Oe,v),w(Pr,Oe,null),e(Oe,w_),e(Oe,Ar),e(Ar,y_),e(Ar,bd),e(bd,F_),e(Ar,D_),e(Oe,x_),e(Oe,Or),e(Or,B_),e(Or,Gi),e(Gi,M_),e(Or,E_),e(Oe,z_),e(Oe,Nr),e(Nr,j_),e(Nr,Lr),e(Lr,C_),e(Nr,q_),e(Oe,P_),w(Do,Oe,null),e(Oe,A_),e(Oe,Xe),w(Sr,Xe,null),e(Xe,O_),e(Xe,ks),e(ks,N_),e(ks,Xi),e(Xi,L_),e(ks,S_),e(ks,vd),e(vd,I_),e(ks,W_),e(Xe,Q_),w(xo,Xe,null),e(Xe,R_),w(Bo,Xe,null),e(Xe,U_),w(Mo,Xe,null),b(s,Xp,v),b(s,Ts,v),e(Ts,Eo),e(Eo,kd),w(Ir,kd,null),e(Ts,H_),e(Ts,Td),e(Td,V_),b(s,Yp,v),b(s,Ne,v),w(Wr,Ne,null),e(Ne,K_),e(Ne,$d),e($d,J_),e(Ne,G_),e(Ne,Qr),e(Qr,X_),e(Qr,Yi),e(Yi,Y_),e(Qr,Z_),e(Ne,eb),e(Ne,Rr),e(Rr,tb),e(Rr,Ur),e(Ur,sb),e(Rr,ob),e(Ne,nb),w(zo,Ne,null),e(Ne,rb),e(Ne,Ye),w(Hr,Ye,null),e(Ye,ab),e(Ye,$s),e($s,ib),e($s,Zi),e(Zi,lb),e($s,db),e($s,wd),e(wd,cb),e($s,pb),e(Ye,hb),w(jo,Ye,null),e(Ye,ub),w(Co,Ye,null),e(Ye,fb),w(qo,Ye,null),b(s,Zp,v),b(s,ws,v),e(ws,Po),e(Po,yd),w(Vr,yd,null),e(ws,mb),e(ws,Fd),e(Fd,gb),b(s,eh,v),b(s,Le,v),w(Kr,Le,null),e(Le,_b),e(Le,Dd),e(Dd,bb),e(Le,vb),e(Le,Jr),e(Jr,kb),e(Jr,el),e(el,Tb),e(Jr,$b),e(Le,wb),e(Le,Gr),e(Gr,yb),e(Gr,Xr),e(Xr,Fb),e(Gr,Db),e(Le,xb),w(Ao,Le,null),e(Le,Bb),e(Le,mt),w(Yr,mt,null),e(mt,Mb),e(mt,ys),e(ys,Eb),e(ys,tl),e(tl,zb),e(ys,jb),e(ys,xd),e(xd,Cb),e(ys,qb),e(mt,Pb),w(Oo,mt,null),e(mt,Ab),w(No,mt,null),b(s,th,v),b(s,Fs,v),e(Fs,Lo),e(Lo,Bd),w(Zr,Bd,null),e(Fs,Ob),e(Fs,Md),e(Md,Nb),b(s,sh,v),b(s,Se,v),w(ea,Se,null),e(Se,Lb),e(Se,Ed),e(Ed,Sb),e(Se,Ib),e(Se,ta),e(ta,Wb),e(ta,sl),e(sl,Qb),e(ta,Rb),e(Se,Ub),e(Se,sa),e(sa,Hb),e(sa,oa),e(oa,Vb),e(sa,Kb),e(Se,Jb),w(So,Se,null),e(Se,Gb),e(Se,Ze),w(na,Ze,null),e(Ze,Xb),e(Ze,Ds),e(Ds,Yb),e(Ds,ol),e(ol,Zb),e(Ds,ev),e(Ds,zd),e(zd,tv),e(Ds,sv),e(Ze,ov),w(Io,Ze,null),e(Ze,nv),w(Wo,Ze,null),e(Ze,rv),w(Qo,Ze,null),b(s,oh,v),b(s,xs,v),e(xs,Ro),e(Ro,jd),w(ra,jd,null),e(xs,av),e(xs,Cd),e(Cd,iv),b(s,nh,v),b(s,Ie,v),w(aa,Ie,null),e(Ie,lv),e(Ie,Bs),e(Bs,dv),e(Bs,qd),e(qd,cv),e(Bs,pv),e(Bs,Pd),e(Pd,hv),e(Bs,uv),e(Ie,fv),e(Ie,ia),e(ia,mv),e(ia,nl),e(nl,gv),e(ia,_v),e(Ie,bv),e(Ie,la),e(la,vv),e(la,da),e(da,kv),e(la,Tv),e(Ie,$v),w(Uo,Ie,null),e(Ie,wv),e(Ie,et),w(ca,et,null),e(et,yv),e(et,Ms),e(Ms,Fv),e(Ms,rl),e(rl,Dv),e(Ms,xv),e(Ms,Ad),e(Ad,Bv),e(Ms,Mv),e(et,Ev),w(Ho,et,null),e(et,zv),w(Vo,et,null),e(et,jv),w(Ko,et,null),b(s,rh,v),b(s,Es,v),e(Es,Jo),e(Jo,Od),w(pa,Od,null),e(Es,Cv),e(Es,Nd),e(Nd,qv),b(s,ah,v),b(s,Me,v),w(ha,Me,null),e(Me,Pv),e(Me,Ld),e(Ld,Av),e(Me,Ov),e(Me,ua),e(ua,Nv),e(ua,al),e(al,Lv),e(ua,Sv),e(Me,Iv),e(Me,fa),e(fa,Wv),e(fa,ma),e(ma,Qv),e(fa,Rv),e(Me,Uv),e(Me,Sd),e(Sd,Hv),e(Me,Vv),e(Me,Mt),e(Mt,Id),e(Id,ga),e(ga,Kv),e(Mt,Jv),e(Mt,Wd),e(Wd,_a),e(_a,Gv),e(Mt,Xv),e(Mt,Qd),e(Qd,ba),e(ba,Yv),e(Mt,Zv),e(Mt,Rd),e(Rd,va),e(va,ek),e(Me,tk),e(Me,gt),w(ka,gt,null),e(gt,sk),e(gt,zs),e(zs,ok),e(zs,Ud),e(Ud,nk),e(zs,rk),e(zs,Hd),e(Hd,ak),e(zs,ik),e(gt,lk),w(Go,gt,null),e(gt,dk),w(Xo,gt,null),b(s,ih,v),b(s,js,v),e(js,Yo),e(Yo,Vd),w(Ta,Vd,null),e(js,ck),e(js,Kd),e(Kd,pk),b(s,lh,v),b(s,Ee,v),w($a,Ee,null),e(Ee,hk),e(Ee,wa),e(wa,uk),e(wa,Jd),e(Jd,fk),e(wa,mk),e(Ee,gk),e(Ee,ya),e(ya,_k),e(ya,il),e(il,bk),e(ya,vk),e(Ee,kk),e(Ee,Fa),e(Fa,Tk),e(Fa,Da),e(Da,$k),e(Fa,wk),e(Ee,yk),e(Ee,Gd),e(Gd,Fk),e(Ee,Dk),e(Ee,Et),e(Et,Xd),e(Xd,xa),e(xa,xk),e(Et,Bk),e(Et,Yd),e(Yd,Ba),e(Ba,Mk),e(Et,Ek),e(Et,Zd),e(Zd,Ma),e(Ma,zk),e(Et,jk),e(Et,ec),e(ec,Ea),e(Ea,Ck),e(Ee,qk),e(Ee,_t),w(za,_t,null),e(_t,Pk),e(_t,Cs),e(Cs,Ak),e(Cs,tc),e(tc,Ok),e(Cs,Nk),e(Cs,sc),e(sc,Lk),e(Cs,Sk),e(_t,Ik),w(Zo,_t,null),e(_t,Wk),w(en,_t,null),b(s,dh,v),b(s,qs,v),e(qs,tn),e(tn,oc),w(ja,oc,null),e(qs,Qk),e(qs,nc),e(nc,Rk),b(s,ch,v),b(s,ze,v),w(Ca,ze,null),e(ze,Uk),e(ze,rc),e(rc,Hk),e(ze,Vk),e(ze,qa),e(qa,Kk),e(qa,ll),e(ll,Jk),e(qa,Gk),e(ze,Xk),e(ze,Pa),e(Pa,Yk),e(Pa,Aa),e(Aa,Zk),e(Pa,eT),e(ze,tT),e(ze,ac),e(ac,sT),e(ze,oT),e(ze,zt),e(zt,ic),e(ic,Oa),e(Oa,nT),e(zt,rT),e(zt,lc),e(lc,Na),e(Na,aT),e(zt,iT),e(zt,dc),e(dc,La),e(La,lT),e(zt,dT),e(zt,cc),e(cc,Sa),e(Sa,cT),e(ze,pT),e(ze,bt),w(Ia,bt,null),e(bt,hT),e(bt,Ps),e(Ps,uT),e(Ps,pc),e(pc,fT),e(Ps,mT),e(Ps,hc),e(hc,gT),e(Ps,_T),e(bt,bT),w(sn,bt,null),e(bt,vT),w(on,bt,null),b(s,ph,v),b(s,As,v),e(As,nn),e(nn,uc),w(Wa,uc,null),e(As,kT),e(As,fc),e(fc,TT),b(s,hh,v),b(s,je,v),w(Qa,je,null),e(je,$T),e(je,mc),e(mc,wT),e(je,yT),e(je,Ra),e(Ra,FT),e(Ra,dl),e(dl,DT),e(Ra,xT),e(je,BT),e(je,Ua),e(Ua,MT),e(Ua,Ha),e(Ha,ET),e(Ua,zT),e(je,jT),e(je,gc),e(gc,CT),e(je,qT),e(je,jt),e(jt,_c),e(_c,Va),e(Va,PT),e(jt,AT),e(jt,bc),e(bc,Ka),e(Ka,OT),e(jt,NT),e(jt,vc),e(vc,Ja),e(Ja,LT),e(jt,ST),e(jt,kc),e(kc,Ga),e(Ga,IT),e(je,WT),e(je,vt),w(Xa,vt,null),e(vt,QT),e(vt,Os),e(Os,RT),e(Os,Tc),e(Tc,UT),e(Os,HT),e(Os,$c),e($c,VT),e(Os,KT),e(vt,JT),w(rn,vt,null),e(vt,GT),w(an,vt,null),b(s,uh,v),b(s,Ns,v),e(Ns,ln),e(ln,wc),w(Ya,wc,null),e(Ns,XT),e(Ns,yc),e(yc,YT),b(s,fh,v),b(s,Ce,v),w(Za,Ce,null),e(Ce,ZT),e(Ce,Fc),e(Fc,e$),e(Ce,t$),e(Ce,ei),e(ei,s$),e(ei,cl),e(cl,o$),e(ei,n$),e(Ce,r$),e(Ce,ti),e(ti,a$),e(ti,si),e(si,i$),e(ti,l$),e(Ce,d$),e(Ce,Dc),e(Dc,c$),e(Ce,p$),e(Ce,Ct),e(Ct,xc),e(xc,oi),e(oi,h$),e(Ct,u$),e(Ct,Bc),e(Bc,ni),e(ni,f$),e(Ct,m$),e(Ct,Mc),e(Mc,ri),e(ri,g$),e(Ct,_$),e(Ct,Ec),e(Ec,ai),e(ai,b$),e(Ce,v$),e(Ce,kt),w(ii,kt,null),e(kt,k$),e(kt,Ls),e(Ls,T$),e(Ls,zc),e(zc,$$),e(Ls,w$),e(Ls,jc),e(jc,y$),e(Ls,F$),e(kt,D$),w(dn,kt,null),e(kt,x$),w(cn,kt,null),b(s,mh,v),b(s,Ss,v),e(Ss,pn),e(pn,Cc),w(li,Cc,null),e(Ss,B$),e(Ss,qc),e(qc,M$),b(s,gh,v),b(s,qe,v),w(di,qe,null),e(qe,E$),e(qe,Is),e(Is,z$),e(Is,Pc),e(Pc,j$),e(Is,C$),e(Is,Ac),e(Ac,q$),e(Is,P$),e(qe,A$),e(qe,ci),e(ci,O$),e(ci,pl),e(pl,N$),e(ci,L$),e(qe,S$),e(qe,pi),e(pi,I$),e(pi,hi),e(hi,W$),e(pi,Q$),e(qe,R$),e(qe,Oc),e(Oc,U$),e(qe,H$),e(qe,qt),e(qt,Nc),e(Nc,ui),e(ui,V$),e(qt,K$),e(qt,Lc),e(Lc,fi),e(fi,J$),e(qt,G$),e(qt,Sc),e(Sc,mi),e(mi,X$),e(qt,Y$),e(qt,Ic),e(Ic,gi),e(gi,Z$),e(qe,ew),e(qe,Tt),w(_i,Tt,null),e(Tt,tw),e(Tt,Ws),e(Ws,sw),e(Ws,Wc),e(Wc,ow),e(Ws,nw),e(Ws,Qc),e(Qc,rw),e(Ws,aw),e(Tt,iw),w(hn,Tt,null),e(Tt,lw),w(un,Tt,null),_h=!0},p(s,[v]){const bi={};v&2&&(bi.$$scope={dirty:v,ctx:s}),Us.$set(bi);const Rc={};v&2&&(Rc.$$scope={dirty:v,ctx:s}),Xs.$set(Rc);const Uc={};v&2&&(Uc.$$scope={dirty:v,ctx:s}),Ys.$set(Uc);const Hc={};v&2&&(Hc.$$scope={dirty:v,ctx:s}),eo.$set(Hc);const vi={};v&2&&(vi.$$scope={dirty:v,ctx:s}),to.$set(vi);const Vc={};v&2&&(Vc.$$scope={dirty:v,ctx:s}),so.$set(Vc);const Kc={};v&2&&(Kc.$$scope={dirty:v,ctx:s}),no.$set(Kc);const Jc={};v&2&&(Jc.$$scope={dirty:v,ctx:s}),ro.$set(Jc);const Pt={};v&2&&(Pt.$$scope={dirty:v,ctx:s}),ao.$set(Pt);const Gc={};v&2&&(Gc.$$scope={dirty:v,ctx:s}),io.$set(Gc);const Xc={};v&2&&(Xc.$$scope={dirty:v,ctx:s}),lo.$set(Xc);const Yc={};v&2&&(Yc.$$scope={dirty:v,ctx:s}),po.$set(Yc);const Zc={};v&2&&(Zc.$$scope={dirty:v,ctx:s}),ho.$set(Zc);const ep={};v&2&&(ep.$$scope={dirty:v,ctx:s}),fo.$set(ep);const tp={};v&2&&(tp.$$scope={dirty:v,ctx:s}),mo.$set(tp);const sp={};v&2&&(sp.$$scope={dirty:v,ctx:s}),go.$set(sp);const ki={};v&2&&(ki.$$scope={dirty:v,ctx:s}),bo.$set(ki);const At={};v&2&&(At.$$scope={dirty:v,ctx:s}),vo.$set(At);const op={};v&2&&(op.$$scope={dirty:v,ctx:s}),ko.$set(op);const np={};v&2&&(np.$$scope={dirty:v,ctx:s}),$o.$set(np);const rp={};v&2&&(rp.$$scope={dirty:v,ctx:s}),wo.$set(rp);const Ti={};v&2&&(Ti.$$scope={dirty:v,ctx:s}),yo.$set(Ti);const ap={};v&2&&(ap.$$scope={dirty:v,ctx:s}),Do.$set(ap);const Ot={};v&2&&(Ot.$$scope={dirty:v,ctx:s}),xo.$set(Ot);const ip={};v&2&&(ip.$$scope={dirty:v,ctx:s}),Bo.$set(ip);const lp={};v&2&&(lp.$$scope={dirty:v,ctx:s}),Mo.$set(lp);const dp={};v&2&&(dp.$$scope={dirty:v,ctx:s}),zo.$set(dp);const $i={};v&2&&($i.$$scope={dirty:v,ctx:s}),jo.$set($i);const cp={};v&2&&(cp.$$scope={dirty:v,ctx:s}),Co.$set(cp);const pp={};v&2&&(pp.$$scope={dirty:v,ctx:s}),qo.$set(pp);const hp={};v&2&&(hp.$$scope={dirty:v,ctx:s}),Ao.$set(hp);const Nt={};v&2&&(Nt.$$scope={dirty:v,ctx:s}),Oo.$set(Nt);const Lt={};v&2&&(Lt.$$scope={dirty:v,ctx:s}),No.$set(Lt);const up={};v&2&&(up.$$scope={dirty:v,ctx:s}),So.$set(up);const fp={};v&2&&(fp.$$scope={dirty:v,ctx:s}),Io.$set(fp);const mp={};v&2&&(mp.$$scope={dirty:v,ctx:s}),Wo.$set(mp);const Qs={};v&2&&(Qs.$$scope={dirty:v,ctx:s}),Qo.$set(Qs);const gp={};v&2&&(gp.$$scope={dirty:v,ctx:s}),Uo.$set(gp);const _p={};v&2&&(_p.$$scope={dirty:v,ctx:s}),Ho.$set(_p);const wi={};v&2&&(wi.$$scope={dirty:v,ctx:s}),Vo.$set(wi);const bp={};v&2&&(bp.$$scope={dirty:v,ctx:s}),Ko.$set(bp);const vp={};v&2&&(vp.$$scope={dirty:v,ctx:s}),Go.$set(vp);const kp={};v&2&&(kp.$$scope={dirty:v,ctx:s}),Xo.$set(kp);const St={};v&2&&(St.$$scope={dirty:v,ctx:s}),Zo.$set(St);const Tp={};v&2&&(Tp.$$scope={dirty:v,ctx:s}),en.$set(Tp);const fn={};v&2&&(fn.$$scope={dirty:v,ctx:s}),sn.$set(fn);const $p={};v&2&&($p.$$scope={dirty:v,ctx:s}),on.$set($p);const wp={};v&2&&(wp.$$scope={dirty:v,ctx:s}),rn.$set(wp);const yi={};v&2&&(yi.$$scope={dirty:v,ctx:s}),an.$set(yi);const yp={};v&2&&(yp.$$scope={dirty:v,ctx:s}),dn.$set(yp);const Fi={};v&2&&(Fi.$$scope={dirty:v,ctx:s}),cn.$set(Fi);const Fp={};v&2&&(Fp.$$scope={dirty:v,ctx:s}),hn.$set(Fp);const Dp={};v&2&&(Dp.$$scope={dirty:v,ctx:s}),un.$set(Dp)},i(s){_h||(y(l.$$.fragment,s),y(X.$$.fragment,s),y(Pn.$$.fragment,s),y(An.$$.fragment,s),y(Us.$$.fragment,s),y(Nn.$$.fragment,s),y(Ln.$$.fragment,s),y(In.$$.fragment,s),y(Wn.$$.fragment,s),y(Un.$$.fragment,s),y(Hn.$$.fragment,s),y(Gn.$$.fragment,s),y(Xs.$$.fragment,s),y(Ys.$$.fragment,s),y(Xn.$$.fragment,s),y(Yn.$$.fragment,s),y(or.$$.fragment,s),y(eo.$$.fragment,s),y(to.$$.fragment,s),y(so.$$.fragment,s),y(nr.$$.fragment,s),y(rr.$$.fragment,s),y(dr.$$.fragment,s),y(no.$$.fragment,s),y(ro.$$.fragment,s),y(ao.$$.fragment,s),y(io.$$.fragment,s),y(lo.$$.fragment,s),y(cr.$$.fragment,s),y(pr.$$.fragment,s),y(mr.$$.fragment,s),y(po.$$.fragment,s),y(ho.$$.fragment,s),y(gr.$$.fragment,s),y(_r.$$.fragment,s),y(Tr.$$.fragment,s),y(fo.$$.fragment,s),y(mo.$$.fragment,s),y(go.$$.fragment,s),y($r.$$.fragment,s),y(wr.$$.fragment,s),y(xr.$$.fragment,s),y(bo.$$.fragment,s),y(vo.$$.fragment,s),y(ko.$$.fragment,s),y(Br.$$.fragment,s),y(Mr.$$.fragment,s),y($o.$$.fragment,s),y(Cr.$$.fragment,s),y(wo.$$.fragment,s),y(yo.$$.fragment,s),y(qr.$$.fragment,s),y(Pr.$$.fragment,s),y(Do.$$.fragment,s),y(Sr.$$.fragment,s),y(xo.$$.fragment,s),y(Bo.$$.fragment,s),y(Mo.$$.fragment,s),y(Ir.$$.fragment,s),y(Wr.$$.fragment,s),y(zo.$$.fragment,s),y(Hr.$$.fragment,s),y(jo.$$.fragment,s),y(Co.$$.fragment,s),y(qo.$$.fragment,s),y(Vr.$$.fragment,s),y(Kr.$$.fragment,s),y(Ao.$$.fragment,s),y(Yr.$$.fragment,s),y(Oo.$$.fragment,s),y(No.$$.fragment,s),y(Zr.$$.fragment,s),y(ea.$$.fragment,s),y(So.$$.fragment,s),y(na.$$.fragment,s),y(Io.$$.fragment,s),y(Wo.$$.fragment,s),y(Qo.$$.fragment,s),y(ra.$$.fragment,s),y(aa.$$.fragment,s),y(Uo.$$.fragment,s),y(ca.$$.fragment,s),y(Ho.$$.fragment,s),y(Vo.$$.fragment,s),y(Ko.$$.fragment,s),y(pa.$$.fragment,s),y(ha.$$.fragment,s),y(ka.$$.fragment,s),y(Go.$$.fragment,s),y(Xo.$$.fragment,s),y(Ta.$$.fragment,s),y($a.$$.fragment,s),y(za.$$.fragment,s),y(Zo.$$.fragment,s),y(en.$$.fragment,s),y(ja.$$.fragment,s),y(Ca.$$.fragment,s),y(Ia.$$.fragment,s),y(sn.$$.fragment,s),y(on.$$.fragment,s),y(Wa.$$.fragment,s),y(Qa.$$.fragment,s),y(Xa.$$.fragment,s),y(rn.$$.fragment,s),y(an.$$.fragment,s),y(Ya.$$.fragment,s),y(Za.$$.fragment,s),y(ii.$$.fragment,s),y(dn.$$.fragment,s),y(cn.$$.fragment,s),y(li.$$.fragment,s),y(di.$$.fragment,s),y(_i.$$.fragment,s),y(hn.$$.fragment,s),y(un.$$.fragment,s),_h=!0)},o(s){F(l.$$.fragment,s),F(X.$$.fragment,s),F(Pn.$$.fragment,s),F(An.$$.fragment,s),F(Us.$$.fragment,s),F(Nn.$$.fragment,s),F(Ln.$$.fragment,s),F(In.$$.fragment,s),F(Wn.$$.fragment,s),F(Un.$$.fragment,s),F(Hn.$$.fragment,s),F(Gn.$$.fragment,s),F(Xs.$$.fragment,s),F(Ys.$$.fragment,s),F(Xn.$$.fragment,s),F(Yn.$$.fragment,s),F(or.$$.fragment,s),F(eo.$$.fragment,s),F(to.$$.fragment,s),F(so.$$.fragment,s),F(nr.$$.fragment,s),F(rr.$$.fragment,s),F(dr.$$.fragment,s),F(no.$$.fragment,s),F(ro.$$.fragment,s),F(ao.$$.fragment,s),F(io.$$.fragment,s),F(lo.$$.fragment,s),F(cr.$$.fragment,s),F(pr.$$.fragment,s),F(mr.$$.fragment,s),F(po.$$.fragment,s),F(ho.$$.fragment,s),F(gr.$$.fragment,s),F(_r.$$.fragment,s),F(Tr.$$.fragment,s),F(fo.$$.fragment,s),F(mo.$$.fragment,s),F(go.$$.fragment,s),F($r.$$.fragment,s),F(wr.$$.fragment,s),F(xr.$$.fragment,s),F(bo.$$.fragment,s),F(vo.$$.fragment,s),F(ko.$$.fragment,s),F(Br.$$.fragment,s),F(Mr.$$.fragment,s),F($o.$$.fragment,s),F(Cr.$$.fragment,s),F(wo.$$.fragment,s),F(yo.$$.fragment,s),F(qr.$$.fragment,s),F(Pr.$$.fragment,s),F(Do.$$.fragment,s),F(Sr.$$.fragment,s),F(xo.$$.fragment,s),F(Bo.$$.fragment,s),F(Mo.$$.fragment,s),F(Ir.$$.fragment,s),F(Wr.$$.fragment,s),F(zo.$$.fragment,s),F(Hr.$$.fragment,s),F(jo.$$.fragment,s),F(Co.$$.fragment,s),F(qo.$$.fragment,s),F(Vr.$$.fragment,s),F(Kr.$$.fragment,s),F(Ao.$$.fragment,s),F(Yr.$$.fragment,s),F(Oo.$$.fragment,s),F(No.$$.fragment,s),F(Zr.$$.fragment,s),F(ea.$$.fragment,s),F(So.$$.fragment,s),F(na.$$.fragment,s),F(Io.$$.fragment,s),F(Wo.$$.fragment,s),F(Qo.$$.fragment,s),F(ra.$$.fragment,s),F(aa.$$.fragment,s),F(Uo.$$.fragment,s),F(ca.$$.fragment,s),F(Ho.$$.fragment,s),F(Vo.$$.fragment,s),F(Ko.$$.fragment,s),F(pa.$$.fragment,s),F(ha.$$.fragment,s),F(ka.$$.fragment,s),F(Go.$$.fragment,s),F(Xo.$$.fragment,s),F(Ta.$$.fragment,s),F($a.$$.fragment,s),F(za.$$.fragment,s),F(Zo.$$.fragment,s),F(en.$$.fragment,s),F(ja.$$.fragment,s),F(Ca.$$.fragment,s),F(Ia.$$.fragment,s),F(sn.$$.fragment,s),F(on.$$.fragment,s),F(Wa.$$.fragment,s),F(Qa.$$.fragment,s),F(Xa.$$.fragment,s),F(rn.$$.fragment,s),F(an.$$.fragment,s),F(Ya.$$.fragment,s),F(Za.$$.fragment,s),F(ii.$$.fragment,s),F(dn.$$.fragment,s),F(cn.$$.fragment,s),F(li.$$.fragment,s),F(di.$$.fragment,s),F(_i.$$.fragment,s),F(hn.$$.fragment,s),F(un.$$.fragment,s),_h=!1},d(s){t(d),s&&t(g),s&&t(c),D(l),s&&t(V),s&&t(E),D(X),s&&t(le),s&&t(N),s&&t(de),s&&t(ee),s&&t(ce),s&&t(L),s&&t(q),s&&t(te),s&&t(pe),s&&t(k),s&&t(Bp),s&&t(pt),s&&t(Mp),s&&t(es),D(Pn),s&&t(Ep),s&&t(lt),D(An),D(Us),s&&t(zp),s&&t(ss),D(Nn),s&&t(jp),s&&t(dt),D(Ln),s&&t(Cp),s&&t(os),D(In),s&&t(qp),s&&t(ct),D(Wn),s&&t(Pp),s&&t(ns),D(Un),s&&t(Ap),s&&t(We),D(Hn),D(Gn),D(Xs),D(Ys),s&&t(Op),s&&t(as),D(Xn),s&&t(Np),s&&t(Qe),D(Yn),D(or),D(eo),D(to),D(so),s&&t(Lp),s&&t(ls),D(nr),s&&t(Sp),s&&t(Re),D(rr),D(dr),D(no),D(ro),D(ao),D(io),D(lo),s&&t(Ip),s&&t(cs),D(cr),s&&t(Wp),s&&t(Ue),D(pr),D(mr),D(po),D(ho),s&&t(Qp),s&&t(hs),D(gr),s&&t(Rp),s&&t(He),D(_r),D(Tr),D(fo),D(mo),D(go),s&&t(Up),s&&t(fs),D($r),s&&t(Hp),s&&t(Ve),D(wr),D(xr),D(bo),D(vo),D(ko),s&&t(Vp),s&&t(_s),D(Br),s&&t(Kp),s&&t(Ae),D(Mr),D($o),D(Cr),D(wo),D(yo),s&&t(Jp),s&&t(vs),D(qr),s&&t(Gp),s&&t(Oe),D(Pr),D(Do),D(Sr),D(xo),D(Bo),D(Mo),s&&t(Xp),s&&t(Ts),D(Ir),s&&t(Yp),s&&t(Ne),D(Wr),D(zo),D(Hr),D(jo),D(Co),D(qo),s&&t(Zp),s&&t(ws),D(Vr),s&&t(eh),s&&t(Le),D(Kr),D(Ao),D(Yr),D(Oo),D(No),s&&t(th),s&&t(Fs),D(Zr),s&&t(sh),s&&t(Se),D(ea),D(So),D(na),D(Io),D(Wo),D(Qo),s&&t(oh),s&&t(xs),D(ra),s&&t(nh),s&&t(Ie),D(aa),D(Uo),D(ca),D(Ho),D(Vo),D(Ko),s&&t(rh),s&&t(Es),D(pa),s&&t(ah),s&&t(Me),D(ha),D(ka),D(Go),D(Xo),s&&t(ih),s&&t(js),D(Ta),s&&t(lh),s&&t(Ee),D($a),D(za),D(Zo),D(en),s&&t(dh),s&&t(qs),D(ja),s&&t(ch),s&&t(ze),D(Ca),D(Ia),D(sn),D(on),s&&t(ph),s&&t(As),D(Wa),s&&t(hh),s&&t(je),D(Qa),D(Xa),D(rn),D(an),s&&t(uh),s&&t(Ns),D(Ya),s&&t(fh),s&&t(Ce),D(Za),D(ii),D(dn),D(cn),s&&t(mh),s&&t(Ss),D(li),s&&t(gh),s&&t(qe),D(di),D(_i),D(hn),D(un)}}}const nD={local:"distilbert",sections:[{local:"overview",title:"Overview"},{local:"transformers.DistilBertConfig",title:"DistilBertConfig"},{local:"transformers.DistilBertTokenizer",title:"DistilBertTokenizer"},{local:"transformers.DistilBertTokenizerFast",title:"DistilBertTokenizerFast"},{local:"transformers.DistilBertModel",title:"DistilBertModel"},{local:"transformers.DistilBertForMaskedLM",title:"DistilBertForMaskedLM"},{local:"transformers.DistilBertForSequenceClassification",title:"DistilBertForSequenceClassification"},{local:"transformers.DistilBertForMultipleChoice",title:"DistilBertForMultipleChoice"},{local:"transformers.DistilBertForTokenClassification",title:"DistilBertForTokenClassification"},{local:"transformers.DistilBertForQuestionAnswering",title:"DistilBertForQuestionAnswering"},{local:"transformers.TFDistilBertModel",title:"TFDistilBertModel"},{local:"transformers.TFDistilBertForMaskedLM",title:"TFDistilBertForMaskedLM"},{local:"transformers.TFDistilBertForSequenceClassification",title:"TFDistilBertForSequenceClassification"},{local:"transformers.TFDistilBertForMultipleChoice",title:"TFDistilBertForMultipleChoice"},{local:"transformers.TFDistilBertForTokenClassification",title:"TFDistilBertForTokenClassification"},{local:"transformers.TFDistilBertForQuestionAnswering",title:"TFDistilBertForQuestionAnswering"},{local:"transformers.FlaxDistilBertModel",title:"FlaxDistilBertModel"},{local:"transformers.FlaxDistilBertForMaskedLM",title:"FlaxDistilBertForMaskedLM"},{local:"transformers.FlaxDistilBertForSequenceClassification",title:"FlaxDistilBertForSequenceClassification"},{local:"transformers.FlaxDistilBertForMultipleChoice",title:"FlaxDistilBertForMultipleChoice"},{local:"transformers.FlaxDistilBertForTokenClassification",title:"FlaxDistilBertForTokenClassification"},{local:"transformers.FlaxDistilBertForQuestionAnswering",title:"FlaxDistilBertForQuestionAnswering"}],title:"DistilBERT"};function rD(x){return oF(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class hD extends Zy{constructor(d){super();eF(this,d,rD,oD,tF,{})}}export{hD as default,nD as metadata};
