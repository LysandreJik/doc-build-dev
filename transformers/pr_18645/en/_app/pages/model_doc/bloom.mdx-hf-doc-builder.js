import{S as Ll,i as Nl,s as Il,e as n,k as u,w as k,t as i,M as Dl,c as a,d as t,m as f,a as r,x as $,h as d,b as m,G as e,g as v,y,q as w,o as T,B,v as Wl,L as ee}from"../../chunks/vendor-hf-doc-builder.js";import{T as Kt}from"../../chunks/Tip-hf-doc-builder.js";import{D as X}from"../../chunks/Docstring-hf-doc-builder.js";import{C as oe}from"../../chunks/CodeBlock-hf-doc-builder.js";import{I as ge}from"../../chunks/IconCopyLink-hf-doc-builder.js";import{E as Z}from"../../chunks/ExampleCodeBlock-hf-doc-builder.js";function Hl(C){let l,_,c,p,b;return p=new oe({props:{code:`from transformers import BloomModel, BloomConfig

# Initializing a Bloom configuration
configuration = BloomConfig()

# Initializing a model from the configuration
model = BloomModel(configuration)

# Accessing the model configuration
configuration = model.config`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BloomModel, BloomConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Bloom configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = BloomConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BloomModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`}}),{c(){l=n("p"),_=i("Example:"),c=u(),k(p.$$.fragment)},l(s){l=a(s,"P",{});var h=r(l);_=d(h,"Example:"),h.forEach(t),c=f(s),$(p.$$.fragment,s)},m(s,h){v(s,l,h),e(l,_),v(s,c,h),y(p,s,h),b=!0},p:ee,i(s){b||(w(p.$$.fragment,s),b=!0)},o(s){T(p.$$.fragment,s),b=!1},d(s){s&&t(l),s&&t(c),B(p,s)}}}function Ul(C){let l,_,c,p,b;return{c(){l=n("p"),_=i("Although the recipe for forward pass needs to be defined within this function, one should call the "),c=n("code"),p=i("Module"),b=i(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(s){l=a(s,"P",{});var h=r(l);_=d(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),c=a(h,"CODE",{});var F=r(c);p=d(F,"Module"),F.forEach(t),b=d(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(t)},m(s,h){v(s,l,h),e(l,_),e(l,c),e(c,p),e(l,b)},d(s){s&&t(l)}}}function Vl(C){let l,_,c,p,b;return p=new oe({props:{code:`from transformers import BloomTokenizerFast, BloomModel
import torch

tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
model = BloomModel.from_pretrained("bigscience/bloom-560m")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BloomTokenizerFast, BloomModel
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BloomTokenizerFast.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom-560m&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BloomModel.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom-560m&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`}}),{c(){l=n("p"),_=i("Example:"),c=u(),k(p.$$.fragment)},l(s){l=a(s,"P",{});var h=r(l);_=d(h,"Example:"),h.forEach(t),c=f(s),$(p.$$.fragment,s)},m(s,h){v(s,l,h),e(l,_),v(s,c,h),y(p,s,h),b=!0},p:ee,i(s){b||(w(p.$$.fragment,s),b=!0)},o(s){T(p.$$.fragment,s),b=!1},d(s){s&&t(l),s&&t(c),B(p,s)}}}function Rl(C){let l,_,c,p,b;return p=new oe({props:{code:`from transformers import BloomTokenizerFast
tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom")
tokenizer("Hello world")['input_ids']
tokenizer(" Hello world")['input_ids']`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt;</span> <span class="language-python"><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BloomTokenizerFast</span>
<span class="hljs-meta">&gt;&gt;&gt;</span> <span class="language-python">tokenizer = BloomTokenizerFast.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom&quot;</span>)</span>
<span class="hljs-meta">&gt;&gt;&gt;</span> <span class="language-python">tokenizer(<span class="hljs-string">&quot;Hello world&quot;</span>)[<span class="hljs-string">&#x27;input_ids&#x27;</span>]</span>
[15496, 995]
<span class="hljs-meta">&gt;&gt;&gt;</span> <span class="language-python">tokenizer(<span class="hljs-string">&quot; Hello world&quot;</span>)[<span class="hljs-string">&#x27;input_ids&#x27;</span>]</span>
[18435, 995]`}}),{c(){l=n("p"),_=i("be encoded differently whether it is at the beginning of the sentence (without space) or not:"),c=u(),k(p.$$.fragment)},l(s){l=a(s,"P",{});var h=r(l);_=d(h,"be encoded differently whether it is at the beginning of the sentence (without space) or not:"),h.forEach(t),c=f(s),$(p.$$.fragment,s)},m(s,h){v(s,l,h),e(l,_),v(s,c,h),y(p,s,h),b=!0},p:ee,i(s){b||(w(p.$$.fragment,s),b=!0)},o(s){T(p.$$.fragment,s),b=!1},d(s){s&&t(l),s&&t(c),B(p,s)}}}function Yl(C){let l,_,c,p,b,s,h,F;return{c(){l=n("p"),_=i("When used with "),c=n("code"),p=i("is_split_into_words=True"),b=i(", this tokenizer needs to be instantiated with "),s=n("code"),h=i("add_prefix_space=True"),F=i(".")},l(te){l=a(te,"P",{});var I=r(l);_=d(I,"When used with "),c=a(I,"CODE",{});var H=r(c);p=d(H,"is_split_into_words=True"),H.forEach(t),b=d(I,", this tokenizer needs to be instantiated with "),s=a(I,"CODE",{});var R=r(s);h=d(R,"add_prefix_space=True"),R.forEach(t),F=d(I,"."),I.forEach(t)},m(te,I){v(te,l,I),e(l,_),e(l,c),e(c,p),e(l,b),e(l,s),e(s,h),e(l,F)},d(te){te&&t(l)}}}function Gl(C){let l,_,c,p,b;return{c(){l=n("p"),_=i("Although the recipe for forward pass needs to be defined within this function, one should call the "),c=n("code"),p=i("Module"),b=i(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(s){l=a(s,"P",{});var h=r(l);_=d(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),c=a(h,"CODE",{});var F=r(c);p=d(F,"Module"),F.forEach(t),b=d(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(t)},m(s,h){v(s,l,h),e(l,_),e(l,c),e(c,p),e(l,b)},d(s){s&&t(l)}}}function Jl(C){let l,_,c,p,b;return p=new oe({props:{code:`import torch
from transformers import BloomTokenizerFast, BloomForCausalLM

tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
logits = outputs.logits`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BloomTokenizerFast, BloomForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BloomTokenizerFast.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom-560m&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BloomForCausalLM.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom-560m&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs, labels=inputs[<span class="hljs-string">&quot;input_ids&quot;</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`}}),{c(){l=n("p"),_=i("Example:"),c=u(),k(p.$$.fragment)},l(s){l=a(s,"P",{});var h=r(l);_=d(h,"Example:"),h.forEach(t),c=f(s),$(p.$$.fragment,s)},m(s,h){v(s,l,h),e(l,_),v(s,c,h),y(p,s,h),b=!0},p:ee,i(s){b||(w(p.$$.fragment,s),b=!0)},o(s){T(p.$$.fragment,s),b=!1},d(s){s&&t(l),s&&t(c),B(p,s)}}}function Kl(C){let l,_,c,p,b;return{c(){l=n("p"),_=i("Although the recipe for forward pass needs to be defined within this function, one should call the "),c=n("code"),p=i("Module"),b=i(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(s){l=a(s,"P",{});var h=r(l);_=d(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),c=a(h,"CODE",{});var F=r(c);p=d(F,"Module"),F.forEach(t),b=d(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(t)},m(s,h){v(s,l,h),e(l,_),e(l,c),e(c,p),e(l,b)},d(s){s&&t(l)}}}function Ql(C){let l,_,c,p,b;return p=new oe({props:{code:`import torch
from transformers import BloomTokenizerFast, BloomForSequenceClassification

tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
model = BloomForSequenceClassification.from_pretrained("bigscience/bloom-560m")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BloomTokenizerFast, BloomForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BloomTokenizerFast.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom-560m&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BloomForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom-560m&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
`}}),{c(){l=n("p"),_=i("Example of single-label classification:"),c=u(),k(p.$$.fragment)},l(s){l=a(s,"P",{});var h=r(l);_=d(h,"Example of single-label classification:"),h.forEach(t),c=f(s),$(p.$$.fragment,s)},m(s,h){v(s,l,h),e(l,_),v(s,c,h),y(p,s,h),b=!0},p:ee,i(s){b||(w(p.$$.fragment,s),b=!0)},o(s){T(p.$$.fragment,s),b=!1},d(s){s&&t(l),s&&t(c),B(p,s)}}}function Xl(C){let l,_;return l=new oe({props:{code:`# To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`
num_labels = len(model.config.id2label)
model = BloomForSequenceClassification.from_pretrained("bigscience/bloom-560m", num_labels=num_labels)

labels = torch.tensor(1)
loss = model(**inputs, labels=labels).loss
round(loss.item(), 2)
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BloomForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom-560m&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">1</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
`}}),{c(){k(l.$$.fragment)},l(c){$(l.$$.fragment,c)},m(c,p){y(l,c,p),_=!0},p:ee,i(c){_||(w(l.$$.fragment,c),_=!0)},o(c){T(l.$$.fragment,c),_=!1},d(c){B(l,c)}}}function Zl(C){let l,_,c,p,b;return p=new oe({props:{code:`import torch
from transformers import BloomTokenizerFast, BloomForSequenceClassification

tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
model = BloomForSequenceClassification.from_pretrained("bigscience/bloom-560m", problem_type="multi_label_classification")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BloomTokenizerFast, BloomForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BloomTokenizerFast.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom-560m&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BloomForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom-560m&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
`}}),{c(){l=n("p"),_=i("Example of multi-label classification:"),c=u(),k(p.$$.fragment)},l(s){l=a(s,"P",{});var h=r(l);_=d(h,"Example of multi-label classification:"),h.forEach(t),c=f(s),$(p.$$.fragment,s)},m(s,h){v(s,l,h),e(l,_),v(s,c,h),y(p,s,h),b=!0},p:ee,i(s){b||(w(p.$$.fragment,s),b=!0)},o(s){T(p.$$.fragment,s),b=!1},d(s){s&&t(l),s&&t(c),B(p,s)}}}function ei(C){let l,_;return l=new oe({props:{code:`# To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`
num_labels = len(model.config.id2label)
model = BloomForSequenceClassification.from_pretrained(
    "bigscience/bloom-560m", num_labels=num_labels, problem_type="multi_label_classification"
)

labels = torch.nn.functional.one_hot(torch.tensor([predicted_class_id]), num_classes=num_labels).to(
    torch.float
)
loss = model(**inputs, labels=labels).loss
loss.backward()`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BloomForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;bigscience/bloom-560m&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.nn.functional.one_hot(torch.tensor([predicted_class_id]), num_classes=num_labels).to(
<span class="hljs-meta">... </span>    torch.<span class="hljs-built_in">float</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span>loss.backward()`}}),{c(){k(l.$$.fragment)},l(c){$(l.$$.fragment,c)},m(c,p){y(l,c,p),_=!0},p:ee,i(c){_||(w(l.$$.fragment,c),_=!0)},o(c){T(l.$$.fragment,c),_=!1},d(c){B(l,c)}}}function oi(C){let l,_,c,p,b;return{c(){l=n("p"),_=i("Although the recipe for forward pass needs to be defined within this function, one should call the "),c=n("code"),p=i("Module"),b=i(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(s){l=a(s,"P",{});var h=r(l);_=d(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),c=a(h,"CODE",{});var F=r(c);p=d(F,"Module"),F.forEach(t),b=d(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(t)},m(s,h){v(s,l,h),e(l,_),e(l,c),e(c,p),e(l,b)},d(s){s&&t(l)}}}function ti(C){let l,_,c,p,b;return p=new oe({props:{code:`from transformers import BloomTokenizerFast, BloomForTokenClassification
import torch

tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
model = BloomForTokenClassification.from_pretrained("bigscience/bloom-560m")

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
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BloomTokenizerFast, BloomForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BloomTokenizerFast.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom-560m&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BloomForTokenClassification.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom-560m&quot;</span>)

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
`}}),{c(){l=n("p"),_=i("Example:"),c=u(),k(p.$$.fragment)},l(s){l=a(s,"P",{});var h=r(l);_=d(h,"Example:"),h.forEach(t),c=f(s),$(p.$$.fragment,s)},m(s,h){v(s,l,h),e(l,_),v(s,c,h),y(p,s,h),b=!0},p:ee,i(s){b||(w(p.$$.fragment,s),b=!0)},o(s){T(p.$$.fragment,s),b=!1},d(s){s&&t(l),s&&t(c),B(p,s)}}}function si(C){let l,_;return l=new oe({props:{code:`labels = predicted_token_class_ids
loss = model(**inputs, labels=labels).loss
round(loss.item(), 2)
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span>labels = predicted_token_class_ids
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
`}}),{c(){k(l.$$.fragment)},l(c){$(l.$$.fragment,c)},m(c,p){y(l,c,p),_=!0},p:ee,i(c){_||(w(l.$$.fragment,c),_=!0)},o(c){T(l.$$.fragment,c),_=!1},d(c){B(l,c)}}}function ni(C){let l,_,c,p,b,s,h,F,te,I,H,R,st,Ue,js,nt,Ps,Qt,_e,Os,Ve,Ss,As,Xt,q,at,Re,Ls,Ns,rt,Ye,Is,Ds,lt,Ge,Ws,Hs,it,Je,Us,Vs,dt,Ke,Rs,Ys,So,Qe,Gs,Js,Zt,se,be,ct,Xe,Ks,pt,Qs,es,ve,Xs,Ze,Zs,en,os,ne,ke,mt,eo,on,ht,tn,ts,D,oo,sn,ae,nn,Ao,an,rn,to,ln,dn,cn,re,pn,Lo,mn,hn,No,un,fn,gn,$e,ss,le,ye,ut,so,_n,ft,bn,ns,x,no,vn,gt,kn,$n,ao,yn,Io,wn,Tn,Bn,ro,Cn,lo,Fn,zn,En,U,io,Mn,ie,qn,Do,xn,jn,_t,Pn,On,Sn,we,An,Te,as,de,Be,bt,co,Ln,vt,Nn,rs,z,po,In,mo,Dn,kt,Wn,Hn,Un,$t,Vn,Rn,Ce,Yn,ho,Gn,yt,Jn,Kn,Qn,Fe,Xn,uo,Zn,Wo,ea,oa,ls,ce,ze,wt,fo,ta,Tt,sa,is,j,go,na,Bt,aa,ra,_o,la,Ho,ia,da,ca,bo,pa,vo,ma,ha,ua,V,ko,fa,pe,ga,Uo,_a,ba,Ct,va,ka,$a,Ee,ya,Me,ds,me,qe,Ft,$o,wa,zt,Ta,cs,E,yo,Ba,Et,Ca,Fa,Vo,Ro,za,Ea,Ma,W,qa,Mt,xa,ja,qt,Pa,Oa,xt,Sa,Aa,jt,La,Na,Ia,wo,Da,Yo,Wa,Ha,Ua,To,Va,Bo,Ra,Ya,Ga,M,Co,Ja,he,Ka,Go,Qa,Xa,Pt,Za,er,or,xe,tr,je,sr,Pe,nr,Oe,ar,Se,ps,ue,Ae,Ot,Fo,rr,St,lr,ms,P,zo,ir,At,dr,cr,Eo,pr,Jo,mr,hr,ur,Mo,fr,qo,gr,_r,br,S,xo,vr,fe,kr,Ko,$r,yr,Lt,wr,Tr,Br,Le,Cr,Ne,Fr,Ie,hs;return s=new ge({}),Ue=new ge({}),Xe=new ge({}),eo=new ge({}),oo=new X({props:{name:"class transformers.BloomConfig",anchor:"transformers.BloomConfig",parameters:[{name:"vocab_size",val:" = 250880"},{name:"hidden_size",val:" = 64"},{name:"n_layer",val:" = 2"},{name:"n_head",val:" = 8"},{name:"layer_norm_epsilon",val:" = 1e-05"},{name:"initializer_range",val:" = 0.02"},{name:"use_cache",val:" = False"},{name:"bos_token_id",val:" = 1"},{name:"eos_token_id",val:" = 2"},{name:"apply_residual_connection_post_layernorm",val:" = False"},{name:"hidden_dropout",val:" = 0.0"},{name:"attention_dropout",val:" = 0.0"},{name:"pretraining_tp",val:" = 1"},{name:"slow_but_exact",val:" = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BloomConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 50257) &#x2014;
Vocabulary size of the Bloom model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/pr_18645/en/model_doc/bloom#transformers.BloomModel">BloomModel</a>.`,name:"vocab_size"},{anchor:"transformers.BloomConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the embeddings and hidden states.`,name:"hidden_size"},{anchor:"transformers.BloomConfig.n_layer",description:`<strong>n_layer</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"n_layer"},{anchor:"transformers.BloomConfig.n_head",description:`<strong>n_head</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"n_head"},{anchor:"transformers.BloomConfig.attn_pdrop",description:`<strong>attn_pdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention.`,name:"attn_pdrop"},{anchor:"transformers.BloomConfig.layer_norm_epsilon",description:`<strong>layer_norm_epsilon</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-5) &#x2014;
The epsilon to use in the layer normalization layers.`,name:"layer_norm_epsilon"},{anchor:"transformers.BloomConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.BloomConfig.apply_residual_connection_post_layernorm",description:`<strong>apply_residual_connection_post_layernorm</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
If enabled, use the layer norm of the hidden states as the residual in the transformer blocks`,name:"apply_residual_connection_post_layernorm"},{anchor:"transformers.BloomConfig.skip_bias_add",description:`<strong>skip_bias_add</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
If set to <code>True</code>, it will skip bias add for each linear layer in the transformer blocks`,name:"skip_bias_add"},{anchor:"transformers.BloomConfig.skip_bias_add_qkv",description:`<strong>skip_bias_add_qkv</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
If set to <code>True</code>, it will skip bias add for the first linear layer in the transformer blocks`,name:"skip_bias_add_qkv"},{anchor:"transformers.BloomConfig.hidden_dropout",description:`<strong>hidden_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
Dropout rate of the dropout function on the bias dropout.`,name:"hidden_dropout"},{anchor:"transformers.BloomConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
Dropout rate applied to the attention probs`,name:"attention_dropout"},{anchor:"transformers.BloomConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models).`,name:"use_cache"},{anchor:"transformers.BloomConfig.pretraining_tp",description:`<strong>pretraining_tp</strong> (<code>int</code>, <em>optional</em>, defaults to <code>1</code>) &#x2014;
Experimental feature. Tensor parallelism rank used during pretraining with Megatron. Please refer to <a href="https://huggingface.co/docs/transformers/parallelism" rel="nofollow">this
document</a> to understand more about it. This value is
necessary to ensure exact reproducibility of the pretraining results. Please refer to <a href="https://github.com/pytorch/pytorch/issues/76232" rel="nofollow">this
issue</a>. Note also that this is enabled only when
<code>slow_but_exact=True</code>.`,name:"pretraining_tp"},{anchor:"transformers.BloomConfig.slow_but_exact",description:`<strong>slow_but_exact</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Experimental feature. Whether to use slow but exact implementation of the attention mechanism. While
merging the TP rank tensors, due to slicing operations the results may be slightly different between the
model trained on Megatron and our model. Please refer to <a href="https://github.com/pytorch/pytorch/issues/76232" rel="nofollow">this
issue</a>. A solution to obtain more accurate results is to
enable this feature. Enabling this will hurt the computational time of the inference. Will be probably
resolved in the future once the main model has been fine-tuned with TP_rank=1.`,name:"slow_but_exact"}],source:"https://github.com/huggingface/transformers/blob/vr_18645/src/transformers/models/bloom/configuration_bloom.py#L42"}}),$e=new Z({props:{anchor:"transformers.BloomConfig.example",$$slots:{default:[Hl]},$$scope:{ctx:C}}}),so=new ge({}),no=new X({props:{name:"class transformers.BloomModel",anchor:"transformers.BloomModel",parameters:[{name:"config",val:": BloomConfig"}],parametersDescription:[{anchor:"transformers.BloomModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18645/en/model_doc/bloom#transformers.BloomConfig">BloomConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18645/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18645/src/transformers/models/bloom/modeling_bloom.py#L583"}}),io=new X({props:{name:"forward",anchor:"transformers.BloomModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Union[typing.Tuple[typing.Tuple[torch.Tensor, torch.Tensor], ...], NoneType] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**deprecated_arguments",val:""}],parametersDescription:[{anchor:"transformers.BloomModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else <code>past_key_values[0][0].shape[2]</code>
(<code>sequence_length</code> of input past key value states). Indices of input sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18645/en/model_doc/bloom#transformers.BloomTokenizerFast">BloomTokenizerFast</a>. See <a href="/docs/transformers/pr_18645/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18645/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BloomModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Tuple[Tuple[torch.Tensor]]</code> of length <code>config.n_layers</code>) &#x2014;
Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
<code>past_key_values</code> output below). Can be used to speed up sequential decoding. The <code>input_ids</code> which have
their past given to this model should not be passed as <code>input_ids</code> as they have already been computed.</p>
<p>Each element of <code>past_key_values</code> is a tuple (past_key, past_value):</p>
<ul>
<li>past_key: [batch_size * num_heads, head_dim, kv_length]</li>
<li>past_value: [batch_size * num_heads, kv_length, head_dim]</li>
</ul>`,name:"past_key_values"},{anchor:"transformers.BloomModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BloomModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BloomModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>past_key_values</code> is used, optionally only the last <code>inputs_embeds</code> have to be input (see
<code>past_key_values</code>).`,name:"inputs_embeds"},{anchor:"transformers.BloomModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.BloomModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BloomModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BloomModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18645/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18645/src/transformers/models/bloom/modeling_bloom.py#L633",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18645/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18645/en/model_doc/bloom#transformers.BloomConfig"
>BloomConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) \u2014 Sequence of hidden-states at the output of the last layer of the model.</p>
<p>If <code>past_key_values</code> is used only the last hidden-state of the sequences of shape <code>(batch_size, 1, hidden_size)</code> is output.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>tuple(tuple(torch.FloatTensor))</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) \u2014 Tuple of <code>tuple(torch.FloatTensor)</code> of length <code>config.n_layers</code>, with each tuple having 2 tensors of shape
<code>(batch_size, num_heads, sequence_length, embed_size_per_head)</code>) and optionally if
<code>config.is_encoder_decoder=True</code> 2 additional tensors of shape <code>(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)</code>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
<code>config.is_encoder_decoder=True</code> in the cross-attention blocks) that can be used (see <code>past_key_values</code>
input) to speed up sequential decoding.</p>
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
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18645/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),we=new Kt({props:{$$slots:{default:[Ul]},$$scope:{ctx:C}}}),Te=new Z({props:{anchor:"transformers.BloomModel.forward.example",$$slots:{default:[Vl]},$$scope:{ctx:C}}}),co=new ge({}),po=new X({props:{name:"class transformers.BloomTokenizerFast",anchor:"transformers.BloomTokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"merges_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"unk_token",val:" = '<unk>'"},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"pad_token",val:" = '<pad>'"},{name:"add_prefix_space",val:" = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BloomTokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.BloomTokenizerFast.merges_file",description:`<strong>merges_file</strong> (<code>str</code>) &#x2014;
Path to the merges file.`,name:"merges_file"},{anchor:"transformers.BloomTokenizerFast.errors",description:`<strong>errors</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;replace&quot;</code>) &#x2014;
Paradigm to follow when decoding bytes to UTF-8. See
<a href="https://docs.python.org/3/library/stdtypes.html#bytes.decode" rel="nofollow">bytes.decode</a> for more information.`,name:"errors"},{anchor:"transformers.BloomTokenizerFast.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&lt;|endoftext|&gt;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.BloomTokenizerFast.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&lt;|endoftext|&gt;</code>) &#x2014;
The beginning of sequence token.`,name:"bos_token"},{anchor:"transformers.BloomTokenizerFast.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&lt;|endoftext|&gt;</code>) &#x2014;
The end of sequence token.`,name:"eos_token"},{anchor:"transformers.BloomTokenizerFast.add_prefix_space",description:`<strong>add_prefix_space</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to add an initial space to the input. This allows to treat the leading word just as any
other word. (Bloom tokenizer detect beginning of words by the preceding space).`,name:"add_prefix_space"},{anchor:"transformers.BloomTokenizerFast.trim_offsets",description:`<strong>trim_offsets</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the post-processing step should trim offsets to avoid including whitespaces.`,name:"trim_offsets"}],source:"https://github.com/huggingface/transformers/blob/vr_18645/src/transformers/models/bloom/tokenization_bloom_fast.py#L49"}}),Ce=new Z({props:{anchor:"transformers.BloomTokenizerFast.example",$$slots:{default:[Rl]},$$scope:{ctx:C}}}),Fe=new Kt({props:{$$slots:{default:[Yl]},$$scope:{ctx:C}}}),fo=new ge({}),go=new X({props:{name:"class transformers.BloomForCausalLM",anchor:"transformers.BloomForCausalLM",parameters:[{name:"config",val:": BloomConfig"}],parametersDescription:[{anchor:"transformers.BloomForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18645/en/model_doc/bloom#transformers.BloomConfig">BloomConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18645/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18645/src/transformers/models/bloom/modeling_bloom.py#L785"}}),ko=new X({props:{name:"forward",anchor:"transformers.BloomForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Union[typing.Tuple[typing.Tuple[torch.Tensor, torch.Tensor], ...], NoneType] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**deprecated_arguments",val:""}],parametersDescription:[{anchor:"transformers.BloomForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else <code>past_key_values[0][0].shape[2]</code>
(<code>sequence_length</code> of input past key value states). Indices of input sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18645/en/model_doc/bloom#transformers.BloomTokenizerFast">BloomTokenizerFast</a>. See <a href="/docs/transformers/pr_18645/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18645/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BloomForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Tuple[Tuple[torch.Tensor]]</code> of length <code>config.n_layers</code>) &#x2014;
Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
<code>past_key_values</code> output below). Can be used to speed up sequential decoding. The <code>input_ids</code> which have
their past given to this model should not be passed as <code>input_ids</code> as they have already been computed.</p>
<p>Each element of <code>past_key_values</code> is a tuple (past_key, past_value):</p>
<ul>
<li>past_key: [batch_size * num_heads, head_dim, kv_length]</li>
<li>past_value: [batch_size * num_heads, kv_length, head_dim]</li>
</ul>`,name:"past_key_values"},{anchor:"transformers.BloomForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BloomForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BloomForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>past_key_values</code> is used, optionally only the last <code>inputs_embeds</code> have to be input (see
<code>past_key_values</code>).`,name:"inputs_embeds"},{anchor:"transformers.BloomForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.BloomForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BloomForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BloomForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18645/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BloomForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for language modeling. Note that the labels <strong>are shifted</strong> inside the model, i.e. you can set
<code>labels = input_ids</code> Indices are selected in <code>[-100, 0, ..., config.vocab_size]</code> All labels set to <code>-100</code>
are ignored (masked), the loss is only computed for labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18645/src/transformers/models/bloom/modeling_bloom.py#L820",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18645/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18645/en/model_doc/bloom#transformers.BloomConfig"
>BloomConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18645/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Ee=new Kt({props:{$$slots:{default:[Gl]},$$scope:{ctx:C}}}),Me=new Z({props:{anchor:"transformers.BloomForCausalLM.forward.example",$$slots:{default:[Jl]},$$scope:{ctx:C}}}),$o=new ge({}),yo=new X({props:{name:"class transformers.BloomForSequenceClassification",anchor:"transformers.BloomForSequenceClassification",parameters:[{name:"config",val:": BloomConfig"}],parametersDescription:[{anchor:"transformers.BloomForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18645/en/model_doc/bloom#transformers.BloomConfig">BloomConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18645/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18645/src/transformers/models/bloom/modeling_bloom.py#L948"}}),Co=new X({props:{name:"forward",anchor:"transformers.BloomForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Union[typing.Tuple[typing.Tuple[torch.Tensor, torch.Tensor], ...], NoneType] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**deprecated_arguments",val:""}],parametersDescription:[{anchor:"transformers.BloomForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else <code>past_key_values[0][0].shape[2]</code>
(<code>sequence_length</code> of input past key value states). Indices of input sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18645/en/model_doc/bloom#transformers.BloomTokenizerFast">BloomTokenizerFast</a>. See <a href="/docs/transformers/pr_18645/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18645/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BloomForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Tuple[Tuple[torch.Tensor]]</code> of length <code>config.n_layers</code>) &#x2014;
Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
<code>past_key_values</code> output below). Can be used to speed up sequential decoding. The <code>input_ids</code> which have
their past given to this model should not be passed as <code>input_ids</code> as they have already been computed.</p>
<p>Each element of <code>past_key_values</code> is a tuple (past_key, past_value):</p>
<ul>
<li>past_key: [batch_size * num_heads, head_dim, kv_length]</li>
<li>past_value: [batch_size * num_heads, kv_length, head_dim]</li>
</ul>`,name:"past_key_values"},{anchor:"transformers.BloomForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BloomForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BloomForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>past_key_values</code> is used, optionally only the last <code>inputs_embeds</code> have to be input (see
<code>past_key_values</code>).`,name:"inputs_embeds"},{anchor:"transformers.BloomForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.BloomForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BloomForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BloomForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18645/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BloomForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18645/src/transformers/models/bloom/modeling_bloom.py#L960",returnDescription:`
<p>A <code>transformers.modeling_outputs.SequenceClassifierOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18645/en/model_doc/bloom#transformers.BloomConfig"
>BloomConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) \u2014 Classification (or regression if config.num_labels==1) loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.num_labels)</code>) \u2014 Classification (or regression if config.num_labels==1) scores (before SoftMax).</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>tuple(tuple(torch.FloatTensor))</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) \u2014 Tuple of <code>tuple(torch.FloatTensor)</code> of length <code>config.n_layers</code>, with each tuple having 2 tensors of shape
<code>(batch_size, num_heads, sequence_length, embed_size_per_head)</code>)</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
<code>past_key_values</code> input) to speed up sequential decoding.</p>
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
<p><code>transformers.modeling_outputs.SequenceClassifierOutputWithPast</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),xe=new Kt({props:{$$slots:{default:[Kl]},$$scope:{ctx:C}}}),je=new Z({props:{anchor:"transformers.BloomForSequenceClassification.forward.example",$$slots:{default:[Ql]},$$scope:{ctx:C}}}),Pe=new Z({props:{anchor:"transformers.BloomForSequenceClassification.forward.example-2",$$slots:{default:[Xl]},$$scope:{ctx:C}}}),Oe=new Z({props:{anchor:"transformers.BloomForSequenceClassification.forward.example-3",$$slots:{default:[Zl]},$$scope:{ctx:C}}}),Se=new Z({props:{anchor:"transformers.BloomForSequenceClassification.forward.example-4",$$slots:{default:[ei]},$$scope:{ctx:C}}}),Fo=new ge({}),zo=new X({props:{name:"class transformers.BloomForTokenClassification",anchor:"transformers.BloomForTokenClassification",parameters:[{name:"config",val:": BloomConfig"}],parametersDescription:[{anchor:"transformers.BloomForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18645/en/model_doc/bloom#transformers.BloomConfig">BloomConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18645/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18645/src/transformers/models/bloom/modeling_bloom.py#L1077"}}),xo=new X({props:{name:"forward",anchor:"transformers.BloomForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Union[typing.Tuple[typing.Tuple[torch.Tensor, torch.Tensor], ...], NoneType] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**deprecated_arguments",val:""}],parametersDescription:[{anchor:"transformers.BloomForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else <code>past_key_values[0][0].shape[2]</code>
(<code>sequence_length</code> of input past key value states). Indices of input sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18645/en/model_doc/bloom#transformers.BloomTokenizerFast">BloomTokenizerFast</a>. See <a href="/docs/transformers/pr_18645/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18645/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BloomForTokenClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Tuple[Tuple[torch.Tensor]]</code> of length <code>config.n_layers</code>) &#x2014;
Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
<code>past_key_values</code> output below). Can be used to speed up sequential decoding. The <code>input_ids</code> which have
their past given to this model should not be passed as <code>input_ids</code> as they have already been computed.</p>
<p>Each element of <code>past_key_values</code> is a tuple (past_key, past_value):</p>
<ul>
<li>past_key: [batch_size * num_heads, head_dim, kv_length]</li>
<li>past_value: [batch_size * num_heads, kv_length, head_dim]</li>
</ul>`,name:"past_key_values"},{anchor:"transformers.BloomForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BloomForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BloomForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>past_key_values</code> is used, optionally only the last <code>inputs_embeds</code> have to be input (see
<code>past_key_values</code>).`,name:"inputs_embeds"},{anchor:"transformers.BloomForTokenClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.BloomForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BloomForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BloomForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18645/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BloomForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18645/src/transformers/models/bloom/modeling_bloom.py#L1097",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18645/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18645/en/model_doc/bloom#transformers.BloomConfig"
>BloomConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18645/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Le=new Kt({props:{$$slots:{default:[oi]},$$scope:{ctx:C}}}),Ne=new Z({props:{anchor:"transformers.BloomForTokenClassification.forward.example",$$slots:{default:[ti]},$$scope:{ctx:C}}}),Ie=new Z({props:{anchor:"transformers.BloomForTokenClassification.forward.example-2",$$slots:{default:[si]},$$scope:{ctx:C}}}),{c(){l=n("meta"),_=u(),c=n("h1"),p=n("a"),b=n("span"),k(s.$$.fragment),h=u(),F=n("span"),te=i("BLOOM"),I=u(),H=n("h2"),R=n("a"),st=n("span"),k(Ue.$$.fragment),js=u(),nt=n("span"),Ps=i("Overview"),Qt=u(),_e=n("p"),Os=i("The BLOOM model has been proposed with its various versions through the "),Ve=n("a"),Ss=i("BigScience Workshop"),As=i(`. BigScience is inspired by other open science initiatives where researchers have pooled their time and resources to collectively achieve a higher impact.
The architecture of BLOOM is essentially similar to GPT3 (auto-regressive model for next token prediction), but has been trained on 46 different languages and 13 programming languages.
Several smaller versions of the models have been trained on the same dataset. BLOOM is available in the following versions:`),Xt=u(),q=n("ul"),at=n("li"),Re=n("a"),Ls=i("bloom-560m"),Ns=u(),rt=n("li"),Ye=n("a"),Is=i("bloom-1b1"),Ds=u(),lt=n("li"),Ge=n("a"),Ws=i("bloom-1b7"),Hs=u(),it=n("li"),Je=n("a"),Us=i("bloom-3b"),Vs=u(),dt=n("li"),Ke=n("a"),Rs=i("bloom-7b1"),Ys=u(),So=n("li"),Qe=n("a"),Gs=i("bloom"),Js=i(" (176B parameters)"),Zt=u(),se=n("h2"),be=n("a"),ct=n("span"),k(Xe.$$.fragment),Ks=u(),pt=n("span"),Qs=i("Languages"),es=u(),ve=n("p"),Xs=i("BLOOM models listed above have been trained in 45 languages human languages and 13 programming languages. You can find the full list of the trained languages "),Ze=n("a"),Zs=i("here"),en=i(`
For BLOOM models on the Hub that uses this architecture and fine-tuned on other datasets, please refer to the corresponding model card for the trained language.`),os=u(),ne=n("h2"),ke=n("a"),mt=n("span"),k(eo.$$.fragment),on=u(),ht=n("span"),tn=i("BloomConfig"),ts=u(),D=n("div"),k(oo.$$.fragment),sn=u(),ae=n("p"),nn=i("This is the configuration class to store the configuration of a "),Ao=n("a"),an=i("BloomModel"),rn=i(`. It is used to instantiate a Bloom
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to the Bloom architecture
`),to=n("a"),ln=i("bigscience/bloom"),dn=i("."),cn=u(),re=n("p"),pn=i("Configuration objects inherit from "),Lo=n("a"),mn=i("PretrainedConfig"),hn=i(` and can be used to control the model outputs. Read the
documentation from `),No=n("a"),un=i("PretrainedConfig"),fn=i(" for more information."),gn=u(),k($e.$$.fragment),ss=u(),le=n("h2"),ye=n("a"),ut=n("span"),k(so.$$.fragment),_n=u(),ft=n("span"),bn=i("BloomModel"),ns=u(),x=n("div"),k(no.$$.fragment),vn=u(),gt=n("p"),kn=i("The bare Bloom Model transformer outputting raw hidden-states without any specific head on top."),$n=u(),ao=n("p"),yn=i("This model inherits from "),Io=n("a"),wn=i("PreTrainedModel"),Tn=i(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)`),Bn=u(),ro=n("p"),Cn=i("This model is also a PyTorch "),lo=n("a"),Fn=i("torch.nn.Module"),zn=i(` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),En=u(),U=n("div"),k(io.$$.fragment),Mn=u(),ie=n("p"),qn=i("The "),Do=n("a"),xn=i("BloomModel"),jn=i(" forward method, overrides the "),_t=n("code"),Pn=i("__call__"),On=i(" special method."),Sn=u(),k(we.$$.fragment),An=u(),k(Te.$$.fragment),as=u(),de=n("h2"),Be=n("a"),bt=n("span"),k(co.$$.fragment),Ln=u(),vt=n("span"),Nn=i("BloomTokenizerFast"),rs=u(),z=n("div"),k(po.$$.fragment),In=u(),mo=n("p"),Dn=i("Construct a \u201Cfast\u201D Bloom tokenizer (backed by HuggingFace\u2019s "),kt=n("em"),Wn=i("tokenizers"),Hn=i(` library). Based on byte-level
Byte-Pair-Encoding.`),Un=u(),$t=n("p"),Vn=i("This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will"),Rn=u(),k(Ce.$$.fragment),Yn=u(),ho=n("p"),Gn=i("You can get around that behavior by passing "),yt=n("code"),Jn=i("add_prefix_space=True"),Kn=i(` when instantiating this tokenizer, but since
the model was not pretrained this way, it might yield a decrease in performance.`),Qn=u(),k(Fe.$$.fragment),Xn=u(),uo=n("p"),Zn=i("This tokenizer inherits from "),Wo=n("a"),ea=i("PreTrainedTokenizerFast"),oa=i(` which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`),ls=u(),ce=n("h2"),ze=n("a"),wt=n("span"),k(fo.$$.fragment),ta=u(),Tt=n("span"),sa=i("BloomForCausalLM"),is=u(),j=n("div"),k(go.$$.fragment),na=u(),Bt=n("p"),aa=i(`The Bloom Model transformer with a language modeling head on top (linear layer with weights tied to the input
embeddings).`),ra=u(),_o=n("p"),la=i("This model inherits from "),Ho=n("a"),ia=i("PreTrainedModel"),da=i(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)`),ca=u(),bo=n("p"),pa=i("This model is also a PyTorch "),vo=n("a"),ma=i("torch.nn.Module"),ha=i(` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),ua=u(),V=n("div"),k(ko.$$.fragment),fa=u(),pe=n("p"),ga=i("The "),Uo=n("a"),_a=i("BloomForCausalLM"),ba=i(" forward method, overrides the "),Ct=n("code"),va=i("__call__"),ka=i(" special method."),$a=u(),k(Ee.$$.fragment),ya=u(),k(Me.$$.fragment),ds=u(),me=n("h2"),qe=n("a"),Ft=n("span"),k($o.$$.fragment),wa=u(),zt=n("span"),Ta=i("BloomForSequenceClassification"),cs=u(),E=n("div"),k(yo.$$.fragment),Ba=u(),Et=n("p"),Ca=i("The Bloom Model transformer with a sequence classification head on top (linear layer)."),Fa=u(),Vo=n("p"),Ro=n("a"),za=i("BloomForSequenceClassification"),Ea=i(` uses the last token in order to do the classification, as other causal models
(e.g. GPT-1) do.`),Ma=u(),W=n("p"),qa=i(`Since it does classification on the last token, it requires to know the position of the last token. If a
`),Mt=n("code"),xa=i("pad_token_id"),ja=i(` is defined in the configuration, it finds the last token that is not a padding token in each row. If
no `),qt=n("code"),Pa=i("pad_token_id"),Oa=i(` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
padding tokens when `),xt=n("code"),Sa=i("inputs_embeds"),Aa=i(" are passed instead of "),jt=n("code"),La=i("input_ids"),Na=i(`, it does the same (take the last value in
each row of the batch).`),Ia=u(),wo=n("p"),Da=i("This model inherits from "),Yo=n("a"),Wa=i("PreTrainedModel"),Ha=i(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)`),Ua=u(),To=n("p"),Va=i("This model is also a PyTorch "),Bo=n("a"),Ra=i("torch.nn.Module"),Ya=i(` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Ga=u(),M=n("div"),k(Co.$$.fragment),Ja=u(),he=n("p"),Ka=i("The "),Go=n("a"),Qa=i("BloomForSequenceClassification"),Xa=i(" forward method, overrides the "),Pt=n("code"),Za=i("__call__"),er=i(" special method."),or=u(),k(xe.$$.fragment),tr=u(),k(je.$$.fragment),sr=u(),k(Pe.$$.fragment),nr=u(),k(Oe.$$.fragment),ar=u(),k(Se.$$.fragment),ps=u(),ue=n("h2"),Ae=n("a"),Ot=n("span"),k(Fo.$$.fragment),rr=u(),St=n("span"),lr=i("BloomForTokenClassification"),ms=u(),P=n("div"),k(zo.$$.fragment),ir=u(),At=n("p"),dr=i(`Bloom Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
Named-Entity-Recognition (NER) tasks.`),cr=u(),Eo=n("p"),pr=i("This model inherits from "),Jo=n("a"),mr=i("PreTrainedModel"),hr=i(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)`),ur=u(),Mo=n("p"),fr=i("This model is also a PyTorch "),qo=n("a"),gr=i("torch.nn.Module"),_r=i(` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),br=u(),S=n("div"),k(xo.$$.fragment),vr=u(),fe=n("p"),kr=i("The "),Ko=n("a"),$r=i("BloomForTokenClassification"),yr=i(" forward method, overrides the "),Lt=n("code"),wr=i("__call__"),Tr=i(" special method."),Br=u(),k(Le.$$.fragment),Cr=u(),k(Ne.$$.fragment),Fr=u(),k(Ie.$$.fragment),this.h()},l(o){const g=Dl('[data-svelte="svelte-1phssyn"]',document.head);l=a(g,"META",{name:!0,content:!0}),g.forEach(t),_=f(o),c=a(o,"H1",{class:!0});var jo=r(c);p=a(jo,"A",{id:!0,class:!0,href:!0});var Nt=r(p);b=a(Nt,"SPAN",{});var It=r(b);$(s.$$.fragment,It),It.forEach(t),Nt.forEach(t),h=f(jo),F=a(jo,"SPAN",{});var Dt=r(F);te=d(Dt,"BLOOM"),Dt.forEach(t),jo.forEach(t),I=f(o),H=a(o,"H2",{class:!0});var Po=r(H);R=a(Po,"A",{id:!0,class:!0,href:!0});var Wt=r(R);st=a(Wt,"SPAN",{});var Ht=r(st);$(Ue.$$.fragment,Ht),Ht.forEach(t),Wt.forEach(t),js=f(Po),nt=a(Po,"SPAN",{});var Ut=r(nt);Ps=d(Ut,"Overview"),Ut.forEach(t),Po.forEach(t),Qt=f(o),_e=a(o,"P",{});var Oo=r(_e);Os=d(Oo,"The BLOOM model has been proposed with its various versions through the "),Ve=a(Oo,"A",{href:!0,rel:!0});var Vt=r(Ve);Ss=d(Vt,"BigScience Workshop"),Vt.forEach(t),As=d(Oo,`. BigScience is inspired by other open science initiatives where researchers have pooled their time and resources to collectively achieve a higher impact.
The architecture of BLOOM is essentially similar to GPT3 (auto-regressive model for next token prediction), but has been trained on 46 different languages and 13 programming languages.
Several smaller versions of the models have been trained on the same dataset. BLOOM is available in the following versions:`),Oo.forEach(t),Xt=f(o),q=a(o,"UL",{});var O=r(q);at=a(O,"LI",{});var Rt=r(at);Re=a(Rt,"A",{href:!0,rel:!0});var Yt=r(Re);Ls=d(Yt,"bloom-560m"),Yt.forEach(t),Rt.forEach(t),Ns=f(O),rt=a(O,"LI",{});var Gt=r(rt);Ye=a(Gt,"A",{href:!0,rel:!0});var Jt=r(Ye);Is=d(Jt,"bloom-1b1"),Jt.forEach(t),Gt.forEach(t),Ds=f(O),lt=a(O,"LI",{});var Mr=r(lt);Ge=a(Mr,"A",{href:!0,rel:!0});var qr=r(Ge);Ws=d(qr,"bloom-1b7"),qr.forEach(t),Mr.forEach(t),Hs=f(O),it=a(O,"LI",{});var xr=r(it);Je=a(xr,"A",{href:!0,rel:!0});var jr=r(Je);Us=d(jr,"bloom-3b"),jr.forEach(t),xr.forEach(t),Vs=f(O),dt=a(O,"LI",{});var Pr=r(dt);Ke=a(Pr,"A",{href:!0,rel:!0});var Or=r(Ke);Rs=d(Or,"bloom-7b1"),Or.forEach(t),Pr.forEach(t),Ys=f(O),So=a(O,"LI",{});var zr=r(So);Qe=a(zr,"A",{href:!0,rel:!0});var Sr=r(Qe);Gs=d(Sr,"bloom"),Sr.forEach(t),Js=d(zr," (176B parameters)"),zr.forEach(t),O.forEach(t),Zt=f(o),se=a(o,"H2",{class:!0});var us=r(se);be=a(us,"A",{id:!0,class:!0,href:!0});var Ar=r(be);ct=a(Ar,"SPAN",{});var Lr=r(ct);$(Xe.$$.fragment,Lr),Lr.forEach(t),Ar.forEach(t),Ks=f(us),pt=a(us,"SPAN",{});var Nr=r(pt);Qs=d(Nr,"Languages"),Nr.forEach(t),us.forEach(t),es=f(o),ve=a(o,"P",{});var fs=r(ve);Xs=d(fs,"BLOOM models listed above have been trained in 45 languages human languages and 13 programming languages. You can find the full list of the trained languages "),Ze=a(fs,"A",{href:!0,rel:!0});var Ir=r(Ze);Zs=d(Ir,"here"),Ir.forEach(t),en=d(fs,`
For BLOOM models on the Hub that uses this architecture and fine-tuned on other datasets, please refer to the corresponding model card for the trained language.`),fs.forEach(t),os=f(o),ne=a(o,"H2",{class:!0});var gs=r(ne);ke=a(gs,"A",{id:!0,class:!0,href:!0});var Dr=r(ke);mt=a(Dr,"SPAN",{});var Wr=r(mt);$(eo.$$.fragment,Wr),Wr.forEach(t),Dr.forEach(t),on=f(gs),ht=a(gs,"SPAN",{});var Hr=r(ht);tn=d(Hr,"BloomConfig"),Hr.forEach(t),gs.forEach(t),ts=f(o),D=a(o,"DIV",{class:!0});var De=r(D);$(oo.$$.fragment,De),sn=f(De),ae=a(De,"P",{});var Qo=r(ae);nn=d(Qo,"This is the configuration class to store the configuration of a "),Ao=a(Qo,"A",{href:!0});var Ur=r(Ao);an=d(Ur,"BloomModel"),Ur.forEach(t),rn=d(Qo,`. It is used to instantiate a Bloom
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to the Bloom architecture
`),to=a(Qo,"A",{href:!0,rel:!0});var Vr=r(to);ln=d(Vr,"bigscience/bloom"),Vr.forEach(t),dn=d(Qo,"."),Qo.forEach(t),cn=f(De),re=a(De,"P",{});var Xo=r(re);pn=d(Xo,"Configuration objects inherit from "),Lo=a(Xo,"A",{href:!0});var Rr=r(Lo);mn=d(Rr,"PretrainedConfig"),Rr.forEach(t),hn=d(Xo,` and can be used to control the model outputs. Read the
documentation from `),No=a(Xo,"A",{href:!0});var Yr=r(No);un=d(Yr,"PretrainedConfig"),Yr.forEach(t),fn=d(Xo," for more information."),Xo.forEach(t),gn=f(De),$($e.$$.fragment,De),De.forEach(t),ss=f(o),le=a(o,"H2",{class:!0});var _s=r(le);ye=a(_s,"A",{id:!0,class:!0,href:!0});var Gr=r(ye);ut=a(Gr,"SPAN",{});var Jr=r(ut);$(so.$$.fragment,Jr),Jr.forEach(t),Gr.forEach(t),_n=f(_s),ft=a(_s,"SPAN",{});var Kr=r(ft);bn=d(Kr,"BloomModel"),Kr.forEach(t),_s.forEach(t),ns=f(o),x=a(o,"DIV",{class:!0});var Y=r(x);$(no.$$.fragment,Y),vn=f(Y),gt=a(Y,"P",{});var Qr=r(gt);kn=d(Qr,"The bare Bloom Model transformer outputting raw hidden-states without any specific head on top."),Qr.forEach(t),$n=f(Y),ao=a(Y,"P",{});var bs=r(ao);yn=d(bs,"This model inherits from "),Io=a(bs,"A",{href:!0});var Xr=r(Io);wn=d(Xr,"PreTrainedModel"),Xr.forEach(t),Tn=d(bs,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)`),bs.forEach(t),Bn=f(Y),ro=a(Y,"P",{});var vs=r(ro);Cn=d(vs,"This model is also a PyTorch "),lo=a(vs,"A",{href:!0,rel:!0});var Zr=r(lo);Fn=d(Zr,"torch.nn.Module"),Zr.forEach(t),zn=d(vs,` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),vs.forEach(t),En=f(Y),U=a(Y,"DIV",{class:!0});var We=r(U);$(io.$$.fragment,We),Mn=f(We),ie=a(We,"P",{});var Zo=r(ie);qn=d(Zo,"The "),Do=a(Zo,"A",{href:!0});var el=r(Do);xn=d(el,"BloomModel"),el.forEach(t),jn=d(Zo," forward method, overrides the "),_t=a(Zo,"CODE",{});var ol=r(_t);Pn=d(ol,"__call__"),ol.forEach(t),On=d(Zo," special method."),Zo.forEach(t),Sn=f(We),$(we.$$.fragment,We),An=f(We),$(Te.$$.fragment,We),We.forEach(t),Y.forEach(t),as=f(o),de=a(o,"H2",{class:!0});var ks=r(de);Be=a(ks,"A",{id:!0,class:!0,href:!0});var tl=r(Be);bt=a(tl,"SPAN",{});var sl=r(bt);$(co.$$.fragment,sl),sl.forEach(t),tl.forEach(t),Ln=f(ks),vt=a(ks,"SPAN",{});var nl=r(vt);Nn=d(nl,"BloomTokenizerFast"),nl.forEach(t),ks.forEach(t),rs=f(o),z=a(o,"DIV",{class:!0});var A=r(z);$(po.$$.fragment,A),In=f(A),mo=a(A,"P",{});var $s=r(mo);Dn=d($s,"Construct a \u201Cfast\u201D Bloom tokenizer (backed by HuggingFace\u2019s "),kt=a($s,"EM",{});var al=r(kt);Wn=d(al,"tokenizers"),al.forEach(t),Hn=d($s,` library). Based on byte-level
Byte-Pair-Encoding.`),$s.forEach(t),Un=f(A),$t=a(A,"P",{});var rl=r($t);Vn=d(rl,"This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will"),rl.forEach(t),Rn=f(A),$(Ce.$$.fragment,A),Yn=f(A),ho=a(A,"P",{});var ys=r(ho);Gn=d(ys,"You can get around that behavior by passing "),yt=a(ys,"CODE",{});var ll=r(yt);Jn=d(ll,"add_prefix_space=True"),ll.forEach(t),Kn=d(ys,` when instantiating this tokenizer, but since
the model was not pretrained this way, it might yield a decrease in performance.`),ys.forEach(t),Qn=f(A),$(Fe.$$.fragment,A),Xn=f(A),uo=a(A,"P",{});var ws=r(uo);Zn=d(ws,"This tokenizer inherits from "),Wo=a(ws,"A",{href:!0});var il=r(Wo);ea=d(il,"PreTrainedTokenizerFast"),il.forEach(t),oa=d(ws,` which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`),ws.forEach(t),A.forEach(t),ls=f(o),ce=a(o,"H2",{class:!0});var Ts=r(ce);ze=a(Ts,"A",{id:!0,class:!0,href:!0});var dl=r(ze);wt=a(dl,"SPAN",{});var cl=r(wt);$(fo.$$.fragment,cl),cl.forEach(t),dl.forEach(t),ta=f(Ts),Tt=a(Ts,"SPAN",{});var pl=r(Tt);sa=d(pl,"BloomForCausalLM"),pl.forEach(t),Ts.forEach(t),is=f(o),j=a(o,"DIV",{class:!0});var G=r(j);$(go.$$.fragment,G),na=f(G),Bt=a(G,"P",{});var ml=r(Bt);aa=d(ml,`The Bloom Model transformer with a language modeling head on top (linear layer with weights tied to the input
embeddings).`),ml.forEach(t),ra=f(G),_o=a(G,"P",{});var Bs=r(_o);la=d(Bs,"This model inherits from "),Ho=a(Bs,"A",{href:!0});var hl=r(Ho);ia=d(hl,"PreTrainedModel"),hl.forEach(t),da=d(Bs,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)`),Bs.forEach(t),ca=f(G),bo=a(G,"P",{});var Cs=r(bo);pa=d(Cs,"This model is also a PyTorch "),vo=a(Cs,"A",{href:!0,rel:!0});var ul=r(vo);ma=d(ul,"torch.nn.Module"),ul.forEach(t),ha=d(Cs,` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Cs.forEach(t),ua=f(G),V=a(G,"DIV",{class:!0});var He=r(V);$(ko.$$.fragment,He),fa=f(He),pe=a(He,"P",{});var et=r(pe);ga=d(et,"The "),Uo=a(et,"A",{href:!0});var fl=r(Uo);_a=d(fl,"BloomForCausalLM"),fl.forEach(t),ba=d(et," forward method, overrides the "),Ct=a(et,"CODE",{});var gl=r(Ct);va=d(gl,"__call__"),gl.forEach(t),ka=d(et," special method."),et.forEach(t),$a=f(He),$(Ee.$$.fragment,He),ya=f(He),$(Me.$$.fragment,He),He.forEach(t),G.forEach(t),ds=f(o),me=a(o,"H2",{class:!0});var Fs=r(me);qe=a(Fs,"A",{id:!0,class:!0,href:!0});var _l=r(qe);Ft=a(_l,"SPAN",{});var bl=r(Ft);$($o.$$.fragment,bl),bl.forEach(t),_l.forEach(t),wa=f(Fs),zt=a(Fs,"SPAN",{});var vl=r(zt);Ta=d(vl,"BloomForSequenceClassification"),vl.forEach(t),Fs.forEach(t),cs=f(o),E=a(o,"DIV",{class:!0});var L=r(E);$(yo.$$.fragment,L),Ba=f(L),Et=a(L,"P",{});var kl=r(Et);Ca=d(kl,"The Bloom Model transformer with a sequence classification head on top (linear layer)."),kl.forEach(t),Fa=f(L),Vo=a(L,"P",{});var Er=r(Vo);Ro=a(Er,"A",{href:!0});var $l=r(Ro);za=d($l,"BloomForSequenceClassification"),$l.forEach(t),Ea=d(Er,` uses the last token in order to do the classification, as other causal models
(e.g. GPT-1) do.`),Er.forEach(t),Ma=f(L),W=a(L,"P",{});var J=r(W);qa=d(J,`Since it does classification on the last token, it requires to know the position of the last token. If a
`),Mt=a(J,"CODE",{});var yl=r(Mt);xa=d(yl,"pad_token_id"),yl.forEach(t),ja=d(J,` is defined in the configuration, it finds the last token that is not a padding token in each row. If
no `),qt=a(J,"CODE",{});var wl=r(qt);Pa=d(wl,"pad_token_id"),wl.forEach(t),Oa=d(J,` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
padding tokens when `),xt=a(J,"CODE",{});var Tl=r(xt);Sa=d(Tl,"inputs_embeds"),Tl.forEach(t),Aa=d(J," are passed instead of "),jt=a(J,"CODE",{});var Bl=r(jt);La=d(Bl,"input_ids"),Bl.forEach(t),Na=d(J,`, it does the same (take the last value in
each row of the batch).`),J.forEach(t),Ia=f(L),wo=a(L,"P",{});var zs=r(wo);Da=d(zs,"This model inherits from "),Yo=a(zs,"A",{href:!0});var Cl=r(Yo);Wa=d(Cl,"PreTrainedModel"),Cl.forEach(t),Ha=d(zs,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)`),zs.forEach(t),Ua=f(L),To=a(L,"P",{});var Es=r(To);Va=d(Es,"This model is also a PyTorch "),Bo=a(Es,"A",{href:!0,rel:!0});var Fl=r(Bo);Ra=d(Fl,"torch.nn.Module"),Fl.forEach(t),Ya=d(Es,` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Es.forEach(t),Ga=f(L),M=a(L,"DIV",{class:!0});var N=r(M);$(Co.$$.fragment,N),Ja=f(N),he=a(N,"P",{});var ot=r(he);Ka=d(ot,"The "),Go=a(ot,"A",{href:!0});var zl=r(Go);Qa=d(zl,"BloomForSequenceClassification"),zl.forEach(t),Xa=d(ot," forward method, overrides the "),Pt=a(ot,"CODE",{});var El=r(Pt);Za=d(El,"__call__"),El.forEach(t),er=d(ot," special method."),ot.forEach(t),or=f(N),$(xe.$$.fragment,N),tr=f(N),$(je.$$.fragment,N),sr=f(N),$(Pe.$$.fragment,N),nr=f(N),$(Oe.$$.fragment,N),ar=f(N),$(Se.$$.fragment,N),N.forEach(t),L.forEach(t),ps=f(o),ue=a(o,"H2",{class:!0});var Ms=r(ue);Ae=a(Ms,"A",{id:!0,class:!0,href:!0});var Ml=r(Ae);Ot=a(Ml,"SPAN",{});var ql=r(Ot);$(Fo.$$.fragment,ql),ql.forEach(t),Ml.forEach(t),rr=f(Ms),St=a(Ms,"SPAN",{});var xl=r(St);lr=d(xl,"BloomForTokenClassification"),xl.forEach(t),Ms.forEach(t),ms=f(o),P=a(o,"DIV",{class:!0});var K=r(P);$(zo.$$.fragment,K),ir=f(K),At=a(K,"P",{});var jl=r(At);dr=d(jl,`Bloom Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
Named-Entity-Recognition (NER) tasks.`),jl.forEach(t),cr=f(K),Eo=a(K,"P",{});var qs=r(Eo);pr=d(qs,"This model inherits from "),Jo=a(qs,"A",{href:!0});var Pl=r(Jo);mr=d(Pl,"PreTrainedModel"),Pl.forEach(t),hr=d(qs,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)`),qs.forEach(t),ur=f(K),Mo=a(K,"P",{});var xs=r(Mo);fr=d(xs,"This model is also a PyTorch "),qo=a(xs,"A",{href:!0,rel:!0});var Ol=r(qo);gr=d(Ol,"torch.nn.Module"),Ol.forEach(t),_r=d(xs,` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),xs.forEach(t),br=f(K),S=a(K,"DIV",{class:!0});var Q=r(S);$(xo.$$.fragment,Q),vr=f(Q),fe=a(Q,"P",{});var tt=r(fe);kr=d(tt,"The "),Ko=a(tt,"A",{href:!0});var Sl=r(Ko);$r=d(Sl,"BloomForTokenClassification"),Sl.forEach(t),yr=d(tt," forward method, overrides the "),Lt=a(tt,"CODE",{});var Al=r(Lt);wr=d(Al,"__call__"),Al.forEach(t),Tr=d(tt," special method."),tt.forEach(t),Br=f(Q),$(Le.$$.fragment,Q),Cr=f(Q),$(Ne.$$.fragment,Q),Fr=f(Q),$(Ie.$$.fragment,Q),Q.forEach(t),K.forEach(t),this.h()},h(){m(l,"name","hf:doc:metadata"),m(l,"content",JSON.stringify(ai)),m(p,"id","bloom"),m(p,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(p,"href","#bloom"),m(c,"class","relative group"),m(R,"id","overview"),m(R,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(R,"href","#overview"),m(H,"class","relative group"),m(Ve,"href","https://bigscience.huggingface.co/"),m(Ve,"rel","nofollow"),m(Re,"href","https://huggingface.co/bigscience/bloom-560m"),m(Re,"rel","nofollow"),m(Ye,"href","https://huggingface.co/bigscience/bloom-1b1"),m(Ye,"rel","nofollow"),m(Ge,"href","https://huggingface.co/bigscience/bloom-1b7"),m(Ge,"rel","nofollow"),m(Je,"href","https://huggingface.co/bigscience/bloom-3b"),m(Je,"rel","nofollow"),m(Ke,"href","https://huggingface.co/bigscience/bloom-7b1"),m(Ke,"rel","nofollow"),m(Qe,"href","https://huggingface.co/bigscience/bloom"),m(Qe,"rel","nofollow"),m(be,"id","languages"),m(be,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(be,"href","#languages"),m(se,"class","relative group"),m(Ze,"href","https://huggingface.co/bigscience/bloom#languages"),m(Ze,"rel","nofollow"),m(ke,"id","transformers.BloomConfig"),m(ke,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(ke,"href","#transformers.BloomConfig"),m(ne,"class","relative group"),m(Ao,"href","/docs/transformers/pr_18645/en/model_doc/bloom#transformers.BloomModel"),m(to,"href","https://huggingface.co/bigscience/bloom"),m(to,"rel","nofollow"),m(Lo,"href","/docs/transformers/pr_18645/en/main_classes/configuration#transformers.PretrainedConfig"),m(No,"href","/docs/transformers/pr_18645/en/main_classes/configuration#transformers.PretrainedConfig"),m(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(ye,"id","transformers.BloomModel"),m(ye,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(ye,"href","#transformers.BloomModel"),m(le,"class","relative group"),m(Io,"href","/docs/transformers/pr_18645/en/main_classes/model#transformers.PreTrainedModel"),m(lo,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),m(lo,"rel","nofollow"),m(Do,"href","/docs/transformers/pr_18645/en/model_doc/bloom#transformers.BloomModel"),m(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Be,"id","transformers.BloomTokenizerFast"),m(Be,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(Be,"href","#transformers.BloomTokenizerFast"),m(de,"class","relative group"),m(Wo,"href","/docs/transformers/pr_18645/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast"),m(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(ze,"id","transformers.BloomForCausalLM"),m(ze,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(ze,"href","#transformers.BloomForCausalLM"),m(ce,"class","relative group"),m(Ho,"href","/docs/transformers/pr_18645/en/main_classes/model#transformers.PreTrainedModel"),m(vo,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),m(vo,"rel","nofollow"),m(Uo,"href","/docs/transformers/pr_18645/en/model_doc/bloom#transformers.BloomForCausalLM"),m(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(qe,"id","transformers.BloomForSequenceClassification"),m(qe,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(qe,"href","#transformers.BloomForSequenceClassification"),m(me,"class","relative group"),m(Ro,"href","/docs/transformers/pr_18645/en/model_doc/bloom#transformers.BloomForSequenceClassification"),m(Yo,"href","/docs/transformers/pr_18645/en/main_classes/model#transformers.PreTrainedModel"),m(Bo,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),m(Bo,"rel","nofollow"),m(Go,"href","/docs/transformers/pr_18645/en/model_doc/bloom#transformers.BloomForSequenceClassification"),m(M,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Ae,"id","transformers.BloomForTokenClassification"),m(Ae,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(Ae,"href","#transformers.BloomForTokenClassification"),m(ue,"class","relative group"),m(Jo,"href","/docs/transformers/pr_18645/en/main_classes/model#transformers.PreTrainedModel"),m(qo,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),m(qo,"rel","nofollow"),m(Ko,"href","/docs/transformers/pr_18645/en/model_doc/bloom#transformers.BloomForTokenClassification"),m(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(o,g){e(document.head,l),v(o,_,g),v(o,c,g),e(c,p),e(p,b),y(s,b,null),e(c,h),e(c,F),e(F,te),v(o,I,g),v(o,H,g),e(H,R),e(R,st),y(Ue,st,null),e(H,js),e(H,nt),e(nt,Ps),v(o,Qt,g),v(o,_e,g),e(_e,Os),e(_e,Ve),e(Ve,Ss),e(_e,As),v(o,Xt,g),v(o,q,g),e(q,at),e(at,Re),e(Re,Ls),e(q,Ns),e(q,rt),e(rt,Ye),e(Ye,Is),e(q,Ds),e(q,lt),e(lt,Ge),e(Ge,Ws),e(q,Hs),e(q,it),e(it,Je),e(Je,Us),e(q,Vs),e(q,dt),e(dt,Ke),e(Ke,Rs),e(q,Ys),e(q,So),e(So,Qe),e(Qe,Gs),e(So,Js),v(o,Zt,g),v(o,se,g),e(se,be),e(be,ct),y(Xe,ct,null),e(se,Ks),e(se,pt),e(pt,Qs),v(o,es,g),v(o,ve,g),e(ve,Xs),e(ve,Ze),e(Ze,Zs),e(ve,en),v(o,os,g),v(o,ne,g),e(ne,ke),e(ke,mt),y(eo,mt,null),e(ne,on),e(ne,ht),e(ht,tn),v(o,ts,g),v(o,D,g),y(oo,D,null),e(D,sn),e(D,ae),e(ae,nn),e(ae,Ao),e(Ao,an),e(ae,rn),e(ae,to),e(to,ln),e(ae,dn),e(D,cn),e(D,re),e(re,pn),e(re,Lo),e(Lo,mn),e(re,hn),e(re,No),e(No,un),e(re,fn),e(D,gn),y($e,D,null),v(o,ss,g),v(o,le,g),e(le,ye),e(ye,ut),y(so,ut,null),e(le,_n),e(le,ft),e(ft,bn),v(o,ns,g),v(o,x,g),y(no,x,null),e(x,vn),e(x,gt),e(gt,kn),e(x,$n),e(x,ao),e(ao,yn),e(ao,Io),e(Io,wn),e(ao,Tn),e(x,Bn),e(x,ro),e(ro,Cn),e(ro,lo),e(lo,Fn),e(ro,zn),e(x,En),e(x,U),y(io,U,null),e(U,Mn),e(U,ie),e(ie,qn),e(ie,Do),e(Do,xn),e(ie,jn),e(ie,_t),e(_t,Pn),e(ie,On),e(U,Sn),y(we,U,null),e(U,An),y(Te,U,null),v(o,as,g),v(o,de,g),e(de,Be),e(Be,bt),y(co,bt,null),e(de,Ln),e(de,vt),e(vt,Nn),v(o,rs,g),v(o,z,g),y(po,z,null),e(z,In),e(z,mo),e(mo,Dn),e(mo,kt),e(kt,Wn),e(mo,Hn),e(z,Un),e(z,$t),e($t,Vn),e(z,Rn),y(Ce,z,null),e(z,Yn),e(z,ho),e(ho,Gn),e(ho,yt),e(yt,Jn),e(ho,Kn),e(z,Qn),y(Fe,z,null),e(z,Xn),e(z,uo),e(uo,Zn),e(uo,Wo),e(Wo,ea),e(uo,oa),v(o,ls,g),v(o,ce,g),e(ce,ze),e(ze,wt),y(fo,wt,null),e(ce,ta),e(ce,Tt),e(Tt,sa),v(o,is,g),v(o,j,g),y(go,j,null),e(j,na),e(j,Bt),e(Bt,aa),e(j,ra),e(j,_o),e(_o,la),e(_o,Ho),e(Ho,ia),e(_o,da),e(j,ca),e(j,bo),e(bo,pa),e(bo,vo),e(vo,ma),e(bo,ha),e(j,ua),e(j,V),y(ko,V,null),e(V,fa),e(V,pe),e(pe,ga),e(pe,Uo),e(Uo,_a),e(pe,ba),e(pe,Ct),e(Ct,va),e(pe,ka),e(V,$a),y(Ee,V,null),e(V,ya),y(Me,V,null),v(o,ds,g),v(o,me,g),e(me,qe),e(qe,Ft),y($o,Ft,null),e(me,wa),e(me,zt),e(zt,Ta),v(o,cs,g),v(o,E,g),y(yo,E,null),e(E,Ba),e(E,Et),e(Et,Ca),e(E,Fa),e(E,Vo),e(Vo,Ro),e(Ro,za),e(Vo,Ea),e(E,Ma),e(E,W),e(W,qa),e(W,Mt),e(Mt,xa),e(W,ja),e(W,qt),e(qt,Pa),e(W,Oa),e(W,xt),e(xt,Sa),e(W,Aa),e(W,jt),e(jt,La),e(W,Na),e(E,Ia),e(E,wo),e(wo,Da),e(wo,Yo),e(Yo,Wa),e(wo,Ha),e(E,Ua),e(E,To),e(To,Va),e(To,Bo),e(Bo,Ra),e(To,Ya),e(E,Ga),e(E,M),y(Co,M,null),e(M,Ja),e(M,he),e(he,Ka),e(he,Go),e(Go,Qa),e(he,Xa),e(he,Pt),e(Pt,Za),e(he,er),e(M,or),y(xe,M,null),e(M,tr),y(je,M,null),e(M,sr),y(Pe,M,null),e(M,nr),y(Oe,M,null),e(M,ar),y(Se,M,null),v(o,ps,g),v(o,ue,g),e(ue,Ae),e(Ae,Ot),y(Fo,Ot,null),e(ue,rr),e(ue,St),e(St,lr),v(o,ms,g),v(o,P,g),y(zo,P,null),e(P,ir),e(P,At),e(At,dr),e(P,cr),e(P,Eo),e(Eo,pr),e(Eo,Jo),e(Jo,mr),e(Eo,hr),e(P,ur),e(P,Mo),e(Mo,fr),e(Mo,qo),e(qo,gr),e(Mo,_r),e(P,br),e(P,S),y(xo,S,null),e(S,vr),e(S,fe),e(fe,kr),e(fe,Ko),e(Ko,$r),e(fe,yr),e(fe,Lt),e(Lt,wr),e(fe,Tr),e(S,Br),y(Le,S,null),e(S,Cr),y(Ne,S,null),e(S,Fr),y(Ie,S,null),hs=!0},p(o,[g]){const jo={};g&2&&(jo.$$scope={dirty:g,ctx:o}),$e.$set(jo);const Nt={};g&2&&(Nt.$$scope={dirty:g,ctx:o}),we.$set(Nt);const It={};g&2&&(It.$$scope={dirty:g,ctx:o}),Te.$set(It);const Dt={};g&2&&(Dt.$$scope={dirty:g,ctx:o}),Ce.$set(Dt);const Po={};g&2&&(Po.$$scope={dirty:g,ctx:o}),Fe.$set(Po);const Wt={};g&2&&(Wt.$$scope={dirty:g,ctx:o}),Ee.$set(Wt);const Ht={};g&2&&(Ht.$$scope={dirty:g,ctx:o}),Me.$set(Ht);const Ut={};g&2&&(Ut.$$scope={dirty:g,ctx:o}),xe.$set(Ut);const Oo={};g&2&&(Oo.$$scope={dirty:g,ctx:o}),je.$set(Oo);const Vt={};g&2&&(Vt.$$scope={dirty:g,ctx:o}),Pe.$set(Vt);const O={};g&2&&(O.$$scope={dirty:g,ctx:o}),Oe.$set(O);const Rt={};g&2&&(Rt.$$scope={dirty:g,ctx:o}),Se.$set(Rt);const Yt={};g&2&&(Yt.$$scope={dirty:g,ctx:o}),Le.$set(Yt);const Gt={};g&2&&(Gt.$$scope={dirty:g,ctx:o}),Ne.$set(Gt);const Jt={};g&2&&(Jt.$$scope={dirty:g,ctx:o}),Ie.$set(Jt)},i(o){hs||(w(s.$$.fragment,o),w(Ue.$$.fragment,o),w(Xe.$$.fragment,o),w(eo.$$.fragment,o),w(oo.$$.fragment,o),w($e.$$.fragment,o),w(so.$$.fragment,o),w(no.$$.fragment,o),w(io.$$.fragment,o),w(we.$$.fragment,o),w(Te.$$.fragment,o),w(co.$$.fragment,o),w(po.$$.fragment,o),w(Ce.$$.fragment,o),w(Fe.$$.fragment,o),w(fo.$$.fragment,o),w(go.$$.fragment,o),w(ko.$$.fragment,o),w(Ee.$$.fragment,o),w(Me.$$.fragment,o),w($o.$$.fragment,o),w(yo.$$.fragment,o),w(Co.$$.fragment,o),w(xe.$$.fragment,o),w(je.$$.fragment,o),w(Pe.$$.fragment,o),w(Oe.$$.fragment,o),w(Se.$$.fragment,o),w(Fo.$$.fragment,o),w(zo.$$.fragment,o),w(xo.$$.fragment,o),w(Le.$$.fragment,o),w(Ne.$$.fragment,o),w(Ie.$$.fragment,o),hs=!0)},o(o){T(s.$$.fragment,o),T(Ue.$$.fragment,o),T(Xe.$$.fragment,o),T(eo.$$.fragment,o),T(oo.$$.fragment,o),T($e.$$.fragment,o),T(so.$$.fragment,o),T(no.$$.fragment,o),T(io.$$.fragment,o),T(we.$$.fragment,o),T(Te.$$.fragment,o),T(co.$$.fragment,o),T(po.$$.fragment,o),T(Ce.$$.fragment,o),T(Fe.$$.fragment,o),T(fo.$$.fragment,o),T(go.$$.fragment,o),T(ko.$$.fragment,o),T(Ee.$$.fragment,o),T(Me.$$.fragment,o),T($o.$$.fragment,o),T(yo.$$.fragment,o),T(Co.$$.fragment,o),T(xe.$$.fragment,o),T(je.$$.fragment,o),T(Pe.$$.fragment,o),T(Oe.$$.fragment,o),T(Se.$$.fragment,o),T(Fo.$$.fragment,o),T(zo.$$.fragment,o),T(xo.$$.fragment,o),T(Le.$$.fragment,o),T(Ne.$$.fragment,o),T(Ie.$$.fragment,o),hs=!1},d(o){t(l),o&&t(_),o&&t(c),B(s),o&&t(I),o&&t(H),B(Ue),o&&t(Qt),o&&t(_e),o&&t(Xt),o&&t(q),o&&t(Zt),o&&t(se),B(Xe),o&&t(es),o&&t(ve),o&&t(os),o&&t(ne),B(eo),o&&t(ts),o&&t(D),B(oo),B($e),o&&t(ss),o&&t(le),B(so),o&&t(ns),o&&t(x),B(no),B(io),B(we),B(Te),o&&t(as),o&&t(de),B(co),o&&t(rs),o&&t(z),B(po),B(Ce),B(Fe),o&&t(ls),o&&t(ce),B(fo),o&&t(is),o&&t(j),B(go),B(ko),B(Ee),B(Me),o&&t(ds),o&&t(me),B($o),o&&t(cs),o&&t(E),B(yo),B(Co),B(xe),B(je),B(Pe),B(Oe),B(Se),o&&t(ps),o&&t(ue),B(Fo),o&&t(ms),o&&t(P),B(zo),B(xo),B(Le),B(Ne),B(Ie)}}}const ai={local:"bloom",sections:[{local:"overview",title:"Overview"},{local:"languages",title:"Languages"},{local:"transformers.BloomConfig",title:"BloomConfig"},{local:"transformers.BloomModel",title:"BloomModel"},{local:"transformers.BloomTokenizerFast",title:"BloomTokenizerFast"},{local:"transformers.BloomForCausalLM",title:"BloomForCausalLM"},{local:"transformers.BloomForSequenceClassification",title:"BloomForSequenceClassification"},{local:"transformers.BloomForTokenClassification",title:"BloomForTokenClassification"}],title:"BLOOM"};function ri(C){return Wl(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class hi extends Ll{constructor(l){super();Nl(this,l,ri,ni,Il,{})}}export{hi as default,ai as metadata};
