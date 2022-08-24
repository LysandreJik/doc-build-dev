import{S as Mc,i as zc,s as Cc,e as s,k as u,w as k,t as l,M as Ec,c as a,d as o,m as f,a as r,x as y,h as i,b as p,G as e,g as v,y as $,q as w,o as T,B as F,v as jc,L as ee}from"../../chunks/vendor-hf-doc-builder.js";import{T as vt}from"../../chunks/Tip-hf-doc-builder.js";import{D as U}from"../../chunks/Docstring-hf-doc-builder.js";import{C as oe}from"../../chunks/CodeBlock-hf-doc-builder.js";import{I as ce}from"../../chunks/IconCopyLink-hf-doc-builder.js";import{E as Z}from"../../chunks/ExampleCodeBlock-hf-doc-builder.js";function Pc(B){let d,g,c,m,b;return m=new oe({props:{code:`from transformers import BloomModel, BloomConfig

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
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`}}),{c(){d=s("p"),g=l("Example:"),c=u(),k(m.$$.fragment)},l(n){d=a(n,"P",{});var h=r(d);g=i(h,"Example:"),h.forEach(o),c=f(n),y(m.$$.fragment,n)},m(n,h){v(n,d,h),e(d,g),v(n,c,h),$(m,n,h),b=!0},p:ee,i(n){b||(w(m.$$.fragment,n),b=!0)},o(n){T(m.$$.fragment,n),b=!1},d(n){n&&o(d),n&&o(c),F(m,n)}}}function qc(B){let d,g,c,m,b;return{c(){d=s("p"),g=l("Although the recipe for forward pass needs to be defined within this function, one should call the "),c=s("code"),m=l("Module"),b=l(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(n){d=a(n,"P",{});var h=r(d);g=i(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),c=a(h,"CODE",{});var x=r(c);m=i(x,"Module"),x.forEach(o),b=i(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(o)},m(n,h){v(n,d,h),e(d,g),e(d,c),e(c,m),e(d,b)},d(n){n&&o(d)}}}function Oc(B){let d,g,c,m,b;return m=new oe({props:{code:`from transformers import BloomTokenizerFast, BloomModel
import torch

tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-350m")
model = BloomModel.from_pretrained("bigscience/bloom-350m")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BloomTokenizerFast, BloomModel
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BloomTokenizerFast.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom-350m&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BloomModel.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom-350m&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`}}),{c(){d=s("p"),g=l("Example:"),c=u(),k(m.$$.fragment)},l(n){d=a(n,"P",{});var h=r(d);g=i(h,"Example:"),h.forEach(o),c=f(n),y(m.$$.fragment,n)},m(n,h){v(n,d,h),e(d,g),v(n,c,h),$(m,n,h),b=!0},p:ee,i(n){b||(w(m.$$.fragment,n),b=!0)},o(n){T(m.$$.fragment,n),b=!1},d(n){n&&o(d),n&&o(c),F(m,n)}}}function Lc(B){let d,g,c,m,b;return m=new oe({props:{code:`from transformers import BloomTokenizerFast
tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom")
tokenizer("Hello world")['input_ids']
tokenizer(" Hello world")['input_ids']`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt;</span> <span class="language-python"><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BloomTokenizerFast</span>
<span class="hljs-meta">&gt;&gt;&gt;</span> <span class="language-python">tokenizer = BloomTokenizerFast.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom&quot;</span>)</span>
<span class="hljs-meta">&gt;&gt;&gt;</span> <span class="language-python">tokenizer(<span class="hljs-string">&quot;Hello world&quot;</span>)[<span class="hljs-string">&#x27;input_ids&#x27;</span>]</span>
[15496, 995]
<span class="hljs-meta">&gt;&gt;&gt;</span> <span class="language-python">tokenizer(<span class="hljs-string">&quot; Hello world&quot;</span>)[<span class="hljs-string">&#x27;input_ids&#x27;</span>]</span>
[18435, 995]`}}),{c(){d=s("p"),g=l("be encoded differently whether it is at the beginning of the sentence (without space) or not:"),c=u(),k(m.$$.fragment)},l(n){d=a(n,"P",{});var h=r(d);g=i(h,"be encoded differently whether it is at the beginning of the sentence (without space) or not:"),h.forEach(o),c=f(n),y(m.$$.fragment,n)},m(n,h){v(n,d,h),e(d,g),v(n,c,h),$(m,n,h),b=!0},p:ee,i(n){b||(w(m.$$.fragment,n),b=!0)},o(n){T(m.$$.fragment,n),b=!1},d(n){n&&o(d),n&&o(c),F(m,n)}}}function Ac(B){let d,g,c,m,b,n,h,x;return{c(){d=s("p"),g=l("When used with "),c=s("code"),m=l("is_split_into_words=True"),b=l(", this tokenizer needs to be instantiated with "),n=s("code"),h=l("add_prefix_space=True"),x=l(".")},l(pe){d=a(pe,"P",{});var V=r(d);g=i(V,"When used with "),c=a(V,"CODE",{});var R=r(c);m=i(R,"is_split_into_words=True"),R.forEach(o),b=i(V,", this tokenizer needs to be instantiated with "),n=a(V,"CODE",{});var te=r(n);h=i(te,"add_prefix_space=True"),te.forEach(o),x=i(V,"."),V.forEach(o)},m(pe,V){v(pe,d,V),e(d,g),e(d,c),e(c,m),e(d,b),e(d,n),e(n,h),e(d,x)},d(pe){pe&&o(d)}}}function Sc(B){let d,g,c,m,b;return{c(){d=s("p"),g=l("Although the recipe for forward pass needs to be defined within this function, one should call the "),c=s("code"),m=l("Module"),b=l(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(n){d=a(n,"P",{});var h=r(d);g=i(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),c=a(h,"CODE",{});var x=r(c);m=i(x,"Module"),x.forEach(o),b=i(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(o)},m(n,h){v(n,d,h),e(d,g),e(d,c),e(c,m),e(d,b)},d(n){n&&o(d)}}}function Ic(B){let d,g,c,m,b;return m=new oe({props:{code:`import torch
from transformers import BloomTokenizerFast, BloomForCausalLM

tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-350m")
model = BloomForCausalLM.from_pretrained("bigscience/bloom-350m")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
logits = outputs.logits`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BloomTokenizerFast, BloomForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BloomTokenizerFast.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom-350m&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BloomForCausalLM.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom-350m&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs, labels=inputs[<span class="hljs-string">&quot;input_ids&quot;</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`}}),{c(){d=s("p"),g=l("Example:"),c=u(),k(m.$$.fragment)},l(n){d=a(n,"P",{});var h=r(d);g=i(h,"Example:"),h.forEach(o),c=f(n),y(m.$$.fragment,n)},m(n,h){v(n,d,h),e(d,g),v(n,c,h),$(m,n,h),b=!0},p:ee,i(n){b||(w(m.$$.fragment,n),b=!0)},o(n){T(m.$$.fragment,n),b=!1},d(n){n&&o(d),n&&o(c),F(m,n)}}}function Nc(B){let d,g,c,m,b;return{c(){d=s("p"),g=l("Although the recipe for forward pass needs to be defined within this function, one should call the "),c=s("code"),m=l("Module"),b=l(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(n){d=a(n,"P",{});var h=r(d);g=i(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),c=a(h,"CODE",{});var x=r(c);m=i(x,"Module"),x.forEach(o),b=i(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(o)},m(n,h){v(n,d,h),e(d,g),e(d,c),e(c,m),e(d,b)},d(n){n&&o(d)}}}function Dc(B){let d,g,c,m,b;return m=new oe({props:{code:`import torch
from transformers import BloomTokenizerFast, BloomForSequenceClassification

tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-350m")
model = BloomForSequenceClassification.from_pretrained("bigscience/bloom-350m")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BloomTokenizerFast, BloomForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BloomTokenizerFast.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom-350m&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BloomForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom-350m&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
`}}),{c(){d=s("p"),g=l("Example of single-label classification:"),c=u(),k(m.$$.fragment)},l(n){d=a(n,"P",{});var h=r(d);g=i(h,"Example of single-label classification:"),h.forEach(o),c=f(n),y(m.$$.fragment,n)},m(n,h){v(n,d,h),e(d,g),v(n,c,h),$(m,n,h),b=!0},p:ee,i(n){b||(w(m.$$.fragment,n),b=!0)},o(n){T(m.$$.fragment,n),b=!1},d(n){n&&o(d),n&&o(c),F(m,n)}}}function Wc(B){let d,g;return d=new oe({props:{code:`# To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`
num_labels = len(model.config.id2label)
model = BloomForSequenceClassification.from_pretrained("bigscience/bloom-350m", num_labels=num_labels)

labels = torch.tensor(1)
loss = model(**inputs, labels=labels).loss
round(loss.item(), 2)
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BloomForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom-350m&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">1</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
`}}),{c(){k(d.$$.fragment)},l(c){y(d.$$.fragment,c)},m(c,m){$(d,c,m),g=!0},p:ee,i(c){g||(w(d.$$.fragment,c),g=!0)},o(c){T(d.$$.fragment,c),g=!1},d(c){F(d,c)}}}function Hc(B){let d,g,c,m,b;return m=new oe({props:{code:`import torch
from transformers import BloomTokenizerFast, BloomForSequenceClassification

tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-350m")
model = BloomForSequenceClassification.from_pretrained("bigscience/bloom-350m", problem_type="multi_label_classification")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BloomTokenizerFast, BloomForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BloomTokenizerFast.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom-350m&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BloomForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom-350m&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
`}}),{c(){d=s("p"),g=l("Example of multi-label classification:"),c=u(),k(m.$$.fragment)},l(n){d=a(n,"P",{});var h=r(d);g=i(h,"Example of multi-label classification:"),h.forEach(o),c=f(n),y(m.$$.fragment,n)},m(n,h){v(n,d,h),e(d,g),v(n,c,h),$(m,n,h),b=!0},p:ee,i(n){b||(w(m.$$.fragment,n),b=!0)},o(n){T(m.$$.fragment,n),b=!1},d(n){n&&o(d),n&&o(c),F(m,n)}}}function Uc(B){let d,g;return d=new oe({props:{code:`# To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`
num_labels = len(model.config.id2label)
model = BloomForSequenceClassification.from_pretrained(
    "bigscience/bloom-350m", num_labels=num_labels, problem_type="multi_label_classification"
)

labels = torch.nn.functional.one_hot(torch.tensor([predicted_class_id]), num_classes=num_labels).to(
    torch.float
)
loss = model(**inputs, labels=labels).loss
loss.backward()`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BloomForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;bigscience/bloom-350m&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.nn.functional.one_hot(torch.tensor([predicted_class_id]), num_classes=num_labels).to(
<span class="hljs-meta">... </span>    torch.<span class="hljs-built_in">float</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span>loss.backward()`}}),{c(){k(d.$$.fragment)},l(c){y(d.$$.fragment,c)},m(c,m){$(d,c,m),g=!0},p:ee,i(c){g||(w(d.$$.fragment,c),g=!0)},o(c){T(d.$$.fragment,c),g=!1},d(c){F(d,c)}}}function Vc(B){let d,g,c,m,b;return{c(){d=s("p"),g=l("Although the recipe for forward pass needs to be defined within this function, one should call the "),c=s("code"),m=l("Module"),b=l(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(n){d=a(n,"P",{});var h=r(d);g=i(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),c=a(h,"CODE",{});var x=r(c);m=i(x,"Module"),x.forEach(o),b=i(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(o)},m(n,h){v(n,d,h),e(d,g),e(d,c),e(c,m),e(d,b)},d(n){n&&o(d)}}}function Jc(B){let d,g,c,m,b;return m=new oe({props:{code:`from transformers import BloomTokenizerFast, BloomForTokenClassification
import torch

tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-350m")
model = BloomForTokenClassification.from_pretrained("bigscience/bloom-350m")

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

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BloomTokenizerFast.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom-350m&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BloomForTokenClassification.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom-350m&quot;</span>)

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
`}}),{c(){d=s("p"),g=l("Example:"),c=u(),k(m.$$.fragment)},l(n){d=a(n,"P",{});var h=r(d);g=i(h,"Example:"),h.forEach(o),c=f(n),y(m.$$.fragment,n)},m(n,h){v(n,d,h),e(d,g),v(n,c,h),$(m,n,h),b=!0},p:ee,i(n){b||(w(m.$$.fragment,n),b=!0)},o(n){T(m.$$.fragment,n),b=!1},d(n){n&&o(d),n&&o(c),F(m,n)}}}function Gc(B){let d,g;return d=new oe({props:{code:`labels = predicted_token_class_ids
loss = model(**inputs, labels=labels).loss
round(loss.item(), 2)
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span>labels = predicted_token_class_ids
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
`}}),{c(){k(d.$$.fragment)},l(c){y(d.$$.fragment,c)},m(c,m){$(d,c,m),g=!0},p:ee,i(c){g||(w(d.$$.fragment,c),g=!0)},o(c){T(d.$$.fragment,c),g=!1},d(c){F(d,c)}}}function Rc(B){let d,g,c,m,b;return{c(){d=s("p"),g=l("Although the recipe for forward pass needs to be defined within this function, one should call the "),c=s("code"),m=l("Module"),b=l(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(n){d=a(n,"P",{});var h=r(d);g=i(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),c=a(h,"CODE",{});var x=r(c);m=i(x,"Module"),x.forEach(o),b=i(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(o)},m(n,h){v(n,d,h),e(d,g),e(d,c),e(c,m),e(d,b)},d(n){n&&o(d)}}}function Xc(B){let d,g,c,m,b;return m=new oe({props:{code:`from transformers import BloomTokenizerFast, FlaxBloomModel

tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom")
model = FlaxBloomModel.from_pretrained("bigscience/bloom")

inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BloomTokenizerFast, FlaxBloomModel

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BloomTokenizerFast.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FlaxBloomModel.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;jax&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`}}),{c(){d=s("p"),g=l("Example:"),c=u(),k(m.$$.fragment)},l(n){d=a(n,"P",{});var h=r(d);g=i(h,"Example:"),h.forEach(o),c=f(n),y(m.$$.fragment,n)},m(n,h){v(n,d,h),e(d,g),v(n,c,h),$(m,n,h),b=!0},p:ee,i(n){b||(w(m.$$.fragment,n),b=!0)},o(n){T(m.$$.fragment,n),b=!1},d(n){n&&o(d),n&&o(c),F(m,n)}}}function Yc(B){let d,g,c,m,b;return{c(){d=s("p"),g=l("Although the recipe for forward pass needs to be defined within this function, one should call the "),c=s("code"),m=l("Module"),b=l(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(n){d=a(n,"P",{});var h=r(d);g=i(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),c=a(h,"CODE",{});var x=r(c);m=i(x,"Module"),x.forEach(o),b=i(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(o)},m(n,h){v(n,d,h),e(d,g),e(d,c),e(c,m),e(d,b)},d(n){n&&o(d)}}}function Kc(B){let d,g,c,m,b;return m=new oe({props:{code:`from transformers import BloomTokenizerFast, FlaxBloomForCausalLM

tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom")
model = FlaxBloomForCausalLM.from_pretrained("bigscience/bloom")

inputs = tokenizer("Hello, my dog is cute", return_tensors="np")
outputs = model(**inputs)

# retrieve logts for next token
next_token_logits = outputs.logits[:, -1]`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BloomTokenizerFast, FlaxBloomForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BloomTokenizerFast.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FlaxBloomForCausalLM.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;np&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># retrieve logts for next token</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>next_token_logits = outputs.logits[:, -<span class="hljs-number">1</span>]`}}),{c(){d=s("p"),g=l("Example:"),c=u(),k(m.$$.fragment)},l(n){d=a(n,"P",{});var h=r(d);g=i(h,"Example:"),h.forEach(o),c=f(n),y(m.$$.fragment,n)},m(n,h){v(n,d,h),e(d,g),v(n,c,h),$(m,n,h),b=!0},p:ee,i(n){b||(w(m.$$.fragment,n),b=!0)},o(n){T(m.$$.fragment,n),b=!1},d(n){n&&o(d),n&&o(c),F(m,n)}}}function Qc(B){let d,g,c,m,b,n,h,x,pe,V,R,te,Vt,co,Xs,Jt,Ys,ls,Me,Ks,po,Qs,Zs,is,P,Gt,mo,ea,oa,Rt,ho,ta,na,Xt,uo,sa,aa,Yt,fo,ra,la,Kt,go,ia,da,kt,_o,ca,pa,ds,me,ze,Qt,bo,ma,Zt,ha,cs,J,vo,ua,he,fa,yt,ga,_a,ko,ba,va,ka,ue,ya,$t,$a,wa,wt,Ta,Fa,Ba,Ce,ps,fe,Ee,en,yo,xa,on,Ma,ms,q,$o,za,tn,Ca,Ea,wo,ja,Tt,Pa,qa,Oa,To,La,Fo,Aa,Sa,Ia,X,Bo,Na,ge,Da,Ft,Wa,Ha,nn,Ua,Va,Ja,je,Ga,Pe,hs,_e,qe,sn,xo,Ra,an,Xa,us,M,Mo,Ya,zo,Ka,rn,Qa,Za,er,ln,or,tr,Oe,nr,Co,sr,dn,ar,rr,lr,Le,ir,Eo,dr,Bt,cr,pr,fs,be,Ae,cn,jo,mr,pn,hr,gs,O,Po,ur,mn,fr,gr,qo,_r,xt,br,vr,kr,Oo,yr,Lo,$r,wr,Tr,Y,Ao,Fr,ve,Br,Mt,xr,Mr,hn,zr,Cr,Er,Se,jr,Ie,_s,ke,Ne,un,So,Pr,fn,qr,bs,z,Io,Or,gn,Lr,Ar,zt,Ct,Sr,Ir,Nr,G,Dr,_n,Wr,Hr,bn,Ur,Vr,vn,Jr,Gr,kn,Rr,Xr,Yr,No,Kr,Et,Qr,Zr,el,Do,ol,Wo,tl,nl,sl,j,Ho,al,ye,rl,jt,ll,il,yn,dl,cl,pl,De,ml,We,hl,He,ul,Ue,fl,Ve,vs,$e,Je,$n,Uo,gl,wn,_l,ks,L,Vo,bl,Tn,vl,kl,Jo,yl,Pt,$l,wl,Tl,Go,Fl,Ro,Bl,xl,Ml,S,Xo,zl,we,Cl,qt,El,jl,Fn,Pl,ql,Ol,Ge,Ll,Re,Al,Xe,ys,Te,Ye,Bn,Yo,Sl,xn,Il,$s,C,Ko,Nl,Mn,Dl,Wl,Qo,Hl,Ot,Ul,Vl,Jl,Zo,Gl,et,Rl,Xl,Yl,zn,Kl,Ql,ne,Cn,ot,Zl,ei,En,tt,oi,ti,jn,nt,ni,si,Pn,st,ai,ri,K,at,li,Fe,ii,qn,di,ci,On,pi,mi,hi,Ke,ui,Qe,ws,Be,Ze,Ln,rt,fi,An,gi,Ts,E,lt,_i,Sn,bi,vi,it,ki,Lt,yi,$i,wi,dt,Ti,ct,Fi,Bi,xi,In,Mi,zi,se,Nn,pt,Ci,Ei,Dn,mt,ji,Pi,Wn,ht,qi,Oi,Hn,ut,Li,Ai,Q,ft,Si,xe,Ii,Un,Ni,Di,Vn,Wi,Hi,Ui,eo,Vi,oo,Fs;return n=new ce({}),co=new ce({}),bo=new ce({}),vo=new U({props:{name:"class transformers.BloomConfig",anchor:"transformers.BloomConfig",parameters:[{name:"vocab_size",val:" = 250880"},{name:"hidden_size",val:" = 64"},{name:"n_layer",val:" = 2"},{name:"n_head",val:" = 8"},{name:"layer_norm_epsilon",val:" = 1e-05"},{name:"initializer_range",val:" = 0.02"},{name:"use_cache",val:" = False"},{name:"bos_token_id",val:" = 1"},{name:"eos_token_id",val:" = 2"},{name:"apply_residual_connection_post_layernorm",val:" = False"},{name:"hidden_dropout",val:" = 0.0"},{name:"attention_dropout",val:" = 0.0"},{name:"pretraining_tp",val:" = 1"},{name:"slow_but_exact",val:" = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BloomConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 50257) &#x2014;
Vocabulary size of the Bloom model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/pr_18022/en/model_doc/bloom#transformers.BloomModel">BloomModel</a>.`,name:"vocab_size"},{anchor:"transformers.BloomConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
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
resolved in the future once the main model has been fine-tuned with TP_rank=1.`,name:"slow_but_exact"}],source:"https://github.com/huggingface/transformers/blob/vr_18022/src/transformers/models/bloom/configuration_bloom.py#L42"}}),Ce=new Z({props:{anchor:"transformers.BloomConfig.example",$$slots:{default:[Pc]},$$scope:{ctx:B}}}),yo=new ce({}),$o=new U({props:{name:"class transformers.BloomModel",anchor:"transformers.BloomModel",parameters:[{name:"config",val:": BloomConfig"}],parametersDescription:[{anchor:"transformers.BloomModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18022/en/model_doc/bloom#transformers.BloomConfig">BloomConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18022/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18022/src/transformers/models/bloom/modeling_bloom.py#L583"}}),Bo=new U({props:{name:"forward",anchor:"transformers.BloomModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Union[typing.Tuple[typing.Tuple[torch.Tensor, torch.Tensor], ...], NoneType] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**deprecated_arguments",val:""}],parametersDescription:[{anchor:"transformers.BloomModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else <code>past_key_values[0][0].shape[2]</code>
(<code>sequence_length</code> of input past key value states). Indices of input sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18022/en/model_doc/bloom#transformers.BloomTokenizerFast">BloomTokenizerFast</a>. See <a href="/docs/transformers/pr_18022/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18022/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
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
Whether or not to return a <a href="/docs/transformers/pr_18022/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18022/src/transformers/models/bloom/modeling_bloom.py#L633",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18022/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18022/en/model_doc/bloom#transformers.BloomConfig"
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
  href="/docs/transformers/pr_18022/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),je=new vt({props:{$$slots:{default:[qc]},$$scope:{ctx:B}}}),Pe=new Z({props:{anchor:"transformers.BloomModel.forward.example",$$slots:{default:[Oc]},$$scope:{ctx:B}}}),xo=new ce({}),Mo=new U({props:{name:"class transformers.BloomTokenizerFast",anchor:"transformers.BloomTokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"merges_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"unk_token",val:" = '<unk>'"},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"pad_token",val:" = '<pad>'"},{name:"add_prefix_space",val:" = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BloomTokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
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
Whether or not the post-processing step should trim offsets to avoid including whitespaces.`,name:"trim_offsets"}],source:"https://github.com/huggingface/transformers/blob/vr_18022/src/transformers/models/bloom/tokenization_bloom_fast.py#L49"}}),Oe=new Z({props:{anchor:"transformers.BloomTokenizerFast.example",$$slots:{default:[Lc]},$$scope:{ctx:B}}}),Le=new vt({props:{$$slots:{default:[Ac]},$$scope:{ctx:B}}}),jo=new ce({}),Po=new U({props:{name:"class transformers.BloomForCausalLM",anchor:"transformers.BloomForCausalLM",parameters:[{name:"config",val:": BloomConfig"}],parametersDescription:[{anchor:"transformers.BloomForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18022/en/model_doc/bloom#transformers.BloomConfig">BloomConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18022/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18022/src/transformers/models/bloom/modeling_bloom.py#L785"}}),Ao=new U({props:{name:"forward",anchor:"transformers.BloomForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Union[typing.Tuple[typing.Tuple[torch.Tensor, torch.Tensor], ...], NoneType] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**deprecated_arguments",val:""}],parametersDescription:[{anchor:"transformers.BloomForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else <code>past_key_values[0][0].shape[2]</code>
(<code>sequence_length</code> of input past key value states). Indices of input sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18022/en/model_doc/bloom#transformers.BloomTokenizerFast">BloomTokenizerFast</a>. See <a href="/docs/transformers/pr_18022/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18022/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
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
Whether or not to return a <a href="/docs/transformers/pr_18022/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BloomForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for language modeling. Note that the labels <strong>are shifted</strong> inside the model, i.e. you can set
<code>labels = input_ids</code> Indices are selected in <code>[-100, 0, ..., config.vocab_size]</code> All labels set to <code>-100</code>
are ignored (masked), the loss is only computed for labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18022/src/transformers/models/bloom/modeling_bloom.py#L820",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18022/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18022/en/model_doc/bloom#transformers.BloomConfig"
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
  href="/docs/transformers/pr_18022/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Se=new vt({props:{$$slots:{default:[Sc]},$$scope:{ctx:B}}}),Ie=new Z({props:{anchor:"transformers.BloomForCausalLM.forward.example",$$slots:{default:[Ic]},$$scope:{ctx:B}}}),So=new ce({}),Io=new U({props:{name:"class transformers.BloomForSequenceClassification",anchor:"transformers.BloomForSequenceClassification",parameters:[{name:"config",val:": BloomConfig"}],parametersDescription:[{anchor:"transformers.BloomForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18022/en/model_doc/bloom#transformers.BloomConfig">BloomConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18022/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18022/src/transformers/models/bloom/modeling_bloom.py#L948"}}),Ho=new U({props:{name:"forward",anchor:"transformers.BloomForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Union[typing.Tuple[typing.Tuple[torch.Tensor, torch.Tensor], ...], NoneType] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**deprecated_arguments",val:""}],parametersDescription:[{anchor:"transformers.BloomForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else <code>past_key_values[0][0].shape[2]</code>
(<code>sequence_length</code> of input past key value states). Indices of input sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18022/en/model_doc/bloom#transformers.BloomTokenizerFast">BloomTokenizerFast</a>. See <a href="/docs/transformers/pr_18022/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18022/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
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
Whether or not to return a <a href="/docs/transformers/pr_18022/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BloomForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18022/src/transformers/models/bloom/modeling_bloom.py#L960",returnDescription:`
<p>A <code>transformers.modeling_outputs.SequenceClassifierOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18022/en/model_doc/bloom#transformers.BloomConfig"
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
`}}),De=new vt({props:{$$slots:{default:[Nc]},$$scope:{ctx:B}}}),We=new Z({props:{anchor:"transformers.BloomForSequenceClassification.forward.example",$$slots:{default:[Dc]},$$scope:{ctx:B}}}),He=new Z({props:{anchor:"transformers.BloomForSequenceClassification.forward.example-2",$$slots:{default:[Wc]},$$scope:{ctx:B}}}),Ue=new Z({props:{anchor:"transformers.BloomForSequenceClassification.forward.example-3",$$slots:{default:[Hc]},$$scope:{ctx:B}}}),Ve=new Z({props:{anchor:"transformers.BloomForSequenceClassification.forward.example-4",$$slots:{default:[Uc]},$$scope:{ctx:B}}}),Uo=new ce({}),Vo=new U({props:{name:"class transformers.BloomForTokenClassification",anchor:"transformers.BloomForTokenClassification",parameters:[{name:"config",val:": BloomConfig"}],parametersDescription:[{anchor:"transformers.BloomForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18022/en/model_doc/bloom#transformers.BloomConfig">BloomConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18022/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18022/src/transformers/models/bloom/modeling_bloom.py#L1077"}}),Xo=new U({props:{name:"forward",anchor:"transformers.BloomForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Union[typing.Tuple[typing.Tuple[torch.Tensor, torch.Tensor], ...], NoneType] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**deprecated_arguments",val:""}],parametersDescription:[{anchor:"transformers.BloomForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else <code>past_key_values[0][0].shape[2]</code>
(<code>sequence_length</code> of input past key value states). Indices of input sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18022/en/model_doc/bloom#transformers.BloomTokenizerFast">BloomTokenizerFast</a>. See <a href="/docs/transformers/pr_18022/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18022/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
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
Whether or not to return a <a href="/docs/transformers/pr_18022/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BloomForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18022/src/transformers/models/bloom/modeling_bloom.py#L1097",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18022/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18022/en/model_doc/bloom#transformers.BloomConfig"
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
  href="/docs/transformers/pr_18022/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Ge=new vt({props:{$$slots:{default:[Vc]},$$scope:{ctx:B}}}),Re=new Z({props:{anchor:"transformers.BloomForTokenClassification.forward.example",$$slots:{default:[Jc]},$$scope:{ctx:B}}}),Xe=new Z({props:{anchor:"transformers.BloomForTokenClassification.forward.example-2",$$slots:{default:[Gc]},$$scope:{ctx:B}}}),Yo=new ce({}),Ko=new U({props:{name:"class transformers.FlaxBloomModel",anchor:"transformers.FlaxBloomModel",parameters:[{name:"config",val:": BloomConfig"},{name:"input_shape",val:": typing.Tuple = (1, 1)"},{name:"seed",val:": int = 0"},{name:"dtype",val:": dtype = <class 'jax.numpy.float32'>"},{name:"_do_init",val:": bool = True"},{name:"use_scan",val:": bool = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.FlaxBloomModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18022/en/model_doc/bloom#transformers.BloomConfig">BloomConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18022/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.FlaxBloomModel.dtype",description:`<strong>dtype</strong> (<code>jax.numpy.dtype</code>, <em>optional</em>, defaults to <code>jax.numpy.float32</code>) &#x2014;
The data type of the computation. Can be one of <code>jax.numpy.float32</code>, <code>jax.numpy.float16</code> (on GPUs) and
<code>jax.numpy.bfloat16</code> (on TPUs).</p>
<p>This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
specified all the computation will be performed with the given <code>dtype</code>.</p>
<p><strong>Note that this only specifies the dtype of the computation and does not influence the dtype of model
parameters.</strong></p>
<p>If you wish to change the dtype of the model parameters, see <a href="/docs/transformers/pr_18022/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16">to_fp16()</a> and
<a href="/docs/transformers/pr_18022/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16">to_bf16()</a>.`,name:"dtype"}],source:"https://github.com/huggingface/transformers/blob/vr_18022/src/transformers/models/bloom/modeling_flax_bloom.py#L680"}}),at=new U({props:{name:"__call__",anchor:"transformers.FlaxBloomModel.__call__",parameters:[{name:"input_ids",val:""},{name:"attention_mask",val:" = None"},{name:"past_key_values",val:": dict = None"},{name:"params",val:": dict = None"},{name:"dropout_rng",val:": PRNGKey = None"},{name:"train",val:": bool = False"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.FlaxBloomModel.__call__.input_ids",description:`<strong>input_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code>. Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <code>BloomTokenizer</code>. See <a href="/docs/transformers/pr_18022/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18022/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FlaxBloomModel.__call__.attention_mask",description:`<strong>attention_mask</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.FlaxBloomModel.__call__.past_key_values",description:`<strong>past_key_values</strong> (<code>Dict[str, np.ndarray]</code>, <em>optional</em>, returned by <code>init_cache</code> or when passing previous <code>past_key_values</code>) &#x2014;
Dictionary of pre-computed hidden-states (key and values in the attention blocks) that can be used for fast
auto-regressive decoding. Pre-computed key and value hidden-states are of shape <em>[batch_size, max_length]</em>.`,name:"past_key_values"},{anchor:"transformers.FlaxBloomModel.__call__.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.FlaxBloomModel.__call__.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.FlaxBloomModel.__call__.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18022/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18022/src/transformers/models/bloom/modeling_flax_bloom.py#L468",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18022/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutput"
>transformers.modeling_flax_outputs.FlaxBaseModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18022/en/model_doc/bloom#transformers.BloomConfig"
>BloomConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) \u2014 Sequence of hidden-states at the output of the last layer of the model.</p>
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
  href="/docs/transformers/pr_18022/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutput"
>transformers.modeling_flax_outputs.FlaxBaseModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Ke=new vt({props:{$$slots:{default:[Rc]},$$scope:{ctx:B}}}),Qe=new Z({props:{anchor:"transformers.FlaxBloomModel.__call__.example",$$slots:{default:[Xc]},$$scope:{ctx:B}}}),rt=new ce({}),lt=new U({props:{name:"class transformers.FlaxBloomForCausalLM",anchor:"transformers.FlaxBloomForCausalLM",parameters:[{name:"config",val:": BloomConfig"},{name:"input_shape",val:": typing.Tuple = (1, 1)"},{name:"seed",val:": int = 0"},{name:"dtype",val:": dtype = <class 'jax.numpy.float32'>"},{name:"_do_init",val:": bool = True"},{name:"use_scan",val:": bool = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.FlaxBloomForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18022/en/model_doc/bloom#transformers.BloomConfig">BloomConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18022/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.FlaxBloomForCausalLM.dtype",description:`<strong>dtype</strong> (<code>jax.numpy.dtype</code>, <em>optional</em>, defaults to <code>jax.numpy.float32</code>) &#x2014;
The data type of the computation. Can be one of <code>jax.numpy.float32</code>, <code>jax.numpy.float16</code> (on GPUs) and
<code>jax.numpy.bfloat16</code> (on TPUs).</p>
<p>This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
specified all the computation will be performed with the given <code>dtype</code>.</p>
<p><strong>Note that this only specifies the dtype of the computation and does not influence the dtype of model
parameters.</strong></p>
<p>If you wish to change the dtype of the model parameters, see <a href="/docs/transformers/pr_18022/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16">to_fp16()</a> and
<a href="/docs/transformers/pr_18022/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16">to_bf16()</a>.`,name:"dtype"}],source:"https://github.com/huggingface/transformers/blob/vr_18022/src/transformers/models/bloom/modeling_flax_bloom.py#L744"}}),ft=new U({props:{name:"__call__",anchor:"transformers.FlaxBloomForCausalLM.__call__",parameters:[{name:"input_ids",val:""},{name:"attention_mask",val:" = None"},{name:"past_key_values",val:": dict = None"},{name:"params",val:": dict = None"},{name:"dropout_rng",val:": PRNGKey = None"},{name:"train",val:": bool = False"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.FlaxBloomForCausalLM.__call__.input_ids",description:`<strong>input_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code>. Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <code>BloomTokenizer</code>. See <a href="/docs/transformers/pr_18022/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18022/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FlaxBloomForCausalLM.__call__.attention_mask",description:`<strong>attention_mask</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.FlaxBloomForCausalLM.__call__.past_key_values",description:`<strong>past_key_values</strong> (<code>Dict[str, np.ndarray]</code>, <em>optional</em>, returned by <code>init_cache</code> or when passing previous <code>past_key_values</code>) &#x2014;
Dictionary of pre-computed hidden-states (key and values in the attention blocks) that can be used for fast
auto-regressive decoding. Pre-computed key and value hidden-states are of shape <em>[batch_size, max_length]</em>.`,name:"past_key_values"},{anchor:"transformers.FlaxBloomForCausalLM.__call__.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.FlaxBloomForCausalLM.__call__.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.FlaxBloomForCausalLM.__call__.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18022/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18022/src/transformers/models/bloom/modeling_flax_bloom.py#L468",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18022/en/main_classes/output#transformers.modeling_flax_outputs.FlaxMaskedLMOutput"
>transformers.modeling_flax_outputs.FlaxMaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18022/en/model_doc/bloom#transformers.BloomConfig"
>BloomConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18022/en/main_classes/output#transformers.modeling_flax_outputs.FlaxMaskedLMOutput"
>transformers.modeling_flax_outputs.FlaxMaskedLMOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),eo=new vt({props:{$$slots:{default:[Yc]},$$scope:{ctx:B}}}),oo=new Z({props:{anchor:"transformers.FlaxBloomForCausalLM.__call__.example",$$slots:{default:[Kc]},$$scope:{ctx:B}}}),{c(){d=s("meta"),g=u(),c=s("h1"),m=s("a"),b=s("span"),k(n.$$.fragment),h=u(),x=s("span"),pe=l("BLOOM"),V=u(),R=s("h2"),te=s("a"),Vt=s("span"),k(co.$$.fragment),Xs=u(),Jt=s("span"),Ys=l("Overview"),ls=u(),Me=s("p"),Ks=l("The BLOOM model has been proposed with its various versions through the "),po=s("a"),Qs=l("BigScience Workshop"),Zs=l(`. BigScience is inspired by other open science initiatives where researchers have pooled their time and resources to collectively achieve a higher impact.
The architecture of BLOOM is essentially similar to GPT3 (auto-regressive model for next token prediction), but has been trained on 46 different languages and 13 programming languages.
Several smaller versions of the models have been trained on the same dataset. BLOOM is available in the following versions:`),is=u(),P=s("ul"),Gt=s("li"),mo=s("a"),ea=l("bloom-350m"),oa=u(),Rt=s("li"),ho=s("a"),ta=l("bloom-760m"),na=u(),Xt=s("li"),uo=s("a"),sa=l("bloom-1b3"),aa=u(),Yt=s("li"),fo=s("a"),ra=l("bloom-2b5"),la=u(),Kt=s("li"),go=s("a"),ia=l("bloom-6b3"),da=u(),kt=s("li"),_o=s("a"),ca=l("bloom"),pa=l(" (176B parameters)"),ds=u(),me=s("h2"),ze=s("a"),Qt=s("span"),k(bo.$$.fragment),ma=u(),Zt=s("span"),ha=l("BloomConfig"),cs=u(),J=s("div"),k(vo.$$.fragment),ua=u(),he=s("p"),fa=l("This is the configuration class to store the configuration of a "),yt=s("a"),ga=l("BloomModel"),_a=l(`. It is used to instantiate a Bloom
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to the Bloom architecture
`),ko=s("a"),ba=l("bigscience/bloom"),va=l("."),ka=u(),ue=s("p"),ya=l("Configuration objects inherit from "),$t=s("a"),$a=l("PretrainedConfig"),wa=l(` and can be used to control the model outputs. Read the
documentation from `),wt=s("a"),Ta=l("PretrainedConfig"),Fa=l(" for more information."),Ba=u(),k(Ce.$$.fragment),ps=u(),fe=s("h2"),Ee=s("a"),en=s("span"),k(yo.$$.fragment),xa=u(),on=s("span"),Ma=l("BloomModel"),ms=u(),q=s("div"),k($o.$$.fragment),za=u(),tn=s("p"),Ca=l("The bare Bloom Model transformer outputting raw hidden-states without any specific head on top."),Ea=u(),wo=s("p"),ja=l("This model inherits from "),Tt=s("a"),Pa=l("PreTrainedModel"),qa=l(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)`),Oa=u(),To=s("p"),La=l("This model is also a PyTorch "),Fo=s("a"),Aa=l("torch.nn.Module"),Sa=l(` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Ia=u(),X=s("div"),k(Bo.$$.fragment),Na=u(),ge=s("p"),Da=l("The "),Ft=s("a"),Wa=l("BloomModel"),Ha=l(" forward method, overrides the "),nn=s("code"),Ua=l("__call__"),Va=l(" special method."),Ja=u(),k(je.$$.fragment),Ga=u(),k(Pe.$$.fragment),hs=u(),_e=s("h2"),qe=s("a"),sn=s("span"),k(xo.$$.fragment),Ra=u(),an=s("span"),Xa=l("BloomTokenizerFast"),us=u(),M=s("div"),k(Mo.$$.fragment),Ya=u(),zo=s("p"),Ka=l("Construct a \u201Cfast\u201D Bloom tokenizer (backed by HuggingFace\u2019s "),rn=s("em"),Qa=l("tokenizers"),Za=l(` library). Based on byte-level
Byte-Pair-Encoding.`),er=u(),ln=s("p"),or=l("This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will"),tr=u(),k(Oe.$$.fragment),nr=u(),Co=s("p"),sr=l("You can get around that behavior by passing "),dn=s("code"),ar=l("add_prefix_space=True"),rr=l(` when instantiating this tokenizer, but since
the model was not pretrained this way, it might yield a decrease in performance.`),lr=u(),k(Le.$$.fragment),ir=u(),Eo=s("p"),dr=l("This tokenizer inherits from "),Bt=s("a"),cr=l("PreTrainedTokenizerFast"),pr=l(` which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`),fs=u(),be=s("h2"),Ae=s("a"),cn=s("span"),k(jo.$$.fragment),mr=u(),pn=s("span"),hr=l("BloomForCausalLM"),gs=u(),O=s("div"),k(Po.$$.fragment),ur=u(),mn=s("p"),fr=l(`The Bloom Model transformer with a language modeling head on top (linear layer with weights tied to the input
embeddings).`),gr=u(),qo=s("p"),_r=l("This model inherits from "),xt=s("a"),br=l("PreTrainedModel"),vr=l(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)`),kr=u(),Oo=s("p"),yr=l("This model is also a PyTorch "),Lo=s("a"),$r=l("torch.nn.Module"),wr=l(` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Tr=u(),Y=s("div"),k(Ao.$$.fragment),Fr=u(),ve=s("p"),Br=l("The "),Mt=s("a"),xr=l("BloomForCausalLM"),Mr=l(" forward method, overrides the "),hn=s("code"),zr=l("__call__"),Cr=l(" special method."),Er=u(),k(Se.$$.fragment),jr=u(),k(Ie.$$.fragment),_s=u(),ke=s("h2"),Ne=s("a"),un=s("span"),k(So.$$.fragment),Pr=u(),fn=s("span"),qr=l("BloomForSequenceClassification"),bs=u(),z=s("div"),k(Io.$$.fragment),Or=u(),gn=s("p"),Lr=l("The Bloom Model transformer with a sequence classification head on top (linear layer)."),Ar=u(),zt=s("p"),Ct=s("a"),Sr=l("BloomForSequenceClassification"),Ir=l(` uses the last token in order to do the classification, as other causal models
(e.g. GPT-1) do.`),Nr=u(),G=s("p"),Dr=l(`Since it does classification on the last token, it requires to know the position of the last token. If a
`),_n=s("code"),Wr=l("pad_token_id"),Hr=l(` is defined in the configuration, it finds the last token that is not a padding token in each row. If
no `),bn=s("code"),Ur=l("pad_token_id"),Vr=l(` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
padding tokens when `),vn=s("code"),Jr=l("inputs_embeds"),Gr=l(" are passed instead of "),kn=s("code"),Rr=l("input_ids"),Xr=l(`, it does the same (take the last value in
each row of the batch).`),Yr=u(),No=s("p"),Kr=l("This model inherits from "),Et=s("a"),Qr=l("PreTrainedModel"),Zr=l(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)`),el=u(),Do=s("p"),ol=l("This model is also a PyTorch "),Wo=s("a"),tl=l("torch.nn.Module"),nl=l(` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),sl=u(),j=s("div"),k(Ho.$$.fragment),al=u(),ye=s("p"),rl=l("The "),jt=s("a"),ll=l("BloomForSequenceClassification"),il=l(" forward method, overrides the "),yn=s("code"),dl=l("__call__"),cl=l(" special method."),pl=u(),k(De.$$.fragment),ml=u(),k(We.$$.fragment),hl=u(),k(He.$$.fragment),ul=u(),k(Ue.$$.fragment),fl=u(),k(Ve.$$.fragment),vs=u(),$e=s("h2"),Je=s("a"),$n=s("span"),k(Uo.$$.fragment),gl=u(),wn=s("span"),_l=l("BloomForTokenClassification"),ks=u(),L=s("div"),k(Vo.$$.fragment),bl=u(),Tn=s("p"),vl=l(`Bloom Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
Named-Entity-Recognition (NER) tasks.`),kl=u(),Jo=s("p"),yl=l("This model inherits from "),Pt=s("a"),$l=l("PreTrainedModel"),wl=l(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)`),Tl=u(),Go=s("p"),Fl=l("This model is also a PyTorch "),Ro=s("a"),Bl=l("torch.nn.Module"),xl=l(` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Ml=u(),S=s("div"),k(Xo.$$.fragment),zl=u(),we=s("p"),Cl=l("The "),qt=s("a"),El=l("BloomForTokenClassification"),jl=l(" forward method, overrides the "),Fn=s("code"),Pl=l("__call__"),ql=l(" special method."),Ol=u(),k(Ge.$$.fragment),Ll=u(),k(Re.$$.fragment),Al=u(),k(Xe.$$.fragment),ys=u(),Te=s("h2"),Ye=s("a"),Bn=s("span"),k(Yo.$$.fragment),Sl=u(),xn=s("span"),Il=l("FlaxBloomModel"),$s=u(),C=s("div"),k(Ko.$$.fragment),Nl=u(),Mn=s("p"),Dl=l("The bare Bloom Model transformer outputting raw hidden-states without any specific head on top."),Wl=u(),Qo=s("p"),Hl=l("This model inherits from "),Ot=s("a"),Ul=l("FlaxPreTrainedModel"),Vl=l(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Jl=u(),Zo=s("p"),Gl=l(`This model is also a Flax Linen
`),et=s("a"),Rl=l("flax.nn.Module"),Xl=l(` subclass. Use it as a
regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.`),Yl=u(),zn=s("p"),Kl=l("Finally, this model supports inherent JAX features such as:"),Ql=u(),ne=s("ul"),Cn=s("li"),ot=s("a"),Zl=l("Just-In-Time (JIT) compilation"),ei=u(),En=s("li"),tt=s("a"),oi=l("Automatic Differentiation"),ti=u(),jn=s("li"),nt=s("a"),ni=l("Vectorization"),si=u(),Pn=s("li"),st=s("a"),ai=l("Parallelization"),ri=u(),K=s("div"),k(at.$$.fragment),li=u(),Fe=s("p"),ii=l("The "),qn=s("code"),di=l("FlaxBloomPreTrainedModel"),ci=l(" forward method, overrides the "),On=s("code"),pi=l("__call__"),mi=l(" special method."),hi=u(),k(Ke.$$.fragment),ui=u(),k(Qe.$$.fragment),ws=u(),Be=s("h2"),Ze=s("a"),Ln=s("span"),k(rt.$$.fragment),fi=u(),An=s("span"),gi=l("FlaxBloomForCausalLM"),Ts=u(),E=s("div"),k(lt.$$.fragment),_i=u(),Sn=s("p"),bi=l(`The Bloom Model transformer with a language modeling head on top (linear layer with weights tied to the input
embeddings).`),vi=u(),it=s("p"),ki=l("This model inherits from "),Lt=s("a"),yi=l("FlaxPreTrainedModel"),$i=l(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),wi=u(),dt=s("p"),Ti=l(`This model is also a Flax Linen
`),ct=s("a"),Fi=l("flax.nn.Module"),Bi=l(` subclass. Use it as a
regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.`),xi=u(),In=s("p"),Mi=l("Finally, this model supports inherent JAX features such as:"),zi=u(),se=s("ul"),Nn=s("li"),pt=s("a"),Ci=l("Just-In-Time (JIT) compilation"),Ei=u(),Dn=s("li"),mt=s("a"),ji=l("Automatic Differentiation"),Pi=u(),Wn=s("li"),ht=s("a"),qi=l("Vectorization"),Oi=u(),Hn=s("li"),ut=s("a"),Li=l("Parallelization"),Ai=u(),Q=s("div"),k(ft.$$.fragment),Si=u(),xe=s("p"),Ii=l("The "),Un=s("code"),Ni=l("FlaxBloomPreTrainedModel"),Di=l(" forward method, overrides the "),Vn=s("code"),Wi=l("__call__"),Hi=l(" special method."),Ui=u(),k(eo.$$.fragment),Vi=u(),k(oo.$$.fragment),this.h()},l(t){const _=Ec('[data-svelte="svelte-1phssyn"]',document.head);d=a(_,"META",{name:!0,content:!0}),_.forEach(o),g=f(t),c=a(t,"H1",{class:!0});var gt=r(c);m=a(gt,"A",{id:!0,class:!0,href:!0});var Jn=r(m);b=a(Jn,"SPAN",{});var Gn=r(b);y(n.$$.fragment,Gn),Gn.forEach(o),Jn.forEach(o),h=f(gt),x=a(gt,"SPAN",{});var Rn=r(x);pe=i(Rn,"BLOOM"),Rn.forEach(o),gt.forEach(o),V=f(t),R=a(t,"H2",{class:!0});var _t=r(R);te=a(_t,"A",{id:!0,class:!0,href:!0});var Xn=r(te);Vt=a(Xn,"SPAN",{});var Yn=r(Vt);y(co.$$.fragment,Yn),Yn.forEach(o),Xn.forEach(o),Xs=f(_t),Jt=a(_t,"SPAN",{});var Kn=r(Jt);Ys=i(Kn,"Overview"),Kn.forEach(o),_t.forEach(o),ls=f(t),Me=a(t,"P",{});var bt=r(Me);Ks=i(bt,"The BLOOM model has been proposed with its various versions through the "),po=a(bt,"A",{href:!0,rel:!0});var Qn=r(po);Qs=i(Qn,"BigScience Workshop"),Qn.forEach(o),Zs=i(bt,`. BigScience is inspired by other open science initiatives where researchers have pooled their time and resources to collectively achieve a higher impact.
The architecture of BLOOM is essentially similar to GPT3 (auto-regressive model for next token prediction), but has been trained on 46 different languages and 13 programming languages.
Several smaller versions of the models have been trained on the same dataset. BLOOM is available in the following versions:`),bt.forEach(o),is=f(t),P=a(t,"UL",{});var A=r(P);Gt=a(A,"LI",{});var Zn=r(Gt);mo=a(Zn,"A",{href:!0,rel:!0});var es=r(mo);ea=i(es,"bloom-350m"),es.forEach(o),Zn.forEach(o),oa=f(A),Rt=a(A,"LI",{});var os=r(Rt);ho=a(os,"A",{href:!0,rel:!0});var ts=r(ho);ta=i(ts,"bloom-760m"),ts.forEach(o),os.forEach(o),na=f(A),Xt=a(A,"LI",{});var ns=r(Xt);uo=a(ns,"A",{href:!0,rel:!0});var ss=r(uo);sa=i(ss,"bloom-1b3"),ss.forEach(o),ns.forEach(o),aa=f(A),Yt=a(A,"LI",{});var as=r(Yt);fo=a(as,"A",{href:!0,rel:!0});var rs=r(fo);ra=i(rs,"bloom-2b5"),rs.forEach(o),as.forEach(o),la=f(A),Kt=a(A,"LI",{});var Ri=r(Kt);go=a(Ri,"A",{href:!0,rel:!0});var Xi=r(go);ia=i(Xi,"bloom-6b3"),Xi.forEach(o),Ri.forEach(o),da=f(A),kt=a(A,"LI",{});var Ji=r(kt);_o=a(Ji,"A",{href:!0,rel:!0});var Yi=r(_o);ca=i(Yi,"bloom"),Yi.forEach(o),pa=i(Ji," (176B parameters)"),Ji.forEach(o),A.forEach(o),ds=f(t),me=a(t,"H2",{class:!0});var Bs=r(me);ze=a(Bs,"A",{id:!0,class:!0,href:!0});var Ki=r(ze);Qt=a(Ki,"SPAN",{});var Qi=r(Qt);y(bo.$$.fragment,Qi),Qi.forEach(o),Ki.forEach(o),ma=f(Bs),Zt=a(Bs,"SPAN",{});var Zi=r(Zt);ha=i(Zi,"BloomConfig"),Zi.forEach(o),Bs.forEach(o),cs=f(t),J=a(t,"DIV",{class:!0});var to=r(J);y(vo.$$.fragment,to),ua=f(to),he=a(to,"P",{});var At=r(he);fa=i(At,"This is the configuration class to store the configuration of a "),yt=a(At,"A",{href:!0});var ed=r(yt);ga=i(ed,"BloomModel"),ed.forEach(o),_a=i(At,`. It is used to instantiate a Bloom
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to the Bloom architecture
`),ko=a(At,"A",{href:!0,rel:!0});var od=r(ko);ba=i(od,"bigscience/bloom"),od.forEach(o),va=i(At,"."),At.forEach(o),ka=f(to),ue=a(to,"P",{});var St=r(ue);ya=i(St,"Configuration objects inherit from "),$t=a(St,"A",{href:!0});var td=r($t);$a=i(td,"PretrainedConfig"),td.forEach(o),wa=i(St,` and can be used to control the model outputs. Read the
documentation from `),wt=a(St,"A",{href:!0});var nd=r(wt);Ta=i(nd,"PretrainedConfig"),nd.forEach(o),Fa=i(St," for more information."),St.forEach(o),Ba=f(to),y(Ce.$$.fragment,to),to.forEach(o),ps=f(t),fe=a(t,"H2",{class:!0});var xs=r(fe);Ee=a(xs,"A",{id:!0,class:!0,href:!0});var sd=r(Ee);en=a(sd,"SPAN",{});var ad=r(en);y(yo.$$.fragment,ad),ad.forEach(o),sd.forEach(o),xa=f(xs),on=a(xs,"SPAN",{});var rd=r(on);Ma=i(rd,"BloomModel"),rd.forEach(o),xs.forEach(o),ms=f(t),q=a(t,"DIV",{class:!0});var ae=r(q);y($o.$$.fragment,ae),za=f(ae),tn=a(ae,"P",{});var ld=r(tn);Ca=i(ld,"The bare Bloom Model transformer outputting raw hidden-states without any specific head on top."),ld.forEach(o),Ea=f(ae),wo=a(ae,"P",{});var Ms=r(wo);ja=i(Ms,"This model inherits from "),Tt=a(Ms,"A",{href:!0});var id=r(Tt);Pa=i(id,"PreTrainedModel"),id.forEach(o),qa=i(Ms,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)`),Ms.forEach(o),Oa=f(ae),To=a(ae,"P",{});var zs=r(To);La=i(zs,"This model is also a PyTorch "),Fo=a(zs,"A",{href:!0,rel:!0});var dd=r(Fo);Aa=i(dd,"torch.nn.Module"),dd.forEach(o),Sa=i(zs,` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),zs.forEach(o),Ia=f(ae),X=a(ae,"DIV",{class:!0});var no=r(X);y(Bo.$$.fragment,no),Na=f(no),ge=a(no,"P",{});var It=r(ge);Da=i(It,"The "),Ft=a(It,"A",{href:!0});var cd=r(Ft);Wa=i(cd,"BloomModel"),cd.forEach(o),Ha=i(It," forward method, overrides the "),nn=a(It,"CODE",{});var pd=r(nn);Ua=i(pd,"__call__"),pd.forEach(o),Va=i(It," special method."),It.forEach(o),Ja=f(no),y(je.$$.fragment,no),Ga=f(no),y(Pe.$$.fragment,no),no.forEach(o),ae.forEach(o),hs=f(t),_e=a(t,"H2",{class:!0});var Cs=r(_e);qe=a(Cs,"A",{id:!0,class:!0,href:!0});var md=r(qe);sn=a(md,"SPAN",{});var hd=r(sn);y(xo.$$.fragment,hd),hd.forEach(o),md.forEach(o),Ra=f(Cs),an=a(Cs,"SPAN",{});var ud=r(an);Xa=i(ud,"BloomTokenizerFast"),ud.forEach(o),Cs.forEach(o),us=f(t),M=a(t,"DIV",{class:!0});var I=r(M);y(Mo.$$.fragment,I),Ya=f(I),zo=a(I,"P",{});var Es=r(zo);Ka=i(Es,"Construct a \u201Cfast\u201D Bloom tokenizer (backed by HuggingFace\u2019s "),rn=a(Es,"EM",{});var fd=r(rn);Qa=i(fd,"tokenizers"),fd.forEach(o),Za=i(Es,` library). Based on byte-level
Byte-Pair-Encoding.`),Es.forEach(o),er=f(I),ln=a(I,"P",{});var gd=r(ln);or=i(gd,"This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will"),gd.forEach(o),tr=f(I),y(Oe.$$.fragment,I),nr=f(I),Co=a(I,"P",{});var js=r(Co);sr=i(js,"You can get around that behavior by passing "),dn=a(js,"CODE",{});var _d=r(dn);ar=i(_d,"add_prefix_space=True"),_d.forEach(o),rr=i(js,` when instantiating this tokenizer, but since
the model was not pretrained this way, it might yield a decrease in performance.`),js.forEach(o),lr=f(I),y(Le.$$.fragment,I),ir=f(I),Eo=a(I,"P",{});var Ps=r(Eo);dr=i(Ps,"This tokenizer inherits from "),Bt=a(Ps,"A",{href:!0});var bd=r(Bt);cr=i(bd,"PreTrainedTokenizerFast"),bd.forEach(o),pr=i(Ps,` which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`),Ps.forEach(o),I.forEach(o),fs=f(t),be=a(t,"H2",{class:!0});var qs=r(be);Ae=a(qs,"A",{id:!0,class:!0,href:!0});var vd=r(Ae);cn=a(vd,"SPAN",{});var kd=r(cn);y(jo.$$.fragment,kd),kd.forEach(o),vd.forEach(o),mr=f(qs),pn=a(qs,"SPAN",{});var yd=r(pn);hr=i(yd,"BloomForCausalLM"),yd.forEach(o),qs.forEach(o),gs=f(t),O=a(t,"DIV",{class:!0});var re=r(O);y(Po.$$.fragment,re),ur=f(re),mn=a(re,"P",{});var $d=r(mn);fr=i($d,`The Bloom Model transformer with a language modeling head on top (linear layer with weights tied to the input
embeddings).`),$d.forEach(o),gr=f(re),qo=a(re,"P",{});var Os=r(qo);_r=i(Os,"This model inherits from "),xt=a(Os,"A",{href:!0});var wd=r(xt);br=i(wd,"PreTrainedModel"),wd.forEach(o),vr=i(Os,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)`),Os.forEach(o),kr=f(re),Oo=a(re,"P",{});var Ls=r(Oo);yr=i(Ls,"This model is also a PyTorch "),Lo=a(Ls,"A",{href:!0,rel:!0});var Td=r(Lo);$r=i(Td,"torch.nn.Module"),Td.forEach(o),wr=i(Ls,` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Ls.forEach(o),Tr=f(re),Y=a(re,"DIV",{class:!0});var so=r(Y);y(Ao.$$.fragment,so),Fr=f(so),ve=a(so,"P",{});var Nt=r(ve);Br=i(Nt,"The "),Mt=a(Nt,"A",{href:!0});var Fd=r(Mt);xr=i(Fd,"BloomForCausalLM"),Fd.forEach(o),Mr=i(Nt," forward method, overrides the "),hn=a(Nt,"CODE",{});var Bd=r(hn);zr=i(Bd,"__call__"),Bd.forEach(o),Cr=i(Nt," special method."),Nt.forEach(o),Er=f(so),y(Se.$$.fragment,so),jr=f(so),y(Ie.$$.fragment,so),so.forEach(o),re.forEach(o),_s=f(t),ke=a(t,"H2",{class:!0});var As=r(ke);Ne=a(As,"A",{id:!0,class:!0,href:!0});var xd=r(Ne);un=a(xd,"SPAN",{});var Md=r(un);y(So.$$.fragment,Md),Md.forEach(o),xd.forEach(o),Pr=f(As),fn=a(As,"SPAN",{});var zd=r(fn);qr=i(zd,"BloomForSequenceClassification"),zd.forEach(o),As.forEach(o),bs=f(t),z=a(t,"DIV",{class:!0});var N=r(z);y(Io.$$.fragment,N),Or=f(N),gn=a(N,"P",{});var Cd=r(gn);Lr=i(Cd,"The Bloom Model transformer with a sequence classification head on top (linear layer)."),Cd.forEach(o),Ar=f(N),zt=a(N,"P",{});var Gi=r(zt);Ct=a(Gi,"A",{href:!0});var Ed=r(Ct);Sr=i(Ed,"BloomForSequenceClassification"),Ed.forEach(o),Ir=i(Gi,` uses the last token in order to do the classification, as other causal models
(e.g. GPT-1) do.`),Gi.forEach(o),Nr=f(N),G=a(N,"P",{});var le=r(G);Dr=i(le,`Since it does classification on the last token, it requires to know the position of the last token. If a
`),_n=a(le,"CODE",{});var jd=r(_n);Wr=i(jd,"pad_token_id"),jd.forEach(o),Hr=i(le,` is defined in the configuration, it finds the last token that is not a padding token in each row. If
no `),bn=a(le,"CODE",{});var Pd=r(bn);Ur=i(Pd,"pad_token_id"),Pd.forEach(o),Vr=i(le,` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
padding tokens when `),vn=a(le,"CODE",{});var qd=r(vn);Jr=i(qd,"inputs_embeds"),qd.forEach(o),Gr=i(le," are passed instead of "),kn=a(le,"CODE",{});var Od=r(kn);Rr=i(Od,"input_ids"),Od.forEach(o),Xr=i(le,`, it does the same (take the last value in
each row of the batch).`),le.forEach(o),Yr=f(N),No=a(N,"P",{});var Ss=r(No);Kr=i(Ss,"This model inherits from "),Et=a(Ss,"A",{href:!0});var Ld=r(Et);Qr=i(Ld,"PreTrainedModel"),Ld.forEach(o),Zr=i(Ss,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)`),Ss.forEach(o),el=f(N),Do=a(N,"P",{});var Is=r(Do);ol=i(Is,"This model is also a PyTorch "),Wo=a(Is,"A",{href:!0,rel:!0});var Ad=r(Wo);tl=i(Ad,"torch.nn.Module"),Ad.forEach(o),nl=i(Is,` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Is.forEach(o),sl=f(N),j=a(N,"DIV",{class:!0});var D=r(j);y(Ho.$$.fragment,D),al=f(D),ye=a(D,"P",{});var Dt=r(ye);rl=i(Dt,"The "),jt=a(Dt,"A",{href:!0});var Sd=r(jt);ll=i(Sd,"BloomForSequenceClassification"),Sd.forEach(o),il=i(Dt," forward method, overrides the "),yn=a(Dt,"CODE",{});var Id=r(yn);dl=i(Id,"__call__"),Id.forEach(o),cl=i(Dt," special method."),Dt.forEach(o),pl=f(D),y(De.$$.fragment,D),ml=f(D),y(We.$$.fragment,D),hl=f(D),y(He.$$.fragment,D),ul=f(D),y(Ue.$$.fragment,D),fl=f(D),y(Ve.$$.fragment,D),D.forEach(o),N.forEach(o),vs=f(t),$e=a(t,"H2",{class:!0});var Ns=r($e);Je=a(Ns,"A",{id:!0,class:!0,href:!0});var Nd=r(Je);$n=a(Nd,"SPAN",{});var Dd=r($n);y(Uo.$$.fragment,Dd),Dd.forEach(o),Nd.forEach(o),gl=f(Ns),wn=a(Ns,"SPAN",{});var Wd=r(wn);_l=i(Wd,"BloomForTokenClassification"),Wd.forEach(o),Ns.forEach(o),ks=f(t),L=a(t,"DIV",{class:!0});var ie=r(L);y(Vo.$$.fragment,ie),bl=f(ie),Tn=a(ie,"P",{});var Hd=r(Tn);vl=i(Hd,`Bloom Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
Named-Entity-Recognition (NER) tasks.`),Hd.forEach(o),kl=f(ie),Jo=a(ie,"P",{});var Ds=r(Jo);yl=i(Ds,"This model inherits from "),Pt=a(Ds,"A",{href:!0});var Ud=r(Pt);$l=i(Ud,"PreTrainedModel"),Ud.forEach(o),wl=i(Ds,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)`),Ds.forEach(o),Tl=f(ie),Go=a(ie,"P",{});var Ws=r(Go);Fl=i(Ws,"This model is also a PyTorch "),Ro=a(Ws,"A",{href:!0,rel:!0});var Vd=r(Ro);Bl=i(Vd,"torch.nn.Module"),Vd.forEach(o),xl=i(Ws,` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Ws.forEach(o),Ml=f(ie),S=a(ie,"DIV",{class:!0});var de=r(S);y(Xo.$$.fragment,de),zl=f(de),we=a(de,"P",{});var Wt=r(we);Cl=i(Wt,"The "),qt=a(Wt,"A",{href:!0});var Jd=r(qt);El=i(Jd,"BloomForTokenClassification"),Jd.forEach(o),jl=i(Wt," forward method, overrides the "),Fn=a(Wt,"CODE",{});var Gd=r(Fn);Pl=i(Gd,"__call__"),Gd.forEach(o),ql=i(Wt," special method."),Wt.forEach(o),Ol=f(de),y(Ge.$$.fragment,de),Ll=f(de),y(Re.$$.fragment,de),Al=f(de),y(Xe.$$.fragment,de),de.forEach(o),ie.forEach(o),ys=f(t),Te=a(t,"H2",{class:!0});var Hs=r(Te);Ye=a(Hs,"A",{id:!0,class:!0,href:!0});var Rd=r(Ye);Bn=a(Rd,"SPAN",{});var Xd=r(Bn);y(Yo.$$.fragment,Xd),Xd.forEach(o),Rd.forEach(o),Sl=f(Hs),xn=a(Hs,"SPAN",{});var Yd=r(xn);Il=i(Yd,"FlaxBloomModel"),Yd.forEach(o),Hs.forEach(o),$s=f(t),C=a(t,"DIV",{class:!0});var W=r(C);y(Ko.$$.fragment,W),Nl=f(W),Mn=a(W,"P",{});var Kd=r(Mn);Dl=i(Kd,"The bare Bloom Model transformer outputting raw hidden-states without any specific head on top."),Kd.forEach(o),Wl=f(W),Qo=a(W,"P",{});var Us=r(Qo);Hl=i(Us,"This model inherits from "),Ot=a(Us,"A",{href:!0});var Qd=r(Ot);Ul=i(Qd,"FlaxPreTrainedModel"),Qd.forEach(o),Vl=i(Us,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Us.forEach(o),Jl=f(W),Zo=a(W,"P",{});var Vs=r(Zo);Gl=i(Vs,`This model is also a Flax Linen
`),et=a(Vs,"A",{href:!0,rel:!0});var Zd=r(et);Rl=i(Zd,"flax.nn.Module"),Zd.forEach(o),Xl=i(Vs,` subclass. Use it as a
regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.`),Vs.forEach(o),Yl=f(W),zn=a(W,"P",{});var ec=r(zn);Kl=i(ec,"Finally, this model supports inherent JAX features such as:"),ec.forEach(o),Ql=f(W),ne=a(W,"UL",{});var ao=r(ne);Cn=a(ao,"LI",{});var oc=r(Cn);ot=a(oc,"A",{href:!0,rel:!0});var tc=r(ot);Zl=i(tc,"Just-In-Time (JIT) compilation"),tc.forEach(o),oc.forEach(o),ei=f(ao),En=a(ao,"LI",{});var nc=r(En);tt=a(nc,"A",{href:!0,rel:!0});var sc=r(tt);oi=i(sc,"Automatic Differentiation"),sc.forEach(o),nc.forEach(o),ti=f(ao),jn=a(ao,"LI",{});var ac=r(jn);nt=a(ac,"A",{href:!0,rel:!0});var rc=r(nt);ni=i(rc,"Vectorization"),rc.forEach(o),ac.forEach(o),si=f(ao),Pn=a(ao,"LI",{});var lc=r(Pn);st=a(lc,"A",{href:!0,rel:!0});var ic=r(st);ai=i(ic,"Parallelization"),ic.forEach(o),lc.forEach(o),ao.forEach(o),ri=f(W),K=a(W,"DIV",{class:!0});var ro=r(K);y(at.$$.fragment,ro),li=f(ro),Fe=a(ro,"P",{});var Ht=r(Fe);ii=i(Ht,"The "),qn=a(Ht,"CODE",{});var dc=r(qn);di=i(dc,"FlaxBloomPreTrainedModel"),dc.forEach(o),ci=i(Ht," forward method, overrides the "),On=a(Ht,"CODE",{});var cc=r(On);pi=i(cc,"__call__"),cc.forEach(o),mi=i(Ht," special method."),Ht.forEach(o),hi=f(ro),y(Ke.$$.fragment,ro),ui=f(ro),y(Qe.$$.fragment,ro),ro.forEach(o),W.forEach(o),ws=f(t),Be=a(t,"H2",{class:!0});var Js=r(Be);Ze=a(Js,"A",{id:!0,class:!0,href:!0});var pc=r(Ze);Ln=a(pc,"SPAN",{});var mc=r(Ln);y(rt.$$.fragment,mc),mc.forEach(o),pc.forEach(o),fi=f(Js),An=a(Js,"SPAN",{});var hc=r(An);gi=i(hc,"FlaxBloomForCausalLM"),hc.forEach(o),Js.forEach(o),Ts=f(t),E=a(t,"DIV",{class:!0});var H=r(E);y(lt.$$.fragment,H),_i=f(H),Sn=a(H,"P",{});var uc=r(Sn);bi=i(uc,`The Bloom Model transformer with a language modeling head on top (linear layer with weights tied to the input
embeddings).`),uc.forEach(o),vi=f(H),it=a(H,"P",{});var Gs=r(it);ki=i(Gs,"This model inherits from "),Lt=a(Gs,"A",{href:!0});var fc=r(Lt);yi=i(fc,"FlaxPreTrainedModel"),fc.forEach(o),$i=i(Gs,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Gs.forEach(o),wi=f(H),dt=a(H,"P",{});var Rs=r(dt);Ti=i(Rs,`This model is also a Flax Linen
`),ct=a(Rs,"A",{href:!0,rel:!0});var gc=r(ct);Fi=i(gc,"flax.nn.Module"),gc.forEach(o),Bi=i(Rs,` subclass. Use it as a
regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.`),Rs.forEach(o),xi=f(H),In=a(H,"P",{});var _c=r(In);Mi=i(_c,"Finally, this model supports inherent JAX features such as:"),_c.forEach(o),zi=f(H),se=a(H,"UL",{});var lo=r(se);Nn=a(lo,"LI",{});var bc=r(Nn);pt=a(bc,"A",{href:!0,rel:!0});var vc=r(pt);Ci=i(vc,"Just-In-Time (JIT) compilation"),vc.forEach(o),bc.forEach(o),Ei=f(lo),Dn=a(lo,"LI",{});var kc=r(Dn);mt=a(kc,"A",{href:!0,rel:!0});var yc=r(mt);ji=i(yc,"Automatic Differentiation"),yc.forEach(o),kc.forEach(o),Pi=f(lo),Wn=a(lo,"LI",{});var $c=r(Wn);ht=a($c,"A",{href:!0,rel:!0});var wc=r(ht);qi=i(wc,"Vectorization"),wc.forEach(o),$c.forEach(o),Oi=f(lo),Hn=a(lo,"LI",{});var Tc=r(Hn);ut=a(Tc,"A",{href:!0,rel:!0});var Fc=r(ut);Li=i(Fc,"Parallelization"),Fc.forEach(o),Tc.forEach(o),lo.forEach(o),Ai=f(H),Q=a(H,"DIV",{class:!0});var io=r(Q);y(ft.$$.fragment,io),Si=f(io),xe=a(io,"P",{});var Ut=r(xe);Ii=i(Ut,"The "),Un=a(Ut,"CODE",{});var Bc=r(Un);Ni=i(Bc,"FlaxBloomPreTrainedModel"),Bc.forEach(o),Di=i(Ut," forward method, overrides the "),Vn=a(Ut,"CODE",{});var xc=r(Vn);Wi=i(xc,"__call__"),xc.forEach(o),Hi=i(Ut," special method."),Ut.forEach(o),Ui=f(io),y(eo.$$.fragment,io),Vi=f(io),y(oo.$$.fragment,io),io.forEach(o),H.forEach(o),this.h()},h(){p(d,"name","hf:doc:metadata"),p(d,"content",JSON.stringify(Zc)),p(m,"id","bloom"),p(m,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),p(m,"href","#bloom"),p(c,"class","relative group"),p(te,"id","overview"),p(te,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),p(te,"href","#overview"),p(R,"class","relative group"),p(po,"href","https://bigscience.huggingface.co/"),p(po,"rel","nofollow"),p(mo,"href","https://huggingface.co/bigscience/bloom-350m"),p(mo,"rel","nofollow"),p(ho,"href","https://huggingface.co/bigscience/bloom-760m"),p(ho,"rel","nofollow"),p(uo,"href","https://huggingface.co/bigscience/bloom-1b3"),p(uo,"rel","nofollow"),p(fo,"href","https://huggingface.co/bigscience/bloom-2b5"),p(fo,"rel","nofollow"),p(go,"href","https://huggingface.co/bigscience/bloom-6b3"),p(go,"rel","nofollow"),p(_o,"href","https://huggingface.co/bigscience/bloom"),p(_o,"rel","nofollow"),p(ze,"id","transformers.BloomConfig"),p(ze,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),p(ze,"href","#transformers.BloomConfig"),p(me,"class","relative group"),p(yt,"href","/docs/transformers/pr_18022/en/model_doc/bloom#transformers.BloomModel"),p(ko,"href","https://huggingface.co/bigscience/bloom"),p(ko,"rel","nofollow"),p($t,"href","/docs/transformers/pr_18022/en/main_classes/configuration#transformers.PretrainedConfig"),p(wt,"href","/docs/transformers/pr_18022/en/main_classes/configuration#transformers.PretrainedConfig"),p(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),p(Ee,"id","transformers.BloomModel"),p(Ee,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),p(Ee,"href","#transformers.BloomModel"),p(fe,"class","relative group"),p(Tt,"href","/docs/transformers/pr_18022/en/main_classes/model#transformers.PreTrainedModel"),p(Fo,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),p(Fo,"rel","nofollow"),p(Ft,"href","/docs/transformers/pr_18022/en/model_doc/bloom#transformers.BloomModel"),p(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),p(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),p(qe,"id","transformers.BloomTokenizerFast"),p(qe,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),p(qe,"href","#transformers.BloomTokenizerFast"),p(_e,"class","relative group"),p(Bt,"href","/docs/transformers/pr_18022/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast"),p(M,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),p(Ae,"id","transformers.BloomForCausalLM"),p(Ae,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),p(Ae,"href","#transformers.BloomForCausalLM"),p(be,"class","relative group"),p(xt,"href","/docs/transformers/pr_18022/en/main_classes/model#transformers.PreTrainedModel"),p(Lo,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),p(Lo,"rel","nofollow"),p(Mt,"href","/docs/transformers/pr_18022/en/model_doc/bloom#transformers.BloomForCausalLM"),p(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),p(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),p(Ne,"id","transformers.BloomForSequenceClassification"),p(Ne,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),p(Ne,"href","#transformers.BloomForSequenceClassification"),p(ke,"class","relative group"),p(Ct,"href","/docs/transformers/pr_18022/en/model_doc/bloom#transformers.BloomForSequenceClassification"),p(Et,"href","/docs/transformers/pr_18022/en/main_classes/model#transformers.PreTrainedModel"),p(Wo,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),p(Wo,"rel","nofollow"),p(jt,"href","/docs/transformers/pr_18022/en/model_doc/bloom#transformers.BloomForSequenceClassification"),p(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),p(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),p(Je,"id","transformers.BloomForTokenClassification"),p(Je,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),p(Je,"href","#transformers.BloomForTokenClassification"),p($e,"class","relative group"),p(Pt,"href","/docs/transformers/pr_18022/en/main_classes/model#transformers.PreTrainedModel"),p(Ro,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),p(Ro,"rel","nofollow"),p(qt,"href","/docs/transformers/pr_18022/en/model_doc/bloom#transformers.BloomForTokenClassification"),p(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),p(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),p(Ye,"id","transformers.FlaxBloomModel"),p(Ye,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),p(Ye,"href","#transformers.FlaxBloomModel"),p(Te,"class","relative group"),p(Ot,"href","/docs/transformers/pr_18022/en/main_classes/model#transformers.FlaxPreTrainedModel"),p(et,"href","https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html"),p(et,"rel","nofollow"),p(ot,"href","https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit"),p(ot,"rel","nofollow"),p(tt,"href","https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation"),p(tt,"rel","nofollow"),p(nt,"href","https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap"),p(nt,"rel","nofollow"),p(st,"href","https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap"),p(st,"rel","nofollow"),p(K,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),p(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),p(Ze,"id","transformers.FlaxBloomForCausalLM"),p(Ze,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),p(Ze,"href","#transformers.FlaxBloomForCausalLM"),p(Be,"class","relative group"),p(Lt,"href","/docs/transformers/pr_18022/en/main_classes/model#transformers.FlaxPreTrainedModel"),p(ct,"href","https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html"),p(ct,"rel","nofollow"),p(pt,"href","https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit"),p(pt,"rel","nofollow"),p(mt,"href","https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation"),p(mt,"rel","nofollow"),p(ht,"href","https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap"),p(ht,"rel","nofollow"),p(ut,"href","https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap"),p(ut,"rel","nofollow"),p(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),p(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(t,_){e(document.head,d),v(t,g,_),v(t,c,_),e(c,m),e(m,b),$(n,b,null),e(c,h),e(c,x),e(x,pe),v(t,V,_),v(t,R,_),e(R,te),e(te,Vt),$(co,Vt,null),e(R,Xs),e(R,Jt),e(Jt,Ys),v(t,ls,_),v(t,Me,_),e(Me,Ks),e(Me,po),e(po,Qs),e(Me,Zs),v(t,is,_),v(t,P,_),e(P,Gt),e(Gt,mo),e(mo,ea),e(P,oa),e(P,Rt),e(Rt,ho),e(ho,ta),e(P,na),e(P,Xt),e(Xt,uo),e(uo,sa),e(P,aa),e(P,Yt),e(Yt,fo),e(fo,ra),e(P,la),e(P,Kt),e(Kt,go),e(go,ia),e(P,da),e(P,kt),e(kt,_o),e(_o,ca),e(kt,pa),v(t,ds,_),v(t,me,_),e(me,ze),e(ze,Qt),$(bo,Qt,null),e(me,ma),e(me,Zt),e(Zt,ha),v(t,cs,_),v(t,J,_),$(vo,J,null),e(J,ua),e(J,he),e(he,fa),e(he,yt),e(yt,ga),e(he,_a),e(he,ko),e(ko,ba),e(he,va),e(J,ka),e(J,ue),e(ue,ya),e(ue,$t),e($t,$a),e(ue,wa),e(ue,wt),e(wt,Ta),e(ue,Fa),e(J,Ba),$(Ce,J,null),v(t,ps,_),v(t,fe,_),e(fe,Ee),e(Ee,en),$(yo,en,null),e(fe,xa),e(fe,on),e(on,Ma),v(t,ms,_),v(t,q,_),$($o,q,null),e(q,za),e(q,tn),e(tn,Ca),e(q,Ea),e(q,wo),e(wo,ja),e(wo,Tt),e(Tt,Pa),e(wo,qa),e(q,Oa),e(q,To),e(To,La),e(To,Fo),e(Fo,Aa),e(To,Sa),e(q,Ia),e(q,X),$(Bo,X,null),e(X,Na),e(X,ge),e(ge,Da),e(ge,Ft),e(Ft,Wa),e(ge,Ha),e(ge,nn),e(nn,Ua),e(ge,Va),e(X,Ja),$(je,X,null),e(X,Ga),$(Pe,X,null),v(t,hs,_),v(t,_e,_),e(_e,qe),e(qe,sn),$(xo,sn,null),e(_e,Ra),e(_e,an),e(an,Xa),v(t,us,_),v(t,M,_),$(Mo,M,null),e(M,Ya),e(M,zo),e(zo,Ka),e(zo,rn),e(rn,Qa),e(zo,Za),e(M,er),e(M,ln),e(ln,or),e(M,tr),$(Oe,M,null),e(M,nr),e(M,Co),e(Co,sr),e(Co,dn),e(dn,ar),e(Co,rr),e(M,lr),$(Le,M,null),e(M,ir),e(M,Eo),e(Eo,dr),e(Eo,Bt),e(Bt,cr),e(Eo,pr),v(t,fs,_),v(t,be,_),e(be,Ae),e(Ae,cn),$(jo,cn,null),e(be,mr),e(be,pn),e(pn,hr),v(t,gs,_),v(t,O,_),$(Po,O,null),e(O,ur),e(O,mn),e(mn,fr),e(O,gr),e(O,qo),e(qo,_r),e(qo,xt),e(xt,br),e(qo,vr),e(O,kr),e(O,Oo),e(Oo,yr),e(Oo,Lo),e(Lo,$r),e(Oo,wr),e(O,Tr),e(O,Y),$(Ao,Y,null),e(Y,Fr),e(Y,ve),e(ve,Br),e(ve,Mt),e(Mt,xr),e(ve,Mr),e(ve,hn),e(hn,zr),e(ve,Cr),e(Y,Er),$(Se,Y,null),e(Y,jr),$(Ie,Y,null),v(t,_s,_),v(t,ke,_),e(ke,Ne),e(Ne,un),$(So,un,null),e(ke,Pr),e(ke,fn),e(fn,qr),v(t,bs,_),v(t,z,_),$(Io,z,null),e(z,Or),e(z,gn),e(gn,Lr),e(z,Ar),e(z,zt),e(zt,Ct),e(Ct,Sr),e(zt,Ir),e(z,Nr),e(z,G),e(G,Dr),e(G,_n),e(_n,Wr),e(G,Hr),e(G,bn),e(bn,Ur),e(G,Vr),e(G,vn),e(vn,Jr),e(G,Gr),e(G,kn),e(kn,Rr),e(G,Xr),e(z,Yr),e(z,No),e(No,Kr),e(No,Et),e(Et,Qr),e(No,Zr),e(z,el),e(z,Do),e(Do,ol),e(Do,Wo),e(Wo,tl),e(Do,nl),e(z,sl),e(z,j),$(Ho,j,null),e(j,al),e(j,ye),e(ye,rl),e(ye,jt),e(jt,ll),e(ye,il),e(ye,yn),e(yn,dl),e(ye,cl),e(j,pl),$(De,j,null),e(j,ml),$(We,j,null),e(j,hl),$(He,j,null),e(j,ul),$(Ue,j,null),e(j,fl),$(Ve,j,null),v(t,vs,_),v(t,$e,_),e($e,Je),e(Je,$n),$(Uo,$n,null),e($e,gl),e($e,wn),e(wn,_l),v(t,ks,_),v(t,L,_),$(Vo,L,null),e(L,bl),e(L,Tn),e(Tn,vl),e(L,kl),e(L,Jo),e(Jo,yl),e(Jo,Pt),e(Pt,$l),e(Jo,wl),e(L,Tl),e(L,Go),e(Go,Fl),e(Go,Ro),e(Ro,Bl),e(Go,xl),e(L,Ml),e(L,S),$(Xo,S,null),e(S,zl),e(S,we),e(we,Cl),e(we,qt),e(qt,El),e(we,jl),e(we,Fn),e(Fn,Pl),e(we,ql),e(S,Ol),$(Ge,S,null),e(S,Ll),$(Re,S,null),e(S,Al),$(Xe,S,null),v(t,ys,_),v(t,Te,_),e(Te,Ye),e(Ye,Bn),$(Yo,Bn,null),e(Te,Sl),e(Te,xn),e(xn,Il),v(t,$s,_),v(t,C,_),$(Ko,C,null),e(C,Nl),e(C,Mn),e(Mn,Dl),e(C,Wl),e(C,Qo),e(Qo,Hl),e(Qo,Ot),e(Ot,Ul),e(Qo,Vl),e(C,Jl),e(C,Zo),e(Zo,Gl),e(Zo,et),e(et,Rl),e(Zo,Xl),e(C,Yl),e(C,zn),e(zn,Kl),e(C,Ql),e(C,ne),e(ne,Cn),e(Cn,ot),e(ot,Zl),e(ne,ei),e(ne,En),e(En,tt),e(tt,oi),e(ne,ti),e(ne,jn),e(jn,nt),e(nt,ni),e(ne,si),e(ne,Pn),e(Pn,st),e(st,ai),e(C,ri),e(C,K),$(at,K,null),e(K,li),e(K,Fe),e(Fe,ii),e(Fe,qn),e(qn,di),e(Fe,ci),e(Fe,On),e(On,pi),e(Fe,mi),e(K,hi),$(Ke,K,null),e(K,ui),$(Qe,K,null),v(t,ws,_),v(t,Be,_),e(Be,Ze),e(Ze,Ln),$(rt,Ln,null),e(Be,fi),e(Be,An),e(An,gi),v(t,Ts,_),v(t,E,_),$(lt,E,null),e(E,_i),e(E,Sn),e(Sn,bi),e(E,vi),e(E,it),e(it,ki),e(it,Lt),e(Lt,yi),e(it,$i),e(E,wi),e(E,dt),e(dt,Ti),e(dt,ct),e(ct,Fi),e(dt,Bi),e(E,xi),e(E,In),e(In,Mi),e(E,zi),e(E,se),e(se,Nn),e(Nn,pt),e(pt,Ci),e(se,Ei),e(se,Dn),e(Dn,mt),e(mt,ji),e(se,Pi),e(se,Wn),e(Wn,ht),e(ht,qi),e(se,Oi),e(se,Hn),e(Hn,ut),e(ut,Li),e(E,Ai),e(E,Q),$(ft,Q,null),e(Q,Si),e(Q,xe),e(xe,Ii),e(xe,Un),e(Un,Ni),e(xe,Di),e(xe,Vn),e(Vn,Wi),e(xe,Hi),e(Q,Ui),$(eo,Q,null),e(Q,Vi),$(oo,Q,null),Fs=!0},p(t,[_]){const gt={};_&2&&(gt.$$scope={dirty:_,ctx:t}),Ce.$set(gt);const Jn={};_&2&&(Jn.$$scope={dirty:_,ctx:t}),je.$set(Jn);const Gn={};_&2&&(Gn.$$scope={dirty:_,ctx:t}),Pe.$set(Gn);const Rn={};_&2&&(Rn.$$scope={dirty:_,ctx:t}),Oe.$set(Rn);const _t={};_&2&&(_t.$$scope={dirty:_,ctx:t}),Le.$set(_t);const Xn={};_&2&&(Xn.$$scope={dirty:_,ctx:t}),Se.$set(Xn);const Yn={};_&2&&(Yn.$$scope={dirty:_,ctx:t}),Ie.$set(Yn);const Kn={};_&2&&(Kn.$$scope={dirty:_,ctx:t}),De.$set(Kn);const bt={};_&2&&(bt.$$scope={dirty:_,ctx:t}),We.$set(bt);const Qn={};_&2&&(Qn.$$scope={dirty:_,ctx:t}),He.$set(Qn);const A={};_&2&&(A.$$scope={dirty:_,ctx:t}),Ue.$set(A);const Zn={};_&2&&(Zn.$$scope={dirty:_,ctx:t}),Ve.$set(Zn);const es={};_&2&&(es.$$scope={dirty:_,ctx:t}),Ge.$set(es);const os={};_&2&&(os.$$scope={dirty:_,ctx:t}),Re.$set(os);const ts={};_&2&&(ts.$$scope={dirty:_,ctx:t}),Xe.$set(ts);const ns={};_&2&&(ns.$$scope={dirty:_,ctx:t}),Ke.$set(ns);const ss={};_&2&&(ss.$$scope={dirty:_,ctx:t}),Qe.$set(ss);const as={};_&2&&(as.$$scope={dirty:_,ctx:t}),eo.$set(as);const rs={};_&2&&(rs.$$scope={dirty:_,ctx:t}),oo.$set(rs)},i(t){Fs||(w(n.$$.fragment,t),w(co.$$.fragment,t),w(bo.$$.fragment,t),w(vo.$$.fragment,t),w(Ce.$$.fragment,t),w(yo.$$.fragment,t),w($o.$$.fragment,t),w(Bo.$$.fragment,t),w(je.$$.fragment,t),w(Pe.$$.fragment,t),w(xo.$$.fragment,t),w(Mo.$$.fragment,t),w(Oe.$$.fragment,t),w(Le.$$.fragment,t),w(jo.$$.fragment,t),w(Po.$$.fragment,t),w(Ao.$$.fragment,t),w(Se.$$.fragment,t),w(Ie.$$.fragment,t),w(So.$$.fragment,t),w(Io.$$.fragment,t),w(Ho.$$.fragment,t),w(De.$$.fragment,t),w(We.$$.fragment,t),w(He.$$.fragment,t),w(Ue.$$.fragment,t),w(Ve.$$.fragment,t),w(Uo.$$.fragment,t),w(Vo.$$.fragment,t),w(Xo.$$.fragment,t),w(Ge.$$.fragment,t),w(Re.$$.fragment,t),w(Xe.$$.fragment,t),w(Yo.$$.fragment,t),w(Ko.$$.fragment,t),w(at.$$.fragment,t),w(Ke.$$.fragment,t),w(Qe.$$.fragment,t),w(rt.$$.fragment,t),w(lt.$$.fragment,t),w(ft.$$.fragment,t),w(eo.$$.fragment,t),w(oo.$$.fragment,t),Fs=!0)},o(t){T(n.$$.fragment,t),T(co.$$.fragment,t),T(bo.$$.fragment,t),T(vo.$$.fragment,t),T(Ce.$$.fragment,t),T(yo.$$.fragment,t),T($o.$$.fragment,t),T(Bo.$$.fragment,t),T(je.$$.fragment,t),T(Pe.$$.fragment,t),T(xo.$$.fragment,t),T(Mo.$$.fragment,t),T(Oe.$$.fragment,t),T(Le.$$.fragment,t),T(jo.$$.fragment,t),T(Po.$$.fragment,t),T(Ao.$$.fragment,t),T(Se.$$.fragment,t),T(Ie.$$.fragment,t),T(So.$$.fragment,t),T(Io.$$.fragment,t),T(Ho.$$.fragment,t),T(De.$$.fragment,t),T(We.$$.fragment,t),T(He.$$.fragment,t),T(Ue.$$.fragment,t),T(Ve.$$.fragment,t),T(Uo.$$.fragment,t),T(Vo.$$.fragment,t),T(Xo.$$.fragment,t),T(Ge.$$.fragment,t),T(Re.$$.fragment,t),T(Xe.$$.fragment,t),T(Yo.$$.fragment,t),T(Ko.$$.fragment,t),T(at.$$.fragment,t),T(Ke.$$.fragment,t),T(Qe.$$.fragment,t),T(rt.$$.fragment,t),T(lt.$$.fragment,t),T(ft.$$.fragment,t),T(eo.$$.fragment,t),T(oo.$$.fragment,t),Fs=!1},d(t){o(d),t&&o(g),t&&o(c),F(n),t&&o(V),t&&o(R),F(co),t&&o(ls),t&&o(Me),t&&o(is),t&&o(P),t&&o(ds),t&&o(me),F(bo),t&&o(cs),t&&o(J),F(vo),F(Ce),t&&o(ps),t&&o(fe),F(yo),t&&o(ms),t&&o(q),F($o),F(Bo),F(je),F(Pe),t&&o(hs),t&&o(_e),F(xo),t&&o(us),t&&o(M),F(Mo),F(Oe),F(Le),t&&o(fs),t&&o(be),F(jo),t&&o(gs),t&&o(O),F(Po),F(Ao),F(Se),F(Ie),t&&o(_s),t&&o(ke),F(So),t&&o(bs),t&&o(z),F(Io),F(Ho),F(De),F(We),F(He),F(Ue),F(Ve),t&&o(vs),t&&o($e),F(Uo),t&&o(ks),t&&o(L),F(Vo),F(Xo),F(Ge),F(Re),F(Xe),t&&o(ys),t&&o(Te),F(Yo),t&&o($s),t&&o(C),F(Ko),F(at),F(Ke),F(Qe),t&&o(ws),t&&o(Be),F(rt),t&&o(Ts),t&&o(E),F(lt),F(ft),F(eo),F(oo)}}}const Zc={local:"bloom",sections:[{local:"overview",title:"Overview"},{local:"transformers.BloomConfig",title:"BloomConfig"},{local:"transformers.BloomModel",title:"BloomModel"},{local:"transformers.BloomTokenizerFast",title:"BloomTokenizerFast"},{local:"transformers.BloomForCausalLM",title:"BloomForCausalLM"},{local:"transformers.BloomForSequenceClassification",title:"BloomForSequenceClassification"},{local:"transformers.BloomForTokenClassification",title:"BloomForTokenClassification"},{local:"transformers.FlaxBloomModel",title:"FlaxBloomModel"},{local:"transformers.FlaxBloomForCausalLM",title:"FlaxBloomForCausalLM"}],title:"BLOOM"};function ep(B){return jc(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class lp extends Mc{constructor(d){super();zc(this,d,ep,Qc,Cc,{})}}export{lp as default,Zc as metadata};
