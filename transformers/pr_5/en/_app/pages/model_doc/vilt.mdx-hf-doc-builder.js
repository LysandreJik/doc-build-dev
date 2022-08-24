import{S as Md,i as Pd,s as jd,e as n,k as p,w as T,t as a,M as Ld,c as s,d as o,m,a as r,x as k,h as i,b as c,N as zd,G as e,g as _,y as w,q as x,o as y,B as $,v as Id,L as Lo}from"../../chunks/vendor-hf-doc-builder.js";import{T as Gt}from"../../chunks/Tip-hf-doc-builder.js";import{D as E}from"../../chunks/Docstring-hf-doc-builder.js";import{C as zo}from"../../chunks/CodeBlock-hf-doc-builder.js";import{I as H}from"../../chunks/IconCopyLink-hf-doc-builder.js";import{E as jo}from"../../chunks/ExampleCodeBlock-hf-doc-builder.js";function Cd(V){let d,b,g,u,v;return u=new zo({props:{code:`from transformers import ViLTModel, ViLTConfig

# Initializing a ViLT dandelin/vilt-b32-mlm style configuration
configuration = ViLTConfig()

# Initializing a model from the dandelin/vilt-b32-mlm style configuration
model = ViLTModel(configuration)

# Accessing the model configuration
configuration = model.config`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> ViLTModel, ViLTConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a ViLT dandelin/vilt-b32-mlm style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = ViLTConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the dandelin/vilt-b32-mlm style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ViLTModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`}}),{c(){d=n("p"),b=a("Example:"),g=p(),T(u.$$.fragment)},l(l){d=s(l,"P",{});var h=r(d);b=i(h,"Example:"),h.forEach(o),g=m(l),k(u.$$.fragment,l)},m(l,h){_(l,d,h),e(d,b),_(l,g,h),w(u,l,h),v=!0},p:Lo,i(l){v||(x(u.$$.fragment,l),v=!0)},o(l){y(u.$$.fragment,l),v=!1},d(l){l&&o(d),l&&o(g),$(u,l)}}}function Ad(V){let d,b;return{c(){d=n("p"),b=a(`NumPy arrays and PyTorch tensors are converted to PIL images when resizing, so the most efficient is to pass
PIL images.`)},l(g){d=s(g,"P",{});var u=r(d);b=i(u,`NumPy arrays and PyTorch tensors are converted to PIL images when resizing, so the most efficient is to pass
PIL images.`),u.forEach(o)},m(g,u){_(g,d,u),e(d,b)},d(g){g&&o(d)}}}function qd(V){let d,b,g,u,v;return{c(){d=n("p"),b=a("Although the recipe for forward pass needs to be defined within this function, one should call the "),g=n("code"),u=a("Module"),v=a(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=s(l,"P",{});var h=r(d);b=i(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),g=s(h,"CODE",{});var F=r(g);u=i(F,"Module"),F.forEach(o),v=i(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(o)},m(l,h){_(l,d,h),e(d,b),e(d,g),e(g,u),e(d,v)},d(l){l&&o(d)}}}function Od(V){let d,b,g,u,v;return u=new zo({props:{code:`from transformers import ViltProcessor, ViltModel
from PIL import Image
import requests

# prepare image and text
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "hello world"

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")

inputs = processor(image, text, return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> ViltProcessor, ViltModel
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># prepare image and text</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)
<span class="hljs-meta">&gt;&gt;&gt; </span>text = <span class="hljs-string">&quot;hello world&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = ViltProcessor.from_pretrained(<span class="hljs-string">&quot;dandelin/vilt-b32-mlm&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ViltModel.from_pretrained(<span class="hljs-string">&quot;dandelin/vilt-b32-mlm&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(image, text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`}}),{c(){d=n("p"),b=a("Examples:"),g=p(),T(u.$$.fragment)},l(l){d=s(l,"P",{});var h=r(d);b=i(h,"Examples:"),h.forEach(o),g=m(l),k(u.$$.fragment,l)},m(l,h){_(l,d,h),e(d,b),_(l,g,h),w(u,l,h),v=!0},p:Lo,i(l){v||(x(u.$$.fragment,l),v=!0)},o(l){y(u.$$.fragment,l),v=!1},d(l){l&&o(d),l&&o(g),$(u,l)}}}function Nd(V){let d,b,g,u,v;return{c(){d=n("p"),b=a("Although the recipe for forward pass needs to be defined within this function, one should call the "),g=n("code"),u=a("Module"),v=a(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=s(l,"P",{});var h=r(d);b=i(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),g=s(h,"CODE",{});var F=r(g);u=i(F,"Module"),F.forEach(o),v=i(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(o)},m(l,h){_(l,d,h),e(d,b),e(d,g),e(g,u),e(d,v)},d(l){l&&o(d)}}}function Sd(V){let d,b,g,u,v;return u=new zo({props:{code:`from transformers import ViltProcessor, ViltForMaskedLM
import requests
from PIL import Image
import re
import torch

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "a bunch of [MASK] laying on a [MASK]."

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
model = ViltForMaskedLM.from_pretrained("dandelin/vilt-b32-mlm")

# prepare inputs
encoding = processor(image, text, return_tensors="pt")

# forward pass
outputs = model(**encoding)

tl = len(re.findall("\\[MASK\\]", text))
inferred_token = [text]

# gradually fill in the MASK tokens, one by one
with torch.no_grad():
    for i in range(tl):
        encoded = processor.tokenizer(inferred_token)
        input_ids = torch.tensor(encoded.input_ids)
        encoded = encoded["input_ids"][0][1:-1]
        outputs = model(input_ids=input_ids, pixel_values=encoding.pixel_values)
        mlm_logits = outputs.logits[0]  # shape (seq_len, vocab_size)
        # only take into account text features (minus CLS and SEP token)
        mlm_logits = mlm_logits[1 : input_ids.shape[1] - 1, :]
        mlm_values, mlm_ids = mlm_logits.softmax(dim=-1).max(dim=-1)
        # only take into account text
        mlm_values[torch.tensor(encoded) != 103] = 0
        select = mlm_values.argmax().item()
        encoded[select] = mlm_ids[select].item()
        inferred_token = [processor.decode(encoded)]

selected_token = ""
encoded = processor.tokenizer(inferred_token)
output = processor.decode(encoded.input_ids[0], skip_special_tokens=True)
print(output)`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> ViltProcessor, ViltForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> re
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)
<span class="hljs-meta">&gt;&gt;&gt; </span>text = <span class="hljs-string">&quot;a bunch of [MASK] laying on a [MASK].&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = ViltProcessor.from_pretrained(<span class="hljs-string">&quot;dandelin/vilt-b32-mlm&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ViltForMaskedLM.from_pretrained(<span class="hljs-string">&quot;dandelin/vilt-b32-mlm&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># prepare inputs</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = processor(image, text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># forward pass</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**encoding)

<span class="hljs-meta">&gt;&gt;&gt; </span>tl = <span class="hljs-built_in">len</span>(re.findall(<span class="hljs-string">&quot;\\[MASK\\]&quot;</span>, text))
<span class="hljs-meta">&gt;&gt;&gt; </span>inferred_token = [text]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># gradually fill in the MASK tokens, one by one</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    <span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> <span class="hljs-built_in">range</span>(tl):
<span class="hljs-meta">... </span>        encoded = processor.tokenizer(inferred_token)
<span class="hljs-meta">... </span>        input_ids = torch.tensor(encoded.input_ids)
<span class="hljs-meta">... </span>        encoded = encoded[<span class="hljs-string">&quot;input_ids&quot;</span>][<span class="hljs-number">0</span>][<span class="hljs-number">1</span>:-<span class="hljs-number">1</span>]
<span class="hljs-meta">... </span>        outputs = model(input_ids=input_ids, pixel_values=encoding.pixel_values)
<span class="hljs-meta">... </span>        mlm_logits = outputs.logits[<span class="hljs-number">0</span>]  <span class="hljs-comment"># shape (seq_len, vocab_size)</span>
<span class="hljs-meta">... </span>        <span class="hljs-comment"># only take into account text features (minus CLS and SEP token)</span>
<span class="hljs-meta">... </span>        mlm_logits = mlm_logits[<span class="hljs-number">1</span> : input_ids.shape[<span class="hljs-number">1</span>] - <span class="hljs-number">1</span>, :]
<span class="hljs-meta">... </span>        mlm_values, mlm_ids = mlm_logits.softmax(dim=-<span class="hljs-number">1</span>).<span class="hljs-built_in">max</span>(dim=-<span class="hljs-number">1</span>)
<span class="hljs-meta">... </span>        <span class="hljs-comment"># only take into account text</span>
<span class="hljs-meta">... </span>        mlm_values[torch.tensor(encoded) != <span class="hljs-number">103</span>] = <span class="hljs-number">0</span>
<span class="hljs-meta">... </span>        select = mlm_values.argmax().item()
<span class="hljs-meta">... </span>        encoded[select] = mlm_ids[select].item()
<span class="hljs-meta">... </span>        inferred_token = [processor.decode(encoded)]

<span class="hljs-meta">&gt;&gt;&gt; </span>selected_token = <span class="hljs-string">&quot;&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>encoded = processor.tokenizer(inferred_token)
<span class="hljs-meta">&gt;&gt;&gt; </span>output = processor.decode(encoded.input_ids[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(output)
a bunch of cats laying on a couch.`}}),{c(){d=n("p"),b=a("Examples:"),g=p(),T(u.$$.fragment)},l(l){d=s(l,"P",{});var h=r(d);b=i(h,"Examples:"),h.forEach(o),g=m(l),k(u.$$.fragment,l)},m(l,h){_(l,d,h),e(d,b),_(l,g,h),w(u,l,h),v=!0},p:Lo,i(l){v||(x(u.$$.fragment,l),v=!0)},o(l){y(u.$$.fragment,l),v=!1},d(l){l&&o(d),l&&o(g),$(u,l)}}}function Dd(V){let d,b,g,u,v;return{c(){d=n("p"),b=a("Although the recipe for forward pass needs to be defined within this function, one should call the "),g=n("code"),u=a("Module"),v=a(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=s(l,"P",{});var h=r(d);b=i(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),g=s(h,"CODE",{});var F=r(g);u=i(F,"Module"),F.forEach(o),v=i(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(o)},m(l,h){_(l,d,h),e(d,b),e(d,g),e(g,u),e(d,v)},d(l){l&&o(d)}}}function Wd(V){let d,b,g,u,v;return u=new zo({props:{code:`from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "How many cats are there?"

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# prepare inputs
encoding = processor(image, text, return_tensors="pt")

# forward pass
outputs = model(**encoding)
logits = outputs.logits
idx = logits.argmax(-1).item()
print("Predicted answer:", model.config.id2label[idx])`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> ViltProcessor, ViltForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image

<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)
<span class="hljs-meta">&gt;&gt;&gt; </span>text = <span class="hljs-string">&quot;How many cats are there?&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = ViltProcessor.from_pretrained(<span class="hljs-string">&quot;dandelin/vilt-b32-finetuned-vqa&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ViltForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;dandelin/vilt-b32-finetuned-vqa&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># prepare inputs</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = processor(image, text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># forward pass</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**encoding)
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits
<span class="hljs-meta">&gt;&gt;&gt; </span>idx = logits.argmax(-<span class="hljs-number">1</span>).item()
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;Predicted answer:&quot;</span>, model.config.id2label[idx])
Predicted answer: <span class="hljs-number">2</span>`}}),{c(){d=n("p"),b=a("Examples:"),g=p(),T(u.$$.fragment)},l(l){d=s(l,"P",{});var h=r(d);b=i(h,"Examples:"),h.forEach(o),g=m(l),k(u.$$.fragment,l)},m(l,h){_(l,d,h),e(d,b),_(l,g,h),w(u,l,h),v=!0},p:Lo,i(l){v||(x(u.$$.fragment,l),v=!0)},o(l){y(u.$$.fragment,l),v=!1},d(l){l&&o(d),l&&o(g),$(u,l)}}}function Rd(V){let d,b,g,u,v;return{c(){d=n("p"),b=a("Although the recipe for forward pass needs to be defined within this function, one should call the "),g=n("code"),u=a("Module"),v=a(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=s(l,"P",{});var h=r(d);b=i(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),g=s(h,"CODE",{});var F=r(g);u=i(F,"Module"),F.forEach(o),v=i(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(o)},m(l,h){_(l,d,h),e(d,b),e(d,g),e(g,u),e(d,v)},d(l){l&&o(d)}}}function Bd(V){let d,b,g,u,v;return u=new zo({props:{code:`from transformers import ViltProcessor, ViltForImagesAndTextClassification
import requests
from PIL import Image

image1 = Image.open(requests.get("https://lil.nlp.cornell.edu/nlvr/exs/ex0_0.jpg", stream=True).raw)
image2 = Image.open(requests.get("https://lil.nlp.cornell.edu/nlvr/exs/ex0_1.jpg", stream=True).raw)
text = "The left image contains twice the number of dogs as the right image."

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2")
model = ViltForImagesAndTextClassification.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2")

# prepare inputs
encoding = processor([image1, image2], text, return_tensors="pt")

# forward pass
outputs = model(input_ids=encoding.input_ids, pixel_values=encoding.pixel_values.unsqueeze(0))
logits = outputs.logits
idx = logits.argmax(-1).item()
print("Predicted answer:", model.config.id2label[idx])`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> ViltProcessor, ViltForImagesAndTextClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image

<span class="hljs-meta">&gt;&gt;&gt; </span>image1 = Image.<span class="hljs-built_in">open</span>(requests.get(<span class="hljs-string">&quot;https://lil.nlp.cornell.edu/nlvr/exs/ex0_0.jpg&quot;</span>, stream=<span class="hljs-literal">True</span>).raw)
<span class="hljs-meta">&gt;&gt;&gt; </span>image2 = Image.<span class="hljs-built_in">open</span>(requests.get(<span class="hljs-string">&quot;https://lil.nlp.cornell.edu/nlvr/exs/ex0_1.jpg&quot;</span>, stream=<span class="hljs-literal">True</span>).raw)
<span class="hljs-meta">&gt;&gt;&gt; </span>text = <span class="hljs-string">&quot;The left image contains twice the number of dogs as the right image.&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = ViltProcessor.from_pretrained(<span class="hljs-string">&quot;dandelin/vilt-b32-finetuned-nlvr2&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ViltForImagesAndTextClassification.from_pretrained(<span class="hljs-string">&quot;dandelin/vilt-b32-finetuned-nlvr2&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># prepare inputs</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = processor([image1, image2], text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># forward pass</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids=encoding.input_ids, pixel_values=encoding.pixel_values.unsqueeze(<span class="hljs-number">0</span>))
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits
<span class="hljs-meta">&gt;&gt;&gt; </span>idx = logits.argmax(-<span class="hljs-number">1</span>).item()
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;Predicted answer:&quot;</span>, model.config.id2label[idx])
Predicted answer: <span class="hljs-literal">True</span>`}}),{c(){d=n("p"),b=a("Examples:"),g=p(),T(u.$$.fragment)},l(l){d=s(l,"P",{});var h=r(d);b=i(h,"Examples:"),h.forEach(o),g=m(l),k(u.$$.fragment,l)},m(l,h){_(l,d,h),e(d,b),_(l,g,h),w(u,l,h),v=!0},p:Lo,i(l){v||(x(u.$$.fragment,l),v=!0)},o(l){y(u.$$.fragment,l),v=!1},d(l){l&&o(d),l&&o(g),$(u,l)}}}function Qd(V){let d,b,g,u,v;return{c(){d=n("p"),b=a("Although the recipe for forward pass needs to be defined within this function, one should call the "),g=n("code"),u=a("Module"),v=a(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=s(l,"P",{});var h=r(d);b=i(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),g=s(h,"CODE",{});var F=r(g);u=i(F,"Module"),F.forEach(o),v=i(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(o)},m(l,h){_(l,d,h),e(d,b),e(d,g),e(g,u),e(d,v)},d(l){l&&o(d)}}}function Ud(V){let d,b,g,u,v;return u=new zo({props:{code:`from transformers import ViltProcessor, ViltForImageAndTextRetrieval
import requests
from PIL import Image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-coco")
model = ViltForImageAndTextRetrieval.from_pretrained("dandelin/vilt-b32-finetuned-coco")

# forward pass
scores = dict()
for text in texts:
    # prepare inputs
    encoding = processor(image, text, return_tensors="pt")
    outputs = model(**encoding)
    scores[text] = outputs.logits[0, :].item()`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> ViltProcessor, ViltForImageAndTextRetrieval
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image

<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)
<span class="hljs-meta">&gt;&gt;&gt; </span>texts = [<span class="hljs-string">&quot;An image of two cats chilling on a couch&quot;</span>, <span class="hljs-string">&quot;A football player scoring a goal&quot;</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = ViltProcessor.from_pretrained(<span class="hljs-string">&quot;dandelin/vilt-b32-finetuned-coco&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ViltForImageAndTextRetrieval.from_pretrained(<span class="hljs-string">&quot;dandelin/vilt-b32-finetuned-coco&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># forward pass</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>scores = <span class="hljs-built_in">dict</span>()
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> text <span class="hljs-keyword">in</span> texts:
<span class="hljs-meta">... </span>    <span class="hljs-comment"># prepare inputs</span>
<span class="hljs-meta">... </span>    encoding = processor(image, text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">... </span>    outputs = model(**encoding)
<span class="hljs-meta">... </span>    scores[text] = outputs.logits[<span class="hljs-number">0</span>, :].item()`}}),{c(){d=n("p"),b=a("Examples:"),g=p(),T(u.$$.fragment)},l(l){d=s(l,"P",{});var h=r(d);b=i(h,"Examples:"),h.forEach(o),g=m(l),k(u.$$.fragment,l)},m(l,h){_(l,d,h),e(d,b),_(l,g,h),w(u,l,h),v=!0},p:Lo,i(l){v||(x(u.$$.fragment,l),v=!0)},o(l){y(u.$$.fragment,l),v=!1},d(l){l&&o(d),l&&o(g),$(u,l)}}}function Hd(V){let d,b,g,u,v;return{c(){d=n("p"),b=a("Although the recipe for forward pass needs to be defined within this function, one should call the "),g=n("code"),u=a("Module"),v=a(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){d=s(l,"P",{});var h=r(d);b=i(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),g=s(h,"CODE",{});var F=r(g);u=i(F,"Module"),F.forEach(o),v=i(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(o)},m(l,h){_(l,d,h),e(d,b),e(d,g),e(g,u),e(d,v)},d(l){l&&o(d)}}}function Kd(V){let d,b,g,u,v,l,h,F,Is,Dn,Z,ve,Io,at,Cs,Co,As,Wn,be,qs,it,Os,Ns,Rn,Jt,Ss,Bn,Xt,Ao,Ds,Qn,Zt,Ws,Un,q,lt,Rs,dt,Bs,Qs,Us,R,Hs,qo,Ks,Gs,Oo,Js,Xs,Yt,Zs,Ys,er,Y,tr,No,or,nr,eo,sr,rr,ar,So,ir,Hn,Te,al,Kn,ke,lr,to,dr,cr,Gn,K,pr,ct,mr,hr,pt,fr,ur,Jn,ee,we,Do,mt,gr,Wo,_r,Xn,P,ht,vr,te,br,Ro,Tr,kr,ft,wr,xr,yr,oe,$r,oo,Vr,Fr,no,Er,Mr,Pr,xe,Zn,ne,ye,Bo,ut,jr,Qo,Lr,Yn,j,gt,zr,Uo,Ir,Cr,_t,Ar,so,qr,Or,Nr,G,vt,Sr,Ho,Dr,Wr,$e,es,se,Ve,Ko,bt,Rr,Go,Br,ts,L,Tt,Qr,Jo,Ur,Hr,M,ro,Kr,Gr,ao,Jr,Xr,io,Zr,Yr,kt,Xo,ea,ta,oa,Zo,na,sa,ra,J,wt,aa,re,ia,Fe,la,Yo,da,ca,pa,Ee,ma,en,ha,fa,ua,ga,tn,_a,os,ae,Me,on,xt,va,nn,ba,ns,B,yt,Ta,$t,ka,sn,wa,xa,ya,O,Vt,$a,ie,Va,lo,Fa,Ea,rn,Ma,Pa,ja,Pe,La,je,ss,le,Le,an,Ft,za,ln,Ia,rs,z,Et,Ca,dn,Aa,qa,Mt,Oa,cn,Na,Sa,Da,N,Pt,Wa,de,Ra,co,Ba,Qa,pn,Ua,Ha,Ka,ze,Ga,Ie,as,ce,Ce,mn,jt,Ja,hn,Xa,is,I,Lt,Za,fn,Ya,ei,zt,ti,un,oi,ni,si,S,It,ri,pe,ai,po,ii,li,gn,di,ci,pi,Ae,mi,qe,ls,me,Oe,_n,Ct,hi,vn,fi,ds,Q,At,ui,bn,gi,_i,D,qt,vi,he,bi,mo,Ti,ki,Tn,wi,xi,yi,Ne,$i,Se,cs,fe,De,kn,Ot,Vi,wn,Fi,ps,C,Nt,Ei,xn,Mi,Pi,St,ji,yn,Li,zi,Ii,W,Dt,Ci,ue,Ai,ho,qi,Oi,$n,Ni,Si,Di,We,Wi,Re,ms,ge,Be,Vn,Wt,Ri,Fn,Bi,hs,A,Rt,Qi,En,Ui,Hi,Bt,Ki,Mn,Gi,Ji,Xi,X,Qt,Zi,_e,Yi,fo,el,tl,Pn,ol,nl,sl,Qe,fs;return l=new H({}),at=new H({}),mt=new H({}),ht=new E({props:{name:"class transformers.ViltConfig",anchor:"transformers.ViltConfig",parameters:[{name:"vocab_size",val:" = 30522"},{name:"type_vocab_size",val:" = 2"},{name:"modality_type_vocab_size",val:" = 2"},{name:"max_position_embeddings",val:" = 40"},{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.0"},{name:"attention_probs_dropout_prob",val:" = 0.0"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"is_encoder_decoder",val:" = False"},{name:"image_size",val:" = 384"},{name:"patch_size",val:" = 32"},{name:"num_channels",val:" = 3"},{name:"qkv_bias",val:" = True"},{name:"max_image_length",val:" = -1"},{name:"tie_word_embeddings",val:" = False"},{name:"num_images",val:" = -1"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.ViltConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 30522) &#x2014;
Vocabulary size of the text part of the model. Defines the number of different tokens that can be
represented by the <code>inputs_ids</code> passed when calling <a href="/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltModel">ViltModel</a>.`,name:"vocab_size"},{anchor:"transformers.ViltConfig.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed when calling <a href="/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltModel">ViltModel</a>. This is used when encoding
text.`,name:"type_vocab_size"},{anchor:"transformers.ViltConfig.modality_type_vocab_size",description:`<strong>modality_type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The vocabulary size of the modalities passed when calling <a href="/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltModel">ViltModel</a>. This is used after concatening the
embeddings of the text and image modalities.`,name:"modality_type_vocab_size"},{anchor:"transformers.ViltConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 40) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.ViltConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.ViltConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.ViltConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.ViltConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (i.e., feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.ViltConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;selu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.ViltConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.ViltConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.ViltConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.ViltConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.ViltConfig.image_size",description:`<strong>image_size</strong> (<code>int</code>, <em>optional</em>, defaults to 384) &#x2014;
The size (resolution) of each image.`,name:"image_size"},{anchor:"transformers.ViltConfig.patch_size",description:`<strong>patch_size</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
The size (resolution) of each patch.`,name:"patch_size"},{anchor:"transformers.ViltConfig.num_channels",description:`<strong>num_channels</strong> (<code>int</code>, <em>optional</em>, defaults to 3) &#x2014;
The number of input channels.`,name:"num_channels"},{anchor:"transformers.ViltConfig.qkv_bias",description:`<strong>qkv_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to add a bias to the queries, keys and values.`,name:"qkv_bias"},{anchor:"transformers.ViltConfig.max_image_length",description:`<strong>max_image_length</strong> (<code>int</code>, <em>optional</em>, defaults to -1) &#x2014;
The maximum number of patches to take as input for the Transformer encoder. If set to a positive integer,
the encoder will sample <code>max_image_length</code> patches at maximum. If set to -1, will not be taken into
account.`,name:"max_image_length"},{anchor:"transformers.ViltConfig.num_images",description:`<strong>num_images</strong> (<code>int</code>, <em>optional</em>, defaults to -1) &#x2014;
The number of images to use for natural language visual reasoning. If set to a positive integer, will be
used by <a href="/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltForImagesAndTextClassification">ViltForImagesAndTextClassification</a> for defining the classifier head.`,name:"num_images"}],source:"https://github.com/huggingface/transformers/blob/vr_5/src/transformers/models/vilt/configuration_vilt.py#L28"}}),xe=new jo({props:{anchor:"transformers.ViltConfig.example",$$slots:{default:[Cd]},$$scope:{ctx:V}}}),ut=new H({}),gt=new E({props:{name:"class transformers.ViltFeatureExtractor",anchor:"transformers.ViltFeatureExtractor",parameters:[{name:"do_resize",val:" = True"},{name:"size",val:" = 384"},{name:"size_divisor",val:" = 32"},{name:"resample",val:" = <Resampling.BICUBIC: 3>"},{name:"do_normalize",val:" = True"},{name:"image_mean",val:" = None"},{name:"image_std",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.ViltFeatureExtractor.do_resize",description:`<strong>do_resize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to resize the input based on <code>size</code>.`,name:"do_resize"},{anchor:"transformers.ViltFeatureExtractor.size",description:`<strong>size</strong> (<code>int</code>, <em>optional</em>, defaults to 384) &#x2014;
Resize the shorter side of the input to the given size. Should be an integer. The longer side will be
limited to under int((1333 / 800) * size) while preserving the aspect ratio. Only has an effect if
<code>do_resize</code> is set to <code>True</code>.`,name:"size"},{anchor:"transformers.ViltFeatureExtractor.size_divisor",description:`<strong>size_divisor</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
The size by which to make sure both the height and width can be divided.`,name:"size_divisor"},{anchor:"transformers.ViltFeatureExtractor.resample",description:`<strong>resample</strong> (<code>int</code>, <em>optional</em>, defaults to <code>PIL.Image.BICUBIC</code>) &#x2014;
An optional resampling filter. This can be one of <code>PIL.Image.NEAREST</code>, <code>PIL.Image.BOX</code>,
<code>PIL.Image.BILINEAR</code>, <code>PIL.Image.HAMMING</code>, <code>PIL.Image.BICUBIC</code> or <code>PIL.Image.LANCZOS</code>. Only has an effect
if <code>do_resize</code> is set to <code>True</code>.`,name:"resample"},{anchor:"transformers.ViltFeatureExtractor.do_normalize",description:`<strong>do_normalize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to normalize the input with mean and standard deviation.`,name:"do_normalize"},{anchor:"transformers.ViltFeatureExtractor.image_mean",description:`<strong>image_mean</strong> (<code>List[int]</code>, defaults to <code>[0.5, 0.5, 0.5]</code>) &#x2014;
The sequence of means for each channel, to be used when normalizing images.`,name:"image_mean"},{anchor:"transformers.ViltFeatureExtractor.image_std",description:`<strong>image_std</strong> (<code>List[int]</code>, defaults to <code>[0.5, 0.5, 0.5]</code>) &#x2014;
The sequence of standard deviations for each channel, to be used when normalizing images.`,name:"image_std"}],source:"https://github.com/huggingface/transformers/blob/vr_5/src/transformers/models/vilt/feature_extraction_vilt.py#L40"}}),vt=new E({props:{name:"__call__",anchor:"transformers.ViltFeatureExtractor.__call__",parameters:[{name:"images",val:": typing.Union[PIL.Image.Image, numpy.ndarray, ForwardRef('torch.Tensor'), typing.List[PIL.Image.Image], typing.List[numpy.ndarray], typing.List[ForwardRef('torch.Tensor')]]"},{name:"pad_and_return_pixel_mask",val:": typing.Optional[bool] = True"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.ViltFeatureExtractor.__call__.images",description:`<strong>images</strong> (<code>PIL.Image.Image</code>, <code>np.ndarray</code>, <code>torch.Tensor</code>, <code>List[PIL.Image.Image]</code>, <code>List[np.ndarray]</code>, <code>List[torch.Tensor]</code>) &#x2014;
The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
number of channels, H and W are image height and width.`,name:"images"},{anchor:"transformers.ViltFeatureExtractor.__call__.pad_and_return_pixel_mask",description:`<strong>pad_and_return_pixel_mask</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to pad images up to the largest image in a batch and create a pixel mask.</p>
<p>If left to the default, will return a pixel mask that is:</p>
<ul>
<li>1 for pixels that are real (i.e. <strong>not masked</strong>),</li>
<li>0 for pixels that are padding (i.e. <strong>masked</strong>).</li>
</ul>`,name:"pad_and_return_pixel_mask"},{anchor:"transformers.ViltFeatureExtractor.__call__.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/pr_5/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>, defaults to <code>&apos;np&apos;</code>) &#x2014;
If set, will return tensors of a particular framework. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return NumPy <code>np.ndarray</code> objects.</li>
<li><code>&apos;jax&apos;</code>: Return JAX <code>jnp.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"}],source:"https://github.com/huggingface/transformers/blob/vr_5/src/transformers/models/vilt/feature_extraction_vilt.py#L181",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_5/en/main_classes/feature_extractor#transformers.BatchFeature"
>BatchFeature</a> with the following fields:</p>
<ul>
<li><strong>pixel_values</strong> \u2014 Pixel values to be fed to a model, of shape (batch_size, num_channels, height,
width).</li>
<li><strong>pixel_mask</strong> \u2014 Pixel mask to be fed to a model (when <code>return_pixel_mask=True</code> or if <em>\u201Cpixel_mask\u201D</em> is
in <code>self.model_input_names</code>).</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_5/en/main_classes/feature_extractor#transformers.BatchFeature"
>BatchFeature</a></p>
`}}),$e=new Gt({props:{warning:!0,$$slots:{default:[Ad]},$$scope:{ctx:V}}}),bt=new H({}),Tt=new E({props:{name:"class transformers.ViltProcessor",anchor:"transformers.ViltProcessor",parameters:[{name:"feature_extractor",val:""},{name:"tokenizer",val:""}],parametersDescription:[{anchor:"transformers.ViltProcessor.feature_extractor",description:`<strong>feature_extractor</strong> (<code>ViltFeatureExtractor</code>) &#x2014;
An instance of <a href="/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltFeatureExtractor">ViltFeatureExtractor</a>. The feature extractor is a required input.`,name:"feature_extractor"},{anchor:"transformers.ViltProcessor.tokenizer",description:"<strong>tokenizer</strong> (<code>BertTokenizerFast</code>) &#x2014;\nAn instance of [&#x2018;BertTokenizerFast`]. The tokenizer is a required input.",name:"tokenizer"}],source:"https://github.com/huggingface/transformers/blob/vr_5/src/transformers/models/vilt/processing_vilt.py#L26"}}),wt=new E({props:{name:"__call__",anchor:"transformers.ViltProcessor.__call__",parameters:[{name:"images",val:""},{name:"text",val:": typing.Union[str, typing.List[str], typing.List[typing.List[str]]] = None"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy] = False"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"return_token_type_ids",val:": typing.Optional[bool] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_overflowing_tokens",val:": bool = False"},{name:"return_special_tokens_mask",val:": bool = False"},{name:"return_offsets_mapping",val:": bool = False"},{name:"return_length",val:": bool = False"},{name:"verbose",val:": bool = True"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_5/src/transformers/models/vilt/processing_vilt.py#L46"}}),xt=new H({}),yt=new E({props:{name:"class transformers.ViltModel",anchor:"transformers.ViltModel",parameters:[{name:"config",val:""},{name:"add_pooling_layer",val:" = True"}],parametersDescription:[{anchor:"transformers.ViltModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltConfig">ViltConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_5/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_5/src/transformers/models/vilt/modeling_vilt.py#L732"}}),Vt=new E({props:{name:"forward",anchor:"transformers.ViltModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"pixel_values",val:": typing.Optional[torch.FloatTensor] = None"},{name:"pixel_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"image_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"image_token_type_idx",val:": typing.Optional[int] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.ViltModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>({0})</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Indices can be obtained using <a href="/docs/transformers/pr_5/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See
<a href="/docs/transformers/pr_5/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and <a href="/docs/transformers/pr_5/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details. <a href="../glossary#input-ids">What are input
IDs?</a>`,name:"input_ids"},{anchor:"transformers.ViltModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>({0})</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.
<a href="../glossary#attention-mask">What are attention masks?</a></li>
</ul>`,name:"attention_mask"},{anchor:"transformers.ViltModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>({0})</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
<a href="../glossary#token-type-ids">What are token type IDs?</a></li>
</ul>`,name:"token_type_ids"},{anchor:"transformers.ViltModel.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, height, width)</code>) &#x2014;
Pixel values. Pixel values can be obtained using <a href="/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltFeatureExtractor">ViltFeatureExtractor</a>. See
<a href="/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltFeatureExtractor.__call__">ViltFeatureExtractor.<strong>call</strong>()</a> for details.`,name:"pixel_values"},{anchor:"transformers.ViltModel.forward.pixel_mask",description:`<strong>pixel_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, height, width)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding pixel values. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for pixels that are real (i.e. <strong>not masked</strong>),</li>
<li>0 for pixels that are padding (i.e. <strong>masked</strong>).
<code>What are attention masks? &lt;../glossary.html#attention-mask&gt;</code>__</li>
</ul>`,name:"pixel_mask"},{anchor:"transformers.ViltModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ViltModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>({0}, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ViltModel.forward.image_embeds",description:`<strong>image_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_patches, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>pixel_values</code>, you can choose to directly pass an embedded representation.
This is useful if you want more control over how to convert <code>pixel_values</code> into patch embeddings.`,name:"image_embeds"},{anchor:"transformers.ViltModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ViltModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ViltModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_5/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_5/src/transformers/models/vilt/modeling_vilt.py#L760",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_5/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling"
>transformers.modeling_outputs.BaseModelOutputWithPooling</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltConfig"
>ViltConfig</a>) and inputs.</p>
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
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_5/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling"
>transformers.modeling_outputs.BaseModelOutputWithPooling</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Pe=new Gt({props:{$$slots:{default:[qd]},$$scope:{ctx:V}}}),je=new jo({props:{anchor:"transformers.ViltModel.forward.example",$$slots:{default:[Od]},$$scope:{ctx:V}}}),Ft=new H({}),Et=new E({props:{name:"class transformers.ViltForMaskedLM",anchor:"transformers.ViltForMaskedLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.ViltForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltConfig">ViltConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_5/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_5/src/transformers/models/vilt/modeling_vilt.py#L897"}}),Pt=new E({props:{name:"forward",anchor:"transformers.ViltForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"pixel_values",val:": typing.Optional[torch.FloatTensor] = None"},{name:"pixel_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"image_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.ViltForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Indices can be obtained using <a href="/docs/transformers/pr_5/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See
<a href="/docs/transformers/pr_5/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and <a href="/docs/transformers/pr_5/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details. <a href="../glossary#input-ids">What are input
IDs?</a>`,name:"input_ids"},{anchor:"transformers.ViltForMaskedLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.
<a href="../glossary#attention-mask">What are attention masks?</a></li>
</ul>`,name:"attention_mask"},{anchor:"transformers.ViltForMaskedLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
<a href="../glossary#token-type-ids">What are token type IDs?</a></li>
</ul>`,name:"token_type_ids"},{anchor:"transformers.ViltForMaskedLM.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, height, width)</code>) &#x2014;
Pixel values. Pixel values can be obtained using <a href="/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltFeatureExtractor">ViltFeatureExtractor</a>. See
<a href="/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltFeatureExtractor.__call__">ViltFeatureExtractor.<strong>call</strong>()</a> for details.`,name:"pixel_values"},{anchor:"transformers.ViltForMaskedLM.forward.pixel_mask",description:`<strong>pixel_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, height, width)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding pixel values. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for pixels that are real (i.e. <strong>not masked</strong>),</li>
<li>0 for pixels that are padding (i.e. <strong>masked</strong>).
<code>What are attention masks? &lt;../glossary.html#attention-mask&gt;</code>__</li>
</ul>`,name:"pixel_mask"},{anchor:"transformers.ViltForMaskedLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ViltForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ViltForMaskedLM.forward.image_embeds",description:`<strong>image_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_patches, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>pixel_values</code>, you can choose to directly pass an embedded representation.
This is useful if you want more control over how to convert <code>pixel_values</code> into patch embeddings.`,name:"image_embeds"},{anchor:"transformers.ViltForMaskedLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ViltForMaskedLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ViltForMaskedLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_5/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.ViltForMaskedLM.forward.labels",description:`<strong>labels</strong> (<em>torch.LongTensor</em> of shape <em>(batch_size, sequence_length)</em>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <em>[-100, 0, &#x2026;,
config.vocab_size]</em> (see <em>input_ids</em> docstring) Tokens with indices set to <em>-100</em> are ignored (masked), the
loss is only computed for the tokens with labels in <em>[0, &#x2026;, config.vocab_size]</em>`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_5/src/transformers/models/vilt/modeling_vilt.py#L913",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_5/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltConfig"
>ViltConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_5/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ze=new Gt({props:{$$slots:{default:[Nd]},$$scope:{ctx:V}}}),Ie=new jo({props:{anchor:"transformers.ViltForMaskedLM.forward.example",$$slots:{default:[Sd]},$$scope:{ctx:V}}}),jt=new H({}),Lt=new E({props:{name:"class transformers.ViltForQuestionAnswering",anchor:"transformers.ViltForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.ViltForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltConfig">ViltConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_5/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_5/src/transformers/models/vilt/modeling_vilt.py#L1069"}}),It=new E({props:{name:"forward",anchor:"transformers.ViltForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"pixel_values",val:": typing.Optional[torch.FloatTensor] = None"},{name:"pixel_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"image_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.ViltForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>({0})</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Indices can be obtained using <a href="/docs/transformers/pr_5/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See
<a href="/docs/transformers/pr_5/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and <a href="/docs/transformers/pr_5/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details. <a href="../glossary#input-ids">What are input
IDs?</a>`,name:"input_ids"},{anchor:"transformers.ViltForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>({0})</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.
<a href="../glossary#attention-mask">What are attention masks?</a></li>
</ul>`,name:"attention_mask"},{anchor:"transformers.ViltForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>({0})</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
<a href="../glossary#token-type-ids">What are token type IDs?</a></li>
</ul>`,name:"token_type_ids"},{anchor:"transformers.ViltForQuestionAnswering.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, height, width)</code>) &#x2014;
Pixel values. Pixel values can be obtained using <a href="/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltFeatureExtractor">ViltFeatureExtractor</a>. See
<a href="/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltFeatureExtractor.__call__">ViltFeatureExtractor.<strong>call</strong>()</a> for details.`,name:"pixel_values"},{anchor:"transformers.ViltForQuestionAnswering.forward.pixel_mask",description:`<strong>pixel_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, height, width)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding pixel values. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for pixels that are real (i.e. <strong>not masked</strong>),</li>
<li>0 for pixels that are padding (i.e. <strong>masked</strong>).
<code>What are attention masks? &lt;../glossary.html#attention-mask&gt;</code>__</li>
</ul>`,name:"pixel_mask"},{anchor:"transformers.ViltForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ViltForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>({0}, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ViltForQuestionAnswering.forward.image_embeds",description:`<strong>image_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_patches, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>pixel_values</code>, you can choose to directly pass an embedded representation.
This is useful if you want more control over how to convert <code>pixel_values</code> into patch embeddings.`,name:"image_embeds"},{anchor:"transformers.ViltForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ViltForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ViltForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_5/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.ViltForQuestionAnswering.forward.labels",description:`<strong>labels</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_labels)</code>, <em>optional</em>) &#x2014;
Labels for computing the visual question answering loss. This tensor must be either a one-hot encoding of
all answers that are applicable for a given example in the batch, or a soft encoding indicating which
answers are applicable, where 1.0 is the highest score.`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_5/src/transformers/models/vilt/modeling_vilt.py#L1087",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_5/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltConfig"
>ViltConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_5/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Ae=new Gt({props:{$$slots:{default:[Dd]},$$scope:{ctx:V}}}),qe=new jo({props:{anchor:"transformers.ViltForQuestionAnswering.forward.example",$$slots:{default:[Wd]},$$scope:{ctx:V}}}),Ct=new H({}),At=new E({props:{name:"class transformers.ViltForImagesAndTextClassification",anchor:"transformers.ViltForImagesAndTextClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.ViltForImagesAndTextClassification.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>({0})</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Indices can be obtained using <a href="/docs/transformers/pr_5/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See
<a href="/docs/transformers/pr_5/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and <a href="/docs/transformers/pr_5/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details. <a href="../glossary#input-ids">What are input
IDs?</a>`,name:"input_ids"},{anchor:"transformers.ViltForImagesAndTextClassification.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>({0})</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.
<a href="../glossary#attention-mask">What are attention masks?</a></li>
</ul>`,name:"attention_mask"},{anchor:"transformers.ViltForImagesAndTextClassification.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>({0})</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
<a href="../glossary#token-type-ids">What are token type IDs?</a></li>
</ul>`,name:"token_type_ids"},{anchor:"transformers.ViltForImagesAndTextClassification.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_images, num_channels, height, width)</code>) &#x2014;
Pixel values. Pixel values can be obtained using <a href="/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltFeatureExtractor">ViltFeatureExtractor</a>. See
<a href="/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltFeatureExtractor.__call__">ViltFeatureExtractor.<strong>call</strong>()</a> for details.`,name:"pixel_values"},{anchor:"transformers.ViltForImagesAndTextClassification.pixel_mask",description:`<strong>pixel_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_images, height, width)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding pixel values. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for pixels that are real (i.e. <strong>not masked</strong>),</li>
<li>0 for pixels that are padding (i.e. <strong>masked</strong>).
<code>What are attention masks? &lt;../glossary.html#attention-mask&gt;</code>__</li>
</ul>`,name:"pixel_mask"},{anchor:"transformers.ViltForImagesAndTextClassification.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ViltForImagesAndTextClassification.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>({0}, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ViltForImagesAndTextClassification.image_embeds",description:`<strong>image_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_images, num_patches, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>pixel_values</code>, you can choose to directly pass an embedded representation.
This is useful if you want more control over how to convert <code>pixel_values</code> into patch embeddings.`,name:"image_embeds"},{anchor:"transformers.ViltForImagesAndTextClassification.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ViltForImagesAndTextClassification.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ViltForImagesAndTextClassification.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_5/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_5/src/transformers/models/vilt/modeling_vilt.py#L1279"}}),qt=new E({props:{name:"forward",anchor:"transformers.ViltForImagesAndTextClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"pixel_values",val:": typing.Optional[torch.FloatTensor] = None"},{name:"pixel_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"image_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.ViltForImagesAndTextClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>({0})</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Indices can be obtained using <a href="/docs/transformers/pr_5/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See
<a href="/docs/transformers/pr_5/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and <a href="/docs/transformers/pr_5/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details. <a href="../glossary#input-ids">What are input
IDs?</a>`,name:"input_ids"},{anchor:"transformers.ViltForImagesAndTextClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>({0})</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.
<a href="../glossary#attention-mask">What are attention masks?</a></li>
</ul>`,name:"attention_mask"},{anchor:"transformers.ViltForImagesAndTextClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>({0})</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
<a href="../glossary#token-type-ids">What are token type IDs?</a></li>
</ul>`,name:"token_type_ids"},{anchor:"transformers.ViltForImagesAndTextClassification.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, height, width)</code>) &#x2014;
Pixel values. Pixel values can be obtained using <a href="/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltFeatureExtractor">ViltFeatureExtractor</a>. See
<a href="/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltFeatureExtractor.__call__">ViltFeatureExtractor.<strong>call</strong>()</a> for details.`,name:"pixel_values"},{anchor:"transformers.ViltForImagesAndTextClassification.forward.pixel_mask",description:`<strong>pixel_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, height, width)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding pixel values. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for pixels that are real (i.e. <strong>not masked</strong>),</li>
<li>0 for pixels that are padding (i.e. <strong>masked</strong>).
<code>What are attention masks? &lt;../glossary.html#attention-mask&gt;</code>__</li>
</ul>`,name:"pixel_mask"},{anchor:"transformers.ViltForImagesAndTextClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ViltForImagesAndTextClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>({0}, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ViltForImagesAndTextClassification.forward.image_embeds",description:`<strong>image_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_patches, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>pixel_values</code>, you can choose to directly pass an embedded representation.
This is useful if you want more control over how to convert <code>pixel_values</code> into patch embeddings.`,name:"image_embeds"},{anchor:"transformers.ViltForImagesAndTextClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ViltForImagesAndTextClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ViltForImagesAndTextClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_5/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.ViltForImagesAndTextClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Binary classification labels.`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_5/src/transformers/models/vilt/modeling_vilt.py#L1298",returnDescription:`
<p>A <code>transformers.models.vilt.modeling_vilt.ViltForImagesAndTextClassificationOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltConfig"
>ViltConfig</a>) and inputs.</p>
<ul>
<li><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) \u2014 Classification (or regression if config.num_labels==1) loss.</li>
<li><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.num_labels)</code>) \u2014 Classification (or regression if config.num_labels==1) scores (before SoftMax).</li>
<li><strong>hidden_states</strong> (<code>List[tuple(torch.FloatTensor)]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 List of tuples of <code>torch.FloatTensor</code> (one for each image-text pair, each tuple containing the output of
the embeddings + one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.
Hidden-states of the model at the output of each layer plus the initial embedding outputs.</li>
<li><strong>attentions</strong> (<code>List[tuple(torch.FloatTensor)]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 List of tuples of <code>torch.FloatTensor</code> (one for each image-text pair, each tuple containing the attention
weights of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>. Attentions weights after the
attention softmax, used to compute the weighted average in the self-attention heads.</li>
</ul>
`,returnType:`
<p><code>transformers.models.vilt.modeling_vilt.ViltForImagesAndTextClassificationOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Ne=new Gt({props:{$$slots:{default:[Rd]},$$scope:{ctx:V}}}),Se=new jo({props:{anchor:"transformers.ViltForImagesAndTextClassification.forward.example",$$slots:{default:[Bd]},$$scope:{ctx:V}}}),Ot=new H({}),Nt=new E({props:{name:"class transformers.ViltForImageAndTextRetrieval",anchor:"transformers.ViltForImageAndTextRetrieval",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.ViltForImageAndTextRetrieval.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltConfig">ViltConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_5/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_5/src/transformers/models/vilt/modeling_vilt.py#L1180"}}),Dt=new E({props:{name:"forward",anchor:"transformers.ViltForImageAndTextRetrieval.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"pixel_values",val:": typing.Optional[torch.FloatTensor] = None"},{name:"pixel_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"image_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.ViltForImageAndTextRetrieval.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>({0})</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Indices can be obtained using <a href="/docs/transformers/pr_5/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See
<a href="/docs/transformers/pr_5/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and <a href="/docs/transformers/pr_5/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details. <a href="../glossary#input-ids">What are input
IDs?</a>`,name:"input_ids"},{anchor:"transformers.ViltForImageAndTextRetrieval.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>({0})</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.
<a href="../glossary#attention-mask">What are attention masks?</a></li>
</ul>`,name:"attention_mask"},{anchor:"transformers.ViltForImageAndTextRetrieval.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>({0})</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
<a href="../glossary#token-type-ids">What are token type IDs?</a></li>
</ul>`,name:"token_type_ids"},{anchor:"transformers.ViltForImageAndTextRetrieval.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, height, width)</code>) &#x2014;
Pixel values. Pixel values can be obtained using <a href="/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltFeatureExtractor">ViltFeatureExtractor</a>. See
<a href="/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltFeatureExtractor.__call__">ViltFeatureExtractor.<strong>call</strong>()</a> for details.`,name:"pixel_values"},{anchor:"transformers.ViltForImageAndTextRetrieval.forward.pixel_mask",description:`<strong>pixel_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, height, width)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding pixel values. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for pixels that are real (i.e. <strong>not masked</strong>),</li>
<li>0 for pixels that are padding (i.e. <strong>masked</strong>).
<code>What are attention masks? &lt;../glossary.html#attention-mask&gt;</code>__</li>
</ul>`,name:"pixel_mask"},{anchor:"transformers.ViltForImageAndTextRetrieval.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ViltForImageAndTextRetrieval.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>({0}, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ViltForImageAndTextRetrieval.forward.image_embeds",description:`<strong>image_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_patches, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>pixel_values</code>, you can choose to directly pass an embedded representation.
This is useful if you want more control over how to convert <code>pixel_values</code> into patch embeddings.`,name:"image_embeds"},{anchor:"transformers.ViltForImageAndTextRetrieval.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ViltForImageAndTextRetrieval.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ViltForImageAndTextRetrieval.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_5/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.ViltForImageAndTextRetrieval.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels are currently not supported.`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_5/src/transformers/models/vilt/modeling_vilt.py#L1192",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_5/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltConfig"
>ViltConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_5/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),We=new Gt({props:{$$slots:{default:[Qd]},$$scope:{ctx:V}}}),Re=new jo({props:{anchor:"transformers.ViltForImageAndTextRetrieval.forward.example",$$slots:{default:[Ud]},$$scope:{ctx:V}}}),Wt=new H({}),Rt=new E({props:{name:"class transformers.ViltForTokenClassification",anchor:"transformers.ViltForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.ViltForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltConfig">ViltConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_5/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_5/src/transformers/models/vilt/modeling_vilt.py#L1419"}}),Qt=new E({props:{name:"forward",anchor:"transformers.ViltForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"pixel_values",val:": typing.Optional[torch.FloatTensor] = None"},{name:"pixel_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"image_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.ViltForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>({0})</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Indices can be obtained using <a href="/docs/transformers/pr_5/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See
<a href="/docs/transformers/pr_5/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and <a href="/docs/transformers/pr_5/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details. <a href="../glossary#input-ids">What are input
IDs?</a>`,name:"input_ids"},{anchor:"transformers.ViltForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>({0})</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.
<a href="../glossary#attention-mask">What are attention masks?</a></li>
</ul>`,name:"attention_mask"},{anchor:"transformers.ViltForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>({0})</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
<a href="../glossary#token-type-ids">What are token type IDs?</a></li>
</ul>`,name:"token_type_ids"},{anchor:"transformers.ViltForTokenClassification.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, height, width)</code>) &#x2014;
Pixel values. Pixel values can be obtained using <a href="/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltFeatureExtractor">ViltFeatureExtractor</a>. See
<a href="/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltFeatureExtractor.__call__">ViltFeatureExtractor.<strong>call</strong>()</a> for details.`,name:"pixel_values"},{anchor:"transformers.ViltForTokenClassification.forward.pixel_mask",description:`<strong>pixel_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, height, width)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding pixel values. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for pixels that are real (i.e. <strong>not masked</strong>),</li>
<li>0 for pixels that are padding (i.e. <strong>masked</strong>).
<code>What are attention masks? &lt;../glossary.html#attention-mask&gt;</code>__</li>
</ul>`,name:"pixel_mask"},{anchor:"transformers.ViltForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ViltForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>({0}, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ViltForTokenClassification.forward.image_embeds",description:`<strong>image_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_patches, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>pixel_values</code>, you can choose to directly pass an embedded representation.
This is useful if you want more control over how to convert <code>pixel_values</code> into patch embeddings.`,name:"image_embeds"},{anchor:"transformers.ViltForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ViltForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ViltForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_5/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.ViltForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, text_sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_5/src/transformers/models/vilt/modeling_vilt.py#L1435",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_5/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltConfig"
>ViltConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_5/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Qe=new Gt({props:{$$slots:{default:[Hd]},$$scope:{ctx:V}}}),{c(){d=n("meta"),b=p(),g=n("h1"),u=n("a"),v=n("span"),T(l.$$.fragment),h=p(),F=n("span"),Is=a("ViLT"),Dn=p(),Z=n("h2"),ve=n("a"),Io=n("span"),T(at.$$.fragment),Cs=p(),Co=n("span"),As=a("Overview"),Wn=p(),be=n("p"),qs=a("The ViLT model was proposed in "),it=n("a"),Os=a("ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision"),Ns=a(`
by Wonjae Kim, Bokyung Son, Ildoo Kim. ViLT incorporates text embeddings into a Vision Transformer (ViT), allowing it to have a minimal design
for Vision-and-Language Pre-training (VLP).`),Rn=p(),Jt=n("p"),Ss=a("The abstract from the paper is the following:"),Bn=p(),Xt=n("p"),Ao=n("em"),Ds=a(`Vision-and-Language Pre-training (VLP) has improved performance on various joint vision-and-language downstream tasks.
Current approaches to VLP heavily rely on image feature extraction processes, most of which involve region supervision
(e.g., object detection) and the convolutional architecture (e.g., ResNet). Although disregarded in the literature, we
find it problematic in terms of both (1) efficiency/speed, that simply extracting input features requires much more
computation than the multimodal interaction steps; and (2) expressive power, as it is upper bounded to the expressive
power of the visual embedder and its predefined visual vocabulary. In this paper, we present a minimal VLP model,
Vision-and-Language Transformer (ViLT), monolithic in the sense that the processing of visual inputs is drastically
simplified to just the same convolution-free manner that we process textual inputs. We show that ViLT is up to tens of
times faster than previous VLP models, yet with competitive or better downstream task performance.`),Qn=p(),Zt=n("p"),Ws=a("Tips:"),Un=p(),q=n("ul"),lt=n("li"),Rs=a("The quickest way to get started with ViLT is by checking the "),dt=n("a"),Bs=a("example notebooks"),Qs=a(`
(which showcase both inference and fine-tuning on custom data).`),Us=p(),R=n("li"),Hs=a("ViLT is a model that takes both "),qo=n("code"),Ks=a("pixel_values"),Gs=a(" and "),Oo=n("code"),Js=a("input_ids"),Xs=a(" as input. One can use "),Yt=n("a"),Zs=a("ViltProcessor"),Ys=a(` to prepare data for the model.
This processor wraps a feature extractor (for the image modality) and a tokenizer (for the language modality) into one.`),er=p(),Y=n("li"),tr=a(`ViLT is trained with images of various sizes: the authors resize the shorter edge of input images to 384 and limit the longer edge to
under 640 while preserving the aspect ratio. To make batching of images possible, the authors use a `),No=n("code"),or=a("pixel_mask"),nr=a(` that indicates
which pixel values are real and which are padding. `),eo=n("a"),sr=a("ViltProcessor"),rr=a(" automatically creates this for you."),ar=p(),So=n("li"),ir=a(`The design of ViLT is very similar to that of a standard Vision Transformer (ViT). The only difference is that the model includes
additional embedding layers for the language modality.`),Hn=p(),Te=n("img"),Kn=p(),ke=n("small"),lr=a("ViLT architecture. Taken from the "),to=n("a"),dr=a("original paper"),cr=a("."),Gn=p(),K=n("p"),pr=a("This model was contributed by "),ct=n("a"),mr=a("nielsr"),hr=a(". The original code can be found "),pt=n("a"),fr=a("here"),ur=a("."),Jn=p(),ee=n("h2"),we=n("a"),Do=n("span"),T(mt.$$.fragment),gr=p(),Wo=n("span"),_r=a("ViltConfig"),Xn=p(),P=n("div"),T(ht.$$.fragment),vr=p(),te=n("p"),br=a("This is the configuration class to store the configuration of a "),Ro=n("code"),Tr=a("ViLTModel"),kr=a(`. It is used to instantiate an ViLT
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the ViLT
`),ft=n("a"),wr=a("dandelin/vilt-b32-mlm"),xr=a(" architecture."),yr=p(),oe=n("p"),$r=a("Configuration objects inherit from "),oo=n("a"),Vr=a("PretrainedConfig"),Fr=a(` and can be used to control the model outputs. Read the
documentation from `),no=n("a"),Er=a("PretrainedConfig"),Mr=a(" for more information."),Pr=p(),T(xe.$$.fragment),Zn=p(),ne=n("h2"),ye=n("a"),Bo=n("span"),T(ut.$$.fragment),jr=p(),Qo=n("span"),Lr=a("ViltFeatureExtractor"),Yn=p(),j=n("div"),T(gt.$$.fragment),zr=p(),Uo=n("p"),Ir=a("Constructs a ViLT feature extractor."),Cr=p(),_t=n("p"),Ar=a("This feature extractor inherits from "),so=n("a"),qr=a("FeatureExtractionMixin"),Or=a(` which contains most of the main methods. Users
should refer to this superclass for more information regarding those methods.`),Nr=p(),G=n("div"),T(vt.$$.fragment),Sr=p(),Ho=n("p"),Dr=a("Main method to prepare for the model one or several image(s)."),Wr=p(),T($e.$$.fragment),es=p(),se=n("h2"),Ve=n("a"),Ko=n("span"),T(bt.$$.fragment),Rr=p(),Go=n("span"),Br=a("ViltProcessor"),ts=p(),L=n("div"),T(Tt.$$.fragment),Qr=p(),Jo=n("p"),Ur=a("Constructs a ViLT processor which wraps a BERT tokenizer and ViLT feature extractor into a single processor."),Hr=p(),M=n("p"),ro=n("a"),Kr=a("ViltProcessor"),Gr=a(" offers all the functionalities of "),ao=n("a"),Jr=a("ViltFeatureExtractor"),Xr=a(" and "),io=n("a"),Zr=a("BertTokenizerFast"),Yr=a(`. See the
docstring of `),kt=n("a"),Xo=n("strong"),ea=a("call"),ta=a("()"),oa=a(" and "),Zo=n("code"),na=a("decode()"),sa=a(" for more information."),ra=p(),J=n("div"),T(wt.$$.fragment),aa=p(),re=n("p"),ia=a("This method uses "),Fe=n("a"),la=a("ViltFeatureExtractor."),Yo=n("strong"),da=a("call"),ca=a("()"),pa=a(` method to prepare image(s) for the model, and
`),Ee=n("a"),ma=a("BertTokenizerFast."),en=n("strong"),ha=a("call"),fa=a("()"),ua=a(" to prepare text for the model."),ga=p(),tn=n("p"),_a=a("Please refer to the docstring of the above two methods for more information."),os=p(),ae=n("h2"),Me=n("a"),on=n("span"),T(xt.$$.fragment),va=p(),nn=n("span"),ba=a("ViltModel"),ns=p(),B=n("div"),T(yt.$$.fragment),Ta=p(),$t=n("p"),ka=a(`The bare ViLT Model transformer outputting raw hidden-states without any specific head on top.
This model is a PyTorch `),sn=n("code"),wa=a("torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>"),xa=a(`_ subclass. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),ya=p(),O=n("div"),T(Vt.$$.fragment),$a=p(),ie=n("p"),Va=a("The "),lo=n("a"),Fa=a("ViltModel"),Ea=a(" forward method, overrides the "),rn=n("code"),Ma=a("__call__"),Pa=a(" special method."),ja=p(),T(Pe.$$.fragment),La=p(),T(je.$$.fragment),ss=p(),le=n("h2"),Le=n("a"),an=n("span"),T(Ft.$$.fragment),za=p(),ln=n("span"),Ia=a("ViltForMaskedLM"),rs=p(),z=n("div"),T(Et.$$.fragment),Ca=p(),dn=n("p"),Aa=a("ViLT Model with a language modeling head on top as done during pretraining."),qa=p(),Mt=n("p"),Oa=a("This model is a PyTorch "),cn=n("code"),Na=a("torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>"),Sa=a(`_ subclass. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),Da=p(),N=n("div"),T(Pt.$$.fragment),Wa=p(),de=n("p"),Ra=a("The "),co=n("a"),Ba=a("ViltForMaskedLM"),Qa=a(" forward method, overrides the "),pn=n("code"),Ua=a("__call__"),Ha=a(" special method."),Ka=p(),T(ze.$$.fragment),Ga=p(),T(Ie.$$.fragment),as=p(),ce=n("h2"),Ce=n("a"),mn=n("span"),T(jt.$$.fragment),Ja=p(),hn=n("span"),Xa=a("ViltForQuestionAnswering"),is=p(),I=n("div"),T(Lt.$$.fragment),Za=p(),fn=n("p"),Ya=a(`Vilt Model transformer with a classifier head on top (a linear layer on top of the final hidden state of the [CLS]
token) for visual question answering, e.g. for VQAv2.`),ei=p(),zt=n("p"),ti=a("This model is a PyTorch "),un=n("code"),oi=a("torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>"),ni=a(`_ subclass. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),si=p(),S=n("div"),T(It.$$.fragment),ri=p(),pe=n("p"),ai=a("The "),po=n("a"),ii=a("ViltForQuestionAnswering"),li=a(" forward method, overrides the "),gn=n("code"),di=a("__call__"),ci=a(" special method."),pi=p(),T(Ae.$$.fragment),mi=p(),T(qe.$$.fragment),ls=p(),me=n("h2"),Oe=n("a"),_n=n("span"),T(Ct.$$.fragment),hi=p(),vn=n("span"),fi=a("ViltForImagesAndTextClassification"),ds=p(),Q=n("div"),T(At.$$.fragment),ui=p(),bn=n("p"),gi=a("Vilt Model transformer with a classifier head on top for natural language visual reasoning, e.g. NLVR2."),_i=p(),D=n("div"),T(qt.$$.fragment),vi=p(),he=n("p"),bi=a("The "),mo=n("a"),Ti=a("ViltForImagesAndTextClassification"),ki=a(" forward method, overrides the "),Tn=n("code"),wi=a("__call__"),xi=a(" special method."),yi=p(),T(Ne.$$.fragment),$i=p(),T(Se.$$.fragment),cs=p(),fe=n("h2"),De=n("a"),kn=n("span"),T(Ot.$$.fragment),Vi=p(),wn=n("span"),Fi=a("ViltForImageAndTextRetrieval"),ps=p(),C=n("div"),T(Nt.$$.fragment),Ei=p(),xn=n("p"),Mi=a(`Vilt Model transformer with a classifier head on top (a linear layer on top of the final hidden state of the [CLS]
token) for image-to-text or text-to-image retrieval, e.g. MSCOCO and F30K.`),Pi=p(),St=n("p"),ji=a("This model is a PyTorch "),yn=n("code"),Li=a("torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>"),zi=a(`_ subclass. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),Ii=p(),W=n("div"),T(Dt.$$.fragment),Ci=p(),ue=n("p"),Ai=a("The "),ho=n("a"),qi=a("ViltForImageAndTextRetrieval"),Oi=a(" forward method, overrides the "),$n=n("code"),Ni=a("__call__"),Si=a(" special method."),Di=p(),T(We.$$.fragment),Wi=p(),T(Re.$$.fragment),ms=p(),ge=n("h2"),Be=n("a"),Vn=n("span"),T(Wt.$$.fragment),Ri=p(),Fn=n("span"),Bi=a("ViltForTokenClassification"),hs=p(),A=n("div"),T(Rt.$$.fragment),Qi=p(),En=n("p"),Ui=a(`ViLT Model with a token classification head on top (a linear layer on top of the final hidden-states of the text
tokens) e.g. for Named-Entity-Recognition (NER) tasks.`),Hi=p(),Bt=n("p"),Ki=a("This model is a PyTorch "),Mn=n("code"),Gi=a("torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>"),Ji=a(`_ subclass. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),Xi=p(),X=n("div"),T(Qt.$$.fragment),Zi=p(),_e=n("p"),Yi=a("The "),fo=n("a"),el=a("ViltForTokenClassification"),tl=a(" forward method, overrides the "),Pn=n("code"),ol=a("__call__"),nl=a(" special method."),sl=p(),T(Qe.$$.fragment),this.h()},l(t){const f=Ld('[data-svelte="svelte-1phssyn"]',document.head);d=s(f,"META",{name:!0,content:!0}),f.forEach(o),b=m(t),g=s(t,"H1",{class:!0});var Ut=r(g);u=s(Ut,"A",{id:!0,class:!0,href:!0});var jn=r(u);v=s(jn,"SPAN",{});var Ln=r(v);k(l.$$.fragment,Ln),Ln.forEach(o),jn.forEach(o),h=m(Ut),F=s(Ut,"SPAN",{});var zn=r(F);Is=i(zn,"ViLT"),zn.forEach(o),Ut.forEach(o),Dn=m(t),Z=s(t,"H2",{class:!0});var Ht=r(Z);ve=s(Ht,"A",{id:!0,class:!0,href:!0});var In=r(ve);Io=s(In,"SPAN",{});var Cn=r(Io);k(at.$$.fragment,Cn),Cn.forEach(o),In.forEach(o),Cs=m(Ht),Co=s(Ht,"SPAN",{});var An=r(Co);As=i(An,"Overview"),An.forEach(o),Ht.forEach(o),Wn=m(t),be=s(t,"P",{});var Kt=r(be);qs=i(Kt,"The ViLT model was proposed in "),it=s(Kt,"A",{href:!0,rel:!0});var qn=r(it);Os=i(qn,"ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision"),qn.forEach(o),Ns=i(Kt,`
by Wonjae Kim, Bokyung Son, Ildoo Kim. ViLT incorporates text embeddings into a Vision Transformer (ViT), allowing it to have a minimal design
for Vision-and-Language Pre-training (VLP).`),Kt.forEach(o),Rn=m(t),Jt=s(t,"P",{});var On=r(Jt);Ss=i(On,"The abstract from the paper is the following:"),On.forEach(o),Bn=m(t),Xt=s(t,"P",{});var Nn=r(Xt);Ao=s(Nn,"EM",{});var Sn=r(Ao);Ds=i(Sn,`Vision-and-Language Pre-training (VLP) has improved performance on various joint vision-and-language downstream tasks.
Current approaches to VLP heavily rely on image feature extraction processes, most of which involve region supervision
(e.g., object detection) and the convolutional architecture (e.g., ResNet). Although disregarded in the literature, we
find it problematic in terms of both (1) efficiency/speed, that simply extracting input features requires much more
computation than the multimodal interaction steps; and (2) expressive power, as it is upper bounded to the expressive
power of the visual embedder and its predefined visual vocabulary. In this paper, we present a minimal VLP model,
Vision-and-Language Transformer (ViLT), monolithic in the sense that the processing of visual inputs is drastically
simplified to just the same convolution-free manner that we process textual inputs. We show that ViLT is up to tens of
times faster than previous VLP models, yet with competitive or better downstream task performance.`),Sn.forEach(o),Nn.forEach(o),Qn=m(t),Zt=s(t,"P",{});var il=r(Zt);Ws=i(il,"Tips:"),il.forEach(o),Un=m(t),q=s(t,"UL",{});var Ue=r(q);lt=s(Ue,"LI",{});var us=r(lt);Rs=i(us,"The quickest way to get started with ViLT is by checking the "),dt=s(us,"A",{href:!0,rel:!0});var ll=r(dt);Bs=i(ll,"example notebooks"),ll.forEach(o),Qs=i(us,`
(which showcase both inference and fine-tuning on custom data).`),us.forEach(o),Us=m(Ue),R=s(Ue,"LI",{});var He=r(R);Hs=i(He,"ViLT is a model that takes both "),qo=s(He,"CODE",{});var dl=r(qo);Ks=i(dl,"pixel_values"),dl.forEach(o),Gs=i(He," and "),Oo=s(He,"CODE",{});var cl=r(Oo);Js=i(cl,"input_ids"),cl.forEach(o),Xs=i(He," as input. One can use "),Yt=s(He,"A",{href:!0});var pl=r(Yt);Zs=i(pl,"ViltProcessor"),pl.forEach(o),Ys=i(He,` to prepare data for the model.
This processor wraps a feature extractor (for the image modality) and a tokenizer (for the language modality) into one.`),He.forEach(o),er=m(Ue),Y=s(Ue,"LI",{});var uo=r(Y);tr=i(uo,`ViLT is trained with images of various sizes: the authors resize the shorter edge of input images to 384 and limit the longer edge to
under 640 while preserving the aspect ratio. To make batching of images possible, the authors use a `),No=s(uo,"CODE",{});var ml=r(No);or=i(ml,"pixel_mask"),ml.forEach(o),nr=i(uo,` that indicates
which pixel values are real and which are padding. `),eo=s(uo,"A",{href:!0});var hl=r(eo);sr=i(hl,"ViltProcessor"),hl.forEach(o),rr=i(uo," automatically creates this for you."),uo.forEach(o),ar=m(Ue),So=s(Ue,"LI",{});var fl=r(So);ir=i(fl,`The design of ViLT is very similar to that of a standard Vision Transformer (ViT). The only difference is that the model includes
additional embedding layers for the language modality.`),fl.forEach(o),Ue.forEach(o),Hn=m(t),Te=s(t,"IMG",{src:!0,alt:!0,width:!0}),Kn=m(t),ke=s(t,"SMALL",{});var gs=r(ke);lr=i(gs,"ViLT architecture. Taken from the "),to=s(gs,"A",{href:!0});var ul=r(to);dr=i(ul,"original paper"),ul.forEach(o),cr=i(gs,"."),gs.forEach(o),Gn=m(t),K=s(t,"P",{});var go=r(K);pr=i(go,"This model was contributed by "),ct=s(go,"A",{href:!0,rel:!0});var gl=r(ct);mr=i(gl,"nielsr"),gl.forEach(o),hr=i(go,". The original code can be found "),pt=s(go,"A",{href:!0,rel:!0});var _l=r(pt);fr=i(_l,"here"),_l.forEach(o),ur=i(go,"."),go.forEach(o),Jn=m(t),ee=s(t,"H2",{class:!0});var _s=r(ee);we=s(_s,"A",{id:!0,class:!0,href:!0});var vl=r(we);Do=s(vl,"SPAN",{});var bl=r(Do);k(mt.$$.fragment,bl),bl.forEach(o),vl.forEach(o),gr=m(_s),Wo=s(_s,"SPAN",{});var Tl=r(Wo);_r=i(Tl,"ViltConfig"),Tl.forEach(o),_s.forEach(o),Xn=m(t),P=s(t,"DIV",{class:!0});var Ke=r(P);k(ht.$$.fragment,Ke),vr=m(Ke),te=s(Ke,"P",{});var _o=r(te);br=i(_o,"This is the configuration class to store the configuration of a "),Ro=s(_o,"CODE",{});var kl=r(Ro);Tr=i(kl,"ViLTModel"),kl.forEach(o),kr=i(_o,`. It is used to instantiate an ViLT
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the ViLT
`),ft=s(_o,"A",{href:!0,rel:!0});var wl=r(ft);wr=i(wl,"dandelin/vilt-b32-mlm"),wl.forEach(o),xr=i(_o," architecture."),_o.forEach(o),yr=m(Ke),oe=s(Ke,"P",{});var vo=r(oe);$r=i(vo,"Configuration objects inherit from "),oo=s(vo,"A",{href:!0});var xl=r(oo);Vr=i(xl,"PretrainedConfig"),xl.forEach(o),Fr=i(vo,` and can be used to control the model outputs. Read the
documentation from `),no=s(vo,"A",{href:!0});var yl=r(no);Er=i(yl,"PretrainedConfig"),yl.forEach(o),Mr=i(vo," for more information."),vo.forEach(o),Pr=m(Ke),k(xe.$$.fragment,Ke),Ke.forEach(o),Zn=m(t),ne=s(t,"H2",{class:!0});var vs=r(ne);ye=s(vs,"A",{id:!0,class:!0,href:!0});var $l=r(ye);Bo=s($l,"SPAN",{});var Vl=r(Bo);k(ut.$$.fragment,Vl),Vl.forEach(o),$l.forEach(o),jr=m(vs),Qo=s(vs,"SPAN",{});var Fl=r(Qo);Lr=i(Fl,"ViltFeatureExtractor"),Fl.forEach(o),vs.forEach(o),Yn=m(t),j=s(t,"DIV",{class:!0});var Ge=r(j);k(gt.$$.fragment,Ge),zr=m(Ge),Uo=s(Ge,"P",{});var El=r(Uo);Ir=i(El,"Constructs a ViLT feature extractor."),El.forEach(o),Cr=m(Ge),_t=s(Ge,"P",{});var bs=r(_t);Ar=i(bs,"This feature extractor inherits from "),so=s(bs,"A",{href:!0});var Ml=r(so);qr=i(Ml,"FeatureExtractionMixin"),Ml.forEach(o),Or=i(bs,` which contains most of the main methods. Users
should refer to this superclass for more information regarding those methods.`),bs.forEach(o),Nr=m(Ge),G=s(Ge,"DIV",{class:!0});var bo=r(G);k(vt.$$.fragment,bo),Sr=m(bo),Ho=s(bo,"P",{});var Pl=r(Ho);Dr=i(Pl,"Main method to prepare for the model one or several image(s)."),Pl.forEach(o),Wr=m(bo),k($e.$$.fragment,bo),bo.forEach(o),Ge.forEach(o),es=m(t),se=s(t,"H2",{class:!0});var Ts=r(se);Ve=s(Ts,"A",{id:!0,class:!0,href:!0});var jl=r(Ve);Ko=s(jl,"SPAN",{});var Ll=r(Ko);k(bt.$$.fragment,Ll),Ll.forEach(o),jl.forEach(o),Rr=m(Ts),Go=s(Ts,"SPAN",{});var zl=r(Go);Br=i(zl,"ViltProcessor"),zl.forEach(o),Ts.forEach(o),ts=m(t),L=s(t,"DIV",{class:!0});var Je=r(L);k(Tt.$$.fragment,Je),Qr=m(Je),Jo=s(Je,"P",{});var Il=r(Jo);Ur=i(Il,"Constructs a ViLT processor which wraps a BERT tokenizer and ViLT feature extractor into a single processor."),Il.forEach(o),Hr=m(Je),M=s(Je,"P",{});var U=r(M);ro=s(U,"A",{href:!0});var Cl=r(ro);Kr=i(Cl,"ViltProcessor"),Cl.forEach(o),Gr=i(U," offers all the functionalities of "),ao=s(U,"A",{href:!0});var Al=r(ao);Jr=i(Al,"ViltFeatureExtractor"),Al.forEach(o),Xr=i(U," and "),io=s(U,"A",{href:!0});var ql=r(io);Zr=i(ql,"BertTokenizerFast"),ql.forEach(o),Yr=i(U,`. See the
docstring of `),kt=s(U,"A",{href:!0});var rl=r(kt);Xo=s(rl,"STRONG",{});var Ol=r(Xo);ea=i(Ol,"call"),Ol.forEach(o),ta=i(rl,"()"),rl.forEach(o),oa=i(U," and "),Zo=s(U,"CODE",{});var Nl=r(Zo);na=i(Nl,"decode()"),Nl.forEach(o),sa=i(U," for more information."),U.forEach(o),ra=m(Je),J=s(Je,"DIV",{class:!0});var To=r(J);k(wt.$$.fragment,To),aa=m(To),re=s(To,"P",{});var ko=r(re);ia=i(ko,"This method uses "),Fe=s(ko,"A",{href:!0});var ks=r(Fe);la=i(ks,"ViltFeatureExtractor."),Yo=s(ks,"STRONG",{});var Sl=r(Yo);da=i(Sl,"call"),Sl.forEach(o),ca=i(ks,"()"),ks.forEach(o),pa=i(ko,` method to prepare image(s) for the model, and
`),Ee=s(ko,"A",{href:!0});var ws=r(Ee);ma=i(ws,"BertTokenizerFast."),en=s(ws,"STRONG",{});var Dl=r(en);ha=i(Dl,"call"),Dl.forEach(o),fa=i(ws,"()"),ws.forEach(o),ua=i(ko," to prepare text for the model."),ko.forEach(o),ga=m(To),tn=s(To,"P",{});var Wl=r(tn);_a=i(Wl,"Please refer to the docstring of the above two methods for more information."),Wl.forEach(o),To.forEach(o),Je.forEach(o),os=m(t),ae=s(t,"H2",{class:!0});var xs=r(ae);Me=s(xs,"A",{id:!0,class:!0,href:!0});var Rl=r(Me);on=s(Rl,"SPAN",{});var Bl=r(on);k(xt.$$.fragment,Bl),Bl.forEach(o),Rl.forEach(o),va=m(xs),nn=s(xs,"SPAN",{});var Ql=r(nn);ba=i(Ql,"ViltModel"),Ql.forEach(o),xs.forEach(o),ns=m(t),B=s(t,"DIV",{class:!0});var wo=r(B);k(yt.$$.fragment,wo),Ta=m(wo),$t=s(wo,"P",{});var ys=r($t);ka=i(ys,`The bare ViLT Model transformer outputting raw hidden-states without any specific head on top.
This model is a PyTorch `),sn=s(ys,"CODE",{});var Ul=r(sn);wa=i(Ul,"torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>"),Ul.forEach(o),xa=i(ys,`_ subclass. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),ys.forEach(o),ya=m(wo),O=s(wo,"DIV",{class:!0});var Xe=r(O);k(Vt.$$.fragment,Xe),$a=m(Xe),ie=s(Xe,"P",{});var xo=r(ie);Va=i(xo,"The "),lo=s(xo,"A",{href:!0});var Hl=r(lo);Fa=i(Hl,"ViltModel"),Hl.forEach(o),Ea=i(xo," forward method, overrides the "),rn=s(xo,"CODE",{});var Kl=r(rn);Ma=i(Kl,"__call__"),Kl.forEach(o),Pa=i(xo," special method."),xo.forEach(o),ja=m(Xe),k(Pe.$$.fragment,Xe),La=m(Xe),k(je.$$.fragment,Xe),Xe.forEach(o),wo.forEach(o),ss=m(t),le=s(t,"H2",{class:!0});var $s=r(le);Le=s($s,"A",{id:!0,class:!0,href:!0});var Gl=r(Le);an=s(Gl,"SPAN",{});var Jl=r(an);k(Ft.$$.fragment,Jl),Jl.forEach(o),Gl.forEach(o),za=m($s),ln=s($s,"SPAN",{});var Xl=r(ln);Ia=i(Xl,"ViltForMaskedLM"),Xl.forEach(o),$s.forEach(o),rs=m(t),z=s(t,"DIV",{class:!0});var Ze=r(z);k(Et.$$.fragment,Ze),Ca=m(Ze),dn=s(Ze,"P",{});var Zl=r(dn);Aa=i(Zl,"ViLT Model with a language modeling head on top as done during pretraining."),Zl.forEach(o),qa=m(Ze),Mt=s(Ze,"P",{});var Vs=r(Mt);Oa=i(Vs,"This model is a PyTorch "),cn=s(Vs,"CODE",{});var Yl=r(cn);Na=i(Yl,"torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>"),Yl.forEach(o),Sa=i(Vs,`_ subclass. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),Vs.forEach(o),Da=m(Ze),N=s(Ze,"DIV",{class:!0});var Ye=r(N);k(Pt.$$.fragment,Ye),Wa=m(Ye),de=s(Ye,"P",{});var yo=r(de);Ra=i(yo,"The "),co=s(yo,"A",{href:!0});var ed=r(co);Ba=i(ed,"ViltForMaskedLM"),ed.forEach(o),Qa=i(yo," forward method, overrides the "),pn=s(yo,"CODE",{});var td=r(pn);Ua=i(td,"__call__"),td.forEach(o),Ha=i(yo," special method."),yo.forEach(o),Ka=m(Ye),k(ze.$$.fragment,Ye),Ga=m(Ye),k(Ie.$$.fragment,Ye),Ye.forEach(o),Ze.forEach(o),as=m(t),ce=s(t,"H2",{class:!0});var Fs=r(ce);Ce=s(Fs,"A",{id:!0,class:!0,href:!0});var od=r(Ce);mn=s(od,"SPAN",{});var nd=r(mn);k(jt.$$.fragment,nd),nd.forEach(o),od.forEach(o),Ja=m(Fs),hn=s(Fs,"SPAN",{});var sd=r(hn);Xa=i(sd,"ViltForQuestionAnswering"),sd.forEach(o),Fs.forEach(o),is=m(t),I=s(t,"DIV",{class:!0});var et=r(I);k(Lt.$$.fragment,et),Za=m(et),fn=s(et,"P",{});var rd=r(fn);Ya=i(rd,`Vilt Model transformer with a classifier head on top (a linear layer on top of the final hidden state of the [CLS]
token) for visual question answering, e.g. for VQAv2.`),rd.forEach(o),ei=m(et),zt=s(et,"P",{});var Es=r(zt);ti=i(Es,"This model is a PyTorch "),un=s(Es,"CODE",{});var ad=r(un);oi=i(ad,"torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>"),ad.forEach(o),ni=i(Es,`_ subclass. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),Es.forEach(o),si=m(et),S=s(et,"DIV",{class:!0});var tt=r(S);k(It.$$.fragment,tt),ri=m(tt),pe=s(tt,"P",{});var $o=r(pe);ai=i($o,"The "),po=s($o,"A",{href:!0});var id=r(po);ii=i(id,"ViltForQuestionAnswering"),id.forEach(o),li=i($o," forward method, overrides the "),gn=s($o,"CODE",{});var ld=r(gn);di=i(ld,"__call__"),ld.forEach(o),ci=i($o," special method."),$o.forEach(o),pi=m(tt),k(Ae.$$.fragment,tt),mi=m(tt),k(qe.$$.fragment,tt),tt.forEach(o),et.forEach(o),ls=m(t),me=s(t,"H2",{class:!0});var Ms=r(me);Oe=s(Ms,"A",{id:!0,class:!0,href:!0});var dd=r(Oe);_n=s(dd,"SPAN",{});var cd=r(_n);k(Ct.$$.fragment,cd),cd.forEach(o),dd.forEach(o),hi=m(Ms),vn=s(Ms,"SPAN",{});var pd=r(vn);fi=i(pd,"ViltForImagesAndTextClassification"),pd.forEach(o),Ms.forEach(o),ds=m(t),Q=s(t,"DIV",{class:!0});var Vo=r(Q);k(At.$$.fragment,Vo),ui=m(Vo),bn=s(Vo,"P",{});var md=r(bn);gi=i(md,"Vilt Model transformer with a classifier head on top for natural language visual reasoning, e.g. NLVR2."),md.forEach(o),_i=m(Vo),D=s(Vo,"DIV",{class:!0});var ot=r(D);k(qt.$$.fragment,ot),vi=m(ot),he=s(ot,"P",{});var Fo=r(he);bi=i(Fo,"The "),mo=s(Fo,"A",{href:!0});var hd=r(mo);Ti=i(hd,"ViltForImagesAndTextClassification"),hd.forEach(o),ki=i(Fo," forward method, overrides the "),Tn=s(Fo,"CODE",{});var fd=r(Tn);wi=i(fd,"__call__"),fd.forEach(o),xi=i(Fo," special method."),Fo.forEach(o),yi=m(ot),k(Ne.$$.fragment,ot),$i=m(ot),k(Se.$$.fragment,ot),ot.forEach(o),Vo.forEach(o),cs=m(t),fe=s(t,"H2",{class:!0});var Ps=r(fe);De=s(Ps,"A",{id:!0,class:!0,href:!0});var ud=r(De);kn=s(ud,"SPAN",{});var gd=r(kn);k(Ot.$$.fragment,gd),gd.forEach(o),ud.forEach(o),Vi=m(Ps),wn=s(Ps,"SPAN",{});var _d=r(wn);Fi=i(_d,"ViltForImageAndTextRetrieval"),_d.forEach(o),Ps.forEach(o),ps=m(t),C=s(t,"DIV",{class:!0});var nt=r(C);k(Nt.$$.fragment,nt),Ei=m(nt),xn=s(nt,"P",{});var vd=r(xn);Mi=i(vd,`Vilt Model transformer with a classifier head on top (a linear layer on top of the final hidden state of the [CLS]
token) for image-to-text or text-to-image retrieval, e.g. MSCOCO and F30K.`),vd.forEach(o),Pi=m(nt),St=s(nt,"P",{});var js=r(St);ji=i(js,"This model is a PyTorch "),yn=s(js,"CODE",{});var bd=r(yn);Li=i(bd,"torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>"),bd.forEach(o),zi=i(js,`_ subclass. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),js.forEach(o),Ii=m(nt),W=s(nt,"DIV",{class:!0});var st=r(W);k(Dt.$$.fragment,st),Ci=m(st),ue=s(st,"P",{});var Eo=r(ue);Ai=i(Eo,"The "),ho=s(Eo,"A",{href:!0});var Td=r(ho);qi=i(Td,"ViltForImageAndTextRetrieval"),Td.forEach(o),Oi=i(Eo," forward method, overrides the "),$n=s(Eo,"CODE",{});var kd=r($n);Ni=i(kd,"__call__"),kd.forEach(o),Si=i(Eo," special method."),Eo.forEach(o),Di=m(st),k(We.$$.fragment,st),Wi=m(st),k(Re.$$.fragment,st),st.forEach(o),nt.forEach(o),ms=m(t),ge=s(t,"H2",{class:!0});var Ls=r(ge);Be=s(Ls,"A",{id:!0,class:!0,href:!0});var wd=r(Be);Vn=s(wd,"SPAN",{});var xd=r(Vn);k(Wt.$$.fragment,xd),xd.forEach(o),wd.forEach(o),Ri=m(Ls),Fn=s(Ls,"SPAN",{});var yd=r(Fn);Bi=i(yd,"ViltForTokenClassification"),yd.forEach(o),Ls.forEach(o),hs=m(t),A=s(t,"DIV",{class:!0});var rt=r(A);k(Rt.$$.fragment,rt),Qi=m(rt),En=s(rt,"P",{});var $d=r(En);Ui=i($d,`ViLT Model with a token classification head on top (a linear layer on top of the final hidden-states of the text
tokens) e.g. for Named-Entity-Recognition (NER) tasks.`),$d.forEach(o),Hi=m(rt),Bt=s(rt,"P",{});var zs=r(Bt);Ki=i(zs,"This model is a PyTorch "),Mn=s(zs,"CODE",{});var Vd=r(Mn);Gi=i(Vd,"torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>"),Vd.forEach(o),Ji=i(zs,`_ subclass. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),zs.forEach(o),Xi=m(rt),X=s(rt,"DIV",{class:!0});var Mo=r(X);k(Qt.$$.fragment,Mo),Zi=m(Mo),_e=s(Mo,"P",{});var Po=r(_e);Yi=i(Po,"The "),fo=s(Po,"A",{href:!0});var Fd=r(fo);el=i(Fd,"ViltForTokenClassification"),Fd.forEach(o),tl=i(Po," forward method, overrides the "),Pn=s(Po,"CODE",{});var Ed=r(Pn);ol=i(Ed,"__call__"),Ed.forEach(o),nl=i(Po," special method."),Po.forEach(o),sl=m(Mo),k(Qe.$$.fragment,Mo),Mo.forEach(o),rt.forEach(o),this.h()},h(){c(d,"name","hf:doc:metadata"),c(d,"content",JSON.stringify(Gd)),c(u,"id","vilt"),c(u,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(u,"href","#vilt"),c(g,"class","relative group"),c(ve,"id","overview"),c(ve,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(ve,"href","#overview"),c(Z,"class","relative group"),c(it,"href","https://arxiv.org/abs/2102.03334"),c(it,"rel","nofollow"),c(dt,"href","https://github.com/NielsRogge/Transformers-Tutorials/tree/master/ViLT"),c(dt,"rel","nofollow"),c(Yt,"href","/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltProcessor"),c(eo,"href","/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltProcessor"),zd(Te.src,al="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vilt_architecture.jpg")||c(Te,"src",al),c(Te,"alt","drawing"),c(Te,"width","600"),c(to,"href","https://arxiv.org/abs/2102.03334"),c(ct,"href","https://huggingface.co/nielsr"),c(ct,"rel","nofollow"),c(pt,"href","https://github.com/dandelin/ViLT"),c(pt,"rel","nofollow"),c(we,"id","transformers.ViltConfig"),c(we,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(we,"href","#transformers.ViltConfig"),c(ee,"class","relative group"),c(ft,"href","https://huggingface.co/dandelin/vilt-b32-mlm"),c(ft,"rel","nofollow"),c(oo,"href","/docs/transformers/pr_5/en/main_classes/configuration#transformers.PretrainedConfig"),c(no,"href","/docs/transformers/pr_5/en/main_classes/configuration#transformers.PretrainedConfig"),c(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(ye,"id","transformers.ViltFeatureExtractor"),c(ye,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(ye,"href","#transformers.ViltFeatureExtractor"),c(ne,"class","relative group"),c(so,"href","/docs/transformers/pr_5/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin"),c(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(Ve,"id","transformers.ViltProcessor"),c(Ve,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(Ve,"href","#transformers.ViltProcessor"),c(se,"class","relative group"),c(ro,"href","/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltProcessor"),c(ao,"href","/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltFeatureExtractor"),c(io,"href","/docs/transformers/pr_5/en/model_doc/bert#transformers.BertTokenizerFast"),c(kt,"href","/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltProcessor.__call__"),c(Fe,"href","/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltFeatureExtractor.__call__"),c(Ee,"href","/docs/transformers/pr_5/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__"),c(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(Me,"id","transformers.ViltModel"),c(Me,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(Me,"href","#transformers.ViltModel"),c(ae,"class","relative group"),c(lo,"href","/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltModel"),c(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(Le,"id","transformers.ViltForMaskedLM"),c(Le,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(Le,"href","#transformers.ViltForMaskedLM"),c(le,"class","relative group"),c(co,"href","/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltForMaskedLM"),c(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(Ce,"id","transformers.ViltForQuestionAnswering"),c(Ce,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(Ce,"href","#transformers.ViltForQuestionAnswering"),c(ce,"class","relative group"),c(po,"href","/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltForQuestionAnswering"),c(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(Oe,"id","transformers.ViltForImagesAndTextClassification"),c(Oe,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(Oe,"href","#transformers.ViltForImagesAndTextClassification"),c(me,"class","relative group"),c(mo,"href","/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltForImagesAndTextClassification"),c(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(De,"id","transformers.ViltForImageAndTextRetrieval"),c(De,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(De,"href","#transformers.ViltForImageAndTextRetrieval"),c(fe,"class","relative group"),c(ho,"href","/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltForImageAndTextRetrieval"),c(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(Be,"id","transformers.ViltForTokenClassification"),c(Be,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(Be,"href","#transformers.ViltForTokenClassification"),c(ge,"class","relative group"),c(fo,"href","/docs/transformers/pr_5/en/model_doc/vilt#transformers.ViltForTokenClassification"),c(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(t,f){e(document.head,d),_(t,b,f),_(t,g,f),e(g,u),e(u,v),w(l,v,null),e(g,h),e(g,F),e(F,Is),_(t,Dn,f),_(t,Z,f),e(Z,ve),e(ve,Io),w(at,Io,null),e(Z,Cs),e(Z,Co),e(Co,As),_(t,Wn,f),_(t,be,f),e(be,qs),e(be,it),e(it,Os),e(be,Ns),_(t,Rn,f),_(t,Jt,f),e(Jt,Ss),_(t,Bn,f),_(t,Xt,f),e(Xt,Ao),e(Ao,Ds),_(t,Qn,f),_(t,Zt,f),e(Zt,Ws),_(t,Un,f),_(t,q,f),e(q,lt),e(lt,Rs),e(lt,dt),e(dt,Bs),e(lt,Qs),e(q,Us),e(q,R),e(R,Hs),e(R,qo),e(qo,Ks),e(R,Gs),e(R,Oo),e(Oo,Js),e(R,Xs),e(R,Yt),e(Yt,Zs),e(R,Ys),e(q,er),e(q,Y),e(Y,tr),e(Y,No),e(No,or),e(Y,nr),e(Y,eo),e(eo,sr),e(Y,rr),e(q,ar),e(q,So),e(So,ir),_(t,Hn,f),_(t,Te,f),_(t,Kn,f),_(t,ke,f),e(ke,lr),e(ke,to),e(to,dr),e(ke,cr),_(t,Gn,f),_(t,K,f),e(K,pr),e(K,ct),e(ct,mr),e(K,hr),e(K,pt),e(pt,fr),e(K,ur),_(t,Jn,f),_(t,ee,f),e(ee,we),e(we,Do),w(mt,Do,null),e(ee,gr),e(ee,Wo),e(Wo,_r),_(t,Xn,f),_(t,P,f),w(ht,P,null),e(P,vr),e(P,te),e(te,br),e(te,Ro),e(Ro,Tr),e(te,kr),e(te,ft),e(ft,wr),e(te,xr),e(P,yr),e(P,oe),e(oe,$r),e(oe,oo),e(oo,Vr),e(oe,Fr),e(oe,no),e(no,Er),e(oe,Mr),e(P,Pr),w(xe,P,null),_(t,Zn,f),_(t,ne,f),e(ne,ye),e(ye,Bo),w(ut,Bo,null),e(ne,jr),e(ne,Qo),e(Qo,Lr),_(t,Yn,f),_(t,j,f),w(gt,j,null),e(j,zr),e(j,Uo),e(Uo,Ir),e(j,Cr),e(j,_t),e(_t,Ar),e(_t,so),e(so,qr),e(_t,Or),e(j,Nr),e(j,G),w(vt,G,null),e(G,Sr),e(G,Ho),e(Ho,Dr),e(G,Wr),w($e,G,null),_(t,es,f),_(t,se,f),e(se,Ve),e(Ve,Ko),w(bt,Ko,null),e(se,Rr),e(se,Go),e(Go,Br),_(t,ts,f),_(t,L,f),w(Tt,L,null),e(L,Qr),e(L,Jo),e(Jo,Ur),e(L,Hr),e(L,M),e(M,ro),e(ro,Kr),e(M,Gr),e(M,ao),e(ao,Jr),e(M,Xr),e(M,io),e(io,Zr),e(M,Yr),e(M,kt),e(kt,Xo),e(Xo,ea),e(kt,ta),e(M,oa),e(M,Zo),e(Zo,na),e(M,sa),e(L,ra),e(L,J),w(wt,J,null),e(J,aa),e(J,re),e(re,ia),e(re,Fe),e(Fe,la),e(Fe,Yo),e(Yo,da),e(Fe,ca),e(re,pa),e(re,Ee),e(Ee,ma),e(Ee,en),e(en,ha),e(Ee,fa),e(re,ua),e(J,ga),e(J,tn),e(tn,_a),_(t,os,f),_(t,ae,f),e(ae,Me),e(Me,on),w(xt,on,null),e(ae,va),e(ae,nn),e(nn,ba),_(t,ns,f),_(t,B,f),w(yt,B,null),e(B,Ta),e(B,$t),e($t,ka),e($t,sn),e(sn,wa),e($t,xa),e(B,ya),e(B,O),w(Vt,O,null),e(O,$a),e(O,ie),e(ie,Va),e(ie,lo),e(lo,Fa),e(ie,Ea),e(ie,rn),e(rn,Ma),e(ie,Pa),e(O,ja),w(Pe,O,null),e(O,La),w(je,O,null),_(t,ss,f),_(t,le,f),e(le,Le),e(Le,an),w(Ft,an,null),e(le,za),e(le,ln),e(ln,Ia),_(t,rs,f),_(t,z,f),w(Et,z,null),e(z,Ca),e(z,dn),e(dn,Aa),e(z,qa),e(z,Mt),e(Mt,Oa),e(Mt,cn),e(cn,Na),e(Mt,Sa),e(z,Da),e(z,N),w(Pt,N,null),e(N,Wa),e(N,de),e(de,Ra),e(de,co),e(co,Ba),e(de,Qa),e(de,pn),e(pn,Ua),e(de,Ha),e(N,Ka),w(ze,N,null),e(N,Ga),w(Ie,N,null),_(t,as,f),_(t,ce,f),e(ce,Ce),e(Ce,mn),w(jt,mn,null),e(ce,Ja),e(ce,hn),e(hn,Xa),_(t,is,f),_(t,I,f),w(Lt,I,null),e(I,Za),e(I,fn),e(fn,Ya),e(I,ei),e(I,zt),e(zt,ti),e(zt,un),e(un,oi),e(zt,ni),e(I,si),e(I,S),w(It,S,null),e(S,ri),e(S,pe),e(pe,ai),e(pe,po),e(po,ii),e(pe,li),e(pe,gn),e(gn,di),e(pe,ci),e(S,pi),w(Ae,S,null),e(S,mi),w(qe,S,null),_(t,ls,f),_(t,me,f),e(me,Oe),e(Oe,_n),w(Ct,_n,null),e(me,hi),e(me,vn),e(vn,fi),_(t,ds,f),_(t,Q,f),w(At,Q,null),e(Q,ui),e(Q,bn),e(bn,gi),e(Q,_i),e(Q,D),w(qt,D,null),e(D,vi),e(D,he),e(he,bi),e(he,mo),e(mo,Ti),e(he,ki),e(he,Tn),e(Tn,wi),e(he,xi),e(D,yi),w(Ne,D,null),e(D,$i),w(Se,D,null),_(t,cs,f),_(t,fe,f),e(fe,De),e(De,kn),w(Ot,kn,null),e(fe,Vi),e(fe,wn),e(wn,Fi),_(t,ps,f),_(t,C,f),w(Nt,C,null),e(C,Ei),e(C,xn),e(xn,Mi),e(C,Pi),e(C,St),e(St,ji),e(St,yn),e(yn,Li),e(St,zi),e(C,Ii),e(C,W),w(Dt,W,null),e(W,Ci),e(W,ue),e(ue,Ai),e(ue,ho),e(ho,qi),e(ue,Oi),e(ue,$n),e($n,Ni),e(ue,Si),e(W,Di),w(We,W,null),e(W,Wi),w(Re,W,null),_(t,ms,f),_(t,ge,f),e(ge,Be),e(Be,Vn),w(Wt,Vn,null),e(ge,Ri),e(ge,Fn),e(Fn,Bi),_(t,hs,f),_(t,A,f),w(Rt,A,null),e(A,Qi),e(A,En),e(En,Ui),e(A,Hi),e(A,Bt),e(Bt,Ki),e(Bt,Mn),e(Mn,Gi),e(Bt,Ji),e(A,Xi),e(A,X),w(Qt,X,null),e(X,Zi),e(X,_e),e(_e,Yi),e(_e,fo),e(fo,el),e(_e,tl),e(_e,Pn),e(Pn,ol),e(_e,nl),e(X,sl),w(Qe,X,null),fs=!0},p(t,[f]){const Ut={};f&2&&(Ut.$$scope={dirty:f,ctx:t}),xe.$set(Ut);const jn={};f&2&&(jn.$$scope={dirty:f,ctx:t}),$e.$set(jn);const Ln={};f&2&&(Ln.$$scope={dirty:f,ctx:t}),Pe.$set(Ln);const zn={};f&2&&(zn.$$scope={dirty:f,ctx:t}),je.$set(zn);const Ht={};f&2&&(Ht.$$scope={dirty:f,ctx:t}),ze.$set(Ht);const In={};f&2&&(In.$$scope={dirty:f,ctx:t}),Ie.$set(In);const Cn={};f&2&&(Cn.$$scope={dirty:f,ctx:t}),Ae.$set(Cn);const An={};f&2&&(An.$$scope={dirty:f,ctx:t}),qe.$set(An);const Kt={};f&2&&(Kt.$$scope={dirty:f,ctx:t}),Ne.$set(Kt);const qn={};f&2&&(qn.$$scope={dirty:f,ctx:t}),Se.$set(qn);const On={};f&2&&(On.$$scope={dirty:f,ctx:t}),We.$set(On);const Nn={};f&2&&(Nn.$$scope={dirty:f,ctx:t}),Re.$set(Nn);const Sn={};f&2&&(Sn.$$scope={dirty:f,ctx:t}),Qe.$set(Sn)},i(t){fs||(x(l.$$.fragment,t),x(at.$$.fragment,t),x(mt.$$.fragment,t),x(ht.$$.fragment,t),x(xe.$$.fragment,t),x(ut.$$.fragment,t),x(gt.$$.fragment,t),x(vt.$$.fragment,t),x($e.$$.fragment,t),x(bt.$$.fragment,t),x(Tt.$$.fragment,t),x(wt.$$.fragment,t),x(xt.$$.fragment,t),x(yt.$$.fragment,t),x(Vt.$$.fragment,t),x(Pe.$$.fragment,t),x(je.$$.fragment,t),x(Ft.$$.fragment,t),x(Et.$$.fragment,t),x(Pt.$$.fragment,t),x(ze.$$.fragment,t),x(Ie.$$.fragment,t),x(jt.$$.fragment,t),x(Lt.$$.fragment,t),x(It.$$.fragment,t),x(Ae.$$.fragment,t),x(qe.$$.fragment,t),x(Ct.$$.fragment,t),x(At.$$.fragment,t),x(qt.$$.fragment,t),x(Ne.$$.fragment,t),x(Se.$$.fragment,t),x(Ot.$$.fragment,t),x(Nt.$$.fragment,t),x(Dt.$$.fragment,t),x(We.$$.fragment,t),x(Re.$$.fragment,t),x(Wt.$$.fragment,t),x(Rt.$$.fragment,t),x(Qt.$$.fragment,t),x(Qe.$$.fragment,t),fs=!0)},o(t){y(l.$$.fragment,t),y(at.$$.fragment,t),y(mt.$$.fragment,t),y(ht.$$.fragment,t),y(xe.$$.fragment,t),y(ut.$$.fragment,t),y(gt.$$.fragment,t),y(vt.$$.fragment,t),y($e.$$.fragment,t),y(bt.$$.fragment,t),y(Tt.$$.fragment,t),y(wt.$$.fragment,t),y(xt.$$.fragment,t),y(yt.$$.fragment,t),y(Vt.$$.fragment,t),y(Pe.$$.fragment,t),y(je.$$.fragment,t),y(Ft.$$.fragment,t),y(Et.$$.fragment,t),y(Pt.$$.fragment,t),y(ze.$$.fragment,t),y(Ie.$$.fragment,t),y(jt.$$.fragment,t),y(Lt.$$.fragment,t),y(It.$$.fragment,t),y(Ae.$$.fragment,t),y(qe.$$.fragment,t),y(Ct.$$.fragment,t),y(At.$$.fragment,t),y(qt.$$.fragment,t),y(Ne.$$.fragment,t),y(Se.$$.fragment,t),y(Ot.$$.fragment,t),y(Nt.$$.fragment,t),y(Dt.$$.fragment,t),y(We.$$.fragment,t),y(Re.$$.fragment,t),y(Wt.$$.fragment,t),y(Rt.$$.fragment,t),y(Qt.$$.fragment,t),y(Qe.$$.fragment,t),fs=!1},d(t){o(d),t&&o(b),t&&o(g),$(l),t&&o(Dn),t&&o(Z),$(at),t&&o(Wn),t&&o(be),t&&o(Rn),t&&o(Jt),t&&o(Bn),t&&o(Xt),t&&o(Qn),t&&o(Zt),t&&o(Un),t&&o(q),t&&o(Hn),t&&o(Te),t&&o(Kn),t&&o(ke),t&&o(Gn),t&&o(K),t&&o(Jn),t&&o(ee),$(mt),t&&o(Xn),t&&o(P),$(ht),$(xe),t&&o(Zn),t&&o(ne),$(ut),t&&o(Yn),t&&o(j),$(gt),$(vt),$($e),t&&o(es),t&&o(se),$(bt),t&&o(ts),t&&o(L),$(Tt),$(wt),t&&o(os),t&&o(ae),$(xt),t&&o(ns),t&&o(B),$(yt),$(Vt),$(Pe),$(je),t&&o(ss),t&&o(le),$(Ft),t&&o(rs),t&&o(z),$(Et),$(Pt),$(ze),$(Ie),t&&o(as),t&&o(ce),$(jt),t&&o(is),t&&o(I),$(Lt),$(It),$(Ae),$(qe),t&&o(ls),t&&o(me),$(Ct),t&&o(ds),t&&o(Q),$(At),$(qt),$(Ne),$(Se),t&&o(cs),t&&o(fe),$(Ot),t&&o(ps),t&&o(C),$(Nt),$(Dt),$(We),$(Re),t&&o(ms),t&&o(ge),$(Wt),t&&o(hs),t&&o(A),$(Rt),$(Qt),$(Qe)}}}const Gd={local:"vilt",sections:[{local:"overview",title:"Overview"},{local:"transformers.ViltConfig",title:"ViltConfig"},{local:"transformers.ViltFeatureExtractor",title:"ViltFeatureExtractor"},{local:"transformers.ViltProcessor",title:"ViltProcessor"},{local:"transformers.ViltModel",title:"ViltModel"},{local:"transformers.ViltForMaskedLM",title:"ViltForMaskedLM"},{local:"transformers.ViltForQuestionAnswering",title:"ViltForQuestionAnswering"},{local:"transformers.ViltForImagesAndTextClassification",title:"ViltForImagesAndTextClassification"},{local:"transformers.ViltForImageAndTextRetrieval",title:"ViltForImageAndTextRetrieval"},{local:"transformers.ViltForTokenClassification",title:"ViltForTokenClassification"}],title:"ViLT"};function Jd(V){return Id(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class nc extends Md{constructor(d){super();Pd(this,d,Jd,Kd,jd,{})}}export{nc as default,Gd as metadata};
