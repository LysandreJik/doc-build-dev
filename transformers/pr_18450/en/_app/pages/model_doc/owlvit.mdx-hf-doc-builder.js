import{S as mh,i as hh,s as fh,e as n,k as f,w as v,t as i,M as uh,c as r,d as o,m as u,a,x as b,h as l,b as h,G as e,g as T,y as $,q as O,o as y,B as V,v as gh,L as B}from"../../chunks/vendor-hf-doc-builder.js";import{T as W}from"../../chunks/Tip-hf-doc-builder.js";import{D as j}from"../../chunks/Docstring-hf-doc-builder.js";import{C as A}from"../../chunks/CodeBlock-hf-doc-builder.js";import{I as z}from"../../chunks/IconCopyLink-hf-doc-builder.js";import{E as S}from"../../chunks/ExampleCodeBlock-hf-doc-builder.js";function _h(x){let d,w,m,p,_;return p=new A({props:{code:`from transformers import OwlViTTextConfig, OwlViTTextModel

# Initializing a OwlViTTextModel with google/owlvit-base-patch32 style configuration
configuration = OwlViTTextConfig()

# Initializing a OwlViTTextConfig from the google/owlvit-base-patch32 style configuration
model = OwlViTTextModel(configuration)

# Accessing the model configuration
configuration = model.config`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> OwlViTTextConfig, OwlViTTextModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a OwlViTTextModel with google/owlvit-base-patch32 style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = OwlViTTextConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a OwlViTTextConfig from the google/owlvit-base-patch32 style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = OwlViTTextModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`}}),{c(){d=n("p"),w=i("Example:"),m=f(),v(p.$$.fragment)},l(s){d=r(s,"P",{});var c=a(d);w=l(c,"Example:"),c.forEach(o),m=u(s),b(p.$$.fragment,s)},m(s,c){T(s,d,c),e(d,w),T(s,m,c),$(p,s,c),_=!0},p:B,i(s){_||(O(p.$$.fragment,s),_=!0)},o(s){y(p.$$.fragment,s),_=!1},d(s){s&&o(d),s&&o(m),V(p,s)}}}function wh(x){let d,w,m,p,_;return p=new A({props:{code:`from transformers import OwlViTVisionConfig, OwlViTVisionModel

# Initializing a OwlViTVisionModel with google/owlvit-base-patch32 style configuration
configuration = OwlViTVisionConfig()

# Initializing a OwlViTVisionModel model from the google/owlvit-base-patch32 style configuration
model = OwlViTVisionModel(configuration)

# Accessing the model configuration
configuration = model.config`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> OwlViTVisionConfig, OwlViTVisionModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a OwlViTVisionModel with google/owlvit-base-patch32 style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = OwlViTVisionConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a OwlViTVisionModel model from the google/owlvit-base-patch32 style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = OwlViTVisionModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`}}),{c(){d=n("p"),w=i("Example:"),m=f(),v(p.$$.fragment)},l(s){d=r(s,"P",{});var c=a(d);w=l(c,"Example:"),c.forEach(o),m=u(s),b(p.$$.fragment,s)},m(s,c){T(s,d,c),e(d,w),T(s,m,c),$(p,s,c),_=!0},p:B,i(s){_||(O(p.$$.fragment,s),_=!0)},o(s){y(p.$$.fragment,s),_=!1},d(s){s&&o(d),s&&o(m),V(p,s)}}}function Th(x){let d,w;return{c(){d=n("p"),w=i(`NumPy arrays and PyTorch tensors are converted to PIL images when resizing, so the most efficient is to pass
PIL images.`)},l(m){d=r(m,"P",{});var p=a(d);w=l(p,`NumPy arrays and PyTorch tensors are converted to PIL images when resizing, so the most efficient is to pass
PIL images.`),p.forEach(o)},m(m,p){T(m,d,p),e(d,w)},d(m){m&&o(d)}}}function vh(x){let d,w,m,p,_;return{c(){d=n("p"),w=i("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=n("code"),p=i("Module"),_=i(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(s){d=r(s,"P",{});var c=a(d);w=l(c,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r(c,"CODE",{});var k=a(m);p=l(k,"Module"),k.forEach(o),_=l(c,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),c.forEach(o)},m(s,c){T(s,d,c),e(d,w),e(d,m),e(m,p),e(d,_)},d(s){s&&o(d)}}}function bh(x){let d,w,m,p,_;return p=new A({props:{code:`from PIL import Image
import requests
from transformers import OwlViTProcessor, OwlViTModel

model = OwlViTModel.from_pretrained("google/owlvit-base-patch32")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(text=[["a photo of a cat", "a photo of a dog"]], images=image, return_tensors="pt")
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> OwlViTProcessor, OwlViTModel

<span class="hljs-meta">&gt;&gt;&gt; </span>model = OwlViTModel.from_pretrained(<span class="hljs-string">&quot;google/owlvit-base-patch32&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor = OwlViTProcessor.from_pretrained(<span class="hljs-string">&quot;google/owlvit-base-patch32&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(text=[[<span class="hljs-string">&quot;a photo of a cat&quot;</span>, <span class="hljs-string">&quot;a photo of a dog&quot;</span>]], images=image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>logits_per_image = outputs.logits_per_image  <span class="hljs-comment"># this is the image-text similarity score</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>probs = logits_per_image.softmax(dim=<span class="hljs-number">1</span>)  <span class="hljs-comment"># we can take the softmax to get the label probabilities</span>

`}}),{c(){d=n("p"),w=i("Examples:"),m=f(),v(p.$$.fragment)},l(s){d=r(s,"P",{});var c=a(d);w=l(c,"Examples:"),c.forEach(o),m=u(s),b(p.$$.fragment,s)},m(s,c){T(s,d,c),e(d,w),T(s,m,c),$(p,s,c),_=!0},p:B,i(s){_||(O(p.$$.fragment,s),_=!0)},o(s){y(p.$$.fragment,s),_=!1},d(s){s&&o(d),s&&o(m),V(p,s)}}}function $h(x){let d,w,m,p,_;return{c(){d=n("p"),w=i("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=n("code"),p=i("Module"),_=i(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(s){d=r(s,"P",{});var c=a(d);w=l(c,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r(c,"CODE",{});var k=a(m);p=l(k,"Module"),k.forEach(o),_=l(c,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),c.forEach(o)},m(s,c){T(s,d,c),e(d,w),e(d,m),e(m,p),e(d,_)},d(s){s&&o(d)}}}function Oh(x){let d,w,m,p,_;return p=new A({props:{code:`from transformers import OwlViTProcessor, OwlViTModel

model = OwlViTModel.from_pretrained("google/owlvit-base-patch32")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
inputs = processor(
    text=[["a photo of a cat", "a photo of a dog"], ["photo of a astranaut"]], return_tensors="pt"
)
text_features = model.get_text_features(**inputs)

`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> OwlViTProcessor, OwlViTModel

<span class="hljs-meta">&gt;&gt;&gt; </span>model = OwlViTModel.from_pretrained(<span class="hljs-string">&quot;google/owlvit-base-patch32&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor = OwlViTProcessor.from_pretrained(<span class="hljs-string">&quot;google/owlvit-base-patch32&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(
<span class="hljs-meta">... </span>    text=[[<span class="hljs-string">&quot;a photo of a cat&quot;</span>, <span class="hljs-string">&quot;a photo of a dog&quot;</span>], [<span class="hljs-string">&quot;photo of a astranaut&quot;</span>]], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>text_features = model.get_text_features(**inputs)

`}}),{c(){d=n("p"),w=i("Examples:"),m=f(),v(p.$$.fragment)},l(s){d=r(s,"P",{});var c=a(d);w=l(c,"Examples:"),c.forEach(o),m=u(s),b(p.$$.fragment,s)},m(s,c){T(s,d,c),e(d,w),T(s,m,c),$(p,s,c),_=!0},p:B,i(s){_||(O(p.$$.fragment,s),_=!0)},o(s){y(p.$$.fragment,s),_=!1},d(s){s&&o(d),s&&o(m),V(p,s)}}}function yh(x){let d,w,m,p,_;return{c(){d=n("p"),w=i("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=n("code"),p=i("Module"),_=i(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(s){d=r(s,"P",{});var c=a(d);w=l(c,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r(c,"CODE",{});var k=a(m);p=l(k,"Module"),k.forEach(o),_=l(c,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),c.forEach(o)},m(s,c){T(s,d,c),e(d,w),e(d,m),e(m,p),e(d,_)},d(s){s&&o(d)}}}function Vh(x){let d,w,m,p,_;return p=new A({props:{code:`from PIL import Image
import requests
from transformers import OwlViTProcessor, OwlViTModel

model = OwlViTModel.from_pretrained("google/owlvit-base-patch32")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt")
image_features = model.get_image_features(**inputs)

`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> OwlViTProcessor, OwlViTModel

<span class="hljs-meta">&gt;&gt;&gt; </span>model = OwlViTModel.from_pretrained(<span class="hljs-string">&quot;google/owlvit-base-patch32&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor = OwlViTProcessor.from_pretrained(<span class="hljs-string">&quot;google/owlvit-base-patch32&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(images=image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>image_features = model.get_image_features(**inputs)

`}}),{c(){d=n("p"),w=i("Examples:"),m=f(),v(p.$$.fragment)},l(s){d=r(s,"P",{});var c=a(d);w=l(c,"Examples:"),c.forEach(o),m=u(s),b(p.$$.fragment,s)},m(s,c){T(s,d,c),e(d,w),T(s,m,c),$(p,s,c),_=!0},p:B,i(s){_||(O(p.$$.fragment,s),_=!0)},o(s){y(p.$$.fragment,s),_=!1},d(s){s&&o(d),s&&o(m),V(p,s)}}}function xh(x){let d,w,m,p,_;return{c(){d=n("p"),w=i("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=n("code"),p=i("Module"),_=i(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(s){d=r(s,"P",{});var c=a(d);w=l(c,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r(c,"CODE",{});var k=a(m);p=l(k,"Module"),k.forEach(o),_=l(c,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),c.forEach(o)},m(s,c){T(s,d,c),e(d,w),e(d,m),e(m,p),e(d,_)},d(s){s&&o(d)}}}function kh(x){let d,w,m,p,_;return p=new A({props:{code:`from transformers import OwlViTProcessor, OwlViTTextModel

model = OwlViTTextModel.from_pretrained("google/owlvit-base-patch32")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
inputs = processor(
    text=[["a photo of a cat", "a photo of a dog"], ["photo of a astranaut"]], return_tensors="pt"
)
outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
pooled_output = outputs.pooler_output  # pooled (EOS token) states

`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> OwlViTProcessor, OwlViTTextModel

<span class="hljs-meta">&gt;&gt;&gt; </span>model = OwlViTTextModel.from_pretrained(<span class="hljs-string">&quot;google/owlvit-base-patch32&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor = OwlViTProcessor.from_pretrained(<span class="hljs-string">&quot;google/owlvit-base-patch32&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(
<span class="hljs-meta">... </span>    text=[[<span class="hljs-string">&quot;a photo of a cat&quot;</span>, <span class="hljs-string">&quot;a photo of a dog&quot;</span>], [<span class="hljs-string">&quot;photo of a astranaut&quot;</span>]], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_state = outputs.last_hidden_state
<span class="hljs-meta">&gt;&gt;&gt; </span>pooled_output = outputs.pooler_output  <span class="hljs-comment"># pooled (EOS token) states</span>

`}}),{c(){d=n("p"),w=i("Examples:"),m=f(),v(p.$$.fragment)},l(s){d=r(s,"P",{});var c=a(d);w=l(c,"Examples:"),c.forEach(o),m=u(s),b(p.$$.fragment,s)},m(s,c){T(s,d,c),e(d,w),T(s,m,c),$(p,s,c),_=!0},p:B,i(s){_||(O(p.$$.fragment,s),_=!0)},o(s){y(p.$$.fragment,s),_=!1},d(s){s&&o(d),s&&o(m),V(p,s)}}}function jh(x){let d,w,m,p,_;return{c(){d=n("p"),w=i("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=n("code"),p=i("Module"),_=i(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(s){d=r(s,"P",{});var c=a(d);w=l(c,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r(c,"CODE",{});var k=a(m);p=l(k,"Module"),k.forEach(o),_=l(c,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),c.forEach(o)},m(s,c){T(s,d,c),e(d,w),e(d,m),e(m,p),e(d,_)},d(s){s&&o(d)}}}function Mh(x){let d,w,m,p,_;return p=new A({props:{code:`from PIL import Image
import requests
from transformers import OwlViTProcessor, OwlViTVisionModel

model = OwlViTVisionModel.from_pretrained("google/owlvit-base-patch32")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
pooled_output = outputs.pooler_output  # pooled CLS states

`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> OwlViTProcessor, OwlViTVisionModel

<span class="hljs-meta">&gt;&gt;&gt; </span>model = OwlViTVisionModel.from_pretrained(<span class="hljs-string">&quot;google/owlvit-base-patch32&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor = OwlViTProcessor.from_pretrained(<span class="hljs-string">&quot;google/owlvit-base-patch32&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(images=image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_state = outputs.last_hidden_state
<span class="hljs-meta">&gt;&gt;&gt; </span>pooled_output = outputs.pooler_output  <span class="hljs-comment"># pooled CLS states</span>

`}}),{c(){d=n("p"),w=i("Examples:"),m=f(),v(p.$$.fragment)},l(s){d=r(s,"P",{});var c=a(d);w=l(c,"Examples:"),c.forEach(o),m=u(s),b(p.$$.fragment,s)},m(s,c){T(s,d,c),e(d,w),T(s,m,c),$(p,s,c),_=!0},p:B,i(s){_||(O(p.$$.fragment,s),_=!0)},o(s){y(p.$$.fragment,s),_=!1},d(s){s&&o(d),s&&o(m),V(p,s)}}}function Eh(x){let d,w,m,p,_;return{c(){d=n("p"),w=i("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=n("code"),p=i("Module"),_=i(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(s){d=r(s,"P",{});var c=a(d);w=l(c,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r(c,"CODE",{});var k=a(m);p=l(k,"Module"),k.forEach(o),_=l(c,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),c.forEach(o)},m(s,c){T(s,d,c),e(d,w),e(d,m),e(m,p),e(d,_)},d(s){s&&o(d)}}}function Fh(x){let d,w,m,p,_;return p=new A({props:{code:`import requests
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = [["a photo of a cat", "a photo of a dog"]]
inputs = processor(text=texts, images=image, return_tensors="pt")
outputs = model(**inputs)

# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
target_sizes = torch.Tensor([image.size[::-1]])
# Convert outputs (bounding boxes and class logits) to COCO API
results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

i = 0  # Retrieve predictions for the first image for the corresponding text queries
text = texts[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

score_threshold = 0.1
for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    if score >= score_threshold:
        print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> OwlViTProcessor, OwlViTForObjectDetection

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = OwlViTProcessor.from_pretrained(<span class="hljs-string">&quot;google/owlvit-base-patch32&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = OwlViTForObjectDetection.from_pretrained(<span class="hljs-string">&quot;google/owlvit-base-patch32&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)
<span class="hljs-meta">&gt;&gt;&gt; </span>texts = [[<span class="hljs-string">&quot;a photo of a cat&quot;</span>, <span class="hljs-string">&quot;a photo of a dog&quot;</span>]]
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(text=texts, images=image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Target image sizes (height, width) to rescale box predictions [batch_size, 2]</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>target_sizes = torch.Tensor([image.size[::-<span class="hljs-number">1</span>]])
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Convert outputs (bounding boxes and class logits) to COCO API</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

<span class="hljs-meta">&gt;&gt;&gt; </span>i = <span class="hljs-number">0</span>  <span class="hljs-comment"># Retrieve predictions for the first image for the corresponding text queries</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>text = texts[i]
<span class="hljs-meta">&gt;&gt;&gt; </span>boxes, scores, labels = results[i][<span class="hljs-string">&quot;boxes&quot;</span>], results[i][<span class="hljs-string">&quot;scores&quot;</span>], results[i][<span class="hljs-string">&quot;labels&quot;</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>score_threshold = <span class="hljs-number">0.1</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> box, score, label <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(boxes, scores, labels):
<span class="hljs-meta">... </span>    box = [<span class="hljs-built_in">round</span>(i, <span class="hljs-number">2</span>) <span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> box.tolist()]
<span class="hljs-meta">... </span>    <span class="hljs-keyword">if</span> score &gt;= score_threshold:
<span class="hljs-meta">... </span>        <span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;Detected <span class="hljs-subst">{text[label]}</span> with confidence <span class="hljs-subst">{<span class="hljs-built_in">round</span>(score.item(), <span class="hljs-number">3</span>)}</span> at location <span class="hljs-subst">{box}</span>&quot;</span>)
Detected a photo of a cat <span class="hljs-keyword">with</span> confidence <span class="hljs-number">0.707</span> at location [<span class="hljs-number">324.97</span>, <span class="hljs-number">20.44</span>, <span class="hljs-number">640.58</span>, <span class="hljs-number">373.29</span>]
Detected a photo of a cat <span class="hljs-keyword">with</span> confidence <span class="hljs-number">0.717</span> at location [<span class="hljs-number">1.46</span>, <span class="hljs-number">55.26</span>, <span class="hljs-number">315.55</span>, <span class="hljs-number">472.17</span>]

`}}),{c(){d=n("p"),w=i("Examples:"),m=f(),v(p.$$.fragment)},l(s){d=r(s,"P",{});var c=a(d);w=l(c,"Examples:"),c.forEach(o),m=u(s),b(p.$$.fragment,s)},m(s,c){T(s,d,c),e(d,w),T(s,m,c),$(p,s,c),_=!0},p:B,i(s){_||(O(p.$$.fragment,s),_=!0)},o(s){y(p.$$.fragment,s),_=!1},d(s){s&&o(d),s&&o(m),V(p,s)}}}function Ph(x){let d,w,m,p,_,s,c,k,Po,Te,U,Y,ue,ge,zo,je,Q,Tt,N,qo,_e,Me,Co,vt,q,Io,ve,be,Ee,Fe,bt,H,C,Pe,K,Do;return{c(){d=n("p"),w=i("TF 2.0 models accepts two formats as inputs:"),m=f(),p=n("ul"),_=n("li"),s=i("having all inputs as keyword arguments (like PyTorch models), or"),c=f(),k=n("li"),Po=i(`having all inputs as a list, tuple or dict in the first positional arguments.
This second option is useful when using `),Te=n("code"),U=i("tf.keras.Model.fit"),Y=i(` method which currently requires having all the
tensors in the first argument of the model call function: `),ue=n("code"),ge=i("model(inputs)"),zo=i(`. If you choose this second option, there
are three possibilities you can use to gather all the input Tensors in the first positional argument :`),je=f(),Q=n("li"),Tt=i("a single Tensor with "),N=n("code"),qo=i("input_ids"),_e=i(" only and nothing else: "),Me=n("code"),Co=i("model(input_ids)"),vt=f(),q=n("li"),Io=i(`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),ve=n("code"),be=i("model([input_ids, attention_mask])"),Ee=i(" or "),Fe=n("code"),bt=i("model([input_ids, attention_mask, token_type_ids])"),H=f(),C=n("li"),Pe=i(`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),K=n("code"),Do=i('model({"input_ids": input_ids, "token_type_ids": token_type_ids})')},l(F){d=r(F,"P",{});var $e=a(d);w=l($e,"TF 2.0 models accepts two formats as inputs:"),$e.forEach(o),m=u(F),p=r(F,"UL",{});var I=a(p);_=r(I,"LI",{});var we=a(_);s=l(we,"having all inputs as keyword arguments (like PyTorch models), or"),we.forEach(o),c=u(I),k=r(I,"LI",{});var Oe=a(k);Po=l(Oe,`having all inputs as a list, tuple or dict in the first positional arguments.
This second option is useful when using `),Te=r(Oe,"CODE",{});var ze=a(Te);U=l(ze,"tf.keras.Model.fit"),ze.forEach(o),Y=l(Oe,` method which currently requires having all the
tensors in the first argument of the model call function: `),ue=r(Oe,"CODE",{});var Us=a(ue);ge=l(Us,"model(inputs)"),Us.forEach(o),zo=l(Oe,`. If you choose this second option, there
are three possibilities you can use to gather all the input Tensors in the first positional argument :`),Oe.forEach(o),je=u(I),Q=r(I,"LI",{});var qe=a(Q);Tt=l(qe,"a single Tensor with "),N=r(qe,"CODE",{});var Lo=a(N);qo=l(Lo,"input_ids"),Lo.forEach(o),_e=l(qe," only and nothing else: "),Me=r(qe,"CODE",{});var M=a(Me);Co=l(M,"model(input_ids)"),M.forEach(o),qe.forEach(o),vt=u(I),q=r(I,"LI",{});var ee=a(q);Io=l(ee,`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),ve=r(ee,"CODE",{});var Hs=a(ve);be=l(Hs,"model([input_ids, attention_mask])"),Hs.forEach(o),Ee=l(ee," or "),Fe=r(ee,"CODE",{});var Ks=a(Fe);bt=l(Ks,"model([input_ids, attention_mask, token_type_ids])"),Ks.forEach(o),ee.forEach(o),H=u(I),C=r(I,"LI",{});var ye=a(C);Pe=l(ye,`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),K=r(ye,"CODE",{});var Rs=a(K);Do=l(Rs,'model({"input_ids": input_ids, "token_type_ids": token_type_ids})'),Rs.forEach(o),ye.forEach(o),I.forEach(o)},m(F,$e){T(F,d,$e),e(d,w),T(F,m,$e),T(F,p,$e),e(p,_),e(_,s),e(p,c),e(p,k),e(k,Po),e(k,Te),e(Te,U),e(k,Y),e(k,ue),e(ue,ge),e(k,zo),e(p,je),e(p,Q),e(Q,Tt),e(Q,N),e(N,qo),e(Q,_e),e(Q,Me),e(Me,Co),e(p,vt),e(p,q),e(q,Io),e(q,ve),e(ve,be),e(q,Ee),e(q,Fe),e(Fe,bt),e(p,H),e(p,C),e(C,Pe),e(C,K),e(K,Do)},d(F){F&&o(d),F&&o(m),F&&o(p)}}}function zh(x){let d,w,m,p,_;return{c(){d=n("p"),w=i("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=n("code"),p=i("Module"),_=i(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(s){d=r(s,"P",{});var c=a(d);w=l(c,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r(c,"CODE",{});var k=a(m);p=l(k,"Module"),k.forEach(o),_=l(c,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),c.forEach(o)},m(s,c){T(s,d,c),e(d,w),e(d,m),e(m,p),e(d,_)},d(s){s&&o(d)}}}function qh(x){let d,w,m,p,_;return p=new A({props:{code:`from PIL import Image
import requests
from transformers import OwlViTProcessor, TFOwlViTPModel

model = TFOwlViTModel.from_pretrained("google/owlvit-base-patch32")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(
    text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="tf", padding=True
)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = tf.nn.softmax(logits_per_image, axis=1)  # we can take the softmax to get the label probabilities`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> OwlViTProcessor, TFOwlViTPModel

<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFOwlViTModel.from_pretrained(<span class="hljs-string">&quot;google/owlvit-base-patch32&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor = OwlViTProcessor.from_pretrained(<span class="hljs-string">&quot;google/owlvit-base-patch32&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(
<span class="hljs-meta">... </span>    text=[<span class="hljs-string">&quot;a photo of a cat&quot;</span>, <span class="hljs-string">&quot;a photo of a dog&quot;</span>], images=image, return_tensors=<span class="hljs-string">&quot;tf&quot;</span>, padding=<span class="hljs-literal">True</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>logits_per_image = outputs.logits_per_image  <span class="hljs-comment"># this is the image-text similarity score</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>probs = tf.nn.softmax(logits_per_image, axis=<span class="hljs-number">1</span>)  <span class="hljs-comment"># we can take the softmax to get the label probabilities</span>`}}),{c(){d=n("p"),w=i("Examples:"),m=f(),v(p.$$.fragment)},l(s){d=r(s,"P",{});var c=a(d);w=l(c,"Examples:"),c.forEach(o),m=u(s),b(p.$$.fragment,s)},m(s,c){T(s,d,c),e(d,w),T(s,m,c),$(p,s,c),_=!0},p:B,i(s){_||(O(p.$$.fragment,s),_=!0)},o(s){y(p.$$.fragment,s),_=!1},d(s){s&&o(d),s&&o(m),V(p,s)}}}function Ch(x){let d,w,m,p,_;return{c(){d=n("p"),w=i("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=n("code"),p=i("Module"),_=i(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(s){d=r(s,"P",{});var c=a(d);w=l(c,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r(c,"CODE",{});var k=a(m);p=l(k,"Module"),k.forEach(o),_=l(c,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),c.forEach(o)},m(s,c){T(s,d,c),e(d,w),e(d,m),e(m,p),e(d,_)},d(s){s&&o(d)}}}function Ih(x){let d,w,m,p,_;return p=new A({props:{code:`from transformers import OwlViTProcessor, TFOwlViTModel

model = TFOwlViTModel.from_pretrained("google/owlvit-base-patch32")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
inputs = processor(
    text=[["a photo of a cat", "a photo of a dog"], ["photo of a astranaut"]], return_tensors="tf"
)
text_features = model.get_text_features(**inputs)`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> OwlViTProcessor, TFOwlViTModel

<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFOwlViTModel.from_pretrained(<span class="hljs-string">&quot;google/owlvit-base-patch32&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor = OwlViTProcessor.from_pretrained(<span class="hljs-string">&quot;google/owlvit-base-patch32&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(
<span class="hljs-meta">... </span>    text=[[<span class="hljs-string">&quot;a photo of a cat&quot;</span>, <span class="hljs-string">&quot;a photo of a dog&quot;</span>], [<span class="hljs-string">&quot;photo of a astranaut&quot;</span>]], return_tensors=<span class="hljs-string">&quot;tf&quot;</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>text_features = model.get_text_features(**inputs)`}}),{c(){d=n("p"),w=i("Examples:"),m=f(),v(p.$$.fragment)},l(s){d=r(s,"P",{});var c=a(d);w=l(c,"Examples:"),c.forEach(o),m=u(s),b(p.$$.fragment,s)},m(s,c){T(s,d,c),e(d,w),T(s,m,c),$(p,s,c),_=!0},p:B,i(s){_||(O(p.$$.fragment,s),_=!0)},o(s){y(p.$$.fragment,s),_=!1},d(s){s&&o(d),s&&o(m),V(p,s)}}}function Dh(x){let d,w,m,p,_;return{c(){d=n("p"),w=i("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=n("code"),p=i("Module"),_=i(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(s){d=r(s,"P",{});var c=a(d);w=l(c,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r(c,"CODE",{});var k=a(m);p=l(k,"Module"),k.forEach(o),_=l(c,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),c.forEach(o)},m(s,c){T(s,d,c),e(d,w),e(d,m),e(m,p),e(d,_)},d(s){s&&o(d)}}}function Lh(x){let d,w,m,p,_;return p=new A({props:{code:`from PIL import Image
import requests
from transformers import OwlViTProcessor, TFOwlViTModel

model = TFOwlViTModel.from_pretrained("google/owlvit-base-patch32")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="tf")
image_features = model.get_image_features(**inputs)`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> OwlViTProcessor, TFOwlViTModel

<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFOwlViTModel.from_pretrained(<span class="hljs-string">&quot;google/owlvit-base-patch32&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor = OwlViTProcessor.from_pretrained(<span class="hljs-string">&quot;google/owlvit-base-patch32&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(images=image, return_tensors=<span class="hljs-string">&quot;tf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>image_features = model.get_image_features(**inputs)`}}),{c(){d=n("p"),w=i("Examples:"),m=f(),v(p.$$.fragment)},l(s){d=r(s,"P",{});var c=a(d);w=l(c,"Examples:"),c.forEach(o),m=u(s),b(p.$$.fragment,s)},m(s,c){T(s,d,c),e(d,w),T(s,m,c),$(p,s,c),_=!0},p:B,i(s){_||(O(p.$$.fragment,s),_=!0)},o(s){y(p.$$.fragment,s),_=!1},d(s){s&&o(d),s&&o(m),V(p,s)}}}function Ah(x){let d,w,m,p,_;return{c(){d=n("p"),w=i("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=n("code"),p=i("Module"),_=i(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(s){d=r(s,"P",{});var c=a(d);w=l(c,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r(c,"CODE",{});var k=a(m);p=l(k,"Module"),k.forEach(o),_=l(c,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),c.forEach(o)},m(s,c){T(s,d,c),e(d,w),e(d,m),e(m,p),e(d,_)},d(s){s&&o(d)}}}function Nh(x){let d,w,m,p,_;return p=new A({props:{code:`from transformers import OwlViTProcessor, TFOwlViTTextModel

model = TFOwlViTTextModel.from_pretrained("google/owlvit-base-patch32")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
inputs = processor(
    text=[["a photo of a cat", "a photo of a dog"], ["photo of a astranaut"]], return_tensors="tf"
)
outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
pooled_output = outputs.pooler_output  # pooled (EOS token) states`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> OwlViTProcessor, TFOwlViTTextModel

<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFOwlViTTextModel.from_pretrained(<span class="hljs-string">&quot;google/owlvit-base-patch32&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor = OwlViTProcessor.from_pretrained(<span class="hljs-string">&quot;google/owlvit-base-patch32&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(
<span class="hljs-meta">... </span>    text=[[<span class="hljs-string">&quot;a photo of a cat&quot;</span>, <span class="hljs-string">&quot;a photo of a dog&quot;</span>], [<span class="hljs-string">&quot;photo of a astranaut&quot;</span>]], return_tensors=<span class="hljs-string">&quot;tf&quot;</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_state = outputs.last_hidden_state
<span class="hljs-meta">&gt;&gt;&gt; </span>pooled_output = outputs.pooler_output  <span class="hljs-comment"># pooled (EOS token) states</span>`}}),{c(){d=n("p"),w=i("Examples:"),m=f(),v(p.$$.fragment)},l(s){d=r(s,"P",{});var c=a(d);w=l(c,"Examples:"),c.forEach(o),m=u(s),b(p.$$.fragment,s)},m(s,c){T(s,d,c),e(d,w),T(s,m,c),$(p,s,c),_=!0},p:B,i(s){_||(O(p.$$.fragment,s),_=!0)},o(s){y(p.$$.fragment,s),_=!1},d(s){s&&o(d),s&&o(m),V(p,s)}}}function Wh(x){let d,w,m,p,_;return{c(){d=n("p"),w=i("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=n("code"),p=i("Module"),_=i(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(s){d=r(s,"P",{});var c=a(d);w=l(c,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r(c,"CODE",{});var k=a(m);p=l(k,"Module"),k.forEach(o),_=l(c,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),c.forEach(o)},m(s,c){T(s,d,c),e(d,w),e(d,m),e(m,p),e(d,_)},d(s){s&&o(d)}}}function Sh(x){let d,w,m,p,_;return p=new A({props:{code:`from PIL import Image
import requests
from transformers import OwlViTProcessor, TFOwlViTVisionModel

model = TFOwlViTVisionModel.from_pretrained("google/owlvit-base-patch32")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="tf")

outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
pooled_output = outputs.pooler_output  # pooled CLS states`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> OwlViTProcessor, TFOwlViTVisionModel

<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFOwlViTVisionModel.from_pretrained(<span class="hljs-string">&quot;google/owlvit-base-patch32&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor = OwlViTProcessor.from_pretrained(<span class="hljs-string">&quot;google/owlvit-base-patch32&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(images=image, return_tensors=<span class="hljs-string">&quot;tf&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_state = outputs.last_hidden_state
<span class="hljs-meta">&gt;&gt;&gt; </span>pooled_output = outputs.pooler_output  <span class="hljs-comment"># pooled CLS states</span>`}}),{c(){d=n("p"),w=i("Examples:"),m=f(),v(p.$$.fragment)},l(s){d=r(s,"P",{});var c=a(d);w=l(c,"Examples:"),c.forEach(o),m=u(s),b(p.$$.fragment,s)},m(s,c){T(s,d,c),e(d,w),T(s,m,c),$(p,s,c),_=!0},p:B,i(s){_||(O(p.$$.fragment,s),_=!0)},o(s){y(p.$$.fragment,s),_=!1},d(s){s&&o(d),s&&o(m),V(p,s)}}}function Bh(x){let d,w,m,p,_;return{c(){d=n("p"),w=i("Although the recipe for forward pass needs to be defined within this function, one should call the "),m=n("code"),p=i("Module"),_=i(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(s){d=r(s,"P",{});var c=a(d);w=l(c,"Although the recipe for forward pass needs to be defined within this function, one should call the "),m=r(c,"CODE",{});var k=a(m);p=l(k,"Module"),k.forEach(o),_=l(c,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),c.forEach(o)},m(s,c){T(s,d,c),e(d,w),e(d,m),e(m,p),e(d,_)},d(s){s&&o(d)}}}function Uh(x){let d,w,m,p,_;return p=new A({props:{code:`import requests
from PIL import Image
import tensorflow as tf
from transformers import OwlViTProcessor, TFOwlViTForObjectDetection

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = TFOwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = [["a photo of a cat", "a photo of a dog"]]
inputs = processor(text=texts, images=image, return_tensors="tf")
outputs = model(**inputs)`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> tensorflow <span class="hljs-keyword">as</span> tf
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> OwlViTProcessor, TFOwlViTForObjectDetection

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = OwlViTProcessor.from_pretrained(<span class="hljs-string">&quot;google/owlvit-base-patch32&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFOwlViTForObjectDetection.from_pretrained(<span class="hljs-string">&quot;google/owlvit-base-patch32&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)
<span class="hljs-meta">&gt;&gt;&gt; </span>texts = [[<span class="hljs-string">&quot;a photo of a cat&quot;</span>, <span class="hljs-string">&quot;a photo of a dog&quot;</span>]]
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(text=texts, images=image, return_tensors=<span class="hljs-string">&quot;tf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)`}}),{c(){d=n("p"),w=i("Examples:"),m=f(),v(p.$$.fragment)},l(s){d=r(s,"P",{});var c=a(d);w=l(c,"Examples:"),c.forEach(o),m=u(s),b(p.$$.fragment,s)},m(s,c){T(s,d,c),e(d,w),T(s,m,c),$(p,s,c),_=!0},p:B,i(s){_||(O(p.$$.fragment,s),_=!0)},o(s){y(p.$$.fragment,s),_=!1},d(s){s&&o(d),s&&o(m),V(p,s)}}}function Hh(x){let d,w,m,p,_,s,c,k,Po,Te,U,Y,ue,ge,zo,je,Q,Tt,N,qo,_e,Me,Co,vt,q,Io,ve,be,Ee,Fe,bt,H,C,Pe,K,Do,F,$e,I,we,Oe,ze,Us,qe,Lo,M,ee,Hs,Ks,ye,Rs,di,Gs,ci,pi,Xs,mi,hi,Zs,fi,ui,Js,gi,_i,Ys,wi,Ti,aa,Ao,ia,Ve,vi,No,bi,$i,Wo,Oi,yi,la,Ce,$t,Zn,So,Vi,Jn,xi,da,R,Bo,ki,Ot,Qs,ji,Mi,en,Ei,Fi,Pi,Ie,zi,tn,qi,Ci,on,Ii,Di,Li,yt,Uo,Ai,Ho,Ni,sn,Wi,Si,ca,De,Vt,Yn,Ko,Bi,Qn,Ui,pa,G,Ro,Hi,Le,Ki,nn,Ri,Gi,Go,Xi,Zi,Ji,Ae,Yi,rn,Qi,el,an,tl,ol,sl,xt,ma,Ne,kt,er,Xo,nl,tr,rl,ha,X,Zo,al,We,il,ln,ll,dl,Jo,cl,pl,ml,Se,hl,dn,fl,ul,cn,gl,_l,wl,jt,fa,Be,Mt,or,Yo,Tl,sr,vl,ua,Z,Qo,bl,nr,$l,Ol,es,yl,pn,Vl,xl,kl,xe,ts,jl,rr,Ml,El,Et,ga,Ue,Ft,ar,os,Fl,ir,Pl,_a,D,ss,zl,L,ql,mn,Cl,Il,hn,Dl,Ll,fn,Al,Nl,lr,Wl,Sl,un,Bl,Ul,Hl,Pt,ns,Kl,rs,Rl,gn,Gl,Xl,Zl,zt,as,Jl,is,Yl,_n,Ql,ed,td,qt,ls,od,ds,sd,dr,nd,rd,wa,He,Ct,cr,cs,ad,pr,id,Ta,J,ps,ld,te,ms,dd,Ke,cd,wn,pd,md,mr,hd,fd,ud,It,gd,Dt,_d,oe,hs,wd,Re,Td,Tn,vd,bd,hr,$d,Od,yd,Lt,Vd,At,xd,se,fs,kd,Ge,jd,vn,Md,Ed,fr,Fd,Pd,zd,Nt,qd,Wt,va,Xe,St,ur,us,Cd,gr,Id,ba,Ze,gs,Dd,ne,_s,Ld,Je,Ad,bn,Nd,Wd,_r,Sd,Bd,Ud,Bt,Hd,Ut,$a,Ye,Ht,wr,ws,Kd,Tr,Rd,Oa,Qe,Ts,Gd,re,vs,Xd,et,Zd,$n,Jd,Yd,vr,Qd,ec,tc,Kt,oc,Rt,ya,tt,Gt,br,bs,sc,$r,nc,Va,ot,$s,rc,ae,Os,ac,st,ic,On,lc,dc,Or,cc,pc,mc,Xt,hc,Zt,xa,nt,Jt,yr,ys,fc,Vr,uc,ka,P,Vs,gc,rt,_c,yn,wc,Tc,xs,vc,bc,$c,Yt,Oc,ie,ks,yc,at,Vc,Vn,xc,kc,xr,jc,Mc,Ec,Qt,Fc,eo,Pc,le,js,zc,it,qc,xn,Cc,Ic,kr,Dc,Lc,Ac,to,Nc,oo,Wc,de,Ms,Sc,lt,Bc,kn,Uc,Hc,jr,Kc,Rc,Gc,so,Xc,no,ja,dt,ro,Mr,Es,Zc,Er,Jc,Ma,ct,Fs,Yc,ce,Ps,Qc,pt,ep,jn,tp,op,Fr,sp,np,rp,ao,ap,io,Ea,mt,lo,Pr,zs,ip,zr,lp,Fa,ht,qs,dp,pe,Cs,cp,ft,pp,Mn,mp,hp,qr,fp,up,gp,co,_p,po,Pa,ut,mo,Cr,Is,wp,Ir,Tp,za,gt,Ds,vp,me,Ls,bp,_t,$p,En,Op,yp,Dr,Vp,xp,kp,ho,jp,fo,qa;return s=new z({}),ge=new z({}),K=new z({}),Ao=new A({props:{code:`import requests
from PIL import Image
import torch

from transformers import OwlViTProcessor, OwlViTForObjectDetection

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = [["a photo of a cat", "a photo of a dog"]]
inputs = processor(text=texts, images=image, return_tensors="pt")
outputs = model(**inputs)

# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
target_sizes = torch.Tensor([image.size[::-1]])
# Convert outputs (bounding boxes and class logits) to COCO API
results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

i = 0  # Retrieve predictions for the first image for the corresponding text queries
text = texts[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

score_threshold = 0.1
for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    if score >= score_threshold:
        print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> OwlViTProcessor, OwlViTForObjectDetection

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = OwlViTProcessor.from_pretrained(<span class="hljs-string">&quot;google/owlvit-base-patch32&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = OwlViTForObjectDetection.from_pretrained(<span class="hljs-string">&quot;google/owlvit-base-patch32&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)
<span class="hljs-meta">&gt;&gt;&gt; </span>texts = [[<span class="hljs-string">&quot;a photo of a cat&quot;</span>, <span class="hljs-string">&quot;a photo of a dog&quot;</span>]]
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(text=texts, images=image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Target image sizes (height, width) to rescale box predictions [batch_size, 2]</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>target_sizes = torch.Tensor([image.size[::-<span class="hljs-number">1</span>]])
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Convert outputs (bounding boxes and class logits) to COCO API</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

<span class="hljs-meta">&gt;&gt;&gt; </span>i = <span class="hljs-number">0</span>  <span class="hljs-comment"># Retrieve predictions for the first image for the corresponding text queries</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>text = texts[i]
<span class="hljs-meta">&gt;&gt;&gt; </span>boxes, scores, labels = results[i][<span class="hljs-string">&quot;boxes&quot;</span>], results[i][<span class="hljs-string">&quot;scores&quot;</span>], results[i][<span class="hljs-string">&quot;labels&quot;</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>score_threshold = <span class="hljs-number">0.1</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> box, score, label <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(boxes, scores, labels):
<span class="hljs-meta">... </span>    box = [<span class="hljs-built_in">round</span>(i, <span class="hljs-number">2</span>) <span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> box.tolist()]
<span class="hljs-meta">... </span>    <span class="hljs-keyword">if</span> score &gt;= score_threshold:
<span class="hljs-meta">... </span>        <span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;Detected <span class="hljs-subst">{text[label]}</span> with confidence <span class="hljs-subst">{<span class="hljs-built_in">round</span>(score.item(), <span class="hljs-number">3</span>)}</span> at location <span class="hljs-subst">{box}</span>&quot;</span>)
Detected a photo of a cat <span class="hljs-keyword">with</span> confidence <span class="hljs-number">0.707</span> at location [<span class="hljs-number">324.97</span>, <span class="hljs-number">20.44</span>, <span class="hljs-number">640.58</span>, <span class="hljs-number">373.29</span>]
Detected a photo of a cat <span class="hljs-keyword">with</span> confidence <span class="hljs-number">0.717</span> at location [<span class="hljs-number">1.46</span>, <span class="hljs-number">55.26</span>, <span class="hljs-number">315.55</span>, <span class="hljs-number">472.17</span>]`}}),So=new z({}),Bo=new j({props:{name:"class transformers.OwlViTConfig",anchor:"transformers.OwlViTConfig",parameters:[{name:"text_config",val:" = None"},{name:"vision_config",val:" = None"},{name:"projection_dim",val:" = 512"},{name:"logit_scale_init_value",val:" = 2.6592"},{name:"return_dict",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.OwlViTConfig.text_config_dict",description:`<strong>text_config_dict</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
Dictionary of configuration options used to initialize <a href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTTextConfig">OwlViTTextConfig</a>.`,name:"text_config_dict"},{anchor:"transformers.OwlViTConfig.vision_config_dict",description:`<strong>vision_config_dict</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
Dictionary of configuration options used to initialize <a href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTVisionConfig">OwlViTVisionConfig</a>.`,name:"vision_config_dict"},{anchor:"transformers.OwlViTConfig.projection_dim",description:`<strong>projection_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
Dimensionality of text and vision projection layers.`,name:"projection_dim"},{anchor:"transformers.OwlViTConfig.logit_scale_init_value",description:`<strong>logit_scale_init_value</strong> (<code>float</code>, <em>optional</em>, defaults to 2.6592) &#x2014;
The inital value of the <em>logit_scale</em> parameter. Default is used as per the original OWL-ViT
implementation.`,name:"logit_scale_init_value"},{anchor:"transformers.OwlViTConfig.kwargs",description:`<strong>kwargs</strong> (<em>optional</em>) &#x2014;
Dictionary of keyword arguments.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/vr_18450/src/transformers/models/owlvit/configuration_owlvit.py#L245"}}),Uo=new j({props:{name:"from_text_vision_configs",anchor:"transformers.OwlViTConfig.from_text_vision_configs",parameters:[{name:"text_config",val:": typing.Dict"},{name:"vision_config",val:": typing.Dict"},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18450/src/transformers/models/owlvit/configuration_owlvit.py#L310",returnDescription:`
<p>An instance of a configuration object</p>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTConfig"
>OwlViTConfig</a></p>
`}}),Ko=new z({}),Ro=new j({props:{name:"class transformers.OwlViTTextConfig",anchor:"transformers.OwlViTTextConfig",parameters:[{name:"vocab_size",val:" = 49408"},{name:"hidden_size",val:" = 512"},{name:"intermediate_size",val:" = 2048"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 8"},{name:"max_position_embeddings",val:" = 16"},{name:"hidden_act",val:" = 'quick_gelu'"},{name:"layer_norm_eps",val:" = 1e-05"},{name:"dropout",val:" = 0.0"},{name:"attention_dropout",val:" = 0.0"},{name:"initializer_range",val:" = 0.02"},{name:"initializer_factor",val:" = 1.0"},{name:"pad_token_id",val:" = 0"},{name:"bos_token_id",val:" = 49406"},{name:"eos_token_id",val:" = 49407"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.OwlViTTextConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 49408) &#x2014;
Vocabulary size of the OWL-ViT text model. Defines the number of different tokens that can be represented
by the <code>inputs_ids</code> passed when calling <a href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTTextModel">OwlViTTextModel</a>.`,name:"vocab_size"},{anchor:"transformers.OwlViTTextConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.OwlViTTextConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (i.e., feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.OwlViTTextConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.OwlViTTextConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.OwlViTTextConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.OwlViTTextConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;quick_gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;selu&quot;</code> and <code>&quot;gelu_new&quot;</code> \`<code>&quot;quick_gelu&quot;</code> are supported. layer_norm_eps (<code>float</code>, <em>optional</em>,
defaults to 1e-5): The epsilon used by the layer normalization layers.`,name:"hidden_act"},{anchor:"transformers.OwlViTTextConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.OwlViTTextConfig.dropout",description:`<strong>dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.`,name:"dropout"},{anchor:"transformers.OwlViTTextConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.OwlViTTextConfig.initializer_factor",description:`<strong>initializer_factor</strong> (<code>float</code>, <em>optional</em>, defaults to 1) &#x2014;
A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
testing).`,name:"initializer_factor"}],source:"https://github.com/huggingface/transformers/blob/vr_18450/src/transformers/models/owlvit/configuration_owlvit.py#L34"}}),xt=new S({props:{anchor:"transformers.OwlViTTextConfig.example",$$slots:{default:[_h]},$$scope:{ctx:x}}}),Xo=new z({}),Zo=new j({props:{name:"class transformers.OwlViTVisionConfig",anchor:"transformers.OwlViTVisionConfig",parameters:[{name:"hidden_size",val:" = 768"},{name:"intermediate_size",val:" = 3072"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"image_size",val:" = 768"},{name:"patch_size",val:" = 32"},{name:"hidden_act",val:" = 'quick_gelu'"},{name:"layer_norm_eps",val:" = 1e-05"},{name:"dropout",val:" = 0.0"},{name:"attention_dropout",val:" = 0.0"},{name:"initializer_range",val:" = 0.02"},{name:"initializer_factor",val:" = 1.0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.OwlViTVisionConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.OwlViTVisionConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (i.e., feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.OwlViTVisionConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.OwlViTVisionConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.OwlViTVisionConfig.image_size",description:`<strong>image_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
The size (resolution) of each image.`,name:"image_size"},{anchor:"transformers.OwlViTVisionConfig.patch_size",description:`<strong>patch_size</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
The size (resolution) of each patch.`,name:"patch_size"},{anchor:"transformers.OwlViTVisionConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;quick_gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;selu&quot;</code> and <code>&quot;gelu_new&quot;</code> \`<code>&quot;quick_gelu&quot;</code> are supported. layer_norm_eps (<code>float</code>, <em>optional</em>,
defaults to 1e-5): The epsilon used by the layer normalization layers.`,name:"hidden_act"},{anchor:"transformers.OwlViTVisionConfig.dropout",description:`<strong>dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.`,name:"dropout"},{anchor:"transformers.OwlViTVisionConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.OwlViTVisionConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.OwlViTVisionConfig.initializer_factor",description:`<strong>initializer_factor</strong> (\`float&#x201C;, <em>optional</em>, defaults to 1) &#x2014;
A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
testing).`,name:"initializer_factor"}],source:"https://github.com/huggingface/transformers/blob/vr_18450/src/transformers/models/owlvit/configuration_owlvit.py#L142"}}),jt=new S({props:{anchor:"transformers.OwlViTVisionConfig.example",$$slots:{default:[wh]},$$scope:{ctx:x}}}),Yo=new z({}),Qo=new j({props:{name:"class transformers.OwlViTFeatureExtractor",anchor:"transformers.OwlViTFeatureExtractor",parameters:[{name:"do_resize",val:" = True"},{name:"size",val:" = (768, 768)"},{name:"resample",val:" = <Resampling.BICUBIC: 3>"},{name:"crop_size",val:" = 768"},{name:"do_center_crop",val:" = False"},{name:"do_normalize",val:" = True"},{name:"image_mean",val:" = None"},{name:"image_std",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.OwlViTFeatureExtractor.do_resize",description:`<strong>do_resize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to resize the shorter edge of the input to a certain <code>size</code>.`,name:"do_resize"},{anchor:"transformers.OwlViTFeatureExtractor.size",description:`<strong>size</strong> (<code>int</code> or <code>Tuple[int, int]</code>, <em>optional</em>, defaults to (768, 768)) &#x2014;
The size to use for resizing the image. Only has an effect if <code>do_resize</code> is set to <code>True</code>. If <code>size</code> is a
sequence like (h, w), output size will be matched to this. If <code>size</code> is an int, then image will be resized
to (size, size).`,name:"size"},{anchor:"transformers.OwlViTFeatureExtractor.resample",description:`<strong>resample</strong> (<code>int</code>, <em>optional</em>, defaults to <code>PIL.Image.BICUBIC</code>) &#x2014;
An optional resampling filter. This can be one of <code>PIL.Image.NEAREST</code>, <code>PIL.Image.BOX</code>,
<code>PIL.Image.BILINEAR</code>, <code>PIL.Image.HAMMING</code>, <code>PIL.Image.BICUBIC</code> or <code>PIL.Image.LANCZOS</code>. Only has an effect
if <code>do_resize</code> is set to <code>True</code>.`,name:"resample"},{anchor:"transformers.OwlViTFeatureExtractor.do_center_crop",description:`<strong>do_center_crop</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to crop the input at the center. If the input size is smaller than <code>crop_size</code> along any edge, the
image is padded with 0&#x2019;s and then center cropped.`,name:"do_center_crop"},{anchor:"transformers.OwlViTFeatureExtractor.crop_size",description:"<strong>crop_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;",name:"crop_size"},{anchor:"transformers.OwlViTFeatureExtractor.do_normalize",description:`<strong>do_normalize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to normalize the input with <code>image_mean</code> and <code>image_std</code>. Desired output size when applying
center-cropping. Only has an effect if <code>do_center_crop</code> is set to <code>True</code>.`,name:"do_normalize"},{anchor:"transformers.OwlViTFeatureExtractor.image_mean",description:`<strong>image_mean</strong> (<code>List[int]</code>, <em>optional</em>, defaults to <code>[0.48145466, 0.4578275, 0.40821073]</code>) &#x2014;
The sequence of means for each channel, to be used when normalizing images.`,name:"image_mean"},{anchor:"transformers.OwlViTFeatureExtractor.image_std",description:`<strong>image_std</strong> (<code>List[int]</code>, <em>optional</em>, defaults to <code>[0.26862954, 0.26130258, 0.27577711]</code>) &#x2014;
The sequence of standard deviations for each channel, to be used when normalizing images.`,name:"image_std"}],source:"https://github.com/huggingface/transformers/blob/vr_18450/src/transformers/models/owlvit/feature_extraction_owlvit.py#L43"}}),ts=new j({props:{name:"__call__",anchor:"transformers.OwlViTFeatureExtractor.__call__",parameters:[{name:"images",val:": typing.Union[PIL.Image.Image, numpy.ndarray, ForwardRef('torch.Tensor'), typing.List[PIL.Image.Image], typing.List[numpy.ndarray], typing.List[ForwardRef('torch.Tensor')]]"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.OwlViTFeatureExtractor.__call__.images",description:`<strong>images</strong> (<code>PIL.Image.Image</code>, <code>np.ndarray</code>, <code>torch.Tensor</code>, <code>List[PIL.Image.Image]</code>, <code>List[np.ndarray]</code>, <code>List[torch.Tensor]</code>) &#x2014;
The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W) or (H, W, C),
where C is a number of channels, H and W are image height and width.`,name:"images"},{anchor:"transformers.OwlViTFeatureExtractor.__call__.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/pr_18450/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>, defaults to <code>&apos;np&apos;</code>) &#x2014;
If set, will return tensors of a particular framework. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return NumPy <code>np.ndarray</code> objects.</li>
<li><code>&apos;jax&apos;</code>: Return JAX <code>jnp.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"}],source:"https://github.com/huggingface/transformers/blob/vr_18450/src/transformers/models/owlvit/feature_extraction_owlvit.py#L136",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18450/en/main_classes/feature_extractor#transformers.BatchFeature"
>BatchFeature</a> with the following fields:</p>
<ul>
<li><strong>pixel_values</strong> \u2014 Pixel values to be fed to a model.</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18450/en/main_classes/feature_extractor#transformers.BatchFeature"
>BatchFeature</a></p>
`}}),Et=new W({props:{warning:!0,$$slots:{default:[Th]},$$scope:{ctx:x}}}),os=new z({}),ss=new j({props:{name:"class transformers.OwlViTProcessor",anchor:"transformers.OwlViTProcessor",parameters:[{name:"feature_extractor",val:""},{name:"tokenizer",val:""}],parametersDescription:[{anchor:"transformers.OwlViTProcessor.feature_extractor",description:`<strong>feature_extractor</strong> (<a href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTFeatureExtractor">OwlViTFeatureExtractor</a>) &#x2014;
The feature extractor is a required input.`,name:"feature_extractor"},{anchor:"transformers.OwlViTProcessor.tokenizer",description:`<strong>tokenizer</strong> ([<code>CLIPTokenizer</code>, <code>CLIPTokenizerFast</code>]) &#x2014;
The tokenizer is a required input.`,name:"tokenizer"}],source:"https://github.com/huggingface/transformers/blob/vr_18450/src/transformers/models/owlvit/processing_owlvit.py#L28"}}),ns=new j({props:{name:"batch_decode",anchor:"transformers.OwlViTProcessor.batch_decode",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18450/src/transformers/models/owlvit/processing_owlvit.py#L149"}}),as=new j({props:{name:"decode",anchor:"transformers.OwlViTProcessor.decode",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18450/src/transformers/models/owlvit/processing_owlvit.py#L156"}}),ls=new j({props:{name:"post_process",anchor:"transformers.OwlViTProcessor.post_process",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18450/src/transformers/models/owlvit/processing_owlvit.py#L142"}}),cs=new z({}),ps=new j({props:{name:"class transformers.OwlViTModel",anchor:"transformers.OwlViTModel",parameters:[{name:"config",val:": OwlViTConfig"}],parametersDescription:[{anchor:"transformers.OwlViTModel.This",description:`<strong>This</strong> model is a PyTorch [torch.nn.Module](https &#x2014;
//pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it`,name:"This"},{anchor:"transformers.OwlViTModel.as",description:`<strong>as</strong> a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and &#x2014;
behavior. &#x2014;
config (<a href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTConfig">OwlViTConfig</a>): Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18450/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"as"}],source:"https://github.com/huggingface/transformers/blob/vr_18450/src/transformers/models/owlvit/modeling_owlvit.py#L877"}}),ms=new j({props:{name:"forward",anchor:"transformers.OwlViTModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"pixel_values",val:": typing.Optional[torch.FloatTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"return_loss",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.OwlViTModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Indices can be obtained using <a href="/docs/transformers/pr_18450/en/model_doc/clip#transformers.CLIPTokenizer">CLIPTokenizer</a>. See
<a href="/docs/transformers/pr_18450/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and <a href="/docs/transformers/pr_18450/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details. <a href="../glossary#input-ids">What are input
IDs?</a>`,name:"input_ids"},{anchor:"transformers.OwlViTModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.
<a href="../glossary#attention-mask">What are attention masks?</a></li>
</ul>`,name:"attention_mask"},{anchor:"transformers.OwlViTModel.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, height, width)</code>) &#x2014;
Pixel values.`,name:"pixel_values"},{anchor:"transformers.OwlViTModel.forward.return_loss",description:`<strong>return_loss</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the contrastive loss.`,name:"return_loss"},{anchor:"transformers.OwlViTModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.OwlViTModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.OwlViTModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18450/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18450/src/transformers/models/owlvit/modeling_owlvit.py#L1011",returnDescription:`
<p>A <code>transformers.models.owlvit.modeling_owlvit.OwlViTOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<code>&lt;class 'transformers.models.owlvit.configuration_owlvit.OwlViTConfig'&gt;</code>) and inputs.</p>
<ul>
<li><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>return_loss</code> is <code>True</code>) \u2014 Contrastive loss for image-text similarity.</li>
<li><strong>logits_per_image</strong> (<code>torch.FloatTensor</code> of shape <code>(image_batch_size, text_batch_size)</code>) \u2014 The scaled dot product scores between <code>image_embeds</code> and <code>text_embeds</code>. This represents the image-text
similarity scores.</li>
<li><strong>logits_per_text</strong> (<code>torch.FloatTensor</code> of shape <code>(text_batch_size, image_batch_size)</code>) \u2014 The scaled dot product scores between <code>text_embeds</code> and <code>image_embeds</code>. This represents the text-image
similarity scores.</li>
<li><strong>text_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size * num_max_text_queries, output_dim</code>) \u2014 The text embeddings obtained by applying the projection layer to the pooled output of <a
  href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTTextModel"
>OwlViTTextModel</a>.</li>
<li><strong>image_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, output_dim</code>) \u2014 The image embeddings obtained by applying the projection layer to the pooled output of
<a
  href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTVisionModel"
>OwlViTVisionModel</a>.</li>
<li><strong>text_model_output</strong> (Tuple<code>BaseModelOutputWithPooling</code>) \u2014 The output of the <a
  href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTTextModel"
>OwlViTTextModel</a>.</li>
<li><strong>vision_model_output</strong> (<code>BaseModelOutputWithPooling</code>) \u2014 The output of the <a
  href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTVisionModel"
>OwlViTVisionModel</a>.</li>
</ul>
`,returnType:`
<p><code>transformers.models.owlvit.modeling_owlvit.OwlViTOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),It=new W({props:{$$slots:{default:[vh]},$$scope:{ctx:x}}}),Dt=new S({props:{anchor:"transformers.OwlViTModel.forward.example",$$slots:{default:[bh]},$$scope:{ctx:x}}}),hs=new j({props:{name:"get_text_features",anchor:"transformers.OwlViTModel.get_text_features",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.OwlViTModel.get_text_features.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size * num_max_text_queries, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Indices can be obtained using <a href="/docs/transformers/pr_18450/en/model_doc/clip#transformers.CLIPTokenizer">CLIPTokenizer</a>. See
<a href="/docs/transformers/pr_18450/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and <a href="/docs/transformers/pr_18450/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details. <a href="../glossary#input-ids">What are input
IDs?</a>`,name:"input_ids"},{anchor:"transformers.OwlViTModel.get_text_features.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, num_max_text_queries, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.
<a href="../glossary#attention-mask">What are attention masks?</a></li>
</ul>`,name:"attention_mask"},{anchor:"transformers.OwlViTModel.get_text_features.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.OwlViTModel.get_text_features.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.OwlViTModel.get_text_features.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18450/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18450/src/transformers/models/owlvit/modeling_owlvit.py#L912",returnDescription:`
<p>The text embeddings obtained by
applying the projection layer to the pooled output of <a
  href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTTextModel"
>OwlViTTextModel</a>.</p>
`,returnType:`
<p>text_features (<code>torch.FloatTensor</code> of shape <code>(batch_size, output_dim</code>)</p>
`}}),Lt=new W({props:{$$slots:{default:[$h]},$$scope:{ctx:x}}}),At=new S({props:{anchor:"transformers.OwlViTModel.get_text_features.example",$$slots:{default:[Oh]},$$scope:{ctx:x}}}),fs=new j({props:{name:"get_image_features",anchor:"transformers.OwlViTModel.get_image_features",parameters:[{name:"pixel_values",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"return_projected",val:": typing.Optional[bool] = True"}],parametersDescription:[{anchor:"transformers.OwlViTModel.get_image_features.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, height, width)</code>) &#x2014;
Pixel values.`,name:"pixel_values"},{anchor:"transformers.OwlViTModel.get_image_features.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.OwlViTModel.get_image_features.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.OwlViTModel.get_image_features.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18450/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18450/src/transformers/models/owlvit/modeling_owlvit.py#L959",returnDescription:`
<p>The image embeddings obtained by
applying the projection layer to the pooled output of <a
  href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTVisionModel"
>OwlViTVisionModel</a>.</p>
`,returnType:`
<p>image_features (<code>torch.FloatTensor</code> of shape <code>(batch_size, output_dim</code>)</p>
`}}),Nt=new W({props:{$$slots:{default:[yh]},$$scope:{ctx:x}}}),Wt=new S({props:{anchor:"transformers.OwlViTModel.get_image_features.example",$$slots:{default:[Vh]},$$scope:{ctx:x}}}),us=new z({}),gs=new j({props:{name:"class transformers.OwlViTTextModel",anchor:"transformers.OwlViTTextModel",parameters:[{name:"config",val:": OwlViTTextConfig"}],source:"https://github.com/huggingface/transformers/blob/vr_18450/src/transformers/models/owlvit/modeling_owlvit.py#L712"}}),_s=new j({props:{name:"forward",anchor:"transformers.OwlViTTextModel.forward",parameters:[{name:"input_ids",val:": Tensor"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.OwlViTTextModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size * num_max_text_queries, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Indices can be obtained using <a href="/docs/transformers/pr_18450/en/model_doc/clip#transformers.CLIPTokenizer">CLIPTokenizer</a>. See
<a href="/docs/transformers/pr_18450/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and <a href="/docs/transformers/pr_18450/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details. <a href="../glossary#input-ids">What are input
IDs?</a>`,name:"input_ids"},{anchor:"transformers.OwlViTTextModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, num_max_text_queries, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.
<a href="../glossary#attention-mask">What are attention masks?</a></li>
</ul>`,name:"attention_mask"},{anchor:"transformers.OwlViTTextModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.OwlViTTextModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.OwlViTTextModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18450/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18450/src/transformers/models/owlvit/modeling_owlvit.py#L727",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18450/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling"
>transformers.modeling_outputs.BaseModelOutputWithPooling</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<code>&lt;class 'transformers.models.owlvit.configuration_owlvit.OwlViTTextConfig'&gt;</code>) and inputs.</p>
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
  href="/docs/transformers/pr_18450/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling"
>transformers.modeling_outputs.BaseModelOutputWithPooling</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Bt=new W({props:{$$slots:{default:[xh]},$$scope:{ctx:x}}}),Ut=new S({props:{anchor:"transformers.OwlViTTextModel.forward.example",$$slots:{default:[kh]},$$scope:{ctx:x}}}),ws=new z({}),Ts=new j({props:{name:"class transformers.OwlViTVisionModel",anchor:"transformers.OwlViTVisionModel",parameters:[{name:"config",val:": OwlViTVisionConfig"}],source:"https://github.com/huggingface/transformers/blob/vr_18450/src/transformers/models/owlvit/modeling_owlvit.py#L824"}}),vs=new j({props:{name:"forward",anchor:"transformers.OwlViTVisionModel.forward",parameters:[{name:"pixel_values",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.OwlViTVisionModel.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, height, width)</code>) &#x2014;
Pixel values.`,name:"pixel_values"},{anchor:"transformers.OwlViTVisionModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.OwlViTVisionModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.OwlViTVisionModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18450/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18450/src/transformers/models/owlvit/modeling_owlvit.py#L837",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18450/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling"
>transformers.modeling_outputs.BaseModelOutputWithPooling</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<code>&lt;class 'transformers.models.owlvit.configuration_owlvit.OwlViTVisionConfig'&gt;</code>) and inputs.</p>
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
  href="/docs/transformers/pr_18450/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling"
>transformers.modeling_outputs.BaseModelOutputWithPooling</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Kt=new W({props:{$$slots:{default:[jh]},$$scope:{ctx:x}}}),Rt=new S({props:{anchor:"transformers.OwlViTVisionModel.forward.example",$$slots:{default:[Mh]},$$scope:{ctx:x}}}),bs=new z({}),$s=new j({props:{name:"class transformers.OwlViTForObjectDetection",anchor:"transformers.OwlViTForObjectDetection",parameters:[{name:"config",val:": OwlViTConfig"}],source:"https://github.com/huggingface/transformers/blob/vr_18450/src/transformers/models/owlvit/modeling_owlvit.py#L1164"}}),Os=new j({props:{name:"forward",anchor:"transformers.OwlViTForObjectDetection.forward",parameters:[{name:"input_ids",val:": Tensor"},{name:"pixel_values",val:": FloatTensor"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.OwlViTForObjectDetection.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, height, width)</code>) &#x2014;
Pixel values.`,name:"pixel_values"},{anchor:"transformers.OwlViTForObjectDetection.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size * num_max_text_queries, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Indices can be obtained using <a href="/docs/transformers/pr_18450/en/model_doc/clip#transformers.CLIPTokenizer">CLIPTokenizer</a>. See
<a href="/docs/transformers/pr_18450/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and <a href="/docs/transformers/pr_18450/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details. <a href="../glossary#input-ids">What are input
IDs?</a>`,name:"input_ids"},{anchor:"transformers.OwlViTForObjectDetection.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, num_max_text_queries, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.
<a href="../glossary#attention-mask">What are attention masks?</a></li>
</ul>`,name:"attention_mask"}],source:"https://github.com/huggingface/transformers/blob/vr_18450/src/transformers/models/owlvit/modeling_owlvit.py#L1292",returnDescription:`
<p>A <code>transformers.models.owlvit.modeling_owlvit.OwlViTObjectDetectionOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<code>&lt;class 'transformers.models.owlvit.configuration_owlvit.OwlViTConfig'&gt;</code>) and inputs.</p>
<ul>
<li><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> are provided)) \u2014 Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
scale-invariant IoU loss.</li>
<li><strong>loss_dict</strong> (<code>Dict</code>, <em>optional</em>) \u2014 A dictionary containing the individual losses. Useful for logging.</li>
<li><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_patches, num_queries)</code>) \u2014 Classification logits (including no-object) for all queries.</li>
<li><strong>pred_boxes</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_patches, 4)</code>) \u2014 Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
possible padding). You can use <code>post_process()</code> to retrieve the unnormalized
bounding boxes.</li>
<li><strong>text_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_max_text_queries, output_dim</code>) \u2014 The text embeddings obtained by applying the projection layer to the pooled output of <a
  href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTTextModel"
>OwlViTTextModel</a>.</li>
<li><strong>image_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, patch_size, patch_size, output_dim</code>) \u2014 Pooled output of <a
  href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTVisionModel"
>OwlViTVisionModel</a>. OWL-ViT represents images as a set of image patches and computes
image embeddings for each patch.</li>
<li><strong>class_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_patches, hidden_size)</code>) \u2014 Class embeddings of all image patches. OWL-ViT represents images as a set of image patches where the total
number of patches is (image_size / patch_size)**2.</li>
<li><strong>text_model_last_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>)) \u2014 Last hidden states extracted from the <a
  href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTTextModel"
>OwlViTTextModel</a>.</li>
<li><strong>vision_model_last_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_patches + 1, hidden_size)</code>)) \u2014 Last hidden states extracted from the <a
  href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTVisionModel"
>OwlViTVisionModel</a>. OWL-ViT represents images as a set of image
patches where the total number of patches is (image_size / patch_size)**2.</li>
</ul>
`,returnType:`
<p><code>transformers.models.owlvit.modeling_owlvit.OwlViTObjectDetectionOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Xt=new W({props:{$$slots:{default:[Eh]},$$scope:{ctx:x}}}),Zt=new S({props:{anchor:"transformers.OwlViTForObjectDetection.forward.example",$$slots:{default:[Fh]},$$scope:{ctx:x}}}),ys=new z({}),Vs=new j({props:{name:"class transformers.TFOwlViTModel",anchor:"transformers.TFOwlViTModel",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFOwlViTModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTConfig">OwlViTConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18450/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18450/src/transformers/models/owlvit/modeling_tf_owlvit.py#L1253"}}),Yt=new W({props:{$$slots:{default:[Ph]},$$scope:{ctx:x}}}),ks=new j({props:{name:"call",anchor:"transformers.TFOwlViTModel.call",parameters:[{name:"input_ids",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"pixel_values",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"position_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"return_loss",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_base_image_embeds",val:": typing.Optional[bool] = False"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"training",val:": bool = False"}],parametersDescription:[{anchor:"transformers.TFOwlViTModel.call.input_ids",description:`<strong>input_ids</strong> (<code>np.ndarray</code>, <code>tf.Tensor</code>, <code>List[tf.Tensor]</code> \`<code>Dict[str, tf.Tensor]</code> or <code>Dict[str, np.ndarray]</code> and each example must have the shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Indices can be obtained using <a href="/docs/transformers/pr_18450/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See
<a href="/docs/transformers/pr_18450/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> and <a href="/docs/transformers/pr_18450/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> for details. <a href="../glossary#input-ids">What are input
IDs?</a>`,name:"input_ids"},{anchor:"transformers.TFOwlViTModel.call.pixel_values",description:`<strong>pixel_values</strong> (<code>np.ndarray</code>, <code>tf.Tensor</code>, <code>List[tf.Tensor]</code> <code>Dict[str, tf.Tensor]</code> or <code>Dict[str, np.ndarray]</code> and each example must have the shape <code>(batch_size, num_channels, height, width)</code>) &#x2014;
Pixel values. Pixel values can be obtained using <a href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTFeatureExtractor">OwlViTFeatureExtractor</a>. See
<a href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTFeatureExtractor.__call__">OwlViTFeatureExtractor.<strong>call</strong>()</a> for details.`,name:"pixel_values"},{anchor:"transformers.TFOwlViTModel.call.attention_mask",description:`<strong>attention_mask</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.
<a href="../glossary#attention-mask">What are attention masks?</a></li>
</ul>`,name:"attention_mask"},{anchor:"transformers.TFOwlViTModel.call.position_ids",description:`<strong>position_ids</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>. <a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.TFOwlViTModel.call.return_loss",description:`<strong>return_loss</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the contrastive loss.`,name:"return_loss"},{anchor:"transformers.TFOwlViTModel.call.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
config will be used instead.`,name:"output_attentions"},{anchor:"transformers.TFOwlViTModel.call.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
used instead.`,name:"output_hidden_states"},{anchor:"transformers.TFOwlViTModel.call.return_base_image_embeds",description:`<strong>return_base_image_embeds</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return unprojected image embeddings. Set to True when <code>TFOwlViTModel</code> is called within
<code>TFOwlViTForObjectDetection</code>.`,name:"return_base_image_embeds"},{anchor:"transformers.TFOwlViTModel.call.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18450/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple. This argument can be used in
eager mode, in graph mode the value will always be set to True.`,name:"return_dict"},{anchor:"transformers.TFOwlViTModel.call.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to \`False&#x201C;) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"}],source:"https://github.com/huggingface/transformers/blob/vr_18450/src/transformers/models/owlvit/modeling_tf_owlvit.py#L1375",returnDescription:`
<p>A <code>transformers.models.owlvit.modeling_tf_owlvit.TFOwlViTOutput</code> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<code>&lt;class 'transformers.models.owlvit.configuration_owlvit.OwlViTConfig'&gt;</code>) and inputs.</p>
<ul>
<li><strong>loss</strong> (<code>tf.Tensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>return_loss</code> is <code>True</code>) \u2014 Contrastive loss for image-text similarity.</li>
<li><strong>logits_per_image:(<code>tf.Tensor</code></strong> of shape <code>(image_batch_size, text_batch_size)</code>) \u2014 The scaled dot product scores between <code>image_embeds</code> and <code>text_embeds</code>. This represents the image-text
similarity scores.</li>
<li><strong>logits_per_text:(<code>tf.Tensor</code></strong> of shape <code>(text_batch_size, image_batch_size)</code>) \u2014 The scaled dot product scores between <code>text_embeds</code> and <code>image_embeds</code>. This represents the text-image
similarity scores.</li>
<li><strong>text_embeds(<code>tf.Tensor</code></strong> of shape <code>(batch_size * num_max_text_queries, output_dim</code>) \u2014 The text embeddings obtained by applying the projection layer to the pooled output of
<a
  href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.TFOwlViTTextModel"
>TFOwlViTTextModel</a>.</li>
<li><strong>image_embeds(<code>tf.Tensor</code></strong> of shape <code>(batch_size, output_dim</code>) \u2014 The image embeddings obtained by applying the projection layer to the pooled output of
<a
  href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.TFOwlViTVisionModel"
>TFOwlViTVisionModel</a>.</li>
<li><strong>text_model_output(<code>TFBaseModelOutputWithPooling</code>):</strong>
The output of the <a
  href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.TFOwlViTTextModel"
>TFOwlViTTextModel</a>.</li>
<li><strong>vision_model_output(<code>TFBaseModelOutputWithPooling</code>):</strong>
The output of the <a
  href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.TFOwlViTVisionModel"
>TFOwlViTVisionModel</a>.</li>
</ul>
`,returnType:`
<p><code>transformers.models.owlvit.modeling_tf_owlvit.TFOwlViTOutput</code> or <code>tuple(tf.Tensor)</code></p>
`}}),Qt=new W({props:{$$slots:{default:[zh]},$$scope:{ctx:x}}}),eo=new S({props:{anchor:"transformers.TFOwlViTModel.call.example",$$slots:{default:[qh]},$$scope:{ctx:x}}}),js=new j({props:{name:"get_text_features",anchor:"transformers.TFOwlViTModel.get_text_features",parameters:[{name:"input_ids",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"position_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"training",val:": bool = False"}],parametersDescription:[{anchor:"transformers.TFOwlViTModel.get_text_features.input_ids",description:`<strong>input_ids</strong> (<code>np.ndarray</code>, <code>tf.Tensor</code>, <code>List[tf.Tensor]</code> \`<code>Dict[str, tf.Tensor]</code> or <code>Dict[str, np.ndarray]</code> and each example must have the shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Indices can be obtained using <a href="/docs/transformers/pr_18450/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See
<a href="/docs/transformers/pr_18450/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> and <a href="/docs/transformers/pr_18450/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> for details. <a href="../glossary#input-ids">What are input
IDs?</a>`,name:"input_ids"},{anchor:"transformers.TFOwlViTModel.get_text_features.attention_mask",description:`<strong>attention_mask</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.
<a href="../glossary#attention-mask">What are attention masks?</a></li>
</ul>`,name:"attention_mask"},{anchor:"transformers.TFOwlViTModel.get_text_features.position_ids",description:`<strong>position_ids</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>. <a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.TFOwlViTModel.get_text_features.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
config will be used instead.`,name:"output_attentions"},{anchor:"transformers.TFOwlViTModel.get_text_features.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
used instead.`,name:"output_hidden_states"},{anchor:"transformers.TFOwlViTModel.get_text_features.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18450/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple. This argument can be used in
eager mode, in graph mode the value will always be set to True.`,name:"return_dict"},{anchor:"transformers.TFOwlViTModel.get_text_features.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to \`False&#x201C;) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"}],source:"https://github.com/huggingface/transformers/blob/vr_18450/src/transformers/models/owlvit/modeling_tf_owlvit.py#L1296",returnDescription:`
<p>The text embeddings obtained by
applying the projection layer to the pooled output of <a
  href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.TFOwlViTTextModel"
>TFOwlViTTextModel</a>.</p>
`,returnType:`
<p>text_features (<code>tf.Tensor</code> of shape <code>(batch_size, output_dim</code>)</p>
`}}),to=new W({props:{$$slots:{default:[Ch]},$$scope:{ctx:x}}}),oo=new S({props:{anchor:"transformers.TFOwlViTModel.get_text_features.example",$$slots:{default:[Ih]},$$scope:{ctx:x}}}),Ms=new j({props:{name:"get_image_features",anchor:"transformers.TFOwlViTModel.get_image_features",parameters:[{name:"pixel_values",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"return_projected",val:": typing.Optional[bool] = True"},{name:"training",val:": bool = False"}],parametersDescription:[{anchor:"transformers.TFOwlViTModel.get_image_features.pixel_values",description:`<strong>pixel_values</strong> (<code>np.ndarray</code>, <code>tf.Tensor</code>, <code>List[tf.Tensor]</code> \`<code>Dict[str, tf.Tensor]</code> or <code>Dict[str, np.ndarray]</code> and each example must have the shape <code>(batch_size, num_channels, height, width)</code>) &#x2014;
Pixel values. Pixel values can be obtained using <a href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTFeatureExtractor">OwlViTFeatureExtractor</a>. See
<a href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTFeatureExtractor.__call__">OwlViTFeatureExtractor.<strong>call</strong>()</a> for details. output_attentions (<code>bool</code>, <em>optional</em>): Whether or not to
return the attentions tensors of all attention layers. See <code>attentions</code> under returned tensors for more
detail. This argument can be used only in eager mode, in graph mode the value in the config will be used
instead.`,name:"pixel_values"},{anchor:"transformers.TFOwlViTModel.get_image_features.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
used instead.`,name:"output_hidden_states"},{anchor:"transformers.TFOwlViTModel.get_image_features.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18450/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple. This argument can be used in
eager mode, in graph mode the value will always be set to True.`,name:"return_dict"},{anchor:"transformers.TFOwlViTModel.get_image_features.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to \`False&#x201C;) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"}],source:"https://github.com/huggingface/transformers/blob/vr_18450/src/transformers/models/owlvit/modeling_tf_owlvit.py#L1335",returnDescription:`
<p>The image embeddings obtained by
applying the projection layer to the pooled output of <a
  href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.TFOwlViTVisionModel"
>TFOwlViTVisionModel</a>.</p>
`,returnType:`
<p>image_features (<code>torch.FloatTensor</code> of shape <code>(batch_size, output_dim</code>)</p>
`}}),so=new W({props:{$$slots:{default:[Dh]},$$scope:{ctx:x}}}),no=new S({props:{anchor:"transformers.TFOwlViTModel.get_image_features.example",$$slots:{default:[Lh]},$$scope:{ctx:x}}}),Es=new z({}),Fs=new j({props:{name:"class transformers.TFOwlViTTextModel",anchor:"transformers.TFOwlViTTextModel",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18450/src/transformers/models/owlvit/modeling_tf_owlvit.py#L1087"}}),Ps=new j({props:{name:"call",anchor:"transformers.TFOwlViTTextModel.call",parameters:[{name:"input_ids",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"position_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"training",val:": typing.Optional[bool] = False"}],parametersDescription:[{anchor:"transformers.TFOwlViTTextModel.call.input_ids",description:`<strong>input_ids</strong> (<code>np.ndarray</code>, <code>tf.Tensor</code>, <code>List[tf.Tensor]</code> \`<code>Dict[str, tf.Tensor]</code> or <code>Dict[str, np.ndarray]</code> and each example must have the shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Indices can be obtained using <a href="/docs/transformers/pr_18450/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See
<a href="/docs/transformers/pr_18450/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> and <a href="/docs/transformers/pr_18450/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> for details. <a href="../glossary#input-ids">What are input
IDs?</a>`,name:"input_ids"},{anchor:"transformers.TFOwlViTTextModel.call.attention_mask",description:`<strong>attention_mask</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.
<a href="../glossary#attention-mask">What are attention masks?</a></li>
</ul>`,name:"attention_mask"},{anchor:"transformers.TFOwlViTTextModel.call.position_ids",description:`<strong>position_ids</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>. <a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.TFOwlViTTextModel.call.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
config will be used instead.`,name:"output_attentions"},{anchor:"transformers.TFOwlViTTextModel.call.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
used instead.`,name:"output_hidden_states"},{anchor:"transformers.TFOwlViTTextModel.call.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18450/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple. This argument can be used in
eager mode, in graph mode the value will always be set to True.`,name:"return_dict"},{anchor:"transformers.TFOwlViTTextModel.call.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to \`False&#x201C;) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"}],source:"https://github.com/huggingface/transformers/blob/vr_18450/src/transformers/models/owlvit/modeling_tf_owlvit.py#L1095",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18450/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutputWithPooling"
>transformers.modeling_tf_outputs.TFBaseModelOutputWithPooling</a> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<code>&lt;class 'transformers.models.owlvit.configuration_owlvit.OwlViTTextConfig'&gt;</code>) and inputs.</p>
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
  href="/docs/transformers/pr_18450/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutputWithPooling"
>transformers.modeling_tf_outputs.TFBaseModelOutputWithPooling</a> or <code>tuple(tf.Tensor)</code></p>
`}}),ao=new W({props:{$$slots:{default:[Ah]},$$scope:{ctx:x}}}),io=new S({props:{anchor:"transformers.TFOwlViTTextModel.call.example",$$slots:{default:[Nh]},$$scope:{ctx:x}}}),zs=new z({}),qs=new j({props:{name:"class transformers.TFOwlViTVisionModel",anchor:"transformers.TFOwlViTVisionModel",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18450/src/transformers/models/owlvit/modeling_tf_owlvit.py#L1160"}}),Cs=new j({props:{name:"call",anchor:"transformers.TFOwlViTVisionModel.call",parameters:[{name:"pixel_values",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"training",val:": typing.Optional[bool] = False"}],parametersDescription:[{anchor:"transformers.TFOwlViTVisionModel.call.pixel_values",description:`<strong>pixel_values</strong> (<code>np.ndarray</code>, <code>tf.Tensor</code>, <code>List[tf.Tensor]</code> \`<code>Dict[str, tf.Tensor]</code> or <code>Dict[str, np.ndarray]</code> and each example must have the shape <code>(batch_size, num_channels, height, width)</code>) &#x2014;
Pixel values. Pixel values can be obtained using <a href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTFeatureExtractor">OwlViTFeatureExtractor</a>. See
<a href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTFeatureExtractor.__call__">OwlViTFeatureExtractor.<strong>call</strong>()</a> for details. output_attentions (<code>bool</code>, <em>optional</em>): Whether or not to
return the attentions tensors of all attention layers. See <code>attentions</code> under returned tensors for more
detail. This argument can be used only in eager mode, in graph mode the value in the config will be used
instead.`,name:"pixel_values"},{anchor:"transformers.TFOwlViTVisionModel.call.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
used instead.`,name:"output_hidden_states"},{anchor:"transformers.TFOwlViTVisionModel.call.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18450/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple. This argument can be used in
eager mode, in graph mode the value will always be set to True.`,name:"return_dict"},{anchor:"transformers.TFOwlViTVisionModel.call.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to \`False&#x201C;) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"}],source:"https://github.com/huggingface/transformers/blob/vr_18450/src/transformers/models/owlvit/modeling_tf_owlvit.py#L1198",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18450/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutputWithPooling"
>transformers.modeling_tf_outputs.TFBaseModelOutputWithPooling</a> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<code>&lt;class 'transformers.models.owlvit.configuration_owlvit.OwlViTVisionConfig'&gt;</code>) and inputs.</p>
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
  href="/docs/transformers/pr_18450/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutputWithPooling"
>transformers.modeling_tf_outputs.TFBaseModelOutputWithPooling</a> or <code>tuple(tf.Tensor)</code></p>
`}}),co=new W({props:{$$slots:{default:[Wh]},$$scope:{ctx:x}}}),po=new S({props:{anchor:"transformers.TFOwlViTVisionModel.call.example",$$slots:{default:[Sh]},$$scope:{ctx:x}}}),Is=new z({}),Ds=new j({props:{name:"class transformers.TFOwlViTForObjectDetection",anchor:"transformers.TFOwlViTForObjectDetection",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18450/src/transformers/models/owlvit/modeling_tf_owlvit.py#L1492"}}),Ls=new j({props:{name:"call",anchor:"transformers.TFOwlViTForObjectDetection.call",parameters:[{name:"input_ids",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"pixel_values",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.TFOwlViTForObjectDetection.call.pixel_values",description:`<strong>pixel_values</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, num_channels, height, width)</code>) &#x2014;
Pixel values.`,name:"pixel_values"},{anchor:"transformers.TFOwlViTForObjectDetection.call.input_ids",description:`<strong>input_ids</strong> (<code>tf.Tensor</code> of shape <code>(batch_size * num_max_text_queries, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Indices can be obtained using <a href="/docs/transformers/pr_18450/en/model_doc/clip#transformers.CLIPTokenizer">CLIPTokenizer</a>. See
<a href="/docs/transformers/pr_18450/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and <a href="/docs/transformers/pr_18450/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details. <a href="../glossary#input-ids">What are input
IDs?</a>`,name:"input_ids"},{anchor:"transformers.TFOwlViTForObjectDetection.call.attention_mask",description:`<strong>attention_mask</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, num_max_text_queries, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.
<a href="../glossary#attention-mask">What are attention masks?</a></li>
</ul>`,name:"attention_mask"}],source:"https://github.com/huggingface/transformers/blob/vr_18450/src/transformers/models/owlvit/modeling_tf_owlvit.py#L1621",returnDescription:`
<p>A <code>transformers.models.owlvit.modeling_tf_owlvit.TFOwlViTObjectDetectionOutput</code> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<code>&lt;class 'transformers.models.owlvit.configuration_owlvit.OwlViTConfig'&gt;</code>) and inputs.</p>
<ul>
<li><strong>loss</strong> (<code>tf.Tensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> are provided)) \u2014 Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
scale-invariant IoU loss.</li>
<li><strong>loss_dict</strong> (<code>Dict</code>, <em>optional</em>) \u2014 A dictionary containing the individual losses. Useful for logging.</li>
<li><strong>logits</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, num_patches, num_queries)</code>) \u2014 Classification logits (including no-object) for all queries.</li>
<li><strong>pred_boxes</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, num_patches, 4)</code>) \u2014 Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
possible padding). You can use <code>post_process()</code> to retrieve the unnormalized
bounding boxes.</li>
<li><strong>text_embeds</strong> (<code>tf.Tensor\`\` of shape </code>(batch_size, num_max_text_queries, output_dim)\`) \u2014 The text embeddings obtained by applying the projection layer to the pooled output of
<a
  href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.TFOwlViTTextModel"
>TFOwlViTTextModel</a>.</li>
<li><strong>image_embeds</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, patch_size, patch_size, output_dim)</code>) \u2014 Pooled output of <a
  href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.TFOwlViTVisionModel"
>TFOwlViTVisionModel</a>. OWL-ViT represents images as a set of image patches and computes
image embeddings for each patch.</li>
<li><strong>class_embeds</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, num_patches, hidden_size)</code>) \u2014 Class embeddings of all image patches. OWL-ViT represents images as a set of image patches where the total
number of patches is (image_size / patch_size)**2.</li>
<li><strong>text_model_last_hidden_states</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) \u2014 Last hidden states extracted from the <a
  href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.TFOwlViTTextModel"
>TFOwlViTTextModel</a>.</li>
<li><strong>vision_model_last_hidden_states</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, num_patches + 1, hidden_size)</code>) \u2014 Last hidden states extracted from the <a
  href="/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.TFOwlViTVisionModel"
>TFOwlViTVisionModel</a>. OWL-ViT represents images as a set of image
patches where the total number of patches is (image_size / patch_size)**2.</li>
</ul>
`,returnType:`
<p><code>transformers.models.owlvit.modeling_tf_owlvit.TFOwlViTObjectDetectionOutput</code> or <code>tuple(tf.Tensor)</code></p>
`}}),ho=new W({props:{$$slots:{default:[Bh]},$$scope:{ctx:x}}}),fo=new S({props:{anchor:"transformers.TFOwlViTForObjectDetection.call.example",$$slots:{default:[Uh]},$$scope:{ctx:x}}}),{c(){d=n("meta"),w=f(),m=n("h1"),p=n("a"),_=n("span"),v(s.$$.fragment),c=f(),k=n("span"),Po=i("OWL-ViT"),Te=f(),U=n("h2"),Y=n("a"),ue=n("span"),v(ge.$$.fragment),zo=f(),je=n("span"),Q=i("Overview"),Tt=f(),N=n("p"),qo=i("The OWL-ViT (short for Vision Transformer for Open-World Localization) was proposed in "),_e=n("a"),Me=i("Simple Open-Vocabulary Object Detection with Vision Transformers"),Co=i(" by Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, Xiao Wang, Xiaohua Zhai, Thomas Kipf, and Neil Houlsby. OWL-ViT is an open-vocabulary object detection network trained on a variety of (image, text) pairs. It can be used to query an image with one or multiple text queries to search for and detect target objects described in text."),vt=f(),q=n("p"),Io=i("The abstract from the paper is the following:"),ve=f(),be=n("p"),Ee=n("em"),Fe=i("Combining simple architectures with large-scale pre-training has led to massive improvements in image classification. For object detection, pre-training and scaling approaches are less well established, especially in the long-tailed and open-vocabulary setting, where training data is relatively scarce. In this paper, we propose a strong recipe for transferring image-text models to open-vocabulary object detection. We use a standard Vision Transformer architecture with minimal modifications, contrastive image-text pre-training, and end-to-end detection fine-tuning. Our analysis of the scaling properties of this setup shows that increasing image-level pre-training and model size yield consistent improvements on the downstream detection task. We provide the adaptation strategies and regularizations needed to attain very strong performance on zero-shot text-conditioned and one-shot image-conditioned object detection. Code and models are available on GitHub."),bt=f(),H=n("h2"),C=n("a"),Pe=n("span"),v(K.$$.fragment),Do=f(),F=n("span"),$e=i("Usage"),I=f(),we=n("p"),Oe=i("OWL-ViT is a zero-shot text-conditioned object detection model. OWL-ViT uses "),ze=n("a"),Us=i("CLIP"),qe=i(" as its multi-modal backbone, with a ViT-like Transformer to get visual features and a causal language model to get the text features. To use CLIP for detection, OWL-ViT removes the final token pooling layer of the vision model and attaches a lightweight classification and box head to each transformer output token. Open-vocabulary classification is enabled by replacing the fixed classification layer weights with the class-name embeddings obtained from the text model. The authors first train CLIP from scratch and fine-tune it end-to-end with the classification and box heads on standard detection datasets using a bipartite matching loss. One or multiple text queries per image can be used to perform zero-shot text-conditioned object detection."),Lo=f(),M=n("p"),ee=n("a"),Hs=i("OwlViTFeatureExtractor"),Ks=i(" can be used to resize (or rescale) and normalize images for the model and "),ye=n("a"),Rs=i("CLIPTokenizer"),di=i(" is used to encode the text. "),Gs=n("a"),ci=i("OwlViTProcessor"),pi=i(" wraps "),Xs=n("a"),mi=i("OwlViTFeatureExtractor"),hi=i(" and "),Zs=n("a"),fi=i("CLIPTokenizer"),ui=i(" into a single instance to both encode the text and prepare the images. The following example shows how to perform object detection using "),Js=n("a"),gi=i("OwlViTProcessor"),_i=i(" and "),Ys=n("a"),wi=i("OwlViTForObjectDetection"),Ti=i("."),aa=f(),v(Ao.$$.fragment),ia=f(),Ve=n("p"),vi=i("This model was contributed by "),No=n("a"),bi=i("adirik"),$i=i(". The original code can be found "),Wo=n("a"),Oi=i("here"),yi=i("."),la=f(),Ce=n("h2"),$t=n("a"),Zn=n("span"),v(So.$$.fragment),Vi=f(),Jn=n("span"),xi=i("OwlViTConfig"),da=f(),R=n("div"),v(Bo.$$.fragment),ki=f(),Ot=n("p"),Qs=n("a"),ji=i("OwlViTConfig"),Mi=i(" is the configuration class to store the configuration of an "),en=n("a"),Ei=i("OwlViTModel"),Fi=i(`. It is used to
instantiate an OWL-ViT model according to the specified arguments, defining the text model and vision model
configs.`),Pi=f(),Ie=n("p"),zi=i("Configuration objects inherit from "),tn=n("a"),qi=i("PretrainedConfig"),Ci=i(` and can be used to control the model outputs. Read the
documentation from `),on=n("a"),Ii=i("PretrainedConfig"),Di=i(" for more information."),Li=f(),yt=n("div"),v(Uo.$$.fragment),Ai=f(),Ho=n("p"),Ni=i("Instantiate a "),sn=n("a"),Wi=i("OwlViTConfig"),Si=i(` (or a derived class) from owlvit text model configuration and owlvit vision
model configuration.`),ca=f(),De=n("h2"),Vt=n("a"),Yn=n("span"),v(Ko.$$.fragment),Bi=f(),Qn=n("span"),Ui=i("OwlViTTextConfig"),pa=f(),G=n("div"),v(Ro.$$.fragment),Hi=f(),Le=n("p"),Ki=i("This is the configuration class to store the configuration of an "),nn=n("a"),Ri=i("OwlViTTextModel"),Gi=i(`. It is used to instantiate an
OwlViT text encoder according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the OwlViT
`),Go=n("a"),Xi=i("google/owlvit-base-patch32"),Zi=i(" architecture."),Ji=f(),Ae=n("p"),Yi=i("Configuration objects inherit from "),rn=n("a"),Qi=i("PretrainedConfig"),el=i(` and can be used to control the model outputs. Read the
documentation from `),an=n("a"),tl=i("PretrainedConfig"),ol=i(" for more information."),sl=f(),v(xt.$$.fragment),ma=f(),Ne=n("h2"),kt=n("a"),er=n("span"),v(Xo.$$.fragment),nl=f(),tr=n("span"),rl=i("OwlViTVisionConfig"),ha=f(),X=n("div"),v(Zo.$$.fragment),al=f(),We=n("p"),il=i("This is the configuration class to store the configuration of an "),ln=n("a"),ll=i("OwlViTVisionModel"),dl=i(`. It is used to instantiate
an OWL-ViT image encoder according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the OWL-ViT
`),Jo=n("a"),cl=i("google/owlvit-base-patch32"),pl=i(" architecture."),ml=f(),Se=n("p"),hl=i("Configuration objects inherit from "),dn=n("a"),fl=i("PretrainedConfig"),ul=i(` and can be used to control the model outputs. Read the
documentation from `),cn=n("a"),gl=i("PretrainedConfig"),_l=i(" for more information."),wl=f(),v(jt.$$.fragment),fa=f(),Be=n("h2"),Mt=n("a"),or=n("span"),v(Yo.$$.fragment),Tl=f(),sr=n("span"),vl=i("OwlViTFeatureExtractor"),ua=f(),Z=n("div"),v(Qo.$$.fragment),bl=f(),nr=n("p"),$l=i("Constructs an OWL-ViT feature extractor."),Ol=f(),es=n("p"),yl=i("This feature extractor inherits from "),pn=n("a"),Vl=i("FeatureExtractionMixin"),xl=i(` which contains most of the main methods. Users
should refer to this superclass for more information regarding those methods.`),kl=f(),xe=n("div"),v(ts.$$.fragment),jl=f(),rr=n("p"),Ml=i("Main method to prepare for the model one or several image(s)."),El=f(),v(Et.$$.fragment),ga=f(),Ue=n("h2"),Ft=n("a"),ar=n("span"),v(os.$$.fragment),Fl=f(),ir=n("span"),Pl=i("OwlViTProcessor"),_a=f(),D=n("div"),v(ss.$$.fragment),zl=f(),L=n("p"),ql=i("Constructs an OWL-ViT processor which wraps "),mn=n("a"),Cl=i("OwlViTFeatureExtractor"),Il=i(" and "),hn=n("a"),Dl=i("CLIPTokenizer"),Ll=i("/"),fn=n("a"),Al=i("CLIPTokenizerFast"),Nl=i(`
into a single processor that interits both the feature extractor and tokenizer functionalities. See the
`),lr=n("code"),Wl=i("__call__()"),Sl=i(" and "),un=n("a"),Bl=i("decode()"),Ul=i(" for more information."),Hl=f(),Pt=n("div"),v(ns.$$.fragment),Kl=f(),rs=n("p"),Rl=i("This method forwards all its arguments to CLIPTokenizerFast\u2019s "),gn=n("a"),Gl=i("batch_decode()"),Xl=i(`. Please
refer to the docstring of this method for more information.`),Zl=f(),zt=n("div"),v(as.$$.fragment),Jl=f(),is=n("p"),Yl=i("This method forwards all its arguments to CLIPTokenizerFast\u2019s "),_n=n("a"),Ql=i("decode()"),ed=i(`. Please refer to
the docstring of this method for more information.`),td=f(),qt=n("div"),v(ls.$$.fragment),od=f(),ds=n("p"),sd=i("This method forwards all its arguments to "),dr=n("code"),nd=i("OwlViTFeatureExtractor.post_process()"),rd=i(`. Please refer to the
docstring of this method for more information.`),wa=f(),He=n("h2"),Ct=n("a"),cr=n("span"),v(cs.$$.fragment),ad=f(),pr=n("span"),id=i("OwlViTModel"),Ta=f(),J=n("div"),v(ps.$$.fragment),ld=f(),te=n("div"),v(ms.$$.fragment),dd=f(),Ke=n("p"),cd=i("The "),wn=n("a"),pd=i("OwlViTModel"),md=i(" forward method, overrides the "),mr=n("code"),hd=i("__call__"),fd=i(" special method."),ud=f(),v(It.$$.fragment),gd=f(),v(Dt.$$.fragment),_d=f(),oe=n("div"),v(hs.$$.fragment),wd=f(),Re=n("p"),Td=i("The "),Tn=n("a"),vd=i("OwlViTModel"),bd=i(" forward method, overrides the "),hr=n("code"),$d=i("__call__"),Od=i(" special method."),yd=f(),v(Lt.$$.fragment),Vd=f(),v(At.$$.fragment),xd=f(),se=n("div"),v(fs.$$.fragment),kd=f(),Ge=n("p"),jd=i("The "),vn=n("a"),Md=i("OwlViTModel"),Ed=i(" forward method, overrides the "),fr=n("code"),Fd=i("__call__"),Pd=i(" special method."),zd=f(),v(Nt.$$.fragment),qd=f(),v(Wt.$$.fragment),va=f(),Xe=n("h2"),St=n("a"),ur=n("span"),v(us.$$.fragment),Cd=f(),gr=n("span"),Id=i("OwlViTTextModel"),ba=f(),Ze=n("div"),v(gs.$$.fragment),Dd=f(),ne=n("div"),v(_s.$$.fragment),Ld=f(),Je=n("p"),Ad=i("The "),bn=n("a"),Nd=i("OwlViTTextModel"),Wd=i(" forward method, overrides the "),_r=n("code"),Sd=i("__call__"),Bd=i(" special method."),Ud=f(),v(Bt.$$.fragment),Hd=f(),v(Ut.$$.fragment),$a=f(),Ye=n("h2"),Ht=n("a"),wr=n("span"),v(ws.$$.fragment),Kd=f(),Tr=n("span"),Rd=i("OwlViTVisionModel"),Oa=f(),Qe=n("div"),v(Ts.$$.fragment),Gd=f(),re=n("div"),v(vs.$$.fragment),Xd=f(),et=n("p"),Zd=i("The "),$n=n("a"),Jd=i("OwlViTVisionModel"),Yd=i(" forward method, overrides the "),vr=n("code"),Qd=i("__call__"),ec=i(" special method."),tc=f(),v(Kt.$$.fragment),oc=f(),v(Rt.$$.fragment),ya=f(),tt=n("h2"),Gt=n("a"),br=n("span"),v(bs.$$.fragment),sc=f(),$r=n("span"),nc=i("OwlViTForObjectDetection"),Va=f(),ot=n("div"),v($s.$$.fragment),rc=f(),ae=n("div"),v(Os.$$.fragment),ac=f(),st=n("p"),ic=i("The "),On=n("a"),lc=i("OwlViTForObjectDetection"),dc=i(" forward method, overrides the "),Or=n("code"),cc=i("__call__"),pc=i(" special method."),mc=f(),v(Xt.$$.fragment),hc=f(),v(Zt.$$.fragment),xa=f(),nt=n("h2"),Jt=n("a"),yr=n("span"),v(ys.$$.fragment),fc=f(),Vr=n("span"),uc=i("TFOwlViTModel"),ka=f(),P=n("div"),v(Vs.$$.fragment),gc=f(),rt=n("p"),_c=i("This model inherits from "),yn=n("a"),wc=i("TFPreTrainedModel"),Tc=i(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.) This model is also a `),xs=n("a"),vc=i("tf.keras.Model"),bc=i(` subclass.
Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general
usage and behavior.`),$c=f(),v(Yt.$$.fragment),Oc=f(),ie=n("div"),v(ks.$$.fragment),yc=f(),at=n("p"),Vc=i("The "),Vn=n("a"),xc=i("TFOwlViTModel"),kc=i(" forward method, overrides the "),xr=n("code"),jc=i("__call__"),Mc=i(" special method."),Ec=f(),v(Qt.$$.fragment),Fc=f(),v(eo.$$.fragment),Pc=f(),le=n("div"),v(js.$$.fragment),zc=f(),it=n("p"),qc=i("The "),xn=n("a"),Cc=i("TFOwlViTModel"),Ic=i(" forward method, overrides the "),kr=n("code"),Dc=i("__call__"),Lc=i(" special method."),Ac=f(),v(to.$$.fragment),Nc=f(),v(oo.$$.fragment),Wc=f(),de=n("div"),v(Ms.$$.fragment),Sc=f(),lt=n("p"),Bc=i("The "),kn=n("a"),Uc=i("TFOwlViTModel"),Hc=i(" forward method, overrides the "),jr=n("code"),Kc=i("__call__"),Rc=i(" special method."),Gc=f(),v(so.$$.fragment),Xc=f(),v(no.$$.fragment),ja=f(),dt=n("h2"),ro=n("a"),Mr=n("span"),v(Es.$$.fragment),Zc=f(),Er=n("span"),Jc=i("TFOwlViTTextModel"),Ma=f(),ct=n("div"),v(Fs.$$.fragment),Yc=f(),ce=n("div"),v(Ps.$$.fragment),Qc=f(),pt=n("p"),ep=i("The "),jn=n("a"),tp=i("TFOwlViTTextModel"),op=i(" forward method, overrides the "),Fr=n("code"),sp=i("__call__"),np=i(" special method."),rp=f(),v(ao.$$.fragment),ap=f(),v(io.$$.fragment),Ea=f(),mt=n("h2"),lo=n("a"),Pr=n("span"),v(zs.$$.fragment),ip=f(),zr=n("span"),lp=i("TFOwlViTVisionModel"),Fa=f(),ht=n("div"),v(qs.$$.fragment),dp=f(),pe=n("div"),v(Cs.$$.fragment),cp=f(),ft=n("p"),pp=i("The "),Mn=n("a"),mp=i("TFOwlViTVisionModel"),hp=i(" forward method, overrides the "),qr=n("code"),fp=i("__call__"),up=i(" special method."),gp=f(),v(co.$$.fragment),_p=f(),v(po.$$.fragment),Pa=f(),ut=n("h2"),mo=n("a"),Cr=n("span"),v(Is.$$.fragment),wp=f(),Ir=n("span"),Tp=i("TFOwlViTForObjectDetection"),za=f(),gt=n("div"),v(Ds.$$.fragment),vp=f(),me=n("div"),v(Ls.$$.fragment),bp=f(),_t=n("p"),$p=i("The "),En=n("a"),Op=i("TFOwlViTForObjectDetection"),yp=i(" forward method, overrides the "),Dr=n("code"),Vp=i("__call__"),xp=i(" special method."),kp=f(),v(ho.$$.fragment),jp=f(),v(fo.$$.fragment),this.h()},l(t){const g=uh('[data-svelte="svelte-1phssyn"]',document.head);d=r(g,"META",{name:!0,content:!0}),g.forEach(o),w=u(t),m=r(t,"H1",{class:!0});var As=a(m);p=r(As,"A",{id:!0,class:!0,href:!0});var Lr=a(p);_=r(Lr,"SPAN",{});var Ar=a(_);b(s.$$.fragment,Ar),Ar.forEach(o),Lr.forEach(o),c=u(As),k=r(As,"SPAN",{});var Nr=a(k);Po=l(Nr,"OWL-ViT"),Nr.forEach(o),As.forEach(o),Te=u(t),U=r(t,"H2",{class:!0});var Ns=a(U);Y=r(Ns,"A",{id:!0,class:!0,href:!0});var Wr=a(Y);ue=r(Wr,"SPAN",{});var Sr=a(ue);b(ge.$$.fragment,Sr),Sr.forEach(o),Wr.forEach(o),zo=u(Ns),je=r(Ns,"SPAN",{});var Br=a(je);Q=l(Br,"Overview"),Br.forEach(o),Ns.forEach(o),Tt=u(t),N=r(t,"P",{});var Ws=a(N);qo=l(Ws,"The OWL-ViT (short for Vision Transformer for Open-World Localization) was proposed in "),_e=r(Ws,"A",{href:!0,rel:!0});var Ur=a(_e);Me=l(Ur,"Simple Open-Vocabulary Object Detection with Vision Transformers"),Ur.forEach(o),Co=l(Ws," by Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, Xiao Wang, Xiaohua Zhai, Thomas Kipf, and Neil Houlsby. OWL-ViT is an open-vocabulary object detection network trained on a variety of (image, text) pairs. It can be used to query an image with one or multiple text queries to search for and detect target objects described in text."),Ws.forEach(o),vt=u(t),q=r(t,"P",{});var Hr=a(q);Io=l(Hr,"The abstract from the paper is the following:"),Hr.forEach(o),ve=u(t),be=r(t,"P",{});var Kr=a(be);Ee=r(Kr,"EM",{});var Rr=a(Ee);Fe=l(Rr,"Combining simple architectures with large-scale pre-training has led to massive improvements in image classification. For object detection, pre-training and scaling approaches are less well established, especially in the long-tailed and open-vocabulary setting, where training data is relatively scarce. In this paper, we propose a strong recipe for transferring image-text models to open-vocabulary object detection. We use a standard Vision Transformer architecture with minimal modifications, contrastive image-text pre-training, and end-to-end detection fine-tuning. Our analysis of the scaling properties of this setup shows that increasing image-level pre-training and model size yield consistent improvements on the downstream detection task. We provide the adaptation strategies and regularizations needed to attain very strong performance on zero-shot text-conditioned and one-shot image-conditioned object detection. Code and models are available on GitHub."),Rr.forEach(o),Kr.forEach(o),bt=u(t),H=r(t,"H2",{class:!0});var Ss=a(H);C=r(Ss,"A",{id:!0,class:!0,href:!0});var Gr=a(C);Pe=r(Gr,"SPAN",{});var Xr=a(Pe);b(K.$$.fragment,Xr),Xr.forEach(o),Gr.forEach(o),Do=u(Ss),F=r(Ss,"SPAN",{});var Zr=a(F);$e=l(Zr,"Usage"),Zr.forEach(o),Ss.forEach(o),I=u(t),we=r(t,"P",{});var Bs=a(we);Oe=l(Bs,"OWL-ViT is a zero-shot text-conditioned object detection model. OWL-ViT uses "),ze=r(Bs,"A",{href:!0});var Jr=a(ze);Us=l(Jr,"CLIP"),Jr.forEach(o),qe=l(Bs," as its multi-modal backbone, with a ViT-like Transformer to get visual features and a causal language model to get the text features. To use CLIP for detection, OWL-ViT removes the final token pooling layer of the vision model and attaches a lightweight classification and box head to each transformer output token. Open-vocabulary classification is enabled by replacing the fixed classification layer weights with the class-name embeddings obtained from the text model. The authors first train CLIP from scratch and fine-tune it end-to-end with the classification and box heads on standard detection datasets using a bipartite matching loss. One or multiple text queries per image can be used to perform zero-shot text-conditioned object detection."),Bs.forEach(o),Lo=u(t),M=r(t,"P",{});var E=a(M);ee=r(E,"A",{href:!0});var Yr=a(ee);Hs=l(Yr,"OwlViTFeatureExtractor"),Yr.forEach(o),Ks=l(E," can be used to resize (or rescale) and normalize images for the model and "),ye=r(E,"A",{href:!0});var Qr=a(ye);Rs=l(Qr,"CLIPTokenizer"),Qr.forEach(o),di=l(E," is used to encode the text. "),Gs=r(E,"A",{href:!0});var ea=a(Gs);ci=l(ea,"OwlViTProcessor"),ea.forEach(o),pi=l(E," wraps "),Xs=r(E,"A",{href:!0});var ta=a(Xs);mi=l(ta,"OwlViTFeatureExtractor"),ta.forEach(o),hi=l(E," and "),Zs=r(E,"A",{href:!0});var oa=a(Zs);fi=l(oa,"CLIPTokenizer"),oa.forEach(o),ui=l(E," into a single instance to both encode the text and prepare the images. The following example shows how to perform object detection using "),Js=r(E,"A",{href:!0});var sa=a(Js);gi=l(sa,"OwlViTProcessor"),sa.forEach(o),_i=l(E," and "),Ys=r(E,"A",{href:!0});var na=a(Ys);wi=l(na,"OwlViTForObjectDetection"),na.forEach(o),Ti=l(E,"."),E.forEach(o),aa=u(t),b(Ao.$$.fragment,t),ia=u(t),Ve=r(t,"P",{});var wt=a(Ve);vi=l(wt,"This model was contributed by "),No=r(wt,"A",{href:!0,rel:!0});var Mp=a(No);bi=l(Mp,"adirik"),Mp.forEach(o),$i=l(wt,". The original code can be found "),Wo=r(wt,"A",{href:!0,rel:!0});var Ep=a(Wo);Oi=l(Ep,"here"),Ep.forEach(o),yi=l(wt,"."),wt.forEach(o),la=u(t),Ce=r(t,"H2",{class:!0});var Ca=a(Ce);$t=r(Ca,"A",{id:!0,class:!0,href:!0});var Fp=a($t);Zn=r(Fp,"SPAN",{});var Pp=a(Zn);b(So.$$.fragment,Pp),Pp.forEach(o),Fp.forEach(o),Vi=u(Ca),Jn=r(Ca,"SPAN",{});var zp=a(Jn);xi=l(zp,"OwlViTConfig"),zp.forEach(o),Ca.forEach(o),da=u(t),R=r(t,"DIV",{class:!0});var uo=a(R);b(Bo.$$.fragment,uo),ki=u(uo),Ot=r(uo,"P",{});var ra=a(Ot);Qs=r(ra,"A",{href:!0});var qp=a(Qs);ji=l(qp,"OwlViTConfig"),qp.forEach(o),Mi=l(ra," is the configuration class to store the configuration of an "),en=r(ra,"A",{href:!0});var Cp=a(en);Ei=l(Cp,"OwlViTModel"),Cp.forEach(o),Fi=l(ra,`. It is used to
instantiate an OWL-ViT model according to the specified arguments, defining the text model and vision model
configs.`),ra.forEach(o),Pi=u(uo),Ie=r(uo,"P",{});var Fn=a(Ie);zi=l(Fn,"Configuration objects inherit from "),tn=r(Fn,"A",{href:!0});var Ip=a(tn);qi=l(Ip,"PretrainedConfig"),Ip.forEach(o),Ci=l(Fn,` and can be used to control the model outputs. Read the
documentation from `),on=r(Fn,"A",{href:!0});var Dp=a(on);Ii=l(Dp,"PretrainedConfig"),Dp.forEach(o),Di=l(Fn," for more information."),Fn.forEach(o),Li=u(uo),yt=r(uo,"DIV",{class:!0});var Ia=a(yt);b(Uo.$$.fragment,Ia),Ai=u(Ia),Ho=r(Ia,"P",{});var Da=a(Ho);Ni=l(Da,"Instantiate a "),sn=r(Da,"A",{href:!0});var Lp=a(sn);Wi=l(Lp,"OwlViTConfig"),Lp.forEach(o),Si=l(Da,` (or a derived class) from owlvit text model configuration and owlvit vision
model configuration.`),Da.forEach(o),Ia.forEach(o),uo.forEach(o),ca=u(t),De=r(t,"H2",{class:!0});var La=a(De);Vt=r(La,"A",{id:!0,class:!0,href:!0});var Ap=a(Vt);Yn=r(Ap,"SPAN",{});var Np=a(Yn);b(Ko.$$.fragment,Np),Np.forEach(o),Ap.forEach(o),Bi=u(La),Qn=r(La,"SPAN",{});var Wp=a(Qn);Ui=l(Wp,"OwlViTTextConfig"),Wp.forEach(o),La.forEach(o),pa=u(t),G=r(t,"DIV",{class:!0});var go=a(G);b(Ro.$$.fragment,go),Hi=u(go),Le=r(go,"P",{});var Pn=a(Le);Ki=l(Pn,"This is the configuration class to store the configuration of an "),nn=r(Pn,"A",{href:!0});var Sp=a(nn);Ri=l(Sp,"OwlViTTextModel"),Sp.forEach(o),Gi=l(Pn,`. It is used to instantiate an
OwlViT text encoder according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the OwlViT
`),Go=r(Pn,"A",{href:!0,rel:!0});var Bp=a(Go);Xi=l(Bp,"google/owlvit-base-patch32"),Bp.forEach(o),Zi=l(Pn," architecture."),Pn.forEach(o),Ji=u(go),Ae=r(go,"P",{});var zn=a(Ae);Yi=l(zn,"Configuration objects inherit from "),rn=r(zn,"A",{href:!0});var Up=a(rn);Qi=l(Up,"PretrainedConfig"),Up.forEach(o),el=l(zn,` and can be used to control the model outputs. Read the
documentation from `),an=r(zn,"A",{href:!0});var Hp=a(an);tl=l(Hp,"PretrainedConfig"),Hp.forEach(o),ol=l(zn," for more information."),zn.forEach(o),sl=u(go),b(xt.$$.fragment,go),go.forEach(o),ma=u(t),Ne=r(t,"H2",{class:!0});var Aa=a(Ne);kt=r(Aa,"A",{id:!0,class:!0,href:!0});var Kp=a(kt);er=r(Kp,"SPAN",{});var Rp=a(er);b(Xo.$$.fragment,Rp),Rp.forEach(o),Kp.forEach(o),nl=u(Aa),tr=r(Aa,"SPAN",{});var Gp=a(tr);rl=l(Gp,"OwlViTVisionConfig"),Gp.forEach(o),Aa.forEach(o),ha=u(t),X=r(t,"DIV",{class:!0});var _o=a(X);b(Zo.$$.fragment,_o),al=u(_o),We=r(_o,"P",{});var qn=a(We);il=l(qn,"This is the configuration class to store the configuration of an "),ln=r(qn,"A",{href:!0});var Xp=a(ln);ll=l(Xp,"OwlViTVisionModel"),Xp.forEach(o),dl=l(qn,`. It is used to instantiate
an OWL-ViT image encoder according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the OWL-ViT
`),Jo=r(qn,"A",{href:!0,rel:!0});var Zp=a(Jo);cl=l(Zp,"google/owlvit-base-patch32"),Zp.forEach(o),pl=l(qn," architecture."),qn.forEach(o),ml=u(_o),Se=r(_o,"P",{});var Cn=a(Se);hl=l(Cn,"Configuration objects inherit from "),dn=r(Cn,"A",{href:!0});var Jp=a(dn);fl=l(Jp,"PretrainedConfig"),Jp.forEach(o),ul=l(Cn,` and can be used to control the model outputs. Read the
documentation from `),cn=r(Cn,"A",{href:!0});var Yp=a(cn);gl=l(Yp,"PretrainedConfig"),Yp.forEach(o),_l=l(Cn," for more information."),Cn.forEach(o),wl=u(_o),b(jt.$$.fragment,_o),_o.forEach(o),fa=u(t),Be=r(t,"H2",{class:!0});var Na=a(Be);Mt=r(Na,"A",{id:!0,class:!0,href:!0});var Qp=a(Mt);or=r(Qp,"SPAN",{});var em=a(or);b(Yo.$$.fragment,em),em.forEach(o),Qp.forEach(o),Tl=u(Na),sr=r(Na,"SPAN",{});var tm=a(sr);vl=l(tm,"OwlViTFeatureExtractor"),tm.forEach(o),Na.forEach(o),ua=u(t),Z=r(t,"DIV",{class:!0});var wo=a(Z);b(Qo.$$.fragment,wo),bl=u(wo),nr=r(wo,"P",{});var om=a(nr);$l=l(om,"Constructs an OWL-ViT feature extractor."),om.forEach(o),Ol=u(wo),es=r(wo,"P",{});var Wa=a(es);yl=l(Wa,"This feature extractor inherits from "),pn=r(Wa,"A",{href:!0});var sm=a(pn);Vl=l(sm,"FeatureExtractionMixin"),sm.forEach(o),xl=l(Wa,` which contains most of the main methods. Users
should refer to this superclass for more information regarding those methods.`),Wa.forEach(o),kl=u(wo),xe=r(wo,"DIV",{class:!0});var In=a(xe);b(ts.$$.fragment,In),jl=u(In),rr=r(In,"P",{});var nm=a(rr);Ml=l(nm,"Main method to prepare for the model one or several image(s)."),nm.forEach(o),El=u(In),b(Et.$$.fragment,In),In.forEach(o),wo.forEach(o),ga=u(t),Ue=r(t,"H2",{class:!0});var Sa=a(Ue);Ft=r(Sa,"A",{id:!0,class:!0,href:!0});var rm=a(Ft);ar=r(rm,"SPAN",{});var am=a(ar);b(os.$$.fragment,am),am.forEach(o),rm.forEach(o),Fl=u(Sa),ir=r(Sa,"SPAN",{});var im=a(ir);Pl=l(im,"OwlViTProcessor"),im.forEach(o),Sa.forEach(o),_a=u(t),D=r(t,"DIV",{class:!0});var ke=a(D);b(ss.$$.fragment,ke),zl=u(ke),L=r(ke,"P",{});var he=a(L);ql=l(he,"Constructs an OWL-ViT processor which wraps "),mn=r(he,"A",{href:!0});var lm=a(mn);Cl=l(lm,"OwlViTFeatureExtractor"),lm.forEach(o),Il=l(he," and "),hn=r(he,"A",{href:!0});var dm=a(hn);Dl=l(dm,"CLIPTokenizer"),dm.forEach(o),Ll=l(he,"/"),fn=r(he,"A",{href:!0});var cm=a(fn);Al=l(cm,"CLIPTokenizerFast"),cm.forEach(o),Nl=l(he,`
into a single processor that interits both the feature extractor and tokenizer functionalities. See the
`),lr=r(he,"CODE",{});var pm=a(lr);Wl=l(pm,"__call__()"),pm.forEach(o),Sl=l(he," and "),un=r(he,"A",{href:!0});var mm=a(un);Bl=l(mm,"decode()"),mm.forEach(o),Ul=l(he," for more information."),he.forEach(o),Hl=u(ke),Pt=r(ke,"DIV",{class:!0});var Ba=a(Pt);b(ns.$$.fragment,Ba),Kl=u(Ba),rs=r(Ba,"P",{});var Ua=a(rs);Rl=l(Ua,"This method forwards all its arguments to CLIPTokenizerFast\u2019s "),gn=r(Ua,"A",{href:!0});var hm=a(gn);Gl=l(hm,"batch_decode()"),hm.forEach(o),Xl=l(Ua,`. Please
refer to the docstring of this method for more information.`),Ua.forEach(o),Ba.forEach(o),Zl=u(ke),zt=r(ke,"DIV",{class:!0});var Ha=a(zt);b(as.$$.fragment,Ha),Jl=u(Ha),is=r(Ha,"P",{});var Ka=a(is);Yl=l(Ka,"This method forwards all its arguments to CLIPTokenizerFast\u2019s "),_n=r(Ka,"A",{href:!0});var fm=a(_n);Ql=l(fm,"decode()"),fm.forEach(o),ed=l(Ka,`. Please refer to
the docstring of this method for more information.`),Ka.forEach(o),Ha.forEach(o),td=u(ke),qt=r(ke,"DIV",{class:!0});var Ra=a(qt);b(ls.$$.fragment,Ra),od=u(Ra),ds=r(Ra,"P",{});var Ga=a(ds);sd=l(Ga,"This method forwards all its arguments to "),dr=r(Ga,"CODE",{});var um=a(dr);nd=l(um,"OwlViTFeatureExtractor.post_process()"),um.forEach(o),rd=l(Ga,`. Please refer to the
docstring of this method for more information.`),Ga.forEach(o),Ra.forEach(o),ke.forEach(o),wa=u(t),He=r(t,"H2",{class:!0});var Xa=a(He);Ct=r(Xa,"A",{id:!0,class:!0,href:!0});var gm=a(Ct);cr=r(gm,"SPAN",{});var _m=a(cr);b(cs.$$.fragment,_m),_m.forEach(o),gm.forEach(o),ad=u(Xa),pr=r(Xa,"SPAN",{});var wm=a(pr);id=l(wm,"OwlViTModel"),wm.forEach(o),Xa.forEach(o),Ta=u(t),J=r(t,"DIV",{class:!0});var To=a(J);b(ps.$$.fragment,To),ld=u(To),te=r(To,"DIV",{class:!0});var vo=a(te);b(ms.$$.fragment,vo),dd=u(vo),Ke=r(vo,"P",{});var Dn=a(Ke);cd=l(Dn,"The "),wn=r(Dn,"A",{href:!0});var Tm=a(wn);pd=l(Tm,"OwlViTModel"),Tm.forEach(o),md=l(Dn," forward method, overrides the "),mr=r(Dn,"CODE",{});var vm=a(mr);hd=l(vm,"__call__"),vm.forEach(o),fd=l(Dn," special method."),Dn.forEach(o),ud=u(vo),b(It.$$.fragment,vo),gd=u(vo),b(Dt.$$.fragment,vo),vo.forEach(o),_d=u(To),oe=r(To,"DIV",{class:!0});var bo=a(oe);b(hs.$$.fragment,bo),wd=u(bo),Re=r(bo,"P",{});var Ln=a(Re);Td=l(Ln,"The "),Tn=r(Ln,"A",{href:!0});var bm=a(Tn);vd=l(bm,"OwlViTModel"),bm.forEach(o),bd=l(Ln," forward method, overrides the "),hr=r(Ln,"CODE",{});var $m=a(hr);$d=l($m,"__call__"),$m.forEach(o),Od=l(Ln," special method."),Ln.forEach(o),yd=u(bo),b(Lt.$$.fragment,bo),Vd=u(bo),b(At.$$.fragment,bo),bo.forEach(o),xd=u(To),se=r(To,"DIV",{class:!0});var $o=a(se);b(fs.$$.fragment,$o),kd=u($o),Ge=r($o,"P",{});var An=a(Ge);jd=l(An,"The "),vn=r(An,"A",{href:!0});var Om=a(vn);Md=l(Om,"OwlViTModel"),Om.forEach(o),Ed=l(An," forward method, overrides the "),fr=r(An,"CODE",{});var ym=a(fr);Fd=l(ym,"__call__"),ym.forEach(o),Pd=l(An," special method."),An.forEach(o),zd=u($o),b(Nt.$$.fragment,$o),qd=u($o),b(Wt.$$.fragment,$o),$o.forEach(o),To.forEach(o),va=u(t),Xe=r(t,"H2",{class:!0});var Za=a(Xe);St=r(Za,"A",{id:!0,class:!0,href:!0});var Vm=a(St);ur=r(Vm,"SPAN",{});var xm=a(ur);b(us.$$.fragment,xm),xm.forEach(o),Vm.forEach(o),Cd=u(Za),gr=r(Za,"SPAN",{});var km=a(gr);Id=l(km,"OwlViTTextModel"),km.forEach(o),Za.forEach(o),ba=u(t),Ze=r(t,"DIV",{class:!0});var Ja=a(Ze);b(gs.$$.fragment,Ja),Dd=u(Ja),ne=r(Ja,"DIV",{class:!0});var Oo=a(ne);b(_s.$$.fragment,Oo),Ld=u(Oo),Je=r(Oo,"P",{});var Nn=a(Je);Ad=l(Nn,"The "),bn=r(Nn,"A",{href:!0});var jm=a(bn);Nd=l(jm,"OwlViTTextModel"),jm.forEach(o),Wd=l(Nn," forward method, overrides the "),_r=r(Nn,"CODE",{});var Mm=a(_r);Sd=l(Mm,"__call__"),Mm.forEach(o),Bd=l(Nn," special method."),Nn.forEach(o),Ud=u(Oo),b(Bt.$$.fragment,Oo),Hd=u(Oo),b(Ut.$$.fragment,Oo),Oo.forEach(o),Ja.forEach(o),$a=u(t),Ye=r(t,"H2",{class:!0});var Ya=a(Ye);Ht=r(Ya,"A",{id:!0,class:!0,href:!0});var Em=a(Ht);wr=r(Em,"SPAN",{});var Fm=a(wr);b(ws.$$.fragment,Fm),Fm.forEach(o),Em.forEach(o),Kd=u(Ya),Tr=r(Ya,"SPAN",{});var Pm=a(Tr);Rd=l(Pm,"OwlViTVisionModel"),Pm.forEach(o),Ya.forEach(o),Oa=u(t),Qe=r(t,"DIV",{class:!0});var Qa=a(Qe);b(Ts.$$.fragment,Qa),Gd=u(Qa),re=r(Qa,"DIV",{class:!0});var yo=a(re);b(vs.$$.fragment,yo),Xd=u(yo),et=r(yo,"P",{});var Wn=a(et);Zd=l(Wn,"The "),$n=r(Wn,"A",{href:!0});var zm=a($n);Jd=l(zm,"OwlViTVisionModel"),zm.forEach(o),Yd=l(Wn," forward method, overrides the "),vr=r(Wn,"CODE",{});var qm=a(vr);Qd=l(qm,"__call__"),qm.forEach(o),ec=l(Wn," special method."),Wn.forEach(o),tc=u(yo),b(Kt.$$.fragment,yo),oc=u(yo),b(Rt.$$.fragment,yo),yo.forEach(o),Qa.forEach(o),ya=u(t),tt=r(t,"H2",{class:!0});var ei=a(tt);Gt=r(ei,"A",{id:!0,class:!0,href:!0});var Cm=a(Gt);br=r(Cm,"SPAN",{});var Im=a(br);b(bs.$$.fragment,Im),Im.forEach(o),Cm.forEach(o),sc=u(ei),$r=r(ei,"SPAN",{});var Dm=a($r);nc=l(Dm,"OwlViTForObjectDetection"),Dm.forEach(o),ei.forEach(o),Va=u(t),ot=r(t,"DIV",{class:!0});var ti=a(ot);b($s.$$.fragment,ti),rc=u(ti),ae=r(ti,"DIV",{class:!0});var Vo=a(ae);b(Os.$$.fragment,Vo),ac=u(Vo),st=r(Vo,"P",{});var Sn=a(st);ic=l(Sn,"The "),On=r(Sn,"A",{href:!0});var Lm=a(On);lc=l(Lm,"OwlViTForObjectDetection"),Lm.forEach(o),dc=l(Sn," forward method, overrides the "),Or=r(Sn,"CODE",{});var Am=a(Or);cc=l(Am,"__call__"),Am.forEach(o),pc=l(Sn," special method."),Sn.forEach(o),mc=u(Vo),b(Xt.$$.fragment,Vo),hc=u(Vo),b(Zt.$$.fragment,Vo),Vo.forEach(o),ti.forEach(o),xa=u(t),nt=r(t,"H2",{class:!0});var oi=a(nt);Jt=r(oi,"A",{id:!0,class:!0,href:!0});var Nm=a(Jt);yr=r(Nm,"SPAN",{});var Wm=a(yr);b(ys.$$.fragment,Wm),Wm.forEach(o),Nm.forEach(o),fc=u(oi),Vr=r(oi,"SPAN",{});var Sm=a(Vr);uc=l(Sm,"TFOwlViTModel"),Sm.forEach(o),oi.forEach(o),ka=u(t),P=r(t,"DIV",{class:!0});var fe=a(P);b(Vs.$$.fragment,fe),gc=u(fe),rt=r(fe,"P",{});var Bn=a(rt);_c=l(Bn,"This model inherits from "),yn=r(Bn,"A",{href:!0});var Bm=a(yn);wc=l(Bm,"TFPreTrainedModel"),Bm.forEach(o),Tc=l(Bn,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.) This model is also a `),xs=r(Bn,"A",{href:!0,rel:!0});var Um=a(xs);vc=l(Um,"tf.keras.Model"),Um.forEach(o),bc=l(Bn,` subclass.
Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general
usage and behavior.`),Bn.forEach(o),$c=u(fe),b(Yt.$$.fragment,fe),Oc=u(fe),ie=r(fe,"DIV",{class:!0});var xo=a(ie);b(ks.$$.fragment,xo),yc=u(xo),at=r(xo,"P",{});var Un=a(at);Vc=l(Un,"The "),Vn=r(Un,"A",{href:!0});var Hm=a(Vn);xc=l(Hm,"TFOwlViTModel"),Hm.forEach(o),kc=l(Un," forward method, overrides the "),xr=r(Un,"CODE",{});var Km=a(xr);jc=l(Km,"__call__"),Km.forEach(o),Mc=l(Un," special method."),Un.forEach(o),Ec=u(xo),b(Qt.$$.fragment,xo),Fc=u(xo),b(eo.$$.fragment,xo),xo.forEach(o),Pc=u(fe),le=r(fe,"DIV",{class:!0});var ko=a(le);b(js.$$.fragment,ko),zc=u(ko),it=r(ko,"P",{});var Hn=a(it);qc=l(Hn,"The "),xn=r(Hn,"A",{href:!0});var Rm=a(xn);Cc=l(Rm,"TFOwlViTModel"),Rm.forEach(o),Ic=l(Hn," forward method, overrides the "),kr=r(Hn,"CODE",{});var Gm=a(kr);Dc=l(Gm,"__call__"),Gm.forEach(o),Lc=l(Hn," special method."),Hn.forEach(o),Ac=u(ko),b(to.$$.fragment,ko),Nc=u(ko),b(oo.$$.fragment,ko),ko.forEach(o),Wc=u(fe),de=r(fe,"DIV",{class:!0});var jo=a(de);b(Ms.$$.fragment,jo),Sc=u(jo),lt=r(jo,"P",{});var Kn=a(lt);Bc=l(Kn,"The "),kn=r(Kn,"A",{href:!0});var Xm=a(kn);Uc=l(Xm,"TFOwlViTModel"),Xm.forEach(o),Hc=l(Kn," forward method, overrides the "),jr=r(Kn,"CODE",{});var Zm=a(jr);Kc=l(Zm,"__call__"),Zm.forEach(o),Rc=l(Kn," special method."),Kn.forEach(o),Gc=u(jo),b(so.$$.fragment,jo),Xc=u(jo),b(no.$$.fragment,jo),jo.forEach(o),fe.forEach(o),ja=u(t),dt=r(t,"H2",{class:!0});var si=a(dt);ro=r(si,"A",{id:!0,class:!0,href:!0});var Jm=a(ro);Mr=r(Jm,"SPAN",{});var Ym=a(Mr);b(Es.$$.fragment,Ym),Ym.forEach(o),Jm.forEach(o),Zc=u(si),Er=r(si,"SPAN",{});var Qm=a(Er);Jc=l(Qm,"TFOwlViTTextModel"),Qm.forEach(o),si.forEach(o),Ma=u(t),ct=r(t,"DIV",{class:!0});var ni=a(ct);b(Fs.$$.fragment,ni),Yc=u(ni),ce=r(ni,"DIV",{class:!0});var Mo=a(ce);b(Ps.$$.fragment,Mo),Qc=u(Mo),pt=r(Mo,"P",{});var Rn=a(pt);ep=l(Rn,"The "),jn=r(Rn,"A",{href:!0});var eh=a(jn);tp=l(eh,"TFOwlViTTextModel"),eh.forEach(o),op=l(Rn," forward method, overrides the "),Fr=r(Rn,"CODE",{});var th=a(Fr);sp=l(th,"__call__"),th.forEach(o),np=l(Rn," special method."),Rn.forEach(o),rp=u(Mo),b(ao.$$.fragment,Mo),ap=u(Mo),b(io.$$.fragment,Mo),Mo.forEach(o),ni.forEach(o),Ea=u(t),mt=r(t,"H2",{class:!0});var ri=a(mt);lo=r(ri,"A",{id:!0,class:!0,href:!0});var oh=a(lo);Pr=r(oh,"SPAN",{});var sh=a(Pr);b(zs.$$.fragment,sh),sh.forEach(o),oh.forEach(o),ip=u(ri),zr=r(ri,"SPAN",{});var nh=a(zr);lp=l(nh,"TFOwlViTVisionModel"),nh.forEach(o),ri.forEach(o),Fa=u(t),ht=r(t,"DIV",{class:!0});var ai=a(ht);b(qs.$$.fragment,ai),dp=u(ai),pe=r(ai,"DIV",{class:!0});var Eo=a(pe);b(Cs.$$.fragment,Eo),cp=u(Eo),ft=r(Eo,"P",{});var Gn=a(ft);pp=l(Gn,"The "),Mn=r(Gn,"A",{href:!0});var rh=a(Mn);mp=l(rh,"TFOwlViTVisionModel"),rh.forEach(o),hp=l(Gn," forward method, overrides the "),qr=r(Gn,"CODE",{});var ah=a(qr);fp=l(ah,"__call__"),ah.forEach(o),up=l(Gn," special method."),Gn.forEach(o),gp=u(Eo),b(co.$$.fragment,Eo),_p=u(Eo),b(po.$$.fragment,Eo),Eo.forEach(o),ai.forEach(o),Pa=u(t),ut=r(t,"H2",{class:!0});var ii=a(ut);mo=r(ii,"A",{id:!0,class:!0,href:!0});var ih=a(mo);Cr=r(ih,"SPAN",{});var lh=a(Cr);b(Is.$$.fragment,lh),lh.forEach(o),ih.forEach(o),wp=u(ii),Ir=r(ii,"SPAN",{});var dh=a(Ir);Tp=l(dh,"TFOwlViTForObjectDetection"),dh.forEach(o),ii.forEach(o),za=u(t),gt=r(t,"DIV",{class:!0});var li=a(gt);b(Ds.$$.fragment,li),vp=u(li),me=r(li,"DIV",{class:!0});var Fo=a(me);b(Ls.$$.fragment,Fo),bp=u(Fo),_t=r(Fo,"P",{});var Xn=a(_t);$p=l(Xn,"The "),En=r(Xn,"A",{href:!0});var ch=a(En);Op=l(ch,"TFOwlViTForObjectDetection"),ch.forEach(o),yp=l(Xn," forward method, overrides the "),Dr=r(Xn,"CODE",{});var ph=a(Dr);Vp=l(ph,"__call__"),ph.forEach(o),xp=l(Xn," special method."),Xn.forEach(o),kp=u(Fo),b(ho.$$.fragment,Fo),jp=u(Fo),b(fo.$$.fragment,Fo),Fo.forEach(o),li.forEach(o),this.h()},h(){h(d,"name","hf:doc:metadata"),h(d,"content",JSON.stringify(Kh)),h(p,"id","owlvit"),h(p,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),h(p,"href","#owlvit"),h(m,"class","relative group"),h(Y,"id","overview"),h(Y,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),h(Y,"href","#overview"),h(U,"class","relative group"),h(_e,"href","https://arxiv.org/abs/2205.06230"),h(_e,"rel","nofollow"),h(C,"id","usage"),h(C,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),h(C,"href","#usage"),h(H,"class","relative group"),h(ze,"href","clip"),h(ee,"href","/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTFeatureExtractor"),h(ye,"href","/docs/transformers/pr_18450/en/model_doc/clip#transformers.CLIPTokenizer"),h(Gs,"href","/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTProcessor"),h(Xs,"href","/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTFeatureExtractor"),h(Zs,"href","/docs/transformers/pr_18450/en/model_doc/clip#transformers.CLIPTokenizer"),h(Js,"href","/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTProcessor"),h(Ys,"href","/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTForObjectDetection"),h(No,"href","https://huggingface.co/adirik"),h(No,"rel","nofollow"),h(Wo,"href","https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit"),h(Wo,"rel","nofollow"),h($t,"id","transformers.OwlViTConfig"),h($t,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),h($t,"href","#transformers.OwlViTConfig"),h(Ce,"class","relative group"),h(Qs,"href","/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTConfig"),h(en,"href","/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTModel"),h(tn,"href","/docs/transformers/pr_18450/en/main_classes/configuration#transformers.PretrainedConfig"),h(on,"href","/docs/transformers/pr_18450/en/main_classes/configuration#transformers.PretrainedConfig"),h(sn,"href","/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTConfig"),h(yt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(Vt,"id","transformers.OwlViTTextConfig"),h(Vt,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),h(Vt,"href","#transformers.OwlViTTextConfig"),h(De,"class","relative group"),h(nn,"href","/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTTextModel"),h(Go,"href","https://huggingface.co/google/owlvit-base-patch32"),h(Go,"rel","nofollow"),h(rn,"href","/docs/transformers/pr_18450/en/main_classes/configuration#transformers.PretrainedConfig"),h(an,"href","/docs/transformers/pr_18450/en/main_classes/configuration#transformers.PretrainedConfig"),h(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(kt,"id","transformers.OwlViTVisionConfig"),h(kt,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),h(kt,"href","#transformers.OwlViTVisionConfig"),h(Ne,"class","relative group"),h(ln,"href","/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTVisionModel"),h(Jo,"href","https://huggingface.co/google/owlvit-base-patch32"),h(Jo,"rel","nofollow"),h(dn,"href","/docs/transformers/pr_18450/en/main_classes/configuration#transformers.PretrainedConfig"),h(cn,"href","/docs/transformers/pr_18450/en/main_classes/configuration#transformers.PretrainedConfig"),h(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(Mt,"id","transformers.OwlViTFeatureExtractor"),h(Mt,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),h(Mt,"href","#transformers.OwlViTFeatureExtractor"),h(Be,"class","relative group"),h(pn,"href","/docs/transformers/pr_18450/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin"),h(xe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(Ft,"id","transformers.OwlViTProcessor"),h(Ft,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),h(Ft,"href","#transformers.OwlViTProcessor"),h(Ue,"class","relative group"),h(mn,"href","/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTFeatureExtractor"),h(hn,"href","/docs/transformers/pr_18450/en/model_doc/clip#transformers.CLIPTokenizer"),h(fn,"href","/docs/transformers/pr_18450/en/model_doc/clip#transformers.CLIPTokenizerFast"),h(un,"href","/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTProcessor.decode"),h(gn,"href","/docs/transformers/pr_18450/en/model_doc/speech_to_text_2#transformers.Speech2Text2Tokenizer.batch_decode"),h(Pt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(_n,"href","/docs/transformers/pr_18450/en/model_doc/speech_to_text_2#transformers.Speech2Text2Tokenizer.decode"),h(zt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(qt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(Ct,"id","transformers.OwlViTModel"),h(Ct,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),h(Ct,"href","#transformers.OwlViTModel"),h(He,"class","relative group"),h(wn,"href","/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTModel"),h(te,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(Tn,"href","/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTModel"),h(oe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(vn,"href","/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTModel"),h(se,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(St,"id","transformers.OwlViTTextModel"),h(St,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),h(St,"href","#transformers.OwlViTTextModel"),h(Xe,"class","relative group"),h(bn,"href","/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTTextModel"),h(ne,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(Ze,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(Ht,"id","transformers.OwlViTVisionModel"),h(Ht,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),h(Ht,"href","#transformers.OwlViTVisionModel"),h(Ye,"class","relative group"),h($n,"href","/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTVisionModel"),h(re,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(Qe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(Gt,"id","transformers.OwlViTForObjectDetection"),h(Gt,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),h(Gt,"href","#transformers.OwlViTForObjectDetection"),h(tt,"class","relative group"),h(On,"href","/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.OwlViTForObjectDetection"),h(ae,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(ot,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(Jt,"id","transformers.TFOwlViTModel"),h(Jt,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),h(Jt,"href","#transformers.TFOwlViTModel"),h(nt,"class","relative group"),h(yn,"href","/docs/transformers/pr_18450/en/main_classes/model#transformers.TFPreTrainedModel"),h(xs,"href","https://www.tensorflow.org/api_docs/python/tf/keras/Model"),h(xs,"rel","nofollow"),h(Vn,"href","/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.TFOwlViTModel"),h(ie,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(xn,"href","/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.TFOwlViTModel"),h(le,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(kn,"href","/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.TFOwlViTModel"),h(de,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(ro,"id","transformers.TFOwlViTTextModel"),h(ro,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),h(ro,"href","#transformers.TFOwlViTTextModel"),h(dt,"class","relative group"),h(jn,"href","/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.TFOwlViTTextModel"),h(ce,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(ct,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(lo,"id","transformers.TFOwlViTVisionModel"),h(lo,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),h(lo,"href","#transformers.TFOwlViTVisionModel"),h(mt,"class","relative group"),h(Mn,"href","/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.TFOwlViTVisionModel"),h(pe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(ht,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(mo,"id","transformers.TFOwlViTForObjectDetection"),h(mo,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),h(mo,"href","#transformers.TFOwlViTForObjectDetection"),h(ut,"class","relative group"),h(En,"href","/docs/transformers/pr_18450/en/model_doc/owlvit#transformers.TFOwlViTForObjectDetection"),h(me,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(gt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(t,g){e(document.head,d),T(t,w,g),T(t,m,g),e(m,p),e(p,_),$(s,_,null),e(m,c),e(m,k),e(k,Po),T(t,Te,g),T(t,U,g),e(U,Y),e(Y,ue),$(ge,ue,null),e(U,zo),e(U,je),e(je,Q),T(t,Tt,g),T(t,N,g),e(N,qo),e(N,_e),e(_e,Me),e(N,Co),T(t,vt,g),T(t,q,g),e(q,Io),T(t,ve,g),T(t,be,g),e(be,Ee),e(Ee,Fe),T(t,bt,g),T(t,H,g),e(H,C),e(C,Pe),$(K,Pe,null),e(H,Do),e(H,F),e(F,$e),T(t,I,g),T(t,we,g),e(we,Oe),e(we,ze),e(ze,Us),e(we,qe),T(t,Lo,g),T(t,M,g),e(M,ee),e(ee,Hs),e(M,Ks),e(M,ye),e(ye,Rs),e(M,di),e(M,Gs),e(Gs,ci),e(M,pi),e(M,Xs),e(Xs,mi),e(M,hi),e(M,Zs),e(Zs,fi),e(M,ui),e(M,Js),e(Js,gi),e(M,_i),e(M,Ys),e(Ys,wi),e(M,Ti),T(t,aa,g),$(Ao,t,g),T(t,ia,g),T(t,Ve,g),e(Ve,vi),e(Ve,No),e(No,bi),e(Ve,$i),e(Ve,Wo),e(Wo,Oi),e(Ve,yi),T(t,la,g),T(t,Ce,g),e(Ce,$t),e($t,Zn),$(So,Zn,null),e(Ce,Vi),e(Ce,Jn),e(Jn,xi),T(t,da,g),T(t,R,g),$(Bo,R,null),e(R,ki),e(R,Ot),e(Ot,Qs),e(Qs,ji),e(Ot,Mi),e(Ot,en),e(en,Ei),e(Ot,Fi),e(R,Pi),e(R,Ie),e(Ie,zi),e(Ie,tn),e(tn,qi),e(Ie,Ci),e(Ie,on),e(on,Ii),e(Ie,Di),e(R,Li),e(R,yt),$(Uo,yt,null),e(yt,Ai),e(yt,Ho),e(Ho,Ni),e(Ho,sn),e(sn,Wi),e(Ho,Si),T(t,ca,g),T(t,De,g),e(De,Vt),e(Vt,Yn),$(Ko,Yn,null),e(De,Bi),e(De,Qn),e(Qn,Ui),T(t,pa,g),T(t,G,g),$(Ro,G,null),e(G,Hi),e(G,Le),e(Le,Ki),e(Le,nn),e(nn,Ri),e(Le,Gi),e(Le,Go),e(Go,Xi),e(Le,Zi),e(G,Ji),e(G,Ae),e(Ae,Yi),e(Ae,rn),e(rn,Qi),e(Ae,el),e(Ae,an),e(an,tl),e(Ae,ol),e(G,sl),$(xt,G,null),T(t,ma,g),T(t,Ne,g),e(Ne,kt),e(kt,er),$(Xo,er,null),e(Ne,nl),e(Ne,tr),e(tr,rl),T(t,ha,g),T(t,X,g),$(Zo,X,null),e(X,al),e(X,We),e(We,il),e(We,ln),e(ln,ll),e(We,dl),e(We,Jo),e(Jo,cl),e(We,pl),e(X,ml),e(X,Se),e(Se,hl),e(Se,dn),e(dn,fl),e(Se,ul),e(Se,cn),e(cn,gl),e(Se,_l),e(X,wl),$(jt,X,null),T(t,fa,g),T(t,Be,g),e(Be,Mt),e(Mt,or),$(Yo,or,null),e(Be,Tl),e(Be,sr),e(sr,vl),T(t,ua,g),T(t,Z,g),$(Qo,Z,null),e(Z,bl),e(Z,nr),e(nr,$l),e(Z,Ol),e(Z,es),e(es,yl),e(es,pn),e(pn,Vl),e(es,xl),e(Z,kl),e(Z,xe),$(ts,xe,null),e(xe,jl),e(xe,rr),e(rr,Ml),e(xe,El),$(Et,xe,null),T(t,ga,g),T(t,Ue,g),e(Ue,Ft),e(Ft,ar),$(os,ar,null),e(Ue,Fl),e(Ue,ir),e(ir,Pl),T(t,_a,g),T(t,D,g),$(ss,D,null),e(D,zl),e(D,L),e(L,ql),e(L,mn),e(mn,Cl),e(L,Il),e(L,hn),e(hn,Dl),e(L,Ll),e(L,fn),e(fn,Al),e(L,Nl),e(L,lr),e(lr,Wl),e(L,Sl),e(L,un),e(un,Bl),e(L,Ul),e(D,Hl),e(D,Pt),$(ns,Pt,null),e(Pt,Kl),e(Pt,rs),e(rs,Rl),e(rs,gn),e(gn,Gl),e(rs,Xl),e(D,Zl),e(D,zt),$(as,zt,null),e(zt,Jl),e(zt,is),e(is,Yl),e(is,_n),e(_n,Ql),e(is,ed),e(D,td),e(D,qt),$(ls,qt,null),e(qt,od),e(qt,ds),e(ds,sd),e(ds,dr),e(dr,nd),e(ds,rd),T(t,wa,g),T(t,He,g),e(He,Ct),e(Ct,cr),$(cs,cr,null),e(He,ad),e(He,pr),e(pr,id),T(t,Ta,g),T(t,J,g),$(ps,J,null),e(J,ld),e(J,te),$(ms,te,null),e(te,dd),e(te,Ke),e(Ke,cd),e(Ke,wn),e(wn,pd),e(Ke,md),e(Ke,mr),e(mr,hd),e(Ke,fd),e(te,ud),$(It,te,null),e(te,gd),$(Dt,te,null),e(J,_d),e(J,oe),$(hs,oe,null),e(oe,wd),e(oe,Re),e(Re,Td),e(Re,Tn),e(Tn,vd),e(Re,bd),e(Re,hr),e(hr,$d),e(Re,Od),e(oe,yd),$(Lt,oe,null),e(oe,Vd),$(At,oe,null),e(J,xd),e(J,se),$(fs,se,null),e(se,kd),e(se,Ge),e(Ge,jd),e(Ge,vn),e(vn,Md),e(Ge,Ed),e(Ge,fr),e(fr,Fd),e(Ge,Pd),e(se,zd),$(Nt,se,null),e(se,qd),$(Wt,se,null),T(t,va,g),T(t,Xe,g),e(Xe,St),e(St,ur),$(us,ur,null),e(Xe,Cd),e(Xe,gr),e(gr,Id),T(t,ba,g),T(t,Ze,g),$(gs,Ze,null),e(Ze,Dd),e(Ze,ne),$(_s,ne,null),e(ne,Ld),e(ne,Je),e(Je,Ad),e(Je,bn),e(bn,Nd),e(Je,Wd),e(Je,_r),e(_r,Sd),e(Je,Bd),e(ne,Ud),$(Bt,ne,null),e(ne,Hd),$(Ut,ne,null),T(t,$a,g),T(t,Ye,g),e(Ye,Ht),e(Ht,wr),$(ws,wr,null),e(Ye,Kd),e(Ye,Tr),e(Tr,Rd),T(t,Oa,g),T(t,Qe,g),$(Ts,Qe,null),e(Qe,Gd),e(Qe,re),$(vs,re,null),e(re,Xd),e(re,et),e(et,Zd),e(et,$n),e($n,Jd),e(et,Yd),e(et,vr),e(vr,Qd),e(et,ec),e(re,tc),$(Kt,re,null),e(re,oc),$(Rt,re,null),T(t,ya,g),T(t,tt,g),e(tt,Gt),e(Gt,br),$(bs,br,null),e(tt,sc),e(tt,$r),e($r,nc),T(t,Va,g),T(t,ot,g),$($s,ot,null),e(ot,rc),e(ot,ae),$(Os,ae,null),e(ae,ac),e(ae,st),e(st,ic),e(st,On),e(On,lc),e(st,dc),e(st,Or),e(Or,cc),e(st,pc),e(ae,mc),$(Xt,ae,null),e(ae,hc),$(Zt,ae,null),T(t,xa,g),T(t,nt,g),e(nt,Jt),e(Jt,yr),$(ys,yr,null),e(nt,fc),e(nt,Vr),e(Vr,uc),T(t,ka,g),T(t,P,g),$(Vs,P,null),e(P,gc),e(P,rt),e(rt,_c),e(rt,yn),e(yn,wc),e(rt,Tc),e(rt,xs),e(xs,vc),e(rt,bc),e(P,$c),$(Yt,P,null),e(P,Oc),e(P,ie),$(ks,ie,null),e(ie,yc),e(ie,at),e(at,Vc),e(at,Vn),e(Vn,xc),e(at,kc),e(at,xr),e(xr,jc),e(at,Mc),e(ie,Ec),$(Qt,ie,null),e(ie,Fc),$(eo,ie,null),e(P,Pc),e(P,le),$(js,le,null),e(le,zc),e(le,it),e(it,qc),e(it,xn),e(xn,Cc),e(it,Ic),e(it,kr),e(kr,Dc),e(it,Lc),e(le,Ac),$(to,le,null),e(le,Nc),$(oo,le,null),e(P,Wc),e(P,de),$(Ms,de,null),e(de,Sc),e(de,lt),e(lt,Bc),e(lt,kn),e(kn,Uc),e(lt,Hc),e(lt,jr),e(jr,Kc),e(lt,Rc),e(de,Gc),$(so,de,null),e(de,Xc),$(no,de,null),T(t,ja,g),T(t,dt,g),e(dt,ro),e(ro,Mr),$(Es,Mr,null),e(dt,Zc),e(dt,Er),e(Er,Jc),T(t,Ma,g),T(t,ct,g),$(Fs,ct,null),e(ct,Yc),e(ct,ce),$(Ps,ce,null),e(ce,Qc),e(ce,pt),e(pt,ep),e(pt,jn),e(jn,tp),e(pt,op),e(pt,Fr),e(Fr,sp),e(pt,np),e(ce,rp),$(ao,ce,null),e(ce,ap),$(io,ce,null),T(t,Ea,g),T(t,mt,g),e(mt,lo),e(lo,Pr),$(zs,Pr,null),e(mt,ip),e(mt,zr),e(zr,lp),T(t,Fa,g),T(t,ht,g),$(qs,ht,null),e(ht,dp),e(ht,pe),$(Cs,pe,null),e(pe,cp),e(pe,ft),e(ft,pp),e(ft,Mn),e(Mn,mp),e(ft,hp),e(ft,qr),e(qr,fp),e(ft,up),e(pe,gp),$(co,pe,null),e(pe,_p),$(po,pe,null),T(t,Pa,g),T(t,ut,g),e(ut,mo),e(mo,Cr),$(Is,Cr,null),e(ut,wp),e(ut,Ir),e(Ir,Tp),T(t,za,g),T(t,gt,g),$(Ds,gt,null),e(gt,vp),e(gt,me),$(Ls,me,null),e(me,bp),e(me,_t),e(_t,$p),e(_t,En),e(En,Op),e(_t,yp),e(_t,Dr),e(Dr,Vp),e(_t,xp),e(me,kp),$(ho,me,null),e(me,jp),$(fo,me,null),qa=!0},p(t,[g]){const As={};g&2&&(As.$$scope={dirty:g,ctx:t}),xt.$set(As);const Lr={};g&2&&(Lr.$$scope={dirty:g,ctx:t}),jt.$set(Lr);const Ar={};g&2&&(Ar.$$scope={dirty:g,ctx:t}),Et.$set(Ar);const Nr={};g&2&&(Nr.$$scope={dirty:g,ctx:t}),It.$set(Nr);const Ns={};g&2&&(Ns.$$scope={dirty:g,ctx:t}),Dt.$set(Ns);const Wr={};g&2&&(Wr.$$scope={dirty:g,ctx:t}),Lt.$set(Wr);const Sr={};g&2&&(Sr.$$scope={dirty:g,ctx:t}),At.$set(Sr);const Br={};g&2&&(Br.$$scope={dirty:g,ctx:t}),Nt.$set(Br);const Ws={};g&2&&(Ws.$$scope={dirty:g,ctx:t}),Wt.$set(Ws);const Ur={};g&2&&(Ur.$$scope={dirty:g,ctx:t}),Bt.$set(Ur);const Hr={};g&2&&(Hr.$$scope={dirty:g,ctx:t}),Ut.$set(Hr);const Kr={};g&2&&(Kr.$$scope={dirty:g,ctx:t}),Kt.$set(Kr);const Rr={};g&2&&(Rr.$$scope={dirty:g,ctx:t}),Rt.$set(Rr);const Ss={};g&2&&(Ss.$$scope={dirty:g,ctx:t}),Xt.$set(Ss);const Gr={};g&2&&(Gr.$$scope={dirty:g,ctx:t}),Zt.$set(Gr);const Xr={};g&2&&(Xr.$$scope={dirty:g,ctx:t}),Yt.$set(Xr);const Zr={};g&2&&(Zr.$$scope={dirty:g,ctx:t}),Qt.$set(Zr);const Bs={};g&2&&(Bs.$$scope={dirty:g,ctx:t}),eo.$set(Bs);const Jr={};g&2&&(Jr.$$scope={dirty:g,ctx:t}),to.$set(Jr);const E={};g&2&&(E.$$scope={dirty:g,ctx:t}),oo.$set(E);const Yr={};g&2&&(Yr.$$scope={dirty:g,ctx:t}),so.$set(Yr);const Qr={};g&2&&(Qr.$$scope={dirty:g,ctx:t}),no.$set(Qr);const ea={};g&2&&(ea.$$scope={dirty:g,ctx:t}),ao.$set(ea);const ta={};g&2&&(ta.$$scope={dirty:g,ctx:t}),io.$set(ta);const oa={};g&2&&(oa.$$scope={dirty:g,ctx:t}),co.$set(oa);const sa={};g&2&&(sa.$$scope={dirty:g,ctx:t}),po.$set(sa);const na={};g&2&&(na.$$scope={dirty:g,ctx:t}),ho.$set(na);const wt={};g&2&&(wt.$$scope={dirty:g,ctx:t}),fo.$set(wt)},i(t){qa||(O(s.$$.fragment,t),O(ge.$$.fragment,t),O(K.$$.fragment,t),O(Ao.$$.fragment,t),O(So.$$.fragment,t),O(Bo.$$.fragment,t),O(Uo.$$.fragment,t),O(Ko.$$.fragment,t),O(Ro.$$.fragment,t),O(xt.$$.fragment,t),O(Xo.$$.fragment,t),O(Zo.$$.fragment,t),O(jt.$$.fragment,t),O(Yo.$$.fragment,t),O(Qo.$$.fragment,t),O(ts.$$.fragment,t),O(Et.$$.fragment,t),O(os.$$.fragment,t),O(ss.$$.fragment,t),O(ns.$$.fragment,t),O(as.$$.fragment,t),O(ls.$$.fragment,t),O(cs.$$.fragment,t),O(ps.$$.fragment,t),O(ms.$$.fragment,t),O(It.$$.fragment,t),O(Dt.$$.fragment,t),O(hs.$$.fragment,t),O(Lt.$$.fragment,t),O(At.$$.fragment,t),O(fs.$$.fragment,t),O(Nt.$$.fragment,t),O(Wt.$$.fragment,t),O(us.$$.fragment,t),O(gs.$$.fragment,t),O(_s.$$.fragment,t),O(Bt.$$.fragment,t),O(Ut.$$.fragment,t),O(ws.$$.fragment,t),O(Ts.$$.fragment,t),O(vs.$$.fragment,t),O(Kt.$$.fragment,t),O(Rt.$$.fragment,t),O(bs.$$.fragment,t),O($s.$$.fragment,t),O(Os.$$.fragment,t),O(Xt.$$.fragment,t),O(Zt.$$.fragment,t),O(ys.$$.fragment,t),O(Vs.$$.fragment,t),O(Yt.$$.fragment,t),O(ks.$$.fragment,t),O(Qt.$$.fragment,t),O(eo.$$.fragment,t),O(js.$$.fragment,t),O(to.$$.fragment,t),O(oo.$$.fragment,t),O(Ms.$$.fragment,t),O(so.$$.fragment,t),O(no.$$.fragment,t),O(Es.$$.fragment,t),O(Fs.$$.fragment,t),O(Ps.$$.fragment,t),O(ao.$$.fragment,t),O(io.$$.fragment,t),O(zs.$$.fragment,t),O(qs.$$.fragment,t),O(Cs.$$.fragment,t),O(co.$$.fragment,t),O(po.$$.fragment,t),O(Is.$$.fragment,t),O(Ds.$$.fragment,t),O(Ls.$$.fragment,t),O(ho.$$.fragment,t),O(fo.$$.fragment,t),qa=!0)},o(t){y(s.$$.fragment,t),y(ge.$$.fragment,t),y(K.$$.fragment,t),y(Ao.$$.fragment,t),y(So.$$.fragment,t),y(Bo.$$.fragment,t),y(Uo.$$.fragment,t),y(Ko.$$.fragment,t),y(Ro.$$.fragment,t),y(xt.$$.fragment,t),y(Xo.$$.fragment,t),y(Zo.$$.fragment,t),y(jt.$$.fragment,t),y(Yo.$$.fragment,t),y(Qo.$$.fragment,t),y(ts.$$.fragment,t),y(Et.$$.fragment,t),y(os.$$.fragment,t),y(ss.$$.fragment,t),y(ns.$$.fragment,t),y(as.$$.fragment,t),y(ls.$$.fragment,t),y(cs.$$.fragment,t),y(ps.$$.fragment,t),y(ms.$$.fragment,t),y(It.$$.fragment,t),y(Dt.$$.fragment,t),y(hs.$$.fragment,t),y(Lt.$$.fragment,t),y(At.$$.fragment,t),y(fs.$$.fragment,t),y(Nt.$$.fragment,t),y(Wt.$$.fragment,t),y(us.$$.fragment,t),y(gs.$$.fragment,t),y(_s.$$.fragment,t),y(Bt.$$.fragment,t),y(Ut.$$.fragment,t),y(ws.$$.fragment,t),y(Ts.$$.fragment,t),y(vs.$$.fragment,t),y(Kt.$$.fragment,t),y(Rt.$$.fragment,t),y(bs.$$.fragment,t),y($s.$$.fragment,t),y(Os.$$.fragment,t),y(Xt.$$.fragment,t),y(Zt.$$.fragment,t),y(ys.$$.fragment,t),y(Vs.$$.fragment,t),y(Yt.$$.fragment,t),y(ks.$$.fragment,t),y(Qt.$$.fragment,t),y(eo.$$.fragment,t),y(js.$$.fragment,t),y(to.$$.fragment,t),y(oo.$$.fragment,t),y(Ms.$$.fragment,t),y(so.$$.fragment,t),y(no.$$.fragment,t),y(Es.$$.fragment,t),y(Fs.$$.fragment,t),y(Ps.$$.fragment,t),y(ao.$$.fragment,t),y(io.$$.fragment,t),y(zs.$$.fragment,t),y(qs.$$.fragment,t),y(Cs.$$.fragment,t),y(co.$$.fragment,t),y(po.$$.fragment,t),y(Is.$$.fragment,t),y(Ds.$$.fragment,t),y(Ls.$$.fragment,t),y(ho.$$.fragment,t),y(fo.$$.fragment,t),qa=!1},d(t){o(d),t&&o(w),t&&o(m),V(s),t&&o(Te),t&&o(U),V(ge),t&&o(Tt),t&&o(N),t&&o(vt),t&&o(q),t&&o(ve),t&&o(be),t&&o(bt),t&&o(H),V(K),t&&o(I),t&&o(we),t&&o(Lo),t&&o(M),t&&o(aa),V(Ao,t),t&&o(ia),t&&o(Ve),t&&o(la),t&&o(Ce),V(So),t&&o(da),t&&o(R),V(Bo),V(Uo),t&&o(ca),t&&o(De),V(Ko),t&&o(pa),t&&o(G),V(Ro),V(xt),t&&o(ma),t&&o(Ne),V(Xo),t&&o(ha),t&&o(X),V(Zo),V(jt),t&&o(fa),t&&o(Be),V(Yo),t&&o(ua),t&&o(Z),V(Qo),V(ts),V(Et),t&&o(ga),t&&o(Ue),V(os),t&&o(_a),t&&o(D),V(ss),V(ns),V(as),V(ls),t&&o(wa),t&&o(He),V(cs),t&&o(Ta),t&&o(J),V(ps),V(ms),V(It),V(Dt),V(hs),V(Lt),V(At),V(fs),V(Nt),V(Wt),t&&o(va),t&&o(Xe),V(us),t&&o(ba),t&&o(Ze),V(gs),V(_s),V(Bt),V(Ut),t&&o($a),t&&o(Ye),V(ws),t&&o(Oa),t&&o(Qe),V(Ts),V(vs),V(Kt),V(Rt),t&&o(ya),t&&o(tt),V(bs),t&&o(Va),t&&o(ot),V($s),V(Os),V(Xt),V(Zt),t&&o(xa),t&&o(nt),V(ys),t&&o(ka),t&&o(P),V(Vs),V(Yt),V(ks),V(Qt),V(eo),V(js),V(to),V(oo),V(Ms),V(so),V(no),t&&o(ja),t&&o(dt),V(Es),t&&o(Ma),t&&o(ct),V(Fs),V(Ps),V(ao),V(io),t&&o(Ea),t&&o(mt),V(zs),t&&o(Fa),t&&o(ht),V(qs),V(Cs),V(co),V(po),t&&o(Pa),t&&o(ut),V(Is),t&&o(za),t&&o(gt),V(Ds),V(Ls),V(ho),V(fo)}}}const Kh={local:"owlvit",sections:[{local:"overview",title:"Overview"},{local:"usage",title:"Usage"},{local:"transformers.OwlViTConfig",title:"OwlViTConfig"},{local:"transformers.OwlViTTextConfig",title:"OwlViTTextConfig"},{local:"transformers.OwlViTVisionConfig",title:"OwlViTVisionConfig"},{local:"transformers.OwlViTFeatureExtractor",title:"OwlViTFeatureExtractor"},{local:"transformers.OwlViTProcessor",title:"OwlViTProcessor"},{local:"transformers.OwlViTModel",title:"OwlViTModel"},{local:"transformers.OwlViTTextModel",title:"OwlViTTextModel"},{local:"transformers.OwlViTVisionModel",title:"OwlViTVisionModel"},{local:"transformers.OwlViTForObjectDetection",title:"OwlViTForObjectDetection"},{local:"transformers.TFOwlViTModel",title:"TFOwlViTModel"},{local:"transformers.TFOwlViTTextModel",title:"TFOwlViTTextModel"},{local:"transformers.TFOwlViTVisionModel",title:"TFOwlViTVisionModel"},{local:"transformers.TFOwlViTForObjectDetection",title:"TFOwlViTForObjectDetection"}],title:"OWL-ViT"};function Rh(x){return gh(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class ef extends mh{constructor(d){super();hh(this,d,Rh,Hh,fh,{})}}export{ef as default,Kh as metadata};
