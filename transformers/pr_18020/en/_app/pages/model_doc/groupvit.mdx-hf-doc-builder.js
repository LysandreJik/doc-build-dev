import{S as Dd,i as Od,s as Nd,e as s,k as f,w as $,t as i,M as Wd,c as a,d as o,m as h,a as d,x as y,h as l,b as g,G as e,g as v,y as b,q as w,o as V,B as x,v as Sd,L as se}from"../../chunks/vendor-hf-doc-builder.js";import{T as ue}from"../../chunks/Tip-hf-doc-builder.js";import{D as j}from"../../chunks/Docstring-hf-doc-builder.js";import{C as ae}from"../../chunks/CodeBlock-hf-doc-builder.js";import{I as me}from"../../chunks/IconCopyLink-hf-doc-builder.js";import{E as re}from"../../chunks/ExampleCodeBlock-hf-doc-builder.js";function Bd(k){let r,T,u,c,m;return c=new ae({props:{code:`from transformers import GroupViTTextConfig, GroupViTTextModel

# Initializing a GroupViTTextModel with nvidia/groupvit-gcc-yfcc style configuration
configuration = GroupViTTextConfig()

model = GroupViTTextModel(configuration)

# Accessing the model configuration
configuration = model.config`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> GroupViTTextConfig, GroupViTTextModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a GroupViTTextModel with nvidia/groupvit-gcc-yfcc style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = GroupViTTextConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span>model = GroupViTTextModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`}}),{c(){r=s("p"),T=i("Example:"),u=f(),$(c.$$.fragment)},l(n){r=a(n,"P",{});var p=d(r);T=l(p,"Example:"),p.forEach(o),u=h(n),y(c.$$.fragment,n)},m(n,p){v(n,r,p),e(r,T),v(n,u,p),b(c,n,p),m=!0},p:se,i(n){m||(w(c.$$.fragment,n),m=!0)},o(n){V(c.$$.fragment,n),m=!1},d(n){n&&o(r),n&&o(u),x(c,n)}}}function Ud(k){let r,T,u,c,m;return c=new ae({props:{code:`from transformers import GroupViTVisionConfig, GroupViTVisionModel

# Initializing a GroupViTVisionModel with nvidia/groupvit-gcc-yfcc style configuration
configuration = GroupViTVisionConfig()

model = GroupViTVisionModel(configuration)

# Accessing the model configuration
configuration = model.config`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> GroupViTVisionConfig, GroupViTVisionModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a GroupViTVisionModel with nvidia/groupvit-gcc-yfcc style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = GroupViTVisionConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span>model = GroupViTVisionModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`}}),{c(){r=s("p"),T=i("Example:"),u=f(),$(c.$$.fragment)},l(n){r=a(n,"P",{});var p=d(r);T=l(p,"Example:"),p.forEach(o),u=h(n),y(c.$$.fragment,n)},m(n,p){v(n,r,p),e(r,T),v(n,u,p),b(c,n,p),m=!0},p:se,i(n){m||(w(c.$$.fragment,n),m=!0)},o(n){V(c.$$.fragment,n),m=!1},d(n){n&&o(r),n&&o(u),x(c,n)}}}function Kd(k){let r,T,u,c,m;return{c(){r=s("p"),T=i("Although the recipe for forward pass needs to be defined within this function, one should call the "),u=s("code"),c=i("Module"),m=i(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(n){r=a(n,"P",{});var p=d(r);T=l(p,"Although the recipe for forward pass needs to be defined within this function, one should call the "),u=a(p,"CODE",{});var G=d(u);c=l(G,"Module"),G.forEach(o),m=l(p,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),p.forEach(o)},m(n,p){v(n,r,p),e(r,T),e(r,u),e(u,c),e(r,m)},d(n){n&&o(r)}}}function Hd(k){let r,T,u,c,m;return c=new ae({props:{code:`from PIL import Image
import requests
from transformers import AutoProcessor, GroupViTModel

model = GroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")
processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(
    text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, GroupViTModel

<span class="hljs-meta">&gt;&gt;&gt; </span>model = GroupViTModel.from_pretrained(<span class="hljs-string">&quot;nvidia/groupvit-gcc-yfcc&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(<span class="hljs-string">&quot;nvidia/groupvit-gcc-yfcc&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(
<span class="hljs-meta">... </span>    text=[<span class="hljs-string">&quot;a photo of a cat&quot;</span>, <span class="hljs-string">&quot;a photo of a dog&quot;</span>], images=image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>logits_per_image = outputs.logits_per_image  <span class="hljs-comment"># this is the image-text similarity score</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>probs = logits_per_image.softmax(dim=<span class="hljs-number">1</span>)  <span class="hljs-comment"># we can take the softmax to get the label probabilities</span>`}}),{c(){r=s("p"),T=i("Examples:"),u=f(),$(c.$$.fragment)},l(n){r=a(n,"P",{});var p=d(r);T=l(p,"Examples:"),p.forEach(o),u=h(n),y(c.$$.fragment,n)},m(n,p){v(n,r,p),e(r,T),v(n,u,p),b(c,n,p),m=!0},p:se,i(n){m||(w(c.$$.fragment,n),m=!0)},o(n){V(c.$$.fragment,n),m=!1},d(n){n&&o(r),n&&o(u),x(c,n)}}}function Rd(k){let r,T,u,c,m;return{c(){r=s("p"),T=i("Although the recipe for forward pass needs to be defined within this function, one should call the "),u=s("code"),c=i("Module"),m=i(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(n){r=a(n,"P",{});var p=d(r);T=l(p,"Although the recipe for forward pass needs to be defined within this function, one should call the "),u=a(p,"CODE",{});var G=d(u);c=l(G,"Module"),G.forEach(o),m=l(p,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),p.forEach(o)},m(n,p){v(n,r,p),e(r,T),e(r,u),e(u,c),e(r,m)},d(n){n&&o(r)}}}function Jd(k){let r,T,u,c,m;return c=new ae({props:{code:`from transformers import CLIPTokenizer, GroupViTModel

model = GroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")
tokenizer = CLIPTokenizer.from_pretrained("nvidia/groupvit-gcc-yfcc")

inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
text_features = model.get_text_features(**inputs)`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> CLIPTokenizer, GroupViTModel

<span class="hljs-meta">&gt;&gt;&gt; </span>model = GroupViTModel.from_pretrained(<span class="hljs-string">&quot;nvidia/groupvit-gcc-yfcc&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = CLIPTokenizer.from_pretrained(<span class="hljs-string">&quot;nvidia/groupvit-gcc-yfcc&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer([<span class="hljs-string">&quot;a photo of a cat&quot;</span>, <span class="hljs-string">&quot;a photo of a dog&quot;</span>], padding=<span class="hljs-literal">True</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>text_features = model.get_text_features(**inputs)`}}),{c(){r=s("p"),T=i("Examples:"),u=f(),$(c.$$.fragment)},l(n){r=a(n,"P",{});var p=d(r);T=l(p,"Examples:"),p.forEach(o),u=h(n),y(c.$$.fragment,n)},m(n,p){v(n,r,p),e(r,T),v(n,u,p),b(c,n,p),m=!0},p:se,i(n){m||(w(c.$$.fragment,n),m=!0)},o(n){V(c.$$.fragment,n),m=!1},d(n){n&&o(r),n&&o(u),x(c,n)}}}function Xd(k){let r,T,u,c,m;return{c(){r=s("p"),T=i("Although the recipe for forward pass needs to be defined within this function, one should call the "),u=s("code"),c=i("Module"),m=i(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(n){r=a(n,"P",{});var p=d(r);T=l(p,"Although the recipe for forward pass needs to be defined within this function, one should call the "),u=a(p,"CODE",{});var G=d(u);c=l(G,"Module"),G.forEach(o),m=l(p,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),p.forEach(o)},m(n,p){v(n,r,p),e(r,T),e(r,u),e(u,c),e(r,m)},d(n){n&&o(r)}}}function Yd(k){let r,T,u,c,m;return c=new ae({props:{code:`from PIL import Image
import requests
from transformers import AutoProcessor, GroupViTModel

model = GroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")
processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")

image_features = model.get_image_features(**inputs)`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, GroupViTModel

<span class="hljs-meta">&gt;&gt;&gt; </span>model = GroupViTModel.from_pretrained(<span class="hljs-string">&quot;nvidia/groupvit-gcc-yfcc&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(<span class="hljs-string">&quot;nvidia/groupvit-gcc-yfcc&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(images=image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>image_features = model.get_image_features(**inputs)`}}),{c(){r=s("p"),T=i("Examples:"),u=f(),$(c.$$.fragment)},l(n){r=a(n,"P",{});var p=d(r);T=l(p,"Examples:"),p.forEach(o),u=h(n),y(c.$$.fragment,n)},m(n,p){v(n,r,p),e(r,T),v(n,u,p),b(c,n,p),m=!0},p:se,i(n){m||(w(c.$$.fragment,n),m=!0)},o(n){V(c.$$.fragment,n),m=!1},d(n){n&&o(r),n&&o(u),x(c,n)}}}function Qd(k){let r,T,u,c,m;return{c(){r=s("p"),T=i("Although the recipe for forward pass needs to be defined within this function, one should call the "),u=s("code"),c=i("Module"),m=i(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(n){r=a(n,"P",{});var p=d(r);T=l(p,"Although the recipe for forward pass needs to be defined within this function, one should call the "),u=a(p,"CODE",{});var G=d(u);c=l(G,"Module"),G.forEach(o),m=l(p,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),p.forEach(o)},m(n,p){v(n,r,p),e(r,T),e(r,u),e(u,c),e(r,m)},d(n){n&&o(r)}}}function Zd(k){let r,T,u,c,m;return c=new ae({props:{code:`from transformers import CLIPTokenizer, GroupViTTextModel

tokenizer = CLIPTokenizer.from_pretrained("nvidia/groupvit-gcc-yfcc")
model = GroupViTTextModel.from_pretrained("nvidia/groupvit-gcc-yfcc")

inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
pooled_output = outputs.pooler_output  # pooled (EOS token) states`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> CLIPTokenizer, GroupViTTextModel

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = CLIPTokenizer.from_pretrained(<span class="hljs-string">&quot;nvidia/groupvit-gcc-yfcc&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = GroupViTTextModel.from_pretrained(<span class="hljs-string">&quot;nvidia/groupvit-gcc-yfcc&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer([<span class="hljs-string">&quot;a photo of a cat&quot;</span>, <span class="hljs-string">&quot;a photo of a dog&quot;</span>], padding=<span class="hljs-literal">True</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_state = outputs.last_hidden_state
<span class="hljs-meta">&gt;&gt;&gt; </span>pooled_output = outputs.pooler_output  <span class="hljs-comment"># pooled (EOS token) states</span>`}}),{c(){r=s("p"),T=i("Examples:"),u=f(),$(c.$$.fragment)},l(n){r=a(n,"P",{});var p=d(r);T=l(p,"Examples:"),p.forEach(o),u=h(n),y(c.$$.fragment,n)},m(n,p){v(n,r,p),e(r,T),v(n,u,p),b(c,n,p),m=!0},p:se,i(n){m||(w(c.$$.fragment,n),m=!0)},o(n){V(c.$$.fragment,n),m=!1},d(n){n&&o(r),n&&o(u),x(c,n)}}}function ep(k){let r,T,u,c,m;return{c(){r=s("p"),T=i("Although the recipe for forward pass needs to be defined within this function, one should call the "),u=s("code"),c=i("Module"),m=i(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(n){r=a(n,"P",{});var p=d(r);T=l(p,"Although the recipe for forward pass needs to be defined within this function, one should call the "),u=a(p,"CODE",{});var G=d(u);c=l(G,"Module"),G.forEach(o),m=l(p,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),p.forEach(o)},m(n,p){v(n,r,p),e(r,T),e(r,u),e(u,c),e(r,m)},d(n){n&&o(r)}}}function tp(k){let r,T,u,c,m;return c=new ae({props:{code:`from PIL import Image
import requests
from transformers import AutoProcessor, GroupViTVisionModel

processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")
model = GroupViTVisionModel.from_pretrained("nvidia/groupvit-gcc-yfcc")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
pooled_output = outputs.pooler_output  # pooled CLS states`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, GroupViTVisionModel

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(<span class="hljs-string">&quot;nvidia/groupvit-gcc-yfcc&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = GroupViTVisionModel.from_pretrained(<span class="hljs-string">&quot;nvidia/groupvit-gcc-yfcc&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(images=image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_state = outputs.last_hidden_state
<span class="hljs-meta">&gt;&gt;&gt; </span>pooled_output = outputs.pooler_output  <span class="hljs-comment"># pooled CLS states</span>`}}),{c(){r=s("p"),T=i("Examples:"),u=f(),$(c.$$.fragment)},l(n){r=a(n,"P",{});var p=d(r);T=l(p,"Examples:"),p.forEach(o),u=h(n),y(c.$$.fragment,n)},m(n,p){v(n,r,p),e(r,T),v(n,u,p),b(c,n,p),m=!0},p:se,i(n){m||(w(c.$$.fragment,n),m=!0)},o(n){V(c.$$.fragment,n),m=!1},d(n){n&&o(r),n&&o(u),x(c,n)}}}function op(k){let r,T,u,c,m,n,p,G,ao,xe,E,U,ie,le,io,de,lo,lt,q,fe,pe,dt,L,F,po,ke,pt,he,Ge,ct,ge,A,co,_e,Te,uo,ve,K,N,$e,Me,Ee,mo;return{c(){r=s("p"),T=i("TF 2.0 models accepts two formats as inputs:"),u=f(),c=s("ul"),m=s("li"),n=i("having all inputs as keyword arguments (like PyTorch models), or"),p=f(),G=s("li"),ao=i("having all inputs as a list, tuple or dict in the first positional arguments."),xe=f(),E=s("p"),U=i("This second option is useful when using "),ie=s("code"),le=i("tf.keras.Model.fit"),io=i(` method which currently requires having all the
tensors in the first argument of the model call function: `),de=s("code"),lo=i("model(inputs)"),lt=i("."),q=f(),fe=s("p"),pe=i(`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),dt=f(),L=s("ul"),F=s("li"),po=i("a single Tensor with "),ke=s("code"),pt=i("input_ids"),he=i(" only and nothing else: "),Ge=s("code"),ct=i("model(input_ids)"),ge=f(),A=s("li"),co=i(`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),_e=s("code"),Te=i("model([input_ids, attention_mask])"),uo=i(" or "),ve=s("code"),K=i("model([input_ids, attention_mask, token_type_ids])"),N=f(),$e=s("li"),Me=i(`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),Ee=s("code"),mo=i('model({"input_ids": input_ids, "token_type_ids": token_type_ids})')},l(M){r=a(M,"P",{});var z=d(r);T=l(z,"TF 2.0 models accepts two formats as inputs:"),z.forEach(o),u=h(M),c=a(M,"UL",{});var ut=d(c);m=a(ut,"LI",{});var rn=d(m);n=l(rn,"having all inputs as keyword arguments (like PyTorch models), or"),rn.forEach(o),p=h(ut),G=a(ut,"LI",{});var H=d(G);ao=l(H,"having all inputs as a list, tuple or dict in the first positional arguments."),H.forEach(o),ut.forEach(o),xe=h(M),E=a(M,"P",{});var ye=d(E);U=l(ye,"This second option is useful when using "),ie=a(ye,"CODE",{});var be=d(ie);le=l(be,"tf.keras.Model.fit"),be.forEach(o),io=l(ye,` method which currently requires having all the
tensors in the first argument of the model call function: `),de=a(ye,"CODE",{});var sn=d(de);lo=l(sn,"model(inputs)"),sn.forEach(o),lt=l(ye,"."),ye.forEach(o),q=h(M),fe=a(M,"P",{});var an=d(fe);pe=l(an,`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),an.forEach(o),dt=h(M),L=a(M,"UL",{});var D=d(L);F=a(D,"LI",{});var je=d(F);po=l(je,"a single Tensor with "),ke=a(je,"CODE",{});var ln=d(ke);pt=l(ln,"input_ids"),ln.forEach(o),he=l(je," only and nothing else: "),Ge=a(je,"CODE",{});var fo=d(Ge);ct=l(fo,"model(input_ids)"),fo.forEach(o),je.forEach(o),ge=h(D),A=a(D,"LI",{});var C=d(A);co=l(C,`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),_e=a(C,"CODE",{});var dn=d(_e);Te=l(dn,"model([input_ids, attention_mask])"),dn.forEach(o),uo=l(C," or "),ve=a(C,"CODE",{});var we=d(ve);K=l(we,"model([input_ids, attention_mask, token_type_ids])"),we.forEach(o),C.forEach(o),N=h(D),$e=a(D,"LI",{});var ho=d($e);Me=l(ho,`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),Ee=a(ho,"CODE",{});var pn=d(Ee);mo=l(pn,'model({"input_ids": input_ids, "token_type_ids": token_type_ids})'),pn.forEach(o),ho.forEach(o),D.forEach(o)},m(M,z){v(M,r,z),e(r,T),v(M,u,z),v(M,c,z),e(c,m),e(m,n),e(c,p),e(c,G),e(G,ao),v(M,xe,z),v(M,E,z),e(E,U),e(E,ie),e(ie,le),e(E,io),e(E,de),e(de,lo),e(E,lt),v(M,q,z),v(M,fe,z),e(fe,pe),v(M,dt,z),v(M,L,z),e(L,F),e(F,po),e(F,ke),e(ke,pt),e(F,he),e(F,Ge),e(Ge,ct),e(L,ge),e(L,A),e(A,co),e(A,_e),e(_e,Te),e(A,uo),e(A,ve),e(ve,K),e(L,N),e(L,$e),e($e,Me),e($e,Ee),e(Ee,mo)},d(M){M&&o(r),M&&o(u),M&&o(c),M&&o(xe),M&&o(E),M&&o(q),M&&o(fe),M&&o(dt),M&&o(L)}}}function np(k){let r,T,u,c,m;return{c(){r=s("p"),T=i("Although the recipe for forward pass needs to be defined within this function, one should call the "),u=s("code"),c=i("Module"),m=i(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(n){r=a(n,"P",{});var p=d(r);T=l(p,"Although the recipe for forward pass needs to be defined within this function, one should call the "),u=a(p,"CODE",{});var G=d(u);c=l(G,"Module"),G.forEach(o),m=l(p,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),p.forEach(o)},m(n,p){v(n,r,p),e(r,T),e(r,u),e(u,c),e(r,m)},d(n){n&&o(r)}}}function rp(k){let r,T,u,c,m;return c=new ae({props:{code:`from PIL import Image
import requests
from transformers import AutoProcessor, TFGroupViTModel

model = TFGroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")
processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(
    text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="tf", padding=True
)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, TFGroupViTModel

<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFGroupViTModel.from_pretrained(<span class="hljs-string">&quot;nvidia/groupvit-gcc-yfcc&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(<span class="hljs-string">&quot;nvidia/groupvit-gcc-yfcc&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(
<span class="hljs-meta">... </span>    text=[<span class="hljs-string">&quot;a photo of a cat&quot;</span>, <span class="hljs-string">&quot;a photo of a dog&quot;</span>], images=image, return_tensors=<span class="hljs-string">&quot;tf&quot;</span>, padding=<span class="hljs-literal">True</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>logits_per_image = outputs.logits_per_image  <span class="hljs-comment"># this is the image-text similarity score</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>probs = logits_per_image.softmax(dim=<span class="hljs-number">1</span>)  <span class="hljs-comment"># we can take the softmax to get the label probabilities</span>`}}),{c(){r=s("p"),T=i("Examples:"),u=f(),$(c.$$.fragment)},l(n){r=a(n,"P",{});var p=d(r);T=l(p,"Examples:"),p.forEach(o),u=h(n),y(c.$$.fragment,n)},m(n,p){v(n,r,p),e(r,T),v(n,u,p),b(c,n,p),m=!0},p:se,i(n){m||(w(c.$$.fragment,n),m=!0)},o(n){V(c.$$.fragment,n),m=!1},d(n){n&&o(r),n&&o(u),x(c,n)}}}function sp(k){let r,T,u,c,m;return{c(){r=s("p"),T=i("Although the recipe for forward pass needs to be defined within this function, one should call the "),u=s("code"),c=i("Module"),m=i(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(n){r=a(n,"P",{});var p=d(r);T=l(p,"Although the recipe for forward pass needs to be defined within this function, one should call the "),u=a(p,"CODE",{});var G=d(u);c=l(G,"Module"),G.forEach(o),m=l(p,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),p.forEach(o)},m(n,p){v(n,r,p),e(r,T),e(r,u),e(u,c),e(r,m)},d(n){n&&o(r)}}}function ap(k){let r,T,u,c,m;return c=new ae({props:{code:`from transformers import CLIPTokenizer, TFGroupViTModel

model = TFGroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")
tokenizer = CLIPTokenizer.from_pretrained("nvidia/groupvit-gcc-yfcc")

inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="tf")
text_features = model.get_text_features(**inputs)`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> CLIPTokenizer, TFGroupViTModel

<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFGroupViTModel.from_pretrained(<span class="hljs-string">&quot;nvidia/groupvit-gcc-yfcc&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = CLIPTokenizer.from_pretrained(<span class="hljs-string">&quot;nvidia/groupvit-gcc-yfcc&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer([<span class="hljs-string">&quot;a photo of a cat&quot;</span>, <span class="hljs-string">&quot;a photo of a dog&quot;</span>], padding=<span class="hljs-literal">True</span>, return_tensors=<span class="hljs-string">&quot;tf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>text_features = model.get_text_features(**inputs)`}}),{c(){r=s("p"),T=i("Examples:"),u=f(),$(c.$$.fragment)},l(n){r=a(n,"P",{});var p=d(r);T=l(p,"Examples:"),p.forEach(o),u=h(n),y(c.$$.fragment,n)},m(n,p){v(n,r,p),e(r,T),v(n,u,p),b(c,n,p),m=!0},p:se,i(n){m||(w(c.$$.fragment,n),m=!0)},o(n){V(c.$$.fragment,n),m=!1},d(n){n&&o(r),n&&o(u),x(c,n)}}}function ip(k){let r,T,u,c,m;return{c(){r=s("p"),T=i("Although the recipe for forward pass needs to be defined within this function, one should call the "),u=s("code"),c=i("Module"),m=i(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(n){r=a(n,"P",{});var p=d(r);T=l(p,"Although the recipe for forward pass needs to be defined within this function, one should call the "),u=a(p,"CODE",{});var G=d(u);c=l(G,"Module"),G.forEach(o),m=l(p,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),p.forEach(o)},m(n,p){v(n,r,p),e(r,T),e(r,u),e(u,c),e(r,m)},d(n){n&&o(r)}}}function lp(k){let r,T,u,c,m;return c=new ae({props:{code:`from PIL import Image
import requests
from transformers import AutoProcessor, TFGroupViTModel

model = TFGroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")
processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="tf")

image_features = model.get_image_features(**inputs)`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, TFGroupViTModel

<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFGroupViTModel.from_pretrained(<span class="hljs-string">&quot;nvidia/groupvit-gcc-yfcc&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(<span class="hljs-string">&quot;nvidia/groupvit-gcc-yfcc&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(images=image, return_tensors=<span class="hljs-string">&quot;tf&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>image_features = model.get_image_features(**inputs)`}}),{c(){r=s("p"),T=i("Examples:"),u=f(),$(c.$$.fragment)},l(n){r=a(n,"P",{});var p=d(r);T=l(p,"Examples:"),p.forEach(o),u=h(n),y(c.$$.fragment,n)},m(n,p){v(n,r,p),e(r,T),v(n,u,p),b(c,n,p),m=!0},p:se,i(n){m||(w(c.$$.fragment,n),m=!0)},o(n){V(c.$$.fragment,n),m=!1},d(n){n&&o(r),n&&o(u),x(c,n)}}}function dp(k){let r,T,u,c,m;return{c(){r=s("p"),T=i("Although the recipe for forward pass needs to be defined within this function, one should call the "),u=s("code"),c=i("Module"),m=i(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(n){r=a(n,"P",{});var p=d(r);T=l(p,"Although the recipe for forward pass needs to be defined within this function, one should call the "),u=a(p,"CODE",{});var G=d(u);c=l(G,"Module"),G.forEach(o),m=l(p,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),p.forEach(o)},m(n,p){v(n,r,p),e(r,T),e(r,u),e(u,c),e(r,m)},d(n){n&&o(r)}}}function pp(k){let r,T,u,c,m;return c=new ae({props:{code:`from transformers import CLIPTokenizer, GroupViTTextModel

tokenizer = CLIPTokenizer.from_pretrained("nvidia/groupvit-gcc-yfcc")
model = GroupViTTextModel.from_pretrained("nvidia/groupvit-gcc-yfcc")

inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
pooled_output = outputs.pooler_output  # pooled (EOS token) states`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> CLIPTokenizer, GroupViTTextModel

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = CLIPTokenizer.from_pretrained(<span class="hljs-string">&quot;nvidia/groupvit-gcc-yfcc&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = GroupViTTextModel.from_pretrained(<span class="hljs-string">&quot;nvidia/groupvit-gcc-yfcc&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer([<span class="hljs-string">&quot;a photo of a cat&quot;</span>, <span class="hljs-string">&quot;a photo of a dog&quot;</span>], padding=<span class="hljs-literal">True</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_state = outputs.last_hidden_state
<span class="hljs-meta">&gt;&gt;&gt; </span>pooled_output = outputs.pooler_output  <span class="hljs-comment"># pooled (EOS token) states</span>`}}),{c(){r=s("p"),T=i("Examples:"),u=f(),$(c.$$.fragment)},l(n){r=a(n,"P",{});var p=d(r);T=l(p,"Examples:"),p.forEach(o),u=h(n),y(c.$$.fragment,n)},m(n,p){v(n,r,p),e(r,T),v(n,u,p),b(c,n,p),m=!0},p:se,i(n){m||(w(c.$$.fragment,n),m=!0)},o(n){V(c.$$.fragment,n),m=!1},d(n){n&&o(r),n&&o(u),x(c,n)}}}function cp(k){let r,T,u,c,m;return{c(){r=s("p"),T=i("Although the recipe for forward pass needs to be defined within this function, one should call the "),u=s("code"),c=i("Module"),m=i(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(n){r=a(n,"P",{});var p=d(r);T=l(p,"Although the recipe for forward pass needs to be defined within this function, one should call the "),u=a(p,"CODE",{});var G=d(u);c=l(G,"Module"),G.forEach(o),m=l(p,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),p.forEach(o)},m(n,p){v(n,r,p),e(r,T),e(r,u),e(u,c),e(r,m)},d(n){n&&o(r)}}}function up(k){let r,T,u,c,m;return c=new ae({props:{code:`from PIL import Image
import requests
from transformers import AutoProcessor, TFGroupViTVisionModel

processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")
model = TFGroupViTVisionModel.from_pretrained("nvidia/groupvit-gcc-yfcc")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
pooled_output = outputs.pooler_output  # pooled CLS states`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, TFGroupViTVisionModel

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(<span class="hljs-string">&quot;nvidia/groupvit-gcc-yfcc&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFGroupViTVisionModel.from_pretrained(<span class="hljs-string">&quot;nvidia/groupvit-gcc-yfcc&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(images=image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_state = outputs.last_hidden_state
<span class="hljs-meta">&gt;&gt;&gt; </span>pooled_output = outputs.pooler_output  <span class="hljs-comment"># pooled CLS states</span>`}}),{c(){r=s("p"),T=i("Examples:"),u=f(),$(c.$$.fragment)},l(n){r=a(n,"P",{});var p=d(r);T=l(p,"Examples:"),p.forEach(o),u=h(n),y(c.$$.fragment,n)},m(n,p){v(n,r,p),e(r,T),v(n,u,p),b(c,n,p),m=!0},p:se,i(n){m||(w(c.$$.fragment,n),m=!0)},o(n){V(c.$$.fragment,n),m=!1},d(n){n&&o(r),n&&o(u),x(c,n)}}}function mp(k){let r,T,u,c,m,n,p,G,ao,xe,E,U,ie,le,io,de,lo,lt,q,fe,pe,dt,L,F,po,ke,pt,he,Ge,ct,ge,A,co,_e,Te,uo,ve,K,N,$e,Me,Ee,mo,M,z,ut,rn,H,ye,be,sn,an,D,je,ln,fo,C,dn,we,ho,pn,go,bs,ws,_o,Vs,xs,Dr,Pe,mt,Rn,To,ks,Jn,Gs,Or,W,vo,Ms,ft,cn,Es,js,un,Ps,Cs,zs,Ce,qs,mn,Fs,Is,fn,Ls,As,Ds,ht,$o,Os,yo,Ns,hn,Ws,Ss,Nr,ze,gt,Xn,bo,Bs,Yn,Us,Wr,S,wo,Ks,qe,Hs,gn,Rs,Js,Vo,Xs,Ys,Qs,Fe,Zs,_n,ea,ta,Tn,oa,na,ra,_t,Sr,Ie,Tt,Qn,xo,sa,Zn,aa,Br,B,ko,ia,Le,la,vn,da,pa,Go,ca,ua,ma,Ae,fa,$n,ha,ga,yn,_a,Ta,va,vt,Ur,De,$t,er,Mo,$a,tr,ya,Kr,I,Eo,ba,jo,wa,Po,Va,xa,ka,R,Co,Ga,Oe,Ma,bn,Ea,ja,or,Pa,Ca,za,yt,qa,bt,Fa,J,zo,Ia,Ne,La,wn,Aa,Da,nr,Oa,Na,Wa,wt,Sa,Vt,Ba,X,qo,Ua,We,Ka,Vn,Ha,Ra,rr,Ja,Xa,Ya,xt,Qa,kt,Hr,Se,Gt,sr,Fo,Za,ar,ei,Rr,Be,Io,ti,Y,Lo,oi,Ue,ni,xn,ri,si,ir,ai,ii,li,Mt,di,Et,Jr,Ke,jt,lr,Ao,pi,dr,ci,Xr,He,Do,ui,Q,Oo,mi,Re,fi,kn,hi,gi,pr,_i,Ti,vi,Pt,$i,Ct,Yr,Je,zt,cr,No,yi,ur,bi,Qr,P,Wo,wi,So,Vi,Gn,xi,ki,Gi,Bo,Mi,Uo,Ei,ji,Pi,qt,Ci,Z,Ko,zi,Xe,qi,Mn,Fi,Ii,mr,Li,Ai,Di,Ft,Oi,It,Ni,ee,Ho,Wi,Ye,Si,En,Bi,Ui,fr,Ki,Hi,Ri,Lt,Ji,At,Xi,te,Ro,Yi,Qe,Qi,jn,Zi,el,hr,tl,ol,nl,Dt,rl,Ot,Zr,Ze,Nt,gr,Jo,sl,_r,al,es,et,Xo,il,oe,Yo,ll,tt,dl,Pn,pl,cl,Tr,ul,ml,fl,Wt,hl,St,ts,ot,Bt,vr,Qo,gl,$r,_l,os,nt,Zo,Tl,ne,en,vl,rt,$l,Cn,yl,bl,yr,wl,Vl,xl,Ut,kl,Kt,ns;return n=new me({}),le=new me({}),To=new me({}),vo=new j({props:{name:"class transformers.GroupViTConfig",anchor:"transformers.GroupViTConfig",parameters:[{name:"text_config_dict",val:" = None"},{name:"vision_config_dict",val:" = None"},{name:"projection_dim",val:" = 256"},{name:"projection_intermediate_dim",val:" = 4096"},{name:"logit_scale_init_value",val:" = 2.6592"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.GroupViTConfig.text_config_dict",description:`<strong>text_config_dict</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
Dictionary of configuration options used to initialize <a href="/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.GroupViTTextConfig">GroupViTTextConfig</a>.`,name:"text_config_dict"},{anchor:"transformers.GroupViTConfig.vision_config_dict",description:`<strong>vision_config_dict</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
Dictionary of configuration options used to initialize <a href="/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.GroupViTVisionConfig">GroupViTVisionConfig</a>.`,name:"vision_config_dict"},{anchor:"transformers.GroupViTConfig.projection_dim",description:`<strong>projection_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 256) &#x2014;
Dimentionality of text and vision projection layers.`,name:"projection_dim"},{anchor:"transformers.GroupViTConfig.projection_intermediate_dim",description:`<strong>projection_intermediate_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimentionality of intermediate layer of text and vision projection layers.`,name:"projection_intermediate_dim"},{anchor:"transformers.GroupViTConfig.logit_scale_init_value",description:`<strong>logit_scale_init_value</strong> (<code>float</code>, <em>optional</em>, defaults to 2.6592) &#x2014;
The inital value of the <em>logit_scale</em> parameter. Default is used as per the original GroupViT
implementation.`,name:"logit_scale_init_value"},{anchor:"transformers.GroupViTConfig.kwargs",description:`<strong>kwargs</strong> (<em>optional</em>) &#x2014;
Dictionary of keyword arguments.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/vr_18020/src/transformers/models/groupvit/configuration_groupvit.py#L265"}}),$o=new j({props:{name:"from_text_vision_configs",anchor:"transformers.GroupViTConfig.from_text_vision_configs",parameters:[{name:"text_config",val:": GroupViTTextConfig"},{name:"vision_config",val:": GroupViTVisionConfig"},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18020/src/transformers/models/groupvit/configuration_groupvit.py#L322",returnDescription:`
<p>An instance of a configuration object</p>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.GroupViTConfig"
>GroupViTConfig</a></p>
`}}),bo=new me({}),wo=new j({props:{name:"class transformers.GroupViTTextConfig",anchor:"transformers.GroupViTTextConfig",parameters:[{name:"vocab_size",val:" = 49408"},{name:"hidden_size",val:" = 256"},{name:"intermediate_size",val:" = 1024"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 4"},{name:"max_position_embeddings",val:" = 77"},{name:"hidden_act",val:" = 'quick_gelu'"},{name:"layer_norm_eps",val:" = 1e-05"},{name:"dropout",val:" = 0.0"},{name:"attention_dropout",val:" = 0.0"},{name:"initializer_range",val:" = 0.02"},{name:"initializer_factor",val:" = 1.0"},{name:"pad_token_id",val:" = 1"},{name:"bos_token_id",val:" = 0"},{name:"eos_token_id",val:" = 2"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.GroupViTTextConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 49408) &#x2014;
Vocabulary size of the GroupViT text model. Defines the number of different tokens that can be represented
by the <code>inputs_ids</code> passed when calling <a href="/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.GroupViTModel">GroupViTModel</a>.`,name:"vocab_size"},{anchor:"transformers.GroupViTTextConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 256) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.GroupViTTextConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (i.e., feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.GroupViTTextConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.GroupViTTextConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 4) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.GroupViTTextConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 77) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.GroupViTTextConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;quick_gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;selu&quot;</code> and <code>&quot;gelu_new&quot;</code> \`<code>&quot;quick_gelu&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.GroupViTTextConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-5) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.GroupViTTextConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.GroupViTTextConfig.dropout",description:`<strong>dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"dropout"},{anchor:"transformers.GroupViTTextConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.GroupViTTextConfig.initializer_factor",description:`<strong>initializer_factor</strong> (<code>float</code>, <em>optional</em>, defaults to 1.0) &#x2014;
A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
testing).`,name:"initializer_factor"}],source:"https://github.com/huggingface/transformers/blob/vr_18020/src/transformers/models/groupvit/configuration_groupvit.py#L32"}}),_t=new re({props:{anchor:"transformers.GroupViTTextConfig.example",$$slots:{default:[Bd]},$$scope:{ctx:k}}}),xo=new me({}),ko=new j({props:{name:"class transformers.GroupViTVisionConfig",anchor:"transformers.GroupViTVisionConfig",parameters:[{name:"hidden_size",val:" = 384"},{name:"intermediate_size",val:" = 1536"},{name:"depths",val:" = [6, 3, 3]"},{name:"num_hidden_layers",val:" = 12"},{name:"num_group_tokens",val:" = [64, 8, 0]"},{name:"num_output_groups",val:" = [64, 8, 8]"},{name:"num_attention_heads",val:" = 6"},{name:"image_size",val:" = 224"},{name:"patch_size",val:" = 16"},{name:"num_channels",val:" = 3"},{name:"hidden_act",val:" = 'gelu'"},{name:"layer_norm_eps",val:" = 1e-05"},{name:"dropout",val:" = 0.0"},{name:"attention_dropout",val:" = 0.0"},{name:"initializer_range",val:" = 0.02"},{name:"initializer_factor",val:" = 1.0"},{name:"assign_eps",val:" = 1.0"},{name:"assign_mlp_ratio",val:" = [0.5, 4]"},{name:"qkv_bias",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.GroupViTVisionConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 384) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.GroupViTVisionConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 1536) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (i.e., feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.GroupViTVisionConfig.depths",description:`<strong>depths</strong> (<code>List[int]</code>, <em>optional</em>, defaults to [6, 3, 3]) &#x2014;
The number of layers in each encoder block.`,name:"depths"},{anchor:"transformers.GroupViTVisionConfig.num_group_tokens",description:`<strong>num_group_tokens</strong> (<code>List[int]</code>, <em>optional</em>, defaults to [64, 8, 0]) &#x2014;
The number of group tokens for each stage.`,name:"num_group_tokens"},{anchor:"transformers.GroupViTVisionConfig.num_output_groups",description:`<strong>num_output_groups</strong> (<code>List[int]</code>, <em>optional</em>, defaults to [64, 8, 8]) &#x2014;
The number of output groups for each stage, 0 means no group.`,name:"num_output_groups"},{anchor:"transformers.GroupViTVisionConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 6) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.GroupViTVisionConfig.image_size",description:`<strong>image_size</strong> (<code>int</code>, <em>optional</em>, defaults to 224) &#x2014;
The size (resolution) of each image.`,name:"image_size"},{anchor:"transformers.GroupViTVisionConfig.patch_size",description:`<strong>patch_size</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
The size (resolution) of each patch.`,name:"patch_size"},{anchor:"transformers.GroupViTVisionConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;selu&quot;</code> and <code>&quot;gelu_new&quot;</code> \`<code>&quot;quick_gelu&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.GroupViTVisionConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-5) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.GroupViTVisionConfig.dropout",description:`<strong>dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.`,name:"dropout"},{anchor:"transformers.GroupViTVisionConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.GroupViTVisionConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.GroupViTVisionConfig.initializer_factor",description:`<strong>initializer_factor</strong> (<code>float</code>, <em>optional</em>, defaults to 1.0) &#x2014;
A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
testing).`,name:"initializer_factor"}],source:"https://github.com/huggingface/transformers/blob/vr_18020/src/transformers/models/groupvit/configuration_groupvit.py#L139"}}),vt=new re({props:{anchor:"transformers.GroupViTVisionConfig.example",$$slots:{default:[Ud]},$$scope:{ctx:k}}}),Mo=new me({}),Eo=new j({props:{name:"class transformers.GroupViTModel",anchor:"transformers.GroupViTModel",parameters:[{name:"config",val:": GroupViTConfig"}],parametersDescription:[{anchor:"transformers.GroupViTModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.GroupViTConfig">GroupViTConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18020/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18020/src/transformers/models/groupvit/modeling_groupvit.py#L1324"}}),Co=new j({props:{name:"forward",anchor:"transformers.GroupViTModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"pixel_values",val:": typing.Optional[torch.FloatTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"return_loss",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"output_segmentation",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.GroupViTModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
it.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18020/en/model_doc/clip#transformers.CLIPTokenizer">CLIPTokenizer</a>. See <a href="/docs/transformers/pr_18020/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18020/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.GroupViTModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.GroupViTModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.GroupViTModel.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, height, width)</code>) &#x2014;
Pixel values. Pixel values can be obtained using <a href="/docs/transformers/pr_18020/en/model_doc/clip#transformers.CLIPFeatureExtractor">CLIPFeatureExtractor</a>. See
<code>CLIPFeatureExtractor.__call__()</code> for details.`,name:"pixel_values"},{anchor:"transformers.GroupViTModel.forward.return_loss",description:`<strong>return_loss</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the contrastive loss.`,name:"return_loss"},{anchor:"transformers.GroupViTModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.GroupViTModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.GroupViTModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18020/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18020/src/transformers/models/groupvit/modeling_groupvit.py#L1466",returnDescription:`
<p>A <code>transformers.models.groupvit.modeling_groupvit.GroupViTModelOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<code>&lt;class 'transformers.models.groupvit.configuration_groupvit.GroupViTConfig'&gt;</code>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>return_loss</code> is <code>True</code>) \u2014 Contrastive loss for image-text similarity.</p>
</li>
<li>
<p><strong>logits_per_image</strong> (<code>torch.FloatTensor</code> of shape <code>(image_batch_size, text_batch_size)</code>) \u2014 The scaled dot product scores between <code>image_embeds</code> and <code>text_embeds</code>. This represents the image-text
similarity scores.</p>
</li>
<li>
<p><strong>logits_per_text</strong> (<code>torch.FloatTensor</code> of shape <code>(text_batch_size, image_batch_size)</code>) \u2014 The scaled dot product scores between <code>text_embeds</code> and <code>image_embeds</code>. This represents the text-image
similarity scores.</p>
</li>
<li>
<p><strong>segmentation_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.num_labels, logits_height, logits_width)</code>) \u2014 Classification scores for each pixel.</p>
<Tip warning={true}>
<p>The logits returned do not necessarily have the same size as the <code>pixel_values</code> passed as inputs. This is
to avoid doing two interpolations and lose some quality when a user needs to resize the logits to the
original image size as post-processing. You should always check your logits shape and resize as needed.</p>
</Tip>
</li>
<li>
<p><strong>text_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, output_dim</code>) \u2014 The text embeddings obtained by applying the projection layer to the pooled output of
<a
  href="/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.GroupViTTextModel"
>GroupViTTextModel</a>.</p>
</li>
<li>
<p><strong>image_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, output_dim</code>) \u2014 The image embeddings obtained by applying the projection layer to the pooled output of
<a
  href="/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.GroupViTVisionModel"
>GroupViTVisionModel</a>.</p>
</li>
<li>
<p><strong>text_model_output</strong> (<code>BaseModelOutputWithPooling</code>) \u2014 The output of the <a
  href="/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.GroupViTTextModel"
>GroupViTTextModel</a>.</p>
</li>
<li>
<p><strong>vision_model_output</strong> (<code>BaseModelOutputWithPooling</code>) \u2014 The output of the <a
  href="/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.GroupViTVisionModel"
>GroupViTVisionModel</a>.</p>
</li>
</ul>
`,returnType:`
<p><code>transformers.models.groupvit.modeling_groupvit.GroupViTModelOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),yt=new ue({props:{$$slots:{default:[Kd]},$$scope:{ctx:k}}}),bt=new re({props:{anchor:"transformers.GroupViTModel.forward.example",$$slots:{default:[Hd]},$$scope:{ctx:k}}}),zo=new j({props:{name:"get_text_features",anchor:"transformers.GroupViTModel.get_text_features",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.GroupViTModel.get_text_features.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
it.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18020/en/model_doc/clip#transformers.CLIPTokenizer">CLIPTokenizer</a>. See <a href="/docs/transformers/pr_18020/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18020/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.GroupViTModel.get_text_features.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.GroupViTModel.get_text_features.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.GroupViTModel.get_text_features.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.GroupViTModel.get_text_features.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.GroupViTModel.get_text_features.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18020/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18020/src/transformers/models/groupvit/modeling_groupvit.py#L1370",returnDescription:`
<p>The text embeddings obtained by
applying the projection layer to the pooled output of <a
  href="/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.GroupViTTextModel"
>GroupViTTextModel</a>.</p>
`,returnType:`
<p>text_features (<code>torch.FloatTensor</code> of shape <code>(batch_size, output_dim</code>)</p>
`}}),wt=new ue({props:{$$slots:{default:[Rd]},$$scope:{ctx:k}}}),Vt=new re({props:{anchor:"transformers.GroupViTModel.get_text_features.example",$$slots:{default:[Jd]},$$scope:{ctx:k}}}),qo=new j({props:{name:"get_image_features",anchor:"transformers.GroupViTModel.get_image_features",parameters:[{name:"pixel_values",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.GroupViTModel.get_image_features.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, height, width)</code>) &#x2014;
Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
<a href="/docs/transformers/pr_18020/en/model_doc/clip#transformers.CLIPFeatureExtractor">CLIPFeatureExtractor</a>. See <code>CLIPFeatureExtractor.__call__()</code> for details.`,name:"pixel_values"},{anchor:"transformers.GroupViTModel.get_image_features.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.GroupViTModel.get_image_features.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.GroupViTModel.get_image_features.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18020/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18020/src/transformers/models/groupvit/modeling_groupvit.py#L1417",returnDescription:`
<p>The image embeddings obtained by
applying the projection layer to the pooled output of <a
  href="/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.GroupViTVisionModel"
>GroupViTVisionModel</a>.</p>
`,returnType:`
<p>image_features (<code>torch.FloatTensor</code> of shape <code>(batch_size, output_dim</code>)</p>
`}}),xt=new ue({props:{$$slots:{default:[Xd]},$$scope:{ctx:k}}}),kt=new re({props:{anchor:"transformers.GroupViTModel.get_image_features.example",$$slots:{default:[Yd]},$$scope:{ctx:k}}}),Fo=new me({}),Io=new j({props:{name:"class transformers.GroupViTTextModel",anchor:"transformers.GroupViTTextModel",parameters:[{name:"config",val:": GroupViTTextConfig"}],source:"https://github.com/huggingface/transformers/blob/vr_18020/src/transformers/models/groupvit/modeling_groupvit.py#L1160"}}),Lo=new j({props:{name:"forward",anchor:"transformers.GroupViTTextModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.GroupViTTextModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
it.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18020/en/model_doc/clip#transformers.CLIPTokenizer">CLIPTokenizer</a>. See <a href="/docs/transformers/pr_18020/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18020/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.GroupViTTextModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.GroupViTTextModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.GroupViTTextModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.GroupViTTextModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.GroupViTTextModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18020/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18020/src/transformers/models/groupvit/modeling_groupvit.py#L1175",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18020/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling"
>transformers.modeling_outputs.BaseModelOutputWithPooling</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<code>&lt;class 'transformers.models.groupvit.configuration_groupvit.GroupViTTextConfig'&gt;</code>) and inputs.</p>
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
  href="/docs/transformers/pr_18020/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling"
>transformers.modeling_outputs.BaseModelOutputWithPooling</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Mt=new ue({props:{$$slots:{default:[Qd]},$$scope:{ctx:k}}}),Et=new re({props:{anchor:"transformers.GroupViTTextModel.forward.example",$$slots:{default:[Zd]},$$scope:{ctx:k}}}),Ao=new me({}),Do=new j({props:{name:"class transformers.GroupViTVisionModel",anchor:"transformers.GroupViTVisionModel",parameters:[{name:"config",val:": GroupViTVisionConfig"}],source:"https://github.com/huggingface/transformers/blob/vr_18020/src/transformers/models/groupvit/modeling_groupvit.py#L1271"}}),Oo=new j({props:{name:"forward",anchor:"transformers.GroupViTVisionModel.forward",parameters:[{name:"pixel_values",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.GroupViTVisionModel.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, height, width)</code>) &#x2014;
Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
<a href="/docs/transformers/pr_18020/en/model_doc/clip#transformers.CLIPFeatureExtractor">CLIPFeatureExtractor</a>. See <code>CLIPFeatureExtractor.__call__()</code> for details.`,name:"pixel_values"},{anchor:"transformers.GroupViTVisionModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.GroupViTVisionModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.GroupViTVisionModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18020/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18020/src/transformers/models/groupvit/modeling_groupvit.py#L1284",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18020/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling"
>transformers.modeling_outputs.BaseModelOutputWithPooling</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<code>&lt;class 'transformers.models.groupvit.configuration_groupvit.GroupViTVisionConfig'&gt;</code>) and inputs.</p>
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
  href="/docs/transformers/pr_18020/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling"
>transformers.modeling_outputs.BaseModelOutputWithPooling</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Pt=new ue({props:{$$slots:{default:[ep]},$$scope:{ctx:k}}}),Ct=new re({props:{anchor:"transformers.GroupViTVisionModel.forward.example",$$slots:{default:[tp]},$$scope:{ctx:k}}}),No=new me({}),Wo=new j({props:{name:"class transformers.TFGroupViTModel",anchor:"transformers.TFGroupViTModel",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFGroupViTModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.GroupViTConfig">GroupViTConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18020/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18020/src/transformers/models/groupvit/modeling_tf_groupvit.py#L1783"}}),qt=new ue({props:{$$slots:{default:[op]},$$scope:{ctx:k}}}),Ko=new j({props:{name:"call",anchor:"transformers.TFGroupViTModel.call",parameters:[{name:"input_ids",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"pixel_values",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"position_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"return_loss",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"output_segmentation",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"training",val:": bool = False"}],parametersDescription:[{anchor:"transformers.TFGroupViTModel.call.input_ids",description:`<strong>input_ids</strong> (<code>np.ndarray</code>, <code>tf.Tensor</code>, <code>List[tf.Tensor]</code> \`<code>Dict[str, tf.Tensor]</code> or <code>Dict[str, np.ndarray]</code> and each example must have the shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18020/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18020/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> and
<a href="/docs/transformers/pr_18020/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.TFGroupViTModel.call.pixel_values",description:`<strong>pixel_values</strong> (<code>np.ndarray</code>, <code>tf.Tensor</code>, <code>List[tf.Tensor]</code> <code>Dict[str, tf.Tensor]</code> or <code>Dict[str, np.ndarray]</code> and each example must have the shape <code>(batch_size, num_channels, height, width)</code>) &#x2014;
Pixel values. Pixel values can be obtained using <a href="/docs/transformers/pr_18020/en/model_doc/clip#transformers.CLIPFeatureExtractor">CLIPFeatureExtractor</a>. See
<code>CLIPFeatureExtractor.__call__()</code> for details.`,name:"pixel_values"},{anchor:"transformers.TFGroupViTModel.call.attention_mask",description:`<strong>attention_mask</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.TFGroupViTModel.call.position_ids",description:`<strong>position_ids</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.TFGroupViTModel.call.return_loss",description:`<strong>return_loss</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the contrastive loss.`,name:"return_loss"},{anchor:"transformers.TFGroupViTModel.call.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
config will be used instead.`,name:"output_attentions"},{anchor:"transformers.TFGroupViTModel.call.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
used instead.`,name:"output_hidden_states"},{anchor:"transformers.TFGroupViTModel.call.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18020/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple. This argument can be used in
eager mode, in graph mode the value will always be set to True.`,name:"return_dict"},{anchor:"transformers.TFGroupViTModel.call.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to \`False&#x201C;) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"}],source:"https://github.com/huggingface/transformers/blob/vr_18020/src/transformers/models/groupvit/modeling_tf_groupvit.py#L1911",returnDescription:`
<p>A <code>transformers.models.groupvit.modeling_tf_groupvit.TFGroupViTModelOutput</code> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<code>&lt;class 'transformers.models.groupvit.configuration_groupvit.GroupViTConfig'&gt;</code>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>tf.Tensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>return_loss</code> is <code>True</code>) \u2014 Contrastive loss for image-text similarity.</p>
</li>
<li>
<p><strong>logits_per_image</strong> (<code>tf.Tensor</code> of shape <code>(image_batch_size, text_batch_size)</code>) \u2014 The scaled dot product scores between <code>image_embeds</code> and <code>text_embeds</code>. This represents the image-text
similarity scores.</p>
</li>
<li>
<p><strong>logits_per_text</strong> (<code>tf.Tensor</code> of shape <code>(text_batch_size, image_batch_size)</code>) \u2014 The scaled dot product scores between <code>text_embeds</code> and <code>image_embeds</code>. This represents the text-image
similarity scores.</p>
</li>
<li>
<p><strong>segmentation_logits</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, config.num_labels, logits_height, logits_width)</code>) \u2014 Classification scores for each pixel.</p>
<Tip warning={true}>
<p>The logits returned do not necessarily have the same size as the <code>pixel_values</code> passed as inputs. This is
to avoid doing two interpolations and lose some quality when a user needs to resize the logits to the
original image size as post-processing. You should always check your logits shape and resize as needed.</p>
</Tip>
</li>
<li>
<p><strong>text_embeds</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, output_dim</code>) \u2014 The text embeddings obtained by applying the projection layer to the pooled output of
<a
  href="/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.TFGroupViTTextModel"
>TFGroupViTTextModel</a>.</p>
</li>
<li>
<p><strong>image_embeds</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, output_dim</code>) \u2014 The image embeddings obtained by applying the projection layer to the pooled output of
<a
  href="/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.TFGroupViTVisionModel"
>TFGroupViTVisionModel</a>.</p>
</li>
<li>
<p><strong>text_model_output</strong> (<code>TFBaseModelOutputWithPooling</code>) \u2014 The output of the <a
  href="/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.TFGroupViTTextModel"
>TFGroupViTTextModel</a>.</p>
</li>
<li>
<p><strong>vision_model_output</strong> (<code>TFBaseModelOutputWithPooling</code>) \u2014 The output of the <a
  href="/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.TFGroupViTVisionModel"
>TFGroupViTVisionModel</a>.</p>
</li>
</ul>
`,returnType:`
<p><code>transformers.models.groupvit.modeling_tf_groupvit.TFGroupViTModelOutput</code> or <code>tuple(tf.Tensor)</code></p>
`}}),Ft=new ue({props:{$$slots:{default:[np]},$$scope:{ctx:k}}}),It=new re({props:{anchor:"transformers.TFGroupViTModel.call.example",$$slots:{default:[rp]},$$scope:{ctx:k}}}),Ho=new j({props:{name:"get_text_features",anchor:"transformers.TFGroupViTModel.get_text_features",parameters:[{name:"input_ids",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"position_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"training",val:": bool = False"}],parametersDescription:[{anchor:"transformers.TFGroupViTModel.get_text_features.input_ids",description:`<strong>input_ids</strong> (<code>np.ndarray</code>, <code>tf.Tensor</code>, <code>List[tf.Tensor]</code> \`<code>Dict[str, tf.Tensor]</code> or <code>Dict[str, np.ndarray]</code> and each example must have the shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18020/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18020/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> and
<a href="/docs/transformers/pr_18020/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.TFGroupViTModel.get_text_features.attention_mask",description:`<strong>attention_mask</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.TFGroupViTModel.get_text_features.position_ids",description:`<strong>position_ids</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.TFGroupViTModel.get_text_features.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
config will be used instead.`,name:"output_attentions"},{anchor:"transformers.TFGroupViTModel.get_text_features.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
used instead.`,name:"output_hidden_states"},{anchor:"transformers.TFGroupViTModel.get_text_features.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18020/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple. This argument can be used in
eager mode, in graph mode the value will always be set to True.`,name:"return_dict"},{anchor:"transformers.TFGroupViTModel.get_text_features.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to \`False&#x201C;) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"}],source:"https://github.com/huggingface/transformers/blob/vr_18020/src/transformers/models/groupvit/modeling_tf_groupvit.py#L1829",returnDescription:`
<p>The text embeddings obtained by applying
the projection layer to the pooled output of <a
  href="/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.TFGroupViTTextModel"
>TFGroupViTTextModel</a>.</p>
`,returnType:`
<p>text_features (<code>tf.Tensor</code> of shape <code>(batch_size, output_dim</code>)</p>
`}}),Lt=new ue({props:{$$slots:{default:[sp]},$$scope:{ctx:k}}}),At=new re({props:{anchor:"transformers.TFGroupViTModel.get_text_features.example",$$slots:{default:[ap]},$$scope:{ctx:k}}}),Ro=new j({props:{name:"get_image_features",anchor:"transformers.TFGroupViTModel.get_image_features",parameters:[{name:"pixel_values",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"training",val:": bool = False"}],parametersDescription:[{anchor:"transformers.TFGroupViTModel.get_image_features.pixel_values",description:`<strong>pixel_values</strong> (<code>np.ndarray</code>, <code>tf.Tensor</code>, <code>List[tf.Tensor]</code> \`<code>Dict[str, tf.Tensor]</code> or <code>Dict[str, np.ndarray]</code> and each example must have the shape <code>(batch_size, num_channels, height, width)</code>) &#x2014;
Pixel values. Pixel values can be obtained using <a href="/docs/transformers/pr_18020/en/model_doc/clip#transformers.CLIPFeatureExtractor">CLIPFeatureExtractor</a>. See
<code>CLIPFeatureExtractor.__call__()</code> for details. output_attentions (<code>bool</code>, <em>optional</em>): Whether or not to
return the attentions tensors of all attention layers. See <code>attentions</code> under returned tensors for more
detail. This argument can be used only in eager mode, in graph mode the value in the config will be used
instead.`,name:"pixel_values"},{anchor:"transformers.TFGroupViTModel.get_image_features.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
used instead.`,name:"output_hidden_states"},{anchor:"transformers.TFGroupViTModel.get_image_features.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18020/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple. This argument can be used in
eager mode, in graph mode the value will always be set to True.`,name:"return_dict"},{anchor:"transformers.TFGroupViTModel.get_image_features.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to \`False&#x201C;) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"}],source:"https://github.com/huggingface/transformers/blob/vr_18020/src/transformers/models/groupvit/modeling_tf_groupvit.py#L1869",returnDescription:`
<p>The image embeddings obtained by applying
the projection layer to the pooled output of <a
  href="/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.TFGroupViTVisionModel"
>TFGroupViTVisionModel</a>.</p>
`,returnType:`
<p>image_features (<code>tf.Tensor</code> of shape <code>(batch_size, output_dim</code>)</p>
`}}),Dt=new ue({props:{$$slots:{default:[ip]},$$scope:{ctx:k}}}),Ot=new re({props:{anchor:"transformers.TFGroupViTModel.get_image_features.example",$$slots:{default:[lp]},$$scope:{ctx:k}}}),Jo=new me({}),Xo=new j({props:{name:"class transformers.TFGroupViTTextModel",anchor:"transformers.TFGroupViTTextModel",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18020/src/transformers/models/groupvit/modeling_tf_groupvit.py#L1601"}}),Yo=new j({props:{name:"call",anchor:"transformers.TFGroupViTTextModel.call",parameters:[{name:"input_ids",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"attention_mask",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"position_ids",val:": typing.Union[numpy.ndarray, tensorflow.python.framework.ops.Tensor, NoneType] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"training",val:": bool = False"}],parametersDescription:[{anchor:"transformers.TFGroupViTTextModel.call.input_ids",description:`<strong>input_ids</strong> (<code>np.ndarray</code>, <code>tf.Tensor</code>, <code>List[tf.Tensor]</code> \`<code>Dict[str, tf.Tensor]</code> or <code>Dict[str, np.ndarray]</code> and each example must have the shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18020/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18020/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> and
<a href="/docs/transformers/pr_18020/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.TFGroupViTTextModel.call.attention_mask",description:`<strong>attention_mask</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.TFGroupViTTextModel.call.position_ids",description:`<strong>position_ids</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.TFGroupViTTextModel.call.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
config will be used instead.`,name:"output_attentions"},{anchor:"transformers.TFGroupViTTextModel.call.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
used instead.`,name:"output_hidden_states"},{anchor:"transformers.TFGroupViTTextModel.call.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18020/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple. This argument can be used in
eager mode, in graph mode the value will always be set to True.`,name:"return_dict"},{anchor:"transformers.TFGroupViTTextModel.call.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to \`False&#x201C;) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"}],source:"https://github.com/huggingface/transformers/blob/vr_18020/src/transformers/models/groupvit/modeling_tf_groupvit.py#L1634",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18020/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutputWithPooling"
>transformers.modeling_tf_outputs.TFBaseModelOutputWithPooling</a> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<code>&lt;class 'transformers.models.groupvit.configuration_groupvit.GroupViTTextConfig'&gt;</code>) and inputs.</p>
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
  href="/docs/transformers/pr_18020/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutputWithPooling"
>transformers.modeling_tf_outputs.TFBaseModelOutputWithPooling</a> or <code>tuple(tf.Tensor)</code></p>
`}}),Wt=new ue({props:{$$slots:{default:[dp]},$$scope:{ctx:k}}}),St=new re({props:{anchor:"transformers.TFGroupViTTextModel.call.example",$$slots:{default:[pp]},$$scope:{ctx:k}}}),Qo=new me({}),Zo=new j({props:{name:"class transformers.TFGroupViTVisionModel",anchor:"transformers.TFGroupViTVisionModel",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18020/src/transformers/models/groupvit/modeling_tf_groupvit.py#L1688"}}),en=new j({props:{name:"call",anchor:"transformers.TFGroupViTVisionModel.call",parameters:[{name:"pixel_values",val:": typing.Union[typing.List[tensorflow.python.framework.ops.Tensor], typing.List[numpy.ndarray], typing.List[tensorflow.python.keras.engine.keras_tensor.KerasTensor], typing.Dict[str, tensorflow.python.framework.ops.Tensor], typing.Dict[str, numpy.ndarray], typing.Dict[str, tensorflow.python.keras.engine.keras_tensor.KerasTensor], tensorflow.python.framework.ops.Tensor, numpy.ndarray, tensorflow.python.keras.engine.keras_tensor.KerasTensor, NoneType] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"training",val:": bool = False"}],parametersDescription:[{anchor:"transformers.TFGroupViTVisionModel.call.pixel_values",description:`<strong>pixel_values</strong> (<code>np.ndarray</code>, <code>tf.Tensor</code>, <code>List[tf.Tensor]</code> \`<code>Dict[str, tf.Tensor]</code> or <code>Dict[str, np.ndarray]</code> and each example must have the shape <code>(batch_size, num_channels, height, width)</code>) &#x2014;
Pixel values. Pixel values can be obtained using <a href="/docs/transformers/pr_18020/en/model_doc/clip#transformers.CLIPFeatureExtractor">CLIPFeatureExtractor</a>. See
<code>CLIPFeatureExtractor.__call__()</code> for details. output_attentions (<code>bool</code>, <em>optional</em>): Whether or not to
return the attentions tensors of all attention layers. See <code>attentions</code> under returned tensors for more
detail. This argument can be used only in eager mode, in graph mode the value in the config will be used
instead.`,name:"pixel_values"},{anchor:"transformers.TFGroupViTVisionModel.call.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
used instead.`,name:"output_hidden_states"},{anchor:"transformers.TFGroupViTVisionModel.call.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18020/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple. This argument can be used in
eager mode, in graph mode the value will always be set to True.`,name:"return_dict"},{anchor:"transformers.TFGroupViTVisionModel.call.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to \`False&#x201C;) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"}],source:"https://github.com/huggingface/transformers/blob/vr_18020/src/transformers/models/groupvit/modeling_tf_groupvit.py#L1729",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18020/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutputWithPooling"
>transformers.modeling_tf_outputs.TFBaseModelOutputWithPooling</a> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<code>&lt;class 'transformers.models.groupvit.configuration_groupvit.GroupViTVisionConfig'&gt;</code>) and inputs.</p>
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
  href="/docs/transformers/pr_18020/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutputWithPooling"
>transformers.modeling_tf_outputs.TFBaseModelOutputWithPooling</a> or <code>tuple(tf.Tensor)</code></p>
`}}),Ut=new ue({props:{$$slots:{default:[cp]},$$scope:{ctx:k}}}),Kt=new re({props:{anchor:"transformers.TFGroupViTVisionModel.call.example",$$slots:{default:[up]},$$scope:{ctx:k}}}),{c(){r=s("meta"),T=f(),u=s("h1"),c=s("a"),m=s("span"),$(n.$$.fragment),p=f(),G=s("span"),ao=i("GroupViT"),xe=f(),E=s("h2"),U=s("a"),ie=s("span"),$(le.$$.fragment),io=f(),de=s("span"),lo=i("Overview"),lt=f(),q=s("p"),fe=i("The GroupViT model was proposed in "),pe=s("a"),dt=i("GroupViT: Semantic Segmentation Emerges from Text Supervision"),L=i(` by Jiarui Xu, Shalini De Mello, Sifei Liu, Wonmin Byeon, Thomas Breuel, Jan Kautz, Xiaolong Wang.
Inspired by `),F=s("a"),po=i("CLIP"),ke=i(", GroupViT is a vision-language model that can perform zero-shot semantic segmentation on any given vocabulary categories."),pt=f(),he=s("p"),Ge=i("The abstract from the paper is the following:"),ct=f(),ge=s("p"),A=s("em"),co=i("Grouping and recognition are important components of visual scene understanding, e.g., for object detection and semantic segmentation. With end-to-end deep learning systems, grouping of image regions usually happens implicitly via top-down supervision from pixel-level recognition labels. Instead, in this paper, we propose to bring back the grouping mechanism into deep networks, which allows semantic segments to emerge automatically with only text supervision. We propose a hierarchical Grouping Vision Transformer (GroupViT), which goes beyond the regular grid structure representation and learns to group image regions into progressively larger arbitrary-shaped segments. We train GroupViT jointly with a text encoder on a large-scale image-text dataset via contrastive losses. With only text supervision and without any pixel-level annotations, GroupViT learns to group together semantic regions and successfully transfers to the task of semantic segmentation in a zero-shot manner, i.e., without any further fine-tuning. It achieves a zero-shot accuracy of 52.3% mIoU on the PASCAL VOC 2012 and 22.4% mIoU on PASCAL Context datasets, and performs competitively to state-of-the-art transfer-learning methods requiring greater levels of supervision."),_e=f(),Te=s("p"),uo=i("Tips:"),ve=f(),K=s("ul"),N=s("li"),$e=i("You may specify "),Me=s("code"),Ee=i("output_segmentation=True"),mo=i(" in the forward of "),M=s("code"),z=i("GroupViTModel"),ut=i(" to get the segmentation logits of input texts."),rn=f(),H=s("li"),ye=i("The quickest way to get started with GroupViT is by checking the "),be=s("a"),sn=i("example notebooks"),an=i(" (which showcase zero-shot segmentation inference). One can also check out the "),D=s("a"),je=i("HuggingFace Spaces demo"),ln=i(" to play with GroupViT."),fo=f(),C=s("p"),dn=i("This model was contributed by "),we=s("a"),ho=i("xvjiarui"),pn=i(". The TensorFlow version was contributed by "),go=s("a"),bs=i("ariG23498"),ws=i(`.
The original code can be found `),_o=s("a"),Vs=i("here"),xs=i("."),Dr=f(),Pe=s("h2"),mt=s("a"),Rn=s("span"),$(To.$$.fragment),ks=f(),Jn=s("span"),Gs=i("GroupViTConfig"),Or=f(),W=s("div"),$(vo.$$.fragment),Ms=f(),ft=s("p"),cn=s("a"),Es=i("GroupViTConfig"),js=i(" is the configuration class to store the configuration of a "),un=s("a"),Ps=i("GroupViTModel"),Cs=i(`. It is used to
instantiate a GroupViT model according to the specified arguments, defining the text model and vision model
configs.`),zs=f(),Ce=s("p"),qs=i("Configuration objects inherit from "),mn=s("a"),Fs=i("PretrainedConfig"),Is=i(` and can be used to control the model outputs. Read the
documentation from `),fn=s("a"),Ls=i("PretrainedConfig"),As=i(" for more information."),Ds=f(),ht=s("div"),$($o.$$.fragment),Os=f(),yo=s("p"),Ns=i("Instantiate a "),hn=s("a"),Ws=i("GroupViTConfig"),Ss=i(` (or a derived class) from groupvit text model configuration and groupvit
vision model configuration.`),Nr=f(),ze=s("h2"),gt=s("a"),Xn=s("span"),$(bo.$$.fragment),Bs=f(),Yn=s("span"),Us=i("GroupViTTextConfig"),Wr=f(),S=s("div"),$(wo.$$.fragment),Ks=f(),qe=s("p"),Hs=i("This is the configuration class to store the configuration of a "),gn=s("a"),Rs=i("GroupViTTextModel"),Js=i(`. It is used to instantiate an
GroupViT model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the GroupViT
`),Vo=s("a"),Xs=i("nvidia/groupvit-gcc-yfcc"),Ys=i(" architecture."),Qs=f(),Fe=s("p"),Zs=i("Configuration objects inherit from "),_n=s("a"),ea=i("PretrainedConfig"),ta=i(` and can be used to control the model outputs. Read the
documentation from `),Tn=s("a"),oa=i("PretrainedConfig"),na=i(" for more information."),ra=f(),$(_t.$$.fragment),Sr=f(),Ie=s("h2"),Tt=s("a"),Qn=s("span"),$(xo.$$.fragment),sa=f(),Zn=s("span"),aa=i("GroupViTVisionConfig"),Br=f(),B=s("div"),$(ko.$$.fragment),ia=f(),Le=s("p"),la=i("This is the configuration class to store the configuration of a "),vn=s("a"),da=i("GroupViTVisionModel"),pa=i(`. It is used to instantiate
an GroupViT model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the GroupViT
`),Go=s("a"),ca=i("nvidia/groupvit-gcc-yfcc"),ua=i(" architecture."),ma=f(),Ae=s("p"),fa=i("Configuration objects inherit from "),$n=s("a"),ha=i("PretrainedConfig"),ga=i(` and can be used to control the model outputs. Read the
documentation from `),yn=s("a"),_a=i("PretrainedConfig"),Ta=i(" for more information."),va=f(),$(vt.$$.fragment),Ur=f(),De=s("h2"),$t=s("a"),er=s("span"),$(Mo.$$.fragment),$a=f(),tr=s("span"),ya=i("GroupViTModel"),Kr=f(),I=s("div"),$(Eo.$$.fragment),ba=f(),jo=s("p"),wa=i("This model is a PyTorch "),Po=s("a"),Va=i("torch.nn.Module"),xa=i(` subclass. Use it
as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),ka=f(),R=s("div"),$(Co.$$.fragment),Ga=f(),Oe=s("p"),Ma=i("The "),bn=s("a"),Ea=i("GroupViTModel"),ja=i(" forward method, overrides the "),or=s("code"),Pa=i("__call__"),Ca=i(" special method."),za=f(),$(yt.$$.fragment),qa=f(),$(bt.$$.fragment),Fa=f(),J=s("div"),$(zo.$$.fragment),Ia=f(),Ne=s("p"),La=i("The "),wn=s("a"),Aa=i("GroupViTModel"),Da=i(" forward method, overrides the "),nr=s("code"),Oa=i("__call__"),Na=i(" special method."),Wa=f(),$(wt.$$.fragment),Sa=f(),$(Vt.$$.fragment),Ba=f(),X=s("div"),$(qo.$$.fragment),Ua=f(),We=s("p"),Ka=i("The "),Vn=s("a"),Ha=i("GroupViTModel"),Ra=i(" forward method, overrides the "),rr=s("code"),Ja=i("__call__"),Xa=i(" special method."),Ya=f(),$(xt.$$.fragment),Qa=f(),$(kt.$$.fragment),Hr=f(),Se=s("h2"),Gt=s("a"),sr=s("span"),$(Fo.$$.fragment),Za=f(),ar=s("span"),ei=i("GroupViTTextModel"),Rr=f(),Be=s("div"),$(Io.$$.fragment),ti=f(),Y=s("div"),$(Lo.$$.fragment),oi=f(),Ue=s("p"),ni=i("The "),xn=s("a"),ri=i("GroupViTTextModel"),si=i(" forward method, overrides the "),ir=s("code"),ai=i("__call__"),ii=i(" special method."),li=f(),$(Mt.$$.fragment),di=f(),$(Et.$$.fragment),Jr=f(),Ke=s("h2"),jt=s("a"),lr=s("span"),$(Ao.$$.fragment),pi=f(),dr=s("span"),ci=i("GroupViTVisionModel"),Xr=f(),He=s("div"),$(Do.$$.fragment),ui=f(),Q=s("div"),$(Oo.$$.fragment),mi=f(),Re=s("p"),fi=i("The "),kn=s("a"),hi=i("GroupViTVisionModel"),gi=i(" forward method, overrides the "),pr=s("code"),_i=i("__call__"),Ti=i(" special method."),vi=f(),$(Pt.$$.fragment),$i=f(),$(Ct.$$.fragment),Yr=f(),Je=s("h2"),zt=s("a"),cr=s("span"),$(No.$$.fragment),yi=f(),ur=s("span"),bi=i("TFGroupViTModel"),Qr=f(),P=s("div"),$(Wo.$$.fragment),wi=f(),So=s("p"),Vi=i("This model inherits from "),Gn=s("a"),xi=i("TFPreTrainedModel"),ki=i(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Gi=f(),Bo=s("p"),Mi=i("This model is also a "),Uo=s("a"),Ei=i("tf.keras.Model"),ji=i(` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),Pi=f(),$(qt.$$.fragment),Ci=f(),Z=s("div"),$(Ko.$$.fragment),zi=f(),Xe=s("p"),qi=i("The "),Mn=s("a"),Fi=i("TFGroupViTModel"),Ii=i(" forward method, overrides the "),mr=s("code"),Li=i("__call__"),Ai=i(" special method."),Di=f(),$(Ft.$$.fragment),Oi=f(),$(It.$$.fragment),Ni=f(),ee=s("div"),$(Ho.$$.fragment),Wi=f(),Ye=s("p"),Si=i("The "),En=s("a"),Bi=i("TFGroupViTModel"),Ui=i(" forward method, overrides the "),fr=s("code"),Ki=i("__call__"),Hi=i(" special method."),Ri=f(),$(Lt.$$.fragment),Ji=f(),$(At.$$.fragment),Xi=f(),te=s("div"),$(Ro.$$.fragment),Yi=f(),Qe=s("p"),Qi=i("The "),jn=s("a"),Zi=i("TFGroupViTModel"),el=i(" forward method, overrides the "),hr=s("code"),tl=i("__call__"),ol=i(" special method."),nl=f(),$(Dt.$$.fragment),rl=f(),$(Ot.$$.fragment),Zr=f(),Ze=s("h2"),Nt=s("a"),gr=s("span"),$(Jo.$$.fragment),sl=f(),_r=s("span"),al=i("TFGroupViTTextModel"),es=f(),et=s("div"),$(Xo.$$.fragment),il=f(),oe=s("div"),$(Yo.$$.fragment),ll=f(),tt=s("p"),dl=i("The "),Pn=s("a"),pl=i("TFGroupViTTextModel"),cl=i(" forward method, overrides the "),Tr=s("code"),ul=i("__call__"),ml=i(" special method."),fl=f(),$(Wt.$$.fragment),hl=f(),$(St.$$.fragment),ts=f(),ot=s("h2"),Bt=s("a"),vr=s("span"),$(Qo.$$.fragment),gl=f(),$r=s("span"),_l=i("TFGroupViTVisionModel"),os=f(),nt=s("div"),$(Zo.$$.fragment),Tl=f(),ne=s("div"),$(en.$$.fragment),vl=f(),rt=s("p"),$l=i("The "),Cn=s("a"),yl=i("TFGroupViTVisionModel"),bl=i(" forward method, overrides the "),yr=s("code"),wl=i("__call__"),Vl=i(" special method."),xl=f(),$(Ut.$$.fragment),kl=f(),$(Kt.$$.fragment),this.h()},l(t){const _=Wd('[data-svelte="svelte-1phssyn"]',document.head);r=a(_,"META",{name:!0,content:!0}),_.forEach(o),T=h(t),u=a(t,"H1",{class:!0});var tn=d(u);c=a(tn,"A",{id:!0,class:!0,href:!0});var br=d(c);m=a(br,"SPAN",{});var wr=d(m);y(n.$$.fragment,wr),wr.forEach(o),br.forEach(o),p=h(tn),G=a(tn,"SPAN",{});var Vr=d(G);ao=l(Vr,"GroupViT"),Vr.forEach(o),tn.forEach(o),xe=h(t),E=a(t,"H2",{class:!0});var on=d(E);U=a(on,"A",{id:!0,class:!0,href:!0});var xr=d(U);ie=a(xr,"SPAN",{});var kr=d(ie);y(le.$$.fragment,kr),kr.forEach(o),xr.forEach(o),io=h(on),de=a(on,"SPAN",{});var Gr=d(de);lo=l(Gr,"Overview"),Gr.forEach(o),on.forEach(o),lt=h(t),q=a(t,"P",{});var st=d(q);fe=l(st,"The GroupViT model was proposed in "),pe=a(st,"A",{href:!0,rel:!0});var Mr=d(pe);dt=l(Mr,"GroupViT: Semantic Segmentation Emerges from Text Supervision"),Mr.forEach(o),L=l(st,` by Jiarui Xu, Shalini De Mello, Sifei Liu, Wonmin Byeon, Thomas Breuel, Jan Kautz, Xiaolong Wang.
Inspired by `),F=a(st,"A",{href:!0});var Er=d(F);po=l(Er,"CLIP"),Er.forEach(o),ke=l(st,", GroupViT is a vision-language model that can perform zero-shot semantic segmentation on any given vocabulary categories."),st.forEach(o),pt=h(t),he=a(t,"P",{});var jr=d(he);Ge=l(jr,"The abstract from the paper is the following:"),jr.forEach(o),ct=h(t),ge=a(t,"P",{});var Pr=d(ge);A=a(Pr,"EM",{});var Cr=d(A);co=l(Cr,"Grouping and recognition are important components of visual scene understanding, e.g., for object detection and semantic segmentation. With end-to-end deep learning systems, grouping of image regions usually happens implicitly via top-down supervision from pixel-level recognition labels. Instead, in this paper, we propose to bring back the grouping mechanism into deep networks, which allows semantic segments to emerge automatically with only text supervision. We propose a hierarchical Grouping Vision Transformer (GroupViT), which goes beyond the regular grid structure representation and learns to group image regions into progressively larger arbitrary-shaped segments. We train GroupViT jointly with a text encoder on a large-scale image-text dataset via contrastive losses. With only text supervision and without any pixel-level annotations, GroupViT learns to group together semantic regions and successfully transfers to the task of semantic segmentation in a zero-shot manner, i.e., without any further fine-tuning. It achieves a zero-shot accuracy of 52.3% mIoU on the PASCAL VOC 2012 and 22.4% mIoU on PASCAL Context datasets, and performs competitively to state-of-the-art transfer-learning methods requiring greater levels of supervision."),Cr.forEach(o),Pr.forEach(o),_e=h(t),Te=a(t,"P",{});var zr=d(Te);uo=l(zr,"Tips:"),zr.forEach(o),ve=h(t),K=a(t,"UL",{});var nn=d(K);N=a(nn,"LI",{});var at=d(N);$e=l(at,"You may specify "),Me=a(at,"CODE",{});var qr=d(Me);Ee=l(qr,"output_segmentation=True"),qr.forEach(o),mo=l(at," in the forward of "),M=a(at,"CODE",{});var Fr=d(M);z=l(Fr,"GroupViTModel"),Fr.forEach(o),ut=l(at," to get the segmentation logits of input texts."),at.forEach(o),rn=h(nn),H=a(nn,"LI",{});var it=d(H);ye=l(it,"The quickest way to get started with GroupViT is by checking the "),be=a(it,"A",{href:!0,rel:!0});var Ir=d(be);sn=l(Ir,"example notebooks"),Ir.forEach(o),an=l(it," (which showcase zero-shot segmentation inference). One can also check out the "),D=a(it,"A",{href:!0,rel:!0});var Lr=d(D);je=l(Lr,"HuggingFace Spaces demo"),Lr.forEach(o),ln=l(it," to play with GroupViT."),it.forEach(o),nn.forEach(o),fo=h(t),C=a(t,"P",{});var ce=d(C);dn=l(ce,"This model was contributed by "),we=a(ce,"A",{href:!0,rel:!0});var Gl=d(we);ho=l(Gl,"xvjiarui"),Gl.forEach(o),pn=l(ce,". The TensorFlow version was contributed by "),go=a(ce,"A",{href:!0,rel:!0});var Ml=d(go);bs=l(Ml,"ariG23498"),Ml.forEach(o),ws=l(ce,`.
The original code can be found `),_o=a(ce,"A",{href:!0,rel:!0});var El=d(_o);Vs=l(El,"here"),El.forEach(o),xs=l(ce,"."),ce.forEach(o),Dr=h(t),Pe=a(t,"H2",{class:!0});var rs=d(Pe);mt=a(rs,"A",{id:!0,class:!0,href:!0});var jl=d(mt);Rn=a(jl,"SPAN",{});var Pl=d(Rn);y(To.$$.fragment,Pl),Pl.forEach(o),jl.forEach(o),ks=h(rs),Jn=a(rs,"SPAN",{});var Cl=d(Jn);Gs=l(Cl,"GroupViTConfig"),Cl.forEach(o),rs.forEach(o),Or=h(t),W=a(t,"DIV",{class:!0});var Ht=d(W);y(vo.$$.fragment,Ht),Ms=h(Ht),ft=a(Ht,"P",{});var Ar=d(ft);cn=a(Ar,"A",{href:!0});var zl=d(cn);Es=l(zl,"GroupViTConfig"),zl.forEach(o),js=l(Ar," is the configuration class to store the configuration of a "),un=a(Ar,"A",{href:!0});var ql=d(un);Ps=l(ql,"GroupViTModel"),ql.forEach(o),Cs=l(Ar,`. It is used to
instantiate a GroupViT model according to the specified arguments, defining the text model and vision model
configs.`),Ar.forEach(o),zs=h(Ht),Ce=a(Ht,"P",{});var zn=d(Ce);qs=l(zn,"Configuration objects inherit from "),mn=a(zn,"A",{href:!0});var Fl=d(mn);Fs=l(Fl,"PretrainedConfig"),Fl.forEach(o),Is=l(zn,` and can be used to control the model outputs. Read the
documentation from `),fn=a(zn,"A",{href:!0});var Il=d(fn);Ls=l(Il,"PretrainedConfig"),Il.forEach(o),As=l(zn," for more information."),zn.forEach(o),Ds=h(Ht),ht=a(Ht,"DIV",{class:!0});var ss=d(ht);y($o.$$.fragment,ss),Os=h(ss),yo=a(ss,"P",{});var as=d(yo);Ns=l(as,"Instantiate a "),hn=a(as,"A",{href:!0});var Ll=d(hn);Ws=l(Ll,"GroupViTConfig"),Ll.forEach(o),Ss=l(as,` (or a derived class) from groupvit text model configuration and groupvit
vision model configuration.`),as.forEach(o),ss.forEach(o),Ht.forEach(o),Nr=h(t),ze=a(t,"H2",{class:!0});var is=d(ze);gt=a(is,"A",{id:!0,class:!0,href:!0});var Al=d(gt);Xn=a(Al,"SPAN",{});var Dl=d(Xn);y(bo.$$.fragment,Dl),Dl.forEach(o),Al.forEach(o),Bs=h(is),Yn=a(is,"SPAN",{});var Ol=d(Yn);Us=l(Ol,"GroupViTTextConfig"),Ol.forEach(o),is.forEach(o),Wr=h(t),S=a(t,"DIV",{class:!0});var Rt=d(S);y(wo.$$.fragment,Rt),Ks=h(Rt),qe=a(Rt,"P",{});var qn=d(qe);Hs=l(qn,"This is the configuration class to store the configuration of a "),gn=a(qn,"A",{href:!0});var Nl=d(gn);Rs=l(Nl,"GroupViTTextModel"),Nl.forEach(o),Js=l(qn,`. It is used to instantiate an
GroupViT model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the GroupViT
`),Vo=a(qn,"A",{href:!0,rel:!0});var Wl=d(Vo);Xs=l(Wl,"nvidia/groupvit-gcc-yfcc"),Wl.forEach(o),Ys=l(qn," architecture."),qn.forEach(o),Qs=h(Rt),Fe=a(Rt,"P",{});var Fn=d(Fe);Zs=l(Fn,"Configuration objects inherit from "),_n=a(Fn,"A",{href:!0});var Sl=d(_n);ea=l(Sl,"PretrainedConfig"),Sl.forEach(o),ta=l(Fn,` and can be used to control the model outputs. Read the
documentation from `),Tn=a(Fn,"A",{href:!0});var Bl=d(Tn);oa=l(Bl,"PretrainedConfig"),Bl.forEach(o),na=l(Fn," for more information."),Fn.forEach(o),ra=h(Rt),y(_t.$$.fragment,Rt),Rt.forEach(o),Sr=h(t),Ie=a(t,"H2",{class:!0});var ls=d(Ie);Tt=a(ls,"A",{id:!0,class:!0,href:!0});var Ul=d(Tt);Qn=a(Ul,"SPAN",{});var Kl=d(Qn);y(xo.$$.fragment,Kl),Kl.forEach(o),Ul.forEach(o),sa=h(ls),Zn=a(ls,"SPAN",{});var Hl=d(Zn);aa=l(Hl,"GroupViTVisionConfig"),Hl.forEach(o),ls.forEach(o),Br=h(t),B=a(t,"DIV",{class:!0});var Jt=d(B);y(ko.$$.fragment,Jt),ia=h(Jt),Le=a(Jt,"P",{});var In=d(Le);la=l(In,"This is the configuration class to store the configuration of a "),vn=a(In,"A",{href:!0});var Rl=d(vn);da=l(Rl,"GroupViTVisionModel"),Rl.forEach(o),pa=l(In,`. It is used to instantiate
an GroupViT model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the GroupViT
`),Go=a(In,"A",{href:!0,rel:!0});var Jl=d(Go);ca=l(Jl,"nvidia/groupvit-gcc-yfcc"),Jl.forEach(o),ua=l(In," architecture."),In.forEach(o),ma=h(Jt),Ae=a(Jt,"P",{});var Ln=d(Ae);fa=l(Ln,"Configuration objects inherit from "),$n=a(Ln,"A",{href:!0});var Xl=d($n);ha=l(Xl,"PretrainedConfig"),Xl.forEach(o),ga=l(Ln,` and can be used to control the model outputs. Read the
documentation from `),yn=a(Ln,"A",{href:!0});var Yl=d(yn);_a=l(Yl,"PretrainedConfig"),Yl.forEach(o),Ta=l(Ln," for more information."),Ln.forEach(o),va=h(Jt),y(vt.$$.fragment,Jt),Jt.forEach(o),Ur=h(t),De=a(t,"H2",{class:!0});var ds=d(De);$t=a(ds,"A",{id:!0,class:!0,href:!0});var Ql=d($t);er=a(Ql,"SPAN",{});var Zl=d(er);y(Mo.$$.fragment,Zl),Zl.forEach(o),Ql.forEach(o),$a=h(ds),tr=a(ds,"SPAN",{});var ed=d(tr);ya=l(ed,"GroupViTModel"),ed.forEach(o),ds.forEach(o),Kr=h(t),I=a(t,"DIV",{class:!0});var Ve=d(I);y(Eo.$$.fragment,Ve),ba=h(Ve),jo=a(Ve,"P",{});var ps=d(jo);wa=l(ps,"This model is a PyTorch "),Po=a(ps,"A",{href:!0,rel:!0});var td=d(Po);Va=l(td,"torch.nn.Module"),td.forEach(o),xa=l(ps,` subclass. Use it
as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),ps.forEach(o),ka=h(Ve),R=a(Ve,"DIV",{class:!0});var Xt=d(R);y(Co.$$.fragment,Xt),Ga=h(Xt),Oe=a(Xt,"P",{});var An=d(Oe);Ma=l(An,"The "),bn=a(An,"A",{href:!0});var od=d(bn);Ea=l(od,"GroupViTModel"),od.forEach(o),ja=l(An," forward method, overrides the "),or=a(An,"CODE",{});var nd=d(or);Pa=l(nd,"__call__"),nd.forEach(o),Ca=l(An," special method."),An.forEach(o),za=h(Xt),y(yt.$$.fragment,Xt),qa=h(Xt),y(bt.$$.fragment,Xt),Xt.forEach(o),Fa=h(Ve),J=a(Ve,"DIV",{class:!0});var Yt=d(J);y(zo.$$.fragment,Yt),Ia=h(Yt),Ne=a(Yt,"P",{});var Dn=d(Ne);La=l(Dn,"The "),wn=a(Dn,"A",{href:!0});var rd=d(wn);Aa=l(rd,"GroupViTModel"),rd.forEach(o),Da=l(Dn," forward method, overrides the "),nr=a(Dn,"CODE",{});var sd=d(nr);Oa=l(sd,"__call__"),sd.forEach(o),Na=l(Dn," special method."),Dn.forEach(o),Wa=h(Yt),y(wt.$$.fragment,Yt),Sa=h(Yt),y(Vt.$$.fragment,Yt),Yt.forEach(o),Ba=h(Ve),X=a(Ve,"DIV",{class:!0});var Qt=d(X);y(qo.$$.fragment,Qt),Ua=h(Qt),We=a(Qt,"P",{});var On=d(We);Ka=l(On,"The "),Vn=a(On,"A",{href:!0});var ad=d(Vn);Ha=l(ad,"GroupViTModel"),ad.forEach(o),Ra=l(On," forward method, overrides the "),rr=a(On,"CODE",{});var id=d(rr);Ja=l(id,"__call__"),id.forEach(o),Xa=l(On," special method."),On.forEach(o),Ya=h(Qt),y(xt.$$.fragment,Qt),Qa=h(Qt),y(kt.$$.fragment,Qt),Qt.forEach(o),Ve.forEach(o),Hr=h(t),Se=a(t,"H2",{class:!0});var cs=d(Se);Gt=a(cs,"A",{id:!0,class:!0,href:!0});var ld=d(Gt);sr=a(ld,"SPAN",{});var dd=d(sr);y(Fo.$$.fragment,dd),dd.forEach(o),ld.forEach(o),Za=h(cs),ar=a(cs,"SPAN",{});var pd=d(ar);ei=l(pd,"GroupViTTextModel"),pd.forEach(o),cs.forEach(o),Rr=h(t),Be=a(t,"DIV",{class:!0});var us=d(Be);y(Io.$$.fragment,us),ti=h(us),Y=a(us,"DIV",{class:!0});var Zt=d(Y);y(Lo.$$.fragment,Zt),oi=h(Zt),Ue=a(Zt,"P",{});var Nn=d(Ue);ni=l(Nn,"The "),xn=a(Nn,"A",{href:!0});var cd=d(xn);ri=l(cd,"GroupViTTextModel"),cd.forEach(o),si=l(Nn," forward method, overrides the "),ir=a(Nn,"CODE",{});var ud=d(ir);ai=l(ud,"__call__"),ud.forEach(o),ii=l(Nn," special method."),Nn.forEach(o),li=h(Zt),y(Mt.$$.fragment,Zt),di=h(Zt),y(Et.$$.fragment,Zt),Zt.forEach(o),us.forEach(o),Jr=h(t),Ke=a(t,"H2",{class:!0});var ms=d(Ke);jt=a(ms,"A",{id:!0,class:!0,href:!0});var md=d(jt);lr=a(md,"SPAN",{});var fd=d(lr);y(Ao.$$.fragment,fd),fd.forEach(o),md.forEach(o),pi=h(ms),dr=a(ms,"SPAN",{});var hd=d(dr);ci=l(hd,"GroupViTVisionModel"),hd.forEach(o),ms.forEach(o),Xr=h(t),He=a(t,"DIV",{class:!0});var fs=d(He);y(Do.$$.fragment,fs),ui=h(fs),Q=a(fs,"DIV",{class:!0});var eo=d(Q);y(Oo.$$.fragment,eo),mi=h(eo),Re=a(eo,"P",{});var Wn=d(Re);fi=l(Wn,"The "),kn=a(Wn,"A",{href:!0});var gd=d(kn);hi=l(gd,"GroupViTVisionModel"),gd.forEach(o),gi=l(Wn," forward method, overrides the "),pr=a(Wn,"CODE",{});var _d=d(pr);_i=l(_d,"__call__"),_d.forEach(o),Ti=l(Wn," special method."),Wn.forEach(o),vi=h(eo),y(Pt.$$.fragment,eo),$i=h(eo),y(Ct.$$.fragment,eo),eo.forEach(o),fs.forEach(o),Yr=h(t),Je=a(t,"H2",{class:!0});var hs=d(Je);zt=a(hs,"A",{id:!0,class:!0,href:!0});var Td=d(zt);cr=a(Td,"SPAN",{});var vd=d(cr);y(No.$$.fragment,vd),vd.forEach(o),Td.forEach(o),yi=h(hs),ur=a(hs,"SPAN",{});var $d=d(ur);bi=l($d,"TFGroupViTModel"),$d.forEach(o),hs.forEach(o),Qr=h(t),P=a(t,"DIV",{class:!0});var O=d(P);y(Wo.$$.fragment,O),wi=h(O),So=a(O,"P",{});var gs=d(So);Vi=l(gs,"This model inherits from "),Gn=a(gs,"A",{href:!0});var yd=d(Gn);xi=l(yd,"TFPreTrainedModel"),yd.forEach(o),ki=l(gs,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),gs.forEach(o),Gi=h(O),Bo=a(O,"P",{});var _s=d(Bo);Mi=l(_s,"This model is also a "),Uo=a(_s,"A",{href:!0,rel:!0});var bd=d(Uo);Ei=l(bd,"tf.keras.Model"),bd.forEach(o),ji=l(_s,` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),_s.forEach(o),Pi=h(O),y(qt.$$.fragment,O),Ci=h(O),Z=a(O,"DIV",{class:!0});var to=d(Z);y(Ko.$$.fragment,to),zi=h(to),Xe=a(to,"P",{});var Sn=d(Xe);qi=l(Sn,"The "),Mn=a(Sn,"A",{href:!0});var wd=d(Mn);Fi=l(wd,"TFGroupViTModel"),wd.forEach(o),Ii=l(Sn," forward method, overrides the "),mr=a(Sn,"CODE",{});var Vd=d(mr);Li=l(Vd,"__call__"),Vd.forEach(o),Ai=l(Sn," special method."),Sn.forEach(o),Di=h(to),y(Ft.$$.fragment,to),Oi=h(to),y(It.$$.fragment,to),to.forEach(o),Ni=h(O),ee=a(O,"DIV",{class:!0});var oo=d(ee);y(Ho.$$.fragment,oo),Wi=h(oo),Ye=a(oo,"P",{});var Bn=d(Ye);Si=l(Bn,"The "),En=a(Bn,"A",{href:!0});var xd=d(En);Bi=l(xd,"TFGroupViTModel"),xd.forEach(o),Ui=l(Bn," forward method, overrides the "),fr=a(Bn,"CODE",{});var kd=d(fr);Ki=l(kd,"__call__"),kd.forEach(o),Hi=l(Bn," special method."),Bn.forEach(o),Ri=h(oo),y(Lt.$$.fragment,oo),Ji=h(oo),y(At.$$.fragment,oo),oo.forEach(o),Xi=h(O),te=a(O,"DIV",{class:!0});var no=d(te);y(Ro.$$.fragment,no),Yi=h(no),Qe=a(no,"P",{});var Un=d(Qe);Qi=l(Un,"The "),jn=a(Un,"A",{href:!0});var Gd=d(jn);Zi=l(Gd,"TFGroupViTModel"),Gd.forEach(o),el=l(Un," forward method, overrides the "),hr=a(Un,"CODE",{});var Md=d(hr);tl=l(Md,"__call__"),Md.forEach(o),ol=l(Un," special method."),Un.forEach(o),nl=h(no),y(Dt.$$.fragment,no),rl=h(no),y(Ot.$$.fragment,no),no.forEach(o),O.forEach(o),Zr=h(t),Ze=a(t,"H2",{class:!0});var Ts=d(Ze);Nt=a(Ts,"A",{id:!0,class:!0,href:!0});var Ed=d(Nt);gr=a(Ed,"SPAN",{});var jd=d(gr);y(Jo.$$.fragment,jd),jd.forEach(o),Ed.forEach(o),sl=h(Ts),_r=a(Ts,"SPAN",{});var Pd=d(_r);al=l(Pd,"TFGroupViTTextModel"),Pd.forEach(o),Ts.forEach(o),es=h(t),et=a(t,"DIV",{class:!0});var vs=d(et);y(Xo.$$.fragment,vs),il=h(vs),oe=a(vs,"DIV",{class:!0});var ro=d(oe);y(Yo.$$.fragment,ro),ll=h(ro),tt=a(ro,"P",{});var Kn=d(tt);dl=l(Kn,"The "),Pn=a(Kn,"A",{href:!0});var Cd=d(Pn);pl=l(Cd,"TFGroupViTTextModel"),Cd.forEach(o),cl=l(Kn," forward method, overrides the "),Tr=a(Kn,"CODE",{});var zd=d(Tr);ul=l(zd,"__call__"),zd.forEach(o),ml=l(Kn," special method."),Kn.forEach(o),fl=h(ro),y(Wt.$$.fragment,ro),hl=h(ro),y(St.$$.fragment,ro),ro.forEach(o),vs.forEach(o),ts=h(t),ot=a(t,"H2",{class:!0});var $s=d(ot);Bt=a($s,"A",{id:!0,class:!0,href:!0});var qd=d(Bt);vr=a(qd,"SPAN",{});var Fd=d(vr);y(Qo.$$.fragment,Fd),Fd.forEach(o),qd.forEach(o),gl=h($s),$r=a($s,"SPAN",{});var Id=d($r);_l=l(Id,"TFGroupViTVisionModel"),Id.forEach(o),$s.forEach(o),os=h(t),nt=a(t,"DIV",{class:!0});var ys=d(nt);y(Zo.$$.fragment,ys),Tl=h(ys),ne=a(ys,"DIV",{class:!0});var so=d(ne);y(en.$$.fragment,so),vl=h(so),rt=a(so,"P",{});var Hn=d(rt);$l=l(Hn,"The "),Cn=a(Hn,"A",{href:!0});var Ld=d(Cn);yl=l(Ld,"TFGroupViTVisionModel"),Ld.forEach(o),bl=l(Hn," forward method, overrides the "),yr=a(Hn,"CODE",{});var Ad=d(yr);wl=l(Ad,"__call__"),Ad.forEach(o),Vl=l(Hn," special method."),Hn.forEach(o),xl=h(so),y(Ut.$$.fragment,so),kl=h(so),y(Kt.$$.fragment,so),so.forEach(o),ys.forEach(o),this.h()},h(){g(r,"name","hf:doc:metadata"),g(r,"content",JSON.stringify(fp)),g(c,"id","groupvit"),g(c,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),g(c,"href","#groupvit"),g(u,"class","relative group"),g(U,"id","overview"),g(U,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),g(U,"href","#overview"),g(E,"class","relative group"),g(pe,"href","https://arxiv.org/abs/2202.11094"),g(pe,"rel","nofollow"),g(F,"href","clip"),g(be,"href","https://github.com/xvjiarui/GroupViT/blob/main/demo/GroupViT_hf_inference_notebook.ipynb"),g(be,"rel","nofollow"),g(D,"href","https://huggingface.co/spaces/xvjiarui/GroupViT"),g(D,"rel","nofollow"),g(we,"href","https://huggingface.co/xvjiarui"),g(we,"rel","nofollow"),g(go,"href","https://huggingface.co/ariG23498"),g(go,"rel","nofollow"),g(_o,"href","https://github.com/NVlabs/GroupViT"),g(_o,"rel","nofollow"),g(mt,"id","transformers.GroupViTConfig"),g(mt,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),g(mt,"href","#transformers.GroupViTConfig"),g(Pe,"class","relative group"),g(cn,"href","/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.GroupViTConfig"),g(un,"href","/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.GroupViTModel"),g(mn,"href","/docs/transformers/pr_18020/en/main_classes/configuration#transformers.PretrainedConfig"),g(fn,"href","/docs/transformers/pr_18020/en/main_classes/configuration#transformers.PretrainedConfig"),g(hn,"href","/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.GroupViTConfig"),g(ht,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),g(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),g(gt,"id","transformers.GroupViTTextConfig"),g(gt,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),g(gt,"href","#transformers.GroupViTTextConfig"),g(ze,"class","relative group"),g(gn,"href","/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.GroupViTTextModel"),g(Vo,"href","https://huggingface.co/nvidia/groupvit-gcc-yfcc"),g(Vo,"rel","nofollow"),g(_n,"href","/docs/transformers/pr_18020/en/main_classes/configuration#transformers.PretrainedConfig"),g(Tn,"href","/docs/transformers/pr_18020/en/main_classes/configuration#transformers.PretrainedConfig"),g(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),g(Tt,"id","transformers.GroupViTVisionConfig"),g(Tt,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),g(Tt,"href","#transformers.GroupViTVisionConfig"),g(Ie,"class","relative group"),g(vn,"href","/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.GroupViTVisionModel"),g(Go,"href","https://huggingface.co/nvidia/groupvit-gcc-yfcc"),g(Go,"rel","nofollow"),g($n,"href","/docs/transformers/pr_18020/en/main_classes/configuration#transformers.PretrainedConfig"),g(yn,"href","/docs/transformers/pr_18020/en/main_classes/configuration#transformers.PretrainedConfig"),g(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),g($t,"id","transformers.GroupViTModel"),g($t,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),g($t,"href","#transformers.GroupViTModel"),g(De,"class","relative group"),g(Po,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),g(Po,"rel","nofollow"),g(bn,"href","/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.GroupViTModel"),g(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),g(wn,"href","/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.GroupViTModel"),g(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),g(Vn,"href","/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.GroupViTModel"),g(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),g(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),g(Gt,"id","transformers.GroupViTTextModel"),g(Gt,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),g(Gt,"href","#transformers.GroupViTTextModel"),g(Se,"class","relative group"),g(xn,"href","/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.GroupViTTextModel"),g(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),g(Be,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),g(jt,"id","transformers.GroupViTVisionModel"),g(jt,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),g(jt,"href","#transformers.GroupViTVisionModel"),g(Ke,"class","relative group"),g(kn,"href","/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.GroupViTVisionModel"),g(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),g(He,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),g(zt,"id","transformers.TFGroupViTModel"),g(zt,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),g(zt,"href","#transformers.TFGroupViTModel"),g(Je,"class","relative group"),g(Gn,"href","/docs/transformers/pr_18020/en/main_classes/model#transformers.TFPreTrainedModel"),g(Uo,"href","https://www.tensorflow.org/api_docs/python/tf/keras/Model"),g(Uo,"rel","nofollow"),g(Mn,"href","/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.TFGroupViTModel"),g(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),g(En,"href","/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.TFGroupViTModel"),g(ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),g(jn,"href","/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.TFGroupViTModel"),g(te,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),g(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),g(Nt,"id","transformers.TFGroupViTTextModel"),g(Nt,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),g(Nt,"href","#transformers.TFGroupViTTextModel"),g(Ze,"class","relative group"),g(Pn,"href","/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.TFGroupViTTextModel"),g(oe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),g(et,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),g(Bt,"id","transformers.TFGroupViTVisionModel"),g(Bt,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),g(Bt,"href","#transformers.TFGroupViTVisionModel"),g(ot,"class","relative group"),g(Cn,"href","/docs/transformers/pr_18020/en/model_doc/groupvit#transformers.TFGroupViTVisionModel"),g(ne,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),g(nt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(t,_){e(document.head,r),v(t,T,_),v(t,u,_),e(u,c),e(c,m),b(n,m,null),e(u,p),e(u,G),e(G,ao),v(t,xe,_),v(t,E,_),e(E,U),e(U,ie),b(le,ie,null),e(E,io),e(E,de),e(de,lo),v(t,lt,_),v(t,q,_),e(q,fe),e(q,pe),e(pe,dt),e(q,L),e(q,F),e(F,po),e(q,ke),v(t,pt,_),v(t,he,_),e(he,Ge),v(t,ct,_),v(t,ge,_),e(ge,A),e(A,co),v(t,_e,_),v(t,Te,_),e(Te,uo),v(t,ve,_),v(t,K,_),e(K,N),e(N,$e),e(N,Me),e(Me,Ee),e(N,mo),e(N,M),e(M,z),e(N,ut),e(K,rn),e(K,H),e(H,ye),e(H,be),e(be,sn),e(H,an),e(H,D),e(D,je),e(H,ln),v(t,fo,_),v(t,C,_),e(C,dn),e(C,we),e(we,ho),e(C,pn),e(C,go),e(go,bs),e(C,ws),e(C,_o),e(_o,Vs),e(C,xs),v(t,Dr,_),v(t,Pe,_),e(Pe,mt),e(mt,Rn),b(To,Rn,null),e(Pe,ks),e(Pe,Jn),e(Jn,Gs),v(t,Or,_),v(t,W,_),b(vo,W,null),e(W,Ms),e(W,ft),e(ft,cn),e(cn,Es),e(ft,js),e(ft,un),e(un,Ps),e(ft,Cs),e(W,zs),e(W,Ce),e(Ce,qs),e(Ce,mn),e(mn,Fs),e(Ce,Is),e(Ce,fn),e(fn,Ls),e(Ce,As),e(W,Ds),e(W,ht),b($o,ht,null),e(ht,Os),e(ht,yo),e(yo,Ns),e(yo,hn),e(hn,Ws),e(yo,Ss),v(t,Nr,_),v(t,ze,_),e(ze,gt),e(gt,Xn),b(bo,Xn,null),e(ze,Bs),e(ze,Yn),e(Yn,Us),v(t,Wr,_),v(t,S,_),b(wo,S,null),e(S,Ks),e(S,qe),e(qe,Hs),e(qe,gn),e(gn,Rs),e(qe,Js),e(qe,Vo),e(Vo,Xs),e(qe,Ys),e(S,Qs),e(S,Fe),e(Fe,Zs),e(Fe,_n),e(_n,ea),e(Fe,ta),e(Fe,Tn),e(Tn,oa),e(Fe,na),e(S,ra),b(_t,S,null),v(t,Sr,_),v(t,Ie,_),e(Ie,Tt),e(Tt,Qn),b(xo,Qn,null),e(Ie,sa),e(Ie,Zn),e(Zn,aa),v(t,Br,_),v(t,B,_),b(ko,B,null),e(B,ia),e(B,Le),e(Le,la),e(Le,vn),e(vn,da),e(Le,pa),e(Le,Go),e(Go,ca),e(Le,ua),e(B,ma),e(B,Ae),e(Ae,fa),e(Ae,$n),e($n,ha),e(Ae,ga),e(Ae,yn),e(yn,_a),e(Ae,Ta),e(B,va),b(vt,B,null),v(t,Ur,_),v(t,De,_),e(De,$t),e($t,er),b(Mo,er,null),e(De,$a),e(De,tr),e(tr,ya),v(t,Kr,_),v(t,I,_),b(Eo,I,null),e(I,ba),e(I,jo),e(jo,wa),e(jo,Po),e(Po,Va),e(jo,xa),e(I,ka),e(I,R),b(Co,R,null),e(R,Ga),e(R,Oe),e(Oe,Ma),e(Oe,bn),e(bn,Ea),e(Oe,ja),e(Oe,or),e(or,Pa),e(Oe,Ca),e(R,za),b(yt,R,null),e(R,qa),b(bt,R,null),e(I,Fa),e(I,J),b(zo,J,null),e(J,Ia),e(J,Ne),e(Ne,La),e(Ne,wn),e(wn,Aa),e(Ne,Da),e(Ne,nr),e(nr,Oa),e(Ne,Na),e(J,Wa),b(wt,J,null),e(J,Sa),b(Vt,J,null),e(I,Ba),e(I,X),b(qo,X,null),e(X,Ua),e(X,We),e(We,Ka),e(We,Vn),e(Vn,Ha),e(We,Ra),e(We,rr),e(rr,Ja),e(We,Xa),e(X,Ya),b(xt,X,null),e(X,Qa),b(kt,X,null),v(t,Hr,_),v(t,Se,_),e(Se,Gt),e(Gt,sr),b(Fo,sr,null),e(Se,Za),e(Se,ar),e(ar,ei),v(t,Rr,_),v(t,Be,_),b(Io,Be,null),e(Be,ti),e(Be,Y),b(Lo,Y,null),e(Y,oi),e(Y,Ue),e(Ue,ni),e(Ue,xn),e(xn,ri),e(Ue,si),e(Ue,ir),e(ir,ai),e(Ue,ii),e(Y,li),b(Mt,Y,null),e(Y,di),b(Et,Y,null),v(t,Jr,_),v(t,Ke,_),e(Ke,jt),e(jt,lr),b(Ao,lr,null),e(Ke,pi),e(Ke,dr),e(dr,ci),v(t,Xr,_),v(t,He,_),b(Do,He,null),e(He,ui),e(He,Q),b(Oo,Q,null),e(Q,mi),e(Q,Re),e(Re,fi),e(Re,kn),e(kn,hi),e(Re,gi),e(Re,pr),e(pr,_i),e(Re,Ti),e(Q,vi),b(Pt,Q,null),e(Q,$i),b(Ct,Q,null),v(t,Yr,_),v(t,Je,_),e(Je,zt),e(zt,cr),b(No,cr,null),e(Je,yi),e(Je,ur),e(ur,bi),v(t,Qr,_),v(t,P,_),b(Wo,P,null),e(P,wi),e(P,So),e(So,Vi),e(So,Gn),e(Gn,xi),e(So,ki),e(P,Gi),e(P,Bo),e(Bo,Mi),e(Bo,Uo),e(Uo,Ei),e(Bo,ji),e(P,Pi),b(qt,P,null),e(P,Ci),e(P,Z),b(Ko,Z,null),e(Z,zi),e(Z,Xe),e(Xe,qi),e(Xe,Mn),e(Mn,Fi),e(Xe,Ii),e(Xe,mr),e(mr,Li),e(Xe,Ai),e(Z,Di),b(Ft,Z,null),e(Z,Oi),b(It,Z,null),e(P,Ni),e(P,ee),b(Ho,ee,null),e(ee,Wi),e(ee,Ye),e(Ye,Si),e(Ye,En),e(En,Bi),e(Ye,Ui),e(Ye,fr),e(fr,Ki),e(Ye,Hi),e(ee,Ri),b(Lt,ee,null),e(ee,Ji),b(At,ee,null),e(P,Xi),e(P,te),b(Ro,te,null),e(te,Yi),e(te,Qe),e(Qe,Qi),e(Qe,jn),e(jn,Zi),e(Qe,el),e(Qe,hr),e(hr,tl),e(Qe,ol),e(te,nl),b(Dt,te,null),e(te,rl),b(Ot,te,null),v(t,Zr,_),v(t,Ze,_),e(Ze,Nt),e(Nt,gr),b(Jo,gr,null),e(Ze,sl),e(Ze,_r),e(_r,al),v(t,es,_),v(t,et,_),b(Xo,et,null),e(et,il),e(et,oe),b(Yo,oe,null),e(oe,ll),e(oe,tt),e(tt,dl),e(tt,Pn),e(Pn,pl),e(tt,cl),e(tt,Tr),e(Tr,ul),e(tt,ml),e(oe,fl),b(Wt,oe,null),e(oe,hl),b(St,oe,null),v(t,ts,_),v(t,ot,_),e(ot,Bt),e(Bt,vr),b(Qo,vr,null),e(ot,gl),e(ot,$r),e($r,_l),v(t,os,_),v(t,nt,_),b(Zo,nt,null),e(nt,Tl),e(nt,ne),b(en,ne,null),e(ne,vl),e(ne,rt),e(rt,$l),e(rt,Cn),e(Cn,yl),e(rt,bl),e(rt,yr),e(yr,wl),e(rt,Vl),e(ne,xl),b(Ut,ne,null),e(ne,kl),b(Kt,ne,null),ns=!0},p(t,[_]){const tn={};_&2&&(tn.$$scope={dirty:_,ctx:t}),_t.$set(tn);const br={};_&2&&(br.$$scope={dirty:_,ctx:t}),vt.$set(br);const wr={};_&2&&(wr.$$scope={dirty:_,ctx:t}),yt.$set(wr);const Vr={};_&2&&(Vr.$$scope={dirty:_,ctx:t}),bt.$set(Vr);const on={};_&2&&(on.$$scope={dirty:_,ctx:t}),wt.$set(on);const xr={};_&2&&(xr.$$scope={dirty:_,ctx:t}),Vt.$set(xr);const kr={};_&2&&(kr.$$scope={dirty:_,ctx:t}),xt.$set(kr);const Gr={};_&2&&(Gr.$$scope={dirty:_,ctx:t}),kt.$set(Gr);const st={};_&2&&(st.$$scope={dirty:_,ctx:t}),Mt.$set(st);const Mr={};_&2&&(Mr.$$scope={dirty:_,ctx:t}),Et.$set(Mr);const Er={};_&2&&(Er.$$scope={dirty:_,ctx:t}),Pt.$set(Er);const jr={};_&2&&(jr.$$scope={dirty:_,ctx:t}),Ct.$set(jr);const Pr={};_&2&&(Pr.$$scope={dirty:_,ctx:t}),qt.$set(Pr);const Cr={};_&2&&(Cr.$$scope={dirty:_,ctx:t}),Ft.$set(Cr);const zr={};_&2&&(zr.$$scope={dirty:_,ctx:t}),It.$set(zr);const nn={};_&2&&(nn.$$scope={dirty:_,ctx:t}),Lt.$set(nn);const at={};_&2&&(at.$$scope={dirty:_,ctx:t}),At.$set(at);const qr={};_&2&&(qr.$$scope={dirty:_,ctx:t}),Dt.$set(qr);const Fr={};_&2&&(Fr.$$scope={dirty:_,ctx:t}),Ot.$set(Fr);const it={};_&2&&(it.$$scope={dirty:_,ctx:t}),Wt.$set(it);const Ir={};_&2&&(Ir.$$scope={dirty:_,ctx:t}),St.$set(Ir);const Lr={};_&2&&(Lr.$$scope={dirty:_,ctx:t}),Ut.$set(Lr);const ce={};_&2&&(ce.$$scope={dirty:_,ctx:t}),Kt.$set(ce)},i(t){ns||(w(n.$$.fragment,t),w(le.$$.fragment,t),w(To.$$.fragment,t),w(vo.$$.fragment,t),w($o.$$.fragment,t),w(bo.$$.fragment,t),w(wo.$$.fragment,t),w(_t.$$.fragment,t),w(xo.$$.fragment,t),w(ko.$$.fragment,t),w(vt.$$.fragment,t),w(Mo.$$.fragment,t),w(Eo.$$.fragment,t),w(Co.$$.fragment,t),w(yt.$$.fragment,t),w(bt.$$.fragment,t),w(zo.$$.fragment,t),w(wt.$$.fragment,t),w(Vt.$$.fragment,t),w(qo.$$.fragment,t),w(xt.$$.fragment,t),w(kt.$$.fragment,t),w(Fo.$$.fragment,t),w(Io.$$.fragment,t),w(Lo.$$.fragment,t),w(Mt.$$.fragment,t),w(Et.$$.fragment,t),w(Ao.$$.fragment,t),w(Do.$$.fragment,t),w(Oo.$$.fragment,t),w(Pt.$$.fragment,t),w(Ct.$$.fragment,t),w(No.$$.fragment,t),w(Wo.$$.fragment,t),w(qt.$$.fragment,t),w(Ko.$$.fragment,t),w(Ft.$$.fragment,t),w(It.$$.fragment,t),w(Ho.$$.fragment,t),w(Lt.$$.fragment,t),w(At.$$.fragment,t),w(Ro.$$.fragment,t),w(Dt.$$.fragment,t),w(Ot.$$.fragment,t),w(Jo.$$.fragment,t),w(Xo.$$.fragment,t),w(Yo.$$.fragment,t),w(Wt.$$.fragment,t),w(St.$$.fragment,t),w(Qo.$$.fragment,t),w(Zo.$$.fragment,t),w(en.$$.fragment,t),w(Ut.$$.fragment,t),w(Kt.$$.fragment,t),ns=!0)},o(t){V(n.$$.fragment,t),V(le.$$.fragment,t),V(To.$$.fragment,t),V(vo.$$.fragment,t),V($o.$$.fragment,t),V(bo.$$.fragment,t),V(wo.$$.fragment,t),V(_t.$$.fragment,t),V(xo.$$.fragment,t),V(ko.$$.fragment,t),V(vt.$$.fragment,t),V(Mo.$$.fragment,t),V(Eo.$$.fragment,t),V(Co.$$.fragment,t),V(yt.$$.fragment,t),V(bt.$$.fragment,t),V(zo.$$.fragment,t),V(wt.$$.fragment,t),V(Vt.$$.fragment,t),V(qo.$$.fragment,t),V(xt.$$.fragment,t),V(kt.$$.fragment,t),V(Fo.$$.fragment,t),V(Io.$$.fragment,t),V(Lo.$$.fragment,t),V(Mt.$$.fragment,t),V(Et.$$.fragment,t),V(Ao.$$.fragment,t),V(Do.$$.fragment,t),V(Oo.$$.fragment,t),V(Pt.$$.fragment,t),V(Ct.$$.fragment,t),V(No.$$.fragment,t),V(Wo.$$.fragment,t),V(qt.$$.fragment,t),V(Ko.$$.fragment,t),V(Ft.$$.fragment,t),V(It.$$.fragment,t),V(Ho.$$.fragment,t),V(Lt.$$.fragment,t),V(At.$$.fragment,t),V(Ro.$$.fragment,t),V(Dt.$$.fragment,t),V(Ot.$$.fragment,t),V(Jo.$$.fragment,t),V(Xo.$$.fragment,t),V(Yo.$$.fragment,t),V(Wt.$$.fragment,t),V(St.$$.fragment,t),V(Qo.$$.fragment,t),V(Zo.$$.fragment,t),V(en.$$.fragment,t),V(Ut.$$.fragment,t),V(Kt.$$.fragment,t),ns=!1},d(t){o(r),t&&o(T),t&&o(u),x(n),t&&o(xe),t&&o(E),x(le),t&&o(lt),t&&o(q),t&&o(pt),t&&o(he),t&&o(ct),t&&o(ge),t&&o(_e),t&&o(Te),t&&o(ve),t&&o(K),t&&o(fo),t&&o(C),t&&o(Dr),t&&o(Pe),x(To),t&&o(Or),t&&o(W),x(vo),x($o),t&&o(Nr),t&&o(ze),x(bo),t&&o(Wr),t&&o(S),x(wo),x(_t),t&&o(Sr),t&&o(Ie),x(xo),t&&o(Br),t&&o(B),x(ko),x(vt),t&&o(Ur),t&&o(De),x(Mo),t&&o(Kr),t&&o(I),x(Eo),x(Co),x(yt),x(bt),x(zo),x(wt),x(Vt),x(qo),x(xt),x(kt),t&&o(Hr),t&&o(Se),x(Fo),t&&o(Rr),t&&o(Be),x(Io),x(Lo),x(Mt),x(Et),t&&o(Jr),t&&o(Ke),x(Ao),t&&o(Xr),t&&o(He),x(Do),x(Oo),x(Pt),x(Ct),t&&o(Yr),t&&o(Je),x(No),t&&o(Qr),t&&o(P),x(Wo),x(qt),x(Ko),x(Ft),x(It),x(Ho),x(Lt),x(At),x(Ro),x(Dt),x(Ot),t&&o(Zr),t&&o(Ze),x(Jo),t&&o(es),t&&o(et),x(Xo),x(Yo),x(Wt),x(St),t&&o(ts),t&&o(ot),x(Qo),t&&o(os),t&&o(nt),x(Zo),x(en),x(Ut),x(Kt)}}}const fp={local:"groupvit",sections:[{local:"overview",title:"Overview"},{local:"transformers.GroupViTConfig",title:"GroupViTConfig"},{local:"transformers.GroupViTTextConfig",title:"GroupViTTextConfig"},{local:"transformers.GroupViTVisionConfig",title:"GroupViTVisionConfig"},{local:"transformers.GroupViTModel",title:"GroupViTModel"},{local:"transformers.GroupViTTextModel",title:"GroupViTTextModel"},{local:"transformers.GroupViTVisionModel",title:"GroupViTVisionModel"},{local:"transformers.TFGroupViTModel",title:"TFGroupViTModel"},{local:"transformers.TFGroupViTTextModel",title:"TFGroupViTTextModel"},{local:"transformers.TFGroupViTVisionModel",title:"TFGroupViTVisionModel"}],title:"GroupViT"};function hp(k){return Sd(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class bp extends Dd{constructor(r){super();Od(this,r,hp,mp,Nd,{})}}export{bp as default,fp as metadata};
