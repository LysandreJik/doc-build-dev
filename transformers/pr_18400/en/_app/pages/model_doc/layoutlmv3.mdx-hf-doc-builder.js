import{S as Fp,i as Pp,s as Cp,e as s,k as c,w as y,t as n,M as Ap,c as r,d as t,m as u,a as i,x as b,h as a,b as l,N as jp,G as e,g,y as L,q as k,o as x,B as w,v as Ip,L as ln}from"../../chunks/vendor-hf-doc-builder.js";import{T as xs}from"../../chunks/Tip-hf-doc-builder.js";import{D as P}from"../../chunks/Docstring-hf-doc-builder.js";import{C as dn}from"../../chunks/CodeBlock-hf-doc-builder.js";import{I as te}from"../../chunks/IconCopyLink-hf-doc-builder.js";import{E as rn}from"../../chunks/ExampleCodeBlock-hf-doc-builder.js";function Np(q){let p,T,_,f,v;return f=new dn({props:{code:`from transformers import LayoutLMv3Model, LayoutLMv3Config

# Initializing a LayoutLMv3 microsoft/layoutlmv3-base style configuration
configuration = LayoutLMv3Config()

# Initializing a model from the microsoft/layoutlmv3-base style configuration
model = LayoutLMv3Model(configuration)

# Accessing the model configuration
configuration = model.config`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> LayoutLMv3Model, LayoutLMv3Config

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a LayoutLMv3 microsoft/layoutlmv3-base style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = LayoutLMv3Config()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the microsoft/layoutlmv3-base style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LayoutLMv3Model(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`}}),{c(){p=s("p"),T=n("Example:"),_=c(),y(f.$$.fragment)},l(d){p=r(d,"P",{});var h=i(p);T=a(h,"Example:"),h.forEach(t),_=u(d),b(f.$$.fragment,d)},m(d,h){g(d,p,h),e(p,T),g(d,_,h),L(f,d,h),v=!0},p:ln,i(d){v||(k(f.$$.fragment,d),v=!0)},o(d){x(f.$$.fragment,d),v=!1},d(d){d&&t(p),d&&t(_),w(f,d)}}}function Dp(q){let p,T,_,f,v;return f=new dn({props:{code:`from transformers import LayoutLMv3FeatureExtractor
from PIL import Image

image = Image.open("name_of_your_document - can be a png file, pdf, etc.").convert("RGB")

# option 1: with apply_ocr=True (default)
feature_extractor = LayoutLMv3FeatureExtractor()
encoding = feature_extractor(image, return_tensors="pt")
print(encoding.keys())
# dict_keys(['pixel_values', 'words', 'boxes'])

# option 2: with apply_ocr=False
feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
encoding = feature_extractor(image, return_tensors="pt")
print(encoding.keys())
# dict_keys(['pixel_values'])`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> LayoutLMv3FeatureExtractor
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image

<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(<span class="hljs-string">&quot;name_of_your_document - can be a png file, pdf, etc.&quot;</span>).convert(<span class="hljs-string">&quot;RGB&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># option 1: with apply_ocr=True (default)</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>feature_extractor = LayoutLMv3FeatureExtractor()
<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = feature_extractor(image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(encoding.keys())
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># dict_keys([&#x27;pixel_values&#x27;, &#x27;words&#x27;, &#x27;boxes&#x27;])</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># option 2: with apply_ocr=False</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=<span class="hljs-literal">False</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = feature_extractor(image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(encoding.keys())
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># dict_keys([&#x27;pixel_values&#x27;])</span>`}}),{c(){p=s("p"),T=n("Examples:"),_=c(),y(f.$$.fragment)},l(d){p=r(d,"P",{});var h=i(p);T=a(h,"Examples:"),h.forEach(t),_=u(d),b(f.$$.fragment,d)},m(d,h){g(d,p,h),e(p,T),g(d,_,h),L(f,d,h),v=!0},p:ln,i(d){v||(k(f.$$.fragment,d),v=!0)},o(d){x(f.$$.fragment,d),v=!1},d(d){d&&t(p),d&&t(_),w(f,d)}}}function Sp(q){let p,T,_,f,v;return{c(){p=s("p"),T=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),_=s("code"),f=n("Module"),v=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(d){p=r(d,"P",{});var h=i(p);T=a(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),_=r(h,"CODE",{});var C=i(_);f=a(C,"Module"),C.forEach(t),v=a(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(t)},m(d,h){g(d,p,h),e(p,T),e(p,_),e(_,f),e(p,v)},d(d){d&&t(p)}}}function Op(q){let p,T,_,f,v;return f=new dn({props:{code:`from transformers import AutoProcessor, AutoModel
from datasets import load_dataset

processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = AutoModel.from_pretrained("microsoft/layoutlmv3-base")

dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
example = dataset[0]
image = example["image"]
words = example["tokens"]
boxes = example["bboxes"]

encoding = processor(image, words, boxes=boxes, return_tensors="pt")

outputs = model(**encoding)
last_hidden_states = outputs.last_hidden_state`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, AutoModel
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlmv3-base&quot;</span>, apply_ocr=<span class="hljs-literal">False</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = AutoModel.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlmv3-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = load_dataset(<span class="hljs-string">&quot;nielsr/funsd-layoutlmv3&quot;</span>, split=<span class="hljs-string">&quot;train&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>example = dataset[<span class="hljs-number">0</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>image = example[<span class="hljs-string">&quot;image&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>words = example[<span class="hljs-string">&quot;tokens&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>boxes = example[<span class="hljs-string">&quot;bboxes&quot;</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = processor(image, words, boxes=boxes, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**encoding)
<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`}}),{c(){p=s("p"),T=n("Examples:"),_=c(),y(f.$$.fragment)},l(d){p=r(d,"P",{});var h=i(p);T=a(h,"Examples:"),h.forEach(t),_=u(d),b(f.$$.fragment,d)},m(d,h){g(d,p,h),e(p,T),g(d,_,h),L(f,d,h),v=!0},p:ln,i(d){v||(k(f.$$.fragment,d),v=!0)},o(d){x(f.$$.fragment,d),v=!1},d(d){d&&t(p),d&&t(_),w(f,d)}}}function Rp(q){let p,T,_,f,v;return{c(){p=s("p"),T=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),_=s("code"),f=n("Module"),v=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(d){p=r(d,"P",{});var h=i(p);T=a(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),_=r(h,"CODE",{});var C=i(_);f=a(C,"Module"),C.forEach(t),v=a(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(t)},m(d,h){g(d,p,h),e(p,T),e(p,_),e(_,f),e(p,v)},d(d){d&&t(p)}}}function Wp(q){let p,T,_,f,v;return f=new dn({props:{code:`from transformers import AutoProcessor, AutoModelForSequenceClassification
from datasets import load_dataset
import torch

processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = AutoModelForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base")

dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
example = dataset[0]
image = example["image"]
words = example["tokens"]
boxes = example["bboxes"]

encoding = processor(image, words, boxes=boxes, return_tensors="pt")
sequence_label = torch.tensor([1])

outputs = model(**encoding, labels=sequence_label)
loss = outputs.loss
logits = outputs.logits`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, AutoModelForSequenceClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlmv3-base&quot;</span>, apply_ocr=<span class="hljs-literal">False</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = AutoModelForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlmv3-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = load_dataset(<span class="hljs-string">&quot;nielsr/funsd-layoutlmv3&quot;</span>, split=<span class="hljs-string">&quot;train&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>example = dataset[<span class="hljs-number">0</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>image = example[<span class="hljs-string">&quot;image&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>words = example[<span class="hljs-string">&quot;tokens&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>boxes = example[<span class="hljs-string">&quot;bboxes&quot;</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = processor(image, words, boxes=boxes, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>sequence_label = torch.tensor([<span class="hljs-number">1</span>])

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**encoding, labels=sequence_label)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`}}),{c(){p=s("p"),T=n("Examples:"),_=c(),y(f.$$.fragment)},l(d){p=r(d,"P",{});var h=i(p);T=a(h,"Examples:"),h.forEach(t),_=u(d),b(f.$$.fragment,d)},m(d,h){g(d,p,h),e(p,T),g(d,_,h),L(f,d,h),v=!0},p:ln,i(d){v||(k(f.$$.fragment,d),v=!0)},o(d){x(f.$$.fragment,d),v=!1},d(d){d&&t(p),d&&t(_),w(f,d)}}}function Bp(q){let p,T,_,f,v;return{c(){p=s("p"),T=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),_=s("code"),f=n("Module"),v=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(d){p=r(d,"P",{});var h=i(p);T=a(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),_=r(h,"CODE",{});var C=i(_);f=a(C,"Module"),C.forEach(t),v=a(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(t)},m(d,h){g(d,p,h),e(p,T),e(p,_),e(_,f),e(p,v)},d(d){d&&t(p)}}}function Up(q){let p,T,_,f,v;return f=new dn({props:{code:`from transformers import AutoProcessor, AutoModelForTokenClassification
from datasets import load_dataset

processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = AutoModelForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=7)

dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
example = dataset[0]
image = example["image"]
words = example["tokens"]
boxes = example["bboxes"]
word_labels = example["ner_tags"]

encoding = processor(image, words, boxes=boxes, word_labels=word_labels, return_tensors="pt")

outputs = model(**encoding)
loss = outputs.loss
logits = outputs.logits`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, AutoModelForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlmv3-base&quot;</span>, apply_ocr=<span class="hljs-literal">False</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = AutoModelForTokenClassification.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlmv3-base&quot;</span>, num_labels=<span class="hljs-number">7</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = load_dataset(<span class="hljs-string">&quot;nielsr/funsd-layoutlmv3&quot;</span>, split=<span class="hljs-string">&quot;train&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>example = dataset[<span class="hljs-number">0</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>image = example[<span class="hljs-string">&quot;image&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>words = example[<span class="hljs-string">&quot;tokens&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>boxes = example[<span class="hljs-string">&quot;bboxes&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>word_labels = example[<span class="hljs-string">&quot;ner_tags&quot;</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = processor(image, words, boxes=boxes, word_labels=word_labels, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**encoding)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`}}),{c(){p=s("p"),T=n("Examples:"),_=c(),y(f.$$.fragment)},l(d){p=r(d,"P",{});var h=i(p);T=a(h,"Examples:"),h.forEach(t),_=u(d),b(f.$$.fragment,d)},m(d,h){g(d,p,h),e(p,T),g(d,_,h),L(f,d,h),v=!0},p:ln,i(d){v||(k(f.$$.fragment,d),v=!0)},o(d){x(f.$$.fragment,d),v=!1},d(d){d&&t(p),d&&t(_),w(f,d)}}}function Qp(q){let p,T,_,f,v;return{c(){p=s("p"),T=n("Although the recipe for forward pass needs to be defined within this function, one should call the "),_=s("code"),f=n("Module"),v=n(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(d){p=r(d,"P",{});var h=i(p);T=a(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),_=r(h,"CODE",{});var C=i(_);f=a(C,"Module"),C.forEach(t),v=a(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(t)},m(d,h){g(d,p,h),e(p,T),e(p,_),e(_,f),e(p,v)},d(d){d&&t(p)}}}function Vp(q){let p,T,_,f,v;return f=new dn({props:{code:`from transformers import AutoProcessor, AutoModelForQuestionAnswering
from datasets import load_dataset
import torch

processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = AutoModelForQuestionAnswering.from_pretrained("microsoft/layoutlmv3-base")

dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
example = dataset[0]
image = example["image"]
question = "what's his name?"
words = example["tokens"]
boxes = example["bboxes"]

encoding = processor(image, question, words, boxes=boxes, return_tensors="pt")
start_positions = torch.tensor([1])
end_positions = torch.tensor([3])

outputs = model(**encoding, start_positions=start_positions, end_positions=end_positions)
loss = outputs.loss
start_scores = outputs.start_logits
end_scores = outputs.end_logits`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, AutoModelForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlmv3-base&quot;</span>, apply_ocr=<span class="hljs-literal">False</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = AutoModelForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlmv3-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = load_dataset(<span class="hljs-string">&quot;nielsr/funsd-layoutlmv3&quot;</span>, split=<span class="hljs-string">&quot;train&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>example = dataset[<span class="hljs-number">0</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>image = example[<span class="hljs-string">&quot;image&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>question = <span class="hljs-string">&quot;what&#x27;s his name?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>words = example[<span class="hljs-string">&quot;tokens&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>boxes = example[<span class="hljs-string">&quot;bboxes&quot;</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = processor(image, question, words, boxes=boxes, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>start_positions = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>end_positions = torch.tensor([<span class="hljs-number">3</span>])

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**encoding, start_positions=start_positions, end_positions=end_positions)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>start_scores = outputs.start_logits
<span class="hljs-meta">&gt;&gt;&gt; </span>end_scores = outputs.end_logits`}}),{c(){p=s("p"),T=n("Examples:"),_=c(),y(f.$$.fragment)},l(d){p=r(d,"P",{});var h=i(p);T=a(h,"Examples:"),h.forEach(t),_=u(d),b(f.$$.fragment,d)},m(d,h){g(d,p,h),e(p,T),g(d,_,h),L(f,d,h),v=!0},p:ln,i(d){v||(k(f.$$.fragment,d),v=!0)},o(d){x(f.$$.fragment,d),v=!1},d(d){d&&t(p),d&&t(_),w(f,d)}}}function Hp(q){let p,T,_,f,v,d,h,C,ws,Ta,le,Te,cn,lt,Ts,un,Ms,Ma,Q,$s,dt,zs,qs,go,Es,Fs,_o,Ps,Cs,$a,vo,As,za,yo,pn,js,qa,bo,Is,Ea,V,Me,Ns,Lo,Ds,Ss,ct,mn,Os,Rs,N,Ws,ko,Bs,Us,xo,Qs,Vs,wo,Hs,Gs,To,Xs,Ys,Js,de,Ks,Mo,Zs,er,$o,tr,or,nr,ut,ar,pt,sr,rr,ir,mt,lr,ht,dr,cr,Fa,$e,wc,Pa,ze,ur,zo,pr,mr,Ca,oe,hr,ft,fr,gr,gt,_r,vr,Aa,ce,qe,hn,_t,yr,fn,br,ja,D,vt,Lr,ue,kr,qo,xr,wr,yt,Tr,Mr,$r,pe,zr,Eo,qr,Er,Fo,Fr,Pr,Cr,Ee,Ia,me,Fe,gn,bt,Ar,_n,jr,Na,S,Lt,Ir,vn,Nr,Dr,kt,Sr,yn,Or,Rr,Wr,ne,xt,Br,bn,Ur,Qr,Pe,Da,he,Ce,Ln,wt,Vr,kn,Hr,Sa,A,Tt,Gr,F,Xr,xn,Yr,Jr,Po,Kr,Zr,wn,ei,ti,Tn,oi,ni,Mn,ai,si,$n,ri,ii,zn,li,di,ci,Mt,ui,Co,pi,mi,hi,Ao,jo,fi,gi,_i,Ae,$t,vi,qn,yi,bi,Io,zt,Oa,fe,je,En,qt,Li,Fn,ki,Ra,O,Et,xi,Ft,wi,Pn,Ti,Mi,$i,Pt,zi,No,qi,Ei,Fi,Ie,Ct,Pi,Cn,Ci,Wa,ge,Ne,An,At,Ai,jn,ji,Ba,I,jt,Ii,In,Ni,Di,Do,So,Si,Oi,Ri,z,Wi,Oo,Bi,Ui,Ro,Qi,Vi,Wo,Hi,Gi,Nn,Xi,Yi,Dn,Ji,Ki,Sn,Zi,el,On,tl,ol,Rn,nl,al,Wn,sl,rl,il,ae,It,ll,M,dl,Bn,cl,ul,Nt,Un,pl,ml,hl,Bo,fl,gl,Qn,_l,vl,Vn,yl,bl,Dt,Hn,Ll,kl,xl,Gn,wl,Tl,Uo,Ml,$l,Xn,zl,ql,Yn,El,Fl,Jn,Pl,Cl,Kn,Al,jl,Zn,Il,Nl,St,ea,Dl,Sl,Ol,ta,Rl,Wl,Bl,oa,Ul,Ua,_e,De,na,Ot,Ql,aa,Vl,Qa,K,Rt,Hl,Wt,Gl,Bt,Xl,Yl,Jl,H,Ut,Kl,ve,Zl,Qo,ed,td,sa,od,nd,ad,Se,sd,Oe,Va,ye,Re,ra,Qt,rd,ia,id,Ha,R,Vt,ld,Ht,dd,Gt,cd,ud,pd,Xt,md,Yt,hd,fd,gd,G,Jt,_d,be,vd,Vo,yd,bd,la,Ld,kd,xd,We,wd,Be,Ga,Le,Ue,da,Kt,Td,ca,Md,Xa,W,Zt,$d,B,zd,eo,qd,Ed,to,Fd,Pd,oo,Cd,Ad,no,jd,Id,Nd,ao,Dd,so,Sd,Od,Rd,X,ro,Wd,ke,Bd,Ho,Ud,Qd,ua,Vd,Hd,Gd,Qe,Xd,Ve,Ya,xe,He,pa,io,Yd,ma,Jd,Ja,U,lo,Kd,Z,Zd,co,ec,tc,ha,oc,nc,fa,ac,sc,rc,uo,ic,po,lc,dc,cc,Y,mo,uc,we,pc,Go,mc,hc,ga,fc,gc,_c,Ge,vc,Xe,Ka;return d=new te({}),lt=new te({}),_t=new te({}),vt=new P({props:{name:"class transformers.LayoutLMv3Config",anchor:"transformers.LayoutLMv3Config",parameters:[{name:"vocab_size",val:" = 50265"},{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 512"},{name:"type_vocab_size",val:" = 2"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-05"},{name:"pad_token_id",val:" = 1"},{name:"bos_token_id",val:" = 0"},{name:"eos_token_id",val:" = 2"},{name:"max_2d_position_embeddings",val:" = 1024"},{name:"coordinate_size",val:" = 128"},{name:"shape_size",val:" = 128"},{name:"has_relative_attention_bias",val:" = True"},{name:"rel_pos_bins",val:" = 32"},{name:"max_rel_pos",val:" = 128"},{name:"rel_2d_pos_bins",val:" = 64"},{name:"max_rel_2d_pos",val:" = 256"},{name:"has_spatial_attention_bias",val:" = True"},{name:"text_embed",val:" = True"},{name:"visual_embed",val:" = True"},{name:"input_size",val:" = 224"},{name:"num_channels",val:" = 3"},{name:"patch_size",val:" = 16"},{name:"classifier_dropout",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.LayoutLMv3Config.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 50265) &#x2014;
Vocabulary size of the LayoutLMv3 model. Defines the number of different tokens that can be represented by
the <code>inputs_ids</code> passed when calling <a href="/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3Model">LayoutLMv3Model</a>.`,name:"vocab_size"},{anchor:"transformers.LayoutLMv3Config.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimension of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.LayoutLMv3Config.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.LayoutLMv3Config.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.LayoutLMv3Config.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimension of the &#x201C;intermediate&#x201D; (i.e., feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.LayoutLMv3Config.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;selu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.LayoutLMv3Config.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.LayoutLMv3Config.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.LayoutLMv3Config.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.LayoutLMv3Config.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed when calling <a href="/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3Model">LayoutLMv3Model</a>.`,name:"type_vocab_size"},{anchor:"transformers.LayoutLMv3Config.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.LayoutLMv3Config.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-5) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.LayoutLMv3Config.max_2d_position_embeddings",description:`<strong>max_2d_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
The maximum value that the 2D position embedding might ever be used with. Typically set this to something
large just in case (e.g., 1024).`,name:"max_2d_position_embeddings"},{anchor:"transformers.LayoutLMv3Config.coordinate_size",description:`<strong>coordinate_size</strong> (<code>int</code>, <em>optional</em>, defaults to <code>128</code>) &#x2014;
Dimension of the coordinate embeddings.`,name:"coordinate_size"},{anchor:"transformers.LayoutLMv3Config.shape_size",description:`<strong>shape_size</strong> (<code>int</code>, <em>optional</em>, defaults to <code>128</code>) &#x2014;
Dimension of the width and height embeddings.`,name:"shape_size"},{anchor:"transformers.LayoutLMv3Config.has_relative_attention_bias",description:`<strong>has_relative_attention_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to use a relative attention bias in the self-attention mechanism.`,name:"has_relative_attention_bias"},{anchor:"transformers.LayoutLMv3Config.rel_pos_bins",description:`<strong>rel_pos_bins</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
The number of relative position bins to be used in the self-attention mechanism.`,name:"rel_pos_bins"},{anchor:"transformers.LayoutLMv3Config.max_rel_pos",description:`<strong>max_rel_pos</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
The maximum number of relative positions to be used in the self-attention mechanism.`,name:"max_rel_pos"},{anchor:"transformers.LayoutLMv3Config.max_rel_2d_pos",description:`<strong>max_rel_2d_pos</strong> (<code>int</code>, <em>optional</em>, defaults to 256) &#x2014;
The maximum number of relative 2D positions in the self-attention mechanism.`,name:"max_rel_2d_pos"},{anchor:"transformers.LayoutLMv3Config.rel_2d_pos_bins",description:`<strong>rel_2d_pos_bins</strong> (<code>int</code>, <em>optional</em>, defaults to 64) &#x2014;
The number of 2D relative position bins in the self-attention mechanism.`,name:"rel_2d_pos_bins"},{anchor:"transformers.LayoutLMv3Config.has_spatial_attention_bias",description:`<strong>has_spatial_attention_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to use a spatial attention bias in the self-attention mechanism.`,name:"has_spatial_attention_bias"},{anchor:"transformers.LayoutLMv3Config.visual_embed",description:`<strong>visual_embed</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to add patch embeddings.`,name:"visual_embed"},{anchor:"transformers.LayoutLMv3Config.input_size",description:`<strong>input_size</strong> (<code>int</code>, <em>optional</em>, defaults to <code>224</code>) &#x2014;
The size (resolution) of the images.`,name:"input_size"},{anchor:"transformers.LayoutLMv3Config.num_channels",description:`<strong>num_channels</strong> (<code>int</code>, <em>optional</em>, defaults to <code>3</code>) &#x2014;
The number of channels of the images.`,name:"num_channels"},{anchor:"transformers.LayoutLMv3Config.patch_size",description:`<strong>patch_size</strong> (<code>int</code>, <em>optional</em>, defaults to <code>16</code>) &#x2014;
The size (resolution) of the patches.`,name:"patch_size"},{anchor:"transformers.LayoutLMv3Config.classifier_dropout",description:`<strong>classifier_dropout</strong> (<code>float</code>, <em>optional</em>) &#x2014;
The dropout ratio for the classification head.`,name:"classifier_dropout"}],source:"https://github.com/huggingface/transformers/blob/vr_18400/src/transformers/models/layoutlmv3/configuration_layoutlmv3.py#L40"}}),Ee=new rn({props:{anchor:"transformers.LayoutLMv3Config.example",$$slots:{default:[Np]},$$scope:{ctx:q}}}),bt=new te({}),Lt=new P({props:{name:"class transformers.LayoutLMv3FeatureExtractor",anchor:"transformers.LayoutLMv3FeatureExtractor",parameters:[{name:"do_resize",val:" = True"},{name:"size",val:" = 224"},{name:"resample",val:" = <Resampling.BILINEAR: 2>"},{name:"do_normalize",val:" = True"},{name:"image_mean",val:" = None"},{name:"image_std",val:" = None"},{name:"apply_ocr",val:" = True"},{name:"ocr_lang",val:" = None"},{name:"tesseract_config",val:" = ''"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.LayoutLMv3FeatureExtractor.do_resize",description:`<strong>do_resize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to resize the input to a certain <code>size</code>.`,name:"do_resize"},{anchor:"transformers.LayoutLMv3FeatureExtractor.size",description:`<strong>size</strong> (<code>int</code> or <code>Tuple(int)</code>, <em>optional</em>, defaults to 224) &#x2014;
Resize the input to the given size. If a tuple is provided, it should be (width, height). If only an
integer is provided, then the input will be resized to (size, size). Only has an effect if <code>do_resize</code> is
set to <code>True</code>.`,name:"size"},{anchor:"transformers.LayoutLMv3FeatureExtractor.resample",description:`<strong>resample</strong> (<code>int</code>, <em>optional</em>, defaults to <code>PIL.Image.BILINEAR</code>) &#x2014;
An optional resampling filter. This can be one of <code>PIL.Image.NEAREST</code>, <code>PIL.Image.BOX</code>,
<code>PIL.Image.BILINEAR</code>, <code>PIL.Image.HAMMING</code>, <code>PIL.Image.BICUBIC</code> or <code>PIL.Image.LANCZOS</code>. Only has an effect
if <code>do_resize</code> is set to <code>True</code>.`,name:"resample"},{anchor:"transformers.LayoutLMv3FeatureExtractor.do_normalize",description:`<strong>do_normalize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to normalize the input with mean and standard deviation.`,name:"do_normalize"},{anchor:"transformers.LayoutLMv3FeatureExtractor.image_mean",description:`<strong>image_mean</strong> (<code>List[int]</code>, defaults to <code>[0.5, 0.5, 0.5]</code>) &#x2014;
The sequence of means for each channel, to be used when normalizing images.`,name:"image_mean"},{anchor:"transformers.LayoutLMv3FeatureExtractor.image_std",description:`<strong>image_std</strong> (<code>List[int]</code>, defaults to <code>[0.5, 0.5, 0.5]</code>) &#x2014;
The sequence of standard deviations for each channel, to be used when normalizing images.`,name:"image_std"},{anchor:"transformers.LayoutLMv3FeatureExtractor.apply_ocr",description:`<strong>apply_ocr</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to apply the Tesseract OCR engine to get words + normalized bounding boxes.`,name:"apply_ocr"},{anchor:"transformers.LayoutLMv3FeatureExtractor.ocr_lang",description:`<strong>ocr_lang</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The language, specified by its ISO code, to be used by the Tesseract OCR engine. By default, English is
used.`,name:"ocr_lang"},{anchor:"transformers.LayoutLMv3FeatureExtractor.tesseract_config",description:`<strong>tesseract_config</strong> (<code>str</code>, <em>optional</em>) &#x2014;
Any additional custom configuration flags that are forwarded to the <code>config</code> parameter when calling
Tesseract. For example: &#x2018;&#x2014;psm 6&#x2019;.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>LayoutLMv3FeatureExtractor uses Google&#x2019;s Tesseract OCR engine under the hood.</p>

					</div>`,name:"tesseract_config"}],source:"https://github.com/huggingface/transformers/blob/vr_18400/src/transformers/models/layoutlmv3/feature_extraction_layoutlmv3.py#L81"}}),xt=new P({props:{name:"__call__",anchor:"transformers.LayoutLMv3FeatureExtractor.__call__",parameters:[{name:"images",val:": typing.Union[PIL.Image.Image, numpy.ndarray, ForwardRef('torch.Tensor'), typing.List[PIL.Image.Image], typing.List[numpy.ndarray], typing.List[ForwardRef('torch.Tensor')]]"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.LayoutLMv3FeatureExtractor.__call__.images",description:`<strong>images</strong> (<code>PIL.Image.Image</code>, <code>np.ndarray</code>, <code>torch.Tensor</code>, <code>List[PIL.Image.Image]</code>, <code>List[np.ndarray]</code>, <code>List[torch.Tensor]</code>) &#x2014;
The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
number of channels, H and W are image height and width.`,name:"images"},{anchor:"transformers.LayoutLMv3FeatureExtractor.__call__.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/pr_18400/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>, defaults to <code>&apos;np&apos;</code>) &#x2014;
If set, will return tensors of a particular framework. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return NumPy <code>np.ndarray</code> objects.</li>
<li><code>&apos;jax&apos;</code>: Return JAX <code>jnp.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"}],source:"https://github.com/huggingface/transformers/blob/vr_18400/src/transformers/models/layoutlmv3/feature_extraction_layoutlmv3.py#L147",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18400/en/main_classes/feature_extractor#transformers.BatchFeature"
>BatchFeature</a> with the following fields:</p>
<ul>
<li><strong>pixel_values</strong> \u2014 Pixel values to be fed to a model, of shape (batch_size, num_channels, height,
width).</li>
<li><strong>words</strong> \u2014 Optional words as identified by Tesseract OCR (only when <a
  href="/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3FeatureExtractor"
>LayoutLMv3FeatureExtractor</a> was
initialized with <code>apply_ocr</code> set to <code>True</code>).</li>
<li><strong>boxes</strong> \u2014 Optional bounding boxes as identified by Tesseract OCR, normalized based on the image size
(only when <a
  href="/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3FeatureExtractor"
>LayoutLMv3FeatureExtractor</a> was initialized with <code>apply_ocr</code> set to <code>True</code>).</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18400/en/main_classes/feature_extractor#transformers.BatchFeature"
>BatchFeature</a></p>
`}}),Pe=new rn({props:{anchor:"transformers.LayoutLMv3FeatureExtractor.__call__.example",$$slots:{default:[Dp]},$$scope:{ctx:q}}}),wt=new te({}),Tt=new P({props:{name:"class transformers.LayoutLMv3Tokenizer",anchor:"transformers.LayoutLMv3Tokenizer",parameters:[{name:"vocab_file",val:""},{name:"merges_file",val:""},{name:"errors",val:" = 'replace'"},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"sep_token",val:" = '</s>'"},{name:"cls_token",val:" = '<s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"mask_token",val:" = '<mask>'"},{name:"add_prefix_space",val:" = True"},{name:"cls_token_box",val:" = [0, 0, 0, 0]"},{name:"sep_token_box",val:" = [0, 0, 0, 0]"},{name:"pad_token_box",val:" = [0, 0, 0, 0]"},{name:"pad_token_label",val:" = -100"},{name:"only_label_first_subword",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.LayoutLMv3Tokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.LayoutLMv3Tokenizer.merges_file",description:`<strong>merges_file</strong> (<code>str</code>) &#x2014;
Path to the merges file.`,name:"merges_file"},{anchor:"transformers.LayoutLMv3Tokenizer.errors",description:`<strong>errors</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;replace&quot;</code>) &#x2014;
Paradigm to follow when decoding bytes to UTF-8. See
<a href="https://docs.python.org/3/library/stdtypes.html#bytes.decode" rel="nofollow">bytes.decode</a> for more information.`,name:"errors"},{anchor:"transformers.LayoutLMv3Tokenizer.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the beginning of
sequence. The token used is the <code>cls_token</code>.</p>

					</div>`,name:"bos_token"},{anchor:"transformers.LayoutLMv3Tokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the end of sequence.
The token used is the <code>sep_token</code>.</p>

					</div>`,name:"eos_token"},{anchor:"transformers.LayoutLMv3Tokenizer.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.LayoutLMv3Tokenizer.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.LayoutLMv3Tokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.LayoutLMv3Tokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.LayoutLMv3Tokenizer.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;mask&gt;&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.LayoutLMv3Tokenizer.add_prefix_space",description:`<strong>add_prefix_space</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to add an initial space to the input. This allows to treat the leading word just as any
other word. (RoBERTa tokenizer detect beginning of words by the preceding space).`,name:"add_prefix_space"},{anchor:"transformers.LayoutLMv3Tokenizer.cls_token_box",description:`<strong>cls_token_box</strong> (<code>List[int]</code>, <em>optional</em>, defaults to <code>[0, 0, 0, 0]</code>) &#x2014;
The bounding box to use for the special [CLS] token.`,name:"cls_token_box"},{anchor:"transformers.LayoutLMv3Tokenizer.sep_token_box",description:`<strong>sep_token_box</strong> (<code>List[int]</code>, <em>optional</em>, defaults to <code>[0, 0, 0, 0]</code>) &#x2014;
The bounding box to use for the special [SEP] token.`,name:"sep_token_box"},{anchor:"transformers.LayoutLMv3Tokenizer.pad_token_box",description:`<strong>pad_token_box</strong> (<code>List[int]</code>, <em>optional</em>, defaults to <code>[0, 0, 0, 0]</code>) &#x2014;
The bounding box to use for the special [PAD] token.`,name:"pad_token_box"},{anchor:"transformers.LayoutLMv3Tokenizer.pad_token_label",description:`<strong>pad_token_label</strong> (<code>int</code>, <em>optional</em>, defaults to -100) &#x2014;
The label to use for padding tokens. Defaults to -100, which is the <code>ignore_index</code> of PyTorch&#x2019;s
CrossEntropyLoss.`,name:"pad_token_label"},{anchor:"transformers.LayoutLMv3Tokenizer.only_label_first_subword",description:`<strong>only_label_first_subword</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to only label the first subword, in case word labels are provided.`,name:"only_label_first_subword"}],source:"https://github.com/huggingface/transformers/blob/vr_18400/src/transformers/models/layoutlmv3/tokenization_layoutlmv3.py#L200"}}),$t=new P({props:{name:"__call__",anchor:"transformers.LayoutLMv3Tokenizer.__call__",parameters:[{name:"text",val:": typing.Union[str, typing.List[str], typing.List[typing.List[str]]]"},{name:"text_pair",val:": typing.Union[typing.List[str], typing.List[typing.List[str]], NoneType] = None"},{name:"boxes",val:": typing.Union[typing.List[typing.List[int]], typing.List[typing.List[typing.List[int]]]] = None"},{name:"word_labels",val:": typing.Union[typing.List[int], typing.List[typing.List[int]], NoneType] = None"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy] = False"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"return_token_type_ids",val:": typing.Optional[bool] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_overflowing_tokens",val:": bool = False"},{name:"return_special_tokens_mask",val:": bool = False"},{name:"return_offsets_mapping",val:": bool = False"},{name:"return_length",val:": bool = False"},{name:"verbose",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.LayoutLMv3Tokenizer.__call__.text",description:`<strong>text</strong> (<code>str</code>, <code>List[str]</code>, <code>List[List[str]]</code>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence can be a string, a list of strings
(words of a single example or questions of a batch of examples) or a list of list of strings (batch of
words).`,name:"text"},{anchor:"transformers.LayoutLMv3Tokenizer.__call__.text_pair",description:`<strong>text_pair</strong> (<code>List[str]</code>, <code>List[List[str]]</code>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence should be a list of strings
(pretokenized string).`,name:"text_pair"},{anchor:"transformers.LayoutLMv3Tokenizer.__call__.boxes",description:`<strong>boxes</strong> (<code>List[List[int]]</code>, <code>List[List[List[int]]]</code>) &#x2014;
Word-level bounding boxes. Each bounding box should be normalized to be on a 0-1000 scale.`,name:"boxes"},{anchor:"transformers.LayoutLMv3Tokenizer.__call__.word_labels",description:`<strong>word_labels</strong> (<code>List[int]</code>, <code>List[List[int]]</code>, <em>optional</em>) &#x2014;
Word-level integer labels (for token classification tasks such as FUNSD, CORD).`,name:"word_labels"},{anchor:"transformers.LayoutLMv3Tokenizer.__call__.add_special_tokens",description:`<strong>add_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to encode the sequences with the special tokens relative to their model.`,name:"add_special_tokens"},{anchor:"transformers.LayoutLMv3Tokenizer.__call__.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/pr_18400/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls padding. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence if provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.LayoutLMv3Tokenizer.__call__.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/pr_18400/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls truncation. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or
to the maximum acceptable input length for the model if that argument is not provided. This will
truncate token by token, removing a token from the longest sequence in the pair if a pair of
sequences (or a batch of pairs) is provided.</li>
<li><code>&apos;only_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</li>
<li><code>&apos;only_second&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</li>
<li><code>False</code> or <code>&apos;do_not_truncate&apos;</code> (default): No truncation (i.e., can output batch with sequence lengths
greater than the model maximum admissible input size).</li>
</ul>`,name:"truncation"},{anchor:"transformers.LayoutLMv3Tokenizer.__call__.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length to use by one of the truncation/padding parameters.</p>
<p>If left unset or set to <code>None</code>, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.LayoutLMv3Tokenizer.__call__.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a number along with <code>max_length</code>, the overflowing tokens returned when
<code>return_overflowing_tokens=True</code> will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.`,name:"stride"},{anchor:"transformers.LayoutLMv3Tokenizer.__call__.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
the use of Tensor Cores on NVIDIA hardware with compute capability &gt;= 7.5 (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.LayoutLMv3Tokenizer.__call__.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/pr_18400/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.LayoutLMv3Tokenizer.__call__.add_special_tokens",description:`<strong>add_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to encode the sequences with the special tokens relative to their model.`,name:"add_special_tokens"},{anchor:"transformers.LayoutLMv3Tokenizer.__call__.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/pr_18400/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls padding. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence if provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.LayoutLMv3Tokenizer.__call__.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/pr_18400/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls truncation. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or
to the maximum acceptable input length for the model if that argument is not provided. This will
truncate token by token, removing a token from the longest sequence in the pair if a pair of
sequences (or a batch of pairs) is provided.</li>
<li><code>&apos;only_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</li>
<li><code>&apos;only_second&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</li>
<li><code>False</code> or <code>&apos;do_not_truncate&apos;</code> (default): No truncation (i.e., can output batch with sequence lengths
greater than the model maximum admissible input size).</li>
</ul>`,name:"truncation"},{anchor:"transformers.LayoutLMv3Tokenizer.__call__.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length to use by one of the truncation/padding parameters. If left unset or set to
<code>None</code>, this will use the predefined model maximum length if a maximum length is required by one of the
truncation/padding parameters. If the model has no specific maximum input length (like XLNet)
truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.LayoutLMv3Tokenizer.__call__.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a number along with <code>max_length</code>, the overflowing tokens returned when
<code>return_overflowing_tokens=True</code> will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.`,name:"stride"},{anchor:"transformers.LayoutLMv3Tokenizer.__call__.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
the use of Tensor Cores on NVIDIA hardware with compute capability &gt;= 7.5 (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.LayoutLMv3Tokenizer.__call__.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/pr_18400/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"}],source:"https://github.com/huggingface/transformers/blob/vr_18400/src/transformers/models/layoutlmv3/tokenization_layoutlmv3.py#L547"}}),zt=new P({props:{name:"save_vocabulary",anchor:"transformers.LayoutLMv3Tokenizer.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/vr_18400/src/transformers/models/layoutlmv3/tokenization_layoutlmv3.py#L428"}}),qt=new te({}),Et=new P({props:{name:"class transformers.LayoutLMv3TokenizerFast",anchor:"transformers.LayoutLMv3TokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"merges_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"errors",val:" = 'replace'"},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"sep_token",val:" = '</s>'"},{name:"cls_token",val:" = '<s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"mask_token",val:" = '<mask>'"},{name:"add_prefix_space",val:" = True"},{name:"trim_offsets",val:" = True"},{name:"cls_token_box",val:" = [0, 0, 0, 0]"},{name:"sep_token_box",val:" = [0, 0, 0, 0]"},{name:"pad_token_box",val:" = [0, 0, 0, 0]"},{name:"pad_token_label",val:" = -100"},{name:"only_label_first_subword",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.LayoutLMv3TokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.LayoutLMv3TokenizerFast.merges_file",description:`<strong>merges_file</strong> (<code>str</code>) &#x2014;
Path to the merges file.`,name:"merges_file"},{anchor:"transformers.LayoutLMv3TokenizerFast.errors",description:`<strong>errors</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;replace&quot;</code>) &#x2014;
Paradigm to follow when decoding bytes to UTF-8. See
<a href="https://docs.python.org/3/library/stdtypes.html#bytes.decode" rel="nofollow">bytes.decode</a> for more information.`,name:"errors"},{anchor:"transformers.LayoutLMv3TokenizerFast.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the beginning of
sequence. The token used is the <code>cls_token</code>.</p>

					</div>`,name:"bos_token"},{anchor:"transformers.LayoutLMv3TokenizerFast.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the end of sequence.
The token used is the <code>sep_token</code>.</p>

					</div>`,name:"eos_token"},{anchor:"transformers.LayoutLMv3TokenizerFast.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.LayoutLMv3TokenizerFast.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.LayoutLMv3TokenizerFast.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.LayoutLMv3TokenizerFast.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.LayoutLMv3TokenizerFast.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;mask&gt;&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.LayoutLMv3TokenizerFast.add_prefix_space",description:`<strong>add_prefix_space</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to add an initial space to the input. This allows to treat the leading word just as any
other word. (RoBERTa tokenizer detect beginning of words by the preceding space).`,name:"add_prefix_space"},{anchor:"transformers.LayoutLMv3TokenizerFast.trim_offsets",description:`<strong>trim_offsets</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether the post processing step should trim offsets to avoid including whitespaces.`,name:"trim_offsets"},{anchor:"transformers.LayoutLMv3TokenizerFast.cls_token_box",description:`<strong>cls_token_box</strong> (<code>List[int]</code>, <em>optional</em>, defaults to <code>[0, 0, 0, 0]</code>) &#x2014;
The bounding box to use for the special [CLS] token.`,name:"cls_token_box"},{anchor:"transformers.LayoutLMv3TokenizerFast.sep_token_box",description:`<strong>sep_token_box</strong> (<code>List[int]</code>, <em>optional</em>, defaults to <code>[0, 0, 0, 0]</code>) &#x2014;
The bounding box to use for the special [SEP] token.`,name:"sep_token_box"},{anchor:"transformers.LayoutLMv3TokenizerFast.pad_token_box",description:`<strong>pad_token_box</strong> (<code>List[int]</code>, <em>optional</em>, defaults to <code>[0, 0, 0, 0]</code>) &#x2014;
The bounding box to use for the special [PAD] token.`,name:"pad_token_box"},{anchor:"transformers.LayoutLMv3TokenizerFast.pad_token_label",description:`<strong>pad_token_label</strong> (<code>int</code>, <em>optional</em>, defaults to -100) &#x2014;
The label to use for padding tokens. Defaults to -100, which is the <code>ignore_index</code> of PyTorch&#x2019;s
CrossEntropyLoss.`,name:"pad_token_label"},{anchor:"transformers.LayoutLMv3TokenizerFast.only_label_first_subword",description:`<strong>only_label_first_subword</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to only label the first subword, in case word labels are provided.`,name:"only_label_first_subword"}],source:"https://github.com/huggingface/transformers/blob/vr_18400/src/transformers/models/layoutlmv3/tokenization_layoutlmv3_fast.py#L65"}}),Ct=new P({props:{name:"__call__",anchor:"transformers.LayoutLMv3TokenizerFast.__call__",parameters:[{name:"text",val:": typing.Union[str, typing.List[str], typing.List[typing.List[str]]]"},{name:"text_pair",val:": typing.Union[typing.List[str], typing.List[typing.List[str]], NoneType] = None"},{name:"boxes",val:": typing.Union[typing.List[typing.List[int]], typing.List[typing.List[typing.List[int]]]] = None"},{name:"word_labels",val:": typing.Union[typing.List[int], typing.List[typing.List[int]], NoneType] = None"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy] = False"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"return_token_type_ids",val:": typing.Optional[bool] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_overflowing_tokens",val:": bool = False"},{name:"return_special_tokens_mask",val:": bool = False"},{name:"return_offsets_mapping",val:": bool = False"},{name:"return_length",val:": bool = False"},{name:"verbose",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.LayoutLMv3TokenizerFast.__call__.text",description:`<strong>text</strong> (<code>str</code>, <code>List[str]</code>, <code>List[List[str]]</code>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence can be a string, a list of strings
(words of a single example or questions of a batch of examples) or a list of list of strings (batch of
words).`,name:"text"},{anchor:"transformers.LayoutLMv3TokenizerFast.__call__.text_pair",description:`<strong>text_pair</strong> (<code>List[str]</code>, <code>List[List[str]]</code>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence should be a list of strings
(pretokenized string).`,name:"text_pair"},{anchor:"transformers.LayoutLMv3TokenizerFast.__call__.boxes",description:`<strong>boxes</strong> (<code>List[List[int]]</code>, <code>List[List[List[int]]]</code>) &#x2014;
Word-level bounding boxes. Each bounding box should be normalized to be on a 0-1000 scale.`,name:"boxes"},{anchor:"transformers.LayoutLMv3TokenizerFast.__call__.word_labels",description:`<strong>word_labels</strong> (<code>List[int]</code>, <code>List[List[int]]</code>, <em>optional</em>) &#x2014;
Word-level integer labels (for token classification tasks such as FUNSD, CORD).`,name:"word_labels"},{anchor:"transformers.LayoutLMv3TokenizerFast.__call__.add_special_tokens",description:`<strong>add_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to encode the sequences with the special tokens relative to their model.`,name:"add_special_tokens"},{anchor:"transformers.LayoutLMv3TokenizerFast.__call__.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/pr_18400/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls padding. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence if provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.LayoutLMv3TokenizerFast.__call__.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/pr_18400/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls truncation. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or
to the maximum acceptable input length for the model if that argument is not provided. This will
truncate token by token, removing a token from the longest sequence in the pair if a pair of
sequences (or a batch of pairs) is provided.</li>
<li><code>&apos;only_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</li>
<li><code>&apos;only_second&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</li>
<li><code>False</code> or <code>&apos;do_not_truncate&apos;</code> (default): No truncation (i.e., can output batch with sequence lengths
greater than the model maximum admissible input size).</li>
</ul>`,name:"truncation"},{anchor:"transformers.LayoutLMv3TokenizerFast.__call__.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length to use by one of the truncation/padding parameters.</p>
<p>If left unset or set to <code>None</code>, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.LayoutLMv3TokenizerFast.__call__.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a number along with <code>max_length</code>, the overflowing tokens returned when
<code>return_overflowing_tokens=True</code> will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.`,name:"stride"},{anchor:"transformers.LayoutLMv3TokenizerFast.__call__.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
the use of Tensor Cores on NVIDIA hardware with compute capability &gt;= 7.5 (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.LayoutLMv3TokenizerFast.__call__.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/pr_18400/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.LayoutLMv3TokenizerFast.__call__.add_special_tokens",description:`<strong>add_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to encode the sequences with the special tokens relative to their model.`,name:"add_special_tokens"},{anchor:"transformers.LayoutLMv3TokenizerFast.__call__.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/pr_18400/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls padding. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence if provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.LayoutLMv3TokenizerFast.__call__.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/pr_18400/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls truncation. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or
to the maximum acceptable input length for the model if that argument is not provided. This will
truncate token by token, removing a token from the longest sequence in the pair if a pair of
sequences (or a batch of pairs) is provided.</li>
<li><code>&apos;only_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</li>
<li><code>&apos;only_second&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</li>
<li><code>False</code> or <code>&apos;do_not_truncate&apos;</code> (default): No truncation (i.e., can output batch with sequence lengths
greater than the model maximum admissible input size).</li>
</ul>`,name:"truncation"},{anchor:"transformers.LayoutLMv3TokenizerFast.__call__.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length to use by one of the truncation/padding parameters. If left unset or set to
<code>None</code>, this will use the predefined model maximum length if a maximum length is required by one of the
truncation/padding parameters. If the model has no specific maximum input length (like XLNet)
truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.LayoutLMv3TokenizerFast.__call__.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a number along with <code>max_length</code>, the overflowing tokens returned when
<code>return_overflowing_tokens=True</code> will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.`,name:"stride"},{anchor:"transformers.LayoutLMv3TokenizerFast.__call__.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
the use of Tensor Cores on NVIDIA hardware with compute capability &gt;= 7.5 (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.LayoutLMv3TokenizerFast.__call__.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/pr_18400/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"}],source:"https://github.com/huggingface/transformers/blob/vr_18400/src/transformers/models/layoutlmv3/tokenization_layoutlmv3_fast.py#L224"}}),At=new te({}),jt=new P({props:{name:"class transformers.LayoutLMv3Processor",anchor:"transformers.LayoutLMv3Processor",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.LayoutLMv3Processor.feature_extractor",description:`<strong>feature_extractor</strong> (<code>LayoutLMv3FeatureExtractor</code>) &#x2014;
An instance of <a href="/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3FeatureExtractor">LayoutLMv3FeatureExtractor</a>. The feature extractor is a required input.`,name:"feature_extractor"},{anchor:"transformers.LayoutLMv3Processor.tokenizer",description:`<strong>tokenizer</strong> (<code>LayoutLMv3Tokenizer</code> or <code>LayoutLMv3TokenizerFast</code>) &#x2014;
An instance of <a href="/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3Tokenizer">LayoutLMv3Tokenizer</a> or <a href="/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3TokenizerFast">LayoutLMv3TokenizerFast</a>. The tokenizer is a required input.`,name:"tokenizer"}],source:"https://github.com/huggingface/transformers/blob/vr_18400/src/transformers/models/layoutlmv3/processing_layoutlmv3.py#L25"}}),It=new P({props:{name:"__call__",anchor:"transformers.LayoutLMv3Processor.__call__",parameters:[{name:"images",val:""},{name:"text",val:": typing.Union[str, typing.List[str], typing.List[typing.List[str]]] = None"},{name:"text_pair",val:": typing.Union[typing.List[str], typing.List[typing.List[str]], NoneType] = None"},{name:"boxes",val:": typing.Union[typing.List[typing.List[int]], typing.List[typing.List[typing.List[int]]]] = None"},{name:"word_labels",val:": typing.Union[typing.List[int], typing.List[typing.List[int]], NoneType] = None"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy] = False"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"return_token_type_ids",val:": typing.Optional[bool] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_overflowing_tokens",val:": bool = False"},{name:"return_special_tokens_mask",val:": bool = False"},{name:"return_offsets_mapping",val:": bool = False"},{name:"return_length",val:": bool = False"},{name:"verbose",val:": bool = True"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18400/src/transformers/models/layoutlmv3/processing_layoutlmv3.py#L47"}}),Ot=new te({}),Rt=new P({props:{name:"class transformers.LayoutLMv3Model",anchor:"transformers.LayoutLMv3Model",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.LayoutLMv3Model.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18400/en/model_doc/layoutlmv2#transformers.LayoutLMv2Config">LayoutLMv2Config</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18400/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18400/src/transformers/models/layoutlmv3/modeling_layoutlmv3.py#L675"}}),Ut=new P({props:{name:"forward",anchor:"transformers.LayoutLMv3Model.forward",parameters:[{name:"input_ids",val:" = None"},{name:"bbox",val:" = None"},{name:"attention_mask",val:" = None"},{name:"token_type_ids",val:" = None"},{name:"position_ids",val:" = None"},{name:"head_mask",val:" = None"},{name:"inputs_embeds",val:" = None"},{name:"pixel_values",val:" = None"},{name:"output_attentions",val:" = None"},{name:"output_hidden_states",val:" = None"},{name:"return_dict",val:" = None"}],parametersDescription:[{anchor:"transformers.LayoutLMv3Model.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18400/en/model_doc/layoutlmv2#transformers.LayoutLMv2Tokenizer">LayoutLMv2Tokenizer</a>. See <a href="/docs/transformers/pr_18400/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18400/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LayoutLMv3Model.forward.bbox",description:`<strong>bbox</strong> (<code>torch.LongTensor</code> of shape <code>((batch_size, sequence_length), 4)</code>, <em>optional</em>) &#x2014;
Bounding boxes of each input sequence tokens. Selected in the range <code>[0, config.max_2d_position_embeddings-1]</code>. Each bounding box should be a normalized version in (x0, y0, x1, y1)
format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
y1) represents the position of the lower right corner.`,name:"bbox"},{anchor:"transformers.LayoutLMv3Model.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, height, width)</code>) &#x2014;
Batch of document images.`,name:"pixel_values"},{anchor:"transformers.LayoutLMv3Model.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LayoutLMv3Model.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.LayoutLMv3Model.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LayoutLMv3Model.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.LayoutLMv3Model.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <em>input_ids</em> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LayoutLMv3Model.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.LayoutLMv3Model.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.LayoutLMv3Model.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18400/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18400/src/transformers/models/layoutlmv3/modeling_layoutlmv3.py#L766",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18400/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput"
>transformers.modeling_outputs.BaseModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3Config"
>LayoutLMv3Config</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18400/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput"
>transformers.modeling_outputs.BaseModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Se=new xs({props:{$$slots:{default:[Sp]},$$scope:{ctx:q}}}),Oe=new rn({props:{anchor:"transformers.LayoutLMv3Model.forward.example",$$slots:{default:[Op]},$$scope:{ctx:q}}}),Qt=new te({}),Vt=new P({props:{name:"class transformers.LayoutLMv3ForSequenceClassification",anchor:"transformers.LayoutLMv3ForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.LayoutLMv3ForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18400/en/model_doc/layoutlmv2#transformers.LayoutLMv2Config">LayoutLMv2Config</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18400/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18400/src/transformers/models/layoutlmv3/modeling_layoutlmv3.py#L1205"}}),Jt=new P({props:{name:"forward",anchor:"transformers.LayoutLMv3ForSequenceClassification.forward",parameters:[{name:"input_ids",val:" = None"},{name:"attention_mask",val:" = None"},{name:"token_type_ids",val:" = None"},{name:"position_ids",val:" = None"},{name:"head_mask",val:" = None"},{name:"inputs_embeds",val:" = None"},{name:"labels",val:" = None"},{name:"output_attentions",val:" = None"},{name:"output_hidden_states",val:" = None"},{name:"return_dict",val:" = None"},{name:"bbox",val:" = None"},{name:"pixel_values",val:" = None"}],parametersDescription:[{anchor:"transformers.LayoutLMv3ForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>batch_size, sequence_length</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18400/en/model_doc/layoutlmv2#transformers.LayoutLMv2Tokenizer">LayoutLMv2Tokenizer</a>. See <a href="/docs/transformers/pr_18400/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18400/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LayoutLMv3ForSequenceClassification.forward.bbox",description:`<strong>bbox</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length, 4)</code>, <em>optional</em>) &#x2014;
Bounding boxes of each input sequence tokens. Selected in the range <code>[0, config.max_2d_position_embeddings-1]</code>. Each bounding box should be a normalized version in (x0, y0, x1, y1)
format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
y1) represents the position of the lower right corner.`,name:"bbox"},{anchor:"transformers.LayoutLMv3ForSequenceClassification.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, height, width)</code>) &#x2014;
Batch of document images.`,name:"pixel_values"},{anchor:"transformers.LayoutLMv3ForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>batch_size, sequence_length</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LayoutLMv3ForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>batch_size, sequence_length</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.LayoutLMv3ForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>batch_size, sequence_length</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LayoutLMv3ForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.LayoutLMv3ForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <em>input_ids</em> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LayoutLMv3ForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.LayoutLMv3ForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.LayoutLMv3ForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18400/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18400/src/transformers/models/layoutlmv3/modeling_layoutlmv3.py#L1217",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18400/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3Config"
>LayoutLMv3Config</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18400/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),We=new xs({props:{$$slots:{default:[Rp]},$$scope:{ctx:q}}}),Be=new rn({props:{anchor:"transformers.LayoutLMv3ForSequenceClassification.forward.example",$$slots:{default:[Wp]},$$scope:{ctx:q}}}),Kt=new te({}),Zt=new P({props:{name:"class transformers.LayoutLMv3ForTokenClassification",anchor:"transformers.LayoutLMv3ForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.LayoutLMv3ForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18400/en/model_doc/layoutlmv2#transformers.LayoutLMv2Config">LayoutLMv2Config</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18400/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18400/src/transformers/models/layoutlmv3/modeling_layoutlmv3.py#L961"}}),ro=new P({props:{name:"forward",anchor:"transformers.LayoutLMv3ForTokenClassification.forward",parameters:[{name:"input_ids",val:" = None"},{name:"bbox",val:" = None"},{name:"attention_mask",val:" = None"},{name:"token_type_ids",val:" = None"},{name:"position_ids",val:" = None"},{name:"head_mask",val:" = None"},{name:"inputs_embeds",val:" = None"},{name:"labels",val:" = None"},{name:"output_attentions",val:" = None"},{name:"output_hidden_states",val:" = None"},{name:"return_dict",val:" = None"},{name:"pixel_values",val:" = None"}],parametersDescription:[{anchor:"transformers.LayoutLMv3ForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>batch_size, sequence_length</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18400/en/model_doc/layoutlmv2#transformers.LayoutLMv2Tokenizer">LayoutLMv2Tokenizer</a>. See <a href="/docs/transformers/pr_18400/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18400/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LayoutLMv3ForTokenClassification.forward.bbox",description:`<strong>bbox</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length, 4)</code>, <em>optional</em>) &#x2014;
Bounding boxes of each input sequence tokens. Selected in the range <code>[0, config.max_2d_position_embeddings-1]</code>. Each bounding box should be a normalized version in (x0, y0, x1, y1)
format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
y1) represents the position of the lower right corner.`,name:"bbox"},{anchor:"transformers.LayoutLMv3ForTokenClassification.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, height, width)</code>) &#x2014;
Batch of document images.`,name:"pixel_values"},{anchor:"transformers.LayoutLMv3ForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>batch_size, sequence_length</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LayoutLMv3ForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>batch_size, sequence_length</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.LayoutLMv3ForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>batch_size, sequence_length</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LayoutLMv3ForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.LayoutLMv3ForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <em>input_ids</em> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LayoutLMv3ForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.LayoutLMv3ForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.LayoutLMv3ForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18400/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.LayoutLMv3ForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18400/src/transformers/models/layoutlmv3/modeling_layoutlmv3.py#L978",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18400/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3Config"
>LayoutLMv3Config</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18400/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Qe=new xs({props:{$$slots:{default:[Bp]},$$scope:{ctx:q}}}),Ve=new rn({props:{anchor:"transformers.LayoutLMv3ForTokenClassification.forward.example",$$slots:{default:[Up]},$$scope:{ctx:q}}}),io=new te({}),lo=new P({props:{name:"class transformers.LayoutLMv3ForQuestionAnswering",anchor:"transformers.LayoutLMv3ForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.LayoutLMv3ForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18400/en/model_doc/layoutlmv2#transformers.LayoutLMv2Config">LayoutLMv2Config</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18400/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18400/src/transformers/models/layoutlmv3/modeling_layoutlmv3.py#L1074"}}),mo=new P({props:{name:"forward",anchor:"transformers.LayoutLMv3ForQuestionAnswering.forward",parameters:[{name:"input_ids",val:" = None"},{name:"attention_mask",val:" = None"},{name:"token_type_ids",val:" = None"},{name:"position_ids",val:" = None"},{name:"head_mask",val:" = None"},{name:"inputs_embeds",val:" = None"},{name:"start_positions",val:" = None"},{name:"end_positions",val:" = None"},{name:"output_attentions",val:" = None"},{name:"output_hidden_states",val:" = None"},{name:"return_dict",val:" = None"},{name:"bbox",val:" = None"},{name:"pixel_values",val:" = None"}],parametersDescription:[{anchor:"transformers.LayoutLMv3ForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>batch_size, sequence_length</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18400/en/model_doc/layoutlmv2#transformers.LayoutLMv2Tokenizer">LayoutLMv2Tokenizer</a>. See <a href="/docs/transformers/pr_18400/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18400/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LayoutLMv3ForQuestionAnswering.forward.bbox",description:`<strong>bbox</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length, 4)</code>, <em>optional</em>) &#x2014;
Bounding boxes of each input sequence tokens. Selected in the range <code>[0, config.max_2d_position_embeddings-1]</code>. Each bounding box should be a normalized version in (x0, y0, x1, y1)
format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
y1) represents the position of the lower right corner.`,name:"bbox"},{anchor:"transformers.LayoutLMv3ForQuestionAnswering.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, height, width)</code>) &#x2014;
Batch of document images.`,name:"pixel_values"},{anchor:"transformers.LayoutLMv3ForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>batch_size, sequence_length</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LayoutLMv3ForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>batch_size, sequence_length</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.LayoutLMv3ForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>batch_size, sequence_length</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LayoutLMv3ForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.LayoutLMv3ForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <em>input_ids</em> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LayoutLMv3ForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.LayoutLMv3ForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.LayoutLMv3ForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18400/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.LayoutLMv3ForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.LayoutLMv3ForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"}],source:"https://github.com/huggingface/transformers/blob/vr_18400/src/transformers/models/layoutlmv3/modeling_layoutlmv3.py#L1087",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18400/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3Config"
>LayoutLMv3Config</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18400/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Ge=new xs({props:{$$slots:{default:[Qp]},$$scope:{ctx:q}}}),Xe=new rn({props:{anchor:"transformers.LayoutLMv3ForQuestionAnswering.forward.example",$$slots:{default:[Vp]},$$scope:{ctx:q}}}),{c(){p=s("meta"),T=c(),_=s("h1"),f=s("a"),v=s("span"),y(d.$$.fragment),h=c(),C=s("span"),ws=n("LayoutLMv3"),Ta=c(),le=s("h2"),Te=s("a"),cn=s("span"),y(lt.$$.fragment),Ts=c(),un=s("span"),Ms=n("Overview"),Ma=c(),Q=s("p"),$s=n("The LayoutLMv3 model was proposed in "),dt=s("a"),zs=n("LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking"),qs=n(` by Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, Furu Wei.
LayoutLMv3 simplifies `),go=s("a"),Es=n("LayoutLMv2"),Fs=n(" by using patch embeddings (as in "),_o=s("a"),Ps=n("ViT"),Cs=n(`) instead of leveraging a CNN backbone, and pre-trains the model on 3 objectives: masked language modeling (MLM), masked image modeling (MIM)
and word-patch alignment (WPA).`),$a=c(),vo=s("p"),As=n("The abstract from the paper is the following:"),za=c(),yo=s("p"),pn=s("em"),js=n("Self-supervised pre-training techniques have achieved remarkable progress in Document AI. Most multimodal pre-trained models use a masked language modeling objective to learn bidirectional representations on the text modality, but they differ in pre-training objectives for the image modality. This discrepancy adds difficulty to multimodal representation learning. In this paper, we propose LayoutLMv3 to pre-train multimodal Transformers for Document AI with unified text and image masking. Additionally, LayoutLMv3 is pre-trained with a word-patch alignment objective to learn cross-modal alignment by predicting whether the corresponding image patch of a text word is masked. The simple unified architecture and training objectives make LayoutLMv3 a general-purpose pre-trained model for both text-centric and image-centric Document AI tasks. Experimental results show that LayoutLMv3 achieves state-of-the-art performance not only in text-centric tasks, including form understanding, receipt understanding, and document visual question answering, but also in image-centric tasks such as document image classification and document layout analysis."),qa=c(),bo=s("p"),Is=n("Tips:"),Ea=c(),V=s("ul"),Me=s("li"),Ns=n("In terms of data processing, LayoutLMv3 is identical to its predecessor "),Lo=s("a"),Ds=n("LayoutLMv2"),Ss=n(", except that:"),ct=s("ul"),mn=s("li"),Os=n("images need to be resized and normalized with channels in regular RGB format. LayoutLMv2 on the other hand normalizes the images internally and expects the channels in BGR format."),Rs=c(),N=s("li"),Ws=n(`text is tokenized using byte-pair encoding (BPE), as opposed to WordPiece.
Due to these differences in data preprocessing, one can use `),ko=s("a"),Bs=n("LayoutLMv3Processor"),Us=n(" which internally combines a "),xo=s("a"),Qs=n("LayoutLMv3FeatureExtractor"),Vs=n(" (for the image modality) and a "),wo=s("a"),Hs=n("LayoutLMv3Tokenizer"),Gs=n("/"),To=s("a"),Xs=n("LayoutLMv3TokenizerFast"),Ys=n(" (for the text modality) to prepare all data for the model."),Js=c(),de=s("li"),Ks=n("Regarding usage of "),Mo=s("a"),Zs=n("LayoutLMv3Processor"),er=n(", we refer to the "),$o=s("a"),tr=n("usage guide"),or=n(" of its predecessor."),nr=c(),ut=s("li"),ar=n("Demo notebooks for LayoutLMv3 can be found "),pt=s("a"),sr=n("here"),rr=n("."),ir=c(),mt=s("li"),lr=n("Demo scripts can be found "),ht=s("a"),dr=n("here"),cr=n("."),Fa=c(),$e=s("img"),Pa=c(),ze=s("small"),ur=n("LayoutLMv3 architecture. Taken from the "),zo=s("a"),pr=n("original paper"),mr=n("."),Ca=c(),oe=s("p"),hr=n("This model was contributed by "),ft=s("a"),fr=n("nielsr"),gr=n(". The original code can be found "),gt=s("a"),_r=n("here"),vr=n("."),Aa=c(),ce=s("h2"),qe=s("a"),hn=s("span"),y(_t.$$.fragment),yr=c(),fn=s("span"),br=n("LayoutLMv3Config"),ja=c(),D=s("div"),y(vt.$$.fragment),Lr=c(),ue=s("p"),kr=n("This is the configuration class to store the configuration of a "),qo=s("a"),xr=n("LayoutLMv3Model"),wr=n(`. It is used to instantiate an
LayoutLMv3 model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the LayoutLMv3
`),yt=s("a"),Tr=n("microsoft/layoutlmv3-base"),Mr=n(" architecture."),$r=c(),pe=s("p"),zr=n("Configuration objects inherit from "),Eo=s("a"),qr=n("PretrainedConfig"),Er=n(` and can be used to control the model outputs. Read the
documentation from `),Fo=s("a"),Fr=n("PretrainedConfig"),Pr=n(" for more information."),Cr=c(),y(Ee.$$.fragment),Ia=c(),me=s("h2"),Fe=s("a"),gn=s("span"),y(bt.$$.fragment),Ar=c(),_n=s("span"),jr=n("LayoutLMv3FeatureExtractor"),Na=c(),S=s("div"),y(Lt.$$.fragment),Ir=c(),vn=s("p"),Nr=n(`Constructs a LayoutLMv3 feature extractor. This can be used to resize + normalize document images, as well as to
apply OCR on them in order to get a list of words and normalized bounding boxes.`),Dr=c(),kt=s("p"),Sr=n("This feature extractor inherits from "),yn=s("code"),Or=n("PreTrainedFeatureExtractor()"),Rr=n(` which contains most
of the main methods. Users should refer to this superclass for more information regarding those methods.`),Wr=c(),ne=s("div"),y(xt.$$.fragment),Br=c(),bn=s("p"),Ur=n("Main method to prepare for the model one or several image(s)."),Qr=c(),y(Pe.$$.fragment),Da=c(),he=s("h2"),Ce=s("a"),Ln=s("span"),y(wt.$$.fragment),Vr=c(),kn=s("span"),Hr=n("LayoutLMv3Tokenizer"),Sa=c(),A=s("div"),y(Tt.$$.fragment),Gr=c(),F=s("p"),Xr=n("Construct a LayoutLMv3 tokenizer. Based on "),xn=s("code"),Yr=n("RoBERTatokenizer"),Jr=n(` (Byte Pair Encoding or BPE).
`),Po=s("a"),Kr=n("LayoutLMv3Tokenizer"),Zr=n(` can be used to turn words, word-level bounding boxes and optional word labels to
token-level `),wn=s("code"),ei=n("input_ids"),ti=n(", "),Tn=s("code"),oi=n("attention_mask"),ni=n(", "),Mn=s("code"),ai=n("token_type_ids"),si=n(", "),$n=s("code"),ri=n("bbox"),ii=n(", and optional "),zn=s("code"),li=n("labels"),di=n(` (for token
classification).`),ci=c(),Mt=s("p"),ui=n("This tokenizer inherits from "),Co=s("a"),pi=n("PreTrainedTokenizer"),mi=n(` which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`),hi=c(),Ao=s("p"),jo=s("a"),fi=n("LayoutLMv3Tokenizer"),gi=n(` runs end-to-end tokenization: punctuation splitting and wordpiece. It also turns the
word-level bounding boxes into token-level bounding boxes.`),_i=c(),Ae=s("div"),y($t.$$.fragment),vi=c(),qn=s("p"),yi=n(`Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences with word-level normalized bounding boxes and optional labels.`),bi=c(),Io=s("div"),y(zt.$$.fragment),Oa=c(),fe=s("h2"),je=s("a"),En=s("span"),y(qt.$$.fragment),Li=c(),Fn=s("span"),ki=n("LayoutLMv3TokenizerFast"),Ra=c(),O=s("div"),y(Et.$$.fragment),xi=c(),Ft=s("p"),wi=n("Construct a \u201Cfast\u201D LayoutLMv3 tokenizer (backed by HuggingFace\u2019s "),Pn=s("em"),Ti=n("tokenizers"),Mi=n(" library). Based on BPE."),$i=c(),Pt=s("p"),zi=n("This tokenizer inherits from "),No=s("a"),qi=n("PreTrainedTokenizerFast"),Ei=n(` which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`),Fi=c(),Ie=s("div"),y(Ct.$$.fragment),Pi=c(),Cn=s("p"),Ci=n(`Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences with word-level normalized bounding boxes and optional labels.`),Wa=c(),ge=s("h2"),Ne=s("a"),An=s("span"),y(At.$$.fragment),Ai=c(),jn=s("span"),ji=n("LayoutLMv3Processor"),Ba=c(),I=s("div"),y(jt.$$.fragment),Ii=c(),In=s("p"),Ni=n(`Constructs a LayoutLMv3 processor which combines a LayoutLMv3 feature extractor and a LayoutLMv3 tokenizer into a
single processor.`),Di=c(),Do=s("p"),So=s("a"),Si=n("LayoutLMv3Processor"),Oi=n(" offers all the functionalities you need to prepare data for the model."),Ri=c(),z=s("p"),Wi=n("It first uses "),Oo=s("a"),Bi=n("LayoutLMv3FeatureExtractor"),Ui=n(` to resize and normalize document images, and optionally applies OCR to
get words and normalized bounding boxes. These are then provided to `),Ro=s("a"),Qi=n("LayoutLMv3Tokenizer"),Vi=n(` or
`),Wo=s("a"),Hi=n("LayoutLMv3TokenizerFast"),Gi=n(", which turns the words and bounding boxes into token-level "),Nn=s("code"),Xi=n("input_ids"),Yi=n(`,
`),Dn=s("code"),Ji=n("attention_mask"),Ki=n(", "),Sn=s("code"),Zi=n("token_type_ids"),el=n(", "),On=s("code"),tl=n("bbox"),ol=n(". Optionally, one can provide integer "),Rn=s("code"),nl=n("word_labels"),al=n(`, which are turned
into token-level `),Wn=s("code"),sl=n("labels"),rl=n(" for token classification tasks (such as FUNSD, CORD)."),il=c(),ae=s("div"),y(It.$$.fragment),ll=c(),M=s("p"),dl=n("This method first forwards the "),Bn=s("code"),cl=n("images"),ul=n(" argument to "),Nt=s("a"),Un=s("strong"),pl=n("call"),ml=n("()"),hl=n(`. In case
`),Bo=s("a"),fl=n("LayoutLMv3FeatureExtractor"),gl=n(" was initialized with "),Qn=s("code"),_l=n("apply_ocr"),vl=n(" set to "),Vn=s("code"),yl=n("True"),bl=n(`, it passes the obtained words and
bounding boxes along with the additional arguments to `),Dt=s("a"),Hn=s("strong"),Ll=n("call"),kl=n("()"),xl=n(` and returns the output,
together with resized and normalized `),Gn=s("code"),wl=n("pixel_values"),Tl=n(". In case "),Uo=s("a"),Ml=n("LayoutLMv3FeatureExtractor"),$l=n(` was initialized
with `),Xn=s("code"),zl=n("apply_ocr"),ql=n(" set to "),Yn=s("code"),El=n("False"),Fl=n(", it passes the words ("),Jn=s("code"),Pl=n("text"),Cl=n("/`"),Kn=s("code"),Al=n("text_pair"),jl=n(") and "),Zn=s("code"),Il=n("boxes"),Nl=n(` specified by the user
along with the additional arguments to `),St=s("a"),ea=s("strong"),Dl=n("call"),Sl=n("()"),Ol=n(` and returns the output, together with
resized and normalized `),ta=s("code"),Rl=n("pixel_values"),Wl=n("."),Bl=c(),oa=s("p"),Ul=n("Please refer to the docstring of the above two methods for more information."),Ua=c(),_e=s("h2"),De=s("a"),na=s("span"),y(Ot.$$.fragment),Ql=c(),aa=s("span"),Vl=n("LayoutLMv3Model"),Qa=c(),K=s("div"),y(Rt.$$.fragment),Hl=c(),Wt=s("p"),Gl=n(`The bare LayoutLMv3 Model transformer outputting raw hidden-states without any specific head on top.
This model is a PyTorch `),Bt=s("a"),Xl=n("torch.nn.Module"),Yl=n(` sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),Jl=c(),H=s("div"),y(Ut.$$.fragment),Kl=c(),ve=s("p"),Zl=n("The "),Qo=s("a"),ed=n("LayoutLMv3Model"),td=n(" forward method, overrides the "),sa=s("code"),od=n("__call__"),nd=n(" special method."),ad=c(),y(Se.$$.fragment),sd=c(),y(Oe.$$.fragment),Va=c(),ye=s("h2"),Re=s("a"),ra=s("span"),y(Qt.$$.fragment),rd=c(),ia=s("span"),id=n("LayoutLMv3ForSequenceClassification"),Ha=c(),R=s("div"),y(Vt.$$.fragment),ld=c(),Ht=s("p"),dd=n(`LayoutLMv3 Model with a sequence classification head on top (a linear layer on top of the final hidden state of the
[CLS] token) e.g. for document image classification tasks such as the
`),Gt=s("a"),cd=n("RVL-CDIP"),ud=n(" dataset."),pd=c(),Xt=s("p"),md=n("This model is a PyTorch "),Yt=s("a"),hd=n("torch.nn.Module"),fd=n(` sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),gd=c(),G=s("div"),y(Jt.$$.fragment),_d=c(),be=s("p"),vd=n("The "),Vo=s("a"),yd=n("LayoutLMv3ForSequenceClassification"),bd=n(" forward method, overrides the "),la=s("code"),Ld=n("__call__"),kd=n(" special method."),xd=c(),y(We.$$.fragment),wd=c(),y(Be.$$.fragment),Ga=c(),Le=s("h2"),Ue=s("a"),da=s("span"),y(Kt.$$.fragment),Td=c(),ca=s("span"),Md=n("LayoutLMv3ForTokenClassification"),Xa=c(),W=s("div"),y(Zt.$$.fragment),$d=c(),B=s("p"),zd=n(`LayoutLMv3 Model with a token classification head on top (a linear layer on top of the final hidden states) e.g.
for sequence labeling (information extraction) tasks such as `),eo=s("a"),qd=n("FUNSD"),Ed=n(`,
`),to=s("a"),Fd=n("SROIE"),Pd=n(", "),oo=s("a"),Cd=n("CORD"),Ad=n(` and
`),no=s("a"),jd=n("Kleister-NDA"),Id=n("."),Nd=c(),ao=s("p"),Dd=n("This model is a PyTorch "),so=s("a"),Sd=n("torch.nn.Module"),Od=n(` sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),Rd=c(),X=s("div"),y(ro.$$.fragment),Wd=c(),ke=s("p"),Bd=n("The "),Ho=s("a"),Ud=n("LayoutLMv3ForTokenClassification"),Qd=n(" forward method, overrides the "),ua=s("code"),Vd=n("__call__"),Hd=n(" special method."),Gd=c(),y(Qe.$$.fragment),Xd=c(),y(Ve.$$.fragment),Ya=c(),xe=s("h2"),He=s("a"),pa=s("span"),y(io.$$.fragment),Yd=c(),ma=s("span"),Jd=n("LayoutLMv3ForQuestionAnswering"),Ja=c(),U=s("div"),y(lo.$$.fragment),Kd=c(),Z=s("p"),Zd=n(`LayoutLMv3 Model with a span classification head on top for extractive question-answering tasks such as
`),co=s("a"),ec=n("DocVQA"),tc=n(` (a linear layer on top of the text part of the hidden-states output to
compute `),ha=s("code"),oc=n("span start logits"),nc=n(" and "),fa=s("code"),ac=n("span end logits"),sc=n(")."),rc=c(),uo=s("p"),ic=n("This model is a PyTorch "),po=s("a"),lc=n("torch.nn.Module"),dc=n(` sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),cc=c(),Y=s("div"),y(mo.$$.fragment),uc=c(),we=s("p"),pc=n("The "),Go=s("a"),mc=n("LayoutLMv3ForQuestionAnswering"),hc=n(" forward method, overrides the "),ga=s("code"),fc=n("__call__"),gc=n(" special method."),_c=c(),y(Ge.$$.fragment),vc=c(),y(Xe.$$.fragment),this.h()},l(o){const m=Ap('[data-svelte="svelte-1phssyn"]',document.head);p=r(m,"META",{name:!0,content:!0}),m.forEach(t),T=u(o),_=r(o,"H1",{class:!0});var ho=i(_);f=r(ho,"A",{id:!0,class:!0,href:!0});var _a=i(f);v=r(_a,"SPAN",{});var va=i(v);b(d.$$.fragment,va),va.forEach(t),_a.forEach(t),h=u(ho),C=r(ho,"SPAN",{});var ya=i(C);ws=a(ya,"LayoutLMv3"),ya.forEach(t),ho.forEach(t),Ta=u(o),le=r(o,"H2",{class:!0});var fo=i(le);Te=r(fo,"A",{id:!0,class:!0,href:!0});var ba=i(Te);cn=r(ba,"SPAN",{});var La=i(cn);b(lt.$$.fragment,La),La.forEach(t),ba.forEach(t),Ts=u(fo),un=r(fo,"SPAN",{});var ka=i(un);Ms=a(ka,"Overview"),ka.forEach(t),fo.forEach(t),Ma=u(o),Q=r(o,"P",{});var ee=i(Q);$s=a(ee,"The LayoutLMv3 model was proposed in "),dt=r(ee,"A",{href:!0,rel:!0});var xa=i(dt);zs=a(xa,"LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking"),xa.forEach(t),qs=a(ee,` by Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, Furu Wei.
LayoutLMv3 simplifies `),go=r(ee,"A",{href:!0});var Tc=i(go);Es=a(Tc,"LayoutLMv2"),Tc.forEach(t),Fs=a(ee," by using patch embeddings (as in "),_o=r(ee,"A",{href:!0});var Mc=i(_o);Ps=a(Mc,"ViT"),Mc.forEach(t),Cs=a(ee,`) instead of leveraging a CNN backbone, and pre-trains the model on 3 objectives: masked language modeling (MLM), masked image modeling (MIM)
and word-patch alignment (WPA).`),ee.forEach(t),$a=u(o),vo=r(o,"P",{});var $c=i(vo);As=a($c,"The abstract from the paper is the following:"),$c.forEach(t),za=u(o),yo=r(o,"P",{});var zc=i(yo);pn=r(zc,"EM",{});var qc=i(pn);js=a(qc,"Self-supervised pre-training techniques have achieved remarkable progress in Document AI. Most multimodal pre-trained models use a masked language modeling objective to learn bidirectional representations on the text modality, but they differ in pre-training objectives for the image modality. This discrepancy adds difficulty to multimodal representation learning. In this paper, we propose LayoutLMv3 to pre-train multimodal Transformers for Document AI with unified text and image masking. Additionally, LayoutLMv3 is pre-trained with a word-patch alignment objective to learn cross-modal alignment by predicting whether the corresponding image patch of a text word is masked. The simple unified architecture and training objectives make LayoutLMv3 a general-purpose pre-trained model for both text-centric and image-centric Document AI tasks. Experimental results show that LayoutLMv3 achieves state-of-the-art performance not only in text-centric tasks, including form understanding, receipt understanding, and document visual question answering, but also in image-centric tasks such as document image classification and document layout analysis."),qc.forEach(t),zc.forEach(t),qa=u(o),bo=r(o,"P",{});var Ec=i(bo);Is=a(Ec,"Tips:"),Ec.forEach(t),Ea=u(o),V=r(o,"UL",{});var Ye=i(V);Me=r(Ye,"LI",{});var wa=i(Me);Ns=a(wa,"In terms of data processing, LayoutLMv3 is identical to its predecessor "),Lo=r(wa,"A",{href:!0});var Fc=i(Lo);Ds=a(Fc,"LayoutLMv2"),Fc.forEach(t),Ss=a(wa,", except that:"),ct=r(wa,"UL",{});var Za=i(ct);mn=r(Za,"LI",{});var Pc=i(mn);Os=a(Pc,"images need to be resized and normalized with channels in regular RGB format. LayoutLMv2 on the other hand normalizes the images internally and expects the channels in BGR format."),Pc.forEach(t),Rs=u(Za),N=r(Za,"LI",{});var se=i(N);Ws=a(se,`text is tokenized using byte-pair encoding (BPE), as opposed to WordPiece.
Due to these differences in data preprocessing, one can use `),ko=r(se,"A",{href:!0});var Cc=i(ko);Bs=a(Cc,"LayoutLMv3Processor"),Cc.forEach(t),Us=a(se," which internally combines a "),xo=r(se,"A",{href:!0});var Ac=i(xo);Qs=a(Ac,"LayoutLMv3FeatureExtractor"),Ac.forEach(t),Vs=a(se," (for the image modality) and a "),wo=r(se,"A",{href:!0});var jc=i(wo);Hs=a(jc,"LayoutLMv3Tokenizer"),jc.forEach(t),Gs=a(se,"/"),To=r(se,"A",{href:!0});var Ic=i(To);Xs=a(Ic,"LayoutLMv3TokenizerFast"),Ic.forEach(t),Ys=a(se," (for the text modality) to prepare all data for the model."),se.forEach(t),Za.forEach(t),wa.forEach(t),Js=u(Ye),de=r(Ye,"LI",{});var Xo=i(de);Ks=a(Xo,"Regarding usage of "),Mo=r(Xo,"A",{href:!0});var Nc=i(Mo);Zs=a(Nc,"LayoutLMv3Processor"),Nc.forEach(t),er=a(Xo,", we refer to the "),$o=r(Xo,"A",{href:!0});var Dc=i($o);tr=a(Dc,"usage guide"),Dc.forEach(t),or=a(Xo," of its predecessor."),Xo.forEach(t),nr=u(Ye),ut=r(Ye,"LI",{});var es=i(ut);ar=a(es,"Demo notebooks for LayoutLMv3 can be found "),pt=r(es,"A",{href:!0,rel:!0});var Sc=i(pt);sr=a(Sc,"here"),Sc.forEach(t),rr=a(es,"."),es.forEach(t),ir=u(Ye),mt=r(Ye,"LI",{});var ts=i(mt);lr=a(ts,"Demo scripts can be found "),ht=r(ts,"A",{href:!0,rel:!0});var Oc=i(ht);dr=a(Oc,"here"),Oc.forEach(t),cr=a(ts,"."),ts.forEach(t),Ye.forEach(t),Fa=u(o),$e=r(o,"IMG",{src:!0,alt:!0,width:!0}),Pa=u(o),ze=r(o,"SMALL",{});var os=i(ze);ur=a(os,"LayoutLMv3 architecture. Taken from the "),zo=r(os,"A",{href:!0});var Rc=i(zo);pr=a(Rc,"original paper"),Rc.forEach(t),mr=a(os,"."),os.forEach(t),Ca=u(o),oe=r(o,"P",{});var Yo=i(oe);hr=a(Yo,"This model was contributed by "),ft=r(Yo,"A",{href:!0,rel:!0});var Wc=i(ft);fr=a(Wc,"nielsr"),Wc.forEach(t),gr=a(Yo,". The original code can be found "),gt=r(Yo,"A",{href:!0,rel:!0});var Bc=i(gt);_r=a(Bc,"here"),Bc.forEach(t),vr=a(Yo,"."),Yo.forEach(t),Aa=u(o),ce=r(o,"H2",{class:!0});var ns=i(ce);qe=r(ns,"A",{id:!0,class:!0,href:!0});var Uc=i(qe);hn=r(Uc,"SPAN",{});var Qc=i(hn);b(_t.$$.fragment,Qc),Qc.forEach(t),Uc.forEach(t),yr=u(ns),fn=r(ns,"SPAN",{});var Vc=i(fn);br=a(Vc,"LayoutLMv3Config"),Vc.forEach(t),ns.forEach(t),ja=u(o),D=r(o,"DIV",{class:!0});var Je=i(D);b(vt.$$.fragment,Je),Lr=u(Je),ue=r(Je,"P",{});var Jo=i(ue);kr=a(Jo,"This is the configuration class to store the configuration of a "),qo=r(Jo,"A",{href:!0});var Hc=i(qo);xr=a(Hc,"LayoutLMv3Model"),Hc.forEach(t),wr=a(Jo,`. It is used to instantiate an
LayoutLMv3 model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the LayoutLMv3
`),yt=r(Jo,"A",{href:!0,rel:!0});var Gc=i(yt);Tr=a(Gc,"microsoft/layoutlmv3-base"),Gc.forEach(t),Mr=a(Jo," architecture."),Jo.forEach(t),$r=u(Je),pe=r(Je,"P",{});var Ko=i(pe);zr=a(Ko,"Configuration objects inherit from "),Eo=r(Ko,"A",{href:!0});var Xc=i(Eo);qr=a(Xc,"PretrainedConfig"),Xc.forEach(t),Er=a(Ko,` and can be used to control the model outputs. Read the
documentation from `),Fo=r(Ko,"A",{href:!0});var Yc=i(Fo);Fr=a(Yc,"PretrainedConfig"),Yc.forEach(t),Pr=a(Ko," for more information."),Ko.forEach(t),Cr=u(Je),b(Ee.$$.fragment,Je),Je.forEach(t),Ia=u(o),me=r(o,"H2",{class:!0});var as=i(me);Fe=r(as,"A",{id:!0,class:!0,href:!0});var Jc=i(Fe);gn=r(Jc,"SPAN",{});var Kc=i(gn);b(bt.$$.fragment,Kc),Kc.forEach(t),Jc.forEach(t),Ar=u(as),_n=r(as,"SPAN",{});var Zc=i(_n);jr=a(Zc,"LayoutLMv3FeatureExtractor"),Zc.forEach(t),as.forEach(t),Na=u(o),S=r(o,"DIV",{class:!0});var Ke=i(S);b(Lt.$$.fragment,Ke),Ir=u(Ke),vn=r(Ke,"P",{});var eu=i(vn);Nr=a(eu,`Constructs a LayoutLMv3 feature extractor. This can be used to resize + normalize document images, as well as to
apply OCR on them in order to get a list of words and normalized bounding boxes.`),eu.forEach(t),Dr=u(Ke),kt=r(Ke,"P",{});var ss=i(kt);Sr=a(ss,"This feature extractor inherits from "),yn=r(ss,"CODE",{});var tu=i(yn);Or=a(tu,"PreTrainedFeatureExtractor()"),tu.forEach(t),Rr=a(ss,` which contains most
of the main methods. Users should refer to this superclass for more information regarding those methods.`),ss.forEach(t),Wr=u(Ke),ne=r(Ke,"DIV",{class:!0});var Zo=i(ne);b(xt.$$.fragment,Zo),Br=u(Zo),bn=r(Zo,"P",{});var ou=i(bn);Ur=a(ou,"Main method to prepare for the model one or several image(s)."),ou.forEach(t),Qr=u(Zo),b(Pe.$$.fragment,Zo),Zo.forEach(t),Ke.forEach(t),Da=u(o),he=r(o,"H2",{class:!0});var rs=i(he);Ce=r(rs,"A",{id:!0,class:!0,href:!0});var nu=i(Ce);Ln=r(nu,"SPAN",{});var au=i(Ln);b(wt.$$.fragment,au),au.forEach(t),nu.forEach(t),Vr=u(rs),kn=r(rs,"SPAN",{});var su=i(kn);Hr=a(su,"LayoutLMv3Tokenizer"),su.forEach(t),rs.forEach(t),Sa=u(o),A=r(o,"DIV",{class:!0});var J=i(A);b(Tt.$$.fragment,J),Gr=u(J),F=r(J,"P",{});var j=i(F);Xr=a(j,"Construct a LayoutLMv3 tokenizer. Based on "),xn=r(j,"CODE",{});var ru=i(xn);Yr=a(ru,"RoBERTatokenizer"),ru.forEach(t),Jr=a(j,` (Byte Pair Encoding or BPE).
`),Po=r(j,"A",{href:!0});var iu=i(Po);Kr=a(iu,"LayoutLMv3Tokenizer"),iu.forEach(t),Zr=a(j,` can be used to turn words, word-level bounding boxes and optional word labels to
token-level `),wn=r(j,"CODE",{});var lu=i(wn);ei=a(lu,"input_ids"),lu.forEach(t),ti=a(j,", "),Tn=r(j,"CODE",{});var du=i(Tn);oi=a(du,"attention_mask"),du.forEach(t),ni=a(j,", "),Mn=r(j,"CODE",{});var cu=i(Mn);ai=a(cu,"token_type_ids"),cu.forEach(t),si=a(j,", "),$n=r(j,"CODE",{});var uu=i($n);ri=a(uu,"bbox"),uu.forEach(t),ii=a(j,", and optional "),zn=r(j,"CODE",{});var pu=i(zn);li=a(pu,"labels"),pu.forEach(t),di=a(j,` (for token
classification).`),j.forEach(t),ci=u(J),Mt=r(J,"P",{});var is=i(Mt);ui=a(is,"This tokenizer inherits from "),Co=r(is,"A",{href:!0});var mu=i(Co);pi=a(mu,"PreTrainedTokenizer"),mu.forEach(t),mi=a(is,` which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`),is.forEach(t),hi=u(J),Ao=r(J,"P",{});var yc=i(Ao);jo=r(yc,"A",{href:!0});var hu=i(jo);fi=a(hu,"LayoutLMv3Tokenizer"),hu.forEach(t),gi=a(yc,` runs end-to-end tokenization: punctuation splitting and wordpiece. It also turns the
word-level bounding boxes into token-level bounding boxes.`),yc.forEach(t),_i=u(J),Ae=r(J,"DIV",{class:!0});var ls=i(Ae);b($t.$$.fragment,ls),vi=u(ls),qn=r(ls,"P",{});var fu=i(qn);yi=a(fu,`Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences with word-level normalized bounding boxes and optional labels.`),fu.forEach(t),ls.forEach(t),bi=u(J),Io=r(J,"DIV",{class:!0});var gu=i(Io);b(zt.$$.fragment,gu),gu.forEach(t),J.forEach(t),Oa=u(o),fe=r(o,"H2",{class:!0});var ds=i(fe);je=r(ds,"A",{id:!0,class:!0,href:!0});var _u=i(je);En=r(_u,"SPAN",{});var vu=i(En);b(qt.$$.fragment,vu),vu.forEach(t),_u.forEach(t),Li=u(ds),Fn=r(ds,"SPAN",{});var yu=i(Fn);ki=a(yu,"LayoutLMv3TokenizerFast"),yu.forEach(t),ds.forEach(t),Ra=u(o),O=r(o,"DIV",{class:!0});var Ze=i(O);b(Et.$$.fragment,Ze),xi=u(Ze),Ft=r(Ze,"P",{});var cs=i(Ft);wi=a(cs,"Construct a \u201Cfast\u201D LayoutLMv3 tokenizer (backed by HuggingFace\u2019s "),Pn=r(cs,"EM",{});var bu=i(Pn);Ti=a(bu,"tokenizers"),bu.forEach(t),Mi=a(cs," library). Based on BPE."),cs.forEach(t),$i=u(Ze),Pt=r(Ze,"P",{});var us=i(Pt);zi=a(us,"This tokenizer inherits from "),No=r(us,"A",{href:!0});var Lu=i(No);qi=a(Lu,"PreTrainedTokenizerFast"),Lu.forEach(t),Ei=a(us,` which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`),us.forEach(t),Fi=u(Ze),Ie=r(Ze,"DIV",{class:!0});var ps=i(Ie);b(Ct.$$.fragment,ps),Pi=u(ps),Cn=r(ps,"P",{});var ku=i(Cn);Ci=a(ku,`Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences with word-level normalized bounding boxes and optional labels.`),ku.forEach(t),ps.forEach(t),Ze.forEach(t),Wa=u(o),ge=r(o,"H2",{class:!0});var ms=i(ge);Ne=r(ms,"A",{id:!0,class:!0,href:!0});var xu=i(Ne);An=r(xu,"SPAN",{});var wu=i(An);b(At.$$.fragment,wu),wu.forEach(t),xu.forEach(t),Ai=u(ms),jn=r(ms,"SPAN",{});var Tu=i(jn);ji=a(Tu,"LayoutLMv3Processor"),Tu.forEach(t),ms.forEach(t),Ba=u(o),I=r(o,"DIV",{class:!0});var re=i(I);b(jt.$$.fragment,re),Ii=u(re),In=r(re,"P",{});var Mu=i(In);Ni=a(Mu,`Constructs a LayoutLMv3 processor which combines a LayoutLMv3 feature extractor and a LayoutLMv3 tokenizer into a
single processor.`),Mu.forEach(t),Di=u(re),Do=r(re,"P",{});var bc=i(Do);So=r(bc,"A",{href:!0});var $u=i(So);Si=a($u,"LayoutLMv3Processor"),$u.forEach(t),Oi=a(bc," offers all the functionalities you need to prepare data for the model."),bc.forEach(t),Ri=u(re),z=r(re,"P",{});var E=i(z);Wi=a(E,"It first uses "),Oo=r(E,"A",{href:!0});var zu=i(Oo);Bi=a(zu,"LayoutLMv3FeatureExtractor"),zu.forEach(t),Ui=a(E,` to resize and normalize document images, and optionally applies OCR to
get words and normalized bounding boxes. These are then provided to `),Ro=r(E,"A",{href:!0});var qu=i(Ro);Qi=a(qu,"LayoutLMv3Tokenizer"),qu.forEach(t),Vi=a(E,` or
`),Wo=r(E,"A",{href:!0});var Eu=i(Wo);Hi=a(Eu,"LayoutLMv3TokenizerFast"),Eu.forEach(t),Gi=a(E,", which turns the words and bounding boxes into token-level "),Nn=r(E,"CODE",{});var Fu=i(Nn);Xi=a(Fu,"input_ids"),Fu.forEach(t),Yi=a(E,`,
`),Dn=r(E,"CODE",{});var Pu=i(Dn);Ji=a(Pu,"attention_mask"),Pu.forEach(t),Ki=a(E,", "),Sn=r(E,"CODE",{});var Cu=i(Sn);Zi=a(Cu,"token_type_ids"),Cu.forEach(t),el=a(E,", "),On=r(E,"CODE",{});var Au=i(On);tl=a(Au,"bbox"),Au.forEach(t),ol=a(E,". Optionally, one can provide integer "),Rn=r(E,"CODE",{});var ju=i(Rn);nl=a(ju,"word_labels"),ju.forEach(t),al=a(E,`, which are turned
into token-level `),Wn=r(E,"CODE",{});var Iu=i(Wn);sl=a(Iu,"labels"),Iu.forEach(t),rl=a(E," for token classification tasks (such as FUNSD, CORD)."),E.forEach(t),il=u(re),ae=r(re,"DIV",{class:!0});var en=i(ae);b(It.$$.fragment,en),ll=u(en),M=r(en,"P",{});var $=i(M);dl=a($,"This method first forwards the "),Bn=r($,"CODE",{});var Nu=i(Bn);cl=a(Nu,"images"),Nu.forEach(t),ul=a($," argument to "),Nt=r($,"A",{href:!0});var Lc=i(Nt);Un=r(Lc,"STRONG",{});var Du=i(Un);pl=a(Du,"call"),Du.forEach(t),ml=a(Lc,"()"),Lc.forEach(t),hl=a($,`. In case
`),Bo=r($,"A",{href:!0});var Su=i(Bo);fl=a(Su,"LayoutLMv3FeatureExtractor"),Su.forEach(t),gl=a($," was initialized with "),Qn=r($,"CODE",{});var Ou=i(Qn);_l=a(Ou,"apply_ocr"),Ou.forEach(t),vl=a($," set to "),Vn=r($,"CODE",{});var Ru=i(Vn);yl=a(Ru,"True"),Ru.forEach(t),bl=a($,`, it passes the obtained words and
bounding boxes along with the additional arguments to `),Dt=r($,"A",{href:!0});var kc=i(Dt);Hn=r(kc,"STRONG",{});var Wu=i(Hn);Ll=a(Wu,"call"),Wu.forEach(t),kl=a(kc,"()"),kc.forEach(t),xl=a($,` and returns the output,
together with resized and normalized `),Gn=r($,"CODE",{});var Bu=i(Gn);wl=a(Bu,"pixel_values"),Bu.forEach(t),Tl=a($,". In case "),Uo=r($,"A",{href:!0});var Uu=i(Uo);Ml=a(Uu,"LayoutLMv3FeatureExtractor"),Uu.forEach(t),$l=a($,` was initialized
with `),Xn=r($,"CODE",{});var Qu=i(Xn);zl=a(Qu,"apply_ocr"),Qu.forEach(t),ql=a($," set to "),Yn=r($,"CODE",{});var Vu=i(Yn);El=a(Vu,"False"),Vu.forEach(t),Fl=a($,", it passes the words ("),Jn=r($,"CODE",{});var Hu=i(Jn);Pl=a(Hu,"text"),Hu.forEach(t),Cl=a($,"/`"),Kn=r($,"CODE",{});var Gu=i(Kn);Al=a(Gu,"text_pair"),Gu.forEach(t),jl=a($,") and "),Zn=r($,"CODE",{});var Xu=i(Zn);Il=a(Xu,"boxes"),Xu.forEach(t),Nl=a($,` specified by the user
along with the additional arguments to `),St=r($,"A",{href:!0});var xc=i(St);ea=r(xc,"STRONG",{});var Yu=i(ea);Dl=a(Yu,"call"),Yu.forEach(t),Sl=a(xc,"()"),xc.forEach(t),Ol=a($,` and returns the output, together with
resized and normalized `),ta=r($,"CODE",{});var Ju=i(ta);Rl=a(Ju,"pixel_values"),Ju.forEach(t),Wl=a($,"."),$.forEach(t),Bl=u(en),oa=r(en,"P",{});var Ku=i(oa);Ul=a(Ku,"Please refer to the docstring of the above two methods for more information."),Ku.forEach(t),en.forEach(t),re.forEach(t),Ua=u(o),_e=r(o,"H2",{class:!0});var hs=i(_e);De=r(hs,"A",{id:!0,class:!0,href:!0});var Zu=i(De);na=r(Zu,"SPAN",{});var ep=i(na);b(Ot.$$.fragment,ep),ep.forEach(t),Zu.forEach(t),Ql=u(hs),aa=r(hs,"SPAN",{});var tp=i(aa);Vl=a(tp,"LayoutLMv3Model"),tp.forEach(t),hs.forEach(t),Qa=u(o),K=r(o,"DIV",{class:!0});var tn=i(K);b(Rt.$$.fragment,tn),Hl=u(tn),Wt=r(tn,"P",{});var fs=i(Wt);Gl=a(fs,`The bare LayoutLMv3 Model transformer outputting raw hidden-states without any specific head on top.
This model is a PyTorch `),Bt=r(fs,"A",{href:!0,rel:!0});var op=i(Bt);Xl=a(op,"torch.nn.Module"),op.forEach(t),Yl=a(fs,` sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),fs.forEach(t),Jl=u(tn),H=r(tn,"DIV",{class:!0});var et=i(H);b(Ut.$$.fragment,et),Kl=u(et),ve=r(et,"P",{});var on=i(ve);Zl=a(on,"The "),Qo=r(on,"A",{href:!0});var np=i(Qo);ed=a(np,"LayoutLMv3Model"),np.forEach(t),td=a(on," forward method, overrides the "),sa=r(on,"CODE",{});var ap=i(sa);od=a(ap,"__call__"),ap.forEach(t),nd=a(on," special method."),on.forEach(t),ad=u(et),b(Se.$$.fragment,et),sd=u(et),b(Oe.$$.fragment,et),et.forEach(t),tn.forEach(t),Va=u(o),ye=r(o,"H2",{class:!0});var gs=i(ye);Re=r(gs,"A",{id:!0,class:!0,href:!0});var sp=i(Re);ra=r(sp,"SPAN",{});var rp=i(ra);b(Qt.$$.fragment,rp),rp.forEach(t),sp.forEach(t),rd=u(gs),ia=r(gs,"SPAN",{});var ip=i(ia);id=a(ip,"LayoutLMv3ForSequenceClassification"),ip.forEach(t),gs.forEach(t),Ha=u(o),R=r(o,"DIV",{class:!0});var tt=i(R);b(Vt.$$.fragment,tt),ld=u(tt),Ht=r(tt,"P",{});var _s=i(Ht);dd=a(_s,`LayoutLMv3 Model with a sequence classification head on top (a linear layer on top of the final hidden state of the
[CLS] token) e.g. for document image classification tasks such as the
`),Gt=r(_s,"A",{href:!0,rel:!0});var lp=i(Gt);cd=a(lp,"RVL-CDIP"),lp.forEach(t),ud=a(_s," dataset."),_s.forEach(t),pd=u(tt),Xt=r(tt,"P",{});var vs=i(Xt);md=a(vs,"This model is a PyTorch "),Yt=r(vs,"A",{href:!0,rel:!0});var dp=i(Yt);hd=a(dp,"torch.nn.Module"),dp.forEach(t),fd=a(vs,` sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),vs.forEach(t),gd=u(tt),G=r(tt,"DIV",{class:!0});var ot=i(G);b(Jt.$$.fragment,ot),_d=u(ot),be=r(ot,"P",{});var nn=i(be);vd=a(nn,"The "),Vo=r(nn,"A",{href:!0});var cp=i(Vo);yd=a(cp,"LayoutLMv3ForSequenceClassification"),cp.forEach(t),bd=a(nn," forward method, overrides the "),la=r(nn,"CODE",{});var up=i(la);Ld=a(up,"__call__"),up.forEach(t),kd=a(nn," special method."),nn.forEach(t),xd=u(ot),b(We.$$.fragment,ot),wd=u(ot),b(Be.$$.fragment,ot),ot.forEach(t),tt.forEach(t),Ga=u(o),Le=r(o,"H2",{class:!0});var ys=i(Le);Ue=r(ys,"A",{id:!0,class:!0,href:!0});var pp=i(Ue);da=r(pp,"SPAN",{});var mp=i(da);b(Kt.$$.fragment,mp),mp.forEach(t),pp.forEach(t),Td=u(ys),ca=r(ys,"SPAN",{});var hp=i(ca);Md=a(hp,"LayoutLMv3ForTokenClassification"),hp.forEach(t),ys.forEach(t),Xa=u(o),W=r(o,"DIV",{class:!0});var nt=i(W);b(Zt.$$.fragment,nt),$d=u(nt),B=r(nt,"P",{});var ie=i(B);zd=a(ie,`LayoutLMv3 Model with a token classification head on top (a linear layer on top of the final hidden states) e.g.
for sequence labeling (information extraction) tasks such as `),eo=r(ie,"A",{href:!0,rel:!0});var fp=i(eo);qd=a(fp,"FUNSD"),fp.forEach(t),Ed=a(ie,`,
`),to=r(ie,"A",{href:!0,rel:!0});var gp=i(to);Fd=a(gp,"SROIE"),gp.forEach(t),Pd=a(ie,", "),oo=r(ie,"A",{href:!0,rel:!0});var _p=i(oo);Cd=a(_p,"CORD"),_p.forEach(t),Ad=a(ie,` and
`),no=r(ie,"A",{href:!0,rel:!0});var vp=i(no);jd=a(vp,"Kleister-NDA"),vp.forEach(t),Id=a(ie,"."),ie.forEach(t),Nd=u(nt),ao=r(nt,"P",{});var bs=i(ao);Dd=a(bs,"This model is a PyTorch "),so=r(bs,"A",{href:!0,rel:!0});var yp=i(so);Sd=a(yp,"torch.nn.Module"),yp.forEach(t),Od=a(bs,` sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),bs.forEach(t),Rd=u(nt),X=r(nt,"DIV",{class:!0});var at=i(X);b(ro.$$.fragment,at),Wd=u(at),ke=r(at,"P",{});var an=i(ke);Bd=a(an,"The "),Ho=r(an,"A",{href:!0});var bp=i(Ho);Ud=a(bp,"LayoutLMv3ForTokenClassification"),bp.forEach(t),Qd=a(an," forward method, overrides the "),ua=r(an,"CODE",{});var Lp=i(ua);Vd=a(Lp,"__call__"),Lp.forEach(t),Hd=a(an," special method."),an.forEach(t),Gd=u(at),b(Qe.$$.fragment,at),Xd=u(at),b(Ve.$$.fragment,at),at.forEach(t),nt.forEach(t),Ya=u(o),xe=r(o,"H2",{class:!0});var Ls=i(xe);He=r(Ls,"A",{id:!0,class:!0,href:!0});var kp=i(He);pa=r(kp,"SPAN",{});var xp=i(pa);b(io.$$.fragment,xp),xp.forEach(t),kp.forEach(t),Yd=u(Ls),ma=r(Ls,"SPAN",{});var wp=i(ma);Jd=a(wp,"LayoutLMv3ForQuestionAnswering"),wp.forEach(t),Ls.forEach(t),Ja=u(o),U=r(o,"DIV",{class:!0});var st=i(U);b(lo.$$.fragment,st),Kd=u(st),Z=r(st,"P",{});var rt=i(Z);Zd=a(rt,`LayoutLMv3 Model with a span classification head on top for extractive question-answering tasks such as
`),co=r(rt,"A",{href:!0,rel:!0});var Tp=i(co);ec=a(Tp,"DocVQA"),Tp.forEach(t),tc=a(rt,` (a linear layer on top of the text part of the hidden-states output to
compute `),ha=r(rt,"CODE",{});var Mp=i(ha);oc=a(Mp,"span start logits"),Mp.forEach(t),nc=a(rt," and "),fa=r(rt,"CODE",{});var $p=i(fa);ac=a($p,"span end logits"),$p.forEach(t),sc=a(rt,")."),rt.forEach(t),rc=u(st),uo=r(st,"P",{});var ks=i(uo);ic=a(ks,"This model is a PyTorch "),po=r(ks,"A",{href:!0,rel:!0});var zp=i(po);lc=a(zp,"torch.nn.Module"),zp.forEach(t),dc=a(ks,` sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),ks.forEach(t),cc=u(st),Y=r(st,"DIV",{class:!0});var it=i(Y);b(mo.$$.fragment,it),uc=u(it),we=r(it,"P",{});var sn=i(we);pc=a(sn,"The "),Go=r(sn,"A",{href:!0});var qp=i(Go);mc=a(qp,"LayoutLMv3ForQuestionAnswering"),qp.forEach(t),hc=a(sn," forward method, overrides the "),ga=r(sn,"CODE",{});var Ep=i(ga);fc=a(Ep,"__call__"),Ep.forEach(t),gc=a(sn," special method."),sn.forEach(t),_c=u(it),b(Ge.$$.fragment,it),vc=u(it),b(Xe.$$.fragment,it),it.forEach(t),st.forEach(t),this.h()},h(){l(p,"name","hf:doc:metadata"),l(p,"content",JSON.stringify(Gp)),l(f,"id","layoutlmv3"),l(f,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),l(f,"href","#layoutlmv3"),l(_,"class","relative group"),l(Te,"id","overview"),l(Te,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),l(Te,"href","#overview"),l(le,"class","relative group"),l(dt,"href","https://arxiv.org/abs/2204.08387"),l(dt,"rel","nofollow"),l(go,"href","layoutlmv2"),l(_o,"href","vit"),l(Lo,"href","layoutlmv2"),l(ko,"href","/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3Processor"),l(xo,"href","/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3FeatureExtractor"),l(wo,"href","/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3Tokenizer"),l(To,"href","/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3TokenizerFast"),l(Mo,"href","/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3Processor"),l($o,"href","layoutlmv2#usage-layoutlmv2processor"),l(pt,"href","https://github.com/NielsRogge/Transformers-Tutorials/tree/master/LayoutLMv3"),l(pt,"rel","nofollow"),l(ht,"href","https://github.com/huggingface/transformers/tree/main/examples/research_projects/layoutlmv3"),l(ht,"rel","nofollow"),jp($e.src,wc="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/layoutlmv3_architecture.png")||l($e,"src",wc),l($e,"alt","drawing"),l($e,"width","600"),l(zo,"href","https://arxiv.org/abs/2204.08387"),l(ft,"href","https://huggingface.co/nielsr"),l(ft,"rel","nofollow"),l(gt,"href","https://github.com/microsoft/unilm/tree/master/layoutlmv3"),l(gt,"rel","nofollow"),l(qe,"id","transformers.LayoutLMv3Config"),l(qe,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),l(qe,"href","#transformers.LayoutLMv3Config"),l(ce,"class","relative group"),l(qo,"href","/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3Model"),l(yt,"href","https://huggingface.co/microsoft/layoutlmv3-base"),l(yt,"rel","nofollow"),l(Eo,"href","/docs/transformers/pr_18400/en/main_classes/configuration#transformers.PretrainedConfig"),l(Fo,"href","/docs/transformers/pr_18400/en/main_classes/configuration#transformers.PretrainedConfig"),l(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),l(Fe,"id","transformers.LayoutLMv3FeatureExtractor"),l(Fe,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),l(Fe,"href","#transformers.LayoutLMv3FeatureExtractor"),l(me,"class","relative group"),l(ne,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),l(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),l(Ce,"id","transformers.LayoutLMv3Tokenizer"),l(Ce,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),l(Ce,"href","#transformers.LayoutLMv3Tokenizer"),l(he,"class","relative group"),l(Po,"href","/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3Tokenizer"),l(Co,"href","/docs/transformers/pr_18400/en/main_classes/tokenizer#transformers.PreTrainedTokenizer"),l(jo,"href","/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3Tokenizer"),l(Ae,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),l(Io,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),l(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),l(je,"id","transformers.LayoutLMv3TokenizerFast"),l(je,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),l(je,"href","#transformers.LayoutLMv3TokenizerFast"),l(fe,"class","relative group"),l(No,"href","/docs/transformers/pr_18400/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast"),l(Ie,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),l(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),l(Ne,"id","transformers.LayoutLMv3Processor"),l(Ne,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),l(Ne,"href","#transformers.LayoutLMv3Processor"),l(ge,"class","relative group"),l(So,"href","/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3Processor"),l(Oo,"href","/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3FeatureExtractor"),l(Ro,"href","/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3Tokenizer"),l(Wo,"href","/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3TokenizerFast"),l(Nt,"href","/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3FeatureExtractor.__call__"),l(Bo,"href","/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3FeatureExtractor"),l(Dt,"href","/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3Tokenizer.__call__"),l(Uo,"href","/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3FeatureExtractor"),l(St,"href","/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3Tokenizer.__call__"),l(ae,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),l(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),l(De,"id","transformers.LayoutLMv3Model"),l(De,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),l(De,"href","#transformers.LayoutLMv3Model"),l(_e,"class","relative group"),l(Bt,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),l(Bt,"rel","nofollow"),l(Qo,"href","/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3Model"),l(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),l(K,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),l(Re,"id","transformers.LayoutLMv3ForSequenceClassification"),l(Re,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),l(Re,"href","#transformers.LayoutLMv3ForSequenceClassification"),l(ye,"class","relative group"),l(Gt,"href","https://www.cs.cmu.edu/~aharley/rvl-cdip/"),l(Gt,"rel","nofollow"),l(Yt,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),l(Yt,"rel","nofollow"),l(Vo,"href","/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3ForSequenceClassification"),l(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),l(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),l(Ue,"id","transformers.LayoutLMv3ForTokenClassification"),l(Ue,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),l(Ue,"href","#transformers.LayoutLMv3ForTokenClassification"),l(Le,"class","relative group"),l(eo,"href","https://guillaumejaume.github.io/FUNSD/"),l(eo,"rel","nofollow"),l(to,"href","https://rrc.cvc.uab.es/?ch=13"),l(to,"rel","nofollow"),l(oo,"href","https://github.com/clovaai/cord"),l(oo,"rel","nofollow"),l(no,"href","https://github.com/applicaai/kleister-nda"),l(no,"rel","nofollow"),l(so,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),l(so,"rel","nofollow"),l(Ho,"href","/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3ForTokenClassification"),l(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),l(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),l(He,"id","transformers.LayoutLMv3ForQuestionAnswering"),l(He,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),l(He,"href","#transformers.LayoutLMv3ForQuestionAnswering"),l(xe,"class","relative group"),l(co,"href","https://rrc.cvc.uab.es/?ch=17"),l(co,"rel","nofollow"),l(po,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),l(po,"rel","nofollow"),l(Go,"href","/docs/transformers/pr_18400/en/model_doc/layoutlmv3#transformers.LayoutLMv3ForQuestionAnswering"),l(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),l(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(o,m){e(document.head,p),g(o,T,m),g(o,_,m),e(_,f),e(f,v),L(d,v,null),e(_,h),e(_,C),e(C,ws),g(o,Ta,m),g(o,le,m),e(le,Te),e(Te,cn),L(lt,cn,null),e(le,Ts),e(le,un),e(un,Ms),g(o,Ma,m),g(o,Q,m),e(Q,$s),e(Q,dt),e(dt,zs),e(Q,qs),e(Q,go),e(go,Es),e(Q,Fs),e(Q,_o),e(_o,Ps),e(Q,Cs),g(o,$a,m),g(o,vo,m),e(vo,As),g(o,za,m),g(o,yo,m),e(yo,pn),e(pn,js),g(o,qa,m),g(o,bo,m),e(bo,Is),g(o,Ea,m),g(o,V,m),e(V,Me),e(Me,Ns),e(Me,Lo),e(Lo,Ds),e(Me,Ss),e(Me,ct),e(ct,mn),e(mn,Os),e(ct,Rs),e(ct,N),e(N,Ws),e(N,ko),e(ko,Bs),e(N,Us),e(N,xo),e(xo,Qs),e(N,Vs),e(N,wo),e(wo,Hs),e(N,Gs),e(N,To),e(To,Xs),e(N,Ys),e(V,Js),e(V,de),e(de,Ks),e(de,Mo),e(Mo,Zs),e(de,er),e(de,$o),e($o,tr),e(de,or),e(V,nr),e(V,ut),e(ut,ar),e(ut,pt),e(pt,sr),e(ut,rr),e(V,ir),e(V,mt),e(mt,lr),e(mt,ht),e(ht,dr),e(mt,cr),g(o,Fa,m),g(o,$e,m),g(o,Pa,m),g(o,ze,m),e(ze,ur),e(ze,zo),e(zo,pr),e(ze,mr),g(o,Ca,m),g(o,oe,m),e(oe,hr),e(oe,ft),e(ft,fr),e(oe,gr),e(oe,gt),e(gt,_r),e(oe,vr),g(o,Aa,m),g(o,ce,m),e(ce,qe),e(qe,hn),L(_t,hn,null),e(ce,yr),e(ce,fn),e(fn,br),g(o,ja,m),g(o,D,m),L(vt,D,null),e(D,Lr),e(D,ue),e(ue,kr),e(ue,qo),e(qo,xr),e(ue,wr),e(ue,yt),e(yt,Tr),e(ue,Mr),e(D,$r),e(D,pe),e(pe,zr),e(pe,Eo),e(Eo,qr),e(pe,Er),e(pe,Fo),e(Fo,Fr),e(pe,Pr),e(D,Cr),L(Ee,D,null),g(o,Ia,m),g(o,me,m),e(me,Fe),e(Fe,gn),L(bt,gn,null),e(me,Ar),e(me,_n),e(_n,jr),g(o,Na,m),g(o,S,m),L(Lt,S,null),e(S,Ir),e(S,vn),e(vn,Nr),e(S,Dr),e(S,kt),e(kt,Sr),e(kt,yn),e(yn,Or),e(kt,Rr),e(S,Wr),e(S,ne),L(xt,ne,null),e(ne,Br),e(ne,bn),e(bn,Ur),e(ne,Qr),L(Pe,ne,null),g(o,Da,m),g(o,he,m),e(he,Ce),e(Ce,Ln),L(wt,Ln,null),e(he,Vr),e(he,kn),e(kn,Hr),g(o,Sa,m),g(o,A,m),L(Tt,A,null),e(A,Gr),e(A,F),e(F,Xr),e(F,xn),e(xn,Yr),e(F,Jr),e(F,Po),e(Po,Kr),e(F,Zr),e(F,wn),e(wn,ei),e(F,ti),e(F,Tn),e(Tn,oi),e(F,ni),e(F,Mn),e(Mn,ai),e(F,si),e(F,$n),e($n,ri),e(F,ii),e(F,zn),e(zn,li),e(F,di),e(A,ci),e(A,Mt),e(Mt,ui),e(Mt,Co),e(Co,pi),e(Mt,mi),e(A,hi),e(A,Ao),e(Ao,jo),e(jo,fi),e(Ao,gi),e(A,_i),e(A,Ae),L($t,Ae,null),e(Ae,vi),e(Ae,qn),e(qn,yi),e(A,bi),e(A,Io),L(zt,Io,null),g(o,Oa,m),g(o,fe,m),e(fe,je),e(je,En),L(qt,En,null),e(fe,Li),e(fe,Fn),e(Fn,ki),g(o,Ra,m),g(o,O,m),L(Et,O,null),e(O,xi),e(O,Ft),e(Ft,wi),e(Ft,Pn),e(Pn,Ti),e(Ft,Mi),e(O,$i),e(O,Pt),e(Pt,zi),e(Pt,No),e(No,qi),e(Pt,Ei),e(O,Fi),e(O,Ie),L(Ct,Ie,null),e(Ie,Pi),e(Ie,Cn),e(Cn,Ci),g(o,Wa,m),g(o,ge,m),e(ge,Ne),e(Ne,An),L(At,An,null),e(ge,Ai),e(ge,jn),e(jn,ji),g(o,Ba,m),g(o,I,m),L(jt,I,null),e(I,Ii),e(I,In),e(In,Ni),e(I,Di),e(I,Do),e(Do,So),e(So,Si),e(Do,Oi),e(I,Ri),e(I,z),e(z,Wi),e(z,Oo),e(Oo,Bi),e(z,Ui),e(z,Ro),e(Ro,Qi),e(z,Vi),e(z,Wo),e(Wo,Hi),e(z,Gi),e(z,Nn),e(Nn,Xi),e(z,Yi),e(z,Dn),e(Dn,Ji),e(z,Ki),e(z,Sn),e(Sn,Zi),e(z,el),e(z,On),e(On,tl),e(z,ol),e(z,Rn),e(Rn,nl),e(z,al),e(z,Wn),e(Wn,sl),e(z,rl),e(I,il),e(I,ae),L(It,ae,null),e(ae,ll),e(ae,M),e(M,dl),e(M,Bn),e(Bn,cl),e(M,ul),e(M,Nt),e(Nt,Un),e(Un,pl),e(Nt,ml),e(M,hl),e(M,Bo),e(Bo,fl),e(M,gl),e(M,Qn),e(Qn,_l),e(M,vl),e(M,Vn),e(Vn,yl),e(M,bl),e(M,Dt),e(Dt,Hn),e(Hn,Ll),e(Dt,kl),e(M,xl),e(M,Gn),e(Gn,wl),e(M,Tl),e(M,Uo),e(Uo,Ml),e(M,$l),e(M,Xn),e(Xn,zl),e(M,ql),e(M,Yn),e(Yn,El),e(M,Fl),e(M,Jn),e(Jn,Pl),e(M,Cl),e(M,Kn),e(Kn,Al),e(M,jl),e(M,Zn),e(Zn,Il),e(M,Nl),e(M,St),e(St,ea),e(ea,Dl),e(St,Sl),e(M,Ol),e(M,ta),e(ta,Rl),e(M,Wl),e(ae,Bl),e(ae,oa),e(oa,Ul),g(o,Ua,m),g(o,_e,m),e(_e,De),e(De,na),L(Ot,na,null),e(_e,Ql),e(_e,aa),e(aa,Vl),g(o,Qa,m),g(o,K,m),L(Rt,K,null),e(K,Hl),e(K,Wt),e(Wt,Gl),e(Wt,Bt),e(Bt,Xl),e(Wt,Yl),e(K,Jl),e(K,H),L(Ut,H,null),e(H,Kl),e(H,ve),e(ve,Zl),e(ve,Qo),e(Qo,ed),e(ve,td),e(ve,sa),e(sa,od),e(ve,nd),e(H,ad),L(Se,H,null),e(H,sd),L(Oe,H,null),g(o,Va,m),g(o,ye,m),e(ye,Re),e(Re,ra),L(Qt,ra,null),e(ye,rd),e(ye,ia),e(ia,id),g(o,Ha,m),g(o,R,m),L(Vt,R,null),e(R,ld),e(R,Ht),e(Ht,dd),e(Ht,Gt),e(Gt,cd),e(Ht,ud),e(R,pd),e(R,Xt),e(Xt,md),e(Xt,Yt),e(Yt,hd),e(Xt,fd),e(R,gd),e(R,G),L(Jt,G,null),e(G,_d),e(G,be),e(be,vd),e(be,Vo),e(Vo,yd),e(be,bd),e(be,la),e(la,Ld),e(be,kd),e(G,xd),L(We,G,null),e(G,wd),L(Be,G,null),g(o,Ga,m),g(o,Le,m),e(Le,Ue),e(Ue,da),L(Kt,da,null),e(Le,Td),e(Le,ca),e(ca,Md),g(o,Xa,m),g(o,W,m),L(Zt,W,null),e(W,$d),e(W,B),e(B,zd),e(B,eo),e(eo,qd),e(B,Ed),e(B,to),e(to,Fd),e(B,Pd),e(B,oo),e(oo,Cd),e(B,Ad),e(B,no),e(no,jd),e(B,Id),e(W,Nd),e(W,ao),e(ao,Dd),e(ao,so),e(so,Sd),e(ao,Od),e(W,Rd),e(W,X),L(ro,X,null),e(X,Wd),e(X,ke),e(ke,Bd),e(ke,Ho),e(Ho,Ud),e(ke,Qd),e(ke,ua),e(ua,Vd),e(ke,Hd),e(X,Gd),L(Qe,X,null),e(X,Xd),L(Ve,X,null),g(o,Ya,m),g(o,xe,m),e(xe,He),e(He,pa),L(io,pa,null),e(xe,Yd),e(xe,ma),e(ma,Jd),g(o,Ja,m),g(o,U,m),L(lo,U,null),e(U,Kd),e(U,Z),e(Z,Zd),e(Z,co),e(co,ec),e(Z,tc),e(Z,ha),e(ha,oc),e(Z,nc),e(Z,fa),e(fa,ac),e(Z,sc),e(U,rc),e(U,uo),e(uo,ic),e(uo,po),e(po,lc),e(uo,dc),e(U,cc),e(U,Y),L(mo,Y,null),e(Y,uc),e(Y,we),e(we,pc),e(we,Go),e(Go,mc),e(we,hc),e(we,ga),e(ga,fc),e(we,gc),e(Y,_c),L(Ge,Y,null),e(Y,vc),L(Xe,Y,null),Ka=!0},p(o,[m]){const ho={};m&2&&(ho.$$scope={dirty:m,ctx:o}),Ee.$set(ho);const _a={};m&2&&(_a.$$scope={dirty:m,ctx:o}),Pe.$set(_a);const va={};m&2&&(va.$$scope={dirty:m,ctx:o}),Se.$set(va);const ya={};m&2&&(ya.$$scope={dirty:m,ctx:o}),Oe.$set(ya);const fo={};m&2&&(fo.$$scope={dirty:m,ctx:o}),We.$set(fo);const ba={};m&2&&(ba.$$scope={dirty:m,ctx:o}),Be.$set(ba);const La={};m&2&&(La.$$scope={dirty:m,ctx:o}),Qe.$set(La);const ka={};m&2&&(ka.$$scope={dirty:m,ctx:o}),Ve.$set(ka);const ee={};m&2&&(ee.$$scope={dirty:m,ctx:o}),Ge.$set(ee);const xa={};m&2&&(xa.$$scope={dirty:m,ctx:o}),Xe.$set(xa)},i(o){Ka||(k(d.$$.fragment,o),k(lt.$$.fragment,o),k(_t.$$.fragment,o),k(vt.$$.fragment,o),k(Ee.$$.fragment,o),k(bt.$$.fragment,o),k(Lt.$$.fragment,o),k(xt.$$.fragment,o),k(Pe.$$.fragment,o),k(wt.$$.fragment,o),k(Tt.$$.fragment,o),k($t.$$.fragment,o),k(zt.$$.fragment,o),k(qt.$$.fragment,o),k(Et.$$.fragment,o),k(Ct.$$.fragment,o),k(At.$$.fragment,o),k(jt.$$.fragment,o),k(It.$$.fragment,o),k(Ot.$$.fragment,o),k(Rt.$$.fragment,o),k(Ut.$$.fragment,o),k(Se.$$.fragment,o),k(Oe.$$.fragment,o),k(Qt.$$.fragment,o),k(Vt.$$.fragment,o),k(Jt.$$.fragment,o),k(We.$$.fragment,o),k(Be.$$.fragment,o),k(Kt.$$.fragment,o),k(Zt.$$.fragment,o),k(ro.$$.fragment,o),k(Qe.$$.fragment,o),k(Ve.$$.fragment,o),k(io.$$.fragment,o),k(lo.$$.fragment,o),k(mo.$$.fragment,o),k(Ge.$$.fragment,o),k(Xe.$$.fragment,o),Ka=!0)},o(o){x(d.$$.fragment,o),x(lt.$$.fragment,o),x(_t.$$.fragment,o),x(vt.$$.fragment,o),x(Ee.$$.fragment,o),x(bt.$$.fragment,o),x(Lt.$$.fragment,o),x(xt.$$.fragment,o),x(Pe.$$.fragment,o),x(wt.$$.fragment,o),x(Tt.$$.fragment,o),x($t.$$.fragment,o),x(zt.$$.fragment,o),x(qt.$$.fragment,o),x(Et.$$.fragment,o),x(Ct.$$.fragment,o),x(At.$$.fragment,o),x(jt.$$.fragment,o),x(It.$$.fragment,o),x(Ot.$$.fragment,o),x(Rt.$$.fragment,o),x(Ut.$$.fragment,o),x(Se.$$.fragment,o),x(Oe.$$.fragment,o),x(Qt.$$.fragment,o),x(Vt.$$.fragment,o),x(Jt.$$.fragment,o),x(We.$$.fragment,o),x(Be.$$.fragment,o),x(Kt.$$.fragment,o),x(Zt.$$.fragment,o),x(ro.$$.fragment,o),x(Qe.$$.fragment,o),x(Ve.$$.fragment,o),x(io.$$.fragment,o),x(lo.$$.fragment,o),x(mo.$$.fragment,o),x(Ge.$$.fragment,o),x(Xe.$$.fragment,o),Ka=!1},d(o){t(p),o&&t(T),o&&t(_),w(d),o&&t(Ta),o&&t(le),w(lt),o&&t(Ma),o&&t(Q),o&&t($a),o&&t(vo),o&&t(za),o&&t(yo),o&&t(qa),o&&t(bo),o&&t(Ea),o&&t(V),o&&t(Fa),o&&t($e),o&&t(Pa),o&&t(ze),o&&t(Ca),o&&t(oe),o&&t(Aa),o&&t(ce),w(_t),o&&t(ja),o&&t(D),w(vt),w(Ee),o&&t(Ia),o&&t(me),w(bt),o&&t(Na),o&&t(S),w(Lt),w(xt),w(Pe),o&&t(Da),o&&t(he),w(wt),o&&t(Sa),o&&t(A),w(Tt),w($t),w(zt),o&&t(Oa),o&&t(fe),w(qt),o&&t(Ra),o&&t(O),w(Et),w(Ct),o&&t(Wa),o&&t(ge),w(At),o&&t(Ba),o&&t(I),w(jt),w(It),o&&t(Ua),o&&t(_e),w(Ot),o&&t(Qa),o&&t(K),w(Rt),w(Ut),w(Se),w(Oe),o&&t(Va),o&&t(ye),w(Qt),o&&t(Ha),o&&t(R),w(Vt),w(Jt),w(We),w(Be),o&&t(Ga),o&&t(Le),w(Kt),o&&t(Xa),o&&t(W),w(Zt),w(ro),w(Qe),w(Ve),o&&t(Ya),o&&t(xe),w(io),o&&t(Ja),o&&t(U),w(lo),w(mo),w(Ge),w(Xe)}}}const Gp={local:"layoutlmv3",sections:[{local:"overview",title:"Overview"},{local:"transformers.LayoutLMv3Config",title:"LayoutLMv3Config"},{local:"transformers.LayoutLMv3FeatureExtractor",title:"LayoutLMv3FeatureExtractor"},{local:"transformers.LayoutLMv3Tokenizer",title:"LayoutLMv3Tokenizer"},{local:"transformers.LayoutLMv3TokenizerFast",title:"LayoutLMv3TokenizerFast"},{local:"transformers.LayoutLMv3Processor",title:"LayoutLMv3Processor"},{local:"transformers.LayoutLMv3Model",title:"LayoutLMv3Model"},{local:"transformers.LayoutLMv3ForSequenceClassification",title:"LayoutLMv3ForSequenceClassification"},{local:"transformers.LayoutLMv3ForTokenClassification",title:"LayoutLMv3ForTokenClassification"},{local:"transformers.LayoutLMv3ForQuestionAnswering",title:"LayoutLMv3ForQuestionAnswering"}],title:"LayoutLMv3"};function Xp(q){return Ip(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class om extends Fp{constructor(p){super();Pp(this,p,Xp,Hp,Cp,{})}}export{om as default,Gp as metadata};
