import{S as ir,i as lr,s as dr,e as a,k as h,w as b,t as i,M as cr,c as s,d as o,m as f,a as r,x as M,h as l,b as c,N as pr,G as t,g as _,y as w,q as A,o as $,B as V,v as mr,L as yo}from"../../chunks/vendor-hf-doc-builder.js";import{T as $o}from"../../chunks/Tip-hf-doc-builder.js";import{D as Z}from"../../chunks/Docstring-hf-doc-builder.js";import{C as xo}from"../../chunks/CodeBlock-hf-doc-builder.js";import{I as He}from"../../chunks/IconCopyLink-hf-doc-builder.js";import{E as Vo}from"../../chunks/ExampleCodeBlock-hf-doc-builder.js";function hr(y){let d,E,u,m,v;return m=new xo({props:{code:`from transformers import VideoMAEConfig, VideoMAEModel

# Initializing a VideoMAE videomae-base style configuration
configuration = VideoMAEConfig()

# Randomly initializing a model from the configuration
model = VideoMAEModel(configuration)

# Accessing the model configuration
configuration = model.config`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> VideoMAEConfig, VideoMAEModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a VideoMAE videomae-base style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = VideoMAEConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Randomly initializing a model from the configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = VideoMAEModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`}}),{c(){d=a("p"),E=i("Example:"),u=h(),b(m.$$.fragment)},l(n){d=s(n,"P",{});var g=r(d);E=l(g,"Example:"),g.forEach(o),u=f(n),M(m.$$.fragment,n)},m(n,g){_(n,d,g),t(d,E),_(n,u,g),w(m,n,g),v=!0},p:yo,i(n){v||(A(m.$$.fragment,n),v=!0)},o(n){$(m.$$.fragment,n),v=!1},d(n){n&&o(d),n&&o(u),V(m,n)}}}function fr(y){let d,E;return{c(){d=a("p"),E=i("NumPy arrays are converted to PIL images when resizing, so the most efficient is to pass PIL images.")},l(u){d=s(u,"P",{});var m=r(d);E=l(m,"NumPy arrays are converted to PIL images when resizing, so the most efficient is to pass PIL images."),m.forEach(o)},m(u,m){_(u,d,m),t(d,E)},d(u){u&&o(d)}}}function ur(y){let d,E,u,m,v;return{c(){d=a("p"),E=i("Although the recipe for forward pass needs to be defined within this function, one should call the "),u=a("code"),m=i("Module"),v=i(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(n){d=s(n,"P",{});var g=r(d);E=l(g,"Although the recipe for forward pass needs to be defined within this function, one should call the "),u=s(g,"CODE",{});var x=r(u);m=l(x,"Module"),x.forEach(o),v=l(g,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),g.forEach(o)},m(n,g){_(n,d,g),t(d,E),t(d,u),t(u,m),t(d,v)},d(n){n&&o(d)}}}function gr(y){let d,E,u,m,v;return m=new xo({props:{code:`from decord import VideoReader, cpu
import numpy as np

from transformers import VideoMAEFeatureExtractor, VideoMAEModel
from huggingface_hub import hf_hub_download


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


# video clip consists of 300 frames (10 seconds at 30 FPS)
file_path = hf_hub_download(
    repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
)
vr = VideoReader(file_path, num_threads=1, ctx=cpu(0))

# sample 16 frames
vr.seek(0)
indices = sample_frame_indices(clip_len=16, frame_sample_rate=4, seg_len=len(vr))
buffer = vr.get_batch(indices).asnumpy()

# create a list of NumPy arrays
video = [buffer[i] for i in range(buffer.shape[0])]

feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base")
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")

# prepare video for the model
inputs = feature_extractor(video, return_tensors="pt")

# forward pass
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
list(last_hidden_states.shape)`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> decord <span class="hljs-keyword">import</span> VideoReader, cpu
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> numpy <span class="hljs-keyword">as</span> np

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> VideoMAEFeatureExtractor, VideoMAEModel
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> huggingface_hub <span class="hljs-keyword">import</span> hf_hub_download


<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">def</span> <span class="hljs-title function_">sample_frame_indices</span>(<span class="hljs-params">clip_len, frame_sample_rate, seg_len</span>):
<span class="hljs-meta">... </span>    converted_len = <span class="hljs-built_in">int</span>(clip_len * frame_sample_rate)
<span class="hljs-meta">... </span>    end_idx = np.random.randint(converted_len, seg_len)
<span class="hljs-meta">... </span>    start_idx = end_idx - converted_len
<span class="hljs-meta">... </span>    indices = np.linspace(start_idx, end_idx, num=clip_len)
<span class="hljs-meta">... </span>    indices = np.clip(indices, start_idx, end_idx - <span class="hljs-number">1</span>).astype(np.int64)
<span class="hljs-meta">... </span>    <span class="hljs-keyword">return</span> indices


<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># video clip consists of 300 frames (10 seconds at 30 FPS)</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>file_path = hf_hub_download(
<span class="hljs-meta">... </span>    repo_id=<span class="hljs-string">&quot;nielsr/video-demo&quot;</span>, filename=<span class="hljs-string">&quot;eating_spaghetti.mp4&quot;</span>, repo_type=<span class="hljs-string">&quot;dataset&quot;</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>vr = VideoReader(file_path, num_threads=<span class="hljs-number">1</span>, ctx=cpu(<span class="hljs-number">0</span>))

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># sample 16 frames</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>vr.seek(<span class="hljs-number">0</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>indices = sample_frame_indices(clip_len=<span class="hljs-number">16</span>, frame_sample_rate=<span class="hljs-number">4</span>, seg_len=<span class="hljs-built_in">len</span>(vr))
<span class="hljs-meta">&gt;&gt;&gt; </span>buffer = vr.get_batch(indices).asnumpy()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># create a list of NumPy arrays</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>video = [buffer[i] <span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> <span class="hljs-built_in">range</span>(buffer.shape[<span class="hljs-number">0</span>])]

<span class="hljs-meta">&gt;&gt;&gt; </span>feature_extractor = VideoMAEFeatureExtractor.from_pretrained(<span class="hljs-string">&quot;MCG-NJU/videomae-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = VideoMAEModel.from_pretrained(<span class="hljs-string">&quot;MCG-NJU/videomae-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># prepare video for the model</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = feature_extractor(video, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># forward pass</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(last_hidden_states.shape)
[<span class="hljs-number">1</span>, <span class="hljs-number">1568</span>, <span class="hljs-number">768</span>]`}}),{c(){d=a("p"),E=i("Examples:"),u=h(),b(m.$$.fragment)},l(n){d=s(n,"P",{});var g=r(d);E=l(g,"Examples:"),g.forEach(o),u=f(n),M(m.$$.fragment,n)},m(n,g){_(n,d,g),t(d,E),_(n,u,g),w(m,n,g),v=!0},p:yo,i(n){v||(A(m.$$.fragment,n),v=!0)},o(n){$(m.$$.fragment,n),v=!1},d(n){n&&o(d),n&&o(u),V(m,n)}}}function _r(y){let d,E,u,m,v;return{c(){d=a("p"),E=i("Although the recipe for forward pass needs to be defined within this function, one should call the "),u=a("code"),m=i("Module"),v=i(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(n){d=s(n,"P",{});var g=r(d);E=l(g,"Although the recipe for forward pass needs to be defined within this function, one should call the "),u=s(g,"CODE",{});var x=r(u);m=l(x,"Module"),x.forEach(o),v=l(g,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),g.forEach(o)},m(n,g){_(n,d,g),t(d,E),t(d,u),t(u,m),t(d,v)},d(n){n&&o(d)}}}function vr(y){let d,E,u,m,v;return m=new xo({props:{code:`from transformers import VideoMAEFeatureExtractor, VideoMAEForPreTraining
import numpy as np
import torch

num_frames = 16
video = list(np.random.randn(16, 3, 224, 224))

feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base")
model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base")

pixel_values = feature_extractor(video, return_tensors="pt").pixel_values

num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
seq_length = (num_frames // model.config.tubelet_size) * num_patches_per_frame
bool_masked_pos = torch.randint(0, 2, (1, seq_length)).bool()

outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
loss = outputs.loss`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> VideoMAEFeatureExtractor, VideoMAEForPreTraining
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> numpy <span class="hljs-keyword">as</span> np
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>num_frames = <span class="hljs-number">16</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>video = <span class="hljs-built_in">list</span>(np.random.randn(<span class="hljs-number">16</span>, <span class="hljs-number">3</span>, <span class="hljs-number">224</span>, <span class="hljs-number">224</span>))

<span class="hljs-meta">&gt;&gt;&gt; </span>feature_extractor = VideoMAEFeatureExtractor.from_pretrained(<span class="hljs-string">&quot;MCG-NJU/videomae-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = VideoMAEForPreTraining.from_pretrained(<span class="hljs-string">&quot;MCG-NJU/videomae-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>pixel_values = feature_extractor(video, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).pixel_values

<span class="hljs-meta">&gt;&gt;&gt; </span>num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** <span class="hljs-number">2</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>seq_length = (num_frames // model.config.tubelet_size) * num_patches_per_frame
<span class="hljs-meta">&gt;&gt;&gt; </span>bool_masked_pos = torch.randint(<span class="hljs-number">0</span>, <span class="hljs-number">2</span>, (<span class="hljs-number">1</span>, seq_length)).<span class="hljs-built_in">bool</span>()

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss`}}),{c(){d=a("p"),E=i("Examples:"),u=h(),b(m.$$.fragment)},l(n){d=s(n,"P",{});var g=r(d);E=l(g,"Examples:"),g.forEach(o),u=f(n),M(m.$$.fragment,n)},m(n,g){_(n,d,g),t(d,E),_(n,u,g),w(m,n,g),v=!0},p:yo,i(n){v||(A(m.$$.fragment,n),v=!0)},o(n){$(m.$$.fragment,n),v=!1},d(n){n&&o(d),n&&o(u),V(m,n)}}}function Er(y){let d,E,u,m,v;return{c(){d=a("p"),E=i("Although the recipe for forward pass needs to be defined within this function, one should call the "),u=a("code"),m=i("Module"),v=i(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(n){d=s(n,"P",{});var g=r(d);E=l(g,"Although the recipe for forward pass needs to be defined within this function, one should call the "),u=s(g,"CODE",{});var x=r(u);m=l(x,"Module"),x.forEach(o),v=l(g,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),g.forEach(o)},m(n,g){_(n,d,g),t(d,E),t(d,u),t(u,m),t(d,v)},d(n){n&&o(d)}}}function br(y){let d,E,u,m,v;return m=new xo({props:{code:`from decord import VideoReader, cpu
import torch

from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification
from huggingface_hub import hf_hub_download


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


# video clip consists of 300 frames (10 seconds at 30 FPS)
file_path = hf_hub_download(
    repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
)
vr = VideoReader(file_path, num_threads=1, ctx=cpu(0))

# sample 16 frames
vr.seek(0)
indices = sample_frame_indices(clip_len=16, frame_sample_rate=4, seg_len=len(vr))
buffer = vr.get_batch(indices).asnumpy()

# create a list of NumPy arrays
video = [buffer[i] for i in range(buffer.shape[0])]

feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

inputs = feature_extractor(video, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# model predicts one of the 400 Kinetics-400 classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> decord <span class="hljs-keyword">import</span> VideoReader, cpu
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> VideoMAEFeatureExtractor, VideoMAEForVideoClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> huggingface_hub <span class="hljs-keyword">import</span> hf_hub_download


<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">def</span> <span class="hljs-title function_">sample_frame_indices</span>(<span class="hljs-params">clip_len, frame_sample_rate, seg_len</span>):
<span class="hljs-meta">... </span>    converted_len = <span class="hljs-built_in">int</span>(clip_len * frame_sample_rate)
<span class="hljs-meta">... </span>    end_idx = np.random.randint(converted_len, seg_len)
<span class="hljs-meta">... </span>    start_idx = end_idx - converted_len
<span class="hljs-meta">... </span>    indices = np.linspace(start_idx, end_idx, num=clip_len)
<span class="hljs-meta">... </span>    indices = np.clip(indices, start_idx, end_idx - <span class="hljs-number">1</span>).astype(np.int64)
<span class="hljs-meta">... </span>    <span class="hljs-keyword">return</span> indices


<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># video clip consists of 300 frames (10 seconds at 30 FPS)</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>file_path = hf_hub_download(
<span class="hljs-meta">... </span>    repo_id=<span class="hljs-string">&quot;nielsr/video-demo&quot;</span>, filename=<span class="hljs-string">&quot;eating_spaghetti.mp4&quot;</span>, repo_type=<span class="hljs-string">&quot;dataset&quot;</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>vr = VideoReader(file_path, num_threads=<span class="hljs-number">1</span>, ctx=cpu(<span class="hljs-number">0</span>))

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># sample 16 frames</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>vr.seek(<span class="hljs-number">0</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>indices = sample_frame_indices(clip_len=<span class="hljs-number">16</span>, frame_sample_rate=<span class="hljs-number">4</span>, seg_len=<span class="hljs-built_in">len</span>(vr))
<span class="hljs-meta">&gt;&gt;&gt; </span>buffer = vr.get_batch(indices).asnumpy()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># create a list of NumPy arrays</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>video = [buffer[i] <span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> <span class="hljs-built_in">range</span>(buffer.shape[<span class="hljs-number">0</span>])]

<span class="hljs-meta">&gt;&gt;&gt; </span>feature_extractor = VideoMAEFeatureExtractor.from_pretrained(<span class="hljs-string">&quot;MCG-NJU/videomae-base-finetuned-kinetics&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = VideoMAEForVideoClassification.from_pretrained(<span class="hljs-string">&quot;MCG-NJU/videomae-base-finetuned-kinetics&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = feature_extractor(video, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    outputs = model(**inputs)
<span class="hljs-meta">... </span>    logits = outputs.logits

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># model predicts one of the 400 Kinetics-400 classes</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_label = logits.argmax(-<span class="hljs-number">1</span>).item()
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(model.config.id2label[predicted_label])
eating spaghetti`}}),{c(){d=a("p"),E=i("Examples:"),u=h(),b(m.$$.fragment)},l(n){d=s(n,"P",{});var g=r(d);E=l(g,"Examples:"),g.forEach(o),u=f(n),M(m.$$.fragment,n)},m(n,g){_(n,d,g),t(d,E),_(n,u,g),w(m,n,g),v=!0},p:yo,i(n){v||(A(m.$$.fragment,n),v=!0)},o(n){$(m.$$.fragment,n),v=!1},d(n){n&&o(d),n&&o(u),V(m,n)}}}function Mr(y){let d,E,u,m,v,n,g,x,To,Jt,L,X,Mt,Ee,jo,wt,ko,Bt,N,Fo,be,Po,Co,Ke,zo,Io,Gt,Ze,No,Ht,Xe,At,qo,Kt,Ye,So,Zt,Y,Me,Lo,Qe,Do,Oo,Uo,et,tt,Ro,Wo,Xt,Q,vs,Yt,ee,Jo,ot,Bo,Go,Qt,q,Ho,we,Ko,Zo,Ae,Xo,Yo,eo,D,te,$t,$e,Qo,Vt,ea,to,T,Ve,ta,O,oa,at,aa,sa,ye,ra,na,ia,U,la,st,da,ca,rt,pa,ma,ha,oe,oo,R,ae,yt,xe,fa,xt,ua,ao,j,Te,ga,Tt,_a,va,je,Ea,nt,ba,Ma,wa,S,ke,Aa,jt,$a,Va,se,so,W,re,kt,Fe,ya,Ft,xa,ro,C,Pe,Ta,Ce,ja,ze,ka,Fa,Pa,k,Ie,Ca,J,za,it,Ia,Na,Pt,qa,Sa,La,ne,Da,ie,no,B,le,Ct,Ne,Oa,zt,Ua,io,z,qe,Ra,Se,Wa,Le,Ja,Ba,Ga,F,De,Ha,G,Ka,lt,Za,Xa,It,Ya,Qa,es,de,ts,ce,lo,H,pe,Nt,Oe,os,qt,as,co,I,Ue,ss,Re,rs,We,ns,is,ls,P,Je,ds,K,cs,dt,ps,ms,St,hs,fs,us,me,gs,he,po;return n=new He({}),Ee=new He({}),$e=new He({}),Ve=new Z({props:{name:"class transformers.VideoMAEConfig",anchor:"transformers.VideoMAEConfig",parameters:[{name:"image_size",val:" = 224"},{name:"patch_size",val:" = 16"},{name:"num_channels",val:" = 3"},{name:"num_frames",val:" = 16"},{name:"tubelet_size",val:" = 2"},{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.0"},{name:"attention_probs_dropout_prob",val:" = 0.0"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"qkv_bias",val:" = True"},{name:"use_mean_pooling",val:" = True"},{name:"decoder_num_attention_heads",val:" = 6"},{name:"decoder_hidden_size",val:" = 384"},{name:"decoder_num_hidden_layers",val:" = 4"},{name:"decoder_intermediate_size",val:" = 1536"},{name:"norm_pix_loss",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.VideoMAEConfig.image_size",description:`<strong>image_size</strong> (<code>int</code>, <em>optional</em>, defaults to 224) &#x2014;
The size (resolution) of each image.`,name:"image_size"},{anchor:"transformers.VideoMAEConfig.patch_size",description:`<strong>patch_size</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
The size (resolution) of each patch.`,name:"patch_size"},{anchor:"transformers.VideoMAEConfig.num_channels",description:`<strong>num_channels</strong> (<code>int</code>, <em>optional</em>, defaults to 3) &#x2014;
The number of input channels.`,name:"num_channels"},{anchor:"transformers.VideoMAEConfig.num_frames",description:`<strong>num_frames</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
The number of frames in each video.`,name:"num_frames"},{anchor:"transformers.VideoMAEConfig.tubelet_size",description:`<strong>tubelet_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The number of tubelets.`,name:"tubelet_size"},{anchor:"transformers.VideoMAEConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.VideoMAEConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.VideoMAEConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.VideoMAEConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (i.e., feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.VideoMAEConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;selu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.VideoMAEConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.VideoMAEConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.VideoMAEConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.VideoMAEConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.VideoMAEConfig.qkv_bias",description:`<strong>qkv_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to add a bias to the queries, keys and values.`,name:"qkv_bias"},{anchor:"transformers.VideoMAEConfig.use_mean_pooling",description:`<strong>use_mean_pooling</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to mean pool the final hidden states instead of using the final hidden state of the [CLS] token.`,name:"use_mean_pooling"},{anchor:"transformers.VideoMAEConfig.decoder_num_attention_heads",description:`<strong>decoder_num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 6) &#x2014;
Number of attention heads for each attention layer in the decoder.`,name:"decoder_num_attention_heads"},{anchor:"transformers.VideoMAEConfig.decoder_hidden_size",description:`<strong>decoder_hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 384) &#x2014;
Dimensionality of the decoder.`,name:"decoder_hidden_size"},{anchor:"transformers.VideoMAEConfig.decoder_num_hidden_layers",description:`<strong>decoder_num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 4) &#x2014;
Number of hidden layers in the decoder.`,name:"decoder_num_hidden_layers"},{anchor:"transformers.VideoMAEConfig.decoder_intermediate_size",description:`<strong>decoder_intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 1536) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (i.e., feed-forward) layer in the decoder.`,name:"decoder_intermediate_size"},{anchor:"transformers.VideoMAEConfig.norm_pix_loss",description:`<strong>norm_pix_loss</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to normalize the target patch pixels.`,name:"norm_pix_loss"}],source:"https://github.com/huggingface/transformers/blob/vr_18658/src/transformers/models/videomae/configuration_videomae.py#L28"}}),oe=new Vo({props:{anchor:"transformers.VideoMAEConfig.example",$$slots:{default:[hr]},$$scope:{ctx:y}}}),xe=new He({}),Te=new Z({props:{name:"class transformers.VideoMAEFeatureExtractor",anchor:"transformers.VideoMAEFeatureExtractor",parameters:[{name:"do_resize",val:" = True"},{name:"size",val:" = 224"},{name:"resample",val:" = <Resampling.BILINEAR: 2>"},{name:"do_center_crop",val:" = True"},{name:"do_normalize",val:" = True"},{name:"image_mean",val:" = None"},{name:"image_std",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.VideoMAEFeatureExtractor.do_resize",description:`<strong>do_resize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to resize the shorter edge of the input to a certain <code>size</code>.`,name:"do_resize"},{anchor:"transformers.VideoMAEFeatureExtractor.size",description:`<strong>size</strong> (<code>int</code>, <em>optional</em>, defaults to 224) &#x2014;
Resize the shorter edge of the input to the given size. Only has an effect if <code>do_resize</code> is set to <code>True</code>.`,name:"size"},{anchor:"transformers.VideoMAEFeatureExtractor.resample",description:`<strong>resample</strong> (<code>int</code>, <em>optional</em>, defaults to <code>PIL.Image.BILINEAR</code>) &#x2014;
An optional resampling filter. This can be one of <code>PIL.Image.NEAREST</code>, <code>PIL.Image.BOX</code>,
<code>PIL.Image.BILINEAR</code>, <code>PIL.Image.HAMMING</code>, <code>PIL.Image.BICUBIC</code> or <code>PIL.Image.LANCZOS</code>. Only has an effect
if <code>do_resize</code> is set to <code>True</code>.`,name:"resample"},{anchor:"transformers.VideoMAEFeatureExtractor.do_center_crop",description:`<strong>do_center_crop</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to center crop the input to a certain <code>size</code>.`,name:"do_center_crop"},{anchor:"transformers.VideoMAEFeatureExtractor.do_normalize",description:`<strong>do_normalize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to normalize the input with mean and standard deviation.`,name:"do_normalize"},{anchor:"transformers.VideoMAEFeatureExtractor.image_mean",description:`<strong>image_mean</strong> (<code>List[int]</code>, defaults to <code>[0.485, 0.456, 0.406]</code>) &#x2014;
The sequence of means for each channel, to be used when normalizing images.`,name:"image_mean"},{anchor:"transformers.VideoMAEFeatureExtractor.image_std",description:`<strong>image_std</strong> (<code>List[int]</code>, defaults to <code>[0.229, 0.224, 0.225]</code>) &#x2014;
The sequence of standard deviations for each channel, to be used when normalizing images.`,name:"image_std"}],source:"https://github.com/huggingface/transformers/blob/vr_18658/src/transformers/models/videomae/feature_extraction_videomae.py#L30"}}),ke=new Z({props:{name:"__call__",anchor:"transformers.VideoMAEFeatureExtractor.__call__",parameters:[{name:"videos",val:": typing.Union[PIL.Image.Image, numpy.ndarray, ForwardRef('torch.Tensor'), typing.List[PIL.Image.Image], typing.List[numpy.ndarray], typing.List[ForwardRef('torch.Tensor')]]"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.VideoMAEFeatureExtractor.__call__.videos",description:`<strong>videos</strong> (<code>List[PIL.Image.Image]</code>, <code>List[np.ndarray]</code>, <code>List[torch.Tensor]</code>, <code>List[List[PIL.Image.Image]]</code>, <code>List[List[np.ndarrray]]</code>, &#x2014;
<code>List[List[torch.Tensor]]</code>): The video or batch of videos to be prepared. Each video should be a list
of frames, which can be either PIL images or NumPy arrays. In case of NumPy arrays/PyTorch tensors,
each frame should be of shape (H, W, C), where H and W are frame height and width, and C is a number of
channels.`,name:"videos"},{anchor:"transformers.VideoMAEFeatureExtractor.__call__.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/pr_18658/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>, defaults to <code>&apos;np&apos;</code>) &#x2014;
If set, will return tensors of a particular framework. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return NumPy <code>np.ndarray</code> objects.</li>
<li><code>&apos;jax&apos;</code>: Return JAX <code>jnp.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"}],source:"https://github.com/huggingface/transformers/blob/vr_18658/src/transformers/models/videomae/feature_extraction_videomae.py#L100",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18658/en/main_classes/feature_extractor#transformers.BatchFeature"
>BatchFeature</a> with the following fields:</p>
<ul>
<li><strong>pixel_values</strong> \u2014 Pixel values to be fed to a model, of shape (batch_size, num_channels, num_frames,
height, width).</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18658/en/main_classes/feature_extractor#transformers.BatchFeature"
>BatchFeature</a></p>
`}}),se=new $o({props:{warning:!0,$$slots:{default:[fr]},$$scope:{ctx:y}}}),Fe=new He({}),Pe=new Z({props:{name:"class transformers.VideoMAEModel",anchor:"transformers.VideoMAEModel",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.VideoMAEModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18658/en/model_doc/videomae#transformers.VideoMAEConfig">VideoMAEConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18658/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18658/src/transformers/models/videomae/modeling_videomae.py#L537"}}),Ie=new Z({props:{name:"forward",anchor:"transformers.VideoMAEModel.forward",parameters:[{name:"pixel_values",val:""},{name:"bool_masked_pos",val:" = None"},{name:"head_mask",val:" = None"},{name:"output_attentions",val:" = None"},{name:"output_hidden_states",val:" = None"},{name:"return_dict",val:" = None"}],parametersDescription:[{anchor:"transformers.VideoMAEModel.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_frames, num_channels, height, width)</code>) &#x2014;
Pixel values. Pixel values can be obtained using <a href="/docs/transformers/pr_18658/en/model_doc/videomae#transformers.VideoMAEFeatureExtractor">VideoMAEFeatureExtractor</a>. See
<a href="/docs/transformers/pr_18658/en/model_doc/videomae#transformers.VideoMAEFeatureExtractor.__call__">VideoMAEFeatureExtractor.<strong>call</strong>()</a> for details.`,name:"pixel_values"},{anchor:"transformers.VideoMAEModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.VideoMAEModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.VideoMAEModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.VideoMAEModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18658/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18658/src/transformers/models/videomae/modeling_videomae.py#L564",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18658/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput"
>transformers.modeling_outputs.BaseModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18658/en/model_doc/videomae#transformers.VideoMAEConfig"
>VideoMAEConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18658/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput"
>transformers.modeling_outputs.BaseModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ne=new $o({props:{$$slots:{default:[ur]},$$scope:{ctx:y}}}),ie=new Vo({props:{anchor:"transformers.VideoMAEModel.forward.example",$$slots:{default:[gr]},$$scope:{ctx:y}}}),Ne=new He({}),qe=new Z({props:{name:"class transformers.VideoMAEForPreTraining",anchor:"transformers.VideoMAEForPreTraining",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.VideoMAEForPreTraining.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18658/en/model_doc/videomae#transformers.VideoMAEConfig">VideoMAEConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18658/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18658/src/transformers/models/videomae/modeling_videomae.py#L737"}}),De=new Z({props:{name:"forward",anchor:"transformers.VideoMAEForPreTraining.forward",parameters:[{name:"pixel_values",val:""},{name:"bool_masked_pos",val:""},{name:"head_mask",val:" = None"},{name:"output_attentions",val:" = None"},{name:"output_hidden_states",val:" = None"},{name:"return_dict",val:" = None"}],parametersDescription:[{anchor:"transformers.VideoMAEForPreTraining.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_frames, num_channels, height, width)</code>) &#x2014;
Pixel values. Pixel values can be obtained using <a href="/docs/transformers/pr_18658/en/model_doc/videomae#transformers.VideoMAEFeatureExtractor">VideoMAEFeatureExtractor</a>. See
<a href="/docs/transformers/pr_18658/en/model_doc/videomae#transformers.VideoMAEFeatureExtractor.__call__">VideoMAEFeatureExtractor.<strong>call</strong>()</a> for details.`,name:"pixel_values"},{anchor:"transformers.VideoMAEForPreTraining.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.VideoMAEForPreTraining.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.VideoMAEForPreTraining.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.VideoMAEForPreTraining.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18658/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18658/src/transformers/models/videomae/modeling_videomae.py#L755",returnDescription:`
<p>A <code>transformers.models.videomae.modeling_videomae.VideoMAEForPreTrainingOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18658/en/model_doc/videomae#transformers.VideoMAEConfig"
>VideoMAEConfig</a>) and inputs.</p>
<ul>
<li><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>) \u2014 Pixel reconstruction loss.</li>
<li><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, patch_size ** 2 * num_channels)</code>) \u2014 Pixel reconstruction logits.</li>
<li><strong>hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, sequence_length, hidden_size)</code>. Hidden-states of the model at the output of each layer
plus the initial embedding outputs.</li>
<li><strong>attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>. Attentions weights after the attention softmax, used to compute the weighted average in
the self-attention heads.</li>
</ul>
`,returnType:`
<p><code>transformers.models.videomae.modeling_videomae.VideoMAEForPreTrainingOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),de=new $o({props:{$$slots:{default:[_r]},$$scope:{ctx:y}}}),ce=new Vo({props:{anchor:"transformers.VideoMAEForPreTraining.forward.example",$$slots:{default:[vr]},$$scope:{ctx:y}}}),Oe=new He({}),Ue=new Z({props:{name:"class transformers.VideoMAEForVideoClassification",anchor:"transformers.VideoMAEForVideoClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.VideoMAEForVideoClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18658/en/model_doc/videomae#transformers.VideoMAEConfig">VideoMAEConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18658/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18658/src/transformers/models/videomae/modeling_videomae.py#L908"}}),Je=new Z({props:{name:"forward",anchor:"transformers.VideoMAEForVideoClassification.forward",parameters:[{name:"pixel_values",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.VideoMAEForVideoClassification.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_frames, num_channels, height, width)</code>) &#x2014;
Pixel values. Pixel values can be obtained using <a href="/docs/transformers/pr_18658/en/model_doc/videomae#transformers.VideoMAEFeatureExtractor">VideoMAEFeatureExtractor</a>. See
<a href="/docs/transformers/pr_18658/en/model_doc/videomae#transformers.VideoMAEFeatureExtractor.__call__">VideoMAEFeatureExtractor.<strong>call</strong>()</a> for details.`,name:"pixel_values"},{anchor:"transformers.VideoMAEForVideoClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.VideoMAEForVideoClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.VideoMAEForVideoClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.VideoMAEForVideoClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18658/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.VideoMAEForVideoClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the image classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18658/src/transformers/models/videomae/modeling_videomae.py#L922",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18658/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput"
>transformers.modeling_outputs.ImageClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18658/en/model_doc/videomae#transformers.VideoMAEConfig"
>VideoMAEConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) \u2014 Classification (or regression if config.num_labels==1) loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.num_labels)</code>) \u2014 Classification (or regression if config.num_labels==1) scores (before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each stage) of shape <code>(batch_size, sequence_length, hidden_size)</code>. Hidden-states
(also called feature maps) of the model at the output of each stage.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, patch_size, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18658/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput"
>transformers.modeling_outputs.ImageClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),me=new $o({props:{$$slots:{default:[Er]},$$scope:{ctx:y}}}),he=new Vo({props:{anchor:"transformers.VideoMAEForVideoClassification.forward.example",$$slots:{default:[br]},$$scope:{ctx:y}}}),{c(){d=a("meta"),E=h(),u=a("h1"),m=a("a"),v=a("span"),b(n.$$.fragment),g=h(),x=a("span"),To=i("VideoMAE"),Jt=h(),L=a("h2"),X=a("a"),Mt=a("span"),b(Ee.$$.fragment),jo=h(),wt=a("span"),ko=i("Overview"),Bt=h(),N=a("p"),Fo=i("The VideoMAE model was proposed in "),be=a("a"),Po=i("VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training"),Co=i(` by Zhan Tong, Yibing Song, Jue Wang, Limin Wang.
VideoMAE extends masked auto encoders (`),Ke=a("a"),zo=i("MAE"),Io=i(") to video, claiming state-of-the-art performance on several video classification benchmarks."),Gt=h(),Ze=a("p"),No=i("The abstract from the paper is the following:"),Ht=h(),Xe=a("p"),At=a("em"),qo=i("Pre-training video transformers on extra large-scale datasets is generally required to achieve premier performance on relatively small datasets. In this paper, we show that video masked autoencoders (VideoMAE) are data-efficient learners for self-supervised video pre-training (SSVP). We are inspired by the recent ImageMAE and propose customized video tube masking and reconstruction. These simple designs turn out to be effective for overcoming information leakage caused by the temporal correlation during video reconstruction. We obtain three important findings on SSVP: (1) An extremely high proportion of masking ratio (i.e., 90% to 95%) still yields favorable performance of VideoMAE. The temporally redundant video content enables higher masking ratio than that of images. (2) VideoMAE achieves impressive results on very small datasets (i.e., around 3k-4k videos) without using any extra data. This is partially ascribed to the challenging task of video reconstruction to enforce high-level structure learning. (3) VideoMAE shows that data quality is more important than data quantity for SSVP. Domain shift between pre-training and target datasets are important issues in SSVP. Notably, our VideoMAE with the vanilla ViT backbone can achieve 83.9% on Kinects-400, 75.3% on Something-Something V2, 90.8% on UCF101, and 61.1% on HMDB51 without using any extra data."),Kt=h(),Ye=a("p"),So=i("Tips:"),Zt=h(),Y=a("ul"),Me=a("li"),Lo=i("One can use "),Qe=a("a"),Do=i("VideoMAEFeatureExtractor"),Oo=i(" to prepare videos for the model. It will resize + normalize all frames of a video for you."),Uo=h(),et=a("li"),tt=a("a"),Ro=i("VideoMAEForPreTraining"),Wo=i(" includes the decoder on top for self-supervised pre-training."),Xt=h(),Q=a("img"),Yt=h(),ee=a("small"),Jo=i("VideoMAE pre-training. Taken from the "),ot=a("a"),Bo=i("original paper"),Go=i("."),Qt=h(),q=a("p"),Ho=i("This model was contributed by "),we=a("a"),Ko=i("nielsr"),Zo=i(`.
The original code can be found `),Ae=a("a"),Xo=i("here"),Yo=i("."),eo=h(),D=a("h2"),te=a("a"),$t=a("span"),b($e.$$.fragment),Qo=h(),Vt=a("span"),ea=i("VideoMAEConfig"),to=h(),T=a("div"),b(Ve.$$.fragment),ta=h(),O=a("p"),oa=i("This is the configuration class to store the configuration of a "),at=a("a"),aa=i("VideoMAEModel"),sa=i(`. It is used to instantiate a
VideoMAE model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the VideoMAE
`),ye=a("a"),ra=i("MCG-NJU/videomae-base"),na=i(" architecture."),ia=h(),U=a("p"),la=i("Configuration objects inherit from "),st=a("a"),da=i("PretrainedConfig"),ca=i(` and can be used to control the model outputs. Read the
documentation from `),rt=a("a"),pa=i("PretrainedConfig"),ma=i(" for more information."),ha=h(),b(oe.$$.fragment),oo=h(),R=a("h2"),ae=a("a"),yt=a("span"),b(xe.$$.fragment),fa=h(),xt=a("span"),ua=i("VideoMAEFeatureExtractor"),ao=h(),j=a("div"),b(Te.$$.fragment),ga=h(),Tt=a("p"),_a=i("Constructs a VideoMAE feature extractor. This feature extractor can be used to prepare videos for the model."),va=h(),je=a("p"),Ea=i("This feature extractor inherits from "),nt=a("a"),ba=i("FeatureExtractionMixin"),Ma=i(` which contains most of the main methods. Users
should refer to this superclass for more information regarding those methods.`),wa=h(),S=a("div"),b(ke.$$.fragment),Aa=h(),jt=a("p"),$a=i("Main method to prepare for the model one or several video(s)."),Va=h(),b(se.$$.fragment),so=h(),W=a("h2"),re=a("a"),kt=a("span"),b(Fe.$$.fragment),ya=h(),Ft=a("span"),xa=i("VideoMAEModel"),ro=h(),C=a("div"),b(Pe.$$.fragment),Ta=h(),Ce=a("p"),ja=i(`The bare VideoMAE Model transformer outputting raw hidden-states without any specific head on top.
This model is a PyTorch `),ze=a("a"),ka=i("torch.nn.Module"),Fa=i(` subclass. Use it
as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),Pa=h(),k=a("div"),b(Ie.$$.fragment),Ca=h(),J=a("p"),za=i("The "),it=a("a"),Ia=i("VideoMAEModel"),Na=i(" forward method, overrides the "),Pt=a("code"),qa=i("__call__"),Sa=i(" special method."),La=h(),b(ne.$$.fragment),Da=h(),b(ie.$$.fragment),no=h(),B=a("h2"),le=a("a"),Ct=a("span"),b(Ne.$$.fragment),Oa=h(),zt=a("span"),Ua=i("VideoMAEForPreTraining"),io=h(),z=a("div"),b(qe.$$.fragment),Ra=h(),Se=a("p"),Wa=i(`The VideoMAE Model transformer with the decoder on top for self-supervised pre-training.
This model is a PyTorch `),Le=a("a"),Ja=i("torch.nn.Module"),Ba=i(` subclass. Use it
as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),Ga=h(),F=a("div"),b(De.$$.fragment),Ha=h(),G=a("p"),Ka=i("The "),lt=a("a"),Za=i("VideoMAEForPreTraining"),Xa=i(" forward method, overrides the "),It=a("code"),Ya=i("__call__"),Qa=i(" special method."),es=h(),b(de.$$.fragment),ts=h(),b(ce.$$.fragment),lo=h(),H=a("h2"),pe=a("a"),Nt=a("span"),b(Oe.$$.fragment),os=h(),qt=a("span"),as=i("VideoMAEForVideoClassification"),co=h(),I=a("div"),b(Ue.$$.fragment),ss=h(),Re=a("p"),rs=i(`VideoMAE Model transformer with a video classification head on top (a linear layer on top of the final hidden state of
the [CLS] token) e.g. for ImageNet.
This model is a PyTorch `),We=a("a"),ns=i("torch.nn.Module"),is=i(` subclass. Use it
as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),ls=h(),P=a("div"),b(Je.$$.fragment),ds=h(),K=a("p"),cs=i("The "),dt=a("a"),ps=i("VideoMAEForVideoClassification"),ms=i(" forward method, overrides the "),St=a("code"),hs=i("__call__"),fs=i(" special method."),us=h(),b(me.$$.fragment),gs=h(),b(he.$$.fragment),this.h()},l(e){const p=cr('[data-svelte="svelte-1phssyn"]',document.head);d=s(p,"META",{name:!0,content:!0}),p.forEach(o),E=f(e),u=s(e,"H1",{class:!0});var Be=r(u);m=s(Be,"A",{id:!0,class:!0,href:!0});var Lt=r(m);v=s(Lt,"SPAN",{});var Dt=r(v);M(n.$$.fragment,Dt),Dt.forEach(o),Lt.forEach(o),g=f(Be),x=s(Be,"SPAN",{});var Ot=r(x);To=l(Ot,"VideoMAE"),Ot.forEach(o),Be.forEach(o),Jt=f(e),L=s(e,"H2",{class:!0});var Ge=r(L);X=s(Ge,"A",{id:!0,class:!0,href:!0});var Ut=r(X);Mt=s(Ut,"SPAN",{});var Rt=r(Mt);M(Ee.$$.fragment,Rt),Rt.forEach(o),Ut.forEach(o),jo=f(Ge),wt=s(Ge,"SPAN",{});var Wt=r(wt);ko=l(Wt,"Overview"),Wt.forEach(o),Ge.forEach(o),Bt=f(e),N=s(e,"P",{});var ct=r(N);Fo=l(ct,"The VideoMAE model was proposed in "),be=s(ct,"A",{href:!0,rel:!0});var Es=r(be);Po=l(Es,"VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training"),Es.forEach(o),Co=l(ct,` by Zhan Tong, Yibing Song, Jue Wang, Limin Wang.
VideoMAE extends masked auto encoders (`),Ke=s(ct,"A",{href:!0});var bs=r(Ke);zo=l(bs,"MAE"),bs.forEach(o),Io=l(ct,") to video, claiming state-of-the-art performance on several video classification benchmarks."),ct.forEach(o),Gt=f(e),Ze=s(e,"P",{});var Ms=r(Ze);No=l(Ms,"The abstract from the paper is the following:"),Ms.forEach(o),Ht=f(e),Xe=s(e,"P",{});var ws=r(Xe);At=s(ws,"EM",{});var As=r(At);qo=l(As,"Pre-training video transformers on extra large-scale datasets is generally required to achieve premier performance on relatively small datasets. In this paper, we show that video masked autoencoders (VideoMAE) are data-efficient learners for self-supervised video pre-training (SSVP). We are inspired by the recent ImageMAE and propose customized video tube masking and reconstruction. These simple designs turn out to be effective for overcoming information leakage caused by the temporal correlation during video reconstruction. We obtain three important findings on SSVP: (1) An extremely high proportion of masking ratio (i.e., 90% to 95%) still yields favorable performance of VideoMAE. The temporally redundant video content enables higher masking ratio than that of images. (2) VideoMAE achieves impressive results on very small datasets (i.e., around 3k-4k videos) without using any extra data. This is partially ascribed to the challenging task of video reconstruction to enforce high-level structure learning. (3) VideoMAE shows that data quality is more important than data quantity for SSVP. Domain shift between pre-training and target datasets are important issues in SSVP. Notably, our VideoMAE with the vanilla ViT backbone can achieve 83.9% on Kinects-400, 75.3% on Something-Something V2, 90.8% on UCF101, and 61.1% on HMDB51 without using any extra data."),As.forEach(o),ws.forEach(o),Kt=f(e),Ye=s(e,"P",{});var $s=r(Ye);So=l($s,"Tips:"),$s.forEach(o),Zt=f(e),Y=s(e,"UL",{});var mo=r(Y);Me=s(mo,"LI",{});var ho=r(Me);Lo=l(ho,"One can use "),Qe=s(ho,"A",{href:!0});var Vs=r(Qe);Do=l(Vs,"VideoMAEFeatureExtractor"),Vs.forEach(o),Oo=l(ho," to prepare videos for the model. It will resize + normalize all frames of a video for you."),ho.forEach(o),Uo=f(mo),et=s(mo,"LI",{});var _s=r(et);tt=s(_s,"A",{href:!0});var ys=r(tt);Ro=l(ys,"VideoMAEForPreTraining"),ys.forEach(o),Wo=l(_s," includes the decoder on top for self-supervised pre-training."),_s.forEach(o),mo.forEach(o),Xt=f(e),Q=s(e,"IMG",{src:!0,alt:!0,width:!0}),Yt=f(e),ee=s(e,"SMALL",{});var fo=r(ee);Jo=l(fo,"VideoMAE pre-training. Taken from the "),ot=s(fo,"A",{href:!0});var xs=r(ot);Bo=l(xs,"original paper"),xs.forEach(o),Go=l(fo,"."),fo.forEach(o),Qt=f(e),q=s(e,"P",{});var pt=r(q);Ho=l(pt,"This model was contributed by "),we=s(pt,"A",{href:!0,rel:!0});var Ts=r(we);Ko=l(Ts,"nielsr"),Ts.forEach(o),Zo=l(pt,`.
The original code can be found `),Ae=s(pt,"A",{href:!0,rel:!0});var js=r(Ae);Xo=l(js,"here"),js.forEach(o),Yo=l(pt,"."),pt.forEach(o),eo=f(e),D=s(e,"H2",{class:!0});var uo=r(D);te=s(uo,"A",{id:!0,class:!0,href:!0});var ks=r(te);$t=s(ks,"SPAN",{});var Fs=r($t);M($e.$$.fragment,Fs),Fs.forEach(o),ks.forEach(o),Qo=f(uo),Vt=s(uo,"SPAN",{});var Ps=r(Vt);ea=l(Ps,"VideoMAEConfig"),Ps.forEach(o),uo.forEach(o),to=f(e),T=s(e,"DIV",{class:!0});var fe=r(T);M(Ve.$$.fragment,fe),ta=f(fe),O=s(fe,"P",{});var mt=r(O);oa=l(mt,"This is the configuration class to store the configuration of a "),at=s(mt,"A",{href:!0});var Cs=r(at);aa=l(Cs,"VideoMAEModel"),Cs.forEach(o),sa=l(mt,`. It is used to instantiate a
VideoMAE model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the VideoMAE
`),ye=s(mt,"A",{href:!0,rel:!0});var zs=r(ye);ra=l(zs,"MCG-NJU/videomae-base"),zs.forEach(o),na=l(mt," architecture."),mt.forEach(o),ia=f(fe),U=s(fe,"P",{});var ht=r(U);la=l(ht,"Configuration objects inherit from "),st=s(ht,"A",{href:!0});var Is=r(st);da=l(Is,"PretrainedConfig"),Is.forEach(o),ca=l(ht,` and can be used to control the model outputs. Read the
documentation from `),rt=s(ht,"A",{href:!0});var Ns=r(rt);pa=l(Ns,"PretrainedConfig"),Ns.forEach(o),ma=l(ht," for more information."),ht.forEach(o),ha=f(fe),M(oe.$$.fragment,fe),fe.forEach(o),oo=f(e),R=s(e,"H2",{class:!0});var go=r(R);ae=s(go,"A",{id:!0,class:!0,href:!0});var qs=r(ae);yt=s(qs,"SPAN",{});var Ss=r(yt);M(xe.$$.fragment,Ss),Ss.forEach(o),qs.forEach(o),fa=f(go),xt=s(go,"SPAN",{});var Ls=r(xt);ua=l(Ls,"VideoMAEFeatureExtractor"),Ls.forEach(o),go.forEach(o),ao=f(e),j=s(e,"DIV",{class:!0});var ue=r(j);M(Te.$$.fragment,ue),ga=f(ue),Tt=s(ue,"P",{});var Ds=r(Tt);_a=l(Ds,"Constructs a VideoMAE feature extractor. This feature extractor can be used to prepare videos for the model."),Ds.forEach(o),va=f(ue),je=s(ue,"P",{});var _o=r(je);Ea=l(_o,"This feature extractor inherits from "),nt=s(_o,"A",{href:!0});var Os=r(nt);ba=l(Os,"FeatureExtractionMixin"),Os.forEach(o),Ma=l(_o,` which contains most of the main methods. Users
should refer to this superclass for more information regarding those methods.`),_o.forEach(o),wa=f(ue),S=s(ue,"DIV",{class:!0});var ft=r(S);M(ke.$$.fragment,ft),Aa=f(ft),jt=s(ft,"P",{});var Us=r(jt);$a=l(Us,"Main method to prepare for the model one or several video(s)."),Us.forEach(o),Va=f(ft),M(se.$$.fragment,ft),ft.forEach(o),ue.forEach(o),so=f(e),W=s(e,"H2",{class:!0});var vo=r(W);re=s(vo,"A",{id:!0,class:!0,href:!0});var Rs=r(re);kt=s(Rs,"SPAN",{});var Ws=r(kt);M(Fe.$$.fragment,Ws),Ws.forEach(o),Rs.forEach(o),ya=f(vo),Ft=s(vo,"SPAN",{});var Js=r(Ft);xa=l(Js,"VideoMAEModel"),Js.forEach(o),vo.forEach(o),ro=f(e),C=s(e,"DIV",{class:!0});var ut=r(C);M(Pe.$$.fragment,ut),Ta=f(ut),Ce=s(ut,"P",{});var Eo=r(Ce);ja=l(Eo,`The bare VideoMAE Model transformer outputting raw hidden-states without any specific head on top.
This model is a PyTorch `),ze=s(Eo,"A",{href:!0,rel:!0});var Bs=r(ze);ka=l(Bs,"torch.nn.Module"),Bs.forEach(o),Fa=l(Eo,` subclass. Use it
as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),Eo.forEach(o),Pa=f(ut),k=s(ut,"DIV",{class:!0});var ge=r(k);M(Ie.$$.fragment,ge),Ca=f(ge),J=s(ge,"P",{});var gt=r(J);za=l(gt,"The "),it=s(gt,"A",{href:!0});var Gs=r(it);Ia=l(Gs,"VideoMAEModel"),Gs.forEach(o),Na=l(gt," forward method, overrides the "),Pt=s(gt,"CODE",{});var Hs=r(Pt);qa=l(Hs,"__call__"),Hs.forEach(o),Sa=l(gt," special method."),gt.forEach(o),La=f(ge),M(ne.$$.fragment,ge),Da=f(ge),M(ie.$$.fragment,ge),ge.forEach(o),ut.forEach(o),no=f(e),B=s(e,"H2",{class:!0});var bo=r(B);le=s(bo,"A",{id:!0,class:!0,href:!0});var Ks=r(le);Ct=s(Ks,"SPAN",{});var Zs=r(Ct);M(Ne.$$.fragment,Zs),Zs.forEach(o),Ks.forEach(o),Oa=f(bo),zt=s(bo,"SPAN",{});var Xs=r(zt);Ua=l(Xs,"VideoMAEForPreTraining"),Xs.forEach(o),bo.forEach(o),io=f(e),z=s(e,"DIV",{class:!0});var _t=r(z);M(qe.$$.fragment,_t),Ra=f(_t),Se=s(_t,"P",{});var Mo=r(Se);Wa=l(Mo,`The VideoMAE Model transformer with the decoder on top for self-supervised pre-training.
This model is a PyTorch `),Le=s(Mo,"A",{href:!0,rel:!0});var Ys=r(Le);Ja=l(Ys,"torch.nn.Module"),Ys.forEach(o),Ba=l(Mo,` subclass. Use it
as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),Mo.forEach(o),Ga=f(_t),F=s(_t,"DIV",{class:!0});var _e=r(F);M(De.$$.fragment,_e),Ha=f(_e),G=s(_e,"P",{});var vt=r(G);Ka=l(vt,"The "),lt=s(vt,"A",{href:!0});var Qs=r(lt);Za=l(Qs,"VideoMAEForPreTraining"),Qs.forEach(o),Xa=l(vt," forward method, overrides the "),It=s(vt,"CODE",{});var er=r(It);Ya=l(er,"__call__"),er.forEach(o),Qa=l(vt," special method."),vt.forEach(o),es=f(_e),M(de.$$.fragment,_e),ts=f(_e),M(ce.$$.fragment,_e),_e.forEach(o),_t.forEach(o),lo=f(e),H=s(e,"H2",{class:!0});var wo=r(H);pe=s(wo,"A",{id:!0,class:!0,href:!0});var tr=r(pe);Nt=s(tr,"SPAN",{});var or=r(Nt);M(Oe.$$.fragment,or),or.forEach(o),tr.forEach(o),os=f(wo),qt=s(wo,"SPAN",{});var ar=r(qt);as=l(ar,"VideoMAEForVideoClassification"),ar.forEach(o),wo.forEach(o),co=f(e),I=s(e,"DIV",{class:!0});var Et=r(I);M(Ue.$$.fragment,Et),ss=f(Et),Re=s(Et,"P",{});var Ao=r(Re);rs=l(Ao,`VideoMAE Model transformer with a video classification head on top (a linear layer on top of the final hidden state of
the [CLS] token) e.g. for ImageNet.
This model is a PyTorch `),We=s(Ao,"A",{href:!0,rel:!0});var sr=r(We);ns=l(sr,"torch.nn.Module"),sr.forEach(o),is=l(Ao,` subclass. Use it
as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),Ao.forEach(o),ls=f(Et),P=s(Et,"DIV",{class:!0});var ve=r(P);M(Je.$$.fragment,ve),ds=f(ve),K=s(ve,"P",{});var bt=r(K);cs=l(bt,"The "),dt=s(bt,"A",{href:!0});var rr=r(dt);ps=l(rr,"VideoMAEForVideoClassification"),rr.forEach(o),ms=l(bt," forward method, overrides the "),St=s(bt,"CODE",{});var nr=r(St);hs=l(nr,"__call__"),nr.forEach(o),fs=l(bt," special method."),bt.forEach(o),us=f(ve),M(me.$$.fragment,ve),gs=f(ve),M(he.$$.fragment,ve),ve.forEach(o),Et.forEach(o),this.h()},h(){c(d,"name","hf:doc:metadata"),c(d,"content",JSON.stringify(wr)),c(m,"id","videomae"),c(m,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(m,"href","#videomae"),c(u,"class","relative group"),c(X,"id","overview"),c(X,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(X,"href","#overview"),c(L,"class","relative group"),c(be,"href","https://arxiv.org/abs/2203.12602"),c(be,"rel","nofollow"),c(Ke,"href","vit_mae"),c(Qe,"href","/docs/transformers/pr_18658/en/model_doc/videomae#transformers.VideoMAEFeatureExtractor"),c(tt,"href","/docs/transformers/pr_18658/en/model_doc/videomae#transformers.VideoMAEForPreTraining"),pr(Q.src,vs="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/videomae_architecture.jpeg")||c(Q,"src",vs),c(Q,"alt","drawing"),c(Q,"width","600"),c(ot,"href","https://arxiv.org/abs/2203.12602"),c(we,"href","https://huggingface.co/nielsr"),c(we,"rel","nofollow"),c(Ae,"href","https://github.com/MCG-NJU/VideoMAE"),c(Ae,"rel","nofollow"),c(te,"id","transformers.VideoMAEConfig"),c(te,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(te,"href","#transformers.VideoMAEConfig"),c(D,"class","relative group"),c(at,"href","/docs/transformers/pr_18658/en/model_doc/videomae#transformers.VideoMAEModel"),c(ye,"href","https://huggingface.co/MCG-NJU/videomae-base"),c(ye,"rel","nofollow"),c(st,"href","/docs/transformers/pr_18658/en/main_classes/configuration#transformers.PretrainedConfig"),c(rt,"href","/docs/transformers/pr_18658/en/main_classes/configuration#transformers.PretrainedConfig"),c(T,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(ae,"id","transformers.VideoMAEFeatureExtractor"),c(ae,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(ae,"href","#transformers.VideoMAEFeatureExtractor"),c(R,"class","relative group"),c(nt,"href","/docs/transformers/pr_18658/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin"),c(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(re,"id","transformers.VideoMAEModel"),c(re,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(re,"href","#transformers.VideoMAEModel"),c(W,"class","relative group"),c(ze,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),c(ze,"rel","nofollow"),c(it,"href","/docs/transformers/pr_18658/en/model_doc/videomae#transformers.VideoMAEModel"),c(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(le,"id","transformers.VideoMAEForPreTraining"),c(le,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(le,"href","#transformers.VideoMAEForPreTraining"),c(B,"class","relative group"),c(Le,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),c(Le,"rel","nofollow"),c(lt,"href","/docs/transformers/pr_18658/en/model_doc/videomae#transformers.VideoMAEForPreTraining"),c(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(pe,"id","transformers.VideoMAEForVideoClassification"),c(pe,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),c(pe,"href","#transformers.VideoMAEForVideoClassification"),c(H,"class","relative group"),c(We,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),c(We,"rel","nofollow"),c(dt,"href","/docs/transformers/pr_18658/en/model_doc/videomae#transformers.VideoMAEForVideoClassification"),c(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),c(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,p){t(document.head,d),_(e,E,p),_(e,u,p),t(u,m),t(m,v),w(n,v,null),t(u,g),t(u,x),t(x,To),_(e,Jt,p),_(e,L,p),t(L,X),t(X,Mt),w(Ee,Mt,null),t(L,jo),t(L,wt),t(wt,ko),_(e,Bt,p),_(e,N,p),t(N,Fo),t(N,be),t(be,Po),t(N,Co),t(N,Ke),t(Ke,zo),t(N,Io),_(e,Gt,p),_(e,Ze,p),t(Ze,No),_(e,Ht,p),_(e,Xe,p),t(Xe,At),t(At,qo),_(e,Kt,p),_(e,Ye,p),t(Ye,So),_(e,Zt,p),_(e,Y,p),t(Y,Me),t(Me,Lo),t(Me,Qe),t(Qe,Do),t(Me,Oo),t(Y,Uo),t(Y,et),t(et,tt),t(tt,Ro),t(et,Wo),_(e,Xt,p),_(e,Q,p),_(e,Yt,p),_(e,ee,p),t(ee,Jo),t(ee,ot),t(ot,Bo),t(ee,Go),_(e,Qt,p),_(e,q,p),t(q,Ho),t(q,we),t(we,Ko),t(q,Zo),t(q,Ae),t(Ae,Xo),t(q,Yo),_(e,eo,p),_(e,D,p),t(D,te),t(te,$t),w($e,$t,null),t(D,Qo),t(D,Vt),t(Vt,ea),_(e,to,p),_(e,T,p),w(Ve,T,null),t(T,ta),t(T,O),t(O,oa),t(O,at),t(at,aa),t(O,sa),t(O,ye),t(ye,ra),t(O,na),t(T,ia),t(T,U),t(U,la),t(U,st),t(st,da),t(U,ca),t(U,rt),t(rt,pa),t(U,ma),t(T,ha),w(oe,T,null),_(e,oo,p),_(e,R,p),t(R,ae),t(ae,yt),w(xe,yt,null),t(R,fa),t(R,xt),t(xt,ua),_(e,ao,p),_(e,j,p),w(Te,j,null),t(j,ga),t(j,Tt),t(Tt,_a),t(j,va),t(j,je),t(je,Ea),t(je,nt),t(nt,ba),t(je,Ma),t(j,wa),t(j,S),w(ke,S,null),t(S,Aa),t(S,jt),t(jt,$a),t(S,Va),w(se,S,null),_(e,so,p),_(e,W,p),t(W,re),t(re,kt),w(Fe,kt,null),t(W,ya),t(W,Ft),t(Ft,xa),_(e,ro,p),_(e,C,p),w(Pe,C,null),t(C,Ta),t(C,Ce),t(Ce,ja),t(Ce,ze),t(ze,ka),t(Ce,Fa),t(C,Pa),t(C,k),w(Ie,k,null),t(k,Ca),t(k,J),t(J,za),t(J,it),t(it,Ia),t(J,Na),t(J,Pt),t(Pt,qa),t(J,Sa),t(k,La),w(ne,k,null),t(k,Da),w(ie,k,null),_(e,no,p),_(e,B,p),t(B,le),t(le,Ct),w(Ne,Ct,null),t(B,Oa),t(B,zt),t(zt,Ua),_(e,io,p),_(e,z,p),w(qe,z,null),t(z,Ra),t(z,Se),t(Se,Wa),t(Se,Le),t(Le,Ja),t(Se,Ba),t(z,Ga),t(z,F),w(De,F,null),t(F,Ha),t(F,G),t(G,Ka),t(G,lt),t(lt,Za),t(G,Xa),t(G,It),t(It,Ya),t(G,Qa),t(F,es),w(de,F,null),t(F,ts),w(ce,F,null),_(e,lo,p),_(e,H,p),t(H,pe),t(pe,Nt),w(Oe,Nt,null),t(H,os),t(H,qt),t(qt,as),_(e,co,p),_(e,I,p),w(Ue,I,null),t(I,ss),t(I,Re),t(Re,rs),t(Re,We),t(We,ns),t(Re,is),t(I,ls),t(I,P),w(Je,P,null),t(P,ds),t(P,K),t(K,cs),t(K,dt),t(dt,ps),t(K,ms),t(K,St),t(St,hs),t(K,fs),t(P,us),w(me,P,null),t(P,gs),w(he,P,null),po=!0},p(e,[p]){const Be={};p&2&&(Be.$$scope={dirty:p,ctx:e}),oe.$set(Be);const Lt={};p&2&&(Lt.$$scope={dirty:p,ctx:e}),se.$set(Lt);const Dt={};p&2&&(Dt.$$scope={dirty:p,ctx:e}),ne.$set(Dt);const Ot={};p&2&&(Ot.$$scope={dirty:p,ctx:e}),ie.$set(Ot);const Ge={};p&2&&(Ge.$$scope={dirty:p,ctx:e}),de.$set(Ge);const Ut={};p&2&&(Ut.$$scope={dirty:p,ctx:e}),ce.$set(Ut);const Rt={};p&2&&(Rt.$$scope={dirty:p,ctx:e}),me.$set(Rt);const Wt={};p&2&&(Wt.$$scope={dirty:p,ctx:e}),he.$set(Wt)},i(e){po||(A(n.$$.fragment,e),A(Ee.$$.fragment,e),A($e.$$.fragment,e),A(Ve.$$.fragment,e),A(oe.$$.fragment,e),A(xe.$$.fragment,e),A(Te.$$.fragment,e),A(ke.$$.fragment,e),A(se.$$.fragment,e),A(Fe.$$.fragment,e),A(Pe.$$.fragment,e),A(Ie.$$.fragment,e),A(ne.$$.fragment,e),A(ie.$$.fragment,e),A(Ne.$$.fragment,e),A(qe.$$.fragment,e),A(De.$$.fragment,e),A(de.$$.fragment,e),A(ce.$$.fragment,e),A(Oe.$$.fragment,e),A(Ue.$$.fragment,e),A(Je.$$.fragment,e),A(me.$$.fragment,e),A(he.$$.fragment,e),po=!0)},o(e){$(n.$$.fragment,e),$(Ee.$$.fragment,e),$($e.$$.fragment,e),$(Ve.$$.fragment,e),$(oe.$$.fragment,e),$(xe.$$.fragment,e),$(Te.$$.fragment,e),$(ke.$$.fragment,e),$(se.$$.fragment,e),$(Fe.$$.fragment,e),$(Pe.$$.fragment,e),$(Ie.$$.fragment,e),$(ne.$$.fragment,e),$(ie.$$.fragment,e),$(Ne.$$.fragment,e),$(qe.$$.fragment,e),$(De.$$.fragment,e),$(de.$$.fragment,e),$(ce.$$.fragment,e),$(Oe.$$.fragment,e),$(Ue.$$.fragment,e),$(Je.$$.fragment,e),$(me.$$.fragment,e),$(he.$$.fragment,e),po=!1},d(e){o(d),e&&o(E),e&&o(u),V(n),e&&o(Jt),e&&o(L),V(Ee),e&&o(Bt),e&&o(N),e&&o(Gt),e&&o(Ze),e&&o(Ht),e&&o(Xe),e&&o(Kt),e&&o(Ye),e&&o(Zt),e&&o(Y),e&&o(Xt),e&&o(Q),e&&o(Yt),e&&o(ee),e&&o(Qt),e&&o(q),e&&o(eo),e&&o(D),V($e),e&&o(to),e&&o(T),V(Ve),V(oe),e&&o(oo),e&&o(R),V(xe),e&&o(ao),e&&o(j),V(Te),V(ke),V(se),e&&o(so),e&&o(W),V(Fe),e&&o(ro),e&&o(C),V(Pe),V(Ie),V(ne),V(ie),e&&o(no),e&&o(B),V(Ne),e&&o(io),e&&o(z),V(qe),V(De),V(de),V(ce),e&&o(lo),e&&o(H),V(Oe),e&&o(co),e&&o(I),V(Ue),V(Je),V(me),V(he)}}}const wr={local:"videomae",sections:[{local:"overview",title:"Overview"},{local:"transformers.VideoMAEConfig",title:"VideoMAEConfig"},{local:"transformers.VideoMAEFeatureExtractor",title:"VideoMAEFeatureExtractor"},{local:"transformers.VideoMAEModel",title:"VideoMAEModel"},{local:"transformers.VideoMAEForPreTraining",title:"VideoMAEForPreTraining"},{local:"transformers.VideoMAEForVideoClassification",title:"VideoMAEForVideoClassification"}],title:"VideoMAE"};function Ar(y){return mr(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class kr extends ir{constructor(d){super();lr(this,d,Ar,Mr,dr,{})}}export{kr as default,wr as metadata};
