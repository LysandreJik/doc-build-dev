import{S as Yy,i as ek,s as tk,e as a,k as d,w,t as r,M as ok,c as s,d as t,m as p,a as n,x as y,h as i,b as m,G as e,g as _,y as k,q as T,o as x,B as $,v as ak,L as _e}from"../../chunks/vendor-hf-doc-builder.js";import{T as ke}from"../../chunks/Tip-hf-doc-builder.js";import{D as F}from"../../chunks/Docstring-hf-doc-builder.js";import{C as ve}from"../../chunks/CodeBlock-hf-doc-builder.js";import{I as oe}from"../../chunks/IconCopyLink-hf-doc-builder.js";import{E as ge}from"../../chunks/ExampleCodeBlock-hf-doc-builder.js";function sk(W){let c,b,u,f,v;return f=new ve({props:{code:`from transformers import Wav2Vec2Model, Wav2Vec2Config

# Initializing a Wav2Vec2 facebook/wav2vec2-base-960h style configuration
configuration = Wav2Vec2Config()

# Initializing a model from the facebook/wav2vec2-base-960h style configuration
model = Wav2Vec2Model(configuration)

# Accessing the model configuration
configuration = model.config`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Wav2Vec2Model, Wav2Vec2Config

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Wav2Vec2 facebook/wav2vec2-base-960h style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = Wav2Vec2Config()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the facebook/wav2vec2-base-960h style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Wav2Vec2Model(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`}}),{c(){c=a("p"),b=r("Example:"),u=d(),w(f.$$.fragment)},l(l){c=s(l,"P",{});var h=n(c);b=i(h,"Example:"),h.forEach(t),u=p(l),y(f.$$.fragment,l)},m(l,h){_(l,c,h),e(c,b),_(l,u,h),k(f,l,h),v=!0},p:_e,i(l){v||(T(f.$$.fragment,l),v=!0)},o(l){x(f.$$.fragment,l),v=!1},d(l){l&&t(c),l&&t(u),$(f,l)}}}function nk(W){let c,b,u,f,v;return f=new ve({props:{code:`# Let's see how to retrieve time steps for a model
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC
from datasets import load_dataset
import datasets
import torch

# import model, feature extractor, tokenizer
model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

# load first sample of English common_voice
dataset = load_dataset("common_voice", "en", split="train", streaming=True)
dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16_000))
dataset_iter = iter(dataset)
sample = next(dataset_iter)

# forward sample through model to get greedily predicted transcription ids
input_values = feature_extractor(sample["audio"]["array"], return_tensors="pt").input_values
logits = model(input_values).logits[0]
pred_ids = torch.argmax(logits, axis=-1)

# retrieve word stamps (analogous commands for \`output_char_offsets\`)
outputs = tokenizer.decode(pred_ids, output_word_offsets=True)
# compute \`time_offset\` in seconds as product of downsampling ratio and sampling_rate
time_offset = model.config.inputs_to_logits_ratio / feature_extractor.sampling_rate

word_offsets = [
    {
        "word": d["word"],
        "start_time": round(d["start_offset"] * time_offset, 2),
        "end_time": round(d["end_offset"] * time_offset, 2),
    }
    for d in outputs.word_offsets
]
# compare word offsets with audio \`common_voice_en_100038.mp3\` online on the dataset viewer:
# https://huggingface.co/datasets/common_voice/viewer/en/train
word_offsets[:3]`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Let&#x27;s see how to retrieve time steps for a model</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> datasets
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># import model, feature extractor, tokenizer</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = AutoModelForCTC.from_pretrained(<span class="hljs-string">&quot;facebook/wav2vec2-base-960h&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/wav2vec2-base-960h&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>feature_extractor = AutoFeatureExtractor.from_pretrained(<span class="hljs-string">&quot;facebook/wav2vec2-base-960h&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># load first sample of English common_voice</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = load_dataset(<span class="hljs-string">&quot;common_voice&quot;</span>, <span class="hljs-string">&quot;en&quot;</span>, split=<span class="hljs-string">&quot;train&quot;</span>, streaming=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = dataset.cast_column(<span class="hljs-string">&quot;audio&quot;</span>, datasets.Audio(sampling_rate=<span class="hljs-number">16_000</span>))
<span class="hljs-meta">&gt;&gt;&gt; </span>dataset_iter = <span class="hljs-built_in">iter</span>(dataset)
<span class="hljs-meta">&gt;&gt;&gt; </span>sample = <span class="hljs-built_in">next</span>(dataset_iter)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># forward sample through model to get greedily predicted transcription ids</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>input_values = feature_extractor(sample[<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;array&quot;</span>], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).input_values
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = model(input_values).logits[<span class="hljs-number">0</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>pred_ids = torch.argmax(logits, axis=-<span class="hljs-number">1</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># retrieve word stamps (analogous commands for \`output_char_offsets\`)</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = tokenizer.decode(pred_ids, output_word_offsets=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># compute \`time_offset\` in seconds as product of downsampling ratio and sampling_rate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>time_offset = model.config.inputs_to_logits_ratio / feature_extractor.sampling_rate

<span class="hljs-meta">&gt;&gt;&gt; </span>word_offsets = [
<span class="hljs-meta">... </span>    {
<span class="hljs-meta">... </span>        <span class="hljs-string">&quot;word&quot;</span>: d[<span class="hljs-string">&quot;word&quot;</span>],
<span class="hljs-meta">... </span>        <span class="hljs-string">&quot;start_time&quot;</span>: <span class="hljs-built_in">round</span>(d[<span class="hljs-string">&quot;start_offset&quot;</span>] * time_offset, <span class="hljs-number">2</span>),
<span class="hljs-meta">... </span>        <span class="hljs-string">&quot;end_time&quot;</span>: <span class="hljs-built_in">round</span>(d[<span class="hljs-string">&quot;end_offset&quot;</span>] * time_offset, <span class="hljs-number">2</span>),
<span class="hljs-meta">... </span>    }
<span class="hljs-meta">... </span>    <span class="hljs-keyword">for</span> d <span class="hljs-keyword">in</span> outputs.word_offsets
<span class="hljs-meta">... </span>]
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># compare word offsets with audio \`common_voice_en_100038.mp3\` online on the dataset viewer:</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># https://huggingface.co/datasets/common_voice/viewer/en/train</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>word_offsets[:<span class="hljs-number">3</span>]
[{<span class="hljs-string">&#x27;word&#x27;</span>: <span class="hljs-string">&#x27;WHY&#x27;</span>, <span class="hljs-string">&#x27;start_time&#x27;</span>: <span class="hljs-number">1.42</span>, <span class="hljs-string">&#x27;end_time&#x27;</span>: <span class="hljs-number">1.54</span>}, {<span class="hljs-string">&#x27;word&#x27;</span>: <span class="hljs-string">&#x27;DOES&#x27;</span>, <span class="hljs-string">&#x27;start_time&#x27;</span>: <span class="hljs-number">1.64</span>, <span class="hljs-string">&#x27;end_time&#x27;</span>: <span class="hljs-number">1.9</span>}, {<span class="hljs-string">&#x27;word&#x27;</span>: <span class="hljs-string">&#x27;MILISANDRA&#x27;</span>, <span class="hljs-string">&#x27;start_time&#x27;</span>: <span class="hljs-number">2.26</span>, <span class="hljs-string">&#x27;end_time&#x27;</span>: <span class="hljs-number">2.9</span>}]`}}),{c(){c=a("p"),b=r("Example:"),u=d(),w(f.$$.fragment)},l(l){c=s(l,"P",{});var h=n(c);b=i(h,"Example:"),h.forEach(t),u=p(l),y(f.$$.fragment,l)},m(l,h){_(l,c,h),e(c,b),_(l,u,h),k(f,l,h),v=!0},p:_e,i(l){v||(T(f.$$.fragment,l),v=!0)},o(l){x(f.$$.fragment,l),v=!1},d(l){l&&t(c),l&&t(u),$(f,l)}}}function rk(W){let c,b,u,f,v,l,h,V;return{c(){c=a("p"),b=r("This class method is simply calling "),u=a("a"),f=r("save_pretrained()"),v=r(` and
`),l=a("code"),h=r("save_pretrained"),V=r(`. Please refer to the docstrings of the methods
above for more information.`),this.h()},l(A){c=s(A,"P",{});var M=n(c);b=i(M,"This class method is simply calling "),u=s(M,"A",{href:!0});var C=n(u);f=i(C,"save_pretrained()"),C.forEach(t),v=i(M,` and
`),l=s(M,"CODE",{});var z=n(l);h=i(z,"save_pretrained"),z.forEach(t),V=i(M,`. Please refer to the docstrings of the methods
above for more information.`),M.forEach(t),this.h()},h(){m(u,"href","/docs/transformers/pr_18351/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained")},m(A,M){_(A,c,M),e(c,b),e(c,u),e(u,f),e(c,v),e(c,l),e(l,h),e(c,V)},d(A){A&&t(c)}}}function ik(W){let c,b,u,f,v,l,h,V,A,M,C,z,L,U;return{c(){c=a("p"),b=r(`This class method is simply calling Wav2Vec2FeatureExtractor\u2019s
`),u=a("a"),f=r("from_pretrained()"),v=r(`, Wav2Vec2CTCTokenizer\u2019s
`),l=a("code"),h=r("from_pretrained"),V=r(`, and
`),A=a("code"),M=r("pyctcdecode.BeamSearchDecoderCTC.load_from_hf_hub"),C=r("."),z=d(),L=a("p"),U=r("Please refer to the docstrings of the methods above for more information."),this.h()},l(O){c=s(O,"P",{});var E=n(c);b=i(E,`This class method is simply calling Wav2Vec2FeatureExtractor\u2019s
`),u=s(E,"A",{href:!0});var ae=n(u);f=i(ae,"from_pretrained()"),ae.forEach(t),v=i(E,`, Wav2Vec2CTCTokenizer\u2019s
`),l=s(E,"CODE",{});var X=n(l);h=i(X,"from_pretrained"),X.forEach(t),V=i(E,`, and
`),A=s(E,"CODE",{});var I=n(A);M=i(I,"pyctcdecode.BeamSearchDecoderCTC.load_from_hf_hub"),I.forEach(t),C=i(E,"."),E.forEach(t),z=p(O),L=s(O,"P",{});var N=n(L);U=i(N,"Please refer to the docstrings of the methods above for more information."),N.forEach(t),this.h()},h(){m(u,"href","/docs/transformers/pr_18351/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained")},m(O,E){_(O,c,E),e(c,b),e(c,u),e(u,f),e(c,v),e(c,l),e(l,h),e(c,V),e(c,A),e(A,M),e(c,C),_(O,z,E),_(O,L,E),e(L,U)},d(O){O&&t(c),O&&t(z),O&&t(L)}}}function lk(W){let c,b,u,f,v,l,h,V,A,M,C,z,L,U,O,E,ae,X,I,N;return{c(){c=a("p"),b=r(`This function makes use of Python\u2019s multiprocessing. Currently, multiprocessing is available only on Unix
systems (see this `),u=a("a"),f=r("issue"),v=r(")."),l=d(),h=a("p"),V=r("If you are decoding multiple batches, consider creating a "),A=a("code"),M=r("Pool"),C=r(" and passing it to "),z=a("code"),L=r("batch_decode"),U=r(`. Otherwise,
`),O=a("code"),E=r("batch_decode"),ae=r(" will be very slow since it will create a fresh "),X=a("code"),I=r("Pool"),N=r(" for each call. See usage example below."),this.h()},l(S){c=s(S,"P",{});var H=n(c);b=i(H,`This function makes use of Python\u2019s multiprocessing. Currently, multiprocessing is available only on Unix
systems (see this `),u=s(H,"A",{href:!0,rel:!0});var D=n(u);f=i(D,"issue"),D.forEach(t),v=i(H,")."),H.forEach(t),l=p(S),h=s(S,"P",{});var P=n(h);V=i(P,"If you are decoding multiple batches, consider creating a "),A=s(P,"CODE",{});var re=n(A);M=i(re,"Pool"),re.forEach(t),C=i(P," and passing it to "),z=s(P,"CODE",{});var se=n(z);L=i(se,"batch_decode"),se.forEach(t),U=i(P,`. Otherwise,
`),O=s(P,"CODE",{});var Te=n(O);E=i(Te,"batch_decode"),Te.forEach(t),ae=i(P," will be very slow since it will create a fresh "),X=s(P,"CODE",{});var ie=n(X);I=i(ie,"Pool"),ie.forEach(t),N=i(P," for each call. See usage example below."),P.forEach(t),this.h()},h(){m(u,"href","https://github.com/kensho-technologies/pyctcdecode/issues/65"),m(u,"rel","nofollow")},m(S,H){_(S,c,H),e(c,b),e(c,u),e(u,f),e(c,v),_(S,l,H),_(S,h,H),e(h,V),e(h,A),e(A,M),e(h,C),e(h,z),e(z,L),e(h,U),e(h,O),e(O,E),e(h,ae),e(h,X),e(X,I),e(h,N)},d(S){S&&t(c),S&&t(l),S&&t(h)}}}function ck(W){let c,b,u,f,v;return f=new ve({props:{code:`# Let's see how to use a user-managed pool for batch decoding multiple audios
from multiprocessing import get_context
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCTC
from datasets import load_dataset
import datasets
import torch

# import model, feature extractor, tokenizer
model = AutoModelForCTC.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm").to("cuda")
processor = AutoProcessor.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")

# load example dataset
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16_000))


def map_to_array(batch):
    batch["speech"] = batch["audio"]["array"]
    return batch


# prepare speech data for batch inference
dataset = dataset.map(map_to_array, remove_columns=["audio"])


def map_to_pred(batch, pool):
    inputs = processor(batch["speech"], sampling_rate=16_000, padding=True, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    transcription = processor.batch_decode(logits.cpu().numpy(), pool).text
    batch["transcription"] = transcription
    return batch


# note: pool should be instantiated *after* \`Wav2Vec2ProcessorWithLM\`.
#       otherwise, the LM won't be available to the pool's sub-processes
# select number of processes and batch_size based on number of CPU cores available and on dataset size
with get_context("fork").Pool(processes=2) as pool:
    result = dataset.map(
        map_to_pred, batched=True, batch_size=2, fn_kwargs={"pool": pool}, remove_columns=["speech"]
    )

result["transcription"][:2]`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Let&#x27;s see how to use a user-managed pool for batch decoding multiple audios</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> multiprocessing <span class="hljs-keyword">import</span> get_context
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, AutoProcessor, AutoModelForCTC
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> datasets
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># import model, feature extractor, tokenizer</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = AutoModelForCTC.from_pretrained(<span class="hljs-string">&quot;patrickvonplaten/wav2vec2-base-100h-with-lm&quot;</span>).to(<span class="hljs-string">&quot;cuda&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(<span class="hljs-string">&quot;patrickvonplaten/wav2vec2-base-100h-with-lm&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># load example dataset</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/librispeech_asr_dummy&quot;</span>, <span class="hljs-string">&quot;clean&quot;</span>, split=<span class="hljs-string">&quot;validation&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = dataset.cast_column(<span class="hljs-string">&quot;audio&quot;</span>, datasets.Audio(sampling_rate=<span class="hljs-number">16_000</span>))


<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">def</span> <span class="hljs-title function_">map_to_array</span>(<span class="hljs-params">batch</span>):
<span class="hljs-meta">... </span>    batch[<span class="hljs-string">&quot;speech&quot;</span>] = batch[<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;array&quot;</span>]
<span class="hljs-meta">... </span>    <span class="hljs-keyword">return</span> batch


<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># prepare speech data for batch inference</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = dataset.<span class="hljs-built_in">map</span>(map_to_array, remove_columns=[<span class="hljs-string">&quot;audio&quot;</span>])


<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">def</span> <span class="hljs-title function_">map_to_pred</span>(<span class="hljs-params">batch, pool</span>):
<span class="hljs-meta">... </span>    inputs = processor(batch[<span class="hljs-string">&quot;speech&quot;</span>], sampling_rate=<span class="hljs-number">16_000</span>, padding=<span class="hljs-literal">True</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">... </span>    inputs = {k: v.to(<span class="hljs-string">&quot;cuda&quot;</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> inputs.items()}

<span class="hljs-meta">... </span>    <span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>        logits = model(**inputs).logits

<span class="hljs-meta">... </span>    transcription = processor.batch_decode(logits.cpu().numpy(), pool).text
<span class="hljs-meta">... </span>    batch[<span class="hljs-string">&quot;transcription&quot;</span>] = transcription
<span class="hljs-meta">... </span>    <span class="hljs-keyword">return</span> batch


<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># note: pool should be instantiated *after* \`Wav2Vec2ProcessorWithLM\`.</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment">#       otherwise, the LM won&#x27;t be available to the pool&#x27;s sub-processes</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># select number of processes and batch_size based on number of CPU cores available and on dataset size</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> get_context(<span class="hljs-string">&quot;fork&quot;</span>).Pool(processes=<span class="hljs-number">2</span>) <span class="hljs-keyword">as</span> pool:
<span class="hljs-meta">... </span>    result = dataset.<span class="hljs-built_in">map</span>(
<span class="hljs-meta">... </span>        map_to_pred, batched=<span class="hljs-literal">True</span>, batch_size=<span class="hljs-number">2</span>, fn_kwargs={<span class="hljs-string">&quot;pool&quot;</span>: pool}, remove_columns=[<span class="hljs-string">&quot;speech&quot;</span>]
<span class="hljs-meta">... </span>    )

<span class="hljs-meta">&gt;&gt;&gt; </span>result[<span class="hljs-string">&quot;transcription&quot;</span>][:<span class="hljs-number">2</span>]
[<span class="hljs-string">&#x27;MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL&#x27;</span>, <span class="hljs-string">&quot;NOR IS MISTER COULTER&#x27;S MANNER LESS INTERESTING THAN HIS MATTER&quot;</span>]`}}),{c(){c=a("p"),b=r("Example:"),u=d(),w(f.$$.fragment)},l(l){c=s(l,"P",{});var h=n(c);b=i(h,"Example:"),h.forEach(t),u=p(l),y(f.$$.fragment,l)},m(l,h){_(l,c,h),e(c,b),_(l,u,h),k(f,l,h),v=!0},p:_e,i(l){v||(T(f.$$.fragment,l),v=!0)},o(l){x(f.$$.fragment,l),v=!1},d(l){l&&t(c),l&&t(u),$(f,l)}}}function dk(W){let c,b,u,f,v;return f=new ve({props:{code:`# Let's see how to retrieve time steps for a model
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCTC
from datasets import load_dataset
import datasets
import torch

# import model, feature extractor, tokenizer
model = AutoModelForCTC.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")
processor = AutoProcessor.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")

# load first sample of English common_voice
dataset = load_dataset("common_voice", "en", split="train", streaming=True)
dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16_000))
dataset_iter = iter(dataset)
sample = next(dataset_iter)

# forward sample through model to get greedily predicted transcription ids
input_values = processor(sample["audio"]["array"], return_tensors="pt").input_values
with torch.no_grad():
    logits = model(input_values).logits[0].cpu().numpy()

# retrieve word stamps (analogous commands for \`output_char_offsets\`)
outputs = processor.decode(logits, output_word_offsets=True)
# compute \`time_offset\` in seconds as product of downsampling ratio and sampling_rate
time_offset = model.config.inputs_to_logits_ratio / processor.feature_extractor.sampling_rate

word_offsets = [
    {
        "word": d["word"],
        "start_time": round(d["start_offset"] * time_offset, 2),
        "end_time": round(d["end_offset"] * time_offset, 2),
    }
    for d in outputs.word_offsets
]
# compare word offsets with audio \`common_voice_en_100038.mp3\` online on the dataset viewer:
# https://huggingface.co/datasets/common_voice/viewer/en/train
word_offsets[:4]`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Let&#x27;s see how to retrieve time steps for a model</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, AutoProcessor, AutoModelForCTC
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> datasets
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># import model, feature extractor, tokenizer</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = AutoModelForCTC.from_pretrained(<span class="hljs-string">&quot;patrickvonplaten/wav2vec2-base-100h-with-lm&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(<span class="hljs-string">&quot;patrickvonplaten/wav2vec2-base-100h-with-lm&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># load first sample of English common_voice</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = load_dataset(<span class="hljs-string">&quot;common_voice&quot;</span>, <span class="hljs-string">&quot;en&quot;</span>, split=<span class="hljs-string">&quot;train&quot;</span>, streaming=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = dataset.cast_column(<span class="hljs-string">&quot;audio&quot;</span>, datasets.Audio(sampling_rate=<span class="hljs-number">16_000</span>))
<span class="hljs-meta">&gt;&gt;&gt; </span>dataset_iter = <span class="hljs-built_in">iter</span>(dataset)
<span class="hljs-meta">&gt;&gt;&gt; </span>sample = <span class="hljs-built_in">next</span>(dataset_iter)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># forward sample through model to get greedily predicted transcription ids</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>input_values = processor(sample[<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;array&quot;</span>], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).input_values
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(input_values).logits[<span class="hljs-number">0</span>].cpu().numpy()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># retrieve word stamps (analogous commands for \`output_char_offsets\`)</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = processor.decode(logits, output_word_offsets=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># compute \`time_offset\` in seconds as product of downsampling ratio and sampling_rate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>time_offset = model.config.inputs_to_logits_ratio / processor.feature_extractor.sampling_rate

<span class="hljs-meta">&gt;&gt;&gt; </span>word_offsets = [
<span class="hljs-meta">... </span>    {
<span class="hljs-meta">... </span>        <span class="hljs-string">&quot;word&quot;</span>: d[<span class="hljs-string">&quot;word&quot;</span>],
<span class="hljs-meta">... </span>        <span class="hljs-string">&quot;start_time&quot;</span>: <span class="hljs-built_in">round</span>(d[<span class="hljs-string">&quot;start_offset&quot;</span>] * time_offset, <span class="hljs-number">2</span>),
<span class="hljs-meta">... </span>        <span class="hljs-string">&quot;end_time&quot;</span>: <span class="hljs-built_in">round</span>(d[<span class="hljs-string">&quot;end_offset&quot;</span>] * time_offset, <span class="hljs-number">2</span>),
<span class="hljs-meta">... </span>    }
<span class="hljs-meta">... </span>    <span class="hljs-keyword">for</span> d <span class="hljs-keyword">in</span> outputs.word_offsets
<span class="hljs-meta">... </span>]
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># compare word offsets with audio \`common_voice_en_100038.mp3\` online on the dataset viewer:</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># https://huggingface.co/datasets/common_voice/viewer/en/train</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>word_offsets[:<span class="hljs-number">4</span>]
[{<span class="hljs-string">&#x27;word&#x27;</span>: <span class="hljs-string">&#x27;WHY&#x27;</span>, <span class="hljs-string">&#x27;start_time&#x27;</span>: <span class="hljs-number">1.42</span>, <span class="hljs-string">&#x27;end_time&#x27;</span>: <span class="hljs-number">1.54</span>}, {<span class="hljs-string">&#x27;word&#x27;</span>: <span class="hljs-string">&#x27;DOES&#x27;</span>, <span class="hljs-string">&#x27;start_time&#x27;</span>: <span class="hljs-number">1.64</span>, <span class="hljs-string">&#x27;end_time&#x27;</span>: <span class="hljs-number">1.88</span>}, {<span class="hljs-string">&#x27;word&#x27;</span>: <span class="hljs-string">&#x27;A&#x27;</span>, <span class="hljs-string">&#x27;start_time&#x27;</span>: <span class="hljs-number">2.12</span>, <span class="hljs-string">&#x27;end_time&#x27;</span>: <span class="hljs-number">2.14</span>}, {<span class="hljs-string">&#x27;word&#x27;</span>: <span class="hljs-string">&#x27;MILE&#x27;</span>, <span class="hljs-string">&#x27;start_time&#x27;</span>: <span class="hljs-number">2.26</span>, <span class="hljs-string">&#x27;end_time&#x27;</span>: <span class="hljs-number">2.46</span>}]`}}),{c(){c=a("p"),b=r("Example:"),u=d(),w(f.$$.fragment)},l(l){c=s(l,"P",{});var h=n(c);b=i(h,"Example:"),h.forEach(t),u=p(l),y(f.$$.fragment,l)},m(l,h){_(l,c,h),e(c,b),_(l,u,h),k(f,l,h),v=!0},p:_e,i(l){v||(T(f.$$.fragment,l),v=!0)},o(l){x(f.$$.fragment,l),v=!1},d(l){l&&t(c),l&&t(u),$(f,l)}}}function pk(W){let c,b,u,f,v;return{c(){c=a("p"),b=r("Although the recipe for forward pass needs to be defined within this function, one should call the "),u=a("code"),f=r("Module"),v=r(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){c=s(l,"P",{});var h=n(c);b=i(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),u=s(h,"CODE",{});var V=n(u);f=i(V,"Module"),V.forEach(t),v=i(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(t)},m(l,h){_(l,c,h),e(c,b),e(c,u),e(u,f),e(c,v)},d(l){l&&t(c)}}}function mk(W){let c,b,u,f,v;return f=new ve({props:{code:`from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
from datasets import load_dataset

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# audio file is decoded on the fly
inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
list(last_hidden_states.shape)`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Wav2Vec2Processor, Wav2Vec2Model
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset

<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/librispeech_asr_demo&quot;</span>, <span class="hljs-string">&quot;clean&quot;</span>, split=<span class="hljs-string">&quot;validation&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = dataset.sort(<span class="hljs-string">&quot;id&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>sampling_rate = dataset.features[<span class="hljs-string">&quot;audio&quot;</span>].sampling_rate

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = Wav2Vec2Processor.from_pretrained(<span class="hljs-string">&quot;facebook/wav2vec2-base-960h&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Wav2Vec2Model.from_pretrained(<span class="hljs-string">&quot;facebook/wav2vec2-base-960h&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># audio file is decoded on the fly</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(dataset[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;array&quot;</span>], sampling_rate=sampling_rate, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(last_hidden_states.shape)
[<span class="hljs-number">1</span>, <span class="hljs-number">292</span>, <span class="hljs-number">768</span>]`}}),{c(){c=a("p"),b=r("Example:"),u=d(),w(f.$$.fragment)},l(l){c=s(l,"P",{});var h=n(c);b=i(h,"Example:"),h.forEach(t),u=p(l),y(f.$$.fragment,l)},m(l,h){_(l,c,h),e(c,b),_(l,u,h),k(f,l,h),v=!0},p:_e,i(l){v||(T(f.$$.fragment,l),v=!0)},o(l){x(f.$$.fragment,l),v=!1},d(l){l&&t(c),l&&t(u),$(f,l)}}}function hk(W){let c,b,u,f,v;return{c(){c=a("p"),b=r("Although the recipe for forward pass needs to be defined within this function, one should call the "),u=a("code"),f=r("Module"),v=r(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){c=s(l,"P",{});var h=n(c);b=i(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),u=s(h,"CODE",{});var V=n(u);f=i(V,"Module"),V.forEach(t),v=i(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(t)},m(l,h){_(l,c,h),e(c,b),e(c,u),e(u,f),e(c,v)},d(l){l&&t(c)}}}function fk(W){let c,b,u,f,v;return f=new ve({props:{code:`from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# audio file is decoded on the fly
inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)

# transcribe speech
transcription = processor.batch_decode(predicted_ids)
transcription[0]`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Wav2Vec2Processor, Wav2Vec2ForCTC
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/librispeech_asr_demo&quot;</span>, <span class="hljs-string">&quot;clean&quot;</span>, split=<span class="hljs-string">&quot;validation&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = dataset.sort(<span class="hljs-string">&quot;id&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>sampling_rate = dataset.features[<span class="hljs-string">&quot;audio&quot;</span>].sampling_rate

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = Wav2Vec2Processor.from_pretrained(<span class="hljs-string">&quot;facebook/wav2vec2-base-960h&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Wav2Vec2ForCTC.from_pretrained(<span class="hljs-string">&quot;facebook/wav2vec2-base-960h&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># audio file is decoded on the fly</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(dataset[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;array&quot;</span>], sampling_rate=sampling_rate, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_ids = torch.argmax(logits, dim=-<span class="hljs-number">1</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># transcribe speech</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>transcription = processor.batch_decode(predicted_ids)
<span class="hljs-meta">&gt;&gt;&gt; </span>transcription[<span class="hljs-number">0</span>]
<span class="hljs-string">&#x27;MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL&#x27;</span>`}}),{c(){c=a("p"),b=r("Example:"),u=d(),w(f.$$.fragment)},l(l){c=s(l,"P",{});var h=n(c);b=i(h,"Example:"),h.forEach(t),u=p(l),y(f.$$.fragment,l)},m(l,h){_(l,c,h),e(c,b),_(l,u,h),k(f,l,h),v=!0},p:_e,i(l){v||(T(f.$$.fragment,l),v=!0)},o(l){x(f.$$.fragment,l),v=!1},d(l){l&&t(c),l&&t(u),$(f,l)}}}function uk(W){let c,b;return c=new ve({props:{code:`inputs["labels"] = processor(text=dataset[0]["text"], return_tensors="pt").input_ids

# compute loss
loss = model(**inputs).loss
round(loss.item(), 2)`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span>inputs[<span class="hljs-string">&quot;labels&quot;</span>] = processor(text=dataset[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;text&quot;</span>], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).input_ids

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># compute loss</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
<span class="hljs-number">53.48</span>`}}),{c(){w(c.$$.fragment)},l(u){y(c.$$.fragment,u)},m(u,f){k(c,u,f),b=!0},p:_e,i(u){b||(T(c.$$.fragment,u),b=!0)},o(u){x(c.$$.fragment,u),b=!1},d(u){$(c,u)}}}function gk(W){let c,b,u,f,v;return{c(){c=a("p"),b=r("Although the recipe for forward pass needs to be defined within this function, one should call the "),u=a("code"),f=r("Module"),v=r(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){c=s(l,"P",{});var h=n(c);b=i(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),u=s(h,"CODE",{});var V=n(u);f=i(V,"Module"),V.forEach(t),v=i(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(t)},m(l,h){_(l,c,h),e(c,b),e(c,u),e(u,f),e(c,v)},d(l){l&&t(c)}}}function _k(W){let c,b,u,f,v;return f=new ve({props:{code:`from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from datasets import load_dataset
import torch

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-ks")
model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-ks")

# audio file is decoded on the fly
inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_ids = torch.argmax(logits, dim=-1).item()
predicted_label = model.config.id2label[predicted_class_ids]
predicted_label`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/librispeech_asr_demo&quot;</span>, <span class="hljs-string">&quot;clean&quot;</span>, split=<span class="hljs-string">&quot;validation&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = dataset.sort(<span class="hljs-string">&quot;id&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>sampling_rate = dataset.features[<span class="hljs-string">&quot;audio&quot;</span>].sampling_rate

<span class="hljs-meta">&gt;&gt;&gt; </span>feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(<span class="hljs-string">&quot;superb/wav2vec2-base-superb-ks&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Wav2Vec2ForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;superb/wav2vec2-base-superb-ks&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># audio file is decoded on the fly</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = feature_extractor(dataset[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;array&quot;</span>], sampling_rate=sampling_rate, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.argmax(logits, dim=-<span class="hljs-number">1</span>).item()
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_label = model.config.id2label[predicted_class_ids]
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_label
<span class="hljs-string">&#x27;_unknown_&#x27;</span>`}}),{c(){c=a("p"),b=r("Example:"),u=d(),w(f.$$.fragment)},l(l){c=s(l,"P",{});var h=n(c);b=i(h,"Example:"),h.forEach(t),u=p(l),y(f.$$.fragment,l)},m(l,h){_(l,c,h),e(c,b),_(l,u,h),k(f,l,h),v=!0},p:_e,i(l){v||(T(f.$$.fragment,l),v=!0)},o(l){x(f.$$.fragment,l),v=!1},d(l){l&&t(c),l&&t(u),$(f,l)}}}function vk(W){let c,b;return c=new ve({props:{code:`# compute loss - target_label is e.g. "down"
target_label = model.config.id2label[0]
inputs["labels"] = torch.tensor([model.config.label2id[target_label]])
loss = model(**inputs).loss
round(loss.item(), 2)`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># compute loss - target_label is e.g. &quot;down&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>target_label = model.config.id2label[<span class="hljs-number">0</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs[<span class="hljs-string">&quot;labels&quot;</span>] = torch.tensor([model.config.label2id[target_label]])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
<span class="hljs-number">6.54</span>`}}),{c(){w(c.$$.fragment)},l(u){y(c.$$.fragment,u)},m(u,f){k(c,u,f),b=!0},p:_e,i(u){b||(T(c.$$.fragment,u),b=!0)},o(u){x(c.$$.fragment,u),b=!1},d(u){$(c,u)}}}function bk(W){let c,b,u,f,v;return{c(){c=a("p"),b=r("Although the recipe for forward pass needs to be defined within this function, one should call the "),u=a("code"),f=r("Module"),v=r(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){c=s(l,"P",{});var h=n(c);b=i(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),u=s(h,"CODE",{});var V=n(u);f=i(V,"Module"),V.forEach(t),v=i(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(t)},m(l,h){_(l,c,h),e(c,b),e(c,u),e(u,f),e(c,v)},d(l){l&&t(c)}}}function wk(W){let c,b,u,f,v;return f=new ve({props:{code:`from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForAudioFrameClassification
from datasets import load_dataset
import torch

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("anton-l/wav2vec2-base-superb-sd")
model = Wav2Vec2ForAudioFrameClassification.from_pretrained("anton-l/wav2vec2-base-superb-sd")

# audio file is decoded on the fly
inputs = feature_extractor(dataset[0]["audio"]["array"], return_tensors="pt", sampling_rate=sampling_rate)
with torch.no_grad():
    logits = model(**inputs).logits

probabilities = torch.sigmoid(logits[0])
# labels is a one-hot array of shape (num_frames, num_speakers)
labels = (probabilities > 0.5).long()
labels[0].tolist()`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Wav2Vec2FeatureExtractor, Wav2Vec2ForAudioFrameClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/librispeech_asr_demo&quot;</span>, <span class="hljs-string">&quot;clean&quot;</span>, split=<span class="hljs-string">&quot;validation&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = dataset.sort(<span class="hljs-string">&quot;id&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>sampling_rate = dataset.features[<span class="hljs-string">&quot;audio&quot;</span>].sampling_rate

<span class="hljs-meta">&gt;&gt;&gt; </span>feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(<span class="hljs-string">&quot;anton-l/wav2vec2-base-superb-sd&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Wav2Vec2ForAudioFrameClassification.from_pretrained(<span class="hljs-string">&quot;anton-l/wav2vec2-base-superb-sd&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># audio file is decoded on the fly</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = feature_extractor(dataset[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;array&quot;</span>], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, sampling_rate=sampling_rate)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>probabilities = torch.sigmoid(logits[<span class="hljs-number">0</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># labels is a one-hot array of shape (num_frames, num_speakers)</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = (probabilities &gt; <span class="hljs-number">0.5</span>).long()
<span class="hljs-meta">&gt;&gt;&gt; </span>labels[<span class="hljs-number">0</span>].tolist()
[<span class="hljs-number">0</span>, <span class="hljs-number">0</span>]`}}),{c(){c=a("p"),b=r("Example:"),u=d(),w(f.$$.fragment)},l(l){c=s(l,"P",{});var h=n(c);b=i(h,"Example:"),h.forEach(t),u=p(l),y(f.$$.fragment,l)},m(l,h){_(l,c,h),e(c,b),_(l,u,h),k(f,l,h),v=!0},p:_e,i(l){v||(T(f.$$.fragment,l),v=!0)},o(l){x(f.$$.fragment,l),v=!1},d(l){l&&t(c),l&&t(u),$(f,l)}}}function yk(W){let c,b,u,f,v;return{c(){c=a("p"),b=r("Although the recipe for forward pass needs to be defined within this function, one should call the "),u=a("code"),f=r("Module"),v=r(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){c=s(l,"P",{});var h=n(c);b=i(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),u=s(h,"CODE",{});var V=n(u);f=i(V,"Module"),V.forEach(t),v=i(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(t)},m(l,h){_(l,c,h),e(c,b),e(c,u),e(u,f),e(c,v)},d(l){l&&t(c)}}}function kk(W){let c,b,u,f,v;return f=new ve({props:{code:`from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForXVector
from datasets import load_dataset
import torch

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("anton-l/wav2vec2-base-superb-sv")
model = Wav2Vec2ForXVector.from_pretrained("anton-l/wav2vec2-base-superb-sv")

# audio file is decoded on the fly
inputs = feature_extractor(
    [d["array"] for d in dataset[:2]["audio"]], sampling_rate=sampling_rate, return_tensors="pt", padding=True
)
with torch.no_grad():
    embeddings = model(**inputs).embeddings

embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()

# the resulting embeddings can be used for cosine similarity-based retrieval
cosine_sim = torch.nn.CosineSimilarity(dim=-1)
similarity = cosine_sim(embeddings[0], embeddings[1])
threshold = 0.7  # the optimal threshold is dataset-dependent
if similarity < threshold:
    print("Speakers are not the same!")
round(similarity.item(), 2)`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Wav2Vec2FeatureExtractor, Wav2Vec2ForXVector
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/librispeech_asr_demo&quot;</span>, <span class="hljs-string">&quot;clean&quot;</span>, split=<span class="hljs-string">&quot;validation&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = dataset.sort(<span class="hljs-string">&quot;id&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>sampling_rate = dataset.features[<span class="hljs-string">&quot;audio&quot;</span>].sampling_rate

<span class="hljs-meta">&gt;&gt;&gt; </span>feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(<span class="hljs-string">&quot;anton-l/wav2vec2-base-superb-sv&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Wav2Vec2ForXVector.from_pretrained(<span class="hljs-string">&quot;anton-l/wav2vec2-base-superb-sv&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># audio file is decoded on the fly</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = feature_extractor(
<span class="hljs-meta">... </span>    [d[<span class="hljs-string">&quot;array&quot;</span>] <span class="hljs-keyword">for</span> d <span class="hljs-keyword">in</span> dataset[:<span class="hljs-number">2</span>][<span class="hljs-string">&quot;audio&quot;</span>]], sampling_rate=sampling_rate, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    embeddings = model(**inputs).embeddings

<span class="hljs-meta">&gt;&gt;&gt; </span>embeddings = torch.nn.functional.normalize(embeddings, dim=-<span class="hljs-number">1</span>).cpu()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the resulting embeddings can be used for cosine similarity-based retrieval</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>cosine_sim = torch.nn.CosineSimilarity(dim=-<span class="hljs-number">1</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>similarity = cosine_sim(embeddings[<span class="hljs-number">0</span>], embeddings[<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>threshold = <span class="hljs-number">0.7</span>  <span class="hljs-comment"># the optimal threshold is dataset-dependent</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">if</span> similarity &lt; threshold:
<span class="hljs-meta">... </span>    <span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;Speakers are not the same!&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(similarity.item(), <span class="hljs-number">2</span>)
<span class="hljs-number">0.98</span>`}}),{c(){c=a("p"),b=r("Example:"),u=d(),w(f.$$.fragment)},l(l){c=s(l,"P",{});var h=n(c);b=i(h,"Example:"),h.forEach(t),u=p(l),y(f.$$.fragment,l)},m(l,h){_(l,c,h),e(c,b),_(l,u,h),k(f,l,h),v=!0},p:_e,i(l){v||(T(f.$$.fragment,l),v=!0)},o(l){x(f.$$.fragment,l),v=!1},d(l){l&&t(c),l&&t(u),$(f,l)}}}function Tk(W){let c,b,u,f,v;return{c(){c=a("p"),b=r("Although the recipe for forward pass needs to be defined within this function, one should call the "),u=a("code"),f=r("Module"),v=r(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){c=s(l,"P",{});var h=n(c);b=i(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),u=s(h,"CODE",{});var V=n(u);f=i(V,"Module"),V.forEach(t),v=i(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(t)},m(l,h){_(l,c,h),e(c,b),e(c,u),e(u,f),e(c,v)},d(l){l&&t(c)}}}function xk(W){let c,b,u,f,v;return f=new ve({props:{code:`import torch
from transformers import AutoFeatureExtractor, Wav2Vec2ForPreTraining
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices
from datasets import load_dataset

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
input_values = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt").input_values  # Batch size 1

# compute masked indices
batch_size, raw_sequence_length = input_values.shape
sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length)
mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob=0.2, mask_length=2)
mask_time_indices = torch.tensor(mask_time_indices, device=input_values.device, dtype=torch.long)

with torch.no_grad():
    outputs = model(input_values, mask_time_indices=mask_time_indices)

# compute cosine similarity between predicted (=projected_states) and target (=projected_quantized_states)
cosine_sim = torch.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states, dim=-1)

# show that cosine similarity is much higher than random
cosine_sim[mask_time_indices.to(torch.bool)].mean() > 0.5

# for contrastive loss training model should be put into train mode
model = model.train()
loss = model(input_values, mask_time_indices=mask_time_indices).loss`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoFeatureExtractor, Wav2Vec2ForPreTraining
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers.models.wav2vec2.modeling_wav2vec2 <span class="hljs-keyword">import</span> _compute_mask_indices
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset

<span class="hljs-meta">&gt;&gt;&gt; </span>feature_extractor = AutoFeatureExtractor.from_pretrained(<span class="hljs-string">&quot;facebook/wav2vec2-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Wav2Vec2ForPreTraining.from_pretrained(<span class="hljs-string">&quot;facebook/wav2vec2-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>ds = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/librispeech_asr_dummy&quot;</span>, <span class="hljs-string">&quot;clean&quot;</span>, split=<span class="hljs-string">&quot;validation&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>input_values = feature_extractor(ds[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;array&quot;</span>], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).input_values  <span class="hljs-comment"># Batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># compute masked indices</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>batch_size, raw_sequence_length = input_values.shape
<span class="hljs-meta">&gt;&gt;&gt; </span>sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length)
<span class="hljs-meta">&gt;&gt;&gt; </span>mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob=<span class="hljs-number">0.2</span>, mask_length=<span class="hljs-number">2</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>mask_time_indices = torch.tensor(mask_time_indices, device=input_values.device, dtype=torch.long)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    outputs = model(input_values, mask_time_indices=mask_time_indices)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># compute cosine similarity between predicted (=projected_states) and target (=projected_quantized_states)</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>cosine_sim = torch.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states, dim=-<span class="hljs-number">1</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># show that cosine similarity is much higher than random</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>cosine_sim[mask_time_indices.to(torch.<span class="hljs-built_in">bool</span>)].mean() &gt; <span class="hljs-number">0.5</span>
tensor(<span class="hljs-literal">True</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># for contrastive loss training model should be put into train mode</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = model.train()
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(input_values, mask_time_indices=mask_time_indices).loss`}}),{c(){c=a("p"),b=r("Example:"),u=d(),w(f.$$.fragment)},l(l){c=s(l,"P",{});var h=n(c);b=i(h,"Example:"),h.forEach(t),u=p(l),y(f.$$.fragment,l)},m(l,h){_(l,c,h),e(c,b),_(l,u,h),k(f,l,h),v=!0},p:_e,i(l){v||(T(f.$$.fragment,l),v=!0)},o(l){x(f.$$.fragment,l),v=!1},d(l){l&&t(c),l&&t(u),$(f,l)}}}function $k(W){let c,b,u,f,v,l,h,V,A,M,C,z,L,U,O,E,ae,X,I,N,S,H,D,P,re,se,Te,ie,le,ct,He,B,dt,be,we,Ne,xe,pt,$e,ce,Pe,We,mt;return{c(){c=a("p"),b=r("TF 2.0 models accepts two formats as inputs:"),u=d(),f=a("ul"),v=a("li"),l=r("having all inputs as keyword arguments (like PyTorch models), or"),h=d(),V=a("li"),A=r("having all inputs as a list, tuple or dict in the first positional arguments."),M=d(),C=a("p"),z=r("This second option is useful when using "),L=a("code"),U=r("tf.keras.Model.fit"),O=r(` method which currently requires having all the
tensors in the first argument of the model call function: `),E=a("code"),ae=r("model(inputs)"),X=r("."),I=d(),N=a("p"),S=r(`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),H=d(),D=a("ul"),P=a("li"),re=r("a single Tensor with "),se=a("code"),Te=r("input_values"),ie=r(" only and nothing else: "),le=a("code"),ct=r("model(inputs_ids)"),He=d(),B=a("li"),dt=r(`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),be=a("code"),we=r("model([input_values, attention_mask])"),Ne=r(" or "),xe=a("code"),pt=r("model([input_values, attention_mask, token_type_ids])"),$e=d(),ce=a("li"),Pe=r(`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),We=a("code"),mt=r('model({"input_values": input_values, "token_type_ids": token_type_ids})')},l(j){c=s(j,"P",{});var q=n(c);b=i(q,"TF 2.0 models accepts two formats as inputs:"),q.forEach(t),u=p(j),f=s(j,"UL",{});var Be=n(f);v=s(Be,"LI",{});var Ue=n(v);l=i(Ue,"having all inputs as keyword arguments (like PyTorch models), or"),Ue.forEach(t),h=p(Be),V=s(Be,"LI",{});var zt=n(V);A=i(zt,"having all inputs as a list, tuple or dict in the first positional arguments."),zt.forEach(t),Be.forEach(t),M=p(j),C=s(j,"P",{});var de=n(C);z=i(de,"This second option is useful when using "),L=s(de,"CODE",{});var $t=n(L);U=i($t,"tf.keras.Model.fit"),$t.forEach(t),O=i(de,` method which currently requires having all the
tensors in the first argument of the model call function: `),E=s(de,"CODE",{});var je=n(E);ae=i(je,"model(inputs)"),je.forEach(t),X=i(de,"."),de.forEach(t),I=p(j),N=s(j,"P",{});var Me=n(N);S=i(Me,`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),Me.forEach(t),H=p(j),D=s(j,"UL",{});var J=n(D);P=s(J,"LI",{});var Z=n(P);re=i(Z,"a single Tensor with "),se=s(Z,"CODE",{});var At=n(se);Te=i(At,"input_values"),At.forEach(t),ie=i(Z," only and nothing else: "),le=s(Z,"CODE",{});var ht=n(le);ct=i(ht,"model(inputs_ids)"),ht.forEach(t),Z.forEach(t),He=p(J),B=s(J,"LI",{});var Ve=n(B);dt=i(Ve,`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),be=s(Ve,"CODE",{});var Wt=n(be);we=i(Wt,"model([input_values, attention_mask])"),Wt.forEach(t),Ne=i(Ve," or "),xe=s(Ve,"CODE",{});var K=n(xe);pt=i(K,"model([input_values, attention_mask, token_type_ids])"),K.forEach(t),Ve.forEach(t),$e=p(J),ce=s(J,"LI",{});var Fe=n(ce);Pe=i(Fe,`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),We=s(Fe,"CODE",{});var Ot=n(We);mt=i(Ot,'model({"input_values": input_values, "token_type_ids": token_type_ids})'),Ot.forEach(t),Fe.forEach(t),J.forEach(t)},m(j,q){_(j,c,q),e(c,b),_(j,u,q),_(j,f,q),e(f,v),e(v,l),e(f,h),e(f,V),e(V,A),_(j,M,q),_(j,C,q),e(C,z),e(C,L),e(L,U),e(C,O),e(C,E),e(E,ae),e(C,X),_(j,I,q),_(j,N,q),e(N,S),_(j,H,q),_(j,D,q),e(D,P),e(P,re),e(P,se),e(se,Te),e(P,ie),e(P,le),e(le,ct),e(D,He),e(D,B),e(B,dt),e(B,be),e(be,we),e(B,Ne),e(B,xe),e(xe,pt),e(D,$e),e(D,ce),e(ce,Pe),e(ce,We),e(We,mt)},d(j){j&&t(c),j&&t(u),j&&t(f),j&&t(M),j&&t(C),j&&t(I),j&&t(N),j&&t(H),j&&t(D)}}}function Wk(W){let c,b,u,f,v;return{c(){c=a("p"),b=r("Although the recipe for forward pass needs to be defined within this function, one should call the "),u=a("code"),f=r("Module"),v=r(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){c=s(l,"P",{});var h=n(c);b=i(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),u=s(h,"CODE",{});var V=n(u);f=i(V,"Module"),V.forEach(t),v=i(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(t)},m(l,h){_(l,c,h),e(c,b),e(c,u),e(u,f),e(c,v)},d(l){l&&t(c)}}}function jk(W){let c,b,u,f,v;return f=new ve({props:{code:`from transformers import Wav2Vec2Processor, TFWav2Vec2Model
from datasets import load_dataset
import soundfile as sf

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = TFWav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")


def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch


ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.map(map_to_array)

input_values = processor(ds["speech"][0], return_tensors="tf").input_values  # Batch size 1
hidden_states = model(input_values).last_hidden_state`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Wav2Vec2Processor, TFWav2Vec2Model
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> soundfile <span class="hljs-keyword">as</span> sf

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = Wav2Vec2Processor.from_pretrained(<span class="hljs-string">&quot;facebook/wav2vec2-base-960h&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFWav2Vec2Model.from_pretrained(<span class="hljs-string">&quot;facebook/wav2vec2-base-960h&quot;</span>)


<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">def</span> <span class="hljs-title function_">map_to_array</span>(<span class="hljs-params">batch</span>):
<span class="hljs-meta">... </span>    speech, _ = sf.read(batch[<span class="hljs-string">&quot;file&quot;</span>])
<span class="hljs-meta">... </span>    batch[<span class="hljs-string">&quot;speech&quot;</span>] = speech
<span class="hljs-meta">... </span>    <span class="hljs-keyword">return</span> batch


<span class="hljs-meta">&gt;&gt;&gt; </span>ds = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/librispeech_asr_dummy&quot;</span>, <span class="hljs-string">&quot;clean&quot;</span>, split=<span class="hljs-string">&quot;validation&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>ds = ds.<span class="hljs-built_in">map</span>(map_to_array)

<span class="hljs-meta">&gt;&gt;&gt; </span>input_values = processor(ds[<span class="hljs-string">&quot;speech&quot;</span>][<span class="hljs-number">0</span>], return_tensors=<span class="hljs-string">&quot;tf&quot;</span>).input_values  <span class="hljs-comment"># Batch size 1</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>hidden_states = model(input_values).last_hidden_state`}}),{c(){c=a("p"),b=r("Example:"),u=d(),w(f.$$.fragment)},l(l){c=s(l,"P",{});var h=n(c);b=i(h,"Example:"),h.forEach(t),u=p(l),y(f.$$.fragment,l)},m(l,h){_(l,c,h),e(c,b),_(l,u,h),k(f,l,h),v=!0},p:_e,i(l){v||(T(f.$$.fragment,l),v=!0)},o(l){x(f.$$.fragment,l),v=!1},d(l){l&&t(c),l&&t(u),$(f,l)}}}function Vk(W){let c,b,u,f,v,l,h,V,A,M,C,z,L,U,O,E,ae,X,I,N,S,H,D,P,re,se,Te,ie,le,ct,He,B,dt,be,we,Ne,xe,pt,$e,ce,Pe,We,mt;return{c(){c=a("p"),b=r("TF 2.0 models accepts two formats as inputs:"),u=d(),f=a("ul"),v=a("li"),l=r("having all inputs as keyword arguments (like PyTorch models), or"),h=d(),V=a("li"),A=r("having all inputs as a list, tuple or dict in the first positional arguments."),M=d(),C=a("p"),z=r("This second option is useful when using "),L=a("code"),U=r("tf.keras.Model.fit"),O=r(` method which currently requires having all the
tensors in the first argument of the model call function: `),E=a("code"),ae=r("model(inputs)"),X=r("."),I=d(),N=a("p"),S=r(`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),H=d(),D=a("ul"),P=a("li"),re=r("a single Tensor with "),se=a("code"),Te=r("input_values"),ie=r(" only and nothing else: "),le=a("code"),ct=r("model(inputs_ids)"),He=d(),B=a("li"),dt=r(`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),be=a("code"),we=r("model([input_values, attention_mask])"),Ne=r(" or "),xe=a("code"),pt=r("model([input_values, attention_mask, token_type_ids])"),$e=d(),ce=a("li"),Pe=r(`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),We=a("code"),mt=r('model({"input_values": input_values, "token_type_ids": token_type_ids})')},l(j){c=s(j,"P",{});var q=n(c);b=i(q,"TF 2.0 models accepts two formats as inputs:"),q.forEach(t),u=p(j),f=s(j,"UL",{});var Be=n(f);v=s(Be,"LI",{});var Ue=n(v);l=i(Ue,"having all inputs as keyword arguments (like PyTorch models), or"),Ue.forEach(t),h=p(Be),V=s(Be,"LI",{});var zt=n(V);A=i(zt,"having all inputs as a list, tuple or dict in the first positional arguments."),zt.forEach(t),Be.forEach(t),M=p(j),C=s(j,"P",{});var de=n(C);z=i(de,"This second option is useful when using "),L=s(de,"CODE",{});var $t=n(L);U=i($t,"tf.keras.Model.fit"),$t.forEach(t),O=i(de,` method which currently requires having all the
tensors in the first argument of the model call function: `),E=s(de,"CODE",{});var je=n(E);ae=i(je,"model(inputs)"),je.forEach(t),X=i(de,"."),de.forEach(t),I=p(j),N=s(j,"P",{});var Me=n(N);S=i(Me,`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),Me.forEach(t),H=p(j),D=s(j,"UL",{});var J=n(D);P=s(J,"LI",{});var Z=n(P);re=i(Z,"a single Tensor with "),se=s(Z,"CODE",{});var At=n(se);Te=i(At,"input_values"),At.forEach(t),ie=i(Z," only and nothing else: "),le=s(Z,"CODE",{});var ht=n(le);ct=i(ht,"model(inputs_ids)"),ht.forEach(t),Z.forEach(t),He=p(J),B=s(J,"LI",{});var Ve=n(B);dt=i(Ve,`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),be=s(Ve,"CODE",{});var Wt=n(be);we=i(Wt,"model([input_values, attention_mask])"),Wt.forEach(t),Ne=i(Ve," or "),xe=s(Ve,"CODE",{});var K=n(xe);pt=i(K,"model([input_values, attention_mask, token_type_ids])"),K.forEach(t),Ve.forEach(t),$e=p(J),ce=s(J,"LI",{});var Fe=n(ce);Pe=i(Fe,`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),We=s(Fe,"CODE",{});var Ot=n(We);mt=i(Ot,'model({"input_values": input_values, "token_type_ids": token_type_ids})'),Ot.forEach(t),Fe.forEach(t),J.forEach(t)},m(j,q){_(j,c,q),e(c,b),_(j,u,q),_(j,f,q),e(f,v),e(v,l),e(f,h),e(f,V),e(V,A),_(j,M,q),_(j,C,q),e(C,z),e(C,L),e(L,U),e(C,O),e(C,E),e(E,ae),e(C,X),_(j,I,q),_(j,N,q),e(N,S),_(j,H,q),_(j,D,q),e(D,P),e(P,re),e(P,se),e(se,Te),e(P,ie),e(P,le),e(le,ct),e(D,He),e(D,B),e(B,dt),e(B,be),e(be,we),e(B,Ne),e(B,xe),e(xe,pt),e(D,$e),e(D,ce),e(ce,Pe),e(ce,We),e(We,mt)},d(j){j&&t(c),j&&t(u),j&&t(f),j&&t(M),j&&t(C),j&&t(I),j&&t(N),j&&t(H),j&&t(D)}}}function Fk(W){let c,b,u,f,v;return{c(){c=a("p"),b=r("Although the recipe for forward pass needs to be defined within this function, one should call the "),u=a("code"),f=r("Module"),v=r(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){c=s(l,"P",{});var h=n(c);b=i(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),u=s(h,"CODE",{});var V=n(u);f=i(V,"Module"),V.forEach(t),v=i(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(t)},m(l,h){_(l,c,h),e(c,b),e(c,u),e(u,f),e(c,v)},d(l){l&&t(c)}}}function Ck(W){let c,b,u,f,v;return f=new ve({props:{code:`import tensorflow as tf
from transformers import Wav2Vec2Processor, TFWav2Vec2ForCTC
from datasets import load_dataset
import soundfile as sf

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = TFWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch


ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.map(map_to_array)

input_values = processor(ds["speech"][0], return_tensors="tf").input_values  # Batch size 1
logits = model(input_values).logits
predicted_ids = tf.argmax(logits, axis=-1)

transcription = processor.decode(predicted_ids[0])

# compute loss
target_transcription = "A MAN SAID TO THE UNIVERSE SIR I EXIST"

# Pass transcription as \`text\` to encode labels
labels = processor(text=transcription, return_tensors="tf").input_ids

loss = model(input_values, labels=labels).loss`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> tensorflow <span class="hljs-keyword">as</span> tf
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Wav2Vec2Processor, TFWav2Vec2ForCTC
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> soundfile <span class="hljs-keyword">as</span> sf

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = Wav2Vec2Processor.from_pretrained(<span class="hljs-string">&quot;facebook/wav2vec2-base-960h&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFWav2Vec2ForCTC.from_pretrained(<span class="hljs-string">&quot;facebook/wav2vec2-base-960h&quot;</span>)


<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">def</span> <span class="hljs-title function_">map_to_array</span>(<span class="hljs-params">batch</span>):
<span class="hljs-meta">... </span>    speech, _ = sf.read(batch[<span class="hljs-string">&quot;file&quot;</span>])
<span class="hljs-meta">... </span>    batch[<span class="hljs-string">&quot;speech&quot;</span>] = speech
<span class="hljs-meta">... </span>    <span class="hljs-keyword">return</span> batch


<span class="hljs-meta">&gt;&gt;&gt; </span>ds = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/librispeech_asr_dummy&quot;</span>, <span class="hljs-string">&quot;clean&quot;</span>, split=<span class="hljs-string">&quot;validation&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>ds = ds.<span class="hljs-built_in">map</span>(map_to_array)

<span class="hljs-meta">&gt;&gt;&gt; </span>input_values = processor(ds[<span class="hljs-string">&quot;speech&quot;</span>][<span class="hljs-number">0</span>], return_tensors=<span class="hljs-string">&quot;tf&quot;</span>).input_values  <span class="hljs-comment"># Batch size 1</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = model(input_values).logits
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_ids = tf.argmax(logits, axis=-<span class="hljs-number">1</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>transcription = processor.decode(predicted_ids[<span class="hljs-number">0</span>])

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># compute loss</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>target_transcription = <span class="hljs-string">&quot;A MAN SAID TO THE UNIVERSE SIR I EXIST&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Pass transcription as \`text\` to encode labels</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = processor(text=transcription, return_tensors=<span class="hljs-string">&quot;tf&quot;</span>).input_ids

<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(input_values, labels=labels).loss`}}),{c(){c=a("p"),b=r("Example:"),u=d(),w(f.$$.fragment)},l(l){c=s(l,"P",{});var h=n(c);b=i(h,"Example:"),h.forEach(t),u=p(l),y(f.$$.fragment,l)},m(l,h){_(l,c,h),e(c,b),_(l,u,h),k(f,l,h),v=!0},p:_e,i(l){v||(T(f.$$.fragment,l),v=!0)},o(l){x(f.$$.fragment,l),v=!1},d(l){l&&t(c),l&&t(u),$(f,l)}}}function Ek(W){let c,b,u,f,v;return{c(){c=a("p"),b=r("Although the recipe for forward pass needs to be defined within this function, one should call the "),u=a("code"),f=r("Module"),v=r(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){c=s(l,"P",{});var h=n(c);b=i(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),u=s(h,"CODE",{});var V=n(u);f=i(V,"Module"),V.forEach(t),v=i(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(t)},m(l,h){_(l,c,h),e(c,b),e(c,u),e(u,f),e(c,v)},d(l){l&&t(c)}}}function qk(W){let c,b,u,f,v;return f=new ve({props:{code:`from transformers import Wav2Vec2Processor, FlaxWav2Vec2Model
from datasets import load_dataset
import soundfile as sf

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-lv60")
model = FlaxWav2Vec2Model.from_pretrained("facebook/wav2vec2-large-lv60")


def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch


ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.map(map_to_array)

input_values = processor(
    ds["speech"][0], sampling_rate=16_000, return_tensors="np"
).input_values  # Batch size 1
hidden_states = model(input_values).last_hidden_state`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Wav2Vec2Processor, FlaxWav2Vec2Model
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> soundfile <span class="hljs-keyword">as</span> sf

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = Wav2Vec2Processor.from_pretrained(<span class="hljs-string">&quot;facebook/wav2vec2-large-lv60&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FlaxWav2Vec2Model.from_pretrained(<span class="hljs-string">&quot;facebook/wav2vec2-large-lv60&quot;</span>)


<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">def</span> <span class="hljs-title function_">map_to_array</span>(<span class="hljs-params">batch</span>):
<span class="hljs-meta">... </span>    speech, _ = sf.read(batch[<span class="hljs-string">&quot;file&quot;</span>])
<span class="hljs-meta">... </span>    batch[<span class="hljs-string">&quot;speech&quot;</span>] = speech
<span class="hljs-meta">... </span>    <span class="hljs-keyword">return</span> batch


<span class="hljs-meta">&gt;&gt;&gt; </span>ds = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/librispeech_asr_dummy&quot;</span>, <span class="hljs-string">&quot;clean&quot;</span>, split=<span class="hljs-string">&quot;validation&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>ds = ds.<span class="hljs-built_in">map</span>(map_to_array)

<span class="hljs-meta">&gt;&gt;&gt; </span>input_values = processor(
<span class="hljs-meta">... </span>    ds[<span class="hljs-string">&quot;speech&quot;</span>][<span class="hljs-number">0</span>], sampling_rate=<span class="hljs-number">16_000</span>, return_tensors=<span class="hljs-string">&quot;np&quot;</span>
<span class="hljs-meta">... </span>).input_values  <span class="hljs-comment"># Batch size 1</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>hidden_states = model(input_values).last_hidden_state`}}),{c(){c=a("p"),b=r("Example:"),u=d(),w(f.$$.fragment)},l(l){c=s(l,"P",{});var h=n(c);b=i(h,"Example:"),h.forEach(t),u=p(l),y(f.$$.fragment,l)},m(l,h){_(l,c,h),e(c,b),_(l,u,h),k(f,l,h),v=!0},p:_e,i(l){v||(T(f.$$.fragment,l),v=!0)},o(l){x(f.$$.fragment,l),v=!1},d(l){l&&t(c),l&&t(u),$(f,l)}}}function Pk(W){let c,b,u,f,v;return{c(){c=a("p"),b=r("Although the recipe for forward pass needs to be defined within this function, one should call the "),u=a("code"),f=r("Module"),v=r(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){c=s(l,"P",{});var h=n(c);b=i(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),u=s(h,"CODE",{});var V=n(u);f=i(V,"Module"),V.forEach(t),v=i(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(t)},m(l,h){_(l,c,h),e(c,b),e(c,u),e(u,f),e(c,v)},d(l){l&&t(c)}}}function Mk(W){let c,b,u,f,v;return f=new ve({props:{code:`import jax.numpy as jnp
from transformers import Wav2Vec2Processor, FlaxWav2Vec2ForCTC
from datasets import load_dataset
import soundfile as sf

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60")
model = FlaxWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60")


def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch


ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.map(map_to_array)

input_values = processor(
    ds["speech"][0], sampling_rate=16_000, return_tensors="np"
).input_values  # Batch size 1
logits = model(input_values).logits
predicted_ids = jnp.argmax(logits, axis=-1)

transcription = processor.decode(predicted_ids[0])
# should give:  "A MAN SAID TO THE UNIVERSE SIR I EXIST"`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> jax.numpy <span class="hljs-keyword">as</span> jnp
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Wav2Vec2Processor, FlaxWav2Vec2ForCTC
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> soundfile <span class="hljs-keyword">as</span> sf

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = Wav2Vec2Processor.from_pretrained(<span class="hljs-string">&quot;facebook/wav2vec2-large-960h-lv60&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FlaxWav2Vec2ForCTC.from_pretrained(<span class="hljs-string">&quot;facebook/wav2vec2-large-960h-lv60&quot;</span>)


<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">def</span> <span class="hljs-title function_">map_to_array</span>(<span class="hljs-params">batch</span>):
<span class="hljs-meta">... </span>    speech, _ = sf.read(batch[<span class="hljs-string">&quot;file&quot;</span>])
<span class="hljs-meta">... </span>    batch[<span class="hljs-string">&quot;speech&quot;</span>] = speech
<span class="hljs-meta">... </span>    <span class="hljs-keyword">return</span> batch


<span class="hljs-meta">&gt;&gt;&gt; </span>ds = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/librispeech_asr_dummy&quot;</span>, <span class="hljs-string">&quot;clean&quot;</span>, split=<span class="hljs-string">&quot;validation&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>ds = ds.<span class="hljs-built_in">map</span>(map_to_array)

<span class="hljs-meta">&gt;&gt;&gt; </span>input_values = processor(
<span class="hljs-meta">... </span>    ds[<span class="hljs-string">&quot;speech&quot;</span>][<span class="hljs-number">0</span>], sampling_rate=<span class="hljs-number">16_000</span>, return_tensors=<span class="hljs-string">&quot;np&quot;</span>
<span class="hljs-meta">... </span>).input_values  <span class="hljs-comment"># Batch size 1</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = model(input_values).logits
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_ids = jnp.argmax(logits, axis=-<span class="hljs-number">1</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>transcription = processor.decode(predicted_ids[<span class="hljs-number">0</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># should give:  &quot;A MAN SAID TO THE UNIVERSE SIR I EXIST&quot;</span>`}}),{c(){c=a("p"),b=r("Example:"),u=d(),w(f.$$.fragment)},l(l){c=s(l,"P",{});var h=n(c);b=i(h,"Example:"),h.forEach(t),u=p(l),y(f.$$.fragment,l)},m(l,h){_(l,c,h),e(c,b),_(l,u,h),k(f,l,h),v=!0},p:_e,i(l){v||(T(f.$$.fragment,l),v=!0)},o(l){x(f.$$.fragment,l),v=!1},d(l){l&&t(c),l&&t(u),$(f,l)}}}function zk(W){let c,b,u,f,v;return{c(){c=a("p"),b=r("Although the recipe for forward pass needs to be defined within this function, one should call the "),u=a("code"),f=r("Module"),v=r(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){c=s(l,"P",{});var h=n(c);b=i(h,"Although the recipe for forward pass needs to be defined within this function, one should call the "),u=s(h,"CODE",{});var V=n(u);f=i(V,"Module"),V.forEach(t),v=i(h,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),h.forEach(t)},m(l,h){_(l,c,h),e(c,b),e(c,u),e(u,f),e(c,v)},d(l){l&&t(c)}}}function Ak(W){let c,b,u,f,v;return f=new ve({props:{code:`import optax
import numpy as np
import jax.numpy as jnp
from transformers import Wav2Vec2FeatureExtractor, FlaxWav2Vec2ForPreTraining
from transformers.models.wav2vec2.modeling_flax_wav2vec2 import _compute_mask_indices
from datasets import load_dataset
import soundfile as sf

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-lv60")
model = FlaxWav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-large-lv60")


def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch


ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.map(map_to_array)

input_values = feature_extractor(ds["speech"][0], return_tensors="np").input_values  # Batch size 1

# compute masked indices
batch_size, raw_sequence_length = input_values.shape
sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length)
mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob=0.2, mask_length=2)

outputs = model(input_values, mask_time_indices=mask_time_indices)

# compute cosine similarity between predicted (=projected_states) and target (=projected_quantized_states)
cosine_sim = optax.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states)

# show that cosine similarity is much higher than random
assert np.asarray(cosine_sim)[mask_time_indices].mean() > 0.5`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> optax
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> numpy <span class="hljs-keyword">as</span> np
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> jax.numpy <span class="hljs-keyword">as</span> jnp
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Wav2Vec2FeatureExtractor, FlaxWav2Vec2ForPreTraining
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers.models.wav2vec2.modeling_flax_wav2vec2 <span class="hljs-keyword">import</span> _compute_mask_indices
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> soundfile <span class="hljs-keyword">as</span> sf

<span class="hljs-meta">&gt;&gt;&gt; </span>feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(<span class="hljs-string">&quot;facebook/wav2vec2-large-lv60&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FlaxWav2Vec2ForPreTraining.from_pretrained(<span class="hljs-string">&quot;facebook/wav2vec2-large-lv60&quot;</span>)


<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">def</span> <span class="hljs-title function_">map_to_array</span>(<span class="hljs-params">batch</span>):
<span class="hljs-meta">... </span>    speech, _ = sf.read(batch[<span class="hljs-string">&quot;file&quot;</span>])
<span class="hljs-meta">... </span>    batch[<span class="hljs-string">&quot;speech&quot;</span>] = speech
<span class="hljs-meta">... </span>    <span class="hljs-keyword">return</span> batch


<span class="hljs-meta">&gt;&gt;&gt; </span>ds = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/librispeech_asr_dummy&quot;</span>, <span class="hljs-string">&quot;clean&quot;</span>, split=<span class="hljs-string">&quot;validation&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>ds = ds.<span class="hljs-built_in">map</span>(map_to_array)

<span class="hljs-meta">&gt;&gt;&gt; </span>input_values = feature_extractor(ds[<span class="hljs-string">&quot;speech&quot;</span>][<span class="hljs-number">0</span>], return_tensors=<span class="hljs-string">&quot;np&quot;</span>).input_values  <span class="hljs-comment"># Batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># compute masked indices</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>batch_size, raw_sequence_length = input_values.shape
<span class="hljs-meta">&gt;&gt;&gt; </span>sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length)
<span class="hljs-meta">&gt;&gt;&gt; </span>mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob=<span class="hljs-number">0.2</span>, mask_length=<span class="hljs-number">2</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_values, mask_time_indices=mask_time_indices)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># compute cosine similarity between predicted (=projected_states) and target (=projected_quantized_states)</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>cosine_sim = optax.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># show that cosine similarity is much higher than random</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">assert</span> np.asarray(cosine_sim)[mask_time_indices].mean() &gt; <span class="hljs-number">0.5</span>`}}),{c(){c=a("p"),b=r("Example:"),u=d(),w(f.$$.fragment)},l(l){c=s(l,"P",{});var h=n(c);b=i(h,"Example:"),h.forEach(t),u=p(l),y(f.$$.fragment,l)},m(l,h){_(l,c,h),e(c,b),_(l,u,h),k(f,l,h),v=!0},p:_e,i(l){v||(T(f.$$.fragment,l),v=!0)},o(l){x(f.$$.fragment,l),v=!1},d(l){l&&t(c),l&&t(u),$(f,l)}}}function Ok(W){let c,b,u,f,v,l,h,V,A,M,C,z,L,U,O,E,ae,X,I,N,S,H,D,P,re,se,Te,ie,le,ct,He,B,dt,be,we,Ne,xe,pt,$e,ce,Pe,We,mt,j,q,Be,Ue,zt,de,$t,je,Me,J,Z,At,ht,Ve,Wt,K,Fe,Ot,Lt,Kp,Lr,Qp,Yp,Ga,em,tm,om,Dt,am,Dr,sm,nm,Sr,rm,im,lm,To,ed,St,xo,Ri,Ja,cm,Hi,dm,td,Q,Za,pm,Xi,mm,hm,Ka,fm,Ir,um,gm,_m,$o,Qa,vm,Gi,bm,wm,Nr,Ya,ym,Xe,es,km,Ji,Tm,xm,ts,$m,Zi,Wm,jm,Vm,Wo,Fm,jo,os,Cm,Ki,Em,od,It,Vo,Qi,as,qm,Yi,Pm,ad,Re,ss,Mm,el,zm,Am,ns,Om,Br,Lm,Dm,Sm,Fo,rs,Im,tl,Nm,sd,Nt,Co,ol,is,Bm,al,Um,nd,R,ls,Rm,sl,Hm,Xm,ze,Ur,Gm,Jm,Rr,Zm,Km,Hr,Qm,Ym,cs,nl,eh,th,oh,Xr,ah,sh,nh,Eo,ds,rh,ft,ih,ps,rl,lh,ch,dh,il,ph,mh,ms,ll,hh,fh,uh,gh,qo,hs,_h,ut,vh,Gr,bh,wh,cl,yh,kh,Jr,Th,xh,$h,Zr,fs,Wh,jt,us,jh,gs,Vh,Kr,Fh,Ch,Eh,Po,qh,Mo,_s,Ph,vs,Mh,Qr,zh,Ah,Oh,zo,bs,Lh,ws,Dh,Yr,Sh,Ih,rd,Bt,Ao,dl,ys,Nh,pl,Bh,id,G,ks,Uh,ml,Rh,Hh,Oo,Ts,Xh,gt,Gh,xs,hl,Jh,Zh,Kh,fl,Qh,Yh,$s,ul,ef,tf,of,af,Lo,Ws,sf,_t,nf,ei,rf,lf,gl,cf,df,ti,pf,mf,hf,Vt,js,ff,Vs,uf,oi,gf,_f,vf,Do,bf,ai,Fs,wf,Ge,Cs,yf,_l,kf,Tf,So,xf,Io,$f,Ft,Es,Wf,vl,jf,Vf,No,ld,Ut,Bo,bl,qs,Ff,wl,Cf,cd,Rt,Ps,Ef,Ms,qf,yl,Pf,Mf,dd,Ht,zs,zf,kl,Af,pd,Xt,As,Of,Os,Lf,si,Df,Sf,md,vt,Ls,If,Ds,Nf,Tl,Bf,Uf,Rf,Uo,Ss,Hf,xl,Xf,hd,bt,Is,Gf,Ns,Jf,$l,Zf,Kf,Qf,Ro,Bs,Yf,Wl,eu,fd,Gt,Ho,jl,Us,tu,Vl,ou,ud,Ce,Rs,au,Hs,su,Xs,nu,ru,iu,Gs,lu,ni,cu,du,pu,Js,mu,Zs,hu,fu,uu,Je,Ks,gu,Jt,_u,ri,vu,bu,Fl,wu,yu,ku,Xo,Tu,Go,gd,Zt,Jo,Cl,Qs,xu,El,$u,_d,Ee,Ys,Wu,Kt,ju,ql,Vu,Fu,en,Cu,Eu,qu,tn,Pu,ii,Mu,zu,Au,on,Ou,an,Lu,Du,Su,Ae,sn,Iu,Qt,Nu,li,Bu,Uu,Pl,Ru,Hu,Xu,Zo,Gu,Ko,Ju,Qo,vd,Yt,Yo,Ml,nn,Zu,zl,Ku,bd,pe,rn,Qu,Al,Yu,eg,ln,tg,cn,og,ag,sg,dn,ng,ci,rg,ig,lg,pn,cg,mn,dg,pg,mg,Oe,hn,hg,eo,fg,di,ug,gg,Ol,_g,vg,bg,ea,wg,ta,yg,oa,wd,to,aa,Ll,fn,kg,Dl,Tg,yd,me,un,xg,Sl,$g,Wg,gn,jg,_n,Vg,Fg,Cg,vn,Eg,pi,qg,Pg,Mg,bn,zg,wn,Ag,Og,Lg,Ze,yn,Dg,oo,Sg,mi,Ig,Ng,Il,Bg,Ug,Rg,sa,Hg,na,kd,ao,ra,Nl,kn,Xg,Bl,Gg,Td,he,Tn,Jg,Ul,Zg,Kg,xn,Qg,$n,Yg,e_,t_,Wn,o_,hi,a_,s_,n_,jn,r_,Vn,i_,l_,c_,Ke,Fn,d_,so,p_,fi,m_,h_,Rl,f_,u_,g_,ia,__,la,xd,no,ca,Hl,Cn,v_,Xl,b_,$d,qe,En,w_,ro,y_,Gl,k_,T_,qn,x_,$_,W_,Pn,j_,ui,V_,F_,C_,Mn,E_,zn,q_,P_,M_,Qe,An,z_,io,A_,gi,O_,L_,Jl,D_,S_,I_,da,N_,pa,Wd,lo,ma,Zl,On,B_,Kl,U_,jd,fe,Ln,R_,Ql,H_,X_,Dn,G_,_i,J_,Z_,K_,Sn,Q_,In,Y_,ev,tv,ha,ov,Ye,Nn,av,co,sv,vi,nv,rv,Yl,iv,lv,cv,fa,dv,ua,Vd,po,ga,ec,Bn,pv,tc,mv,Fd,ue,Un,hv,Rn,fv,oc,uv,gv,_v,Hn,vv,bi,bv,wv,yv,Xn,kv,Gn,Tv,xv,$v,_a,Wv,et,Jn,jv,mo,Vv,wi,Fv,Cv,ac,Ev,qv,Pv,va,Mv,ba,Cd,ho,wa,sc,Zn,zv,nc,Av,Ed,Y,Kn,Ov,Qn,Lv,Yn,Dv,Sv,Iv,er,Nv,yi,Bv,Uv,Rv,tr,Hv,or,Xv,Gv,Jv,rc,Zv,Kv,wt,ic,ar,Qv,Yv,lc,sr,e2,t2,cc,nr,o2,a2,dc,rr,s2,n2,tt,ir,r2,fo,i2,pc,l2,c2,mc,d2,p2,m2,ya,h2,ka,qd,uo,Ta,hc,lr,f2,fc,u2,Pd,ee,cr,g2,go,_2,uc,v2,b2,dr,w2,y2,k2,pr,T2,ki,x2,$2,W2,mr,j2,hr,V2,F2,C2,gc,E2,q2,yt,_c,fr,P2,M2,vc,ur,z2,A2,bc,gr,O2,L2,wc,_r,D2,S2,ot,vr,I2,_o,N2,yc,B2,U2,kc,R2,H2,X2,xa,G2,$a,Md,vo,Wa,Tc,br,J2,xc,Z2,zd,te,wr,K2,bo,Q2,$c,Y2,eb,yr,tb,ob,ab,kr,sb,Ti,nb,rb,ib,Tr,lb,xr,cb,db,pb,Wc,mb,hb,kt,jc,$r,fb,ub,Vc,Wr,gb,_b,Fc,jr,vb,bb,Cc,Vr,wb,yb,at,Fr,kb,wo,Tb,xi,xb,$b,Ec,Wb,jb,Vb,ja,Fb,Va,Ad;return l=new oe({}),U=new oe({}),Z=new oe({}),Fe=new F({props:{name:"class transformers.Wav2Vec2Config",anchor:"transformers.Wav2Vec2Config",parameters:[{name:"vocab_size",val:" = 32"},{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout",val:" = 0.1"},{name:"activation_dropout",val:" = 0.1"},{name:"attention_dropout",val:" = 0.1"},{name:"feat_proj_dropout",val:" = 0.0"},{name:"feat_quantizer_dropout",val:" = 0.0"},{name:"final_dropout",val:" = 0.1"},{name:"layerdrop",val:" = 0.1"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-05"},{name:"feat_extract_norm",val:" = 'group'"},{name:"feat_extract_activation",val:" = 'gelu'"},{name:"conv_dim",val:" = (512, 512, 512, 512, 512, 512, 512)"},{name:"conv_stride",val:" = (5, 2, 2, 2, 2, 2, 2)"},{name:"conv_kernel",val:" = (10, 3, 3, 3, 3, 2, 2)"},{name:"conv_bias",val:" = False"},{name:"num_conv_pos_embeddings",val:" = 128"},{name:"num_conv_pos_embedding_groups",val:" = 16"},{name:"do_stable_layer_norm",val:" = False"},{name:"apply_spec_augment",val:" = True"},{name:"mask_time_prob",val:" = 0.05"},{name:"mask_time_length",val:" = 10"},{name:"mask_time_min_masks",val:" = 2"},{name:"mask_feature_prob",val:" = 0.0"},{name:"mask_feature_length",val:" = 10"},{name:"mask_feature_min_masks",val:" = 0"},{name:"num_codevectors_per_group",val:" = 320"},{name:"num_codevector_groups",val:" = 2"},{name:"contrastive_logits_temperature",val:" = 0.1"},{name:"num_negatives",val:" = 100"},{name:"codevector_dim",val:" = 256"},{name:"proj_codevector_dim",val:" = 256"},{name:"diversity_loss_weight",val:" = 0.1"},{name:"ctc_loss_reduction",val:" = 'sum'"},{name:"ctc_zero_infinity",val:" = False"},{name:"use_weighted_layer_sum",val:" = False"},{name:"classifier_proj_size",val:" = 256"},{name:"tdnn_dim",val:" = (512, 512, 512, 512, 1500)"},{name:"tdnn_kernel",val:" = (5, 3, 3, 1, 1)"},{name:"tdnn_dilation",val:" = (1, 2, 3, 1, 1)"},{name:"xvector_output_dim",val:" = 512"},{name:"pad_token_id",val:" = 0"},{name:"bos_token_id",val:" = 1"},{name:"eos_token_id",val:" = 2"},{name:"add_adapter",val:" = False"},{name:"adapter_kernel_size",val:" = 3"},{name:"adapter_stride",val:" = 2"},{name:"num_adapter_layers",val:" = 3"},{name:"output_hidden_size",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Wav2Vec2Config.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Vocabulary size of the Wav2Vec2 model. Defines the number of different tokens that can be represented by
the <code>inputs_ids</code> passed when calling <a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Model">Wav2Vec2Model</a> or <a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.TFWav2Vec2Model">TFWav2Vec2Model</a>. Vocabulary size of the
model. Defines the different tokens that can be represented by the <em>inputs_ids</em> passed to the forward
method of <a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Model">Wav2Vec2Model</a>.`,name:"vocab_size"},{anchor:"transformers.Wav2Vec2Config.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.Wav2Vec2Config.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.Wav2Vec2Config.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.Wav2Vec2Config.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (i.e., feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.Wav2Vec2Config.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;selu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.Wav2Vec2Config.hidden_dropout",description:`<strong>hidden_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout"},{anchor:"transformers.Wav2Vec2Config.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.Wav2Vec2Config.final_dropout",description:`<strong>final_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for the final projection layer of <a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC">Wav2Vec2ForCTC</a>.`,name:"final_dropout"},{anchor:"transformers.Wav2Vec2Config.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.Wav2Vec2Config.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.Wav2Vec2Config.feat_extract_norm",description:`<strong>feat_extract_norm</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;group&quot;</code>) &#x2014;
The norm to be applied to 1D convolutional layers in feature encoder. One of <code>&quot;group&quot;</code> for group
normalization of only the first 1D convolutional layer or <code>&quot;layer&quot;</code> for layer normalization of all 1D
convolutional layers.`,name:"feat_extract_norm"},{anchor:"transformers.Wav2Vec2Config.feat_proj_dropout",description:`<strong>feat_proj_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout probability for output of the feature encoder.`,name:"feat_proj_dropout"},{anchor:"transformers.Wav2Vec2Config.feat_extract_activation",description:"<strong>feat_extract_activation</strong> (<code>str, </code>optional<code>, defaults to </code>&#x201C;gelu&#x201D;<code>) -- The non-linear activation function (function or string) in the 1D convolutional layers of the feature extractor. If string, </code>&#x201C;gelu&#x201D;<code>, </code>&#x201C;relu&#x201D;<code>, </code>&#x201C;selu&#x201D;<code>and</code>&#x201C;gelu_new&#x201D;` are supported.",name:"feat_extract_activation"},{anchor:"transformers.Wav2Vec2Config.feat_quantizer_dropout",description:`<strong>feat_quantizer_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout probabilitiy for quantized feature encoder states.`,name:"feat_quantizer_dropout"},{anchor:"transformers.Wav2Vec2Config.conv_dim",description:`<strong>conv_dim</strong> (<code>Tuple[int]</code> or <code>List[int]</code>, <em>optional</em>, defaults to <code>(512, 512, 512, 512, 512, 512, 512)</code>) &#x2014;
A tuple of integers defining the number of input and output channels of each 1D convolutional layer in the
feature encoder. The length of <em>conv_dim</em> defines the number of 1D convolutional layers.`,name:"conv_dim"},{anchor:"transformers.Wav2Vec2Config.conv_stride",description:`<strong>conv_stride</strong> (<code>Tuple[int]</code> or <code>List[int]</code>, <em>optional</em>, defaults to <code>(5, 2, 2, 2, 2, 2, 2)</code>) &#x2014;
A tuple of integers defining the stride of each 1D convolutional layer in the feature encoder. The length
of <em>conv_stride</em> defines the number of convolutional layers and has to match the length of <em>conv_dim</em>.`,name:"conv_stride"},{anchor:"transformers.Wav2Vec2Config.conv_kernel",description:`<strong>conv_kernel</strong> (<code>Tuple[int]</code> or <code>List[int]</code>, <em>optional</em>, defaults to <code>(10, 3, 3, 3, 3, 3, 3)</code>) &#x2014;
A tuple of integers defining the kernel size of each 1D convolutional layer in the feature encoder. The
length of <em>conv_kernel</em> defines the number of convolutional layers and has to match the length of
<em>conv_dim</em>.`,name:"conv_kernel"},{anchor:"transformers.Wav2Vec2Config.conv_bias",description:`<strong>conv_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether the 1D convolutional layers have a bias.`,name:"conv_bias"},{anchor:"transformers.Wav2Vec2Config.num_conv_pos_embeddings",description:`<strong>num_conv_pos_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
Number of convolutional positional embeddings. Defines the kernel size of 1D convolutional positional
embeddings layer.`,name:"num_conv_pos_embeddings"},{anchor:"transformers.Wav2Vec2Config.num_conv_pos_embedding_groups",description:`<strong>num_conv_pos_embedding_groups</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of groups of 1D convolutional positional embeddings layer.`,name:"num_conv_pos_embedding_groups"},{anchor:"transformers.Wav2Vec2Config.do_stable_layer_norm",description:`<strong>do_stable_layer_norm</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to apply <em>stable</em> layer norm architecture of the Transformer encoder. <code>do_stable_layer_norm is True</code> corresponds to applying layer norm before the attention layer, whereas <code>do_stable_layer_norm is False</code> corresponds to applying layer norm after the attention layer.`,name:"do_stable_layer_norm"},{anchor:"transformers.Wav2Vec2Config.apply_spec_augment",description:`<strong>apply_spec_augment</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to apply <em>SpecAugment</em> data augmentation to the outputs of the feature encoder. For reference see
<a href="https://arxiv.org/abs/1904.08779" rel="nofollow">SpecAugment: A Simple Data Augmentation Method for Automatic Speech
Recognition</a>.`,name:"apply_spec_augment"},{anchor:"transformers.Wav2Vec2Config.mask_time_prob",description:`<strong>mask_time_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.05) &#x2014;
Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked. The masking
procecure generates &#x201D;mask_time_prob<em>len(time_axis)/mask_time_length&#x201D; independent masks over the axis. If
reasoning from the propability of each feature vector to be chosen as the start of the vector span to be
masked, </em>mask_time_prob<em> should be \`prob_vector_start</em>mask_time_length<code>. Note that overlap may decrease the actual percentage of masked vectors. This is only relevant if </code>apply_spec_augment is True\`.`,name:"mask_time_prob"},{anchor:"transformers.Wav2Vec2Config.mask_time_length",description:`<strong>mask_time_length</strong> (<code>int</code>, <em>optional</em>, defaults to 10) &#x2014;
Length of vector span along the time axis.`,name:"mask_time_length"},{anchor:"transformers.Wav2Vec2Config.mask_time_min_masks",description:`<strong>mask_time_min_masks</strong> (<code>int</code>, <em>optional</em>, defaults to 2), &#x2014;
The minimum number of masks of length <code>mask_feature_length</code> generated along the time axis, each time step,
irrespectively of <code>mask_feature_prob</code>. Only relevant if &#x201D;mask_time_prob*len(time_axis)/mask_time_length &lt;
mask_time_min_masks&#x201D;`,name:"mask_time_min_masks"},{anchor:"transformers.Wav2Vec2Config.mask_feature_prob",description:`<strong>mask_feature_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
Percentage (between 0 and 1) of all feature vectors along the feature axis which will be masked. The
masking procecure generates &#x201D;mask_feature_prob<em>len(feature_axis)/mask_time_length&#x201D; independent masks over
the axis. If reasoning from the propability of each feature vector to be chosen as the start of the vector
span to be masked, </em>mask_feature_prob<em> should be \`prob_vector_start</em>mask_feature_length<code>. Note that overlap may decrease the actual percentage of masked vectors. This is only relevant if </code>apply_spec_augment is
True\`.`,name:"mask_feature_prob"},{anchor:"transformers.Wav2Vec2Config.mask_feature_length",description:`<strong>mask_feature_length</strong> (<code>int</code>, <em>optional</em>, defaults to 10) &#x2014;
Length of vector span along the feature axis.`,name:"mask_feature_length"},{anchor:"transformers.Wav2Vec2Config.mask_feature_min_masks",description:`<strong>mask_feature_min_masks</strong> (<code>int</code>, <em>optional</em>, defaults to 0), &#x2014;
The minimum number of masks of length <code>mask_feature_length</code> generated along the feature axis, each time
step, irrespectively of <code>mask_feature_prob</code>. Only relevant if
&#x201D;mask_feature_prob*len(feature_axis)/mask_feature_length &lt; mask_feature_min_masks&#x201D;`,name:"mask_feature_min_masks"},{anchor:"transformers.Wav2Vec2Config.num_codevectors_per_group",description:`<strong>num_codevectors_per_group</strong> (<code>int</code>, <em>optional</em>, defaults to 320) &#x2014;
Number of entries in each quantization codebook (group).`,name:"num_codevectors_per_group"},{anchor:"transformers.Wav2Vec2Config.num_codevector_groups",description:`<strong>num_codevector_groups</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Number of codevector groups for product codevector quantization.`,name:"num_codevector_groups"},{anchor:"transformers.Wav2Vec2Config.contrastive_logits_temperature",description:`<strong>contrastive_logits_temperature</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The temperature <em>kappa</em> in the contrastive loss.`,name:"contrastive_logits_temperature"},{anchor:"transformers.Wav2Vec2Config.feat_quantizer_dropout",description:`<strong>feat_quantizer_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout probabilitiy for the output of the feature encoder that&#x2019;s used by the quantizer.`,name:"feat_quantizer_dropout"},{anchor:"transformers.Wav2Vec2Config.num_negatives",description:`<strong>num_negatives</strong> (<code>int</code>, <em>optional</em>, defaults to 100) &#x2014;
Number of negative samples for the contrastive loss.`,name:"num_negatives"},{anchor:"transformers.Wav2Vec2Config.codevector_dim",description:`<strong>codevector_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 256) &#x2014;
Dimensionality of the quantized feature vectors.`,name:"codevector_dim"},{anchor:"transformers.Wav2Vec2Config.proj_codevector_dim",description:`<strong>proj_codevector_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 256) &#x2014;
Dimensionality of the final projection of both the quantized and the transformer features.`,name:"proj_codevector_dim"},{anchor:"transformers.Wav2Vec2Config.diversity_loss_weight",description:`<strong>diversity_loss_weight</strong> (<code>int</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The weight of the codebook diversity loss component.`,name:"diversity_loss_weight"},{anchor:"transformers.Wav2Vec2Config.ctc_loss_reduction",description:`<strong>ctc_loss_reduction</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;sum&quot;</code>) &#x2014;
Specifies the reduction to apply to the output of <code>torch.nn.CTCLoss</code>. Only relevant when training an
instance of <a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC">Wav2Vec2ForCTC</a>.`,name:"ctc_loss_reduction"},{anchor:"transformers.Wav2Vec2Config.ctc_zero_infinity",description:`<strong>ctc_zero_infinity</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to zero infinite losses and the associated gradients of <code>torch.nn.CTCLoss</code>. Infinite losses mainly
occur when the inputs are too short to be aligned to the targets. Only relevant when training an instance
of <a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC">Wav2Vec2ForCTC</a>.`,name:"ctc_zero_infinity"},{anchor:"transformers.Wav2Vec2Config.use_weighted_layer_sum",description:`<strong>use_weighted_layer_sum</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a weighted average of layer outputs with learned weights. Only relevant when using an
instance of <a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2ForSequenceClassification">Wav2Vec2ForSequenceClassification</a>.`,name:"use_weighted_layer_sum"},{anchor:"transformers.Wav2Vec2Config.classifier_proj_size",description:`<strong>classifier_proj_size</strong> (<code>int</code>, <em>optional</em>, defaults to 256) &#x2014;
Dimensionality of the projection before token mean-pooling for classification.`,name:"classifier_proj_size"},{anchor:"transformers.Wav2Vec2Config.tdnn_dim",description:`<strong>tdnn_dim</strong> (<code>Tuple[int]</code> or <code>List[int]</code>, <em>optional</em>, defaults to <code>(512, 512, 512, 512, 1500)</code>) &#x2014;
A tuple of integers defining the number of output channels of each 1D convolutional layer in the <em>TDNN</em>
module of the <em>XVector</em> model. The length of <em>tdnn_dim</em> defines the number of <em>TDNN</em> layers.`,name:"tdnn_dim"},{anchor:"transformers.Wav2Vec2Config.tdnn_kernel",description:`<strong>tdnn_kernel</strong> (<code>Tuple[int]</code> or <code>List[int]</code>, <em>optional</em>, defaults to <code>(5, 3, 3, 1, 1)</code>) &#x2014;
A tuple of integers defining the kernel size of each 1D convolutional layer in the <em>TDNN</em> module of the
<em>XVector</em> model. The length of <em>tdnn_kernel</em> has to match the length of <em>tdnn_dim</em>.`,name:"tdnn_kernel"},{anchor:"transformers.Wav2Vec2Config.tdnn_dilation",description:`<strong>tdnn_dilation</strong> (<code>Tuple[int]</code> or <code>List[int]</code>, <em>optional</em>, defaults to <code>(1, 2, 3, 1, 1)</code>) &#x2014;
A tuple of integers defining the dilation factor of each 1D convolutional layer in <em>TDNN</em> module of the
<em>XVector</em> model. The length of <em>tdnn_dilation</em> has to match the length of <em>tdnn_dim</em>.`,name:"tdnn_dilation"},{anchor:"transformers.Wav2Vec2Config.xvector_output_dim",description:`<strong>xvector_output_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
Dimensionality of the <em>XVector</em> embedding vectors.`,name:"xvector_output_dim"},{anchor:"transformers.Wav2Vec2Config.add_adapter",description:`<strong>add_adapter</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether a convolutional network should be stacked on top of the Wav2Vec2 Encoder. Can be very useful for
warm-starting Wav2Vec2 for SpeechEncoderDecoder models.`,name:"add_adapter"},{anchor:"transformers.Wav2Vec2Config.adapter_kernel_size",description:`<strong>adapter_kernel_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3) &#x2014;
Kernel size of the convolutional layers in the adapter network. Only relevant if <code>add_adapter is True</code>.`,name:"adapter_kernel_size"},{anchor:"transformers.Wav2Vec2Config.adapter_stride",description:`<strong>adapter_stride</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Stride of the convolutional layers in the adapter network. Only relevant if <code>add_adapter is True</code>.`,name:"adapter_stride"},{anchor:"transformers.Wav2Vec2Config.num_adapter_layers",description:`<strong>num_adapter_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 3) &#x2014;
Number of convolutional layers that should be used in the adapter network. Only relevant if <code>add_adapter is True</code>.`,name:"num_adapter_layers"},{anchor:"transformers.Wav2Vec2Config.output_hidden_size",description:`<strong>output_hidden_size</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Dimensionality of the encoder output layer. If not defined, this defaults to <em>hidden-size</em>. Only relevant
if <code>add_adapter is True</code>.`,name:"output_hidden_size"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/configuration_wav2vec2.py#L32"}}),To=new ge({props:{anchor:"transformers.Wav2Vec2Config.example",$$slots:{default:[sk]},$$scope:{ctx:W}}}),Ja=new oe({}),Za=new F({props:{name:"class transformers.Wav2Vec2CTCTokenizer",anchor:"transformers.Wav2Vec2CTCTokenizer",parameters:[{name:"vocab_file",val:""},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"word_delimiter_token",val:" = '|'"},{name:"replace_word_delimiter_char",val:" = ' '"},{name:"do_lower_case",val:" = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Wav2Vec2CTCTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
File containing the vocabulary.`,name:"vocab_file"},{anchor:"transformers.Wav2Vec2CTCTokenizer.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sentence token.`,name:"bos_token"},{anchor:"transformers.Wav2Vec2CTCTokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sentence token.`,name:"eos_token"},{anchor:"transformers.Wav2Vec2CTCTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.Wav2Vec2CTCTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.Wav2Vec2CTCTokenizer.word_delimiter_token",description:`<strong>word_delimiter_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;|&quot;</code>) &#x2014;
The token used for defining the end of a word.`,name:"word_delimiter_token"},{anchor:"transformers.Wav2Vec2CTCTokenizer.do_lower_case",description:`<strong>do_lower_case</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to accept lowercase input and lowercase the output when decoding.</p>
<p>**kwargs &#x2014;
Additional keyword arguments passed along to <a href="/docs/transformers/pr_18351/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a>`,name:"do_lower_case"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/tokenization_wav2vec2.py#L127"}}),Qa=new F({props:{name:"__call__",anchor:"transformers.Wav2Vec2CTCTokenizer.__call__",parameters:[{name:"text",val:": typing.Union[str, typing.List[str], typing.List[typing.List[str]]] = None"},{name:"text_pair",val:": typing.Union[str, typing.List[str], typing.List[typing.List[str]], NoneType] = None"},{name:"text_target",val:": typing.Union[str, typing.List[str], typing.List[typing.List[str]]] = None"},{name:"text_pair_target",val:": typing.Union[str, typing.List[str], typing.List[typing.List[str]], NoneType] = None"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy] = False"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"is_split_into_words",val:": bool = False"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"return_token_type_ids",val:": typing.Optional[bool] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_overflowing_tokens",val:": bool = False"},{name:"return_special_tokens_mask",val:": bool = False"},{name:"return_offsets_mapping",val:": bool = False"},{name:"return_length",val:": bool = False"},{name:"verbose",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Wav2Vec2CTCTokenizer.__call__.text",description:`<strong>text</strong> (<code>str</code>, <code>List[str]</code>, <code>List[List[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
(pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
<code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text"},{anchor:"transformers.Wav2Vec2CTCTokenizer.__call__.text_pair",description:`<strong>text_pair</strong> (<code>str</code>, <code>List[str]</code>, <code>List[List[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
(pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
<code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text_pair"},{anchor:"transformers.Wav2Vec2CTCTokenizer.__call__.text_target",description:`<strong>text_target</strong> (<code>str</code>, <code>List[str]</code>, <code>List[List[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
you must set <code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text_target"},{anchor:"transformers.Wav2Vec2CTCTokenizer.__call__.text_pair_target",description:`<strong>text_pair_target</strong> (<code>str</code>, <code>List[str]</code>, <code>List[List[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
you must set <code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text_pair_target"},{anchor:"transformers.Wav2Vec2CTCTokenizer.__call__.add_special_tokens",description:`<strong>add_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to encode the sequences with the special tokens relative to their model.`,name:"add_special_tokens"},{anchor:"transformers.Wav2Vec2CTCTokenizer.__call__.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/pr_18351/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls padding. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence if provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.Wav2Vec2CTCTokenizer.__call__.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/pr_18351/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
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
</ul>`,name:"truncation"},{anchor:"transformers.Wav2Vec2CTCTokenizer.__call__.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length to use by one of the truncation/padding parameters.</p>
<p>If left unset or set to <code>None</code>, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.Wav2Vec2CTCTokenizer.__call__.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a number along with <code>max_length</code>, the overflowing tokens returned when
<code>return_overflowing_tokens=True</code> will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.`,name:"stride"},{anchor:"transformers.Wav2Vec2CTCTokenizer.__call__.is_split_into_words",description:`<strong>is_split_into_words</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the input is already pre-tokenized (e.g., split into words). If set to <code>True</code>, the
tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
which it will tokenize. This is useful for NER or token classification.`,name:"is_split_into_words"},{anchor:"transformers.Wav2Vec2CTCTokenizer.__call__.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
the use of Tensor Cores on NVIDIA hardware with compute capability &gt;= 7.5 (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.Wav2Vec2CTCTokenizer.__call__.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/pr_18351/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.Wav2Vec2CTCTokenizer.__call__.return_token_type_ids",description:`<strong>return_token_type_ids</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return token type IDs. If left to the default, will return the token type IDs according to
the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"return_token_type_ids"},{anchor:"transformers.Wav2Vec2CTCTokenizer.__call__.return_attention_mask",description:`<strong>return_attention_mask</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return the attention mask. If left to the default, will return the attention mask according
to the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"return_attention_mask"},{anchor:"transformers.Wav2Vec2CTCTokenizer.__call__.return_overflowing_tokens",description:`<strong>return_overflowing_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
of pairs) is provided with <code>truncation_strategy = longest_first</code> or <code>True</code>, an error is raised instead
of returning overflowing tokens.`,name:"return_overflowing_tokens"},{anchor:"transformers.Wav2Vec2CTCTokenizer.__call__.return_special_tokens_mask",description:`<strong>return_special_tokens_mask</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return special tokens mask information.`,name:"return_special_tokens_mask"},{anchor:"transformers.Wav2Vec2CTCTokenizer.__call__.return_offsets_mapping",description:`<strong>return_offsets_mapping</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return <code>(char_start, char_end)</code> for each token.</p>
<p>This is only available on fast tokenizers inheriting from <a href="/docs/transformers/pr_18351/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a>, if using
Python&#x2019;s tokenizer, this method will raise <code>NotImplementedError</code>.`,name:"return_offsets_mapping"},{anchor:"transformers.Wav2Vec2CTCTokenizer.__call__.return_length",description:`<strong>return_length</strong>  (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return the lengths of the encoded inputs.`,name:"return_length"},{anchor:"transformers.Wav2Vec2CTCTokenizer.__call__.verbose",description:`<strong>verbose</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to print more information and warnings.
**kwargs &#x2014; passed to the <code>self.tokenize()</code> method`,name:"verbose"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/tokenization_utils_base.py#L2401",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18351/en/main_classes/tokenizer#transformers.BatchEncoding"
>BatchEncoding</a> with the following fields:</p>
<ul>
<li>
<p><strong>input_ids</strong> \u2014 List of token ids to be fed to a model.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
</li>
<li>
<p><strong>token_type_ids</strong> \u2014 List of token type ids to be fed to a model (when <code>return_token_type_ids=True</code> or
if <em>\u201Ctoken_type_ids\u201D</em> is in <code>self.model_input_names</code>).</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a></p>
</li>
<li>
<p><strong>attention_mask</strong> \u2014 List of indices specifying which tokens should be attended to by the model (when
<code>return_attention_mask=True</code> or if <em>\u201Cattention_mask\u201D</em> is in <code>self.model_input_names</code>).</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a></p>
</li>
<li>
<p><strong>overflowing_tokens</strong> \u2014 List of overflowing tokens sequences (when a <code>max_length</code> is specified and
<code>return_overflowing_tokens=True</code>).</p>
</li>
<li>
<p><strong>num_truncated_tokens</strong> \u2014 Number of tokens truncated (when a <code>max_length</code> is specified and
<code>return_overflowing_tokens=True</code>).</p>
</li>
<li>
<p><strong>special_tokens_mask</strong> \u2014 List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
regular sequence tokens (when <code>add_special_tokens=True</code> and <code>return_special_tokens_mask=True</code>).</p>
</li>
<li>
<p><strong>length</strong> \u2014 The length of the inputs (when <code>return_length=True</code>)</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18351/en/main_classes/tokenizer#transformers.BatchEncoding"
>BatchEncoding</a></p>
`}}),Ya=new F({props:{name:"save_vocabulary",anchor:"transformers.Wav2Vec2CTCTokenizer.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/tokenization_wav2vec2.py#L597"}}),es=new F({props:{name:"decode",anchor:"transformers.Wav2Vec2CTCTokenizer.decode",parameters:[{name:"token_ids",val:": typing.Union[int, typing.List[int], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')]"},{name:"skip_special_tokens",val:": bool = False"},{name:"clean_up_tokenization_spaces",val:": bool = True"},{name:"output_char_offsets",val:": bool = False"},{name:"output_word_offsets",val:": bool = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Wav2Vec2CTCTokenizer.decode.token_ids",description:`<strong>token_ids</strong> (<code>Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]</code>) &#x2014;
List of tokenized input ids. Can be obtained using the <code>__call__</code> method.`,name:"token_ids"},{anchor:"transformers.Wav2Vec2CTCTokenizer.decode.skip_special_tokens",description:`<strong>skip_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to remove special tokens in the decoding.`,name:"skip_special_tokens"},{anchor:"transformers.Wav2Vec2CTCTokenizer.decode.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to clean up the tokenization spaces.`,name:"clean_up_tokenization_spaces"},{anchor:"transformers.Wav2Vec2CTCTokenizer.decode.output_char_offsets",description:`<strong>output_char_offsets</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to output character offsets. Character offsets can be used in combination with the
sampling rate and model downsampling rate to compute the time-stamps of transcribed characters.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>Please take a look at the example of <code>decode</code> to better
understand how to make use of <code>output_word_offsets</code>.</p>

					</div>`,name:"output_char_offsets"},{anchor:"transformers.Wav2Vec2CTCTokenizer.decode.output_word_offsets",description:`<strong>output_word_offsets</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to output word offsets. Word offsets can be used in combination with the sampling rate
and model downsampling rate to compute the time-stamps of transcribed words.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>Please take a look at the example of <code>decode</code> to better
understand how to make use of <code>output_word_offsets</code>.</p>

					</div>`,name:"output_word_offsets"},{anchor:"transformers.Wav2Vec2CTCTokenizer.decode.kwargs",description:`<strong>kwargs</strong> (additional keyword arguments, <em>optional</em>) &#x2014;
Will be passed to the underlying model specific decode method.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/tokenization_wav2vec2.py#L490",returnDescription:`
<p>The list of decoded
sentences. Will be a <code>Wav2Vec2CTCTokenizerOutput</code> when
<code>output_char_offsets == True</code> or <code>output_word_offsets == True</code>.</p>
`,returnType:`
<p><code>str</code> or <code>Wav2Vec2CTCTokenizerOutput</code></p>
`}}),Wo=new ge({props:{anchor:"transformers.Wav2Vec2CTCTokenizer.decode.example",$$slots:{default:[nk]},$$scope:{ctx:W}}}),os=new F({props:{name:"batch_decode",anchor:"transformers.Wav2Vec2CTCTokenizer.batch_decode",parameters:[{name:"sequences",val:": typing.Union[typing.List[int], typing.List[typing.List[int]], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')]"},{name:"skip_special_tokens",val:": bool = False"},{name:"clean_up_tokenization_spaces",val:": bool = True"},{name:"output_char_offsets",val:": bool = False"},{name:"output_word_offsets",val:": bool = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Wav2Vec2CTCTokenizer.batch_decode.sequences",description:`<strong>sequences</strong> (<code>Union[List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor]</code>) &#x2014;
List of tokenized input ids. Can be obtained using the <code>__call__</code> method.`,name:"sequences"},{anchor:"transformers.Wav2Vec2CTCTokenizer.batch_decode.skip_special_tokens",description:`<strong>skip_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to remove special tokens in the decoding.`,name:"skip_special_tokens"},{anchor:"transformers.Wav2Vec2CTCTokenizer.batch_decode.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to clean up the tokenization spaces.`,name:"clean_up_tokenization_spaces"},{anchor:"transformers.Wav2Vec2CTCTokenizer.batch_decode.output_char_offsets",description:`<strong>output_char_offsets</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to output character offsets. Character offsets can be used in combination with the
sampling rate and model downsampling rate to compute the time-stamps of transcribed characters.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>Please take a look at the Example of <code>decode</code> to better
understand how to make use of <code>output_word_offsets</code>.
<code>batch_decode</code> works the same way with batched output.</p>

					</div>`,name:"output_char_offsets"},{anchor:"transformers.Wav2Vec2CTCTokenizer.batch_decode.output_word_offsets",description:`<strong>output_word_offsets</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to output word offsets. Word offsets can be used in combination with the sampling rate
and model downsampling rate to compute the time-stamps of transcribed words.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>Please take a look at the Example of <code>decode</code> to better
understand how to make use of <code>output_word_offsets</code>.
<code>batch_decode</code> works the same way with batched output.</p>

					</div>`,name:"output_word_offsets"},{anchor:"transformers.Wav2Vec2CTCTokenizer.batch_decode.kwargs",description:`<strong>kwargs</strong> (additional keyword arguments, <em>optional</em>) &#x2014;
Will be passed to the underlying model specific decode method.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/tokenization_wav2vec2.py#L420",returnDescription:`
<p>The list of decoded
sentences. Will be a <code>Wav2Vec2CTCTokenizerOutput</code> when
<code>output_char_offsets == True</code> or <code>output_word_offsets == True</code>.</p>
`,returnType:`
<p><code>List[str]</code> or <code>Wav2Vec2CTCTokenizerOutput</code></p>
`}}),as=new oe({}),ss=new F({props:{name:"class transformers.Wav2Vec2FeatureExtractor",anchor:"transformers.Wav2Vec2FeatureExtractor",parameters:[{name:"feature_size",val:" = 1"},{name:"sampling_rate",val:" = 16000"},{name:"padding_value",val:" = 0.0"},{name:"return_attention_mask",val:" = False"},{name:"do_normalize",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Wav2Vec2FeatureExtractor.feature_size",description:`<strong>feature_size</strong> (<code>int</code>, defaults to 1) &#x2014;
The feature dimension of the extracted features.`,name:"feature_size"},{anchor:"transformers.Wav2Vec2FeatureExtractor.sampling_rate",description:`<strong>sampling_rate</strong> (<code>int</code>, defaults to 16000) &#x2014;
The sampling rate at which the audio files should be digitalized expressed in Hertz per second (Hz).`,name:"sampling_rate"},{anchor:"transformers.Wav2Vec2FeatureExtractor.padding_value",description:`<strong>padding_value</strong> (<code>float</code>, defaults to 0.0) &#x2014;
The value that is used to fill the padding values.`,name:"padding_value"},{anchor:"transformers.Wav2Vec2FeatureExtractor.do_normalize",description:`<strong>do_normalize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to zero-mean unit-variance normalize the input. Normalizing can help to significantly
improve the performance for some models, <em>e.g.</em>,
<a href="https://huggingface.co/models?search=lv60" rel="nofollow">wav2vec2-lv60</a>.`,name:"do_normalize"},{anchor:"transformers.Wav2Vec2FeatureExtractor.return_attention_mask",description:`<strong>return_attention_mask</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not <a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor.__call__"><strong>call</strong>()</a> should return <code>attention_mask</code>.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>Wav2Vec2 models that have set <code>config.feat_extract_norm == &quot;group&quot;</code>, such as
<a href="https://huggingface.co/facebook/wav2vec2-base-960h" rel="nofollow">wav2vec2-base</a>, have <strong>not</strong> been trained using
<code>attention_mask</code>. For such models, <code>input_values</code> should simply be padded with 0 and no <code>attention_mask</code>
should be passed.</p>
<p>For Wav2Vec2 models that have set <code>config.feat_extract_norm == &quot;layer&quot;</code>, such as
<a href="https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self" rel="nofollow">wav2vec2-lv60</a>, <code>attention_mask</code> should be
passed for batched inference.</p>

					</div>`,name:"return_attention_mask"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/feature_extraction_wav2vec2.py#L31"}}),rs=new F({props:{name:"__call__",anchor:"transformers.Wav2Vec2FeatureExtractor.__call__",parameters:[{name:"raw_speech",val:": typing.Union[numpy.ndarray, typing.List[float], typing.List[numpy.ndarray], typing.List[typing.List[float]]]"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"truncation",val:": bool = False"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"sampling_rate",val:": typing.Optional[int] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Wav2Vec2FeatureExtractor.__call__.raw_speech",description:`<strong>raw_speech</strong> (<code>np.ndarray</code>, <code>List[float]</code>, <code>List[np.ndarray]</code>, <code>List[List[float]]</code>) &#x2014;
The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
values, a list of numpy arrays or a list of list of float values.`,name:"raw_speech"},{anchor:"transformers.Wav2Vec2FeatureExtractor.__call__.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/pr_18351/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Select a strategy to pad the returned sequences (according to the model&#x2019;s padding side and padding
index) among:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence if provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.Wav2Vec2FeatureExtractor.__call__.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Maximum length of the returned list and optionally padding length (see above).`,name:"max_length"},{anchor:"transformers.Wav2Vec2FeatureExtractor.__call__.truncation",description:`<strong>truncation</strong> (<code>bool</code>) &#x2014;
Activates truncation to cut input sequences longer than <em>max_length</em> to <em>max_length</em>.`,name:"truncation"},{anchor:"transformers.Wav2Vec2FeatureExtractor.__call__.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value.</p>
<p>This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability</p>
<blockquote>
<p>= 7.5 (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.</p>
</blockquote>`,name:"pad_to_multiple_of"},{anchor:"transformers.Wav2Vec2FeatureExtractor.__call__.return_attention_mask",description:`<strong>return_attention_mask</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return the attention mask. If left to the default, will return the attention mask according
to the specific feature_extractor&#x2019;s default.</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a></p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>Wav2Vec2 models that have set <code>config.feat_extract_norm == &quot;group&quot;</code>, such as
<a href="https://huggingface.co/facebook/wav2vec2-base-960h" rel="nofollow">wav2vec2-base</a>, have <strong>not</strong> been trained using
<code>attention_mask</code>. For such models, <code>input_values</code> should simply be padded with 0 and no
<code>attention_mask</code> should be passed.</p>
<p>For Wav2Vec2 models that have set <code>config.feat_extract_norm == &quot;layer&quot;</code>, such as
<a href="https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self" rel="nofollow">wav2vec2-lv60</a>, <code>attention_mask</code> should
be passed for batched inference.</p>

					</div>`,name:"return_attention_mask"},{anchor:"transformers.Wav2Vec2FeatureExtractor.__call__.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/pr_18351/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.Wav2Vec2FeatureExtractor.__call__.sampling_rate",description:`<strong>sampling_rate</strong> (<code>int</code>, <em>optional</em>) &#x2014;
The sampling rate at which the <code>raw_speech</code> input was sampled. It is strongly recommended to pass
<code>sampling_rate</code> at the forward call to prevent silent errors.`,name:"sampling_rate"},{anchor:"transformers.Wav2Vec2FeatureExtractor.__call__.padding_value",description:"<strong>padding_value</strong> (<code>float</code>, defaults to 0.0) &#x2014;",name:"padding_value"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/feature_extraction_wav2vec2.py#L102"}}),is=new oe({}),ls=new F({props:{name:"class transformers.Wav2Vec2Processor",anchor:"transformers.Wav2Vec2Processor",parameters:[{name:"feature_extractor",val:""},{name:"tokenizer",val:""}],parametersDescription:[{anchor:"transformers.Wav2Vec2Processor.feature_extractor",description:`<strong>feature_extractor</strong> (<code>Wav2Vec2FeatureExtractor</code>) &#x2014;
An instance of <a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor">Wav2Vec2FeatureExtractor</a>. The feature extractor is a required input.`,name:"feature_extractor"},{anchor:"transformers.Wav2Vec2Processor.tokenizer",description:`<strong>tokenizer</strong> (<a href="/docs/transformers/pr_18351/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a>) &#x2014;
An instance of <a href="/docs/transformers/pr_18351/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a>. The tokenizer is a required input.`,name:"tokenizer"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/processing_wav2vec2.py#L26"}}),ds=new F({props:{name:"__call__",anchor:"transformers.Wav2Vec2Processor.__call__",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/processing_wav2vec2.py#L67"}}),hs=new F({props:{name:"pad",anchor:"transformers.Wav2Vec2Processor.pad",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/processing_wav2vec2.py#L104"}}),fs=new F({props:{name:"from_pretrained",anchor:"transformers.Wav2Vec2Processor.from_pretrained",parameters:[{name:"pretrained_model_name_or_path",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/processing_wav2vec2.py#L48"}}),us=new F({props:{name:"save_pretrained",anchor:"transformers.Wav2Vec2Processor.save_pretrained",parameters:[{name:"save_directory",val:""},{name:"push_to_hub",val:": bool = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Wav2Vec2Processor.save_pretrained.save_directory",description:`<strong>save_directory</strong> (<code>str</code> or <code>os.PathLike</code>) &#x2014;
Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
be created if it does not exist).`,name:"save_directory"},{anchor:"transformers.Wav2Vec2Processor.save_pretrained.push_to_hub",description:`<strong>push_to_hub</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
repository you want to push to with <code>repo_id</code> (will default to the name of <code>save_directory</code> in your
namespace).
kwargs &#x2014;
Additional key word arguments passed along to the <a href="/docs/transformers/pr_18351/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub">push_to_hub()</a> method.`,name:"push_to_hub"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/processing_utils.py#L94"}}),Po=new ke({props:{$$slots:{default:[rk]},$$scope:{ctx:W}}}),_s=new F({props:{name:"batch_decode",anchor:"transformers.Wav2Vec2Processor.batch_decode",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/processing_wav2vec2.py#L134"}}),bs=new F({props:{name:"decode",anchor:"transformers.Wav2Vec2Processor.decode",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/processing_wav2vec2.py#L141"}}),ys=new oe({}),ks=new F({props:{name:"class transformers.Wav2Vec2ProcessorWithLM",anchor:"transformers.Wav2Vec2ProcessorWithLM",parameters:[{name:"feature_extractor",val:": FeatureExtractionMixin"},{name:"tokenizer",val:": PreTrainedTokenizerBase"},{name:"decoder",val:": BeamSearchDecoderCTC"}],parametersDescription:[{anchor:"transformers.Wav2Vec2ProcessorWithLM.feature_extractor",description:`<strong>feature_extractor</strong> (<a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor">Wav2Vec2FeatureExtractor</a>) &#x2014;
An instance of <a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor">Wav2Vec2FeatureExtractor</a>. The feature extractor is a required input.`,name:"feature_extractor"},{anchor:"transformers.Wav2Vec2ProcessorWithLM.tokenizer",description:`<strong>tokenizer</strong> (<a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer">Wav2Vec2CTCTokenizer</a>) &#x2014;
An instance of <a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer">Wav2Vec2CTCTokenizer</a>. The tokenizer is a required input.`,name:"tokenizer"},{anchor:"transformers.Wav2Vec2ProcessorWithLM.decoder",description:`<strong>decoder</strong> (<code>pyctcdecode.BeamSearchDecoderCTC</code>) &#x2014;
An instance of <code>pyctcdecode.BeamSearchDecoderCTC</code>. The decoder is a required input.`,name:"decoder"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py#L67"}}),Ts=new F({props:{name:"__call__",anchor:"transformers.Wav2Vec2ProcessorWithLM.__call__",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py#L214"}}),Ws=new F({props:{name:"pad",anchor:"transformers.Wav2Vec2ProcessorWithLM.pad",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py#L252"}}),js=new F({props:{name:"from_pretrained",anchor:"transformers.Wav2Vec2ProcessorWithLM.from_pretrained",parameters:[{name:"pretrained_model_name_or_path",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Wav2Vec2ProcessorWithLM.from_pretrained.pretrained_model_name_or_path",description:`<strong>pretrained_model_name_or_path</strong> (<code>str</code> or <code>os.PathLike</code>) &#x2014;
This can be either:</p>
<ul>
<li>a string, the <em>model id</em> of a pretrained feature_extractor hosted inside a model repo on
huggingface.co. Valid model ids can be located at the root-level, like <code>bert-base-uncased</code>, or
namespaced under a user or organization name, like <code>dbmdz/bert-base-german-cased</code>.</li>
<li>a path to a <em>directory</em> containing a feature extractor file saved using the
<a href="/docs/transformers/pr_18351/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained">save_pretrained()</a> method, e.g., <code>./my_model_directory/</code>.</li>
<li>a path or url to a saved feature extractor JSON <em>file</em>, e.g.,
<code>./my_model_directory/preprocessor_config.json</code>.
**kwargs &#x2014;
Additional keyword arguments passed along to both <a href="/docs/transformers/pr_18351/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor">SequenceFeatureExtractor</a> and
<a href="/docs/transformers/pr_18351/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a></li>
</ul>`,name:"pretrained_model_name_or_path"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py#L112"}}),Do=new ke({props:{$$slots:{default:[ik]},$$scope:{ctx:W}}}),Fs=new F({props:{name:"save_pretrained",anchor:"transformers.Wav2Vec2ProcessorWithLM.save_pretrained",parameters:[{name:"save_directory",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py#L108"}}),Cs=new F({props:{name:"batch_decode",anchor:"transformers.Wav2Vec2ProcessorWithLM.batch_decode",parameters:[{name:"logits",val:": ndarray"},{name:"pool",val:": typing.Union[<bound method BaseContext.Pool of <multiprocessing.context.DefaultContext object at 0x7fbf30f1bf10>>, NoneType] = None"},{name:"num_processes",val:": typing.Optional[int] = None"},{name:"beam_width",val:": typing.Optional[int] = None"},{name:"beam_prune_logp",val:": typing.Optional[float] = None"},{name:"token_min_logp",val:": typing.Optional[float] = None"},{name:"hotwords",val:": typing.Optional[typing.Iterable[str]] = None"},{name:"hotword_weight",val:": typing.Optional[float] = None"},{name:"alpha",val:": typing.Optional[float] = None"},{name:"beta",val:": typing.Optional[float] = None"},{name:"unk_score_offset",val:": typing.Optional[float] = None"},{name:"lm_score_boundary",val:": typing.Optional[bool] = None"},{name:"output_word_offsets",val:": bool = False"}],parametersDescription:[{anchor:"transformers.Wav2Vec2ProcessorWithLM.batch_decode.logits",description:`<strong>logits</strong> (<code>np.ndarray</code>) &#x2014;
The logits output vector of the model representing the log probabilities for each token.`,name:"logits"},{anchor:"transformers.Wav2Vec2ProcessorWithLM.batch_decode.pool",description:`<strong>pool</strong> (<code>multiprocessing.Pool</code>, <em>optional</em>) &#x2014;
An optional user-managed pool. If not set, one will be automatically created and closed. The pool
should be instantiated <em>after</em> <code>Wav2Vec2ProcessorWithLM</code>. Otherwise, the LM won&#x2019;t be available to the
pool&#x2019;s sub-processes.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>Currently, only pools created with a &#x2018;fork&#x2019; context can be used. If a &#x2018;spawn&#x2019; pool is passed, it will
be ignored and sequential decoding will be used instead.</p>

					</div>`,name:"pool"},{anchor:"transformers.Wav2Vec2ProcessorWithLM.batch_decode.num_processes",description:`<strong>num_processes</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If <code>pool</code> is not set, number of processes on which the function should be parallelized over. Defaults
to the number of available CPUs.`,name:"num_processes"},{anchor:"transformers.Wav2Vec2ProcessorWithLM.batch_decode.beam_width",description:`<strong>beam_width</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Maximum number of beams at each step in decoding. Defaults to pyctcdecode&#x2019;s DEFAULT_BEAM_WIDTH.`,name:"beam_width"},{anchor:"transformers.Wav2Vec2ProcessorWithLM.batch_decode.beam_prune_logp",description:`<strong>beam_prune_logp</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Beams that are much worse than best beam will be pruned Defaults to pyctcdecode&#x2019;s DEFAULT_PRUNE_LOGP.`,name:"beam_prune_logp"},{anchor:"transformers.Wav2Vec2ProcessorWithLM.batch_decode.token_min_logp",description:`<strong>token_min_logp</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Tokens below this logp are skipped unless they are argmax of frame Defaults to pyctcdecode&#x2019;s
DEFAULT_MIN_TOKEN_LOGP.`,name:"token_min_logp"},{anchor:"transformers.Wav2Vec2ProcessorWithLM.batch_decode.hotwords",description:`<strong>hotwords</strong> (<code>List[str]</code>, <em>optional</em>) &#x2014;
List of words with extra importance, can be OOV for LM`,name:"hotwords"},{anchor:"transformers.Wav2Vec2ProcessorWithLM.batch_decode.hotword_weight",description:`<strong>hotword_weight</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Weight factor for hotword importance Defaults to pyctcdecode&#x2019;s DEFAULT_HOTWORD_WEIGHT.`,name:"hotword_weight"},{anchor:"transformers.Wav2Vec2ProcessorWithLM.batch_decode.alpha",description:`<strong>alpha</strong> (<code>float</code>, <em>optional</em>) &#x2014;
Weight for language model during shallow fusion`,name:"alpha"},{anchor:"transformers.Wav2Vec2ProcessorWithLM.batch_decode.beta",description:`<strong>beta</strong> (<code>float</code>, <em>optional</em>) &#x2014;
Weight for length score adjustment of during scoring`,name:"beta"},{anchor:"transformers.Wav2Vec2ProcessorWithLM.batch_decode.unk_score_offset",description:`<strong>unk_score_offset</strong> (<code>float</code>, <em>optional</em>) &#x2014;
Amount of log score offset for unknown tokens`,name:"unk_score_offset"},{anchor:"transformers.Wav2Vec2ProcessorWithLM.batch_decode.lm_score_boundary",description:`<strong>lm_score_boundary</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to have kenlm respect boundaries when scoring`,name:"lm_score_boundary"},{anchor:"transformers.Wav2Vec2ProcessorWithLM.batch_decode.output_word_offsets",description:`<strong>output_word_offsets</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to output word offsets. Word offsets can be used in combination with the sampling rate
and model downsampling rate to compute the time-stamps of transcribed words.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>Please take a look at the Example of <code>decode</code> to
better understand how to make use of <code>output_word_offsets</code>.
<code>batch_decode</code> works the same way with batched
output.</p>

					</div>`,name:"output_word_offsets"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py#L283",returnDescription:`
<p><code>Wav2Vec2DecoderWithLMOutput</code>.</p>
`}}),So=new ke({props:{$$slots:{default:[lk]},$$scope:{ctx:W}}}),Io=new ge({props:{anchor:"transformers.Wav2Vec2ProcessorWithLM.batch_decode.example",$$slots:{default:[ck]},$$scope:{ctx:W}}}),Es=new F({props:{name:"decode",anchor:"transformers.Wav2Vec2ProcessorWithLM.decode",parameters:[{name:"logits",val:": ndarray"},{name:"beam_width",val:": typing.Optional[int] = None"},{name:"beam_prune_logp",val:": typing.Optional[float] = None"},{name:"token_min_logp",val:": typing.Optional[float] = None"},{name:"hotwords",val:": typing.Optional[typing.Iterable[str]] = None"},{name:"hotword_weight",val:": typing.Optional[float] = None"},{name:"alpha",val:": typing.Optional[float] = None"},{name:"beta",val:": typing.Optional[float] = None"},{name:"unk_score_offset",val:": typing.Optional[float] = None"},{name:"lm_score_boundary",val:": typing.Optional[bool] = None"},{name:"output_word_offsets",val:": bool = False"}],parametersDescription:[{anchor:"transformers.Wav2Vec2ProcessorWithLM.decode.logits",description:`<strong>logits</strong> (<code>np.ndarray</code>) &#x2014;
The logits output vector of the model representing the log probabilities for each token.`,name:"logits"},{anchor:"transformers.Wav2Vec2ProcessorWithLM.decode.beam_width",description:`<strong>beam_width</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Maximum number of beams at each step in decoding. Defaults to pyctcdecode&#x2019;s DEFAULT_BEAM_WIDTH.`,name:"beam_width"},{anchor:"transformers.Wav2Vec2ProcessorWithLM.decode.beam_prune_logp",description:`<strong>beam_prune_logp</strong> (<code>int</code>, <em>optional</em>) &#x2014;
A threshold to prune beams with log-probs less than best_beam_logp + beam_prune_logp. The value should
be &lt;= 0. Defaults to pyctcdecode&#x2019;s DEFAULT_PRUNE_LOGP.`,name:"beam_prune_logp"},{anchor:"transformers.Wav2Vec2ProcessorWithLM.decode.token_min_logp",description:`<strong>token_min_logp</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Tokens with log-probs below token_min_logp are skipped unless they are have the maximum log-prob for an
utterance. Defaults to pyctcdecode&#x2019;s DEFAULT_MIN_TOKEN_LOGP.`,name:"token_min_logp"},{anchor:"transformers.Wav2Vec2ProcessorWithLM.decode.hotwords",description:`<strong>hotwords</strong> (<code>List[str]</code>, <em>optional</em>) &#x2014;
List of words with extra importance which can be missing from the LM&#x2019;s vocabulary, e.g. [&#x201C;huggingface&#x201D;]`,name:"hotwords"},{anchor:"transformers.Wav2Vec2ProcessorWithLM.decode.hotword_weight",description:`<strong>hotword_weight</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Weight multiplier that boosts hotword scores. Defaults to pyctcdecode&#x2019;s DEFAULT_HOTWORD_WEIGHT.`,name:"hotword_weight"},{anchor:"transformers.Wav2Vec2ProcessorWithLM.decode.alpha",description:`<strong>alpha</strong> (<code>float</code>, <em>optional</em>) &#x2014;
Weight for language model during shallow fusion`,name:"alpha"},{anchor:"transformers.Wav2Vec2ProcessorWithLM.decode.beta",description:`<strong>beta</strong> (<code>float</code>, <em>optional</em>) &#x2014;
Weight for length score adjustment of during scoring`,name:"beta"},{anchor:"transformers.Wav2Vec2ProcessorWithLM.decode.unk_score_offset",description:`<strong>unk_score_offset</strong> (<code>float</code>, <em>optional</em>) &#x2014;
Amount of log score offset for unknown tokens`,name:"unk_score_offset"},{anchor:"transformers.Wav2Vec2ProcessorWithLM.decode.lm_score_boundary",description:`<strong>lm_score_boundary</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to have kenlm respect boundaries when scoring`,name:"lm_score_boundary"},{anchor:"transformers.Wav2Vec2ProcessorWithLM.decode.output_word_offsets",description:`<strong>output_word_offsets</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to output word offsets. Word offsets can be used in combination with the sampling rate
and model downsampling rate to compute the time-stamps of transcribed words.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>Please take a look at the example of <code>decode</code> to
better understand how to make use of <code>output_word_offsets</code>.</p>

					</div>`,name:"output_word_offsets"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py#L488",returnDescription:`
<p><code>Wav2Vec2DecoderWithLMOutput</code>.</p>
`}}),No=new ge({props:{anchor:"transformers.Wav2Vec2ProcessorWithLM.decode.example",$$slots:{default:[dk]},$$scope:{ctx:W}}}),qs=new oe({}),Ps=new F({props:{name:"class transformers.models.wav2vec2_with_lm.processing_wav2vec2_with_lm.Wav2Vec2DecoderWithLMOutput",anchor:"transformers.models.wav2vec2_with_lm.processing_wav2vec2_with_lm.Wav2Vec2DecoderWithLMOutput",parameters:[{name:"text",val:": typing.Union[typing.List[str], str]"},{name:"logit_score",val:": typing.Union[typing.List[float], float] = None"},{name:"lm_score",val:": typing.Union[typing.List[float], float] = None"},{name:"word_offsets",val:": typing.Union[typing.List[typing.List[typing.Dict[str, typing.Union[int, str]]]], typing.List[typing.Dict[str, typing.Union[int, str]]]] = None"}],parametersDescription:[{anchor:"transformers.models.wav2vec2_with_lm.processing_wav2vec2_with_lm.Wav2Vec2DecoderWithLMOutput.text",description:`<strong>text</strong> (list of <code>str</code> or <code>str</code>) &#x2014;
Decoded logits in text from. Usually the speech transcription.`,name:"text"},{anchor:"transformers.models.wav2vec2_with_lm.processing_wav2vec2_with_lm.Wav2Vec2DecoderWithLMOutput.logit_score",description:`<strong>logit_score</strong> (list of <code>float</code> or <code>float</code>) &#x2014;
Total logit score of the beam associated with produced text.`,name:"logit_score"},{anchor:"transformers.models.wav2vec2_with_lm.processing_wav2vec2_with_lm.Wav2Vec2DecoderWithLMOutput.lm_score",description:`<strong>lm_score</strong> (list of <code>float</code>) &#x2014;
Fused lm_score of the beam associated with produced text.`,name:"lm_score"},{anchor:"transformers.models.wav2vec2_with_lm.processing_wav2vec2_with_lm.Wav2Vec2DecoderWithLMOutput.word_offsets",description:`<strong>word_offsets</strong> (list of <code>List[Dict[str, Union[int, str]]]</code> or <code>List[Dict[str, Union[int, str]]]</code>) &#x2014;
Offsets of the decoded words. In combination with sampling rate and model downsampling rate word offsets
can be used to compute time stamps for each word.`,name:"word_offsets"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py#L45"}}),zs=new F({props:{name:"class transformers.modeling_outputs.Wav2Vec2BaseModelOutput",anchor:"transformers.modeling_outputs.Wav2Vec2BaseModelOutput",parameters:[{name:"last_hidden_state",val:": FloatTensor = None"},{name:"extract_features",val:": FloatTensor = None"},{name:"hidden_states",val:": typing.Optional[typing.Tuple[torch.FloatTensor]] = None"},{name:"attentions",val:": typing.Optional[typing.Tuple[torch.FloatTensor]] = None"}],parametersDescription:[{anchor:"transformers.modeling_outputs.Wav2Vec2BaseModelOutput.last_hidden_state",description:`<strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) &#x2014;
Sequence of hidden-states at the output of the last layer of the model.`,name:"last_hidden_state"},{anchor:"transformers.modeling_outputs.Wav2Vec2BaseModelOutput.extract_features",description:`<strong>extract_features</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, conv_dim[-1])</code>) &#x2014;
Sequence of extracted feature vectors of the last convolutional layer of the model.`,name:"extract_features"},{anchor:"transformers.modeling_outputs.Wav2Vec2BaseModelOutput.hidden_states",description:`<strong>hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the initial embedding outputs.`,name:"hidden_states"},{anchor:"transformers.modeling_outputs.Wav2Vec2BaseModelOutput.attentions",description:`<strong>attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.`,name:"attentions"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/modeling_outputs.py#L976"}}),As=new F({props:{name:"class transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput",anchor:"transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput",parameters:[{name:"loss",val:": typing.Optional[torch.FloatTensor] = None"},{name:"projected_states",val:": FloatTensor = None"},{name:"projected_quantized_states",val:": FloatTensor = None"},{name:"codevector_perplexity",val:": FloatTensor = None"},{name:"hidden_states",val:": typing.Optional[typing.Tuple[torch.FloatTensor]] = None"},{name:"attentions",val:": typing.Optional[typing.Tuple[torch.FloatTensor]] = None"},{name:"contrastive_loss",val:": typing.Optional[torch.FloatTensor] = None"},{name:"diversity_loss",val:": typing.Optional[torch.FloatTensor] = None"}],parametersDescription:[{anchor:"transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput.loss",description:`<strong>loss</strong> (<em>optional</em>, returned when <code>sample_negative_indices</code> are passed, <code>torch.FloatTensor</code> of shape <code>(1,)</code>) &#x2014;
Total loss as the sum of the contrastive loss (L_m) and the diversity loss (L_d) as stated in the <a href="https://arxiv.org/pdf/2006.11477.pdf" rel="nofollow">official
paper</a> . (classification) loss.`,name:"loss"},{anchor:"transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput.projected_states",description:`<strong>projected_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.proj_codevector_dim)</code>) &#x2014;
Hidden-states of the model projected to <em>config.proj_codevector_dim</em> that can be used to predict the masked
projected quantized states.`,name:"projected_states"},{anchor:"transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput.projected_quantized_states",description:`<strong>projected_quantized_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.proj_codevector_dim)</code>) &#x2014;
Quantized extracted feature vectors projected to <em>config.proj_codevector_dim</em> representing the positive
target vectors for contrastive loss.`,name:"projected_quantized_states"},{anchor:"transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput.hidden_states",description:`<strong>hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the initial embedding outputs.`,name:"hidden_states"},{anchor:"transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput.attentions",description:`<strong>attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.`,name:"attentions"},{anchor:"transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput.contrastive_loss",description:`<strong>contrastive_loss</strong> (<em>optional</em>, returned when <code>sample_negative_indices</code> are passed, <code>torch.FloatTensor</code> of shape <code>(1,)</code>) &#x2014;
The contrastive loss (L_m) as stated in the <a href="https://arxiv.org/pdf/2006.11477.pdf" rel="nofollow">official paper</a> .`,name:"contrastive_loss"},{anchor:"transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput.diversity_loss",description:`<strong>diversity_loss</strong> (<em>optional</em>, returned when <code>sample_negative_indices</code> are passed, <code>torch.FloatTensor</code> of shape <code>(1,)</code>) &#x2014;
The diversity loss (L_d) as stated in the <a href="https://arxiv.org/pdf/2006.11477.pdf" rel="nofollow">official paper</a> .`,name:"diversity_loss"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L94"}}),Ls=new F({props:{name:"class transformers.models.wav2vec2.modeling_flax_wav2vec2.FlaxWav2Vec2BaseModelOutput",anchor:"transformers.models.wav2vec2.modeling_flax_wav2vec2.FlaxWav2Vec2BaseModelOutput",parameters:[{name:"last_hidden_state",val:": ndarray = None"},{name:"extract_features",val:": ndarray = None"},{name:"hidden_states",val:": typing.Optional[typing.Tuple[jax._src.numpy.ndarray.ndarray]] = None"},{name:"attentions",val:": typing.Optional[typing.Tuple[jax._src.numpy.ndarray.ndarray]] = None"}],parametersDescription:[{anchor:"transformers.models.wav2vec2.modeling_flax_wav2vec2.FlaxWav2Vec2BaseModelOutput.last_hidden_state",description:`<strong>last_hidden_state</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) &#x2014;
Sequence of hidden-states at the output of the last layer of the model.`,name:"last_hidden_state"},{anchor:"transformers.models.wav2vec2.modeling_flax_wav2vec2.FlaxWav2Vec2BaseModelOutput.extract_features",description:`<strong>extract_features</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length, last_conv_dim)</code>) &#x2014;
Sequence of extracted feature vectors of the last convolutional layer of the model with <code>last_conv_dim</code>
being the dimension of the last convolutional layer.`,name:"extract_features"},{anchor:"transformers.models.wav2vec2.modeling_flax_wav2vec2.FlaxWav2Vec2BaseModelOutput.hidden_states",description:`<strong>hidden_states</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>jnp.ndarray</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the initial embedding outputs.`,name:"hidden_states"},{anchor:"transformers.models.wav2vec2.modeling_flax_wav2vec2.FlaxWav2Vec2BaseModelOutput.attentions",description:`<strong>attentions</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>jnp.ndarray</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.`,name:"attentions"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/modeling_flax_wav2vec2.py#L46"}}),Ss=new F({props:{name:"replace",anchor:"transformers.models.wav2vec2.modeling_flax_wav2vec2.FlaxWav2Vec2BaseModelOutput.replace",parameters:[{name:"**updates",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/flax/struct.py#L108"}}),Is=new F({props:{name:"class transformers.models.wav2vec2.modeling_flax_wav2vec2.FlaxWav2Vec2ForPreTrainingOutput",anchor:"transformers.models.wav2vec2.modeling_flax_wav2vec2.FlaxWav2Vec2ForPreTrainingOutput",parameters:[{name:"projected_states",val:": ndarray = None"},{name:"projected_quantized_states",val:": ndarray = None"},{name:"codevector_perplexity",val:": ndarray = None"},{name:"hidden_states",val:": typing.Optional[typing.Tuple[jax._src.numpy.ndarray.ndarray]] = None"},{name:"attentions",val:": typing.Optional[typing.Tuple[jax._src.numpy.ndarray.ndarray]] = None"}],parametersDescription:[{anchor:"transformers.models.wav2vec2.modeling_flax_wav2vec2.FlaxWav2Vec2ForPreTrainingOutput.loss",description:`<strong>loss</strong> (<em>optional</em>, returned when model is in train mode, <code>jnp.ndarray</code> of shape <code>(1,)</code>) &#x2014;
Total loss as the sum of the contrastive loss (L_m) and the diversity loss (L_d) as stated in the <a href="https://arxiv.org/pdf/2006.11477.pdf" rel="nofollow">official
paper</a> . (classification) loss.`,name:"loss"},{anchor:"transformers.models.wav2vec2.modeling_flax_wav2vec2.FlaxWav2Vec2ForPreTrainingOutput.projected_states",description:`<strong>projected_states</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length, config.proj_codevector_dim)</code>) &#x2014;
Hidden-states of the model projected to <em>config.proj_codevector_dim</em> that can be used to predict the masked
projected quantized states.`,name:"projected_states"},{anchor:"transformers.models.wav2vec2.modeling_flax_wav2vec2.FlaxWav2Vec2ForPreTrainingOutput.projected_quantized_states",description:`<strong>projected_quantized_states</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length, config.proj_codevector_dim)</code>) &#x2014;
Quantized extracted feature vectors projected to <em>config.proj_codevector_dim</em> representing the positive
target vectors for contrastive loss.`,name:"projected_quantized_states"},{anchor:"transformers.models.wav2vec2.modeling_flax_wav2vec2.FlaxWav2Vec2ForPreTrainingOutput.hidden_states",description:`<strong>hidden_states</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>jnp.ndarray</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the initial embedding outputs.`,name:"hidden_states"},{anchor:"transformers.models.wav2vec2.modeling_flax_wav2vec2.FlaxWav2Vec2ForPreTrainingOutput.attentions",description:`<strong>attentions</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>jnp.ndarray</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.`,name:"attentions"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/modeling_flax_wav2vec2.py#L76"}}),Bs=new F({props:{name:"replace",anchor:"transformers.models.wav2vec2.modeling_flax_wav2vec2.FlaxWav2Vec2ForPreTrainingOutput.replace",parameters:[{name:"**updates",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/flax/struct.py#L108"}}),Us=new oe({}),Rs=new F({props:{name:"class transformers.Wav2Vec2Model",anchor:"transformers.Wav2Vec2Model",parameters:[{name:"config",val:": Wav2Vec2Config"}],parametersDescription:[{anchor:"transformers.Wav2Vec2Model.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Config">Wav2Vec2Config</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18351/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1182"}}),Ks=new F({props:{name:"forward",anchor:"transformers.Wav2Vec2Model.forward",parameters:[{name:"input_values",val:": typing.Optional[torch.Tensor]"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"mask_time_indices",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.Wav2Vec2Model.forward.input_values",description:`<strong>input_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Float values of input raw speech waveform. Values can be obtained by loading a <em>.flac</em> or <em>.wav</em> audio file
into an array of type <em>List[float]</em> or a <em>numpy.ndarray</em>, <em>e.g.</em> via the soundfile library (<em>pip install
soundfile</em>). To prepare the array into <em>input_values</em>, the <a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor">Wav2Vec2Processor</a> should be used for padding
and conversion into a tensor of type <em>torch.FloatTensor</em>. See <a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__">Wav2Vec2Processor.<strong>call</strong>()</a> for details.`,name:"input_values"},{anchor:"transformers.Wav2Vec2Model.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing convolution and attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a></p>
<div class="course-tip course-tip-orange bg-gradient-to-br dark:bg-gradient-to-r before:border-orange-500 dark:before:border-orange-800 from-orange-50 dark:from-gray-900 to-white dark:to-gray-950 border border-orange-50 text-orange-700 dark:text-gray-400">
						
<p><code>attention_mask</code> should only be passed if the corresponding processor has <code>config.return_attention_mask == True</code>. For all models whose processor has <code>config.return_attention_mask == False</code>, such as
<a href="https://huggingface.co/facebook/wav2vec2-base-960h" rel="nofollow">wav2vec2-base</a>, <code>attention_mask</code> should <strong>not</strong> be
passed to avoid degraded performance when doing batched inference. For such models <code>input_values</code> should
simply be padded with 0 and passed without <code>attention_mask</code>. Be aware that these models also yield slightly
different results depending on whether <code>input_values</code> is padded or not.</p>

					</div>`,name:"attention_mask"},{anchor:"transformers.Wav2Vec2Model.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Wav2Vec2Model.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Wav2Vec2Model.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18351/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1268",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.modeling_outputs.Wav2Vec2BaseModelOutput"
>transformers.modeling_outputs.Wav2Vec2BaseModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Config"
>Wav2Vec2Config</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) \u2014 Sequence of hidden-states at the output of the last layer of the model.</p>
</li>
<li>
<p><strong>extract_features</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, conv_dim[-1])</code>) \u2014 Sequence of extracted feature vectors of the last convolutional layer of the model.</p>
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
  href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.modeling_outputs.Wav2Vec2BaseModelOutput"
>transformers.modeling_outputs.Wav2Vec2BaseModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Xo=new ke({props:{$$slots:{default:[pk]},$$scope:{ctx:W}}}),Go=new ge({props:{anchor:"transformers.Wav2Vec2Model.forward.example",$$slots:{default:[mk]},$$scope:{ctx:W}}}),Qs=new oe({}),Ys=new F({props:{name:"class transformers.Wav2Vec2ForCTC",anchor:"transformers.Wav2Vec2ForCTC",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.Wav2Vec2ForCTC.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Config">Wav2Vec2Config</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18351/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1591"}}),sn=new F({props:{name:"forward",anchor:"transformers.Wav2Vec2ForCTC.forward",parameters:[{name:"input_values",val:": typing.Optional[torch.Tensor]"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.Wav2Vec2ForCTC.forward.input_values",description:`<strong>input_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Float values of input raw speech waveform. Values can be obtained by loading a <em>.flac</em> or <em>.wav</em> audio file
into an array of type <em>List[float]</em> or a <em>numpy.ndarray</em>, <em>e.g.</em> via the soundfile library (<em>pip install
soundfile</em>). To prepare the array into <em>input_values</em>, the <a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor">Wav2Vec2Processor</a> should be used for padding
and conversion into a tensor of type <em>torch.FloatTensor</em>. See <a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__">Wav2Vec2Processor.<strong>call</strong>()</a> for details.`,name:"input_values"},{anchor:"transformers.Wav2Vec2ForCTC.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing convolution and attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a></p>
<div class="course-tip course-tip-orange bg-gradient-to-br dark:bg-gradient-to-r before:border-orange-500 dark:before:border-orange-800 from-orange-50 dark:from-gray-900 to-white dark:to-gray-950 border border-orange-50 text-orange-700 dark:text-gray-400">
						
<p><code>attention_mask</code> should only be passed if the corresponding processor has <code>config.return_attention_mask == True</code>. For all models whose processor has <code>config.return_attention_mask == False</code>, such as
<a href="https://huggingface.co/facebook/wav2vec2-base-960h" rel="nofollow">wav2vec2-base</a>, <code>attention_mask</code> should <strong>not</strong> be
passed to avoid degraded performance when doing batched inference. For such models <code>input_values</code> should
simply be padded with 0 and passed without <code>attention_mask</code>. Be aware that these models also yield slightly
different results depending on whether <code>input_values</code> is padded or not.</p>

					</div>`,name:"attention_mask"},{anchor:"transformers.Wav2Vec2ForCTC.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Wav2Vec2ForCTC.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Wav2Vec2ForCTC.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18351/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.Wav2Vec2ForCTC.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_length)</code>, <em>optional</em>) &#x2014;
Labels for connectionist temporal classification. Note that <code>target_length</code> has to be smaller or equal to
the sequence length of the output logits. Indices are selected in <code>[-100, 0, ..., config.vocab_size - 1]</code>.
All labels set to <code>-100</code> are ignored (masked), the loss is only computed for labels in <code>[0, ..., config.vocab_size - 1]</code>.`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1632",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18351/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput"
>transformers.modeling_outputs.CausalLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Config"
>Wav2Vec2Config</a>) and inputs.</p>
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
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18351/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput"
>transformers.modeling_outputs.CausalLMOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Zo=new ke({props:{$$slots:{default:[hk]},$$scope:{ctx:W}}}),Ko=new ge({props:{anchor:"transformers.Wav2Vec2ForCTC.forward.example",$$slots:{default:[fk]},$$scope:{ctx:W}}}),Qo=new ge({props:{anchor:"transformers.Wav2Vec2ForCTC.forward.example-2",$$slots:{default:[uk]},$$scope:{ctx:W}}}),nn=new oe({}),rn=new F({props:{name:"class transformers.Wav2Vec2ForSequenceClassification",anchor:"transformers.Wav2Vec2ForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.Wav2Vec2ForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Config">Wav2Vec2Config</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18351/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1721"}}),hn=new F({props:{name:"forward",anchor:"transformers.Wav2Vec2ForSequenceClassification.forward",parameters:[{name:"input_values",val:": typing.Optional[torch.Tensor]"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.Wav2Vec2ForSequenceClassification.forward.input_values",description:`<strong>input_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Float values of input raw speech waveform. Values can be obtained by loading a <em>.flac</em> or <em>.wav</em> audio file
into an array of type <em>List[float]</em> or a <em>numpy.ndarray</em>, <em>e.g.</em> via the soundfile library (<em>pip install
soundfile</em>). To prepare the array into <em>input_values</em>, the <a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor">Wav2Vec2Processor</a> should be used for padding
and conversion into a tensor of type <em>torch.FloatTensor</em>. See <a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__">Wav2Vec2Processor.<strong>call</strong>()</a> for details.`,name:"input_values"},{anchor:"transformers.Wav2Vec2ForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing convolution and attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a></p>
<div class="course-tip course-tip-orange bg-gradient-to-br dark:bg-gradient-to-r before:border-orange-500 dark:before:border-orange-800 from-orange-50 dark:from-gray-900 to-white dark:to-gray-950 border border-orange-50 text-orange-700 dark:text-gray-400">
						
<p><code>attention_mask</code> should only be passed if the corresponding processor has <code>config.return_attention_mask == True</code>. For all models whose processor has <code>config.return_attention_mask == False</code>, such as
<a href="https://huggingface.co/facebook/wav2vec2-base-960h" rel="nofollow">wav2vec2-base</a>, <code>attention_mask</code> should <strong>not</strong> be
passed to avoid degraded performance when doing batched inference. For such models <code>input_values</code> should
simply be padded with 0 and passed without <code>attention_mask</code>. Be aware that these models also yield slightly
different results depending on whether <code>input_values</code> is padded or not.</p>

					</div>`,name:"attention_mask"},{anchor:"transformers.Wav2Vec2ForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Wav2Vec2ForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Wav2Vec2ForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18351/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.Wav2Vec2ForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1766",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18351/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Config"
>Wav2Vec2Config</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18351/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ea=new ke({props:{$$slots:{default:[gk]},$$scope:{ctx:W}}}),ta=new ge({props:{anchor:"transformers.Wav2Vec2ForSequenceClassification.forward.example",$$slots:{default:[_k]},$$scope:{ctx:W}}}),oa=new ge({props:{anchor:"transformers.Wav2Vec2ForSequenceClassification.forward.example-2",$$slots:{default:[vk]},$$scope:{ctx:W}}}),fn=new oe({}),un=new F({props:{name:"class transformers.Wav2Vec2ForAudioFrameClassification",anchor:"transformers.Wav2Vec2ForAudioFrameClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.Wav2Vec2ForAudioFrameClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Config">Wav2Vec2Config</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18351/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1844"}}),yn=new F({props:{name:"forward",anchor:"transformers.Wav2Vec2ForAudioFrameClassification.forward",parameters:[{name:"input_values",val:": typing.Optional[torch.Tensor]"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.Wav2Vec2ForAudioFrameClassification.forward.input_values",description:`<strong>input_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Float values of input raw speech waveform. Values can be obtained by loading a <em>.flac</em> or <em>.wav</em> audio file
into an array of type <em>List[float]</em> or a <em>numpy.ndarray</em>, <em>e.g.</em> via the soundfile library (<em>pip install
soundfile</em>). To prepare the array into <em>input_values</em>, the <a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor">Wav2Vec2Processor</a> should be used for padding
and conversion into a tensor of type <em>torch.FloatTensor</em>. See <a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__">Wav2Vec2Processor.<strong>call</strong>()</a> for details.`,name:"input_values"},{anchor:"transformers.Wav2Vec2ForAudioFrameClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing convolution and attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a></p>
<div class="course-tip course-tip-orange bg-gradient-to-br dark:bg-gradient-to-r before:border-orange-500 dark:before:border-orange-800 from-orange-50 dark:from-gray-900 to-white dark:to-gray-950 border border-orange-50 text-orange-700 dark:text-gray-400">
						
<p><code>attention_mask</code> should only be passed if the corresponding processor has <code>config.return_attention_mask == True</code>. For all models whose processor has <code>config.return_attention_mask == False</code>, such as
<a href="https://huggingface.co/facebook/wav2vec2-base-960h" rel="nofollow">wav2vec2-base</a>, <code>attention_mask</code> should <strong>not</strong> be
passed to avoid degraded performance when doing batched inference. For such models <code>input_values</code> should
simply be padded with 0 and passed without <code>attention_mask</code>. Be aware that these models also yield slightly
different results depending on whether <code>input_values</code> is padded or not.</p>

					</div>`,name:"attention_mask"},{anchor:"transformers.Wav2Vec2ForAudioFrameClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Wav2Vec2ForAudioFrameClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Wav2Vec2ForAudioFrameClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18351/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.Wav2Vec2ForAudioFrameClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1888",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18351/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Config"
>Wav2Vec2Config</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18351/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),sa=new ke({props:{$$slots:{default:[bk]},$$scope:{ctx:W}}}),na=new ge({props:{anchor:"transformers.Wav2Vec2ForAudioFrameClassification.forward.example",$$slots:{default:[wk]},$$scope:{ctx:W}}}),kn=new oe({}),Tn=new F({props:{name:"class transformers.Wav2Vec2ForXVector",anchor:"transformers.Wav2Vec2ForXVector",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.Wav2Vec2ForXVector.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Config">Wav2Vec2Config</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18351/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L2006"}}),Fn=new F({props:{name:"forward",anchor:"transformers.Wav2Vec2ForXVector.forward",parameters:[{name:"input_values",val:": typing.Optional[torch.Tensor]"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.Wav2Vec2ForXVector.forward.input_values",description:`<strong>input_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Float values of input raw speech waveform. Values can be obtained by loading a <em>.flac</em> or <em>.wav</em> audio file
into an array of type <em>List[float]</em> or a <em>numpy.ndarray</em>, <em>e.g.</em> via the soundfile library (<em>pip install
soundfile</em>). To prepare the array into <em>input_values</em>, the <a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor">Wav2Vec2Processor</a> should be used for padding
and conversion into a tensor of type <em>torch.FloatTensor</em>. See <a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__">Wav2Vec2Processor.<strong>call</strong>()</a> for details.`,name:"input_values"},{anchor:"transformers.Wav2Vec2ForXVector.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing convolution and attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a></p>
<div class="course-tip course-tip-orange bg-gradient-to-br dark:bg-gradient-to-r before:border-orange-500 dark:before:border-orange-800 from-orange-50 dark:from-gray-900 to-white dark:to-gray-950 border border-orange-50 text-orange-700 dark:text-gray-400">
						
<p><code>attention_mask</code> should only be passed if the corresponding processor has <code>config.return_attention_mask == True</code>. For all models whose processor has <code>config.return_attention_mask == False</code>, such as
<a href="https://huggingface.co/facebook/wav2vec2-base-960h" rel="nofollow">wav2vec2-base</a>, <code>attention_mask</code> should <strong>not</strong> be
passed to avoid degraded performance when doing batched inference. For such models <code>input_values</code> should
simply be padded with 0 and passed without <code>attention_mask</code>. Be aware that these models also yield slightly
different results depending on whether <code>input_values</code> is padded or not.</p>

					</div>`,name:"attention_mask"},{anchor:"transformers.Wav2Vec2ForXVector.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Wav2Vec2ForXVector.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Wav2Vec2ForXVector.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18351/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.Wav2Vec2ForXVector.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L2068",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18351/en/main_classes/output#transformers.modeling_outputs.XVectorOutput"
>transformers.modeling_outputs.XVectorOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Config"
>Wav2Vec2Config</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) \u2014 Classification loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.xvector_output_dim)</code>) \u2014 Classification hidden states before AMSoftmax.</p>
</li>
<li>
<p><strong>embeddings</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.xvector_output_dim)</code>) \u2014 Utterance embeddings used for vector similarity-based retrieval.</p>
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
  href="/docs/transformers/pr_18351/en/main_classes/output#transformers.modeling_outputs.XVectorOutput"
>transformers.modeling_outputs.XVectorOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ia=new ke({props:{$$slots:{default:[yk]},$$scope:{ctx:W}}}),la=new ge({props:{anchor:"transformers.Wav2Vec2ForXVector.forward.example",$$slots:{default:[kk]},$$scope:{ctx:W}}}),Cn=new oe({}),En=new F({props:{name:"class transformers.Wav2Vec2ForPreTraining",anchor:"transformers.Wav2Vec2ForPreTraining",parameters:[{name:"config",val:": Wav2Vec2Config"}],parametersDescription:[{anchor:"transformers.Wav2Vec2ForPreTraining.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Config">Wav2Vec2Config</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18351/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1331"}}),An=new F({props:{name:"forward",anchor:"transformers.Wav2Vec2ForPreTraining.forward",parameters:[{name:"input_values",val:": typing.Optional[torch.Tensor]"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"mask_time_indices",val:": typing.Optional[torch.BoolTensor] = None"},{name:"sampled_negative_indices",val:": typing.Optional[torch.BoolTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.Wav2Vec2ForPreTraining.forward.input_values",description:`<strong>input_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Float values of input raw speech waveform. Values can be obtained by loading a <em>.flac</em> or <em>.wav</em> audio file
into an array of type <em>List[float]</em> or a <em>numpy.ndarray</em>, <em>e.g.</em> via the soundfile library (<em>pip install
soundfile</em>). To prepare the array into <em>input_values</em>, the <a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor">Wav2Vec2Processor</a> should be used for padding
and conversion into a tensor of type <em>torch.FloatTensor</em>. See <a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__">Wav2Vec2Processor.<strong>call</strong>()</a> for details.`,name:"input_values"},{anchor:"transformers.Wav2Vec2ForPreTraining.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing convolution and attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a></p>
<div class="course-tip course-tip-orange bg-gradient-to-br dark:bg-gradient-to-r before:border-orange-500 dark:before:border-orange-800 from-orange-50 dark:from-gray-900 to-white dark:to-gray-950 border border-orange-50 text-orange-700 dark:text-gray-400">
						
<p><code>attention_mask</code> should only be passed if the corresponding processor has <code>config.return_attention_mask == True</code>. For all models whose processor has <code>config.return_attention_mask == False</code>, such as
<a href="https://huggingface.co/facebook/wav2vec2-base-960h" rel="nofollow">wav2vec2-base</a>, <code>attention_mask</code> should <strong>not</strong> be
passed to avoid degraded performance when doing batched inference. For such models <code>input_values</code> should
simply be padded with 0 and passed without <code>attention_mask</code>. Be aware that these models also yield slightly
different results depending on whether <code>input_values</code> is padded or not.</p>

					</div>`,name:"attention_mask"},{anchor:"transformers.Wav2Vec2ForPreTraining.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Wav2Vec2ForPreTraining.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Wav2Vec2ForPreTraining.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18351/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.Wav2Vec2ForPreTraining.forward.mask_time_indices",description:`<strong>mask_time_indices</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict
masked extracted features in <em>config.proj_codevector_dim</em> space.`,name:"mask_time_indices"},{anchor:"transformers.Wav2Vec2ForPreTraining.forward.sampled_negative_indices",description:`<strong>sampled_negative_indices</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, sequence_length, num_negatives)</code>, <em>optional</em>) &#x2014;
Indices indicating which quantized target vectors are used as negative sampled vectors in contrastive loss.
Required input for pre-training.`,name:"sampled_negative_indices"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1392",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput"
>transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Config"
>Wav2Vec2Config</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<em>optional</em>, returned when <code>sample_negative_indices</code> are passed, <code>torch.FloatTensor</code> of shape <code>(1,)</code>) \u2014 Total loss as the sum of the contrastive loss (L_m) and the diversity loss (L_d) as stated in the <a
  href="https://arxiv.org/pdf/2006.11477.pdf"
  rel="nofollow"
>official
paper</a> . (classification) loss.</p>
</li>
<li>
<p><strong>projected_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.proj_codevector_dim)</code>) \u2014 Hidden-states of the model projected to <em>config.proj_codevector_dim</em> that can be used to predict the masked
projected quantized states.</p>
</li>
<li>
<p><strong>projected_quantized_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.proj_codevector_dim)</code>) \u2014 Quantized extracted feature vectors projected to <em>config.proj_codevector_dim</em> representing the positive
target vectors for contrastive loss.</p>
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
<li>
<p><strong>contrastive_loss</strong> (<em>optional</em>, returned when <code>sample_negative_indices</code> are passed, <code>torch.FloatTensor</code> of shape <code>(1,)</code>) \u2014 The contrastive loss (L_m) as stated in the <a
  href="https://arxiv.org/pdf/2006.11477.pdf"
  rel="nofollow"
>official paper</a> .</p>
</li>
<li>
<p><strong>diversity_loss</strong> (<em>optional</em>, returned when <code>sample_negative_indices</code> are passed, <code>torch.FloatTensor</code> of shape <code>(1,)</code>) \u2014 The diversity loss (L_d) as stated in the <a
  href="https://arxiv.org/pdf/2006.11477.pdf"
  rel="nofollow"
>official paper</a> .</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput"
>transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),da=new ke({props:{$$slots:{default:[Tk]},$$scope:{ctx:W}}}),pa=new ge({props:{anchor:"transformers.Wav2Vec2ForPreTraining.forward.example",$$slots:{default:[xk]},$$scope:{ctx:W}}}),On=new oe({}),Ln=new F({props:{name:"class transformers.TFWav2Vec2Model",anchor:"transformers.TFWav2Vec2Model",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFWav2Vec2Model.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Config">Wav2Vec2Config</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18351/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/modeling_tf_wav2vec2.py#L1468"}}),ha=new ke({props:{$$slots:{default:[$k]},$$scope:{ctx:W}}}),Nn=new F({props:{name:"call",anchor:"transformers.TFWav2Vec2Model.call",parameters:[{name:"input_values",val:": Tensor"},{name:"attention_mask",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"position_ids",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"head_mask",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"training",val:": bool = False"}],parametersDescription:[{anchor:"transformers.TFWav2Vec2Model.call.input_values",description:`<strong>input_values</strong> (<code>np.ndarray</code>, <code>tf.Tensor</code>, <code>List[tf.Tensor]</code> \`<code>Dict[str, tf.Tensor]</code> or <code>Dict[str, np.ndarray]</code> and each example must have the shape <code>({0})</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18351/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> and
<a href="/docs/transformers/pr_18351/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_values"},{anchor:"transformers.TFWav2Vec2Model.call.attention_mask",description:`<strong>attention_mask</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>({0})</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.TFWav2Vec2Model.call.token_type_ids",description:`<strong>token_type_ids</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>({0})</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.TFWav2Vec2Model.call.position_ids",description:`<strong>position_ids</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>({0})</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.TFWav2Vec2Model.call.head_mask",description:`<strong>head_mask</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.TFWav2Vec2Model.call.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>({0}, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_values</code> you can choose to directly pass an embedded representation.
This is useful if you want more control over how to convert <code>input_values</code> indices into associated vectors
than the model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.TFWav2Vec2Model.call.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
config will be used instead.`,name:"output_attentions"},{anchor:"transformers.TFWav2Vec2Model.call.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
used instead.`,name:"output_hidden_states"},{anchor:"transformers.TFWav2Vec2Model.call.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18351/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple. This argument can be used in
eager mode, in graph mode the value will always be set to True.`,name:"return_dict"},{anchor:"transformers.TFWav2Vec2Model.call.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to \`False&#x201C;) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/modeling_tf_wav2vec2.py#L1474",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18351/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutput"
>transformers.modeling_tf_outputs.TFBaseModelOutput</a> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<a
  href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Config"
>Wav2Vec2Config</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18351/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutput"
>transformers.modeling_tf_outputs.TFBaseModelOutput</a> or <code>tuple(tf.Tensor)</code></p>
`}}),fa=new ke({props:{$$slots:{default:[Wk]},$$scope:{ctx:W}}}),ua=new ge({props:{anchor:"transformers.TFWav2Vec2Model.call.example",$$slots:{default:[jk]},$$scope:{ctx:W}}}),Bn=new oe({}),Un=new F({props:{name:"class transformers.TFWav2Vec2ForCTC",anchor:"transformers.TFWav2Vec2ForCTC",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFWav2Vec2ForCTC.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Config">Wav2Vec2Config</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18351/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/modeling_tf_wav2vec2.py#L1571"}}),_a=new ke({props:{$$slots:{default:[Vk]},$$scope:{ctx:W}}}),Jn=new F({props:{name:"call",anchor:"transformers.TFWav2Vec2ForCTC.call",parameters:[{name:"input_values",val:": Tensor"},{name:"attention_mask",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"position_ids",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"head_mask",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"labels",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"training",val:": typing.Optional[bool] = False"}],parametersDescription:[{anchor:"transformers.TFWav2Vec2ForCTC.call.input_values",description:`<strong>input_values</strong> (<code>np.ndarray</code>, <code>tf.Tensor</code>, <code>List[tf.Tensor]</code> \`<code>Dict[str, tf.Tensor]</code> or <code>Dict[str, np.ndarray]</code> and each example must have the shape <code>({0})</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18351/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a>. See <a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> and
<a href="/docs/transformers/pr_18351/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_values"},{anchor:"transformers.TFWav2Vec2ForCTC.call.attention_mask",description:`<strong>attention_mask</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>({0})</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.TFWav2Vec2ForCTC.call.token_type_ids",description:`<strong>token_type_ids</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>({0})</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.TFWav2Vec2ForCTC.call.position_ids",description:`<strong>position_ids</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>({0})</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.TFWav2Vec2ForCTC.call.head_mask",description:`<strong>head_mask</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.TFWav2Vec2ForCTC.call.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>np.ndarray</code> or <code>tf.Tensor</code> of shape <code>({0}, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_values</code> you can choose to directly pass an embedded representation.
This is useful if you want more control over how to convert <code>input_values</code> indices into associated vectors
than the model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.TFWav2Vec2ForCTC.call.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
config will be used instead.`,name:"output_attentions"},{anchor:"transformers.TFWav2Vec2ForCTC.call.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
used instead.`,name:"output_hidden_states"},{anchor:"transformers.TFWav2Vec2ForCTC.call.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18351/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple. This argument can be used in
eager mode, in graph mode the value will always be set to True.`,name:"return_dict"},{anchor:"transformers.TFWav2Vec2ForCTC.call.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to \`False&#x201C;) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"},{anchor:"transformers.TFWav2Vec2ForCTC.call.labels",description:`<strong>labels</strong> (<code>tf.Tensor</code> or <code>np.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_values</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked),
the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/modeling_tf_wav2vec2.py#L1598",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18351/en/main_classes/output#transformers.modeling_tf_outputs.TFCausalLMOutput"
>transformers.modeling_tf_outputs.TFCausalLMOutput</a> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<a
  href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Config"
>Wav2Vec2Config</a>) and inputs.</p>
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
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18351/en/main_classes/output#transformers.modeling_tf_outputs.TFCausalLMOutput"
>transformers.modeling_tf_outputs.TFCausalLMOutput</a> or <code>tuple(tf.Tensor)</code></p>
`}}),va=new ke({props:{$$slots:{default:[Fk]},$$scope:{ctx:W}}}),ba=new ge({props:{anchor:"transformers.TFWav2Vec2ForCTC.call.example",$$slots:{default:[Ck]},$$scope:{ctx:W}}}),Zn=new oe({}),Kn=new F({props:{name:"class transformers.FlaxWav2Vec2Model",anchor:"transformers.FlaxWav2Vec2Model",parameters:[{name:"config",val:": Wav2Vec2Config"},{name:"input_shape",val:": typing.Tuple = (1, 1024)"},{name:"seed",val:": int = 0"},{name:"dtype",val:": dtype = <class 'jax.numpy.float32'>"},{name:"_do_init",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.FlaxWav2Vec2Model.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Config">Wav2Vec2Config</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18351/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.FlaxWav2Vec2Model.dtype",description:`<strong>dtype</strong> (<code>jax.numpy.dtype</code>, <em>optional</em>, defaults to <code>jax.numpy.float32</code>) &#x2014;
The data type of the computation. Can be one of <code>jax.numpy.float32</code>, <code>jax.numpy.float16</code> (on GPUs) and
<code>jax.numpy.bfloat16</code> (on TPUs).</p>
<p>This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
specified all the computation will be performed with the given <code>dtype</code>.</p>
<p><strong>Note that this only specifies the dtype of the computation and does not influence the dtype of model
parameters.</strong></p>
<p>If you wish to change the dtype of the model parameters, see <a href="/docs/transformers/pr_18351/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16">to_fp16()</a> and
<a href="/docs/transformers/pr_18351/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16">to_bf16()</a>.`,name:"dtype"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/modeling_flax_wav2vec2.py#L1058"}}),ir=new F({props:{name:"__call__",anchor:"transformers.FlaxWav2Vec2Model.__call__",parameters:[{name:"input_values",val:""},{name:"attention_mask",val:" = None"},{name:"mask_time_indices",val:" = None"},{name:"params",val:": dict = None"},{name:"dropout_rng",val:": PRNGKey = None"},{name:"train",val:": bool = False"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"freeze_feature_encoder",val:": bool = False"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.FlaxWav2Vec2Model.__call__.input_values",description:`<strong>input_values</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Float values of input raw speech waveform. Values can be obtained by loading a <em>.flac</em> or <em>.wav</em> audio file
into an array of type <em>List[float]</em> or a <em>numpy.ndarray</em>, <em>e.g.</em> via the soundfile library (<em>pip install
soundfile</em>). To prepare the array into <em>input_values</em>, the <a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor">Wav2Vec2Processor</a> should be used for padding
and conversion into a tensor of type <em>jnp.ndarray</em>. See <a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__">Wav2Vec2Processor.<strong>call</strong>()</a> for details.`,name:"input_values"},{anchor:"transformers.FlaxWav2Vec2Model.__call__.attention_mask",description:`<strong>attention_mask</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing convolution and attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a> .. warning:: <code>attention_mask</code> should only be passed
if the corresponding processor has <code>config.return_attention_mask == True</code>. For all models whose processor
has <code>config.return_attention_mask == False</code>, such as
<a href="https://huggingface.co/facebook/wav2vec2-base-960h" rel="nofollow">wav2vec2-base</a>, <code>attention_mask</code> should <strong>not</strong> be
passed to avoid degraded performance when doing batched inference. For such models <code>input_values</code> should
simply be padded with 0 and passed without <code>attention_mask</code>. Be aware that these models also yield slightly
different results depending on whether <code>input_values</code> is padded or not.`,name:"attention_mask"},{anchor:"transformers.FlaxWav2Vec2Model.__call__.mask_time_indices",description:`<strong>mask_time_indices</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict
masked extracted features in <em>config.proj_codevector_dim</em> space.`,name:"mask_time_indices"},{anchor:"transformers.FlaxWav2Vec2Model.__call__.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.FlaxWav2Vec2Model.__call__.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.FlaxWav2Vec2Model.__call__.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18351/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/modeling_flax_wav2vec2.py#L890",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.models.wav2vec2.modeling_flax_wav2vec2.FlaxWav2Vec2BaseModelOutput"
>transformers.models.wav2vec2.modeling_flax_wav2vec2.FlaxWav2Vec2BaseModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<code>&lt;class 'transformers.models.wav2vec2.configuration_wav2vec2.Wav2Vec2Config'&gt;</code>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) \u2014 Sequence of hidden-states at the output of the last layer of the model.</p>
</li>
<li>
<p><strong>extract_features</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length, last_conv_dim)</code>) \u2014 Sequence of extracted feature vectors of the last convolutional layer of the model with <code>last_conv_dim</code>
being the dimension of the last convolutional layer.</p>
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
  href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.models.wav2vec2.modeling_flax_wav2vec2.FlaxWav2Vec2BaseModelOutput"
>transformers.models.wav2vec2.modeling_flax_wav2vec2.FlaxWav2Vec2BaseModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ya=new ke({props:{$$slots:{default:[Ek]},$$scope:{ctx:W}}}),ka=new ge({props:{anchor:"transformers.FlaxWav2Vec2Model.__call__.example",$$slots:{default:[qk]},$$scope:{ctx:W}}}),lr=new oe({}),cr=new F({props:{name:"class transformers.FlaxWav2Vec2ForCTC",anchor:"transformers.FlaxWav2Vec2ForCTC",parameters:[{name:"config",val:": Wav2Vec2Config"},{name:"input_shape",val:": typing.Tuple = (1, 1024)"},{name:"seed",val:": int = 0"},{name:"dtype",val:": dtype = <class 'jax.numpy.float32'>"},{name:"_do_init",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.FlaxWav2Vec2ForCTC.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Config">Wav2Vec2Config</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18351/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.FlaxWav2Vec2ForCTC.dtype",description:`<strong>dtype</strong> (<code>jax.numpy.dtype</code>, <em>optional</em>, defaults to <code>jax.numpy.float32</code>) &#x2014;
The data type of the computation. Can be one of <code>jax.numpy.float32</code>, <code>jax.numpy.float16</code> (on GPUs) and
<code>jax.numpy.bfloat16</code> (on TPUs).</p>
<p>This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
specified all the computation will be performed with the given <code>dtype</code>.</p>
<p><strong>Note that this only specifies the dtype of the computation and does not influence the dtype of model
parameters.</strong></p>
<p>If you wish to change the dtype of the model parameters, see <a href="/docs/transformers/pr_18351/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16">to_fp16()</a> and
<a href="/docs/transformers/pr_18351/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16">to_bf16()</a>.`,name:"dtype"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/modeling_flax_wav2vec2.py#L1176"}}),vr=new F({props:{name:"__call__",anchor:"transformers.FlaxWav2Vec2ForCTC.__call__",parameters:[{name:"input_values",val:""},{name:"attention_mask",val:" = None"},{name:"mask_time_indices",val:" = None"},{name:"params",val:": dict = None"},{name:"dropout_rng",val:": PRNGKey = None"},{name:"train",val:": bool = False"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"freeze_feature_encoder",val:": bool = False"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.FlaxWav2Vec2ForCTC.__call__.input_values",description:`<strong>input_values</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Float values of input raw speech waveform. Values can be obtained by loading a <em>.flac</em> or <em>.wav</em> audio file
into an array of type <em>List[float]</em> or a <em>numpy.ndarray</em>, <em>e.g.</em> via the soundfile library (<em>pip install
soundfile</em>). To prepare the array into <em>input_values</em>, the <a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor">Wav2Vec2Processor</a> should be used for padding
and conversion into a tensor of type <em>jnp.ndarray</em>. See <a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__">Wav2Vec2Processor.<strong>call</strong>()</a> for details.`,name:"input_values"},{anchor:"transformers.FlaxWav2Vec2ForCTC.__call__.attention_mask",description:`<strong>attention_mask</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing convolution and attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a> .. warning:: <code>attention_mask</code> should only be passed
if the corresponding processor has <code>config.return_attention_mask == True</code>. For all models whose processor
has <code>config.return_attention_mask == False</code>, such as
<a href="https://huggingface.co/facebook/wav2vec2-base-960h" rel="nofollow">wav2vec2-base</a>, <code>attention_mask</code> should <strong>not</strong> be
passed to avoid degraded performance when doing batched inference. For such models <code>input_values</code> should
simply be padded with 0 and passed without <code>attention_mask</code>. Be aware that these models also yield slightly
different results depending on whether <code>input_values</code> is padded or not.`,name:"attention_mask"},{anchor:"transformers.FlaxWav2Vec2ForCTC.__call__.mask_time_indices",description:`<strong>mask_time_indices</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict
masked extracted features in <em>config.proj_codevector_dim</em> space.`,name:"mask_time_indices"},{anchor:"transformers.FlaxWav2Vec2ForCTC.__call__.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.FlaxWav2Vec2ForCTC.__call__.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.FlaxWav2Vec2ForCTC.__call__.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18351/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/modeling_flax_wav2vec2.py#L890",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18351/en/main_classes/output#transformers.modeling_flax_outputs.FlaxMaskedLMOutput"
>transformers.modeling_flax_outputs.FlaxMaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<code>&lt;class 'transformers.models.wav2vec2.configuration_wav2vec2.Wav2Vec2Config'&gt;</code>) and inputs.</p>
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
  href="/docs/transformers/pr_18351/en/main_classes/output#transformers.modeling_flax_outputs.FlaxMaskedLMOutput"
>transformers.modeling_flax_outputs.FlaxMaskedLMOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),xa=new ke({props:{$$slots:{default:[Pk]},$$scope:{ctx:W}}}),$a=new ge({props:{anchor:"transformers.FlaxWav2Vec2ForCTC.__call__.example",$$slots:{default:[Mk]},$$scope:{ctx:W}}}),br=new oe({}),wr=new F({props:{name:"class transformers.FlaxWav2Vec2ForPreTraining",anchor:"transformers.FlaxWav2Vec2ForPreTraining",parameters:[{name:"config",val:": Wav2Vec2Config"},{name:"input_shape",val:": typing.Tuple = (1, 1024)"},{name:"seed",val:": int = 0"},{name:"dtype",val:": dtype = <class 'jax.numpy.float32'>"},{name:"_do_init",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.FlaxWav2Vec2ForPreTraining.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Config">Wav2Vec2Config</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18351/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.FlaxWav2Vec2ForPreTraining.dtype",description:`<strong>dtype</strong> (<code>jax.numpy.dtype</code>, <em>optional</em>, defaults to <code>jax.numpy.float32</code>) &#x2014;
The data type of the computation. Can be one of <code>jax.numpy.float32</code>, <code>jax.numpy.float16</code> (on GPUs) and
<code>jax.numpy.bfloat16</code> (on TPUs).</p>
<p>This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
specified all the computation will be performed with the given <code>dtype</code>.</p>
<p><strong>Note that this only specifies the dtype of the computation and does not influence the dtype of model
parameters.</strong></p>
<p>If you wish to change the dtype of the model parameters, see <a href="/docs/transformers/pr_18351/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16">to_fp16()</a> and
<a href="/docs/transformers/pr_18351/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16">to_bf16()</a>.`,name:"dtype"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/modeling_flax_wav2vec2.py#L1322"}}),Fr=new F({props:{name:"__call__",anchor:"transformers.FlaxWav2Vec2ForPreTraining.__call__",parameters:[{name:"input_values",val:""},{name:"attention_mask",val:" = None"},{name:"mask_time_indices",val:" = None"},{name:"gumbel_temperature",val:": int = 1"},{name:"params",val:": dict = None"},{name:"dropout_rng",val:": PRNGKey = None"},{name:"gumbel_rng",val:": PRNGKey = None"},{name:"train",val:": bool = False"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"freeze_feature_encoder",val:": bool = False"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.FlaxWav2Vec2ForPreTraining.__call__.input_values",description:`<strong>input_values</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Float values of input raw speech waveform. Values can be obtained by loading a <em>.flac</em> or <em>.wav</em> audio file
into an array of type <em>List[float]</em> or a <em>numpy.ndarray</em>, <em>e.g.</em> via the soundfile library (<em>pip install
soundfile</em>). To prepare the array into <em>input_values</em>, the <a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor">Wav2Vec2Processor</a> should be used for padding
and conversion into a tensor of type <em>jnp.ndarray</em>. See <a href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__">Wav2Vec2Processor.<strong>call</strong>()</a> for details.`,name:"input_values"},{anchor:"transformers.FlaxWav2Vec2ForPreTraining.__call__.attention_mask",description:`<strong>attention_mask</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing convolution and attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a> .. warning:: <code>attention_mask</code> should only be passed
if the corresponding processor has <code>config.return_attention_mask == True</code>. For all models whose processor
has <code>config.return_attention_mask == False</code>, such as
<a href="https://huggingface.co/facebook/wav2vec2-base-960h" rel="nofollow">wav2vec2-base</a>, <code>attention_mask</code> should <strong>not</strong> be
passed to avoid degraded performance when doing batched inference. For such models <code>input_values</code> should
simply be padded with 0 and passed without <code>attention_mask</code>. Be aware that these models also yield slightly
different results depending on whether <code>input_values</code> is padded or not.`,name:"attention_mask"},{anchor:"transformers.FlaxWav2Vec2ForPreTraining.__call__.mask_time_indices",description:`<strong>mask_time_indices</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict
masked extracted features in <em>config.proj_codevector_dim</em> space.`,name:"mask_time_indices"},{anchor:"transformers.FlaxWav2Vec2ForPreTraining.__call__.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.FlaxWav2Vec2ForPreTraining.__call__.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.FlaxWav2Vec2ForPreTraining.__call__.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18351/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18351/src/transformers/models/wav2vec2/modeling_flax_wav2vec2.py#L1325",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.models.wav2vec2.modeling_flax_wav2vec2.FlaxWav2Vec2ForPreTrainingOutput"
>transformers.models.wav2vec2.modeling_flax_wav2vec2.FlaxWav2Vec2ForPreTrainingOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<code>&lt;class 'transformers.models.wav2vec2.configuration_wav2vec2.Wav2Vec2Config'&gt;</code>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<em>optional</em>, returned when model is in train mode, <code>jnp.ndarray</code> of shape <code>(1,)</code>) \u2014 Total loss as the sum of the contrastive loss (L_m) and the diversity loss (L_d) as stated in the <a
  href="https://arxiv.org/pdf/2006.11477.pdf"
  rel="nofollow"
>official
paper</a> . (classification) loss.</p>
</li>
<li>
<p><strong>projected_states</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length, config.proj_codevector_dim)</code>) \u2014 Hidden-states of the model projected to <em>config.proj_codevector_dim</em> that can be used to predict the masked
projected quantized states.</p>
</li>
<li>
<p><strong>projected_quantized_states</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length, config.proj_codevector_dim)</code>) \u2014 Quantized extracted feature vectors projected to <em>config.proj_codevector_dim</em> representing the positive
target vectors for contrastive loss.</p>
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
  href="/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.models.wav2vec2.modeling_flax_wav2vec2.FlaxWav2Vec2ForPreTrainingOutput"
>transformers.models.wav2vec2.modeling_flax_wav2vec2.FlaxWav2Vec2ForPreTrainingOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ja=new ke({props:{$$slots:{default:[zk]},$$scope:{ctx:W}}}),Va=new ge({props:{anchor:"transformers.FlaxWav2Vec2ForPreTraining.__call__.example",$$slots:{default:[Ak]},$$scope:{ctx:W}}}),{c(){c=a("meta"),b=d(),u=a("h1"),f=a("a"),v=a("span"),w(l.$$.fragment),h=d(),V=a("span"),A=r("Wav2Vec2"),M=d(),C=a("h2"),z=a("a"),L=a("span"),w(U.$$.fragment),O=d(),E=a("span"),ae=r("Overview"),X=d(),I=a("p"),N=r("The Wav2Vec2 model was proposed in "),S=a("a"),H=r("wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"),D=r(" by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli."),P=d(),re=a("p"),se=r("The abstract from the paper is the following:"),Te=d(),ie=a("p"),le=a("em"),ct=r(`We show for the first time that learning powerful representations from speech audio alone followed by fine-tuning on
transcribed speech can outperform the best semi-supervised methods while being conceptually simpler. wav2vec 2.0 masks
the speech input in the latent space and solves a contrastive task defined over a quantization of the latent
representations which are jointly learned. Experiments using all labeled data of Librispeech achieve 1.8/3.3 WER on the
clean/other test sets. When lowering the amount of labeled data to one hour, wav2vec 2.0 outperforms the previous state
of the art on the 100 hour subset while using 100 times less labeled data. Using just ten minutes of labeled data and
pre-training on 53k hours of unlabeled data still achieves 4.8/8.2 WER. This demonstrates the feasibility of speech
recognition with limited amounts of labeled data.`),He=d(),B=a("p"),dt=r("Tips:"),be=d(),we=a("ul"),Ne=a("li"),xe=r("Wav2Vec2 is a speech model that accepts a float array corresponding to the raw waveform of the speech signal."),pt=d(),$e=a("li"),ce=r(`Wav2Vec2 model was trained using connectionist temporal classification (CTC) so the model output has to be decoded
using `),Pe=a("a"),We=r("Wav2Vec2CTCTokenizer"),mt=r("."),j=d(),q=a("p"),Be=r("This model was contributed by "),Ue=a("a"),zt=r("patrickvonplaten"),de=r("."),$t=d(),je=a("h2"),Me=a("a"),J=a("span"),w(Z.$$.fragment),At=d(),ht=a("span"),Ve=r("Wav2Vec2Config"),Wt=d(),K=a("div"),w(Fe.$$.fragment),Ot=d(),Lt=a("p"),Kp=r("This is the configuration class to store the configuration of a "),Lr=a("a"),Qp=r("Wav2Vec2Model"),Yp=r(`. It is used to instantiate an
Wav2Vec2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the Wav2Vec2
`),Ga=a("a"),em=r("facebook/wav2vec2-base-960h"),tm=r(" architecture."),om=d(),Dt=a("p"),am=r("Configuration objects inherit from "),Dr=a("a"),sm=r("PretrainedConfig"),nm=r(` and can be used to control the model outputs. Read the
documentation from `),Sr=a("a"),rm=r("PretrainedConfig"),im=r(" for more information."),lm=d(),w(To.$$.fragment),ed=d(),St=a("h2"),xo=a("a"),Ri=a("span"),w(Ja.$$.fragment),cm=d(),Hi=a("span"),dm=r("Wav2Vec2CTCTokenizer"),td=d(),Q=a("div"),w(Za.$$.fragment),pm=d(),Xi=a("p"),mm=r("Constructs a Wav2Vec2CTC tokenizer."),hm=d(),Ka=a("p"),fm=r("This tokenizer inherits from "),Ir=a("a"),um=r("PreTrainedTokenizer"),gm=r(` which contains some of the main methods. Users should refer to
the superclass for more information regarding such methods.`),_m=d(),$o=a("div"),w(Qa.$$.fragment),vm=d(),Gi=a("p"),bm=r(`Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences.`),wm=d(),Nr=a("div"),w(Ya.$$.fragment),ym=d(),Xe=a("div"),w(es.$$.fragment),km=d(),Ji=a("p"),Tm=r(`Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
tokens and clean up tokenization spaces.`),xm=d(),ts=a("p"),$m=r("Similar to doing "),Zi=a("code"),Wm=r("self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))"),jm=r("."),Vm=d(),w(Wo.$$.fragment),Fm=d(),jo=a("div"),w(os.$$.fragment),Cm=d(),Ki=a("p"),Em=r("Convert a list of lists of token ids into a list of strings by calling decode."),od=d(),It=a("h2"),Vo=a("a"),Qi=a("span"),w(as.$$.fragment),qm=d(),Yi=a("span"),Pm=r("Wav2Vec2FeatureExtractor"),ad=d(),Re=a("div"),w(ss.$$.fragment),Mm=d(),el=a("p"),zm=r("Constructs a Wav2Vec2 feature extractor."),Am=d(),ns=a("p"),Om=r("This feature extractor inherits from "),Br=a("a"),Lm=r("SequenceFeatureExtractor"),Dm=r(` which contains
most of the main methods. Users should refer to this superclass for more information regarding those methods.`),Sm=d(),Fo=a("div"),w(rs.$$.fragment),Im=d(),tl=a("p"),Nm=r("Main method to featurize and prepare for the model one or several sequence(s). sequences."),sd=d(),Nt=a("h2"),Co=a("a"),ol=a("span"),w(is.$$.fragment),Bm=d(),al=a("span"),Um=r("Wav2Vec2Processor"),nd=d(),R=a("div"),w(ls.$$.fragment),Rm=d(),sl=a("p"),Hm=r(`Constructs a Wav2Vec2 processor which wraps a Wav2Vec2 feature extractor and a Wav2Vec2 CTC tokenizer into a single
processor.`),Xm=d(),ze=a("p"),Ur=a("a"),Gm=r("Wav2Vec2Processor"),Jm=r(" offers all the functionalities of "),Rr=a("a"),Zm=r("Wav2Vec2FeatureExtractor"),Km=r(" and "),Hr=a("a"),Qm=r("PreTrainedTokenizer"),Ym=r(`.
See the docstring of `),cs=a("a"),nl=a("strong"),eh=r("call"),th=r("()"),oh=r(" and "),Xr=a("a"),ah=r("decode()"),sh=r(" for more information."),nh=d(),Eo=a("div"),w(ds.$$.fragment),rh=d(),ft=a("p"),ih=r(`When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor\u2019s
`),ps=a("a"),rl=a("strong"),lh=r("call"),ch=r("()"),dh=r(` and returns its output. If used in the context
`),il=a("code"),ph=r("as_target_processor()"),mh=r(` this method forwards all its arguments to PreTrainedTokenizer\u2019s
`),ms=a("a"),ll=a("strong"),hh=r("call"),fh=r("()"),uh=r(". Please refer to the docstring of the above two methods for more information."),gh=d(),qo=a("div"),w(hs.$$.fragment),_h=d(),ut=a("p"),vh=r(`When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor\u2019s
`),Gr=a("a"),bh=r("pad()"),wh=r(` and returns its output. If used in the context
`),cl=a("code"),yh=r("as_target_processor()"),kh=r(` this method forwards all its arguments to PreTrainedTokenizer\u2019s
`),Jr=a("a"),Th=r("pad()"),xh=r(". Please refer to the docstring of the above two methods for more information."),$h=d(),Zr=a("div"),w(fs.$$.fragment),Wh=d(),jt=a("div"),w(us.$$.fragment),jh=d(),gs=a("p"),Vh=r(`Saves the attributes of this processor (feature extractor, tokenizer\u2026) in the specified directory so that it
can be reloaded using the `),Kr=a("a"),Fh=r("from_pretrained()"),Ch=r(" method."),Eh=d(),w(Po.$$.fragment),qh=d(),Mo=a("div"),w(_s.$$.fragment),Ph=d(),vs=a("p"),Mh=r("This method forwards all its arguments to PreTrainedTokenizer\u2019s "),Qr=a("a"),zh=r("batch_decode()"),Ah=r(`. Please
refer to the docstring of this method for more information.`),Oh=d(),zo=a("div"),w(bs.$$.fragment),Lh=d(),ws=a("p"),Dh=r("This method forwards all its arguments to PreTrainedTokenizer\u2019s "),Yr=a("a"),Sh=r("decode()"),Ih=r(`. Please refer
to the docstring of this method for more information.`),rd=d(),Bt=a("h2"),Ao=a("a"),dl=a("span"),w(ys.$$.fragment),Nh=d(),pl=a("span"),Bh=r("Wav2Vec2ProcessorWithLM"),id=d(),G=a("div"),w(ks.$$.fragment),Uh=d(),ml=a("p"),Rh=r(`Constructs a Wav2Vec2 processor which wraps a Wav2Vec2 feature extractor, a Wav2Vec2 CTC tokenizer and a decoder
with language model support into a single processor for language model boosted speech recognition decoding.`),Hh=d(),Oo=a("div"),w(Ts.$$.fragment),Xh=d(),gt=a("p"),Gh=r(`When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor\u2019s
`),xs=a("a"),hl=a("strong"),Jh=r("call"),Zh=r("()"),Kh=r(` and returns its output. If used in the context
`),fl=a("code"),Qh=r("as_target_processor()"),Yh=r(` this method forwards all its arguments to
Wav2Vec2CTCTokenizer\u2019s `),$s=a("a"),ul=a("strong"),ef=r("call"),tf=r("()"),of=r(`. Please refer to the docstring of the above two
methods for more information.`),af=d(),Lo=a("div"),w(Ws.$$.fragment),sf=d(),_t=a("p"),nf=r(`When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor\u2019s
`),ei=a("a"),rf=r("pad()"),lf=r(` and returns its output. If used in the context
`),gl=a("code"),cf=r("as_target_processor()"),df=r(` this method forwards all its arguments to
Wav2Vec2CTCTokenizer\u2019s `),ti=a("a"),pf=r("pad()"),mf=r(`. Please refer to the docstring of the above two methods
for more information.`),hf=d(),Vt=a("div"),w(js.$$.fragment),ff=d(),Vs=a("p"),uf=r("Instantiate a "),oi=a("a"),gf=r("Wav2Vec2ProcessorWithLM"),_f=r(" from a pretrained Wav2Vec2 processor."),vf=d(),w(Do.$$.fragment),bf=d(),ai=a("div"),w(Fs.$$.fragment),wf=d(),Ge=a("div"),w(Cs.$$.fragment),yf=d(),_l=a("p"),kf=r("Batch decode output logits to audio transcription with language model support."),Tf=d(),w(So.$$.fragment),xf=d(),w(Io.$$.fragment),$f=d(),Ft=a("div"),w(Es.$$.fragment),Wf=d(),vl=a("p"),jf=r("Decode output logits to audio transcription with language model support."),Vf=d(),w(No.$$.fragment),ld=d(),Ut=a("h2"),Bo=a("a"),bl=a("span"),w(qs.$$.fragment),Ff=d(),wl=a("span"),Cf=r("Wav2Vec2 specific outputs"),cd=d(),Rt=a("div"),w(Ps.$$.fragment),Ef=d(),Ms=a("p"),qf=r("Output type of "),yl=a("code"),Pf=r("Wav2Vec2DecoderWithLM"),Mf=r(", with transcription."),dd=d(),Ht=a("div"),w(zs.$$.fragment),zf=d(),kl=a("p"),Af=r("Base class for models that have been trained with the Wav2Vec2 loss objective."),pd=d(),Xt=a("div"),w(As.$$.fragment),Of=d(),Os=a("p"),Lf=r("Output type of "),si=a("a"),Df=r("Wav2Vec2ForPreTraining"),Sf=r(", with potential hidden states and attentions."),md=d(),vt=a("div"),w(Ls.$$.fragment),If=d(),Ds=a("p"),Nf=r("Output type of "),Tl=a("code"),Bf=r("FlaxWav2Vec2BaseModelOutput"),Uf=r(", with potential hidden states and attentions."),Rf=d(),Uo=a("div"),w(Ss.$$.fragment),Hf=d(),xl=a("p"),Xf=r("\u201CReturns a new object replacing the specified fields with new values."),hd=d(),bt=a("div"),w(Is.$$.fragment),Gf=d(),Ns=a("p"),Jf=r("Output type of "),$l=a("code"),Zf=r("FlaxWav2Vec2ForPreTrainingOutput"),Kf=r(", with potential hidden states and attentions."),Qf=d(),Ro=a("div"),w(Bs.$$.fragment),Yf=d(),Wl=a("p"),eu=r("\u201CReturns a new object replacing the specified fields with new values."),fd=d(),Gt=a("h2"),Ho=a("a"),jl=a("span"),w(Us.$$.fragment),tu=d(),Vl=a("span"),ou=r("Wav2Vec2Model"),ud=d(),Ce=a("div"),w(Rs.$$.fragment),au=d(),Hs=a("p"),su=r(`The bare Wav2Vec2 Model transformer outputting raw hidden-states without any specific head on top.
Wav2Vec2 was proposed in `),Xs=a("a"),nu=r(`wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
Representations`),ru=r(` by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael
Auli.`),iu=d(),Gs=a("p"),lu=r("This model inherits from "),ni=a("a"),cu=r("PreTrainedModel"),du=r(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving etc.).`),pu=d(),Js=a("p"),mu=r("This model is a PyTorch "),Zs=a("a"),hu=r("torch.nn.Module"),fu=r(` sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),uu=d(),Je=a("div"),w(Ks.$$.fragment),gu=d(),Jt=a("p"),_u=r("The "),ri=a("a"),vu=r("Wav2Vec2Model"),bu=r(" forward method, overrides the "),Fl=a("code"),wu=r("__call__"),yu=r(" special method."),ku=d(),w(Xo.$$.fragment),Tu=d(),w(Go.$$.fragment),gd=d(),Zt=a("h2"),Jo=a("a"),Cl=a("span"),w(Qs.$$.fragment),xu=d(),El=a("span"),$u=r("Wav2Vec2ForCTC"),_d=d(),Ee=a("div"),w(Ys.$$.fragment),Wu=d(),Kt=a("p"),ju=r("Wav2Vec2 Model with a "),ql=a("code"),Vu=r("language modeling"),Fu=r(` head on top for Connectionist Temporal Classification (CTC).
Wav2Vec2 was proposed in `),en=a("a"),Cu=r(`wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
Representations`),Eu=r(` by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael
Auli.`),qu=d(),tn=a("p"),Pu=r("This model inherits from "),ii=a("a"),Mu=r("PreTrainedModel"),zu=r(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving etc.).`),Au=d(),on=a("p"),Ou=r("This model is a PyTorch "),an=a("a"),Lu=r("torch.nn.Module"),Du=r(` sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),Su=d(),Ae=a("div"),w(sn.$$.fragment),Iu=d(),Qt=a("p"),Nu=r("The "),li=a("a"),Bu=r("Wav2Vec2ForCTC"),Uu=r(" forward method, overrides the "),Pl=a("code"),Ru=r("__call__"),Hu=r(" special method."),Xu=d(),w(Zo.$$.fragment),Gu=d(),w(Ko.$$.fragment),Ju=d(),w(Qo.$$.fragment),vd=d(),Yt=a("h2"),Yo=a("a"),Ml=a("span"),w(nn.$$.fragment),Zu=d(),zl=a("span"),Ku=r("Wav2Vec2ForSequenceClassification"),bd=d(),pe=a("div"),w(rn.$$.fragment),Qu=d(),Al=a("p"),Yu=r(`Wav2Vec2 Model with a sequence classification head on top (a linear layer over the pooled output) for tasks like
SUPERB Keyword Spotting.`),eg=d(),ln=a("p"),tg=r("Wav2Vec2 was proposed in "),cn=a("a"),og=r(`wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
Representations`),ag=r(` by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael
Auli.`),sg=d(),dn=a("p"),ng=r("This model inherits from "),ci=a("a"),rg=r("PreTrainedModel"),ig=r(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving etc.).`),lg=d(),pn=a("p"),cg=r("This model is a PyTorch "),mn=a("a"),dg=r("torch.nn.Module"),pg=r(` sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),mg=d(),Oe=a("div"),w(hn.$$.fragment),hg=d(),eo=a("p"),fg=r("The "),di=a("a"),ug=r("Wav2Vec2ForSequenceClassification"),gg=r(" forward method, overrides the "),Ol=a("code"),_g=r("__call__"),vg=r(" special method."),bg=d(),w(ea.$$.fragment),wg=d(),w(ta.$$.fragment),yg=d(),w(oa.$$.fragment),wd=d(),to=a("h2"),aa=a("a"),Ll=a("span"),w(fn.$$.fragment),kg=d(),Dl=a("span"),Tg=r("Wav2Vec2ForAudioFrameClassification"),yd=d(),me=a("div"),w(un.$$.fragment),xg=d(),Sl=a("p"),$g=r("Wav2Vec2 Model with a frame classification head on top for tasks like Speaker Diarization."),Wg=d(),gn=a("p"),jg=r("Wav2Vec2 was proposed in "),_n=a("a"),Vg=r(`wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
Representations`),Fg=r(` by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael
Auli.`),Cg=d(),vn=a("p"),Eg=r("This model inherits from "),pi=a("a"),qg=r("PreTrainedModel"),Pg=r(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving etc.).`),Mg=d(),bn=a("p"),zg=r("This model is a PyTorch "),wn=a("a"),Ag=r("torch.nn.Module"),Og=r(` sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),Lg=d(),Ze=a("div"),w(yn.$$.fragment),Dg=d(),oo=a("p"),Sg=r("The "),mi=a("a"),Ig=r("Wav2Vec2ForAudioFrameClassification"),Ng=r(" forward method, overrides the "),Il=a("code"),Bg=r("__call__"),Ug=r(" special method."),Rg=d(),w(sa.$$.fragment),Hg=d(),w(na.$$.fragment),kd=d(),ao=a("h2"),ra=a("a"),Nl=a("span"),w(kn.$$.fragment),Xg=d(),Bl=a("span"),Gg=r("Wav2Vec2ForXVector"),Td=d(),he=a("div"),w(Tn.$$.fragment),Jg=d(),Ul=a("p"),Zg=r("Wav2Vec2 Model with an XVector feature extraction head on top for tasks like Speaker Verification."),Kg=d(),xn=a("p"),Qg=r("Wav2Vec2 was proposed in "),$n=a("a"),Yg=r(`wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
Representations`),e_=r(` by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael
Auli.`),t_=d(),Wn=a("p"),o_=r("This model inherits from "),hi=a("a"),a_=r("PreTrainedModel"),s_=r(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving etc.).`),n_=d(),jn=a("p"),r_=r("This model is a PyTorch "),Vn=a("a"),i_=r("torch.nn.Module"),l_=r(` sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),c_=d(),Ke=a("div"),w(Fn.$$.fragment),d_=d(),so=a("p"),p_=r("The "),fi=a("a"),m_=r("Wav2Vec2ForXVector"),h_=r(" forward method, overrides the "),Rl=a("code"),f_=r("__call__"),u_=r(" special method."),g_=d(),w(ia.$$.fragment),__=d(),w(la.$$.fragment),xd=d(),no=a("h2"),ca=a("a"),Hl=a("span"),w(Cn.$$.fragment),v_=d(),Xl=a("span"),b_=r("Wav2Vec2ForPreTraining"),$d=d(),qe=a("div"),w(En.$$.fragment),w_=d(),ro=a("p"),y_=r("Wav2Vec2 Model with a quantizer and "),Gl=a("code"),k_=r("VQ"),T_=r(` head on top.
Wav2Vec2 was proposed in `),qn=a("a"),x_=r(`wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
Representations`),$_=r(` by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael
Auli.`),W_=d(),Pn=a("p"),j_=r("This model inherits from "),ui=a("a"),V_=r("PreTrainedModel"),F_=r(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving etc.).`),C_=d(),Mn=a("p"),E_=r("This model is a PyTorch "),zn=a("a"),q_=r("torch.nn.Module"),P_=r(` sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),M_=d(),Qe=a("div"),w(An.$$.fragment),z_=d(),io=a("p"),A_=r("The "),gi=a("a"),O_=r("Wav2Vec2ForPreTraining"),L_=r(" forward method, overrides the "),Jl=a("code"),D_=r("__call__"),S_=r(" special method."),I_=d(),w(da.$$.fragment),N_=d(),w(pa.$$.fragment),Wd=d(),lo=a("h2"),ma=a("a"),Zl=a("span"),w(On.$$.fragment),B_=d(),Kl=a("span"),U_=r("TFWav2Vec2Model"),jd=d(),fe=a("div"),w(Ln.$$.fragment),R_=d(),Ql=a("p"),H_=r("The bare TFWav2Vec2 Model transformer outputing raw hidden-states without any specific head on top."),X_=d(),Dn=a("p"),G_=r("This model inherits from "),_i=a("a"),J_=r("TFPreTrainedModel"),Z_=r(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),K_=d(),Sn=a("p"),Q_=r("This model is also a "),In=a("a"),Y_=r("tf.keras.Model"),ev=r(` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),tv=d(),w(ha.$$.fragment),ov=d(),Ye=a("div"),w(Nn.$$.fragment),av=d(),co=a("p"),sv=r("The "),vi=a("a"),nv=r("TFWav2Vec2Model"),rv=r(" forward method, overrides the "),Yl=a("code"),iv=r("__call__"),lv=r(" special method."),cv=d(),w(fa.$$.fragment),dv=d(),w(ua.$$.fragment),Vd=d(),po=a("h2"),ga=a("a"),ec=a("span"),w(Bn.$$.fragment),pv=d(),tc=a("span"),mv=r("TFWav2Vec2ForCTC"),Fd=d(),ue=a("div"),w(Un.$$.fragment),hv=d(),Rn=a("p"),fv=r("TFWav2Vec2 Model with a "),oc=a("code"),uv=r("language modeling"),gv=r(" head on top for Connectionist Temporal Classification (CTC)."),_v=d(),Hn=a("p"),vv=r("This model inherits from "),bi=a("a"),bv=r("TFPreTrainedModel"),wv=r(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),yv=d(),Xn=a("p"),kv=r("This model is also a "),Gn=a("a"),Tv=r("tf.keras.Model"),xv=r(` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),$v=d(),w(_a.$$.fragment),Wv=d(),et=a("div"),w(Jn.$$.fragment),jv=d(),mo=a("p"),Vv=r("The "),wi=a("a"),Fv=r("TFWav2Vec2ForCTC"),Cv=r(" forward method, overrides the "),ac=a("code"),Ev=r("__call__"),qv=r(" special method."),Pv=d(),w(va.$$.fragment),Mv=d(),w(ba.$$.fragment),Cd=d(),ho=a("h2"),wa=a("a"),sc=a("span"),w(Zn.$$.fragment),zv=d(),nc=a("span"),Av=r("FlaxWav2Vec2Model"),Ed=d(),Y=a("div"),w(Kn.$$.fragment),Ov=d(),Qn=a("p"),Lv=r(`The bare Wav2Vec2 Model transformer outputting raw hidden-states without any specific head on top.
Wav2Vec2 was proposed in `),Yn=a("a"),Dv=r(`wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
Representations`),Sv=r(` by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael
Auli.`),Iv=d(),er=a("p"),Nv=r("This model inherits from "),yi=a("a"),Bv=r("FlaxPreTrainedModel"),Uv=r(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Rv=d(),tr=a("p"),Hv=r(`This model is also a Flax Linen
`),or=a("a"),Xv=r("flax.nn.Module"),Gv=r(` subclass. Use it as a
regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.`),Jv=d(),rc=a("p"),Zv=r("Finally, this model supports inherent JAX features such as:"),Kv=d(),wt=a("ul"),ic=a("li"),ar=a("a"),Qv=r("Just-In-Time (JIT) compilation"),Yv=d(),lc=a("li"),sr=a("a"),e2=r("Automatic Differentiation"),t2=d(),cc=a("li"),nr=a("a"),o2=r("Vectorization"),a2=d(),dc=a("li"),rr=a("a"),s2=r("Parallelization"),n2=d(),tt=a("div"),w(ir.$$.fragment),r2=d(),fo=a("p"),i2=r("The "),pc=a("code"),l2=r("FlaxWav2Vec2PreTrainedModel"),c2=r(" forward method, overrides the "),mc=a("code"),d2=r("__call__"),p2=r(" special method."),m2=d(),w(ya.$$.fragment),h2=d(),w(ka.$$.fragment),qd=d(),uo=a("h2"),Ta=a("a"),hc=a("span"),w(lr.$$.fragment),f2=d(),fc=a("span"),u2=r("FlaxWav2Vec2ForCTC"),Pd=d(),ee=a("div"),w(cr.$$.fragment),g2=d(),go=a("p"),_2=r("Wav2Vec2 Model with a "),uc=a("code"),v2=r("language modeling"),b2=r(` head on top for Connectionist Temporal Classification (CTC).
Wav2Vec2 was proposed in `),dr=a("a"),w2=r(`wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
Representations`),y2=r(` by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael
Auli.`),k2=d(),pr=a("p"),T2=r("This model inherits from "),ki=a("a"),x2=r("FlaxPreTrainedModel"),$2=r(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),W2=d(),mr=a("p"),j2=r(`This model is also a Flax Linen
`),hr=a("a"),V2=r("flax.nn.Module"),F2=r(` subclass. Use it as a
regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.`),C2=d(),gc=a("p"),E2=r("Finally, this model supports inherent JAX features such as:"),q2=d(),yt=a("ul"),_c=a("li"),fr=a("a"),P2=r("Just-In-Time (JIT) compilation"),M2=d(),vc=a("li"),ur=a("a"),z2=r("Automatic Differentiation"),A2=d(),bc=a("li"),gr=a("a"),O2=r("Vectorization"),L2=d(),wc=a("li"),_r=a("a"),D2=r("Parallelization"),S2=d(),ot=a("div"),w(vr.$$.fragment),I2=d(),_o=a("p"),N2=r("The "),yc=a("code"),B2=r("FlaxWav2Vec2PreTrainedModel"),U2=r(" forward method, overrides the "),kc=a("code"),R2=r("__call__"),H2=r(" special method."),X2=d(),w(xa.$$.fragment),G2=d(),w($a.$$.fragment),Md=d(),vo=a("h2"),Wa=a("a"),Tc=a("span"),w(br.$$.fragment),J2=d(),xc=a("span"),Z2=r("FlaxWav2Vec2ForPreTraining"),zd=d(),te=a("div"),w(wr.$$.fragment),K2=d(),bo=a("p"),Q2=r("Wav2Vec2 Model with a quantizer and "),$c=a("code"),Y2=r("VQ"),eb=r(` head on top.
Wav2Vec2 was proposed in `),yr=a("a"),tb=r(`wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
Representations`),ob=r(` by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael
Auli.`),ab=d(),kr=a("p"),sb=r("This model inherits from "),Ti=a("a"),nb=r("FlaxPreTrainedModel"),rb=r(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),ib=d(),Tr=a("p"),lb=r(`This model is also a Flax Linen
`),xr=a("a"),cb=r("flax.nn.Module"),db=r(` subclass. Use it as a
regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.`),pb=d(),Wc=a("p"),mb=r("Finally, this model supports inherent JAX features such as:"),hb=d(),kt=a("ul"),jc=a("li"),$r=a("a"),fb=r("Just-In-Time (JIT) compilation"),ub=d(),Vc=a("li"),Wr=a("a"),gb=r("Automatic Differentiation"),_b=d(),Fc=a("li"),jr=a("a"),vb=r("Vectorization"),bb=d(),Cc=a("li"),Vr=a("a"),wb=r("Parallelization"),yb=d(),at=a("div"),w(Fr.$$.fragment),kb=d(),wo=a("p"),Tb=r("The "),xi=a("a"),xb=r("FlaxWav2Vec2ForPreTraining"),$b=r(" forward method, overrides the "),Ec=a("code"),Wb=r("__call__"),jb=r(" special method."),Vb=d(),w(ja.$$.fragment),Fb=d(),w(Va.$$.fragment),this.h()},l(o){const g=ok('[data-svelte="svelte-1phssyn"]',document.head);c=s(g,"META",{name:!0,content:!0}),g.forEach(t),b=p(o),u=s(o,"H1",{class:!0});var Cr=n(u);f=s(Cr,"A",{id:!0,class:!0,href:!0});var qc=n(f);v=s(qc,"SPAN",{});var Pc=n(v);y(l.$$.fragment,Pc),Pc.forEach(t),qc.forEach(t),h=p(Cr),V=s(Cr,"SPAN",{});var Mc=n(V);A=i(Mc,"Wav2Vec2"),Mc.forEach(t),Cr.forEach(t),M=p(o),C=s(o,"H2",{class:!0});var Er=n(C);z=s(Er,"A",{id:!0,class:!0,href:!0});var zc=n(z);L=s(zc,"SPAN",{});var Ac=n(L);y(U.$$.fragment,Ac),Ac.forEach(t),zc.forEach(t),O=p(Er),E=s(Er,"SPAN",{});var Oc=n(E);ae=i(Oc,"Overview"),Oc.forEach(t),Er.forEach(t),X=p(o),I=s(o,"P",{});var qr=n(I);N=i(qr,"The Wav2Vec2 model was proposed in "),S=s(qr,"A",{href:!0,rel:!0});var Lc=n(S);H=i(Lc,"wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"),Lc.forEach(t),D=i(qr," by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli."),qr.forEach(t),P=p(o),re=s(o,"P",{});var Dc=n(re);se=i(Dc,"The abstract from the paper is the following:"),Dc.forEach(t),Te=p(o),ie=s(o,"P",{});var Sc=n(ie);le=s(Sc,"EM",{});var Ic=n(le);ct=i(Ic,`We show for the first time that learning powerful representations from speech audio alone followed by fine-tuning on
transcribed speech can outperform the best semi-supervised methods while being conceptually simpler. wav2vec 2.0 masks
the speech input in the latent space and solves a contrastive task defined over a quantization of the latent
representations which are jointly learned. Experiments using all labeled data of Librispeech achieve 1.8/3.3 WER on the
clean/other test sets. When lowering the amount of labeled data to one hour, wav2vec 2.0 outperforms the previous state
of the art on the 100 hour subset while using 100 times less labeled data. Using just ten minutes of labeled data and
pre-training on 53k hours of unlabeled data still achieves 4.8/8.2 WER. This demonstrates the feasibility of speech
recognition with limited amounts of labeled data.`),Ic.forEach(t),Sc.forEach(t),He=p(o),B=s(o,"P",{});var Nc=n(B);dt=i(Nc,"Tips:"),Nc.forEach(t),be=p(o),we=s(o,"UL",{});var Pr=n(we);Ne=s(Pr,"LI",{});var Bc=n(Ne);xe=i(Bc,"Wav2Vec2 is a speech model that accepts a float array corresponding to the raw waveform of the speech signal."),Bc.forEach(t),pt=p(Pr),$e=s(Pr,"LI",{});var Mr=n($e);ce=i(Mr,`Wav2Vec2 model was trained using connectionist temporal classification (CTC) so the model output has to be decoded
using `),Pe=s(Mr,"A",{href:!0});var Uc=n(Pe);We=i(Uc,"Wav2Vec2CTCTokenizer"),Uc.forEach(t),mt=i(Mr,"."),Mr.forEach(t),Pr.forEach(t),j=p(o),q=s(o,"P",{});var zr=n(q);Be=i(zr,"This model was contributed by "),Ue=s(zr,"A",{href:!0,rel:!0});var Rc=n(Ue);zt=i(Rc,"patrickvonplaten"),Rc.forEach(t),de=i(zr,"."),zr.forEach(t),$t=p(o),je=s(o,"H2",{class:!0});var Ar=n(je);Me=s(Ar,"A",{id:!0,class:!0,href:!0});var Hc=n(Me);J=s(Hc,"SPAN",{});var Xc=n(J);y(Z.$$.fragment,Xc),Xc.forEach(t),Hc.forEach(t),At=p(Ar),ht=s(Ar,"SPAN",{});var Gc=n(ht);Ve=i(Gc,"Wav2Vec2Config"),Gc.forEach(t),Ar.forEach(t),Wt=p(o),K=s(o,"DIV",{class:!0});var Tt=n(K);y(Fe.$$.fragment,Tt),Ot=p(Tt),Lt=s(Tt,"P",{});var yo=n(Lt);Kp=i(yo,"This is the configuration class to store the configuration of a "),Lr=s(yo,"A",{href:!0});var Jc=n(Lr);Qp=i(Jc,"Wav2Vec2Model"),Jc.forEach(t),Yp=i(yo,`. It is used to instantiate an
Wav2Vec2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the Wav2Vec2
`),Ga=s(yo,"A",{href:!0,rel:!0});var Zc=n(Ga);em=i(Zc,"facebook/wav2vec2-base-960h"),Zc.forEach(t),tm=i(yo," architecture."),yo.forEach(t),om=p(Tt),Dt=s(Tt,"P",{});var ko=n(Dt);am=i(ko,"Configuration objects inherit from "),Dr=s(ko,"A",{href:!0});var Kc=n(Dr);sm=i(Kc,"PretrainedConfig"),Kc.forEach(t),nm=i(ko,` and can be used to control the model outputs. Read the
documentation from `),Sr=s(ko,"A",{href:!0});var Qc=n(Sr);rm=i(Qc,"PretrainedConfig"),Qc.forEach(t),im=i(ko," for more information."),ko.forEach(t),lm=p(Tt),y(To.$$.fragment,Tt),Tt.forEach(t),ed=p(o),St=s(o,"H2",{class:!0});var Or=n(St);xo=s(Or,"A",{id:!0,class:!0,href:!0});var Yc=n(xo);Ri=s(Yc,"SPAN",{});var zb=n(Ri);y(Ja.$$.fragment,zb),zb.forEach(t),Yc.forEach(t),cm=p(Or),Hi=s(Or,"SPAN",{});var Ab=n(Hi);dm=i(Ab,"Wav2Vec2CTCTokenizer"),Ab.forEach(t),Or.forEach(t),td=p(o),Q=s(o,"DIV",{class:!0});var Le=n(Q);y(Za.$$.fragment,Le),pm=p(Le),Xi=s(Le,"P",{});var Ob=n(Xi);mm=i(Ob,"Constructs a Wav2Vec2CTC tokenizer."),Ob.forEach(t),hm=p(Le),Ka=s(Le,"P",{});var Od=n(Ka);fm=i(Od,"This tokenizer inherits from "),Ir=s(Od,"A",{href:!0});var Lb=n(Ir);um=i(Lb,"PreTrainedTokenizer"),Lb.forEach(t),gm=i(Od,` which contains some of the main methods. Users should refer to
the superclass for more information regarding such methods.`),Od.forEach(t),_m=p(Le),$o=s(Le,"DIV",{class:!0});var Ld=n($o);y(Qa.$$.fragment,Ld),vm=p(Ld),Gi=s(Ld,"P",{});var Db=n(Gi);bm=i(Db,`Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences.`),Db.forEach(t),Ld.forEach(t),wm=p(Le),Nr=s(Le,"DIV",{class:!0});var Sb=n(Nr);y(Ya.$$.fragment,Sb),Sb.forEach(t),ym=p(Le),Xe=s(Le,"DIV",{class:!0});var Fa=n(Xe);y(es.$$.fragment,Fa),km=p(Fa),Ji=s(Fa,"P",{});var Ib=n(Ji);Tm=i(Ib,`Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
tokens and clean up tokenization spaces.`),Ib.forEach(t),xm=p(Fa),ts=s(Fa,"P",{});var Dd=n(ts);$m=i(Dd,"Similar to doing "),Zi=s(Dd,"CODE",{});var Nb=n(Zi);Wm=i(Nb,"self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))"),Nb.forEach(t),jm=i(Dd,"."),Dd.forEach(t),Vm=p(Fa),y(Wo.$$.fragment,Fa),Fa.forEach(t),Fm=p(Le),jo=s(Le,"DIV",{class:!0});var Sd=n(jo);y(os.$$.fragment,Sd),Cm=p(Sd),Ki=s(Sd,"P",{});var Bb=n(Ki);Em=i(Bb,"Convert a list of lists of token ids into a list of strings by calling decode."),Bb.forEach(t),Sd.forEach(t),Le.forEach(t),od=p(o),It=s(o,"H2",{class:!0});var Id=n(It);Vo=s(Id,"A",{id:!0,class:!0,href:!0});var Ub=n(Vo);Qi=s(Ub,"SPAN",{});var Rb=n(Qi);y(as.$$.fragment,Rb),Rb.forEach(t),Ub.forEach(t),qm=p(Id),Yi=s(Id,"SPAN",{});var Hb=n(Yi);Pm=i(Hb,"Wav2Vec2FeatureExtractor"),Hb.forEach(t),Id.forEach(t),ad=p(o),Re=s(o,"DIV",{class:!0});var Ca=n(Re);y(ss.$$.fragment,Ca),Mm=p(Ca),el=s(Ca,"P",{});var Xb=n(el);zm=i(Xb,"Constructs a Wav2Vec2 feature extractor."),Xb.forEach(t),Am=p(Ca),ns=s(Ca,"P",{});var Nd=n(ns);Om=i(Nd,"This feature extractor inherits from "),Br=s(Nd,"A",{href:!0});var Gb=n(Br);Lm=i(Gb,"SequenceFeatureExtractor"),Gb.forEach(t),Dm=i(Nd,` which contains
most of the main methods. Users should refer to this superclass for more information regarding those methods.`),Nd.forEach(t),Sm=p(Ca),Fo=s(Ca,"DIV",{class:!0});var Bd=n(Fo);y(rs.$$.fragment,Bd),Im=p(Bd),tl=s(Bd,"P",{});var Jb=n(tl);Nm=i(Jb,"Main method to featurize and prepare for the model one or several sequence(s). sequences."),Jb.forEach(t),Bd.forEach(t),Ca.forEach(t),sd=p(o),Nt=s(o,"H2",{class:!0});var Ud=n(Nt);Co=s(Ud,"A",{id:!0,class:!0,href:!0});var Zb=n(Co);ol=s(Zb,"SPAN",{});var Kb=n(ol);y(is.$$.fragment,Kb),Kb.forEach(t),Zb.forEach(t),Bm=p(Ud),al=s(Ud,"SPAN",{});var Qb=n(al);Um=i(Qb,"Wav2Vec2Processor"),Qb.forEach(t),Ud.forEach(t),nd=p(o),R=s(o,"DIV",{class:!0});var ne=n(R);y(ls.$$.fragment,ne),Rm=p(ne),sl=s(ne,"P",{});var Yb=n(sl);Hm=i(Yb,`Constructs a Wav2Vec2 processor which wraps a Wav2Vec2 feature extractor and a Wav2Vec2 CTC tokenizer into a single
processor.`),Yb.forEach(t),Xm=p(ne),ze=s(ne,"P",{});var xt=n(ze);Ur=s(xt,"A",{href:!0});var ew=n(Ur);Gm=i(ew,"Wav2Vec2Processor"),ew.forEach(t),Jm=i(xt," offers all the functionalities of "),Rr=s(xt,"A",{href:!0});var tw=n(Rr);Zm=i(tw,"Wav2Vec2FeatureExtractor"),tw.forEach(t),Km=i(xt," and "),Hr=s(xt,"A",{href:!0});var ow=n(Hr);Qm=i(ow,"PreTrainedTokenizer"),ow.forEach(t),Ym=i(xt,`.
See the docstring of `),cs=s(xt,"A",{href:!0});var Cb=n(cs);nl=s(Cb,"STRONG",{});var aw=n(nl);eh=i(aw,"call"),aw.forEach(t),th=i(Cb,"()"),Cb.forEach(t),oh=i(xt," and "),Xr=s(xt,"A",{href:!0});var sw=n(Xr);ah=i(sw,"decode()"),sw.forEach(t),sh=i(xt," for more information."),xt.forEach(t),nh=p(ne),Eo=s(ne,"DIV",{class:!0});var Rd=n(Eo);y(ds.$$.fragment,Rd),rh=p(Rd),ft=s(Rd,"P",{});var Ea=n(ft);ih=i(Ea,`When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor\u2019s
`),ps=s(Ea,"A",{href:!0});var Eb=n(ps);rl=s(Eb,"STRONG",{});var nw=n(rl);lh=i(nw,"call"),nw.forEach(t),ch=i(Eb,"()"),Eb.forEach(t),dh=i(Ea,` and returns its output. If used in the context
`),il=s(Ea,"CODE",{});var rw=n(il);ph=i(rw,"as_target_processor()"),rw.forEach(t),mh=i(Ea,` this method forwards all its arguments to PreTrainedTokenizer\u2019s
`),ms=s(Ea,"A",{href:!0});var qb=n(ms);ll=s(qb,"STRONG",{});var iw=n(ll);hh=i(iw,"call"),iw.forEach(t),fh=i(qb,"()"),qb.forEach(t),uh=i(Ea,". Please refer to the docstring of the above two methods for more information."),Ea.forEach(t),Rd.forEach(t),gh=p(ne),qo=s(ne,"DIV",{class:!0});var Hd=n(qo);y(hs.$$.fragment,Hd),_h=p(Hd),ut=s(Hd,"P",{});var qa=n(ut);vh=i(qa,`When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor\u2019s
`),Gr=s(qa,"A",{href:!0});var lw=n(Gr);bh=i(lw,"pad()"),lw.forEach(t),wh=i(qa,` and returns its output. If used in the context
`),cl=s(qa,"CODE",{});var cw=n(cl);yh=i(cw,"as_target_processor()"),cw.forEach(t),kh=i(qa,` this method forwards all its arguments to PreTrainedTokenizer\u2019s
`),Jr=s(qa,"A",{href:!0});var dw=n(Jr);Th=i(dw,"pad()"),dw.forEach(t),xh=i(qa,". Please refer to the docstring of the above two methods for more information."),qa.forEach(t),Hd.forEach(t),$h=p(ne),Zr=s(ne,"DIV",{class:!0});var pw=n(Zr);y(fs.$$.fragment,pw),pw.forEach(t),Wh=p(ne),jt=s(ne,"DIV",{class:!0});var $i=n(jt);y(us.$$.fragment,$i),jh=p($i),gs=s($i,"P",{});var Xd=n(gs);Vh=i(Xd,`Saves the attributes of this processor (feature extractor, tokenizer\u2026) in the specified directory so that it
can be reloaded using the `),Kr=s(Xd,"A",{href:!0});var mw=n(Kr);Fh=i(mw,"from_pretrained()"),mw.forEach(t),Ch=i(Xd," method."),Xd.forEach(t),Eh=p($i),y(Po.$$.fragment,$i),$i.forEach(t),qh=p(ne),Mo=s(ne,"DIV",{class:!0});var Gd=n(Mo);y(_s.$$.fragment,Gd),Ph=p(Gd),vs=s(Gd,"P",{});var Jd=n(vs);Mh=i(Jd,"This method forwards all its arguments to PreTrainedTokenizer\u2019s "),Qr=s(Jd,"A",{href:!0});var hw=n(Qr);zh=i(hw,"batch_decode()"),hw.forEach(t),Ah=i(Jd,`. Please
refer to the docstring of this method for more information.`),Jd.forEach(t),Gd.forEach(t),Oh=p(ne),zo=s(ne,"DIV",{class:!0});var Zd=n(zo);y(bs.$$.fragment,Zd),Lh=p(Zd),ws=s(Zd,"P",{});var Kd=n(ws);Dh=i(Kd,"This method forwards all its arguments to PreTrainedTokenizer\u2019s "),Yr=s(Kd,"A",{href:!0});var fw=n(Yr);Sh=i(fw,"decode()"),fw.forEach(t),Ih=i(Kd,`. Please refer
to the docstring of this method for more information.`),Kd.forEach(t),Zd.forEach(t),ne.forEach(t),rd=p(o),Bt=s(o,"H2",{class:!0});var Qd=n(Bt);Ao=s(Qd,"A",{id:!0,class:!0,href:!0});var uw=n(Ao);dl=s(uw,"SPAN",{});var gw=n(dl);y(ys.$$.fragment,gw),gw.forEach(t),uw.forEach(t),Nh=p(Qd),pl=s(Qd,"SPAN",{});var _w=n(pl);Bh=i(_w,"Wav2Vec2ProcessorWithLM"),_w.forEach(t),Qd.forEach(t),id=p(o),G=s(o,"DIV",{class:!0});var ye=n(G);y(ks.$$.fragment,ye),Uh=p(ye),ml=s(ye,"P",{});var vw=n(ml);Rh=i(vw,`Constructs a Wav2Vec2 processor which wraps a Wav2Vec2 feature extractor, a Wav2Vec2 CTC tokenizer and a decoder
with language model support into a single processor for language model boosted speech recognition decoding.`),vw.forEach(t),Hh=p(ye),Oo=s(ye,"DIV",{class:!0});var Yd=n(Oo);y(Ts.$$.fragment,Yd),Xh=p(Yd),gt=s(Yd,"P",{});var Pa=n(gt);Gh=i(Pa,`When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor\u2019s
`),xs=s(Pa,"A",{href:!0});var Pb=n(xs);hl=s(Pb,"STRONG",{});var bw=n(hl);Jh=i(bw,"call"),bw.forEach(t),Zh=i(Pb,"()"),Pb.forEach(t),Kh=i(Pa,` and returns its output. If used in the context
`),fl=s(Pa,"CODE",{});var ww=n(fl);Qh=i(ww,"as_target_processor()"),ww.forEach(t),Yh=i(Pa,` this method forwards all its arguments to
Wav2Vec2CTCTokenizer\u2019s `),$s=s(Pa,"A",{href:!0});var Mb=n($s);ul=s(Mb,"STRONG",{});var yw=n(ul);ef=i(yw,"call"),yw.forEach(t),tf=i(Mb,"()"),Mb.forEach(t),of=i(Pa,`. Please refer to the docstring of the above two
methods for more information.`),Pa.forEach(t),Yd.forEach(t),af=p(ye),Lo=s(ye,"DIV",{class:!0});var ep=n(Lo);y(Ws.$$.fragment,ep),sf=p(ep),_t=s(ep,"P",{});var Ma=n(_t);nf=i(Ma,`When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor\u2019s
`),ei=s(Ma,"A",{href:!0});var kw=n(ei);rf=i(kw,"pad()"),kw.forEach(t),lf=i(Ma,` and returns its output. If used in the context
`),gl=s(Ma,"CODE",{});var Tw=n(gl);cf=i(Tw,"as_target_processor()"),Tw.forEach(t),df=i(Ma,` this method forwards all its arguments to
Wav2Vec2CTCTokenizer\u2019s `),ti=s(Ma,"A",{href:!0});var xw=n(ti);pf=i(xw,"pad()"),xw.forEach(t),mf=i(Ma,`. Please refer to the docstring of the above two methods
for more information.`),Ma.forEach(t),ep.forEach(t),hf=p(ye),Vt=s(ye,"DIV",{class:!0});var Wi=n(Vt);y(js.$$.fragment,Wi),ff=p(Wi),Vs=s(Wi,"P",{});var tp=n(Vs);uf=i(tp,"Instantiate a "),oi=s(tp,"A",{href:!0});var $w=n(oi);gf=i($w,"Wav2Vec2ProcessorWithLM"),$w.forEach(t),_f=i(tp," from a pretrained Wav2Vec2 processor."),tp.forEach(t),vf=p(Wi),y(Do.$$.fragment,Wi),Wi.forEach(t),bf=p(ye),ai=s(ye,"DIV",{class:!0});var Ww=n(ai);y(Fs.$$.fragment,Ww),Ww.forEach(t),wf=p(ye),Ge=s(ye,"DIV",{class:!0});var za=n(Ge);y(Cs.$$.fragment,za),yf=p(za),_l=s(za,"P",{});var jw=n(_l);kf=i(jw,"Batch decode output logits to audio transcription with language model support."),jw.forEach(t),Tf=p(za),y(So.$$.fragment,za),xf=p(za),y(Io.$$.fragment,za),za.forEach(t),$f=p(ye),Ft=s(ye,"DIV",{class:!0});var ji=n(Ft);y(Es.$$.fragment,ji),Wf=p(ji),vl=s(ji,"P",{});var Vw=n(vl);jf=i(Vw,"Decode output logits to audio transcription with language model support."),Vw.forEach(t),Vf=p(ji),y(No.$$.fragment,ji),ji.forEach(t),ye.forEach(t),ld=p(o),Ut=s(o,"H2",{class:!0});var op=n(Ut);Bo=s(op,"A",{id:!0,class:!0,href:!0});var Fw=n(Bo);bl=s(Fw,"SPAN",{});var Cw=n(bl);y(qs.$$.fragment,Cw),Cw.forEach(t),Fw.forEach(t),Ff=p(op),wl=s(op,"SPAN",{});var Ew=n(wl);Cf=i(Ew,"Wav2Vec2 specific outputs"),Ew.forEach(t),op.forEach(t),cd=p(o),Rt=s(o,"DIV",{class:!0});var ap=n(Rt);y(Ps.$$.fragment,ap),Ef=p(ap),Ms=s(ap,"P",{});var sp=n(Ms);qf=i(sp,"Output type of "),yl=s(sp,"CODE",{});var qw=n(yl);Pf=i(qw,"Wav2Vec2DecoderWithLM"),qw.forEach(t),Mf=i(sp,", with transcription."),sp.forEach(t),ap.forEach(t),dd=p(o),Ht=s(o,"DIV",{class:!0});var np=n(Ht);y(zs.$$.fragment,np),zf=p(np),kl=s(np,"P",{});var Pw=n(kl);Af=i(Pw,"Base class for models that have been trained with the Wav2Vec2 loss objective."),Pw.forEach(t),np.forEach(t),pd=p(o),Xt=s(o,"DIV",{class:!0});var rp=n(Xt);y(As.$$.fragment,rp),Of=p(rp),Os=s(rp,"P",{});var ip=n(Os);Lf=i(ip,"Output type of "),si=s(ip,"A",{href:!0});var Mw=n(si);Df=i(Mw,"Wav2Vec2ForPreTraining"),Mw.forEach(t),Sf=i(ip,", with potential hidden states and attentions."),ip.forEach(t),rp.forEach(t),md=p(o),vt=s(o,"DIV",{class:!0});var Vi=n(vt);y(Ls.$$.fragment,Vi),If=p(Vi),Ds=s(Vi,"P",{});var lp=n(Ds);Nf=i(lp,"Output type of "),Tl=s(lp,"CODE",{});var zw=n(Tl);Bf=i(zw,"FlaxWav2Vec2BaseModelOutput"),zw.forEach(t),Uf=i(lp,", with potential hidden states and attentions."),lp.forEach(t),Rf=p(Vi),Uo=s(Vi,"DIV",{class:!0});var cp=n(Uo);y(Ss.$$.fragment,cp),Hf=p(cp),xl=s(cp,"P",{});var Aw=n(xl);Xf=i(Aw,"\u201CReturns a new object replacing the specified fields with new values."),Aw.forEach(t),cp.forEach(t),Vi.forEach(t),hd=p(o),bt=s(o,"DIV",{class:!0});var Fi=n(bt);y(Is.$$.fragment,Fi),Gf=p(Fi),Ns=s(Fi,"P",{});var dp=n(Ns);Jf=i(dp,"Output type of "),$l=s(dp,"CODE",{});var Ow=n($l);Zf=i(Ow,"FlaxWav2Vec2ForPreTrainingOutput"),Ow.forEach(t),Kf=i(dp,", with potential hidden states and attentions."),dp.forEach(t),Qf=p(Fi),Ro=s(Fi,"DIV",{class:!0});var pp=n(Ro);y(Bs.$$.fragment,pp),Yf=p(pp),Wl=s(pp,"P",{});var Lw=n(Wl);eu=i(Lw,"\u201CReturns a new object replacing the specified fields with new values."),Lw.forEach(t),pp.forEach(t),Fi.forEach(t),fd=p(o),Gt=s(o,"H2",{class:!0});var mp=n(Gt);Ho=s(mp,"A",{id:!0,class:!0,href:!0});var Dw=n(Ho);jl=s(Dw,"SPAN",{});var Sw=n(jl);y(Us.$$.fragment,Sw),Sw.forEach(t),Dw.forEach(t),tu=p(mp),Vl=s(mp,"SPAN",{});var Iw=n(Vl);ou=i(Iw,"Wav2Vec2Model"),Iw.forEach(t),mp.forEach(t),ud=p(o),Ce=s(o,"DIV",{class:!0});var Ct=n(Ce);y(Rs.$$.fragment,Ct),au=p(Ct),Hs=s(Ct,"P",{});var hp=n(Hs);su=i(hp,`The bare Wav2Vec2 Model transformer outputting raw hidden-states without any specific head on top.
Wav2Vec2 was proposed in `),Xs=s(hp,"A",{href:!0,rel:!0});var Nw=n(Xs);nu=i(Nw,`wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
Representations`),Nw.forEach(t),ru=i(hp,` by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael
Auli.`),hp.forEach(t),iu=p(Ct),Gs=s(Ct,"P",{});var fp=n(Gs);lu=i(fp,"This model inherits from "),ni=s(fp,"A",{href:!0});var Bw=n(ni);cu=i(Bw,"PreTrainedModel"),Bw.forEach(t),du=i(fp,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving etc.).`),fp.forEach(t),pu=p(Ct),Js=s(Ct,"P",{});var up=n(Js);mu=i(up,"This model is a PyTorch "),Zs=s(up,"A",{href:!0,rel:!0});var Uw=n(Zs);hu=i(Uw,"torch.nn.Module"),Uw.forEach(t),fu=i(up,` sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),up.forEach(t),uu=p(Ct),Je=s(Ct,"DIV",{class:!0});var Aa=n(Je);y(Ks.$$.fragment,Aa),gu=p(Aa),Jt=s(Aa,"P",{});var Ci=n(Jt);_u=i(Ci,"The "),ri=s(Ci,"A",{href:!0});var Rw=n(ri);vu=i(Rw,"Wav2Vec2Model"),Rw.forEach(t),bu=i(Ci," forward method, overrides the "),Fl=s(Ci,"CODE",{});var Hw=n(Fl);wu=i(Hw,"__call__"),Hw.forEach(t),yu=i(Ci," special method."),Ci.forEach(t),ku=p(Aa),y(Xo.$$.fragment,Aa),Tu=p(Aa),y(Go.$$.fragment,Aa),Aa.forEach(t),Ct.forEach(t),gd=p(o),Zt=s(o,"H2",{class:!0});var gp=n(Zt);Jo=s(gp,"A",{id:!0,class:!0,href:!0});var Xw=n(Jo);Cl=s(Xw,"SPAN",{});var Gw=n(Cl);y(Qs.$$.fragment,Gw),Gw.forEach(t),Xw.forEach(t),xu=p(gp),El=s(gp,"SPAN",{});var Jw=n(El);$u=i(Jw,"Wav2Vec2ForCTC"),Jw.forEach(t),gp.forEach(t),_d=p(o),Ee=s(o,"DIV",{class:!0});var Et=n(Ee);y(Ys.$$.fragment,Et),Wu=p(Et),Kt=s(Et,"P",{});var Ei=n(Kt);ju=i(Ei,"Wav2Vec2 Model with a "),ql=s(Ei,"CODE",{});var Zw=n(ql);Vu=i(Zw,"language modeling"),Zw.forEach(t),Fu=i(Ei,` head on top for Connectionist Temporal Classification (CTC).
Wav2Vec2 was proposed in `),en=s(Ei,"A",{href:!0,rel:!0});var Kw=n(en);Cu=i(Kw,`wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
Representations`),Kw.forEach(t),Eu=i(Ei,` by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael
Auli.`),Ei.forEach(t),qu=p(Et),tn=s(Et,"P",{});var _p=n(tn);Pu=i(_p,"This model inherits from "),ii=s(_p,"A",{href:!0});var Qw=n(ii);Mu=i(Qw,"PreTrainedModel"),Qw.forEach(t),zu=i(_p,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving etc.).`),_p.forEach(t),Au=p(Et),on=s(Et,"P",{});var vp=n(on);Ou=i(vp,"This model is a PyTorch "),an=s(vp,"A",{href:!0,rel:!0});var Yw=n(an);Lu=i(Yw,"torch.nn.Module"),Yw.forEach(t),Du=i(vp,` sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),vp.forEach(t),Su=p(Et),Ae=s(Et,"DIV",{class:!0});var qt=n(Ae);y(sn.$$.fragment,qt),Iu=p(qt),Qt=s(qt,"P",{});var qi=n(Qt);Nu=i(qi,"The "),li=s(qi,"A",{href:!0});var e1=n(li);Bu=i(e1,"Wav2Vec2ForCTC"),e1.forEach(t),Uu=i(qi," forward method, overrides the "),Pl=s(qi,"CODE",{});var t1=n(Pl);Ru=i(t1,"__call__"),t1.forEach(t),Hu=i(qi," special method."),qi.forEach(t),Xu=p(qt),y(Zo.$$.fragment,qt),Gu=p(qt),y(Ko.$$.fragment,qt),Ju=p(qt),y(Qo.$$.fragment,qt),qt.forEach(t),Et.forEach(t),vd=p(o),Yt=s(o,"H2",{class:!0});var bp=n(Yt);Yo=s(bp,"A",{id:!0,class:!0,href:!0});var o1=n(Yo);Ml=s(o1,"SPAN",{});var a1=n(Ml);y(nn.$$.fragment,a1),a1.forEach(t),o1.forEach(t),Zu=p(bp),zl=s(bp,"SPAN",{});var s1=n(zl);Ku=i(s1,"Wav2Vec2ForSequenceClassification"),s1.forEach(t),bp.forEach(t),bd=p(o),pe=s(o,"DIV",{class:!0});var st=n(pe);y(rn.$$.fragment,st),Qu=p(st),Al=s(st,"P",{});var n1=n(Al);Yu=i(n1,`Wav2Vec2 Model with a sequence classification head on top (a linear layer over the pooled output) for tasks like
SUPERB Keyword Spotting.`),n1.forEach(t),eg=p(st),ln=s(st,"P",{});var wp=n(ln);tg=i(wp,"Wav2Vec2 was proposed in "),cn=s(wp,"A",{href:!0,rel:!0});var r1=n(cn);og=i(r1,`wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
Representations`),r1.forEach(t),ag=i(wp,` by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael
Auli.`),wp.forEach(t),sg=p(st),dn=s(st,"P",{});var yp=n(dn);ng=i(yp,"This model inherits from "),ci=s(yp,"A",{href:!0});var i1=n(ci);rg=i(i1,"PreTrainedModel"),i1.forEach(t),ig=i(yp,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving etc.).`),yp.forEach(t),lg=p(st),pn=s(st,"P",{});var kp=n(pn);cg=i(kp,"This model is a PyTorch "),mn=s(kp,"A",{href:!0,rel:!0});var l1=n(mn);dg=i(l1,"torch.nn.Module"),l1.forEach(t),pg=i(kp,` sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),kp.forEach(t),mg=p(st),Oe=s(st,"DIV",{class:!0});var Pt=n(Oe);y(hn.$$.fragment,Pt),hg=p(Pt),eo=s(Pt,"P",{});var Pi=n(eo);fg=i(Pi,"The "),di=s(Pi,"A",{href:!0});var c1=n(di);ug=i(c1,"Wav2Vec2ForSequenceClassification"),c1.forEach(t),gg=i(Pi," forward method, overrides the "),Ol=s(Pi,"CODE",{});var d1=n(Ol);_g=i(d1,"__call__"),d1.forEach(t),vg=i(Pi," special method."),Pi.forEach(t),bg=p(Pt),y(ea.$$.fragment,Pt),wg=p(Pt),y(ta.$$.fragment,Pt),yg=p(Pt),y(oa.$$.fragment,Pt),Pt.forEach(t),st.forEach(t),wd=p(o),to=s(o,"H2",{class:!0});var Tp=n(to);aa=s(Tp,"A",{id:!0,class:!0,href:!0});var p1=n(aa);Ll=s(p1,"SPAN",{});var m1=n(Ll);y(fn.$$.fragment,m1),m1.forEach(t),p1.forEach(t),kg=p(Tp),Dl=s(Tp,"SPAN",{});var h1=n(Dl);Tg=i(h1,"Wav2Vec2ForAudioFrameClassification"),h1.forEach(t),Tp.forEach(t),yd=p(o),me=s(o,"DIV",{class:!0});var nt=n(me);y(un.$$.fragment,nt),xg=p(nt),Sl=s(nt,"P",{});var f1=n(Sl);$g=i(f1,"Wav2Vec2 Model with a frame classification head on top for tasks like Speaker Diarization."),f1.forEach(t),Wg=p(nt),gn=s(nt,"P",{});var xp=n(gn);jg=i(xp,"Wav2Vec2 was proposed in "),_n=s(xp,"A",{href:!0,rel:!0});var u1=n(_n);Vg=i(u1,`wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
Representations`),u1.forEach(t),Fg=i(xp,` by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael
Auli.`),xp.forEach(t),Cg=p(nt),vn=s(nt,"P",{});var $p=n(vn);Eg=i($p,"This model inherits from "),pi=s($p,"A",{href:!0});var g1=n(pi);qg=i(g1,"PreTrainedModel"),g1.forEach(t),Pg=i($p,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving etc.).`),$p.forEach(t),Mg=p(nt),bn=s(nt,"P",{});var Wp=n(bn);zg=i(Wp,"This model is a PyTorch "),wn=s(Wp,"A",{href:!0,rel:!0});var _1=n(wn);Ag=i(_1,"torch.nn.Module"),_1.forEach(t),Og=i(Wp,` sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),Wp.forEach(t),Lg=p(nt),Ze=s(nt,"DIV",{class:!0});var Oa=n(Ze);y(yn.$$.fragment,Oa),Dg=p(Oa),oo=s(Oa,"P",{});var Mi=n(oo);Sg=i(Mi,"The "),mi=s(Mi,"A",{href:!0});var v1=n(mi);Ig=i(v1,"Wav2Vec2ForAudioFrameClassification"),v1.forEach(t),Ng=i(Mi," forward method, overrides the "),Il=s(Mi,"CODE",{});var b1=n(Il);Bg=i(b1,"__call__"),b1.forEach(t),Ug=i(Mi," special method."),Mi.forEach(t),Rg=p(Oa),y(sa.$$.fragment,Oa),Hg=p(Oa),y(na.$$.fragment,Oa),Oa.forEach(t),nt.forEach(t),kd=p(o),ao=s(o,"H2",{class:!0});var jp=n(ao);ra=s(jp,"A",{id:!0,class:!0,href:!0});var w1=n(ra);Nl=s(w1,"SPAN",{});var y1=n(Nl);y(kn.$$.fragment,y1),y1.forEach(t),w1.forEach(t),Xg=p(jp),Bl=s(jp,"SPAN",{});var k1=n(Bl);Gg=i(k1,"Wav2Vec2ForXVector"),k1.forEach(t),jp.forEach(t),Td=p(o),he=s(o,"DIV",{class:!0});var rt=n(he);y(Tn.$$.fragment,rt),Jg=p(rt),Ul=s(rt,"P",{});var T1=n(Ul);Zg=i(T1,"Wav2Vec2 Model with an XVector feature extraction head on top for tasks like Speaker Verification."),T1.forEach(t),Kg=p(rt),xn=s(rt,"P",{});var Vp=n(xn);Qg=i(Vp,"Wav2Vec2 was proposed in "),$n=s(Vp,"A",{href:!0,rel:!0});var x1=n($n);Yg=i(x1,`wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
Representations`),x1.forEach(t),e_=i(Vp,` by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael
Auli.`),Vp.forEach(t),t_=p(rt),Wn=s(rt,"P",{});var Fp=n(Wn);o_=i(Fp,"This model inherits from "),hi=s(Fp,"A",{href:!0});var $1=n(hi);a_=i($1,"PreTrainedModel"),$1.forEach(t),s_=i(Fp,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving etc.).`),Fp.forEach(t),n_=p(rt),jn=s(rt,"P",{});var Cp=n(jn);r_=i(Cp,"This model is a PyTorch "),Vn=s(Cp,"A",{href:!0,rel:!0});var W1=n(Vn);i_=i(W1,"torch.nn.Module"),W1.forEach(t),l_=i(Cp,` sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),Cp.forEach(t),c_=p(rt),Ke=s(rt,"DIV",{class:!0});var La=n(Ke);y(Fn.$$.fragment,La),d_=p(La),so=s(La,"P",{});var zi=n(so);p_=i(zi,"The "),fi=s(zi,"A",{href:!0});var j1=n(fi);m_=i(j1,"Wav2Vec2ForXVector"),j1.forEach(t),h_=i(zi," forward method, overrides the "),Rl=s(zi,"CODE",{});var V1=n(Rl);f_=i(V1,"__call__"),V1.forEach(t),u_=i(zi," special method."),zi.forEach(t),g_=p(La),y(ia.$$.fragment,La),__=p(La),y(la.$$.fragment,La),La.forEach(t),rt.forEach(t),xd=p(o),no=s(o,"H2",{class:!0});var Ep=n(no);ca=s(Ep,"A",{id:!0,class:!0,href:!0});var F1=n(ca);Hl=s(F1,"SPAN",{});var C1=n(Hl);y(Cn.$$.fragment,C1),C1.forEach(t),F1.forEach(t),v_=p(Ep),Xl=s(Ep,"SPAN",{});var E1=n(Xl);b_=i(E1,"Wav2Vec2ForPreTraining"),E1.forEach(t),Ep.forEach(t),$d=p(o),qe=s(o,"DIV",{class:!0});var Mt=n(qe);y(En.$$.fragment,Mt),w_=p(Mt),ro=s(Mt,"P",{});var Ai=n(ro);y_=i(Ai,"Wav2Vec2 Model with a quantizer and "),Gl=s(Ai,"CODE",{});var q1=n(Gl);k_=i(q1,"VQ"),q1.forEach(t),T_=i(Ai,` head on top.
Wav2Vec2 was proposed in `),qn=s(Ai,"A",{href:!0,rel:!0});var P1=n(qn);x_=i(P1,`wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
Representations`),P1.forEach(t),$_=i(Ai,` by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael
Auli.`),Ai.forEach(t),W_=p(Mt),Pn=s(Mt,"P",{});var qp=n(Pn);j_=i(qp,"This model inherits from "),ui=s(qp,"A",{href:!0});var M1=n(ui);V_=i(M1,"PreTrainedModel"),M1.forEach(t),F_=i(qp,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving etc.).`),qp.forEach(t),C_=p(Mt),Mn=s(Mt,"P",{});var Pp=n(Mn);E_=i(Pp,"This model is a PyTorch "),zn=s(Pp,"A",{href:!0,rel:!0});var z1=n(zn);q_=i(z1,"torch.nn.Module"),z1.forEach(t),P_=i(Pp,` sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.`),Pp.forEach(t),M_=p(Mt),Qe=s(Mt,"DIV",{class:!0});var Da=n(Qe);y(An.$$.fragment,Da),z_=p(Da),io=s(Da,"P",{});var Oi=n(io);A_=i(Oi,"The "),gi=s(Oi,"A",{href:!0});var A1=n(gi);O_=i(A1,"Wav2Vec2ForPreTraining"),A1.forEach(t),L_=i(Oi," forward method, overrides the "),Jl=s(Oi,"CODE",{});var O1=n(Jl);D_=i(O1,"__call__"),O1.forEach(t),S_=i(Oi," special method."),Oi.forEach(t),I_=p(Da),y(da.$$.fragment,Da),N_=p(Da),y(pa.$$.fragment,Da),Da.forEach(t),Mt.forEach(t),Wd=p(o),lo=s(o,"H2",{class:!0});var Mp=n(lo);ma=s(Mp,"A",{id:!0,class:!0,href:!0});var L1=n(ma);Zl=s(L1,"SPAN",{});var D1=n(Zl);y(On.$$.fragment,D1),D1.forEach(t),L1.forEach(t),B_=p(Mp),Kl=s(Mp,"SPAN",{});var S1=n(Kl);U_=i(S1,"TFWav2Vec2Model"),S1.forEach(t),Mp.forEach(t),jd=p(o),fe=s(o,"DIV",{class:!0});var it=n(fe);y(Ln.$$.fragment,it),R_=p(it),Ql=s(it,"P",{});var I1=n(Ql);H_=i(I1,"The bare TFWav2Vec2 Model transformer outputing raw hidden-states without any specific head on top."),I1.forEach(t),X_=p(it),Dn=s(it,"P",{});var zp=n(Dn);G_=i(zp,"This model inherits from "),_i=s(zp,"A",{href:!0});var N1=n(_i);J_=i(N1,"TFPreTrainedModel"),N1.forEach(t),Z_=i(zp,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),zp.forEach(t),K_=p(it),Sn=s(it,"P",{});var Ap=n(Sn);Q_=i(Ap,"This model is also a "),In=s(Ap,"A",{href:!0,rel:!0});var B1=n(In);Y_=i(B1,"tf.keras.Model"),B1.forEach(t),ev=i(Ap,` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),Ap.forEach(t),tv=p(it),y(ha.$$.fragment,it),ov=p(it),Ye=s(it,"DIV",{class:!0});var Sa=n(Ye);y(Nn.$$.fragment,Sa),av=p(Sa),co=s(Sa,"P",{});var Li=n(co);sv=i(Li,"The "),vi=s(Li,"A",{href:!0});var U1=n(vi);nv=i(U1,"TFWav2Vec2Model"),U1.forEach(t),rv=i(Li," forward method, overrides the "),Yl=s(Li,"CODE",{});var R1=n(Yl);iv=i(R1,"__call__"),R1.forEach(t),lv=i(Li," special method."),Li.forEach(t),cv=p(Sa),y(fa.$$.fragment,Sa),dv=p(Sa),y(ua.$$.fragment,Sa),Sa.forEach(t),it.forEach(t),Vd=p(o),po=s(o,"H2",{class:!0});var Op=n(po);ga=s(Op,"A",{id:!0,class:!0,href:!0});var H1=n(ga);ec=s(H1,"SPAN",{});var X1=n(ec);y(Bn.$$.fragment,X1),X1.forEach(t),H1.forEach(t),pv=p(Op),tc=s(Op,"SPAN",{});var G1=n(tc);mv=i(G1,"TFWav2Vec2ForCTC"),G1.forEach(t),Op.forEach(t),Fd=p(o),ue=s(o,"DIV",{class:!0});var lt=n(ue);y(Un.$$.fragment,lt),hv=p(lt),Rn=s(lt,"P",{});var Lp=n(Rn);fv=i(Lp,"TFWav2Vec2 Model with a "),oc=s(Lp,"CODE",{});var J1=n(oc);uv=i(J1,"language modeling"),J1.forEach(t),gv=i(Lp," head on top for Connectionist Temporal Classification (CTC)."),Lp.forEach(t),_v=p(lt),Hn=s(lt,"P",{});var Dp=n(Hn);vv=i(Dp,"This model inherits from "),bi=s(Dp,"A",{href:!0});var Z1=n(bi);bv=i(Z1,"TFPreTrainedModel"),Z1.forEach(t),wv=i(Dp,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Dp.forEach(t),yv=p(lt),Xn=s(lt,"P",{});var Sp=n(Xn);kv=i(Sp,"This model is also a "),Gn=s(Sp,"A",{href:!0,rel:!0});var K1=n(Gn);Tv=i(K1,"tf.keras.Model"),K1.forEach(t),xv=i(Sp,` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),Sp.forEach(t),$v=p(lt),y(_a.$$.fragment,lt),Wv=p(lt),et=s(lt,"DIV",{class:!0});var Ia=n(et);y(Jn.$$.fragment,Ia),jv=p(Ia),mo=s(Ia,"P",{});var Di=n(mo);Vv=i(Di,"The "),wi=s(Di,"A",{href:!0});var Q1=n(wi);Fv=i(Q1,"TFWav2Vec2ForCTC"),Q1.forEach(t),Cv=i(Di," forward method, overrides the "),ac=s(Di,"CODE",{});var Y1=n(ac);Ev=i(Y1,"__call__"),Y1.forEach(t),qv=i(Di," special method."),Di.forEach(t),Pv=p(Ia),y(va.$$.fragment,Ia),Mv=p(Ia),y(ba.$$.fragment,Ia),Ia.forEach(t),lt.forEach(t),Cd=p(o),ho=s(o,"H2",{class:!0});var Ip=n(ho);wa=s(Ip,"A",{id:!0,class:!0,href:!0});var ey=n(wa);sc=s(ey,"SPAN",{});var ty=n(sc);y(Zn.$$.fragment,ty),ty.forEach(t),ey.forEach(t),zv=p(Ip),nc=s(Ip,"SPAN",{});var oy=n(nc);Av=i(oy,"FlaxWav2Vec2Model"),oy.forEach(t),Ip.forEach(t),Ed=p(o),Y=s(o,"DIV",{class:!0});var De=n(Y);y(Kn.$$.fragment,De),Ov=p(De),Qn=s(De,"P",{});var Np=n(Qn);Lv=i(Np,`The bare Wav2Vec2 Model transformer outputting raw hidden-states without any specific head on top.
Wav2Vec2 was proposed in `),Yn=s(Np,"A",{href:!0,rel:!0});var ay=n(Yn);Dv=i(ay,`wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
Representations`),ay.forEach(t),Sv=i(Np,` by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael
Auli.`),Np.forEach(t),Iv=p(De),er=s(De,"P",{});var Bp=n(er);Nv=i(Bp,"This model inherits from "),yi=s(Bp,"A",{href:!0});var sy=n(yi);Bv=i(sy,"FlaxPreTrainedModel"),sy.forEach(t),Uv=i(Bp,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Bp.forEach(t),Rv=p(De),tr=s(De,"P",{});var Up=n(tr);Hv=i(Up,`This model is also a Flax Linen
`),or=s(Up,"A",{href:!0,rel:!0});var ny=n(or);Xv=i(ny,"flax.nn.Module"),ny.forEach(t),Gv=i(Up,` subclass. Use it as a
regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.`),Up.forEach(t),Jv=p(De),rc=s(De,"P",{});var ry=n(rc);Zv=i(ry,"Finally, this model supports inherent JAX features such as:"),ry.forEach(t),Kv=p(De),wt=s(De,"UL",{});var Na=n(wt);ic=s(Na,"LI",{});var iy=n(ic);ar=s(iy,"A",{href:!0,rel:!0});var ly=n(ar);Qv=i(ly,"Just-In-Time (JIT) compilation"),ly.forEach(t),iy.forEach(t),Yv=p(Na),lc=s(Na,"LI",{});var cy=n(lc);sr=s(cy,"A",{href:!0,rel:!0});var dy=n(sr);e2=i(dy,"Automatic Differentiation"),dy.forEach(t),cy.forEach(t),t2=p(Na),cc=s(Na,"LI",{});var py=n(cc);nr=s(py,"A",{href:!0,rel:!0});var my=n(nr);o2=i(my,"Vectorization"),my.forEach(t),py.forEach(t),a2=p(Na),dc=s(Na,"LI",{});var hy=n(dc);rr=s(hy,"A",{href:!0,rel:!0});var fy=n(rr);s2=i(fy,"Parallelization"),fy.forEach(t),hy.forEach(t),Na.forEach(t),n2=p(De),tt=s(De,"DIV",{class:!0});var Ba=n(tt);y(ir.$$.fragment,Ba),r2=p(Ba),fo=s(Ba,"P",{});var Si=n(fo);i2=i(Si,"The "),pc=s(Si,"CODE",{});var uy=n(pc);l2=i(uy,"FlaxWav2Vec2PreTrainedModel"),uy.forEach(t),c2=i(Si," forward method, overrides the "),mc=s(Si,"CODE",{});var gy=n(mc);d2=i(gy,"__call__"),gy.forEach(t),p2=i(Si," special method."),Si.forEach(t),m2=p(Ba),y(ya.$$.fragment,Ba),h2=p(Ba),y(ka.$$.fragment,Ba),Ba.forEach(t),De.forEach(t),qd=p(o),uo=s(o,"H2",{class:!0});var Rp=n(uo);Ta=s(Rp,"A",{id:!0,class:!0,href:!0});var _y=n(Ta);hc=s(_y,"SPAN",{});var vy=n(hc);y(lr.$$.fragment,vy),vy.forEach(t),_y.forEach(t),f2=p(Rp),fc=s(Rp,"SPAN",{});var by=n(fc);u2=i(by,"FlaxWav2Vec2ForCTC"),by.forEach(t),Rp.forEach(t),Pd=p(o),ee=s(o,"DIV",{class:!0});var Se=n(ee);y(cr.$$.fragment,Se),g2=p(Se),go=s(Se,"P",{});var Ii=n(go);_2=i(Ii,"Wav2Vec2 Model with a "),uc=s(Ii,"CODE",{});var wy=n(uc);v2=i(wy,"language modeling"),wy.forEach(t),b2=i(Ii,` head on top for Connectionist Temporal Classification (CTC).
Wav2Vec2 was proposed in `),dr=s(Ii,"A",{href:!0,rel:!0});var yy=n(dr);w2=i(yy,`wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
Representations`),yy.forEach(t),y2=i(Ii,` by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael
Auli.`),Ii.forEach(t),k2=p(Se),pr=s(Se,"P",{});var Hp=n(pr);T2=i(Hp,"This model inherits from "),ki=s(Hp,"A",{href:!0});var ky=n(ki);x2=i(ky,"FlaxPreTrainedModel"),ky.forEach(t),$2=i(Hp,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Hp.forEach(t),W2=p(Se),mr=s(Se,"P",{});var Xp=n(mr);j2=i(Xp,`This model is also a Flax Linen
`),hr=s(Xp,"A",{href:!0,rel:!0});var Ty=n(hr);V2=i(Ty,"flax.nn.Module"),Ty.forEach(t),F2=i(Xp,` subclass. Use it as a
regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.`),Xp.forEach(t),C2=p(Se),gc=s(Se,"P",{});var xy=n(gc);E2=i(xy,"Finally, this model supports inherent JAX features such as:"),xy.forEach(t),q2=p(Se),yt=s(Se,"UL",{});var Ua=n(yt);_c=s(Ua,"LI",{});var $y=n(_c);fr=s($y,"A",{href:!0,rel:!0});var Wy=n(fr);P2=i(Wy,"Just-In-Time (JIT) compilation"),Wy.forEach(t),$y.forEach(t),M2=p(Ua),vc=s(Ua,"LI",{});var jy=n(vc);ur=s(jy,"A",{href:!0,rel:!0});var Vy=n(ur);z2=i(Vy,"Automatic Differentiation"),Vy.forEach(t),jy.forEach(t),A2=p(Ua),bc=s(Ua,"LI",{});var Fy=n(bc);gr=s(Fy,"A",{href:!0,rel:!0});var Cy=n(gr);O2=i(Cy,"Vectorization"),Cy.forEach(t),Fy.forEach(t),L2=p(Ua),wc=s(Ua,"LI",{});var Ey=n(wc);_r=s(Ey,"A",{href:!0,rel:!0});var qy=n(_r);D2=i(qy,"Parallelization"),qy.forEach(t),Ey.forEach(t),Ua.forEach(t),S2=p(Se),ot=s(Se,"DIV",{class:!0});var Ra=n(ot);y(vr.$$.fragment,Ra),I2=p(Ra),_o=s(Ra,"P",{});var Ni=n(_o);N2=i(Ni,"The "),yc=s(Ni,"CODE",{});var Py=n(yc);B2=i(Py,"FlaxWav2Vec2PreTrainedModel"),Py.forEach(t),U2=i(Ni," forward method, overrides the "),kc=s(Ni,"CODE",{});var My=n(kc);R2=i(My,"__call__"),My.forEach(t),H2=i(Ni," special method."),Ni.forEach(t),X2=p(Ra),y(xa.$$.fragment,Ra),G2=p(Ra),y($a.$$.fragment,Ra),Ra.forEach(t),Se.forEach(t),Md=p(o),vo=s(o,"H2",{class:!0});var Gp=n(vo);Wa=s(Gp,"A",{id:!0,class:!0,href:!0});var zy=n(Wa);Tc=s(zy,"SPAN",{});var Ay=n(Tc);y(br.$$.fragment,Ay),Ay.forEach(t),zy.forEach(t),J2=p(Gp),xc=s(Gp,"SPAN",{});var Oy=n(xc);Z2=i(Oy,"FlaxWav2Vec2ForPreTraining"),Oy.forEach(t),Gp.forEach(t),zd=p(o),te=s(o,"DIV",{class:!0});var Ie=n(te);y(wr.$$.fragment,Ie),K2=p(Ie),bo=s(Ie,"P",{});var Bi=n(bo);Q2=i(Bi,"Wav2Vec2 Model with a quantizer and "),$c=s(Bi,"CODE",{});var Ly=n($c);Y2=i(Ly,"VQ"),Ly.forEach(t),eb=i(Bi,` head on top.
Wav2Vec2 was proposed in `),yr=s(Bi,"A",{href:!0,rel:!0});var Dy=n(yr);tb=i(Dy,`wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
Representations`),Dy.forEach(t),ob=i(Bi,` by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael
Auli.`),Bi.forEach(t),ab=p(Ie),kr=s(Ie,"P",{});var Jp=n(kr);sb=i(Jp,"This model inherits from "),Ti=s(Jp,"A",{href:!0});var Sy=n(Ti);nb=i(Sy,"FlaxPreTrainedModel"),Sy.forEach(t),rb=i(Jp,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Jp.forEach(t),ib=p(Ie),Tr=s(Ie,"P",{});var Zp=n(Tr);lb=i(Zp,`This model is also a Flax Linen
`),xr=s(Zp,"A",{href:!0,rel:!0});var Iy=n(xr);cb=i(Iy,"flax.nn.Module"),Iy.forEach(t),db=i(Zp,` subclass. Use it as a
regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.`),Zp.forEach(t),pb=p(Ie),Wc=s(Ie,"P",{});var Ny=n(Wc);mb=i(Ny,"Finally, this model supports inherent JAX features such as:"),Ny.forEach(t),hb=p(Ie),kt=s(Ie,"UL",{});var Ha=n(kt);jc=s(Ha,"LI",{});var By=n(jc);$r=s(By,"A",{href:!0,rel:!0});var Uy=n($r);fb=i(Uy,"Just-In-Time (JIT) compilation"),Uy.forEach(t),By.forEach(t),ub=p(Ha),Vc=s(Ha,"LI",{});var Ry=n(Vc);Wr=s(Ry,"A",{href:!0,rel:!0});var Hy=n(Wr);gb=i(Hy,"Automatic Differentiation"),Hy.forEach(t),Ry.forEach(t),_b=p(Ha),Fc=s(Ha,"LI",{});var Xy=n(Fc);jr=s(Xy,"A",{href:!0,rel:!0});var Gy=n(jr);vb=i(Gy,"Vectorization"),Gy.forEach(t),Xy.forEach(t),bb=p(Ha),Cc=s(Ha,"LI",{});var Jy=n(Cc);Vr=s(Jy,"A",{href:!0,rel:!0});var Zy=n(Vr);wb=i(Zy,"Parallelization"),Zy.forEach(t),Jy.forEach(t),Ha.forEach(t),yb=p(Ie),at=s(Ie,"DIV",{class:!0});var Xa=n(at);y(Fr.$$.fragment,Xa),kb=p(Xa),wo=s(Xa,"P",{});var Ui=n(wo);Tb=i(Ui,"The "),xi=s(Ui,"A",{href:!0});var Ky=n(xi);xb=i(Ky,"FlaxWav2Vec2ForPreTraining"),Ky.forEach(t),$b=i(Ui," forward method, overrides the "),Ec=s(Ui,"CODE",{});var Qy=n(Ec);Wb=i(Qy,"__call__"),Qy.forEach(t),jb=i(Ui," special method."),Ui.forEach(t),Vb=p(Xa),y(ja.$$.fragment,Xa),Fb=p(Xa),y(Va.$$.fragment,Xa),Xa.forEach(t),Ie.forEach(t),this.h()},h(){m(c,"name","hf:doc:metadata"),m(c,"content",JSON.stringify(Lk)),m(f,"id","wav2vec2"),m(f,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(f,"href","#wav2vec2"),m(u,"class","relative group"),m(z,"id","overview"),m(z,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(z,"href","#overview"),m(C,"class","relative group"),m(S,"href","https://arxiv.org/abs/2006.11477"),m(S,"rel","nofollow"),m(Pe,"href","/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer"),m(Ue,"href","https://huggingface.co/patrickvonplaten"),m(Ue,"rel","nofollow"),m(Me,"id","transformers.Wav2Vec2Config"),m(Me,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(Me,"href","#transformers.Wav2Vec2Config"),m(je,"class","relative group"),m(Lr,"href","/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Model"),m(Ga,"href","https://huggingface.co/facebook/wav2vec2-base-960h"),m(Ga,"rel","nofollow"),m(Dr,"href","/docs/transformers/pr_18351/en/main_classes/configuration#transformers.PretrainedConfig"),m(Sr,"href","/docs/transformers/pr_18351/en/main_classes/configuration#transformers.PretrainedConfig"),m(K,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(xo,"id","transformers.Wav2Vec2CTCTokenizer"),m(xo,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(xo,"href","#transformers.Wav2Vec2CTCTokenizer"),m(St,"class","relative group"),m(Ir,"href","/docs/transformers/pr_18351/en/main_classes/tokenizer#transformers.PreTrainedTokenizer"),m($o,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Nr,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Xe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(jo,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Vo,"id","transformers.Wav2Vec2FeatureExtractor"),m(Vo,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(Vo,"href","#transformers.Wav2Vec2FeatureExtractor"),m(It,"class","relative group"),m(Br,"href","/docs/transformers/pr_18351/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor"),m(Fo,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Re,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Co,"id","transformers.Wav2Vec2Processor"),m(Co,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(Co,"href","#transformers.Wav2Vec2Processor"),m(Nt,"class","relative group"),m(Ur,"href","/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor"),m(Rr,"href","/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor"),m(Hr,"href","/docs/transformers/pr_18351/en/main_classes/tokenizer#transformers.PreTrainedTokenizer"),m(cs,"href","/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__"),m(Xr,"href","/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.decode"),m(ps,"href","/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor.__call__"),m(ms,"href","/docs/transformers/pr_18351/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__"),m(Eo,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Gr,"href","/docs/transformers/pr_18351/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor.pad"),m(Jr,"href","/docs/transformers/pr_18351/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.pad"),m(qo,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Zr,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Kr,"href","/docs/transformers/pr_18351/en/model_doc/speech_to_text_2#transformers.Speech2Text2Processor.from_pretrained"),m(jt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Qr,"href","/docs/transformers/pr_18351/en/model_doc/speech_to_text_2#transformers.Speech2Text2Tokenizer.batch_decode"),m(Mo,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Yr,"href","/docs/transformers/pr_18351/en/model_doc/speech_to_text_2#transformers.Speech2Text2Tokenizer.decode"),m(zo,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Ao,"id","transformers.Wav2Vec2ProcessorWithLM"),m(Ao,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(Ao,"href","#transformers.Wav2Vec2ProcessorWithLM"),m(Bt,"class","relative group"),m(xs,"href","/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor.__call__"),m($s,"href","/docs/transformers/pr_18351/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__"),m(Oo,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(ei,"href","/docs/transformers/pr_18351/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor.pad"),m(ti,"href","/docs/transformers/pr_18351/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.pad"),m(Lo,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(oi,"href","/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2ProcessorWithLM"),m(Vt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(ai,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Ge,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Ft,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Bo,"id","transformers.models.wav2vec2_with_lm.processing_wav2vec2_with_lm.Wav2Vec2DecoderWithLMOutput"),m(Bo,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(Bo,"href","#transformers.models.wav2vec2_with_lm.processing_wav2vec2_with_lm.Wav2Vec2DecoderWithLMOutput"),m(Ut,"class","relative group"),m(Rt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Ht,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(si,"href","/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2ForPreTraining"),m(Xt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Uo,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(vt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Ro,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(bt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Ho,"id","transformers.Wav2Vec2Model"),m(Ho,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(Ho,"href","#transformers.Wav2Vec2Model"),m(Gt,"class","relative group"),m(Xs,"href","https://arxiv.org/abs/2006.11477"),m(Xs,"rel","nofollow"),m(ni,"href","/docs/transformers/pr_18351/en/main_classes/model#transformers.PreTrainedModel"),m(Zs,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),m(Zs,"rel","nofollow"),m(ri,"href","/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2Model"),m(Je,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Ce,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Jo,"id","transformers.Wav2Vec2ForCTC"),m(Jo,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(Jo,"href","#transformers.Wav2Vec2ForCTC"),m(Zt,"class","relative group"),m(en,"href","https://arxiv.org/abs/2006.11477"),m(en,"rel","nofollow"),m(ii,"href","/docs/transformers/pr_18351/en/main_classes/model#transformers.PreTrainedModel"),m(an,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),m(an,"rel","nofollow"),m(li,"href","/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC"),m(Ae,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Yo,"id","transformers.Wav2Vec2ForSequenceClassification"),m(Yo,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(Yo,"href","#transformers.Wav2Vec2ForSequenceClassification"),m(Yt,"class","relative group"),m(cn,"href","https://arxiv.org/abs/2006.11477"),m(cn,"rel","nofollow"),m(ci,"href","/docs/transformers/pr_18351/en/main_classes/model#transformers.PreTrainedModel"),m(mn,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),m(mn,"rel","nofollow"),m(di,"href","/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2ForSequenceClassification"),m(Oe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(pe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(aa,"id","transformers.Wav2Vec2ForAudioFrameClassification"),m(aa,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(aa,"href","#transformers.Wav2Vec2ForAudioFrameClassification"),m(to,"class","relative group"),m(_n,"href","https://arxiv.org/abs/2006.11477"),m(_n,"rel","nofollow"),m(pi,"href","/docs/transformers/pr_18351/en/main_classes/model#transformers.PreTrainedModel"),m(wn,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),m(wn,"rel","nofollow"),m(mi,"href","/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2ForAudioFrameClassification"),m(Ze,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(me,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(ra,"id","transformers.Wav2Vec2ForXVector"),m(ra,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(ra,"href","#transformers.Wav2Vec2ForXVector"),m(ao,"class","relative group"),m($n,"href","https://arxiv.org/abs/2006.11477"),m($n,"rel","nofollow"),m(hi,"href","/docs/transformers/pr_18351/en/main_classes/model#transformers.PreTrainedModel"),m(Vn,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),m(Vn,"rel","nofollow"),m(fi,"href","/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2ForXVector"),m(Ke,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(he,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(ca,"id","transformers.Wav2Vec2ForPreTraining"),m(ca,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(ca,"href","#transformers.Wav2Vec2ForPreTraining"),m(no,"class","relative group"),m(qn,"href","https://arxiv.org/abs/2006.11477"),m(qn,"rel","nofollow"),m(ui,"href","/docs/transformers/pr_18351/en/main_classes/model#transformers.PreTrainedModel"),m(zn,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),m(zn,"rel","nofollow"),m(gi,"href","/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.Wav2Vec2ForPreTraining"),m(Qe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(qe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(ma,"id","transformers.TFWav2Vec2Model"),m(ma,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(ma,"href","#transformers.TFWav2Vec2Model"),m(lo,"class","relative group"),m(_i,"href","/docs/transformers/pr_18351/en/main_classes/model#transformers.TFPreTrainedModel"),m(In,"href","https://www.tensorflow.org/api_docs/python/tf/keras/Model"),m(In,"rel","nofollow"),m(vi,"href","/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.TFWav2Vec2Model"),m(Ye,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(fe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(ga,"id","transformers.TFWav2Vec2ForCTC"),m(ga,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(ga,"href","#transformers.TFWav2Vec2ForCTC"),m(po,"class","relative group"),m(bi,"href","/docs/transformers/pr_18351/en/main_classes/model#transformers.TFPreTrainedModel"),m(Gn,"href","https://www.tensorflow.org/api_docs/python/tf/keras/Model"),m(Gn,"rel","nofollow"),m(wi,"href","/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.TFWav2Vec2ForCTC"),m(et,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(ue,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(wa,"id","transformers.FlaxWav2Vec2Model"),m(wa,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(wa,"href","#transformers.FlaxWav2Vec2Model"),m(ho,"class","relative group"),m(Yn,"href","https://arxiv.org/abs/2006.11477"),m(Yn,"rel","nofollow"),m(yi,"href","/docs/transformers/pr_18351/en/main_classes/model#transformers.FlaxPreTrainedModel"),m(or,"href","https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html"),m(or,"rel","nofollow"),m(ar,"href","https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit"),m(ar,"rel","nofollow"),m(sr,"href","https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation"),m(sr,"rel","nofollow"),m(nr,"href","https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap"),m(nr,"rel","nofollow"),m(rr,"href","https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap"),m(rr,"rel","nofollow"),m(tt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Ta,"id","transformers.FlaxWav2Vec2ForCTC"),m(Ta,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(Ta,"href","#transformers.FlaxWav2Vec2ForCTC"),m(uo,"class","relative group"),m(dr,"href","https://arxiv.org/abs/2006.11477"),m(dr,"rel","nofollow"),m(ki,"href","/docs/transformers/pr_18351/en/main_classes/model#transformers.FlaxPreTrainedModel"),m(hr,"href","https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html"),m(hr,"rel","nofollow"),m(fr,"href","https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit"),m(fr,"rel","nofollow"),m(ur,"href","https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation"),m(ur,"rel","nofollow"),m(gr,"href","https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap"),m(gr,"rel","nofollow"),m(_r,"href","https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap"),m(_r,"rel","nofollow"),m(ot,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Wa,"id","transformers.FlaxWav2Vec2ForPreTraining"),m(Wa,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(Wa,"href","#transformers.FlaxWav2Vec2ForPreTraining"),m(vo,"class","relative group"),m(yr,"href","https://arxiv.org/abs/2006.11477"),m(yr,"rel","nofollow"),m(Ti,"href","/docs/transformers/pr_18351/en/main_classes/model#transformers.FlaxPreTrainedModel"),m(xr,"href","https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html"),m(xr,"rel","nofollow"),m($r,"href","https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit"),m($r,"rel","nofollow"),m(Wr,"href","https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation"),m(Wr,"rel","nofollow"),m(jr,"href","https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap"),m(jr,"rel","nofollow"),m(Vr,"href","https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap"),m(Vr,"rel","nofollow"),m(xi,"href","/docs/transformers/pr_18351/en/model_doc/wav2vec2#transformers.FlaxWav2Vec2ForPreTraining"),m(at,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(te,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(o,g){e(document.head,c),_(o,b,g),_(o,u,g),e(u,f),e(f,v),k(l,v,null),e(u,h),e(u,V),e(V,A),_(o,M,g),_(o,C,g),e(C,z),e(z,L),k(U,L,null),e(C,O),e(C,E),e(E,ae),_(o,X,g),_(o,I,g),e(I,N),e(I,S),e(S,H),e(I,D),_(o,P,g),_(o,re,g),e(re,se),_(o,Te,g),_(o,ie,g),e(ie,le),e(le,ct),_(o,He,g),_(o,B,g),e(B,dt),_(o,be,g),_(o,we,g),e(we,Ne),e(Ne,xe),e(we,pt),e(we,$e),e($e,ce),e($e,Pe),e(Pe,We),e($e,mt),_(o,j,g),_(o,q,g),e(q,Be),e(q,Ue),e(Ue,zt),e(q,de),_(o,$t,g),_(o,je,g),e(je,Me),e(Me,J),k(Z,J,null),e(je,At),e(je,ht),e(ht,Ve),_(o,Wt,g),_(o,K,g),k(Fe,K,null),e(K,Ot),e(K,Lt),e(Lt,Kp),e(Lt,Lr),e(Lr,Qp),e(Lt,Yp),e(Lt,Ga),e(Ga,em),e(Lt,tm),e(K,om),e(K,Dt),e(Dt,am),e(Dt,Dr),e(Dr,sm),e(Dt,nm),e(Dt,Sr),e(Sr,rm),e(Dt,im),e(K,lm),k(To,K,null),_(o,ed,g),_(o,St,g),e(St,xo),e(xo,Ri),k(Ja,Ri,null),e(St,cm),e(St,Hi),e(Hi,dm),_(o,td,g),_(o,Q,g),k(Za,Q,null),e(Q,pm),e(Q,Xi),e(Xi,mm),e(Q,hm),e(Q,Ka),e(Ka,fm),e(Ka,Ir),e(Ir,um),e(Ka,gm),e(Q,_m),e(Q,$o),k(Qa,$o,null),e($o,vm),e($o,Gi),e(Gi,bm),e(Q,wm),e(Q,Nr),k(Ya,Nr,null),e(Q,ym),e(Q,Xe),k(es,Xe,null),e(Xe,km),e(Xe,Ji),e(Ji,Tm),e(Xe,xm),e(Xe,ts),e(ts,$m),e(ts,Zi),e(Zi,Wm),e(ts,jm),e(Xe,Vm),k(Wo,Xe,null),e(Q,Fm),e(Q,jo),k(os,jo,null),e(jo,Cm),e(jo,Ki),e(Ki,Em),_(o,od,g),_(o,It,g),e(It,Vo),e(Vo,Qi),k(as,Qi,null),e(It,qm),e(It,Yi),e(Yi,Pm),_(o,ad,g),_(o,Re,g),k(ss,Re,null),e(Re,Mm),e(Re,el),e(el,zm),e(Re,Am),e(Re,ns),e(ns,Om),e(ns,Br),e(Br,Lm),e(ns,Dm),e(Re,Sm),e(Re,Fo),k(rs,Fo,null),e(Fo,Im),e(Fo,tl),e(tl,Nm),_(o,sd,g),_(o,Nt,g),e(Nt,Co),e(Co,ol),k(is,ol,null),e(Nt,Bm),e(Nt,al),e(al,Um),_(o,nd,g),_(o,R,g),k(ls,R,null),e(R,Rm),e(R,sl),e(sl,Hm),e(R,Xm),e(R,ze),e(ze,Ur),e(Ur,Gm),e(ze,Jm),e(ze,Rr),e(Rr,Zm),e(ze,Km),e(ze,Hr),e(Hr,Qm),e(ze,Ym),e(ze,cs),e(cs,nl),e(nl,eh),e(cs,th),e(ze,oh),e(ze,Xr),e(Xr,ah),e(ze,sh),e(R,nh),e(R,Eo),k(ds,Eo,null),e(Eo,rh),e(Eo,ft),e(ft,ih),e(ft,ps),e(ps,rl),e(rl,lh),e(ps,ch),e(ft,dh),e(ft,il),e(il,ph),e(ft,mh),e(ft,ms),e(ms,ll),e(ll,hh),e(ms,fh),e(ft,uh),e(R,gh),e(R,qo),k(hs,qo,null),e(qo,_h),e(qo,ut),e(ut,vh),e(ut,Gr),e(Gr,bh),e(ut,wh),e(ut,cl),e(cl,yh),e(ut,kh),e(ut,Jr),e(Jr,Th),e(ut,xh),e(R,$h),e(R,Zr),k(fs,Zr,null),e(R,Wh),e(R,jt),k(us,jt,null),e(jt,jh),e(jt,gs),e(gs,Vh),e(gs,Kr),e(Kr,Fh),e(gs,Ch),e(jt,Eh),k(Po,jt,null),e(R,qh),e(R,Mo),k(_s,Mo,null),e(Mo,Ph),e(Mo,vs),e(vs,Mh),e(vs,Qr),e(Qr,zh),e(vs,Ah),e(R,Oh),e(R,zo),k(bs,zo,null),e(zo,Lh),e(zo,ws),e(ws,Dh),e(ws,Yr),e(Yr,Sh),e(ws,Ih),_(o,rd,g),_(o,Bt,g),e(Bt,Ao),e(Ao,dl),k(ys,dl,null),e(Bt,Nh),e(Bt,pl),e(pl,Bh),_(o,id,g),_(o,G,g),k(ks,G,null),e(G,Uh),e(G,ml),e(ml,Rh),e(G,Hh),e(G,Oo),k(Ts,Oo,null),e(Oo,Xh),e(Oo,gt),e(gt,Gh),e(gt,xs),e(xs,hl),e(hl,Jh),e(xs,Zh),e(gt,Kh),e(gt,fl),e(fl,Qh),e(gt,Yh),e(gt,$s),e($s,ul),e(ul,ef),e($s,tf),e(gt,of),e(G,af),e(G,Lo),k(Ws,Lo,null),e(Lo,sf),e(Lo,_t),e(_t,nf),e(_t,ei),e(ei,rf),e(_t,lf),e(_t,gl),e(gl,cf),e(_t,df),e(_t,ti),e(ti,pf),e(_t,mf),e(G,hf),e(G,Vt),k(js,Vt,null),e(Vt,ff),e(Vt,Vs),e(Vs,uf),e(Vs,oi),e(oi,gf),e(Vs,_f),e(Vt,vf),k(Do,Vt,null),e(G,bf),e(G,ai),k(Fs,ai,null),e(G,wf),e(G,Ge),k(Cs,Ge,null),e(Ge,yf),e(Ge,_l),e(_l,kf),e(Ge,Tf),k(So,Ge,null),e(Ge,xf),k(Io,Ge,null),e(G,$f),e(G,Ft),k(Es,Ft,null),e(Ft,Wf),e(Ft,vl),e(vl,jf),e(Ft,Vf),k(No,Ft,null),_(o,ld,g),_(o,Ut,g),e(Ut,Bo),e(Bo,bl),k(qs,bl,null),e(Ut,Ff),e(Ut,wl),e(wl,Cf),_(o,cd,g),_(o,Rt,g),k(Ps,Rt,null),e(Rt,Ef),e(Rt,Ms),e(Ms,qf),e(Ms,yl),e(yl,Pf),e(Ms,Mf),_(o,dd,g),_(o,Ht,g),k(zs,Ht,null),e(Ht,zf),e(Ht,kl),e(kl,Af),_(o,pd,g),_(o,Xt,g),k(As,Xt,null),e(Xt,Of),e(Xt,Os),e(Os,Lf),e(Os,si),e(si,Df),e(Os,Sf),_(o,md,g),_(o,vt,g),k(Ls,vt,null),e(vt,If),e(vt,Ds),e(Ds,Nf),e(Ds,Tl),e(Tl,Bf),e(Ds,Uf),e(vt,Rf),e(vt,Uo),k(Ss,Uo,null),e(Uo,Hf),e(Uo,xl),e(xl,Xf),_(o,hd,g),_(o,bt,g),k(Is,bt,null),e(bt,Gf),e(bt,Ns),e(Ns,Jf),e(Ns,$l),e($l,Zf),e(Ns,Kf),e(bt,Qf),e(bt,Ro),k(Bs,Ro,null),e(Ro,Yf),e(Ro,Wl),e(Wl,eu),_(o,fd,g),_(o,Gt,g),e(Gt,Ho),e(Ho,jl),k(Us,jl,null),e(Gt,tu),e(Gt,Vl),e(Vl,ou),_(o,ud,g),_(o,Ce,g),k(Rs,Ce,null),e(Ce,au),e(Ce,Hs),e(Hs,su),e(Hs,Xs),e(Xs,nu),e(Hs,ru),e(Ce,iu),e(Ce,Gs),e(Gs,lu),e(Gs,ni),e(ni,cu),e(Gs,du),e(Ce,pu),e(Ce,Js),e(Js,mu),e(Js,Zs),e(Zs,hu),e(Js,fu),e(Ce,uu),e(Ce,Je),k(Ks,Je,null),e(Je,gu),e(Je,Jt),e(Jt,_u),e(Jt,ri),e(ri,vu),e(Jt,bu),e(Jt,Fl),e(Fl,wu),e(Jt,yu),e(Je,ku),k(Xo,Je,null),e(Je,Tu),k(Go,Je,null),_(o,gd,g),_(o,Zt,g),e(Zt,Jo),e(Jo,Cl),k(Qs,Cl,null),e(Zt,xu),e(Zt,El),e(El,$u),_(o,_d,g),_(o,Ee,g),k(Ys,Ee,null),e(Ee,Wu),e(Ee,Kt),e(Kt,ju),e(Kt,ql),e(ql,Vu),e(Kt,Fu),e(Kt,en),e(en,Cu),e(Kt,Eu),e(Ee,qu),e(Ee,tn),e(tn,Pu),e(tn,ii),e(ii,Mu),e(tn,zu),e(Ee,Au),e(Ee,on),e(on,Ou),e(on,an),e(an,Lu),e(on,Du),e(Ee,Su),e(Ee,Ae),k(sn,Ae,null),e(Ae,Iu),e(Ae,Qt),e(Qt,Nu),e(Qt,li),e(li,Bu),e(Qt,Uu),e(Qt,Pl),e(Pl,Ru),e(Qt,Hu),e(Ae,Xu),k(Zo,Ae,null),e(Ae,Gu),k(Ko,Ae,null),e(Ae,Ju),k(Qo,Ae,null),_(o,vd,g),_(o,Yt,g),e(Yt,Yo),e(Yo,Ml),k(nn,Ml,null),e(Yt,Zu),e(Yt,zl),e(zl,Ku),_(o,bd,g),_(o,pe,g),k(rn,pe,null),e(pe,Qu),e(pe,Al),e(Al,Yu),e(pe,eg),e(pe,ln),e(ln,tg),e(ln,cn),e(cn,og),e(ln,ag),e(pe,sg),e(pe,dn),e(dn,ng),e(dn,ci),e(ci,rg),e(dn,ig),e(pe,lg),e(pe,pn),e(pn,cg),e(pn,mn),e(mn,dg),e(pn,pg),e(pe,mg),e(pe,Oe),k(hn,Oe,null),e(Oe,hg),e(Oe,eo),e(eo,fg),e(eo,di),e(di,ug),e(eo,gg),e(eo,Ol),e(Ol,_g),e(eo,vg),e(Oe,bg),k(ea,Oe,null),e(Oe,wg),k(ta,Oe,null),e(Oe,yg),k(oa,Oe,null),_(o,wd,g),_(o,to,g),e(to,aa),e(aa,Ll),k(fn,Ll,null),e(to,kg),e(to,Dl),e(Dl,Tg),_(o,yd,g),_(o,me,g),k(un,me,null),e(me,xg),e(me,Sl),e(Sl,$g),e(me,Wg),e(me,gn),e(gn,jg),e(gn,_n),e(_n,Vg),e(gn,Fg),e(me,Cg),e(me,vn),e(vn,Eg),e(vn,pi),e(pi,qg),e(vn,Pg),e(me,Mg),e(me,bn),e(bn,zg),e(bn,wn),e(wn,Ag),e(bn,Og),e(me,Lg),e(me,Ze),k(yn,Ze,null),e(Ze,Dg),e(Ze,oo),e(oo,Sg),e(oo,mi),e(mi,Ig),e(oo,Ng),e(oo,Il),e(Il,Bg),e(oo,Ug),e(Ze,Rg),k(sa,Ze,null),e(Ze,Hg),k(na,Ze,null),_(o,kd,g),_(o,ao,g),e(ao,ra),e(ra,Nl),k(kn,Nl,null),e(ao,Xg),e(ao,Bl),e(Bl,Gg),_(o,Td,g),_(o,he,g),k(Tn,he,null),e(he,Jg),e(he,Ul),e(Ul,Zg),e(he,Kg),e(he,xn),e(xn,Qg),e(xn,$n),e($n,Yg),e(xn,e_),e(he,t_),e(he,Wn),e(Wn,o_),e(Wn,hi),e(hi,a_),e(Wn,s_),e(he,n_),e(he,jn),e(jn,r_),e(jn,Vn),e(Vn,i_),e(jn,l_),e(he,c_),e(he,Ke),k(Fn,Ke,null),e(Ke,d_),e(Ke,so),e(so,p_),e(so,fi),e(fi,m_),e(so,h_),e(so,Rl),e(Rl,f_),e(so,u_),e(Ke,g_),k(ia,Ke,null),e(Ke,__),k(la,Ke,null),_(o,xd,g),_(o,no,g),e(no,ca),e(ca,Hl),k(Cn,Hl,null),e(no,v_),e(no,Xl),e(Xl,b_),_(o,$d,g),_(o,qe,g),k(En,qe,null),e(qe,w_),e(qe,ro),e(ro,y_),e(ro,Gl),e(Gl,k_),e(ro,T_),e(ro,qn),e(qn,x_),e(ro,$_),e(qe,W_),e(qe,Pn),e(Pn,j_),e(Pn,ui),e(ui,V_),e(Pn,F_),e(qe,C_),e(qe,Mn),e(Mn,E_),e(Mn,zn),e(zn,q_),e(Mn,P_),e(qe,M_),e(qe,Qe),k(An,Qe,null),e(Qe,z_),e(Qe,io),e(io,A_),e(io,gi),e(gi,O_),e(io,L_),e(io,Jl),e(Jl,D_),e(io,S_),e(Qe,I_),k(da,Qe,null),e(Qe,N_),k(pa,Qe,null),_(o,Wd,g),_(o,lo,g),e(lo,ma),e(ma,Zl),k(On,Zl,null),e(lo,B_),e(lo,Kl),e(Kl,U_),_(o,jd,g),_(o,fe,g),k(Ln,fe,null),e(fe,R_),e(fe,Ql),e(Ql,H_),e(fe,X_),e(fe,Dn),e(Dn,G_),e(Dn,_i),e(_i,J_),e(Dn,Z_),e(fe,K_),e(fe,Sn),e(Sn,Q_),e(Sn,In),e(In,Y_),e(Sn,ev),e(fe,tv),k(ha,fe,null),e(fe,ov),e(fe,Ye),k(Nn,Ye,null),e(Ye,av),e(Ye,co),e(co,sv),e(co,vi),e(vi,nv),e(co,rv),e(co,Yl),e(Yl,iv),e(co,lv),e(Ye,cv),k(fa,Ye,null),e(Ye,dv),k(ua,Ye,null),_(o,Vd,g),_(o,po,g),e(po,ga),e(ga,ec),k(Bn,ec,null),e(po,pv),e(po,tc),e(tc,mv),_(o,Fd,g),_(o,ue,g),k(Un,ue,null),e(ue,hv),e(ue,Rn),e(Rn,fv),e(Rn,oc),e(oc,uv),e(Rn,gv),e(ue,_v),e(ue,Hn),e(Hn,vv),e(Hn,bi),e(bi,bv),e(Hn,wv),e(ue,yv),e(ue,Xn),e(Xn,kv),e(Xn,Gn),e(Gn,Tv),e(Xn,xv),e(ue,$v),k(_a,ue,null),e(ue,Wv),e(ue,et),k(Jn,et,null),e(et,jv),e(et,mo),e(mo,Vv),e(mo,wi),e(wi,Fv),e(mo,Cv),e(mo,ac),e(ac,Ev),e(mo,qv),e(et,Pv),k(va,et,null),e(et,Mv),k(ba,et,null),_(o,Cd,g),_(o,ho,g),e(ho,wa),e(wa,sc),k(Zn,sc,null),e(ho,zv),e(ho,nc),e(nc,Av),_(o,Ed,g),_(o,Y,g),k(Kn,Y,null),e(Y,Ov),e(Y,Qn),e(Qn,Lv),e(Qn,Yn),e(Yn,Dv),e(Qn,Sv),e(Y,Iv),e(Y,er),e(er,Nv),e(er,yi),e(yi,Bv),e(er,Uv),e(Y,Rv),e(Y,tr),e(tr,Hv),e(tr,or),e(or,Xv),e(tr,Gv),e(Y,Jv),e(Y,rc),e(rc,Zv),e(Y,Kv),e(Y,wt),e(wt,ic),e(ic,ar),e(ar,Qv),e(wt,Yv),e(wt,lc),e(lc,sr),e(sr,e2),e(wt,t2),e(wt,cc),e(cc,nr),e(nr,o2),e(wt,a2),e(wt,dc),e(dc,rr),e(rr,s2),e(Y,n2),e(Y,tt),k(ir,tt,null),e(tt,r2),e(tt,fo),e(fo,i2),e(fo,pc),e(pc,l2),e(fo,c2),e(fo,mc),e(mc,d2),e(fo,p2),e(tt,m2),k(ya,tt,null),e(tt,h2),k(ka,tt,null),_(o,qd,g),_(o,uo,g),e(uo,Ta),e(Ta,hc),k(lr,hc,null),e(uo,f2),e(uo,fc),e(fc,u2),_(o,Pd,g),_(o,ee,g),k(cr,ee,null),e(ee,g2),e(ee,go),e(go,_2),e(go,uc),e(uc,v2),e(go,b2),e(go,dr),e(dr,w2),e(go,y2),e(ee,k2),e(ee,pr),e(pr,T2),e(pr,ki),e(ki,x2),e(pr,$2),e(ee,W2),e(ee,mr),e(mr,j2),e(mr,hr),e(hr,V2),e(mr,F2),e(ee,C2),e(ee,gc),e(gc,E2),e(ee,q2),e(ee,yt),e(yt,_c),e(_c,fr),e(fr,P2),e(yt,M2),e(yt,vc),e(vc,ur),e(ur,z2),e(yt,A2),e(yt,bc),e(bc,gr),e(gr,O2),e(yt,L2),e(yt,wc),e(wc,_r),e(_r,D2),e(ee,S2),e(ee,ot),k(vr,ot,null),e(ot,I2),e(ot,_o),e(_o,N2),e(_o,yc),e(yc,B2),e(_o,U2),e(_o,kc),e(kc,R2),e(_o,H2),e(ot,X2),k(xa,ot,null),e(ot,G2),k($a,ot,null),_(o,Md,g),_(o,vo,g),e(vo,Wa),e(Wa,Tc),k(br,Tc,null),e(vo,J2),e(vo,xc),e(xc,Z2),_(o,zd,g),_(o,te,g),k(wr,te,null),e(te,K2),e(te,bo),e(bo,Q2),e(bo,$c),e($c,Y2),e(bo,eb),e(bo,yr),e(yr,tb),e(bo,ob),e(te,ab),e(te,kr),e(kr,sb),e(kr,Ti),e(Ti,nb),e(kr,rb),e(te,ib),e(te,Tr),e(Tr,lb),e(Tr,xr),e(xr,cb),e(Tr,db),e(te,pb),e(te,Wc),e(Wc,mb),e(te,hb),e(te,kt),e(kt,jc),e(jc,$r),e($r,fb),e(kt,ub),e(kt,Vc),e(Vc,Wr),e(Wr,gb),e(kt,_b),e(kt,Fc),e(Fc,jr),e(jr,vb),e(kt,bb),e(kt,Cc),e(Cc,Vr),e(Vr,wb),e(te,yb),e(te,at),k(Fr,at,null),e(at,kb),e(at,wo),e(wo,Tb),e(wo,xi),e(xi,xb),e(wo,$b),e(wo,Ec),e(Ec,Wb),e(wo,jb),e(at,Vb),k(ja,at,null),e(at,Fb),k(Va,at,null),Ad=!0},p(o,[g]){const Cr={};g&2&&(Cr.$$scope={dirty:g,ctx:o}),To.$set(Cr);const qc={};g&2&&(qc.$$scope={dirty:g,ctx:o}),Wo.$set(qc);const Pc={};g&2&&(Pc.$$scope={dirty:g,ctx:o}),Po.$set(Pc);const Mc={};g&2&&(Mc.$$scope={dirty:g,ctx:o}),Do.$set(Mc);const Er={};g&2&&(Er.$$scope={dirty:g,ctx:o}),So.$set(Er);const zc={};g&2&&(zc.$$scope={dirty:g,ctx:o}),Io.$set(zc);const Ac={};g&2&&(Ac.$$scope={dirty:g,ctx:o}),No.$set(Ac);const Oc={};g&2&&(Oc.$$scope={dirty:g,ctx:o}),Xo.$set(Oc);const qr={};g&2&&(qr.$$scope={dirty:g,ctx:o}),Go.$set(qr);const Lc={};g&2&&(Lc.$$scope={dirty:g,ctx:o}),Zo.$set(Lc);const Dc={};g&2&&(Dc.$$scope={dirty:g,ctx:o}),Ko.$set(Dc);const Sc={};g&2&&(Sc.$$scope={dirty:g,ctx:o}),Qo.$set(Sc);const Ic={};g&2&&(Ic.$$scope={dirty:g,ctx:o}),ea.$set(Ic);const Nc={};g&2&&(Nc.$$scope={dirty:g,ctx:o}),ta.$set(Nc);const Pr={};g&2&&(Pr.$$scope={dirty:g,ctx:o}),oa.$set(Pr);const Bc={};g&2&&(Bc.$$scope={dirty:g,ctx:o}),sa.$set(Bc);const Mr={};g&2&&(Mr.$$scope={dirty:g,ctx:o}),na.$set(Mr);const Uc={};g&2&&(Uc.$$scope={dirty:g,ctx:o}),ia.$set(Uc);const zr={};g&2&&(zr.$$scope={dirty:g,ctx:o}),la.$set(zr);const Rc={};g&2&&(Rc.$$scope={dirty:g,ctx:o}),da.$set(Rc);const Ar={};g&2&&(Ar.$$scope={dirty:g,ctx:o}),pa.$set(Ar);const Hc={};g&2&&(Hc.$$scope={dirty:g,ctx:o}),ha.$set(Hc);const Xc={};g&2&&(Xc.$$scope={dirty:g,ctx:o}),fa.$set(Xc);const Gc={};g&2&&(Gc.$$scope={dirty:g,ctx:o}),ua.$set(Gc);const Tt={};g&2&&(Tt.$$scope={dirty:g,ctx:o}),_a.$set(Tt);const yo={};g&2&&(yo.$$scope={dirty:g,ctx:o}),va.$set(yo);const Jc={};g&2&&(Jc.$$scope={dirty:g,ctx:o}),ba.$set(Jc);const Zc={};g&2&&(Zc.$$scope={dirty:g,ctx:o}),ya.$set(Zc);const ko={};g&2&&(ko.$$scope={dirty:g,ctx:o}),ka.$set(ko);const Kc={};g&2&&(Kc.$$scope={dirty:g,ctx:o}),xa.$set(Kc);const Qc={};g&2&&(Qc.$$scope={dirty:g,ctx:o}),$a.$set(Qc);const Or={};g&2&&(Or.$$scope={dirty:g,ctx:o}),ja.$set(Or);const Yc={};g&2&&(Yc.$$scope={dirty:g,ctx:o}),Va.$set(Yc)},i(o){Ad||(T(l.$$.fragment,o),T(U.$$.fragment,o),T(Z.$$.fragment,o),T(Fe.$$.fragment,o),T(To.$$.fragment,o),T(Ja.$$.fragment,o),T(Za.$$.fragment,o),T(Qa.$$.fragment,o),T(Ya.$$.fragment,o),T(es.$$.fragment,o),T(Wo.$$.fragment,o),T(os.$$.fragment,o),T(as.$$.fragment,o),T(ss.$$.fragment,o),T(rs.$$.fragment,o),T(is.$$.fragment,o),T(ls.$$.fragment,o),T(ds.$$.fragment,o),T(hs.$$.fragment,o),T(fs.$$.fragment,o),T(us.$$.fragment,o),T(Po.$$.fragment,o),T(_s.$$.fragment,o),T(bs.$$.fragment,o),T(ys.$$.fragment,o),T(ks.$$.fragment,o),T(Ts.$$.fragment,o),T(Ws.$$.fragment,o),T(js.$$.fragment,o),T(Do.$$.fragment,o),T(Fs.$$.fragment,o),T(Cs.$$.fragment,o),T(So.$$.fragment,o),T(Io.$$.fragment,o),T(Es.$$.fragment,o),T(No.$$.fragment,o),T(qs.$$.fragment,o),T(Ps.$$.fragment,o),T(zs.$$.fragment,o),T(As.$$.fragment,o),T(Ls.$$.fragment,o),T(Ss.$$.fragment,o),T(Is.$$.fragment,o),T(Bs.$$.fragment,o),T(Us.$$.fragment,o),T(Rs.$$.fragment,o),T(Ks.$$.fragment,o),T(Xo.$$.fragment,o),T(Go.$$.fragment,o),T(Qs.$$.fragment,o),T(Ys.$$.fragment,o),T(sn.$$.fragment,o),T(Zo.$$.fragment,o),T(Ko.$$.fragment,o),T(Qo.$$.fragment,o),T(nn.$$.fragment,o),T(rn.$$.fragment,o),T(hn.$$.fragment,o),T(ea.$$.fragment,o),T(ta.$$.fragment,o),T(oa.$$.fragment,o),T(fn.$$.fragment,o),T(un.$$.fragment,o),T(yn.$$.fragment,o),T(sa.$$.fragment,o),T(na.$$.fragment,o),T(kn.$$.fragment,o),T(Tn.$$.fragment,o),T(Fn.$$.fragment,o),T(ia.$$.fragment,o),T(la.$$.fragment,o),T(Cn.$$.fragment,o),T(En.$$.fragment,o),T(An.$$.fragment,o),T(da.$$.fragment,o),T(pa.$$.fragment,o),T(On.$$.fragment,o),T(Ln.$$.fragment,o),T(ha.$$.fragment,o),T(Nn.$$.fragment,o),T(fa.$$.fragment,o),T(ua.$$.fragment,o),T(Bn.$$.fragment,o),T(Un.$$.fragment,o),T(_a.$$.fragment,o),T(Jn.$$.fragment,o),T(va.$$.fragment,o),T(ba.$$.fragment,o),T(Zn.$$.fragment,o),T(Kn.$$.fragment,o),T(ir.$$.fragment,o),T(ya.$$.fragment,o),T(ka.$$.fragment,o),T(lr.$$.fragment,o),T(cr.$$.fragment,o),T(vr.$$.fragment,o),T(xa.$$.fragment,o),T($a.$$.fragment,o),T(br.$$.fragment,o),T(wr.$$.fragment,o),T(Fr.$$.fragment,o),T(ja.$$.fragment,o),T(Va.$$.fragment,o),Ad=!0)},o(o){x(l.$$.fragment,o),x(U.$$.fragment,o),x(Z.$$.fragment,o),x(Fe.$$.fragment,o),x(To.$$.fragment,o),x(Ja.$$.fragment,o),x(Za.$$.fragment,o),x(Qa.$$.fragment,o),x(Ya.$$.fragment,o),x(es.$$.fragment,o),x(Wo.$$.fragment,o),x(os.$$.fragment,o),x(as.$$.fragment,o),x(ss.$$.fragment,o),x(rs.$$.fragment,o),x(is.$$.fragment,o),x(ls.$$.fragment,o),x(ds.$$.fragment,o),x(hs.$$.fragment,o),x(fs.$$.fragment,o),x(us.$$.fragment,o),x(Po.$$.fragment,o),x(_s.$$.fragment,o),x(bs.$$.fragment,o),x(ys.$$.fragment,o),x(ks.$$.fragment,o),x(Ts.$$.fragment,o),x(Ws.$$.fragment,o),x(js.$$.fragment,o),x(Do.$$.fragment,o),x(Fs.$$.fragment,o),x(Cs.$$.fragment,o),x(So.$$.fragment,o),x(Io.$$.fragment,o),x(Es.$$.fragment,o),x(No.$$.fragment,o),x(qs.$$.fragment,o),x(Ps.$$.fragment,o),x(zs.$$.fragment,o),x(As.$$.fragment,o),x(Ls.$$.fragment,o),x(Ss.$$.fragment,o),x(Is.$$.fragment,o),x(Bs.$$.fragment,o),x(Us.$$.fragment,o),x(Rs.$$.fragment,o),x(Ks.$$.fragment,o),x(Xo.$$.fragment,o),x(Go.$$.fragment,o),x(Qs.$$.fragment,o),x(Ys.$$.fragment,o),x(sn.$$.fragment,o),x(Zo.$$.fragment,o),x(Ko.$$.fragment,o),x(Qo.$$.fragment,o),x(nn.$$.fragment,o),x(rn.$$.fragment,o),x(hn.$$.fragment,o),x(ea.$$.fragment,o),x(ta.$$.fragment,o),x(oa.$$.fragment,o),x(fn.$$.fragment,o),x(un.$$.fragment,o),x(yn.$$.fragment,o),x(sa.$$.fragment,o),x(na.$$.fragment,o),x(kn.$$.fragment,o),x(Tn.$$.fragment,o),x(Fn.$$.fragment,o),x(ia.$$.fragment,o),x(la.$$.fragment,o),x(Cn.$$.fragment,o),x(En.$$.fragment,o),x(An.$$.fragment,o),x(da.$$.fragment,o),x(pa.$$.fragment,o),x(On.$$.fragment,o),x(Ln.$$.fragment,o),x(ha.$$.fragment,o),x(Nn.$$.fragment,o),x(fa.$$.fragment,o),x(ua.$$.fragment,o),x(Bn.$$.fragment,o),x(Un.$$.fragment,o),x(_a.$$.fragment,o),x(Jn.$$.fragment,o),x(va.$$.fragment,o),x(ba.$$.fragment,o),x(Zn.$$.fragment,o),x(Kn.$$.fragment,o),x(ir.$$.fragment,o),x(ya.$$.fragment,o),x(ka.$$.fragment,o),x(lr.$$.fragment,o),x(cr.$$.fragment,o),x(vr.$$.fragment,o),x(xa.$$.fragment,o),x($a.$$.fragment,o),x(br.$$.fragment,o),x(wr.$$.fragment,o),x(Fr.$$.fragment,o),x(ja.$$.fragment,o),x(Va.$$.fragment,o),Ad=!1},d(o){t(c),o&&t(b),o&&t(u),$(l),o&&t(M),o&&t(C),$(U),o&&t(X),o&&t(I),o&&t(P),o&&t(re),o&&t(Te),o&&t(ie),o&&t(He),o&&t(B),o&&t(be),o&&t(we),o&&t(j),o&&t(q),o&&t($t),o&&t(je),$(Z),o&&t(Wt),o&&t(K),$(Fe),$(To),o&&t(ed),o&&t(St),$(Ja),o&&t(td),o&&t(Q),$(Za),$(Qa),$(Ya),$(es),$(Wo),$(os),o&&t(od),o&&t(It),$(as),o&&t(ad),o&&t(Re),$(ss),$(rs),o&&t(sd),o&&t(Nt),$(is),o&&t(nd),o&&t(R),$(ls),$(ds),$(hs),$(fs),$(us),$(Po),$(_s),$(bs),o&&t(rd),o&&t(Bt),$(ys),o&&t(id),o&&t(G),$(ks),$(Ts),$(Ws),$(js),$(Do),$(Fs),$(Cs),$(So),$(Io),$(Es),$(No),o&&t(ld),o&&t(Ut),$(qs),o&&t(cd),o&&t(Rt),$(Ps),o&&t(dd),o&&t(Ht),$(zs),o&&t(pd),o&&t(Xt),$(As),o&&t(md),o&&t(vt),$(Ls),$(Ss),o&&t(hd),o&&t(bt),$(Is),$(Bs),o&&t(fd),o&&t(Gt),$(Us),o&&t(ud),o&&t(Ce),$(Rs),$(Ks),$(Xo),$(Go),o&&t(gd),o&&t(Zt),$(Qs),o&&t(_d),o&&t(Ee),$(Ys),$(sn),$(Zo),$(Ko),$(Qo),o&&t(vd),o&&t(Yt),$(nn),o&&t(bd),o&&t(pe),$(rn),$(hn),$(ea),$(ta),$(oa),o&&t(wd),o&&t(to),$(fn),o&&t(yd),o&&t(me),$(un),$(yn),$(sa),$(na),o&&t(kd),o&&t(ao),$(kn),o&&t(Td),o&&t(he),$(Tn),$(Fn),$(ia),$(la),o&&t(xd),o&&t(no),$(Cn),o&&t($d),o&&t(qe),$(En),$(An),$(da),$(pa),o&&t(Wd),o&&t(lo),$(On),o&&t(jd),o&&t(fe),$(Ln),$(ha),$(Nn),$(fa),$(ua),o&&t(Vd),o&&t(po),$(Bn),o&&t(Fd),o&&t(ue),$(Un),$(_a),$(Jn),$(va),$(ba),o&&t(Cd),o&&t(ho),$(Zn),o&&t(Ed),o&&t(Y),$(Kn),$(ir),$(ya),$(ka),o&&t(qd),o&&t(uo),$(lr),o&&t(Pd),o&&t(ee),$(cr),$(vr),$(xa),$($a),o&&t(Md),o&&t(vo),$(br),o&&t(zd),o&&t(te),$(wr),$(Fr),$(ja),$(Va)}}}const Lk={local:"wav2vec2",sections:[{local:"overview",title:"Overview"},{local:"transformers.Wav2Vec2Config",title:"Wav2Vec2Config"},{local:"transformers.Wav2Vec2CTCTokenizer",title:"Wav2Vec2CTCTokenizer"},{local:"transformers.Wav2Vec2FeatureExtractor",title:"Wav2Vec2FeatureExtractor"},{local:"transformers.Wav2Vec2Processor",title:"Wav2Vec2Processor"},{local:"transformers.Wav2Vec2ProcessorWithLM",title:"Wav2Vec2ProcessorWithLM"},{local:"transformers.models.wav2vec2_with_lm.processing_wav2vec2_with_lm.Wav2Vec2DecoderWithLMOutput",title:"Wav2Vec2 specific outputs"},{local:"transformers.Wav2Vec2Model",title:"Wav2Vec2Model"},{local:"transformers.Wav2Vec2ForCTC",title:"Wav2Vec2ForCTC"},{local:"transformers.Wav2Vec2ForSequenceClassification",title:"Wav2Vec2ForSequenceClassification"},{local:"transformers.Wav2Vec2ForAudioFrameClassification",title:"Wav2Vec2ForAudioFrameClassification"},{local:"transformers.Wav2Vec2ForXVector",title:"Wav2Vec2ForXVector"},{local:"transformers.Wav2Vec2ForPreTraining",title:"Wav2Vec2ForPreTraining"},{local:"transformers.TFWav2Vec2Model",title:"TFWav2Vec2Model"},{local:"transformers.TFWav2Vec2ForCTC",title:"TFWav2Vec2ForCTC"},{local:"transformers.FlaxWav2Vec2Model",title:"FlaxWav2Vec2Model"},{local:"transformers.FlaxWav2Vec2ForCTC",title:"FlaxWav2Vec2ForCTC"},{local:"transformers.FlaxWav2Vec2ForPreTraining",title:"FlaxWav2Vec2ForPreTraining"}],title:"Wav2Vec2"};function Dk(W){return ak(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Hk extends Yy{constructor(c){super();ek(this,c,Dk,Ok,tk,{})}}export{Hk as default,Lk as metadata};
