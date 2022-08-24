import{S as lu,i as iu,s as cu,e as n,k as i,w as v,t as s,M as pu,c as a,d as o,m as c,a as r,x as y,h as d,b as m,G as e,g as _,y as T,q as w,o as S,B as $,v as mu,L as Xe}from"../../chunks/vendor-hf-doc-builder.js";import{T as qn}from"../../chunks/Tip-hf-doc-builder.js";import{D as C}from"../../chunks/Docstring-hf-doc-builder.js";import{C as Ye}from"../../chunks/CodeBlock-hf-doc-builder.js";import{I as xe}from"../../chunks/IconCopyLink-hf-doc-builder.js";import{E as Je}from"../../chunks/ExampleCodeBlock-hf-doc-builder.js";function hu(z){let p,k,f,h,b;return h=new Ye({props:{code:`from transformers import BlenderbotSmallModel, BlenderbotSmallConfig

# Initializing a BlenderbotSmall facebook/blenderbot_small-90M style configuration
configuration = BlenderbotSmallConfig()

# Initializing a model from the facebook/blenderbot_small-90M style configuration
model = BlenderbotSmallModel(configuration)

# Accessing the model configuration
configuration = model.config`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BlenderbotSmallModel, BlenderbotSmallConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a BlenderbotSmall facebook/blenderbot_small-90M style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = BlenderbotSmallConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the facebook/blenderbot_small-90M style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BlenderbotSmallModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`}}),{c(){p=n("p"),k=s("Example:"),f=i(),v(h.$$.fragment)},l(l){p=a(l,"P",{});var u=r(p);k=d(u,"Example:"),u.forEach(o),f=c(l),y(h.$$.fragment,l)},m(l,u){_(l,p,u),e(p,k),_(l,f,u),T(h,l,u),b=!0},p:Xe,i(l){b||(w(h.$$.fragment,l),b=!0)},o(l){S(h.$$.fragment,l),b=!1},d(l){l&&o(p),l&&o(f),$(h,l)}}}function uu(z){let p,k,f,h,b;return{c(){p=n("p"),k=s("Although the recipe for forward pass needs to be defined within this function, one should call the "),f=n("code"),h=s("Module"),b=s(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){p=a(l,"P",{});var u=r(p);k=d(u,"Although the recipe for forward pass needs to be defined within this function, one should call the "),f=a(u,"CODE",{});var F=r(f);h=d(F,"Module"),F.forEach(o),b=d(u,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),u.forEach(o)},m(l,u){_(l,p,u),e(p,k),e(p,f),e(f,h),e(p,b)},d(l){l&&o(p)}}}function fu(z){let p,k,f,h,b;return h=new Ye({props:{code:`from transformers import BlenderbotSmallTokenizer, BlenderbotSmallModel

model = BlenderbotSmallModel.from_pretrained("facebook/blenderbot_small-90M")
tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")

inputs = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt")
decoder_inputs = tokenizer("Studies show that", return_tensors="pt")  # Batch size 1
outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_inputs.input_ids)

last_hidden_states = outputs.last_hidden_state
list(last_hidden_states.shape)`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BlenderbotSmallTokenizer, BlenderbotSmallModel

<span class="hljs-meta">&gt;&gt;&gt; </span>model = BlenderbotSmallModel.from_pretrained(<span class="hljs-string">&quot;facebook/blenderbot_small-90M&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BlenderbotSmallTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/blenderbot_small-90M&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Studies have been shown that owning a dog is good for you&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>decoder_inputs = tokenizer(<span class="hljs-string">&quot;Studies show that&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)  <span class="hljs-comment"># Batch size 1</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_inputs.input_ids)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(last_hidden_states.shape)
[<span class="hljs-number">1</span>, <span class="hljs-number">3</span>, <span class="hljs-number">512</span>]`}}),{c(){p=n("p"),k=s("Example:"),f=i(),v(h.$$.fragment)},l(l){p=a(l,"P",{});var u=r(p);k=d(u,"Example:"),u.forEach(o),f=c(l),y(h.$$.fragment,l)},m(l,u){_(l,p,u),e(p,k),_(l,f,u),T(h,l,u),b=!0},p:Xe,i(l){b||(w(h.$$.fragment,l),b=!0)},o(l){S(h.$$.fragment,l),b=!1},d(l){l&&o(p),l&&o(f),$(h,l)}}}function _u(z){let p,k,f,h,b;return{c(){p=n("p"),k=s("Although the recipe for forward pass needs to be defined within this function, one should call the "),f=n("code"),h=s("Module"),b=s(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){p=a(l,"P",{});var u=r(p);k=d(u,"Although the recipe for forward pass needs to be defined within this function, one should call the "),f=a(u,"CODE",{});var F=r(f);h=d(F,"Module"),F.forEach(o),b=d(u,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),u.forEach(o)},m(l,u){_(l,p,u),e(p,k),e(p,f),e(f,h),e(p,b)},d(l){l&&o(p)}}}function gu(z){let p,k,f,h,b;return h=new Ye({props:{code:`from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration

mname = "facebook/blenderbot_small-90M"
model = BlenderbotSmallForConditionalGeneration.from_pretrained(mname)
tokenizer = BlenderbotSmallTokenizer.from_pretrained(mname)
UTTERANCE = "My friends are cool but they eat too many carbs."
print("Human: ", UTTERANCE)

inputs = tokenizer([UTTERANCE], return_tensors="pt")
reply_ids = model.generate(**inputs)
print("Bot: ", tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0])

REPLY = "I'm not sure"
print("Human: ", REPLY)

NEXT_UTTERANCE = (
    "My friends are cool but they eat too many carbs.</s> <s>what kind of carbs do they eat? "
    "i don't know much about carbs</s> "
    "<s> I'm not sure."
)
inputs = tokenizer([NEXT_UTTERANCE], return_tensors="pt")
next_reply_ids = model.generate(**inputs)
print("Bot: ", tokenizer.batch_decode(next_reply_ids, skip_special_tokens=True)[0])`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>mname = <span class="hljs-string">&quot;facebook/blenderbot_small-90M&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BlenderbotSmallForConditionalGeneration.from_pretrained(mname)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BlenderbotSmallTokenizer.from_pretrained(mname)
<span class="hljs-meta">&gt;&gt;&gt; </span>UTTERANCE = <span class="hljs-string">&quot;My friends are cool but they eat too many carbs.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;Human: &quot;</span>, UTTERANCE)
Human:  My friends are cool but they eat too many carbs.

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer([UTTERANCE], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>reply_ids = model.generate(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;Bot: &quot;</span>, tokenizer.batch_decode(reply_ids, skip_special_tokens=<span class="hljs-literal">True</span>)[<span class="hljs-number">0</span>])
Bot:  what kind of carbs do they eat? i don<span class="hljs-string">&#x27;t know much about carbs.

&gt;&gt;&gt; REPLY = &quot;I&#x27;</span>m <span class="hljs-keyword">not</span> sure<span class="hljs-string">&quot;
&gt;&gt;&gt; print(&quot;</span>Human: <span class="hljs-string">&quot;, REPLY)
Human: I&#x27;m not sure

&gt;&gt;&gt; NEXT_UTTERANCE = (
...     &quot;</span>My friends are cool but they eat too many carbs.&lt;/s&gt; &lt;s&gt;what kind of carbs do they eat? <span class="hljs-string">&quot;
...     &quot;</span>i don<span class="hljs-string">&#x27;t know much about carbs&lt;/s&gt; &quot;
...     &quot;&lt;s&gt; I&#x27;</span>m <span class="hljs-keyword">not</span> sure.<span class="hljs-string">&quot;
... )
&gt;&gt;&gt; inputs = tokenizer([NEXT_UTTERANCE], return_tensors=&quot;</span>pt<span class="hljs-string">&quot;)
&gt;&gt;&gt; next_reply_ids = model.generate(**inputs)
&gt;&gt;&gt; print(&quot;</span>Bot: <span class="hljs-string">&quot;, tokenizer.batch_decode(next_reply_ids, skip_special_tokens=True)[0])
Bot:  they eat a lot of carbs. carbs are high in fat, protein, and carbohydrates.</span>`}}),{c(){p=n("p"),k=s("Conversation example:"),f=i(),v(h.$$.fragment)},l(l){p=a(l,"P",{});var u=r(p);k=d(u,"Conversation example:"),u.forEach(o),f=c(l),y(h.$$.fragment,l)},m(l,u){_(l,p,u),e(p,k),_(l,f,u),T(h,l,u),b=!0},p:Xe,i(l){b||(w(h.$$.fragment,l),b=!0)},o(l){S(h.$$.fragment,l),b=!1},d(l){l&&o(p),l&&o(f),$(h,l)}}}function bu(z){let p,k,f,h,b;return h=new Ye({props:{code:`from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForCausalLM

tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")
model = BlenderbotSmallForCausalLM.from_pretrained(
    "facebook/blenderbot_small-90M", add_cross_attention=False
)
assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

logits = outputs.logits
expected_shape = [1, inputs.input_ids.shape[-1], model.config.vocab_size]
list(logits.shape) == expected_shape`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BlenderbotSmallTokenizer, BlenderbotSmallForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BlenderbotSmallTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/blenderbot_small-90M&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BlenderbotSmallForCausalLM.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;facebook/blenderbot_small-90M&quot;</span>, add_cross_attention=<span class="hljs-literal">False</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">assert</span> model.config.is_decoder, <span class="hljs-string">f&quot;<span class="hljs-subst">{model.__class__}</span> has to be configured as a decoder.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits
<span class="hljs-meta">&gt;&gt;&gt; </span>expected_shape = [<span class="hljs-number">1</span>, inputs.input_ids.shape[-<span class="hljs-number">1</span>], model.config.vocab_size]
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(logits.shape) == expected_shape
<span class="hljs-literal">True</span>`}}),{c(){p=n("p"),k=s("Example:"),f=i(),v(h.$$.fragment)},l(l){p=a(l,"P",{});var u=r(p);k=d(u,"Example:"),u.forEach(o),f=c(l),y(h.$$.fragment,l)},m(l,u){_(l,p,u),e(p,k),_(l,f,u),T(h,l,u),b=!0},p:Xe,i(l){b||(w(h.$$.fragment,l),b=!0)},o(l){S(h.$$.fragment,l),b=!1},d(l){l&&o(p),l&&o(f),$(h,l)}}}function ku(z){let p,k,f,h,b,l,u,F,Ce,he,B,je,W,Oe,Pe,R,Le,Ne,V,Q,Ie,Z,j,L,ue,ae,Be,J,D,be,re,N,ke,se,ze,ee,de,le,Ae,X,Fe,H,De;return{c(){p=n("p"),k=s("TF 2.0 models accepts two formats as inputs:"),f=i(),h=n("ul"),b=n("li"),l=s("having all inputs as keyword arguments (like PyTorch models), or"),u=i(),F=n("li"),Ce=s("having all inputs as a list, tuple or dict in the first positional arguments."),he=i(),B=n("p"),je=s("This second option is useful when using "),W=n("code"),Oe=s("tf.keras.Model.fit"),Pe=s(` method which currently requires having all the
tensors in the first argument of the model call function: `),R=n("code"),Le=s("model(inputs)"),Ne=s("."),V=i(),Q=n("p"),Ie=s(`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),Z=i(),j=n("ul"),L=n("li"),ue=s("a single Tensor with "),ae=n("code"),Be=s("input_ids"),J=s(" only and nothing else: "),D=n("code"),be=s("model(input_ids)"),re=i(),N=n("li"),ke=s(`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),se=n("code"),ze=s("model([input_ids, attention_mask])"),ee=s(" or "),de=n("code"),le=s("model([input_ids, attention_mask, token_type_ids])"),Ae=i(),X=n("li"),Fe=s(`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),H=n("code"),De=s('model({"input_ids": input_ids, "token_type_ids": token_type_ids})')},l(x){p=a(x,"P",{});var q=r(p);k=d(q,"TF 2.0 models accepts two formats as inputs:"),q.forEach(o),f=c(x),h=a(x,"UL",{});var ie=r(h);b=a(ie,"LI",{});var Ze=r(b);l=d(Ze,"having all inputs as keyword arguments (like PyTorch models), or"),Ze.forEach(o),u=c(ie),F=a(ie,"LI",{});var Re=r(F);Ce=d(Re,"having all inputs as a list, tuple or dict in the first positional arguments."),Re.forEach(o),ie.forEach(o),he=c(x),B=a(x,"P",{});var O=r(B);je=d(O,"This second option is useful when using "),W=a(O,"CODE",{});var eo=r(W);Oe=d(eo,"tf.keras.Model.fit"),eo.forEach(o),Pe=d(O,` method which currently requires having all the
tensors in the first argument of the model call function: `),R=a(O,"CODE",{});var ve=r(R);Le=d(ve,"model(inputs)"),ve.forEach(o),Ne=d(O,"."),O.forEach(o),V=c(x),Q=a(x,"P",{});var oo=r(Q);Ie=d(oo,`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),oo.forEach(o),Z=c(x),j=a(x,"UL",{});var Y=r(j);L=a(Y,"LI",{});var G=r(L);ue=d(G,"a single Tensor with "),ae=a(G,"CODE",{});var to=r(ae);Be=d(to,"input_ids"),to.forEach(o),J=d(G," only and nothing else: "),D=a(G,"CODE",{});var no=r(D);be=d(no,"model(input_ids)"),no.forEach(o),G.forEach(o),re=c(Y),N=a(Y,"LI",{});var oe=r(N);ke=d(oe,`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),se=a(oe,"CODE",{});var ce=r(se);ze=d(ce,"model([input_ids, attention_mask])"),ce.forEach(o),ee=d(oe," or "),de=a(oe,"CODE",{});var fe=r(de);le=d(fe,"model([input_ids, attention_mask, token_type_ids])"),fe.forEach(o),oe.forEach(o),Ae=c(Y),X=a(Y,"LI",{});var ye=r(X);Fe=d(ye,`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),H=a(ye,"CODE",{});var Te=r(H);De=d(Te,'model({"input_ids": input_ids, "token_type_ids": token_type_ids})'),Te.forEach(o),ye.forEach(o),Y.forEach(o)},m(x,q){_(x,p,q),e(p,k),_(x,f,q),_(x,h,q),e(h,b),e(b,l),e(h,u),e(h,F),e(F,Ce),_(x,he,q),_(x,B,q),e(B,je),e(B,W),e(W,Oe),e(B,Pe),e(B,R),e(R,Le),e(B,Ne),_(x,V,q),_(x,Q,q),e(Q,Ie),_(x,Z,q),_(x,j,q),e(j,L),e(L,ue),e(L,ae),e(ae,Be),e(L,J),e(L,D),e(D,be),e(j,re),e(j,N),e(N,ke),e(N,se),e(se,ze),e(N,ee),e(N,de),e(de,le),e(j,Ae),e(j,X),e(X,Fe),e(X,H),e(H,De)},d(x){x&&o(p),x&&o(f),x&&o(h),x&&o(he),x&&o(B),x&&o(V),x&&o(Q),x&&o(Z),x&&o(j)}}}function vu(z){let p,k,f,h,b;return{c(){p=n("p"),k=s("Although the recipe for forward pass needs to be defined within this function, one should call the "),f=n("code"),h=s("Module"),b=s(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){p=a(l,"P",{});var u=r(p);k=d(u,"Although the recipe for forward pass needs to be defined within this function, one should call the "),f=a(u,"CODE",{});var F=r(f);h=d(F,"Module"),F.forEach(o),b=d(u,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),u.forEach(o)},m(l,u){_(l,p,u),e(p,k),e(p,f),e(f,h),e(p,b)},d(l){l&&o(p)}}}function yu(z){let p,k,f,h,b;return h=new Ye({props:{code:`from transformers import BlenderbotSmallTokenizer, TFBlenderbotSmallModel
import tensorflow as tf

tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")
model = TFBlenderbotSmallModel.from_pretrained("facebook/blenderbot_small-90M")

inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
outputs = model(inputs)

last_hidden_states = outputs.last_hidden_state`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BlenderbotSmallTokenizer, TFBlenderbotSmallModel
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> tensorflow <span class="hljs-keyword">as</span> tf

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BlenderbotSmallTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/blenderbot_small-90M&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = TFBlenderbotSmallModel.from_pretrained(<span class="hljs-string">&quot;facebook/blenderbot_small-90M&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;tf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`}}),{c(){p=n("p"),k=s("Example:"),f=i(),v(h.$$.fragment)},l(l){p=a(l,"P",{});var u=r(p);k=d(u,"Example:"),u.forEach(o),f=c(l),y(h.$$.fragment,l)},m(l,u){_(l,p,u),e(p,k),_(l,f,u),T(h,l,u),b=!0},p:Xe,i(l){b||(w(h.$$.fragment,l),b=!0)},o(l){S(h.$$.fragment,l),b=!1},d(l){l&&o(p),l&&o(f),$(h,l)}}}function Tu(z){let p,k,f,h,b,l,u,F,Ce,he,B,je,W,Oe,Pe,R,Le,Ne,V,Q,Ie,Z,j,L,ue,ae,Be,J,D,be,re,N,ke,se,ze,ee,de,le,Ae,X,Fe,H,De;return{c(){p=n("p"),k=s("TF 2.0 models accepts two formats as inputs:"),f=i(),h=n("ul"),b=n("li"),l=s("having all inputs as keyword arguments (like PyTorch models), or"),u=i(),F=n("li"),Ce=s("having all inputs as a list, tuple or dict in the first positional arguments."),he=i(),B=n("p"),je=s("This second option is useful when using "),W=n("code"),Oe=s("tf.keras.Model.fit"),Pe=s(` method which currently requires having all the
tensors in the first argument of the model call function: `),R=n("code"),Le=s("model(inputs)"),Ne=s("."),V=i(),Q=n("p"),Ie=s(`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),Z=i(),j=n("ul"),L=n("li"),ue=s("a single Tensor with "),ae=n("code"),Be=s("input_ids"),J=s(" only and nothing else: "),D=n("code"),be=s("model(input_ids)"),re=i(),N=n("li"),ke=s(`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),se=n("code"),ze=s("model([input_ids, attention_mask])"),ee=s(" or "),de=n("code"),le=s("model([input_ids, attention_mask, token_type_ids])"),Ae=i(),X=n("li"),Fe=s(`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),H=n("code"),De=s('model({"input_ids": input_ids, "token_type_ids": token_type_ids})')},l(x){p=a(x,"P",{});var q=r(p);k=d(q,"TF 2.0 models accepts two formats as inputs:"),q.forEach(o),f=c(x),h=a(x,"UL",{});var ie=r(h);b=a(ie,"LI",{});var Ze=r(b);l=d(Ze,"having all inputs as keyword arguments (like PyTorch models), or"),Ze.forEach(o),u=c(ie),F=a(ie,"LI",{});var Re=r(F);Ce=d(Re,"having all inputs as a list, tuple or dict in the first positional arguments."),Re.forEach(o),ie.forEach(o),he=c(x),B=a(x,"P",{});var O=r(B);je=d(O,"This second option is useful when using "),W=a(O,"CODE",{});var eo=r(W);Oe=d(eo,"tf.keras.Model.fit"),eo.forEach(o),Pe=d(O,` method which currently requires having all the
tensors in the first argument of the model call function: `),R=a(O,"CODE",{});var ve=r(R);Le=d(ve,"model(inputs)"),ve.forEach(o),Ne=d(O,"."),O.forEach(o),V=c(x),Q=a(x,"P",{});var oo=r(Q);Ie=d(oo,`If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
first positional argument :`),oo.forEach(o),Z=c(x),j=a(x,"UL",{});var Y=r(j);L=a(Y,"LI",{});var G=r(L);ue=d(G,"a single Tensor with "),ae=a(G,"CODE",{});var to=r(ae);Be=d(to,"input_ids"),to.forEach(o),J=d(G," only and nothing else: "),D=a(G,"CODE",{});var no=r(D);be=d(no,"model(input_ids)"),no.forEach(o),G.forEach(o),re=c(Y),N=a(Y,"LI",{});var oe=r(N);ke=d(oe,`a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
`),se=a(oe,"CODE",{});var ce=r(se);ze=d(ce,"model([input_ids, attention_mask])"),ce.forEach(o),ee=d(oe," or "),de=a(oe,"CODE",{});var fe=r(de);le=d(fe,"model([input_ids, attention_mask, token_type_ids])"),fe.forEach(o),oe.forEach(o),Ae=c(Y),X=a(Y,"LI",{});var ye=r(X);Fe=d(ye,`a dictionary with one or several input Tensors associated to the input names given in the docstring:
`),H=a(ye,"CODE",{});var Te=r(H);De=d(Te,'model({"input_ids": input_ids, "token_type_ids": token_type_ids})'),Te.forEach(o),ye.forEach(o),Y.forEach(o)},m(x,q){_(x,p,q),e(p,k),_(x,f,q),_(x,h,q),e(h,b),e(b,l),e(h,u),e(h,F),e(F,Ce),_(x,he,q),_(x,B,q),e(B,je),e(B,W),e(W,Oe),e(B,Pe),e(B,R),e(R,Le),e(B,Ne),_(x,V,q),_(x,Q,q),e(Q,Ie),_(x,Z,q),_(x,j,q),e(j,L),e(L,ue),e(L,ae),e(ae,Be),e(L,J),e(L,D),e(D,be),e(j,re),e(j,N),e(N,ke),e(N,se),e(se,ze),e(N,ee),e(N,de),e(de,le),e(j,Ae),e(j,X),e(X,Fe),e(X,H),e(H,De)},d(x){x&&o(p),x&&o(f),x&&o(h),x&&o(he),x&&o(B),x&&o(V),x&&o(Q),x&&o(Z),x&&o(j)}}}function wu(z){let p,k,f,h,b;return{c(){p=n("p"),k=s("Although the recipe for forward pass needs to be defined within this function, one should call the "),f=n("code"),h=s("Module"),b=s(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){p=a(l,"P",{});var u=r(p);k=d(u,"Although the recipe for forward pass needs to be defined within this function, one should call the "),f=a(u,"CODE",{});var F=r(f);h=d(F,"Module"),F.forEach(o),b=d(u,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),u.forEach(o)},m(l,u){_(l,p,u),e(p,k),e(p,f),e(f,h),e(p,b)},d(l){l&&o(p)}}}function Su(z){let p,k,f,h,b;return h=new Ye({props:{code:`from transformers import BlenderbotSmallTokenizer, FlaxBlenderbotSmallModel

tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")
model = FlaxBlenderbotSmallModel.from_pretrained("facebook/blenderbot_small-90M")

inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BlenderbotSmallTokenizer, FlaxBlenderbotSmallModel

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BlenderbotSmallTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/blenderbot_small-90M&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FlaxBlenderbotSmallModel.from_pretrained(<span class="hljs-string">&quot;facebook/blenderbot_small-90M&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;jax&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`}}),{c(){p=n("p"),k=s("Example:"),f=i(),v(h.$$.fragment)},l(l){p=a(l,"P",{});var u=r(p);k=d(u,"Example:"),u.forEach(o),f=c(l),y(h.$$.fragment,l)},m(l,u){_(l,p,u),e(p,k),_(l,f,u),T(h,l,u),b=!0},p:Xe,i(l){b||(w(h.$$.fragment,l),b=!0)},o(l){S(h.$$.fragment,l),b=!1},d(l){l&&o(p),l&&o(f),$(h,l)}}}function $u(z){let p,k,f,h,b;return h=new Ye({props:{code:`from transformers import BlenderbotSmallTokenizer, FlaxBlenderbotSmallForConditionalGeneration

model = FlaxBlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M")
tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")

text = "My friends are cool but they eat too many carbs."
inputs = tokenizer(text, max_length=1024, return_tensors="np")
encoder_outputs = model.encode(**inputs)`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BlenderbotSmallTokenizer, FlaxBlenderbotSmallForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>model = FlaxBlenderbotSmallForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;facebook/blenderbot_small-90M&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BlenderbotSmallTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/blenderbot_small-90M&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>text = <span class="hljs-string">&quot;My friends are cool but they eat too many carbs.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(text, max_length=<span class="hljs-number">1024</span>, return_tensors=<span class="hljs-string">&quot;np&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>encoder_outputs = model.encode(**inputs)`}}),{c(){p=n("p"),k=s("Example:"),f=i(),v(h.$$.fragment)},l(l){p=a(l,"P",{});var u=r(p);k=d(u,"Example:"),u.forEach(o),f=c(l),y(h.$$.fragment,l)},m(l,u){_(l,p,u),e(p,k),_(l,f,u),T(h,l,u),b=!0},p:Xe,i(l){b||(w(h.$$.fragment,l),b=!0)},o(l){S(h.$$.fragment,l),b=!1},d(l){l&&o(p),l&&o(f),$(h,l)}}}function xu(z){let p,k,f,h,b;return h=new Ye({props:{code:`import jax.numpy as jnp
from transformers import BlenderbotSmallTokenizer, FlaxBlenderbotSmallForConditionalGeneration

model = FlaxBlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M")
tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")

text = "My friends are cool but they eat too many carbs."
inputs = tokenizer(text, max_length=1024, return_tensors="np")
encoder_outputs = model.encode(**inputs)

decoder_start_token_id = model.config.decoder_start_token_id
decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

outputs = model.decode(decoder_input_ids, encoder_outputs)
last_decoder_hidden_states = outputs.last_hidden_state`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> jax.numpy <span class="hljs-keyword">as</span> jnp
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BlenderbotSmallTokenizer, FlaxBlenderbotSmallForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>model = FlaxBlenderbotSmallForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;facebook/blenderbot_small-90M&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BlenderbotSmallTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/blenderbot_small-90M&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>text = <span class="hljs-string">&quot;My friends are cool but they eat too many carbs.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(text, max_length=<span class="hljs-number">1024</span>, return_tensors=<span class="hljs-string">&quot;np&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>encoder_outputs = model.encode(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>decoder_start_token_id = model.config.decoder_start_token_id
<span class="hljs-meta">&gt;&gt;&gt; </span>decoder_input_ids = jnp.ones((inputs.input_ids.shape[<span class="hljs-number">0</span>], <span class="hljs-number">1</span>), dtype=<span class="hljs-string">&quot;i4&quot;</span>) * decoder_start_token_id

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model.decode(decoder_input_ids, encoder_outputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>last_decoder_hidden_states = outputs.last_hidden_state`}}),{c(){p=n("p"),k=s("Example:"),f=i(),v(h.$$.fragment)},l(l){p=a(l,"P",{});var u=r(p);k=d(u,"Example:"),u.forEach(o),f=c(l),y(h.$$.fragment,l)},m(l,u){_(l,p,u),e(p,k),_(l,f,u),T(h,l,u),b=!0},p:Xe,i(l){b||(w(h.$$.fragment,l),b=!0)},o(l){S(h.$$.fragment,l),b=!1},d(l){l&&o(p),l&&o(f),$(h,l)}}}function Bu(z){let p,k,f,h,b;return{c(){p=n("p"),k=s("Although the recipe for forward pass needs to be defined within this function, one should call the "),f=n("code"),h=s("Module"),b=s(`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`)},l(l){p=a(l,"P",{});var u=r(p);k=d(u,"Although the recipe for forward pass needs to be defined within this function, one should call the "),f=a(u,"CODE",{});var F=r(f);h=d(F,"Module"),F.forEach(o),b=d(u,`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`),u.forEach(o)},m(l,u){_(l,p,u),e(p,k),e(p,f),e(f,h),e(p,b)},d(l){l&&o(p)}}}function zu(z){let p,k,f,h,b;return h=new Ye({props:{code:`from transformers import BlenderbotSmallTokenizer, FlaxBlenderbotSmallForConditionalGeneration

model = FlaxBlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M")
tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")

text = "My friends are cool but they eat too many carbs."
inputs = tokenizer(text, max_length=1024, return_tensors="np")
encoder_outputs = model.encode(**inputs)`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BlenderbotSmallTokenizer, FlaxBlenderbotSmallForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>model = FlaxBlenderbotSmallForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;facebook/blenderbot_small-90M&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BlenderbotSmallTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/blenderbot_small-90M&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>text = <span class="hljs-string">&quot;My friends are cool but they eat too many carbs.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(text, max_length=<span class="hljs-number">1024</span>, return_tensors=<span class="hljs-string">&quot;np&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>encoder_outputs = model.encode(**inputs)`}}),{c(){p=n("p"),k=s("Example:"),f=i(),v(h.$$.fragment)},l(l){p=a(l,"P",{});var u=r(p);k=d(u,"Example:"),u.forEach(o),f=c(l),y(h.$$.fragment,l)},m(l,u){_(l,p,u),e(p,k),_(l,f,u),T(h,l,u),b=!0},p:Xe,i(l){b||(w(h.$$.fragment,l),b=!0)},o(l){S(h.$$.fragment,l),b=!1},d(l){l&&o(p),l&&o(f),$(h,l)}}}function Fu(z){let p,k,f,h,b;return h=new Ye({props:{code:`import jax.numpy as jnp
from transformers import BlenderbotSmallTokenizer, FlaxBlenderbotSmallForConditionalGeneration

model = FlaxBlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M")
tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")

text = "My friends are cool but they eat too many carbs."
inputs = tokenizer(text, max_length=1024, return_tensors="np")
encoder_outputs = model.encode(**inputs)

decoder_start_token_id = model.config.decoder_start_token_id
decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

outputs = model.decode(decoder_input_ids, encoder_outputs)
logits = outputs.logits`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> jax.numpy <span class="hljs-keyword">as</span> jnp
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BlenderbotSmallTokenizer, FlaxBlenderbotSmallForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>model = FlaxBlenderbotSmallForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;facebook/blenderbot_small-90M&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BlenderbotSmallTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/blenderbot_small-90M&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>text = <span class="hljs-string">&quot;My friends are cool but they eat too many carbs.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(text, max_length=<span class="hljs-number">1024</span>, return_tensors=<span class="hljs-string">&quot;np&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>encoder_outputs = model.encode(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>decoder_start_token_id = model.config.decoder_start_token_id
<span class="hljs-meta">&gt;&gt;&gt; </span>decoder_input_ids = jnp.ones((inputs.input_ids.shape[<span class="hljs-number">0</span>], <span class="hljs-number">1</span>), dtype=<span class="hljs-string">&quot;i4&quot;</span>) * decoder_start_token_id

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model.decode(decoder_input_ids, encoder_outputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`}}),{c(){p=n("p"),k=s("Example:"),f=i(),v(h.$$.fragment)},l(l){p=a(l,"P",{});var u=r(p);k=d(u,"Example:"),u.forEach(o),f=c(l),y(h.$$.fragment,l)},m(l,u){_(l,p,u),e(p,k),_(l,f,u),T(h,l,u),b=!0},p:Xe,i(l){b||(w(h.$$.fragment,l),b=!0)},o(l){S(h.$$.fragment,l),b=!1},d(l){l&&o(p),l&&o(f),$(h,l)}}}function qu(z){let p,k,f,h,b,l,u,F,Ce,he,B,je,W,Oe,Pe,R,Le,Ne,V,Q,Ie,Z,j,L,ue,ae,Be,J,D,be,re,N,ke,se,ze,ee,de,le,Ae,X,Fe,H,De,x,q,ie,Ze,Re,O,eo,ve,oo,Y,G,to,no,oe,ce,fe,ye,Te,md,sa,hd,cs,we,ht,ud,ao,fd,En,_d,gd,ut,bd,kd,vd,ro,yd,Mn,Td,wd,Cn,Sd,$d,xd,wo,ps,so,So,da,ft,Bd,la,zd,ms,U,_t,Fd,ia,qd,Ed,gt,Md,jn,Cd,jd,Od,He,bt,Pd,ca,Ld,Nd,pa,Id,Ad,$o,kt,Dd,lo,Gd,ma,Ud,Wd,ha,Rd,Hd,Kd,Ke,vt,Vd,On,Qd,Pn,Jd,Xd,ua,Yd,Zd,Ln,yt,hs,io,xo,fa,Tt,el,_a,ol,us,Ge,wt,tl,St,nl,ga,al,rl,sl,Bo,$t,dl,ba,ll,fs,co,zo,ka,xt,il,va,cl,_s,Se,Bt,pl,zt,ml,Nn,hl,ul,fl,Ft,_l,qt,gl,bl,kl,qe,Et,vl,po,yl,In,Tl,wl,ya,Sl,$l,xl,Fo,Bl,qo,gs,mo,Eo,Ta,Mt,zl,wa,Fl,bs,$e,Ct,ql,jt,El,An,Ml,Cl,jl,Ot,Ol,Pt,Pl,Ll,Nl,Ee,Lt,Il,ho,Al,Dn,Dl,Gl,Sa,Ul,Wl,Rl,Mo,Hl,Co,ks,uo,jo,$a,Nt,Kl,xa,Vl,vs,fo,It,Ql,Oo,At,Jl,Po,ys,_o,Lo,Ba,Dt,Xl,za,Yl,Ts,pe,Gt,Zl,Ut,ei,Gn,oi,ti,ni,Wt,ai,Rt,ri,si,di,No,li,Me,Ht,ii,go,ci,Un,pi,mi,Fa,hi,ui,fi,Io,_i,Ao,ws,bo,Do,qa,Kt,gi,Ea,bi,Ss,me,Vt,ki,Qt,vi,Wn,yi,Ti,wi,Jt,Si,Xt,$i,xi,Bi,Go,zi,P,Yt,Fi,ko,qi,Rn,Ei,Mi,Ma,Ci,ji,Oi,Uo,Pi,Ca,Li,Ni,ja,Oa,Pa,La,Ii,Ai,Na,Ia,Aa,Da,Di,Gi,Ga,Ua,Wa,Ra,Ui,Wi,Ha,Ka,Va,Zt,Ri,Qa,Hi,Ki,Vi,Ja,Xa,Ya,Za,Qi,$s,vo,Wo,er,en,Ji,or,Xi,xs,I,on,Yi,tn,Zi,Hn,ec,oc,tc,nn,nc,an,ac,rc,sc,tr,dc,lc,Ue,nr,rn,ic,cc,ar,sn,pc,mc,rr,dn,hc,uc,sr,ln,fc,_c,Ro,cn,gc,Ho,bc,Ko,pn,kc,Vo,vc,Qo,mn,yc,Jo,Bs,yo,Xo,dr,hn,Tc,lr,wc,zs,A,un,Sc,fn,$c,Kn,xc,Bc,zc,_n,Fc,gn,qc,Ec,Mc,ir,Cc,jc,We,cr,bn,Oc,Pc,pr,kn,Lc,Nc,mr,vn,Ic,Ac,hr,yn,Dc,Gc,E,Tn,Uc,To,Wc,ur,Rc,Hc,fr,Kc,Vc,Qc,Yo,Jc,_r,Xc,Yc,gr,br,kr,vr,Zc,ep,yr,Tr,wr,Sr,op,tp,$r,xr,Br,zr,np,ap,Fr,qr,wn,Zo,et,Er,Sn,rp,Mr,sp,dp,Cr,lp,ip,jr,cp,pp,Or,Pr,Lr,Nr,mp,hp,Ir,Ar,Dr,Gr,up,fp,Ur,Wr,Rr,Hr,_p,gp,Kr,Vr,Qr,Jr,bp,kp,ot,$n,vp,tt,yp,nt,xn,Tp,at,Fs;return l=new xe({}),re=new xe({}),Te=new xe({}),ht=new C({props:{name:"class transformers.BlenderbotSmallConfig",anchor:"transformers.BlenderbotSmallConfig",parameters:[{name:"vocab_size",val:" = 50265"},{name:"max_position_embeddings",val:" = 512"},{name:"encoder_layers",val:" = 8"},{name:"encoder_ffn_dim",val:" = 2048"},{name:"encoder_attention_heads",val:" = 16"},{name:"decoder_layers",val:" = 8"},{name:"decoder_ffn_dim",val:" = 2048"},{name:"decoder_attention_heads",val:" = 16"},{name:"encoder_layerdrop",val:" = 0.0"},{name:"decoder_layerdrop",val:" = 0.0"},{name:"use_cache",val:" = True"},{name:"is_encoder_decoder",val:" = True"},{name:"activation_function",val:" = 'gelu'"},{name:"d_model",val:" = 512"},{name:"dropout",val:" = 0.1"},{name:"attention_dropout",val:" = 0.0"},{name:"activation_dropout",val:" = 0.0"},{name:"init_std",val:" = 0.02"},{name:"decoder_start_token_id",val:" = 1"},{name:"classifier_dropout",val:" = 0.0"},{name:"scale_embedding",val:" = False"},{name:"pad_token_id",val:" = 0"},{name:"bos_token_id",val:" = 1"},{name:"eos_token_id",val:" = 2"},{name:"forced_eos_token_id",val:" = 2"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BlenderbotSmallConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 50265) &#x2014;
Vocabulary size of the BlenderbotSmall model. Defines the number of different tokens that can be
represented by the <code>inputs_ids</code> passed when calling <a href="/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallModel">BlenderbotSmallModel</a> or <a href="/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.TFBlenderbotSmallModel">TFBlenderbotSmallModel</a>.`,name:"vocab_size"},{anchor:"transformers.BlenderbotSmallConfig.d_model",description:`<strong>d_model</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
Dimensionality of the layers and the pooler layer.`,name:"d_model"},{anchor:"transformers.BlenderbotSmallConfig.encoder_layers",description:`<strong>encoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
Number of encoder layers.`,name:"encoder_layers"},{anchor:"transformers.BlenderbotSmallConfig.decoder_layers",description:`<strong>decoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
Number of decoder layers.`,name:"decoder_layers"},{anchor:"transformers.BlenderbotSmallConfig.encoder_attention_heads",description:`<strong>encoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"encoder_attention_heads"},{anchor:"transformers.BlenderbotSmallConfig.decoder_attention_heads",description:`<strong>decoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"decoder_attention_heads"},{anchor:"transformers.BlenderbotSmallConfig.decoder_ffn_dim",description:`<strong>decoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"decoder_ffn_dim"},{anchor:"transformers.BlenderbotSmallConfig.encoder_ffn_dim",description:`<strong>encoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"encoder_ffn_dim"},{anchor:"transformers.BlenderbotSmallConfig.activation_function",description:`<strong>activation_function</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"activation_function"},{anchor:"transformers.BlenderbotSmallConfig.dropout",description:`<strong>dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"dropout"},{anchor:"transformers.BlenderbotSmallConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.BlenderbotSmallConfig.activation_dropout",description:`<strong>activation_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for activations inside the fully connected layer.`,name:"activation_dropout"},{anchor:"transformers.BlenderbotSmallConfig.classifier_dropout",description:`<strong>classifier_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for classifier.`,name:"classifier_dropout"},{anchor:"transformers.BlenderbotSmallConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.BlenderbotSmallConfig.init_std",description:`<strong>init_std</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
encoder_layerdrop &#x2014; (<code>float</code>, <em>optional</em>, defaults to 0.0):
The LayerDrop probability for the encoder. See the [LayerDrop paper](see <a href="https://arxiv.org/abs/1909.11556" rel="nofollow">https://arxiv.org/abs/1909.11556</a>)
for more details.
decoder_layerdrop &#x2014; (<code>float</code>, <em>optional</em>, defaults to 0.0):
The LayerDrop probability for the decoder. See the [LayerDrop paper](see <a href="https://arxiv.org/abs/1909.11556" rel="nofollow">https://arxiv.org/abs/1909.11556</a>)
for more details.`,name:"init_std"},{anchor:"transformers.BlenderbotSmallConfig.scale_embedding",description:`<strong>scale_embedding</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Scale embeddings by diving by sqrt(d_model).`,name:"scale_embedding"},{anchor:"transformers.BlenderbotSmallConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models)`,name:"use_cache"},{anchor:"transformers.BlenderbotSmallConfig.forced_eos_token_id",description:`<strong>forced_eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The id of the token to force as the last generated token when <code>max_length</code> is reached. Usually set to
<code>eos_token_id</code>.`,name:"forced_eos_token_id"}],source:"https://github.com/huggingface/transformers/blob/vr_18694/src/transformers/models/blenderbot_small/configuration_blenderbot_small.py#L36"}}),wo=new Je({props:{anchor:"transformers.BlenderbotSmallConfig.example",$$slots:{default:[hu]},$$scope:{ctx:z}}}),ft=new xe({}),_t=new C({props:{name:"class transformers.BlenderbotSmallTokenizer",anchor:"transformers.BlenderbotSmallTokenizer",parameters:[{name:"vocab_file",val:""},{name:"merges_file",val:""},{name:"bos_token",val:" = '__start__'"},{name:"eos_token",val:" = '__end__'"},{name:"unk_token",val:" = '__unk__'"},{name:"pad_token",val:" = '__null__'"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BlenderbotSmallTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
File containing the vocabulary.`,name:"vocab_file"},{anchor:"transformers.BlenderbotSmallTokenizer.merges_file",description:`<strong>merges_file</strong> (<code>str</code>) &#x2014;
Path to the merges file.`,name:"merges_file"},{anchor:"transformers.BlenderbotSmallTokenizer.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;__start__&quot;</code>) &#x2014;
The beginning of sentence token.`,name:"bos_token"},{anchor:"transformers.BlenderbotSmallTokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;__end__&quot;</code>) &#x2014;
The end of sentence token.`,name:"eos_token"},{anchor:"transformers.BlenderbotSmallTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;__unk__&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.BlenderbotSmallTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;__pad__&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.
**kwargs &#x2014;
Additional keyword arguments passed along to <a href="/docs/transformers/pr_18694/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a>`,name:"pad_token"}],source:"https://github.com/huggingface/transformers/blob/vr_18694/src/transformers/models/blenderbot_small/tokenization_blenderbot_small.py#L69"}}),bt=new C({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.BlenderbotSmallTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": typing.List[int]"},{name:"token_ids_1",val:": typing.Optional[typing.List[int]] = None"}],parametersDescription:[{anchor:"transformers.BlenderbotSmallTokenizer.build_inputs_with_special_tokens.token_ids_0",description:"<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014; The first tokenized sequence.",name:"token_ids_0"},{anchor:"transformers.BlenderbotSmallTokenizer.build_inputs_with_special_tokens.token_ids_1",description:"<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014; The second tokenized sequence.",name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/vr_18694/src/transformers/tokenization_utils_base.py#L2982",returnDescription:`
<p>The model input with special tokens.</p>
`,returnType:`
<p><code>List[int]</code></p>
`}}),kt=new C({props:{name:"get_special_tokens_mask",anchor:"transformers.BlenderbotSmallTokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": typing.List"},{name:"token_ids_1",val:": typing.Optional[typing.List] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.BlenderbotSmallTokenizer.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of ids of the first sequence.`,name:"token_ids_0"},{anchor:"transformers.BlenderbotSmallTokenizer.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
List of ids of the second sequence.`,name:"token_ids_1"},{anchor:"transformers.BlenderbotSmallTokenizer.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/vr_18694/src/transformers/tokenization_utils.py#L845",returnDescription:`
<p>1 for a special token, 0 for a sequence token.</p>
`,returnType:`
<p>A list of integers in the range [0, 1]</p>
`}}),vt=new C({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.BlenderbotSmallTokenizer.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": typing.List[int]"},{name:"token_ids_1",val:": typing.Optional[typing.List[int]] = None"}],parametersDescription:[{anchor:"transformers.BlenderbotSmallTokenizer.create_token_type_ids_from_sequences.token_ids_0",description:"<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014; The first tokenized sequence.",name:"token_ids_0"},{anchor:"transformers.BlenderbotSmallTokenizer.create_token_type_ids_from_sequences.token_ids_1",description:"<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014; The second tokenized sequence.",name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/vr_18694/src/transformers/tokenization_utils_base.py#L2962",returnDescription:`
<p>The token type ids.</p>
`,returnType:`
<p><code>List[int]</code></p>
`}}),yt=new C({props:{name:"save_vocabulary",anchor:"transformers.BlenderbotSmallTokenizer.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/vr_18694/src/transformers/models/blenderbot_small/tokenization_blenderbot_small.py#L211"}}),Tt=new xe({}),wt=new C({props:{name:"class transformers.BlenderbotSmallTokenizerFast",anchor:"transformers.BlenderbotSmallTokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"merges_file",val:" = None"},{name:"unk_token",val:" = '<|endoftext|>'"},{name:"bos_token",val:" = '<|endoftext|>'"},{name:"eos_token",val:" = '<|endoftext|>'"},{name:"add_prefix_space",val:" = False"},{name:"trim_offsets",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BlenderbotSmallTokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"}],source:"https://github.com/huggingface/transformers/blob/vr_18694/src/transformers/models/blenderbot_small/tokenization_blenderbot_small_fast.py#L52"}}),$t=new C({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.BlenderbotSmallTokenizerFast.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": typing.List[int]"},{name:"token_ids_1",val:": typing.Optional[typing.List[int]] = None"}],parametersDescription:[{anchor:"transformers.BlenderbotSmallTokenizerFast.create_token_type_ids_from_sequences.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.BlenderbotSmallTokenizerFast.create_token_type_ids_from_sequences.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/vr_18694/src/transformers/models/blenderbot_small/tokenization_blenderbot_small_fast.py#L98",returnDescription:`
<p>List of zeros.</p>
`,returnType:`
<p><code>List[int]</code></p>
`}}),xt=new xe({}),Bt=new C({props:{name:"class transformers.BlenderbotSmallModel",anchor:"transformers.BlenderbotSmallModel",parameters:[{name:"config",val:": BlenderbotSmallConfig"}],parametersDescription:[{anchor:"transformers.BlenderbotSmallModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallConfig">BlenderbotSmallConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/pr_18694/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18694/src/transformers/models/blenderbot_small/modeling_blenderbot_small.py#L1083"}}),Et=new C({props:{name:"forward",anchor:"transformers.BlenderbotSmallModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Union[typing.Tuple, transformers.modeling_outputs.BaseModelOutput, NoneType] = None"},{name:"past_key_values",val:": typing.Optional[typing.List[torch.FloatTensor]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.BlenderbotSmallModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
it.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallTokenizer">BlenderbotSmallTokenizer</a>. See <a href="/docs/transformers/pr_18694/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18694/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BlenderbotSmallModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BlenderbotSmallModel.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallTokenizer">BlenderbotSmallTokenizer</a>. See <a href="/docs/transformers/pr_18694/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18694/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>BlenderbotSmall uses the <code>bos_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_ids</code> have to be input (see
<code>past_key_values</code>).`,name:"decoder_input_ids"},{anchor:"transformers.BlenderbotSmallModel.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.BlenderbotSmallModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(encoder_layers, encoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BlenderbotSmallModel.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.BlenderbotSmallModel.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.BlenderbotSmallModel.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple(tuple(torch.FloatTensor)</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.BlenderbotSmallModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple(tuple(torch.FloatTensor))</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) &#x2014;
Tuple of <code>tuple(torch.FloatTensor)</code> of length <code>config.n_layers</code>, with each tuple having 2 tensors of shape
<code>(batch_size, num_heads, sequence_length, embed_size_per_head)</code>) and 2 additional tensors of shape
<code>(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)</code>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
<p>If <code>past_key_values</code> are used, the user can optionally input only the last <code>decoder_input_ids</code> (those that
don&#x2019;t have their past key value states given to this model) of shape <code>(batch_size, 1)</code> instead of all
<code>decoder_input_ids</code> of shape <code>(batch_size, sequence_length)</code>. inputs_embeds (<code>torch.FloatTensor</code> of shape
<code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>): Optionally, instead of passing <code>input_ids</code> you
can choose to directly pass an embedded representation. This is useful if you want more control over how to
convert <code>input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.`,name:"past_key_values"},{anchor:"transformers.BlenderbotSmallModel.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.BlenderbotSmallModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.BlenderbotSmallModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BlenderbotSmallModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BlenderbotSmallModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18694/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18694/src/transformers/models/blenderbot_small/modeling_blenderbot_small.py#L1110",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18694/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput"
>transformers.modeling_outputs.Seq2SeqModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallConfig"
>BlenderbotSmallConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) \u2014 Sequence of hidden-states at the output of the last layer of the decoder of the model.</p>
<p>If <code>past_key_values</code> is used only the last hidden-state of the sequences of shape <code>(batch_size, 1, hidden_size)</code> is output.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>tuple(tuple(torch.FloatTensor))</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) \u2014 Tuple of <code>tuple(torch.FloatTensor)</code> of length <code>config.n_layers</code>, with each tuple having 2 tensors of shape
<code>(batch_size, num_heads, sequence_length, embed_size_per_head)</code>) and 2 additional tensors of shape
<code>(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)</code>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>decoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>decoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder\u2019s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
<li>
<p><strong>encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) \u2014 Sequence of hidden-states at the output of the last layer of the encoder of the model.</p>
</li>
<li>
<p><strong>encoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>encoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18694/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput"
>transformers.modeling_outputs.Seq2SeqModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Fo=new qn({props:{$$slots:{default:[uu]},$$scope:{ctx:z}}}),qo=new Je({props:{anchor:"transformers.BlenderbotSmallModel.forward.example",$$slots:{default:[fu]},$$scope:{ctx:z}}}),Mt=new xe({}),Ct=new C({props:{name:"class transformers.BlenderbotSmallForConditionalGeneration",anchor:"transformers.BlenderbotSmallForConditionalGeneration",parameters:[{name:"config",val:": BlenderbotSmallConfig"}],parametersDescription:[{anchor:"transformers.BlenderbotSmallForConditionalGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallConfig">BlenderbotSmallConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/pr_18694/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18694/src/transformers/models/blenderbot_small/modeling_blenderbot_small.py#L1209"}}),Lt=new C({props:{name:"forward",anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Union[typing.Tuple, transformers.modeling_outputs.BaseModelOutput, NoneType] = None"},{name:"past_key_values",val:": typing.Optional[typing.List[torch.FloatTensor]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
it.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallTokenizer">BlenderbotSmallTokenizer</a>. See <a href="/docs/transformers/pr_18694/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18694/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallTokenizer">BlenderbotSmallTokenizer</a>. See <a href="/docs/transformers/pr_18694/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18694/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>BlenderbotSmall uses the <code>bos_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_ids</code> have to be input (see
<code>past_key_values</code>).`,name:"decoder_input_ids"},{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(encoder_layers, encoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple(tuple(torch.FloatTensor)</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple(tuple(torch.FloatTensor))</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) &#x2014;
Tuple of <code>tuple(torch.FloatTensor)</code> of length <code>config.n_layers</code>, with each tuple having 2 tensors of shape
<code>(batch_size, num_heads, sequence_length, embed_size_per_head)</code>) and 2 additional tensors of shape
<code>(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)</code>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
<p>If <code>past_key_values</code> are used, the user can optionally input only the last <code>decoder_input_ids</code> (those that
don&#x2019;t have their past key value states given to this model) of shape <code>(batch_size, 1)</code> instead of all
<code>decoder_input_ids</code> of shape <code>(batch_size, sequence_length)</code>. inputs_embeds (<code>torch.FloatTensor</code> of shape
<code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>): Optionally, instead of passing <code>input_ids</code> you
can choose to directly pass an embedded representation. This is useful if you want more control over how to
convert <code>input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.`,name:"past_key_values"},{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18694/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18694/src/transformers/models/blenderbot_small/modeling_blenderbot_small.py#L1253",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18694/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput"
>transformers.modeling_outputs.Seq2SeqLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallConfig"
>BlenderbotSmallConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) \u2014 Language modeling loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) \u2014 Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>tuple(tuple(torch.FloatTensor))</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) \u2014 Tuple of <code>tuple(torch.FloatTensor)</code> of length <code>config.n_layers</code>, with each tuple having 2 tensors of shape
<code>(batch_size, num_heads, sequence_length, embed_size_per_head)</code>) and 2 additional tensors of shape
<code>(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)</code>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>decoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>decoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder\u2019s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
<li>
<p><strong>encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) \u2014 Sequence of hidden-states at the output of the last layer of the encoder of the model.</p>
</li>
<li>
<p><strong>encoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>encoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18694/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput"
>transformers.modeling_outputs.Seq2SeqLMOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Mo=new qn({props:{$$slots:{default:[_u]},$$scope:{ctx:z}}}),Co=new Je({props:{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.example",$$slots:{default:[gu]},$$scope:{ctx:z}}}),Nt=new xe({}),It=new C({props:{name:"class transformers.BlenderbotSmallForCausalLM",anchor:"transformers.BlenderbotSmallForCausalLM",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/vr_18694/src/transformers/models/blenderbot_small/modeling_blenderbot_small.py#L1389"}}),At=new C({props:{name:"forward",anchor:"transformers.BlenderbotSmallForCausalLM.forward",parameters:[{name:"input_ids",val:": LongTensor = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[typing.List[torch.FloatTensor]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.BlenderbotSmallForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
provide it.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallTokenizer">BlenderbotSmallTokenizer</a>. See <a href="/docs/transformers/pr_18694/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18694/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BlenderbotSmallForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BlenderbotSmallForCausalLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong>  (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.BlenderbotSmallForCausalLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used
in the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:`,name:"encoder_attention_mask"},{anchor:"transformers.BlenderbotSmallForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BlenderbotSmallForCausalLM.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.BlenderbotSmallForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple(tuple(torch.FloatTensor))</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) &#x2014;
Tuple of <code>tuple(torch.FloatTensor)</code> of length <code>config.n_layers</code>, with each tuple having 2 tensors of
shape <code>(batch_size, num_heads, sequence_length, embed_size_per_head)</code>) and 2 additional tensors of
shape <code>(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)</code>. The two additional
tensors are only required when the model is used as a decoder in a Sequence to Sequence model.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
cross-attention blocks) that can be used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
<p>If <code>past_key_values</code> are used, the user can optionally input only the last <code>decoder_input_ids</code> (those
that don&#x2019;t have their past key value states given to this model) of shape <code>(batch_size, 1)</code> instead of
all <code>decoder_input_ids</code> of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.BlenderbotSmallForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.BlenderbotSmallForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding
(see <code>past_key_values</code>).</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"use_cache"},{anchor:"transformers.BlenderbotSmallForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under
returned tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BlenderbotSmallForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors
for more detail.`,name:"output_hidden_states"},{anchor:"transformers.BlenderbotSmallForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18694/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18694/src/transformers/models/blenderbot_small/modeling_blenderbot_small.py#L1420",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18694/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallConfig"
>BlenderbotSmallConfig</a>) and inputs.</p>
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
  href="/docs/transformers/pr_18694/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Po=new Je({props:{anchor:"transformers.BlenderbotSmallForCausalLM.forward.example",$$slots:{default:[bu]},$$scope:{ctx:z}}}),Dt=new xe({}),Gt=new C({props:{name:"class transformers.TFBlenderbotSmallModel",anchor:"transformers.TFBlenderbotSmallModel",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFBlenderbotSmallModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallConfig">BlenderbotSmallConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18694/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18694/src/transformers/models/blenderbot_small/modeling_tf_blenderbot_small.py#L1144"}}),No=new qn({props:{$$slots:{default:[ku]},$$scope:{ctx:z}}}),Ht=new C({props:{name:"call",anchor:"transformers.TFBlenderbotSmallModel.call",parameters:[{name:"input_ids",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"decoder_position_ids",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"head_mask",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"encoder_outputs",val:": typing.Union[typing.Tuple, transformers.modeling_tf_outputs.TFBaseModelOutput, NoneType] = None"},{name:"past_key_values",val:": typing.Optional[typing.List[tensorflow.python.framework.ops.Tensor]] = None"},{name:"inputs_embeds",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"training",val:": typing.Optional[bool] = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFBlenderbotSmallModel.call.input_ids",description:`<strong>input_ids</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallTokenizer">BlenderbotSmallTokenizer</a>. See <a href="/docs/transformers/pr_18694/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18694/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.TFBlenderbotSmallModel.call.attention_mask",description:`<strong>attention_mask</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.TFBlenderbotSmallModel.call.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallTokenizer">BlenderbotSmallTokenizer</a>. See <a href="/docs/transformers/pr_18694/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18694/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>BlenderbotSmall uses the <code>bos_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_ids</code> have to be input (see
<code>past_key_values</code>).`,name:"decoder_input_ids"},{anchor:"transformers.TFBlenderbotSmallModel.call.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
will be made by default and ignore pad tokens. It is not recommended to set this for most use cases.`,name:"decoder_attention_mask"},{anchor:"transformers.TFBlenderbotSmallModel.call.decoder_position_ids",description:`<strong>decoder_position_ids</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the
range <code>[0, config.max_position_embeddings - 1]</code>.`,name:"decoder_position_ids"},{anchor:"transformers.TFBlenderbotSmallModel.call.head_mask",description:`<strong>head_mask</strong> (<code>tf.Tensor</code> of shape <code>(encoder_layers, encoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.TFBlenderbotSmallModel.call.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>tf.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.TFBlenderbotSmallModel.call.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>tf.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.TFBlenderbotSmallModel.call.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tf.FloatTensor</code>, <em>optional</em>) &#x2014;
hidden states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
of shape <code>(batch_size, sequence_length, hidden_size)</code> is a sequence of`,name:"encoder_outputs"},{anchor:"transformers.TFBlenderbotSmallModel.call.past_key_values",description:`<strong>past_key_values</strong> (<code>Tuple[Tuple[tf.Tensor]]</code> of length <code>config.n_layers</code>) &#x2014;
contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
If <code>past_key_values</code> are used, the user can optionally input only the last <code>decoder_input_ids</code> (those that
don&#x2019;t have their past key value states given to this model) of shape <code>(batch_size, 1)</code> instead of all
<code>decoder_input_ids</code> of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.TFBlenderbotSmallModel.call.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>). Set to <code>False</code> during training, <code>True</code> during generation`,name:"use_cache"},{anchor:"transformers.TFBlenderbotSmallModel.call.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
config will be used instead.`,name:"output_attentions"},{anchor:"transformers.TFBlenderbotSmallModel.call.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
used instead.`,name:"output_hidden_states"},{anchor:"transformers.TFBlenderbotSmallModel.call.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18694/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple. This argument can be used in
eager mode, in graph mode the value will always be set to True.`,name:"return_dict"},{anchor:"transformers.TFBlenderbotSmallModel.call.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"}],source:"https://github.com/huggingface/transformers/blob/vr_18694/src/transformers/models/blenderbot_small/modeling_tf_blenderbot_small.py#L1156",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18694/en/main_classes/output#transformers.modeling_tf_outputs.TFSeq2SeqModelOutput"
>transformers.modeling_tf_outputs.TFSeq2SeqModelOutput</a> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<a
  href="/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallConfig"
>BlenderbotSmallConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) \u2014 Sequence of hidden-states at the output of the last layer of the decoder of the model.</p>
<p>If <code>past_key_values</code> is used only the last hidden-state of the sequences of shape <code>(batch_size, 1, hidden_size)</code> is output.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>List[tf.Tensor]</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) \u2014 List of <code>tf.Tensor</code> of length <code>config.n_layers</code>, with each tensor of shape <code>(2, batch_size, num_heads, sequence_length, embed_size_per_head)</code>).</p>
<p>Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>decoder_hidden_states</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>decoder_attentions</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder\u2019s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
<li>
<p><strong>encoder_last_hidden_state</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) \u2014 Sequence of hidden-states at the output of the last layer of the encoder of the model.</p>
</li>
<li>
<p><strong>encoder_hidden_states</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>encoder_attentions</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18694/en/main_classes/output#transformers.modeling_tf_outputs.TFSeq2SeqModelOutput"
>transformers.modeling_tf_outputs.TFSeq2SeqModelOutput</a> or <code>tuple(tf.Tensor)</code></p>
`}}),Io=new qn({props:{$$slots:{default:[vu]},$$scope:{ctx:z}}}),Ao=new Je({props:{anchor:"transformers.TFBlenderbotSmallModel.call.example",$$slots:{default:[yu]},$$scope:{ctx:z}}}),Kt=new xe({}),Vt=new C({props:{name:"class transformers.TFBlenderbotSmallForConditionalGeneration",anchor:"transformers.TFBlenderbotSmallForConditionalGeneration",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TFBlenderbotSmallForConditionalGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallConfig">BlenderbotSmallConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18694/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/vr_18694/src/transformers/models/blenderbot_small/modeling_tf_blenderbot_small.py#L1233"}}),Go=new qn({props:{$$slots:{default:[Tu]},$$scope:{ctx:z}}}),Yt=new C({props:{name:"call",anchor:"transformers.TFBlenderbotSmallForConditionalGeneration.call",parameters:[{name:"input_ids",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"decoder_position_ids",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"head_mask",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[transformers.modeling_tf_outputs.TFBaseModelOutput] = None"},{name:"past_key_values",val:": typing.Optional[typing.List[tensorflow.python.framework.ops.Tensor]] = None"},{name:"inputs_embeds",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"labels",val:": typing.Optional[tensorflow.python.framework.ops.Tensor] = None"},{name:"training",val:": typing.Optional[bool] = False"}],parametersDescription:[{anchor:"transformers.TFBlenderbotSmallForConditionalGeneration.call.input_ids",description:`<strong>input_ids</strong> (<code>tf.Tensor</code> of shape <code>({0})</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallTokenizer">BlenderbotSmallTokenizer</a>. See <a href="/docs/transformers/pr_18694/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18694/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.TFBlenderbotSmallForConditionalGeneration.call.attention_mask",description:`<strong>attention_mask</strong> (<code>tf.Tensor</code> of shape <code>({0})</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.TFBlenderbotSmallForConditionalGeneration.call.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallTokenizer">BlenderbotSmallTokenizer</a>. See <a href="/docs/transformers/pr_18694/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18694/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>BlenderbotSmall uses the <code>bos_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_ids</code> have to be input (see
<code>past_key_values</code>).`,name:"decoder_input_ids"},{anchor:"transformers.TFBlenderbotSmallForConditionalGeneration.call.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
will be made by default and ignore pad tokens. It is not recommended to set this for most use cases.`,name:"decoder_attention_mask"},{anchor:"transformers.TFBlenderbotSmallForConditionalGeneration.call.decoder_position_ids",description:`<strong>decoder_position_ids</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the
range <code>[0, config.max_position_embeddings - 1]</code>.`,name:"decoder_position_ids"},{anchor:"transformers.TFBlenderbotSmallForConditionalGeneration.call.head_mask",description:`<strong>head_mask</strong> (<code>tf.Tensor</code> of shape <code>(encoder_layers, encoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.TFBlenderbotSmallForConditionalGeneration.call.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>tf.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.TFBlenderbotSmallForConditionalGeneration.call.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>tf.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.TFBlenderbotSmallForConditionalGeneration.call.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tf.FloatTensor</code>, <em>optional</em>) &#x2014;
hidden states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
of shape <code>(batch_size, sequence_length, hidden_size)</code> is a sequence of`,name:"encoder_outputs"},{anchor:"transformers.TFBlenderbotSmallForConditionalGeneration.call.past_key_values",description:`<strong>past_key_values</strong> (<code>Tuple[Tuple[tf.Tensor]]</code> of length <code>config.n_layers</code>) &#x2014;
contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
If <code>past_key_values</code> are used, the user can optionally input only the last <code>decoder_input_ids</code> (those that
don&#x2019;t have their past key value states given to this model) of shape <code>(batch_size, 1)</code> instead of all
<code>decoder_input_ids</code> of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.TFBlenderbotSmallForConditionalGeneration.call.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>). Set to <code>False</code> during training, <code>True</code> during generation`,name:"use_cache"},{anchor:"transformers.TFBlenderbotSmallForConditionalGeneration.call.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
config will be used instead.`,name:"output_attentions"},{anchor:"transformers.TFBlenderbotSmallForConditionalGeneration.call.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
used instead.`,name:"output_hidden_states"},{anchor:"transformers.TFBlenderbotSmallForConditionalGeneration.call.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18694/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple. This argument can be used in
eager mode, in graph mode the value will always be set to True.`,name:"return_dict"},{anchor:"transformers.TFBlenderbotSmallForConditionalGeneration.call.training",description:`<strong>training</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to use the model in training mode (some modules like dropout modules have different
behaviors between training and evaluation).`,name:"training"},{anchor:"transformers.TFBlenderbotSmallForConditionalGeneration.call.labels",description:`<strong>labels</strong> (<code>tf.tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/vr_18694/src/transformers/models/blenderbot_small/modeling_tf_blenderbot_small.py#L1266",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18694/en/main_classes/output#transformers.modeling_tf_outputs.TFSeq2SeqLMOutput"
>transformers.modeling_tf_outputs.TFSeq2SeqLMOutput</a> or a tuple of <code>tf.Tensor</code> (if
<code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various elements depending on the
configuration (<a
  href="/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallConfig"
>BlenderbotSmallConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>tf.Tensor</code> of shape <code>(n,)</code>, <em>optional</em>, where n is the number of non-masked labels, returned when <code>labels</code> is provided) \u2014 Language modeling loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) \u2014 Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>List[tf.Tensor]</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) \u2014 List of <code>tf.Tensor</code> of length <code>config.n_layers</code>, with each tensor of shape <code>(2, batch_size, num_heads, sequence_length, embed_size_per_head)</code>).</p>
<p>Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>decoder_hidden_states</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>decoder_attentions</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder\u2019s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
<li>
<p><strong>encoder_last_hidden_state</strong> (<code>tf.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) \u2014 Sequence of hidden-states at the output of the last layer of the encoder of the model.</p>
</li>
<li>
<p><strong>encoder_hidden_states</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>encoder_attentions</strong> (<code>tuple(tf.Tensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>tf.Tensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18694/en/main_classes/output#transformers.modeling_tf_outputs.TFSeq2SeqLMOutput"
>transformers.modeling_tf_outputs.TFSeq2SeqLMOutput</a> or <code>tuple(tf.Tensor)</code></p>
`}}),Uo=new qn({props:{$$slots:{default:[wu]},$$scope:{ctx:z}}}),en=new xe({}),on=new C({props:{name:"class transformers.FlaxBlenderbotSmallModel",anchor:"transformers.FlaxBlenderbotSmallModel",parameters:[{name:"config",val:": BlenderbotSmallConfig"},{name:"input_shape",val:": typing.Tuple[int] = (1, 1)"},{name:"seed",val:": int = 0"},{name:"dtype",val:": dtype = <class 'jax.numpy.float32'>"},{name:"_do_init",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.FlaxBlenderbotSmallModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallConfig">BlenderbotSmallConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18694/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.FlaxBlenderbotSmallModel.dtype",description:`<strong>dtype</strong> (<code>jax.numpy.dtype</code>, <em>optional</em>, defaults to <code>jax.numpy.float32</code>) &#x2014;
The data type of the computation. Can be one of <code>jax.numpy.float32</code>, <code>jax.numpy.float16</code> (on GPUs) and
<code>jax.numpy.bfloat16</code> (on TPUs).</p>
<p>This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
specified all the computation will be performed with the given <code>dtype</code>.</p>
<p><strong>Note that this only specifies the dtype of the computation and does not influence the dtype of model
parameters.</strong></p>
<p>If you wish to change the dtype of the model parameters, see <a href="/docs/transformers/pr_18694/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16">to_fp16()</a> and
<a href="/docs/transformers/pr_18694/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16">to_bf16()</a>.`,name:"dtype"}],source:"https://github.com/huggingface/transformers/blob/vr_18694/src/transformers/models/blenderbot_small/modeling_flax_blenderbot_small.py#L1215"}}),cn=new C({props:{name:"__call__",anchor:"transformers.FlaxBlenderbotSmallModel.__call__",parameters:[{name:"input_ids",val:": ndarray"},{name:"attention_mask",val:": typing.Optional[jax._src.numpy.ndarray.ndarray] = None"},{name:"decoder_input_ids",val:": typing.Optional[jax._src.numpy.ndarray.ndarray] = None"},{name:"decoder_attention_mask",val:": typing.Optional[jax._src.numpy.ndarray.ndarray] = None"},{name:"position_ids",val:": typing.Optional[jax._src.numpy.ndarray.ndarray] = None"},{name:"decoder_position_ids",val:": typing.Optional[jax._src.numpy.ndarray.ndarray] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"train",val:": bool = False"},{name:"params",val:": dict = None"},{name:"dropout_rng",val:": PRNGKey = None"}],source:"https://github.com/huggingface/transformers/blob/vr_18694/src/transformers/models/blenderbot_small/modeling_flax_blenderbot_small.py#L1151",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18694/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSeq2SeqModelOutput"
>transformers.modeling_flax_outputs.FlaxSeq2SeqModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallConfig"
>BlenderbotSmallConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) \u2014 Sequence of hidden-states at the output of the last layer of the decoder of the model.</p>
<p>If <code>past_key_values</code> is used only the last hidden-state of the sequences of shape <code>(batch_size, 1, hidden_size)</code> is output.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>tuple(tuple(jnp.ndarray))</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) \u2014 Tuple of <code>tuple(jnp.ndarray)</code> of length <code>config.n_layers</code>, with each tuple having 2 tensors of shape
<code>(batch_size, num_heads, sequence_length, embed_size_per_head)</code>) and 2 additional tensors of shape
<code>(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)</code>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>decoder_hidden_states</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>decoder_attentions</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder\u2019s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
<li>
<p><strong>encoder_last_hidden_state</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) \u2014 Sequence of hidden-states at the output of the last layer of the encoder of the model.</p>
</li>
<li>
<p><strong>encoder_hidden_states</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>encoder_attentions</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18694/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSeq2SeqModelOutput"
>transformers.modeling_flax_outputs.FlaxSeq2SeqModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Ho=new Je({props:{anchor:"transformers.FlaxBlenderbotSmallModel.__call__.example",$$slots:{default:[Su]},$$scope:{ctx:z}}}),pn=new C({props:{name:"encode",anchor:"transformers.FlaxBlenderbotSmallModel.encode",parameters:[{name:"input_ids",val:": ndarray"},{name:"attention_mask",val:": typing.Optional[jax._src.numpy.ndarray.ndarray] = None"},{name:"position_ids",val:": typing.Optional[jax._src.numpy.ndarray.ndarray] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"train",val:": bool = False"},{name:"params",val:": dict = None"},{name:"dropout_rng",val:": PRNGKey = None"}],parametersDescription:[{anchor:"transformers.FlaxBlenderbotSmallModel.encode.input_ids",description:`<strong>input_ids</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
it.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallTokenizer">BlenderbotSmallTokenizer</a>. See <a href="/docs/transformers/pr_18694/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18694/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FlaxBlenderbotSmallModel.encode.attention_mask",description:`<strong>attention_mask</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.FlaxBlenderbotSmallModel.encode.position_ids",description:`<strong>position_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.`,name:"position_ids"},{anchor:"transformers.FlaxBlenderbotSmallModel.encode.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.FlaxBlenderbotSmallModel.encode.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.FlaxBlenderbotSmallModel.encode.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18694/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18694/src/transformers/models/blenderbot_small/modeling_flax_blenderbot_small.py#L972",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18694/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutput"
>transformers.modeling_flax_outputs.FlaxBaseModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<code>&lt;class 'transformers.models.blenderbot_small.configuration_blenderbot_small.BlenderbotSmallConfig'&gt;</code>) and inputs.</p>
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
  href="/docs/transformers/pr_18694/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutput"
>transformers.modeling_flax_outputs.FlaxBaseModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Vo=new Je({props:{anchor:"transformers.FlaxBlenderbotSmallModel.encode.example",$$slots:{default:[$u]},$$scope:{ctx:z}}}),mn=new C({props:{name:"decode",anchor:"transformers.FlaxBlenderbotSmallModel.decode",parameters:[{name:"decoder_input_ids",val:""},{name:"encoder_outputs",val:""},{name:"encoder_attention_mask",val:": typing.Optional[jax._src.numpy.ndarray.ndarray] = None"},{name:"decoder_attention_mask",val:": typing.Optional[jax._src.numpy.ndarray.ndarray] = None"},{name:"decoder_position_ids",val:": typing.Optional[jax._src.numpy.ndarray.ndarray] = None"},{name:"past_key_values",val:": dict = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"train",val:": bool = False"},{name:"params",val:": dict = None"},{name:"dropout_rng",val:": PRNGKey = None"}],parametersDescription:[{anchor:"transformers.FlaxBlenderbotSmallModel.decode.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, target_sequence_length)</code>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallTokenizer">BlenderbotSmallTokenizer</a>. See <a href="/docs/transformers/pr_18694/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18694/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>For translation and summarization training, <code>decoder_input_ids</code> should be provided. If no
<code>decoder_input_ids</code> is provided, the model will create this tensor by shifting the <code>input_ids</code> to the right
for denoising pre-training following the paper.`,name:"decoder_input_ids"},{anchor:"transformers.FlaxBlenderbotSmallModel.decode.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple(tuple(jnp.ndarray)</code>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.FlaxBlenderbotSmallModel.decode.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"encoder_attention_mask"},{anchor:"transformers.FlaxBlenderbotSmallModel.decode.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.</p>
<p>If you want to change padding behavior, you should modify to your needs. See diagram 1 in <a href="https://arxiv.org/abs/1910.13461" rel="nofollow">the
paper</a> for more information on the default strategy.`,name:"decoder_attention_mask"},{anchor:"transformers.FlaxBlenderbotSmallModel.decode.decoder_position_ids",description:`<strong>decoder_position_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the
range <code>[0, config.max_position_embeddings - 1]</code>.`,name:"decoder_position_ids"},{anchor:"transformers.FlaxBlenderbotSmallModel.decode.past_key_values",description:`<strong>past_key_values</strong> (<code>Dict[str, np.ndarray]</code>, <em>optional</em>, returned by <code>init_cache</code> or when passing previous <code>past_key_values</code>) &#x2014;
Dictionary of pre-computed hidden-states (key and values in the attention blocks) that can be used for fast
auto-regressive decoding. Pre-computed key and value hidden-states are of shape <em>[batch_size, max_length]</em>.`,name:"past_key_values"},{anchor:"transformers.FlaxBlenderbotSmallModel.decode.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.FlaxBlenderbotSmallModel.decode.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.FlaxBlenderbotSmallModel.decode.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18694/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18694/src/transformers/models/blenderbot_small/modeling_flax_blenderbot_small.py#L1035",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18694/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutputWithPastAndCrossAttentions"
>transformers.modeling_flax_outputs.FlaxBaseModelOutputWithPastAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<code>&lt;class 'transformers.models.blenderbot_small.configuration_blenderbot_small.BlenderbotSmallConfig'&gt;</code>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) \u2014 Sequence of hidden-states at the output of the last layer of the model.</p>
<p>If <code>past_key_values</code> is used only the last hidden-state of the sequences of shape <code>(batch_size, 1, hidden_size)</code> is output.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>tuple(tuple(jnp.ndarray))</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) \u2014 Tuple of <code>tuple(jnp.ndarray)</code> of length <code>config.n_layers</code>, with each tuple having 2 tensors of shape
<code>(batch_size, num_heads, sequence_length, embed_size_per_head)</code>) and optionally if
<code>config.is_encoder_decoder=True</code> 2 additional tensors of shape <code>(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)</code>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
<code>config.is_encoder_decoder=True</code> in the cross-attention blocks) that can be used (see <code>past_key_values</code>
input) to speed up sequential decoding.</p>
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
<li>
<p><strong>cross_attentions</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> and <code>config.add_cross_attention=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder\u2019s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18694/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutputWithPastAndCrossAttentions"
>transformers.modeling_flax_outputs.FlaxBaseModelOutputWithPastAndCrossAttentions</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Jo=new Je({props:{anchor:"transformers.FlaxBlenderbotSmallModel.decode.example",$$slots:{default:[xu]},$$scope:{ctx:z}}}),hn=new xe({}),un=new C({props:{name:"class transformers.FlaxBlenderbotSmallForConditionalGeneration",anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration",parameters:[{name:"config",val:": BlenderbotSmallConfig"},{name:"input_shape",val:": typing.Tuple[int] = (1, 1)"},{name:"seed",val:": int = 0"},{name:"dtype",val:": dtype = <class 'jax.numpy.float32'>"},{name:"_do_init",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallConfig">BlenderbotSmallConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/pr_18694/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration.dtype",description:`<strong>dtype</strong> (<code>jax.numpy.dtype</code>, <em>optional</em>, defaults to <code>jax.numpy.float32</code>) &#x2014;
The data type of the computation. Can be one of <code>jax.numpy.float32</code>, <code>jax.numpy.float16</code> (on GPUs) and
<code>jax.numpy.bfloat16</code> (on TPUs).</p>
<p>This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
specified all the computation will be performed with the given <code>dtype</code>.</p>
<p><strong>Note that this only specifies the dtype of the computation and does not influence the dtype of model
parameters.</strong></p>
<p>If you wish to change the dtype of the model parameters, see <a href="/docs/transformers/pr_18694/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16">to_fp16()</a> and
<a href="/docs/transformers/pr_18694/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16">to_bf16()</a>.`,name:"dtype"}],source:"https://github.com/huggingface/transformers/blob/vr_18694/src/transformers/models/blenderbot_small/modeling_flax_blenderbot_small.py#L1303"}}),Tn=new C({props:{name:"__call__",anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration.__call__",parameters:[{name:"input_ids",val:": ndarray"},{name:"attention_mask",val:": typing.Optional[jax._src.numpy.ndarray.ndarray] = None"},{name:"decoder_input_ids",val:": typing.Optional[jax._src.numpy.ndarray.ndarray] = None"},{name:"decoder_attention_mask",val:": typing.Optional[jax._src.numpy.ndarray.ndarray] = None"},{name:"position_ids",val:": typing.Optional[jax._src.numpy.ndarray.ndarray] = None"},{name:"decoder_position_ids",val:": typing.Optional[jax._src.numpy.ndarray.ndarray] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"train",val:": bool = False"},{name:"params",val:": dict = None"},{name:"dropout_rng",val:": PRNGKey = None"}],parametersDescription:[{anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration.__call__.input_ids",description:`<strong>input_ids</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
it.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallTokenizer">BlenderbotSmallTokenizer</a>. See <a href="/docs/transformers/pr_18694/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18694/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration.__call__.attention_mask",description:`<strong>attention_mask</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration.__call__.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallTokenizer">BlenderbotSmallTokenizer</a>. See <a href="/docs/transformers/pr_18694/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18694/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>For translation and summarization training, <code>decoder_input_ids</code> should be provided. If no
<code>decoder_input_ids</code> is provided, the model will create this tensor by shifting the <code>input_ids</code> to the right
for denoising pre-training following the paper.`,name:"decoder_input_ids"},{anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration.__call__.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.</p>
<p>If you want to change padding behavior, you should modify to your needs. See diagram 1 in <a href="https://arxiv.org/abs/1910.13461" rel="nofollow">the
paper</a> for more information on the default strategy.`,name:"decoder_attention_mask"},{anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration.__call__.position_ids",description:`<strong>position_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.`,name:"position_ids"},{anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration.__call__.decoder_position_ids",description:`<strong>decoder_position_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the
range <code>[0, config.max_position_embeddings - 1]</code>.`,name:"decoder_position_ids"},{anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration.__call__.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration.__call__.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration.__call__.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18694/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18694/src/transformers/models/blenderbot_small/modeling_flax_blenderbot_small.py#L1151",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18694/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSeq2SeqLMOutput"
>transformers.modeling_flax_outputs.FlaxSeq2SeqLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallConfig"
>BlenderbotSmallConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>logits</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) \u2014 Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>tuple(tuple(jnp.ndarray))</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) \u2014 Tuple of <code>tuple(jnp.ndarray)</code> of length <code>config.n_layers</code>, with each tuple having 2 tensors of shape
<code>(batch_size, num_heads, sequence_length, embed_size_per_head)</code>) and 2 additional tensors of shape
<code>(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)</code>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>decoder_hidden_states</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>decoder_attentions</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder\u2019s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
<li>
<p><strong>encoder_last_hidden_state</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) \u2014 Sequence of hidden-states at the output of the last layer of the encoder of the model.</p>
</li>
<li>
<p><strong>encoder_hidden_states</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> (one for the output of the embeddings + one for the output of each layer) of shape
<code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>encoder_attentions</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18694/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSeq2SeqLMOutput"
>transformers.modeling_flax_outputs.FlaxSeq2SeqLMOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Yo=new qn({props:{$$slots:{default:[Bu]},$$scope:{ctx:z}}}),Sn=new xe({}),$n=new C({props:{name:"encode",anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration.encode",parameters:[{name:"input_ids",val:": ndarray"},{name:"attention_mask",val:": typing.Optional[jax._src.numpy.ndarray.ndarray] = None"},{name:"position_ids",val:": typing.Optional[jax._src.numpy.ndarray.ndarray] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"train",val:": bool = False"},{name:"params",val:": dict = None"},{name:"dropout_rng",val:": PRNGKey = None"}],parametersDescription:[{anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration.encode.input_ids",description:`<strong>input_ids</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
it.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallTokenizer">BlenderbotSmallTokenizer</a>. See <a href="/docs/transformers/pr_18694/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18694/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration.encode.attention_mask",description:`<strong>attention_mask</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration.encode.position_ids",description:`<strong>position_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.`,name:"position_ids"},{anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration.encode.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration.encode.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration.encode.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18694/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18694/src/transformers/models/blenderbot_small/modeling_flax_blenderbot_small.py#L972",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18694/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutput"
>transformers.modeling_flax_outputs.FlaxBaseModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<code>&lt;class 'transformers.models.blenderbot_small.configuration_blenderbot_small.BlenderbotSmallConfig'&gt;</code>) and inputs.</p>
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
  href="/docs/transformers/pr_18694/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutput"
>transformers.modeling_flax_outputs.FlaxBaseModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),tt=new Je({props:{anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration.encode.example",$$slots:{default:[zu]},$$scope:{ctx:z}}}),xn=new C({props:{name:"decode",anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration.decode",parameters:[{name:"decoder_input_ids",val:""},{name:"encoder_outputs",val:""},{name:"encoder_attention_mask",val:": typing.Optional[jax._src.numpy.ndarray.ndarray] = None"},{name:"decoder_attention_mask",val:": typing.Optional[jax._src.numpy.ndarray.ndarray] = None"},{name:"decoder_position_ids",val:": typing.Optional[jax._src.numpy.ndarray.ndarray] = None"},{name:"past_key_values",val:": dict = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"deterministic",val:": bool = True"},{name:"params",val:": dict = None"},{name:"dropout_rng",val:": PRNGKey = None"}],parametersDescription:[{anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration.decode.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, target_sequence_length)</code>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallTokenizer">BlenderbotSmallTokenizer</a>. See <a href="/docs/transformers/pr_18694/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/pr_18694/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>For translation and summarization training, <code>decoder_input_ids</code> should be provided. If no
<code>decoder_input_ids</code> is provided, the model will create this tensor by shifting the <code>input_ids</code> to the right
for denoising pre-training following the paper.`,name:"decoder_input_ids"},{anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration.decode.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple(tuple(jnp.ndarray)</code>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration.decode.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"encoder_attention_mask"},{anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration.decode.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>jnp.ndarray</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.</p>
<p>If you want to change padding behavior, you should modify to your needs. See diagram 1 in <a href="https://arxiv.org/abs/1910.13461" rel="nofollow">the
paper</a> for more information on the default strategy.`,name:"decoder_attention_mask"},{anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration.decode.decoder_position_ids",description:`<strong>decoder_position_ids</strong> (<code>numpy.ndarray</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the
range <code>[0, config.max_position_embeddings - 1]</code>.`,name:"decoder_position_ids"},{anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration.decode.past_key_values",description:`<strong>past_key_values</strong> (<code>Dict[str, np.ndarray]</code>, <em>optional</em>, returned by <code>init_cache</code> or when passing previous <code>past_key_values</code>) &#x2014;
Dictionary of pre-computed hidden-states (key and values in the attention blocks) that can be used for fast
auto-regressive decoding. Pre-computed key and value hidden-states are of shape <em>[batch_size, max_length]</em>.`,name:"past_key_values"},{anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration.decode.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration.decode.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration.decode.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/pr_18694/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/vr_18694/src/transformers/models/blenderbot_small/modeling_flax_blenderbot_small.py#L1307",returnDescription:`
<p>A <a
  href="/docs/transformers/pr_18694/en/main_classes/output#transformers.modeling_flax_outputs.FlaxCausalLMOutputWithCrossAttentions"
>transformers.modeling_flax_outputs.FlaxCausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<code>&lt;class 'transformers.models.blenderbot_small.configuration_blenderbot_small.BlenderbotSmallConfig'&gt;</code>) and inputs.</p>
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
<li>
<p><strong>cross_attentions</strong> (<code>tuple(jnp.ndarray)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Cross attentions weights after the attention softmax, used to compute the weighted average in the
cross-attention heads.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>tuple(tuple(jnp.ndarray))</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) \u2014 Tuple of <code>jnp.ndarray</code> tuples of length <code>config.n_layers</code>, with each tuple containing the cached key, value
states of the self-attention and the cross-attention layers if model is used in encoder-decoder setting.
Only relevant if <code>config.is_decoder = True</code>.</p>
<p>Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
<code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
</ul>
`,returnType:`
<p><a
  href="/docs/transformers/pr_18694/en/main_classes/output#transformers.modeling_flax_outputs.FlaxCausalLMOutputWithCrossAttentions"
>transformers.modeling_flax_outputs.FlaxCausalLMOutputWithCrossAttentions</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),at=new Je({props:{anchor:"transformers.FlaxBlenderbotSmallForConditionalGeneration.decode.example",$$slots:{default:[Fu]},$$scope:{ctx:z}}}),{c(){p=n("meta"),k=i(),f=n("h1"),h=n("a"),b=n("span"),v(l.$$.fragment),u=i(),F=n("span"),Ce=s("Blenderbot Small"),he=i(),B=n("p"),je=s("Note that "),W=n("a"),Oe=s("BlenderbotSmallModel"),Pe=s(` and
`),R=n("a"),Le=s("BlenderbotSmallForConditionalGeneration"),Ne=s(` are only used in combination with the checkpoint
`),V=n("a"),Q=s("facebook/blenderbot-90M"),Ie=s(`. Larger Blenderbot checkpoints should
instead be used with `),Z=n("a"),j=s("BlenderbotModel"),L=s(` and
`),ue=n("a"),ae=s("BlenderbotForConditionalGeneration"),Be=i(),J=n("h2"),D=n("a"),be=n("span"),v(re.$$.fragment),N=i(),ke=n("span"),se=s("Overview"),ze=i(),ee=n("p"),de=s("The Blender chatbot model was proposed in "),le=n("a"),Ae=s("Recipes for building an open-domain chatbot"),X=s(` Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu,
Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston on 30 Apr 2020.`),Fe=i(),H=n("p"),De=s("The abstract of the paper is the following:"),x=i(),q=n("p"),ie=n("em"),Ze=s(`Building open-domain chatbots is a challenging area for machine learning research. While prior work has shown that
scaling neural models in the number of parameters and the size of the data they are trained on gives improved results,
we show that other ingredients are important for a high-performing chatbot. Good conversation requires a number of
skills that an expert conversationalist blends in a seamless way: providing engaging talking points and listening to
their partners, and displaying knowledge, empathy and personality appropriately, while maintaining a consistent
persona. We show that large scale models can learn these skills when given appropriate training data and choice of
generation strategy. We build variants of these recipes with 90M, 2.7B and 9.4B parameter models, and make our models
and code publicly available. Human evaluations show our best models are superior to existing approaches in multi-turn
dialogue in terms of engagingness and humanness measurements. We then discuss the limitations of this work by analyzing
failure cases of our models.`),Re=i(),O=n("p"),eo=s("This model was contributed by "),ve=n("a"),oo=s("patrickvonplaten"),Y=s(`. The authors\u2019 code can be
found `),G=n("a"),to=s("here"),no=s(" ."),oe=i(),ce=n("h2"),fe=n("a"),ye=n("span"),v(Te.$$.fragment),md=i(),sa=n("span"),hd=s("BlenderbotSmallConfig"),cs=i(),we=n("div"),v(ht.$$.fragment),ud=i(),ao=n("p"),fd=s("This is the configuration class to store the configuration of a "),En=n("a"),_d=s("BlenderbotSmallModel"),gd=s(`. It is used to instantiate
an BlenderbotSmall model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the BlenderbotSmall
`),ut=n("a"),bd=s("facebook/blenderbot_small-90M"),kd=s(" architecture."),vd=i(),ro=n("p"),yd=s("Configuration objects inherit from "),Mn=n("a"),Td=s("PretrainedConfig"),wd=s(` and can be used to control the model outputs. Read the
documentation from `),Cn=n("a"),Sd=s("PretrainedConfig"),$d=s(" for more information."),xd=i(),v(wo.$$.fragment),ps=i(),so=n("h2"),So=n("a"),da=n("span"),v(ft.$$.fragment),Bd=i(),la=n("span"),zd=s("BlenderbotSmallTokenizer"),ms=i(),U=n("div"),v(_t.$$.fragment),Fd=i(),ia=n("p"),qd=s("Constructs a Blenderbot-90M tokenizer based on BPE (Byte-Pair-Encoding)"),Ed=i(),gt=n("p"),Md=s("This tokenizer inherits from "),jn=n("a"),Cd=s("PreTrainedTokenizer"),jd=s(` which contains most of the main methods. Users should refer to
the superclass for more information regarding methods.`),Od=i(),He=n("div"),v(bt.$$.fragment),Pd=i(),ca=n("p"),Ld=s(`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens.`),Nd=i(),pa=n("p"),Id=s("This implementation does not add special tokens and this method should be overridden in a subclass."),Ad=i(),$o=n("div"),v(kt.$$.fragment),Dd=i(),lo=n("p"),Gd=s(`Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer `),ma=n("code"),Ud=s("prepare_for_model"),Wd=s(" or "),ha=n("code"),Rd=s("encode_plus"),Hd=s(" methods."),Kd=i(),Ke=n("div"),v(vt.$$.fragment),Vd=i(),On=n("p"),Qd=s("Create the token type IDs corresponding to the sequences passed. "),Pn=n("a"),Jd=s(`What are token type
IDs?`),Xd=i(),ua=n("p"),Yd=s("Should be overridden in a subclass if the model has a special way of building those."),Zd=i(),Ln=n("div"),v(yt.$$.fragment),hs=i(),io=n("h2"),xo=n("a"),fa=n("span"),v(Tt.$$.fragment),el=i(),_a=n("span"),ol=s("BlenderbotSmallTokenizerFast"),us=i(),Ge=n("div"),v(wt.$$.fragment),tl=i(),St=n("p"),nl=s("Construct a \u201Cfast\u201D BlenderbotSmall tokenizer (backed by HuggingFace\u2019s "),ga=n("em"),al=s("tokenizers"),rl=s(" library)."),sl=i(),Bo=n("div"),v($t.$$.fragment),dl=i(),ba=n("p"),ll=s(`Create a mask from the two sequences passed to be used in a sequence-pair classification task. BlenderbotSmall
does not make use of token type ids, therefore a list of zeros is returned.`),fs=i(),co=n("h2"),zo=n("a"),ka=n("span"),v(xt.$$.fragment),il=i(),va=n("span"),cl=s("BlenderbotSmallModel"),_s=i(),Se=n("div"),v(Bt.$$.fragment),pl=i(),zt=n("p"),ml=s(`The bare BlenderbotSmall Model outputting raw hidden-states without any specific head on top.
This model inherits from `),Nn=n("a"),hl=s("PreTrainedModel"),ul=s(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),fl=i(),Ft=n("p"),_l=s("This model is also a PyTorch "),qt=n("a"),gl=s("torch.nn.Module"),bl=s(` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),kl=i(),qe=n("div"),v(Et.$$.fragment),vl=i(),po=n("p"),yl=s("The "),In=n("a"),Tl=s("BlenderbotSmallModel"),wl=s(" forward method, overrides the "),ya=n("code"),Sl=s("__call__"),$l=s(" special method."),xl=i(),v(Fo.$$.fragment),Bl=i(),v(qo.$$.fragment),gs=i(),mo=n("h2"),Eo=n("a"),Ta=n("span"),v(Mt.$$.fragment),zl=i(),wa=n("span"),Fl=s("BlenderbotSmallForConditionalGeneration"),bs=i(),$e=n("div"),v(Ct.$$.fragment),ql=i(),jt=n("p"),El=s(`The BlenderbotSmall Model with a language modeling head. Can be used for summarization.
This model inherits from `),An=n("a"),Ml=s("PreTrainedModel"),Cl=s(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),jl=i(),Ot=n("p"),Ol=s("This model is also a PyTorch "),Pt=n("a"),Pl=s("torch.nn.Module"),Ll=s(` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Nl=i(),Ee=n("div"),v(Lt.$$.fragment),Il=i(),ho=n("p"),Al=s("The "),Dn=n("a"),Dl=s("BlenderbotSmallForConditionalGeneration"),Gl=s(" forward method, overrides the "),Sa=n("code"),Ul=s("__call__"),Wl=s(" special method."),Rl=i(),v(Mo.$$.fragment),Hl=i(),v(Co.$$.fragment),ks=i(),uo=n("h2"),jo=n("a"),$a=n("span"),v(Nt.$$.fragment),Kl=i(),xa=n("span"),Vl=s("BlenderbotSmallForCausalLM"),vs=i(),fo=n("div"),v(It.$$.fragment),Ql=i(),Oo=n("div"),v(At.$$.fragment),Jl=i(),v(Po.$$.fragment),ys=i(),_o=n("h2"),Lo=n("a"),Ba=n("span"),v(Dt.$$.fragment),Xl=i(),za=n("span"),Yl=s("TFBlenderbotSmallModel"),Ts=i(),pe=n("div"),v(Gt.$$.fragment),Zl=i(),Ut=n("p"),ei=s(`The bare BLENDERBOT_SMALL Model outputting raw hidden-states without any specific head on top.
This model inherits from `),Gn=n("a"),oi=s("TFPreTrainedModel"),ti=s(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),ni=i(),Wt=n("p"),ai=s("This model is also a "),Rt=n("a"),ri=s("tf.keras.Model"),si=s(` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),di=i(),v(No.$$.fragment),li=i(),Me=n("div"),v(Ht.$$.fragment),ii=i(),go=n("p"),ci=s("The "),Un=n("a"),pi=s("TFBlenderbotSmallModel"),mi=s(" forward method, overrides the "),Fa=n("code"),hi=s("__call__"),ui=s(" special method."),fi=i(),v(Io.$$.fragment),_i=i(),v(Ao.$$.fragment),ws=i(),bo=n("h2"),Do=n("a"),qa=n("span"),v(Kt.$$.fragment),gi=i(),Ea=n("span"),bi=s("TFBlenderbotSmallForConditionalGeneration"),Ss=i(),me=n("div"),v(Vt.$$.fragment),ki=i(),Qt=n("p"),vi=s(`The BLENDERBOT_SMALL Model with a language modeling head. Can be used for summarization.
This model inherits from `),Wn=n("a"),yi=s("TFPreTrainedModel"),Ti=s(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),wi=i(),Jt=n("p"),Si=s("This model is also a "),Xt=n("a"),$i=s("tf.keras.Model"),xi=s(` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),Bi=i(),v(Go.$$.fragment),zi=i(),P=n("div"),v(Yt.$$.fragment),Fi=i(),ko=n("p"),qi=s("The "),Rn=n("a"),Ei=s("TFBlenderbotSmallForConditionalGeneration"),Mi=s(" forward method, overrides the "),Ma=n("code"),Ci=s("__call__"),ji=s(" special method."),Oi=i(),v(Uo.$$.fragment),Pi=i(),Ca=n("p"),Li=s("Conversation example::"),Ni=i(),ja=n("blockquote"),Oa=n("blockquote"),Pa=n("blockquote"),La=n("p"),Ii=s(`from transformers import BlenderbotSmallTokenizer, TFBlenderbotSmallForConditionalGeneration >>> mname =
\u2018facebook/blenderbot_small-90M\u2019 >>> model = BlenderbotSmallForConditionalGeneration.from_pretrained(mname) >>>
tokenizer = BlenderbotSmallTokenizer.from_pretrained(mname)`),Ai=i(),Na=n("blockquote"),Ia=n("blockquote"),Aa=n("blockquote"),Da=n("p"),Di=s(`UTTERANCE = \u201CMy friends are cool but they eat too many carbs.\u201D >>> print(\u201CHuman: \u201D, UTTERANCE) >>> inputs =
tokenizer([UTTERANCE], return_tensors=\u2018tf\u2019)`),Gi=i(),Ga=n("blockquote"),Ua=n("blockquote"),Wa=n("blockquote"),Ra=n("p"),Ui=s(`reply_ids = model.generate(**inputs) >>> print(\u201CBot: \u201D, tokenizer.batch_decode(reply_ids,
skip_special_tokens=True)[0]) what kind of carbs do they eat? i don\u2019t know much about carbs.`),Wi=i(),Ha=n("blockquote"),Ka=n("blockquote"),Va=n("blockquote"),Zt=n("p"),Ri=s(`REPLY = \u201CI\u2019m not sure\u201D >>> print(\u201CHuman: \u201D, REPLY) >>> NEXT_UTTERANCE = ( \u2026 \u201CMy friends are cool but they
eat too many carbs.</s> \u201D \u2026 \u201D`),Qa=n("s"),Hi=s("what kind of carbs do they eat? i don\u2019t know much about carbs."),Ki=s(` \u201D \u2026
\u201D<s>I\u2019m not sure.\u201D \u2026 )`),Vi=i(),Ja=n("blockquote"),Xa=n("blockquote"),Ya=n("blockquote"),Za=n("p"),Qi=s(`inputs = tokenizer([NEXT_UTTERANCE], return_tensors=\u2018tf\u2019) >>> inputs.pop(\u201Ctoken_type_ids\u201D) >>>
next_reply_ids = model.generate(**inputs) >>> print(\u201CBot: \u201D, tokenizer.batch_decode(next_reply_ids,
skip_special_tokens=True)[0])`),$s=i(),vo=n("h2"),Wo=n("a"),er=n("span"),v(en.$$.fragment),Ji=i(),or=n("span"),Xi=s("FlaxBlenderbotSmallModel"),xs=i(),I=n("div"),v(on.$$.fragment),Yi=i(),tn=n("p"),Zi=s(`The bare BlenderbotSmall Model transformer outputting raw hidden-states without any specific head on top.
This model inherits from `),Hn=n("a"),ec=s("FlaxPreTrainedModel"),oc=s(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),tc=i(),nn=n("p"),nc=s(`This model is also a Flax Linen
`),an=n("a"),ac=s("flax.nn.Module"),rc=s(` subclass. Use it as a
regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.`),sc=i(),tr=n("p"),dc=s("Finally, this model supports inherent JAX features such as:"),lc=i(),Ue=n("ul"),nr=n("li"),rn=n("a"),ic=s("Just-In-Time (JIT) compilation"),cc=i(),ar=n("li"),sn=n("a"),pc=s("Automatic Differentiation"),mc=i(),rr=n("li"),dn=n("a"),hc=s("Vectorization"),uc=i(),sr=n("li"),ln=n("a"),fc=s("Parallelization"),_c=i(),Ro=n("div"),v(cn.$$.fragment),gc=i(),v(Ho.$$.fragment),bc=i(),Ko=n("div"),v(pn.$$.fragment),kc=i(),v(Vo.$$.fragment),vc=i(),Qo=n("div"),v(mn.$$.fragment),yc=i(),v(Jo.$$.fragment),Bs=i(),yo=n("h2"),Xo=n("a"),dr=n("span"),v(hn.$$.fragment),Tc=i(),lr=n("span"),wc=s("FlaxBlenderbotForConditionalGeneration"),zs=i(),A=n("div"),v(un.$$.fragment),Sc=i(),fn=n("p"),$c=s(`The BLENDERBOT_SMALL Model with a language modeling head. Can be used for summarization.
This model inherits from `),Kn=n("a"),xc=s("FlaxPreTrainedModel"),Bc=s(`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),zc=i(),_n=n("p"),Fc=s(`This model is also a Flax Linen
`),gn=n("a"),qc=s("flax.nn.Module"),Ec=s(` subclass. Use it as a
regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.`),Mc=i(),ir=n("p"),Cc=s("Finally, this model supports inherent JAX features such as:"),jc=i(),We=n("ul"),cr=n("li"),bn=n("a"),Oc=s("Just-In-Time (JIT) compilation"),Pc=i(),pr=n("li"),kn=n("a"),Lc=s("Automatic Differentiation"),Nc=i(),mr=n("li"),vn=n("a"),Ic=s("Vectorization"),Ac=i(),hr=n("li"),yn=n("a"),Dc=s("Parallelization"),Gc=i(),E=n("div"),v(Tn.$$.fragment),Uc=i(),To=n("p"),Wc=s("The "),ur=n("code"),Rc=s("FlaxBlenderbotSmallPreTrainedModel"),Hc=s(" forward method, overrides the "),fr=n("code"),Kc=s("__call__"),Vc=s(" special method."),Qc=i(),v(Yo.$$.fragment),Jc=i(),_r=n("p"),Xc=s("Summarization example:"),Yc=i(),gr=n("blockquote"),br=n("blockquote"),kr=n("blockquote"),vr=n("p"),Zc=s("from transformers import BlenderbotSmallTokenizer, FlaxBlenderbotSmallForConditionalGeneration"),ep=i(),yr=n("blockquote"),Tr=n("blockquote"),wr=n("blockquote"),Sr=n("p"),op=s(`model = FlaxBlenderbotSmallForConditionalGeneration.from_pretrained(\u2018facebook/blenderbot_small-90M\u2019) >>>
tokenizer = BlenderbotSmallTokenizer.from_pretrained(\u2018facebook/blenderbot_small-90M\u2019)`),tp=i(),$r=n("blockquote"),xr=n("blockquote"),Br=n("blockquote"),zr=n("p"),np=s(`ARTICLE_TO_SUMMARIZE = \u201CMy friends are cool but they eat too many carbs.\u201D >>> inputs =
tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors=\u2018np\u2019)`),ap=i(),Fr=n("blockquote"),qr=n("blockquote"),wn=n("blockquote"),Zo=n("h1"),et=n("a"),Er=n("span"),v(Sn.$$.fragment),rp=i(),Mr=n("span"),sp=s("Generate Summary >>> summary_ids = model.generate(inputs[\u2018input_ids\u2019]).sequences >>>"),dp=i(),Cr=n("p"),lp=s("print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))"),ip=i(),jr=n("p"),cp=s("Mask filling example:"),pp=i(),Or=n("blockquote"),Pr=n("blockquote"),Lr=n("blockquote"),Nr=n("p"),mp=s(`from transformers import BlenderbotSmallTokenizer, FlaxBlenderbotSmallForConditionalGeneration >>>
tokenizer = BlenderbotSmallTokenizer.from_pretrained(\u2018facebook/blenderbot_small-90M\u2019) >>> TXT = \u201CMy friends are
<mask> but they eat too many carbs.\u201D`),hp=i(),Ir=n("blockquote"),Ar=n("blockquote"),Dr=n("blockquote"),Gr=n("p"),up=s(`model = FlaxBlenderbotSmallForConditionalGeneration.from_pretrained(\u2018facebook/blenderbot_small-90M\u2019) >>>
input_ids = tokenizer([TXT], return_tensors=\u2018np\u2019)[\u2018input_ids\u2019] >>> logits = model(input_ids).logits`),fp=i(),Ur=n("blockquote"),Wr=n("blockquote"),Rr=n("blockquote"),Hr=n("p"),_p=s(`masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item() >>> probs =
jax.nn.softmax(logits[0, masked_index], axis=0) >>> values, predictions = jax.lax.top_k(probs)`),gp=i(),Kr=n("blockquote"),Vr=n("blockquote"),Qr=n("blockquote"),Jr=n("p"),bp=s("tokenizer.decode(predictions).split()"),kp=i(),ot=n("div"),v($n.$$.fragment),vp=i(),v(tt.$$.fragment),yp=i(),nt=n("div"),v(xn.$$.fragment),Tp=i(),v(at.$$.fragment),this.h()},l(t){const g=pu('[data-svelte="svelte-1phssyn"]',document.head);p=a(g,"META",{name:!0,content:!0}),g.forEach(o),k=c(t),f=a(t,"H1",{class:!0});var Bn=r(f);h=a(Bn,"A",{id:!0,class:!0,href:!0});var Xr=r(h);b=a(Xr,"SPAN",{});var Yr=r(b);y(l.$$.fragment,Yr),Yr.forEach(o),Xr.forEach(o),u=c(Bn),F=a(Bn,"SPAN",{});var Zr=r(F);Ce=d(Zr,"Blenderbot Small"),Zr.forEach(o),Bn.forEach(o),he=c(t),B=a(t,"P",{});var _e=r(B);je=d(_e,"Note that "),W=a(_e,"A",{href:!0});var es=r(W);Oe=d(es,"BlenderbotSmallModel"),es.forEach(o),Pe=d(_e,` and
`),R=a(_e,"A",{href:!0});var os=r(R);Le=d(os,"BlenderbotSmallForConditionalGeneration"),os.forEach(o),Ne=d(_e,` are only used in combination with the checkpoint
`),V=a(_e,"A",{href:!0,rel:!0});var ts=r(V);Q=d(ts,"facebook/blenderbot-90M"),ts.forEach(o),Ie=d(_e,`. Larger Blenderbot checkpoints should
instead be used with `),Z=a(_e,"A",{href:!0});var ns=r(Z);j=d(ns,"BlenderbotModel"),ns.forEach(o),L=d(_e,` and
`),ue=a(_e,"A",{href:!0});var as=r(ue);ae=d(as,"BlenderbotForConditionalGeneration"),as.forEach(o),_e.forEach(o),Be=c(t),J=a(t,"H2",{class:!0});var zn=r(J);D=a(zn,"A",{id:!0,class:!0,href:!0});var rs=r(D);be=a(rs,"SPAN",{});var ss=r(be);y(re.$$.fragment,ss),ss.forEach(o),rs.forEach(o),N=c(zn),ke=a(zn,"SPAN",{});var ds=r(ke);se=d(ds,"Overview"),ds.forEach(o),zn.forEach(o),ze=c(t),ee=a(t,"P",{});var Fn=r(ee);de=d(Fn,"The Blender chatbot model was proposed in "),le=a(Fn,"A",{href:!0,rel:!0});var ls=r(le);Ae=d(ls,"Recipes for building an open-domain chatbot"),ls.forEach(o),X=d(Fn,` Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu,
Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston on 30 Apr 2020.`),Fn.forEach(o),Fe=c(t),H=a(t,"P",{});var is=r(H);De=d(is,"The abstract of the paper is the following:"),is.forEach(o),x=c(t),q=a(t,"P",{});var Sp=r(q);ie=a(Sp,"EM",{});var $p=r(ie);Ze=d($p,`Building open-domain chatbots is a challenging area for machine learning research. While prior work has shown that
scaling neural models in the number of parameters and the size of the data they are trained on gives improved results,
we show that other ingredients are important for a high-performing chatbot. Good conversation requires a number of
skills that an expert conversationalist blends in a seamless way: providing engaging talking points and listening to
their partners, and displaying knowledge, empathy and personality appropriately, while maintaining a consistent
persona. We show that large scale models can learn these skills when given appropriate training data and choice of
generation strategy. We build variants of these recipes with 90M, 2.7B and 9.4B parameter models, and make our models
and code publicly available. Human evaluations show our best models are superior to existing approaches in multi-turn
dialogue in terms of engagingness and humanness measurements. We then discuss the limitations of this work by analyzing
failure cases of our models.`),$p.forEach(o),Sp.forEach(o),Re=c(t),O=a(t,"P",{});var Vn=r(O);eo=d(Vn,"This model was contributed by "),ve=a(Vn,"A",{href:!0,rel:!0});var xp=r(ve);oo=d(xp,"patrickvonplaten"),xp.forEach(o),Y=d(Vn,`. The authors\u2019 code can be
found `),G=a(Vn,"A",{href:!0,rel:!0});var Bp=r(G);to=d(Bp,"here"),Bp.forEach(o),no=d(Vn," ."),Vn.forEach(o),oe=c(t),ce=a(t,"H2",{class:!0});var qs=r(ce);fe=a(qs,"A",{id:!0,class:!0,href:!0});var zp=r(fe);ye=a(zp,"SPAN",{});var Fp=r(ye);y(Te.$$.fragment,Fp),Fp.forEach(o),zp.forEach(o),md=c(qs),sa=a(qs,"SPAN",{});var qp=r(sa);hd=d(qp,"BlenderbotSmallConfig"),qp.forEach(o),qs.forEach(o),cs=c(t),we=a(t,"DIV",{class:!0});var rt=r(we);y(ht.$$.fragment,rt),ud=c(rt),ao=a(rt,"P",{});var Qn=r(ao);fd=d(Qn,"This is the configuration class to store the configuration of a "),En=a(Qn,"A",{href:!0});var Ep=r(En);_d=d(Ep,"BlenderbotSmallModel"),Ep.forEach(o),gd=d(Qn,`. It is used to instantiate
an BlenderbotSmall model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the BlenderbotSmall
`),ut=a(Qn,"A",{href:!0,rel:!0});var Mp=r(ut);bd=d(Mp,"facebook/blenderbot_small-90M"),Mp.forEach(o),kd=d(Qn," architecture."),Qn.forEach(o),vd=c(rt),ro=a(rt,"P",{});var Jn=r(ro);yd=d(Jn,"Configuration objects inherit from "),Mn=a(Jn,"A",{href:!0});var Cp=r(Mn);Td=d(Cp,"PretrainedConfig"),Cp.forEach(o),wd=d(Jn,` and can be used to control the model outputs. Read the
documentation from `),Cn=a(Jn,"A",{href:!0});var jp=r(Cn);Sd=d(jp,"PretrainedConfig"),jp.forEach(o),$d=d(Jn," for more information."),Jn.forEach(o),xd=c(rt),y(wo.$$.fragment,rt),rt.forEach(o),ps=c(t),so=a(t,"H2",{class:!0});var Es=r(so);So=a(Es,"A",{id:!0,class:!0,href:!0});var Op=r(So);da=a(Op,"SPAN",{});var Pp=r(da);y(ft.$$.fragment,Pp),Pp.forEach(o),Op.forEach(o),Bd=c(Es),la=a(Es,"SPAN",{});var Lp=r(la);zd=d(Lp,"BlenderbotSmallTokenizer"),Lp.forEach(o),Es.forEach(o),ms=c(t),U=a(t,"DIV",{class:!0});var ge=r(U);y(_t.$$.fragment,ge),Fd=c(ge),ia=a(ge,"P",{});var Np=r(ia);qd=d(Np,"Constructs a Blenderbot-90M tokenizer based on BPE (Byte-Pair-Encoding)"),Np.forEach(o),Ed=c(ge),gt=a(ge,"P",{});var Ms=r(gt);Md=d(Ms,"This tokenizer inherits from "),jn=a(Ms,"A",{href:!0});var Ip=r(jn);Cd=d(Ip,"PreTrainedTokenizer"),Ip.forEach(o),jd=d(Ms,` which contains most of the main methods. Users should refer to
the superclass for more information regarding methods.`),Ms.forEach(o),Od=c(ge),He=a(ge,"DIV",{class:!0});var Xn=r(He);y(bt.$$.fragment,Xn),Pd=c(Xn),ca=a(Xn,"P",{});var Ap=r(ca);Ld=d(Ap,`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens.`),Ap.forEach(o),Nd=c(Xn),pa=a(Xn,"P",{});var Dp=r(pa);Id=d(Dp,"This implementation does not add special tokens and this method should be overridden in a subclass."),Dp.forEach(o),Xn.forEach(o),Ad=c(ge),$o=a(ge,"DIV",{class:!0});var Cs=r($o);y(kt.$$.fragment,Cs),Dd=c(Cs),lo=a(Cs,"P",{});var Yn=r(lo);Gd=d(Yn,`Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer `),ma=a(Yn,"CODE",{});var Gp=r(ma);Ud=d(Gp,"prepare_for_model"),Gp.forEach(o),Wd=d(Yn," or "),ha=a(Yn,"CODE",{});var Up=r(ha);Rd=d(Up,"encode_plus"),Up.forEach(o),Hd=d(Yn," methods."),Yn.forEach(o),Cs.forEach(o),Kd=c(ge),Ke=a(ge,"DIV",{class:!0});var Zn=r(Ke);y(vt.$$.fragment,Zn),Vd=c(Zn),On=a(Zn,"P",{});var wp=r(On);Qd=d(wp,"Create the token type IDs corresponding to the sequences passed. "),Pn=a(wp,"A",{href:!0});var Wp=r(Pn);Jd=d(Wp,`What are token type
IDs?`),Wp.forEach(o),wp.forEach(o),Xd=c(Zn),ua=a(Zn,"P",{});var Rp=r(ua);Yd=d(Rp,"Should be overridden in a subclass if the model has a special way of building those."),Rp.forEach(o),Zn.forEach(o),Zd=c(ge),Ln=a(ge,"DIV",{class:!0});var Hp=r(Ln);y(yt.$$.fragment,Hp),Hp.forEach(o),ge.forEach(o),hs=c(t),io=a(t,"H2",{class:!0});var js=r(io);xo=a(js,"A",{id:!0,class:!0,href:!0});var Kp=r(xo);fa=a(Kp,"SPAN",{});var Vp=r(fa);y(Tt.$$.fragment,Vp),Vp.forEach(o),Kp.forEach(o),el=c(js),_a=a(js,"SPAN",{});var Qp=r(_a);ol=d(Qp,"BlenderbotSmallTokenizerFast"),Qp.forEach(o),js.forEach(o),us=c(t),Ge=a(t,"DIV",{class:!0});var ea=r(Ge);y(wt.$$.fragment,ea),tl=c(ea),St=a(ea,"P",{});var Os=r(St);nl=d(Os,"Construct a \u201Cfast\u201D BlenderbotSmall tokenizer (backed by HuggingFace\u2019s "),ga=a(Os,"EM",{});var Jp=r(ga);al=d(Jp,"tokenizers"),Jp.forEach(o),rl=d(Os," library)."),Os.forEach(o),sl=c(ea),Bo=a(ea,"DIV",{class:!0});var Ps=r(Bo);y($t.$$.fragment,Ps),dl=c(Ps),ba=a(Ps,"P",{});var Xp=r(ba);ll=d(Xp,`Create a mask from the two sequences passed to be used in a sequence-pair classification task. BlenderbotSmall
does not make use of token type ids, therefore a list of zeros is returned.`),Xp.forEach(o),Ps.forEach(o),ea.forEach(o),fs=c(t),co=a(t,"H2",{class:!0});var Ls=r(co);zo=a(Ls,"A",{id:!0,class:!0,href:!0});var Yp=r(zo);ka=a(Yp,"SPAN",{});var Zp=r(ka);y(xt.$$.fragment,Zp),Zp.forEach(o),Yp.forEach(o),il=c(Ls),va=a(Ls,"SPAN",{});var em=r(va);cl=d(em,"BlenderbotSmallModel"),em.forEach(o),Ls.forEach(o),_s=c(t),Se=a(t,"DIV",{class:!0});var st=r(Se);y(Bt.$$.fragment,st),pl=c(st),zt=a(st,"P",{});var Ns=r(zt);ml=d(Ns,`The bare BlenderbotSmall Model outputting raw hidden-states without any specific head on top.
This model inherits from `),Nn=a(Ns,"A",{href:!0});var om=r(Nn);hl=d(om,"PreTrainedModel"),om.forEach(o),ul=d(Ns,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Ns.forEach(o),fl=c(st),Ft=a(st,"P",{});var Is=r(Ft);_l=d(Is,"This model is also a PyTorch "),qt=a(Is,"A",{href:!0,rel:!0});var tm=r(qt);gl=d(tm,"torch.nn.Module"),tm.forEach(o),bl=d(Is,` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Is.forEach(o),kl=c(st),qe=a(st,"DIV",{class:!0});var dt=r(qe);y(Et.$$.fragment,dt),vl=c(dt),po=a(dt,"P",{});var oa=r(po);yl=d(oa,"The "),In=a(oa,"A",{href:!0});var nm=r(In);Tl=d(nm,"BlenderbotSmallModel"),nm.forEach(o),wl=d(oa," forward method, overrides the "),ya=a(oa,"CODE",{});var am=r(ya);Sl=d(am,"__call__"),am.forEach(o),$l=d(oa," special method."),oa.forEach(o),xl=c(dt),y(Fo.$$.fragment,dt),Bl=c(dt),y(qo.$$.fragment,dt),dt.forEach(o),st.forEach(o),gs=c(t),mo=a(t,"H2",{class:!0});var As=r(mo);Eo=a(As,"A",{id:!0,class:!0,href:!0});var rm=r(Eo);Ta=a(rm,"SPAN",{});var sm=r(Ta);y(Mt.$$.fragment,sm),sm.forEach(o),rm.forEach(o),zl=c(As),wa=a(As,"SPAN",{});var dm=r(wa);Fl=d(dm,"BlenderbotSmallForConditionalGeneration"),dm.forEach(o),As.forEach(o),bs=c(t),$e=a(t,"DIV",{class:!0});var lt=r($e);y(Ct.$$.fragment,lt),ql=c(lt),jt=a(lt,"P",{});var Ds=r(jt);El=d(Ds,`The BlenderbotSmall Model with a language modeling head. Can be used for summarization.
This model inherits from `),An=a(Ds,"A",{href:!0});var lm=r(An);Ml=d(lm,"PreTrainedModel"),lm.forEach(o),Cl=d(Ds,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Ds.forEach(o),jl=c(lt),Ot=a(lt,"P",{});var Gs=r(Ot);Ol=d(Gs,"This model is also a PyTorch "),Pt=a(Gs,"A",{href:!0,rel:!0});var im=r(Pt);Pl=d(im,"torch.nn.Module"),im.forEach(o),Ll=d(Gs,` subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`),Gs.forEach(o),Nl=c(lt),Ee=a(lt,"DIV",{class:!0});var it=r(Ee);y(Lt.$$.fragment,it),Il=c(it),ho=a(it,"P",{});var ta=r(ho);Al=d(ta,"The "),Dn=a(ta,"A",{href:!0});var cm=r(Dn);Dl=d(cm,"BlenderbotSmallForConditionalGeneration"),cm.forEach(o),Gl=d(ta," forward method, overrides the "),Sa=a(ta,"CODE",{});var pm=r(Sa);Ul=d(pm,"__call__"),pm.forEach(o),Wl=d(ta," special method."),ta.forEach(o),Rl=c(it),y(Mo.$$.fragment,it),Hl=c(it),y(Co.$$.fragment,it),it.forEach(o),lt.forEach(o),ks=c(t),uo=a(t,"H2",{class:!0});var Us=r(uo);jo=a(Us,"A",{id:!0,class:!0,href:!0});var mm=r(jo);$a=a(mm,"SPAN",{});var hm=r($a);y(Nt.$$.fragment,hm),hm.forEach(o),mm.forEach(o),Kl=c(Us),xa=a(Us,"SPAN",{});var um=r(xa);Vl=d(um,"BlenderbotSmallForCausalLM"),um.forEach(o),Us.forEach(o),vs=c(t),fo=a(t,"DIV",{class:!0});var Ws=r(fo);y(It.$$.fragment,Ws),Ql=c(Ws),Oo=a(Ws,"DIV",{class:!0});var Rs=r(Oo);y(At.$$.fragment,Rs),Jl=c(Rs),y(Po.$$.fragment,Rs),Rs.forEach(o),Ws.forEach(o),ys=c(t),_o=a(t,"H2",{class:!0});var Hs=r(_o);Lo=a(Hs,"A",{id:!0,class:!0,href:!0});var fm=r(Lo);Ba=a(fm,"SPAN",{});var _m=r(Ba);y(Dt.$$.fragment,_m),_m.forEach(o),fm.forEach(o),Xl=c(Hs),za=a(Hs,"SPAN",{});var gm=r(za);Yl=d(gm,"TFBlenderbotSmallModel"),gm.forEach(o),Hs.forEach(o),Ts=c(t),pe=a(t,"DIV",{class:!0});var Ve=r(pe);y(Gt.$$.fragment,Ve),Zl=c(Ve),Ut=a(Ve,"P",{});var Ks=r(Ut);ei=d(Ks,`The bare BLENDERBOT_SMALL Model outputting raw hidden-states without any specific head on top.
This model inherits from `),Gn=a(Ks,"A",{href:!0});var bm=r(Gn);oi=d(bm,"TFPreTrainedModel"),bm.forEach(o),ti=d(Ks,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Ks.forEach(o),ni=c(Ve),Wt=a(Ve,"P",{});var Vs=r(Wt);ai=d(Vs,"This model is also a "),Rt=a(Vs,"A",{href:!0,rel:!0});var km=r(Rt);ri=d(km,"tf.keras.Model"),km.forEach(o),si=d(Vs,` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),Vs.forEach(o),di=c(Ve),y(No.$$.fragment,Ve),li=c(Ve),Me=a(Ve,"DIV",{class:!0});var ct=r(Me);y(Ht.$$.fragment,ct),ii=c(ct),go=a(ct,"P",{});var na=r(go);ci=d(na,"The "),Un=a(na,"A",{href:!0});var vm=r(Un);pi=d(vm,"TFBlenderbotSmallModel"),vm.forEach(o),mi=d(na," forward method, overrides the "),Fa=a(na,"CODE",{});var ym=r(Fa);hi=d(ym,"__call__"),ym.forEach(o),ui=d(na," special method."),na.forEach(o),fi=c(ct),y(Io.$$.fragment,ct),_i=c(ct),y(Ao.$$.fragment,ct),ct.forEach(o),Ve.forEach(o),ws=c(t),bo=a(t,"H2",{class:!0});var Qs=r(bo);Do=a(Qs,"A",{id:!0,class:!0,href:!0});var Tm=r(Do);qa=a(Tm,"SPAN",{});var wm=r(qa);y(Kt.$$.fragment,wm),wm.forEach(o),Tm.forEach(o),gi=c(Qs),Ea=a(Qs,"SPAN",{});var Sm=r(Ea);bi=d(Sm,"TFBlenderbotSmallForConditionalGeneration"),Sm.forEach(o),Qs.forEach(o),Ss=c(t),me=a(t,"DIV",{class:!0});var Qe=r(me);y(Vt.$$.fragment,Qe),ki=c(Qe),Qt=a(Qe,"P",{});var Js=r(Qt);vi=d(Js,`The BLENDERBOT_SMALL Model with a language modeling head. Can be used for summarization.
This model inherits from `),Wn=a(Js,"A",{href:!0});var $m=r(Wn);yi=d($m,"TFPreTrainedModel"),$m.forEach(o),Ti=d(Js,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),Js.forEach(o),wi=c(Qe),Jt=a(Qe,"P",{});var Xs=r(Jt);Si=d(Xs,"This model is also a "),Xt=a(Xs,"A",{href:!0,rel:!0});var xm=r(Xt);$i=d(xm,"tf.keras.Model"),xm.forEach(o),xi=d(Xs,` subclass. Use it
as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
behavior.`),Xs.forEach(o),Bi=c(Qe),y(Go.$$.fragment,Qe),zi=c(Qe),P=a(Qe,"DIV",{class:!0});var K=r(P);y(Yt.$$.fragment,K),Fi=c(K),ko=a(K,"P",{});var aa=r(ko);qi=d(aa,"The "),Rn=a(aa,"A",{href:!0});var Bm=r(Rn);Ei=d(Bm,"TFBlenderbotSmallForConditionalGeneration"),Bm.forEach(o),Mi=d(aa," forward method, overrides the "),Ma=a(aa,"CODE",{});var zm=r(Ma);Ci=d(zm,"__call__"),zm.forEach(o),ji=d(aa," special method."),aa.forEach(o),Oi=c(K),y(Uo.$$.fragment,K),Pi=c(K),Ca=a(K,"P",{});var Fm=r(Ca);Li=d(Fm,"Conversation example::"),Fm.forEach(o),Ni=c(K),ja=a(K,"BLOCKQUOTE",{});var qm=r(ja);Oa=a(qm,"BLOCKQUOTE",{});var Em=r(Oa);Pa=a(Em,"BLOCKQUOTE",{});var Mm=r(Pa);La=a(Mm,"P",{});var Cm=r(La);Ii=d(Cm,`from transformers import BlenderbotSmallTokenizer, TFBlenderbotSmallForConditionalGeneration >>> mname =
\u2018facebook/blenderbot_small-90M\u2019 >>> model = BlenderbotSmallForConditionalGeneration.from_pretrained(mname) >>>
tokenizer = BlenderbotSmallTokenizer.from_pretrained(mname)`),Cm.forEach(o),Mm.forEach(o),Em.forEach(o),qm.forEach(o),Ai=c(K),Na=a(K,"BLOCKQUOTE",{});var jm=r(Na);Ia=a(jm,"BLOCKQUOTE",{});var Om=r(Ia);Aa=a(Om,"BLOCKQUOTE",{});var Pm=r(Aa);Da=a(Pm,"P",{});var Lm=r(Da);Di=d(Lm,`UTTERANCE = \u201CMy friends are cool but they eat too many carbs.\u201D >>> print(\u201CHuman: \u201D, UTTERANCE) >>> inputs =
tokenizer([UTTERANCE], return_tensors=\u2018tf\u2019)`),Lm.forEach(o),Pm.forEach(o),Om.forEach(o),jm.forEach(o),Gi=c(K),Ga=a(K,"BLOCKQUOTE",{});var Nm=r(Ga);Ua=a(Nm,"BLOCKQUOTE",{});var Im=r(Ua);Wa=a(Im,"BLOCKQUOTE",{});var Am=r(Wa);Ra=a(Am,"P",{});var Dm=r(Ra);Ui=d(Dm,`reply_ids = model.generate(**inputs) >>> print(\u201CBot: \u201D, tokenizer.batch_decode(reply_ids,
skip_special_tokens=True)[0]) what kind of carbs do they eat? i don\u2019t know much about carbs.`),Dm.forEach(o),Am.forEach(o),Im.forEach(o),Nm.forEach(o),Wi=c(K),Ha=a(K,"BLOCKQUOTE",{});var Gm=r(Ha);Ka=a(Gm,"BLOCKQUOTE",{});var Um=r(Ka);Va=a(Um,"BLOCKQUOTE",{});var Wm=r(Va);Zt=a(Wm,"P",{});var Ys=r(Zt);Ri=d(Ys,`REPLY = \u201CI\u2019m not sure\u201D >>> print(\u201CHuman: \u201D, REPLY) >>> NEXT_UTTERANCE = ( \u2026 \u201CMy friends are cool but they
eat too many carbs.</s> \u201D \u2026 \u201D`),Qa=a(Ys,"S",{});var Rm=r(Qa);Hi=d(Rm,"what kind of carbs do they eat? i don\u2019t know much about carbs."),Rm.forEach(o),Ki=d(Ys,` \u201D \u2026
\u201D<s>I\u2019m not sure.\u201D \u2026 )`),Ys.forEach(o),Wm.forEach(o),Um.forEach(o),Gm.forEach(o),Vi=c(K),Ja=a(K,"BLOCKQUOTE",{});var Hm=r(Ja);Xa=a(Hm,"BLOCKQUOTE",{});var Km=r(Xa);Ya=a(Km,"BLOCKQUOTE",{});var Vm=r(Ya);Za=a(Vm,"P",{});var Qm=r(Za);Qi=d(Qm,`inputs = tokenizer([NEXT_UTTERANCE], return_tensors=\u2018tf\u2019) >>> inputs.pop(\u201Ctoken_type_ids\u201D) >>>
next_reply_ids = model.generate(**inputs) >>> print(\u201CBot: \u201D, tokenizer.batch_decode(next_reply_ids,
skip_special_tokens=True)[0])`),Qm.forEach(o),Vm.forEach(o),Km.forEach(o),Hm.forEach(o),K.forEach(o),Qe.forEach(o),$s=c(t),vo=a(t,"H2",{class:!0});var Zs=r(vo);Wo=a(Zs,"A",{id:!0,class:!0,href:!0});var Jm=r(Wo);er=a(Jm,"SPAN",{});var Xm=r(er);y(en.$$.fragment,Xm),Xm.forEach(o),Jm.forEach(o),Ji=c(Zs),or=a(Zs,"SPAN",{});var Ym=r(or);Xi=d(Ym,"FlaxBlenderbotSmallModel"),Ym.forEach(o),Zs.forEach(o),xs=c(t),I=a(t,"DIV",{class:!0});var te=r(I);y(on.$$.fragment,te),Yi=c(te),tn=a(te,"P",{});var ed=r(tn);Zi=d(ed,`The bare BlenderbotSmall Model transformer outputting raw hidden-states without any specific head on top.
This model inherits from `),Hn=a(ed,"A",{href:!0});var Zm=r(Hn);ec=d(Zm,"FlaxPreTrainedModel"),Zm.forEach(o),oc=d(ed,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),ed.forEach(o),tc=c(te),nn=a(te,"P",{});var od=r(nn);nc=d(od,`This model is also a Flax Linen
`),an=a(od,"A",{href:!0,rel:!0});var eh=r(an);ac=d(eh,"flax.nn.Module"),eh.forEach(o),rc=d(od,` subclass. Use it as a
regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.`),od.forEach(o),sc=c(te),tr=a(te,"P",{});var oh=r(tr);dc=d(oh,"Finally, this model supports inherent JAX features such as:"),oh.forEach(o),lc=c(te),Ue=a(te,"UL",{});var pt=r(Ue);nr=a(pt,"LI",{});var th=r(nr);rn=a(th,"A",{href:!0,rel:!0});var nh=r(rn);ic=d(nh,"Just-In-Time (JIT) compilation"),nh.forEach(o),th.forEach(o),cc=c(pt),ar=a(pt,"LI",{});var ah=r(ar);sn=a(ah,"A",{href:!0,rel:!0});var rh=r(sn);pc=d(rh,"Automatic Differentiation"),rh.forEach(o),ah.forEach(o),mc=c(pt),rr=a(pt,"LI",{});var sh=r(rr);dn=a(sh,"A",{href:!0,rel:!0});var dh=r(dn);hc=d(dh,"Vectorization"),dh.forEach(o),sh.forEach(o),uc=c(pt),sr=a(pt,"LI",{});var lh=r(sr);ln=a(lh,"A",{href:!0,rel:!0});var ih=r(ln);fc=d(ih,"Parallelization"),ih.forEach(o),lh.forEach(o),pt.forEach(o),_c=c(te),Ro=a(te,"DIV",{class:!0});var td=r(Ro);y(cn.$$.fragment,td),gc=c(td),y(Ho.$$.fragment,td),td.forEach(o),bc=c(te),Ko=a(te,"DIV",{class:!0});var nd=r(Ko);y(pn.$$.fragment,nd),kc=c(nd),y(Vo.$$.fragment,nd),nd.forEach(o),vc=c(te),Qo=a(te,"DIV",{class:!0});var ad=r(Qo);y(mn.$$.fragment,ad),yc=c(ad),y(Jo.$$.fragment,ad),ad.forEach(o),te.forEach(o),Bs=c(t),yo=a(t,"H2",{class:!0});var rd=r(yo);Xo=a(rd,"A",{id:!0,class:!0,href:!0});var ch=r(Xo);dr=a(ch,"SPAN",{});var ph=r(dr);y(hn.$$.fragment,ph),ph.forEach(o),ch.forEach(o),Tc=c(rd),lr=a(rd,"SPAN",{});var mh=r(lr);wc=d(mh,"FlaxBlenderbotForConditionalGeneration"),mh.forEach(o),rd.forEach(o),zs=c(t),A=a(t,"DIV",{class:!0});var ne=r(A);y(un.$$.fragment,ne),Sc=c(ne),fn=a(ne,"P",{});var sd=r(fn);$c=d(sd,`The BLENDERBOT_SMALL Model with a language modeling head. Can be used for summarization.
This model inherits from `),Kn=a(sd,"A",{href:!0});var hh=r(Kn);xc=d(hh,"FlaxPreTrainedModel"),hh.forEach(o),Bc=d(sd,`. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`),sd.forEach(o),zc=c(ne),_n=a(ne,"P",{});var dd=r(_n);Fc=d(dd,`This model is also a Flax Linen
`),gn=a(dd,"A",{href:!0,rel:!0});var uh=r(gn);qc=d(uh,"flax.nn.Module"),uh.forEach(o),Ec=d(dd,` subclass. Use it as a
regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.`),dd.forEach(o),Mc=c(ne),ir=a(ne,"P",{});var fh=r(ir);Cc=d(fh,"Finally, this model supports inherent JAX features such as:"),fh.forEach(o),jc=c(ne),We=a(ne,"UL",{});var mt=r(We);cr=a(mt,"LI",{});var _h=r(cr);bn=a(_h,"A",{href:!0,rel:!0});var gh=r(bn);Oc=d(gh,"Just-In-Time (JIT) compilation"),gh.forEach(o),_h.forEach(o),Pc=c(mt),pr=a(mt,"LI",{});var bh=r(pr);kn=a(bh,"A",{href:!0,rel:!0});var kh=r(kn);Lc=d(kh,"Automatic Differentiation"),kh.forEach(o),bh.forEach(o),Nc=c(mt),mr=a(mt,"LI",{});var vh=r(mr);vn=a(vh,"A",{href:!0,rel:!0});var yh=r(vn);Ic=d(yh,"Vectorization"),yh.forEach(o),vh.forEach(o),Ac=c(mt),hr=a(mt,"LI",{});var Th=r(hr);yn=a(Th,"A",{href:!0,rel:!0});var wh=r(yn);Dc=d(wh,"Parallelization"),wh.forEach(o),Th.forEach(o),mt.forEach(o),Gc=c(ne),E=a(ne,"DIV",{class:!0});var M=r(E);y(Tn.$$.fragment,M),Uc=c(M),To=a(M,"P",{});var ra=r(To);Wc=d(ra,"The "),ur=a(ra,"CODE",{});var Sh=r(ur);Rc=d(Sh,"FlaxBlenderbotSmallPreTrainedModel"),Sh.forEach(o),Hc=d(ra," forward method, overrides the "),fr=a(ra,"CODE",{});var $h=r(fr);Kc=d($h,"__call__"),$h.forEach(o),Vc=d(ra," special method."),ra.forEach(o),Qc=c(M),y(Yo.$$.fragment,M),Jc=c(M),_r=a(M,"P",{});var xh=r(_r);Xc=d(xh,"Summarization example:"),xh.forEach(o),Yc=c(M),gr=a(M,"BLOCKQUOTE",{});var Bh=r(gr);br=a(Bh,"BLOCKQUOTE",{});var zh=r(br);kr=a(zh,"BLOCKQUOTE",{});var Fh=r(kr);vr=a(Fh,"P",{});var qh=r(vr);Zc=d(qh,"from transformers import BlenderbotSmallTokenizer, FlaxBlenderbotSmallForConditionalGeneration"),qh.forEach(o),Fh.forEach(o),zh.forEach(o),Bh.forEach(o),ep=c(M),yr=a(M,"BLOCKQUOTE",{});var Eh=r(yr);Tr=a(Eh,"BLOCKQUOTE",{});var Mh=r(Tr);wr=a(Mh,"BLOCKQUOTE",{});var Ch=r(wr);Sr=a(Ch,"P",{});var jh=r(Sr);op=d(jh,`model = FlaxBlenderbotSmallForConditionalGeneration.from_pretrained(\u2018facebook/blenderbot_small-90M\u2019) >>>
tokenizer = BlenderbotSmallTokenizer.from_pretrained(\u2018facebook/blenderbot_small-90M\u2019)`),jh.forEach(o),Ch.forEach(o),Mh.forEach(o),Eh.forEach(o),tp=c(M),$r=a(M,"BLOCKQUOTE",{});var Oh=r($r);xr=a(Oh,"BLOCKQUOTE",{});var Ph=r(xr);Br=a(Ph,"BLOCKQUOTE",{});var Lh=r(Br);zr=a(Lh,"P",{});var Nh=r(zr);np=d(Nh,`ARTICLE_TO_SUMMARIZE = \u201CMy friends are cool but they eat too many carbs.\u201D >>> inputs =
tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors=\u2018np\u2019)`),Nh.forEach(o),Lh.forEach(o),Ph.forEach(o),Oh.forEach(o),ap=c(M),Fr=a(M,"BLOCKQUOTE",{});var Ih=r(Fr);qr=a(Ih,"BLOCKQUOTE",{});var Ah=r(qr);wn=a(Ah,"BLOCKQUOTE",{});var ld=r(wn);Zo=a(ld,"H1",{class:!0});var id=r(Zo);et=a(id,"A",{id:!0,class:!0,href:!0});var Dh=r(et);Er=a(Dh,"SPAN",{});var Gh=r(Er);y(Sn.$$.fragment,Gh),Gh.forEach(o),Dh.forEach(o),rp=c(id),Mr=a(id,"SPAN",{});var Uh=r(Mr);sp=d(Uh,"Generate Summary >>> summary_ids = model.generate(inputs[\u2018input_ids\u2019]).sequences >>>"),Uh.forEach(o),id.forEach(o),dp=c(ld),Cr=a(ld,"P",{});var Wh=r(Cr);lp=d(Wh,"print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))"),Wh.forEach(o),ld.forEach(o),Ah.forEach(o),Ih.forEach(o),ip=c(M),jr=a(M,"P",{});var Rh=r(jr);cp=d(Rh,"Mask filling example:"),Rh.forEach(o),pp=c(M),Or=a(M,"BLOCKQUOTE",{});var Hh=r(Or);Pr=a(Hh,"BLOCKQUOTE",{});var Kh=r(Pr);Lr=a(Kh,"BLOCKQUOTE",{});var Vh=r(Lr);Nr=a(Vh,"P",{});var Qh=r(Nr);mp=d(Qh,`from transformers import BlenderbotSmallTokenizer, FlaxBlenderbotSmallForConditionalGeneration >>>
tokenizer = BlenderbotSmallTokenizer.from_pretrained(\u2018facebook/blenderbot_small-90M\u2019) >>> TXT = \u201CMy friends are
<mask> but they eat too many carbs.\u201D`),Qh.forEach(o),Vh.forEach(o),Kh.forEach(o),Hh.forEach(o),hp=c(M),Ir=a(M,"BLOCKQUOTE",{});var Jh=r(Ir);Ar=a(Jh,"BLOCKQUOTE",{});var Xh=r(Ar);Dr=a(Xh,"BLOCKQUOTE",{});var Yh=r(Dr);Gr=a(Yh,"P",{});var Zh=r(Gr);up=d(Zh,`model = FlaxBlenderbotSmallForConditionalGeneration.from_pretrained(\u2018facebook/blenderbot_small-90M\u2019) >>>
input_ids = tokenizer([TXT], return_tensors=\u2018np\u2019)[\u2018input_ids\u2019] >>> logits = model(input_ids).logits`),Zh.forEach(o),Yh.forEach(o),Xh.forEach(o),Jh.forEach(o),fp=c(M),Ur=a(M,"BLOCKQUOTE",{});var eu=r(Ur);Wr=a(eu,"BLOCKQUOTE",{});var ou=r(Wr);Rr=a(ou,"BLOCKQUOTE",{});var tu=r(Rr);Hr=a(tu,"P",{});var nu=r(Hr);_p=d(nu,`masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item() >>> probs =
jax.nn.softmax(logits[0, masked_index], axis=0) >>> values, predictions = jax.lax.top_k(probs)`),nu.forEach(o),tu.forEach(o),ou.forEach(o),eu.forEach(o),gp=c(M),Kr=a(M,"BLOCKQUOTE",{});var au=r(Kr);Vr=a(au,"BLOCKQUOTE",{});var ru=r(Vr);Qr=a(ru,"BLOCKQUOTE",{});var su=r(Qr);Jr=a(su,"P",{});var du=r(Jr);bp=d(du,"tokenizer.decode(predictions).split()"),du.forEach(o),su.forEach(o),ru.forEach(o),au.forEach(o),M.forEach(o),kp=c(ne),ot=a(ne,"DIV",{class:!0});var cd=r(ot);y($n.$$.fragment,cd),vp=c(cd),y(tt.$$.fragment,cd),cd.forEach(o),yp=c(ne),nt=a(ne,"DIV",{class:!0});var pd=r(nt);y(xn.$$.fragment,pd),Tp=c(pd),y(at.$$.fragment,pd),pd.forEach(o),ne.forEach(o),this.h()},h(){m(p,"name","hf:doc:metadata"),m(p,"content",JSON.stringify(Eu)),m(h,"id","blenderbot-small"),m(h,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(h,"href","#blenderbot-small"),m(f,"class","relative group"),m(W,"href","/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallModel"),m(R,"href","/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallForConditionalGeneration"),m(V,"href","https://huggingface.co/facebook/blenderbot-90M"),m(V,"rel","nofollow"),m(Z,"href","/docs/transformers/pr_18694/en/model_doc/blenderbot#transformers.BlenderbotModel"),m(ue,"href","/docs/transformers/pr_18694/en/model_doc/blenderbot#transformers.BlenderbotForConditionalGeneration"),m(D,"id","overview"),m(D,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(D,"href","#overview"),m(J,"class","relative group"),m(le,"href","https://arxiv.org/pdf/2004.13637.pdf"),m(le,"rel","nofollow"),m(ve,"href","https://huggingface.co/patrickvonplaten"),m(ve,"rel","nofollow"),m(G,"href","https://github.com/facebookresearch/ParlAI"),m(G,"rel","nofollow"),m(fe,"id","transformers.BlenderbotSmallConfig"),m(fe,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(fe,"href","#transformers.BlenderbotSmallConfig"),m(ce,"class","relative group"),m(En,"href","/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallModel"),m(ut,"href","https://huggingface.co/facebook/blenderbot_small-90M"),m(ut,"rel","nofollow"),m(Mn,"href","/docs/transformers/pr_18694/en/main_classes/configuration#transformers.PretrainedConfig"),m(Cn,"href","/docs/transformers/pr_18694/en/main_classes/configuration#transformers.PretrainedConfig"),m(we,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(So,"id","transformers.BlenderbotSmallTokenizer"),m(So,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(So,"href","#transformers.BlenderbotSmallTokenizer"),m(so,"class","relative group"),m(jn,"href","/docs/transformers/pr_18694/en/main_classes/tokenizer#transformers.PreTrainedTokenizer"),m(He,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m($o,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Pn,"href","../glossary#token-type-ids"),m(Ke,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Ln,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(xo,"id","transformers.BlenderbotSmallTokenizerFast"),m(xo,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(xo,"href","#transformers.BlenderbotSmallTokenizerFast"),m(io,"class","relative group"),m(Bo,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Ge,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(zo,"id","transformers.BlenderbotSmallModel"),m(zo,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(zo,"href","#transformers.BlenderbotSmallModel"),m(co,"class","relative group"),m(Nn,"href","/docs/transformers/pr_18694/en/main_classes/model#transformers.PreTrainedModel"),m(qt,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),m(qt,"rel","nofollow"),m(In,"href","/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallModel"),m(qe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Se,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Eo,"id","transformers.BlenderbotSmallForConditionalGeneration"),m(Eo,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(Eo,"href","#transformers.BlenderbotSmallForConditionalGeneration"),m(mo,"class","relative group"),m(An,"href","/docs/transformers/pr_18694/en/main_classes/model#transformers.PreTrainedModel"),m(Pt,"href","https://pytorch.org/docs/stable/nn.html#torch.nn.Module"),m(Pt,"rel","nofollow"),m(Dn,"href","/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.BlenderbotSmallForConditionalGeneration"),m(Ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m($e,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(jo,"id","transformers.BlenderbotSmallForCausalLM"),m(jo,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(jo,"href","#transformers.BlenderbotSmallForCausalLM"),m(uo,"class","relative group"),m(Oo,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(fo,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Lo,"id","transformers.TFBlenderbotSmallModel"),m(Lo,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(Lo,"href","#transformers.TFBlenderbotSmallModel"),m(_o,"class","relative group"),m(Gn,"href","/docs/transformers/pr_18694/en/main_classes/model#transformers.TFPreTrainedModel"),m(Rt,"href","https://www.tensorflow.org/api_docs/python/tf/keras/Model"),m(Rt,"rel","nofollow"),m(Un,"href","/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.TFBlenderbotSmallModel"),m(Me,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(pe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Do,"id","transformers.TFBlenderbotSmallForConditionalGeneration"),m(Do,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(Do,"href","#transformers.TFBlenderbotSmallForConditionalGeneration"),m(bo,"class","relative group"),m(Wn,"href","/docs/transformers/pr_18694/en/main_classes/model#transformers.TFPreTrainedModel"),m(Xt,"href","https://www.tensorflow.org/api_docs/python/tf/keras/Model"),m(Xt,"rel","nofollow"),m(Rn,"href","/docs/transformers/pr_18694/en/model_doc/blenderbot-small#transformers.TFBlenderbotSmallForConditionalGeneration"),m(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(me,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Wo,"id","transformers.FlaxBlenderbotSmallModel"),m(Wo,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(Wo,"href","#transformers.FlaxBlenderbotSmallModel"),m(vo,"class","relative group"),m(Hn,"href","/docs/transformers/pr_18694/en/main_classes/model#transformers.FlaxPreTrainedModel"),m(an,"href","https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html"),m(an,"rel","nofollow"),m(rn,"href","https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit"),m(rn,"rel","nofollow"),m(sn,"href","https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation"),m(sn,"rel","nofollow"),m(dn,"href","https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap"),m(dn,"rel","nofollow"),m(ln,"href","https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap"),m(ln,"rel","nofollow"),m(Ro,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Ko,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Qo,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(Xo,"id","transformers.FlaxBlenderbotSmallForConditionalGeneration"),m(Xo,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(Xo,"href","#transformers.FlaxBlenderbotSmallForConditionalGeneration"),m(yo,"class","relative group"),m(Kn,"href","/docs/transformers/pr_18694/en/main_classes/model#transformers.FlaxPreTrainedModel"),m(gn,"href","https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html"),m(gn,"rel","nofollow"),m(bn,"href","https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit"),m(bn,"rel","nofollow"),m(kn,"href","https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation"),m(kn,"rel","nofollow"),m(vn,"href","https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap"),m(vn,"rel","nofollow"),m(yn,"href","https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap"),m(yn,"rel","nofollow"),m(et,"id","generate-summary->>>-summary_ids-=-model.generate(inputs[\u2018input_ids\u2019]).sequences->>>"),m(et,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),m(et,"href","#generate-summary->>>-summary_ids-=-model.generate(inputs[\u2018input_ids\u2019]).sequences->>>"),m(Zo,"class","relative group"),m(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(ot,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(nt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),m(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(t,g){e(document.head,p),_(t,k,g),_(t,f,g),e(f,h),e(h,b),T(l,b,null),e(f,u),e(f,F),e(F,Ce),_(t,he,g),_(t,B,g),e(B,je),e(B,W),e(W,Oe),e(B,Pe),e(B,R),e(R,Le),e(B,Ne),e(B,V),e(V,Q),e(B,Ie),e(B,Z),e(Z,j),e(B,L),e(B,ue),e(ue,ae),_(t,Be,g),_(t,J,g),e(J,D),e(D,be),T(re,be,null),e(J,N),e(J,ke),e(ke,se),_(t,ze,g),_(t,ee,g),e(ee,de),e(ee,le),e(le,Ae),e(ee,X),_(t,Fe,g),_(t,H,g),e(H,De),_(t,x,g),_(t,q,g),e(q,ie),e(ie,Ze),_(t,Re,g),_(t,O,g),e(O,eo),e(O,ve),e(ve,oo),e(O,Y),e(O,G),e(G,to),e(O,no),_(t,oe,g),_(t,ce,g),e(ce,fe),e(fe,ye),T(Te,ye,null),e(ce,md),e(ce,sa),e(sa,hd),_(t,cs,g),_(t,we,g),T(ht,we,null),e(we,ud),e(we,ao),e(ao,fd),e(ao,En),e(En,_d),e(ao,gd),e(ao,ut),e(ut,bd),e(ao,kd),e(we,vd),e(we,ro),e(ro,yd),e(ro,Mn),e(Mn,Td),e(ro,wd),e(ro,Cn),e(Cn,Sd),e(ro,$d),e(we,xd),T(wo,we,null),_(t,ps,g),_(t,so,g),e(so,So),e(So,da),T(ft,da,null),e(so,Bd),e(so,la),e(la,zd),_(t,ms,g),_(t,U,g),T(_t,U,null),e(U,Fd),e(U,ia),e(ia,qd),e(U,Ed),e(U,gt),e(gt,Md),e(gt,jn),e(jn,Cd),e(gt,jd),e(U,Od),e(U,He),T(bt,He,null),e(He,Pd),e(He,ca),e(ca,Ld),e(He,Nd),e(He,pa),e(pa,Id),e(U,Ad),e(U,$o),T(kt,$o,null),e($o,Dd),e($o,lo),e(lo,Gd),e(lo,ma),e(ma,Ud),e(lo,Wd),e(lo,ha),e(ha,Rd),e(lo,Hd),e(U,Kd),e(U,Ke),T(vt,Ke,null),e(Ke,Vd),e(Ke,On),e(On,Qd),e(On,Pn),e(Pn,Jd),e(Ke,Xd),e(Ke,ua),e(ua,Yd),e(U,Zd),e(U,Ln),T(yt,Ln,null),_(t,hs,g),_(t,io,g),e(io,xo),e(xo,fa),T(Tt,fa,null),e(io,el),e(io,_a),e(_a,ol),_(t,us,g),_(t,Ge,g),T(wt,Ge,null),e(Ge,tl),e(Ge,St),e(St,nl),e(St,ga),e(ga,al),e(St,rl),e(Ge,sl),e(Ge,Bo),T($t,Bo,null),e(Bo,dl),e(Bo,ba),e(ba,ll),_(t,fs,g),_(t,co,g),e(co,zo),e(zo,ka),T(xt,ka,null),e(co,il),e(co,va),e(va,cl),_(t,_s,g),_(t,Se,g),T(Bt,Se,null),e(Se,pl),e(Se,zt),e(zt,ml),e(zt,Nn),e(Nn,hl),e(zt,ul),e(Se,fl),e(Se,Ft),e(Ft,_l),e(Ft,qt),e(qt,gl),e(Ft,bl),e(Se,kl),e(Se,qe),T(Et,qe,null),e(qe,vl),e(qe,po),e(po,yl),e(po,In),e(In,Tl),e(po,wl),e(po,ya),e(ya,Sl),e(po,$l),e(qe,xl),T(Fo,qe,null),e(qe,Bl),T(qo,qe,null),_(t,gs,g),_(t,mo,g),e(mo,Eo),e(Eo,Ta),T(Mt,Ta,null),e(mo,zl),e(mo,wa),e(wa,Fl),_(t,bs,g),_(t,$e,g),T(Ct,$e,null),e($e,ql),e($e,jt),e(jt,El),e(jt,An),e(An,Ml),e(jt,Cl),e($e,jl),e($e,Ot),e(Ot,Ol),e(Ot,Pt),e(Pt,Pl),e(Ot,Ll),e($e,Nl),e($e,Ee),T(Lt,Ee,null),e(Ee,Il),e(Ee,ho),e(ho,Al),e(ho,Dn),e(Dn,Dl),e(ho,Gl),e(ho,Sa),e(Sa,Ul),e(ho,Wl),e(Ee,Rl),T(Mo,Ee,null),e(Ee,Hl),T(Co,Ee,null),_(t,ks,g),_(t,uo,g),e(uo,jo),e(jo,$a),T(Nt,$a,null),e(uo,Kl),e(uo,xa),e(xa,Vl),_(t,vs,g),_(t,fo,g),T(It,fo,null),e(fo,Ql),e(fo,Oo),T(At,Oo,null),e(Oo,Jl),T(Po,Oo,null),_(t,ys,g),_(t,_o,g),e(_o,Lo),e(Lo,Ba),T(Dt,Ba,null),e(_o,Xl),e(_o,za),e(za,Yl),_(t,Ts,g),_(t,pe,g),T(Gt,pe,null),e(pe,Zl),e(pe,Ut),e(Ut,ei),e(Ut,Gn),e(Gn,oi),e(Ut,ti),e(pe,ni),e(pe,Wt),e(Wt,ai),e(Wt,Rt),e(Rt,ri),e(Wt,si),e(pe,di),T(No,pe,null),e(pe,li),e(pe,Me),T(Ht,Me,null),e(Me,ii),e(Me,go),e(go,ci),e(go,Un),e(Un,pi),e(go,mi),e(go,Fa),e(Fa,hi),e(go,ui),e(Me,fi),T(Io,Me,null),e(Me,_i),T(Ao,Me,null),_(t,ws,g),_(t,bo,g),e(bo,Do),e(Do,qa),T(Kt,qa,null),e(bo,gi),e(bo,Ea),e(Ea,bi),_(t,Ss,g),_(t,me,g),T(Vt,me,null),e(me,ki),e(me,Qt),e(Qt,vi),e(Qt,Wn),e(Wn,yi),e(Qt,Ti),e(me,wi),e(me,Jt),e(Jt,Si),e(Jt,Xt),e(Xt,$i),e(Jt,xi),e(me,Bi),T(Go,me,null),e(me,zi),e(me,P),T(Yt,P,null),e(P,Fi),e(P,ko),e(ko,qi),e(ko,Rn),e(Rn,Ei),e(ko,Mi),e(ko,Ma),e(Ma,Ci),e(ko,ji),e(P,Oi),T(Uo,P,null),e(P,Pi),e(P,Ca),e(Ca,Li),e(P,Ni),e(P,ja),e(ja,Oa),e(Oa,Pa),e(Pa,La),e(La,Ii),e(P,Ai),e(P,Na),e(Na,Ia),e(Ia,Aa),e(Aa,Da),e(Da,Di),e(P,Gi),e(P,Ga),e(Ga,Ua),e(Ua,Wa),e(Wa,Ra),e(Ra,Ui),e(P,Wi),e(P,Ha),e(Ha,Ka),e(Ka,Va),e(Va,Zt),e(Zt,Ri),e(Zt,Qa),e(Qa,Hi),e(Zt,Ki),e(P,Vi),e(P,Ja),e(Ja,Xa),e(Xa,Ya),e(Ya,Za),e(Za,Qi),_(t,$s,g),_(t,vo,g),e(vo,Wo),e(Wo,er),T(en,er,null),e(vo,Ji),e(vo,or),e(or,Xi),_(t,xs,g),_(t,I,g),T(on,I,null),e(I,Yi),e(I,tn),e(tn,Zi),e(tn,Hn),e(Hn,ec),e(tn,oc),e(I,tc),e(I,nn),e(nn,nc),e(nn,an),e(an,ac),e(nn,rc),e(I,sc),e(I,tr),e(tr,dc),e(I,lc),e(I,Ue),e(Ue,nr),e(nr,rn),e(rn,ic),e(Ue,cc),e(Ue,ar),e(ar,sn),e(sn,pc),e(Ue,mc),e(Ue,rr),e(rr,dn),e(dn,hc),e(Ue,uc),e(Ue,sr),e(sr,ln),e(ln,fc),e(I,_c),e(I,Ro),T(cn,Ro,null),e(Ro,gc),T(Ho,Ro,null),e(I,bc),e(I,Ko),T(pn,Ko,null),e(Ko,kc),T(Vo,Ko,null),e(I,vc),e(I,Qo),T(mn,Qo,null),e(Qo,yc),T(Jo,Qo,null),_(t,Bs,g),_(t,yo,g),e(yo,Xo),e(Xo,dr),T(hn,dr,null),e(yo,Tc),e(yo,lr),e(lr,wc),_(t,zs,g),_(t,A,g),T(un,A,null),e(A,Sc),e(A,fn),e(fn,$c),e(fn,Kn),e(Kn,xc),e(fn,Bc),e(A,zc),e(A,_n),e(_n,Fc),e(_n,gn),e(gn,qc),e(_n,Ec),e(A,Mc),e(A,ir),e(ir,Cc),e(A,jc),e(A,We),e(We,cr),e(cr,bn),e(bn,Oc),e(We,Pc),e(We,pr),e(pr,kn),e(kn,Lc),e(We,Nc),e(We,mr),e(mr,vn),e(vn,Ic),e(We,Ac),e(We,hr),e(hr,yn),e(yn,Dc),e(A,Gc),e(A,E),T(Tn,E,null),e(E,Uc),e(E,To),e(To,Wc),e(To,ur),e(ur,Rc),e(To,Hc),e(To,fr),e(fr,Kc),e(To,Vc),e(E,Qc),T(Yo,E,null),e(E,Jc),e(E,_r),e(_r,Xc),e(E,Yc),e(E,gr),e(gr,br),e(br,kr),e(kr,vr),e(vr,Zc),e(E,ep),e(E,yr),e(yr,Tr),e(Tr,wr),e(wr,Sr),e(Sr,op),e(E,tp),e(E,$r),e($r,xr),e(xr,Br),e(Br,zr),e(zr,np),e(E,ap),e(E,Fr),e(Fr,qr),e(qr,wn),e(wn,Zo),e(Zo,et),e(et,Er),T(Sn,Er,null),e(Zo,rp),e(Zo,Mr),e(Mr,sp),e(wn,dp),e(wn,Cr),e(Cr,lp),e(E,ip),e(E,jr),e(jr,cp),e(E,pp),e(E,Or),e(Or,Pr),e(Pr,Lr),e(Lr,Nr),e(Nr,mp),e(E,hp),e(E,Ir),e(Ir,Ar),e(Ar,Dr),e(Dr,Gr),e(Gr,up),e(E,fp),e(E,Ur),e(Ur,Wr),e(Wr,Rr),e(Rr,Hr),e(Hr,_p),e(E,gp),e(E,Kr),e(Kr,Vr),e(Vr,Qr),e(Qr,Jr),e(Jr,bp),e(A,kp),e(A,ot),T($n,ot,null),e(ot,vp),T(tt,ot,null),e(A,yp),e(A,nt),T(xn,nt,null),e(nt,Tp),T(at,nt,null),Fs=!0},p(t,[g]){const Bn={};g&2&&(Bn.$$scope={dirty:g,ctx:t}),wo.$set(Bn);const Xr={};g&2&&(Xr.$$scope={dirty:g,ctx:t}),Fo.$set(Xr);const Yr={};g&2&&(Yr.$$scope={dirty:g,ctx:t}),qo.$set(Yr);const Zr={};g&2&&(Zr.$$scope={dirty:g,ctx:t}),Mo.$set(Zr);const _e={};g&2&&(_e.$$scope={dirty:g,ctx:t}),Co.$set(_e);const es={};g&2&&(es.$$scope={dirty:g,ctx:t}),Po.$set(es);const os={};g&2&&(os.$$scope={dirty:g,ctx:t}),No.$set(os);const ts={};g&2&&(ts.$$scope={dirty:g,ctx:t}),Io.$set(ts);const ns={};g&2&&(ns.$$scope={dirty:g,ctx:t}),Ao.$set(ns);const as={};g&2&&(as.$$scope={dirty:g,ctx:t}),Go.$set(as);const zn={};g&2&&(zn.$$scope={dirty:g,ctx:t}),Uo.$set(zn);const rs={};g&2&&(rs.$$scope={dirty:g,ctx:t}),Ho.$set(rs);const ss={};g&2&&(ss.$$scope={dirty:g,ctx:t}),Vo.$set(ss);const ds={};g&2&&(ds.$$scope={dirty:g,ctx:t}),Jo.$set(ds);const Fn={};g&2&&(Fn.$$scope={dirty:g,ctx:t}),Yo.$set(Fn);const ls={};g&2&&(ls.$$scope={dirty:g,ctx:t}),tt.$set(ls);const is={};g&2&&(is.$$scope={dirty:g,ctx:t}),at.$set(is)},i(t){Fs||(w(l.$$.fragment,t),w(re.$$.fragment,t),w(Te.$$.fragment,t),w(ht.$$.fragment,t),w(wo.$$.fragment,t),w(ft.$$.fragment,t),w(_t.$$.fragment,t),w(bt.$$.fragment,t),w(kt.$$.fragment,t),w(vt.$$.fragment,t),w(yt.$$.fragment,t),w(Tt.$$.fragment,t),w(wt.$$.fragment,t),w($t.$$.fragment,t),w(xt.$$.fragment,t),w(Bt.$$.fragment,t),w(Et.$$.fragment,t),w(Fo.$$.fragment,t),w(qo.$$.fragment,t),w(Mt.$$.fragment,t),w(Ct.$$.fragment,t),w(Lt.$$.fragment,t),w(Mo.$$.fragment,t),w(Co.$$.fragment,t),w(Nt.$$.fragment,t),w(It.$$.fragment,t),w(At.$$.fragment,t),w(Po.$$.fragment,t),w(Dt.$$.fragment,t),w(Gt.$$.fragment,t),w(No.$$.fragment,t),w(Ht.$$.fragment,t),w(Io.$$.fragment,t),w(Ao.$$.fragment,t),w(Kt.$$.fragment,t),w(Vt.$$.fragment,t),w(Go.$$.fragment,t),w(Yt.$$.fragment,t),w(Uo.$$.fragment,t),w(en.$$.fragment,t),w(on.$$.fragment,t),w(cn.$$.fragment,t),w(Ho.$$.fragment,t),w(pn.$$.fragment,t),w(Vo.$$.fragment,t),w(mn.$$.fragment,t),w(Jo.$$.fragment,t),w(hn.$$.fragment,t),w(un.$$.fragment,t),w(Tn.$$.fragment,t),w(Yo.$$.fragment,t),w(Sn.$$.fragment,t),w($n.$$.fragment,t),w(tt.$$.fragment,t),w(xn.$$.fragment,t),w(at.$$.fragment,t),Fs=!0)},o(t){S(l.$$.fragment,t),S(re.$$.fragment,t),S(Te.$$.fragment,t),S(ht.$$.fragment,t),S(wo.$$.fragment,t),S(ft.$$.fragment,t),S(_t.$$.fragment,t),S(bt.$$.fragment,t),S(kt.$$.fragment,t),S(vt.$$.fragment,t),S(yt.$$.fragment,t),S(Tt.$$.fragment,t),S(wt.$$.fragment,t),S($t.$$.fragment,t),S(xt.$$.fragment,t),S(Bt.$$.fragment,t),S(Et.$$.fragment,t),S(Fo.$$.fragment,t),S(qo.$$.fragment,t),S(Mt.$$.fragment,t),S(Ct.$$.fragment,t),S(Lt.$$.fragment,t),S(Mo.$$.fragment,t),S(Co.$$.fragment,t),S(Nt.$$.fragment,t),S(It.$$.fragment,t),S(At.$$.fragment,t),S(Po.$$.fragment,t),S(Dt.$$.fragment,t),S(Gt.$$.fragment,t),S(No.$$.fragment,t),S(Ht.$$.fragment,t),S(Io.$$.fragment,t),S(Ao.$$.fragment,t),S(Kt.$$.fragment,t),S(Vt.$$.fragment,t),S(Go.$$.fragment,t),S(Yt.$$.fragment,t),S(Uo.$$.fragment,t),S(en.$$.fragment,t),S(on.$$.fragment,t),S(cn.$$.fragment,t),S(Ho.$$.fragment,t),S(pn.$$.fragment,t),S(Vo.$$.fragment,t),S(mn.$$.fragment,t),S(Jo.$$.fragment,t),S(hn.$$.fragment,t),S(un.$$.fragment,t),S(Tn.$$.fragment,t),S(Yo.$$.fragment,t),S(Sn.$$.fragment,t),S($n.$$.fragment,t),S(tt.$$.fragment,t),S(xn.$$.fragment,t),S(at.$$.fragment,t),Fs=!1},d(t){o(p),t&&o(k),t&&o(f),$(l),t&&o(he),t&&o(B),t&&o(Be),t&&o(J),$(re),t&&o(ze),t&&o(ee),t&&o(Fe),t&&o(H),t&&o(x),t&&o(q),t&&o(Re),t&&o(O),t&&o(oe),t&&o(ce),$(Te),t&&o(cs),t&&o(we),$(ht),$(wo),t&&o(ps),t&&o(so),$(ft),t&&o(ms),t&&o(U),$(_t),$(bt),$(kt),$(vt),$(yt),t&&o(hs),t&&o(io),$(Tt),t&&o(us),t&&o(Ge),$(wt),$($t),t&&o(fs),t&&o(co),$(xt),t&&o(_s),t&&o(Se),$(Bt),$(Et),$(Fo),$(qo),t&&o(gs),t&&o(mo),$(Mt),t&&o(bs),t&&o($e),$(Ct),$(Lt),$(Mo),$(Co),t&&o(ks),t&&o(uo),$(Nt),t&&o(vs),t&&o(fo),$(It),$(At),$(Po),t&&o(ys),t&&o(_o),$(Dt),t&&o(Ts),t&&o(pe),$(Gt),$(No),$(Ht),$(Io),$(Ao),t&&o(ws),t&&o(bo),$(Kt),t&&o(Ss),t&&o(me),$(Vt),$(Go),$(Yt),$(Uo),t&&o($s),t&&o(vo),$(en),t&&o(xs),t&&o(I),$(on),$(cn),$(Ho),$(pn),$(Vo),$(mn),$(Jo),t&&o(Bs),t&&o(yo),$(hn),t&&o(zs),t&&o(A),$(un),$(Tn),$(Yo),$(Sn),$($n),$(tt),$(xn),$(at)}}}const Eu={local:"blenderbot-small",sections:[{local:"overview",title:"Overview"},{local:"transformers.BlenderbotSmallConfig",title:"BlenderbotSmallConfig"},{local:"transformers.BlenderbotSmallTokenizer",title:"BlenderbotSmallTokenizer"},{local:"transformers.BlenderbotSmallTokenizerFast",title:"BlenderbotSmallTokenizerFast"},{local:"transformers.BlenderbotSmallModel",title:"BlenderbotSmallModel"},{local:"transformers.BlenderbotSmallForConditionalGeneration",title:"BlenderbotSmallForConditionalGeneration"},{local:"transformers.BlenderbotSmallForCausalLM",title:"BlenderbotSmallForCausalLM"},{local:"transformers.TFBlenderbotSmallModel",title:"TFBlenderbotSmallModel"},{local:"transformers.TFBlenderbotSmallForConditionalGeneration",title:"TFBlenderbotSmallForConditionalGeneration"},{local:"transformers.FlaxBlenderbotSmallModel",title:"FlaxBlenderbotSmallModel"},{local:"transformers.FlaxBlenderbotSmallForConditionalGeneration",title:"FlaxBlenderbotForConditionalGeneration"}],title:"Blenderbot Small"};function Mu(z){return mu(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Iu extends lu{constructor(p){super();iu(this,p,Mu,qu,cu,{})}}export{Iu as default,Eu as metadata};
