import{S as ss,i as ts,s as ns,e as o,k as d,w as sa,t as n,M as es,c as r,d as t,m,a as i,x as ta,h as e,b as u,G as s,g as c,y as na,L as os,q as ea,o as oa,B as ra,v as rs}from"../chunks/vendor-hf-doc-builder.js";import{I as is}from"../chunks/IconCopyLink-hf-doc-builder.js";import{C as Na}from"../chunks/CodeBlock-hf-doc-builder.js";function ls(La){let g,Q,_,j,H,q,ia,S,la,U,w,pa,I,ua,ca,J,v,x,A,ha,da,ma,C,P,fa,ga,B,p,_a,N,ja,wa,T,va,qa,L,Ta,ba,M,ka,ya,b,$a,Ga,k,Ea,xa,R,y,Y,f,Aa,D,Ca,Pa,$,Oa,za,F,G,K,O,Ha,V,E,W;return q=new is({}),y=new Na({props:{code:`-from transformers import Trainer, TrainingArguments
+from optimum.habana import GaudiTrainer, GaudiTrainingArguments

# define the training arguments
-training_args = TrainingArguments(
+training_args = GaudiTrainingArguments(
+  use_habana=True,
+  use_lazy_mode=True,
+  gaudi_config_name=gaudi_config_name,
  ...
)

# Initialize our Trainer
-trainer = Trainer(
+trainer = GaudiTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
    ... # other arguments
)`,highlighted:`<span class="hljs-deletion">-from transformers import Trainer, TrainingArguments</span>
<span class="hljs-addition">+from optimum.habana import GaudiTrainer, GaudiTrainingArguments</span>

# define the training arguments
<span class="hljs-deletion">-training_args = TrainingArguments(</span>
<span class="hljs-addition">+training_args = GaudiTrainingArguments(</span>
<span class="hljs-addition">+  use_habana=True,</span>
<span class="hljs-addition">+  use_lazy_mode=True,</span>
<span class="hljs-addition">+  gaudi_config_name=gaudi_config_name,</span>
  ...
)

# Initialize our Trainer
<span class="hljs-deletion">-trainer = Trainer(</span>
<span class="hljs-addition">+trainer = GaudiTrainer(</span>
    model=model,
    args=training_args,
    train_dataset=train_dataset
    ... # other arguments
)`}}),G=new Na({props:{code:`{
  "use_habana_mixed_precision": true,
  "hmp_opt_level": "O1",
  "hmp_is_verbose": false,
  "use_fused_adam": true,
  "use_fused_clip_norm": true,
  "hmp_bf16_ops": [
    "add",
    "addmm",
    "bmm",
    "div",
    "dropout",
    "gelu",
    "iadd",
    "linear",
    "layer_norm",
    "matmul",
    "mm",
    "rsub",
    "softmax",
    "truediv"
  ],
  "hmp_fp32_ops": [
    "embedding",
    "nll_loss",
    "log_softmax"
  ]
}`,highlighted:`<span class="hljs-punctuation">{</span>
  <span class="hljs-attr">&quot;use_habana_mixed_precision&quot;</span><span class="hljs-punctuation">:</span> <span class="hljs-keyword">true</span><span class="hljs-punctuation">,</span>
  <span class="hljs-attr">&quot;hmp_opt_level&quot;</span><span class="hljs-punctuation">:</span> <span class="hljs-string">&quot;O1&quot;</span><span class="hljs-punctuation">,</span>
  <span class="hljs-attr">&quot;hmp_is_verbose&quot;</span><span class="hljs-punctuation">:</span> <span class="hljs-keyword">false</span><span class="hljs-punctuation">,</span>
  <span class="hljs-attr">&quot;use_fused_adam&quot;</span><span class="hljs-punctuation">:</span> <span class="hljs-keyword">true</span><span class="hljs-punctuation">,</span>
  <span class="hljs-attr">&quot;use_fused_clip_norm&quot;</span><span class="hljs-punctuation">:</span> <span class="hljs-keyword">true</span><span class="hljs-punctuation">,</span>
  <span class="hljs-attr">&quot;hmp_bf16_ops&quot;</span><span class="hljs-punctuation">:</span> <span class="hljs-punctuation">[</span>
    <span class="hljs-string">&quot;add&quot;</span><span class="hljs-punctuation">,</span>
    <span class="hljs-string">&quot;addmm&quot;</span><span class="hljs-punctuation">,</span>
    <span class="hljs-string">&quot;bmm&quot;</span><span class="hljs-punctuation">,</span>
    <span class="hljs-string">&quot;div&quot;</span><span class="hljs-punctuation">,</span>
    <span class="hljs-string">&quot;dropout&quot;</span><span class="hljs-punctuation">,</span>
    <span class="hljs-string">&quot;gelu&quot;</span><span class="hljs-punctuation">,</span>
    <span class="hljs-string">&quot;iadd&quot;</span><span class="hljs-punctuation">,</span>
    <span class="hljs-string">&quot;linear&quot;</span><span class="hljs-punctuation">,</span>
    <span class="hljs-string">&quot;layer_norm&quot;</span><span class="hljs-punctuation">,</span>
    <span class="hljs-string">&quot;matmul&quot;</span><span class="hljs-punctuation">,</span>
    <span class="hljs-string">&quot;mm&quot;</span><span class="hljs-punctuation">,</span>
    <span class="hljs-string">&quot;rsub&quot;</span><span class="hljs-punctuation">,</span>
    <span class="hljs-string">&quot;softmax&quot;</span><span class="hljs-punctuation">,</span>
    <span class="hljs-string">&quot;truediv&quot;</span>
  <span class="hljs-punctuation">]</span><span class="hljs-punctuation">,</span>
  <span class="hljs-attr">&quot;hmp_fp32_ops&quot;</span><span class="hljs-punctuation">:</span> <span class="hljs-punctuation">[</span>
    <span class="hljs-string">&quot;embedding&quot;</span><span class="hljs-punctuation">,</span>
    <span class="hljs-string">&quot;nll_loss&quot;</span><span class="hljs-punctuation">,</span>
    <span class="hljs-string">&quot;log_softmax&quot;</span>
  <span class="hljs-punctuation">]</span>
<span class="hljs-punctuation">}</span>`}}),E=new Na({props:{code:`gaudi_config = GaudiConfig.from_pretrained(
    gaudi_config_name,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)`,highlighted:`gaudi_config = GaudiConfig.from_pretrained(
    gaudi_config_name,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=<span class="hljs-literal">True</span> <span class="hljs-keyword">if</span> model_args.use_auth_token <span class="hljs-keyword">else</span> <span class="hljs-literal">None</span>,
)`}}),{c(){g=o("meta"),Q=d(),_=o("h1"),j=o("a"),H=o("span"),sa(q.$$.fragment),ia=d(),S=o("span"),la=n("Quickstart"),U=d(),w=o("p"),pa=n("\u{1F917} Optimum Habana was designed with one goal in mind: "),I=o("strong"),ua=n("making training and evaluation straightforward for any \u{1F917} Transformers user while leveraging the complete power of Gaudi processors"),ca=n(`.
There are two main classes one needs to know:`),J=d(),v=o("ul"),x=o("li"),A=o("a"),ha=n("GaudiTrainer"),da=n(": the trainer class that takes care of compiling (lazy or eager mode) and distributing the model to run on HPUs, and of performing traning and evaluation."),ma=d(),C=o("li"),P=o("a"),fa=n("GaudiConfig"),ga=n(": the class that enables to configure Habana Mixed Precision and to decide whether optimized operators and optimizers should be used or not."),B=d(),p=o("p"),_a=n("The "),N=o("code"),ja=n("GaudiTrainer"),wa=n(" is very similar to the "),T=o("a"),va=n("\u{1F917} Transformers Trainer"),qa=n(", and adapting a script using the Trainer to make it work with Gaudi will mostly consist in simply swapping the "),L=o("code"),Ta=n("Trainer"),ba=n(" class for the "),M=o("code"),ka=n("GaudiTrainer"),ya=n(` one.
That is how most of the `),b=o("a"),$a=n("example scripts"),Ga=n(" were adapted from their "),k=o("a"),Ea=n("original counterparts"),xa=n("."),R=d(),sa(y.$$.fragment),Y=d(),f=o("p"),Aa=n("where "),D=o("code"),Ca=n("gaudi_config_name"),Pa=n(" is the name of a model from the "),$=o("a"),Oa=n("Hub"),za=n(" (Gaudi configurations are stored in model repositories). You can also give the path to a custom Gaudi configuration written in a JSON file such as this one:"),F=d(),sa(G.$$.fragment),K=d(),O=o("p"),Ha=n("If you prefer to instantiate a Gaudi configuration to work on it before giving it to the trainer, you can do it as follows:"),V=d(),sa(E.$$.fragment),this.h()},l(a){const l=es('[data-svelte="svelte-1phssyn"]',document.head);g=r(l,"META",{name:!0,content:!0}),l.forEach(t),Q=m(a),_=r(a,"H1",{class:!0});var X=i(_);j=r(X,"A",{id:!0,class:!0,href:!0});var Ma=i(j);H=r(Ma,"SPAN",{});var Da=i(H);ta(q.$$.fragment,Da),Da.forEach(t),Ma.forEach(t),ia=m(X),S=r(X,"SPAN",{});var Qa=i(S);la=e(Qa,"Quickstart"),Qa.forEach(t),X.forEach(t),U=m(a),w=r(a,"P",{});var Z=i(w);pa=e(Z,"\u{1F917} Optimum Habana was designed with one goal in mind: "),I=r(Z,"STRONG",{});var Ua=i(I);ua=e(Ua,"making training and evaluation straightforward for any \u{1F917} Transformers user while leveraging the complete power of Gaudi processors"),Ua.forEach(t),ca=e(Z,`.
There are two main classes one needs to know:`),Z.forEach(t),J=m(a),v=r(a,"UL",{});var aa=i(v);x=r(aa,"LI",{});var Sa=i(x);A=r(Sa,"A",{href:!0});var Ja=i(A);ha=e(Ja,"GaudiTrainer"),Ja.forEach(t),da=e(Sa,": the trainer class that takes care of compiling (lazy or eager mode) and distributing the model to run on HPUs, and of performing traning and evaluation."),Sa.forEach(t),ma=m(aa),C=r(aa,"LI",{});var Ia=i(C);P=r(Ia,"A",{href:!0});var Ba=i(P);fa=e(Ba,"GaudiConfig"),Ba.forEach(t),ga=e(Ia,": the class that enables to configure Habana Mixed Precision and to decide whether optimized operators and optimizers should be used or not."),Ia.forEach(t),aa.forEach(t),B=m(a),p=r(a,"P",{});var h=i(p);_a=e(h,"The "),N=r(h,"CODE",{});var Ra=i(N);ja=e(Ra,"GaudiTrainer"),Ra.forEach(t),wa=e(h," is very similar to the "),T=r(h,"A",{href:!0,rel:!0});var Ya=i(T);va=e(Ya,"\u{1F917} Transformers Trainer"),Ya.forEach(t),qa=e(h,", and adapting a script using the Trainer to make it work with Gaudi will mostly consist in simply swapping the "),L=r(h,"CODE",{});var Fa=i(L);Ta=e(Fa,"Trainer"),Fa.forEach(t),ba=e(h," class for the "),M=r(h,"CODE",{});var Ka=i(M);ka=e(Ka,"GaudiTrainer"),Ka.forEach(t),ya=e(h,` one.
That is how most of the `),b=r(h,"A",{href:!0,rel:!0});var Va=i(b);$a=e(Va,"example scripts"),Va.forEach(t),Ga=e(h," were adapted from their "),k=r(h,"A",{href:!0,rel:!0});var Wa=i(k);Ea=e(Wa,"original counterparts"),Wa.forEach(t),xa=e(h,"."),h.forEach(t),R=m(a),ta(y.$$.fragment,a),Y=m(a),f=r(a,"P",{});var z=i(f);Aa=e(z,"where "),D=r(z,"CODE",{});var Xa=i(D);Ca=e(Xa,"gaudi_config_name"),Xa.forEach(t),Pa=e(z," is the name of a model from the "),$=r(z,"A",{href:!0,rel:!0});var Za=i($);Oa=e(Za,"Hub"),Za.forEach(t),za=e(z," (Gaudi configurations are stored in model repositories). You can also give the path to a custom Gaudi configuration written in a JSON file such as this one:"),z.forEach(t),F=m(a),ta(G.$$.fragment,a),K=m(a),O=r(a,"P",{});var as=i(O);Ha=e(as,"If you prefer to instantiate a Gaudi configuration to work on it before giving it to the trainer, you can do it as follows:"),as.forEach(t),V=m(a),ta(E.$$.fragment,a),this.h()},h(){u(g,"name","hf:doc:metadata"),u(g,"content",JSON.stringify(ps)),u(j,"id","quickstart"),u(j,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),u(j,"href","#quickstart"),u(_,"class","relative group"),u(A,"href","/docs/optimum.habana/pr_89/en/trainer#optimum.habana.GaudiTrainer"),u(P,"href","/docs/optimum.habana/pr_89/en/gaudi_config#optimum.habana.GaudiConfig"),u(T,"href","https://huggingface.co/docs/transformers/main_classes/trainer"),u(T,"rel","nofollow"),u(b,"href","https://github.com/huggingface/optimum-habana/tree/main/examples"),u(b,"rel","nofollow"),u(k,"href","https://github.com/huggingface/transformers/tree/main/examples/pytorch"),u(k,"rel","nofollow"),u($,"href","https://huggingface.co/Habana"),u($,"rel","nofollow")},m(a,l){s(document.head,g),c(a,Q,l),c(a,_,l),s(_,j),s(j,H),na(q,H,null),s(_,ia),s(_,S),s(S,la),c(a,U,l),c(a,w,l),s(w,pa),s(w,I),s(I,ua),s(w,ca),c(a,J,l),c(a,v,l),s(v,x),s(x,A),s(A,ha),s(x,da),s(v,ma),s(v,C),s(C,P),s(P,fa),s(C,ga),c(a,B,l),c(a,p,l),s(p,_a),s(p,N),s(N,ja),s(p,wa),s(p,T),s(T,va),s(p,qa),s(p,L),s(L,Ta),s(p,ba),s(p,M),s(M,ka),s(p,ya),s(p,b),s(b,$a),s(p,Ga),s(p,k),s(k,Ea),s(p,xa),c(a,R,l),na(y,a,l),c(a,Y,l),c(a,f,l),s(f,Aa),s(f,D),s(D,Ca),s(f,Pa),s(f,$),s($,Oa),s(f,za),c(a,F,l),na(G,a,l),c(a,K,l),c(a,O,l),s(O,Ha),c(a,V,l),na(E,a,l),W=!0},p:os,i(a){W||(ea(q.$$.fragment,a),ea(y.$$.fragment,a),ea(G.$$.fragment,a),ea(E.$$.fragment,a),W=!0)},o(a){oa(q.$$.fragment,a),oa(y.$$.fragment,a),oa(G.$$.fragment,a),oa(E.$$.fragment,a),W=!1},d(a){t(g),a&&t(Q),a&&t(_),ra(q),a&&t(U),a&&t(w),a&&t(J),a&&t(v),a&&t(B),a&&t(p),a&&t(R),ra(y,a),a&&t(Y),a&&t(f),a&&t(F),ra(G,a),a&&t(K),a&&t(O),a&&t(V),ra(E,a)}}}const ps={local:"quickstart",title:"Quickstart"};function us(La){return rs(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class ms extends ss{constructor(g){super();ts(this,g,us,ls,ns,{})}}export{ms as default,ps as metadata};
